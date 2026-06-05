"""
Information Sufficiency Ratio (ISR) — pre-compression safety gate.

Background
----------
Chlon's Information Sufficiency Ratio estimates how close a prompt sits to
its Minimum Description Length (MDL). A prompt at MDL has no redundancy left
to remove; compressing it further trades meaningful tokens for shorter text
and is empirically linked to higher hallucination rates downstream. ISR is
intentionally cheap (no LLM calls) so it can fire before the expensive
Bayesian optimisation loop.

Score interpretation
--------------------
    isr ≈ 0.0  highly redundant prompt — aggressive compression is safe
    isr ≈ 1.0  near MDL, do not compress

Components (weighted sum)
-------------------------
    a) Lexical diversity  (weight 0.3)
       Type-token ratio = unique_tokens / total_tokens. High diversity ⇒
       fewer repeated words ⇒ less redundancy to remove.

    b) Shannon entropy    (weight 0.3)
       H = -Σ p(t) log2 p(t) over token frequencies, normalised by log2(V)
       where V is the vocabulary size. High entropy ⇒ flat distribution ⇒
       information-dense prompt.

    c) Semantic variance  (weight 0.4)
       1 - mean pairwise cosine similarity between sentence embeddings
       (all-MiniLM-L6-v2). High variance ⇒ sentences cover distinct ideas ⇒
       hard to compress without losing meaning. (Implemented as
       "1 - mean_similarity" so that the resulting score is directly
       additive in the ISR formula — high consistency among sentences would
       lower this component, signalling redundancy.)

The TF-IDF rareness bonus is left as an optional add-on if a reference
corpus is supplied; the default path is corpus-free so this module can run
on a single prompt without external data.
"""

from __future__ import annotations

import re
import math
from typing import Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


# Component weights — sum to 1.0.
_W_LEXICAL = 0.3
_W_ENTROPY = 0.3
_W_SEMANTIC = 0.4

_EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
_shared_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    global _shared_model
    if _shared_model is None:
        _shared_model = SentenceTransformer(_EMBEDDING_MODEL_NAME)
    return _shared_model


def _tokenise(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _split_sentences(text: str) -> list[str]:
    # Protect numbered-list markers so "1. foo" doesn't become two sentences.
    protected = re.sub(r'(\d+)\.\s+([A-Z])', r'\1NUMMARKER \2', text)
    parts = re.split(r'(?<=[.!?])\s+', protected)
    parts = [p.replace('NUMMARKER', '.') for p in parts]
    return [p.strip() for p in parts if p.strip()]


def _lexical_diversity(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def _normalised_shannon_entropy(tokens: list[str]) -> float:
    """
    Shannon entropy of the token distribution, divided by log2(V) so the
    result lives in [0, 1]. Returns 0 for empty / single-token text.
    """
    if len(tokens) <= 1:
        return 0.0
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    total = len(tokens)
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log2(p)
    vocab = len(counts)
    if vocab <= 1:
        return 0.0
    return entropy / math.log2(vocab)


def _semantic_variance(sentences: list[str]) -> float:
    """
    1 - mean pairwise cosine similarity of sentence embeddings. High variance
    means sentences cover distinct ideas (information-dense); low variance
    means the prompt is repetitive.

    Returns 0.0 when there are fewer than two sentences — too little signal
    to score, so we don't penalise short prompts on this axis.
    """
    if len(sentences) < 2:
        return 0.0
    model = _get_model()
    embeddings = model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    normed = embeddings / norms
    sim = normed @ normed.T
    iu = np.triu_indices_from(sim, k=1)
    if iu[0].size == 0:
        return 0.0
    mean_sim = float(sim[iu].mean())
    return max(0.0, min(1.0, 1.0 - mean_sim))


def _tfidf_rareness_bonus(prompt_text: str, reference_corpus: str) -> float:
    """
    Optional bonus in [0, 0.1]: fraction of prompt tokens that are rare in
    the reference corpus, where rareness = log(1 + N / df). Bounded so it
    doesn't dominate the three core components.
    """
    prompt_tokens = _tokenise(prompt_text)
    if not prompt_tokens:
        return 0.0
    # Treat each sentence in the reference corpus as one "document".
    ref_sentences = _split_sentences(reference_corpus) or [reference_corpus]
    n_docs = len(ref_sentences)
    df: dict[str, int] = {}
    for s in ref_sentences:
        for t in set(_tokenise(s)):
            df[t] = df.get(t, 0) + 1
    rare_score = 0.0
    for t in set(prompt_tokens):
        rare_score += math.log(1 + n_docs / (df.get(t, 0) + 1))
    # Normalise by vocab size of the prompt; squash into [0, 0.1].
    rare_score /= max(1, len(set(prompt_tokens)))
    return float(min(0.1, rare_score / 10.0))


def compute_isr(prompt_text: str, reference_corpus: Optional[str] = None) -> float:
    """
    Information Sufficiency Ratio in [0, 1].

    Args:
        prompt_text:       Prompt to score.
        reference_corpus:  Optional text used to weight rare terms via TF-IDF.
                           When supplied, contributes a small bounded bonus
                           (≤ 0.1) on top of the core three components.

    Returns:
        Float in [0, 1]. Higher = closer to MDL = leave alone.
    """
    if not prompt_text or not prompt_text.strip():
        return 0.0

    tokens = _tokenise(prompt_text)
    sentences = _split_sentences(prompt_text)

    lex = _lexical_diversity(tokens)
    ent = _normalised_shannon_entropy(tokens)
    sem = _semantic_variance(sentences)

    isr = _W_LEXICAL * lex + _W_ENTROPY * ent + _W_SEMANTIC * sem

    if reference_corpus:
        isr += _tfidf_rareness_bonus(prompt_text, reference_corpus)

    return float(np.clip(isr, 0.0, 1.0))


class ISRGate:
    """
    Pre-compression gate driven by the Information Sufficiency Ratio.

    The gate produces one of three decisions:
        - skip       (isr > high_threshold): prompt is near MDL.
        - aggressive (isr < low_threshold):  prompt is highly redundant.
        - moderate   (otherwise):            standard compression.

    The thresholds are exposed so different deployments can trade off
    safety vs. compression yield.
    """

    def __init__(self, high_threshold: float = 0.85, low_threshold: float = 0.40):
        if not 0.0 <= low_threshold <= high_threshold <= 1.0:
            raise ValueError(
                f"Thresholds must satisfy 0 <= low ({low_threshold}) "
                f"<= high ({high_threshold}) <= 1"
            )
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

    def should_compress(self, prompt_text: str) -> Tuple[bool, float, str]:
        """
        Args:
            prompt_text: Prompt to score.

        Returns:
            (should_compress, isr_score, reason)
        """
        isr = compute_isr(prompt_text)
        if isr > self.high_threshold:
            return False, isr, "skip: information-dense near MDL"
        if isr < self.low_threshold:
            return True, isr, "aggressive: high redundancy"
        return True, isr, "moderate: standard compression with safety bounds"


if __name__ == "__main__":
    # Two contrasting test prompts: one dense, one highly redundant.
    dense_prompt = (
        "You are an expert astrologer with deep knowledge of natal charts, "
        "transits, and synastry. Given a person's birth date, time, and "
        "location, compute the positions of the Sun, Moon, ascendant, and "
        "every planet across the twelve houses. Interpret each placement in "
        "terms of psychological archetype, life domain, and contemporary "
        "predictive context. Identify major aspects (conjunction, opposition, "
        "trine, square, sextile) and explain their developmental tension. "
        "Cross-reference current transits to surface timing-sensitive themes. "
        "Conclude with three concrete behavioural suggestions tailored to "
        "the chart's dominant element and modality."
    )

    redundant_prompt = (
        "Please help the user. Please help the user. Please help the user. "
        "Please help the user. Please help the user. Please help the user. "
        "Help the user please. Help the user please. Help the user please. "
        "The user needs help. The user needs help. The user needs help. "
        "Be helpful to the user. Be helpful to the user. Be helpful to the user."
    )

    gate = ISRGate(high_threshold=0.85, low_threshold=0.40)

    print("Testing dense prompt (Astrologer):")
    should, isr, reason = gate.should_compress(dense_prompt)
    band = "high" if isr > gate.high_threshold else ("low" if isr < gate.low_threshold else "moderate")
    print(f"  ISR = {isr:.2f} ({band})")
    print(f"  Decision: {'compress' if should else 'skip compression'} ({reason})")
    print()

    print("Testing redundant prompt (Prompt Generator):")
    should, isr, reason = gate.should_compress(redundant_prompt)
    band = "high" if isr > gate.high_threshold else ("low" if isr < gate.low_threshold else "moderate")
    print(f"  ISR = {isr:.2f} ({band})")
    print(f"  Decision: {'compress' if should else 'skip compression'} ({reason})")
