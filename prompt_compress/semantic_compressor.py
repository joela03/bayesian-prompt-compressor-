"""
TextRank semantic compression for mid-density prompts.

Builds a sentence-similarity graph from sentence-transformers embeddings,
runs PageRank, and keeps the top-N sentences in their original order.
"""

import re
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer


class TextRankCompressor:
    """
    Extractive sentence selection via PageRank over a cosine-similarity graph.
    """

    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

    _shared_model: SentenceTransformer | None = None

    def __init__(self, keep_ratio: float = 0.775):
        """
        Args:
            keep_ratio: Fraction of original sentences to retain (default 0.775,
                        i.e. ~22.5% reduction — within the 75-80% target band).
        """
        self.keep_ratio = keep_ratio
        if TextRankCompressor._shared_model is None:
            TextRankCompressor._shared_model = SentenceTransformer(self.EMBEDDING_MODEL)
        self.model = TextRankCompressor._shared_model

    def compress(self, text: str) -> str:
        """
        Select top-ranked sentences and return them in original order.
        """
        sentences = self._split_sentences(text)
        if len(sentences) <= 2:
            return text  # nothing to gain

        embeddings = self.model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
        sim_matrix = self._cosine_matrix(embeddings)
        np.fill_diagonal(sim_matrix, 0.0)

        graph = nx.from_numpy_array(sim_matrix)
        scores = nx.pagerank(graph)

        n_keep = max(2, int(round(len(sentences) * self.keep_ratio)))
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n_keep]
        keep_indices = sorted(idx for idx, _ in ranked)

        return ' '.join(sentences[i] for i in keep_indices)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        # Protect numbered list markers, then split on sentence-ending punctuation.
        protected = re.sub(r'(\d+)\.\s+([A-Z])', r'\1NUMMARKER \2', text)
        parts = re.split(r'(?<=[.!?])\s+', protected)
        parts = [p.replace('NUMMARKER', '.') for p in parts]
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _cosine_matrix(embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        normalised = embeddings / norms
        return normalised @ normalised.T


def compute_density(text: str, model: SentenceTransformer | None = None) -> dict:
    """
    Information density score for routing decisions.

      density = lexical_diversity * (1 - mean pairwise sentence similarity)

    Higher density => less redundancy => less safe to compress aggressively.
    """
    tokens = re.findall(r"\b\w+\b", text.lower())
    if not tokens:
        return {'lexical_diversity': 0.0, 'sentence_similarity': 0.0, 'density': 0.0}

    lexical_diversity = len(set(tokens)) / len(tokens)

    sentences = TextRankCompressor._split_sentences(text)
    if len(sentences) < 2:
        sentence_similarity = 0.0
    else:
        if model is None:
            if TextRankCompressor._shared_model is None:
                TextRankCompressor._shared_model = SentenceTransformer(
                    TextRankCompressor.EMBEDDING_MODEL
                )
            model = TextRankCompressor._shared_model
        embeddings = model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
        sim = TextRankCompressor._cosine_matrix(embeddings)
        # Mean of upper triangle (off-diagonal) — symmetric matrix, exclude self-similarity.
        iu = np.triu_indices_from(sim, k=1)
        sentence_similarity = float(sim[iu].mean()) if iu[0].size > 0 else 0.0

    density = lexical_diversity * (1.0 - sentence_similarity)
    return {
        'lexical_diversity': lexical_diversity,
        'sentence_similarity': sentence_similarity,
        'density': density,
    }


if __name__ == "__main__":
    sample = (
        "You are a helpful assistant. "
        "Your job is to answer user questions. "
        "Provide clear, concise responses. "
        "Avoid unnecessary preamble. "
        "Cite sources when relevant. "
        "Keep responses under 200 words. "
        "Use bullet points for lists. "
        "Be professional and accurate."
    )
    print(compute_density(sample))
    print()
    print("--- compressed ---")
    print(TextRankCompressor(keep_ratio=0.6).compress(sample))
