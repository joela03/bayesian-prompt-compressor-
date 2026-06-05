"""Evaluators that score a PromptStructure for the optimiser."""

import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from .encoders import PromptStructure, create_test_structure

logger = logging.getLogger(__name__)

# Conditional .env load — silently no-op when no .env file exists so the
# library can be imported in environments that don't use dotenv.
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent / '.env'
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path)
except ImportError:
    pass


class MockEvaluator:
    """Synthetic landscape for testing the BO machinery without LLM calls."""

    def __init__(self, noise_level: float = 0.05):
        self.noise_level = noise_level
        self.call_count = 0

    def evaluate(self, structure: PromptStructure) -> float:
        self.call_count += 1

        score = 0.0

        if structure.has_instruction:
            if 0.5 <= structure.instruction_length <= 0.7:
                score += 0.4
            elif structure.instruction_length < 0.5:
                score += 0.4 * (structure.instruction_length / 0.5)
            else:
                score += 0.4 * (1.0 - (structure.instruction_length - 0.7) / 0.3)
        else:
            return 0.0  # instruction is mandatory in this synthetic landscape

        if structure.has_examples:
            if 0.3 <= structure.num_examples <= 0.5:
                score += 0.2
            else:
                distance = min(abs(structure.num_examples - 0.3),
                               abs(structure.num_examples - 0.5))
                score += 0.2 * (1.0 - distance * 2)
        else:
            score += 0.1

        if structure.has_constraints:
            score += 0.3

        if structure.has_style:
            score += 0.02
        else:
            score += 0.05

        if structure.has_context:
            score += 0.02
        else:
            score += 0.05

        score += np.random.randn() * self.noise_level

        # Compression incentive: keeping all tokens costs ~25% of the score so
        # the BO is forced to explore the quality/compression tradeoff.
        compression_bonus = 1.0 - (0.25 * structure.total_tokens)
        score = score * compression_bonus

        score = np.clip(score, 0.0, 1.0)
        time.sleep(0.01)
        return float(score)

    def get_stats(self) -> Dict:
        return {'total_calls': self.call_count}


class RealEvaluator:
    """Evaluator that calls a live LLM. Costs money — use sparingly."""

    def __init__(self, model: str = "gpt-4o-mini", test_query: str = None):
        self.model = model
        self.test_query = test_query or "What are the key principles of effective communication?"
        self.call_count = 0

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY environment variable")

        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "RealEvaluator requires the optional OpenAI dependency. "
                "Install it with `pip install prompt-compress[openai]`."
            ) from exc

        self.client = openai.OpenAI(api_key=api_key)

    def structure_to_prompt_text(self, structure: PromptStructure) -> str:
        parts = []
        if structure.has_instruction:
            if structure.instruction_length > 0.7:
                parts.append("You are an expert assistant. Provide thorough, comprehensive answers.")
            elif structure.instruction_length > 0.3:
                parts.append("You are a helpful assistant. Provide clear answers.")
            else:
                parts.append("You are an assistant.")
        if structure.has_examples:
            n_examples = int(structure.num_examples * 5)
            if n_examples > 0:
                parts.append("\nFor example, good answers are well-structured and detailed.")
        if structure.has_constraints:
            parts.append("\nConstraints: Keep responses concise and factual.")
        if structure.has_style:
            parts.append("\nStyle: Professional and clear.")
        if structure.has_context:
            parts.append("\nContext: This is for a technical audience.")
        return '\n'.join(parts)

    def evaluate(self, structure: PromptStructure, query: str = None) -> float:
        self.call_count += 1
        prompt_text = self.structure_to_prompt_text(structure)
        metrics = self.evaluate_prompt_text(prompt_text, query)
        if metrics is None:
            return 0.5
        quality = metrics['overall_quality']
        tokens = len(prompt_text.split())
        normalised_tokens = tokens / 100
        return quality / (1 + normalised_tokens * 0.5)

    def evaluate_prompt_text(self, prompt_text: str, query: str = None, reference_answer: str = None):
        if query is None:
            query = self.test_query
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt_text},
                    {"role": "user", "content": query},
                ],
                max_tokens=300,
                temperature=0.7,
            )
            answer = response.choices[0].message.content
            metrics = {
                'answer': answer,
                'answer_length': len(answer.split()),
                'response_time': response.usage.total_tokens,
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
            }
            metrics.update(self.compute_quality_metrics(answer, query, reference_answer))
            return metrics
        except Exception:
            logger.exception("OpenAI evaluation failed")
            return None

    def compute_quality_metrics(self, answer, query, reference):
        metrics = {}

        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        query_coverage = len(query_words & answer_words) / len(query_words)
        metrics['query_coverage'] = query_coverage

        word_count = len(answer.split())
        if 50 <= word_count <= 200:
            conciseness_score = 1.0
        elif word_count < 50:
            conciseness_score = word_count / 50
        else:
            conciseness_score = max(0, 1.0 - (word_count - 200) / 200)
        metrics['conciseness'] = conciseness_score

        structure_score = 0.0
        if answer and answer[0].isupper():
            structure_score += 0.3
        if any(p in answer for p in ['.', '!', '?']):
            structure_score += 0.3
        sentences = answer.split('.')
        if len(sentences) >= 2:
            structure_score += 0.2
        if '\n' in answer:
            structure_score += 0.2
        metrics['structure'] = min(structure_score, 1.0)

        if reference:
            ref_words = set(reference.lower().split())
            overlap = len(ref_words & answer_words) / len(ref_words)
            metrics['reference_similarity'] = overlap

        weights = {
            'query_coverage': 0.3,
            'conciseness': 0.3,
            'structure': 0.2,
            'reference_similarity': 0.2 if reference else 0.0,
        }
        if not reference:
            weights = {k: v / 0.8 for k, v in weights.items() if k != 'reference_similarity'}
        metrics['overall_quality'] = sum(metrics.get(k, 0) * v for k, v in weights.items())
        return metrics

    def get_stats(self) -> Dict:
        return {
            'total_calls': self.call_count,
            'model': self.model,
            'approx_cost': self.call_count * 0.01,
        }


class SemanticEvaluator:
    """
    BO objective = cosine(original, candidate) - alpha * length_ratio.

    Without the length penalty the trivial maximum is "keep everything"
    (similarity ~ 1.0). Larger alpha shifts the optimum toward shorter prompts.
    """

    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

    # alpha="auto" resolves to this. 0.3 is the validated benchmark default —
    # it produced the headline numbers (23% avg compression, 84.2 LLM judge).
    # Higher values (1.0+) push BO toward structures the validator rejects.
    AUTO_ALPHA = 0.3

    _shared_model: Optional[SentenceTransformer] = None

    def __init__(self, prompt_text: str, builder, components: Dict, alpha=AUTO_ALPHA):
        """
        Args:
            prompt_text: Original prompt (the target we measure similarity against).
            builder:     PromptBuilder used to materialise candidate text from a structure.
            components:  Parsed components dict for the original prompt.
            alpha:       "auto" → AUTO_ALPHA (0.3), or a float. Larger penalises length more.
        """
        self.prompt_text = prompt_text
        self.builder = builder
        self.components = components
        self.call_count = 0
        self.alpha = self._resolve_alpha(alpha)

        if SemanticEvaluator._shared_model is None:
            SemanticEvaluator._shared_model = SentenceTransformer(self.EMBEDDING_MODEL)
        self.model = SemanticEvaluator._shared_model

        # Cache the original embedding — invariant across evaluate() calls.
        self._original_embedding = self._embed(prompt_text)

    @classmethod
    def _resolve_alpha(cls, alpha) -> float:
        if isinstance(alpha, str):
            if alpha == "auto":
                return cls.AUTO_ALPHA
            raise ValueError(f"alpha must be 'auto' or numeric, got {alpha!r}")
        return float(alpha)

    def evaluate(self, structure: PromptStructure) -> float:
        self.call_count += 1
        candidate = self.builder.build(structure, self.components)
        if not candidate.strip():
            return 0.0
        candidate_embedding = self._embed(candidate)
        similarity = self._cosine(self._original_embedding, candidate_embedding)

        original_len = len(self.prompt_text.split())
        candidate_len = len(candidate.split())
        length_ratio = candidate_len / original_len if original_len > 0 else 1.0

        score = similarity - self.alpha * length_ratio
        return float(np.clip(score, 0.0, 1.0))

    def get_stats(self) -> Dict:
        return {'total_calls': self.call_count, 'model': self.EMBEDDING_MODEL}

    def _embed(self, text: str) -> np.ndarray:
        return self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)
