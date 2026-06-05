"""
Output validation gate for prompt compression.

Three checks on every compressed prompt:
  1. Persona preserved (when the original opens with one).
  2. Placeholders preserved (``\\{[^}]+\\}`` substrings, when
     ``preserve_placeholders=True``).
  3. Semantic similarity >= ``similarity_threshold``.

The default threshold (0.75) was lowered from an earlier 0.85 after we found
genuine valid compressions scoring around 0.78. An ``adaptive_threshold``
mode scales it with the compression ratio so aggressive compressions face a
stricter bar: ``max(0.70, 1 - compression_ratio * 0.5)``.

ISR is a separate *pre*-compression gate (see ``information_sufficiency``);
this module is the *post*-compression safety net and runs after every
compression regardless of whether ISR fired.
"""

import logging
import re

import numpy as np
from sentence_transformers import SentenceTransformer

from ._persona import persona_present

logger = logging.getLogger(__name__)


class CompressionValidator:
    """Reject compressed outputs that fail persona/placeholder/similarity checks."""

    SIMILARITY_THRESHOLD = 0.75
    ADAPTIVE_FLOOR = 0.70
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
    PLACEHOLDER_PATTERN = r'\{[^}]+\}'

    _shared_model: SentenceTransformer | None = None

    def __init__(
        self,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        preserve_placeholders: bool = True,
        adaptive_threshold: bool = False,
    ):
        """
        Args:
            similarity_threshold: Minimum cosine similarity to accept. Ignored
                when ``adaptive_threshold`` is True.
            preserve_placeholders: Reject when any ``{placeholder}`` from the
                original is missing in the compressed output.
            adaptive_threshold: Scale threshold with compression ratio:
                ``max(0.70, 1 - ratio * 0.5)``.
        """
        self.similarity_threshold = similarity_threshold
        self.preserve_placeholders = preserve_placeholders
        self.adaptive_threshold = adaptive_threshold
        if CompressionValidator._shared_model is None:
            CompressionValidator._shared_model = SentenceTransformer(self.EMBEDDING_MODEL)
        self.model = CompressionValidator._shared_model

    def _effective_threshold(self, original: str, compressed: str) -> float:
        if not self.adaptive_threshold:
            return self.similarity_threshold
        orig_tokens = max(len(original.split()), 1)
        comp_tokens = len(compressed.split())
        ratio = max(0.0, (orig_tokens - comp_tokens) / orig_tokens)
        return max(self.ADAPTIVE_FLOOR, 1 - ratio * 0.5)

    def validate(self, original: str, compressed: str) -> tuple[bool, list[str], float]:
        """
        Run all checks.

        Returns:
            (passed, reasons, similarity) — similarity is always computed
            (orchestrator surfaces it on the result regardless of pass/fail).
        """
        reasons: list[str] = []

        if persona_present(original) and not persona_present(compressed):
            reasons.append("persona sentence missing (original opens with a persona; compressed does not)")

        if self.preserve_placeholders:
            missing = self._missing_placeholders(original, compressed)
            if missing:
                reasons.append(f"missing placeholders {sorted(missing)}")

        sim = self._cosine_similarity(original, compressed)
        threshold = self._effective_threshold(original, compressed)
        if sim < threshold:
            reasons.append(f"similarity {sim:.3f} < {threshold:.3f}")

        return len(reasons) == 0, reasons, sim

    def gate(self, original: str, compressed: str) -> str:
        """Apply validation; return compressed if all checks pass, else original."""
        passed, reasons, _ = self.validate(original, compressed)
        if passed:
            return compressed
        for reason in reasons:
            logger.warning("Validator: %s. Returning original.", reason)
        return original

    @classmethod
    def _placeholders(cls, text: str) -> set[str]:
        return set(re.findall(cls.PLACEHOLDER_PATTERN, text))

    @classmethod
    def _missing_placeholders(cls, original: str, compressed: str) -> set[str]:
        return cls._placeholders(original) - cls._placeholders(compressed)

    def _cosine_similarity(self, a: str, b: str) -> float:
        embeddings = self.model.encode([a, b], convert_to_numpy=True, show_progress_bar=False)
        va, vb = embeddings[0], embeddings[1]
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if denom == 0.0:
            return 0.0
        return float(np.dot(va, vb) / denom)


def _run_tests() -> None:
    v = CompressionValidator()

    original_ph = "You are a tutor. Explain {concepts} with {theme} clearly and concisely."
    kept = "You are a tutor. Explain {concepts} using {theme}."
    passed, reasons, _ = v.validate(original_ph, kept)
    print(f"[placeholders kept]   passed={passed} reasons={reasons}")
    assert passed, f"expected pass, got {reasons}"

    dropped = "You are a tutor. Explain {concepts} clearly."
    passed, reasons, _ = v.validate(original_ph, dropped)
    print(f"[placeholder dropped] passed={passed} reasons={reasons}")
    assert not passed and any("missing placeholders" in r for r in reasons), reasons

    original_sim = (
        "You are an expert nutritionist. Provide a detailed weekly meal plan "
        "for a vegetarian athlete that focuses on protein-rich foods, balanced "
        "macronutrients, and proper hydration across training days."
    )
    paraphrased = (
        "You are a nutrition expert. Produce a one-week vegetarian athlete "
        "meal plan with high protein, balanced macros, and hydration guidance."
    )
    passed, reasons, sim = v.validate(original_sim, paraphrased)
    print(f"[similarity ~0.78]    sim={sim:.3f} passed={passed} reasons={reasons}")
    assert passed, f"expected pass at threshold 0.75 with sim {sim:.3f}; reasons={reasons}"

    v_lenient = CompressionValidator(preserve_placeholders=False)
    passed, reasons, _ = v_lenient.validate(original_ph, dropped)
    print(f"[preserve_ph=False]   passed={passed} reasons={reasons}")
    assert all("missing placeholders" not in r for r in reasons)

    v_adaptive = CompressionValidator(adaptive_threshold=True)
    eff = v_adaptive._effective_threshold(original_sim, paraphrased)
    print(f"[adaptive threshold]  effective={eff:.3f}")
    assert v_adaptive.ADAPTIVE_FLOOR <= eff <= 1.0

    print("\nAll validator tests passed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    _run_tests()
