"""End-to-end prompt compression pipeline + public `PromptCompressor` API."""

from __future__ import annotations

import logging
import time
from typing import Literal, Optional, Union

from .attention_optimiser import AttentionInformedOptimiser
from .encoders import PromptEncoder
from .evaluators import MockEvaluator, RealEvaluator, SemanticEvaluator
from .informed_optimiser import InformedBayesianOptimiser
from .optimiser import BayesianPromptOptimiser, OptimisationConfig
from .parser import PromptBuilder, PromptParser
from .result import CompressionResult
from .semantic_compressor import TextRankCompressor, compute_density
from .validators import CompressionValidator


logger = logging.getLogger(__name__)


# Long reason strings → short tags used on CompressionResult.validator_failures.
_REASON_TO_TAG = (
    ('persona', 'persona'),
    ('missing placeholders', 'placeholder'),
    ('similarity', 'similarity'),
)


def _tag_failures(reasons: list[str]) -> list[str]:
    tags: list[str] = []
    for reason in reasons:
        for needle, tag in _REASON_TO_TAG:
            if needle in reason and tag not in tags:
                tags.append(tag)
                break
    return tags


class CompressionFailedError(Exception):
    """
    Raised when ``on_failure='raise'`` and the validator rejects the
    compressed output. Inspect ``.failures`` for the failing checks
    (subset of {'persona', 'placeholder', 'similarity'}) and ``.reasons``
    for the full human-readable explanations.
    """

    def __init__(self, failures: list[str], reasons: list[str]):
        self.failures = failures
        self.reasons = reasons
        super().__init__("Compression rejected: " + "; ".join(reasons))


class PromptCompressor:
    """
    End-to-end structural prompt compressor.

    Routes by density:
      tier 1 (density < 0.5):  Bayesian optimisation over structure space
      tier 2 (0.5–0.85):       TextRank extractive sentence selection
      tier 3 (>= 0.85):        pass-through (already information-dense)

    The post-compression validator gates every result; on failure the
    behaviour is controlled by ``on_failure`` (see ``compress``).
    """

    # alpha="auto" resolves to this. 0.3 is the validated benchmark default;
    # see SemanticEvaluator.AUTO_ALPHA for rationale.
    AUTO_ALPHA = 0.3

    def __init__(
        self,
        use_real_evaluator: bool = False,
        use_mock_evaluator: bool = False,
        use_informed_prior: bool = False,
        use_attention_prior: bool = False,
        optimisation_config: Optional[OptimisationConfig] = None,
        alpha: Union[float, str] = "auto",
    ):
        """
        Args:
            use_real_evaluator: Use a live LLM evaluator (costs API spend).
            use_mock_evaluator: Use the legacy structural MockEvaluator.
            use_informed_prior: Use P3-derived prior for BO.
            use_attention_prior: Use per-prompt attention prior + ISR gate.
            optimisation_config: Knobs for the BO loop.
            alpha:              Quality/compression trade-off. "auto" →
                                AUTO_ALPHA (0.3). Larger penalises length more,
                                but >> 0.3 pushes BO toward structures the
                                validator rejects.
        """
        self.alpha = self._resolve_alpha(alpha)
        self.parser = PromptParser()
        self.builder = PromptBuilder()
        self.encoder = PromptEncoder()
        self.textrank = TextRankCompressor()

        # Evaluator: real > mock > semantic (default). SemanticEvaluator is
        # built per-compress() because it binds to the prompt being scored.
        self.use_real_evaluator = use_real_evaluator
        self.use_mock_evaluator = use_mock_evaluator

        if use_real_evaluator:
            self.evaluator: object | None = RealEvaluator()
        elif use_mock_evaluator:
            self.evaluator = MockEvaluator()
        else:
            self.evaluator = None

        # Optimiser instantiated per-compress() so BO state doesn't leak
        # across prompts.
        self.use_informed_prior = use_informed_prior
        self.use_attention_prior = use_attention_prior
        self.optimisation_config = optimisation_config

    @classmethod
    def _resolve_alpha(cls, alpha) -> float:
        if isinstance(alpha, str):
            if alpha == "auto":
                return cls.AUTO_ALPHA
            raise ValueError(f"alpha must be 'auto' or numeric, got {alpha!r}")
        return float(alpha)

    def compress(
        self,
        prompt: str,
        min_similarity: float = 0.75,
        on_failure: Literal['fallback', 'raise', 'warn'] = 'fallback',
    ) -> CompressionResult:
        """
        Compress a prompt and return a :class:`CompressionResult`.

        Args:
            prompt:         The system prompt to compress.
            min_similarity: Per-call override of the validator's similarity
                            threshold. Useful when different applications want
                            different safety bars.
            on_failure:     What to do when validation rejects the compression.
                            * 'fallback' (default) — silently return a result
                              whose ``compressed_text`` equals ``original_text``;
                              ``gate_passed`` will be False.
                            * 'raise' — raise :class:`CompressionFailedError`.
                            * 'warn'  — log a warning and return the fallback.
        """
        if on_failure not in ('fallback', 'raise', 'warn'):
            raise ValueError(
                f"on_failure must be 'fallback', 'raise', or 'warn'; got {on_failure!r}"
            )

        # Validator built per-call so per-call min_similarity is honoured
        # without mutating shared state.
        validator = CompressionValidator(similarity_threshold=min_similarity)

        t_start = time.time()
        prompt_text = prompt

        original_structure, components = self.parser.parse(prompt_text)
        density_info = compute_density(prompt_text)
        density = density_info['density']
        original_tokens = len(prompt_text.split())

        # Tier 3 — pass-through for information-dense prompts.
        if density >= 0.85:
            return self._build_passthrough_result(
                prompt_text, original_structure, components, validator,
                density, tier=3, t_start=t_start,
            )

        # Tier 2 — TextRank extractive selection.
        if density >= 0.50:
            try:
                compressed_text = self.textrank.compress(prompt_text)
            except Exception:
                # TextRank can fail to converge; fall through to Tier 1.
                logger.debug("TextRank failed; falling back to Tier 1", exc_info=True)
            else:
                protected = components.get('__protected__', {})
                if protected:
                    compressed_text = PromptParser.restore_protected_regions(
                        compressed_text, protected
                    )
                return self._gate_and_build(
                    prompt_text, compressed_text, original_structure,
                    optimised_structure=original_structure,
                    components=components, validator=validator,
                    density=density, tier=2, t_start=t_start,
                    on_failure=on_failure,
                    best_score=None, n_evaluations=None,
                )

        # Tier 1 — Bayesian optimisation.
        if not self.use_real_evaluator and not self.use_mock_evaluator:
            self.evaluator = SemanticEvaluator(
                prompt_text, self.builder, components, alpha=self.alpha,
            )

        config = self.optimisation_config or OptimisationConfig(
            n_iterations=20, n_init=5, beta=2.0, random_seed=42,
        )
        if self.use_attention_prior:
            optimiser = AttentionInformedOptimiser(
                self.encoder, self.evaluator, config,
                components=components, prompt_text=prompt_text, alpha=self.alpha,
            )
        elif self.use_informed_prior:
            optimiser = InformedBayesianOptimiser(self.encoder, self.evaluator, config)
        else:
            optimiser = BayesianPromptOptimiser(self.encoder, self.evaluator, config)

        opt_result = optimiser.optimise()

        # ISR gate short-circuit: identity result, no further compression.
        if getattr(opt_result, 'skipped', False):
            return self._build_passthrough_result(
                prompt_text, original_structure, components, validator,
                density, tier=3, t_start=t_start,
            )

        compressed_text = self.builder.build(opt_result.best_structure, components)
        return self._gate_and_build(
            prompt_text, compressed_text, original_structure,
            optimised_structure=opt_result.best_structure,
            components=components, validator=validator,
            density=density, tier=1, t_start=t_start,
            on_failure=on_failure,
            best_score=opt_result.best_score,
            n_evaluations=opt_result.total_evaluations,
        )

    def _gate_and_build(
        self, original_text, compressed_text, original_structure,
        optimised_structure, components, validator, density, tier, t_start,
        on_failure, best_score, n_evaluations,
    ) -> CompressionResult:
        """Run the validator, apply on_failure policy, and pack the result."""
        passed, reasons, similarity = validator.validate(original_text, compressed_text)
        failures = [] if passed else _tag_failures(reasons)

        if passed:
            final_text = compressed_text
            gate_passed = True
        else:
            if on_failure == 'raise':
                raise CompressionFailedError(failures, reasons)
            if on_failure == 'warn':
                for reason in reasons:
                    logger.warning("Compression rejected: %s", reason)
            final_text = original_text
            gate_passed = False

        return CompressionResult(
            original_text=original_text,
            compressed_text=final_text,
            original_tokens=len(original_text.split()),
            compressed_tokens=len(final_text.split()),
            semantic_similarity=similarity,
            gate_passed=gate_passed,
            validator_failures=failures,
            density=density,
            tier=tier,
            best_score=best_score,
            n_evaluations=n_evaluations,
            alpha=self.alpha if tier == 1 else None,
            components_original=_structure_to_flags(original_structure),
            components_compressed=_structure_to_flags(optimised_structure),
            components_text={},  # see CompressionResult docstring — flag-only
            time_seconds=time.time() - t_start,
        )

    def _build_passthrough_result(
        self, prompt_text, structure, components, validator,
        density, tier, t_start,
    ) -> CompressionResult:
        """Tier 3 / ISR-skip result: compressed == original, validator still runs."""
        passed, _reasons, similarity = validator.validate(prompt_text, prompt_text)
        return CompressionResult(
            original_text=prompt_text,
            compressed_text=prompt_text,
            original_tokens=len(prompt_text.split()),
            compressed_tokens=len(prompt_text.split()),
            semantic_similarity=similarity,
            gate_passed=passed,
            validator_failures=[],
            density=density,
            tier=tier,
            best_score=None,
            n_evaluations=None,
            alpha=None,
            components_original=_structure_to_flags(structure),
            components_compressed=_structure_to_flags(structure),
            components_text={},
            time_seconds=time.time() - t_start,
        )


def _structure_to_flags(structure) -> dict:
    """Flatten a PromptStructure to a flag dict for CompressionResult."""
    if structure is None:
        return {}
    return {
        'has_instruction': bool(structure.has_instruction),
        'has_examples': bool(structure.has_examples),
        'has_constraints': bool(structure.has_constraints),
        'has_style': bool(structure.has_style),
        'has_context': bool(structure.has_context),
        'num_examples': float(structure.num_examples),
        'instruction_length': float(structure.instruction_length),
        'total_tokens': float(structure.total_tokens),
        'component_ordering': list(structure.component_ordering),
    }
