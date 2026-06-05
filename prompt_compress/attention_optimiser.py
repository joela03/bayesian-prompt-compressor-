"""
Attention-informed Bayesian optimiser.

Subclass of InformedBayesianOptimiser that swaps the static P3-JSON prior
for a per-prompt prior generated from component-to-component attention,
and adds an Information Sufficiency Ratio (ISR) pre-compression gate so
prompts already near their Minimum Description Length are not compressed.
"""

import logging
from typing import Dict, Optional

from .attention_priors import AttentionPriorGenerator
from .encoders import PromptStructure
from .information_sufficiency import ISRGate
from .informed_optimiser import InformedBayesianOptimiser
from .optimiser import OptimisationConfig, OptimisationResult

logger = logging.getLogger(__name__)


class AttentionInformedOptimiser(InformedBayesianOptimiser):
    """
    Use a fresh attention-derived prior for each prompt instead of a static
    dataset-wide prior, with an ISR gate that can short-circuit BO when the
    prompt is information-dense (near MDL) or flag aggressive compression
    when the prompt is highly redundant.
    """

    AGGRESSIVE_BETA_MULTIPLIER = 1.2  # widen UCB exploration when ISR is low
    AUTO_ALPHA = 0.3                  # default for alpha="auto"; matches the
                                      # validated benchmark (see SemanticEvaluator.AUTO_ALPHA)

    def __init__(
        self,
        encoder,
        evaluator,
        config: Optional[OptimisationConfig] = None,
        components: Optional[Dict] = None,
        prior_generator: Optional[AttentionPriorGenerator] = None,
        prompt_text: Optional[str] = None,
        use_isr_gate: bool = True,
        isr_high_threshold: float = 0.85,
        isr_low_threshold: float = 0.40,
        alpha=AUTO_ALPHA,
    ):
        """
        Args:
            components:         Parsed components dict (PromptParser output) for
                                the current prompt. Required to generate the prior.
            prior_generator:    Optional shared AttentionPriorGenerator instance.
            prompt_text:        Raw prompt text. Required for the ISR gate to
                                fire; if None, the gate is silently skipped.
            use_isr_gate:       Master switch for the ISR pre-check.
            isr_high_threshold: ISR above this ⇒ skip compression entirely.
            isr_low_threshold:  ISR below this ⇒ enable aggressive mode (the
                                UCB beta is multiplied by AGGRESSIVE_BETA_MULTIPLIER
                                to favour exploration of compressive structures).
            alpha:              Quality/compression trade-off. Larger alpha
                                penalises length more heavily. Accepts "auto"
                                (resolves to AUTO_ALPHA=1.0) or any float.
                                The objective lives on SemanticEvaluator —
                                this kwarg is stored for logging and is
                                surfaced on the OptimisationResult.
        """
        self.alpha = self._resolve_alpha(alpha)
        if components is None:
            raise ValueError(
                "AttentionInformedOptimiser requires the parsed components dict"
            )

        self._components = components
        self._prior_generator = prior_generator or AttentionPriorGenerator()
        self._prompt_text = prompt_text

        if config is None:
            config = OptimisationConfig(
                n_iterations=20, n_init=5, beta=2.0, random_seed=42
            )

        # Honour both the explicit kwargs and the OptimisationConfig fields.
        # Explicit kwargs win so callers can override per-instance.
        self.use_isr_gate = use_isr_gate and config.enable_isr_gate
        self.isr_gate = ISRGate(
            high_threshold=isr_high_threshold if isr_high_threshold is not None else config.isr_high,
            low_threshold=isr_low_threshold if isr_low_threshold is not None else config.isr_low,
        )

        # Skip the parent's JSON-loading constructor by initialising the
        # grandparent (BayesianPromptOptimiser) directly.
        from .optimiser import BayesianPromptOptimiser
        BayesianPromptOptimiser.__init__(self, encoder, evaluator, config)

        self.prior = self._prior_generator.generate(components)
        logger.info(
            "Loaded attention-informed prior "
            "(mean_attention=%s)",
            {k: round(v, 2) for k, v in self.prior['mean_attention'].items()},
        )

    @classmethod
    def _resolve_alpha(cls, alpha) -> float:
        if isinstance(alpha, str):
            if alpha == "auto":
                return cls.AUTO_ALPHA
            raise ValueError(f"alpha must be 'auto' or numeric, got {alpha!r}")
        return float(alpha)

    def check_isr(self) -> tuple[bool, float, str]:
        """
        Run the ISR gate against the stored prompt text. Returns the same
        triple as ISRGate.should_compress.

        Returns (True, 0.0, "isr disabled") if the gate is off or no prompt
        text was supplied.
        """
        if not self.use_isr_gate or self._prompt_text is None:
            return True, 0.0, "isr disabled"
        return self.isr_gate.should_compress(self._prompt_text)

    def optimise(self) -> OptimisationResult:
        """
        Run ISR gate first, then dispatch to the parent BO loop.

        Behaviour:
          - ISR > high_threshold: return a `skipped=True` result; the
            orchestrator (PromptCompressor) treats this as "return original".
          - ISR < low_threshold:  temporarily inflate UCB beta to favour
            exploration of compressive structures.
          - Otherwise:           run BO unchanged.
        """
        should_compress, isr, reason = self.check_isr()

        if not should_compress:
            logger.info(
                "ISR=%.2f exceeds threshold. Prompt already near Minimum "
                "Description Length. Preserving original.",
                isr,
            )
            return self._skipped_result(isr, reason)

        if self.use_isr_gate and self._prompt_text is not None:
            band = "low" if isr < self.isr_gate.low_threshold else (
                "high" if isr > self.isr_gate.high_threshold else "moderate"
            )
            logger.info("ISR=%.2f (%s). %s.", isr, band, reason.capitalize())

        # Aggressive mode: temporarily widen UCB exploration. Restore beta
        # after the run so the optimiser stays reusable.
        original_beta = self.config.beta
        if isr < self.isr_gate.low_threshold:
            self.config.beta = original_beta * self.AGGRESSIVE_BETA_MULTIPLIER
            logger.info(
                "ISR=%.2f (low). Enabling aggressive compression "
                "(beta %.2f -> %.2f).",
                isr,
                original_beta,
                self.config.beta,
            )

        try:
            result = super().optimise()
        finally:
            self.config.beta = original_beta

        result.skipped = False
        result.isr_score = isr if self.use_isr_gate and self._prompt_text else None
        result.isr_reason = reason if self.use_isr_gate and self._prompt_text else None
        result.alpha_used = self.alpha
        return result

    def _skipped_result(self, isr: float, reason: str) -> OptimisationResult:
        """
        Build a sentinel OptimisationResult signalling "do not compress".

        The structure is set to "keep everything" so that if a caller ignores
        `skipped` and materialises this structure anyway, the output is at
        worst the full original prompt — not a corrupted compression.
        """
        identity_structure = PromptStructure(
            has_instruction=True,
            has_examples=True,
            has_constraints=True,
            has_style=True,
            has_context=True,
            num_examples=1.0,
            instruction_length=1.0,
            total_tokens=1.0,
            component_ordering=[1, 2, 3, 4, 5],
        )
        return OptimisationResult(
            best_structure=identity_structure,
            best_score=1.0,
            all_scores=[1.0],
            all_structures=[identity_structure],
            total_evaluations=0,
            skipped=True,
            isr_score=isr,
            isr_reason=reason,
            alpha_used=self.alpha,
        )
