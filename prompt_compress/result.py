"""Public result type returned by ``PromptCompressor.compress``."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


_TIER_LABELS = {1: 'Bayesian BO', 2: 'TextRank', 3: 'Preserved'}


@dataclass
class CompressionResult:
    """
    Outcome of a single ``compress()`` call.

    Stored fields are raw measurements only. Anything that can be derived
    from them (compression ratio, efficiency, etc.) is a ``@property`` so
    the source of truth is unambiguous.
    """

    # Inputs / outputs
    original_text: str
    compressed_text: str

    # Raw measurements
    original_tokens: int
    compressed_tokens: int
    semantic_similarity: float

    # Validator output
    gate_passed: bool
    validator_failures: list[str]  # short tags: 'persona', 'placeholder', 'similarity'

    # Routing
    density: float
    tier: int  # 1, 2, 3

    # BO internal state (None when tier != 1)
    best_score: Optional[float] = None
    n_evaluations: Optional[int] = None
    alpha: Optional[float] = None

    # Component breakdown — populated with best_structure flags only (per the
    # spec: components_compressed is the kept-flag map, not per-section text).
    components_original: dict = field(default_factory=dict)
    components_compressed: dict = field(default_factory=dict)
    components_text: dict = field(default_factory=dict)

    time_seconds: float = 0.0

    @property
    def compression_ratio(self) -> float:
        if self.original_tokens <= 0:
            return 0.0
        return (self.original_tokens - self.compressed_tokens) / self.original_tokens

    @property
    def tokens_saved(self) -> int:
        return self.original_tokens - self.compressed_tokens

    @property
    def compression_efficiency(self) -> float:
        """compression_ratio * semantic_similarity. Both axes must be high."""
        return self.compression_ratio * self.semantic_similarity

    @property
    def tier_label(self) -> str:
        return _TIER_LABELS.get(self.tier, f'unknown({self.tier})')

    @property
    def persona_preserved(self) -> bool:
        return 'persona' not in self.validator_failures

    @property
    def placeholders_preserved(self) -> bool:
        return 'placeholder' not in self.validator_failures

    @property
    def safe_to_use(self) -> bool:
        """True when the compressed text passed every validator check."""
        return self.gate_passed

    def summary(self) -> str:
        """One-screen terminal summary."""
        lines = [
            "CompressionResult",
            "─" * 60,
            f"  tier               {self.tier} ({self.tier_label})",
            f"  density            {self.density:.3f}",
            f"  tokens             {self.original_tokens} → {self.compressed_tokens}"
            f"  (saved {self.tokens_saved}, {self.compression_ratio:.1%})",
            f"  semantic_sim       {self.semantic_similarity:.3f}",
            f"  efficiency         {self.compression_efficiency:.3f}"
            "  (compression × sim)",
            f"  safe_to_use        {self.safe_to_use}",
        ]
        if self.validator_failures:
            lines.append(f"  validator_failures {self.validator_failures}")
        if self.tier == 1:
            lines.append(
                f"  BO                 best_score={self.best_score:.3f}, "
                f"evals={self.n_evaluations}, alpha={self.alpha}"
            )
        if self.time_seconds:
            lines.append(f"  time               {self.time_seconds:.2f}s")
        return "\n".join(lines)

    def diff(self) -> str:
        """Side-by-side original vs compressed, line-wrapped to ~40 cols each."""
        import textwrap
        left = textwrap.wrap(self.original_text, width=40) or [""]
        right = textwrap.wrap(self.compressed_text, width=40) or [""]
        rows = max(len(left), len(right))
        left += [""] * (rows - len(left))
        right += [""] * (rows - len(right))
        lines = [
            f"{'ORIGINAL':<42}│ COMPRESSED",
            "─" * 42 + "┼" + "─" * 42,
        ]
        for l, r in zip(left, right):
            lines.append(f"{l:<42}│ {r}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """JSON-serialisable dict. Includes properties for downstream consumers."""
        d = asdict(self)
        d['compression_ratio'] = self.compression_ratio
        d['tokens_saved'] = self.tokens_saved
        d['compression_efficiency'] = self.compression_efficiency
        d['tier_label'] = self.tier_label
        d['persona_preserved'] = self.persona_preserved
        d['placeholders_preserved'] = self.placeholders_preserved
        d['safe_to_use'] = self.safe_to_use
        return d
