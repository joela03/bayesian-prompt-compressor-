"""
prompt-compress — structural prompt compression with safety gating.

Public API:
    PromptCompressor       — the compressor
    CompressionResult      — what compress() returns
    CompressionFailedError — raised when on_failure='raise' and validation fails
    OptimisationConfig     — knobs for the Bayesian optimiser

Internal classes (parser, validator, evaluator, etc.) are importable from
their submodules for advanced use but are intentionally not promoted here.
"""

from .compressor import CompressionFailedError, PromptCompressor
from .optimiser import OptimisationConfig
from .result import CompressionResult

__version__ = '0.1.0'

__all__ = [
    'PromptCompressor',
    'CompressionFailedError',
    'OptimisationConfig',
    'CompressionResult',
    '__version__',
]
