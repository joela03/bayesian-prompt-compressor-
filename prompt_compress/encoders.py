"""Encode prompt structures as fixed-length vectors for the GP optimiser."""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class PromptStructure:
    """A prompt decomposed into the search space the optimiser explores."""

    has_instruction: bool
    has_examples: bool
    has_constraints: bool
    has_style: bool
    has_context: bool

    num_examples: float          # normalised 0-1
    instruction_length: float    # normalised 0-1 (token fraction)
    total_tokens: float          # normalised 0-1

    component_ordering: List[int]  # permutation of 1..5

    def __post_init__(self):
        # Clip so the GP never sees out-of-range candidates. The 0.1 floor on
        # instruction_length prevents the builder from producing an empty
        # instruction section.
        self.instruction_length = np.clip(self.instruction_length, 0.1, 1.0)
        self.num_examples = np.clip(self.num_examples, 0.0, 1.0)
        self.total_tokens = np.clip(self.total_tokens, 0.0, 1.0)


class PromptEncoder:
    """Convert PromptStructure to/from a 14-dimensional vector."""

    def __init__(self):
        self.encoding_dim = 14

    def encode(self, structure: PromptStructure) -> np.ndarray:
        categorical = np.array([
            float(structure.has_instruction),
            float(structure.has_examples),
            float(structure.has_constraints),
            float(structure.has_style),
            float(structure.has_context),
        ])

        num_components = sum([
            structure.has_instruction,
            structure.has_examples,
            structure.has_constraints,
            structure.has_style,
            structure.has_context,
        ])
        avg_component_len = (
            structure.total_tokens / num_components if num_components > 0 else 0.0
        )

        continuous = np.array([
            structure.num_examples,
            structure.instruction_length,
            structure.total_tokens,
            avg_component_len,
        ])
        ordering = np.array(structure.component_ordering, dtype=float)
        vector = np.concatenate([categorical, continuous, ordering])
        assert vector.shape == (self.encoding_dim,)
        return vector

    def decode_partial(self, vector: np.ndarray) -> Dict:
        categorical = vector[:5]
        continuous = vector[5:9]
        ordering = vector[9:14]
        return {
            'categorical': {
                'has_instruction': bool(categorical[0]),
                'has_examples': bool(categorical[1]),
                'has_constraints': bool(categorical[2]),
                'has_style': bool(categorical[3]),
                'has_context': bool(categorical[4]),
            },
            'continuous': {
                'num_examples': continuous[0],
                'instruction_length': continuous[1],
                'total_tokens': continuous[2],
                'avg_component_len': continuous[3],
            },
            'ordering': ordering.tolist(),
        }


def create_test_structure() -> PromptStructure:
    """Helper used by tests/demos to construct a known-shape structure."""
    return PromptStructure(
        has_instruction=True,
        has_examples=True,
        has_constraints=False,
        has_style=True,
        has_context=False,
        num_examples=0.3,
        instruction_length=0.254,
        total_tokens=0.487,
        component_ordering=[1, 2, 4, 3, 5],
    )
