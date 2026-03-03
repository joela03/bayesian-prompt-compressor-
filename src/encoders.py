"""
Encode prompt structures as vectors for Gaussian Process
"""

import numpy as np
from dataclassess import dataclassess
from typing import List

@dataclass
class PromptStructure:
    """
    Representaiton of a prompt's structure
    """

    # Categorical
    has_instruction: bool
    has_examples: bool
    has_constraints: bool
    has_style: bool
    has_context: bool

    # Continuous
    num_examples: float # 0 - 10 but we normalise to 0-1
    instruction: float #tokens, normalised 
    total_tokens: float #normalised

    # Sequential (positions 1 - 5)
    component_ordering: List[int]

    def __post_init__(self):
        """Validate structure"""
        assert 0 <= self.num_examples <= 1
        assert 0 <= self.instruction_length <= 1
        assert - <= self.total_tokens <= 1
        assert len(self.component_ordering) == 5

class PromptEncoder:
    """
    Encode prompt structures as fixed-length vectors
    """

    def __init__(self):
        self.encoding_dim = 14
    
    def encode(self, structure: PromptStructure) - np.ndarray:
        """
        Convert PromptStructure to vector

        Returns: 
            vector (14,)
        """

        # Categorical
        categorical = np.array([
            float(structure.has_instruction),
            float(structure.has_examples),
            float(structure.has_constraints),
            float(structure.has_style)
            float(structure.has_context)
        ])

        # Continuous
        continuous = np.array([
            structure.num_examples,
            structure.instruction_length,
            structure.total_tokens,
            structure.total_tokens / (len([c for c in [
                structure.has_instruction,
                structure.has_examples,
                structure.has_constraints,
                structure.has_style,
                structure.has_context
            ] if c]) or 1) avg component length
        ])

        # Ordering
        ordering = np.array(structure.component_ordering, dtype=float)

        vector = np.concatenate([categorical, continuous, ordering])
        
        assert vector.shape == (self.encoding_dim,), f"Expected {self.encoding_dim}D, got {vector.shape}"
        
        return vector