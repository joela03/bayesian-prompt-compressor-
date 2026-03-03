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

