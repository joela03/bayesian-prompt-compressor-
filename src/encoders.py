"""
Encode prompt structures as vectors for Gaussian Process
"""

import numpy as np
from dataclasses import dataclass
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
    instruction_length: float #tokens, normalised 
    total_tokens: float #normalised

    # Sequential (positions 1 - 5)
    component_ordering: List[int]

    def __post_init__(self):
        """Validate structure"""
        assert 0 <= self.num_examples <= 1
        assert 0 <= self.instruction_length <= 1
        assert 0 <= self.total_tokens <= 1
        assert len(self.component_ordering) == 5

class PromptEncoder:
    """
    Encode prompt structures as fixed-length vectors
    """

    def __init__(self):
        self.encoding_dim = 14
    
    def encode(self, structure: PromptStructure) -> np.ndarray:
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
            float(structure.has_style),
            float(structure.has_context),
        ])

        # Continuous
        num_components = sum([
            structure.has_instruction,
            structure.has_examples,
            structure.has_constraints,
            structure.has_style,
            structure.has_context
        ])

        avg_component_len = structure.total_tokens / num_components if num_components > 0 else 0.0

        continuous = np.array([
            structure.num_examples,
            structure.instruction_length,
            structure.total_tokens,
            avg_component_len
        ])

        # Ordering
        ordering = np.array(structure.component_ordering, dtype=float)

        vector = np.concatenate([categorical, continuous, ordering])
        
        assert vector.shape == (self.encoding_dim,), f"Expected {self.encoding_dim}D, got {vector.shape}"
        
        return vector

    def decode_partial(self, vector: np.ndarray) -> Dict:
        """
        Shows what a vector represents
        """
        categorical = vector[:5]
        continuous = vector[5:9]
        ordering = vector[9:14]
        
        return {
            'categorical': {
                'has_instruction': bool(categorical[0]),
                'has_examples': bool(categorical[1]),
                'has_constraints': bool(categorical[2]),
                'has_style': bool(categorical[3]),
                'has_context': bool(categorical[4])
            },
            'continuous': {
                'num_examples': continuous[0],
                'instruction_length': continuous[1],
                'total_tokens': continuous[2],
                'avg_component_len': continuous[3]
            },
            'ordering': ordering.tolist()
        }


def create_test_structure() -> PromptStructure:
    """
    Helper: create a simple test structure
    """
    return PromptStructure(
        has_instruction=True,
        has_examples=True,
        has_constraints=False,
        has_style=True,
        has_context=False,
        num_examples=0.3,  
        instruction_length=0.254, 
        total_tokens=0.487,
        component_ordering=[1, 2, 4, 3, 5]
    )

if __name__ == "__main__":
    encoder = PromptEncoder()
    test_structure = create_test_structure()
    
    # Encode
    vector = encoder.encode(test_structure)
    print(f"Encoded vector shape: {vector.shape}")
    print(f"Vector: {vector}")
    
    # Decode
    decoded = encoder.decode_partial(vector)
    print(f"\nDecoded:")
    for key, value in decoded.items():
        print(f"  {key}: {value}")
    
    print("True")