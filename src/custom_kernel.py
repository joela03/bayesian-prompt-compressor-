"""
Custom compositional kernel for prompt structure optimization

Combines three kernel types:
1. Hamming kernel for categorical features (has_instruction, has_examples, etc.)
2. RBF kernel for continuous features (num_examples, instruction_length, total_tokens)
3. Kendall-Tau kernel for ordering (component_ordering)

Weighted by P3-derived feature importance
"""

import numpy as np
from sklearn.gaussian_process.kernels import Kernel, StationaryKernelMixin, NormalizedKernelMixin
from scipy.stats import kendalltau


class PromptStructureKernel(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """
    Custom kernel for prompt structure optimization
    
    Combines three similarity measures weighted by empirical importance
    """
    
    def __init__(self, 
                 categorical_weight=0.4,
                 continuous_weight=0.4,
                 ordering_weight=0.2,
                 length_scale=1.0,
                 use_p3_weights=True):
        """
        Args:
            categorical_weight: Weight for Hamming kernel (binary features)
            continuous_weight: Weight for RBF kernel (continuous features)
            ordering_weight: Weight for Kendall-Tau kernel (ordering)
            length_scale: RBF kernel length scale
            use_p3_weights: If True, load feature weights from P3 analysis
        """
        self.categorical_weight = categorical_weight
        self.continuous_weight = continuous_weight
        self.ordering_weight = ordering_weight
        self.length_scale = length_scale
        self.use_p3_weights = use_p3_weights
        
        # Load P3-derived feature importance if available
        if use_p3_weights:
            self._load_p3_weights()