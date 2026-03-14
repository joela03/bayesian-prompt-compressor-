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
import json
from pathlib import Path


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

    def _load_p3_weights(self):
        """
        Load feature importance from P3 analysis
        """
        findings_path = Path('data/results/p3_findings_adjusted.json')
        
        if findings_path.exists():
            with open(findings_path, 'r') as f:
                findings = json.load(f)
            
            # Extract component importance
            self.feature_weights = {
                'has_instruction': findings.get('instruction_importance', 0.95),
                'has_examples': findings.get('examples_importance', 0.68),
                'has_constraints': findings.get('constraints_importance', 0.39),
                'has_style': findings.get('style_importance', 0.09),
                'has_context': findings.get('context_importance', 0.06),
            }
            
            print(f"Loaded P3 feature weights:")
            for feat, weight in self.feature_weights.items():
                print(f"   {feat}: {weight:.2f}")
        else:
            # Default uniform weights
            self.feature_weights = {
                'has_instruction': 1.0,
                'has_examples': 1.0,
                'has_constraints': 1.0,
                'has_style': 1.0,
                'has_context': 1.0,
            }
            print("P3 findings not found, using uniform weights")