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

        def hamming_kernel(self, X1, X2):
        """
        Hamming kernel for categorical features (5D: has_instruction, etc.)
        
        Weighted by P3-derived importance
        
        Args:
            X1, X2: Arrays of shape (n_samples, 14)
                    First 5 dimensions are categorical
        
        Returns:
            Kernel matrix of shape (n1, n2)
        """
        # Extract categorical features (dimensions 0-4)
        cat1 = X1[:, :5]
        cat2 = X2[:, :5]
        
        # Feature names in order
        feature_names = ['has_instruction', 'has_examples', 'has_constraints', 
                        'has_style', 'has_context']
        
        # Get weights for each feature
        weights = np.array([self.feature_weights[name] for name in feature_names])
        
        # Weighted Hamming similarity
        # For each pair, compute weighted agreement
        n1, n2 = cat1.shape[0], cat2.shape[0]
        K = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                # Agreement: 1 if same, 0 if different
                agreement = (cat1[i] == cat2[j]).astype(float)
                # Weighted agreement
                weighted_agreement = agreement * weights
                # Normalise by sum of weights
                K[i, j] = weighted_agreement.sum() / weights.sum()
        
        return K

    def rbf_kernel(self, X1, X2):
        """
        RBF kernel for continuous features (4D: num_examples, instruction_length, 
        total_tokens, mean_example_length)
        
        Args:
            X1, X2: Arrays of shape (n_samples, 14)
                    Dimensions 5-8 are continuous
        
        Returns:
            Kernel matrix of shape (n1, n2)
        """
        # Extract continuous features (dimensions 5-8)
        cont1 = X1[:, 5:9]
        cont2 = X2[:, 5:9]
        
        # Standard RBF: k(x, x') = exp(-||x - x'||^2 / (2 * length_scale^2))
        n1, n2 = cont1.shape[0], cont2.shape[0]
        K = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                diff = cont1[i] - cont2[j]
                sq_dist = np.sum(diff ** 2)
                K[i, j] = np.exp(-sq_dist / (2 * self.length_scale ** 2))
        
        return K
    
    def kendall_tau_kernel(self, X1, X2):
        """
        Kendall-Tau correlation kernel for ordinal features (5D: component_ordering)
        
        Measures rank correlation between orderings
        
        Args:
            X1, X2: Arrays of shape (n_samples, 14)
                    Dimensions 9-13 are ordering positions
        
        Returns:
            Kernel matrix of shape (n1, n2)
        """
        # Extract ordering features (dimensions 9-13)
        ord1 = X1[:, 9:14]
        ord2 = X2[:, 9:14]
        
        n1, n2 = ord1.shape[0], ord2.shape[0]
        K = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                # Kendall-Tau correlation (-1 to 1)
                tau, _ = kendalltau(ord1[i], ord2[j])
                
                # Convert to similarity (0 to 1)
                # tau=-1 (opposite) → 0, tau=0 (random) → 0.5, tau=1 (same) → 1
                K[i, j] = (tau + 1) / 2
        
        return K
    
    def __call__(self, X, Y=None, eval_gradient=False):
        """
        Compute kernel between X and Y
        
        Args:
            X: Array of shape (n_samples_X, 14)
            Y: Array of shape (n_samples_Y, 14) or None
            eval_gradient: Whether to compute gradient (not implemented)
        
        Returns:
            K: Kernel matrix of shape (n_samples_X, n_samples_Y)
        """
        if Y is None:
            Y = X
        
        # Ensure arrays
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        
        # Compute each kernel component
        K_cat = self.hamming_kernel(X, Y)
        K_cont = self.rbf_kernel(X, Y)
        K_ord = self.kendall_tau_kernel(X, Y)
        
        # Combine with weights (multiplicative - captures interactions)
        K = (self.categorical_weight * K_cat + 
             self.continuous_weight * K_cont + 
             self.ordering_weight * K_ord)
        
        if eval_gradient:
            raise NotImplementedError("Gradient computation not implemented")
        
        return K