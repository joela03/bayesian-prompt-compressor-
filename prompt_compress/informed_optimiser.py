"""Informed Bayesian Optimiser using P3 dataset priors."""

import json
import logging
from pathlib import Path

import numpy as np

from .encoders import PromptStructure
from .optimiser import BayesianPromptOptimiser, OptimisationConfig

logger = logging.getLogger(__name__)

class InformedBayesianOptimiser(BayesianPromptOptimiser):
    """
    Bayesian optimizer with informed priors from P3 analysis
    
    Overrides random_structure() to sample from learned priors
    instead of uniform random distribution.
    """
    
    def __init__(self, encoder, evaluator, config=None, prior_path=None):
        """
        Initialize with P3 priors
        """
        # Default to the bundled package data file. Works whether installed
        # via wheel or in editable mode.
        if prior_path is None:
            prior_path = Path(__file__).parent / 'data' / 'p3_findings_adjusted.json'
        
        # Call parent init
        if config is None:
            config = OptimisationConfig(
                n_iterations=20,
                n_init=5,
                beta=2.0,
                random_seed=42
            )
        
        super().__init__(encoder, evaluator, config)
        
        self.prior = self._load_prior(prior_path)
        
        if self.prior:
            logger.info("Loaded P3 informed priors")
        else:
            logger.info("P3 priors not found, using uniform sampling")
    
    def _load_prior(self, path):
        """
        Load P3 analysis results as prior distribution
        
        Returns:
            Dict with prior mean and variance, or None if not found
        """
        try:
            prior_file = Path(path)
            if not prior_file.exists():
                return None
            
            with open(prior_file) as f:
                findings = json.load(f)
            
            # Convert P3 findings to prior distribution
            prior_mean = {
                'instruction_length': findings.get('optimal_instruction_length', 0.5),
                'num_examples': findings.get('optimal_num_examples', 0.4),
                'has_constraints': findings.get('constraints_importance', 0.8),
                'has_style': 1 - findings.get('style_importance', 0.3),
                'has_context': 1 - findings.get('context_importance', 0.3),
            }
            
            # Moderate confidence in prior (allows exploration)
            prior_variance = 0.25
            
            return {
                'mean': prior_mean,
                'variance': prior_variance,
                'source': 'P3 dataset analysis'
            }
        
        except Exception:
            logger.warning("Could not load prior from %s", path, exc_info=True)
            return None
    
    def random_structure(self) -> PromptStructure:
        """
        Generate prompt structure sampling from P3 priors
        
        Overrides parent's uniform random sampling.
        This is called during both initialization and BO iterations.
        """
        if self.prior is None:
            # Fallback to parent's uniform sampling
            return super().random_structure()
        
        prior_mean = self.prior['mean']
        prior_var = self.prior['variance']
        
        # Sample around prior distributions
        return PromptStructure(
            has_instruction=True,  # Always keep instruction
            
            # Sample examples probabilistically (50% from P3)
            has_examples=np.random.rand() < 0.5,
            
            # High probability of keeping constraints (80% from P3)
            has_constraints=np.random.rand() < 0.8,
            
            # Low probability of keeping style/context (30% from P3)
            has_style=np.random.rand() < 0.3,
            has_context=np.random.rand() < 0.3,
            
            # Sample continuous values around prior mean with Gaussian noise
            instruction_length=np.clip(
                prior_mean['instruction_length'] + np.random.normal(0, prior_var),
                0.3, 1.0  # Keep at least 30%
            ),
            
            num_examples=np.clip(
                prior_mean['num_examples'] + np.random.normal(0, prior_var),
                0.0, 1.0
            ),
            
            total_tokens=np.random.rand() * 0.8 + 0.2,  # Same as parent
            
            component_ordering=np.random.permutation([1, 2, 3, 4, 5]).tolist()
        )
