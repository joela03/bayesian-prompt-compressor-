"""
Complete prompt compression pipeline with enhanced metrics display
"""
import os
from encoders import PromptEncoder
from evaluators import MockEvaluator, RealEvaluator
from optimiser import BayesianPromptOptimiser, OptimisationConfig
from prompt_parser import PromptParser, PromptBuilder

class PromptCompressor:
    """
    End-to-end prompt compression
    """
    
    def __init__(self, 
                 use_real_evaluator: bool = False,
                 optimisation_config: OptimisationConfig = None):
        """
        Args:
            use_real_evaluator: If True, use real LLM (costs money)
            optimisation_config: Config for Bayesian optimisation
        """
        self.parser = PromptParser()
        self.builder = PromptBuilder()
        self.encoder = PromptEncoder()
        
        self.evaluator = MockEvaluator(noise_level=0.05)
        print("Using MockEvaluator")
        
        # Setup optimiser
        self.config = optimisation_config or OptimisationConfig(
            n_iterations=20,
            n_init=5,
            beta=2.0,
            random_seed=42
        )
        
        self.optimiser = BayesianPromptOptimiser(
            self.encoder,
            self.evaluator,
            self.config
        )