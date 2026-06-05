"""Bayesian Optimiser over the 14-D prompt structure space."""

from dataclasses import dataclass
import logging
from typing import List, Optional

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.preprocessing import StandardScaler

from .encoders import PromptEncoder, PromptStructure
from .evaluators import MockEvaluator

logger = logging.getLogger(__name__)


# ISR (Information Sufficiency Ratio) fields are consumed by the attention-
# informed subclass to gate compression before BO runs. Plain BayesianPromptOptimiser
# ignores them; they're configured here so all optimiser variants share one
# config surface.


@dataclass
class OptimisationConfig:
    """Configuration for Bayesian Optimisation"""

    n_iterations: int = 30
    n_init: int = 10
    beta: float = 4.0
    n_candidates: int = 100
    random_seed: int = 42
    verbose: bool = False

    # ISR pre-compression gate (consumed by AttentionInformedOptimiser).
    # See information_sufficiency.ISRGate for semantics.
    enable_isr_gate: bool = True
    isr_high: float = 0.85
    isr_low: float = 0.40

@dataclass
class OptimisationResult:
    """Result of optimisation run"""

    best_structure: PromptStructure
    best_score: float
    all_scores: List[float]
    all_structures: List[PromptStructure]
    total_evaluations: int

    # Set by AttentionInformedOptimiser when the ISR gate triggers a skip
    # or an aggressive-mode override. None when the gate didn't run.
    skipped: bool = False
    isr_score: Optional[float] = None
    isr_reason: Optional[str] = None

    # Resolved alpha (quality/compression trade-off) used by this run.
    alpha_used: Optional[float] = None

    def summary(self) -> str:
        """Print-friendly summary"""
        return f"""
Optimisation Results:
{'='*60}
Best Score: {self.best_score:.3f}
Total Evaluations: {self.total_evaluations}

Best Structure:
  Components:
    - Instruction:  {self.best_structure.has_instruction}
    - Examples:     {self.best_structure.has_examples} (n={self.best_structure.num_examples:.2f})
    - Constraints:  {self.best_structure.has_constraints}
    - Style:        {self.best_structure.has_style}
    - Context:      {self.best_structure.has_context}
  
  Metrics:
    - Instruction length: {self.best_structure.instruction_length:.2f}
    - Total tokens:       {self.best_structure.total_tokens:.2f}
    - Ordering:           {self.best_structure.component_ordering}

Score Statistics:
  - Mean:   {np.mean(self.all_scores):.3f}
  - Std:    {np.std(self.all_scores):.3f}
  - Min:    {min(self.all_scores):.3f}
  - Max:    {max(self.all_scores):.3f}
{'='*60}
"""

class BayesianPromptOptimiser:
    """
    BO with RBF kernel
    """

    def __init__(
        self,
        encoder: PromptEncoder,
        evaluator: MockEvaluator,
        config: Optional[OptimisationConfig] = None
    ):
        
        self.encoder = encoder
        self.evaluator = evaluator
        self.config = config or OptimisationConfig()

        np.random.seed(self.config.random_seed)
        
        # GP with simple RBF kernel. Widen the length-scale bounds because
        # SemanticEvaluator produces a narrow score range (~0.7-1.0) which
        # pushes the optimiser against the default upper bound.
        kernel = ConstantKernel(1.0) * RBF(
            length_scale=1.0, length_scale_bounds=(1e-5, 1e8)
        )
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.01,
            n_restarts_optimizer=5,
            random_state=self.config.random_seed,
        )

        # Scale features before fitting the GP. The 14D feature vectors mix
        # binary, continuous, and ordering scales, which makes lbfgs hard to
        # converge without z-scoring.
        self.scaler = StandardScaler()
        self.scaler_fitted = False

        self.X_observed = []
        self.y_observed = []
        self.structures_tested = []

    def random_structure(self) -> PromptStructure:
        """
        Generate random prompt structure for initialisation
        """

        has_instruction = np.random.rand() > 0.1 
        has_examples = np.random.rand() > 0.3
        has_constraints = np.random.rand() > 0.5
        has_style = np.random.rand() > 0.5
        has_context = np.random.rand() > 0.7
        
        num_examples = np.random.rand()
        instruction_length = np.random.rand()
        total_tokens = np.random.rand() * 0.8 + 0.2 
        
        component_ordering = np.random.permutation([1, 2, 3, 4, 5]).tolist()
        
        return PromptStructure(
            has_instruction=has_instruction,
            has_examples=has_examples,
            has_constraints=has_constraints,
            has_style=has_style,
            has_context=has_context,
            num_examples=num_examples,
            instruction_length=instruction_length,
            total_tokens=total_tokens,
            component_ordering=component_ordering
        )

    def ucb_acquisition(self, X_candidates: np.ndarray, beta: Optional[float] = None) -> int:
        """
        Upper Confidence Bound acquisition function
        
        Args:
            X_candidates: Array of candidate vectors (n_candidates, dim)
            beta: Exploration parameter
        
        Returns:
            index of best candidate
        """
        if beta is None:
            beta = self.config.beta
        
        # return random candidate if no data
        if len(self.X_observed) == 0:
            return np.random.randint(len(X_candidates))
        
        # GP predictions (scale candidates with the same fitted scaler)
        X_scaled = self.scaler.transform(X_candidates)
        mu, sigma = self.gp.predict(X_scaled, return_std=True)
        
        # UCB = mean + beta * std
        ucb_values = mu + beta * sigma
        
        # Return index of maximum UCB
        return np.argmax(ucb_values)    

    def optimise(self) -> OptimisationResult:
        """
        Run Bayesian Optimisation using config 

        Return:
            OptimisationResult with all details
        """

        cfg = self.config

        self._log_progress("Starting Bayesian Optimisation")
        self._log_progress("  Initialisation: %s random points", cfg.n_init)
        self._log_progress("  BO iterations: %s", cfg.n_iterations)
        self._log_progress("  Total evaluations: %s", cfg.n_init + cfg.n_iterations)
        self._log_progress("  UCB beta: %s", cfg.beta)
        self._log_progress("  Random seed: %s", cfg.random_seed)

        # PHASE 1: Random initialisation
        self._log_progress("Random Initialisation")
        for i in range(cfg.n_init):
            structure = self.random_structure()
            vector = self.encoder.encode(structure)
            score = self.evaluator.evaluate(structure)
            
            self.X_observed.append(vector)
            self.y_observed.append(score)
            self.structures_tested.append(structure)
            
            self._log_progress("  Init %s/%s: score = %.3f", i + 1, cfg.n_init, score)

        # Fit initial GP (with feature scaling)
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        if not self.scaler_fitted:
            X_scaled = self.scaler.fit_transform(X)
            self.scaler_fitted = True
        else:
            X_scaled = self.scaler.transform(X)
        self.gp.fit(X_scaled, y)

        self._log_progress("Bayesian Optimisation")

        
        # PHASE 2: BO iterations
        for i in range(cfg.n_iterations):
            # Generate candidate structures
            candidates = [self.random_structure() for _ in range(cfg.n_candidates)]
            candidate_vectors = np.array([self.encoder.encode(s) for s in candidates])
            
            # Acquisition function selects best candidate
            best_idx = self.ucb_acquisition(candidate_vectors)
            next_structure = candidates[best_idx]
            next_vector = candidate_vectors[best_idx]
            
            # Evaluate
            score = self.evaluator.evaluate(next_structure)
            
            # Update observations
            self.X_observed.append(next_vector)
            self.y_observed.append(score)
            self.structures_tested.append(next_structure)
            
            # Update GP (scaler already fitted from the initial fit)
            X = np.array(self.X_observed)
            y = np.array(self.y_observed)
            X_scaled = self.scaler.transform(X)
            self.gp.fit(X_scaled, y)
            
            # Get current best
            best_score_so_far = max(self.y_observed)
            
            self._log_progress(
                "  Iter %s/%s: score = %.3f | best so far = %.3f",
                i + 1,
                cfg.n_iterations,
                score,
                best_score_so_far,
            )
        
        # Build result
        best_idx = np.argmax(self.y_observed)
        
        result = OptimisationResult(
            best_structure=self.structures_tested[best_idx],
            best_score=self.y_observed[best_idx],
            all_scores=self.y_observed.copy(),
            all_structures=self.structures_tested.copy(),
            total_evaluations=len(self.y_observed)
        )
        
        self._log_progress("Optimisation complete")
        self._log_progress("  Best score: %.3f", result.best_score)
        self._log_progress("  Total evaluations: %s", result.total_evaluations)
        
        return result

    def _log_progress(self, message: str, *args) -> None:
        if self.config.verbose:
            logger.info(message, *args)

    def plot_progress(self, result: OptimisationResult, save_path: Optional[str] = None):
        """
        Visualise optimisation progress. Requires matplotlib (research-only).
        """
        # Deferred import: matplotlib is in the [research] extras, not core deps.
        import matplotlib.pyplot as plt

        scores = result.all_scores

        plt.figure(figsize=(12, 4))
        
        # Plot 1: Scores over time
        plt.subplot(1, 3, 1)
        plt.plot(scores, 'o-', alpha=0.6, label='Observed', markersize=4)
        
        # Running best
        running_best = [max(scores[:i+1]) for i in range(len(scores))]
        plt.plot(running_best, 'r-', linewidth=2, label='Best so far')
        
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.title('Optimisation Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Score distribution
        plt.subplot(1, 3, 2)
        plt.hist(scores, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(max(scores), color='r', linestyle='--', linewidth=2, label='Best')
        plt.axvline(np.mean(scores), color='g', linestyle='--', linewidth=2, label='Mean')
        plt.xlabel('Score')
        plt.ylabel('Count')
        plt.title('Score Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Improvement over iterations
        plt.subplot(1, 3, 3)
        improvements = [running_best[i] - running_best[0] for i in range(len(running_best))]
        plt.plot(improvements, 'g-', linewidth=2)
        plt.fill_between(range(len(improvements)), 0, improvements, alpha=0.3)
        plt.xlabel('Iteration')
        plt.ylabel('Improvement from Initial')
        plt.title('Cumulative Improvement')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info("Plot saved to: %s", save_path)
        
        plt.show()

if __name__ == "__main__":
    import os
    
    # Create results directory if it doesn't exist
    os.makedirs('data/results', exist_ok=True)
    
    # Setup with dataclass config
    config = OptimisationConfig(
        n_iterations=20,
        n_init=10,
        beta=2.0,
        n_candidates=100,
        random_seed=42
    )
    
    encoder = PromptEncoder()
    evaluator = MockEvaluator(noise_level=0.05)
    optimiser = BayesianPromptOptimiser(encoder, evaluator, config)
    
    # Run optimisation
    result = optimiser.optimise()
    
    # Print summary
    print(result.summary())
    
    # Plot
    optimiser.plot_progress(result, save_path='data/results/v0_optimisation_progress.png')
