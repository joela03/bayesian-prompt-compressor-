"""
The actual Bayesian Optimiser
"""

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from typing import List, Tuple, Callable
import matplotlip.pyplot as pyplot

from dataclasses import dataclass, field

from src.encoders import PromptStructure
from src.evaluators import MockEvaluator

class BayesianPromptOptimiser:
    """
    Simple BO with RBF kernel
    """

@dataclass
class OptimisationConfig:
    """Configuration for Bayesian Optimisation"""

    n_iterations: int = 30
    n_init: int = 10
    bet: float = 2.0
    n_candidates: int = 100
    random_seed: int = 42

@dataclass
class OptimisationResult:
    """Result of optimisation run"""

    best_structure: PromptStructure
    best_score: floatall_scores: List[float]
    all_scores: List[float]
    all_structures: List[PromptStructure]
    total_evaluation: int

    def summary(self) -> str:
        """Print-friendly summary"""
        return f"""
Optimization Results:
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
