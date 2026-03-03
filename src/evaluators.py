"""
Evaluate prompt performance
"""
import numpy as np
from encoders import PromptStructure, create_test_structure
import time
from typing import Dict

class MockEvaluator:
    """
    Fake evaluator for testing

    Creates a synthetic landscape:
    - Having instruction is good 
    - Having examples is good more examples is better
    - Shorter is better (we want it to be compressed)
    - Ordering: instruction-first is best
    """

    def __init__(self, noise_level: float = 0.05):
        self.noise_level = noise_level
        self.call_count = 0

    def evaluate(self, structure: PromptStructure) -> float:
        """
        Synthetic evaluation function
        
        Returns:
            score: higher is better (roughly 0-1 range)
        """
        self.call_count += 1
        
        score = 0.0
        
        # Component bonuses
        if structure.has_instruction:
            score += 0.3
        if structure.has_examples:
            score += 0.2
            score += structure.num_examples * 0.1  # max +0.1
        if structure.has_style:
            score += 0.1
        if structure.has_constraints:
            score += 0.05
        
        # Compression bonus: shorter is better
        compression_bonus = (1 - structure.total_tokens) * 0.2
        score += compression_bonus
        
        # Ordering bonus: instruction should be first
        if structure.component_ordering[0] == 1:
            score += 0.15
        
        # Add noise 
        noise = np.random.randn() * self.noise_level
        score += noise
        
        # Clip to reasonable range
        score = np.clip(score, 0, 1)
        
        time.sleep(0.01)
        
        return score
    
    def get_stats(self) -> Dict:
        return {
            'total_calls': self.call_count
        }

if __name__ == "__main__":
    evaluator = MockEvaluator(noise_level=0.05)
    
    # Test 1: Evaluate a good structure
    print("Test 1: Good structure")
    good_structure = create_test_structure()
    score1 = evaluator.evaluate(good_structure)
    print(f"  Score: {score1:.3f}")
    
    # Test 2: Evaluate a bad structure
    print("\nTest 2: Bad structure")
    bad_structure = PromptStructure(
        has_instruction=True, 
        has_examples=False,
        has_constraints=False,
        has_style=False,
        has_context=False,
        num_examples=0.0,
        instruction_length=0.0,
        total_tokens=0.9,
        component_ordering=[5, 4, 3, 2, 1]
    )
    score2 = evaluator.evaluate(bad_structure)
    print(f"  Score: {score2:.3f}")
    
    # Test 3: Check consistency with noise
    print("\nTest 3: Consistency (10 evaluations of same structure)")
    scores = [evaluator.evaluate(good_structure) for _ in range(10)]
    print(f"  Mean: {np.mean(scores):.3f}")
    print(f"  Std: {np.std(scores):.3f}")
    print(f"  Range: [{min(scores):.3f}, {max(scores):.3f}]")
    
    print(f"\nTotal calls: {evaluator.call_count}")
    print("\n Evaluation works!")