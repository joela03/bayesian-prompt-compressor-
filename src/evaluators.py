"""
Evaluate prompt performance
"""
import os
import numpy as np
from encoders import PromptStructure, create_test_structure
import time
from typing import Dict
import openai

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

class RealEvaluator:
    """
    Real evaluator that calls GPT-4o-mini
    
    Tests prompts with real LLM and measures output quality
    """
    
    def __init__(self, model: str = "gpt-4o-mini", test_query: str = None):
        self.model = model
        self.test_query = test_query or "What are the key principles of effective communication?"
        self.call_count = 0
        
        # Initialise OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY environment variable")
        
        self.client = openai.OpenAI(api_key=api_key)

    def structure_to_prompt_text(self, structure: PromptStructure) -> str:
        """
        Convert PromptStructure to actual text prompt
        
        This is a simplified version - in production you'd use
        the actual prompt components from the parser
        """
        parts = []
        
        # Instruction
        if structure.has_instruction:
            if structure.instruction_length > 0.7:
                parts.append("You are an expert assistant. Provide thorough, comprehensive answers.")
            elif structure.instruction_length > 0.3:
                parts.append("You are a helpful assistant. Provide clear answers.")
            else:
                parts.append("You are an assistant.")
        
        # Examples
        if structure.has_examples:
            n_examples = int(structure.num_examples * 5)
            if n_examples > 0:
                parts.append(f"\nFor example, good answers are well-structured and detailed.")
        
        # Constraints
        if structure.has_constraints:
            parts.append("\nConstraints: Keep responses concise and factual.")
        
        # Style
        if structure.has_style:
            parts.append("\nStyle: Professional and clear.")
        
        # Context
        if structure.has_context:
            parts.append("\nContext: This is for a technical audience.")
        
        return '\n'.join(parts)

    def evaluate(self, structure: PromptStructure, query: str = None) -> float:
        """
        Evaluate a PromptStructure (for optimisation loop)
        
        Args:
            structure: PromptStructure to evaluate
            query: Optional test query (uses default if not provided)
        
        Returns:
            score: 0-1, optimisation objective (quality/tokens)
        """
        self.call_count += 1
        
        # Build prompt text
        prompt_text = self.structure_to_prompt_text(structure)
        
        # Evaluate with full metrics
        metrics = self.evaluate_prompt_text(prompt_text, query)
        
        if metrics is None:
            return 0.5  # Neutral on error
        
        # Optimisation objective: quality / token_cost
        quality = metrics['overall_quality']
        tokens = len(prompt_text.split())
        normalised_tokens = tokens / 100
        
        objective = quality / (1 + normalised_tokens * 0.5)
        
        return objective
    
    def evaluate_prompt_text(self, prompt_text:str, query: str = None, reference_answer: str = None):
        """
        Evaluate a prompt with quantitative metrics
        
        Args:
            prompt_text: The system prompt to test
            test_query: A question to ask
        
        Returns:
            metrics: Dict of quantitative scores
        """

        if query is None:
            query = self.test_query
        
        # Generate response
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": structure},
                    {"role": "user", "content": test_query}
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
            
            # Compute metrics
            metrics = {
                'answer': answer,
                'answer_length': len(answer.split()),
                'response_time': response.usage.total_tokens,
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
            }
            
            # Quality metrics
            metrics.update(self._compute_quality_metrics(answer, test_query, reference_answer))
            
            return metrics
            
        except Exception as e:
            print(f"Error: {e}")
            return None

    def _compute_quality_metrics(self, answer, query, reference):
        """
        Compute quality scores
        """
        metrics = {}
        
        # Completeness
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        query_coverage = len(query_words & answer_words) / len(query_words)
        metrics['query_coverage'] = query_coverage
        
        # Conciseness
        word_count = len(answer.split())
        if 50 <= word_count <= 200:
            conciseness_score = 1.0
        elif word_count < 50:
            conciseness_score = word_count / 50
        else:
            conciseness_score = max(0, 1.0 - (word_count - 200) / 200)
        metrics['conciseness'] = conciseness_score
        
        #Structure
        structure_score = 0.0
        if answer and answer[0].isupper():
            structure_score += 0.3
        if any(p in answer for p in ['.', '!', '?']):
            structure_score += 0.3
        sentences = answer.split('.')
        if len(sentences) >= 2:
            structure_score += 0.2
        if '\n' in answer:  # Has paragraphs
            structure_score += 0.2
        metrics['structure'] = min(structure_score, 1.0)
        
        # 4. If we have reference, compute similarity
        if reference:
            # Response Overlap with reference
            ref_words = set(reference.lower().split())
            answer_words = set(answer.lower().split())
            overlap = len(ref_words & answer_words) / len(ref_words)
            metrics['reference_similarity'] = overlap
        
        # 5. Overall quality
        weights = {
            'query_coverage': 0.3,
            'conciseness': 0.3,
            'structure': 0.2,
            'reference_similarity': 0.2 if reference else 0.0
        }
        
        # Normalise weights if no reference
        if not reference:
            weights = {k: v / 0.8 for k, v in weights.items() if k != 'reference_similarity'}
        
        overall = sum(metrics.get(k, 0) * v for k, v in weights.items())
        metrics['overall_quality'] = overall
        
        return metrics

    def get_stats(self) -> Dict:
        """Get evaluator statistics"""
        return {
            'total_calls': self.call_count,
            'model': self.model,
            'approx_cost': self.call_count * 0.01
        }
