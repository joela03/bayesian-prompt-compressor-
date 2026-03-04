"""
Test compression on real GPT-4o-mini with quantitative metrics

This validates that:
1. Compressed prompts actually work with real LLMs
2. We can measure quality objectively
3. Compression-performance tradeoff is real
"""

import os
import openai
from compress_prompt import PromptCompressor
from evaluators import RealEvaluator

class QuantitativeEvaluator:
    """
    Measure actual LLM output quality with multiple metrics
    """
    
    def __init__(self, model="gpt-4o-mini"):
        self.model = model
        self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    def evaluate_prompt_quality(self, prompt_text, test_query, reference_answer=None):
        """
        Evaluate a prompt with quantitative metrics
        
        Args:
            prompt_text: The system prompt to test
            test_query: A question to ask
            reference_answer: Optional gold standard answer
        
        Returns:
            metrics: Dict of quantitative scores
        """
        # Generate response
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt_text},
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