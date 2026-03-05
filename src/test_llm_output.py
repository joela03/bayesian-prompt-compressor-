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
        if answer[0].isupper():
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
