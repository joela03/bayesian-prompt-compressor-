"""
Real-world validation using GPT API with semantic similarity & entropy
"""

import openai
import random
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

class RealPromptValidator:
    """
    Validates compressed prompts using real GPT API calls
    Measures semantic similarity and entropy
    """
    
    def __init__(self, test_queries: List[str] = None):
        """
        Args:
            test_queries: List of test queries to evaluate prompts on
                         If None, uses default set
        """
        # Default test queries
        self.test_queries = test_queries or [
            "What is machine learning?",
            "Explain quantum computing in simple terms.",
            "How do I start a startup?",
            "What are the benefits of meditation?",
            "Summarise the causes of climate change."
        ]
        
        # Load sentence transformer for semantic similarity
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def validate(self, 
                 original_prompt: str, 
                 compressed_prompt: str,
                 n_queries: int = 5) -> Dict:
        """
        Compare original vs compressed prompt on real queries
        
        Returns:
            {
                'semantic_similarity': float,    # 0-1, average across queries
                'semantic_entropy': float,        # Shannon entropy of similarity
                'compression_rate': float,        # 0-1
                'quality_retained': float,        # 0-1
                'outputs_original': List[str],
                'outputs_compressed': List[str],
                'similarities_per_query': List[float],
                'entropies_per_query': List[float]
            }
        """
        # Select random test queries
        test_set = random.sample(self.test_queries, 
                                min(n_queries, len(self.test_queries)))
        
        # Generate outputs with both prompts
        outputs_original = []
        outputs_compressed = []
        
        print(f"\n Testing on {len(test_set)} queries...")
        
        for i, query in enumerate(test_set):
            print(f"  Query {i+1}/{len(test_set)}: {query[:50]}...")
            
            # Original prompt
            response_orig = self._call_gpt(original_prompt, query)
            outputs_original.append(response_orig)
            
            # Compressed prompt
            response_comp = self._call_gpt(compressed_prompt, query)
            outputs_compressed.append(response_comp)
        
        # Compute semantic similarity and entropy for each pair
        similarities = []
        entropies = []
        
        for orig, comp in zip(outputs_original, outputs_compressed):
            # Your semantic similarity calculation
            similarity = self._compute_similarity(orig, comp)
            similarities.append(similarity)
            
            # Your entropy calculation
            entropy = self._calculate_shannon_entropy(similarity)
            entropies.append(entropy)
        
        avg_similarity = np.mean(similarities)
        avg_entropy = np.mean(entropies)
        
        # Compute compression rate
        original_tokens = len(original_prompt.split())
        compressed_tokens = len(compressed_prompt.split())
        compression_rate = 1 - (compressed_tokens / original_tokens)
        
        # Quality score (semantic similarity is the quality measure)
        quality_retained = avg_similarity
        
        return {
            'semantic_similarity': float(avg_similarity),
            'semantic_entropy': float(avg_entropy),
            'compression_rate': float(compression_rate),
            'quality_retained': float(quality_retained),
            'outputs_original': outputs_original,
            'outputs_compressed': outputs_compressed,
            'similarities_per_query': [float(s) for s in similarities],
            'entropies_per_query': [float(e) for e in entropies],
            'token_reduction': original_tokens - compressed_tokens,
            'original_tokens': original_tokens,
            'compressed_tokens': compressed_tokens
        }
    
    def _call_gpt(self, system_prompt: str, query: str) -> str:
        """Call GPT API with system prompt and query"""
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f" API error: {e}")
            return ""
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity using sentence embeddings
        (Your implementation)
        """
        if not text1 or not text2:
            return 0.0
        
        # Generate embeddings
        embeddings = self.model.encode([text1, text2])
        
        # Calculate cosine similarity
        similarity = 1 - cosine(embeddings[0], embeddings[1])
        
        return float(similarity)
    
    def _calculate_shannon_entropy(self, similarity: float) -> float:
        """
        Calculate Shannon entropy from similarity score
        (Your implementation)
        """
        # Normalize similarity to probability [0, 1]
        p = np.clip((1 + similarity) / 2, 1e-9, 1 - 1e-9)
        
        # Shannon entropy: H = -p*log2(p) - (1-p)*log2(1-p)
        entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
        
        return float(entropy)


if __name__ == "__main__":
    # Test
    validator = RealPromptValidator()
    
    original = """You are a helpful AI assistant.

When answering questions:
1. Be thorough and comprehensive
2. Cite sources when possible
3. Use clear, professional language

For example, a good response starts with a direct answer, then provides supporting evidence.

Constraints: Keep responses under 500 words. Avoid speculation.

Style: Use an academic but accessible tone."""

    compressed = """You're a helpful AI assistant.

Guidelines:
1. Be thorough
2. Cite sources
3. Use clear language

Example: Direct answer, then evidence.

Constraints: Under 500 words. No speculation.

Style: Academic but accessible."""
    
    print("="*70)
    print("VALIDATION TEST")
    print("="*70)
    
    results = validator.validate(original, compressed, n_queries=3)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nSemantic Similarity: {results['semantic_similarity']:.2%}")
    print(f"Semantic Entropy:    {results['semantic_entropy']:.4f}")
    print(f"Compression Rate:    {results['compression_rate']:.2%}")
    print(f"Quality Retained:    {results['quality_retained']:.2%}")
    print(f"Token Reduction:     {results['token_reduction']} tokens")
    
    print(f"\nPer-Query Results:")
    for i, (sim, ent) in enumerate(zip(results['similarities_per_query'], 
                                       results['entropies_per_query'])):
        print(f"  Query {i+1}: similarity={sim:.2%}, entropy={ent:.4f}")