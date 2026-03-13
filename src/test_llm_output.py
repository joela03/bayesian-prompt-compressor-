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
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

def test_compression_on_real_llm():
    """
    Complete test: compress a prompt and measure real LLM performance
    """
    print("="*70)
    print("LLM COMPRESSION TEST")
    print("="*70)
    
    load_dotenv()
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY environment variable")
        return
    
    # Test prompt
    original_prompt = """
You are an expert AI assistant helping with research questions.

When answering questions, please follow these comprehensive guidelines:
1. Provide thorough, well-researched answers
2. Include relevant citations and sources when available
3. Use clear, professional academic language
4. Structure your response logically with clear sections

For example, a good response begins with a direct answer to the question, followed by supporting evidence, relevant context, and acknowledgment of any limitations or alternative perspectives.

Constraints:
- Keep responses under 300 words
- Maintain an objective, balanced tone
- Avoid speculation or unsupported claims
- Do not make recommendations without sufficient evidence

Style: Use an academic but accessible writing style. Be precise and professional while remaining approachable.

Context: You are assisting with academic research and your responses should meet scholarly standards.
"""
    
    # Test queries
    test_queries = [
        "What are the main causes of climate change?",
        "Explain the concept of machine learning in simple terms.",
        "What is the difference between correlation and causation?"
    ]
    
    # Reference answers (optional, for similarity scoring)
    reference_answers = {
        test_queries[0]: "Climate change is primarily caused by greenhouse gas emissions from human activities, particularly the burning of fossil fuels (coal, oil, and natural gas) which releases carbon dioxide and other gases into the atmosphere. These gases trap heat, leading to global warming. Other significant contributors include deforestation, industrial processes, and agriculture.",
        test_queries[1]: "Machine learning is a subset of artificial intelligence where computers learn patterns from data without being explicitly programmed. Instead of following fixed rules, ML algorithms improve their performance through experience. For example, a spam filter learns to identify spam by analyzing thousands of emails, gradually getting better at distinguishing spam from legitimate messages.",
        test_queries[2]: "Correlation means two variables change together, while causation means one variable directly causes changes in another. For example, ice cream sales and drowning rates are correlated (both increase in summer), but ice cream doesn't cause drowning—the relationship is coincidental. Establishing causation requires controlled experiments or rigorous statistical methods."
    }
    
    print("\n Original Prompt:")
    print(f"   {len(original_prompt.split())} words, {len(original_prompt)} characters")
    
    # Step 1: Compress the prompt
    print("\n Compressing prompt")
    compressor = PromptCompressor(use_real_evaluator=True)
    compression_result = compressor.compress(original_prompt)
    
    compressed_prompt = compression_result['compressed_text']
    
    print(f"   Original:   {compression_result['metrics']['original_tokens']} tokens")
    print(f"   Compressed: {compression_result['metrics']['compressed_tokens']} tokens")
    print(f"   Saved:      {compression_result['metrics']['tokens_saved']} tokens ({compression_result['metrics']['compression_ratio']:.1%})")
    
    # Step 2: Test both prompts on real queries
    print(" Step 2: Testing both prompts on real queries")
    
    evaluator = RealEvaluator(model="gpt-4o-mini")
    
    results = {
        'original': [],
        'compressed': []
    }
    
    for i, query in enumerate(test_queries):
        print(f"\n   Query {i+1}: {query[:50]}")
        
        # Test original
        print("      Testing original prompt")
        orig_metrics = evaluator.evaluate_prompt_text(
            original_prompt,
            query,
            reference_answers.get(query)
        )
        results['original'].append(orig_metrics)
        
        # Test compressed
        print("      Testing compressed prompt...")
        comp_metrics = evaluator.evaluate_prompt_text(
            compressed_prompt,
            query,
            reference_answers.get(query)
        )
        results['compressed'].append(comp_metrics)
    
    # Step 3: Compare results
    print("\n" + "="*70)
    print(" QUANTITATIVE COMPARISON")
    print("="*70)
    
    # Average metrics
    metrics_to_compare = ['overall_quality', 'query_coverage', 'conciseness', 'structure', 'answer_length']
    
    print(f"\n{'Metric':<25} {'Original':<15} {'Compressed':<15} {'Difference':<15}")
    print("-"*70)
    
    for metric in metrics_to_compare:
        orig_avg = sum(r[metric] for r in results['original']) / len(results['original'])
        comp_avg = sum(r[metric] for r in results['compressed']) / len(results['compressed'])
        diff = comp_avg - orig_avg
        
        diff_str = f"{diff:+.3f}" if metric != 'answer_length' else f"{diff:+.1f}"
        
        print(f"{metric:<25} {orig_avg:<15.3f} {comp_avg:<15.3f} {diff_str:<15}")
    
    # Token usage
    orig_prompt_tokens = sum(r['prompt_tokens'] for r in results['original'])
    comp_prompt_tokens = sum(r['prompt_tokens'] for r in results['compressed'])
    
    print(f"\n{'Token Usage (3 queries):':<25} {orig_prompt_tokens:<15} {comp_prompt_tokens:<15} {comp_prompt_tokens - orig_prompt_tokens:+<15}")
    
    # Cost analysis
    cost_per_1m_tokens = 0.15
    orig_cost = (orig_prompt_tokens / 1_000_000) * cost_per_1m_tokens
    comp_cost = (comp_prompt_tokens / 1_000_000) * cost_per_1m_tokens
    
    print(f"\n💰 Cost Analysis (3 queries):")
    print(f"   Original:   ${orig_cost:.6f}")
    print(f"   Compressed: ${comp_cost:.6f}")
    print(f"   Savings:    ${orig_cost - comp_cost:.6f}")
    
    # Extrapolate
    queries_per_day = 10000
    days_per_month = 30
    
    daily_savings = (orig_cost - comp_cost) / 3 * queries_per_day
    monthly_savings = daily_savings * days_per_month
    
    print(f"\n Projected Savings:")
    print(f"   At {queries_per_day:,} queries/day:")
    print(f"   Daily:   ${daily_savings:.2f}")
    print(f"   Monthly: ${monthly_savings:.2f}")
    print(f"   Yearly:  ${monthly_savings * 12:.2f}")
    
    # Quality verdict
    print(f"\n Quality Assessment:")
    orig_quality = sum(r['overall_quality'] for r in results['original']) / len(results['original'])
    comp_quality = sum(r['overall_quality'] for r in results['compressed']) / len(results['compressed'])
    
    quality_retention = comp_quality / orig_quality if orig_quality > 0 else 0
    
    print(f"   Original quality:     {orig_quality:.3f}")
    print(f"   Compressed quality:   {comp_quality:.3f}")
    print(f"   Quality retention:    {quality_retention:.1%}")
    
    if quality_retention >= 0.95:
        print(f"   Verdict: EXCELLENT - Quality maintained with {compression_result['metrics']['compression_ratio']:.0%} compression")
    elif quality_retention >= 0.85:
        print(f"   Verdict: GOOD - Minor quality loss, significant token savings")
    elif quality_retention >= 0.75:
        print(f"   Verdict:  ACCEPTABLE - Noticeable quality loss, evaluate tradeoff")
    else:
        print(f"   Verdict: POOR - Too much quality loss, compression too aggressive")
    
    # Save detailed results
    import json
    detailed_results = {
        'original_prompt': original_prompt,
        'compressed_prompt': compressed_prompt,
        'compression_metrics': compression_result['metrics'],
        'test_queries': test_queries,
        'original_responses': results['original'],
        'compressed_responses': results['compressed'],
        'summary': {
            'quality_retention': quality_retention,
            'token_savings': compression_result['metrics']['tokens_saved'],
            'monthly_cost_savings': monthly_savings
        }
    }
    
    with open('data/results/real_llm_test_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"   Saved to: data/results/real_llm_test_results.json")
    
    # Print example responses
    print(f"\n📄 Example Response Comparison:")
    print(f"   Query: {test_queries[0]}")
    print(f"\n   Original response:")
    print(f"   {results['original'][0]['answer'][:200]}...")
    print(f"\n   Compressed response:")
    print(f"   {results['compressed'][0]['answer'][:200]}...")

if __name__ == "__main__":
    test_compression_on_real_llm()