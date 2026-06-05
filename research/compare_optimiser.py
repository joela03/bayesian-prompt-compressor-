# File: src/compare_optimizers.py

"""
Compare naive vs informed Bayesian optimization
Shows speedup from P3 priors
"""

import json
import sys
from pathlib import Path
from prompt_compress import PromptCompressor

def compare_optimizers():
    """Run comparison on test prompts"""
    
    # Load test prompts
    test_prompts_file = Path(__file__).parent.parent / 'data' / 'test_prompts' / 'long_prompts_test_set.json'
    
    with open(test_prompts_file) as f:
        prompts = json.load(f)
    
    # Test on first 3 prompts (enough to show pattern)
    test_prompts = prompts[:3]
    
    print("="*70)
    print("OPTIMIZER COMPARISON: NAIVE vs INFORMED")
    print("="*70)
    
    results = {
        'naive': [],
        'informed': []
    }
    
    for prompt_data in test_prompts:
        print(f"\n{'='*70}")
        print(f"Testing: {prompt_data['role']} ({prompt_data['word_count']} words)")
        print(f"{'='*70}")
        
        # NAIVE
        print("\n[1/2] NAIVE Optimizer (uniform random initialization)...")
        naive = PromptCompressor(use_informed_prior=False)
        result_naive = naive.compress(prompt_data['text'], output=False)
        
        results['naive'].append({
            'id': prompt_data['id'],
            'evaluations': result_naive['optimisation_result'].total_evaluations,
            'compression': result_naive['metrics']['compression_ratio'],
            'best_score': result_naive['optimisation_result'].best_score
        })
        
        print(f"   ✅ Evaluations: {result_naive['optimisation_result'].total_evaluations}")
        print(f"   ✅ Best score: {result_naive['optimisation_result'].best_score:.3f}")
        print(f"   ✅ Compression: {result_naive['metrics']['compression_ratio']:.1%}")
        
        # INFORMED
        print("\n[2/2] INFORMED Optimizer (P3 priors)...")
        informed = PromptCompressor(use_informed_prior=True)
        result_informed = informed.compress(prompt_data['text'], output=False)
        
        results['informed'].append({
            'id': prompt_data['id'],
            'evaluations': result_informed['optimisation_result'].total_evaluations,
            'compression': result_informed['metrics']['compression_ratio'],
            'best_score': result_informed['optimisation_result'].best_score
        })
        
        print(f"   ✅ Evaluations: {result_informed['optimisation_result'].total_evaluations}")
        print(f"   ✅ Best score: {result_informed['optimisation_result'].best_score:.3f}")
        print(f"   ✅ Compression: {result_informed['metrics']['compression_ratio']:.1%}")
        
        # Compare
        if result_naive['optimisation_result'].total_evaluations > 0:
            speedup = result_naive['optimisation_result'].total_evaluations / result_informed['optimisation_result'].total_evaluations
            print(f"\n   🚀 SPEEDUP: {speedup:.1f}× faster!")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    avg_naive_evals = sum(r['evaluations'] for r in results['naive']) / len(results['naive'])
    avg_informed_evals = sum(r['evaluations'] for r in results['informed']) / len(results['informed'])
    avg_speedup = avg_naive_evals / avg_informed_evals if avg_informed_evals > 0 else 1.0
    
    avg_naive_compression = sum(r['compression'] for r in results['naive']) / len(results['naive'])
    avg_informed_compression = sum(r['compression'] for r in results['informed']) / len(results['informed'])
    
    avg_naive_score = sum(r['best_score'] for r in results['naive']) / len(results['naive'])
    avg_informed_score = sum(r['best_score'] for r in results['informed']) / len(results['informed'])
    
    print(f"\n{'Metric':<25} {'Naive':<15} {'Informed':<15}")
    print("-"*70)
    print(f"{'Average evaluations:':<25} {avg_naive_evals:<15.0f} {avg_informed_evals:<15.0f}")
    print(f"{'Average best score:':<25} {avg_naive_score:<15.3f} {avg_informed_score:<15.3f}")
    print(f"{'Average compression:':<25} {avg_naive_compression:<15.1%} {avg_informed_compression:<15.1%}")
    
    print(f"\n🎯 OVERALL SPEEDUP: {avg_speedup:.1f}× faster")
    
    quality_ratio = avg_informed_score / avg_naive_score if avg_naive_score > 0 else 1.0
    print(f"   Quality: {quality_ratio:.1%} of naive")
    
    compression_ratio = avg_informed_compression / avg_naive_compression if avg_naive_compression > 0 else 1.0
    print(f"   Compression: {compression_ratio:.1%} of naive")
    
    # Save results
    output_file = Path(__file__).parent.parent / 'data' / 'results' / 'optimizer_comparison.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Saved results to: {output_file}")
    
    return results

def graph():
    scores = result['optimisation_result'].all_scores
    running_best = [max(scores[:i+1]) for i in range(len(scores))]

    plt.figure(figsize=(10, 6))
    plt.plot(scores, 'o-', alpha=0.6, label='Observed')
    plt.plot(running_best, 'r-', linewidth=2, label='Best so far')
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.title('Bayesian Optimization Progress')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('bo_progress.png', dpi=150)

if __name__ == "__main__":
    compare_optimizers()
