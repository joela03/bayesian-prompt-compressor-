"""
Complete prompt compression pipeline with enhanced metrics display
"""
import os
from encoders import PromptEncoder
from evaluators import MockEvaluator
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

    def compress(self, prompt_text: str, output: bool = True):
        """
        Compress a prompt
        
        Args:
            prompt_text: Original prompt (string)
            output: Print progress
        
        Returns:
            result: Dict with compression details
        """
        if output:
            print("="*70)
            print("PROMPT COMPRESSION")
            print("="*70)
            print(f"\n Original Prompt ({len(prompt_text.split())} words):")
            print("-"*70)
            print(prompt_text)
            print("-"*70)
        
        # Step 1: Parse prompt
        if output:
            print("\n Step 1: Parsing prompt structure...")
        
        original_structure, components = self.parser.parse(prompt_text)
        
        if output:
            print(f"  Components found:")
            print(f"    Instruction:  {original_structure.has_instruction}")
            print(f"    Examples:     {original_structure.has_examples}")
            print(f"    Constraints:  {original_structure.has_constraints}")
            print(f"    Style:        {original_structure.has_style}")
            print(f"    Context:      {original_structure.has_context}")
        
        # Step 2: Evaluate original
        if output:
            print(f"\n Step 2: Evaluating original prompt...")
        
        original_score = self.evaluator.evaluate(original_structure)
        
        if output:
            print(f"  Original score: {original_score:.3f}")
        
        # Step 3: Optimise structure
        if output:
            print(f"\n  Step 3: Optimising structure...")
            print(f"  Running Bayesian Optimisation ({self.config.n_init + self.config.n_iterations} evaluations)...")
        
        optimisation_result = self.optimiser.optimise()
        
        best_structure = optimisation_result.best_structure
        
        if output:
            print(f"\n  Optimisation complete!")
            print(f"  Best score found: {optimisation_result.best_score:.3f}")
        
        # Step 4: Build compressed prompt
        if output:
            print(f"\n Step 4: Building compressed prompt...")
        
        compressed_text = self.builder.build(best_structure, components)
        
        # Step 5: Compute metrics
        original_tokens = len(prompt_text.split())
        compressed_tokens = len(compressed_text.split())
        tokens_saved = original_tokens - compressed_tokens
        compression_ratio = tokens_saved / original_tokens if original_tokens > 0 else 0
        performance_retention = optimisation_result.best_score / original_score if original_score > 0 else 0
        
        if output:
            print(f"\n{'='*70}")
            print(" RESULTS")
            print(f"{'='*70}")
            
            # Token Reduction (Big Visual Display)
            print(f"\n Token Reduction:")
            print(f"  {'Original tokens:':<25} {original_tokens:>6} tokens")
            print(f"  {'Compressed tokens:':<25} {compressed_tokens:>6} tokens")
            print(f"  {'─'*35}")
            print(f"  {'TOKENS SAVED:':<25} {tokens_saved:>6} tokens ({'↓' + str(int(compression_ratio * 100)) + '%'})")
            
            # Performance
            print(f"\n Performance:")
            print(f"  {'Original score:':<25} {original_score:>6.3f}")
            print(f"  {'Compressed score:':<25} {optimisation_result.best_score:>6.3f}")
            print(f"  {'Performance retained:':<25} {performance_retention:>6.1%}")
            
            # Efficiency Metric
            compression_efficiency = performance_retention / (1 - compression_ratio) if compression_ratio < 1 else 0
            
            print(f"\n Compression Efficiency:")
            print(f"  {'Quality per token saved:':<25} {compression_efficiency:>6.2f}")
            if compression_efficiency > 3.0:
                print(f"  {'Assessment:':<25} {'Excellent compression!'}")
            elif compression_efficiency > 2.0:
                print(f"  {'Assessment:':<25} {'Good compression'}")
            elif compression_efficiency > 1.0:
                print(f"  {'Assessment:':<25} {'Moderate compression'}")
            else:
                print(f"  {'Assessment:':<25} {'Poor compression'}")
            
            # Optimal Structure
            print(f"\n Optimal Structure:")
            print(f"  {'Instruction:':<18} {best_structure.has_instruction} (length: {best_structure.instruction_length:.2f})")
            print(f"  {'Examples:':<18} {best_structure.has_examples} (count: {int(best_structure.num_examples * 10)})")
            print(f"  {'Constraints:':<18} {best_structure.has_constraints}")
            print(f"  {'Style:':<18} {best_structure.has_style}")
            print(f"  {'Context:':<18} {best_structure.has_context}")
            print(f"  {'Ordering:':<18} {best_structure.component_ordering}")
            
            # Side-by-side comparison
            print(f"\n Compressed Prompt:")
            print("-"*70)
            print(compressed_text)
            print("-"*70)
            
            # Summary box
            print(f"\n{'┌' + '─'*68 + '┐'}")
            print(f"│ {'COMPRESSION SUMMARY':^66} │")
            print(f"│ {' '*66} │")
            print(f"│  {tokens_saved} tokens saved ({compression_ratio:.0%} reduction){' '*(66-len(f'{tokens_saved} tokens saved ({compression_ratio:.0%} reduction)'))} │")
            print(f"│  Performance retained: {performance_retention:.0%}{' '*(66-len(f'Performance retained: {performance_retention:.0%}'))} │")
            print(f"│  Evaluations used: {optimisation_result.total_evaluations}{' '*(66-len(f'Evaluations used: {optimisation_result.total_evaluations}'))} │")
            print(f"└{'─'*68}┘")
        
        # Return all details
        return {
            'original_text': prompt_text,
            'compressed_text': compressed_text,
            'original_structure': original_structure,
            'optimised_structure': best_structure,
            'components': components,
            'metrics': {
                'original_tokens': original_tokens,
                'compressed_tokens': compressed_tokens,
                'tokens_saved': tokens_saved,
                'compression_ratio': compression_ratio,
                'original_score': original_score,
                'compressed_score': optimisation_result.best_score,
                'performance_retention': performance_retention,
                'compression_efficiency': compression_efficiency,
                'total_evaluations': optimisation_result.total_evaluations
            },
            'optimisation_result': optimisation_result
        }
