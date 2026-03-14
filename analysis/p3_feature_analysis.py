"""
P3 Dataset Exploratory Analysis

Analyses BigScience P3 prompts to discover what makes them effective
Extracts data-driven insights for Bayesian optimisation
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets import load_dataset
from src.prompt_parser import PromptParser
from src.encoders import PromptEncoder
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

class P3PromptAnalyser:
    """
    Analyse P3 prompts to discover patterns
    """
    
    def __init__(self):
        self.parser = PromptParser()
        self.encoder = PromptEncoder()
        self.results = []
    
    def load_p3_subset(self, subset_name, n_samples=50):
        """
        Load a P3 subset and extract prompt features
        
        Args:
            subset_name: Name of P3 subset
            n_samples: How many prompts to analyse
        
        Returns:
            List of prompt samples with features
        """
        print(f"Loading {subset_name}...")
        
        try:
            dataset = load_dataset("bigscience/P3", subset_name, split="train")
        except Exception as e:
            print(f"Error loading {subset_name}: {e}")
            return []
        
        samples = []
        for i in range(min(n_samples, len(dataset))):
            example = dataset[i]
            
            # Get the prompt template
            prompt = example.get('inputs_pretokenised', '')
            
            if not prompt or len(prompt) < 10:
                continue
            
            samples.append({
                'id': f'{subset_name}_{i}',
                'prompt': prompt,
                'subset': subset_name,
            })
        
        print(f"  Loaded {len(samples)} prompts")
        return samples

    def extract_features(self, prompt_text):
        """
        Extract all features from a prompt
        
        Returns:
            Dict of features
        """
        # Parse structure
        components = self.parser.parse(prompt_text)
        structure = self.parser.infer_structure(components)
        
        # Get vector encoding
        vector = self.encoder.encode(structure)
        
        # Build feature dict
        features = {
            # Binary features
            'has_instruction': structure.has_instruction,
            'has_examples': structure.has_examples,
            'has_constraints': structure.has_constraints,
            'has_style': structure.has_style,
            'has_context': structure.has_context,
            
            # Continuous features
            'num_examples': structure.num_examples,
            'instruction_length': structure.instruction_length,
            'total_tokens': structure.total_tokens,
            
            # Derived features
            'prompt_length': len(prompt_text),
            'word_count': len(prompt_text.split()),
            'num_sentences': prompt_text.count('.') + prompt_text.count('!') + prompt_text.count('?'),
            
            # Complexity
            'num_components': sum([
                structure.has_instruction,
                structure.has_examples,
                structure.has_constraints,
                structure.has_style,
                structure.has_context
            ])
        }
        
        return features

    def analyse_dataset(self, subset_names, n_per_subset=50):
        """
        Analyse multiple P3 subsets
        
        Args:
            subset_names: List of subset names to analyse
            n_per_subset: Number of prompts per subset
        
        Returns:
            DataFrame with all features
        """
        all_data = []
        
        for subset in subset_names:
            samples = self.load_p3_subset(subset, n_per_subset)
            
            for sample in samples:
                try:
                    features = self.extract_features(sample['prompt'])
                    
                    row = {
                        'id': sample['id'],
                        'subset': subset,
                        'prompt': sample['prompt'],
                        **features
                    }
                    
                    all_data.append(row)
                except Exception as e:
                    print(f"Error extracting features: {e}")
        
        return pd.DataFrame(all_data)

    def length_analysis(self, df):
        """
        Analyse prompt length distributions
        """
        print("LENGTH DISTRIBUTION ANALYSIS")
        
        print(f"\nPrompt Length Statistics:")
        print(f"  Mean:   {df['word_count'].mean():.1f} words")
        print(f"  Median: {df['word_count'].median():.1f} words")
        print(f"  Min:    {df['word_count'].min():.0f} words")
        print(f"  Max:    {df['word_count'].max():.0f} words")
        print(f"  Std:    {df['word_count'].std():.1f} words")
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Word count distribution
        axes[0].hist(df['word_count'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0].axvline(df['word_count'].median(), color='red', linestyle='--', 
                       linewidth=2, label=f"Median: {df['word_count'].median():.0f}")
        axes[0].set_xlabel('Word Count', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Prompt Length Distribution', fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Component count
        axes[1].hist(df['num_components'], bins=6, color='coral', alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Number of Components', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Component Count Distribution', fontsize=13, fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/results/p3_length_distributions.png', dpi=150)

    def example_analysis(self, df):
        """
        Analyse example usage patterns
        """
        print("/n EXAMPLE USAGE PATTERNS")
        
        with_examples = df[df['has_examples'] == True]
        
        if len(with_examples) == 0:
            print("No prompts with examples found")
            return
        
        print(f"\nPrompts with examples: {len(with_examples)} ({len(with_examples)/len(df):.1%})")
        print(f"Average num_examples:  {with_examples['num_examples'].mean():.2f}")
        print(f"Median num_examples:   {with_examples['num_examples'].median():.2f}")
        
        # Examples vs length
        plt.figure(figsize=(10, 6))
        plt.scatter(with_examples['num_examples'], with_examples['word_count'],
                   alpha=0.5, s=50, color='steelblue', edgecolors='black', linewidth=0.5)
        
        # Add correlation
        corr = with_examples[['num_examples', 'word_count']].corr().iloc[0, 1]
        
        plt.xlabel('Number of Examples (normalized)', fontsize=12)
        plt.ylabel('Total Word Count', fontsize=12)
        plt.title(f'Examples vs Prompt Length (correlation: {corr:.2f})', 
                 fontsize=13, fontweight='bold')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/results/p3_examples_vs_length.png', dpi=150)
        
        return corr

    def generate_findings(self, df):
        """
        Generate findings JSON for informed optimiser
        
        Returns:
            Dict of findings to use in BO
        """
        findings = {
            # Component importance (usage rates)
            'instruction_importance': df['has_instruction'].mean(),
            'examples_importance': df['has_examples'].mean(),
            'constraints_importance': df['has_constraints'].mean(),
            'style_importance': df['has_style'].mean(),
            'context_importance': df['has_context'].mean(),
            
            # Optimal values
            'optimal_num_examples': df[df['has_examples']]['num_examples'].median() if df['has_examples'].any() else 0.4,
            'optimal_instruction_length': df[df['has_instruction']]['instruction_length'].median() if df['has_instruction'].any() else 0.5,
            
            # Correlations (for kernel weighting)
            'example_length_correlation': df[['num_examples', 'word_count']].corr().iloc[0, 1] if 'num_examples' in df.columns else 0.5,
            
            # Statistics
            'mean_word_count': df['word_count'].mean(),
            'median_word_count': df['word_count'].median(),
            'mean_components': df['num_components'].mean(),
        }
        
        # Save to JSON
        import json
        with open('data/results/p3_findings.json', 'w') as f:
            json.dump(findings, f, indent=2)
        
        
        return findings

def run_p3_analysis():
    """
    Main analysis pipeline
    """
    print("="*70)
    print("P3 EXPLORATORY ANALYSIS")
    print("="*70)
    
    # Create output directory
    os.makedirs('data/results', exist_ok=True)
    
    # Initialize analyser
    analyser = P3PromptAnalyser()
    
    # Select diverse P3 subsets (classification tasks)
    subsets = [
        "super_glue_cb_GPT_3_style",
        "amazon_polarity_Is_this_product_review_positive",
        "app_reviews_categorize_rating_using_review",
    ]
    
    print(f"\n Analysing {len(subsets)} P3 subsets...")
    print(f"   Sampling 50 prompts per subset = {len(subsets) * 50} total\n")
    
    # Load and analyse
    df = analyser.analyse_dataset(subsets, n_per_subset=50)
    
    if len(df) == 0:
        print(" No data collected. Check internet connection.")
        return None
    
    print(f"\n Loaded {len(df)} prompts for analysis")
    
    # Run analyses
    print("\n Running analyses...")
    
    usage = analyser.component_analysis(df)
    analyser.length_analysis(df)
    corr = analyser.example_analysis(df)
    findings = analyser.generate_findings(df)
    
    # Save dataframe
    df.to_csv('data/results/p3_feature_data.csv', index=False)
    print(f"\n Saved feature data: data/results/p3_feature_data.csv")
    
    print("\n" + "="*70)
    print(" ANALYSIS COMPLETE")
    print("="*70)
    print("\nKey Findings:")
    print(f"  - Instruction usage: {findings['instruction_importance']:.1%}")
    print(f"  - Examples usage: {findings['examples_importance']:.1%}")
    print(f"  - Optimal num examples: {findings['optimal_num_examples']:.2f}")
    print(f"  - Example-length correlation: {findings['example_length_correlation']:.2f}")
    print(f"  - Mean prompt length: {findings['mean_word_count']:.0f} words")
    
    return df, findings


if __name__ == "__main__":
    df, findings = run_p3_analysis()