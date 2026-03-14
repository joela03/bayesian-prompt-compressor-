"""
P3 Dataset Exploratory Analysis

Analyses BigScience P3 prompts to discover what makes them effective
Extracts data-driven insights for Bayesian optimisation
"""

from datasets import load_dataset
from src.prompt_parser import PromptParser
from src.encoders import PromptEncoder

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
