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
