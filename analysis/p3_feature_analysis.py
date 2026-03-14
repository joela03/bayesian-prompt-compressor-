"""
P3 Dataset Exploratory Analysis

Analyses BigScience P3 prompts to discover what makes them effective
Extracts data-driven insights for Bayesian optimisation
"""

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
