"""
Parse real text prompts into PromptStructure
"""

import re
from typing import Dict, List, Tuple
from src.encoders import PromptStructure
from src.text_compressor import TextCompressor

class PromptParser:
    """
    Convert text prompt -> PromptStructure

    Identifies components:
    - Instruction
    - Example
    - Constraints
    - Style
    - Context
    """

    def __init__(self):
        self.instruction_keywords = [
            'write', 'generate', 'create', 'provide', 'explain',
            'summarise', 'answer', 'describe', 'analyse', 'you are'
        ]

        self.example_keywords = [
            'for example', 'e.g.', 'such as', 'for instance',
            'example:', 'examples', '1.', '2.', '3.'
        ]

        self.constraint_keywords = [
            'must', 'should', 'do not', 'avoid', 'ensure',
            'requirements:', 'constraints:', 'rules:'
        ]

        self.style_keywords = [
            'tone', 'style', 'voice', 'format', 'profressional',
            'casual', 'format', 'academic'
        ]

        self.context_keywords = [
            'context:', 'background:', 'note:', 'setting',
            'you are a', 'you are an', 'your role'
        ]
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitter
        text = text.strip()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _classify_sentence(self, sentence: str) -> str:
        """Classify a sentence into component type"""
        sentence_lower = sentence.lower()
        
        # Check each type (priority order matters!)
        if any(kw in sentence_lower for kw in self.example_keywords):
            return 'examples'
        
        if any(kw in sentence_lower for kw in self.constraint_keywords):
            return 'constraints'
        
        if any(kw in sentence_lower for kw in self.style_keywords):
            return 'style'
        
        if any(kw in sentence_lower for kw in self.context_keywords):
            return 'context'
        
        if any(kw in sentence_lower for kw in self.instruction_keywords):
            return 'instruction'
        
        # Default: treat as instruction
        return 'instruction'
    
    def _determine_ordering(self, text: str, components: Dict) -> List[int]:
        """
        Determine order of components in original text
        
        Returns ordering: [position of inst, pos of examples, ...]
        """
        # Find first occurrence of each component type
        positions = {}
        
        # Map component type to position code
        comp_map = {
            'instruction': 1,
            'examples': 2,
            'constraints': 3,
            'style': 4,
            'context': 5
        }
        
        for comp_type, sentences in components.items():
            if sentences:
                # Find position of first sentence of this type
                first_sentence = sentences[0]
                pos = text.lower().find(first_sentence.lower())
                if pos >= 0:
                    positions[comp_map[comp_type]] = pos
        
        # Sort by position
        sorted_comps = sorted(positions.items(), key=lambda x: x[1])
        ordering = [comp for comp, _ in sorted_comps]
        
        # Fill in missing components at end
        for comp_code in [1, 2, 3, 4, 5]:
            if comp_code not in ordering:
                ordering.append(comp_code)
        
        return ordering

    def parse(self, prompt_text:str) -> Tuple[PromptStructure, Dict]:
        """
        Parse text prompt into structure

        Returns:
            structure: PromptStructure object
            components: Dict with text of each component
        """

        # Split into sentences
        sentences = self._split_sentences(prompt_text)

        # Classify each sentence
        components = {
            'instruction': [],
            'examples': [],
            'constraints': [],
            'style': [],
            'context': []
        }
        
        for sentence in sentences:
            comp_type = self._classify_sentence(sentence)
            components[comp_type].append(sentence)
        
        # Count tokens (rough approximation)
        total_tokens = len(prompt_text.split())
        
        # Compute normalised values
        num_examples = len(components['examples'])
        instruction_length = sum(len(s.split()) for s in components['instruction'])
        
        # Determine ordering from original prompt
        ordering = self._determine_ordering(prompt_text, components)
        
        # Create structure
        structure = PromptStructure(
            has_instruction=len(components['instruction']) > 0,
            has_examples=len(components['examples']) > 0,
            has_constraints=len(components['constraints']) > 0,
            has_style=len(components['style']) > 0,
            has_context=len(components['context']) > 0,
            
            num_examples=min(num_examples / 10.0, 1.0),
            instruction_length=min(instruction_length / 200.0, 1.0),
            total_tokens=min(total_tokens / 500.0, 1.0),
            
            component_ordering=ordering
        )
        
        return structure, components

class PromptBuilder:
    """
    Convert PromptStructure → text prompt
    
    Opposite of PromptParser
    """
    
    def __init__(self):
        """Initialize with TextCompressor"""
        self.compressor = TextCompressor()
    
    def build(self, structure: PromptStructure, components: Dict) -> str:
        """
        Build prompt with ACTUAL compression using TextCompressor
        
        Args:
            structure: PromptStructure with optimization results
            components: Dict of component texts from parser
        
        Returns:
            Compressed prompt text
        """
        sections = []
        
        comp_names = {1: 'instruction', 2: 'examples', 3: 'constraints', 
                      4: 'style', 5: 'context'}
        
        for pos in structure.component_ordering:
            comp_name = comp_names[pos]
            include = getattr(structure, f'has_{comp_name}')
            
            if not include or comp_name not in components:
                continue
            
            comp_data = components[comp_name]
            if not comp_data:
                continue
            
            if comp_name == 'instruction':
                # Get text
                if isinstance(comp_data, list):
                    text = ' '.join(comp_data)
                else:
                    text = comp_data
                
                aggressiveness = 1 - structure.instruction_length
                
                compressed = self.compressor.compress_instruction(text, aggressiveness=aggressiveness)
                sections.append(compressed)
            
            elif comp_name == 'examples':
                target_count = max(1, int(structure.num_examples * 5))
                
                if isinstance(comp_data, list):
                    compressed = self.compressor.compress_examples(comp_data, target_count)
                    if compressed:
                        sections.append('\n'.join(compressed))
                else:
                    compressed = self.compressor.compress_examples([comp_data], target_count)
                    if compressed:
                        sections.append(compressed[0])
            
            elif comp_name == 'constraints':
                if isinstance(comp_data, list):
                    text = '\n'.join(comp_data)
                else:
                    text = comp_data
                
                compressed = self.compressor.compress_constraints(text, aggressiveness=0.5)
                if compressed:
                    sections.append(compressed)
            
            elif comp_name == 'style':
                if isinstance(comp_data, list):
                    text = ' '.join(comp_data)
                else:
                    text = comp_data
                
                compressed = self.compressor.compress_style(text, aggressiveness=0.8)
                if compressed:
                    sections.append(compressed)
            
            elif comp_name == 'context':
                if isinstance(comp_data, list):
                    text = ' '.join(comp_data)
                else:
                    text = comp_data
                
                compressed = self.compressor.compress_context(text, aggressiveness=0.6)
                if compressed:
                    sections.append(compressed)
        
        return '\n'.join(sections)
    
    def _compress_text(self, text: str, ratio: float) -> str:
        """
        Simple text compression (keep first and last sentences)
        """
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 2:
            return text
        
        n_keep = max(1, int(len(sentences) * ratio))
        
        # Keep first and last
        if n_keep == 1:
            kept = [sentences[0]]
        else:
            kept = [sentences[0]] + [sentences[-1]]
        
        return '. '.join(kept) + '.'

if __name__ == "__main__":
    # Example prompt
    test_prompt = """
You are an expert AI assistant helping with research questions.

When answering questions, please follow these guidelines:
1. Be thorough and comprehensive
2. Cite sources when possible
3. Use clear, professional language

For example, a good response starts with a direct answer, then provides supporting evidence.

Constraints: Keep responses under 500 words. Avoid speculation.

Style: Use an academic but accessible tone.
"""
    
    print("="*70)
    print("PROMPT PARSER TEST")
    print("="*70)
    
    print("\nOriginal Prompt:")
    print("-"*70)
    print(test_prompt)
    print("-"*70)
    
    # Parse
    parser = PromptParser()
    structure, components = parser.parse(test_prompt)
    
    print("\nParsed Structure:")
    print(f"  has_instruction: {structure.has_instruction}")
    print(f"  has_examples: {structure.has_examples}")
    print(f"  has_constraints: {structure.has_constraints}")
    print(f"  has_style: {structure.has_style}")
    print(f"  has_context: {structure.has_context}")
    print(f"  num_examples: {structure.num_examples:.2f}")
    print(f"  total_tokens: {structure.total_tokens:.2f}")
    print(f"  ordering: {structure.component_ordering}")
    
    print("\nComponent Texts:")
    for comp_type, sentences in components.items():
        if sentences:
            print(f"\n{comp_type.upper()}:")
            for sentence in sentences:
                print(f"  - {sentence}")
    
    # Rebuild
    print("\n" + "="*70)
    builder = PromptBuilder()
    rebuilt = builder.build(structure, components)
    
    print("\nRebuilt Prompt:")
    print("-"*70)
    print(rebuilt)
    print("-"*70)
    
    print("\n✅ Parser and Builder work!")