"""
Parse real text prompts into PromptStructure
"""

import real
from typing import Dict, List, Tuple
from encoders import PromptStructure

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
        pass
    
    def build(self, structure: PromptStructure, components: Dict) -> str:
        """
        Build text prompt from structure and components
        
        Args:
            structure: PromptStructure specifying what to include
            components: Dict with actual text for each component
        
        Returns:
            prompt_text: Reconstructed prompt
        """
        # Map ordering codes to component names
        comp_names = {
            1: 'instruction',
            2: 'examples',
            3: 'constraints',
            4: 'style',
            5: 'context'
        }
        
        # Build prompt in specified order
        sections = []
        
        for pos in structure.component_ordering:
            comp_name = comp_names[pos]
            
            # Check if this component should be included
            include = getattr(structure, f'has_{comp_name}')
            
            if include and comp_name in components and components[comp_name]:
                # Add component text
                comp_text = ' '.join(components[comp_name])
                
                # Apply compression based on structure parameters
                if comp_name == 'examples':
                    # Limit number of examples
                    n_examples_keep = int(structure.num_examples * 10)
                    if n_examples_keep < len(components[comp_name]):
                        comp_text = ' '.join(components[comp_name][:n_examples_keep])
                
                elif comp_name == 'instruction':
                    # Potentially shorten instruction
                    if structure.instruction_length < 0.5:
                        # Make more concise
                        comp_text = self._compress_text(comp_text, ratio=structure.instruction_length)
                
                sections.append(comp_text)
        
        # Join sections
        prompt_text = '\n\n'.join(sections)
        
        return prompt_text
    
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