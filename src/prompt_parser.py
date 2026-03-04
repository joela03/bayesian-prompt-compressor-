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
        