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