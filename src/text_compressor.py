"""
Reduces token count while preserving meaning
"""

import re
from typing import List

class TextCompressor:
    """
    Compress prompt text using multiple strategies
    """
    
    def __init__(self):
        pass
    
    def compress_instruction(self, instruction: str, aggressiveness: float = 0.5) -> str:
        """
        Compress instruction text
        
        Args:
            instruction: Original instruction text
            aggressiveness: 0-1, how much to compress (higher = more aggressive)
        
        Returns:
            Compressed instruction
        """
        # Remove filler words
        filler_words = ['please', 'very', 'really', 'quite', 'just', 'simply', 
                       'comprehensive', 'detailed', 'thorough', 'carefully']
        
        compressed = instruction
        
        if aggressiveness > 0.3:
            for word in filler_words:
                compressed = re.sub(rf'\b{word}\b', '', compressed, flags=re.IGNORECASE)
        
        # Remove phrases
        if aggressiveness > 0.5:
            phrases_to_remove = [
                'For example,', 'In other words,', 'Additionally,',
                'It is important to', 'Make sure to', 'Be sure to'
            ]
            for phrase in phrases_to_remove:
                compressed = compressed.replace(phrase, '')
        
        # Clean up extra spaces
        compressed = re.sub(r'\s+', ' ', compressed)
        compressed = re.sub(r'\s*,\s*', ', ', compressed)
        compressed = re.sub(r'\s*\.\s*', '. ', compressed)
        
        return compressed.strip()