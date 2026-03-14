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

    def compress_examples(self, examples: List[str], target_count: int = 2) -> List[str]:
        """
        Reduce number and length of examples
        
        Args:
            examples: List of example texts
            target_count: How many examples to keep
        
        Returns:
            Compressed examples
        """
            # Keep only target_count examples
            kept = examples[:target_count]
            
            compressed = []
            for ex in kept:
                # Split by period and keep first sentence
                sentences = ex.split('.')
                if sentences:
                    first_sentence = sentences[0].strip()
                    if first_sentence:
                        compressed.append(first_sentence + '.')
            
            return compressed

    def compress_constraints(self, constraints: str, aggressiveness: float = 0.5) -> str:
        """
        Compress constraints to keywords
        
        Args:
            constraints: Original constraints text
            aggressiveness: How much to compress
        
        Returns:
            Compressed constraints
        """
        # Extract bullet points or numbered items
        bullets = re.findall(r'[-•]\s*(.+?)(?:\n|$)', constraints)
        numbers = re.findall(r'\d+[.)]\s*(.+?)(?:\n|$)', constraints)
        
        items = bullets + numbers
        
        if items:
            # Compress each item
            compressed = []
            for item in items:
                # Remove filler
                item = re.sub(r'\b(please|should|must|always|never)\b', '', item, flags=re.IGNORECASE)
                item = item.strip()
                if len(item) > 0:
                    compressed.append(item)
            
            # If aggressive, take only first half
            if aggressiveness > 0.6 and len(compressed) > 2:
                n = max(2, len(compressed) // 2)
                compressed = compressed[:n]
            
            return '\n'.join(f"- {item}" for item in compressed)
        
        # No bullets found, just remove filler
        compressed = constraints
        if aggressiveness > 0.4:
            filler = ['please', 'should', 'must', 'always', 'never']
            for word in filler:
                compressed = re.sub(rf'\b{word}\b', '', compressed, flags=re.IGNORECASE)
        
        compressed = re.sub(r'\s+', ' ', compressed)
        return compressed.strip()

    def compress_context(self, context: str, aggressiveness: float = 0.6) -> str:
        """
        Compress context text
        
        Args:
            context: Original context text
            aggressiveness: How much to compress
        
        Returns:
            Compressed context
        """
        # Context can be moderately compressed
        compressed = context
        
        # Remove "Context:" prefix
        compressed = re.sub(r'^Context:\s*', '', compressed, flags=re.IGNORECASE)
        
        # Remove filler
        filler = ['You are', 'This is for', 'The audience is']
        for phrase in filler:
            compressed = re.sub(rf'^{phrase}\s+', '', compressed, flags=re.IGNORECASE)
        
        # Keep just the core info
        if aggressiveness > 0.5:
            # Extract key nouns
            compressed = re.sub(r'\b(assisting with|helping with|working on)\b', '', compressed, flags=re.IGNORECASE)
        
        compressed = re.sub(r'\s+', ' ', compressed)
        return compressed.strip()
