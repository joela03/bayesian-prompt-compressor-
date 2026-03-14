"""
Aggressive text compression utilities

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
        compressed = instruction
        
        #  Remove common filler words (aggressiveness > 0.3)
        if aggressiveness > 0.3:
            filler_words = [
                'please', 'very', 'really', 'quite', 'just', 'simply',
                'comprehensive', 'detailed', 'thorough', 'carefully',
                'well-researched', 'relevant'
            ]
            
            for word in filler_words:
                # Use word boundaries to avoid partial matches
                compressed = re.sub(rf'\b{word}\b,?\s*', '', compressed, flags=re.IGNORECASE)
        
        # Remove filler phrases (aggressiveness > 0.5)
        if aggressiveness > 0.5:
            phrases_to_remove = [
                'For example,',
                'In other words,', 
                'Additionally,',
                'It is important to',
                'Make sure to',
                'Be sure to',
                'Please note that'
            ]
            for phrase in phrases_to_remove:
                compressed = re.sub(rf'{phrase}\s*', '', compressed, flags=re.IGNORECASE)
        
        # Simplify complex phrases (aggressiveness > 0.6)
        if aggressiveness > 0.6:
            replacements = {
                r'\bprovide\s+(\w+),?\s+and\s+(\w+)\b': r'provide \1 \2',  # "provide X and Y" → "provide X Y"
                r'\bwith\s+clear,?\s+': r'with ',  # "with clear, professional" → "with professional"
                r'\band\s+comprehensive\b': '',  # Remove "and comprehensive"
            }
            for pattern, replacement in replacements.items():
                compressed = re.sub(pattern, replacement, compressed, flags=re.IGNORECASE)
        
        # Clean up spacing and punctuation
        compressed = re.sub(r'\s+', ' ', compressed)  # Multiple spaces → single space
        compressed = re.sub(r'\s*,\s*', ', ', compressed)  # Fix comma spacing
        compressed = re.sub(r',\s*,', ',', compressed)  # Remove double commas
        compressed = re.sub(r'\s*\.\s*', '. ', compressed)  # Fix period spacing
        compressed = re.sub(r',\s*\.', '.', compressed)  # Remove comma before period
        compressed = re.sub(r'\s+([.,!?])', r'\1', compressed)  # Remove space before punctuation
        
        # Ensure starts with capital letter
        compressed = compressed.strip()
        if compressed and compressed[0].islower():
            compressed = compressed[0].upper() + compressed[1:]
        
        # Ensure ends with period if original did
        if instruction.strip().endswith('.') and not compressed.endswith('.'):
            compressed += '.'
        
        return compressed
    
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
        
        # Shorten each example - keep only first sentence
        compressed = []
        for ex in kept:
            # Remove "For example," prefix if exists
            ex = re.sub(r'^For example,?\s*', '', ex, flags=re.IGNORECASE)
            ex = re.sub(r'^Example:?\s*', '', ex, flags=re.IGNORECASE)
            
            # Split by period and keep first sentence
            sentences = [s.strip() for s in ex.split('.') if s.strip()]
            if sentences:
                first_sentence = sentences[0]
                # Remove filler from example
                first_sentence = re.sub(r'\b(please|carefully|make sure to)\b', '', first_sentence, flags=re.IGNORECASE)
                first_sentence = re.sub(r'\s+', ' ', first_sentence).strip()
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
        # Extract bullet points - handle multiple formats
        # Match: "- text", "• text", "* text", or "1. text", "2) text"
        bullets = re.findall(r'^[\s]*[-•*]\s*(.+?)$', constraints, re.MULTILINE)
        numbers = re.findall(r'^[\s]*\d+[.)]\s*(.+?)$', constraints, re.MULTILINE)
        
        items = bullets + numbers
        
        if items:
            # Compress each item
            compressed_items = []
            for item in items:
                # Remove common filler words
                filler = ['please', 'make sure to', 'be sure to', 'always', 'do not', 'should not']
                compressed_item = item
                
                for phrase in filler:
                    compressed_item = re.sub(rf'\b{phrase}\b', '', compressed_item, flags=re.IGNORECASE)
                
                # Clean up
                compressed_item = re.sub(r'\s+', ' ', compressed_item).strip()
                
                # Only keep if not empty and actually different
                if compressed_item and len(compressed_item) > 5:
                    compressed_items.append(compressed_item)
            
            # If aggressive, take only first half of items
            if aggressiveness > 0.6 and len(compressed_items) > 2:
                n = max(2, len(compressed_items) // 2)
                compressed_items = compressed_items[:n]
            
            return '\n'.join(f"- {item}" for item in compressed_items)
        
        # No bullets found, just remove filler from whole text
        compressed = constraints
        if aggressiveness > 0.4:
            filler = ['please', 'should', 'must', 'always', 'never', 'make sure to', 'be sure to']
            for phrase in filler:
                compressed = re.sub(rf'\b{phrase}\b', '', compressed, flags=re.IGNORECASE)
        
        compressed = re.sub(r'\s+', ' ', compressed).strip()
        return compressed
    
    def compress_style(self, style: str, aggressiveness: float = 0.7) -> str:
        """
        Compress style text (style is often unnecessary)
        
        Args:
            style: Original style text
            aggressiveness: How much to compress
        
        Returns:
            Compressed style
        """
        compressed = style
        
        # Remove common filler
        filler = ['Use', 'Be', 'Maintain', 'Keep', 'Ensure', 'Style:']
        for word in filler:
            compressed = re.sub(rf'\b{word}\b:?\s*', '', compressed, flags=re.IGNORECASE)
        
        # Shorten to just keywords
        if aggressiveness > 0.6:
            # Extract key adjectives
            keywords = re.findall(
                r'\b(professional|clear|concise|academic|accessible|formal|informal|friendly|technical)\b',
                compressed,
                flags=re.IGNORECASE
            )
            if keywords:
                # Remove duplicates, keep order
                seen = set()
                unique_keywords = []
                for k in keywords:
                    k_lower = k.lower()
                    if k_lower not in seen:
                        seen.add(k_lower)
                        unique_keywords.append(k_lower)
                compressed = ', '.join(unique_keywords)
        
        compressed = re.sub(r'\s+', ' ', compressed).strip()
        return compressed
    
    def compress_context(self, context: str, aggressiveness: float = 0.6) -> str:
        """
        Compress context text
        
        Args:
            context: Original context text
            aggressiveness: How much to compress
        
        Returns:
            Compressed context
        """
        compressed = context
        
        # Remove "Context:" prefix
        compressed = re.sub(r'^Context:\s*', '', compressed, flags=re.IGNORECASE)
        
        # Remove common filler phrases
        filler_phrases = [
            'You are',
            'This is for',
            'The audience is',
            'assisting with',
            'helping with',
            'working on'
        ]
        
        for phrase in filler_phrases:
            compressed = re.sub(rf'\b{phrase}\b\s*', '', compressed, flags=re.IGNORECASE)
        
        # Clean up
        compressed = re.sub(r'\s+', ' ', compressed).strip()
        
        # Ensure starts with capital
        if compressed and compressed[0].islower():
            compressed = compressed[0].upper() + compressed[1:]
        
        return compressed


# Test the compressor
if __name__ == "__main__":
    compressor = TextCompressor()
    
    print("="*70)
    print("TEXT COMPRESSOR TEST")
    print("="*70)
    
    # Test 1: Instruction compression
    original = "Please provide thorough, well-researched, and comprehensive answers with clear, professional academic language."
    compressed = compressor.compress_instruction(original, aggressiveness=0.7)
    
    print(f"\n1. Instruction Compression:")
    print(f"   Original ({len(original.split())} words):")
    print(f"   {original}")
    print(f"\n   Compressed ({len(compressed.split())} words):")
    print(f"   {compressed}")
    print(f"   Reduction: {(1 - len(compressed.split())/len(original.split())):.1%}")
    
    # Test 2: Examples compression
    examples = [
        "For example, a good response begins with a direct answer to the question, followed by supporting evidence.",
        "Another example would be to provide relevant context and acknowledgment of any limitations.",
        "A third example shows how to structure your response logically with clear sections."
    ]
    compressed_ex = compressor.compress_examples(examples, target_count=2)
    
    print(f"\n2. Examples Compression:")
    print(f"   Original: {len(examples)} examples, {sum(len(e.split()) for e in examples)} words")
    for i, ex in enumerate(examples):
        print(f"     Ex{i+1}: {ex[:60]}...")
    print(f"\n   Compressed: {len(compressed_ex)} examples, {sum(len(e.split()) for e in compressed_ex)} words")
    for i, ex in enumerate(compressed_ex):
        print(f"     Ex{i+1}: {ex}")
    print(f"   Reduction: {(1 - sum(len(e.split()) for e in compressed_ex)/sum(len(e.split()) for e in examples)):.1%}")
    
    # Test 3: Constraints compression
    constraints = """- Keep responses under 300 words
- Maintain an objective, balanced tone
- Avoid speculation or unsupported claims
- Do not make recommendations without sufficient evidence"""
    
    compressed_const = compressor.compress_constraints(constraints, aggressiveness=0.6)
    
    print(f"\n3. Constraints Compression:")
    print(f"   Original ({len(constraints.split())} words):")
    for line in constraints.split('\n'):
        print(f"     {line}")
    print(f"\n   Compressed ({len(compressed_const.split())} words):")
    for line in compressed_const.split('\n'):
        print(f"     {line}")
    print(f"   Reduction: {(1 - len(compressed_const.split())/len(constraints.split())):.1%}")
    
    # Test 4: Full prompt compression
    print(f"\n4. Full Prompt Test:")
    full_original = """Please provide thorough, well-researched answers with clear professional language.
For example, start with a direct answer.
- Keep responses under 300 words
- Avoid speculation
Style: Professional and accessible"""
    
    # Simulate component-by-component compression
    lines = full_original.split('\n')
    full_compressed = []
    
    # Line 1: instruction
    full_compressed.append(compressor.compress_instruction(lines[0], 0.7))
    # Line 2: example
    full_compressed.append(compressor.compress_examples([lines[1]], 1)[0])
    # Lines 3-4: constraints
    const = '\n'.join(lines[2:4])
    full_compressed.append(compressor.compress_constraints(const, 0.6))
    # Line 5: style
    full_compressed.append(compressor.compress_style(lines[4], 0.8))
    
    full_compressed_text = '\n'.join(full_compressed)
    
    print(f"   Original ({len(full_original.split())} words):")
    print(f"   {full_original}")
    print(f"\n   Compressed ({len(full_compressed_text.split())} words):")
    print(f"   {full_compressed_text}")
    print(f"   Reduction: {(1 - len(full_compressed_text.split())/len(full_original.split())):.1%}")
    
    print("\n" + "="*70)
    print("TextCompressor works!")