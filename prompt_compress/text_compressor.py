"""
Component-specific text compression for prompt optimization
"""
import re
from typing import List


class TextCompressor:
    """
    Compress text within prompt components while preserving meaning
    """
    
    @staticmethod
    def compress_instruction(text: str, aggressiveness: float = 0.5) -> str:
        """
        Compress instruction text by removing filler, condensing phrases
        
        Args:
            text: Original instruction text
            aggressiveness: 0-1, how much to compress (0=none, 1=maximum)
        
        Returns:
            Compressed instruction text
        """
        if aggressiveness < 0.1:
            return text  # Minimal compression, return as-is
        
        # Stage 1: Remove filler words (light compression)
        fillers = [
            'very', 'really', 'quite', 'just', 'actually', 'basically',
            'essentially', 'literally', 'simply', 'certainly', 'definitely'
        ]
        for filler in fillers:
            text = re.sub(rf'\b{filler}\b\s*', '', text, flags=re.IGNORECASE)
        
        # Stage 2: Replace verbose phrases (medium compression)
        if aggressiveness > 0.3:
            replacements = {
                r'in order to': 'to',
                r'due to the fact that': 'because',
                r'for the purpose of': 'for',
                r'is able to': 'can',
                r'has the ability to': 'can',
                r'specializing in': 'for',
                r'with the exception of': 'except',
                r'at this point in time': 'now',
                r'in the event that': 'if',
                r'on a regular basis': 'regularly',
            }
            
            for pattern, replacement in replacements.items():
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Stage 3: Aggressive condensing (high compression)
        if aggressiveness > 0.6:
            # Remove examples of what NOT to do (keep only positive guidance)
            text = re.sub(r'Avoid[^.]*\.', '', text, flags=re.IGNORECASE)
            
            # Condense bullet points
            text = re.sub(r'\d+\.\s+', ' ', text)  # Remove numbering, keep spacing
            text = re.sub(r'\s+', ' ', text).strip()  # Clean up
            text = re.sub(r'\s*-\s*', ', ', text)  # Convert bullets to commas
        
        # Stage 4: Clean up whitespace
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces → single space
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)  # Space before punctuation
        text = re.sub(r'([.,!?;:])\s*([.,!?;:])', r'\1', text)  # Double punctuation
        
        return text.strip()
    
    @staticmethod
    def compress_examples(examples: List[str], target_count: int = 2) -> List[str]:
        """
        Reduce number of examples to target count
        
        Strategy: Keep first and last for diversity, or evenly spaced
        
        Args:
            examples: List of example strings
            target_count: How many examples to keep
        
        Returns:
            Subset of examples
        """
        if len(examples) <= target_count:
            return examples  # Already at or below target
        
        if target_count == 0:
            return []
        
        if target_count == 1:
            return [examples[0]]  # Keep first example
        
        if target_count == 2:
            return [examples[0], examples[-1]]  # First and last
        
        # For target_count > 2: Keep evenly spaced examples
        indices = []
        step = len(examples) / target_count
        for i in range(target_count):
            index = int(i * step)
            indices.append(index)
        
        return [examples[i] for i in indices]
    
    @staticmethod
    def compress_constraints(text: str, aggressiveness: float = 0.5) -> str:
        """
        Compress constraints while ALWAYS preserving critical warnings
        
        Critical phrases (MUST keep):
        - DO NOT, MUST, CRITICAL, REQUIRED, NEVER, STRICTLY
        
        Args:
            text: Original constraints text
            aggressiveness: 0-1, how much to compress non-critical parts
        
        Returns:
            Compressed constraints with critical warnings intact
        """
        # Critical markers that indicate must-keep content
        critical_markers = [
            'DO NOT', 'MUST', 'CRITICAL', 'REQUIRED', 
            'NEVER', 'STRICTLY', 'COMPLETELY', 'ALWAYS',
            'FORBIDDEN', 'PROHIBITED', 'MANDATORY'
        ]
        
        # Split into sentences
        sentences = re.split(r'[.!?]\s*', text)
        
        critical_sentences = []
        regular_sentences = []
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            # Check if sentence contains critical markers
            is_critical = any(
                marker in sentence.upper() 
                for marker in critical_markers
            )
            
            if is_critical:
                critical_sentences.append(sentence.strip())
            else:
                regular_sentences.append(sentence.strip())
        
        # Compress regular sentences based on aggressiveness
        compressed_regular = []
        
        if aggressiveness < 0.5:
            # Keep most regular sentences
            keep_count = int(len(regular_sentences) * 0.7)
            compressed_regular = regular_sentences[:keep_count]
        elif aggressiveness < 0.8:
            # Keep some regular sentences
            keep_count = min(3, len(regular_sentences))
            compressed_regular = regular_sentences[:keep_count]
        else:
            # Keep only 1-2 regular sentences
            keep_count = min(2, len(regular_sentences))
            compressed_regular = regular_sentences[:keep_count]
        
        # Combine: ALL critical + selected regular
        all_sentences = critical_sentences + compressed_regular
        
        if not all_sentences:
            return ""
        
        # Join sentences back together
        result = '. '.join(all_sentences)
        
        # Ensure proper ending punctuation
        if result and not result.endswith('.'):
            result += '.'
        
        return result
    
    @staticmethod
    def compress_by_word_count(text: str, target_words: int) -> str:
        """
        Compress text to approximately target word count
        
        Args:
            text: Original text
            target_words: Target word count
        
        Returns:
            Compressed text
        """
        words = text.split()
        current_count = len(words)
        
        if current_count <= target_words:
            return text  # Already short enough
        
        # Calculate compression ratio needed
        ratio = target_words / current_count
        aggressiveness = 1 - ratio  # If need 50% of words, aggressiveness = 0.5
        
        # Use instruction compression (general purpose)
        return TextCompressor.compress_instruction(text, aggressiveness)
    @staticmethod
    def compress_style(text: str, aggressiveness: float = 0.7) -> str:
        """
        Compress style text (style is often unnecessary)
        
        Args:
            text: Original style text
            aggressiveness: How much to compress
        
        Returns:
            Compressed style
        """
        compressed = text
        
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

    @staticmethod
    def compress_context(text: str, aggressiveness: float = 0.6) -> str:
        """
        Compress context text
        
        Args:
            context: Original context text
            aggressiveness: How much to compress
        
        Returns:
            Compressed context
        """
        compressed = text
        
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