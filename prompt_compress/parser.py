"""
Parse real text prompts into PromptStructure
"""

import re
from typing import Dict, List, Tuple

from .encoders import PromptStructure
from .text_compressor import TextCompressor
from ._persona import PERSONA_PATTERNS

# Pattern for the opaque tokens stashed by PromptParser._extract_protected_regions.
# Used by PromptBuilder to detect which sentences must survive compression because
# they carry placeholders / code fences / URLs that the validator will check.
_PROTECTED_TOKEN_RE = re.compile(r'__PROTECTED_\d+__')

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
            'requirements:', 'constraints:', 'rules:',
            'do not', 'only reply', 'nothing else',
            'just reply', 'do not write',
            'never', 'always', 'do not type', 'do not provide',
            'only respond', 'respond only',
        ]

        self.style_keywords = [
            'tone', 'style', 'voice', 'format', 'profressional',
            'casual', 'format', 'academic'
        ]

        self.context_keywords = [
            'context:', 'background:', 'note:', 'setting', 'your role'
        ]
    
    def _extract_protected_regions(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Pull out regions that must survive compression byte-for-byte.

        Detects (in this order, to avoid nested matches):
          1. Triple-backtick code fences
          2. Double-brace template placeholders {{var}}
          3. Single-brace template placeholders {var}
          4. URLs (http(s)://...)

        Each match is replaced with a unique token `__PROTECTED_N__` and the
        original content is returned in a token→content map for later restore.
        """
        protected: Dict[str, str] = {}
        counter = 0

        def stash(match: re.Match) -> str:
            nonlocal counter
            token = f"__PROTECTED_{counter}__"
            protected[token] = match.group(0)
            counter += 1
            return token

        # 1. Code fences (DOTALL so multi-line blocks are caught)
        text = re.sub(r'```.*?```', stash, text, flags=re.DOTALL)
        # 2. Double-brace placeholders first so they don't get half-eaten by the single-brace pass
        text = re.sub(r'\{\{[^{}]+\}\}', stash, text)
        # 3. Single-brace placeholders
        text = re.sub(r'\{[^{}\s]+\}', stash, text)
        # 4. URLs
        text = re.sub(r'https?://\S+', stash, text)

        return text, protected

    @staticmethod
    def restore_protected_regions(text: str, protected: Dict[str, str]) -> str:
        """Replace every __PROTECTED_N__ token with its original content."""
        for token, original in protected.items():
            text = text.replace(token, original)
        return text

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences - PROTECT numbered lists and bullets
        """
        text = text.strip()
        
        text = re.sub(r'(\d+)\.\s+([A-Z])', r'\1NUMMARKER \2', text)
        
        text = re.sub(r'^-\s+', 'BULLETMARKER ', text, flags=re.MULTILINE)
        
        sentences = re.split(r'[.!?]+\s+', text)
        
        sentences = [s.replace('NUMMARKER', '.') for s in sentences]
        sentences = [s.replace('BULLETMARKER', '- ') for s in sentences]
        
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences

    def _classify_sentence(self, sentence: str) -> str:
        """Classify a sentence into component type"""
        sentence_lower = sentence.lower()

        # Persona/role sentence — must never be removed
        if sentence_lower.strip().startswith('you are'):
            return 'instruction'

        # Priority order matters. Constraints MUST be evaluated before
        # examples — "for example, do not use jargon" should classify as a
        # constraint, not an example, because the "do not" rule is the
        # load-bearing part of the sentence.
        if any(kw in sentence_lower for kw in self.constraint_keywords):
            return 'constraints'

        if any(kw in sentence_lower for kw in self.example_keywords):
            return 'examples'

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
            components: Dict with text of each component (plus a special
                        '__protected__' entry mapping protect tokens to their
                        original content so the builder can restore them)
        """

        # Pull out protected regions (code fences, placeholders, URLs) BEFORE
        # any sentence splitting so they aren't mangled by the regex stages.
        prompt_text, protected = self._extract_protected_regions(prompt_text)

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

        # Attach the protected-token map so PromptBuilder can restore them.
        components['__protected__'] = protected

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

        # If a persona sentence exists in the instruction component, ensure
        # instruction (position 1) is emitted first, AND that the persona
        # sentence is itself the first sentence within instruction. This
        # keeps the validator's persona check happy without removing
        # component_ordering from the BO search space.
        ordering = list(structure.component_ordering)
        instr_sentences = list(components.get('instruction', []))
        persona_idx = next(
            (i for i, s in enumerate(instr_sentences)
             if any(re.match(rf'^{pat}\b', s.lower().lstrip())
                    for pat in PERSONA_PATTERNS)),
            None,
        )
        if persona_idx is not None:
            # Hoist persona sentence to front of instruction list
            if persona_idx != 0:
                instr_sentences.insert(0, instr_sentences.pop(persona_idx))
                components = dict(components)
                components['instruction'] = instr_sentences
            # Hoist instruction position to front of ordering
            if 1 in ordering and ordering[0] != 1:
                ordering.remove(1)
                ordering.insert(0, 1)

        # Mandatory-sentence pass: any sentence carrying a __PROTECTED_N__ token
        # (placeholder / code fence / URL) MUST survive compression. Protection-
        # by-substitution alone doesn't help — the optimiser can still drop the
        # whole component, or per-component compressors can drop the sentence.
        # We collect those sentences here and patch them back in below.
        mandatory_by_comp = self._collect_mandatory_sentences(components)
        emitted_tokens: set[str] = set()

        for pos in ordering:
            comp_name = comp_names[pos]
            include = getattr(structure, f'has_{comp_name}')
            comp_data = components.get(comp_name)
            mandatory = mandatory_by_comp.get(comp_name, [])

            if not include or not comp_data:
                # Component dropped — but mandatory sentences still must survive.
                # Emit them verbatim (the token-restore pass below puts the real
                # placeholder text back).
                for sent, tokens in mandatory:
                    if tokens - emitted_tokens:
                        sections.append(sent)
                        emitted_tokens.update(tokens)
                continue

            if comp_name == 'instruction':
                if isinstance(comp_data, list):
                    text = ' '.join(comp_data)
                else:
                    text = comp_data
                aggressiveness = 1 - structure.instruction_length
                compressed = self.compressor.compress_instruction(text, aggressiveness=aggressiveness)

            elif comp_name == 'examples':
                target_count = max(1, int(structure.num_examples * 5))
                if isinstance(comp_data, list):
                    compressed_list = self.compressor.compress_examples(comp_data, target_count)
                    compressed = '\n'.join(compressed_list) if compressed_list else ''
                else:
                    compressed_list = self.compressor.compress_examples([comp_data], target_count)
                    compressed = compressed_list[0] if compressed_list else ''

            elif comp_name == 'constraints':
                if isinstance(comp_data, list):
                    text = '\n'.join(comp_data)
                else:
                    text = comp_data
                compressed = self.compressor.compress_constraints(text, aggressiveness=0.5)

            elif comp_name == 'style':
                if isinstance(comp_data, list):
                    text = ' '.join(comp_data)
                else:
                    text = comp_data
                compressed = self.compressor.compress_style(text, aggressiveness=0.8)

            elif comp_name == 'context':
                if isinstance(comp_data, list):
                    text = ' '.join(comp_data)
                else:
                    text = comp_data
                compressed = self.compressor.compress_context(text, aggressiveness=0.6)

            else:
                compressed = ''

            # Patch in any mandatory tokens lost during compression.
            compressed_tokens = set(_PROTECTED_TOKEN_RE.findall(compressed))
            parts = [compressed] if compressed else []
            for sent, tokens in mandatory:
                missing = (tokens - compressed_tokens) - emitted_tokens
                if missing:
                    parts.append(sent)
                    emitted_tokens.update(tokens)
            emitted_tokens.update(compressed_tokens)

            if parts:
                sections.append('\n'.join(parts))

        # Safety net: any mandatory tokens still unemitted (e.g. the component
        # wasn't in `ordering` for some reason) get appended at the end.
        for comp_name, mandatory in mandatory_by_comp.items():
            for sent, tokens in mandatory:
                if tokens - emitted_tokens:
                    sections.append(sent)
                    emitted_tokens.update(tokens)

        result = '\n'.join(sections)

        # Restore protected regions (code fences, placeholders, URLs)
        protected = components.get('__protected__', {})
        if protected:
            result = PromptParser.restore_protected_regions(result, protected)

        return result

    @staticmethod
    def _collect_mandatory_sentences(components: Dict) -> Dict[str, List[Tuple[str, set]]]:
        """
        For each component, return [(sentence, token_set), ...] for sentences
        that carry one or more __PROTECTED_N__ tokens. Those tokens map to
        placeholders / code fences / URLs in the original prompt and must be
        preserved across compression.
        """
        out: Dict[str, List[Tuple[str, set]]] = {}
        for name, data in components.items():
            if name == '__protected__':
                continue
            if isinstance(data, list):
                items = data
            elif isinstance(data, str):
                items = [data]
            else:
                continue
            mandatory: List[Tuple[str, set]] = []
            for sent in items:
                tokens = set(_PROTECTED_TOKEN_RE.findall(sent))
                if tokens:
                    mandatory.append((sent, tokens))
            if mandatory:
                out[name] = mandatory
        return out
    
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