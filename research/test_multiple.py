from prompt_compress import PromptCompressor

# Test prompts of different styles
test_prompts = {
    "simple_assistant": """You are a helpful AI assistant.

When answering questions, provide clear and accurate information.
Use a friendly, professional tone.

For example, good answers are concise and well-structured.

Keep responses under 200 words.""",

    "technical_writer": """You are an expert technical writer helping with documentation.

Guidelines:
1. Write in clear, simple language
2. Use active voice
3. Include code examples when relevant
4. Break complex topics into steps

Constraints:
- Avoid jargon unless necessary
- Maximum 300 words per response
- Use markdown formatting

Style: Professional but approachable.""",

    "research_assistant": """You are an expert AI assistant helping with research questions.

When answering questions, please follow these comprehensive guidelines:
1. Provide thorough, well-researched answers
2. Include relevant citations and sources when available  
3. Use clear, professional academic language
4. Structure your response logically with clear sections

For example, a good response begins with a direct answer to the question, followed by supporting evidence, relevant context, and acknowledgment of any limitations or alternative perspectives.

Constraints:
- Keep responses under 300 words
- Maintain an objective, balanced tone
- Avoid speculation or unsupported claims
- Do not make recommendations without sufficient evidence

Style: Use an academic but accessible writing style. Be precise and professional while remaining approachable.

Context: You are assisting with academic research and your responses should meet scholarly standards."""
}

print("="*70)
print("TESTING ON MULTIPLE PROMPT TYPES")
print("="*70)

compressor = PromptCompressor(use_real_evaluator=False)

results = {}

for name, prompt in test_prompts.items():
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"{'='*70}")
    print(f"Original length: {len(prompt.split())} words")
    
    result = compressor.compress(prompt, output=False)
    
    # Store results
    results[name] = {
        'original_words': result['metrics']['original_tokens'],
        'compressed_words': result['metrics']['compressed_tokens'],
        'compression_rate': result['metrics']['compression_ratio'],
        'quality_retained': result['metrics']['performance_retention'],
        'compressed_text': result['compressed_text']
    }
    
    # Print summary
    print(f"\n📊 Results:")
    print(f"   Original:    {results[name]['original_words']} words")
    print(f"   Compressed:  {results[name]['compressed_words']} words")
    print(f"   Reduction:   {results[name]['compression_rate']:.1%}")
    print(f"   Quality:     {results[name]['quality_retained']:.1%}")
    
    print(f"\n📄 Compressed output:")
    print("-"*70)
    print(results[name]['compressed_text'])
    print("-"*70)

# Summary table
print(f"\n{'='*70}")
print("SUMMARY TABLE")
print(f"{'='*70}")

print(f"\n{'Prompt Type':<25} {'Original':<10} {'Compressed':<10} {'Reduction':<12} {'Quality':<10}")
print("-"*70)

for name, data in results.items():
    print(f"{name:<25} {data['original_words']:<10} {data['compressed_words']:<10} {data['compression_rate']:.1%}{'':>6} {data['quality_retained']:.0%}{'':>6}")

print(f"\n{'='*70}")
print("ANALYSIS")
print(f"{'='*70}")

avg_compression = sum(r['compression_rate'] for r in results.values()) / len(results)
avg_quality = sum(r['quality_retained'] for r in results.values()) / len(results)

print(f"\nAverage compression: {avg_compression:.1%}")
print(f"Average quality retained: {avg_quality:.1%}")

if avg_compression > 0.25 and avg_quality > 0.85:
    print("\n✅ System performs well on standard prompts!")
    print("   Parser works fine for typical use cases.")
else:
    print("\n⚠️  System struggles on standard prompts")
    print("   May need parser improvements")