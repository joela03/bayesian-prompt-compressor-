# prompt-compress

Structural prompt compression for production LLM apps. Where LLMLingua removes individual low-perplexity tokens, this library parses your system prompt into named components (instruction, examples, constraints, style, context), uses Bayesian optimisation to search which components to keep and how aggressively to compress each, scores candidates by semantic similarity to the original, and gates every output through a post-compression validator (persona / placeholder / similarity). Prompts that are already information-dense are detected up front and passed through unchanged.

## Install

```bash
pip install prompt-compress
```

## Quickstart — production integration

```python
from prompt_compress import PromptCompressor, CompressionFailedError

compressor = PromptCompressor()

try:
    result = compressor.compress(
        SYSTEM_PROMPT,
        min_similarity=0.80,
        on_failure='raise',
    )
    SYSTEM_PROMPT = result.compressed_text
    print(f"Saved {result.tokens_saved} tokens per call ({result.compression_ratio:.1%})")
except CompressionFailedError as e:
    print(f"Compression unsafe, using original: {e}")
```

`on_failure` accepts `'fallback'` (default — return the original silently with `gate_passed=False`), `'raise'` (raise `CompressionFailedError`), or `'warn'` (log a warning and return the fallback). The library never blocks on user input.

## Inspecting results

```python
result = compressor.compress(SYSTEM_PROMPT)

print(result.summary())   # one-screen terminal summary
print(result.diff())      # side-by-side original vs compressed
result.to_dict()          # JSON-serialisable, useful for caching/logging
```

Key properties on `CompressionResult`:

| Property | Description |
|---|---|
| `compressed_text` | the output you should use |
| `compression_ratio` | tokens saved / original tokens |
| `tokens_saved` | absolute token count saved |
| `semantic_similarity` | cosine sim of original vs compressed (MiniLM) |
| `compression_efficiency` | `compression_ratio × semantic_similarity` |
| `safe_to_use` | True iff all validator checks passed |
| `persona_preserved` | True iff the "You are…" line survived |
| `placeholders_preserved` | True iff every `{var}` from the original is in the output |
| `tier` / `tier_label` | which pipeline tier ran (1 BO, 2 TextRank, 3 Preserved) |
| `density` | information density score used for routing |

## Configuration

```python
from prompt_compress import PromptCompressor, OptimisationConfig

compressor = PromptCompressor(
    # Optimiser variants:
    use_informed_prior=False,    # seed BO with P3-derived prior
    use_attention_prior=False,   # per-prompt attention prior + ISR safety gate
    # Trade-off knob:
    alpha=0.3,                   # "auto" → 0.3 (validated benchmark default)
    # Tune BO budget:
    optimisation_config=OptimisationConfig(
        n_iterations=20, n_init=5, beta=2.0, random_seed=42,
    ),
)
```

`min_similarity` and `on_failure` are per-call (`compressor.compress(prompt, min_similarity=…, on_failure=…)`) so different parts of your app can adopt different safety bars without rebuilding the compressor.

## Benchmark results

Matched-subset comparison against LLMLingua on the 38 prompts both systems successfully compressed (see `research/benchmark.py` and `research/evaluate.py` to reproduce):

| Metric                       | Ours    | LLMLingua |
|------------------------------|---------|-----------|
| Compression ratio            | 24.1%   | 24.2%     |
| LLM judge score (0–100)      | 73.3    | 70.2      |
| Persona preservation         | 100%    | 53%       |
| Compression efficiency       | 0.179   | 0.155     |

`Compression efficiency = compression_ratio × output_similarity` — rewards being high on both axes.

## Citation

> *EMNLP manuscript in preparation.*
