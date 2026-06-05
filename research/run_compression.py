"""
Benchmark the new compressor (SemanticEvaluator + tier routing + validator gate)
on two datasets: P3 (short prompts) and Awesome-ChatGPT (long prompts).

For each prompt this reports:
  - words (input length)
  - density (lexical_diversity * (1 - mean sentence cosine sim))
  - tier (T1 = Bayesian, T2 = TextRank, T3 = passthrough)
  - compression ratio (% tokens saved)
  - semantic similarity (cosine of all-MiniLM-L6-v2 embeddings)
  - gate result (whether CompressionValidator accepted the compressed output)
  - whether persona was retained in the output

Usage:
    python src/run_compression.py
    python src/run_compression.py --attention      # use AttentionInformedOptimiser
    python src/run_compression.py --mock           # use legacy MockEvaluator (no semantic obj)
"""

import argparse
import io
import json
import logging
import sys
import contextlib
from pathlib import Path

from prompt_compress import PromptCompressor
from prompt_compress.semantic_compressor import compute_density
from prompt_compress.validators import CompressionValidator


DATASETS = {
    'P3':      'data/test_prompts/p3_test_set.json',
    'Awesome': 'data/test_prompts/long_prompts_test_set.json',
}

TIER_THRESHOLDS = {'T2': 0.50, 'T3': 0.85}


def infer_tier(density: float) -> str:
    if density >= TIER_THRESHOLDS['T3']:
        return 'T3'
    if density >= TIER_THRESHOLDS['T2']:
        return 'T2'
    return 'T1'


def label_for(prompt: dict) -> str:
    return prompt.get('role') or prompt.get('task') or prompt['id']


def run_one(prompt: dict, compressor: PromptCompressor, validator: CompressionValidator) -> dict:
    text = prompt['text']
    density_info = compute_density(text)
    density = density_info['density']

    # Silence the compressor's own logging during the run; we'll print our own table.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = compressor.compress(text, output=False)

    compressed = result['compressed_text']
    passed, reasons = validator.validate(text, compressed)
    similarity = validator._cosine_similarity(text, compressed)
    persona_kept = (
        not validator._persona_present(text)
        or validator._persona_present(compressed)
    )

    return {
        'id': prompt['id'],
        'label': label_for(prompt),
        'source': prompt.get('source'),
        'words_in': result['metrics']['original_tokens'],
        'words_out': result['metrics']['compressed_tokens'],
        'compression': result['metrics']['compression_ratio'],
        'best_score': result['metrics']['compressed_score'],
        'density': density,
        'lexical_diversity': density_info['lexical_diversity'],
        'sentence_similarity': density_info['sentence_similarity'],
        'tier': infer_tier(density),
        'similarity': similarity,
        'gate_passed': passed,
        'gate_reasons': reasons,
        'persona_kept': persona_kept,
        'compressed_preview': compressed[:140].replace('\n', ' '),
    }


def run_dataset(name: str, path: Path, compressor, validator) -> list[dict]:
    print(f'\n[{name}] loading {path}')
    with open(path) as f:
        prompts = json.load(f)
    print(f'[{name}] {len(prompts)} prompts')

    results = []
    for i, p in enumerate(prompts, 1):
        print(f'  [{i:>2}/{len(prompts)}] {label_for(p)[:50]}', flush=True)
        results.append(run_one(p, compressor, validator))
    return results


def print_results_table(name: str, results: list[dict]) -> None:
    print(f'\n{"=" * 110}')
    print(f'  {name}  —  {len(results)} prompts')
    print('=' * 110)
    header = (
        f'  {"#":<3} {"Label":<35} {"Win":<5} {"Wout":<5} '
        f'{"Dens":<5} {"Tier":<5} {"Compr":<7} {"Sim":<6} {"BO":<6} {"Pers":<5} {"Gate":<5}'
    )
    print(header)
    print('-' * 110)
    for i, r in enumerate(results, 1):
        gate = 'PASS' if r['gate_passed'] else 'FAIL'
        pers = 'OK' if r['persona_kept'] else 'NO'
        print(
            f'  {i:<3} {r["label"][:33]:<35} '
            f'{r["words_in"]:<5} {r["words_out"]:<5} '
            f'{r["density"]:.2f}  {r["tier"]:<5} '
            f'{r["compression"]:>6.1%} {r["similarity"]:.3f} '
            f'{r["best_score"]:.3f} {pers:<5} {gate:<5}'
        )
    print_aggregate(results)


def print_aggregate(results: list[dict]) -> None:
    n = len(results)
    if n == 0:
        return
    avg_compr = sum(r['compression'] for r in results) / n
    avg_sim = sum(r['similarity'] for r in results) / n
    avg_words_in = sum(r['words_in'] for r in results) / n
    avg_words_out = sum(r['words_out'] for r in results) / n
    gate_pass = sum(r['gate_passed'] for r in results)
    persona_kept = sum(r['persona_kept'] for r in results)
    tier_dist = {'T1': 0, 'T2': 0, 'T3': 0}
    for r in results:
        tier_dist[r['tier']] += 1

    print('-' * 110)
    print(
        f'  avg words in/out:    {avg_words_in:.0f} -> {avg_words_out:.0f}  '
        f'(avg compression: {avg_compr:.1%})'
    )
    print(f'  avg cosine sim:      {avg_sim:.3f}')
    print(f'  validator pass rate: {gate_pass}/{n}  ({gate_pass/n:.0%})')
    print(f'  persona preserved:   {persona_kept}/{n}  ({persona_kept/n:.0%})')
    print(
        f'  tier distribution:   T1={tier_dist["T1"]}, '
        f'T2={tier_dist["T2"]}, T3={tier_dist["T3"]}'
    )

    # Surface any gate failure reasons
    failures = [r for r in results if not r['gate_passed']]
    if failures:
        print('  validator failure reasons:')
        for r in failures:
            print(f'    [{r["label"][:30]}] {"; ".join(r["gate_reasons"])}')


def print_side_by_side(all_results: dict[str, list[dict]]) -> None:
    print(f'\n{"=" * 90}')
    print('  COMPARISON: P3 vs Awesome-ChatGPT')
    print('=' * 90)
    print(f'  {"Dataset":<12} {"n":<4} {"Avg words":<10} {"Avg compr":<11} {"Avg sim":<9} {"Pass rate":<9}')
    print('-' * 90)
    for name, results in all_results.items():
        n = len(results)
        avg_w = sum(r['words_in'] for r in results) / n if n else 0
        avg_c = sum(r['compression'] for r in results) / n if n else 0
        avg_s = sum(r['similarity'] for r in results) / n if n else 0
        pass_rate = (sum(r['gate_passed'] for r in results) / n) if n else 0
        print(
            f'  {name:<12} {n:<4} {avg_w:>7.0f}    '
            f'{avg_c:>7.1%}      {avg_s:.3f}     {pass_rate:.0%}'
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attention', action='store_true',
                   help='Use AttentionInformedOptimiser (per-prompt attention prior)')
    p.add_argument('--informed', action='store_true',
                   help='Use InformedBayesianOptimiser (static P3 JSON prior)')
    p.add_argument('--mock', action='store_true',
                   help='Use the legacy MockEvaluator (no semantic objective)')
    p.add_argument('--datasets', nargs='+', default=list(DATASETS),
                   choices=list(DATASETS),
                   help='Which datasets to run (default: both)')
    p.add_argument('--out',
                   default='data/results/new_compressor_benchmark.json',
                   help='Where to save results JSON')
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Quiet the validator's WARNING logs (their content is captured in gate_reasons).
    logging.basicConfig(level=logging.ERROR)

    compressor = PromptCompressor(
        use_mock_evaluator=args.mock,
        use_informed_prior=args.informed,
        use_attention_prior=args.attention,
    )
    validator = CompressionValidator()

    print(f'\nConfiguration:')
    print(f'  evaluator:  {"Mock" if args.mock else "Semantic"}')
    print(
        f'  optimiser:  '
        f'{"AttentionInformed" if args.attention else "Informed" if args.informed else "Bayesian"}'
    )
    print(f'  datasets:   {", ".join(args.datasets)}')

    all_results: dict[str, list[dict]] = {}
    for name in args.datasets:
        path = Path(DATASETS[name])
        results = run_dataset(name, path, compressor, validator)
        all_results[name] = results
        print_results_table(name, results)

    print_side_by_side(all_results)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({
            'config': {
                'evaluator': 'mock' if args.mock else 'semantic',
                'optimiser': 'attention' if args.attention else 'informed' if args.informed else 'bayesian',
            },
            'results': all_results,
        }, f, indent=2)
    print(f'\nSaved results to {out_path}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
