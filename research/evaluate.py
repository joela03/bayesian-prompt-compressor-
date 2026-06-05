"""
End-to-end evaluation pipeline. Audits the parser, reruns compression with
the current parser, then measures real LLM output quality on the matched
subset using a probing battery and an LLM-as-judge. Everything is resumable
via cached JSON files.

Usage:
    python src/evaluate.py                     # run all steps
    python src/evaluate.py --audit-only        # Step 1 only, no API calls
    python src/evaluate.py --skip-rerun        # skip Step 2, use existing our_results.json
    python src/evaluate.py --skip-probing      # skip Steps 3-4, use cached probing/judge
    python src/evaluate.py --report-only       # regenerate report from cached data
    python src/evaluate.py --yes               # auto-confirm API-spend prompt
"""

from __future__ import annotations

import argparse
import inspect
import io
import json
import logging
import re
import sys
import time
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / '.env')

BENCHMARK_DIR              = PROJECT_ROOT / 'data' / 'results' / 'benchmark'
ALL_PROMPTS_PATH           = BENCHMARK_DIR / 'all_prompts.json'
CLASSIFICATIONS_PATH       = BENCHMARK_DIR / 'classifications.json'
PROMPTS_CLASSIFIED_PATH    = BENCHMARK_DIR / 'prompts_classified.json'
OUR_RESULTS_PATH           = BENCHMARK_DIR / 'our_results.json'
OUR_RESULTS_OLD_PATH       = BENCHMARK_DIR / 'our_results_old.json'
LLMLINGUA_RESULTS_PATH     = BENCHMARK_DIR / 'llmlingua_results.json'
PROBING_RESPONSES_PATH     = BENCHMARK_DIR / 'probing_responses.json'
PROBING_RESULTS_PATH       = BENCHMARK_DIR / 'probing_results.json'
JUDGE_RESULTS_PATH         = BENCHMARK_DIR / 'judge_results.json'
EVALUATION_REPORT_PATH     = BENCHMARK_DIR / 'evaluation_report.txt'

REQUIRED_CONSTRAINT_KEYWORDS = [
    # originals
    'must', 'should', 'do not', 'avoid', 'ensure',
    'requirements:', 'constraints:', 'rules:',
    # behavioural / format constraints
    'only reply', 'nothing else', 'just reply',
    'do not write', 'never', 'always',
    'do not type', 'do not provide',
    'only respond', 'respond only',
]

PROBING_QUERIES = [
    "What is your role and what can you help me with? Give a brief answer.",
    "Give me one concrete example of how you would respond to a typical request in your role.",
    "What constraints or limits apply to your responses? What won't you do?",
]

JUDGE_SYSTEM_PROMPT = (
    "You are evaluating whether a compressed system prompt produces equivalent LLM behaviour.\n"
    "You will see the original system prompt's intent, then two responses to the same user query:\n"
    "Response A (from original prompt) and Response B (from compressed prompt).\n"
    "Score Response B from 0 to 100, where:\n"
    "  100 = identical quality and intent fulfillment\n"
    "  80  = minor differences, same overall quality\n"
    "  60  = noticeable differences but same task accomplished  \n"
    "  40  = significant quality loss or missing key behaviours\n"
    "  0   = completely wrong or incoherent\n"
    "Reply with ONLY an integer score. No explanation."
)

CONSTRAINT_MARKERS = ('not', 'only', 'never', 'must', 'always', 'limit', 'avoid')


# ----------------------------------------------------------------------------
# Lazy shared singletons

_st_model = None
_openai_client = None


def get_st_model():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _st_model


def get_openai_client():
    global _openai_client
    if _openai_client is None:
        import openai
        import os
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise RuntimeError('OPENAI_API_KEY missing — check .env')
        _openai_client = openai.OpenAI(api_key=api_key)
    return _openai_client


# ----------------------------------------------------------------------------
# Helpers

def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f)


def cosine_sim(a: str, b: str) -> float:
    """Cosine similarity using the shared sentence-transformers model."""
    if not a or not b:
        return 0.0
    from scipy.spatial.distance import cosine as scipy_cosine
    model = get_st_model()
    embs = model.encode([a, b], convert_to_numpy=True, show_progress_bar=False)
    return float(1 - scipy_cosine(embs[0], embs[1]))


def embed(texts: list[str]):
    model = get_st_model()
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def ensure_prompts_classified():
    """Produce data/results/benchmark/prompts_classified.json by merging
    all_prompts.json + classifications.json if it doesn't already exist."""
    if PROMPTS_CLASSIFIED_PATH.exists():
        return load_json(PROMPTS_CLASSIFIED_PATH)
    prompts = load_json(ALL_PROMPTS_PATH)
    classifications = load_json(CLASSIFICATIONS_PATH, {}) or {}
    if not prompts:
        raise RuntimeError(
            f'No source prompts found at {ALL_PROMPTS_PATH}; run benchmark.py first.'
        )
    merged = []
    for p in prompts:
        merged.append({**p, 'domain': classifications.get(p['id'], 'other')})
    save_json(PROMPTS_CLASSIFIED_PATH, merged)
    return merged


# ----------------------------------------------------------------------------
# Step 1 — Audit

def step1_audit():
    print('\n' + '=' * 70)
    print('Step 1 — Audit constraint_keywords + classify priority order')
    print('=' * 70)

    # Re-import so the audit reflects the current file state, not a stale cache.
    import importlib
    if 'prompt_parser' in sys.modules:
        importlib.reload(sys.modules['prompt_compress.parser'])
    from prompt_compress.parser import PromptParser

    parser = PromptParser()
    actual_kws = list(parser.constraint_keywords)

    print('\n[AUDIT] constraint_keywords check:')
    missing: list[str] = []
    for kw in REQUIRED_CONSTRAINT_KEYWORDS:
        present = kw in actual_kws
        marker = '✓' if present else '✗'
        status = 'found' if present else 'MISSING — add this before continuing'
        print(f'  {marker} {repr(kw):<25} {status}')
        if not present:
            missing.append(kw)

    extras = [kw for kw in actual_kws if kw not in REQUIRED_CONSTRAINT_KEYWORDS]
    if extras:
        print(f'\n  Note: parser also contains {len(extras)} keyword(s) not in the required '
              f'list (kept as-is): {extras}')

    if missing:
        print(f'\nWARNING: {len(missing)} constraint keyword(s) missing. To fix, edit '
              f'src/prompt_parser.py and add to self.constraint_keywords:')
        for kw in missing:
            print(f"    {repr(kw)},")
    else:
        print('\nAll required constraint keywords present.')

    # Priority-order audit
    print('\n[AUDIT] _classify_sentence priority order:')
    src = inspect.getsource(PromptParser._classify_sentence)
    type_positions: list[tuple[int, str]] = []
    for kw_type in ('example', 'constraint', 'style', 'context', 'instruction'):
        idx = src.find(f'self.{kw_type}_keywords')
        if idx >= 0:
            type_positions.append((idx, f'{kw_type}_keywords'))
    type_positions.sort()

    order_correct = True
    for i, (_, name) in enumerate(type_positions, 1):
        print(f'  {i}. {name}')
    type_names = [n for _, n in type_positions]
    if 'example_keywords' in type_names and 'constraint_keywords' in type_names:
        if type_names.index('example_keywords') < type_names.index('constraint_keywords'):
            order_correct = False
            print('\nWARNING: constraints should be checked before examples for '
                  "'do not' patterns to work correctly. "
                  "Sentences like 'do not use examples' will currently be misclassified.")

    if order_correct:
        print('\n  Priority order OK (constraints before examples).')

    return {
        'missing_keywords': missing,
        'extra_keywords': extras,
        'priority_order_ok': order_correct,
        'priority_order': type_names,
        'constraint_keywords_present': actual_kws,
    }


# ----------------------------------------------------------------------------
# Step 2 — Rerun compression

def step2_rerun_compression():
    print('\n' + '=' * 70)
    print('Step 2 — Rerun compression with current parser')
    print('=' * 70)

    # Rotate the previous results so we can diff
    if OUR_RESULTS_PATH.exists():
        if OUR_RESULTS_OLD_PATH.exists():
            # Keep the very-first "old" as a stable baseline; overwrite the active.
            OUR_RESULTS_PATH.unlink()
            print(f'  removed stale {OUR_RESULTS_PATH.name} '
                  f'(baseline preserved at {OUR_RESULTS_OLD_PATH.name})')
        else:
            OUR_RESULTS_PATH.rename(OUR_RESULTS_OLD_PATH)
            print(f'  renamed {OUR_RESULTS_PATH.name} -> {OUR_RESULTS_OLD_PATH.name}')

    prompts = ensure_prompts_classified()
    classifications = {p['id']: p['domain'] for p in prompts}

    # Make sure src/ modules pick up the current parser state
    import importlib
    for mod in ('prompt_parser', 'compress_prompt', 'evaluators', 'validators',
                'semantic_compressor', 'optimiser', 'informed_optimiser',
                'attention_optimiser', 'attention_priors'):
        if mod in sys.modules:
            importlib.reload(sys.modules[mod])
    from prompt_compress import PromptCompressor
    from prompt_compress.validators import CompressionValidator
    from prompt_compress.semantic_compressor import compute_density

    logging.basicConfig(level=logging.ERROR, force=True)
    get_st_model()

    compressor = PromptCompressor()
    validator = CompressionValidator()

    results: dict = load_json(OUR_RESULTS_PATH, {}) or {}
    total = len(prompts)
    new_this_run = 0

    for i, p in enumerate(prompts, 1):
        if p['id'] in results:
            continue
        domain = classifications.get(p['id'], 'other')
        text = p['text']
        t0 = time.time()
        record = {'id': p['id'], 'label': p.get('label', p['id']),
                  'source': p.get('source', 'unknown'), 'domain': domain}
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                r = compressor.compress(text, output=False)
            compressed = r['compressed_text']
            elapsed = time.time() - t0
            density_info = compute_density(text)
            density = density_info['density']
            tier = 'T3' if density >= 0.85 else ('T2' if density >= 0.50 else 'T1')
            similarity = cosine_sim(text, compressed)
            gate_passed, gate_reasons = validator.validate(text, compressed)
            persona_preserved = (
                not validator._persona_present(text)
                or validator._persona_present(compressed)
            )
            record.update({
                'words_in': r['metrics']['original_tokens'],
                'words_out': r['metrics']['compressed_tokens'],
                'compression_ratio': r['metrics']['compression_ratio'],
                'density': density,
                'tier': tier,
                'similarity': similarity,
                'gate_passed': gate_passed,
                'gate_reasons': gate_reasons,
                'persona_preserved': persona_preserved,
                'compressed_text': compressed,
                'time_seconds': elapsed,
            })
            print(
                f'  [{i}/{total}] {p.get("label", "")[:30]:<30} '
                f'| {domain[:18]:<18} | {tier} | '
                f'{r["metrics"]["compression_ratio"]:>6.1%} | '
                f'sim={similarity:.3f} | {elapsed:.2f}s',
                flush=True,
            )
        except Exception as e:
            record.update({'error': f'{type(e).__name__}: {e}',
                           'time_seconds': time.time() - t0})
            print(f'  [{i}/{total}] {p.get("label", "")[:30]:<30} | '
                  f'ERROR: {record["error"][:80]}', flush=True)
        results[p['id']] = record
        new_this_run += 1
        save_json(OUR_RESULTS_PATH, results)

    print(f'\n  Done. {new_this_run} new results, {len(results)} total.')
    return results


# ----------------------------------------------------------------------------
# Constraint-fix impact

def step2_diff_constraint_impact():
    print('\n' + '=' * 70)
    print('Step 2b — Constraint-fix impact (old vs new)')
    print('=' * 70)

    old = load_json(OUR_RESULTS_OLD_PATH)
    new = load_json(OUR_RESULTS_PATH)
    if not old or not new:
        print('  No baseline (our_results_old.json) found; cannot diff.')
        return

    markers = ('do not', 'only reply', 'nothing else', 'just reply',
               'never', 'always', 'only respond', 'respond only',
               'do not type', 'do not provide', 'do not write')

    affected = []
    unchanged = 0
    for pid, n in new.items():
        if pid not in old:
            continue
        o = old[pid]
        n_text = (n.get('compressed_text') or '').lower()
        o_text = (o.get('compressed_text') or '').lower()
        # Did any marker appear in the new output that wasn't in the old?
        gained = [m for m in markers if m in n_text and m not in o_text]
        n_ratio = n.get('compression_ratio', 0.0)
        o_ratio = o.get('compression_ratio', 0.0)
        # Affected if a constraint marker now appears OR compression changed materially
        if gained or abs(n_ratio - o_ratio) > 0.005:
            affected.append({'id': pid, 'label': n.get('label', pid),
                             'old_ratio': o_ratio, 'new_ratio': n_ratio,
                             'gained_markers': gained})
        else:
            unchanged += 1

    valid_old = [r for r in old.values() if 'compression_ratio' in r]
    valid_new = [r for r in new.values() if 'compression_ratio' in r]
    avg_old = sum(r['compression_ratio'] for r in valid_old) / len(valid_old) if valid_old else 0
    avg_new = sum(r['compression_ratio'] for r in valid_new) / len(valid_new) if valid_new else 0
    print(f'  Prompts where compression changed: {len(affected)}')
    print(f'  Avg compression before: {avg_old:>5.1%}  ->  after: {avg_new:>5.1%}')
    print(f'  Prompts unchanged: {unchanged}')

    # Surface a sample
    affected.sort(key=lambda x: abs(x['new_ratio'] - x['old_ratio']), reverse=True)
    if affected:
        print('\n  Sample of biggest-divergence changes:')
        for a in affected[:8]:
            gained = a['gained_markers']
            tag = f"(kept {gained[0]!r})" if gained else ''
            print(f"    {a['label'][:35]:<37} was {a['old_ratio']:>5.1%} "
                  f"-> now {a['new_ratio']:>5.1%}  {tag}")

    return {'affected': affected, 'unchanged': unchanged,
            'avg_old': avg_old, 'avg_new': avg_new}


# ----------------------------------------------------------------------------
# Step 3 — Probing battery

def step3_probing(matched_ids: list[str], classifications: dict[str, str],
                  originals: dict[str, str], our_results: dict, llmlingua_results: dict):
    print('\n' + '=' * 70)
    print('Step 3 — Probing battery (LLM output similarity)')
    print('=' * 70)

    cache: dict = load_json(PROBING_RESPONSES_PATH, {}) or {}
    client = get_openai_client()

    total_keys = len(matched_ids) * len(PROBING_QUERIES) * 3
    print(f'  matched prompts: {len(matched_ids)}')
    print(f'  cache: {len(cache)}/{total_keys} responses already present')

    def system_text_for(version: str, pid: str) -> str | None:
        if version == 'original':
            return originals.get(pid)
        if version == 'ours':
            return our_results.get(pid, {}).get('compressed_text')
        if version == 'llmlingua':
            return llmlingua_results.get(pid, {}).get('compressed_text')
        return None

    api_calls = 0
    for pid in matched_ids:
        for q_idx, query in enumerate(PROBING_QUERIES):
            for version in ('original', 'ours', 'llmlingua'):
                key = f'{pid}__q{q_idx}__{version}'
                if key in cache:
                    continue
                sys_text = system_text_for(version, pid)
                if not sys_text:
                    cache[key] = None
                    continue
                try:
                    resp = client.chat.completions.create(
                        model='gpt-4o-mini',
                        messages=[
                            {'role': 'system', 'content': sys_text},
                            {'role': 'user', 'content': query},
                        ],
                        max_tokens=200,
                        temperature=0.0,
                    )
                    cache[key] = resp.choices[0].message.content
                except Exception as e:
                    cache[key] = f'__ERROR__: {type(e).__name__}: {e}'
                api_calls += 1
                if api_calls % 10 == 0:
                    save_json(PROBING_RESPONSES_PATH, cache)
                    print(f'  ...{api_calls} new API calls (cache size: {len(cache)})', flush=True)
    save_json(PROBING_RESPONSES_PATH, cache)
    print(f'  Probing API calls this run: {api_calls}')

    # Compute per-prompt output similarity (ours vs original; llmlingua vs original)
    print('  Computing output similarities...')
    per_prompt: dict = load_json(PROBING_RESULTS_PATH, {}) or {}

    for pid in matched_ids:
        sims_ours, sims_lingua = [], []
        per_query = []
        skip = False
        for q_idx in range(len(PROBING_QUERIES)):
            orig_resp = cache.get(f'{pid}__q{q_idx}__original')
            our_resp  = cache.get(f'{pid}__q{q_idx}__ours')
            lin_resp  = cache.get(f'{pid}__q{q_idx}__llmlingua')
            if not all(isinstance(x, str) and not x.startswith('__ERROR__')
                       for x in (orig_resp, our_resp, lin_resp)):
                skip = True
                continue
            so = cosine_sim(orig_resp, our_resp)
            sl = cosine_sim(orig_resp, lin_resp)
            sims_ours.append(so)
            sims_lingua.append(sl)
            per_query.append({'query': q_idx, 'sim_ours': so, 'sim_llmlingua': sl})
        if sims_ours and sims_lingua:
            per_prompt[pid] = {
                'output_sim_ours': sum(sims_ours) / len(sims_ours),
                'output_sim_llmlingua': sum(sims_lingua) / len(sims_lingua),
                'per_query': per_query,
                'partial': skip,
            }
    save_json(PROBING_RESULTS_PATH, per_prompt)
    print(f'  Saved {PROBING_RESULTS_PATH.name} for {len(per_prompt)} prompts')
    return per_prompt, cache


# ----------------------------------------------------------------------------
# Step 4 — LLM judge

def step4_judge(matched_ids: list[str], originals: dict[str, str],
                probing_cache: dict):
    print('\n' + '=' * 70)
    print('Step 4 — LLM judge (output quality vs original)')
    print('=' * 70)

    judge_cache: dict = load_json(JUDGE_RESULTS_PATH, {}) or {}
    client = get_openai_client()
    total_keys = len(matched_ids) * len(PROBING_QUERIES) * 2
    print(f'  cache: {len(judge_cache)}/{total_keys} judgments already present')

    api_calls = 0
    for pid in matched_ids:
        orig_prompt = originals.get(pid, '')
        intent = ' '.join(orig_prompt.split()[:150])
        for q_idx, query in enumerate(PROBING_QUERIES):
            orig_resp = probing_cache.get(f'{pid}__q{q_idx}__original')
            for version in ('ours', 'llmlingua'):
                key = f'{pid}__q{q_idx}__vs_{version}'
                if key in judge_cache:
                    continue
                compressed_resp = probing_cache.get(f'{pid}__q{q_idx}__{version}')
                if not (isinstance(orig_resp, str) and isinstance(compressed_resp, str)
                        and not orig_resp.startswith('__ERROR__')
                        and not compressed_resp.startswith('__ERROR__')):
                    judge_cache[key] = None
                    continue
                user_msg = (
                    f"Original system prompt intent: {intent}\n\n"
                    f"User query: {query}\n\n"
                    f"Response A (original): {orig_resp}\n"
                    f"Response B (compressed): {compressed_resp}\n\n"
                    f"Score B (0-100):"
                )
                try:
                    resp = client.chat.completions.create(
                        model='gpt-4o-mini',
                        messages=[
                            {'role': 'system', 'content': JUDGE_SYSTEM_PROMPT},
                            {'role': 'user', 'content': user_msg},
                        ],
                        max_tokens=8,
                        temperature=0.0,
                    )
                    raw = resp.choices[0].message.content.strip()
                    m = re.search(r'-?\d+', raw)
                    if m:
                        score = max(0, min(100, int(m.group())))
                    else:
                        score = None
                    judge_cache[key] = score
                except Exception:
                    judge_cache[key] = None
                api_calls += 1
                if api_calls % 10 == 0:
                    save_json(JUDGE_RESULTS_PATH, judge_cache)
                    print(f'  ...{api_calls} judge calls (cache size: {len(judge_cache)})',
                          flush=True)
    save_json(JUDGE_RESULTS_PATH, judge_cache)
    print(f'  Judge API calls this run: {api_calls}')

    # Aggregate per prompt
    per_prompt: dict = {}
    for pid in matched_ids:
        ours_scores, lingua_scores = [], []
        for q_idx in range(len(PROBING_QUERIES)):
            s_ours = judge_cache.get(f'{pid}__q{q_idx}__vs_ours')
            s_lin  = judge_cache.get(f'{pid}__q{q_idx}__vs_llmlingua')
            if isinstance(s_ours, int):
                ours_scores.append(s_ours)
            if isinstance(s_lin, int):
                lingua_scores.append(s_lin)
        if ours_scores or lingua_scores:
            per_prompt[pid] = {
                'judge_score_ours': sum(ours_scores) / len(ours_scores)
                if ours_scores else None,
                'judge_score_llmlingua': sum(lingua_scores) / len(lingua_scores)
                if lingua_scores else None,
                'n_ours': len(ours_scores),
                'n_llmlingua': len(lingua_scores),
            }
    return per_prompt, judge_cache


# ----------------------------------------------------------------------------
# Step 5 — Compression efficiency

def compression_efficiency(output_similarity: float | None,
                           compression_ratio: float) -> float | None:
    """
    Compression efficiency = compression_ratio * output_similarity.

    Both axes must be high. The old formulation divided similarity by
    compression which exploded for low-compression cases (e.g. compression
    0.03 with similarity 0.85 gave 28.3 — meaningless). The product form
    lives in [0, 1] for both systems and rewards being high on both axes.
    """
    if compression_ratio is None or output_similarity is None:
        return None
    return compression_ratio * output_similarity


# ----------------------------------------------------------------------------
# Step 5b — Constraint preservation analysis on Query 3

def _split_sentences(text: str) -> list[str]:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def step5_constraint_analysis(matched_ids: list[str], probing_cache: dict):
    print('\n' + '=' * 70)
    print('Step 5b — Query 3 constraint preservation analysis')
    print('=' * 70)

    per_prompt = {}
    for pid in matched_ids:
        orig = probing_cache.get(f'{pid}__q2__original')
        ours = probing_cache.get(f'{pid}__q2__ours')
        lin  = probing_cache.get(f'{pid}__q2__llmlingua')
        if not (isinstance(orig, str) and isinstance(ours, str) and isinstance(lin, str)):
            continue
        if any(s.startswith('__ERROR__') for s in (orig, ours, lin)):
            continue

        orig_sents = _split_sentences(orig)
        constraint_sents = [
            s for s in orig_sents
            if any(m in s.lower() for m in CONSTRAINT_MARKERS)
        ]
        if not constraint_sents:
            per_prompt[pid] = {
                'n_constraint_phrases_in_original': 0,
                'ours_preserved': None,
                'llmlingua_preserved': None,
            }
            continue

        our_sents = _split_sentences(ours)
        lin_sents = _split_sentences(lin)

        def coverage(target_sents):
            if not target_sents:
                return 0.0
            constraint_embs = embed(constraint_sents)
            target_embs = embed(target_sents)
            from scipy.spatial.distance import cosine as scipy_cosine
            hits = 0
            for ce in constraint_embs:
                max_sim = 0.0
                for te in target_embs:
                    sim = 1 - scipy_cosine(ce, te)
                    if sim > max_sim:
                        max_sim = sim
                if max_sim >= 0.80:
                    hits += 1
            return hits / len(constraint_sents)

        per_prompt[pid] = {
            'n_constraint_phrases_in_original': len(constraint_sents),
            'ours_preserved': coverage(our_sents),
            'llmlingua_preserved': coverage(lin_sents),
            'sample_constraint': constraint_sents[0][:100],
        }

    return per_prompt


# ----------------------------------------------------------------------------
# Step 6 — Report

def step6_report(audit_result: dict | None,
                 diff_result: dict | None,
                 probing: dict, judge: dict,
                 constraint_analysis: dict,
                 our_results: dict, llmlingua_results: dict,
                 prompts: list[dict]) -> None:
    print('\n' + '=' * 70)
    print('Step 6 — Generating evaluation report')
    print('=' * 70)

    classifications = {p['id']: p.get('domain', 'other') for p in prompts}
    pbid = {p['id']: p for p in prompts}

    matched_ids = [pid for pid in llmlingua_results
                   if not llmlingua_results[pid].get('error')
                   and pid in our_results
                   and 'compression_ratio' in our_results[pid]]

    lines: list[str] = []
    def out(s: str = '') -> None:
        lines.append(s)
        print(s)

    out('=' * 100)
    out('  EVALUATION REPORT — constraint audit + fair LLM-output comparison')
    out('=' * 100)

    # ------------- Section 1 — Constraint audit
    out('')
    out('SECTION 1 — Constraint audit result')
    out('-' * 100)
    if audit_result is None:
        out('  (audit step skipped)')
    else:
        out(f'  Missing required keywords: {len(audit_result["missing_keywords"])}')
        if audit_result['missing_keywords']:
            for kw in audit_result['missing_keywords']:
                out(f'    - {kw!r}')
        out(f'  Priority order: ' + ', '.join(audit_result['priority_order']))
        order_status = 'OK' if audit_result['priority_order_ok'] else 'WRONG'
        out(f'  Priority order check: {order_status}')
        overall = (
            'PASS' if not audit_result['missing_keywords']
            and audit_result['priority_order_ok']
            else 'FAIL'
        )
        out(f'  Overall audit: {overall}')

    # ------------- Section 2 — Impact of constraint fix
    out('')
    out('SECTION 2 — Impact of constraint fix (old our_results vs new)')
    out('-' * 100)
    if diff_result is None:
        out('  No baseline available — no diff computed.')
    else:
        out(f'  Before fix:  avg compression {diff_result["avg_old"]:>5.1%}')
        out(f'  After fix:   avg compression {diff_result["avg_new"]:>5.1%}')
        out(f'  Prompts where compression changed: {len(diff_result["affected"])}')
        out(f'  Prompts unchanged: {diff_result["unchanged"]}')
        if diff_result['affected']:
            out('  Prompts where constraints now preserved (sample):')
            for a in diff_result['affected'][:10]:
                gained = a['gained_markers']
                tag = f"(kept {gained[0]!r})" if gained else ''
                out(f"    {a['label'][:35]:<37} was {a['old_ratio']:>5.1%} "
                    f"-> now {a['new_ratio']:>5.1%}  {tag}")

    # ------------- Section 3 — Fair comparison on matched subset
    out('')
    out(f'SECTION 3 — Fair comparison (matched subset, n={len(matched_ids)})')
    out('-' * 100)

    def agg(values):
        vals = [v for v in values if v is not None]
        return sum(vals) / len(vals) if vals else None

    our_compr = agg(our_results[pid]['compression_ratio'] for pid in matched_ids)
    llm_compr = agg(llmlingua_results[pid]['compression_ratio'] for pid in matched_ids)
    our_psim  = agg(our_results[pid]['similarity'] for pid in matched_ids)
    llm_psim  = agg(llmlingua_results[pid]['similarity'] for pid in matched_ids)
    our_osim  = agg(probing.get(pid, {}).get('output_sim_ours') for pid in matched_ids)
    llm_osim  = agg(probing.get(pid, {}).get('output_sim_llmlingua') for pid in matched_ids)
    our_judge = agg(judge.get(pid, {}).get('judge_score_ours') for pid in matched_ids)
    llm_judge = agg(judge.get(pid, {}).get('judge_score_llmlingua') for pid in matched_ids)

    our_eff = agg(compression_efficiency(
        probing.get(pid, {}).get('output_sim_ours'),
        our_results[pid]['compression_ratio']) for pid in matched_ids)
    llm_eff = agg(compression_efficiency(
        probing.get(pid, {}).get('output_sim_llmlingua'),
        llmlingua_results[pid]['compression_ratio']) for pid in matched_ids)

    our_persona = sum(1 for pid in matched_ids
                      if our_results[pid].get('persona_preserved')) / max(len(matched_ids), 1)
    llm_persona = sum(1 for pid in matched_ids
                      if llmlingua_results[pid].get('persona_preserved')) / max(len(matched_ids), 1)

    THRESHOLD = 0.85
    our_gate = sum(1 for pid in matched_ids
                   if (probing.get(pid, {}).get('output_sim_ours') or 0) >= THRESHOLD) \
        / max(len(matched_ids), 1)
    llm_gate = sum(1 for pid in matched_ids
                   if (probing.get(pid, {}).get('output_sim_llmlingua') or 0) >= THRESHOLD) \
        / max(len(matched_ids), 1)

    def fmt_pct(x): return f'{x:>6.1%}' if x is not None else '   n/a'
    def fmt_3(x):   return f'{x:>6.3f}' if x is not None else '   n/a'
    def fmt_f(x):   return f'{x:>6.2f}' if x is not None else '   n/a'

    out(f'  {"Metric":<28} {"Ours":<10} {"LLMLingua":<10}')
    out('  ' + '-' * 70)
    out(f'  {"Avg compression":<28} {fmt_pct(our_compr):<10} {fmt_pct(llm_compr):<10}')
    out(f'  {"Prompt similarity":<28} {fmt_3(our_psim):<10} {fmt_3(llm_psim):<10}  '
        '<- BIASED (favors low-compression)')
    out(f'  {"LLM output similarity":<28} {fmt_3(our_osim):<10} {fmt_3(llm_osim):<10}  '
        '<- FAIR (same queries, fixed judge embedding)')
    out(f'  {"LLM judge score (0-100)":<28} {fmt_f(our_judge):<10} {fmt_f(llm_judge):<10}  '
        '<- FAIR (relative to original intent)')
    out(f'  {"Compression efficiency (comp × out_sim)":<28} {fmt_3(our_eff):<10} {fmt_3(llm_eff):<10}  '
        '<- FAIR (rewards high on both axes)')
    out(f'  {"Persona preserved":<28} {fmt_pct(our_persona):<10} {fmt_pct(llm_persona):<10}  '
        '<- FAIR (binary)')
    out(f'  {"Output-sim gate (>{:.2f})".format(THRESHOLD):<28} '
        f'{fmt_pct(our_gate):<10} {fmt_pct(llm_gate):<10}  '
        '<- FAIR (fixed threshold)')

    # ------------- Section 4 — By domain
    out('')
    out(f'SECTION 4 — By domain (matched subset, n>=3 only)')
    out('-' * 100)
    by_domain: dict[str, list[str]] = defaultdict(list)
    for pid in matched_ids:
        by_domain[classifications.get(pid, 'other')].append(pid)

    out(f'  {"Domain":<22} {"n":<3} {"Ours compr":<10} {"LLM compr":<10} '
        f'{"Ours osim":<10} {"LLM osim":<10} {"Ours judge":<11} {"LLM judge":<10}')
    out('  ' + '-' * 90)
    for domain in sorted(by_domain):
        ids = by_domain[domain]
        if len(ids) < 3:
            continue
        oc = agg(our_results[pid]['compression_ratio'] for pid in ids)
        lc = agg(llmlingua_results[pid]['compression_ratio'] for pid in ids)
        oo = agg(probing.get(pid, {}).get('output_sim_ours') for pid in ids)
        lo = agg(probing.get(pid, {}).get('output_sim_llmlingua') for pid in ids)
        oj = agg(judge.get(pid, {}).get('judge_score_ours') for pid in ids)
        lj = agg(judge.get(pid, {}).get('judge_score_llmlingua') for pid in ids)
        out(f'  {domain:<22} {len(ids):<3} '
            f'{fmt_pct(oc):<10} {fmt_pct(lc):<10} '
            f'{fmt_3(oo):<10} {fmt_3(lo):<10} '
            f'{fmt_f(oj):<11} {fmt_f(lj):<10}')

    # ------------- Section 5 — Query 3 constraint analysis
    out('')
    out('SECTION 5 — Query 3 constraint preservation analysis')
    out('-' * 100)
    out('  (cosine >= 0.80 between an original constraint sentence and any compressed sentence)')

    ours_rates = [v['ours_preserved'] for v in constraint_analysis.values()
                  if v.get('ours_preserved') is not None]
    llm_rates  = [v['llmlingua_preserved'] for v in constraint_analysis.values()
                  if v.get('llmlingua_preserved') is not None]
    out(f'  prompts with constraints in original Q3: '
        f'{sum(1 for v in constraint_analysis.values() if v["n_constraint_phrases_in_original"])}')
    if ours_rates:
        out(f'  ours: avg constraint mention rate     {sum(ours_rates)/len(ours_rates):>5.1%}')
    if llm_rates:
        out(f'  llmlingua: avg constraint mention rate {sum(llm_rates)/len(llm_rates):>5.1%}')

    out('')
    out('  Per-prompt (constraint mention rate; ✓ = >=50% of original constraints captured):')
    out(f'  {"Prompt":<40} {"#constraints":<13} {"Ours":<8} {"LLM":<8}')
    out('  ' + '-' * 80)
    for pid in sorted(constraint_analysis):
        rec = constraint_analysis[pid]
        if rec['n_constraint_phrases_in_original'] == 0:
            continue
        label = pbid.get(pid, {}).get('label', pid)
        our_r = rec.get('ours_preserved')
        lin_r = rec.get('llmlingua_preserved')
        our_s = f'{our_r:>5.0%}' if our_r is not None else ' n/a '
        lin_s = f'{lin_r:>5.0%}' if lin_r is not None else ' n/a '
        our_mark = '✓' if our_r and our_r >= 0.5 else ' '
        lin_mark = '✓' if lin_r and lin_r >= 0.5 else ' '
        out(f'  {label[:38]:<40} {rec["n_constraint_phrases_in_original"]:<13} '
            f'{our_s}{our_mark}  {lin_s}{lin_mark}')

    out('')
    out('=' * 100)
    out(f'Report written to {EVALUATION_REPORT_PATH}')

    EVALUATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVALUATION_REPORT_PATH.write_text('\n'.join(lines) + '\n')


# ----------------------------------------------------------------------------
# Cost estimate + confirmation

def confirm_api_spend(matched_n: int, auto_yes: bool, probing_cache: dict,
                      judge_cache: dict) -> bool:
    probing_total = matched_n * len(PROBING_QUERIES) * 3
    judge_total = matched_n * len(PROBING_QUERIES) * 2
    probing_remaining = max(0, probing_total - len(probing_cache))
    judge_remaining = max(0, judge_total - len(judge_cache))
    total_calls = probing_remaining + judge_remaining
    # Rough GPT-4o-mini cost: ~$0.00015 per call (mix of input/output ~400 tokens)
    est_cost = total_calls * 0.00015
    print('')
    print(f'  Estimated additional API calls: {total_calls}')
    print(f'    probing: {probing_remaining}/{probing_total} remaining')
    print(f'    judge:   {judge_remaining}/{judge_total} remaining')
    print(f'  Estimated cost: ~${est_cost:.3f}')
    if total_calls == 0:
        print('  Everything already cached — no spend required.')
        return True
    if auto_yes:
        print('  --yes flag set; proceeding.')
        return True
    try:
        ans = input('  Proceed? [y/N] ').strip().lower()
    except EOFError:
        ans = ''
    return ans == 'y'


# ----------------------------------------------------------------------------
# Main

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--skip-rerun', action='store_true')
    p.add_argument('--skip-probing', action='store_true')
    p.add_argument('--report-only', action='store_true')
    p.add_argument('--yes', '-y', action='store_true',
                   help='Auto-confirm API-spend prompt')
    return p.parse_args()


def main() -> int:
    args = parse_args()
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1 — Audit always runs (cheap, no API calls)
    # Audit is purely local (no API), so it runs even under --report-only.
    audit_result = step1_audit()

    if args.audit_only:
        return 0

    # Step 2 — Rerun
    diff_result = None
    if not args.skip_rerun and not args.report_only:
        step2_rerun_compression()
        diff_result = step2_diff_constraint_impact()
    else:
        diff_result = step2_diff_constraint_impact() if OUR_RESULTS_OLD_PATH.exists() else None

    # Reload everything from disk for the LLM-eval steps + report
    prompts = ensure_prompts_classified()
    classifications = {p['id']: p.get('domain', 'other') for p in prompts}
    originals = {p['id']: p['text'] for p in prompts}
    our_results = load_json(OUR_RESULTS_PATH, {}) or {}
    llmlingua_results = load_json(LLMLINGUA_RESULTS_PATH, {}) or {}
    matched_ids = [pid for pid in llmlingua_results
                   if not llmlingua_results[pid].get('error')
                   and pid in our_results
                   and 'compression_ratio' in our_results[pid]]

    probing_cache: dict = load_json(PROBING_RESPONSES_PATH, {}) or {}
    judge_cache: dict   = load_json(JUDGE_RESULTS_PATH, {}) or {}

    # Steps 3 + 4 — API spend, confirm together
    if not args.skip_probing and not args.report_only:
        if not confirm_api_spend(len(matched_ids), args.yes, probing_cache, judge_cache):
            print('Aborted before API calls.')
            return 1
        probing, probing_cache = step3_probing(
            matched_ids, classifications, originals, our_results, llmlingua_results
        )
        judge, judge_cache = step4_judge(matched_ids, originals, probing_cache)
    else:
        probing = load_json(PROBING_RESULTS_PATH, {}) or {}
        # Re-aggregate judge from cache if needed
        judge = {}
        for pid in matched_ids:
            ours_scores, lingua_scores = [], []
            for q_idx in range(len(PROBING_QUERIES)):
                s_ours = judge_cache.get(f'{pid}__q{q_idx}__vs_ours')
                s_lin  = judge_cache.get(f'{pid}__q{q_idx}__vs_llmlingua')
                if isinstance(s_ours, int): ours_scores.append(s_ours)
                if isinstance(s_lin, int):  lingua_scores.append(s_lin)
            if ours_scores or lingua_scores:
                judge[pid] = {
                    'judge_score_ours': sum(ours_scores)/len(ours_scores)
                    if ours_scores else None,
                    'judge_score_llmlingua': sum(lingua_scores)/len(lingua_scores)
                    if lingua_scores else None,
                }

    constraint_analysis = step5_constraint_analysis(matched_ids, probing_cache)

    step6_report(
        audit_result=audit_result,
        diff_result=diff_result,
        probing=probing,
        judge=judge,
        constraint_analysis=constraint_analysis,
        our_results=our_results,
        llmlingua_results=llmlingua_results,
        prompts=prompts,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
