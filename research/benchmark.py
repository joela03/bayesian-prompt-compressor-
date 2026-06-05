"""
Benchmark our compression system against LLMLingua across prompt sources.

Pipeline (each step is cacheable + resumable via the matching --skip flag):
  1. Collect  -> data/results/benchmark/all_prompts.json
  2. Classify -> data/results/benchmark/classifications.json
  3. Our run  -> data/results/benchmark/our_results.json     (incremental, resumable)
  4. Subset   -> data/results/benchmark/llmlingua_subset.json
  5. LLMLingua-> data/results/benchmark/llmlingua_results.json (incremental, resumable)
  6. Report   -> data/results/benchmark/report.txt              (stdout + file)

Usage (run from project root):
    python src/benchmark.py                    # run everything
    python src/benchmark.py --skip-collect     # use existing all_prompts.json
    python src/benchmark.py --skip-classify    # use existing classifications.json
    python src/benchmark.py --skip-our         # use cached our_results.json
    python src/benchmark.py --skip-llmlingua   # use cached llmlingua_results.json
    python src/benchmark.py --report-only      # regenerate report from cached files
"""

import argparse
import io
import json
import logging
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path

from dotenv import load_dotenv

# Project layout: this file lives in src/; project root is its parent.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / '.env')

BENCHMARK_DIR = PROJECT_ROOT / 'data' / 'results' / 'benchmark'
ALL_PROMPTS_PATH        = BENCHMARK_DIR / 'all_prompts.json'
CLASSIFICATIONS_PATH    = BENCHMARK_DIR / 'classifications.json'
OUR_RESULTS_PATH        = BENCHMARK_DIR / 'our_results.json'
LLMLINGUA_SUBSET_PATH   = BENCHMARK_DIR / 'llmlingua_subset.json'
LLMLINGUA_RESULTS_PATH  = BENCHMARK_DIR / 'llmlingua_results.json'
REPORT_PATH             = BENCHMARK_DIR / 'report.txt'

LOCAL_LONG_PATH = PROJECT_ROOT / 'data' / 'test_prompts' / 'long_prompts_test_set.json'
LOCAL_P3_PATH   = PROJECT_ROOT / 'data' / 'test_prompts' / 'p3_test_set.json'

# Cap on rows pulled from Awesome ChatGPT Prompts. The dataset has ~1800 rows
# which would make the "our system" step take 2+ hours via BO. 300 keeps the
# total set close to the spec's expected size (~70-450 prompts).
AWESOME_CAP = 300

# Keyword-based domain classifier. First-match wins; ordering reflects priority
# (more specific domains earlier so a "marketing consultant" doesn't fall into
# 'analysis' just because it mentions "interpret").
DOMAIN_KEYWORDS = [
    ('developer_tools', [
        r'\bcode\b', r'\bcoding\b', r'\bdebug', r'\bfunction\b', r'\bprogram',
        r'\bdeveloper\b', r'\bengineer\b', r'\bsoftware\b', r'\bapi\b',
        r'\bgit\b', r'\bsql\b', r'\blinux\b', r'\bshell\b', r'\bcommit\b',
        r'\bpython\b', r'\bjavascript\b', r'\bkernel\b', r'\bcompiler\b',
        r'\bterminal\b', r'\binterpreter\b', r'\bregex\b', r'\bbackend\b',
        r'\bfrontend\b', r'\bdevops\b', r'\bkubernetes\b', r'\bweb\s?browser',
        r'\bsolr\b',
    ]),
    ('creative_writing', [
        r'\bstory(?:teller|writer)?\b', r'\bpoet\b', r'\bpoem\b', r'\bnovelist\b',
        r'\bscreenwrit', r'\bscreenplay\b', r'\bnarrative\b', r'\blyric',
        r'\brapper\b', r'\bcomposer\b', r'\bfiction\b', r'\bessay writ',
        r'\bjournalist\b', r'\bcreative director\b',
    ]),
    ('education', [
        r'\bteacher\b', r'\btutor\b', r'\bstudent\b', r'\blesson\b',
        r'\bprofessor\b', r'\bmentor\b', r'\beducator\b', r'\binstructor\b',
        r'\bacademician\b', r'\bpronunciation\b', r'\bphilosophy teacher\b',
        r'\bmath teacher\b', r'\beducational content\b',
    ]),
    ('business', [
        r'\bconsultant\b', r'\bmarketing\b', r'\bstartup\b', r'\bbusiness\b',
        r'\bmanager\b', r'\bsalesperson\b', r'\badvisor\b', r'\brecruit',
        r'\binvestor\b', r'\bpitch\b', r'\bceo\b', r'\baccountant\b',
        r'\bfinancial analyst\b', r'\breal estate\b', r'\bproduct manager\b',
        r'\bcommercial\b',
    ]),
    ('analysis', [
        r'\banalys', r'\binterpret', r'\bevaluator\b', r'\bresearcher\b',
        r'\bstatistician\b', r'\bcritic\b', r'\breviewer\b', r'\bdebater\b',
        r'\bhistorian\b', r'\bfallacy finder\b', r'\bdata scientist\b',
        r'\bjournal reviewer\b',
    ]),
    ('roleplay', [
        r'\bpretend\b', r'\bpersona\b', r'\bjailbreak\b', r'\bcharacter from\b',
        r'\bdan\b', r'\btime travel\b', r'\bmagician\b', r'\bgnomist\b',
        r'\bstoryteller\b', r'\brole-?play',
    ]),
    ('translation', [
        r'\btranslat', r'\blanguage detect', r'\blinguist\b', r'\betymolog',
        r'\bemoji translator\b', r'\bbiblical translator\b',
    ]),
    ('customer_service', [
        r'\bcustomer\s+(service|support)\b', r'\bsupport agent\b',
        r'\bservice representative\b', r'\bhelpdesk\b', r'\bhelp desk\b',
        r'\bsales representative\b',
    ]),
    ('health_lifestyle', [
        r'\bnutrition', r'\bfitness\b', r'\bdoctor\b', r'\btherapist\b',
        r'\bpsychologist\b', r'\bmedical\b', r'\bdietitian\b', r'\bdentist\b',
        r'\byogi\b', r'\bmental health\b', r'\bpersonal trainer\b',
        r'\bpet behavior', r'\blife coach\b', r'\brelationship coach\b',
    ]),
    ('entertainment', [
        r'\bgame\b', r'\bd&d\b', r'\bdnd\b', r'\brpg\b', r'\bpuzzle\b',
        r'\briddle\b', r'\bcomedian\b', r'\bchess\b', r'\bastrologer\b',
        r'\bdream interpret', r'\bhoroscope\b', r'\btext based adventure\b',
    ]),
]
DEFAULT_DOMAIN = 'other'


# -----------------------------------------------------------------------------
# Helpers

def word_count(s: str) -> int:
    return len(s.split())


def length_band(n: int) -> str:
    if n < 150:
        return 'short'
    if n < 300:
        return 'medium'
    return 'long'


def dedup_key(text: str) -> str:
    return text.lower().strip()[:80]


def classify_domain(text: str) -> str:
    t = text.lower()
    for domain, patterns in DOMAIN_KEYWORDS:
        for pat in patterns:
            if re.search(pat, t):
                return domain
    return DEFAULT_DOMAIN


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f)


def cosine_sim(model, a: str, b: str) -> float:
    """Cosine similarity via a pre-loaded sentence-transformers model."""
    from scipy.spatial.distance import cosine
    embs = model.encode([a, b], convert_to_numpy=True, show_progress_bar=False)
    return float(1 - cosine(embs[0], embs[1]))


# -----------------------------------------------------------------------------
# Step 1 — Collect

def step_collect() -> list[dict]:
    print('\n[1] Collecting prompts...')
    records: list[dict] = []
    src_count: dict[str, int] = {}

    # Source A — Awesome ChatGPT Prompts. The dataset is ~1800 rows now;
    # we cap with AWESOME_CAP to keep the benchmark tractable. Sampling is
    # deterministic (slice from the start) so re-runs reproduce.
    try:
        from datasets import load_dataset
        ds = load_dataset('fka/awesome-chatgpt-prompts', split='train')
        rows = list(ds)
        if AWESOME_CAP and len(rows) > AWESOME_CAP:
            print(f'  awesome_chatgpt: capping {len(rows)} -> {AWESOME_CAP}')
            rows = rows[:AWESOME_CAP]
        a = [{'label': r['act'], 'text': r['prompt'], 'source': 'awesome_chatgpt'} for r in rows]
        records.extend(a)
        src_count['awesome_chatgpt'] = len(a)
        print(f'  awesome_chatgpt loaded: {len(a)}')
    except Exception as e:
        print(f'  awesome_chatgpt FAILED: {e}')
        src_count['awesome_chatgpt'] = 0

    # Source B — ShareGPT (system messages only, cap 300)
    try:
        from datasets import load_dataset
        ds = load_dataset('anon8231489123/ShareGPT_Vicuna_unfiltered', split='train', streaming=True)
        b: list[dict] = []
        for item in ds:
            convs = item.get('conversations', [])
            if convs and convs[0].get('from') == 'system':
                text = convs[0].get('value', '').strip()
                if text:
                    b.append({
                        'label': f"ShareGPT_{item.get('id', '')}",
                        'text': text,
                        'source': 'sharegpt',
                    })
            if len(b) >= 300:
                break
        records.extend(b)
        src_count['sharegpt'] = len(b)
        print(f'  sharegpt loaded: {len(b)}')
    except Exception as e:
        print(f'  sharegpt FAILED: {e}')
        src_count['sharegpt'] = 0

    # Source C — local files
    c: list[dict] = []
    if LOCAL_LONG_PATH.exists():
        for p in load_json(LOCAL_LONG_PATH, []):
            c.append({'label': p['role'], 'text': p['text'], 'source': 'local'})
    if LOCAL_P3_PATH.exists():
        for p in load_json(LOCAL_P3_PATH, []):
            c.append({'label': p['task'], 'text': p['text'], 'source': 'local'})
    records.extend(c)
    src_count['local'] = len(c)
    print(f'  local loaded: {len(c)}')

    # Filter: word count >= 80
    pre = len(records)
    records = [r for r in records if word_count(r['text']) >= 80]
    print(f'  after word_count >= 80 filter: {len(records)} (dropped {pre - len(records)})')

    # Dedup by first 80 chars (lowercase, stripped)
    seen: set[str] = set()
    deduped: list[dict] = []
    for r in records:
        key = dedup_key(r['text'])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    print(f'  after dedup: {len(deduped)} (dropped {len(records) - len(deduped)})')

    # Stable IDs prefixed by source
    counters: dict[str, int] = defaultdict(int)
    prefix_map = {'awesome_chatgpt': 'awesome', 'sharegpt': 'sharegpt', 'local': 'local'}
    out: list[dict] = []
    for r in deduped:
        prefix = prefix_map.get(r['source'], r['source'])
        counters[prefix] += 1
        r['id'] = f"{prefix}_{counters[prefix]:03d}"
        r['word_count'] = word_count(r['text'])
        out.append(r)

    print('\n  Source breakdown after dedup:')
    final_counts = Counter(r['source'] for r in out)
    for src, n in final_counts.items():
        print(f'    {src}: {n}')
    print(f'  TOTAL: {len(out)}')

    save_json(ALL_PROMPTS_PATH, out)
    print(f'\n  Saved {ALL_PROMPTS_PATH}')
    return out


# -----------------------------------------------------------------------------
# Step 2 — Classify

def step_classify(prompts: list[dict]) -> dict[str, str]:
    print('\n[2] Classifying prompts by domain (keyword-based)...')
    classifications: dict[str, str] = {}
    for p in prompts:
        # Classify on label + first 200 chars of body — labels are very signaly
        signal = f"{p['label']} {p['text'][:200]}"
        classifications[p['id']] = classify_domain(signal)

    dist = Counter(classifications.values())
    print('  domain distribution:')
    for domain, n in sorted(dist.items(), key=lambda kv: -kv[1]):
        print(f'    {domain:<20} {n}')

    save_json(CLASSIFICATIONS_PATH, classifications)
    print(f'  Saved {CLASSIFICATIONS_PATH}')
    return classifications


# -----------------------------------------------------------------------------
# Step 3 — Run our system

def step_run_ours(prompts: list[dict], classifications: dict[str, str], st_model) -> dict:
    print('\n[3] Running our compression system...')
    # Silence inner logging — we capture every signal in result dicts
    logging.basicConfig(level=logging.ERROR, force=True)

    from prompt_compress import PromptCompressor
    from prompt_compress.validators import CompressionValidator
    from prompt_compress.semantic_compressor import compute_density

    results: dict = load_json(OUR_RESULTS_PATH, default={}) or {}
    # Cache validation: the resumable cache trusts that {id -> prompt content}
    # is stable across runs. Discard entries that are
    #   (a) no longer in the current all_prompts.json,
    #   (b) whose cached label disagrees with the current prompt at that id,
    #   (c) previous runs that errored out — let them retry.
    prompts_by_id_for_validation = {p['id']: p for p in prompts}
    stale_ids = [pid for pid, r in results.items()
                 if pid not in prompts_by_id_for_validation
                 or (r.get('label') is not None
                     and r['label'] != prompts_by_id_for_validation[pid]['label'])
                 or r.get('error')]
    for pid in stale_ids:
        del results[pid]
    if stale_ids:
        print(f'  invalidated {len(stale_ids)} stale cache entries')
    print(f'  resuming with {len(results)} cached results')

    compressor = PromptCompressor(use_attention_prior=True)
    validator = CompressionValidator()

    total = len(prompts)
    completed_this_run = 0
    for i, p in enumerate(prompts, 1):
        if p['id'] in results:
            continue
        domain = classifications.get(p['id'], DEFAULT_DOMAIN)
        text = p['text']
        t0 = time.time()
        record = {
            'id': p['id'],
            'label': p['label'],
            'source': p['source'],
            'domain': domain,
        }
        try:
            # Suppress per-prompt BO chatter
            buf = io.StringIO()
            with redirect_stdout(buf):
                r = compressor.compress(text)
            compressed = r.compressed_text
            elapsed = time.time() - t0
            density_info = compute_density(text)
            density = density_info['density']
            if density >= 0.85:
                tier = 'T3'
            elif density >= 0.50:
                tier = 'T2'
            else:
                tier = 'T1'

            similarity = cosine_sim(st_model, text, compressed)
            # validator.validate now returns (passed, reasons, similarity) —
            # we already compute similarity above, drop the third element.
            gate_passed, gate_reasons, _val_sim = validator.validate(text, compressed)
            from prompt_compress._persona import persona_present
            persona_preserved = (
                not persona_present(text) or persona_present(compressed)
            )

            record.update({
                'words_in': r.original_tokens,
                'words_out': r.compressed_tokens,
                'compression_ratio': r.compression_ratio,
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
                f'  [{i}/{total}] {p["label"][:30]:<30} | {domain[:18]:<18} '
                f'| {tier} | {r.compression_ratio:>6.1%} '
                f'| sim={similarity:.3f} | {elapsed:.2f}s',
                flush=True,
            )
        except Exception as e:
            record.update({
                'error': f'{type(e).__name__}: {e}',
                'time_seconds': time.time() - t0,
            })
            print(f'  [{i}/{total}] {p["label"][:30]:<30} | ERROR: {record["error"][:80]}', flush=True)

        results[p['id']] = record
        completed_this_run += 1
        # Persist after each — resumable
        save_json(OUR_RESULTS_PATH, results)

    print(f'\n  Done. {completed_this_run} new results, {len(results)} total.')
    return results


# -----------------------------------------------------------------------------
# Step 4 — Select LLMLingua subset (stratified by domain × length band)

def step_select_subset(
    prompts: list[dict],
    classifications: dict[str, str],
    per_band: int = 3,
    cap_per_domain: int = 15,
) -> list[str]:
    print('\n[4] Selecting LLMLingua subset (stratified by domain × length band)...')
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for p in prompts:
        by_domain[classifications.get(p['id'], DEFAULT_DOMAIN)].append(p)

    selected_ids: list[str] = []
    rows: list[tuple[str, int, int, int, int]] = []  # (domain, total, short, medium, long)

    for domain in sorted(by_domain):
        items = by_domain[domain]
        bands = {'short': [], 'medium': [], 'long': []}
        for p in items:
            bands[length_band(p['word_count'])].append(p)
        for b in bands.values():
            b.sort(key=lambda x: x['word_count'])

        picks: list[dict] = []
        for band_name in ('short', 'medium', 'long'):
            picks.extend(bands[band_name][:per_band])

        picks = picks[:cap_per_domain]
        for p in picks:
            selected_ids.append(p['id'])
        rows.append((
            domain,
            len(picks),
            sum(1 for p in picks if length_band(p['word_count']) == 'short'),
            sum(1 for p in picks if length_band(p['word_count']) == 'medium'),
            sum(1 for p in picks if length_band(p['word_count']) == 'long'),
        ))

    print(f'  {"Domain":<22} {"n":<4} {"short":<6} {"medium":<7} {"long":<5}')
    print('  ' + '-' * 50)
    for domain, n, s, m, l in rows:
        print(f'  {domain:<22} {n:<4} {s:<6} {m:<7} {l:<5}')
    print(f'  Total selected: {len(selected_ids)}')

    save_json(LLMLINGUA_SUBSET_PATH, selected_ids)
    print(f'  Saved {LLMLINGUA_SUBSET_PATH}')
    return selected_ids


# -----------------------------------------------------------------------------
# Step 5 — Run LLMLingua on subset

def step_run_llmlingua(selected_ids: list[str], prompts: list[dict], st_model) -> dict:
    print('\n[5] Running LLMLingua on subset...')

    # Lazy import; install if missing
    try:
        from llmlingua import PromptCompressor as LinguaCompressor
    except ImportError:
        print('  llmlingua not installed; installing into current env...')
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', 'llmlingua', '-q'],
            check=False,
        )
        from llmlingua import PromptCompressor as LinguaCompressor

    print('  Loading LLMLingua compressor (GPT-2 backbone, CPU)...')
    lingua = LinguaCompressor(
        model_name='openai-community/gpt2',
        device_map='cpu',
        model_config={},
    )

    from prompt_compress._persona import persona_present as persona_check

    prompts_by_id = {p['id']: p for p in prompts}
    results: dict = load_json(LLMLINGUA_RESULTS_PATH, default={}) or {}
    # Same cache-validation logic as step_run_ours.
    stale_ids = [pid for pid, r in results.items()
                 if pid not in prompts_by_id
                 or (r.get('label') is not None
                     and r['label'] != prompts_by_id[pid]['label'])
                 or r.get('error')]
    for pid in stale_ids:
        del results[pid]
    if stale_ids:
        print(f'  invalidated {len(stale_ids)} stale LLMLingua cache entries')
    print(f'  resuming with {len(results)} cached results')

    total = len(selected_ids)
    completed_this_run = 0
    for i, pid in enumerate(selected_ids, 1):
        if pid in results:
            continue
        if pid not in prompts_by_id:
            print(f'  [{i}/{total}] {pid} not found in prompts')
            continue
        p = prompts_by_id[pid]
        text = p['text']
        win = word_count(text)
        t0 = time.time()
        record = {
            'id': pid,
            'label': p['label'],
            'words_in': win,
        }
        try:
            # LLMLingua's API: rate (not ratio); context wraps the single prompt.
            out = lingua.compress_prompt(text, rate=0.5, force_tokens=['\n'])
            compressed = out['compressed_prompt']
            wout = word_count(compressed)
            similarity = cosine_sim(st_model, text, compressed)
            record.update({
                'words_out': wout,
                'compression_ratio': (win - wout) / win if win else 0.0,
                'similarity': similarity,
                'persona_preserved': (not persona_check(text)) or persona_check(compressed),
                'compressed_text': compressed,
                'time_seconds': time.time() - t0,
                'error': None,
            })
            print(
                f'  [{i}/{total}] {p["label"][:30]:<30} '
                f'| {record["compression_ratio"]:>6.1%} '
                f'| sim={similarity:.3f} | {record["time_seconds"]:.2f}s',
                flush=True,
            )
        except Exception as e:
            record.update({
                'error': f'{type(e).__name__}: {e}',
                'time_seconds': time.time() - t0,
            })
            print(f'  [{i}/{total}] {p["label"][:30]:<30} | ERROR: {record["error"][:80]}', flush=True)

        results[pid] = record
        completed_this_run += 1
        save_json(LLMLINGUA_RESULTS_PATH, results)

    print(f'\n  Done. {completed_this_run} new results, {len(results)} total.')
    return results


# -----------------------------------------------------------------------------
# Step 6 — Diagnostic report

def step_report(
    prompts: list[dict],
    classifications: dict[str, str],
    our_results: dict,
    llmlingua_results: dict,
) -> None:
    print('\n[6] Generating diagnostic report...')
    prompts_by_id = {p['id']: p for p in prompts}

    lines: list[str] = []
    def out(s: str = '') -> None:
        lines.append(s)
        print(s)

    out('=' * 100)
    out('  BENCHMARK REPORT — our compressor vs LLMLingua')
    out('=' * 100)

    # --------- Section 1 — overall summary
    out('')
    out('SECTION 1 — Overall summary')
    out('-' * 100)

    ours_valid = [r for r in our_results.values() if 'compression_ratio' in r]
    if ours_valid:
        n = len(ours_valid)
        avg_c = sum(r['compression_ratio'] for r in ours_valid) / n
        avg_s = sum(r['similarity'] for r in ours_valid) / n
        persona = sum(r.get('persona_preserved', False) for r in ours_valid) / n
        gate = sum(r.get('gate_passed', False) for r in ours_valid) / n
        out(f'  Full set (our system, n={n})')
        out(
            f'    avg compression {avg_c:>6.1%}   avg sim {avg_s:.3f}   '
            f'persona {persona:.0%}   gate pass {gate:.0%}'
        )
    else:
        out('  Full set (our system, n=0): no results')

    # Matched subset: prompts both systems successfully compressed. Reported
    # because the full-set "ours" average above includes prompts LLMLingua
    # never attempted (stratified subsampling), which biases the comparison.
    # On the matched subset both averages cover the same prompts.
    lingua_ok = {
        pid: r for pid, r in llmlingua_results.items()
        if not r.get('error') and 'compression_ratio' in r
    }
    matched_ids = sorted(
        pid for pid in lingua_ok
        if pid in our_results and 'compression_ratio' in our_results[pid]
    )

    out('')
    if matched_ids:
        n = len(matched_ids)
        # Ours on the matched subset
        oc = sum(our_results[pid]['compression_ratio'] for pid in matched_ids) / n
        os_ = sum(our_results[pid]['similarity'] for pid in matched_ids) / n
        o_eff = sum(
            our_results[pid]['compression_ratio'] * our_results[pid]['similarity']
            for pid in matched_ids
        ) / n
        o_persona = sum(
            our_results[pid].get('persona_preserved', False) for pid in matched_ids
        ) / n
        o_gate = sum(
            our_results[pid].get('gate_passed', False) for pid in matched_ids
        ) / n

        # LLMLingua on the matched subset (same IDs by construction)
        lc = sum(lingua_ok[pid]['compression_ratio'] for pid in matched_ids) / n
        ls = sum(lingua_ok[pid]['similarity'] for pid in matched_ids) / n
        l_eff = sum(
            lingua_ok[pid]['compression_ratio'] * lingua_ok[pid]['similarity']
            for pid in matched_ids
        ) / n
        l_persona = sum(
            lingua_ok[pid].get('persona_preserved', False) for pid in matched_ids
        ) / n
        # LLMLingua "quality gate pass" — same bar the validator applies to
        # ours: similarity >= 0.85 AND persona preserved. LLMLingua doesn't
        # have a gate of its own, so we score against the same threshold.
        l_gate = sum(
            (lingua_ok[pid]['similarity'] >= 0.85
             and lingua_ok[pid].get('persona_preserved', False))
            for pid in matched_ids
        ) / n

        out(f'  Matched subset — apples-to-apples (n={n}, prompts both systems ran on)')
        out(
            f'    Our system :  compression {oc:>6.1%}   sim {os_:.3f}   '
            f'efficiency {o_eff:.3f}'
        )
        out(
            f'    LLMLingua  :  compression {lc:>6.1%}   sim {ls:.3f}   '
            f'efficiency {l_eff:.3f}'
        )
        out('')
        out('  Efficiency = compression × similarity (higher is better on both axes)')
        out(f'  Persona preserved:  ours {o_persona:.0%}   LLMLingua {l_persona:.0%}')
        out(
            f'  Quality gate pass:  ours {o_gate:.0%}   '
            f'LLMLingua {l_gate:.0%}  (outputs below sim 0.85 or persona lost)'
        )
    else:
        out('  Matched subset: no prompts with both systems successful — skipping')

    failures = [r for r in llmlingua_results.values() if r.get('error')]
    if failures:
        out('')
        out(f'  LLMLingua errors: {len(failures)}')

    # --------- Section 2 — by domain (subset)
    out('')
    out('SECTION 2 — By domain (subset prompts where both systems ran)')
    out('-' * 100)
    out(
        f'  {"Domain":<22} {"n":<3} '
        f'{"Our compr":<10} {"Our sim":<9} '
        f'{"LLM compr":<10} {"LLM sim":<9} {"Winner"}'
    )
    out('  ' + '-' * 95)

    by_domain: dict[str, list[str]] = defaultdict(list)
    for pid in llmlingua_results:
        if llmlingua_results[pid].get('error'):
            continue
        ours = our_results.get(pid)
        if not ours or 'compression_ratio' not in ours:
            continue
        by_domain[classifications.get(pid, DEFAULT_DOMAIN)].append(pid)

    for domain in sorted(by_domain):
        ids = by_domain[domain]
        n = len(ids)
        oc = sum(our_results[pid]['compression_ratio'] for pid in ids) / n
        os = sum(our_results[pid]['similarity'] for pid in ids) / n
        lc = sum(llmlingua_results[pid]['compression_ratio'] for pid in ids) / n
        ls = sum(llmlingua_results[pid]['similarity'] for pid in ids) / n
        sim_winner = 'ours' if os > ls else 'llmlingua'
        compr_winner = 'ours' if oc > lc else 'llmlingua'
        out(
            f'  {domain:<22} {n:<3} '
            f'{oc:>7.1%}    {os:.3f}     '
            f'{lc:>7.1%}    {ls:.3f}     '
            f'sim: {sim_winner} / compr: {compr_winner}'
        )

    # --------- Section 3 — our system underperforms
    out('')
    out('SECTION 3 — Cases where our system underperforms (our compr < 5% AND LLMLingua > 15%)')
    out('-' * 100)
    out(
        f'  {"Label":<35} {"Domain":<20} {"Words":<6} '
        f'{"Our":<7} {"LLM":<7} {"OurSim":<7} {"LLMSim"}'
    )
    out('  ' + '-' * 95)
    misses = []
    for pid, lr in llmlingua_results.items():
        if lr.get('error'):
            continue
        our = our_results.get(pid)
        if not our or 'compression_ratio' not in our:
            continue
        if our['compression_ratio'] < 0.05 and lr['compression_ratio'] > 0.15:
            misses.append((our, lr))
    misses.sort(key=lambda x: x[1]['compression_ratio'] - x[0]['compression_ratio'], reverse=True)
    for our, lr in misses[:30]:
        out(
            f'  {our["label"][:33]:<35} '
            f'{our.get("domain", "other")[:18]:<20} '
            f'{our["words_in"]:<6} '
            f'{our["compression_ratio"]:>5.1%}  '
            f'{lr["compression_ratio"]:>5.1%}  '
            f'{our["similarity"]:.3f}   {lr["similarity"]:.3f}'
        )
    if not misses:
        out('  (none)')

    # --------- Section 4 — LLMLingua breaks quality
    out('')
    out('SECTION 4 — Cases where LLMLingua breaks quality (sim < 0.85 OR persona dropped)')
    out('-' * 100)
    out(
        f'  {"Label":<35} {"Domain":<20} {"Words":<6} '
        f'{"LLM compr":<10} {"LLM sim":<8} {"Persona"}'
    )
    out('  ' + '-' * 95)
    breaks = []
    for pid, lr in llmlingua_results.items():
        if lr.get('error'):
            continue
        if lr.get('similarity', 1.0) < 0.85 or not lr.get('persona_preserved', True):
            breaks.append(lr)
    breaks.sort(key=lambda r: r['similarity'])
    for lr in breaks[:30]:
        domain = classifications.get(lr['id'], DEFAULT_DOMAIN)
        out(
            f'  {lr["label"][:33]:<35} '
            f'{domain[:18]:<20} '
            f'{lr["words_in"]:<6} '
            f'{lr["compression_ratio"]:>7.1%}   '
            f'{lr["similarity"]:.3f}    '
            f'{lr.get("persona_preserved", True)}'
        )
    if not breaks:
        out('  (none)')

    # --------- Section 5 — side-by-side examples
    out('')
    out('SECTION 5 — Side-by-side examples')
    out('-' * 100)

    pairs = []
    for pid in llmlingua_results:
        if llmlingua_results[pid].get('error'):
            continue
        our = our_results.get(pid)
        lr = llmlingua_results[pid]
        if not our or 'compression_ratio' not in our:
            continue
        pairs.append((pid, our, lr))

    def quality_score(pair):
        _, our, lr = pair
        # higher = better for ours: high compr, high sim
        return our['compression_ratio'] * our['similarity'] - lr['compression_ratio'] * lr['similarity']

    # 1. Ours wins clearly: our compr*sim much higher than LLM's
    ours_win = max(pairs, key=quality_score) if pairs else None
    # 2. LLMLingua wins clearly on compression while still passing sim
    llm_win_candidates = [p for p in pairs if p[2]['similarity'] >= 0.85]
    llm_win = max(llm_win_candidates,
                  key=lambda p: p[2]['compression_ratio'] - p[1]['compression_ratio']) \
        if llm_win_candidates else None
    # 3. Close call: smallest absolute difference in compr*sim
    close = min(pairs, key=lambda p: abs(quality_score(p))) if pairs else None

    examples = [
        ('OURS-WIN', ours_win),
        ('LLMLINGUA-WIN-ON-COMPRESSION', llm_win),
        ('CLOSE-CALL', close),
    ]
    seen_pids = set()
    for tag, pair in examples:
        if pair is None or pair[0] in seen_pids:
            continue
        seen_pids.add(pair[0])
        pid, our, lr = pair
        original = prompts_by_id.get(pid, {}).get('text', '')
        head = ' '.join(original.split()[:100])
        our_head = ' '.join(our.get('compressed_text', '').split()[:100])
        lr_head = ' '.join(lr.get('compressed_text', '').split()[:100])
        out('')
        out(f'  [{tag}]  {our["label"]}  ({classifications.get(pid, DEFAULT_DOMAIN)})')
        out(
            f'    metrics:  ours -> compr {our["compression_ratio"]:.1%}, '
            f'sim {our["similarity"]:.3f};   '
            f'llmlingua -> compr {lr["compression_ratio"]:.1%}, '
            f'sim {lr["similarity"]:.3f}'
        )
        out(f'    original (first 100w):')
        out(f'      {head}')
        out(f'    ours compressed (first 100w):')
        out(f'      {our_head}')
        out(f'    llmlingua compressed (first 100w):')
        out(f'      {lr_head}')

    out('')
    out('=' * 100)
    out(f'Report written to {REPORT_PATH}')

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w') as f:
        f.write('\n'.join(lines) + '\n')


# -----------------------------------------------------------------------------
# Main

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--skip-collect', action='store_true')
    p.add_argument('--skip-classify', action='store_true')
    p.add_argument('--skip-our', action='store_true')
    p.add_argument('--skip-llmlingua', action='store_true')
    p.add_argument('--report-only', action='store_true')
    return p.parse_args()


def main() -> int:
    args = parse_args()
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        prompts = load_json(ALL_PROMPTS_PATH, [])
        classifications = load_json(CLASSIFICATIONS_PATH, {})
        our_results = load_json(OUR_RESULTS_PATH, {})
        llmlingua_results = load_json(LLMLINGUA_RESULTS_PATH, {})
        if not (prompts and classifications and our_results and llmlingua_results):
            print('--report-only requires all cached files to exist:', file=sys.stderr)
            for path in (ALL_PROMPTS_PATH, CLASSIFICATIONS_PATH, OUR_RESULTS_PATH, LLMLINGUA_RESULTS_PATH):
                print(f'  {path}: {"OK" if path.exists() else "MISSING"}', file=sys.stderr)
            return 1
        step_report(prompts, classifications, our_results, llmlingua_results)
        return 0

    # Step 1 — Collect
    if args.skip_collect:
        prompts = load_json(ALL_PROMPTS_PATH)
        if prompts is None:
            print('--skip-collect requires existing all_prompts.json', file=sys.stderr)
            return 1
        print(f'[1] Skipped collect; loaded {len(prompts)} prompts from cache.')
    else:
        prompts = step_collect()

    # Step 2 — Classify
    if args.skip_classify:
        classifications = load_json(CLASSIFICATIONS_PATH)
        if classifications is None:
            print('--skip-classify requires existing classifications.json', file=sys.stderr)
            return 1
        print(f'[2] Skipped classify; loaded {len(classifications)} classifications from cache.')
    else:
        classifications = step_classify(prompts)

    # Load shared embedding model once
    print('\n  Loading sentence-transformers model (shared)...')
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Step 3 — Run our system
    if args.skip_our:
        our_results = load_json(OUR_RESULTS_PATH, {})
        print(f'[3] Skipped our run; loaded {len(our_results)} cached results.')
    else:
        our_results = step_run_ours(prompts, classifications, st_model)

    # Step 4 — Select subset
    selected_ids = load_json(LLMLINGUA_SUBSET_PATH)
    if selected_ids is None:
        selected_ids = step_select_subset(prompts, classifications)
    else:
        print(f'\n[4] Loaded existing subset: {len(selected_ids)} ids')

    # Step 5 — Run LLMLingua
    if args.skip_llmlingua:
        llmlingua_results = load_json(LLMLINGUA_RESULTS_PATH, {})
        print(f'[5] Skipped LLMLingua; loaded {len(llmlingua_results)} cached results.')
    else:
        llmlingua_results = step_run_llmlingua(selected_ids, prompts, st_model)

    # Step 6 — Report
    step_report(prompts, classifications, our_results, llmlingua_results)
    return 0


if __name__ == '__main__':
    sys.exit(main())
