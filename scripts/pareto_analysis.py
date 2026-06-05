"""
Pareto frontier analysis — RESEARCH / DEMO USE ONLY

This script sweeps alpha across multiple values to plot the quality/compression
tradeoff. It is NOT the benchmark configuration. The production default is
alpha=0.3, which produced the validated benchmark results (23% avg compression,
84.2 LLM judge score).

To reproduce the validated benchmark numbers use:
    python research/benchmark.py  (uses alpha=0.3 by default)

To demonstrate tunability for the Pareto chart use:
    python scripts/pareto_analysis.py --prompts-file scripts/tier1_prompts.json

ISR filtering
-------------
Prompts with ISR > --isr-high are excluded by default (--skip-dense). They
are not compressible without quality loss, so leaving them in would distort
the frontier toward the trivial "skip = perfect quality" corner.

Quality measurement
-------------------
Default (--quick, on by default): cosine similarity between MiniLM embeddings
of the original and compressed prompts. Fast, no API keys.

Optional --llm-quality: runs fixed queries against an OpenAI chat model for
both prompts and compares response embeddings. More faithful, more expensive.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent

from prompt_compress import PromptCompressor
from prompt_compress.information_sufficiency import compute_isr
from prompt_compress.optimiser import OptimisationConfig
from prompt_compress.semantic_compressor import compute_density


DEFAULT_ALPHAS = [0.5, 0.75, 1.0, 1.25, 1.5]


DEFAULT_PROMPTS = [
    {
        "label": "Astrologer (dense)",
        "text": (
            "You are an expert astrologer with deep knowledge of natal charts, "
            "transits, and synastry. Given a person's birth date, time, and "
            "location, compute the positions of the Sun, Moon, ascendant, and "
            "every planet across the twelve houses. Interpret each placement "
            "in terms of psychological archetype, life domain, and contemporary "
            "predictive context. Identify major aspects (conjunction, opposition, "
            "trine, square, sextile) and explain their developmental tension. "
            "Cross-reference current transits to surface timing-sensitive themes. "
            "Conclude with three concrete behavioural suggestions tailored to "
            "the chart's dominant element and modality."
        ),
    },
    {
        "label": "Helpful assistant (verbose)",
        "text": (
            "You are a very helpful assistant. You are always extremely helpful "
            "and try your absolute best to help the user with whatever they "
            "need. Be very polite, very courteous, and very respectful in all "
            "of your responses. Always greet the user warmly and thank them "
            "for their question. If you do not know the answer, simply say so "
            "politely and offer to help in another way. Always be patient. "
            "Always be kind. Always be considerate. Always be thorough."
        ),
    },
    {
        "label": "Code reviewer (procedural)",
        "text": (
            "You are a senior code reviewer. For each pull request, perform "
            "the following review steps in order: (1) read the PR description "
            "and confirm the stated intent matches the diff, (2) scan for "
            "obvious bugs — off-by-one, null dereferences, race conditions, "
            "(3) check test coverage for new branches, (4) flag any "
            "performance-sensitive paths that lack benchmarks, (5) verify "
            "logging and error messages are actionable, (6) note style "
            "deviations only if they obscure intent. End with a short verdict: "
            "approve, request changes, or comment."
        ),
    },
    {
        "label": "Translator (templated)",
        "text": (
            "You are a professional English-to-French translator. Translate "
            "the following text faithfully, preserving register and idiom: "
            "{source_text}. If a phrase has no direct French equivalent, "
            "render its closest cultural analogue and append a footnote in "
            "brackets explaining the choice. Maintain the original paragraph "
            "structure. Do not summarise, expand, or interpret — translate."
        ),
    },
    {
        "label": "Cooking helper (chatty)",
        "text": (
            "You are a friendly cooking helper. When the user asks for a "
            "recipe, walk them through it step by step. Ask about dietary "
            "restrictions and available equipment first. Suggest substitutions "
            "for any ingredient they don't have. Estimate prep and cook times. "
            "Mention common mistakes to avoid. After the recipe, offer a "
            "wine pairing if appropriate. Keep the tone warm and encouraging. "
            "Be patient with beginner questions. Convert units on request."
        ),
    },
    {
        "label": "Summariser",
        "text": (
            "You are a precise summariser. Given any document, produce a "
            "summary that is one-tenth the length of the original, preserves "
            "all proper nouns and quantitative claims, and reflects the "
            "author's stated position without editorial commentary. Do not "
            "include phrases like 'the author argues' — write declaratively. "
            "If the document contains contradictions, surface them in a final "
            "'tensions' paragraph."
        ),
    },
    {
        "label": "Polite redundant",
        "text": (
            "Please help the user. Please help the user. Please help the user. "
            "Please help the user. Help the user please. Help the user please. "
            "The user needs help. The user needs help. The user needs help. "
            "Be helpful to the user. Be helpful to the user."
        ),
    },
]


_embedder: Optional[SentenceTransformer] = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedder


def cosine_sim(a: str, b: str) -> float:
    model = get_embedder()
    emb = model.encode([a, b], convert_to_numpy=True, show_progress_bar=False)
    va, vb = emb[0], emb[1]
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.dot(va, vb) / denom) if denom else 0.0


def quality_quick(original_prompt: str, compressed_prompt: str) -> float:
    """Embedding similarity between the two prompts. Fast, no API."""
    return cosine_sim(original_prompt, compressed_prompt)


def quality_llm(original_prompt: str, compressed_prompt: str, queries: list[str], model: str) -> float:
    """
    Run each query against both prompts via OpenAI and average cosine sim of
    the responses. Requires OPENAI_API_KEY. Slow.
    """
    from openai import OpenAI
    client = OpenAI()
    sims: list[float] = []
    for q in queries:
        orig_resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"{original_prompt}\n\nUser: {q}\nAssistant:"}],
            temperature=0.7,
        ).choices[0].message.content
        comp_resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"{compressed_prompt}\n\nUser: {q}\nAssistant:"}],
            temperature=0.7,
        ).choices[0].message.content
        sims.append(cosine_sim(orig_resp, comp_resp))
    return float(np.mean(sims)) if sims else 0.0


def load_prompts(path: Optional[str]) -> list[dict]:
    if not path:
        return DEFAULT_PROMPTS
    with open(path) as f:
        raw = json.load(f)
    # Accept either a list of strings or a list of {label, text} dicts.
    normalised: list[dict] = []
    for i, item in enumerate(raw):
        if isinstance(item, str):
            normalised.append({"label": f"prompt_{i}", "text": item})
        else:
            normalised.append({"label": item.get("label", f"prompt_{i}"), "text": item["text"]})
    return normalised


def run_llmlingua(prompts: list[tuple[str, str]], rate: float, model_name: str, device: str,
                  use_v1: bool, quality_fn) -> Optional[dict]:
    try:
        from llmlingua import PromptCompressor as L
    except Exception as e:
        print(f"LLMLingua unavailable: {e}")
        return None
    try:
        c = L(model_name=model_name, device_map=device, use_llmlingua2=not use_v1)
    except Exception as e:
        print(f"LLMLingua init failed: {e}")
        return None

    comp_list: list[float] = []
    qual_list: list[float] = []
    for label, text in prompts:
        try:
            r = c.compress_prompt([text], rate=rate)
            compressed = r["compressed_prompt"]
        except Exception as e:
            print(f"LLMLingua failed on '{label}': {e}")
            continue
        comp_ratio = 1 - len(compressed.split()) / max(len(text.split()), 1)
        q = quality_fn(text, compressed)
        comp_list.append(comp_ratio)
        qual_list.append(q)
    if not comp_list:
        return None
    return {"compression": comp_list, "quality": qual_list}


def silence_bo_output(fn, *args, **kwargs):
    """Run fn() with stdout suppressed — BO prints per-iteration logs we don't need here."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*args, **kwargs)


def _tier_label(density: float) -> str:
    if density >= 0.85:
        return "T3"
    if density >= 0.50:
        return "T2"
    return "T1"


def run_sweep(prompts: list[dict], alphas: list[float], skip_dense: bool, isr_high: float,
              quality_fn) -> tuple[dict, list[tuple[str, str]], list[dict]]:
    compressible: list[tuple[str, str]] = []
    skipped: list[dict] = []
    for p in prompts:
        text, label = p["text"], p["label"]
        isr = compute_isr(text)
        if skip_dense and isr > isr_high:
            print(f"⏭️ Skipping prompt '{label}' (ISR={isr:.2f}) – too dense.")
            skipped.append({"label": label, "isr": isr})
            continue
        compressible.append((label, text))

    # Diagnostic header: density + tier per prompt. Alpha only affects the
    # BO objective, so prompts routed to Tier 2/3 will appear flat no matter
    # what alpha you use.
    print(f"\nRunning alpha sweep on {len(compressible)} compressible prompts...")
    print(f"{'prompt':<40} {'density':>8} {'tier':>5}")
    for label, text in compressible:
        d = compute_density(text)['density']
        print(f"{label[:40]:<40} {d:>8.3f} {_tier_label(d):>5}")
    print()

    results: dict[float, dict[str, list[float]]] = {
        alpha: {"compression": [], "quality": []} for alpha in alphas
    }
    # Per-prompt per-alpha table so flatness is auditable.
    per_prompt: list[dict] = []

    bo_config = OptimisationConfig(n_iterations=15, n_init=4, beta=2.0, random_seed=42)
    for label, text in compressible:
        row: dict = {"label": label}
        for alpha in alphas:
            compressor = PromptCompressor(
                use_real_evaluator=False,
                use_attention_prior=True,
                optimisation_config=bo_config,
                alpha=alpha,
            )
            r = silence_bo_output(compressor.compress, text, output=False)
            comp_ratio = r["metrics"]["compression_ratio"]
            q = quality_fn(text, r["compressed_text"])
            results[alpha]["compression"].append(comp_ratio)
            results[alpha]["quality"].append(q)
            row[alpha] = comp_ratio
        per_prompt.append(row)

    print(f"{'prompt':<40} " + " ".join(f"α={a:<6}" for a in alphas))
    for row in per_prompt:
        cells = " ".join(f"{row[a]:>6.1%} " for a in alphas)
        print(f"{row['label'][:40]:<40} {cells}")

    # Flat-curve diagnostic. The compression coming out of BO can be invariant
    # in alpha when UCB converges to the same best structure for every alpha
    # (alpha changes the objective's score, not the chosen point). Common
    # causes: short BO budget (n_init + n_iterations), strong attention/random
    # priors that bias candidate generation, or instruction_length saturating
    # at the 0.1 clip floor in PromptStructure.
    per_alpha_means = {a: float(np.mean(results[a]["compression"])) for a in alphas}
    spread = max(per_alpha_means.values()) - min(per_alpha_means.values())
    if spread < 0.01:
        print(
            f"\n[DIAGNOSTIC] Compression varies by <1% across the alpha sweep "
            f"(spread = {spread:.1%}). BO is converging to the same best "
            f"structure regardless of alpha. To widen the Pareto curve: "
            f"increase BO budget (n_iterations / n_candidates), raise UCB "
            f"beta for more exploration, or sweep a smaller alpha range that "
            f"doesn't saturate the structure space.\n"
        )

    return results, compressible, skipped


def summarise_and_plot(results: dict, lingua: Optional[dict], output_plot: str) -> None:
    alphas = sorted(results.keys())

    print()
    print(f"{'Alpha':<10} | {'Compression':<15} | {'Quality':<10}")
    print("-" * 42)
    rows: list[dict] = []
    for alpha in alphas:
        c = np.array(results[alpha]["compression"]) if results[alpha]["compression"] else np.array([0.0])
        q = np.array(results[alpha]["quality"]) if results[alpha]["quality"] else np.array([0.0])
        rows.append({
            "alpha": alpha,
            "comp_mean": float(c.mean()), "comp_std": float(c.std()),
            "qual_mean": float(q.mean()), "qual_std": float(q.std()),
        })
        print(f"{alpha:<10.2f} | {c.mean():>13.1%} | {q.mean():>9.3f}")

    lingua_row = None
    if lingua and lingua["compression"]:
        lc = np.array(lingua["compression"])
        lq = np.array(lingua["quality"])
        lingua_row = {
            "comp_mean": float(lc.mean()), "comp_std": float(lc.std()),
            "qual_mean": float(lq.mean()), "qual_std": float(lq.std()),
        }
        print(f"{'LLMLingua':<10} | {lc.mean():>13.1%} | {lq.mean():>9.3f}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        csv_path = Path(output_plot).with_suffix('.csv')
        print(f"matplotlib not available; writing CSV instead: {csv_path}")
        with open(csv_path, 'w') as f:
            f.write("method,alpha,compression,quality\n")
            for r in rows:
                f.write(f"ours,{r['alpha']},{r['comp_mean']},{r['qual_mean']}\n")
            if lingua_row:
                f.write(f"llmlingua,,{lingua_row['comp_mean']},{lingua_row['qual_mean']}\n")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    xs = [r["comp_mean"] for r in rows]
    ys = [r["qual_mean"] for r in rows]
    yerr = [r["qual_std"] for r in rows]
    xerr = [r["comp_std"] for r in rows]
    ax.errorbar(xs, ys, xerr=xerr, yerr=yerr, fmt='o-', label='Ours (α sweep)', capsize=4, color='#2a7ae2')
    for r, x, y in zip(rows, xs, ys):
        ax.annotate(f"α={r['alpha']}", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)

    if lingua_row:
        ax.errorbar(
            [lingua_row["comp_mean"]], [lingua_row["qual_mean"]],
            xerr=[lingua_row["comp_std"]], yerr=[lingua_row["qual_std"]],
            fmt='s', color='#d62728', label='LLMLingua', capsize=4, markersize=8,
        )

    ax.set_xlabel("Compression rate (1 - compressed_tokens / original_tokens)")
    ax.set_ylabel("Output quality (cosine similarity)")
    ax.set_title("Pareto Frontier: Compression vs. Quality Preservation (ISR-filtered)")
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_plot, dpi=150)
    print(f"\nPareto plot saved to {output_plot}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts-file", default=None,
                        help="Path to JSON list of prompts. Each item: string or {label, text}.")
    parser.add_argument("--output-plot", default="pareto_frontier.png")
    parser.add_argument("--skip-dense", action="store_true", default=True,
                        help="Exclude prompts with ISR > --isr-high (default).")
    parser.add_argument("--no-skip-dense", dest="skip_dense", action="store_false")
    parser.add_argument("--isr-high", type=float, default=0.85)
    parser.add_argument("--alpha-list", default=None,
                        help="Comma-separated alphas, e.g. '0.5,1.0,1.5'. Default: 0.5,0.75,1.0,1.25,1.5.")
    parser.add_argument("--llmlingua-rate", type=float, default=0.5)
    parser.add_argument("--llmlingua-model", default="microsoft/llmlingua-2-xlm-roberta-large-meetingbank")
    parser.add_argument("--llmlingua-device", default="cpu")
    parser.add_argument("--llmlingua-v1", action="store_true",
                        help="Use LLMLingua v1 (broken on transformers >= 4.36).")
    parser.add_argument("--skip-llmlingua", action="store_true",
                        help="Skip LLMLingua entirely (faster, useful for quick demos).")
    parser.add_argument("--llm-quality", action="store_true",
                        help="Measure quality via real LLM responses (needs OPENAI_API_KEY).")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--llm-queries", nargs="+", default=[
        "What can you help me with?",
        "Give me a typical response example.",
    ])
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("[WARNING] Pareto sweep uses non-default alpha values.")
    print("          Benchmark headline numbers (23% compression, judge=84.2)")
    print("          were produced with alpha=0.3. This sweep is for research only.")
    print()

    prompts = load_prompts(args.prompts_file)
    print(f"Loading {len(prompts)} prompts...")

    alphas = (
        [float(a) for a in args.alpha_list.split(",")]
        if args.alpha_list else DEFAULT_ALPHAS
    )

    if args.llm_quality:
        if not os.environ.get("OPENAI_API_KEY"):
            print("--llm-quality requires OPENAI_API_KEY; falling back to --quick mode.")
            quality_fn = quality_quick
        else:
            quality_fn = lambda a, b: quality_llm(a, b, args.llm_queries, args.llm_model)
    else:
        quality_fn = quality_quick

    results, compressible, _ = run_sweep(prompts, alphas, args.skip_dense, args.isr_high, quality_fn)

    lingua = None
    if not args.skip_llmlingua and compressible:
        print("Running LLMLingua...")
        lingua = run_llmlingua(
            compressible, args.llmlingua_rate, args.llmlingua_model,
            args.llmlingua_device, args.llmlingua_v1, quality_fn,
        )

    summarise_and_plot(results, lingua, args.output_plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
