import argparse
import json
from typing import Dict, List

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from prompt_compress import PromptCompressor
from llmlingua import PromptCompressor as LLMLinguaCompressor

embedder = SentenceTransformer('all-MiniLM-L6-v2')
client = OpenAI()


def alpha_type(value: str):
    """argparse type: accepts the literal 'auto' or any float."""
    if value == "auto":
        return value
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--alpha must be 'auto' or a float, got {value!r}")


def cosine_sim(a: str, b: str) -> float:
    emb_a = embedder.encode(a, convert_to_numpy=True)
    emb_b = embedder.encode(b, convert_to_numpy=True)
    return float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b)))


def llm_judge(original_output: str, compressed_output: str, judge_model: str = "gpt-4o-mini") -> float:
    """Return a score 0-100."""
    prompt = f"""Rate how well the compressed output preserves the meaning and intent of the original output.
Original: {original_output}
Compressed: {compressed_output}
Give a score 0-100 (100 = perfect preservation). Respond with only the number."""
    response = client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return float(response.choices[0].message.content.strip())


def get_llm_response(prompt: str, queries: List[str], model: str = "gpt-4o-mini") -> Dict[str, str]:
    """Run the prompt through an LLM for each query and return outputs."""
    outputs = {}
    for q in queries:
        full_prompt = f"{prompt}\n\nUser: {q}\nAssistant:"
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.7,
        )
        outputs[q] = response.choices[0].message.content
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text or path to a .txt file")
    parser.add_argument("--queries", nargs="+", default=[
        "What is your role and what can you help me with?",
        "Give me an example of your best response to a typical request.",
        "What is your approach to handling ambiguous questions?",
        "What won't you do or what are your limitations?",
    ])
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--response-model", default="gpt-4o-mini")
    parser.add_argument("--llmlingua-rate", type=float, default=0.5)
    # LLMLingua-2 (classifier-based) is the default: LLMLingua v1's
    # iterative_compress_prompt is broken on transformers >= 4.36 (DynamicCache
    # changed past_key_values' unpacking shape). v2 is also faster on CPU and
    # has no 1024-token context ceiling like gpt2.
    parser.add_argument(
        "--llmlingua-model",
        default="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    )
    parser.add_argument(
        "--llmlingua-v1",
        action="store_true",
        help="Use LLMLingua v1 (perplexity-based). Broken on transformers >= 4.36.",
    )
    parser.add_argument("--llmlingua-device", default="cpu")
    parser.add_argument("--output", default="comparison_result.json")
    parser.add_argument(
        "--alpha",
        type=alpha_type,
        default="auto",
        help="Quality/compression trade-off. 'auto' uses the validated "
             "benchmark default (0.3). Higher values favour shorter prompts.",
    )
    args = parser.parse_args()

    if args.prompt.endswith('.txt'):
        with open(args.prompt, 'r') as f:
            prompt_text = f.read()
    else:
        prompt_text = args.prompt

    print("Getting original LLM responses...")
    original_outputs = get_llm_response(prompt_text, args.queries, model=args.response_model)

    print("Compressing with PromptCompressor (ours)...")
    # use_attention_prior=True activates the AttentionInformedOptimiser, which
    # is where the ISR gate is wired in. Without this flag the result dict's
    # isr_score / isr_reason stay None.
    compressor = PromptCompressor(
        use_real_evaluator=False, use_attention_prior=True, alpha=args.alpha,
    )
    alpha_label = "auto" if args.alpha == "auto" else f"{float(args.alpha):.2f}"
    print(f"   alpha={alpha_label} (resolved to {compressor.alpha:.2f})")
    result = compressor.compress(prompt_text, output=False)
    compressed_text = result["compressed_text"]
    our_ratio = result["metrics"]["compression_ratio"]
    isr_score = result.get("isr_score")
    isr_reason = result.get("isr_reason")
    print(f"   Compression: {our_ratio:.1%}")
    if isr_score is not None:
        print(f"   ISR: {isr_score:.3f} — {isr_reason}")
    else:
        print("   ISR: n/a (prompt routed away from BO before gate ran)")

    print("Compressing with LLMLingua...")
    llmlingua = LLMLinguaCompressor(
        model_name=args.llmlingua_model,
        device_map=args.llmlingua_device,
        use_llmlingua2=not args.llmlingua_v1,
    )
    llmlingua_result = llmlingua.compress_prompt(
        [prompt_text],
        rate=args.llmlingua_rate,
    )
    compressed_llmlingua = llmlingua_result["compressed_prompt"]

    print("Getting compressed responses (ours)...")
    our_outputs = get_llm_response(compressed_text, args.queries, model=args.response_model)
    print("Getting compressed responses (LLMLingua)...")
    llmlingua_outputs = get_llm_response(compressed_llmlingua, args.queries, model=args.response_model)

    results = {
        "method": "ours",
        "compression": our_ratio,
        "output_similarities": [],
        "judge_scores": [],
    }
    for q in args.queries:
        sim = cosine_sim(original_outputs[q], our_outputs[q])
        judge = llm_judge(original_outputs[q], our_outputs[q], args.judge_model)
        results["output_similarities"].append(sim)
        results["judge_scores"].append(judge)
    results["avg_osim"] = float(np.mean(results["output_similarities"]))
    results["avg_judge"] = float(np.mean(results["judge_scores"]))

    llmlingua_ratio = 1 - (len(compressed_llmlingua.split()) / max(len(prompt_text.split()), 1))
    results_llm = {
        "method": "llmlingua",
        "compression": llmlingua_ratio,
        "output_similarities": [],
        "judge_scores": [],
    }
    for q in args.queries:
        sim = cosine_sim(original_outputs[q], llmlingua_outputs[q])
        judge = llm_judge(original_outputs[q], llmlingua_outputs[q], args.judge_model)
        results_llm["output_similarities"].append(sim)
        results_llm["judge_scores"].append(judge)
    results_llm["avg_osim"] = float(np.mean(results_llm["output_similarities"]))
    results_llm["avg_judge"] = float(np.mean(results_llm["judge_scores"]))

    print("\n" + "=" * 70)
    print("ORIGINAL PROMPT")
    print("=" * 70)
    print(prompt_text)

    print("\n" + "=" * 70)
    print(
        f"COMPRESSED — OURS ({len(compressed_text.split())} words, "
        f"{our_ratio:.1%} reduction, alpha={compressor.alpha:.2f})"
    )
    if isr_score is not None:
        print(f"ISR: {isr_score:.3f} — {isr_reason}")
    print("=" * 70)
    print(compressed_text)

    print("\n" + "=" * 70)
    print(
        f"COMPRESSED — LLMLINGUA "
        f"({len(compressed_llmlingua.split())} words, {llmlingua_ratio:.1%} reduction)"
    )
    print("=" * 70)
    print(compressed_llmlingua)

    print("\n" + "=" * 70)
    print("COMPARISON REPORT")
    print("=" * 70)
    print(f"{'Metric':<25} {'Ours':<15} {'LLMLingua':<15}")
    print("-" * 55)
    print(f"{'Compression rate':<25} {results['compression']:>14.1%} {results_llm['compression']:>14.1%}")
    print(f"{'Avg output similarity':<25} {results['avg_osim']:>14.3f} {results_llm['avg_osim']:>14.3f}")
    print(f"{'Avg judge score (0-100)':<25} {results['avg_judge']:>14.1f} {results_llm['avg_judge']:>14.1f}")
    denom_ours = max(1 - results['compression'], 1e-6)
    denom_llm = max(1 - results_llm['compression'], 1e-6)
    print(f"{'Quality/compression ratio':<25} {results['avg_judge'] / denom_ours:>14.1f} {results_llm['avg_judge'] / denom_llm:>14.1f}")
    if isr_score is not None:
        print(f"{'ISR (ours)':<25} {isr_score:>14.3f}")
        print(f"{'ISR decision':<25} {isr_reason}")

    # Persist everything to JSON so this run can be re-inspected later.
    results["compressed_prompt"] = compressed_text
    results["isr_score"] = isr_score
    results["isr_reason"] = isr_reason
    results["alpha_used"] = compressor.alpha
    results_llm["compressed_prompt"] = compressed_llmlingua

    with open(args.output, "w") as f:
        json.dump(
            {
                "original_prompt": prompt_text,
                "ours": results,
                "llmlingua": results_llm,
            },
            f,
            indent=2,
        )
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
