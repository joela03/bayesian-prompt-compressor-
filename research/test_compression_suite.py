"""
Compression Test Suite
======================
Tests the compression pipeline across 6 diverse prompt types and gives
a diagnostic breakdown: what was parsed, what was kept/removed, and
where the system succeeds or struggles.

Run from the src/ directory:
    python test_compression_suite.py
or with informed priors:
    python test_compression_suite.py --informed
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Inline test prompts – six archetypes covering the real use-case space
# ---------------------------------------------------------------------------

TEST_PROMPTS = {

    "verbose_customer_service": {
        "label": "Verbose Customer Service",
        "expected_compression": "HIGH (30-40%)",
        "text": """You are a helpful, friendly, and professional customer service representative
for a technology company. Your primary goal is to make every customer feel heard,
valued, and supported throughout their experience with our products.

When a customer reaches out, always start by warmly greeting them and acknowledging
their concern. Make sure the customer knows you genuinely care about resolving their
issue as quickly and effectively as possible. Use empathetic language throughout the
conversation and always remain calm and patient, even when customers are frustrated
or upset.

For example, if a customer says they are frustrated, you might say something like:
"I completely understand how frustrating that must be, and I am so sorry for the
inconvenience. Let me do everything I can to help you today."

Another example: if a customer asks about billing, always verify their account
information first before proceeding. For instance, you could ask: "Could you please
confirm your account email address so I can pull up your details?"

Guidelines and rules you must always follow:
- Do not share internal company pricing strategies or unreleased product information
- Always escalate to a senior agent if a customer requests to speak to a manager
- Never promise a specific resolution time without checking with the relevant team first
- Maintain a positive, upbeat tone at all times

Your communication style should be warm, conversational, and approachable. Avoid
overly formal or robotic language. Write the way a real person would speak, not
the way a policy document reads. Short sentences are better than long, complex ones.

Context: You are working in the live chat support channel for our SaaS product.
Most customers are non-technical users who need simple, jargon-free guidance.""",
    },

    "dense_technical_spec": {
        "label": "Dense Technical Specification",
        "expected_compression": "LOW (3-8%)",
        "text": """You are an API integration specialist. Parse the following endpoint specification
and generate a strictly compliant HTTP request.

Endpoint: POST /api/v3/payments/authorize
Authentication: Bearer token in Authorization header, format: "Bearer {token}"
Content-Type: application/json; charset=utf-8

Required fields:
- merchant_id: string, UUID v4 format, no hyphens
- amount: integer, value in lowest currency denomination (pence/cents), max 999999
- currency: string, ISO 4217 three-letter code, uppercase only
- card_token: string, 32-character alphanumeric, returned from /api/v3/tokenize
- idempotency_key: string, UUID v4, must be unique per transaction attempt

Optional fields:
- descriptor: string, max 22 chars, alphanumeric plus spaces only
- metadata: object, max 10 keys, values must be strings, max 255 chars per value

Response codes:
- 200: Authorized. Body contains authorization_code and expires_at (ISO 8601).
- 402: Declined. Body contains decline_code. Do not retry without new card_token.
- 422: Validation error. Body contains field-level errors array.
- 429: Rate limited. Retry after value in Retry-After header (seconds).

Constraints:
- NEVER log or store card_token values
- NEVER retry a 402 response with the same card_token
- Idempotency_key MUST be regenerated for each new authorization attempt
- Amount MUST be validated as a positive integer before sending""",
    },

    "medium_research_assistant": {
        "label": "Research Assistant (Medium Density)",
        "expected_compression": "MEDIUM (15-25%)",
        "text": """You are an expert AI assistant specialising in academic research support.
Your role is to help researchers, students, and academics find, understand,
and synthesise information across a wide range of disciplines.

When answering questions, follow these guidelines:
1. Begin with a direct, clear answer to the question before elaborating
2. Cite credible sources and distinguish between established consensus and active debate
3. Use precise, professional academic language while remaining accessible
4. Acknowledge the limitations of your knowledge and areas of genuine uncertainty
5. Structure longer responses with clear sections when helpful

For example, a good response to "What caused the 2008 financial crisis?" would
start with a two-sentence direct answer covering the core causes, then provide
a structured breakdown of contributing factors with appropriate references.

Constraints:
- Do not provide specific investment, legal, or medical advice
- Do not present contested claims as settled facts
- Avoid speculation beyond what evidence supports
- Keep responses under 400 words unless the question explicitly requires more depth

Style: Academic but accessible. Authoritative without being condescending.
Precision is more important than comprehensiveness — a shorter, accurate answer
beats a longer vague one.""",
    },

    "short_simple": {
        "label": "Short Simple Prompt",
        "expected_compression": "VERY LOW (0-5%) — too short to compress",
        "text": """You are a helpful assistant. Answer questions clearly and concisely.
Be accurate and honest. If you don't know something, say so.
Keep responses under 150 words.""",
    },

    "creative_director": {
        "label": "Creative Director (Long, Repetitive)",
        "expected_compression": "HIGH (25-40%)",
        "text": """You are an expert creative director specializing in crafting multi-layered,
evocative subject lines for artistic content. Your subject lines serve as
conceptual bridges between themes and visual execution.

Core Creative Philosophy:
1. Multi-Dimensional Approach: Create subject lines with multiple interconnected
   layers that contribute to a unified theme. Avoid single-concept descriptions.
2. Tangible Scene Construction: Use specific, concrete details to construct scenes
   rather than abstract statements. Show, don't tell.
3. Meaningful Juxtaposition: Incorporate contrasting or paradoxical elements that
   create tension and depth.
4. Unconventional Connections: Make unexpected associations between visual elements,
   emotions, and concepts.
5. Specificity Over Generality: Replace vague descriptors with precise, sensory-rich
   details.

CRITICAL CREATIVITY RULES:
- The examples provided are STRICTLY FOR INSPIRATION ONLY
- DO NOT copy, paraphrase, or adapt any specific phrases from the examples
- Create something COMPLETELY NEW and ORIGINAL
- Use the examples to understand STRUCTURE and TECHNIQUE, not the content

Subject Line Structure:
1. Opening Hook: Begin with a distinctive visual element or unexpected concept
2. Scene Development: Expand into a mini-narrative with multiple interacting elements
3. Contextual Layering: Incorporate subtle references to the theme through metaphors
4. Technical Precision: Include specific details about composition, lighting, texture

Example Approaches (FOR INSPIRATION ONLY):
1. Theme: White Rain River
   Subject: "Crystalline droplets shatter against obsidian river stones, their
   prismatic fragments dancing in moonlight like scattered diamonds"

2. Theme: Cozy Winter Evening
   Subject: "A tattered copy of A Christmas Carol lies open on a threadbare armchair,
   its pages catching amber glow of a crackling fire"

ORIGINALITY REQUIREMENTS:
- Study examples for TECHNIQUE not content
- DO NOT use same objects, settings, or phrases from examples
- Create entirely new imagery unique to your theme
- Think of completely different visual metaphors and associations

Remember: Every subject line should transport the reader into a rich, visually
compelling world. Be completely original.""",
    },

    "code_reviewer": {
        "label": "Code Reviewer (Structured, Medium)",
        "expected_compression": "MEDIUM (10-20%)",
        "text": """You are a senior software engineer conducting thorough but constructive code reviews.
Your goal is to maintain high code quality while helping developers learn and improve.

Review priorities in order:
1. Correctness: Does the code work? Are edge cases handled?
2. Security: Any vulnerabilities? Is user input properly validated and sanitised?
3. Performance: Obvious inefficiencies? Unscalable patterns?
4. Readability: Will another developer understand this in six months?
5. Testing: Are tests present and meaningful?

When giving feedback, always:
- Be specific — "This could be better" is useless; explain what and why
- Suggest concrete alternatives, not just problems
- Distinguish clearly between blocking issues (must fix) and suggestions (nice to have)
- Acknowledge good work when you see it — positive reinforcement matters

Constraints:
- Never approve code with known security vulnerabilities, regardless of scope
- Never approve untested changes to payment or authentication code
- Flag but do not block on style issues if the codebase has no linter config

Style: Direct and professional. Firm on correctness and security, collaborative on
everything else. Treat the review as a conversation, not an audit.""",
    },

}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def separator(char="=", width=72):
    return char * width


def print_component_breakdown(components: Dict, original_structure, optimised_structure) -> None:
    """Show which components were found and whether the BO kept them."""

    comp_map = {
        "instruction":  ("has_instruction",  "✅"),
        "examples":     ("has_examples",     "📝"),
        "constraints":  ("has_constraints",  "🔒"),
        "style":        ("has_style",        "🎨"),
        "context":      ("has_context",      "📍"),
    }

    print("\n  Component breakdown:")
    print(f"  {'Component':<14} {'Found':<8} {'Original':<12} {'Optimised':<12} {'Decision'}")
    print(f"  {'-'*62}")

    for comp, (attr, icon) in comp_map.items():
        found    = bool(components.get(comp))
        original = getattr(original_structure, attr)
        optimised = getattr(optimised_structure, attr)

        if not found:
            decision = "— not in prompt"
        elif original and optimised:
            decision = "kept ✓"
        elif original and not optimised:
            decision = "REMOVED ✗"
        elif not original and optimised:
            decision = "added (unusual)"
        else:
            decision = "absent"

        sentences = len(components.get(comp, []))
        print(f"  {icon} {comp:<12} {str(found):<8} {str(original):<12} {str(optimised):<12} {decision}  ({sentences} sentences)")


def print_metrics(metrics: Dict) -> None:
    print(f"\n  Token metrics:")
    print(f"    Original:    {metrics['original_tokens']:>5} tokens")
    print(f"    Compressed:  {metrics['compressed_tokens']:>5} tokens")
    print(f"    Saved:       {metrics['tokens_saved']:>5} tokens  ({metrics['compression_ratio']:.1%})")
    print(f"    Score retained: {metrics['performance_retention']:.1%}")


def truncate(text: str, n: int = 200) -> str:
    return text[:n] + "..." if len(text) > n else text


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------

def run_tests(use_informed: bool = False) -> List[Dict]:

    # Import here so the script can live outside src/ if needed
    try:
        from prompt_compress import PromptCompressor
    except ImportError:
        print("ERROR: Could not import PromptCompressor.")
        print("Make sure you run this script from the src/ directory.")
        sys.exit(1)

    prior_label = "INFORMED (P3 priors)" if use_informed else "NAIVE (uniform random)"
    print(separator())
    print(f"  COMPRESSION TEST SUITE  —  {prior_label}")
    print(separator())
    print(f"  Testing {len(TEST_PROMPTS)} prompt archetypes\n")

    compressor = PromptCompressor(
        use_real_evaluator=False,
        use_informed_prior=use_informed,
    )

    summary_rows = []

    for key, prompt_data in TEST_PROMPTS.items():
        print(f"\n{separator()}")
        print(f"  {prompt_data['label'].upper()}")
        print(f"  Expected compression: {prompt_data['expected_compression']}")
        print(separator())

        t0 = time.time()
        result = compressor.compress(prompt_data["text"], output=False)
        elapsed = time.time() - t0

        metrics      = result["metrics"]
        components   = result["components"]
        orig_struct  = result["original_structure"]
        opt_struct   = result["optimised_structure"]
        compressed   = result["compressed_text"]

        print_component_breakdown(components, orig_struct, opt_struct)
        print_metrics(metrics)

        print(f"\n  Time:  {elapsed:.2f}s")
        print(f"  BO evaluations: {result['optimisation_result'].total_evaluations}")

        print(f"\n  Original ({metrics['original_tokens']} tokens):")
        print(f"  {separator('-')}")
        print(f"  {truncate(prompt_data['text'].replace(chr(10), ' '), 300)}")

        print(f"\n  Compressed ({metrics['compressed_tokens']} tokens):")
        print(f"  {separator('-')}")
        print(f"  {truncate(compressed.replace(chr(10), ' '), 300)}")

        # Check if result matched expectation
        cr = metrics["compression_ratio"]
        if "HIGH" in prompt_data["expected_compression"]:
            met = "✅" if cr >= 0.20 else "⚠️ "
        elif "MEDIUM" in prompt_data["expected_compression"]:
            met = "✅" if 0.08 <= cr <= 0.30 else "⚠️ "
        elif "LOW" in prompt_data["expected_compression"]:
            met = "✅" if cr <= 0.12 else "⚠️ "
        else:
            met = "✅" if cr <= 0.05 else "⚠️ "

        summary_rows.append({
            "key":         key,
            "label":       prompt_data["label"],
            "original":    metrics["original_tokens"],
            "compressed":  metrics["compressed_tokens"],
            "ratio":       cr,
            "retained":    metrics["performance_retention"],
            "elapsed":     elapsed,
            "expectation": prompt_data["expected_compression"],
            "met":         met,
        })

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print(f"\n{separator()}")
    print("  SUMMARY TABLE")
    print(separator())

    header = f"  {'Prompt':<30} {'Orig':>5} {'Comp':>5} {'Saved':>6}  {'Ret.':>6}  {'Time':>5}  {'Exp?'}"
    print(header)
    print(f"  {'-'*72}")

    for row in summary_rows:
        print(
            f"  {row['label'][:28]:<30}"
            f"  {row['original']:>4}"
            f"  {row['compressed']:>4}"
            f"  {row['ratio']:>5.1%}"
            f"  {row['retained']:>5.1%}"
            f"  {row['elapsed']:>4.1f}s"
            f"  {row['met']}"
        )

    avg_compression = sum(r["ratio"] for r in summary_rows) / len(summary_rows)
    avg_retention   = sum(r["retained"] for r in summary_rows) / len(summary_rows)
    total_saved     = sum(r["original"] - r["compressed"] for r in summary_rows)
    total_original  = sum(r["original"] for r in summary_rows)

    print(f"  {'-'*72}")
    print(f"  {'AVERAGE':<30}  {total_original:>4}  "
          f"  {total_original - total_saved:>4}  {avg_compression:>5.1%}"
          f"  {avg_retention:>5.1%}")

    print(f"\n  Total tokens saved across all prompts: {total_saved}")

    # ---------------------------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------------------------
    print(f"\n{separator()}")
    print("  DIAGNOSTICS")
    print(separator())

    failed = [r for r in summary_rows if r["met"] == "⚠️ "]
    passed = [r for r in summary_rows if r["met"] == "✅"]

    print(f"\n  ✅ {len(passed)}/{len(summary_rows)} prompts met compression expectations")

    if failed:
        print(f"\n  ⚠️  Prompts that missed expectations:")
        for r in failed:
            print(f"     • {r['label']}: got {r['ratio']:.1%}, expected {r['expectation']}")

    # Check for the known mock-evaluator ceiling effect
    perfect_scores = [r for r in summary_rows if r["retained"] >= 0.99]
    if perfect_scores:
        print(f"\n  ⚠️  Mock evaluator ceiling effect detected:")
        print(f"     {len(perfect_scores)} prompt(s) show 100% retention — the mock evaluator")
        print(f"     cannot penalise semantic loss. Real LLM validation needed to verify quality.")

    # Check if short prompts were over-compressed
    short_compressed = [r for r in summary_rows if r["original"] < 50 and r["ratio"] > 0.05]
    if short_compressed:
        print(f"\n  ⚠️  Short prompts being compressed unnecessarily:")
        for r in short_compressed:
            print(f"     • {r['label']}: only {r['original']} tokens but {r['ratio']:.1%} removed")

    print(f"\n  Prior used: {prior_label}")
    print(f"  Evaluator: MockEvaluator (no real LLM calls)")
    print(f"  Next step: run with --real-eval to validate with GPT-4o-mini")

    return summary_rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run compression test suite")
    parser.add_argument(
        "--informed",
        action="store_true",
        help="Use P3-informed priors instead of naive uniform sampling",
    )
    args = parser.parse_args()

    results = run_tests(use_informed=args.informed)

    # Optionally save results
    output_path = Path("data/results/compression_suite_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            [{k: v for k, v in r.items() if k != "met"} for r in results],
            f,
            indent=2,
        )
    print(f"\n  Results saved to: {output_path}")
