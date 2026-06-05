"""
Reclassify the "other" bucket in classifications.json using GPT-4o-mini.

Why
---
The keyword-based classifier in src/benchmark.py:step_classify leaves a
chunk of prompts in the catch-all "other" bucket. Those prompts drag down
SECTION 2's by-domain analysis because they're aggregated into a single
opaque group. This script asks GPT-4o-mini to assign each "other" prompt
to one of seven defined domains, and writes the result back to
classifications.json.

Only "other" entries are touched. Anything already classified is left as-is.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CLASSIFICATIONS_PATH = ROOT / "data/results/benchmark/classifications.json"
PROMPTS_PATH = ROOT / "data/results/benchmark/all_prompts.json"

# Project convention: .env lives at the repo root (see src/evaluators.py).
# Load it explicitly so OPENAI_API_KEY is available regardless of cwd.
load_dotenv(dotenv_path=ROOT / ".env")

# Valid domains for the GPT classifier. Anything outside this set falls back
# to "general" — see the prompt below.
VALID_DOMAINS = {
    "customer_service",
    "developer_tools",
    "marketing",
    "research",
    "roleplay",
    "business",
    "general",
}

SYSTEM_PROMPT = """Classify this system prompt into exactly one domain. Reply with only the key, nothing else.

customer_service  — support agents, helpdesks, complaint handling, live chat
developer_tools   — coding, debugging, code review, APIs, technical documentation
marketing         — copywriting, content creation, social media, brand voice
research          — academic research, tutoring, explaining concepts, analysis
roleplay          — character personas, fiction, interactive stories, games
business          — project management, consulting, professional services, strategy
general           — genuinely no clear domain fit"""


def first_n_words(text: str, n: int = 200) -> str:
    words = text.split()
    return " ".join(words[:n])


def classify_one(client, prompt_text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": first_n_words(prompt_text)},
        ],
        max_tokens=10,
        temperature=0.0,
    )
    raw = (response.choices[0].message.content or "").strip().lower()
    return raw if raw in VALID_DOMAINS else "general"


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set — aborting.")
        return 1

    classifications: dict[str, str] = json.loads(CLASSIFICATIONS_PATH.read_text())
    prompts_by_id: dict[str, dict] = {p["id"]: p for p in json.loads(PROMPTS_PATH.read_text())}

    other_ids = [pid for pid, dom in classifications.items() if dom == "other"]
    if not other_ids:
        print("No 'other' entries to reclassify.")
        return 0

    # Cost estimate: ~280 input tokens + ~10 output tokens per call.
    # GPT-4o-mini pricing: $0.15 / 1M input, $0.60 / 1M output.
    est_in_tokens = len(other_ids) * 280
    est_out_tokens = len(other_ids) * 10
    est_cost = est_in_tokens * 0.15 / 1_000_000 + est_out_tokens * 0.60 / 1_000_000
    print(f"Estimated cost: {len(other_ids)} prompts × ~100 tokens ≈ ${est_cost:.3f}")
    confirm = input("Proceed? [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return 0

    from openai import OpenAI
    client = OpenAI()

    updated: dict[str, str] = {}
    batch_size = 10
    for i, pid in enumerate(other_ids, start=1):
        entry = prompts_by_id.get(pid)
        if entry is None:
            print(f"[{i}/{len(other_ids)}] {pid}: not found in all_prompts.json, skipping")
            continue
        text = entry.get("text", "")
        label = entry.get("label", pid)
        try:
            new_domain = classify_one(client, text)
        except Exception as e:
            print(f"[{i}/{len(other_ids)}] {label}: ERROR {e}; assigning general")
            new_domain = "general"
        updated[pid] = new_domain
        print(f"[{i}/{len(other_ids)}] {label}  ->  {new_domain}  (was: other)")
        if i % batch_size == 0 and i < len(other_ids):
            time.sleep(0.5)

    # Write back: only the "other" entries are touched.
    for pid, dom in updated.items():
        classifications[pid] = dom
    CLASSIFICATIONS_PATH.write_text(json.dumps(classifications, indent=2))
    print(f"\nWrote {len(updated)} updated classifications to {CLASSIFICATIONS_PATH}")

    counts = Counter(classifications.values())
    print("\nNew domain distribution:")
    for dom, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {dom:<22} {n}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
