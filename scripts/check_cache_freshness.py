"""
Check whether evaluate.py's caches are still aligned with our_results.json.

evaluate.py builds judge scores by running LLM queries against the compressed
prompts in our_results.json and caching the responses in probing_responses.json
and judge_results.json. If our_results.json has been regenerated (e.g. by a
benchmark.py rerun) but those caches haven't been refreshed, the judge scores
end up applying to stale compressed outputs — invisible bug, very misleading
headline numbers.

This script compares mtimes and deletes caches that predate our_results.json
so the next evaluate.py run is forced to regenerate them. It does NOT run
evaluate.py — that's a paid LLM call and should be a deliberate user action.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = ROOT / "data/results/benchmark"
OUR_RESULTS = BENCHMARK_DIR / "our_results.json"
PROBING_RESPONSES = BENCHMARK_DIR / "probing_responses.json"
JUDGE_RESULTS = BENCHMARK_DIR / "judge_results.json"


def _fmt_mtime(path: Path) -> str:
    if not path.exists():
        return "(missing)"
    import datetime as dt
    return dt.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")


def main() -> int:
    if not OUR_RESULTS.exists():
        print(f"{OUR_RESULTS} does not exist — nothing to check.")
        return 1

    our_mtime = OUR_RESULTS.stat().st_mtime
    print(f"our_results.json       {_fmt_mtime(OUR_RESULTS)}  (reference)")
    print(f"probing_responses.json {_fmt_mtime(PROBING_RESPONSES)}")
    print(f"judge_results.json     {_fmt_mtime(JUDGE_RESULTS)}")
    print()

    def maybe_delete(path: Path, label: str) -> None:
        if not path.exists():
            print(f"{label} not present — nothing to delete.")
            return
        if path.stat().st_mtime < our_mtime:
            print(f"{label} is stale — deleting so evaluate.py regenerates it")
            os.remove(path)
            print("Deleted.")
        else:
            print(f"{label} is current — no action needed")

    maybe_delete(PROBING_RESPONSES, "probing_responses.json")
    maybe_delete(JUDGE_RESULTS, "judge_results.json")

    print()
    print("Stale caches cleared. Run `python src/evaluate.py` to regenerate "
          "judge scores")
    print("against the current compressed outputs from our_results.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
