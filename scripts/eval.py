"""
Run RLM root-cause prediction across multiple RCAEval cases and report
accuracy. Each case is billed as one real LLM call (plus tool round-trips) --
use --limit to control cost, especially before running the full suite.

Usage:
    uv run scripts/eval.py --limit 5
    uv run scripts/eval.py --limit 20 --backend openai --model gpt-4o-mini
    uv run scripts/eval.py --pattern re1ob_adservice_cpu_1   # one specific case
    uv run scripts/eval.py --strategy logs --pattern re2*    # log-based RE2/RE3 cases
    uv run scripts/eval.py                                    # ALL 375 cases -- costs real money
"""

import argparse
import csv
import time
from pathlib import Path

from dotenv import load_dotenv

from rlm_gateway.dataset import download_suite, list_cases
from rlm_gateway.pipeline import run_case, run_case_direct, run_case_logs

RESULTS_PATH = Path("logs/eval_results.csv")

STRATEGIES = {
    "metrics": run_case,  # RLM + query_service/list_services tools over RE1-RE3 metrics.parquet
    "direct": run_case_direct,  # bypasses RLM -- single precomputed-summary prompt, no tools
    "logs": run_case_logs,  # RLM + log_change_ranking/etc tools over RE2/RE3 logs.parquet
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", default="anthropic", choices=["anthropic", "openai", "openrouter", "local"])
    parser.add_argument("--model", default=None, help="Override the backend's default model")
    parser.add_argument("--strategy", default="metrics", choices=list(STRATEGIES), help="Which run_case* to use")
    parser.add_argument("--limit", type=int, default=5, help="Max number of cases to run (default 5)")
    parser.add_argument("--pattern", default="re1*", help="HF snapshot_download allow_patterns filter")
    parser.add_argument("--verbose", action="store_true", help="Show full per-iteration REPL trace")
    args = parser.parse_args()

    load_dotenv()

    run_fn = STRATEGIES[args.strategy]

    local_path = download_suite(pattern=args.pattern)
    cases = list_cases(local_path)
    if args.limit:
        cases = cases[: args.limit]

    print(f"Running {len(cases)} case(s) strategy={args.strategy} backend={args.backend} model={args.model or '(default)'}\n")

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    fieldnames = ["case_id", "system", "fault_type", "actual", "predicted", "correct", "elapsed_s", "error"]
    correct_count = 0

    # Write each result as it completes, not just at the end -- so killing
    # the run partway through (e.g. out of API credit) doesn't lose already-
    # completed, already-paid-for results.
    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, case in enumerate(cases, 1):
            print(f"[{i}/{len(cases)}] {case.case_id} (actual={case.root_cause_service}) ... ", end="", flush=True)
            start = time.time()
            try:
                kwargs = {"backend": args.backend, "model_name": args.model}
                if args.strategy != "direct":  # run_case_direct has no REPL loop to trace
                    kwargs["verbose"] = args.verbose
                prediction = run_fn(local_path, case, **kwargs)
                correct = case.root_cause_service in prediction
                error = None
            except Exception as exc:  # one bad case shouldn't kill the whole batch
                prediction = None
                correct = False
                error = str(exc)

            elapsed = time.time() - start
            correct_count += correct
            status = "OK" if error is None else f"ERROR: {error}"
            print(f"predicted={prediction} correct={correct} ({elapsed:.1f}s) {status if error else ''}")

            writer.writerow(
                {
                    "case_id": case.case_id,
                    "system": case.system,
                    "fault_type": case.fault_type,
                    "actual": case.root_cause_service,
                    "predicted": prediction,
                    "correct": correct,
                    "elapsed_s": round(elapsed, 1),
                    "error": error or "",
                }
            )
            f.flush()

    accuracy = correct_count / len(cases) if cases else 0.0
    print(f"\nAccuracy: {correct_count}/{len(cases)} ({accuracy:.1%})")
    print(f"Results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
