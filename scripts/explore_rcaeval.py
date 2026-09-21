"""
Step 1-2 of the RLM Gateway project: pull one real RCAEval case and
understand its actual file layout before writing any RLM code.

Run with:
    uv run scripts/explore_rcaeval.py
"""

import os

import pandas as pd
from huggingface_hub import snapshot_download

# --- Step 1: download just ONE suite's worth of cases, not all 735 ---
# re1* = metrics-only (simplest). Swap to "re2*" once this works, for
# multi-source (metrics + logs + traces).
print("Downloading RE1 suite (metrics-only, smallest)...")
local_path = snapshot_download(
    repo_id="phamquiluan/RCAEval",
    repo_type="dataset",
    allow_patterns="re1*",
)
print(f"Downloaded to: {local_path}\n")

# --- Step 2: walk the actual folder structure — don't guess filenames ---
print("=== Real file layout (first 3 case folders found) ===")
shown = 0
for root, dirs, files in os.walk(local_path):
    if files and shown < 3:
        rel = os.path.relpath(root, local_path)
        print(f"\n{rel}/")
        for f in files:
            print(f"  {f}")
        shown += 1

# --- Step 3: load the index table so we can pick a specific case ---
print("\n=== Loading index table ===")
index_files = [f for f in os.listdir(local_path) if f.endswith(".parquet") and "index" in f.lower()]
if index_files:
    df = pd.read_parquet(os.path.join(local_path, index_files[0]))
    print(df[["case", "system_name", "fault", "root_cause_service", "has_logs", "has_traces"]].head(10))
else:
    print("No obvious index file found at top level — check the walk output above")
    print("for where the case metadata actually lives, then load it directly.")

# --- Once you can see real filenames above, move this into src/rlm_gateway/tools.py ---
def query_service(case_path: str, service_name: str, metric_name: str | None = None):
    """
    This is the tool function RLM's root model will call from its
    generated REPL code instead of getting the whole case dumped in
    at once. Fill in the real path/column logic once step 2's output
    shows you the actual file structure for a case.
    """
    raise NotImplementedError(
        "Fill this in once you see the real per-case file structure above "
        "(likely a metrics file with a service_name or similar column)."
    )
