"""
RCAEval dataset access: download, case-id parsing, and per-case loading.

Each case is a folder named "re{suite}{system}_{service}_{fault}_{run}"
containing metrics.parquet (one row per timestamp, one column per
"{service}_{metric}" pair) and inject_time.txt (a single unix timestamp).
The ground-truth root cause is encoded directly in the folder name.

RE1 is metrics-only. RE2/RE3 also include logs.parquet (raw timestamped
log lines: timestamp, container_name, message) and traces.parquet.
"""

import os
import re
from dataclasses import dataclass

import pandas as pd
from huggingface_hub import snapshot_download

DATASET_REPO_ID = "phamquiluan/RCAEval"

# Fault types are spelled out explicitly (rather than matched greedily)
# because service names can themselves contain hyphens, e.g. "ts-auth-service".
CASE_NAME_RE = re.compile(
    r"^re(?P<suite>[123])(?P<system>[a-z]+)_(?P<service>.+)_(?P<fault>cpu|mem|disk|delay|loss)_(?P<run>\d+)$"
)

SYSTEM_NAMES = {"ob": "online-boutique", "ss": "sock-shop", "tt": "train-ticket"}


@dataclass(frozen=True)
class Case:
    case_id: str
    suite: str
    system: str
    root_cause_service: str
    fault_type: str
    run: int


def download_suite(pattern: str = "re1*") -> str:
    """Download a suite of RCAEval cases and return the local snapshot path."""
    return snapshot_download(repo_id=DATASET_REPO_ID, repo_type="dataset", allow_patterns=pattern)


def parse_case_id(case_id: str) -> Case:
    match = CASE_NAME_RE.match(case_id)
    if not match:
        raise ValueError(f"Unrecognized case id: {case_id!r}")
    return Case(
        case_id=case_id,
        suite=f're{match["suite"]}',
        system=SYSTEM_NAMES.get(match["system"], match["system"]),
        root_cause_service=match["service"],
        fault_type=match["fault"],
        run=int(match["run"]),
    )


def list_cases(local_path: str) -> list[Case]:
    return sorted(
        (parse_case_id(name) for name in os.listdir(local_path) if CASE_NAME_RE.match(name)),
        key=lambda c: c.case_id,
    )


def load_case_metrics(local_path: str, case_id: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(local_path, case_id, "metrics.parquet"))


def load_case_logs(local_path: str, case_id: str) -> pd.DataFrame:
    """RE2/RE3 only. Columns: timestamp, container_name, message."""
    path = os.path.join(local_path, case_id, "logs.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No logs.parquet for {case_id!r}. RE1 cases are metrics-only and have no logs -- "
            f"use download_suite(pattern='re2*') (or 're3*') to get log-bearing cases."
        )
    return pd.read_parquet(path)


def load_inject_time(local_path: str, case_id: str) -> int:
    with open(os.path.join(local_path, case_id, "inject_time.txt")) as f:
        return int(f.read().strip())
