"""
Builds an RLM instance with query_service/list_services as custom tools,
runs it on one RCAEval case, and checks the prediction against ground truth.
"""

import os

from rlm import RLM
from rlm.clients.anthropic import AnthropicClient
from rlm.clients.openai import OpenAIClient

from . import _rlm_compat  # noqa: F401  (patches AnthropicClient's thinking-block bug)
from .dataset import Case, download_suite, load_case_logs, load_case_metrics, load_inject_time
from .tools import list_categories, list_services, make_query_service_tool, summarize_metrics

PROMPT_TEMPLATE = """\
You are diagnosing a microservice incident. A fault was injected at unix \
timestamp {inject_time} into the "{system}" system, which has these \
services: {services}.

You have two tools:
- list_services() -> list[str]
- query_service(service_name, category=None) -> dict with a "time" key and \
one key per metric column. category is one of: {categories}.

Each query can return hundreds of raw data points. Do NOT print raw \
query_service() results directly -- compute and print a summary instead \
(e.g. mean/min/max before vs. after the injection timestamp).

Use them to inspect metrics around the injection time and figure out which \
service is the root cause of the incident. Respond with ONLY the service \
name as your final answer.
"""


# "local" runs a free, zero-API-cost model via Ollama's OpenAI-compatible
# server (rlm routes backend="vllm" to its generic OpenAI-compatible client,
# which just needs base_url pointed at localhost -- Ollama doesn't check
# the api_key value at all, so any placeholder works).
BACKEND_CONFIGS = {
    "anthropic": {"rlm_backend": "anthropic", "api_key_env": "ANTHROPIC_API_KEY", "default_model": "claude-sonnet-5"},
    "openai": {"rlm_backend": "openai", "api_key_env": "OPENAI_API_KEY", "default_model": "gpt-4o-mini"},
    # Free tier: 50 requests/day on an unfunded OpenRouter account. qwen3-coder
    # is benchmarked as OpenRouter's strongest free option for tool-calling/
    # agentic tasks specifically -- a different model family, not just a
    # bigger download of the same one that struggled locally.
    "openrouter": {
        "rlm_backend": "openrouter",
        "api_key_env": "OPENROUTER_API_KEY",
        "default_model": "qwen/qwen3-coder:free",
        # Normally rlm's own routing fills this in for backend="openrouter",
        # but run_case_direct() calls OpenAIClient directly, bypassing that --
        # so it's set explicitly here to work for both code paths.
        "base_url": "https://openrouter.ai/api/v1",
    },
    "local": {
        "rlm_backend": "vllm",
        "api_key_env": None,
        "default_model": "qwen2.5:7b",
        "base_url": "http://localhost:11434/v1",
    },
}


def run_case(
    local_path: str,
    case: Case,
    backend: str = "anthropic",
    model_name: str | None = None,
    verbose: bool = True,
) -> str:
    metrics_df = load_case_metrics(local_path, case.case_id)
    inject_time = load_inject_time(local_path, case.case_id)
    services = list_services(metrics_df)
    categories = list_categories(metrics_df)

    config = BACKEND_CONFIGS[backend]
    model_name = model_name or config["default_model"]
    api_key = os.getenv(config["api_key_env"]) if config["api_key_env"] else "ollama"

    backend_kwargs = {"model_name": model_name, "api_key": api_key}
    if "base_url" in config:
        backend_kwargs["base_url"] = config["base_url"]

    rlm = RLM(
        backend=config["rlm_backend"],
        backend_kwargs=backend_kwargs,
        environment="local",
        custom_tools={
            "list_services": {
                "tool": lambda: services,
                "description": "List every service name present in this case's metrics.",
            },
            "query_service": {
                "tool": make_query_service_tool(metrics_df),
                "description": (
                    "Fetch metrics for one service, optionally filtered to a "
                    "category (cpu, mem, workload, error, latency-50, latency-90)."
                ),
            },
        },
        verbose=verbose,
    )

    prompt = PROMPT_TEMPLATE.format(
        inject_time=inject_time,
        system=case.system,
        services=", ".join(services),
        categories=", ".join(categories),
    )
    result = rlm.completion(prompt)
    return result.response.strip()


DIRECT_PROMPT_TEMPLATE = """\
A fault was injected into the "{system}" system at unix timestamp {inject_time}.
Below is a summary comparing each service's metrics {window}s before vs. \
{window}s after the injection (mean value and percent change per category):

{summary}

Which service is the root cause of the incident? Respond with ONLY the \
service name, nothing else.
"""


def run_case_direct(
    local_path: str,
    case: Case,
    backend: str = "anthropic",
    model_name: str | None = None,
    window: int = 30,
) -> str:
    """
    Single-shot alternative to run_case(): precomputes the pre/post-injection
    summary ourselves and asks the model one direct question, with no tools
    and no multi-turn REPL loop for it to lose track of. Bypasses RLM
    entirely -- calls the underlying model client straight.
    """
    metrics_df = load_case_metrics(local_path, case.case_id)
    inject_time = load_inject_time(local_path, case.case_id)
    summary = summarize_metrics(metrics_df, inject_time, window=window)

    config = BACKEND_CONFIGS[backend]
    model_name = model_name or config["default_model"]
    api_key = os.getenv(config["api_key_env"]) if config["api_key_env"] else "ollama"

    client_kwargs = {"api_key": api_key, "model_name": model_name}
    if "base_url" in config:
        client_kwargs["base_url"] = config["base_url"]

    client = AnthropicClient(**client_kwargs) if config["rlm_backend"] == "anthropic" else OpenAIClient(**client_kwargs)

    prompt = DIRECT_PROMPT_TEMPLATE.format(
        system=case.system, inject_time=inject_time, summary=summary, window=window
    )
    return client.completion(prompt).strip()


LOGS_ROOT_PROMPT = """\
A fault was injected into the "{system}" system at unix timestamp \
{inject_time}. The context is a LIST of {num_services} strings, one chunk \
per service -- index into it like context[0], context[1], etc. Do NOT \
slice it like a string (e.g. context[:1000] returns whole list items, not \
characters, since there are only {num_services} items total). Each chunk \
covers {window}s before and after the injection.

You have three tools -- use them instead of writing your own \
parsing/counting/ranking code, which is error-prone:
- `log_change_ranking() -> list[dict]`: for EVERY service, exact before/after \
log-line counts around the injection plus "percent_change", already sorted \
by |percent_change| descending (most anomalous first). Each dict also has \
an "index" field -- reuse that directly, do NOT try to re-derive an index \
with context.index(...) (that fails: context holds strings, not these dicts). \
ALWAYS call this FIRST and use "percent_change" to judge anomalies, never \
raw count differences -- a busy gateway service can have a huge raw count \
difference just because it logs the most overall, even with a small/normal \
relative change. A service that drops sharply (large negative percent_change) \
is just as suspicious as one that spikes -- some faults (e.g. high CPU load) \
reduce throughput rather than raising it.
- `service_name_at(index: int) -> str`: the exact service name for \
context[index]. ALWAYS get your final answer from this (or straight from \
log_change_ranking()'s "service" field), never by parsing the \
"=== service: X ===" header text yourself (it's easy to include the \
trailing "===" by mistake).
- `count_logs_in_range(index: int, start_time: float, end_time: float) -> int`: \
exact log-line count for context[index]'s service within \
[start_time, end_time] (unix seconds), for any follow-up drill-down you need \
beyond the standard before/after windows.

IMPORTANT: write all code inside ```repl blocks, never ```python blocks -- \
only ```repl blocks are actually executed. A ```python block is silently \
ignored and does nothing, even if it looks correct.

IMPORTANT: this REPL is NOT a Jupyter notebook. A bare expression on the \
last line (e.g. `my_list[:5]`) produces NO visible output and is silently \
discarded -- you will NOT see the result. You MUST wrap anything you want \
to see in print(...), e.g. `print(my_list[:5])`, every single time, \
including for intermediate values you're using to decide your next step.

After calling log_change_ranking(), also skim the actual message content \
(context[i]) of the top few candidates for a shift in what kind of messages \
they're logging (new error types, stalls, repeated retries) to confirm the \
ranking makes sense. Do NOT just search for the literal words "error" or \
"abnormal" -- a fault often produces no log line containing those words.

You MUST respond with exactly one specific service name, from \
log_change_ranking() or service_name_at(), as your final answer, even if no \
service looks obviously broken -- pick the one that looks most different \
from its own earlier behavior. "No anomaly found" is not an acceptable \
final answer.

Respond with ONLY the service name as your final answer.
"""


def run_case_logs(
    local_path: str,
    case: Case,
    backend: str = "anthropic",
    model_name: str | None = None,
    verbose: bool = True,
    window: int = 15,
    max_iterations: int = 15,
) -> str:
    """
    RE2/RE3 only. Uses RLM the way it's actually designed for: a large,
    unstructured context (raw log lines, chunked per service) passed as
    `prompt`, with the actual question passed separately as `root_prompt`.
    """
    logs_df = load_case_logs(local_path, case.case_id)
    inject_time = load_inject_time(local_path, case.case_id)

    windowed = logs_df[
        (logs_df["timestamp"] >= inject_time - window) & (logs_df["timestamp"] <= inject_time + window)
    ]

    service_names = []
    context_chunks = []
    timestamps_by_service = []
    for container, group in windowed.groupby("container_name"):
        lines = "\n".join(f"{row.timestamp} {row.message}" for row in group.itertuples())
        context_chunks.append(f"=== service: {container} ===\n{lines}")
        service_names.append(container)
        timestamps_by_service.append(group["timestamp"].tolist())

    def service_name_at(index: int) -> str:
        if not (0 <= index < len(service_names)):
            return f"Invalid index {index}. Valid range: 0-{len(service_names) - 1}"
        return service_names[index]

    def count_logs_in_range(index: int, start_time: float, end_time: float) -> int:
        if not (0 <= index < len(timestamps_by_service)):
            return -1
        return sum(1 for t in timestamps_by_service[index] if start_time <= t <= end_time)

    def log_change_ranking() -> list[dict]:
        results = []
        for i, name in enumerate(service_names):
            # Half-open on the injection instant so a log line timestamped
            # exactly at inject_time can't be counted in both windows.
            before = sum(1 for t in timestamps_by_service[i] if inject_time - window <= t < inject_time)
            after = sum(1 for t in timestamps_by_service[i] if inject_time <= t <= inject_time + window)
            pct_change = round((after - before) / before * 100, 1) if before > 0 else (100.0 * after if after > 0 else 0.0)
            results.append(
                {"index": i, "service": name, "before_count": before, "after_count": after, "percent_change": pct_change}
            )
        results.sort(key=lambda r: abs(r["percent_change"]), reverse=True)
        return results

    config = BACKEND_CONFIGS[backend]
    model_name = model_name or config["default_model"]
    api_key = os.getenv(config["api_key_env"]) if config["api_key_env"] else "ollama"

    backend_kwargs = {"model_name": model_name, "api_key": api_key}
    if "base_url" in config:
        backend_kwargs["base_url"] = config["base_url"]

    rlm = RLM(
        backend=config["rlm_backend"],
        backend_kwargs=backend_kwargs,
        environment="local",
        max_iterations=max_iterations,
        # Off by default. Without it, the full growing conversation history
        # gets resent every turn, uncompressed -- this is almost certainly
        # why later iterations got dramatically slower (204s by turn 6).
        compaction=True,
        custom_tools={
            "log_change_ranking": {
                "tool": log_change_ranking,
                "description": (
                    "log_change_ranking() -> list[dict]: before/after log counts and "
                    "percent_change for every service, sorted by |percent_change| "
                    "descending. Call this first; use percent_change, not raw count "
                    "differences, to judge anomalies -- raw differences favor "
                    "whichever service logs the most overall."
                ),
            },
            "service_name_at": {
                "tool": service_name_at,
                "description": (
                    "service_name_at(index: int) -> str: the exact service name for "
                    "a context[] index. Always use this for your final answer "
                    "instead of parsing it out of the chunk's "
                    "'=== service: X ===' header text yourself."
                ),
            },
            "count_logs_in_range": {
                "tool": count_logs_in_range,
                "description": (
                    "count_logs_in_range(index: int, start_time: float, end_time: float) -> int: "
                    "exact log-line count for context[index]'s service within "
                    "[start_time, end_time] (unix seconds). Use this instead of "
                    "writing your own regex/counting code."
                ),
            },
        },
        verbose=verbose,
    )

    root_prompt = LOGS_ROOT_PROMPT.format(
        system=case.system, inject_time=inject_time, window=window, num_services=len(context_chunks)
    )
    result = rlm.completion(context_chunks, root_prompt=root_prompt)
    return result.response.strip()


if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv

    load_dotenv()

    backend = sys.argv[1] if len(sys.argv) > 1 else "anthropic"

    local_path = download_suite()
    case = Case(
        case_id="re1ss_carts_mem_4",
        suite="re1",
        system="sock-shop",
        root_cause_service="carts",
        fault_type="mem",
        run=4,
    )

    prediction = run_case(local_path, case, backend=backend)
    print(f"\nPredicted root cause: {prediction}")
    print(f"Actual root cause:    {case.root_cause_service}")
    print(f"Correct: {case.root_cause_service in prediction}")
