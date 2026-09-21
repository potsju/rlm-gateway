# rlm-gateway

Root-cause analysis over [RCAEval](https://huggingface.co/datasets/phamquiluan/RCAEval)
microservice incidents, using [RLM](https://github.com/alexzhang13/rlm) (Recursive
Language Models) as the orchestrator: instead of dumping a whole case's data into
one prompt, the root model explores the case programmatically in a Python REPL,
calling tools to pull in only the slices of data it actually needs.

## Layout

```
scripts/
    explore_rcaeval.py   # print a case's real file layout / schema
    eval.py               # batch-run any strategy over N cases, report accuracy
src/rlm_gateway/
    dataset.py            # download / case-id parsing / per-case loading
    tools.py               # custom_tools for the metrics-based strategies
    pipeline.py             # the three RCA strategies (see below)
    _rlm_compat.py           # workaround for an rlms==0.1.3 AnthropicClient bug
colab_local_eval.ipynb     # self-contained notebook: free T4 GPU via Ollama
colab_openrouter_eval.ipynb # self-contained notebook: free OpenRouter API
```

## Three strategies

RCAEval cases come in three suites. RE1 is metrics-only; RE2/RE3 also include raw
application logs.

- **`run_case`** -- RLM + `list_services`/`query_service` tools over RE1-RE3
  metrics.parquet. The model explores metrics around the fault-injection time and
  picks a root cause.
- **`run_case_direct`** -- bypasses RLM entirely. Precomputes a pre/post-injection
  summary and asks the model one direct question, no tools, no REPL loop. Simple,
  fast, and a useful baseline/fallback when a weaker model struggles with the
  tool-calling loop.
- **`run_case_logs`** -- RE2/RE3 only. RLM over raw per-service log text, with
  `log_change_ranking`/`service_name_at`/`count_logs_in_range` tools so the model
  doesn't have to hand-write its own parsing or counting logic. This is the
  closest fit to what RLM is actually designed for (a large, unstructured context
  explored programmatically) rather than a small structured one.

## Backends

Any strategy takes `backend=`: `anthropic`, `openai`, `openrouter` (50 free
requests/day, no card required), or `local` (free, via [Ollama](https://ollama.com)
running an open-weight model like `qwen2.5:14b` -- see the Colab notebooks for a
zero-cost GPU setup).

## Setup

```bash
cp .env.example .env   # fill in an API key for whichever backend you use
uv sync
```

## Usage

```bash
# Explore a case's real file layout / schema first
uv run scripts/explore_rcaeval.py

# Batch-run and report accuracy (writes logs/eval_results.csv incrementally)
uv run scripts/eval.py --strategy metrics --backend anthropic --limit 5
uv run scripts/eval.py --strategy direct --backend local --limit 10
uv run scripts/eval.py --strategy logs --backend local --pattern 're2*' --limit 10
```

No local GPU? `colab_local_eval.ipynb` runs the same pipeline for free on Colab's
T4 GPU via Ollama -- no API key needed. `colab_openrouter_eval.ipynb` is the
same idea via OpenRouter's free tier instead, if you don't want to deal with GPU
setup at all.

## Status

All three strategies run end-to-end and have each produced correct predictions on
real cases. `run_case_direct` has been batch-tested informally with encouraging
results; `run_case_logs` (real RLM + tool-calling on log data, run on a free local
qwen2.5-14b via Ollama) is implemented and correct on the cases tested, but has
**not** been run over a large enough batch yet to report a statistically
meaningful accuracy number -- that's the natural next step for anyone picking
this up.

Along the way, several real upstream/rlms library issues were found and worked
around by reading the actual library source rather than guessing:
- `AnthropicClient` crashes on extended-thinking responses (`_rlm_compat.py`)
- Only ` ```repl ` fenced blocks are executed; ` ```python ` blocks are silently
  ignored with no error feedback to the model
- `compaction` defaults to `False`, so the full uncompressed conversation history
  gets resent every turn -- causing severe iteration slowdown on longer runs
- The REPL is not a Jupyter cell: a bare expression on the last line produces no
  visible output, only `print(...)` does
