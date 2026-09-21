"""
Custom tools exposed to the RLM root model's REPL (via RLM's
`custom_tools={...}` argument). Each metrics.parquet column is named
"{service}_{metric}", e.g. "carts-db_cpu" or "front-end_latency-90" --
these helpers split on that convention so the model can explore metrics
by service instead of getting all 50-60 raw columns dumped in at once.

The three RCAEval systems don't share one metric vocabulary: sock-shop and
train-ticket use "_workload"/"_latency-50"/"_latency-90", while
online-boutique uses "_load"/"_latency" instead. Both are matched here.
"""

import pandas as pd

# Longest/most specific suffixes first so e.g. "_latency-50" isn't mistaken
# for ending in "_50", and "_latency-50"/"_latency-90" aren't mistaken for
# ending in the shorter "_latency".
METRIC_SUFFIXES = [
    "_latency-50",
    "_latency-90",
    "_workload",
    "_latency",
    "_load",
    "_error",
    "_cpu",
    "_mem",
]


def split_metric_column(column: str) -> tuple[str, str]:
    for suffix in METRIC_SUFFIXES:
        if column.endswith(suffix):
            return column[: -len(suffix)], suffix[1:]
    raise ValueError(f"Unrecognized metric column: {column!r}")


def list_services(metrics_df: pd.DataFrame) -> list[str]:
    services = {split_metric_column(col)[0] for col in metrics_df.columns if col != "time"}
    return sorted(services)


def list_categories(metrics_df: pd.DataFrame) -> list[str]:
    """The metric categories actually present in this case (varies by system)."""
    categories = {split_metric_column(col)[1] for col in metrics_df.columns if col != "time"}
    return sorted(categories)


def summarize_metrics(metrics_df: pd.DataFrame, inject_time: int, window: int = 30) -> str:
    """
    Precompute a pre/post-injection mean-and-percent-change summary for every
    service, so a weaker model can answer in one shot from a finished table
    instead of needing multi-turn exploration to compute this itself.
    """
    times = metrics_df["time"].tolist()
    inject_idx = min(range(len(times)), key=lambda i: abs(times[i] - inject_time))

    lines = []
    for service in list_services(metrics_df):
        cols = [c for c in metrics_df.columns if c != "time" and split_metric_column(c)[0] == service]
        parts = []
        for col in cols:
            category = split_metric_column(col)[1]
            values = metrics_df[col].tolist()
            pre = values[max(0, inject_idx - window) : inject_idx]
            post = values[inject_idx : inject_idx + window]
            if not pre or not post:
                continue
            pre_mean = sum(pre) / len(pre)
            post_mean = sum(post) / len(post)
            if abs(pre_mean) < 1e-6:
                # Percent change is meaningless from a ~zero baseline (e.g.
                # errors going from 0 to nonzero would show as a nonsensical
                # billion-percent spike) -- just show the absolute values.
                parts.append(f"{category}: {pre_mean:.4g}->{post_mean:.4g}")
            else:
                change = (post_mean - pre_mean) / abs(pre_mean)
                parts.append(f"{category}: {pre_mean:.4g}->{post_mean:.4g} ({change:+.1%})")
        lines.append(f"{service}: " + ", ".join(parts) if parts else f"{service}: (no data)")
    return "\n".join(lines)


def make_query_service_tool(metrics_df: pd.DataFrame):
    """Build a query_service(service_name, category=None) tool bound to one case's metrics."""

    def query_service(service_name: str, category: str | None = None) -> dict:
        """
        Return {"time": [...], "<column>": [...], ...} for one service.
        category, if given, filters to one of: cpu, mem, workload, error,
        latency-50, latency-90.
        """
        matched = [
            col
            for col in metrics_df.columns
            if col != "time"
            and split_metric_column(col)[0] == service_name
            and (category is None or split_metric_column(col)[1] == category)
        ]
        if not matched:
            return {"error": f"No metrics for service={service_name!r} category={category!r}"}
        return metrics_df[["time", *matched]].to_dict(orient="list")

    return query_service
