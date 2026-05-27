#!/usr/bin/env python3
"""Summarize PostgreSQL planner cost versus runtime for Hollywood lcm traces."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


WORKLOAD_FILES = (
    ("job_light", "hollywood_job_light.json"),
    ("job", "hollywood_job.json"),
    ("job_complex", "hollywood_job_complex.json"),
)

ROOT_COST_RE = re.compile(r"\(cost=([0-9.]+)\.\.([0-9.]+)\s+rows=([0-9]+)")
EXECUTION_RE = re.compile(r"Execution Time:\s*([0-9.]+)\s*ms")


def flatten_lines(node: Any) -> list[str]:
    if isinstance(node, str):
        return [node]
    if isinstance(node, list):
        out: list[str] = []
        for child in node:
            out.extend(flatten_lines(child))
        return out
    return []


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - rank) + ordered[hi] * (rank - lo)


def qerror(pred: float, label: float) -> float:
    if pred <= 0 or label <= 0:
        return math.inf
    return max(pred / label, label / pred)


def ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    out = [0.0] * len(values)
    idx = 0
    while idx < len(indexed):
        j = idx + 1
        while j < len(indexed) and indexed[j][1] == indexed[idx][1]:
            j += 1
        rank = (idx + j - 1) / 2.0 + 1.0
        for original_idx, _ in indexed[idx:j]:
            out[original_idx] = rank
        idx = j
    return out


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return math.nan
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = sum((x - mean_x) ** 2 for x in xs)
    den_y = sum((y - mean_y) ** 2 for y in ys)
    if den_x <= 0 or den_y <= 0:
        return math.nan
    return num / math.sqrt(den_x * den_y)


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(ranks(xs), ranks(ys))


def extract_row(workload: str, query: dict[str, Any]) -> dict[str, str | float] | None:
    if query.get("timeout") or query.get("error"):
        return None
    plan_sets = query.get("analyze_plans") or []
    if not plan_sets:
        return None
    lines = flatten_lines(plan_sets[0])
    if not lines:
        return None
    root_match = ROOT_COST_RE.search(lines[0])
    if not root_match:
        return None
    total_cost = float(root_match.group(2))
    estimated_rows = float(root_match.group(3))
    execution_ms = math.nan
    for line in lines:
        match = EXECUTION_RE.search(line)
        if match:
            execution_ms = float(match.group(1))
            break
    if total_cost <= 0 or not math.isfinite(execution_ms) or execution_ms <= 0:
        return None
    return {
        "workload": workload,
        "query_id": str(query.get("query_id", "")),
        "total_cost": total_cost,
        "estimated_rows": estimated_rows,
        "execution_ms": execution_ms,
        "execution_s": execution_ms / 1000.0,
        "raw_qerror_cost_vs_ms": qerror(total_cost, execution_ms),
        "raw_qerror_cost_vs_s": qerror(total_cost, execution_ms / 1000.0),
    }


def summarize_workload(workload: str, rows: list[dict[str, str | float]]) -> dict[str, str | float | int]:
    costs = [float(row["total_cost"]) for row in rows]
    runtime_ms = [float(row["execution_ms"]) for row in rows]
    log_costs = [math.log(value) for value in costs]
    log_runtime = [math.log(value) for value in runtime_ms]
    ratios = [runtime / cost for cost, runtime in zip(costs, runtime_ms)]
    scale = math.exp(statistics.fmean(math.log(ratio) for ratio in ratios))
    scaled_q = [qerror(cost * scale, runtime) for cost, runtime in zip(costs, runtime_ms)]
    raw_q_ms = [float(row["raw_qerror_cost_vs_ms"]) for row in rows]
    return {
        "workload": workload,
        "n": len(rows),
        "cost_runtime_pearson": pearson(costs, runtime_ms),
        "cost_runtime_spearman": spearman(costs, runtime_ms),
        "log_cost_runtime_pearson": pearson(log_costs, log_runtime),
        "median_cost_to_ms_scale": scale,
        "scaled_q_median": percentile(scaled_q, 50),
        "scaled_q_p90": percentile(scaled_q, 90),
        "scaled_q_p95": percentile(scaled_q, 95),
        "scaled_q_p99": percentile(scaled_q, 99),
        "scaled_q_max": max(scaled_q) if scaled_q else math.nan,
        "raw_cost_vs_ms_q_median": percentile(raw_q_ms, 50),
        "raw_cost_vs_ms_q_p95": percentile(raw_q_ms, 95),
        "raw_cost_vs_ms_q_max": max(raw_q_ms) if raw_q_ms else math.nan,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = list(rows[0]) if rows else []
    lines = [
        "# PostgreSQL Cost Summary",
        "",
        "Correlation columns compare PostgreSQL root `Total Cost` with actual execution runtime.",
        "`scaled_q_*` uses one log-mean multiplicative scale per workload to map planner cost units to milliseconds.",
        "",
    ]
    if headers:
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(format_value(row[h]) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    all_rows: list[dict[str, str | float]] = []
    summaries: list[dict[str, str | float | int]] = []
    for workload, filename in WORKLOAD_FILES:
        payload = json.loads((args.raw_dir / filename).read_text(encoding="utf-8"))
        rows = [
            row
            for query in payload.get("query_list", [])
            if (row := extract_row(workload, query)) is not None
        ]
        all_rows.extend(rows)
        summaries.append(summarize_workload(workload, rows))

    write_csv(args.out_dir / "postgres_cost_per_query.csv", all_rows)
    write_csv(args.out_dir / "postgres_cost_summary.csv", summaries)
    write_markdown(args.out_dir / "postgres_cost_summary.md", summaries)
    print(f"wrote {args.out_dir / 'postgres_cost_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
