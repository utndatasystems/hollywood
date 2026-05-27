#!/usr/bin/env python3
"""Summarize lcm-eval pretrained ZeroShot prediction CSVs for Hollywood."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


WORKLOADS = (
    ("job_light", "hollywood_job_light"),
    ("job", "hollywood_job"),
    ("job_complex", "hollywood_job_complex"),
)


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


def summarize_rows(workload: str, seed: str, rows: list[dict[str, str]]) -> dict[str, str | int | float]:
    qerrors = [float(row["qerror"]) for row in rows]
    labels = [float(row["label"]) for row in rows]
    preds = [float(row["prediction"]) for row in rows]
    return {
        "workload": workload,
        "seed": seed,
        "n": len(rows),
        "label_nonpositive": sum(value <= 0 for value in labels),
        "prediction_nonpositive": sum(value <= 0 for value in preds),
        "q_mean": statistics.fmean(qerrors) if qerrors else math.nan,
        "q_median": percentile(qerrors, 50),
        "q_p90": percentile(qerrors, 90),
        "q_p95": percentile(qerrors, 95),
        "q_p99": percentile(qerrors, 99),
        "q_max": max(qerrors) if qerrors else math.nan,
    }


def read_prediction_file(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def format_float(value: str | int | float) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_outputs(summaries: list[dict[str, str | int | float]], out_csv: Path, out_md: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "workload",
        "seed",
        "n",
        "label_nonpositive",
        "prediction_nonpositive",
        "q_mean",
        "q_median",
        "q_p90",
        "q_p95",
        "q_p99",
        "q_max",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(summaries)

    lines = [
        "# Pretrained ZeroShot Hollywood Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in summaries:
        lines.append("| " + " | ".join(format_float(row[header]) for header in headers) + " |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--out-md", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summaries: list[dict[str, str | int | float]] = []
    for workload_name, file_prefix in WORKLOADS:
        pooled_rows: list[dict[str, str]] = []
        for seed in args.seeds:
            path = args.run_root / f"seed_{seed}" / f"{file_prefix}_{seed}_test_pred.csv"
            rows = read_prediction_file(path)
            pooled_rows.extend(rows)
            summaries.append(summarize_rows(workload_name, str(seed), rows))
        summaries.append(summarize_rows(workload_name, "pooled", pooled_rows))

    out_csv = args.out_csv or args.run_root / "summary_qerrors.csv"
    out_md = args.out_md or args.run_root / "summary_qerrors.md"
    write_outputs(summaries, out_csv, out_md)
    print(f"wrote {out_csv}")
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
