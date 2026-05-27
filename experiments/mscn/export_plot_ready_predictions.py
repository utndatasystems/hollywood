#!/usr/bin/env python3
"""Export paper MSCN seed outputs into the legacy plot-script layout.

The paper-grade MSCN runner stores predictions under:

    model_seeds/seed_<N>/predictions/<workload>.csv

The existing q-error plotting scripts consume:

    predictions/<workload>.csv

This adapter keeps the original seed outputs untouched and writes a small
plot-ready directory that selects one seed per workload.  By default, the seed
is the one chosen by the runner's aggregate `best_seed_by_p95` field.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def csv_row_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def select_seed(report: dict[str, Any], workload: str, fallback_seed: str) -> str:
    aggregate = report.get("aggregate") or {}
    entry = aggregate.get(workload) or {}
    seed = entry.get("best_seed_by_p95")
    return str(seed if seed not in (None, "") else fallback_seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--workloads",
        default="job_exact,job_complex,job_light,job_exact_nodes,job_complex_nodes,job_light_nodes",
        help="Comma-separated prediction files to export if present.",
    )
    parser.add_argument("--fallback-seed", default="1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paper_run = args.paper_run.resolve()
    report_path = paper_run / "paper_mscn_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing paper MSCN report: {report_path}")
    report = read_json(report_path)
    out_dir = (args.out_dir or (paper_run / "plot_ready_best_p95")).resolve()
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "source_paper_run": str(paper_run),
        "source_report": str(report_path),
        "selection_policy": "best_seed_by_p95_per_workload",
        "workloads": {},
    }
    for workload in [part.strip() for part in args.workloads.split(",") if part.strip()]:
        seed = select_seed(report, workload, args.fallback_seed)
        src = paper_run / "model_seeds" / f"seed_{seed}" / "predictions" / f"{workload}.csv"
        if not src.exists():
            continue
        dst = pred_dir / f"{workload}.csv"
        shutil.copy2(src, dst)
        manifest["workloads"][workload] = {
            "seed": seed,
            "source": str(src),
            "output": str(dst),
            "rows": csv_row_count(dst),
            "aggregate": (report.get("aggregate") or {}).get(workload),
        }

    if not manifest["workloads"]:
        raise RuntimeError(f"No workload prediction CSVs were exported from {paper_run}")
    (out_dir / "plot_ready_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "workloads": manifest["workloads"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
