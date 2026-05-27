from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "generator"))

from run_job_queries_duckdb import run_benchmark  # noqa: E402


WORKLOADS = {
    "job_light": ROOT / "queries" / "job_light",
    "job": ROOT / "queries" / "job",
    "job_complex": ROOT / "queries" / "job_complex",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Hollywood SQL workloads against DuckDB.")
    parser.add_argument("--db", default=str(ROOT / "data" / "hollywood_200k.duckdb"))
    parser.add_argument("--workload", choices=["all", *WORKLOADS.keys()], default="all")
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "duckdb"))
    parser.add_argument("--analyze", action="store_true", help="Run ANALYZE before execution. This opens the DB read-write.")
    args = parser.parse_args()

    selected = WORKLOADS if args.workload == "all" else {args.workload: WORKLOADS[args.workload]}
    out_dir = Path(args.out_dir)
    for name, query_dir in selected.items():
        out_csv = out_dir / f"{name}_duckdb_results.csv"
        run_benchmark(Path(args.db), query_dir, out_csv, analyze=bool(args.analyze))


if __name__ == "__main__":
    main()
