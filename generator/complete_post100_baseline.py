from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from model_defaults import model_for_role


BASE_DIR = Path(__file__).resolve().parent


def _default_query_dir() -> Path:
    candidates = (
        BASE_DIR / "job_adapted",
        BASE_DIR / "job_queries",
        BASE_DIR.parent / "imdb_job_dataset" / "job_queries",
    )
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


DEFAULT_QUERY_DIR = _default_query_dir()


def _run_step(cmd: list[str], *, cwd: Path) -> None:
    print()
    print("=" * 88)
    print("RUN:", " ".join(cmd))
    print("=" * 88)
    subprocess.run(cmd, cwd=str(cwd), check=True, env=os.environ.copy())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Complete the post-step-100 baseline path: TV summaries, strict export, JOB validation, and signoff."
    )
    parser.add_argument("--base-dir", default=str(BASE_DIR), help="Run directory to finish and score.")
    parser.add_argument(
        "--model",
        default=model_for_role("plot_summaries"),
        help="Model for TV summary generation steps.",
    )
    parser.add_argument(
        "--skip-tv",
        action="store_true",
        help="Skip step 120 if TV enrichment has already been completed.",
    )
    parser.add_argument(
        "--skip-job",
        action="store_true",
        help="Skip DuckDB build and JOB query execution.",
    )
    parser.add_argument(
        "--include-extras",
        action="store_true",
        help="Include non-core research extras in the IMDb export. Off by default for benchmark validation.",
    )
    parser.add_argument(
        "--query-dir",
        default=str(DEFAULT_QUERY_DIR),
        help="Directory containing adapted JOB SQL files.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    imdb_schema_dir = base_dir / "imdb_schema"
    duckdb_path = imdb_schema_dir / f"{base_dir.name}_imdb.duckdb"
    job_csv_path = base_dir / "duckdb_job_timing_results.csv"

    if not args.skip_tv:
        _run_step(
            [
                sys.executable,
                "-u",
                str(base_dir / "generate_tv_summaries.py"),
                "--base-dir",
                str(base_dir),
                "--tier1-model",
                args.model,
                "--tier2-model",
                args.model,
            ],
            cwd=base_dir.parent,
        )

    _run_step(
        [
            sys.executable,
            "-u",
            str(base_dir / "export_imdb_schema.py"),
            "--base-dir",
            str(base_dir),
            "--out-dir",
            str(imdb_schema_dir),
            "--strict-job",
        ] + (["--include-extras"] if args.include_extras else []),
        cwd=base_dir.parent,
    )

    _run_step(
        [
            sys.executable,
            "-u",
            str(base_dir / "validate_imdb_schema.py"),
            "--schema-dir",
            str(imdb_schema_dir),
        ],
        cwd=base_dir.parent,
    )

    if not args.skip_job:
        _run_step(
            [
                sys.executable,
                "-u",
                str(base_dir / "build_duckdb_from_imdb_schema.py"),
                "--imdb-schema-dir",
                str(imdb_schema_dir),
                "--out",
                str(duckdb_path),
                "--overwrite",
                "--core-only",
            ],
            cwd=base_dir.parent,
        )
        _run_step(
            [
                sys.executable,
                "-u",
                str(base_dir / "run_job_queries_duckdb.py"),
                "--db",
                str(duckdb_path),
                "--query-dir",
                str(Path(args.query_dir).resolve()),
                "--out-csv",
                str(job_csv_path),
            ],
            cwd=base_dir.parent,
        )

    _run_step(
        [
            sys.executable,
            "-u",
            str(base_dir / "generate_signoff_report.py"),
            "--base-dir",
            str(base_dir),
        ],
        cwd=base_dir.parent,
    )


if __name__ == "__main__":
    main()
