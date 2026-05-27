from __future__ import annotations

import argparse
import re
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = {
    "job_light": ROOT / "queries" / "job_light",
    "job": ROOT / "queries" / "job",
    "job_complex": ROOT / "queries" / "job_complex",
}


def _normalize_sql(sql: str) -> str:
    sql = sql.strip().rstrip(";")
    sql = re.sub(r"\baka_title\s+AS\s+AT\b", "aka_title AS aka_t", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bat\.", "aka_t.", sql, flags=re.IGNORECASE)
    return sql


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small release smoke test.")
    parser.add_argument("--db", default=str(ROOT / "data" / "hollywood_200k.duckdb"))
    args = parser.parse_args()

    con = duckdb.connect(args.db, read_only=True)
    try:
        title_rows = con.execute("SELECT COUNT(*) FROM title").fetchone()[0]
        if title_rows < 200_000:
            raise RuntimeError(f"title table is unexpectedly small: {title_rows}")
        print(f"title rows: {title_rows:,}")

        for workload, query_dir in WORKLOADS.items():
            files = sorted(query_dir.glob("*.sql"))
            if not files:
                raise RuntimeError(f"missing queries for {workload}")
            sql = _normalize_sql(files[0].read_text(encoding="utf-8"))
            row = con.execute(sql).fetchone()
            print(f"{workload}: {files[0].name} -> {row}")
    finally:
        con.close()

    print("smoke test: OK")


if __name__ == "__main__":
    main()
