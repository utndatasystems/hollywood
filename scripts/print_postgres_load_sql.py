from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "generator"))

from imdb_job_contract import JOB_INDEX_DDL, JOB_TABLE_COLUMNS, JOB_TABLE_TYPES  # noqa: E402


TYPE_MAP = {
    "INT": "integer",
    "TEXT": "text",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Print PostgreSQL DDL and psql copy commands for the strict CSVs.")
    parser.add_argument("--csv-dir", default="data/imdb_csv", help="Path as seen by psql, relative or absolute.")
    args = parser.parse_args()

    print("SET max_parallel_workers_per_gather = 0;")
    print("SET client_encoding = 'UTF8';")
    for table, cols in JOB_TABLE_COLUMNS.items():
        print(f'DROP TABLE IF EXISTS "{table}" CASCADE;')
        col_defs = []
        for col in cols:
            typ = TYPE_MAP.get(JOB_TABLE_TYPES[table][col], "text")
            col_defs.append(f'  "{col}" {typ}')
        print(f'CREATE TABLE "{table}" (')
        print(",\n".join(col_defs))
        print(");")
        print(f"\\copy \"{table}\" FROM '{args.csv_dir}/{table}.csv' WITH (FORMAT csv, HEADER true, NULL '');")
    for ddl in JOB_INDEX_DDL:
        print(ddl + ";")
    print("ANALYZE;")


if __name__ == "__main__":
    main()
