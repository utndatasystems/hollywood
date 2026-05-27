from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "generator"))

from imdb_job_contract import JOB_CORE_TABLES, JOB_TABLE_COLUMNS  # noqa: E402


QUERY_COUNTS = {
    "job_light": 70,
    "job": 113,
    "job_complex": 30,
}


def _read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return next(csv.reader(handle))


def validate_csvs(csv_dir: Path) -> None:
    for table in JOB_CORE_TABLES:
        path = csv_dir / f"{table}.csv"
        if not path.exists():
            raise RuntimeError(f"missing strict CSV: {path}")
        header = _read_header(path)
        expected = JOB_TABLE_COLUMNS[table]
        if header != expected:
            raise RuntimeError(f"{table}.csv header mismatch: {header} != {expected}")
    print(f"strict CSV tables: {len(JOB_CORE_TABLES)}/21")


def validate_queries(query_root: Path) -> None:
    for workload, expected in QUERY_COUNTS.items():
        count = len(list((query_root / workload).glob("*.sql")))
        if count != expected:
            raise RuntimeError(f"{workload} query count mismatch: {count} != {expected}")
        print(f"{workload}: {count}/{expected} queries")


def validate_duckdb(db_path: Path) -> None:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        missing = set(JOB_CORE_TABLES) - tables
        if missing:
            raise RuntimeError(f"DuckDB missing tables: {sorted(missing)}")
        for table in JOB_CORE_TABLES:
            count = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            if table in {"title", "name", "cast_info"} and count <= 0:
                raise RuntimeError(f"DuckDB table {table} is empty")
        print("DuckDB strict tables: OK")
    finally:
        con.close()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_checksums(root: Path) -> None:
    checksum_path = root / "data" / "checksums.sha256"
    if not checksum_path.exists():
        raise RuntimeError(f"missing checksum file: {checksum_path}")
    rows = []
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, rel_path = line.split(maxsplit=1)
        rows.append((expected, rel_path))
    for expected, rel_path in rows:
        path = root / rel_path
        if not path.exists():
            raise RuntimeError(f"checksum target missing: {rel_path}")
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"checksum mismatch for {rel_path}: {observed} != {expected}")
    print(f"checksums: {len(rows)}/{len(rows)} files")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the public release folder layout and core artifacts.")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--skip-checksums", action="store_true", help="Skip SHA-256 validation for large data files.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    validate_csvs(root / "data" / "imdb_csv")
    validate_duckdb(root / "data" / "hollywood_200k.duckdb")
    validate_queries(root / "queries")
    if not args.skip_checksums:
        validate_checksums(root)
    print("release validation: OK")


if __name__ == "__main__":
    main()
