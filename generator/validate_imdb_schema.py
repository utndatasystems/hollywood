"""Schema contract and integrity checks for converted JOB/IMDB CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from imdb_job_contract import JOB_CORE_TABLES, JOB_TABLE_COLUMNS


def _must_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, low_memory=False)


def validate(schema_dir: Path):
    # 1) Header contract
    for table, cols in JOB_TABLE_COLUMNS.items():
        p = schema_dir / f"{table}.csv"
        if not p.exists():
            raise RuntimeError(f"Missing table file: {table}.csv")
        header = list(pd.read_csv(p, nrows=0).columns)
        if header != cols:
            raise RuntimeError(f"Header mismatch in {table}.csv: expected {cols}, got {header}")

    # 2) Basic non-empty expectations
    title = _must_read(schema_dir / "title.csv")
    name = _must_read(schema_dir / "name.csv")
    cast_info = _must_read(schema_dir / "cast_info.csv")
    if title.empty or name.empty or cast_info.empty:
        raise RuntimeError("Core tables must be non-empty: title/name/cast_info")

    # 3) FK domain checks
    title_ids = set(pd.to_numeric(title["id"], errors="coerce").dropna().astype(int).tolist())
    person_ids = set(pd.to_numeric(name["id"], errors="coerce").dropna().astype(int).tolist())
    char_ids = set(pd.to_numeric(_must_read(schema_dir / "char_name.csv")["id"], errors="coerce").dropna().astype(int).tolist())

    cast_movie = set(pd.to_numeric(cast_info["movie_id"], errors="coerce").dropna().astype(int).tolist())
    cast_person = set(pd.to_numeric(cast_info["person_id"], errors="coerce").dropna().astype(int).tolist())
    cast_char = set(pd.to_numeric(cast_info["person_role_id"], errors="coerce").dropna().astype(int).tolist())

    if not cast_movie.issubset(title_ids):
        raise RuntimeError("cast_info.movie_id contains values missing in title.id")
    if not cast_person.issubset(person_ids):
        raise RuntimeError("cast_info.person_id contains values missing in name.id")
    if cast_char and not cast_char.issubset(char_ids):
        raise RuntimeError("cast_info.person_role_id contains values missing in char_name.id")

    cc = _must_read(schema_dir / "complete_cast.csv")
    cct = _must_read(schema_dir / "comp_cast_type.csv")
    cct_ids = set(pd.to_numeric(cct["id"], errors="coerce").dropna().astype(int).tolist())
    cc_subject = set(pd.to_numeric(cc["subject_id"], errors="coerce").dropna().astype(int).tolist())
    cc_status = set(pd.to_numeric(cc["status_id"], errors="coerce").dropna().astype(int).tolist())
    if not cc_subject.issubset(cct_ids):
        raise RuntimeError("complete_cast.subject_id has values outside comp_cast_type.id")
    if not cc_status.issubset(cct_ids):
        raise RuntimeError("complete_cast.status_id has values outside comp_cast_type.id")

    print("Schema contract: OK")
    print(f"Core tables: {len(JOB_CORE_TABLES)}")
    print(f"title rows: {len(title):,} | name rows: {len(name):,} | cast_info rows: {len(cast_info):,}")
    print(f"complete_cast rows: {len(cc):,}")


def main():
    parser = argparse.ArgumentParser(description="Validate JOB/IMDB converted schema")
    parser.add_argument("--schema-dir", default=str((Path(__file__).resolve().parent / "imdb_schema").resolve()))
    args = parser.parse_args()

    validate(Path(args.schema_dir).resolve())


if __name__ == "__main__":
    main()
