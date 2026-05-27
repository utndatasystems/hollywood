from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from imdb_job_contract import JOB_CORE_TABLES


def build_duckdb(imdb_schema_dir: Path, out_path: Path, *, overwrite: bool = False, core_only: bool = False) -> Path:
    imdb_schema_dir = imdb_schema_dir.resolve()
    out_path = out_path.resolve()

    if not imdb_schema_dir.exists():
        raise FileNotFoundError(f"IMDb schema directory not found: {imdb_schema_dir}")

    csv_files = sorted(imdb_schema_dir.glob("*.csv"))
    if core_only:
        allowed = {f"{table}.csv" for table in JOB_CORE_TABLES}
        csv_files = [path for path in csv_files if path.name in allowed]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {imdb_schema_dir}")

    if out_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output file already exists: {out_path}")
        out_path.unlink()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(out_path))
    try:
        con.execute("PRAGMA threads=4")
        con.execute("BEGIN TRANSACTION")
        for csv_path in csv_files:
            table_name = csv_path.stem
            csv_literal = str(csv_path).replace("'", "''")
            con.execute(f'DROP TABLE IF EXISTS "{table_name}"')
            con.execute(
                f'''
                CREATE TABLE "{table_name}" AS
                SELECT *
                FROM read_csv_auto(
                    '{csv_literal}',
                    header=true,
                    sample_size=-1,
                    all_varchar=false,
                    ignore_errors=false
                )
                '''
            )

        con.execute(
            """
            CREATE OR REPLACE TABLE _import_manifest AS
            SELECT
                current_timestamp AS imported_at,
                ? AS source_dir,
                ? AS csv_count
            """,
            [str(imdb_schema_dir), len(csv_files)],
        )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.close()

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a DuckDB database file from an IMDb-schema CSV export directory.")
    parser.add_argument("--imdb-schema-dir", default=str(Path(__file__).resolve().parent / "imdb_schema"))
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "imdb_schema" / "imdb.duckdb"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--core-only", action="store_true", help="Import only the strict JOB/IMDb core tables.")
    args = parser.parse_args()

    out_path = build_duckdb(
        Path(args.imdb_schema_dir),
        Path(args.out),
        overwrite=bool(args.overwrite),
        core_only=bool(args.core_only),
    )
    print(out_path)


if __name__ == "__main__":
    main()
