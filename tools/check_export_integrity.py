from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import duckdb


CORE_TABLES = [
    "title",
    "name",
    "cast_info",
    "char_name",
    "company_name",
    "movie_companies",
    "movie_keyword",
    "keyword",
    "movie_link",
    "aka_title",
    "aka_name",
    "movie_info",
    "movie_info_idx",
    "person_info",
    "info_type",
    "kind_type",
    "role_type",
    "company_type",
    "link_type",
    "complete_cast",
    "comp_cast_type",
]

FK_CHECKS = [
    ("cast_info", "person_id", "name", "id", "person_id IS NOT NULL"),
    ("cast_info", "movie_id", "title", "id", "movie_id IS NOT NULL"),
    ("cast_info", "person_role_id", "char_name", "id", "person_role_id IS NOT NULL"),
    ("cast_info", "role_id", "role_type", "id", "role_id IS NOT NULL"),
    ("movie_companies", "movie_id", "title", "id", "movie_id IS NOT NULL"),
    ("movie_companies", "company_id", "company_name", "id", "company_id IS NOT NULL"),
    ("movie_companies", "company_type_id", "company_type", "id", "company_type_id IS NOT NULL"),
    ("movie_keyword", "movie_id", "title", "id", "movie_id IS NOT NULL"),
    ("movie_keyword", "keyword_id", "keyword", "id", "keyword_id IS NOT NULL"),
    ("movie_link", "movie_id", "title", "id", "movie_id IS NOT NULL"),
    ("movie_link", "linked_movie_id", "title", "id", "linked_movie_id IS NOT NULL"),
    ("aka_title", "movie_id", "title", "id", "movie_id IS NOT NULL"),
    ("aka_name", "person_id", "name", "id", "person_id IS NOT NULL"),
    ("movie_info", "movie_id", "title", "id", "movie_id IS NOT NULL"),
    ("movie_info", "info_type_id", "info_type", "id", "info_type_id IS NOT NULL"),
    ("movie_info_idx", "movie_id", "title", "id", "movie_id IS NOT NULL"),
    ("movie_info_idx", "info_type_id", "info_type", "id", "info_type_id IS NOT NULL"),
    ("person_info", "person_id", "name", "id", "person_id IS NOT NULL"),
    ("person_info", "info_type_id", "info_type", "id", "info_type_id IS NOT NULL"),
    ("complete_cast", "movie_id", "title", "id", "movie_id IS NOT NULL"),
    ("complete_cast", "subject_id", "comp_cast_type", "id", "subject_id IS NOT NULL"),
    ("complete_cast", "status_id", "comp_cast_type", "id", "status_id IS NOT NULL"),
]

REQUIRED_CHECKS = [
    ("title", "id"),
    ("title", "title"),
    ("title", "kind_id"),
    ("name", "id"),
    ("name", "name"),
    ("cast_info", "id"),
    ("company_name", "id"),
    ("company_name", "name"),
    ("movie_info", "id"),
    ("movie_info", "movie_id"),
    ("movie_info", "info_type_id"),
]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def scalar(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    return int(con.execute(sql).fetchone()[0])


def display_path(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    db_path = Path(args.db).resolve()
    out_dir = Path(args.out_dir).resolve()
    display_base = Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    table_rows: list[dict[str, Any]] = []
    fk_rows: list[dict[str, Any]] = []
    required_rows: list[dict[str, Any]] = []

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        for table in CORE_TABLES:
            rows = scalar(con, f'SELECT COUNT(*) FROM "{table}"')
            distinct_ids = scalar(con, f'SELECT COUNT(DISTINCT id) FROM "{table}"') if "id" else rows
            null_ids = scalar(con, f'SELECT COUNT(*) FROM "{table}" WHERE id IS NULL')
            table_rows.append(
                {
                    "table": table,
                    "rows": rows,
                    "distinct_ids": distinct_ids,
                    "duplicate_primary_keys": rows - distinct_ids,
                    "null_primary_keys": null_ids,
                }
            )

        for child, child_col, parent, parent_col, predicate in FK_CHECKS:
            missing = scalar(
                con,
                f'''
                SELECT COUNT(*)
                FROM "{child}" c
                LEFT JOIN "{parent}" p
                  ON c."{child_col}" = p."{parent_col}"
                WHERE {predicate} AND p."{parent_col}" IS NULL
                ''',
            )
            fk_rows.append(
                {
                    "child_table": child,
                    "child_column": child_col,
                    "parent_table": parent,
                    "parent_column": parent_col,
                    "missing_parent_rows": missing,
                }
            )

        for table, column in REQUIRED_CHECKS:
            failures = scalar(
                con,
                f'''
                SELECT COUNT(*)
                FROM "{table}"
                WHERE "{column}" IS NULL OR CAST("{column}" AS VARCHAR) = ''
                ''',
            )
            required_rows.append({"table": table, "column": column, "empty_or_null_rows": failures})

        genre_stats = {
            "genre_rows": scalar(con, 'SELECT COUNT(*) FROM movie_info WHERE info_type_id = 3'),
            "duplicate_genre_rows": scalar(
                con,
                """
                SELECT COUNT(*)
                FROM (
                  SELECT movie_id, info_type_id, info, COUNT(*) AS n
                  FROM movie_info
                  WHERE info_type_id = 3
                  GROUP BY movie_id, info_type_id, info
                  HAVING COUNT(*) > 1
                ) d
                """,
            ),
            "primary_movie_rows": scalar(con, "SELECT COUNT(*) FROM title WHERE kind_id = 1"),
            "primary_movies_with_genre": scalar(
                con,
                """
                SELECT COUNT(DISTINCT t.id)
                FROM title t
                JOIN movie_info mi ON mi.movie_id = t.id AND mi.info_type_id = 3
                WHERE t.kind_id = 1
                """,
            ),
            "primary_movies_with_secondary_genre": scalar(
                con,
                """
                SELECT COUNT(*)
                FROM (
                  SELECT t.id
                  FROM title t
                  JOIN movie_info mi ON mi.movie_id = t.id AND mi.info_type_id = 3
                  WHERE t.kind_id = 1
                  GROUP BY t.id
                  HAVING COUNT(*) > 1
                ) g
                """,
            ),
        }
        genre_stats["primary_movies_without_genre"] = (
            genre_stats["primary_movie_rows"] - genre_stats["primary_movies_with_genre"]
        )
    finally:
        con.close()

    write_csv(out_dir / "table_integrity.csv", table_rows, list(table_rows[0]))
    write_csv(out_dir / "fk_integrity.csv", fk_rows, list(fk_rows[0]))
    write_csv(out_dir / "required_field_integrity.csv", required_rows, list(required_rows[0]))
    summary = {
        "db": display_path(db_path, display_base),
        "duplicate_primary_key_violations": sum(int(row["duplicate_primary_keys"]) for row in table_rows),
        "null_primary_key_violations": sum(int(row["null_primary_keys"]) for row in table_rows),
        "checked_fk_violations": sum(int(row["missing_parent_rows"]) for row in fk_rows),
        "checked_required_field_failures": sum(int(row["empty_or_null_rows"]) for row in required_rows),
        "genre": genre_stats,
        "files": {
            "table_integrity": display_path(out_dir / "table_integrity.csv", display_base),
            "fk_integrity": display_path(out_dir / "fk_integrity.csv", display_base),
            "required_field_integrity": display_path(out_dir / "required_field_integrity.csv", display_base),
        },
    }
    (out_dir / "export_integrity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Check strict JOB export integrity.")
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
