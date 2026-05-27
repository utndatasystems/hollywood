from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import time
from pathlib import Path

import duckdb


if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB = BASE_DIR / "imdb_schema" / "imdb.duckdb"
DEFAULT_QUERY_DIR = BASE_DIR / "job_queries"
DEFAULT_OUT_CSV = BASE_DIR / "duckdb_job_timing_results.csv"


def _sort_key(path: Path) -> tuple[int, str]:
    match = re.match(r"(\d+)([a-z]?)", path.stem)
    return (int(match.group(1)), match.group(2)) if match else (999, path.stem)


def _analyze_all_tables(con: duckdb.DuckDBPyConnection) -> int:
    rows = con.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'main'
        ORDER BY table_name
        """
    ).fetchall()
    count = 0
    for (table_name,) in rows:
        con.execute(f'ANALYZE "{table_name}"')
        count += 1
    return count


def _trim_note(text: str, limit: int = 120) -> str:
    raw = " ".join(str(text).split())
    return raw[:limit]


def _normalize_sql_for_duckdb(sql: str) -> str:
    normalized = str(sql or "")
    # DuckDB is stricter about certain reserved-looking aliases such as `AT`.
    # Keep the benchmark query text intact on disk and rewrite only at execution time.
    normalized = re.sub(r"\baka_title\s+AS\s+AT\b", "aka_title AS aka_t", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bat\.", "aka_t.", normalized, flags=re.IGNORECASE)
    return normalized


def run_benchmark(
    db_path: Path,
    query_dir: Path,
    out_csv: Path,
    *,
    analyze: bool = True,
) -> dict[str, float | int]:
    db_path = db_path.resolve()
    query_dir = query_dir.resolve()
    out_csv = out_csv.resolve()

    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB database not found: {db_path}")
    if not query_dir.exists():
        raise FileNotFoundError(f"Query directory not found: {query_dir}")

    sql_files = sorted(query_dir.glob("*.sql"), key=_sort_key)
    if not sql_files:
        raise FileNotFoundError(f"No SQL files found in {query_dir}")

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        con.execute("PRAGMA threads=4")

        analyzed_tables = 0
        if analyze:
            con.close()
            con = duckdb.connect(str(db_path), read_only=False)
            con.execute("PRAGMA threads=4")
            analyzed_tables = _analyze_all_tables(con)

        print(f"DuckDB JOB execution benchmark")
        print(f"Database: {db_path}")
        print(f"Queries:   {query_dir} ({len(sql_files)} files)")
        if analyze:
            print(f"Analyze:   {analyzed_tables} tables analyzed")
        print()
        print(f"  {'Query':<8} {'Status':<8} {'Rows':>8} {'ms':>8}  Note")
        print(f"  {'-'*72}")

        results: list[tuple[str, str, int, float, str]] = []
        ok = 0
        err = 0
        empty = 0

        for sql_file in sql_files:
            query_id = sql_file.stem
            sql = _normalize_sql_for_duckdb(sql_file.read_text(encoding="utf-8").strip().rstrip(";"))
            t0 = time.time()
            try:
                rows = con.execute(sql).fetchall()
                elapsed_ms = (time.time() - t0) * 1000.0
                row_count = len(rows)
                if row_count == 0:
                    status = "EMPTY"
                    note = "(no rows returned)"
                    empty += 1
                else:
                    status = "OK"
                    first = rows[0]
                    note = _trim_note(repr(first))
                ok += 1
                print(f"  {query_id:<8} {status:<8} {row_count:>8,} {elapsed_ms:>7.0f}  {note}", flush=True)
                results.append((query_id, status, row_count, round(elapsed_ms, 1), note))
            except Exception as exc:
                elapsed_ms = (time.time() - t0) * 1000.0
                err += 1
                note = _trim_note(str(exc), limit=140)
                print(f"  {query_id:<8} {'ERROR':<8} {'-':>8} {elapsed_ms:>7.0f}  {note}", flush=True)
                results.append((query_id, "ERROR", 0, round(elapsed_ms, 1), note))

        times = [row[3] for row in results if row[1] != "ERROR"]
        print()
        print("=" * 72)
        print(f"SUMMARY: {len(sql_files)} queries")
        print("=" * 72)
        print(f"  OK (rows > 0):   {ok - empty}")
        print(f"  OK (empty):      {empty}")
        print(f"  ERRORS:          {err}")
        print(f"  Total OK:        {ok}/{len(sql_files)}")
        if times:
            ordered = sorted(times)
            print()
            print("  Timing (successful queries):")
            print(f"    Min:    {ordered[0]:>8.1f} ms")
            print(f"    Max:    {ordered[-1]:>8.1f} ms")
            print(f"    Median: {ordered[len(ordered)//2]:>8.1f} ms")
            print(f"    Total:  {sum(times):>8.1f} ms")

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["query", "status", "row_count", "ms", "note"])
            writer.writerows(results)
        print()
        print(f"CSV -> {out_csv}")

        return {
            "queries": len(sql_files),
            "ok": ok,
            "empty": empty,
            "error": err,
            "timing_total_ms": round(sum(times), 1) if times else 0.0,
            "timing_median_ms": ordered[len(ordered)//2] if times else 0.0,
        }
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run adapted JOB queries against a DuckDB database and record timing.")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="Path to the DuckDB database file.")
    parser.add_argument("--query-dir", default=str(DEFAULT_QUERY_DIR), help="Directory containing adapted JOB SQL files.")
    parser.add_argument("--out-csv", default=str(DEFAULT_OUT_CSV), help="Where to write the timing CSV.")
    parser.add_argument("--skip-analyze", action="store_true", help="Skip ANALYZE before running queries.")
    args = parser.parse_args()

    run_benchmark(
        Path(args.db),
        Path(args.query_dir),
        Path(args.out_csv),
        analyze=not bool(args.skip_analyze),
    )


if __name__ == "__main__":
    main()
