#!/usr/bin/env python3
"""Collect lcm-eval compatible raw PostgreSQL traces for Hollywood workloads.

The lcm-eval ZeroShot cost-model path parses PostgreSQL's raw text EXPLAIN
output, not FORMAT JSON.  This script produces the same high-level JSON shape
as lcm-eval's own run_workload.py:

    {
      "query_list": [
        {
          "analyze_plans": [[[line], ...], ...],
          "verbose_plan": [[line], ...],
          "timeout": false,
          "hint_notices": null,
          "sql": "...",
          "hint": ""
        }
      ],
      "database_stats": {"column_stats": [...], "table_stats": [...]},
      "run_kwargs": {...},
      "total_time_secs": 12.34
    }

It talks to the existing PostgreSQL Docker container through docker exec + psql,
so it does not read local .env files and does not require psycopg2.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FINAL_QUERY_ROOT = ROOT / "queries"
DEFAULT_RUN_DIR = ROOT / "results" / "zeroshot_lcm_raw"
DEFAULT_DOCKER_BIN_CANDIDATES = (
    Path(r"C:\Program Files\Docker\Docker\resources\bin\docker.exe"),
    Path(r"/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe"),
)


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    query_dir: Path
    output_path: Path


def detect_docker_bin(explicit: str | None) -> str:
    if explicit:
        return explicit
    found = shutil.which("docker") or shutil.which("docker.exe")
    if found:
        return found
    for candidate in DEFAULT_DOCKER_BIN_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return "docker"


def natural_key(path: Path) -> list[Any]:
    parts = re.split(r"(\d+)", path.stem.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def read_sql(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig").strip()
    return text.rstrip(";").strip()


def atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True)
    os.replace(tmp, path)


def load_existing(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def rows_from_psql_stdout(stdout: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in stdout.splitlines():
        line = line.rstrip("\r")
        if not line:
            continue
        if line == "SET":
            continue
        rows.append([line])
    return rows


def parse_json_stdout(stdout: str) -> Any:
    stripped = stdout.strip()
    if not stripped:
        return []
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = min([idx for idx in (stripped.find("["), stripped.find("{")) if idx >= 0], default=-1)
        end = max(stripped.rfind("]"), stripped.rfind("}"))
        if start < 0 or end < start:
            raise
        return json.loads(stripped[start : end + 1])


def run_psql(
    *,
    docker_bin: str,
    container: str,
    db_name: str,
    user: str,
    sql: str,
    timeout_sec: int,
) -> tuple[int, str, str, float]:
    cmd = [
        docker_bin,
        "exec",
        container,
        "psql",
        "-X",
        "-U",
        user,
        "-d",
        db_name,
        "-v",
        "ON_ERROR_STOP=1",
        "-P",
        "pager=off",
        "-tAq",
        "-c",
        sql,
    ]
    start = time.perf_counter()
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=timeout_sec + 20,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return completed.returncode, completed.stdout or "", completed.stderr or "", elapsed_ms


def collect_database_stats(
    *,
    docker_bin: str,
    container: str,
    db_name: str,
    user: str,
    timeout_sec: int,
) -> dict[str, Any]:
    column_stats_sql = """
        SELECT COALESCE(json_agg(row_to_json(s)), '[]'::json)
        FROM (
            SELECT
                s.tablename,
                s.attname,
                s.null_frac,
                s.avg_width,
                s.n_distinct,
                s.correlation,
                c.data_type
            FROM pg_stats s
            JOIN information_schema.columns c
                ON s.tablename = c.table_name
               AND s.attname = c.column_name
            WHERE s.schemaname = 'public'
            ORDER BY s.tablename, s.attname
        ) s;
    """
    table_stats_sql = """
        SELECT COALESCE(json_agg(row_to_json(s)), '[]'::json)
        FROM (
            SELECT relname, reltuples, relpages
            FROM pg_class
            WHERE relkind = 'r'
            ORDER BY relname
        ) s;
    """
    column_code, column_out, column_err, _ = run_psql(
        docker_bin=docker_bin,
        container=container,
        db_name=db_name,
        user=user,
        sql=column_stats_sql,
        timeout_sec=timeout_sec,
    )
    if column_code != 0:
        raise RuntimeError(f"Could not collect column stats: {column_err.strip()}")
    table_code, table_out, table_err, _ = run_psql(
        docker_bin=docker_bin,
        container=container,
        db_name=db_name,
        user=user,
        sql=table_stats_sql,
        timeout_sec=timeout_sec,
    )
    if table_code != 0:
        raise RuntimeError(f"Could not collect table stats: {table_err.strip()}")
    column_stats = parse_json_stdout(column_out)
    table_stats = parse_json_stdout(table_out)
    if not column_stats or not table_stats:
        raise RuntimeError(f"No PostgreSQL stats found for database {db_name}; run ANALYZE first.")
    return {"column_stats": column_stats, "table_stats": table_stats}


def setup_statements(timeout_sec: int, disable_parallel: bool, disable_memoize: bool) -> list[str]:
    statements = [f"SET statement_timeout = '{int(timeout_sec * 1000)}ms'"]
    if disable_parallel:
        statements.append("SET max_parallel_workers_per_gather = 0")
    if disable_memoize:
        statements.append("SET enable_memoize = off")
    return statements


def explain_sql(
    *,
    docker_bin: str,
    container: str,
    db_name: str,
    user: str,
    sql: str,
    timeout_sec: int,
    analyze: bool,
    disable_parallel: bool,
    disable_memoize: bool,
) -> tuple[bool, list[list[str]] | None, str, float]:
    mode = "ANALYZE TRUE" if analyze else "VERBOSE TRUE, ANALYZE FALSE"
    full_sql = "; ".join(setup_statements(timeout_sec, disable_parallel, disable_memoize))
    full_sql = f"{full_sql}; EXPLAIN ({mode}) {sql};"
    try:
        code, stdout, stderr, elapsed_ms = run_psql(
            docker_bin=docker_bin,
            container=container,
            db_name=db_name,
            user=user,
            sql=full_sql,
            timeout_sec=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        return True, None, f"subprocess timeout after {exc.timeout}s", float(timeout_sec * 1000)
    if code != 0:
        lower_err = stderr.lower()
        is_timeout = "statement timeout" in lower_err or "canceling statement due to statement timeout" in lower_err
        return is_timeout, None, stderr.strip() or stdout.strip(), elapsed_ms
    return False, rows_from_psql_stdout(stdout), "", elapsed_ms


def iter_sql_files(query_dir: Path, wanted_ids: set[str] | None, limit: int | None) -> Iterable[Path]:
    files = sorted(query_dir.glob("*.sql"), key=natural_key)
    emitted = 0
    for path in files:
        if wanted_ids is not None and path.stem not in wanted_ids:
            continue
        yield path
        emitted += 1
        if limit is not None and emitted >= limit:
            break


def collect_workload(
    spec: WorkloadSpec,
    *,
    docker_bin: str,
    container: str,
    db_name: str,
    user: str,
    repetitions: int,
    timeout_sec: int,
    disable_parallel: bool,
    disable_memoize: bool,
    wanted_ids: set[str] | None,
    limit: int | None,
    resume: bool,
    dedupe_sql: bool,
    fail_fast: bool,
    save_every: int,
) -> None:
    start = time.perf_counter()
    existing = load_existing(spec.output_path) if resume else None
    database_stats = (
        existing.get("database_stats")
        if existing and existing.get("database_stats")
        else collect_database_stats(
            docker_bin=docker_bin,
            container=container,
            db_name=db_name,
            user=user,
            timeout_sec=timeout_sec,
        )
    )
    query_list = list(existing.get("query_list", [])) if existing else []
    seen_ids = {str(q.get("query_id")) for q in query_list if q.get("query_id")}
    seen_sql = {str(q.get("sql")) for q in query_list if q.get("sql")}
    run_kwargs = {
        "collector": "collect_lcm_raw_traces.py",
        "workload": spec.name,
        "query_dir": str(spec.query_dir),
        "db_name": db_name,
        "container": container,
        "repetitions_per_query": repetitions,
        "timeout_sec": timeout_sec,
        "disable_parallel": disable_parallel,
        "disable_memoize": disable_memoize,
        "dedupe_sql": dedupe_sql,
        "format": "lcm_eval_raw_postgres",
    }
    if existing and existing.get("run_kwargs"):
        run_kwargs["previous_run_kwargs"] = existing["run_kwargs"]

    print(f"[{spec.name}] collecting into {spec.output_path}")
    total_seen_at_start = len(query_list)
    for idx, path in enumerate(iter_sql_files(spec.query_dir, wanted_ids, limit), start=1):
        query_id = path.stem
        sql = read_sql(path)
        if query_id in seen_ids:
            continue
        if dedupe_sql and sql in seen_sql:
            continue

        verbose_timeout, verbose_rows, verbose_error, verbose_elapsed = explain_sql(
            docker_bin=docker_bin,
            container=container,
            db_name=db_name,
            user=user,
            sql=sql,
            timeout_sec=timeout_sec,
            analyze=False,
            disable_parallel=disable_parallel,
            disable_memoize=disable_memoize,
        )
        analyze_plans: list[list[list[str]]] | None = []
        analyze_elapsed = 0.0
        analyze_error = ""
        analyze_timeout = False
        if not verbose_timeout and verbose_rows is not None:
            for _ in range(repetitions):
                curr_timeout, curr_rows, curr_error, curr_elapsed = explain_sql(
                    docker_bin=docker_bin,
                    container=container,
                    db_name=db_name,
                    user=user,
                    sql=sql,
                    timeout_sec=timeout_sec,
                    analyze=True,
                    disable_parallel=disable_parallel,
                    disable_memoize=disable_memoize,
                )
                analyze_elapsed += curr_elapsed
                if curr_timeout or curr_rows is None:
                    analyze_timeout = curr_timeout
                    analyze_error = curr_error
                    analyze_plans = None
                    break
                analyze_plans.append(curr_rows)
        timeout = bool(verbose_timeout or analyze_timeout)
        error = verbose_error or analyze_error
        if error and fail_fast:
            raise RuntimeError(f"[{spec.name}/{query_id}] {error}")

        query_list.append(
            {
                "query_id": query_id,
                "sql_path": str(path),
                "analyze_plans": analyze_plans,
                "verbose_plan": verbose_rows,
                "timeout": timeout,
                "hint_notices": None,
                "sql": sql,
                "hint": "",
                "collector_elapsed_ms": verbose_elapsed + analyze_elapsed,
                "error": error,
            }
        )
        seen_ids.add(query_id)
        seen_sql.add(sql)

        done_now = len(query_list) - total_seen_at_start
        print(f"[{spec.name}] {query_id}: timeout={timeout} rows={len(query_list)}")
        if save_every <= 1 or done_now % save_every == 0:
            atomic_json_dump(
                {
                    "query_list": query_list,
                    "database_stats": database_stats,
                    "run_kwargs": run_kwargs,
                    "total_time_secs": time.perf_counter() - start,
                },
                spec.output_path,
            )

    atomic_json_dump(
        {
            "query_list": query_list,
            "database_stats": database_stats,
            "run_kwargs": run_kwargs,
            "total_time_secs": time.perf_counter() - start,
        },
        spec.output_path,
    )
    print(f"[{spec.name}] saved {len(query_list)} queries to {spec.output_path}")


def parse_workload_arg(value: str, out_dir: Path) -> WorkloadSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use NAME=QUERY_DIR for --workload")
    name, raw_dir = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Workload name cannot be empty")
    query_dir = Path(raw_dir).expanduser().resolve()
    return WorkloadSpec(name=name, query_dir=query_dir, output_path=out_dir / f"{name}_lcm_raw.json")


def build_specs(args: argparse.Namespace) -> list[WorkloadSpec]:
    if args.workload:
        out_dir = args.out_dir.resolve()
        return [parse_workload_arg(value, out_dir) for value in args.workload]
    if args.all_final:
        out_dir = args.out_dir.resolve()
        return [
            WorkloadSpec(name=name, query_dir=(DEFAULT_FINAL_QUERY_ROOT / name), output_path=out_dir / f"{name}_lcm_raw.json")
            for name in ("job_light", "job", "job_complex")
        ]
    if args.query_dir is None or args.out is None:
        raise SystemExit("Provide either --query-dir + --out, --workload NAME=DIR, or --all-final.")
    return [
        WorkloadSpec(
            name=args.workload_name or Path(args.query_dir).resolve().name,
            query_dir=Path(args.query_dir).expanduser().resolve(),
            output_path=Path(args.out).expanduser().resolve(),
        )
    ]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-dir", type=Path, help="Directory containing .sql files for one workload.")
    parser.add_argument("--out", type=Path, help="Output JSON path for --query-dir.")
    parser.add_argument("--workload-name", help="Optional name for the single --query-dir workload.")
    parser.add_argument(
        "--workload",
        action="append",
        help="Collect one workload as NAME=QUERY_DIR. Repeat for multiple workloads.",
    )
    parser.add_argument(
        "--all-final",
        action="store_true",
        help="Collect final JOB-Light, JOB, and JOB-Complex SQL dirs from this release.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RUN_DIR, help="Output directory for --workload/--all-final.")
    parser.add_argument("--db-name", default="hollywood_200k")
    parser.add_argument("--pg-container", default="pg_bench")
    parser.add_argument("--pg-user", default="postgres")
    parser.add_argument("--docker-bin", help="Path to docker or docker.exe.")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument("--disable-parallel", action="store_true", default=True)
    parser.add_argument("--allow-parallel", action="store_false", dest="disable_parallel")
    parser.add_argument("--disable-memoize", action="store_true")
    parser.add_argument("--limit", type=int, help="Limit the number of SQL files per workload.")
    parser.add_argument("--query-id", action="append", help="Only collect a specific query id/stem. Repeatable.")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument(
        "--dedupe-sql",
        action="store_true",
        help="Skip exact duplicate SQL strings. Off by default so every benchmark file gets a trace.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--save-every", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    docker_bin = detect_docker_bin(args.docker_bin)
    specs = build_specs(args)
    wanted_ids = set(args.query_id) if args.query_id else None

    for spec in specs:
        if not spec.query_dir.exists():
            raise SystemExit(f"Query directory does not exist: {spec.query_dir}")
        collect_workload(
            spec,
            docker_bin=docker_bin,
            container=args.pg_container,
            db_name=args.db_name,
            user=args.pg_user,
            repetitions=args.repetitions,
            timeout_sec=args.timeout_sec,
            disable_parallel=args.disable_parallel,
            disable_memoize=args.disable_memoize,
            wanted_ids=wanted_ids,
            limit=args.limit,
            resume=args.resume,
            dedupe_sql=args.dedupe_sql,
            fail_fast=args.fail_fast,
            save_every=max(1, args.save_every),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
