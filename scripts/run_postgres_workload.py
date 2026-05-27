from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = {
    "job_light": ROOT / "queries" / "job_light",
    "job": ROOT / "queries" / "job",
    "job_complex": ROOT / "queries" / "job_complex",
}

WRAPPER_NODE_TYPES = {
    "Aggregate",
    "Finalize Aggregate",
    "Partial Aggregate",
    "GroupAggregate",
    "HashAggregate",
    "Result",
    "Limit",
    "Sort",
    "Incremental Sort",
    "Unique",
    "Gather",
    "Gather Merge",
    "Materialize",
}


def natural_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", path.stem)
    return [int(part) if part.isdigit() else part for part in parts]


def strip_to_count(sql: str) -> str:
    match = re.search(r"\bFROM\b", sql, flags=re.IGNORECASE)
    if not match:
        raise ValueError("Could not find FROM clause")
    return "SELECT COUNT(*) " + sql[match.start() :].strip().rstrip(";")


def q_error(actual: float, estimate: float) -> float:
    actual = max(float(actual), 1.0)
    estimate = max(float(estimate), 1.0)
    return max(actual / estimate, estimate / actual)


def signed_log10_error(actual: float, estimate: float) -> float:
    actual = max(float(actual), 1.0)
    estimate = max(float(estimate), 1.0)
    return math.log10(estimate / actual)


def bias(actual: float, estimate: float) -> str:
    if estimate > actual:
        return "over"
    if estimate < actual:
        return "under"
    return "exact"


def flatten_plan(plan: dict[str, Any], rows: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    if rows is None:
        rows = []
    rows.append(plan)
    for child in plan.get("Plans", []) or []:
        flatten_plan(child, rows)
    return rows


def choose_target_node(plan: dict[str, Any]) -> dict[str, Any]:
    nodes = flatten_plan(plan)
    for node in nodes:
        if str(node.get("Node Type") or "") not in WRAPPER_NODE_TYPES:
            return node
    return nodes[0]


def extract_json(stdout: str) -> Any:
    start = stdout.find("[")
    end = stdout.rfind("]")
    if start < 0 or end < start:
        raise ValueError(f"Could not find EXPLAIN JSON in psql output: {stdout[:400]}")
    return json.loads(stdout[start : end + 1])


def run_psql(args: argparse.Namespace, sql: str) -> str:
    cmd = [args.psql, "-X", "-v", "ON_ERROR_STOP=1", "-P", "pager=off", "-tAq"]
    if args.conn:
        cmd.append(args.conn)
    cmd.extend(["-c", sql])
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=args.timeout_sec + 30,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return proc.stdout or ""


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def run_query(args: argparse.Namespace, workload: str, sql_path: Path) -> dict[str, Any]:
    source_sql = sql_path.read_text(encoding="utf-8").strip().rstrip(";")
    count_sql = strip_to_count(source_sql)
    setup = [
        f"SET statement_timeout = '{int(args.timeout_sec * 1000)}ms'",
        "SET max_parallel_workers_per_gather = 0",
    ]
    explain_sql = "; ".join(setup) + f"; EXPLAIN (ANALYZE, VERBOSE, FORMAT JSON) {count_sql};"
    started = time.perf_counter()
    row: dict[str, Any] = {
        "dataset": "Hollywood200K",
        "query_mode": "balanced_duckdb_pg_200k",
        "workload": workload,
        "query_id": sql_path.stem,
        "engine": "PostgreSQL",
        "actual_cardinality": "",
        "estimated_cardinality": "",
        "q_error": "",
        "signed_log10_error": "",
        "zero_estimate_flag": "",
        "runtime_ms": "",
        "plan_total_cost": "",
        "status": "ok",
        "error": "",
    }
    try:
        stdout = run_psql(args, explain_sql)
        explain_doc = extract_json(stdout)[0]
        plan = explain_doc["Plan"]
        target = choose_target_node(plan)
        actual = float(target.get("Actual Rows") or 0.0) * float(target.get("Actual Loops") or 1.0)
        estimate = float(target.get("Plan Rows") or 0.0)
        effective_estimate = estimate if estimate > 0 else 1.0
        row.update(
            {
                "actual_cardinality": actual,
                "estimated_cardinality": estimate,
                "q_error": q_error(actual, effective_estimate),
                "signed_log10_error": signed_log10_error(actual, effective_estimate),
                "zero_estimate_flag": bool(actual > 0 and estimate == 0),
                "runtime_ms": float(explain_doc.get("Execution Time") or 0.0),
                "plan_total_cost": target.get("Total Cost", ""),
            }
        )
        plan_path = Path(args.out_dir) / "postgres" / workload / "raw_plans" / f"{sql_path.stem}.json"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(explain_doc, indent=2), encoding="utf-8")
    except Exception as exc:
        row["status"] = "error"
        row["error"] = str(exc)[:500]
        row["runtime_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect PostgreSQL full-query cardinality estimates and raw JSON plans.")
    parser.add_argument("--psql", default="psql", help="psql executable.")
    parser.add_argument("--conn", default="", help="psql connection string or database name. Empty uses PG* environment variables.")
    parser.add_argument("--workload", choices=["all", *WORKLOADS.keys()], default="all")
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "postgres_full_query"))
    parser.add_argument("--timeout-sec", type=int, default=1800)
    args = parser.parse_args()

    selected = WORKLOADS if args.workload == "all" else {args.workload: WORKLOADS[args.workload]}
    all_rows: list[dict[str, Any]] = []
    for workload, query_dir in selected.items():
        files = sorted(query_dir.glob("*.sql"), key=natural_key)
        rows = [run_query(args, workload, path) for path in files]
        all_rows.extend(rows)
        write_csv(Path(args.out_dir) / "postgres" / workload / "query_summary.csv", rows, list(rows[0]))
        write_csv(Path(args.out_dir) / f"{workload}_postgres_full_query.csv", rows, list(rows[0]))
    write_csv(Path(args.out_dir) / "postgres_full_query_cardinality_raw.csv", all_rows, list(all_rows[0]))
    print(Path(args.out_dir) / "postgres_full_query_cardinality_raw.csv")


if __name__ == "__main__":
    main()
