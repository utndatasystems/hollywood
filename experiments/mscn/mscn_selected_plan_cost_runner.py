#!/usr/bin/env python3
"""Train a selected-plan MSCN-style cost model for Hollywood JOB workloads.

This is intentionally separate from the paper cardinality runner.  It uses the
same SQL/predicate/table bitmap encoder as the Hollywood MSCN cardinality bridge,
but changes the label to PostgreSQL selected-plan execution time.

The target is the cost-model table, not exhaustive bad-plan enumeration:

* train on synthetic SQL queries labeled by EXPLAIN ANALYZE runtime;
* evaluate on the final adapted JOB, JOB-light, and JOB-complex SQL sets;
* report Q-error on runtime labels in microseconds.

Runtime labels are stored in integer microseconds because the reusable MSCN
encoder expects positive integer labels.  Q-error is scale-invariant, so this
does not affect the reported errors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent / "vendor"
MSCN_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(MSCN_DIR) not in sys.path:
    sys.path.insert(0, str(MSCN_DIR))

import run_rich_mscn_experiment as rich  # noqa: E402
import paper_mscn_runner as paper_mscn  # noqa: E402


RUNS_DIR = ROOT / "experiments" / "mscn" / "cost_model_runs"
WORKLOADS = ("job_light", "job", "job_complex")


def now_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def qerror(pred: float, actual: float) -> float:
    pred = max(float(pred), 1.0)
    actual = max(float(actual), 1.0)
    return pred / actual if pred >= actual else actual / pred


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    return float(np.percentile(np.array(values, dtype=np.float64), pct))


def summarize_qerrors(values: list[float]) -> dict[str, Any]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return {"count": 0, "finite": 0}
    return {
        "count": len(values),
        "finite": len(finite),
        "mean": float(np.mean(finite)),
        "median": percentile(finite, 50),
        "p90": percentile(finite, 90),
        "p95": percentile(finite, 95),
        "p99": percentile(finite, 99),
        "max": max(finite),
    }


def parse_model_seeds(raw: str) -> list[int]:
    out = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not out:
        raise ValueError("--model-seeds must contain at least one integer")
    return out


def parse_skip_eval_queries(values: list[str]) -> dict[str, set[str]]:
    skipped: dict[str, set[str]] = {}
    for raw in values:
        if ":" not in raw:
            raise ValueError("--skip-eval-query must use workload:query_id, e.g. job_complex:11")
        workload, query_id = (part.strip() for part in raw.split(":", 1))
        if not workload or not query_id:
            raise ValueError("--skip-eval-query must use workload:query_id, e.g. job_complex:11")
        skipped.setdefault(workload, set()).add(query_id)
    return skipped


def resolve_torch_device(raw: str) -> str:
    if raw == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = raw
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(json.dumps({"warning": "CUDA requested but unavailable; using CPU"}), flush=True)
        return "cpu"
    if device.startswith("cuda"):
        print(json.dumps({"device": device, "cuda_device": torch.cuda.get_device_name(0)}), flush=True)
    else:
        print(json.dumps({"device": device}), flush=True)
    return device


def strip_sql(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig").strip().rstrip(";")


def extract_json_payload(raw: str) -> Any:
    """Parse psql FORMAT JSON output, tolerating notices around the JSON."""

    text = raw.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end < start:
        raise ValueError(f"No JSON plan array found in EXPLAIN output: {text[:300]!r}")
    return json.loads(text[start : end + 1])


def plan_runtime_us(
    *,
    session: rich.PsqlSession,
    sql: str,
    timeout_ms: int,
) -> tuple[int, dict[str, Any] | None, str | None, float]:
    """Return execution runtime in integer microseconds plus the plan payload."""

    start = time.perf_counter()
    wrapped = (
        f"SET statement_timeout TO {int(timeout_ms)};\n"
        # Docker's default /dev/shm can be too small for PostgreSQL parallel
        # query execution. Disabling gather workers keeps the selected-plan
        # runtime labels stable without changing the benchmark SQL text.
        "SET max_parallel_workers_per_gather TO 0;\n"
        "EXPLAIN (ANALYZE TRUE, FORMAT JSON, TIMING FALSE, SUMMARY TRUE) "
        f"{sql.strip().rstrip(';')};\n"
        "SET max_parallel_workers_per_gather TO DEFAULT;\n"
        "SET statement_timeout TO 0;"
    )
    try:
        raw = session.run(wrapped, timeout=max(30.0, timeout_ms / 1000.0 + 30.0))
        payload = extract_json_payload(raw)
        doc = payload[0] if isinstance(payload, list) and payload else payload
        runtime_ms = float(doc.get("Execution Time") or 0.0)
        if runtime_ms <= 0:
            raise ValueError("EXPLAIN payload did not contain positive Execution Time")
        runtime_us = max(1, int(round(runtime_ms * 1000.0)))
        return runtime_us, doc, None, (time.perf_counter() - start) * 1000.0
    except Exception as exc:
        try:
            session.run("SET max_parallel_workers_per_gather TO DEFAULT; SET statement_timeout TO 0")
        except Exception:
            pass
        return 0, None, str(exc).splitlines()[0][:500], (time.perf_counter() - start) * 1000.0


def load_eval_rows_from_query_dirs(
    *,
    session: rich.PsqlSession,
    query_dirs: dict[str, Path],
    out_dir: Path,
    timeout_ms: int,
    skip_eval_queries: dict[str, set[str]] | None = None,
) -> dict[str, list[tuple[str, str, int, str]]]:
    eval_root = out_dir / "eval_runtime_labels"
    eval_root.mkdir(parents=True, exist_ok=True)
    skip_eval_queries = skip_eval_queries or {}
    skipped_rows: list[dict[str, str]] = []
    out: dict[str, list[tuple[str, str, int, str]]] = {}
    for workload, query_dir in query_dirs.items():
        rows: list[tuple[str, str, int, str]] = []
        label_path = eval_root / f"{workload}.csv"
        plan_path = eval_root / f"{workload}_plans.jsonl"
        done: dict[str, dict[str, str]] = {}
        if label_path.exists():
            with label_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if row.get("runtime_us"):
                        done[str(row["query_id"])] = row
        with label_path.open("a", newline="", encoding="utf-8") as label_handle, plan_path.open("a", encoding="utf-8") as plan_handle:
            fieldnames = ["query_id", "runtime_us", "elapsed_ms", "error", "sql_path"]
            writer = csv.DictWriter(label_handle, fieldnames=fieldnames)
            if label_path.stat().st_size == 0:
                writer.writeheader()
            for path in sorted(query_dir.glob("*.sql"), key=rich.sort_key):
                qid = path.stem
                if qid in skip_eval_queries.get(workload, set()):
                    skipped_rows.append({"workload": workload, "query_id": qid, "sql_path": str(path)})
                    continue
                sql = strip_sql(path)
                cached = done.get(qid)
                if cached and not cached.get("error"):
                    rows.append((qid, sql, int(cached["runtime_us"]), str(path)))
                    continue
                runtime_us, plan_doc, error, elapsed_ms = plan_runtime_us(session=session, sql=sql, timeout_ms=timeout_ms)
                writer.writerow(
                    {
                        "query_id": qid,
                        "runtime_us": runtime_us if not error else "",
                        "elapsed_ms": round(elapsed_ms, 3),
                        "error": error or "",
                        "sql_path": str(path),
                    }
                )
                label_handle.flush()
                if plan_doc is not None:
                    plan_handle.write(json.dumps({"workload": workload, "query_id": qid, "plan": plan_doc}) + "\n")
                    plan_handle.flush()
                if error:
                    continue
                rows.append((qid, sql, runtime_us, str(path)))
        if not rows:
            raise RuntimeError(f"No positive runtime labels collected for {workload} from {query_dir}")
        out[workload] = rows
    if skipped_rows:
        skipped_path = eval_root / "skipped_eval_queries.csv"
        with skipped_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["workload", "query_id", "sql_path"])
            writer.writeheader()
            writer.writerows(skipped_rows)
    return out


def load_train_sql_paths(train_query_dir: Path, limit: int) -> list[Path]:
    paths = sorted(train_query_dir.glob("*.sql"), key=rich.sort_key)
    if len(paths) < limit:
        raise RuntimeError(f"{train_query_dir} has only {len(paths)} SQL files; need {limit}")
    return paths[:limit]


def load_or_label_training_rows(
    *,
    session: rich.PsqlSession,
    train_query_dir: Path,
    limit: int,
    out_dir: Path,
    timeout_ms: int,
    exclude_hashes: set[str],
) -> list[tuple[str, str, int, str]]:
    train_root = out_dir / "train_runtime_labels"
    train_root.mkdir(parents=True, exist_ok=True)
    label_path = train_root / "labels.csv"
    plan_path = train_root / "plans.jsonl"
    done: dict[str, dict[str, str]] = {}
    if label_path.exists():
        with label_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("runtime_us"):
                    done[str(row["query_id"])] = row

    rows: list[tuple[str, str, int, str]] = []
    with label_path.open("a", newline="", encoding="utf-8") as label_handle, plan_path.open("a", encoding="utf-8") as plan_handle:
        fieldnames = ["query_id", "runtime_us", "elapsed_ms", "error", "sql_path"]
        writer = csv.DictWriter(label_handle, fieldnames=fieldnames)
        if label_path.stat().st_size == 0:
            writer.writeheader()
        for path in load_train_sql_paths(train_query_dir, limit * 2):
            if len(rows) >= limit:
                break
            sql = strip_sql(path)
            digest = rich.sql_hash(sql)
            if digest in exclude_hashes:
                continue
            qid = path.stem
            cached = done.get(qid)
            if cached and not cached.get("error"):
                rows.append((qid, sql, int(cached["runtime_us"]), str(path)))
                continue
            runtime_us, plan_doc, error, elapsed_ms = plan_runtime_us(session=session, sql=sql, timeout_ms=timeout_ms)
            writer.writerow(
                {
                    "query_id": qid,
                    "runtime_us": runtime_us if not error else "",
                    "elapsed_ms": round(elapsed_ms, 3),
                    "error": error or "",
                    "sql_path": str(path),
                }
            )
            label_handle.flush()
            if plan_doc is not None:
                plan_handle.write(json.dumps({"query_id": qid, "plan": plan_doc}) + "\n")
                plan_handle.flush()
            if error:
                continue
            rows.append((qid, sql, runtime_us, str(path)))
            if len(rows) % 250 == 0:
                print(json.dumps({"runtime_labeled_train": len(rows), "target": limit}), flush=True)
    if len(rows) < limit:
        raise RuntimeError(f"Only collected {len(rows)} / {limit} training runtime labels")
    return rows[:limit]


def write_cost_predictions(
    path: Path,
    items: list[rich.QueryItem],
    preds_us: np.ndarray,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    qerrors: list[float] = []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_id",
                "prediction_us",
                "actual_us",
                "prediction_ms",
                "actual_ms",
                "q_error",
                "source",
                "sql_path",
            ],
        )
        writer.writeheader()
        for item, pred in zip(items, preds_us):
            qe = qerror(float(pred), item.label)
            qerrors.append(qe)
            writer.writerow(
                {
                    "query_id": item.query_id,
                    "prediction_us": float(pred),
                    "actual_us": item.label,
                    "prediction_ms": float(pred) / 1000.0,
                    "actual_ms": item.label / 1000.0,
                    "q_error": qe,
                    "source": item.source,
                    "sql_path": item.sql_path,
                }
            )
    return summarize_qerrors(qerrors)


def train_eval_seed(
    *,
    seed: int,
    run_dir: Path,
    encoder: rich.RichEncoder,
    train_dataset: torch.utils.data.TensorDataset,
    train_items: list[rich.QueryItem],
    val_dataset: torch.utils.data.TensorDataset,
    val_items: list[rich.QueryItem],
    eval_datasets: dict[str, tuple[torch.utils.data.TensorDataset, list[rich.QueryItem]]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_torch_device(args.device)
    seed_dir = run_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    model = rich.train_model(
        encoder=encoder,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hid_units=args.hid,
        device=device,
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "encoder": encoder.to_json(),
            "seed": seed,
            "args": vars(args),
            "label_unit": "microseconds",
        },
        seed_dir / "selected_plan_mscn_cost_model.pt",
    )
    summaries: dict[str, Any] = {}
    train_preds = rich.predict(model, train_dataset, encoder, args.batch_size, device)
    summaries["train_90pct"] = write_cost_predictions(seed_dir / "predictions" / "train_90pct.csv", train_items, train_preds)
    val_preds = rich.predict(model, val_dataset, encoder, args.batch_size, device)
    summaries["validation_10pct"] = write_cost_predictions(seed_dir / "predictions" / "validation_10pct.csv", val_items, val_preds)
    for name, (dataset, items) in eval_datasets.items():
        preds = rich.predict(model, dataset, encoder, args.batch_size, device)
        summaries[name] = write_cost_predictions(seed_dir / "predictions" / f"{name}.csv", items, preds)
    (seed_dir / "qerror_summaries.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return summaries


def aggregate_seed_summaries(seed_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows: dict[str, list[dict[str, Any]]] = {}
    for seed, summaries in seed_summaries.items():
        for split, summary in summaries.items():
            if isinstance(summary, dict) and "median" in summary:
                rows.setdefault(split, []).append({"seed": seed, **summary})
    out: dict[str, Any] = {}
    for split, values in rows.items():
        out[split] = {
            "seed_count": len(values),
            "median_of_medians": float(np.median([v["median"] for v in values])),
            "median_p95": float(np.median([v["p95"] for v in values])),
            "best_seed_by_p95": min(values, key=lambda v: v["p95"])["seed"],
            "seeds": values,
        }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=f"selected_plan_mscn_cost_{now_id()}")
    parser.add_argument("--db-name", default="hollywood_200k")
    parser.add_argument("--pg-container", default="pg_bench")
    parser.add_argument("--pg-user", default="postgres")
    parser.add_argument("--docker-bin", default="docker")
    parser.add_argument("--train-query-dir", type=Path, required=True)
    parser.add_argument("--train-queries", type=int, default=10000)
    parser.add_argument("--job-light-query-dir", type=Path, required=True)
    parser.add_argument("--job-query-dir", type=Path, required=True)
    parser.add_argument("--job-complex-query-dir", type=Path, required=True)
    parser.add_argument("--runtime-timeout-ms", type=int, default=3600000)
    parser.add_argument("--sample-seed", type=int, default=9901)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--model-seeds", default="1,2,3")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hid", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--strict-coverage", action="store_true")
    parser.add_argument(
        "--skip-eval-query",
        action="append",
        default=[],
        help="Skip one final workload eval query as workload:query_id, e.g. job_complex:11. May be repeated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_seeds = parse_model_seeds(args.model_seeds)
    skip_eval_queries = parse_skip_eval_queries(args.skip_eval_query)
    run_dir = RUNS_DIR / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"run_dir": str(run_dir), "model_seeds": model_seeds}), flush=True)

    rich.ensure_container_running(args.docker_bin, args.pg_container)
    session = rich.PsqlSession(
        docker_bin=args.docker_bin,
        container=args.pg_container,
        db_name=args.db_name,
        user=args.pg_user,
    )
    try:
        query_dirs = {
            "job_light": args.job_light_query_dir.resolve(),
            "job": args.job_query_dir.resolve(),
            "job_complex": args.job_complex_query_dir.resolve(),
        }
        eval_rows = load_eval_rows_from_query_dirs(
            session=session,
            query_dirs=query_dirs,
            out_dir=run_dir,
            timeout_ms=args.runtime_timeout_ms,
            skip_eval_queries=skip_eval_queries,
        )
        eval_hashes = {rich.sql_hash(sql) for rows in eval_rows.values() for _, sql, _, _ in rows}
        train_rows = load_or_label_training_rows(
            session=session,
            train_query_dir=args.train_query_dir.resolve(),
            limit=args.train_queries,
            out_dir=run_dir,
            timeout_ms=args.runtime_timeout_ms,
            exclude_hashes=eval_hashes,
        )
        (run_dir / "training_rows_manifest.json").write_text(
            json.dumps(
                {
                    "method": "selected_plan_mscn_cost_runtime_labels_v1",
                    "train_query_dir": str(args.train_query_dir.resolve()),
                    "train_queries_used": len(train_rows),
                    "eval_workloads": {name: len(rows) for name, rows in eval_rows.items()},
                    "label_unit": "microseconds",
                    "runtime_timeout_ms": args.runtime_timeout_ms,
                    "leakage_policy": "Exact eval SQL hashes are excluded from training.",
                    "skipped_eval_queries": {key: sorted(value) for key, value in skip_eval_queries.items()},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        rows_for_tables = {"train": train_rows, **eval_rows}
        sample_tables = paper_mscn.create_random_sample_tables(
            session=session,
            tables=paper_mscn.collect_tables(rows_for_tables),
            sample_id=args.run_id,
            sample_seed=args.sample_seed,
            num_samples=args.num_samples,
            out_dir=run_dir,
        )
        train_items = paper_mscn.write_workload_with_random_samples(
            name="train",
            rows=train_rows,
            out_dir=run_dir / "rich_train",
            session=session,
            sample_tables=sample_tables,
            num_samples=args.num_samples,
        )
        eval_items_by_name: dict[str, list[rich.QueryItem]] = {}
        for name, rows in eval_rows.items():
            eval_items_by_name[name] = paper_mscn.write_workload_with_random_samples(
                name=name,
                rows=rows,
                out_dir=run_dir / f"rich_eval_{name}",
                session=session,
                sample_tables=sample_tables,
                num_samples=args.num_samples,
            )
    finally:
        session.close()

    train_loaded, train_bitmaps = rich.load_workload(run_dir / "rich_train" / "workload.jsonl", run_dir / "rich_train" / "bitmaps.bin", args.num_samples)
    split = int(len(train_loaded) * 0.9)
    train_core_items = train_loaded[:split]
    train_core_bitmaps = train_bitmaps[:split]
    val_items = train_loaded[split:]
    val_bitmaps = train_bitmaps[split:]

    encoder = rich.RichEncoder(train_loaded, train_bitmaps)
    loaded_eval_items: dict[str, list[rich.QueryItem]] = {}
    loaded_eval_bitmaps: dict[str, list[list[np.ndarray]]] = {}
    for name in eval_items_by_name:
        items, bitmaps = rich.load_workload(run_dir / f"rich_eval_{name}" / "workload.jsonl", run_dir / f"rich_eval_{name}" / "bitmaps.bin", args.num_samples)
        loaded_eval_items[name] = items
        loaded_eval_bitmaps[name] = bitmaps
    paper_mscn.extend_encoder_capacity(encoder, [item for items in loaded_eval_items.values() for item in items])

    coverage = encoder.audit_eval_coverage(loaded_eval_items)
    (run_dir / "coverage_audit.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    has_gaps = any(values for audit in coverage.values() for key, values in audit.items() if key.startswith("missing_"))
    if has_gaps and args.strict_coverage:
        raise RuntimeError(f"Coverage gaps detected; see {run_dir / 'coverage_audit.json'}")

    (run_dir / "encoder_manifest.json").write_text(json.dumps(encoder.to_json(), indent=2), encoding="utf-8")
    train_dataset, _ = encoder.encode_dataset(train_core_items, train_core_bitmaps)
    val_dataset, _ = encoder.encode_dataset(val_items, val_bitmaps)
    eval_datasets = {
        name: (encoder.encode_dataset(items, loaded_eval_bitmaps[name])[0], items)
        for name, items in loaded_eval_items.items()
    }

    seed_summaries: dict[str, dict[str, Any]] = {}
    for seed in model_seeds:
        print(json.dumps({"training_model_seed": seed}), flush=True)
        seed_summaries[str(seed)] = train_eval_seed(
            seed=seed,
            run_dir=run_dir,
            encoder=encoder,
            train_dataset=train_dataset,
            train_items=train_core_items,
            val_dataset=val_dataset,
            val_items=val_items,
            eval_datasets=eval_datasets,
            args=args,
        )

    report = {
        "method": "selected_plan_mscn_cost_v1",
        "run_dir": str(run_dir),
        "db_name": args.db_name,
        "label_unit": "microseconds",
        "train_queries": args.train_queries,
        "epochs": args.epochs,
        "num_samples": args.num_samples,
        "sample_seed": args.sample_seed,
        "model_seeds": model_seeds,
        "skipped_eval_queries": {key: sorted(value) for key, value in skip_eval_queries.items()},
        "aggregate": aggregate_seed_summaries(seed_summaries),
        "seed_summaries": seed_summaries,
    }
    (run_dir / "selected_plan_mscn_cost_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"done": True, "run_dir": str(run_dir), "aggregate": report["aggregate"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
