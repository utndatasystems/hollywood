#!/usr/bin/env python3
"""Paper-grade MSCN cardinality runner for Hollywood IMDb exports.

This runner intentionally lives outside the older one-file experiment script.
It reuses the battle-tested parsing/query-generation/model components from
``run_rich_mscn_experiment.py`` while replacing the weak sampling path:

* old path: bitmap sample = first N rows by primary key;
* this path: bitmap sample = deterministic random UNLOGGED sample table.

The default recipe is deliberately closer to the MSCN paper scale: 100k
training queries, 100 epochs, and three model seeds.  Use smaller settings for
smoke tests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent / "vendor"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_rich_mscn_experiment as rich  # noqa: E402


RUNS_DIR = ROOT / "experiments" / "mscn" / "runs"
IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
NON_CARDINALITY_NODE_TYPES = {
    "Aggregate",
    "Gather",
    "Gather Merge",
    "Hash",
    "Limit",
    "Materialize",
    "Memoize",
    "Result",
    "Sort",
    "Unique",
}
SUBPLAN_METADATA_FIELDS = [
    "subplan_query_id",
    "source_workload",
    "source_query_id",
    "node_id",
    "parent_id",
    "depth",
    "node_type",
    "join_type",
    "relation_name",
    "alias",
    "aliases",
    "tables",
    "plan_rows",
    "actual_rows_per_loop",
    "actual_loops",
    "actual_total_rows",
    "postgres_q_error_per_loop",
    "postgres_q_error_total",
    "logical_count",
    "label_source",
    "count_ms",
    "count_error",
    "sql_hash",
    "sql_path",
]


def now_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def qident(name: str) -> str:
    if not IDENT_RE.match(name):
        raise ValueError(f"Unsafe SQL identifier: {name!r}")
    return '"' + name.replace('"', '""') + '"'


def bare_ident(name: str) -> str:
    if not IDENT_RE.match(name):
        raise ValueError(f"Unsafe SQL identifier: {name!r}")
    return name


def compact_hash(text: str, n: int = 10) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def sample_table_name(*, sample_id: str, table: str) -> str:
    # PostgreSQL identifiers are truncated at 63 bytes.  Keep the hash early.
    safe_table = re.sub(r"[^A-Za-z0-9_]+", "_", table.lower())[:34]
    return f"mscn_s_{compact_hash(sample_id + ':' + table)}_{safe_table}"


def index_name(table_name: str) -> str:
    return f"{table_name[:50]}_id_idx"


def set_rich_globals(args: argparse.Namespace) -> None:
    rich.BASE_RUN = args.base_run.resolve()
    rich.MSCN_PAPER_RUN = args.mscn_paper_run.resolve()
    rich.DEFAULT_DB_NAME = args.db_name
    rich.EVAL_EXACT_QUERY_DIR = args.exact_query_dir.resolve()
    rich.EVAL_COMPLEX_QUERY_DIR = args.complex_query_dir.resolve()
    rich.NUM_SAMPLES = int(args.num_samples)


def collect_tables(rows_by_name: dict[str, list[tuple[str, str, int, str]]]) -> list[str]:
    tables: set[str] = set()
    for rows in rows_by_name.values():
        for qid, sql, label, sql_path in rows:
            item = rich.make_query_item(qid, sql, label, sql_path, "collect")
            tables.update(item.tables)
    return sorted(tables)


def load_queryitem_workload(path: Path, *, source_name: str) -> list[tuple[str, str, int, str]]:
    """Load an existing workload.jsonl as eval rows, then rebuild bitmaps here."""

    rows: list[tuple[str, str, int, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = rich.query_from_json(json.loads(line))
            rows.append((item.query_id, item.sql.strip().rstrip(";"), int(item.label), item.sql_path))
    if not rows:
        raise ValueError(f"No workload rows loaded from {path}")
    return rows


def load_labeled_query_dir(query_dir: Path, labels_path: Path, *, source_name: str) -> list[tuple[str, str, int, str]]:
    """Load an eval SQL directory using labels from a plan-trace query summary."""

    labels = rich.read_labels(labels_path)
    rows: list[tuple[str, str, int, str]] = []
    for path in sorted(query_dir.glob("*.sql"), key=rich.sort_key):
        label = labels.get(path.stem)
        if label is None:
            continue
        rows.append((path.stem, path.read_text(encoding="utf-8-sig").strip().rstrip(";"), label, str(path)))
    if not rows:
        raise ValueError(f"No labeled {source_name} rows loaded from {query_dir}; checked labels at {labels_path}")
    return rows


def load_extra_eval_workloads(specs: list[str]) -> dict[str, list[tuple[str, str, int, str]]]:
    extra: dict[str, list[tuple[str, str, int, str]]] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError("--extra-eval-workload must be NAME=PATH")
        name, raw_path = spec.split("=", 1)
        name = name.strip()
        if not IDENT_RE.match(name):
            raise ValueError(f"Unsafe eval workload name: {name!r}")
        path = Path(raw_path).resolve()
        extra[name] = load_queryitem_workload(path, source_name=name)
    return extra


def load_extra_eval_query_dirs(specs: list[str], *, base_run: Path) -> dict[str, list[tuple[str, str, int, str]]]:
    extra: dict[str, list[tuple[str, str, int, str]]] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError("--extra-eval-query-dir must be NAME=PATH")
        name, raw_path = spec.split("=", 1)
        name = name.strip()
        if not IDENT_RE.match(name):
            raise ValueError(f"Unsafe eval workload name: {name!r}")
        query_dir = Path(raw_path).resolve()
        labels_path = base_run / "postgres" / name / "query_summary.csv"
        extra[name] = load_labeled_query_dir(query_dir, labels_path, source_name=name)
    return extra


def parse_name_path_specs(specs: list[str], *, label: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"{label} must be NAME=PATH")
        name, raw_path = spec.split("=", 1)
        name = name.strip()
        if not IDENT_RE.match(name):
            raise ValueError(f"Unsafe workload name in {label}: {name!r}")
        out[name] = Path(raw_path).resolve()
    return out


def default_raw_plan_dir(base_run: Path, workload_name: str) -> Path:
    return base_run / "postgres" / workload_name / "raw_plans"


def qerror(est: Any, actual: Any) -> float | None:
    try:
        est_f = max(float(est), 1.0)
        act_f = max(float(actual), 1.0)
    except Exception:
        return None
    return est_f / act_f if est_f >= act_f else act_f / est_f


def create_random_sample_tables(
    *,
    session: rich.PsqlSession,
    tables: list[str],
    sample_id: str,
    sample_seed: int,
    num_samples: int,
    out_dir: Path,
) -> dict[str, str]:
    """Create deterministic random sample tables and return base->sample map."""

    effective_sample_id = f"{sample_id}:seed={sample_seed}:n={num_samples}"
    samples: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    for table in tables:
        sample_table = sample_table_name(sample_id=effective_sample_id, table=table)
        samples[table] = sample_table
        seed_text = f"{sample_seed}:{table}:{num_samples}"
        ddl = f"""
        CREATE UNLOGGED TABLE IF NOT EXISTS {qident(sample_table)} AS
        SELECT *
        FROM {qident(table)}
        ORDER BY md5(id::text || ':' || {rich.sql_quote(seed_text)})
        LIMIT {int(num_samples)};
        CREATE INDEX IF NOT EXISTS {qident(index_name(sample_table))} ON {qident(sample_table)} (id);
        ANALYZE {qident(sample_table)};
        """
        session.run(ddl)
        count_text = session.run(f"SELECT count(*) FROM {qident(sample_table)};")
        try:
            count = int(float(count_text.strip().splitlines()[-1]))
        except Exception:
            count = None
        records.append(
            {
                "base_table": table,
                "sample_table": sample_table,
                "sample_seed": sample_seed,
                "requested_rows": num_samples,
                "sample_rows": count,
            }
        )
        print(json.dumps({"sample_table": sample_table, "base_table": table, "rows": count}), flush=True)

    manifest = {
        "method": "paper_mscn_random_materialized_samples_v1",
        "sample_id": sample_id,
        "effective_sample_id": effective_sample_id,
        "sample_seed": sample_seed,
        "num_samples": num_samples,
        "tables": records,
    }
    (out_dir / "sample_tables_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return samples


def bitmap_sql_from_sample(sample_table: str, alias: str, conditions: list[str]) -> str:
    condition = " AND ".join(f"({c})" for c in conditions if c.strip()) if conditions else "TRUE"
    return f"""
    SELECT COALESCE(string_agg(CASE WHEN {condition} THEN '1' ELSE '0' END, '' ORDER BY {alias}.id), '')
    FROM {qident(sample_table)} AS {alias}
    """


def write_workload_with_random_samples(
    *,
    name: str,
    rows: list[tuple[str, str, int, str]],
    out_dir: Path,
    session: rich.PsqlSession,
    sample_tables: dict[str, str],
    num_samples: int,
) -> list[rich.QueryItem]:
    out_dir.mkdir(parents=True, exist_ok=True)
    workload_path = out_dir / "workload.jsonl"
    bitmaps_path = out_dir / "bitmaps.bin"
    manifest_path = out_dir / "manifest.json"
    items = [rich.make_query_item(qid, sql, label, sql_path, name) for qid, sql, label, sql_path in rows]
    cache: dict[tuple[str, str, str], str] = {}
    num_bytes = int((num_samples + 7) >> 3)
    query_meta: list[dict[str, Any]] = []

    with workload_path.open("w", encoding="utf-8") as w_handle, bitmaps_path.open("wb") as b_handle:
        for idx, item in enumerate(items, start=1):
            w_handle.write(json.dumps(rich.query_to_json(item), ensure_ascii=False) + "\n")
            b_handle.write(struct.pack("<I", len(item.table_aliases)))
            zero_bitmaps = 0
            densities: list[float] = []
            for alias in item.table_aliases:
                table = item.alias_to_table[alias]
                sample_table = sample_tables.get(table)
                if sample_table is None:
                    raise KeyError(f"No sample table for {table}")
                conditions = item.local_conditions.get(alias, [])
                key = (sample_table, alias, rich.json_dumps(conditions))
                bit_text = cache.get(key)
                if bit_text is None:
                    bit_text = session.run(bitmap_sql_from_sample(sample_table, alias, conditions))
                    cache[key] = bit_text
                bits = [1 if ch == "1" else 0 for ch in bit_text[:num_samples]]
                one_count = sum(bits)
                zero_bitmaps += 1 if one_count == 0 else 0
                densities.append(one_count / float(max(1, len(bits))))
                packed = rich.pack_bits(bits, num_samples)
                if len(packed) != num_bytes:
                    raise RuntimeError(f"Packed bitmap has {len(packed)} bytes, expected {num_bytes}")
                b_handle.write(packed)
            query_meta.append(
                {
                    "query_id": item.query_id,
                    "tables": len(item.tables),
                    "joins": len([j for j in item.joins if j != rich.DUMMY_JOIN]),
                    "predicates": len(item.predicates),
                    "zero_bitmaps": zero_bitmaps,
                    "mean_bitmap_density": float(np.mean(densities)) if densities else None,
                }
            )
            if idx % 500 == 0:
                print(json.dumps({"workload": name, "bitmap_query": idx, "total": len(items)}), flush=True)

    manifest = {
        "method": "paper_mscn_random_sample_workload_v1",
        "name": name,
        "query_count": len(items),
        "workload_jsonl": str(workload_path),
        "bitmaps": str(bitmaps_path),
        "num_materialized_samples": num_samples,
        "sample_policy": "deterministic_random_unlogged_tables",
        "summary": {
            "columns": len({p.column for item in items for p in item.predicates}),
            "operators": sorted({p.operator for item in items for p in item.predicates}),
            "joins": len({j for item in items for j in item.joins}),
            "tables": len({t for item in items for t in item.tables}),
            "median_tables": float(np.median([len(item.tables) for item in items])) if items else 0,
        },
        "queries": query_meta,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return items


def csv_number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def plan_doc_from_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list) and payload:
        return payload[0]
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Unexpected plan JSON payload in {path}")


def flatten_plan_with_aliases(
    plan: dict[str, Any],
    *,
    query_id: str,
    known_aliases: set[str],
    parent_id: str = "",
    depth: int = 0,
    counter: list[int] | None = None,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Flatten a PostgreSQL plan and attach the base aliases in each subtree."""

    if counter is None:
        counter = [0]
    counter[0] += 1
    node_id = f"{query_id}:{counter[0]:04d}"
    child_rows: list[dict[str, Any]] = []
    aliases: set[str] = set()
    alias = str(plan.get("Alias") or "").lower()
    if alias in known_aliases:
        aliases.add(alias)
    for child in plan.get("Plans", []) or []:
        rows, child_aliases = flatten_plan_with_aliases(
            child,
            query_id=query_id,
            known_aliases=known_aliases,
            parent_id=node_id,
            depth=depth + 1,
            counter=counter,
        )
        child_rows.extend(rows)
        aliases.update(child_aliases)

    plan_rows = csv_number(plan.get("Plan Rows"))
    actual_rows = csv_number(plan.get("Actual Rows"))
    actual_loops = csv_number(plan.get("Actual Loops")) or 1.0
    actual_total_rows = None if actual_rows is None else actual_rows * actual_loops
    row = {
        "node_id": node_id,
        "parent_id": parent_id,
        "depth": depth,
        "node_type": str(plan.get("Node Type") or ""),
        "join_type": str(plan.get("Join Type") or ""),
        "relation_name": str(plan.get("Relation Name") or ""),
        "alias": alias,
        "plan_rows": plan_rows,
        "actual_rows_per_loop": actual_rows,
        "actual_loops": actual_loops,
        "actual_total_rows": actual_total_rows,
        "postgres_q_error_per_loop": qerror(plan_rows, actual_rows),
        "postgres_q_error_total": qerror(plan_rows, actual_total_rows),
        "aliases": sorted(aliases),
    }
    return [row] + child_rows, aliases


def raw_join_conditions(sql: str, selected_aliases: set[str], known_aliases: set[str]) -> list[str]:
    _, where_clause = rich.extract_from_where(sql)
    conditions: list[str] = []
    for part in rich.split_top_level_and(where_clause):
        cleaned = rich.strip_outer_parens(part)
        aliases = rich.aliases_in_condition(cleaned, known_aliases)
        if len(aliases) >= 2 and aliases.issubset(selected_aliases):
            conditions.append(cleaned)
    return conditions


def build_logical_subplan_sql(root_item: rich.QueryItem, aliases: list[str]) -> str:
    selected = set(aliases)
    known = set(root_item.alias_to_table)
    from_parts = [
        f"{bare_ident(root_item.alias_to_table[alias])} AS {bare_ident(alias)}"
        for alias in root_item.table_aliases
        if alias in selected
    ]
    conditions: list[str] = []
    for alias in root_item.table_aliases:
        if alias in selected:
            conditions.extend(root_item.local_conditions.get(alias, []))
    conditions.extend(raw_join_conditions(root_item.sql, selected, known))

    seen: set[str] = set()
    deduped: list[str] = []
    for condition in conditions:
        key = " ".join(condition.split()).lower()
        if key not in seen:
            seen.add(key)
            deduped.append(condition)

    sql = "SELECT COUNT(*) FROM " + ", ".join(from_parts)
    if deduped:
        sql += "\nWHERE " + "\n  AND ".join(f"({condition})" for condition in deduped)
    else:
        sql += "\nWHERE TRUE"
    return sql


def _split_generated_count_sql(sql: str) -> tuple[list[str], list[str]] | None:
    """Split SQL emitted by build_logical_subplan_sql into FROM parts/conditions."""

    text = sql.strip().rstrip(";")
    match = re.match(r"(?is)^SELECT\s+COUNT\(\*\)\s+FROM\s+(.+?)\s+WHERE\s+(.+)$", text)
    if not match:
        return None
    from_text = match.group(1).strip()
    where_text = match.group(2).strip()
    from_parts = [part.strip() for part in from_text.split(",") if part.strip()]
    conditions = []
    for raw in re.split(r"\n\s+AND\s+", where_text):
        condition = raw.strip()
        if condition.startswith("(") and condition.endswith(")"):
            condition = condition[1:-1].strip()
        if condition and condition.upper() != "TRUE":
            conditions.append(condition)
    if not from_parts:
        return None
    return from_parts, conditions


def _alias_from_part(from_part: str) -> str | None:
    match = re.search(r"(?i)\s+AS\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", from_part.strip())
    return match.group(1).lower() if match else None


def _condition_aliases(condition: str, aliases: set[str]) -> set[str]:
    seen = {match.group(1).lower() for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.", condition)}
    return seen & aliases


def _connected_components(aliases: list[str], conditions: list[str]) -> list[set[str]]:
    alias_set = {alias.lower() for alias in aliases}
    parent = {alias: alias for alias in alias_set}

    def find(alias: str) -> str:
        while parent[alias] != alias:
            parent[alias] = parent[parent[alias]]
            alias = parent[alias]
        return alias

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for condition in conditions:
        refs = sorted(_condition_aliases(condition, alias_set))
        if len(refs) > 1:
            first = refs[0]
            for other in refs[1:]:
                union(first, other)

    grouped: dict[str, set[str]] = {}
    for alias in alias_set:
        grouped.setdefault(find(alias), set()).add(alias)
    return list(grouped.values())


def exact_count_logical_subplan(
    session: rich.PsqlSession,
    sql: str,
    timeout_ms: int,
) -> tuple[int | float | None, str | None, float]:
    """Count a logical subplan exactly, decomposing independent components.

    PostgreSQL can spend a very long time executing COUNT(*) over disconnected
    Cartesian products such as ``movie_info x movie_info_idx``.  For independent
    components the exact cardinality is the product of exact component counts,
    so this keeps the label identical while avoiding needless cross-product
    execution.
    """

    parsed = _split_generated_count_sql(sql)
    if parsed is None:
        return rich.count_query(session, sql, timeout_ms)
    from_parts, conditions = parsed
    alias_to_from: dict[str, str] = {}
    for part in from_parts:
        alias = _alias_from_part(part)
        if not alias:
            return rich.count_query(session, sql, timeout_ms)
        alias_to_from[alias] = part

    aliases = list(alias_to_from)
    components = _connected_components(aliases, conditions)
    if len(components) <= 1:
        return rich.count_query(session, sql, timeout_ms)

    product = 1
    elapsed_total = 0.0
    alias_set = set(aliases)
    for component in components:
        component_from = [alias_to_from[alias] for alias in aliases if alias in component]
        component_conditions = []
        for condition in conditions:
            refs = _condition_aliases(condition, alias_set)
            if not refs:
                continue
            if refs <= component:
                component_conditions.append(condition)
        component_sql = "SELECT COUNT(*) FROM " + ", ".join(component_from)
        if component_conditions:
            component_sql += "\nWHERE " + "\n  AND ".join(f"({condition})" for condition in component_conditions)
        else:
            component_sql += "\nWHERE TRUE"
        actual, error, elapsed_ms = rich.count_query(session, component_sql, timeout_ms)
        elapsed_total += float(elapsed_ms or 0.0)
        if error:
            return None, error, elapsed_total
        product *= int(round(float(actual)))
    return product, None, elapsed_total


def generate_plan_subplan_rows(
    *,
    workload_name: str,
    root_rows: list[tuple[str, str, int, str]],
    raw_plan_dir: Path,
    out_dir: Path,
    session: rich.PsqlSession,
    timeout_ms: int,
    limit: int,
) -> list[tuple[str, str, int, str]]:
    """Build logical subquery rows from PostgreSQL plan-node alias subtrees."""

    out_dir.mkdir(parents=True, exist_ok=True)
    query_dir = out_dir / "queries"
    query_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / "subplan_node_metadata.csv"
    manifest_path = out_dir / "manifest.json"
    rows: list[tuple[str, str, int, str]] = []
    metadata_rows: list[dict[str, Any]] = []
    count_cache: dict[str, tuple[int, float]] = {}
    count_errors: list[dict[str, Any]] = []
    skipped = {
        "missing_plan": 0,
        "wrapper_node": 0,
        "no_aliases": 0,
        "duplicate_alias_set": 0,
        "count_error": 0,
        "count_session_restarts": 0,
        "limit": 0,
    }

    if manifest_path.exists() and metadata_path.exists():
        with metadata_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                qid = str(row.get("subplan_query_id") or "")
                sql_path = Path(str(row.get("sql_path") or query_dir / f"{qid}.sql"))
                label_raw = row.get("logical_count") or ""
                if not qid or not sql_path.exists() or not label_raw:
                    continue
                rows.append(
                    (
                        qid,
                        sql_path.read_text(encoding="utf-8-sig").strip().rstrip(";"),
                        max(1, int(round(float(label_raw)))),
                        str(sql_path),
                    )
                )
        if rows:
            print(
                json.dumps(
                    {
                        "resume_subplans": workload_name,
                        "subplan_query_count": len(rows),
                        "manifest": str(manifest_path),
                    }
                ),
                flush=True,
            )
            return rows

    for source_qid, sql, _label, sql_path in root_rows:
        plan_path = raw_plan_dir / f"{source_qid}.json"
        if not plan_path.exists():
            skipped["missing_plan"] += 1
            continue
        root_item = rich.make_query_item(source_qid, sql, 1, sql_path, workload_name)
        plan_doc = plan_doc_from_file(plan_path)
        plan_root = plan_doc.get("Plan", plan_doc)
        plan_rows, _ = flatten_plan_with_aliases(
            plan_root,
            query_id=source_qid,
            known_aliases=set(root_item.alias_to_table),
        )
        seen_alias_sets: set[tuple[str, ...]] = set()
        for node in plan_rows:
            if limit and len(rows) >= limit:
                skipped["limit"] += 1
                break
            node_type = str(node.get("node_type") or "")
            aliases = list(node.get("aliases") or [])
            alias_key = tuple(aliases)
            if node_type in NON_CARDINALITY_NODE_TYPES:
                skipped["wrapper_node"] += 1
                continue
            if not aliases:
                skipped["no_aliases"] += 1
                continue
            if alias_key in seen_alias_sets:
                skipped["duplicate_alias_set"] += 1
                continue
            seen_alias_sets.add(alias_key)
            subplan_sql = build_logical_subplan_sql(root_item, aliases)
            digest = rich.sql_hash(subplan_sql)
            cached = count_cache.get(digest)
            if cached is None:
                actual, error, elapsed_ms = exact_count_logical_subplan(session, subplan_sql, timeout_ms)
                if error:
                    if getattr(session, "restart_count", 0):
                        skipped["count_session_restarts"] = int(getattr(session, "restart_count", 0))
                    skipped["count_error"] += 1
                    count_errors.append(
                        {
                            "source_query_id": source_qid,
                            "node_id": node.get("node_id"),
                            "node_type": node_type,
                            "aliases": " ".join(aliases),
                            "sql_hash": digest,
                            "error": str(error).splitlines()[0][:500],
                        }
                    )
                    continue
                cached = (int(round(float(actual))), elapsed_ms)
                count_cache[digest] = cached
            actual, elapsed_ms = cached
            local_index = len(rows) + 1
            subplan_qid = f"{workload_name}_node_{local_index:05d}"
            out_sql_path = query_dir / f"{subplan_qid}.sql"
            out_sql_path.write_text(subplan_sql.strip().rstrip(";") + ";\n", encoding="utf-8")
            label = max(1, int(round(float(actual))))
            rows.append((subplan_qid, subplan_sql.strip().rstrip(";"), label, str(out_sql_path)))
            metadata_rows.append(
                {
                    "subplan_query_id": subplan_qid,
                    "source_workload": workload_name,
                    "source_query_id": source_qid,
                    "node_id": node.get("node_id"),
                    "parent_id": node.get("parent_id"),
                    "depth": node.get("depth"),
                    "node_type": node_type,
                    "join_type": node.get("join_type"),
                    "relation_name": node.get("relation_name"),
                    "alias": node.get("alias"),
                    "aliases": " ".join(aliases),
                    "tables": " ".join(root_item.alias_to_table[a] for a in aliases),
                    "plan_rows": node.get("plan_rows"),
                    "actual_rows_per_loop": node.get("actual_rows_per_loop"),
                    "actual_loops": node.get("actual_loops"),
                    "actual_total_rows": node.get("actual_total_rows"),
                    "postgres_q_error_per_loop": node.get("postgres_q_error_per_loop"),
                    "postgres_q_error_total": node.get("postgres_q_error_total"),
                    "logical_count": label,
                    "label_source": "exact_logical_count",
                    "count_ms": round(elapsed_ms, 3),
                    "count_error": "",
                    "sql_hash": digest,
                    "sql_path": str(out_sql_path),
                }
            )
        if limit and len(rows) >= limit:
            break

    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUBPLAN_METADATA_FIELDS)
        writer.writeheader()
        for row in metadata_rows:
            writer.writerow({field: row.get(field) for field in SUBPLAN_METADATA_FIELDS})
    if count_errors:
        with (out_dir / "count_errors.csv").open("w", encoding="utf-8", newline="") as handle:
            fieldnames = ["source_query_id", "node_id", "node_type", "aliases", "sql_hash", "error"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in count_errors:
                writer.writerow({field: row.get(field) for field in fieldnames})
    manifest = {
        "method": "paper_mscn_logical_subplans_from_postgres_plan_v1",
        "workload_name": workload_name,
        "raw_plan_dir": str(raw_plan_dir),
        "root_query_count": len(root_rows),
        "subplan_query_count": len(rows),
        "limit": limit,
        "timeout_ms": timeout_ms,
        "label_policy": "exact COUNT(*) of the reconstructed logical alias subtree; PostgreSQL node actuals are retained only as metadata.",
        "duplicate_policy": "Plan-node occurrences are preserved; exact COUNT(*) labels are cached by SQL hash for speed.",
        "unique_count_sql": len(count_cache),
        "skipped": skipped,
        "metadata_csv": str(metadata_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return rows


def extend_encoder_capacity(encoder: rich.RichEncoder, eval_items: list[rich.QueryItem]) -> None:
    """Avoid truncating eval tables/predicates while keeping train-only vocab."""

    if not eval_items:
        return
    encoder.max_tables = max(encoder.max_tables, max(len(item.tables) for item in eval_items))
    encoder.max_predicates = max(encoder.max_predicates, max(max(1, len(item.predicates)) for item in eval_items))
    encoder.max_joins = max(encoder.max_joins, max(max(1, len(item.joins)) for item in eval_items))


def parse_model_seeds(text: str) -> list[int]:
    seeds = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not seeds:
        raise ValueError("--model-seeds must contain at least one integer")
    return seeds


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


def load_training_rows(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    session: rich.PsqlSession,
    eval_rows: dict[str, list[tuple[str, str, int, str]]],
) -> list[tuple[str, str, int, str]]:
    eval_hashes = {rich.sql_hash(sql) for rows in eval_rows.values() for _, sql, _, _ in rows}
    for query_dir in args.exclude_query_dir:
        eval_hashes.update(rich.load_sql_hashes_from_dir(query_dir.resolve()))

    if args.train_workload:
        source_rows = load_queryitem_workload(args.train_workload.resolve(), source_name="train_workload")
        relabel_dir = run_dir / "training_relabel"
        relabel_dir.mkdir(parents=True, exist_ok=True)
        relabel_path = relabel_dir / "labels.csv"
        cached: dict[str, dict[str, str]] = {}
        if relabel_path.exists():
            with relabel_path.open(newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    cached[str(row.get("query_id", ""))] = row

        train_rows: list[tuple[str, str, int, str]] = []
        seen_hashes: set[str] = set()
        skipped = {
            "eval_sql_overlap": 0,
            "duplicate_sql": 0,
            "zero_actual": 0,
            "count_error": 0,
        }
        with relabel_path.open("a", newline="", encoding="utf-8") as handle:
            fieldnames = ["query_id", "label", "elapsed_ms", "error", "sql_hash", "sql_path"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if relabel_path.stat().st_size == 0:
                writer.writeheader()
            for qid, sql, old_label, sql_path in source_rows:
                if len(train_rows) >= int(args.train_queries):
                    break
                digest = rich.sql_hash(sql)
                if digest in eval_hashes:
                    skipped["eval_sql_overlap"] += 1
                    continue
                if digest in seen_hashes:
                    skipped["duplicate_sql"] += 1
                    continue
                seen_hashes.add(digest)

                cached_row = cached.get(qid)
                if cached_row and cached_row.get("label") and not cached_row.get("error"):
                    label = int(round(float(cached_row["label"])))
                    train_rows.append((qid, sql, max(1, label), sql_path))
                    continue

                if args.relabel_train_workload:
                    actual, error, elapsed_ms = exact_count_logical_subplan(session, sql, args.statement_timeout_ms)
                    if error:
                        skipped["count_error"] += 1
                        writer.writerow(
                            {
                                "query_id": qid,
                                "label": "",
                                "elapsed_ms": round(elapsed_ms, 3),
                                "error": str(error).splitlines()[0][:500],
                                "sql_hash": digest,
                                "sql_path": sql_path,
                            }
                        )
                        handle.flush()
                        continue
                    label = int(round(float(actual)))
                    if label <= 0:
                        skipped["zero_actual"] += 1
                        writer.writerow(
                            {
                                "query_id": qid,
                                "label": "",
                                "elapsed_ms": round(elapsed_ms, 3),
                                "error": "zero_actual",
                                "sql_hash": digest,
                                "sql_path": sql_path,
                            }
                        )
                        handle.flush()
                        continue
                    writer.writerow(
                        {
                            "query_id": qid,
                            "label": label,
                            "elapsed_ms": round(elapsed_ms, 3),
                            "error": "",
                            "sql_hash": digest,
                            "sql_path": sql_path,
                        }
                    )
                    handle.flush()
                else:
                    label = int(old_label)

                train_rows.append((qid, sql, max(1, int(label)), sql_path))
                if len(train_rows) % 1000 == 0:
                    print(
                        json.dumps(
                            {
                                "training_workload_rows": len(train_rows),
                                "target": int(args.train_queries),
                                "skipped": skipped,
                            }
                        ),
                        flush=True,
                    )

        if len(train_rows) < int(args.train_queries):
            raise RuntimeError(
                f"Only prepared {len(train_rows)} / {args.train_queries} rows from {args.train_workload}; skipped={skipped}"
            )
        random.Random(args.query_seed).shuffle(train_rows)
        train_hashes = {rich.sql_hash(sql) for _, sql, _, _ in train_rows}
        overlap = sorted(train_hashes & eval_hashes)
        if overlap:
            raise RuntimeError(f"Training SQL overlaps eval SQL by hash: {overlap[:5]}")
        manifest = {
            "method": "paper_mscn_training_rows_from_saved_workload_v1",
            "train_target": args.train_queries,
            "train_workload": str(args.train_workload.resolve()),
            "source_rows_loaded": len(source_rows),
            "relabel_train_workload": bool(args.relabel_train_workload),
            "train_queries_used": len(train_rows),
            "exact_eval_sql_hash_overlap": len(overlap),
            "skipped": skipped,
            "label_policy": "Recomputed labels on target DB with COUNT(*)" if args.relabel_train_workload else "Reused labels from saved workload",
        }
        (run_dir / "training_rows_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return train_rows

    raw_existing_rows = rich.load_existing_job_variants(limit=args.existing_job_variants)
    existing_rows = [row for row in raw_existing_rows if rich.sql_hash(row[1]) not in eval_hashes]
    existing_hashes = {rich.sql_hash(sql) for _, sql, _, _ in existing_rows}
    synthetic_needed = max(0, int(args.train_queries) - len(existing_rows))
    synthetic_rows = rich.generate_synthetic_queries(
        session=session,
        out_dir=run_dir / "generation",
        target_count=synthetic_needed,
        eval_hashes=eval_hashes,
        existing_hashes=existing_hashes,
        seed=args.query_seed,
        timeout_ms=args.statement_timeout_ms,
    )
    train_rows = existing_rows + synthetic_rows
    random.Random(args.query_seed).shuffle(train_rows)
    train_hashes = {rich.sql_hash(sql) for _, sql, _, _ in train_rows}
    overlap = sorted(train_hashes & eval_hashes)
    if overlap:
        raise RuntimeError(f"Training SQL overlaps eval SQL by hash: {overlap[:5]}")

    manifest = {
        "method": "paper_mscn_training_rows_v1",
        "train_target": args.train_queries,
        "existing_job_variants_requested": args.existing_job_variants,
        "existing_job_variants_loaded": len(raw_existing_rows),
        "existing_job_variants_used": len(existing_rows),
        "synthetic_queries_used": len(synthetic_rows),
        "train_queries_used": len(train_rows),
        "exact_eval_sql_hash_overlap": len(overlap),
        "leakage_policy": "Exact eval SQL hashes are excluded from training.",
    }
    (run_dir / "training_rows_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return train_rows


def train_and_eval_seed(
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
    seed_dir = run_dir / "model_seeds" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_torch_device(args.device)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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
        },
        seed_dir / "paper_mscn_model.pt",
    )

    summaries: dict[str, Any] = {}
    train_preds = rich.predict(model, train_dataset, encoder, args.batch_size, device)
    summaries["train_90pct"] = rich.write_predictions(seed_dir / "predictions" / "train_90pct.csv", train_items, train_preds)
    val_preds = rich.predict(model, val_dataset, encoder, args.batch_size, device)
    summaries["validation_10pct"] = rich.write_predictions(seed_dir / "predictions" / "validation_10pct.csv", val_items, val_preds)
    for name, (dataset, items) in eval_datasets.items():
        preds = rich.predict(model, dataset, encoder, args.batch_size, device)
        summaries[name] = rich.write_predictions(seed_dir / "predictions" / f"{name}.csv", items, preds)
    (seed_dir / "qerror_summaries.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return summaries


def aggregate_seed_summaries(seed_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows: dict[str, list[dict[str, Any]]] = {}
    for seed, summaries in seed_summaries.items():
        for split, summary in summaries.items():
            if isinstance(summary, dict) and "median" in summary:
                rows.setdefault(split, []).append({"seed": seed, **summary})
    aggregate: dict[str, Any] = {}
    for split, values in rows.items():
        aggregate[split] = {
            "seed_count": len(values),
            "median_of_medians": float(np.median([v["median"] for v in values])),
            "median_p95": float(np.median([v["p95"] for v in values])),
            "best_seed_by_p95": min(values, key=lambda v: v["p95"])["seed"],
            "seeds": values,
        }
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=f"paper_mscn_{now_id()}")
    parser.add_argument("--db-name", default=rich.DEFAULT_DB_NAME)
    parser.add_argument("--pg-container", default=rich.DEFAULT_PG_CONTAINER)
    parser.add_argument("--pg-user", default=rich.DEFAULT_PG_USER)
    parser.add_argument("--docker-bin", default="docker")
    parser.add_argument("--base-run", type=Path, default=rich.BASE_RUN)
    parser.add_argument("--mscn-paper-run", type=Path, default=rich.MSCN_PAPER_RUN)
    parser.add_argument("--exact-query-dir", type=Path, default=rich.EVAL_EXACT_QUERY_DIR)
    parser.add_argument("--complex-query-dir", type=Path, default=rich.EVAL_COMPLEX_QUERY_DIR)
    parser.add_argument(
        "--extra-eval-workload",
        action="append",
        default=[],
        help="Additional eval workload loaded from an existing workload.jsonl as NAME=PATH, e.g. job_light=.../workload.jsonl.",
    )
    parser.add_argument(
        "--extra-eval-query-dir",
        action="append",
        default=[],
        help="Additional eval workload loaded from SQL files and BASE_RUN/postgres/NAME/query_summary.csv labels as NAME=PATH.",
    )
    parser.add_argument(
        "--extra-eval-plan-dir",
        action="append",
        default=[],
        help="Raw PostgreSQL EXPLAIN ANALYZE JSON directory for an extra eval workload as NAME=PATH.",
    )
    parser.add_argument(
        "--no-subplan-eval",
        action="store_true",
        help="Disable derived intermediate/subplan q-error workloads.",
    )
    parser.add_argument(
        "--subplan-limit-per-workload",
        type=int,
        default=0,
        help="Debug cap for generated subplan rows per workload. Default 0 means unlimited.",
    )
    parser.add_argument("--exclude-query-dir", action="append", type=Path, default=[])
    parser.add_argument(
        "--train-workload",
        type=Path,
        help="Optional saved workload.jsonl to use as the training SQL corpus instead of generating fresh synthetic SQL.",
    )
    parser.add_argument(
        "--relabel-train-workload",
        action="store_true",
        help="When --train-workload is set, recompute each training label on the target database with COUNT(*).",
    )
    parser.add_argument("--train-queries", type=int, default=100000)
    parser.add_argument("--existing-job-variants", type=int, default=1110)
    parser.add_argument("--query-seed", type=int, default=424242)
    parser.add_argument("--sample-seed", type=int, default=9001)
    parser.add_argument("--model-seeds", default="1,2,3")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hid", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--statement-timeout-ms", type=int, default=8000)
    parser.add_argument(
        "--subplan-count-timeout-ms",
        type=int,
        default=60000,
        help="Timeout for exact COUNT(*) labels of derived plan-node subqueries.",
    )
    parser.add_argument("--strict-coverage", action="store_true")
    args = parser.parse_args()

    set_rich_globals(args)
    model_seeds = parse_model_seeds(args.model_seeds)
    run_dir = RUNS_DIR / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"run_dir": str(run_dir), "model_seeds": model_seeds}), flush=True)

    rich.ensure_container_running(args.docker_bin, args.pg_container)
    session = rich.PsqlSession(docker_bin=args.docker_bin, container=args.pg_container, db_name=args.db_name, user=args.pg_user)
    try:
        eval_rows = rich.load_eval_queries()
        for name, rows in load_extra_eval_query_dirs(args.extra_eval_query_dir, base_run=args.base_run.resolve()).items():
            if name in eval_rows:
                raise ValueError(f"Duplicate eval workload name from --extra-eval-query-dir: {name}")
            eval_rows[name] = rows
        eval_rows.update(load_extra_eval_workloads(args.extra_eval_workload))
        plan_dirs = {
            name: default_raw_plan_dir(args.base_run.resolve(), name)
            for name in eval_rows
        }
        plan_dirs.update(parse_name_path_specs(args.extra_eval_plan_dir, label="--extra-eval-plan-dir"))
        if not args.no_subplan_eval:
            subplan_rows_by_name: dict[str, list[tuple[str, str, int, str]]] = {}
            for name, rows in list(eval_rows.items()):
                raw_plan_dir = plan_dirs.get(name)
                if raw_plan_dir is None or not raw_plan_dir.exists():
                    print(
                        json.dumps(
                            {
                                "warning": "missing_raw_plan_dir_for_subplans",
                                "workload": name,
                                "raw_plan_dir": str(raw_plan_dir) if raw_plan_dir else None,
                            }
                        ),
                        flush=True,
                    )
                    continue
                subplan_rows = generate_plan_subplan_rows(
                    workload_name=name,
                    root_rows=rows,
                    raw_plan_dir=raw_plan_dir,
                    out_dir=run_dir / "subplans" / name,
                    session=session,
                    timeout_ms=args.subplan_count_timeout_ms,
                    limit=max(0, int(args.subplan_limit_per_workload)),
                )
                if subplan_rows:
                    subplan_rows_by_name[f"{name}_nodes"] = subplan_rows
                else:
                    print(
                        json.dumps(
                            {
                                "warning": "no_subplans_generated",
                                "workload": name,
                                "raw_plan_dir": str(raw_plan_dir),
                            }
                        ),
                        flush=True,
                    )
            eval_rows.update(subplan_rows_by_name)
        train_rows = load_training_rows(args=args, run_dir=run_dir, session=session, eval_rows=eval_rows)
        rows_for_tables = {"train": train_rows, **eval_rows}
        sample_tables = create_random_sample_tables(
            session=session,
            tables=collect_tables(rows_for_tables),
            sample_id=args.run_id,
            sample_seed=args.sample_seed,
            num_samples=args.num_samples,
            out_dir=run_dir,
        )
        train_items = write_workload_with_random_samples(
            name="train",
            rows=train_rows,
            out_dir=run_dir / "rich_train",
            session=session,
            sample_tables=sample_tables,
            num_samples=args.num_samples,
        )
        eval_items_by_name: dict[str, list[rich.QueryItem]] = {}
        for name, rows in eval_rows.items():
            eval_items_by_name[name] = write_workload_with_random_samples(
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
    extend_encoder_capacity(encoder, [item for items in loaded_eval_items.values() for item in items])

    coverage = encoder.audit_eval_coverage(loaded_eval_items)
    (run_dir / "coverage_audit.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    has_gaps = any(
        values
        for audit in coverage.values()
        for key, values in audit.items()
        if key.startswith("missing_")
    )
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
        seed_summaries[str(seed)] = train_and_eval_seed(
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

    aggregate = aggregate_seed_summaries(seed_summaries)
    report = {
        "method": "paper_mscn_random_samples_v1",
        "run_dir": str(run_dir),
        "db_name": args.db_name,
        "train_queries": args.train_queries,
        "epochs": args.epochs,
        "num_samples": args.num_samples,
        "query_seed": args.query_seed,
        "sample_seed": args.sample_seed,
        "model_seeds": model_seeds,
        "aggregate": aggregate,
        "seed_summaries": seed_summaries,
    }
    (run_dir / "paper_mscn_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"done": True, "run_dir": str(run_dir), "aggregate": aggregate}, indent=2), flush=True)


if __name__ == "__main__":
    main()
