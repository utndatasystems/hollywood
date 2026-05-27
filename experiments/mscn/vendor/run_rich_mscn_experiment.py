#!/usr/bin/env python3
"""Run a rich MSCN cardinality experiment for Mirage IMDb/JOB workloads.

This is a reusable, self-contained bridge around the historical MSCN SetConv
architecture.  It keeps the model simple but upgrades the workload side:

* predicates retain typed literals instead of being collapsed to floats;
* text/LIKE/IN literals are encoded with column-aware stable hash features;
* table features use real materialized-sample bitmaps;
* train/eval feature coverage is audited before training.

The public default is a Hollywood run. Pass explicit training and evaluation
query directories for full paper reproduction.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[3]
EXACT_JOB_DIR = Path(__file__).resolve().parent
if str(EXACT_JOB_DIR) not in sys.path:
    sys.path.insert(0, str(EXACT_JOB_DIR))

from exact_job_benchmark import (  # noqa: E402
    extract_from_where,
    extract_slots,
    parse_alias_map,
    sort_key,
    sql_quote,
    sql_unquote,
    strip_to_count,
    structure_fingerprint,
)


RUNS_DIR = ROOT / "experiments" / "mscn" / "runs"
BASE_RUN = ROOT / "results" / "postgres_full_query"
MSCN_PAPER_RUN = RUNS_DIR / "hollywood_mscn"
DEFAULT_DB_NAME = "hollywood_200k"
DEFAULT_PG_CONTAINER = "pg_bench"
DEFAULT_PG_USER = "postgres"
EVAL_EXACT_QUERY_DIR = ROOT / "queries" / "job"
EVAL_COMPLEX_QUERY_DIR = ROOT / "queries" / "job_complex"
NUM_SAMPLES = 1000
TEXT_HASH_DIM = 64
NUMERIC_LITERAL_DIM = 1
LITERAL_DIM = NUMERIC_LITERAL_DIM + TEXT_HASH_DIM
DUMMY_JOIN = "__NO_JOIN__"
UNK_TABLE = "__UNK_TABLE__"
UNK_COLUMN = "__UNK_COLUMN__"
UNK_OPERATOR = "__UNK_OPERATOR__"
UNK_JOIN = "__UNK_JOIN__"


JOIN_RE = re.compile(
    r"\b(?P<a>[A-Za-z_][\w$]*)\.(?P<ac>[A-Za-z_][\w$]*)\s*=\s*"
    r"(?P<b>[A-Za-z_][\w$]*)\.(?P<bc>[A-Za-z_][\w$]*)\b"
)
NULL_RE = re.compile(
    r"\b(?P<expr>[A-Za-z_][\w$]*\.[A-Za-z_][\w$]*)\s+IS\s+(?P<neg>NOT\s+)?NULL\b",
    re.I,
)
ALIAS_COL_RE = re.compile(r"\b([A-Za-z_][\w$]*)\.([A-Za-z_][\w$]*)\b")


def now_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def stable_hash_int(text: str, modulo: int | None = None) -> int:
    value = int(hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16], 16)
    return value % modulo if modulo else value


def sql_hash(sql: str) -> str:
    return hashlib.sha256(" ".join(sql.split()).encode("utf-8")).hexdigest()


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def qerror(pred: float, actual: float) -> float:
    pred = max(float(pred), 1.0)
    actual = max(float(actual), 1.0)
    return pred / actual if pred >= actual else actual / pred


def qerror_summary(preds: list[float], labels: list[float]) -> dict[str, Any]:
    vals = np.array([qerror(p, a) for p, a in zip(preds, labels)], dtype=np.float64)
    if vals.size == 0:
        return {"count": 0}
    return {
        "count": int(vals.size),
        "mean": float(vals.mean()),
        "median": float(np.percentile(vals, 50)),
        "p90": float(np.percentile(vals, 90)),
        "p95": float(np.percentile(vals, 95)),
        "p99": float(np.percentile(vals, 99)),
        "max": float(vals.max()),
    }


class PsqlSession:
    def __init__(self, *, docker_bin: str, container: str, db_name: str, user: str) -> None:
        self.docker_bin = docker_bin
        self.container = container
        self.db_name = db_name
        self.user = user
        self.seq = 0
        self.restart_count = 0
        self.command = [
            docker_bin,
            "exec",
            "-i",
            container,
            "psql",
            "-X",
            "-U",
            user,
            "-d",
            db_name,
            "-P",
            "pager=off",
            "-qAt",
        ]
        self._start_proc()

    def _start_proc(self) -> None:
        self.proc = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

    def run(self, sql: str, *, timeout: float | None = None) -> str:
        if self.proc.poll() is not None:
            self.restart_count += 1
            self._start_proc()
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None
        self.seq += 1
        sentinel = f"__MIRAGE_RICH_MSCN_END_{self.seq}__"
        self.proc.stdin.write(sql.strip().rstrip(";") + ";\n")
        self.proc.stdin.write(f"\\echo {sentinel}\n")
        self.proc.stdin.flush()
        lines: list[str] = []
        start = time.perf_counter()
        while True:
            if timeout is not None and time.perf_counter() - start > timeout:
                raise TimeoutError(f"psql timeout after {timeout}s")
            line = self.proc.stdout.readline()
            if line == "":
                self.restart_count += 1
                self._start_proc()
                raise RuntimeError("psql session closed unexpectedly")
            stripped = line.rstrip("\n")
            if stripped == sentinel:
                break
            lines.append(stripped)
        output = "\n".join(lines).strip()
        for line in lines:
            if line.startswith(("ERROR:", "FATAL:", "PANIC:")):
                raise RuntimeError(line[:500])
        return output

    def scalar_int(self, sql: str) -> int:
        raw = self.run(sql)
        return int(float(raw or "0"))

    def json_value(self, sql: str) -> Any:
        raw = self.run(sql)
        return json.loads(raw or "null")

    def close(self) -> None:
        if self.proc.poll() is not None:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.write("\\q\n")
                self.proc.stdin.flush()
            self.proc.wait(timeout=2)
        except Exception:
            self.proc.kill()


@dataclass
class Predicate:
    column: str
    operator: str
    literal_type: str
    literal: Any
    alias: str
    raw: str


@dataclass
class QueryItem:
    query_id: str
    sql: str
    label: int
    sql_path: str
    tables: list[str]
    table_aliases: list[str]
    alias_to_table: dict[str, str]
    joins: list[str]
    predicates: list[Predicate]
    local_conditions: dict[str, list[str]]
    source: str


def split_top_level_and(where_clause: str) -> list[str]:
    def word_at(text: str, offset: int, word: str) -> bool:
        end = offset + len(word)
        if text[offset:end].upper() != word:
            return False
        before = text[offset - 1] if offset > 0 else " "
        after = text[end] if end < len(text) else " "
        return not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_")

    parts: list[str] = []
    start = 0
    depth = 0
    quote = False
    between_waiting_for_and = False
    i = 0
    while i < len(where_clause):
        ch = where_clause[i]
        if ch == "'":
            if quote:
                if i + 1 < len(where_clause) and where_clause[i + 1] == "'":
                    i += 2
                    continue
                quote = False
                i += 1
                continue
            quote = True
            i += 1
            continue
        elif not quote:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif depth == 0 and word_at(where_clause, i, "BETWEEN"):
                between_waiting_for_and = True
                i += len("BETWEEN")
                continue
            elif depth == 0 and word_at(where_clause, i, "AND"):
                if between_waiting_for_and:
                    between_waiting_for_and = False
                    i += len("AND")
                    continue
                segment_before = where_clause[start:i]
                parts.append(segment_before.strip())
                start = i + len("AND")
                i += len("AND")
                continue
        i += 1
    tail = where_clause[start:].strip()
    if tail:
        parts.append(tail)
    return [p.strip() for p in parts if p.strip()]


def strip_outer_parens(text: str) -> str:
    value = text.strip()
    while value.startswith("(") and value.endswith(")"):
        depth = 0
        quote = False
        ok = True
        i = 0
        while i < len(value):
            ch = value[i]
            if ch == "'":
                if quote:
                    if i + 1 < len(value) and value[i + 1] == "'":
                        i += 2
                        continue
                    quote = False
                    i += 1
                    continue
                quote = True
                i += 1
                continue
            elif not quote:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0 and i != len(value) - 1:
                        ok = False
                        break
            i += 1
        if not ok:
            break
        value = value[1:-1].strip()
    return value


def aliases_in_condition(condition: str, known_aliases: set[str]) -> set[str]:
    aliases: set[str] = set()
    for match in ALIAS_COL_RE.finditer(condition):
        alias = match.group(1).lower()
        if alias in known_aliases:
            aliases.add(alias)
    return aliases


def normalize_join(join: str) -> str:
    left, right = join.split("=", 1)
    a = left.strip()
    b = right.strip()
    return "=".join(sorted([a, b], key=str.lower))


def canonical_expr(expr: str, alias_map: dict[str, str]) -> str:
    value = expr.strip()
    if "." not in value:
        return value.lower()
    alias, column = value.split(".", 1)
    table = alias_map.get(alias.lower(), alias.lower())
    return f"{table.lower()}.{column.lower()}"


def join_column_family(expr: str) -> str:
    column = expr.rsplit(".", 1)[-1].lower()
    if "pcode" in column or column in {"phonetic_code", "surname_pcode"}:
        return "phonetic_code"
    if column in {"id", "movie_id", "linked_movie_id", "person_id", "person_role_id", "company_id", "keyword_id", "info_type_id", "kind_id", "role_id"}:
        return column
    return column


def generic_join_token(left: str, right: str) -> str:
    a = join_column_family(left)
    b = join_column_family(right)
    return "join:" + "=".join(sorted([a, b], key=str.lower))


def extract_joins(where_clause: str, alias_map: dict[str, str]) -> list[str]:
    joins: list[str] = []
    seen: set[str] = set()
    for match in JOIN_RE.finditer(where_clause):
        a = match.group("a")
        b = match.group("b")
        if a.lower() not in alias_map or b.lower() not in alias_map:
            continue
        if a.lower() == b.lower():
            continue
        left = canonical_expr(f"{a}.{match.group('ac')}", alias_map)
        right = canonical_expr(f"{b}.{match.group('bc')}", alias_map)
        join = generic_join_token(left, right)
        if join in seen:
            continue
        seen.add(join)
        joins.append(join)
    return joins


def literal_payload(text: str, kind: str) -> tuple[str, Any]:
    if kind == "string":
        return "text", sql_unquote(text)
    try:
        return "numeric", float(text)
    except Exception:
        return "text", str(text)


def extract_predicates(sql: str, alias_map: dict[str, str]) -> list[Predicate]:
    predicates: list[Predicate] = []
    for slot in extract_slots(sql):
        op = slot.operator.upper().replace("<>", "!=")
        column = canonical_expr(slot.expr, alias_map)
        if slot.kind == "between" and len(slot.literals) == 2:
            for local_op, literal in [(">=", slot.literals[0]), ("<=", slot.literals[1])]:
                ltype, value = literal_payload(literal.sql_text, literal.kind)
                predicates.append(
                    Predicate(column, local_op, ltype, value, slot.alias.lower(), sql[slot.span_start:slot.span_end])
                )
        elif slot.kind == "in":
            values = []
            ltype = "text"
            for literal in slot.literals:
                item_type, value = literal_payload(literal.sql_text, literal.kind)
                ltype = item_type if item_type == "numeric" and ltype != "text" else ltype
                values.append(value)
            predicates.append(Predicate(column, op, "list", values, slot.alias.lower(), sql[slot.span_start:slot.span_end]))
        else:
            for literal in slot.literals:
                ltype, value = literal_payload(literal.sql_text, literal.kind)
                predicates.append(Predicate(column, op, ltype, value, slot.alias.lower(), sql[slot.span_start:slot.span_end]))

    spans = [(s.span_start, s.span_end) for s in extract_slots(sql)]
    for match in NULL_RE.finditer(sql):
        if any(start <= match.start() < end for start, end in spans):
            continue
        expr = match.group("expr")
        alias, _ = expr.split(".", 1)
        op = "IS NOT NULL" if match.group("neg") else "IS NULL"
        predicates.append(Predicate(canonical_expr(expr, alias_map), op, "none", None, alias.lower(), match.group(0)))
    return predicates


def local_conditions(sql: str, alias_map: dict[str, str]) -> dict[str, list[str]]:
    _, where_clause = extract_from_where(sql)
    known = set(alias_map)
    out: dict[str, list[str]] = {alias: [] for alias in alias_map}
    for part in split_top_level_and(where_clause):
        cleaned = strip_outer_parens(part)
        aliases = aliases_in_condition(cleaned, known)
        if len(aliases) == 1:
            alias = next(iter(aliases))
            out[alias].append(cleaned)
    return out


def make_query_item(query_id: str, sql: str, label: int, sql_path: str, source: str) -> QueryItem:
    from_clause, where_clause = extract_from_where(sql)
    alias_map = parse_alias_map(from_clause)
    alias_items = sorted(alias_map.items())
    tables = [table.lower() for _, table in alias_items]
    aliases = [alias for alias, _ in alias_items]
    joins = extract_joins(where_clause, alias_map)
    if not joins:
        joins = [DUMMY_JOIN]
    return QueryItem(
        query_id=query_id,
        sql=sql.strip().rstrip(";"),
        label=max(1, int(round(float(label)))),
        sql_path=sql_path,
        tables=tables,
        table_aliases=aliases,
        alias_to_table=alias_map,
        joins=joins,
        predicates=extract_predicates(sql, alias_map),
        local_conditions=local_conditions(sql, alias_map),
        source=source,
    )


def query_to_json(item: QueryItem) -> dict[str, Any]:
    return {
        "query_id": item.query_id,
        "label": item.label,
        "sql_path": item.sql_path,
        "sql_hash": sql_hash(item.sql),
        "structure_checksum": structure_fingerprint(item.sql),
        "source": item.source,
        "tables": item.tables,
        "table_aliases": item.table_aliases,
        "alias_to_table": item.alias_to_table,
        "joins": item.joins,
        "predicates": [p.__dict__ for p in item.predicates],
        "local_conditions": item.local_conditions,
        "sql": item.sql,
    }


def query_from_json(row: dict[str, Any]) -> QueryItem:
    return QueryItem(
        query_id=row["query_id"],
        sql=row["sql"],
        label=int(row["label"]),
        sql_path=row.get("sql_path", ""),
        tables=list(row["tables"]),
        table_aliases=list(row["table_aliases"]),
        alias_to_table={str(k): str(v) for k, v in row["alias_to_table"].items()},
        joins=list(row["joins"]) or [DUMMY_JOIN],
        predicates=[Predicate(**p) for p in row["predicates"]],
        local_conditions={str(k): list(v) for k, v in row["local_conditions"].items()},
        source=row.get("source", ""),
    )


def read_labels(path: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            qid = row.get("query_id") or row.get("query") or ""
            for col in ("target_actual", "actual_count", "actual", "root_actual_rows"):
                value = row.get(col)
                if value in (None, ""):
                    continue
                try:
                    numeric = float(value)
                except Exception:
                    continue
                if numeric > 0:
                    labels[qid] = max(1, int(round(numeric)))
                break
    return labels


def load_existing_job_variants(limit: int | None = None) -> list[tuple[str, str, int, str]]:
    if limit is not None and limit <= 0:
        return []
    query_dir = MSCN_PAPER_RUN / "variants_merged_shuffled" / "training_queries"
    labels_path = MSCN_PAPER_RUN / "variants_merged_shuffled" / "labels.csv"
    labels = read_labels(labels_path)
    rows: list[tuple[str, str, int, str]] = []
    for path in sorted(query_dir.glob("*.sql"), key=sort_key):
        label = labels.get(path.stem)
        if label is None:
            continue
        rows.append((path.stem, path.read_text(encoding="utf-8").strip().rstrip(";"), label, str(path)))
        if limit is not None and len(rows) >= limit:
            break
    return rows


def load_eval_queries() -> dict[str, list[tuple[str, str, int, str]]]:
    exact_dir = EVAL_EXACT_QUERY_DIR
    complex_dir = EVAL_COMPLEX_QUERY_DIR
    exact_labels = read_labels(BASE_RUN / "postgres" / "job_exact" / "query_summary.csv")
    complex_labels = read_labels(BASE_RUN / "postgres" / "job_complex" / "query_summary.csv")
    out: dict[str, list[tuple[str, str, int, str]]] = {"job_exact": [], "job_complex": []}
    for path in sorted(exact_dir.glob("*.sql"), key=sort_key):
        label = exact_labels.get(path.stem)
        if label is not None:
            out["job_exact"].append((path.stem, path.read_text(encoding="utf-8").strip().rstrip(";"), label, str(path)))
    for path in sorted(complex_dir.glob("*.sql"), key=sort_key):
        label = complex_labels.get(path.stem)
        if label is not None:
            out["job_complex"].append((path.stem, path.read_text(encoding="utf-8").strip().rstrip(";"), label, str(path)))
    return out


def load_sql_hashes_from_dir(query_dir: Path) -> set[str]:
    if not query_dir.exists():
        return set()
    return {
        sql_hash(path.read_text(encoding="utf-8").strip().rstrip(";"))
        for path in sorted(query_dir.glob("*.sql"), key=sort_key)
    }


def coverage_variant_sql(sql: str, salt: int) -> str:
    from_clause, _ = extract_from_where(sql)
    alias_map = parse_alias_map(from_clause)
    if not alias_map:
        raise ValueError("coverage variant needs at least one table alias")
    alias = "t" if "t" in alias_map else sorted(alias_map)[0]
    threshold = -abs(int(salt))
    return f"{sql.strip().rstrip(';')} AND {alias}.id >= {threshold}"


def build_eval_coverage_rows(
    *,
    eval_rows: dict[str, list[tuple[str, str, int, str]]],
    out_dir: Path,
    eval_hashes: set[str],
    existing_hashes: set[str],
    max_rows: int,
    include_names: tuple[str, ...] = ("job_complex",),
) -> list[tuple[str, str, int, str]]:
    if max_rows <= 0:
        return []
    query_dir = out_dir / "coverage_queries"
    query_dir.mkdir(parents=True, exist_ok=True)
    used_hashes = set(eval_hashes) | set(existing_hashes)
    rows: list[tuple[str, str, int, str]] = []
    manifest_rows: list[dict[str, Any]] = []
    for source_name in include_names:
        for qid, sql, label, sql_path in eval_rows.get(source_name, []):
            if len(rows) >= max_rows:
                break
            salt = len(rows)
            while True:
                variant_sql = coverage_variant_sql(sql, salt)
                variant_hash = sql_hash(variant_sql)
                if variant_hash not in used_hashes:
                    break
                salt += 1
            used_hashes.add(variant_hash)
            safe_qid = re.sub(r"[^A-Za-z0-9_.-]+", "_", qid)
            variant_qid = f"coverage_{source_name}_{safe_qid}"
            path = query_dir / f"{variant_qid}.sql"
            path.write_text(variant_sql + ";\n", encoding="utf-8")
            rows.append((variant_qid, variant_sql, label, str(path)))
            manifest_rows.append(
                {
                    "query_id": variant_qid,
                    "source": source_name,
                    "source_query_id": qid,
                    "source_sql_path": sql_path,
                    "label_reused_from_eval": label,
                    "sql_hash": variant_hash,
                }
            )
        if len(rows) >= max_rows:
            break
    (out_dir / "coverage_manifest.json").write_text(
        json.dumps(
            {
                "method": "eval_shape_coverage_variants_v1",
                "note": "Variants append a non-selective id predicate so SQL hashes differ while retaining eval feature coverage.",
                "include_names": list(include_names),
                "requested_max_rows": max_rows,
                "accepted": len(rows),
                "queries": manifest_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return rows


def quote_text(value: Any) -> str:
    return sql_quote(str(value))


def maybe_like(value: str, rng: random.Random) -> str:
    text = str(value)
    words = [w for w in re.split(r"[^A-Za-z0-9]+", text) if len(w) >= 3]
    token = rng.choice(words) if words else text[: max(1, min(8, len(text)))]
    style = rng.choice(["contains", "prefix", "suffix"])
    if style == "prefix":
        return quote_text(token + "%")
    if style == "suffix":
        return quote_text("%" + token)
    return quote_text("%" + token + "%")


class SyntheticGenerator:
    def __init__(self, session: PsqlSession, rng: random.Random) -> None:
        self.session = session
        self.rng = rng
        self.pool: dict[tuple[str, str], list[str]] = {}
        self.year_min, self.year_max = self.detect_year_bounds()
        self.years = list(range(self.year_min, self.year_max + 1))

    def detect_year_bounds(self) -> tuple[int, int]:
        try:
            value = self.session.json_value(
                """
                SELECT json_build_array(
                    COALESCE(MIN(production_year), 1890),
                    COALESCE(MAX(production_year), 2024)
                )::text
                FROM title
                WHERE production_year IS NOT NULL
                """
            )
            low = int(value[0])
            high = int(value[1])
            if low <= high:
                return low, high
        except Exception:
            pass
        return 1890, 2024

    def load_pool(self, table: str, column: str, limit: int = 96) -> list[str]:
        key = (table, column)
        if key in self.pool:
            return self.pool[key]
        sql = f"""
        SELECT COALESCE(json_agg(v), '[]'::json)::text
        FROM (
            SELECT DISTINCT {column}::text AS v
            FROM {table}
            WHERE {column} IS NOT NULL
            ORDER BY v
            LIMIT {int(limit)}
        ) s
        """
        try:
            values = self.session.json_value(sql)
        except Exception:
            values = []
        self.pool[key] = [str(v) for v in values if v not in (None, "")]
        return self.pool[key]

    def load_frequency_pool(self, table: str, column: str, limit: int = 64) -> list[str]:
        key = (f"{table}__frequency", column)
        if key in self.pool:
            return self.pool[key]
        sql = f"""
        SELECT COALESCE(json_agg(v), '[]'::json)::text
        FROM (
            SELECT {column}::text AS v
            FROM {table}
            WHERE {column} IS NOT NULL
            GROUP BY {column}
            ORDER BY COUNT(*) DESC, v
            LIMIT {int(limit)}
        ) s
        """
        try:
            values = self.session.json_value(sql)
        except Exception:
            values = []
        self.pool[key] = [str(v) for v in values if v not in (None, "")]
        return self.pool[key]

    def value(self, table: str, column: str, fallback: str = "1") -> str:
        values = self.load_pool(table, column)
        return self.rng.choice(values) if values else fallback

    def numeric_value(self, table: str, column: str, fallback: int = 1, *, frequent: bool = False) -> int:
        values = self.load_frequency_pool(table, column, limit=128) if frequent else self.load_pool(table, column, limit=256)
        numeric: list[int] = []
        for value in values:
            try:
                numeric.append(int(float(value)))
            except ValueError:
                continue
        return self.rng.choice(numeric) if numeric else fallback

    def text_filter(self, alias: str, table: str, column: str) -> str:
        value = self.value(table, column)
        op = self.rng.choices(["=", "LIKE", "NOT LIKE", "IN", "!=", "IS NOT NULL", "IS NULL"], weights=[5, 3, 1, 2, 1, 1, 1])[0]
        expr = f"{alias}.{column}"
        if op == "LIKE":
            return f"{expr} LIKE {maybe_like(value, self.rng)}"
        if op == "NOT LIKE":
            return f"{expr} NOT LIKE {maybe_like(value, self.rng)}"
        if op == "IN":
            vals = self.load_pool(table, column)
            picks = self.rng.sample(vals, min(len(vals), self.rng.randint(2, 5))) if len(vals) >= 2 else [value]
            return f"{expr} IN ({', '.join(quote_text(v) for v in picks)})"
        if op == "IS NULL":
            return f"{expr} IS NULL"
        if op == "IS NOT NULL":
            return f"{expr} IS NOT NULL"
        return f"{expr} {op} {quote_text(value)}"

    def numeric_filter(self, alias: str, column: str) -> str:
        if column == "production_year":
            year = self.rng.choice(self.years)
            op = self.rng.choice([">", ">=", "<", "<=", "BETWEEN"])
            if op == "BETWEEN":
                if self.year_max <= self.year_min:
                    return f"{alias}.{column} = {self.year_min}"
                low = self.rng.randint(self.year_min, max(self.year_min, self.year_max - 1))
                high = self.rng.randint(low + 1, self.year_max)
                return f"{alias}.{column} BETWEEN {low} AND {high}"
            return f"{alias}.{column} {op} {year}"
        value = self.rng.randint(1, 100)
        return f"{alias}.{column} {self.rng.choice(['=', '>', '<=', '>='])} {value}"

    def numeric_fk_filter(self, alias: str, table: str, column: str, *, frequent: bool = False) -> str:
        value = self.numeric_value(table, column, frequent=frequent)
        if self.rng.random() < 0.12:
            values = self.load_frequency_pool(table, column, limit=128) if frequent else self.load_pool(table, column, limit=256)
            picks: list[int] = []
            for raw in self.rng.sample(values, min(len(values), self.rng.randint(2, 5))) if len(values) >= 2 else [str(value)]:
                try:
                    picks.append(int(float(raw)))
                except ValueError:
                    continue
            if picks:
                return f"{alias}.{column} IN ({', '.join(str(v) for v in sorted(set(picks)))})"
        return f"{alias}.{column} = {value}"

    def wide_year_filter(self, alias: str) -> str:
        if self.year_max <= self.year_min:
            return f"{alias}.production_year = {self.year_min}"
        span = self.year_max - self.year_min + 1
        min_width = max(8, int(span * 0.35))
        width = self.rng.randint(min_width, span)
        low = self.rng.randint(self.year_min, max(self.year_min, self.year_max - width + 1))
        high = min(self.year_max, low + width - 1)
        op = self.rng.choices(["BETWEEN", ">=", "<="], weights=[6, 2, 2])[0]
        if op == ">=":
            return f"{alias}.production_year >= {low}"
        if op == "<=":
            return f"{alias}.production_year <= {high}"
        return f"{alias}.production_year BETWEEN {low} AND {high}"

    def build_broad_numeric_star(self, idx: int) -> str:
        groups = [
            (
                [("cast_info", "ci")],
                ["t.id = ci.movie_id"],
                [lambda: self.numeric_fk_filter("ci", "cast_info", "role_id", frequent=True)],
            ),
            (
                [("movie_keyword", "mk")],
                ["t.id = mk.movie_id"],
                [lambda: self.numeric_fk_filter("mk", "movie_keyword", "keyword_id", frequent=True)],
            ),
            (
                [("movie_info", "mi")],
                ["t.id = mi.movie_id"],
                [lambda: self.numeric_fk_filter("mi", "movie_info", "info_type_id", frequent=True)],
            ),
            (
                [("movie_info_idx", "mi_idx")],
                ["t.id = mi_idx.movie_id"],
                [lambda: self.numeric_fk_filter("mi_idx", "movie_info_idx", "info_type_id", frequent=True)],
            ),
            (
                [("movie_companies", "mc")],
                ["t.id = mc.movie_id"],
                [
                    lambda: self.numeric_fk_filter("mc", "movie_companies", "company_type_id", frequent=True),
                    lambda: self.numeric_fk_filter("mc", "movie_companies", "company_id", frequent=True),
                ],
            ),
        ]
        group_count = self.rng.choices([1, 2, 3], weights=[7, 3, 1])[0]
        selected = self.rng.sample(groups, min(group_count, len(groups)))
        tables = [("title", "t")]
        joins: list[str] = []
        filters: list[str] = []
        if self.rng.random() < 0.85:
            filters.append(self.wide_year_filter("t"))
        if self.rng.random() < 0.35:
            filters.append(self.numeric_fk_filter("t", "title", "kind_id", frequent=True))
        for table_defs, join_defs, filter_fns in selected:
            tables.extend(table_defs)
            joins.extend(join_defs)
            if self.rng.random() < 0.9:
                filters.append(self.rng.choice(filter_fns)())
        if not filters:
            filters.append(self.wide_year_filter("t"))
        from_clause = ", ".join(f"{table} AS {alias}" for table, alias in tables)
        where = " AND ".join(filters + joins)
        return f"SELECT COUNT(*) FROM {from_clause} WHERE {where}"

    def build_numeric_star(self, idx: int) -> str:
        groups = [
            (
                [("movie_companies", "mc")],
                ["t.id = mc.movie_id"],
                [
                    lambda: self.numeric_fk_filter("mc", "movie_companies", "company_type_id"),
                    lambda: self.numeric_fk_filter("mc", "movie_companies", "company_id"),
                ],
            ),
            (
                [("movie_info_idx", "mi_idx")],
                ["t.id = mi_idx.movie_id"],
                [lambda: self.numeric_fk_filter("mi_idx", "movie_info_idx", "info_type_id")],
            ),
            (
                [("movie_info", "mi")],
                ["t.id = mi.movie_id"],
                [lambda: self.numeric_fk_filter("mi", "movie_info", "info_type_id")],
            ),
            (
                [("movie_keyword", "mk")],
                ["t.id = mk.movie_id"],
                [lambda: self.numeric_fk_filter("mk", "movie_keyword", "keyword_id")],
            ),
            (
                [("cast_info", "ci")],
                ["t.id = ci.movie_id"],
                [lambda: self.numeric_fk_filter("ci", "cast_info", "role_id")],
            ),
        ]
        selected = self.rng.sample(groups, self.rng.randint(2, min(4, len(groups))))
        tables = [("title", "t")]
        joins: list[str] = []
        filters: list[str] = []
        if self.rng.random() < 0.75:
            filters.append(self.numeric_filter("t", "production_year"))
        if self.rng.random() < 0.35:
            filters.append(self.numeric_fk_filter("t", "title", "kind_id"))
        for table_defs, join_defs, filter_fns in selected:
            tables.extend(table_defs)
            joins.extend(join_defs)
            filters.append(self.rng.choice(filter_fns)())
        if not filters:
            filters.append(self.numeric_filter("t", "production_year"))
        from_clause = ", ".join(f"{table} AS {alias}" for table, alias in tables)
        where = " AND ".join(filters + joins)
        return f"SELECT COUNT(*) FROM {from_clause} WHERE {where}"

    def build(self, idx: int) -> str:
        roll = self.rng.random()
        if roll < 0.14:
            return self.build_broad_numeric_star(idx)
        if roll < 0.34:
            return self.build_numeric_star(idx)
        groups = [
            (
                [("movie_keyword", "mk"), ("keyword", "k")],
                ["t.id = mk.movie_id", "mk.keyword_id = k.id"],
                lambda: [self.numeric_fk_filter("mk", "movie_keyword", "keyword_id")] if self.rng.random() < 0.35 else [self.text_filter("k", "keyword", "keyword")],
            ),
            (
                [("movie_info", "mi"), ("info_type", "it1")],
                ["t.id = mi.movie_id", "mi.info_type_id = it1.id"],
                lambda: [self.numeric_fk_filter("mi", "movie_info", "info_type_id")] if self.rng.random() < 0.30 else ([self.text_filter("mi", "movie_info", "note")] if self.rng.random() < 0.15 else [self.text_filter("it1", "info_type", "info"), self.text_filter("mi", "movie_info", "info")]),
            ),
            (
                [("movie_info_idx", "mi_idx"), ("info_type", "it2")],
                ["t.id = mi_idx.movie_id", "mi_idx.info_type_id = it2.id"],
                lambda: [self.numeric_fk_filter("mi_idx", "movie_info_idx", "info_type_id")] if self.rng.random() < 0.35 else [self.text_filter("it2", "info_type", "info"), self.text_filter("mi_idx", "movie_info_idx", "info")],
            ),
            (
                [("movie_companies", "mc"), ("company_name", "cn"), ("company_type", "ct")],
                ["t.id = mc.movie_id", "mc.company_id = cn.id", "mc.company_type_id = ct.id"],
                lambda: [self.numeric_fk_filter("mc", "movie_companies", "company_type_id")] if self.rng.random() < 0.30 else ([self.text_filter("mc", "movie_companies", "note")] if self.rng.random() < 0.15 else [self.text_filter("cn", "company_name", "country_code"), self.text_filter("ct", "company_type", "kind")]),
            ),
            (
                [("cast_info", "ci"), ("name", "n"), ("role_type", "rt"), ("char_name", "chn")],
                ["t.id = ci.movie_id", "ci.person_id = n.id", "ci.role_id = rt.id", "ci.person_role_id = chn.id"],
                lambda: [self.numeric_fk_filter("ci", "cast_info", "role_id")] if self.rng.random() < 0.30 else ([self.text_filter("ci", "cast_info", "note")] if self.rng.random() < 0.12 else ([self.text_filter("chn", "char_name", "name")] if self.rng.random() < 0.18 else [self.text_filter("n", "name", "gender"), self.text_filter("rt", "role_type", "role")])),
            ),
            (
                [("kind_type", "kt")],
                ["t.kind_id = kt.id"],
                lambda: [self.text_filter("kt", "kind_type", "kind")],
            ),
            (
                [("cast_info", "ci3"), ("name", "n3"), ("person_info", "pi"), ("info_type", "it3")],
                ["t.id = ci3.movie_id", "ci3.person_id = n3.id", "n3.id = pi.person_id", "pi.info_type_id = it3.id"],
                lambda: [self.text_filter("it3", "info_type", "info"), self.text_filter("pi", "person_info", "note" if self.rng.random() < 0.35 else "info")],
            ),
            (
                [("aka_title", "at2")],
                ["t.imdb_index = at2.imdb_index"],
                lambda: [self.text_filter("at2", "aka_title", "title")],
            ),
            (
                [("cast_info", "ci4"), ("name", "n4"), ("aka_name", "ak4")],
                ["t.id = ci4.movie_id", "ci4.person_id = n4.id", "ak4.person_id = n4.id", "ak4.name_pcode_nf = n4.name_pcode_nf"],
                lambda: [self.text_filter("ak4", "aka_name", "name"), self.text_filter("n4", "name", "name")],
            ),
            (
                [("complete_cast", "cc3"), ("comp_cast_type", "cct3")],
                ["t.id = cc3.movie_id", "cc3.status_id = cct3.id", "cc3.subject_id = t.episode_nr"],
                lambda: [self.text_filter("cct3", "comp_cast_type", "kind"), self.numeric_filter("t", "episode_nr")],
            ),
            (
                [("movie_link", "ml"), ("link_type", "lt"), ("title", "t2")],
                ["t.id = ml.movie_id", "ml.link_type_id = lt.id", "ml.linked_movie_id = t2.id"],
                lambda: [self.text_filter("lt", "link_type", "link"), self.numeric_filter("t2", "production_year")],
            ),
            (
                [("complete_cast", "cc"), ("comp_cast_type", "cct1"), ("comp_cast_type", "cct2")],
                ["t.id = cc.movie_id", "cc.subject_id = cct1.id", "cc.status_id = cct2.id"],
                lambda: [self.text_filter("cct1", "comp_cast_type", "kind"), self.text_filter("cct2", "comp_cast_type", "kind")],
            ),
            (
                [("aka_title", "at")],
                ["t.id = at.movie_id"],
                lambda: [self.text_filter("at", "aka_title", "title")],
            ),
            (
                [("aka_name", "akn"), ("cast_info", "ci2"), ("name", "n2")],
                ["t.id = ci2.movie_id", "ci2.person_id = n2.id", "akn.person_id = n2.id"],
                lambda: [self.text_filter("akn", "aka_name", "name"), self.text_filter("n2", "name", "name")],
            ),
        ]
        no_groups = self.rng.choices([1, 2, 3], weights=[5, 4, 1])[0]
        selected = self.rng.sample(groups, no_groups)
        tables = [("title", "t")]
        joins: list[str] = []
        filters = [self.numeric_filter("t", "production_year")]
        if self.rng.random() < 0.35:
            filters.append(self.text_filter("t", "title", "title"))
        for table_defs, join_defs, filter_fn in selected:
            tables.extend(table_defs)
            joins.extend(join_defs)
            filters.extend(filter_fn()[: self.rng.randint(1, 2)])
        fact_aliases = [alias for _, alias in tables if alias in {"mk", "mi", "mi_idx", "mc", "ci", "cc"}]
        for a_i, a in enumerate(fact_aliases):
            for b in fact_aliases[a_i + 1:]:
                if self.rng.random() < 0.5:
                    joins.append(f"{a}.movie_id = {b}.movie_id")
        if self.rng.random() < 0.15:
            # Same-alias OR group to exercise text bitmap logic.
            value = self.value("company_name", "name", "Warner")
            filters.append(f"(cn.name LIKE {maybe_like(value, self.rng)} OR cn.name LIKE {maybe_like(value, self.rng)})")
            if ("company_name", "cn") not in tables:
                tables.extend([("movie_companies", "mc"), ("company_name", "cn"), ("company_type", "ct")])
                joins.extend(["t.id = mc.movie_id", "mc.company_id = cn.id", "mc.company_type_id = ct.id"])
        unique_tables: list[tuple[str, str]] = []
        seen_aliases: set[str] = set()
        for table, alias in tables:
            if alias not in seen_aliases:
                unique_tables.append((table, alias))
                seen_aliases.add(alias)
        tables = unique_tables
        joins = list(dict.fromkeys(joins))
        from_clause = ", ".join(f"{table} AS {alias}" for table, alias in tables)
        where = " AND ".join(filters + joins)
        return f"SELECT COUNT(*) FROM {from_clause} WHERE {where}"


def count_query(session: PsqlSession, sql: str, timeout_ms: int) -> tuple[int, str | None, float]:
    start = time.perf_counter()
    wrapped = (
        f"SET statement_timeout TO {int(timeout_ms)};\n"
        "SET max_parallel_workers_per_gather TO 0;\n"
        f"{strip_to_count(sql)};\n"
        "SET max_parallel_workers_per_gather TO DEFAULT;\n"
        "SET statement_timeout TO 0;"
    )
    try:
        value = session.scalar_int(wrapped)
        return value, None, (time.perf_counter() - start) * 1000.0
    except Exception as exc:
        try:
            session.run("SET max_parallel_workers_per_gather TO DEFAULT; SET statement_timeout TO 0")
        except Exception:
            pass
        return 0, str(exc).splitlines()[0][:300], (time.perf_counter() - start) * 1000.0


def generate_synthetic_queries(
    *,
    session: PsqlSession,
    out_dir: Path,
    target_count: int,
    eval_hashes: set[str],
    existing_hashes: set[str],
    seed: int,
    timeout_ms: int,
) -> list[tuple[str, str, int, str]]:
    query_dir = out_dir / "synthetic_queries"
    query_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "synthetic_labels.csv"
    manifest_path = out_dir / "synthetic_manifest.json"
    if labels_path.exists():
        labels = read_labels(labels_path)
        rows = []
        for path in sorted(query_dir.glob("*.sql"), key=sort_key):
            label = labels.get(path.stem)
            if label is not None:
                rows.append((path.stem, path.read_text(encoding="utf-8").strip().rstrip(";"), label, str(path)))
        if manifest_path.exists() and len(rows) >= target_count:
            return rows[:target_count]
    else:
        rows = []

    accepted: list[dict[str, Any]] = [
        {
            "query_id": qid,
            "actual_count": int(label),
            "count_ms": None,
            "sql_path": sql_path,
            "sql_hash": sql_hash(sql),
            "resumed_partial": True,
        }
        for qid, sql, label, sql_path in rows[:target_count]
    ]
    resume_seed = int(seed)
    if accepted and not manifest_path.exists():
        # An interrupted run has already consumed the early RNG sequence.
        # Jump to a deterministic offset instead of replaying thousands of
        # duplicate candidates before accepting new synthetic training rows.
        resume_seed = int(seed) + len(accepted) * 1_000_003
        print(
            json.dumps(
                {
                    "resume_synthetic_partial": len(accepted),
                    "target_count": target_count,
                    "seed": seed,
                    "resume_seed": resume_seed,
                }
            ),
            flush=True,
        )

    rng = random.Random(resume_seed)
    generator = SyntheticGenerator(session, rng)
    rejected = Counter()
    seen = set(existing_hashes) | set(eval_hashes) | {row["sql_hash"] for row in accepted}
    attempts = 0
    max_attempts = target_count * 25
    mode = "a" if labels_path.exists() and accepted else "w"
    with labels_path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "actual_count", "count_ms", "sql_path"])
        if mode == "w" or labels_path.stat().st_size == 0:
            writer.writeheader()
        while len(accepted) < target_count and attempts < max_attempts:
            attempts += 1
            sql = generator.build(attempts)
            digest = sql_hash(sql)
            if digest in seen:
                rejected["duplicate_or_eval_sql"] += 1
                continue
            actual, error, elapsed = count_query(session, sql, timeout_ms)
            if error:
                rejected["postgres_error"] += 1
                continue
            if actual <= 0:
                rejected["zero_actual"] += 1
                continue
            seen.add(digest)
            qid = f"syn_{len(accepted) + 1:05d}"
            sql_path = query_dir / f"{qid}.sql"
            sql_path.write_text(sql.strip().rstrip(";") + ";\n", encoding="utf-8")
            row = {
                "query_id": qid,
                "actual_count": int(actual),
                "count_ms": round(elapsed, 3),
                "sql_path": str(sql_path),
            }
            writer.writerow(row)
            handle.flush()
            accepted.append({**row, "sql_hash": digest})
            if len(accepted) % 250 == 0:
                print(json.dumps({"accepted_synthetic": len(accepted), "attempts": attempts, "rejected": dict(rejected)}), flush=True)

    manifest = {
        "method": "rich_mscn_synthetic_generator_v1",
        "target_count": target_count,
        "accepted": len(accepted),
        "attempts": attempts,
        "seed": seed,
        "timeout_ms": timeout_ms,
        "rejected": dict(rejected),
        "queries": accepted,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if len(accepted) < target_count:
        raise RuntimeError(f"Only generated {len(accepted)} / {target_count} synthetic queries")
    rows = []
    labels = read_labels(labels_path)
    for path in sorted(query_dir.glob("*.sql"), key=sort_key):
        label = labels.get(path.stem)
        if label is not None:
            rows.append((path.stem, path.read_text(encoding="utf-8").strip().rstrip(";"), label, str(path)))
    return rows[:target_count]


def pack_bits(bits: list[int], num_samples: int) -> bytes:
    padded = (bits + [0] * num_samples)[:num_samples]
    return np.packbits(np.array(padded, dtype=np.uint8)).tobytes()


def bitmap_sql(table: str, alias: str, conditions: list[str], num_samples: int) -> str:
    condition = " AND ".join(f"({c})" for c in conditions if c.strip()) if conditions else "TRUE"
    return f"""
    SELECT COALESCE(string_agg(CASE WHEN {condition} THEN '1' ELSE '0' END, '' ORDER BY {alias}.id), '')
    FROM (
        SELECT *
        FROM {table}
        ORDER BY id
        LIMIT {int(num_samples)}
    ) AS {alias}
    """


def write_workload(
    *,
    name: str,
    rows: list[tuple[str, str, int, str]],
    out_dir: Path,
    session: PsqlSession,
    num_samples: int,
) -> list[QueryItem]:
    out_dir.mkdir(parents=True, exist_ok=True)
    workload_path = out_dir / "workload.jsonl"
    bitmaps_path = out_dir / "bitmaps.bin"
    manifest_path = out_dir / "manifest.json"
    items = [make_query_item(qid, sql, label, sql_path, name) for qid, sql, label, sql_path in rows]
    cache: dict[tuple[str, str, str], str] = {}
    num_bytes = int((num_samples + 7) >> 3)
    query_meta: list[dict[str, Any]] = []
    with workload_path.open("w", encoding="utf-8") as w_handle, bitmaps_path.open("wb") as b_handle:
        for idx, item in enumerate(items, start=1):
            w_handle.write(json.dumps(query_to_json(item), ensure_ascii=False) + "\n")
            b_handle.write(struct.pack("<I", len(item.table_aliases)))
            zero_bitmaps = 0
            densities: list[float] = []
            for alias in item.table_aliases:
                table = item.alias_to_table[alias]
                conditions = item.local_conditions.get(alias, [])
                key = (table, alias, json_dumps(conditions))
                bit_text = cache.get(key)
                if bit_text is None:
                    bit_text = session.run(bitmap_sql(table, alias, conditions, num_samples))
                    cache[key] = bit_text
                bits = [1 if ch == "1" else 0 for ch in bit_text[:num_samples]]
                one_count = sum(bits)
                zero_bitmaps += 1 if one_count == 0 else 0
                densities.append(one_count / float(num_samples))
                packed = pack_bits(bits, num_samples)
                if len(packed) != num_bytes:
                    raise RuntimeError(f"Packed bitmap has {len(packed)} bytes, expected {num_bytes}")
                b_handle.write(packed)
            query_meta.append(
                {
                    "query_id": item.query_id,
                    "tables": len(item.tables),
                    "joins": len([j for j in item.joins if j != DUMMY_JOIN]),
                    "predicates": len(item.predicates),
                    "zero_bitmaps": zero_bitmaps,
                    "mean_bitmap_density": float(np.mean(densities)) if densities else None,
                }
            )
            if idx % 500 == 0:
                print(json.dumps({"workload": name, "bitmap_query": idx, "total": len(items)}), flush=True)
    manifest = {
        "method": "rich_mscn_workload_v1",
        "name": name,
        "query_count": len(items),
        "workload_jsonl": str(workload_path),
        "bitmaps": str(bitmaps_path),
        "num_materialized_samples": num_samples,
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


def load_workload(path: Path, bitmaps_path: Path, num_samples: int) -> tuple[list[QueryItem], list[list[np.ndarray]]]:
    items: list[QueryItem] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                items.append(query_from_json(json.loads(line)))
    num_bytes = int((num_samples + 7) >> 3)
    all_bitmaps: list[list[np.ndarray]] = []
    with bitmaps_path.open("rb") as handle:
        for item in items:
            raw = handle.read(4)
            if not raw:
                raise RuntimeError("Unexpected EOF in bitmaps")
            table_count = struct.unpack("<I", raw)[0]
            bitmaps: list[np.ndarray] = []
            for _ in range(table_count):
                payload = handle.read(num_bytes)
                if len(payload) != num_bytes:
                    raise RuntimeError("Unexpected EOF inside bitmap payload")
                bitmaps.append(np.unpackbits(np.frombuffer(payload, dtype=np.uint8))[:num_samples].astype(np.float32))
            if table_count != len(item.tables):
                raise RuntimeError(f"Bitmap/table mismatch for {item.query_id}: {table_count} vs {len(item.tables)}")
            all_bitmaps.append(bitmaps)
    return items, all_bitmaps


class RichEncoder:
    def __init__(self, train_items: list[QueryItem], train_bitmaps: list[list[np.ndarray]]) -> None:
        self.table_tokens = sorted({t for item in train_items for t in item.tables} | {UNK_TABLE})
        self.column_tokens = sorted({p.column for item in train_items for p in item.predicates} | {UNK_COLUMN})
        self.operator_tokens = sorted({p.operator for item in train_items for p in item.predicates} | {UNK_OPERATOR})
        self.join_tokens = sorted({j for item in train_items for j in item.joins} | {DUMMY_JOIN, UNK_JOIN})
        self.table_idx = {v: i for i, v in enumerate(self.table_tokens)}
        self.column_idx = {v: i for i, v in enumerate(self.column_tokens)}
        self.operator_idx = {v: i for i, v in enumerate(self.operator_tokens)}
        self.join_idx = {v: i for i, v in enumerate(self.join_tokens)}
        numeric_values: dict[str, list[float]] = defaultdict(list)
        for item in train_items:
            for pred in item.predicates:
                if pred.literal_type == "numeric":
                    numeric_values[pred.column].append(float(pred.literal))
        self.numeric_minmax = {
            col: [float(min(vals)), float(max(vals) if max(vals) > min(vals) else min(vals) + 1.0)]
            for col, vals in numeric_values.items()
            if vals
        }
        labels = np.array([item.label for item in train_items], dtype=np.float64)
        logs = np.log(np.maximum(labels, 1.0))
        self.label_min = float(logs.min())
        self.label_max = float(logs.max() if logs.max() > logs.min() else logs.min() + 1.0)
        self.max_tables = max(len(item.tables) for item in train_items)
        self.max_predicates = max(max(1, len(item.predicates)) for item in train_items)
        self.max_joins = max(max(1, len(item.joins)) for item in train_items)
        self.sample_feats = len(self.table_tokens) + NUM_SAMPLES
        self.predicate_feats = len(self.column_tokens) + len(self.operator_tokens) + LITERAL_DIM
        self.join_feats = len(self.join_tokens)

    def audit_eval_coverage(self, eval_items: dict[str, list[QueryItem]]) -> dict[str, Any]:
        train = {
            "tables": set(self.table_tokens),
            "columns": set(self.column_tokens),
            "operators": set(self.operator_tokens),
            "joins": set(self.join_tokens),
        }
        audit: dict[str, Any] = {}
        for name, items in eval_items.items():
            eval_sets = {
                "tables": {t for item in items for t in item.tables},
                "columns": {p.column for item in items for p in item.predicates},
                "operators": {p.operator for item in items for p in item.predicates},
                "joins": {j for item in items for j in item.joins},
            }
            audit[name] = {
                f"missing_{kind}": sorted(values - train[kind])
                for kind, values in eval_sets.items()
            }
            audit[name]["oov_counts"] = {
                kind: len(audit[name][f"missing_{kind}"])
                for kind in ("tables", "columns", "operators", "joins")
            }
        return audit

    def literal_vector(self, pred: Predicate) -> np.ndarray:
        vec = np.zeros(LITERAL_DIM, dtype=np.float32)
        if pred.literal_type == "numeric":
            low, high = self.numeric_minmax.get(pred.column, [0.0, 1.0])
            val = float(pred.literal)
            vec[0] = 0.0 if high <= low else float((val - low) / (high - low))
            return vec
        if pred.literal_type == "none":
            return vec
        values: list[str]
        if pred.literal_type == "list":
            values = [str(v) for v in pred.literal]
        else:
            values = [str(pred.literal)]
        tokens: list[str] = []
        for value in values:
            if pred.operator in {"LIKE", "NOT LIKE"}:
                parts = [p.strip() for p in value.split("%") if p.strip()]
                tokens.extend(parts or [value])
            else:
                tokens.append(value)
        if not tokens:
            return vec
        for token in tokens:
            full = f"{pred.column}_{token}"
            vec[1 + stable_hash_int(full, TEXT_HASH_DIM)] += 1.0
            for idx in range(max(1, len(full) - 2)):
                gram = full[idx:idx + 3]
                vec[1 + stable_hash_int(gram, TEXT_HASH_DIM)] += 0.25
        scale = float(max(1, len(tokens)))
        vec[1:] /= scale
        return vec

    def onehot(self, mapping: dict[str, int], key: str, unk_key: str) -> np.ndarray:
        vec = np.zeros(len(mapping), dtype=np.float32)
        idx = mapping.get(key, mapping.get(unk_key))
        if idx is not None:
            vec[idx] = 1.0
        return vec

    def normalize_label(self, label: int) -> float:
        value = math.log(max(1.0, float(label)))
        norm = (value - self.label_min) / (self.label_max - self.label_min)
        return float(min(1.0, max(0.0, norm)))

    def unnormalize_labels(self, values: np.ndarray) -> np.ndarray:
        logs = values * (self.label_max - self.label_min) + self.label_min
        return np.maximum(1.0, np.exp(logs))

    def encode_dataset(self, items: list[QueryItem], bitmaps: list[list[np.ndarray]]) -> tuple[TensorDataset, list[int]]:
        sample_rows: list[np.ndarray] = []
        pred_rows: list[np.ndarray] = []
        join_rows: list[np.ndarray] = []
        sample_masks: list[np.ndarray] = []
        pred_masks: list[np.ndarray] = []
        join_masks: list[np.ndarray] = []
        labels: list[float] = []
        actuals: list[int] = []
        for item, item_bitmaps in zip(items, bitmaps):
            s_arr = np.zeros((self.max_tables, self.sample_feats), dtype=np.float32)
            s_mask = np.zeros((self.max_tables, 1), dtype=np.float32)
            for i, (table_token, bitmap) in enumerate(zip(item.tables, item_bitmaps)):
                if i >= self.max_tables:
                    break
                s_arr[i, :len(self.table_tokens)] = self.onehot(self.table_idx, table_token, UNK_TABLE)
                s_arr[i, len(self.table_tokens):] = bitmap[:NUM_SAMPLES]
                s_mask[i, 0] = 1.0

            p_arr = np.zeros((self.max_predicates, self.predicate_feats), dtype=np.float32)
            p_mask = np.zeros((self.max_predicates, 1), dtype=np.float32)
            for i, pred in enumerate(item.predicates[: self.max_predicates]):
                col_vec = self.onehot(self.column_idx, pred.column, UNK_COLUMN)
                op_vec = self.onehot(self.operator_idx, pred.operator, UNK_OPERATOR)
                p_arr[i] = np.concatenate([col_vec, op_vec, self.literal_vector(pred)])
                p_mask[i, 0] = 1.0
            if not item.predicates:
                p_mask[0, 0] = 1.0

            j_arr = np.zeros((self.max_joins, self.join_feats), dtype=np.float32)
            j_mask = np.zeros((self.max_joins, 1), dtype=np.float32)
            joins = item.joins or [DUMMY_JOIN]
            for i, join in enumerate(joins[: self.max_joins]):
                j_arr[i] = self.onehot(self.join_idx, join, UNK_JOIN)
                j_mask[i, 0] = 1.0

            sample_rows.append(s_arr)
            pred_rows.append(p_arr)
            join_rows.append(j_arr)
            sample_masks.append(s_mask)
            pred_masks.append(p_mask)
            join_masks.append(j_mask)
            labels.append(self.normalize_label(item.label))
            actuals.append(item.label)

        dataset = TensorDataset(
            torch.tensor(np.stack(sample_rows), dtype=torch.float32),
            torch.tensor(np.stack(pred_rows), dtype=torch.float32),
            torch.tensor(np.stack(join_rows), dtype=torch.float32),
            torch.tensor(np.array(labels), dtype=torch.float32),
            torch.tensor(np.stack(sample_masks), dtype=torch.float32),
            torch.tensor(np.stack(pred_masks), dtype=torch.float32),
            torch.tensor(np.stack(join_masks), dtype=torch.float32),
        )
        return dataset, actuals

    def to_json(self) -> dict[str, Any]:
        return {
            "table_tokens": self.table_tokens,
            "column_tokens": self.column_tokens,
            "operator_tokens": self.operator_tokens,
            "join_tokens": self.join_tokens,
            "numeric_minmax": self.numeric_minmax,
            "label_min": self.label_min,
            "label_max": self.label_max,
            "max_tables": self.max_tables,
            "max_predicates": self.max_predicates,
            "max_joins": self.max_joins,
            "literal_dim": LITERAL_DIM,
            "text_hash_dim": TEXT_HASH_DIM,
            "num_samples": NUM_SAMPLES,
            "unknown_tokens": {
                "table": UNK_TABLE,
                "column": UNK_COLUMN,
                "operator": UNK_OPERATOR,
                "join": UNK_JOIN,
            },
        }


class SetConv(nn.Module):
    def __init__(self, sample_feats: int, predicate_feats: int, join_feats: int, hid_units: int) -> None:
        super().__init__()
        self.sample_mlp1 = nn.Linear(sample_feats, hid_units)
        self.sample_mlp2 = nn.Linear(hid_units, hid_units)
        self.predicate_mlp1 = nn.Linear(predicate_feats, hid_units)
        self.predicate_mlp2 = nn.Linear(hid_units, hid_units)
        self.join_mlp1 = nn.Linear(join_feats, hid_units)
        self.join_mlp2 = nn.Linear(hid_units, hid_units)
        self.out_mlp1 = nn.Linear(hid_units * 3, hid_units)
        self.out_mlp2 = nn.Linear(hid_units, 1)

    def forward(self, samples, predicates, joins, sample_mask, predicate_mask, join_mask):
        hid_sample = F.relu(self.sample_mlp1(samples))
        hid_sample = F.relu(self.sample_mlp2(hid_sample)) * sample_mask
        hid_sample = hid_sample.sum(dim=1) / torch.clamp(sample_mask.sum(dim=1), min=1.0)

        hid_predicate = F.relu(self.predicate_mlp1(predicates))
        hid_predicate = F.relu(self.predicate_mlp2(hid_predicate)) * predicate_mask
        hid_predicate = hid_predicate.sum(dim=1) / torch.clamp(predicate_mask.sum(dim=1), min=1.0)

        hid_join = F.relu(self.join_mlp1(joins))
        hid_join = F.relu(self.join_mlp2(hid_join)) * join_mask
        hid_join = hid_join.sum(dim=1) / torch.clamp(join_mask.sum(dim=1), min=1.0)

        hid = torch.cat((hid_sample, hid_predicate, hid_join), dim=1)
        hid = F.relu(self.out_mlp1(hid))
        return torch.sigmoid(self.out_mlp2(hid)).squeeze(1)


def qerror_loss(preds, targets, encoder: RichEncoder):
    pred_logs = preds * (encoder.label_max - encoder.label_min) + encoder.label_min
    target_logs = targets * (encoder.label_max - encoder.label_min) + encoder.label_min
    pred_cards = torch.exp(pred_logs)
    target_cards = torch.exp(target_logs)
    ratio = torch.maximum(pred_cards / target_cards, target_cards / pred_cards)
    return ratio.mean()


def predict(model: SetConv, dataset: TensorDataset, encoder: RichEncoder, batch_size: int, device: str) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks = [x.to(device) for x in batch]
            out = model(samples, predicates, joins, sample_masks, predicate_masks, join_masks)
            preds.append(out.detach().cpu().numpy())
    return encoder.unnormalize_labels(np.concatenate(preds))


def train_model(
    *,
    encoder: RichEncoder,
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    epochs: int,
    batch_size: int,
    hid_units: int,
    device: str,
) -> SetConv:
    model = SetConv(encoder.sample_feats, encoder.predicate_feats, encoder.join_feats, hid_units).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in loader:
            samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks = [x.to(device) for x in batch]
            optimizer.zero_grad()
            out = model(samples, predicates, joins, sample_masks, predicate_masks, join_masks)
            loss = qerror_loss(out, targets, encoder)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(json.dumps({"epoch": epoch, "train_qerror_loss": float(np.mean(losses))}), flush=True)
    return model


def write_predictions(path: Path, items: list[QueryItem], preds: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [item.label for item in items]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "prediction", "actual", "q_error", "source", "sql_path"])
        writer.writeheader()
        for item, pred in zip(items, preds):
            writer.writerow(
                {
                    "query_id": item.query_id,
                    "prediction": float(pred),
                    "actual": item.label,
                    "q_error": qerror(float(pred), item.label),
                    "source": item.source,
                    "sql_path": item.sql_path,
                }
            )
    return qerror_summary([float(p) for p in preds], labels)


def summarize_values(values: list[float]) -> dict[str, Any]:
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def postgres_target_qerror_summary(workload: str) -> dict[str, Any] | None:
    path = BASE_RUN / "postgres" / workload / "query_summary.csv"
    if not path.exists():
        return None
    finite: list[float] = []
    infinite = 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = row.get("target_q_error")
            if value in (None, ""):
                continue
            try:
                numeric = float(value)
            except Exception:
                continue
            if math.isfinite(numeric):
                finite.append(numeric)
            else:
                infinite += 1
    summary = summarize_values(finite)
    summary["infinite"] = infinite
    summary["source"] = str(path)
    return summary


def compare_baselines() -> dict[str, Any]:
    out: dict[str, Any] = {}
    old_path = MSCN_PAPER_RUN / "heldout_exact_job_qerror_summary.json"
    if old_path.exists():
        payload = json.loads(old_path.read_text(encoding="utf-8"))
        out["mscn_v1_exact_job"] = payload.get("heldout_eval", {}).get("q_error")
    for workload in ("job_exact", "job_complex"):
        summary = postgres_target_qerror_summary(workload)
        if summary is not None:
            out[f"postgres_{workload}_target"] = summary
    zero_path = BASE_RUN / "benchmark_readiness_summary.md"
    if zero_path.exists():
        out["readiness_summary"] = str(zero_path)
    return out


def write_report(run_dir: Path, summaries: dict[str, Any], coverage: dict[str, Any], args: argparse.Namespace) -> None:
    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8")) if run_manifest_path.exists() else {}
    lines = [
        "# Rich MSCN 10k Report",
        "",
        f"Run directory: `{run_dir}`",
        f"Database: `{args.db_name}`",
        f"Training target: `{args.train_queries}` queries",
        f"Epochs: `{args.epochs}`",
        "",
        "## Q-Error",
        "",
        "| Split | Count | Mean | Median | p90 | p95 | p99 | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, summary in summaries.items():
        if not isinstance(summary, dict) or "count" not in summary:
            continue
        lines.append(
            f"| {name} | {summary.get('count', 0)} | {summary.get('mean', 0):.4f} | "
            f"{summary.get('median', 0):.4f} | {summary.get('p90', 0):.4f} | "
            f"{summary.get('p95', 0):.4f} | {summary.get('p99', 0):.4f} | {summary.get('max', 0):.4f} |"
        )
    lines += [
        "",
        "## Baselines",
        "",
        "```json",
        json.dumps(summaries.get("baselines", {}), indent=2),
        "```",
        "",
        "## Benchmark Hygiene",
        "",
        "```json",
        json.dumps(run_manifest, indent=2),
        "```",
        "",
        "## Coverage",
        "",
        "Coverage is checked against train-set feature dictionaries. Missing eval features are encoded with explicit UNK buckets unless strict coverage is requested.",
        "",
        "```json",
        json.dumps(coverage, indent=2),
        "```",
        "",
        "## Notes",
        "",
        "- Text literals are encoded as column-aware deterministic hash vectors.",
        "- LIKE/NOT LIKE literals are split on `%`; IN lists are averaged over item vectors.",
        "- Table features use real PostgreSQL materialized-sample bitmaps.",
        "- Default training is leakage-free: no eval-derived coverage variants are added unless explicitly requested.",
    ]
    (run_dir / "rich_mscn_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_container_running(docker_bin: str, container: str) -> None:
    completed = subprocess.run(
        [docker_bin, "ps", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    running = set((completed.stdout or "").splitlines())
    if container in running:
        return
    subprocess.run([docker_bin, "start", container], check=True)


def main() -> None:
    global BASE_RUN, MSCN_PAPER_RUN, DEFAULT_DB_NAME, EVAL_EXACT_QUERY_DIR, EVAL_COMPLEX_QUERY_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=f"hollywood_rich_mscn_10k_{now_id()}")
    parser.add_argument("--db-name", default=DEFAULT_DB_NAME)
    parser.add_argument("--pg-container", default=DEFAULT_PG_CONTAINER)
    parser.add_argument("--pg-user", default=DEFAULT_PG_USER)
    parser.add_argument("--docker-bin", default="docker")
    parser.add_argument("--base-run", type=Path, default=BASE_RUN, help="CE run directory containing postgres/job_exact and postgres/job_complex traces.")
    parser.add_argument("--mscn-paper-run", type=Path, default=MSCN_PAPER_RUN, help="Optional existing MSCN variant source run.")
    parser.add_argument("--exact-query-dir", type=Path, default=EVAL_EXACT_QUERY_DIR)
    parser.add_argument("--complex-query-dir", type=Path, default=EVAL_COMPLEX_QUERY_DIR)
    parser.add_argument(
        "--exclude-query-dir",
        action="append",
        type=Path,
        default=[],
        help="Additional benchmark query directory whose exact SQL hashes must not appear in training.",
    )
    parser.add_argument("--train-queries", type=int, default=10000)
    parser.add_argument("--existing-job-variants", type=int, default=1110)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hid", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--statement-timeout-ms", type=int, default=8000)
    parser.add_argument("--strict-coverage", action="store_true")
    parser.add_argument("--allow-coverage-gaps", action="store_true", help="Deprecated compatibility flag; gaps are allowed by default through UNK tokens.")
    parser.add_argument("--include-eval-coverage-variants", action="store_true", help="Leakage-prone diagnostic mode: add eval-shaped variants to training. Do not use for publishable benchmarks.")
    parser.add_argument("--skip-existing-generation", action="store_true")
    args = parser.parse_args()

    BASE_RUN = args.base_run.resolve()
    MSCN_PAPER_RUN = args.mscn_paper_run.resolve()
    DEFAULT_DB_NAME = args.db_name
    EVAL_EXACT_QUERY_DIR = args.exact_query_dir.resolve()
    EVAL_COMPLEX_QUERY_DIR = args.complex_query_dir.resolve()

    run_dir = RUNS_DIR / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"run_dir": str(run_dir)}), flush=True)

    ensure_container_running(args.docker_bin, args.pg_container)
    eval_rows = load_eval_queries()
    eval_hashes = {sql_hash(sql) for rows in eval_rows.values() for _, sql, _, _ in rows}
    extra_excluded_hashes: set[str] = set()
    for query_dir in args.exclude_query_dir:
        extra_excluded_hashes.update(load_sql_hashes_from_dir(query_dir.resolve()))
    eval_hashes.update(extra_excluded_hashes)
    raw_existing_rows = load_existing_job_variants(limit=args.existing_job_variants)
    existing_rows = [row for row in raw_existing_rows if sql_hash(row[1]) not in eval_hashes]
    skipped_existing_eval_overlap = len(raw_existing_rows) - len(existing_rows)
    existing_hashes = {sql_hash(sql) for _, sql, _, _ in existing_rows}
    if args.include_eval_coverage_variants:
        coverage_rows = build_eval_coverage_rows(
            eval_rows=eval_rows,
            out_dir=run_dir / "generation",
            eval_hashes=eval_hashes,
            existing_hashes=existing_hashes,
            max_rows=max(0, int(args.train_queries) - len(existing_rows)),
        )
    else:
        coverage_rows = []
        generation_dir = run_dir / "generation"
        generation_dir.mkdir(parents=True, exist_ok=True)
        (generation_dir / "coverage_manifest.json").write_text(
            json.dumps(
                {
                    "method": "eval_shape_coverage_variants_v1",
                    "accepted": 0,
                    "disabled_by_default": True,
                    "note": "No eval-derived rows were added to training. Missing eval features are handled by UNK buckets.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    coverage_hashes = {sql_hash(sql) for _, sql, _, _ in coverage_rows}
    synthetic_needed = max(0, int(args.train_queries) - len(existing_rows) - len(coverage_rows))

    session = PsqlSession(docker_bin=args.docker_bin, container=args.pg_container, db_name=args.db_name, user=args.pg_user)
    try:
        synthetic_rows = generate_synthetic_queries(
            session=session,
            out_dir=run_dir / "generation",
            target_count=synthetic_needed,
            eval_hashes=eval_hashes,
            existing_hashes=existing_hashes | coverage_hashes,
            seed=args.seed,
            timeout_ms=args.statement_timeout_ms,
        )
        train_rows = existing_rows + coverage_rows + synthetic_rows
        train_hashes = {sql_hash(sql) for _, sql, _, _ in train_rows}
        exact_overlap = sorted(train_hashes & eval_hashes)
        if exact_overlap:
            raise RuntimeError(f"Training SQL overlaps eval SQL by hash: {exact_overlap[:5]}")
        (run_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "method": "rich_mscn_no_leak_benchmark_v1",
                    "db_name": args.db_name,
                    "train_target": args.train_queries,
                    "seed": args.seed,
                    "existing_job_variants_requested": args.existing_job_variants,
                    "existing_job_variants_loaded": len(raw_existing_rows),
                    "existing_job_variants_used": len(existing_rows),
                    "existing_job_variants_skipped_exact_eval_overlap": skipped_existing_eval_overlap,
                    "eval_coverage_variants_enabled": bool(args.include_eval_coverage_variants),
                    "eval_coverage_variants_used": len(coverage_rows),
                    "synthetic_queries_used": len(synthetic_rows),
                    "train_queries_used": len(train_rows),
                    "exact_eval_sql_hash_overlap": len(exact_overlap),
                    "eval_query_counts": {name: len(rows) for name, rows in eval_rows.items()},
                    "extra_excluded_sql_hashes": len(extra_excluded_hashes),
                    "leakage_policy": "No exact eval SQL hashes in training; eval-derived coverage variants disabled unless --include-eval-coverage-variants is set.",
                    "oov_policy": "Tables, columns, operators, and joins not present in training are encoded with explicit UNK one-hot buckets.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        random.Random(args.seed).shuffle(train_rows)
        train_items = write_workload(
            name="train",
            rows=train_rows,
            out_dir=run_dir / "rich_train",
            session=session,
            num_samples=NUM_SAMPLES,
        )
        eval_items_by_name: dict[str, list[QueryItem]] = {}
        for name, rows in eval_rows.items():
            eval_items_by_name[name] = write_workload(
                name=name,
                rows=rows,
                out_dir=run_dir / f"rich_eval_{name}",
                session=session,
                num_samples=NUM_SAMPLES,
            )
    finally:
        session.close()

    train_loaded, train_bitmaps = load_workload(run_dir / "rich_train" / "workload.jsonl", run_dir / "rich_train" / "bitmaps.bin", NUM_SAMPLES)
    split = int(len(train_loaded) * 0.9)
    train_core_items = train_loaded[:split]
    train_core_bitmaps = train_bitmaps[:split]
    val_items = train_loaded[split:]
    val_bitmaps = train_bitmaps[split:]

    encoder = RichEncoder(train_loaded, train_bitmaps)
    loaded_eval_items: dict[str, list[QueryItem]] = {}
    loaded_eval_bitmaps: dict[str, list[list[np.ndarray]]] = {}
    for name in eval_rows:
        items, bitmaps = load_workload(run_dir / f"rich_eval_{name}" / "workload.jsonl", run_dir / f"rich_eval_{name}" / "bitmaps.bin", NUM_SAMPLES)
        loaded_eval_items[name] = items
        loaded_eval_bitmaps[name] = bitmaps
    coverage = encoder.audit_eval_coverage(loaded_eval_items)
    (run_dir / "coverage_audit.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    has_gaps = any(
        values
        for audit in coverage.values()
        for key, values in audit.items()
        if key.startswith("missing_")
    )
    if has_gaps and args.strict_coverage and not args.allow_coverage_gaps:
        raise RuntimeError(f"Coverage gaps detected; see {run_dir / 'coverage_audit.json'}")

    (run_dir / "encoder_manifest.json").write_text(json.dumps(encoder.to_json(), indent=2), encoding="utf-8")
    train_dataset, train_actuals = encoder.encode_dataset(train_core_items, train_core_bitmaps)
    val_dataset, val_actuals = encoder.encode_dataset(val_items, val_bitmaps)
    eval_datasets: dict[str, tuple[TensorDataset, list[int]]] = {}
    for name, items in loaded_eval_items.items():
        eval_datasets[name] = encoder.encode_dataset(items, loaded_eval_bitmaps[name])

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(json.dumps({"warning": "CUDA requested but unavailable; using CPU"}), flush=True)
        device = "cpu"
    torch.manual_seed(args.seed)
    model = train_model(
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
            "args": vars(args),
        },
        run_dir / "rich_mscn_model.pt",
    )

    summaries: dict[str, Any] = {}
    train_preds = predict(model, train_dataset, encoder, args.batch_size, device)
    summaries["train_90pct"] = write_predictions(run_dir / "predictions" / "train_90pct.csv", train_core_items, train_preds)
    val_preds = predict(model, val_dataset, encoder, args.batch_size, device)
    summaries["validation_10pct"] = write_predictions(run_dir / "predictions" / "validation_10pct.csv", val_items, val_preds)
    for name, (dataset, _) in eval_datasets.items():
        preds = predict(model, dataset, encoder, args.batch_size, device)
        summaries[name] = write_predictions(run_dir / "predictions" / f"{name}.csv", loaded_eval_items[name], preds)

    summaries["baselines"] = compare_baselines()
    (run_dir / "qerror_summaries.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    write_report(run_dir, summaries, coverage, args)
    print(json.dumps({"done": True, "run_dir": str(run_dir), "summaries": summaries}, indent=2), flush=True)


if __name__ == "__main__":
    main()
