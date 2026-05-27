from __future__ import annotations

import argparse
import difflib
import hashlib
import itertools
import json
import math
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXCLUDED_QUERY_FILES = {"schema.sql", "fkindexes.sql"}
DEFAULT_PG_CONTAINER = "pg_bench"
DEFAULT_PG_USER = "postgres"
DEFAULT_DUCKDB_BIN_CANDIDATES = (
    Path("/snap/duckdb/9/duckdb"),
    Path("/snap/bin/duckdb"),
)
DEFAULT_MINING_TIMEOUT_MS = 8000
# Keep this as a grouped atom. SLOT_RE embeds it inside larger alternatives; if
# the string/number alternation is not grouped, multi-line IN clauses can be
# partially skipped and the adapter silently fails to search those literals.
SQL_LITERAL_PATTERN = r"(?:'(?:''|[^'])*'|\b\d+(?:\.\d+)?\b)"
STRUCTURE_TOKEN_RE = re.compile(
    r"""
    (?:--[^\n]*)
    |(?:/\*.*?\*/)
    |(?:'(?:''|[^'])*')
    |(?:\b\d+(?:\.\d+)?\b)
    |(?:[A-Za-z_][\w$]*)
    |(?:<=|>=|<>|!=|::)
    |(?:.)
    """,
    re.S | re.X,
)
SLOT_RE = re.compile(
    r"""
    (?P<between>
        (?P<between_expr>[A-Za-z_][\w$]*\.[A-Za-z_][\w$]*)
        \s+BETWEEN\s+
        (?P<between_a>\d+(?:\.\d+)?)
        \s+AND\s+
        (?P<between_b>\d+(?:\.\d+)?)
    )
    |
    (?P<inop>
        (?P<in_expr>[A-Za-z_][\w$]*\.[A-Za-z_][\w$]*)
        \s+(?P<in_op>IN|NOT\s+IN)\s*
        \(
            (?P<in_body>\s*__LITERAL__(?:\s*,\s*__LITERAL__)*\s*)
        \)
    )
    |
    (?P<like>
        (?P<like_expr>[A-Za-z_][\w$]*\.[A-Za-z_][\w$]*)
        \s+(?P<like_op>LIKE|NOT\s+LIKE)\s+
        (?P<like_lit>__LITERAL__)
    )
    |
    (?P<cmp>
        (?P<cmp_expr>[A-Za-z_][\w$]*\.[A-Za-z_][\w$]*)
        \s*(?P<cmp_op>=|!=|<>|<=|>=|<|>)\s*
        (?P<cmp_lit>__LITERAL__)
    )
    """.replace("__LITERAL__", SQL_LITERAL_PATTERN),
    re.I | re.S | re.X,
)
IN_LITERAL_RE = re.compile(SQL_LITERAL_PATTERN, re.S)


@dataclass(frozen=True)
class LiteralRef:
    start: int
    end: int
    sql_text: str
    kind: str


@dataclass(frozen=True)
class Slot:
    index: int
    kind: str
    expr: str
    operator: str
    alias: str
    column: str
    table: str | None
    span_start: int
    span_end: int
    literals: tuple[LiteralRef, ...]
    wildcard_shape: str | None

    @property
    def arity(self) -> int:
        return len(self.literals)

    @property
    def signature(self) -> tuple[str, str, str, int]:
        return (
            self.kind.lower(),
            self.expr.lower(),
            self.operator.upper(),
            self.arity,
        )


@dataclass
class QueryPlanStats:
    actual_count: int
    sub_agg_estimate: float
    sub_agg_actual: float
    sub_agg_qerror: float
    sub_agg_bias: str
    sub_agg_node: str
    planning_ms: float
    execution_ms: float
    total_ms: float


@dataclass
class QueryResult:
    query_id: str
    structure_checksum: str
    structure_guard_passed: bool
    actual_count: int
    status: str
    blocker_notes: list[str]
    changed_slots: int
    postgres_plan: QueryPlanStats | None
    postgres_error: str | None
    duckdb_status: str
    duckdb_actual_count: int | None
    duckdb_ms: float | None
    duckdb_error: str | None
    original_literals: list[dict[str, Any]]
    adapted_literals: list[dict[str, Any]]


@dataclass
class BeamCandidate:
    state: dict[int, tuple[str, ...]]
    actual: int
    error: str | None
    relaxed_support: int


class PostgresRunner:
    def __init__(self, *, container: str, db_name: str, user: str, persistent: bool = False) -> None:
        self.container = container
        self.db_name = db_name
        self.user = user
        self.persistent = persistent
        self._scalar_cache: dict[str, str] = {}
        self._proc: subprocess.Popen[str] | None = None
        self._seq = 0

    def _ensure_proc(self) -> subprocess.Popen[str]:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        cmd = [
            "docker",
            "exec",
            "-i",
            self.container,
            "psql",
            "-X",
            "-U",
            self.user,
            "-d",
            self.db_name,
            "-P",
            "pager=off",
            "-qAt",
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        return self._proc

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None or proc.poll() is not None:
            return
        try:
            if proc.stdin:
                proc.stdin.write("\\q\n")
                proc.stdin.flush()
            proc.wait(timeout=2)
        except Exception:
            proc.kill()

    def _run_sql_persistent(self, sql: str) -> str:
        proc = self._ensure_proc()
        if proc.stdin is None or proc.stdout is None:
            raise RuntimeError("persistent psql session is not available")
        self._seq += 1
        sentinel = f"__JOB_BENCHMARK_END_{self._seq}__"
        proc.stdin.write(sql.strip().rstrip(";") + ";\n")
        proc.stdin.write(f"\\echo {sentinel}\n")
        proc.stdin.flush()

        lines: list[str] = []
        while True:
            line = proc.stdout.readline()
            if line == "":
                raise RuntimeError("persistent psql session exited unexpectedly")
            stripped = line.rstrip("\n")
            if stripped == sentinel:
                break
            lines.append(stripped)

        output = "\n".join(lines).strip()
        for line in lines:
            if re.match(r"^(ERROR|FATAL|PANIC):", line):
                raise RuntimeError(line[:240])
        return output

    def _run_sql(self, sql: str) -> str:
        key = f"{self.db_name}\n{sql}"
        cached = self._scalar_cache.get(key)
        if cached is not None:
            return cached
        if self.persistent:
            try:
                output = self._run_sql_persistent(sql)
                self._scalar_cache[key] = output
                return output
            except RuntimeError as exc:
                message = str(exc)
                if re.match(r"^(ERROR|FATAL|PANIC):", message):
                    raise
                self.close()
        cmd = [
            "docker",
            "exec",
            self.container,
            "psql",
            "-X",
            "-U",
            self.user,
            "-d",
            self.db_name,
            "-v",
            "ON_ERROR_STOP=1",
            "-P",
            "pager=off",
            "-tAq",
            "-c",
            sql,
        ]
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            stdout = (completed.stdout or "").strip()
            message = stderr or stdout or f"psql exited with code {completed.returncode}"
            raise RuntimeError(message)
        output = (completed.stdout or "").strip()
        self._scalar_cache[key] = output
        return output

    def exec(self, sql: str) -> str:
        return self._run_sql(sql)

    def scalar_int(self, sql: str) -> int:
        raw = self._run_sql(sql)
        return int(float(raw or "0"))

    def json_scalar(self, sql: str) -> Any:
        raw = self._run_sql(sql)
        return json.loads(raw or "null")

    def analyze(self) -> None:
        self._run_sql("ANALYZE;")


class DuckDBRunner:
    def __init__(self, *, binary: Path, db_path: Path) -> None:
        self.binary = binary
        self.db_path = db_path

    def count_query(self, sql: str) -> tuple[int, float]:
        cmd = [
            str(self.binary),
            str(self.db_path),
            "-readonly",
            "-json",
            "-c",
            normalize_sql_for_duckdb(strip_to_count(sql)),
        ]
        t0 = time.perf_counter()
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            stdout = (completed.stdout or "").strip()
            message = stderr or stdout or f"duckdb exited with code {completed.returncode}"
            raise RuntimeError(message)
        payload = json.loads((completed.stdout or "[]").strip() or "[]")
        if not payload:
            return 0, elapsed_ms
        row = payload[0]
        value = int(float(row.get("count_star()", 0) or row.get("count", 0) or 0))
        return value, elapsed_ms


def sort_key(path: Path) -> tuple[int, str]:
    match = re.match(r"(\d+)([a-z]?)", path.stem)
    if not match:
        return (999, path.stem)
    return (int(match.group(1)), match.group(2))


def normalize_sql_for_duckdb(sql: str) -> str:
    normalized = str(sql or "")
    normalized = re.sub(r"\baka_title\s+AS\s+AT\b", "aka_title AS aka_t", normalized, flags=re.I)
    normalized = re.sub(r"\bat\.", "aka_t.", normalized, flags=re.I)
    return normalized


def sql_unquote(text: str) -> str:
    raw = str(text or "")
    if len(raw) >= 2 and raw[0] == "'" and raw[-1] == "'":
        return raw[1:-1].replace("''", "'")
    return raw


def sql_quote(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def structure_fingerprint(sql: str) -> str:
    parts: list[str] = []
    for token in STRUCTURE_TOKEN_RE.findall(sql):
        stripped = token.strip()
        if not stripped or stripped.startswith("--") or stripped.startswith("/*"):
            continue
        if re.fullmatch(r"'(?:''|[^'])*'", token, re.S):
            parts.append("<str>")
        elif re.fullmatch(r"\b\d+(?:\.\d+)?\b", token):
            parts.append("<num>")
        elif re.fullmatch(r"[A-Za-z_][\w$]*", token):
            parts.append(token.lower())
        else:
            parts.append(token)
    return hashlib.sha256(" ".join(parts).encode("utf-8")).hexdigest()


def normalized_sql_content_hash(sql: str) -> str:
    """Hash rendered SQL text after whitespace/semicolon normalization."""
    normalized = " ".join(str(sql or "").strip().rstrip(";").split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def extract_from_where(sql: str) -> tuple[str, str]:
    from_match = re.search(r"\bFROM\b", sql, re.I)
    where_match = re.search(r"\bWHERE\b", sql, re.I)
    if not from_match or not where_match:
        raise ValueError("Expected SELECT ... FROM ... WHERE ... query")
    return sql[from_match.start():where_match.start()], sql[where_match.end():].strip().rstrip(";")


def strip_to_count(sql: str) -> str:
    from_match = re.search(r"\bFROM\b", sql, re.I)
    if not from_match:
        raise ValueError("Could not find FROM clause")
    return "SELECT COUNT(*) " + sql[from_match.start():].strip().rstrip(";")


def strip_to_exists(sql: str) -> str:
    from_match = re.search(r"\bFROM\b", sql, re.I)
    if not from_match:
        raise ValueError("Could not find FROM clause")
    return "SELECT 1 " + sql[from_match.start():].strip().rstrip(";") + " LIMIT 1"


def parse_alias_map(from_clause: str) -> dict[str, str]:
    work = from_clause.strip()
    if work[:4].upper() == "FROM":
        work = work[4:]
    parts: list[str] = []
    depth = 0
    start = 0
    for idx, ch in enumerate(work):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            parts.append(work[start:idx].strip())
            start = idx + 1
    tail = work[start:].strip()
    if tail:
        parts.append(tail)

    aliases: dict[str, str] = {}
    for part in parts:
        cleaned = re.sub(r"\s+", " ", part.strip())
        match = re.match(r"([A-Za-z_][\w$]*)(?:\s+(?:AS\s+)?([A-Za-z_][\w$]*))?$", cleaned, re.I)
        if not match:
            continue
        table = match.group(1)
        alias = match.group(2) or table
        aliases[alias.lower()] = table
    return aliases


def wildcard_shape(pattern: str) -> str:
    segments = [segment for segment in re.split(r"[%_]+", pattern) if segment]
    return json.dumps(
        {
            "starts_wild": bool(pattern[:1] in {"%", "_"}),
            "ends_wild": bool(pattern[-1:] in {"%", "_"}),
            "segments": len(segments),
            "segment_lengths": [len(segment) for segment in segments],
        },
        sort_keys=True,
    )


def extract_slots(sql: str) -> list[Slot]:
    from_clause, _ = extract_from_where(sql)
    alias_map = parse_alias_map(from_clause)
    slots: list[Slot] = []
    for index, match in enumerate(SLOT_RE.finditer(sql)):
        gd = match.groupdict()
        if gd.get("between"):
            expr = gd["between_expr"]
            operator = "BETWEEN"
            literals = (
                LiteralRef(match.start("between_a"), match.end("between_a"), gd["between_a"], "number"),
                LiteralRef(match.start("between_b"), match.end("between_b"), gd["between_b"], "number"),
            )
            kind = "between"
            shape = None
        elif gd.get("inop"):
            expr = gd["in_expr"]
            operator = re.sub(r"\s+", " ", gd["in_op"].upper())
            body_start = match.start("in_body")
            literals = tuple(
                LiteralRef(
                    body_start + item.start(),
                    body_start + item.end(),
                    item.group(0),
                    "string" if item.group(0).startswith("'") else "number",
                )
                for item in IN_LITERAL_RE.finditer(gd["in_body"])
            )
            kind = "in"
            shape = None
        elif gd.get("like"):
            expr = gd["like_expr"]
            operator = re.sub(r"\s+", " ", gd["like_op"].upper())
            literal = gd["like_lit"]
            literals = (
                LiteralRef(
                    match.start("like_lit"),
                    match.end("like_lit"),
                    literal,
                    "string" if literal.startswith("'") else "number",
                ),
            )
            kind = "like"
            shape = wildcard_shape(sql_unquote(literal))
        else:
            expr = gd["cmp_expr"]
            operator = gd["cmp_op"].upper()
            literal = gd["cmp_lit"]
            literals = (
                LiteralRef(
                    match.start("cmp_lit"),
                    match.end("cmp_lit"),
                    literal,
                    "string" if literal.startswith("'") else "number",
                ),
            )
            kind = "cmp"
            shape = None

        alias, column = expr.split(".", 1)
        slots.append(
            Slot(
                index=index,
                kind=kind,
                expr=expr,
                operator=operator,
                alias=alias,
                column=column,
                table=alias_map.get(alias.lower()),
                span_start=match.start(),
                span_end=match.end(),
                literals=literals,
                wildcard_shape=shape,
            )
        )
    return slots


def render_sql(
    base_sql: str,
    slots: list[Slot],
    slot_values: dict[int, tuple[str, ...]],
    *,
    slot_overrides: dict[int, str] | None = None,
) -> str:
    replacements: list[tuple[int, int, str]] = []
    overrides = slot_overrides or {}
    for slot in slots:
        if slot.index in overrides:
            replacements.append((slot.span_start, slot.span_end, overrides[slot.index]))
            continue
        values = slot_values.get(slot.index)
        if not values:
            continue
        if len(values) != slot.arity:
            raise ValueError(f"Slot {slot.index} expected {slot.arity} literals, got {len(values)}")
        for literal, replacement in zip(slot.literals, values):
            replacements.append((literal.start, literal.end, replacement))
    sql = base_sql
    for start, end, replacement in sorted(replacements, key=lambda item: item[0], reverse=True):
        sql = sql[:start] + replacement + sql[end:]
    return sql


def original_slot_values(slot: Slot) -> tuple[str, ...]:
    return tuple(literal.sql_text for literal in slot.literals)


def load_historical_state(
    *,
    historical_query_dir: Path | None,
    query_id: str,
    original_sql: str,
    original_slots: list[Slot],
) -> dict[int, tuple[str, ...]]:
    if historical_query_dir is None:
        return {}
    historical_path = historical_query_dir / f"{query_id}.sql"
    if not historical_path.exists():
        return {}
    historical_sql = historical_path.read_text(encoding="utf-8")
    if structure_fingerprint(historical_sql) != structure_fingerprint(original_sql):
        return {}
    historical_slots = extract_slots(historical_sql)
    if len(historical_slots) != len(original_slots):
        return {}
    state: dict[int, tuple[str, ...]] = {}
    for original_slot, historical_slot in zip(original_slots, historical_slots):
        if historical_slot.signature != original_slot.signature:
            return {}
        values = original_slot_values(historical_slot)
        if values != original_slot_values(original_slot):
            state[original_slot.index] = values
    return state


def unique_preserve_order(items: list[tuple[str, ...]]) -> list[tuple[str, ...]]:
    seen: set[tuple[str, ...]] = set()
    out: list[tuple[str, ...]] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def safe_json_sql(value: str) -> str:
    return sql_quote(value)


def with_statement_timeout(sql: str, timeout_ms: int = DEFAULT_MINING_TIMEOUT_MS) -> str:
    timeout = max(1, int(timeout_ms))
    return f"SET statement_timeout TO {timeout};\n{sql.strip().rstrip(';')};\nSET statement_timeout TO 0;"


def mine_values_for_slot(
    *,
    slot: Slot,
    base_sql: str,
    base_state: dict[int, tuple[str, ...]],
    runner: PostgresRunner,
    limit: int,
) -> list[dict[str, Any]]:
    try:
        relaxed_sql = render_sql(base_sql, extract_slots(base_sql), base_state, slot_overrides={slot.index: "TRUE"})
        from_clause, where_clause = extract_from_where(relaxed_sql)
    except Exception:
        return []
    expr = slot.expr
    query = f"""
    SELECT COALESCE(
        json_agg(
            json_build_object('value', value, 'freq', freq)
            ORDER BY freq DESC, value_text
        )::text,
        '[]'
    )
    FROM (
        SELECT
            {expr} AS value,
            COUNT(*) AS freq,
            MIN(({expr})::text) AS value_text
        {from_clause}
        WHERE {where_clause}
          AND {expr} IS NOT NULL
        GROUP BY 1
        ORDER BY freq DESC, value_text
        LIMIT {int(limit)}
    ) s;
    """
    try:
        payload = runner.json_scalar(with_statement_timeout(query))
        return payload if isinstance(payload, list) else []
    except Exception:
        return []


def mine_context_values_for_slot(*, slot: Slot, base_sql: str, runner: PostgresRunner, limit: int) -> list[dict[str, Any]]:
    slots = extract_slots(base_sql)
    try:
        relaxed_sql = render_sql(
            base_sql,
            slots,
            {},
            slot_overrides={other.index: "TRUE" for other in slots},
        )
        from_clause, where_clause = extract_from_where(relaxed_sql)
    except Exception:
        return []
    expr = slot.expr
    query = f"""
    SELECT COALESCE(
        json_agg(
            json_build_object('value', value, 'freq', freq)
            ORDER BY freq DESC, value_text
        )::text,
        '[]'
    )
    FROM (
        SELECT
            {expr} AS value,
            COUNT(*) AS freq,
            MIN(({expr})::text) AS value_text
        {from_clause}
        WHERE {where_clause}
          AND {expr} IS NOT NULL
        GROUP BY 1
        ORDER BY freq DESC, value_text
        LIMIT {int(limit)}
    ) s;
    """
    try:
        payload = runner.json_scalar(with_statement_timeout(query))
        return payload if isinstance(payload, list) else []
    except Exception:
        return []


def literal_tokens(slot: Slot) -> list[str]:
    tokens: list[str] = []
    for literal in slot.literals:
        if literal.kind != "string":
            continue
        text = sql_unquote(literal.sql_text)
        for token in re.split(r"[^A-Za-z0-9]+", text):
            cleaned = token.strip().lower()
            if len(cleaned) >= 3:
                tokens.append(cleaned)
    out: list[str] = []
    seen: set[str] = set()
    for token in sorted(tokens, key=len, reverse=True):
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def mine_token_match_values_for_slot(*, slot: Slot, runner: PostgresRunner, limit: int) -> list[dict[str, Any]]:
    if not slot.table or not slot.literals or slot.literals[0].kind != "string":
        return []
    tokens = literal_tokens(slot)[:2]
    if not tokens:
        return []
    filters = " OR ".join(
        f"LOWER(({slot.column})::text) LIKE {sql_quote('%' + token + '%')}"
        for token in tokens
    )
    query = f"""
    SELECT COALESCE(
        json_agg(
            json_build_object('value', value, 'freq', freq)
            ORDER BY freq DESC, value_text
        )::text,
        '[]'
    )
    FROM (
        SELECT
            {slot.column} AS value,
            COUNT(*) AS freq,
            MIN(({slot.column})::text) AS value_text
        FROM {slot.table}
        WHERE {slot.column} IS NOT NULL
          AND ({filters})
        GROUP BY 1
        ORDER BY freq DESC, value_text
        LIMIT {int(limit)}
    ) s;
    """
    try:
        payload = runner.json_scalar(with_statement_timeout(query))
        return payload if isinstance(payload, list) else []
    except Exception:
        return []


def mine_global_values_for_slot(*, slot: Slot, runner: PostgresRunner, limit: int) -> list[dict[str, Any]]:
    if not slot.table:
        return []
    query = f"""
    SELECT COALESCE(
        json_agg(
            json_build_object('value', value, 'freq', freq)
            ORDER BY freq DESC, value_text
        )::text,
        '[]'
    )
    FROM (
        SELECT
            {slot.column} AS value,
            COUNT(*) AS freq,
            MIN(({slot.column})::text) AS value_text
        FROM {slot.table}
        WHERE {slot.column} IS NOT NULL
        GROUP BY 1
        ORDER BY freq DESC, value_text
        LIMIT {int(limit)}
    ) s;
    """
    try:
        payload = runner.json_scalar(with_statement_timeout(query))
        return payload if isinstance(payload, list) else []
    except Exception:
        return []


def char_class(ch: str) -> str:
    if ch.isdigit():
        return "D"
    if ch.isalpha():
        return "A"
    if ch.isspace():
        return "S"
    return "P"


def best_matching_substrings(text: str, template: str, limit: int = 3) -> list[str]:
    sample = str(text or "")
    target = str(template or "")
    width = max(1, len(target))
    if not sample:
        return []
    if len(sample) <= width:
        return [sample]

    scored: list[tuple[int, int, str]] = []
    for start in range(0, len(sample) - width + 1):
        chunk = sample[start:start + width]
        score = 0
        for source_ch, candidate_ch in zip(target, chunk):
            if source_ch.lower() == candidate_ch.lower():
                score += 4
            elif char_class(source_ch) == char_class(candidate_ch):
                score += 2
        if start == 0:
            score += 1
        if start + width == len(sample):
            score += 1
        scored.append((score, -start, chunk))

    out: list[str] = []
    seen: set[str] = set()
    for _, _, chunk in sorted(scored, reverse=True):
        if chunk in seen:
            continue
        seen.add(chunk)
        out.append(chunk)
        if len(out) >= max(1, limit):
            break
    return out


def like_fragments_for_value(
    raw_value: str,
    segment_lengths: list[int],
    segment_count: int,
    original_segments: list[str],
) -> list[list[str]]:
    text = str(raw_value or "").strip()
    if not text:
        return []
    tokens = [token for token in re.split(r"[^A-Za-z0-9]+", text) if token]
    raw_chunks = [chunk for chunk in re.split(r"[%_]+", text) if chunk]
    fragments: list[list[str]] = []

    if tokens:
        if len(tokens) >= segment_count:
            fragments.append(tokens[:segment_count])
            fragments.append(tokens[-segment_count:])
        elif segment_count == 1:
            fragments.append([tokens[0]])
            fragments.append([tokens[-1]])

    if raw_chunks and segment_count == 1:
        fragments.append([raw_chunks[0]])
        fragments.append([raw_chunks[-1]])

    if segment_count == 1:
        source = tokens[0] if tokens else text
        for width in segment_lengths or [min(6, len(source))]:
            width = max(1, min(len(source), int(width)))
            fragments.append([source[:width]])
            fragments.append([source[-width:]])

    if original_segments:
        per_segment_matches = [
            best_matching_substrings(text, segment, limit=2)
            for segment in original_segments[:segment_count]
        ]
        if len(per_segment_matches) == segment_count and all(per_segment_matches):
            for combo in itertools.product(*per_segment_matches):
                fragments.append([str(item) for item in combo])

    cleaned: list[list[str]] = []
    for candidate in fragments:
        items = [str(item).strip() for item in candidate if str(item).strip()]
        if len(items) != segment_count:
            continue
        cleaned.append(items)
    return cleaned


def build_like_pattern(original: str, fragments: list[str]) -> str:
    starts_wild = bool(original[:1] in {"%", "_"})
    ends_wild = bool(original[-1:] in {"%", "_"})
    body = "%".join(fragments)
    if starts_wild:
        body = "%" + body
    if ends_wild:
        body = body + "%"
    return body


def flexible_like_patterns(original: str, raw_value: str) -> list[str]:
    """Build freer LIKE replacements while keeping the predicate slot intact.

    The benchmark contract is exact SQL structure, not exact literal topology:
    a title predicate like ``t.title LIKE 'Kung Fu Panda%'`` may become another
    prefix/contains pattern mined from the dataset. Keep the original pattern as
    one candidate elsewhere, but add useful alternatives early in the list.
    """
    value = str(raw_value or "").strip()
    if not value:
        return []

    original = str(original or "")
    starts_wild = bool(original[:1] in {"%", "_"})
    ends_wild = bool(original[-1:] in {"%", "_"})
    tokens = [token for token in re.split(r"[^A-Za-z0-9]+", value) if token]
    candidates: list[str] = []

    def add(pattern: str) -> None:
        pattern = str(pattern or "").strip()
        if pattern:
            candidates.append(pattern)

    # Preserve the broad intent of equality-like vs prefix/suffix/contains, but
    # allow the concrete string and even wildcard granularity to move.
    if starts_wild and ends_wild:
        add(f"%{value}%")
        for token in tokens[:3]:
            add(f"%{token}%")
        for token in sorted(tokens, key=len, reverse=True)[:2]:
            add(f"%{token}%")
    elif ends_wild:
        add(f"{value}%")
        if tokens:
            add(f"{tokens[0]}%")
        if len(tokens) >= 2:
            add(f"{tokens[0]} {tokens[1]}%")
    elif starts_wild:
        add(f"%{value}")
        if tokens:
            add(f"%{tokens[-1]}")
        if len(tokens) >= 2:
            add(f"%{tokens[-2]} {tokens[-1]}")
    else:
        add(value)

    # Cross-shape fallbacks are intentionally late: they rescue sparse patterns
    # without dominating the first, more semantically similar candidates.
    if tokens:
        add(f"%{tokens[0]}%")
        add(f"{tokens[0]}%")
    add(f"%{value}%")

    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        out.append(candidate)
    return out


def numeric_value(item: Any) -> float | None:
    if item is None:
        return None
    try:
        return float(item)
    except Exception:
        return None


def build_slot_candidates(
    *,
    slot: Slot,
    query_local_values: list[dict[str, Any]],
    historical_local_values: list[dict[str, Any]],
    query_context_values: list[dict[str, Any]],
    token_match_values: list[dict[str, Any]],
    global_values: list[dict[str, Any]],
    historical_state: dict[int, tuple[str, ...]],
) -> list[tuple[str, ...]]:
    candidates: list[tuple[str, ...]] = [original_slot_values(slot)]
    historical_values = historical_state.get(slot.index)
    if historical_values:
        candidates.append(historical_values)

    def iter_values() -> list[Any]:
        out: list[Any] = []
        for bucket in (
            query_local_values,
            historical_local_values,
            query_context_values,
            token_match_values,
            global_values,
        ):
            for row in bucket:
                out.append(row.get("value"))
        return out

    raw_values = iter_values()

    if slot.kind == "between":
        nums = [value for value in (numeric_value(item) for item in raw_values) if value is not None]
        nums = sorted({int(round(value)) for value in nums})
        if nums:
            width = max(1, int(round(float(slot.literals[1].sql_text) - float(slot.literals[0].sql_text))))
            pivots = []
            pivots.extend(nums[:3])
            pivots.extend(nums[max(0, len(nums) // 2 - 1):len(nums) // 2 + 2])
            pivots.extend(nums[-3:])
            for pivot in pivots:
                high = pivot + width
                if nums:
                    high = min(high, nums[-1])
                low = min(pivot, high)
                candidates.append((str(int(low)), str(int(high))))
            candidates.append((str(nums[0]), str(nums[-1])))
    elif slot.kind == "in":
        values: list[str] = []
        for item in raw_values:
            if item is None:
                continue
            values.append(str(item))
        deduped = []
        seen: set[str] = set()
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            deduped.append(value)
        if deduped:
            arity = slot.arity
            windows = min(6, max(1, len(deduped) - arity + 1))
            for start in range(windows):
                chunk = deduped[start:start + arity]
                if len(chunk) == arity:
                    candidates.append(tuple(sql_quote(item) for item in chunk))
            if len(deduped) >= arity:
                candidates.append(tuple(sql_quote(item) for item in deduped[-arity:]))
    elif slot.kind == "like":
        original = sql_unquote(slot.literals[0].sql_text)
        segments = [segment for segment in re.split(r"[%_]+", original) if segment]
        segment_lengths = [len(segment) for segment in segments] or [max(1, len(original.strip("%_")))]
        segment_count = max(1, len(segments))
        for item in raw_values:
            if item is None:
                continue
            for pattern in flexible_like_patterns(original, str(item)):
                candidates.append((sql_quote(pattern),))
            for fragments in like_fragments_for_value(
                str(item),
                segment_lengths,
                segment_count,
                segments,
            ):
                pattern = build_like_pattern(original, fragments)
                candidates.append((sql_quote(pattern),))
    else:
        if slot.literals[0].kind == "number":
            nums = [value for value in (numeric_value(item) for item in raw_values) if value is not None]
            deduped_nums = []
            seen_num: set[int] = set()
            for value in [int(round(number)) for number in nums]:
                if value in seen_num:
                    continue
                seen_num.add(value)
                deduped_nums.append(value)
            for value in deduped_nums[:8]:
                candidates.append((str(int(value)),))
            if deduped_nums:
                candidates.append((str(deduped_nums[-1]),))
        else:
            values = []
            seen_value: set[str] = set()
            for item in raw_values:
                if item is None:
                    continue
                text = str(item)
                if text in seen_value:
                    continue
                seen_value.add(text)
                values.append(text)
            for value in values[:12]:
                candidates.append((sql_quote(value),))

    return unique_preserve_order(candidates)


def sql_like_matches(value: str, pattern: str) -> bool:
    regex = []
    for ch in str(pattern):
        if ch == "%":
            regex.append(".*")
        elif ch == "_":
            regex.append(".")
        else:
            regex.append(re.escape(ch))
    return re.fullmatch("".join(regex), str(value), flags=re.S) is not None


def slot_values_from_witness(
    slot: Slot,
    raw_value: Any,
    candidates: list[tuple[str, ...]] | None = None,
) -> tuple[str, ...] | None:
    if raw_value is None:
        return None
    if slot.kind == "like":
        # A witness row proves the positive LIKE can be made satisfiable. For
        # NOT LIKE predicates, prefer a real existing literal that does not
        # match the witness row. Only fall back to a sentinel if mining gives
        # us no safe alternative.
        raw_text = str(raw_value)
        if "NOT" in slot.operator.upper():
            original = original_slot_values(slot)
            original_pattern = sql_unquote(original[0])
            if not sql_like_matches(raw_text, original_pattern):
                return None
            for candidate in candidates or []:
                pattern = sql_unquote(candidate[0])
                if not sql_like_matches(raw_text, pattern):
                    return None if candidate == original else candidate
            return (sql_quote("%__mirage_no_such_pattern__%"),)
        original = sql_unquote(slot.literals[0].sql_text)
        patterns = flexible_like_patterns(original, str(raw_value))
        return (sql_quote(patterns[0] if patterns else str(raw_value)),)
    if slot.kind == "between":
        number = numeric_value(raw_value)
        if number is None:
            return None
        try:
            low_original = float(slot.literals[0].sql_text)
            high_original = float(slot.literals[1].sql_text)
        except Exception:
            low_original = high_original = float(number)
        width = max(0, int(round(high_original - low_original)))
        high = int(round(number))
        low = high - width
        return (str(low), str(high))
    if slot.kind == "in":
        literal = slot.literals[0]
        value = sql_quote(str(raw_value)) if literal.kind == "string" else str(int(round(float(raw_value))))
        values = list(original_slot_values(slot))
        values[0] = value
        return tuple(values)

    literal = slot.literals[0]
    if literal.kind == "string":
        number = numeric_value(raw_value)
        if number is not None and slot.operator in {"<", "<=", ">", ">="}:
            if slot.operator == "<":
                number += 0.1
            elif slot.operator == ">":
                number -= 0.1
            return (sql_quote(f"{number:.1f}"),)
        if slot.operator in {"!=", "<>"}:
            original = original_slot_values(slot)
            original_value = sql_unquote(original[0])
            if str(raw_value) != original_value:
                return None
            for candidate in candidates or []:
                candidate_value = sql_unquote(candidate[0])
                if str(raw_value) != candidate_value:
                    return None if candidate == original else candidate
            return (sql_quote(f"__mirage_not_{str(raw_value)}__"),)
        return (sql_quote(str(raw_value)),)

    number = numeric_value(raw_value)
    if number is None:
        return None
    value = int(round(number))
    if slot.operator in {"!=", "<>"}:
        original = original_slot_values(slot)
        try:
            if value != int(round(float(original[0]))):
                return None
        except Exception:
            pass
        for candidate in candidates or []:
            try:
                candidate_value = int(round(float(candidate[0])))
            except Exception:
                continue
            if value != candidate_value:
                return None if candidate == original else candidate
        return (str(value + 1),)
    if slot.operator == ">":
        value -= 1
    elif slot.operator == "<":
        value += 1
    return (str(value),)


def mine_joint_witness_states(
    *,
    base_sql: str,
    slots: list[Slot],
    runner: PostgresRunner,
    limit: int,
    slot_candidates: dict[int, list[tuple[str, ...]]] | None = None,
) -> list[dict[int, tuple[str, ...]]]:
    if not slots:
        return []
    try:
        relaxed_sql = render_sql(
            base_sql,
            slots,
            {},
            slot_overrides={slot.index: "TRUE" for slot in slots},
        )
        from_clause, where_clause = extract_from_where(relaxed_sql)
    except Exception:
        return []

    select_list = ",\n            ".join(f"{slot.expr} AS v{slot.index}" for slot in slots)
    non_null = "\n          AND ".join(f"{slot.expr} IS NOT NULL" for slot in slots)
    query = f"""
    SELECT COALESCE(json_agg(row_to_json(s))::text, '[]')
    FROM (
        SELECT {select_list}
        {from_clause}
        WHERE {where_clause}
          AND {non_null}
        LIMIT {int(limit)}
    ) s;
    """
    try:
        payload = runner.json_scalar(with_statement_timeout(query))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []

    states: list[dict[int, tuple[str, ...]]] = []
    seen: set[tuple[tuple[int, tuple[str, ...]], ...]] = set()
    for row in payload:
        if not isinstance(row, dict):
            continue
        state: dict[int, tuple[str, ...]] = {}
        for slot in slots:
            values = slot_values_from_witness(
                slot,
                row.get(f"v{slot.index}"),
                (slot_candidates or {}).get(slot.index),
            )
            if values and values != original_slot_values(slot):
                state[slot.index] = values
        key = tuple(sorted(state.items()))
        if not state or key in seen:
            continue
        seen.add(key)
        states.append(state)
    return states


def numeric_similarity(original: str, candidate: str) -> float:
    try:
        orig_value = float(original)
        cand_value = float(candidate)
    except Exception:
        return 0.0
    scale = max(abs(orig_value), abs(cand_value), 1.0)
    delta = abs(orig_value - cand_value) / scale
    return max(0.0, 1.0 - delta)


def string_similarity(original: str, candidate: str) -> float:
    orig_text = sql_unquote(original).lower()
    cand_text = sql_unquote(candidate).lower()
    ratio = difflib.SequenceMatcher(None, orig_text, cand_text).ratio()
    orig_tokens = {token for token in re.split(r"[^a-z0-9]+", orig_text) if len(token) >= 2}
    cand_tokens = {token for token in re.split(r"[^a-z0-9]+", cand_text) if len(token) >= 2}
    if not orig_tokens and not cand_tokens:
        return ratio
    overlap = len(orig_tokens & cand_tokens)
    union = len(orig_tokens | cand_tokens) or 1
    jaccard = overlap / union
    return (ratio + jaccard) / 2.0


def slot_value_similarity(slot: Slot, values: tuple[str, ...]) -> float:
    originals = original_slot_values(slot)
    if len(values) != len(originals):
        return 0.0
    scores: list[float] = []
    for original, candidate, literal in zip(originals, values, slot.literals):
        if candidate == original:
            scores.append(1.0)
        elif literal.kind == "number":
            scores.append(numeric_similarity(original, candidate))
        else:
            scores.append(string_similarity(original, candidate))
    return sum(scores) / len(scores) if scores else 1.0


def state_similarity_score(slots: list[Slot], state: dict[int, tuple[str, ...]]) -> float:
    if not slots:
        return 1.0
    total = 0.0
    for slot in slots:
        total += slot_value_similarity(slot, state.get(slot.index, original_slot_values(slot)))
    return total / len(slots)


def state_change_count(slots: list[Slot], state: dict[int, tuple[str, ...]]) -> int:
    changed = 0
    for slot in slots:
        if state.get(slot.index) and state[slot.index] != original_slot_values(slot):
            changed += 1
    return changed


def evaluate_state(
    *,
    query_sql: str,
    slots: list[Slot],
    state: dict[int, tuple[str, ...]],
    runner: PostgresRunner,
    cache: dict[tuple[tuple[int, tuple[str, ...]], ...], tuple[int, str | None]],
) -> tuple[int, str | None]:
    state_key = tuple(sorted(state.items()))
    cached = cache.get(state_key)
    if cached is not None:
        return cached
    sql = render_sql(query_sql, slots, state)
    try:
        has_rows = runner.scalar_int(with_statement_timeout(strip_to_exists(sql)))
        if has_rows:
            actual = runner.scalar_int(strip_to_count(sql) + ";")
        else:
            actual = 0
        result = (actual, None)
    except Exception as exc:
        result = (0, str(exc).splitlines()[0][:240])
    cache[state_key] = result
    return result


def evaluate_relaxed_support(
    *,
    query_sql: str,
    slots: list[Slot],
    state: dict[int, tuple[str, ...]],
    runner: PostgresRunner,
    relax_after_index: int,
    cache: dict[tuple[tuple[tuple[int, tuple[str, ...]], ...], int], tuple[int, str | None]],
) -> tuple[int, str | None]:
    state_key = tuple(sorted(state.items()))
    cache_key = (state_key, int(relax_after_index))
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    overrides = {
        slot.index: "TRUE"
        for slot in slots
        if slot.index > relax_after_index
    }
    sql = render_sql(query_sql, slots, state, slot_overrides=overrides or None)
    try:
        actual = runner.scalar_int(with_statement_timeout(strip_to_exists(sql)))
        result = (actual, None)
    except Exception as exc:
        result = (0, str(exc).splitlines()[0][:240])
    cache[cache_key] = result
    return result


def beam_search_query(
    *,
    query_id: str,
    original_sql: str,
    slots: list[Slot],
    runner: PostgresRunner,
    historical_state: dict[int, tuple[str, ...]],
    candidate_limit: int,
    beam_width: int,
    accept_witness_hit: bool,
    skip_joint_witness: bool,
) -> tuple[dict[int, tuple[str, ...]], int, list[str]]:
    evaluation_cache: dict[tuple[tuple[int, tuple[str, ...]], ...], tuple[int, str | None]] = {}
    relaxed_support_cache: dict[tuple[tuple[tuple[int, tuple[str, ...]], ...], int], tuple[int, str | None]] = {}
    slot_candidates: dict[int, list[tuple[str, ...]]] = {}
    notes: list[str] = []

    for slot in slots:
        local_values = mine_values_for_slot(
            slot=slot,
            base_sql=original_sql,
            base_state={},
            runner=runner,
            limit=candidate_limit,
        )
        historical_local_values = mine_values_for_slot(
            slot=slot,
            base_sql=original_sql,
            base_state=historical_state,
            runner=runner,
            limit=candidate_limit,
        ) if historical_state else []
        query_context_values = mine_context_values_for_slot(
            slot=slot,
            base_sql=original_sql,
            runner=runner,
            limit=candidate_limit,
        )
        token_match_values = mine_token_match_values_for_slot(
            slot=slot,
            runner=runner,
            limit=candidate_limit,
        )
        global_values = mine_global_values_for_slot(slot=slot, runner=runner, limit=candidate_limit)
        candidates = build_slot_candidates(
            slot=slot,
            query_local_values=local_values,
            historical_local_values=historical_local_values,
            query_context_values=query_context_values,
            token_match_values=token_match_values,
            global_values=global_values,
            historical_state=historical_state,
        )
        # LIKE/title/company predicates are often the rescue point for sparse
        # synthetic datasets. Keep a modestly wider candidate list than the
        # mining limit so freer literal replacements are not truncated away.
        per_slot_cap = max(candidate_limit, beam_width)
        if slot.kind in {"like", "cmp", "in"} and slot.literals and slot.literals[0].kind == "string":
            per_slot_cap = max(per_slot_cap, min(len(candidates), candidate_limit * 3))
        slot_candidates[slot.index] = candidates[:per_slot_cap]
        if len(candidates) <= 1:
            notes.append(f"{query_id}:{slot.expr} had no mined alternatives beyond the current literal set")

    seed_states = [dict()]
    if historical_state:
        seed_states.append(dict(historical_state))

    seed_beam: list[BeamCandidate] = []
    for seed in seed_states:
        actual, error = evaluate_state(
            query_sql=original_sql,
            slots=slots,
            state=seed,
            runner=runner,
            cache=evaluation_cache,
        )
        relaxed_support, _ = evaluate_relaxed_support(
            query_sql=original_sql,
            slots=slots,
            state=seed,
            runner=runner,
            relax_after_index=-1,
            cache=relaxed_support_cache,
        )
        seed_beam.append(
            BeamCandidate(
                state=seed,
                actual=actual,
                error=error,
                relaxed_support=relaxed_support,
            )
        )

    if accept_witness_hit:
        seed_hits = [item for item in seed_beam if item.actual > 0 and item.error is None]
        if seed_hits:
            best_seed = sorted(
                seed_hits,
                key=lambda item: (
                    math.log10(item.actual + 1.0),
                    state_similarity_score(slots, item.state),
                    -state_change_count(slots, item.state),
                ),
                reverse=True,
            )[0]
            if best_seed.state:
                notes.append(f"{query_id}: accepted historical/seed state before expensive witness mining")
            return best_seed.state, best_seed.actual, notes

    witness_states = [] if skip_joint_witness else mine_joint_witness_states(
        base_sql=original_sql,
        slots=slots,
        runner=runner,
        limit=max(beam_width * 4, candidate_limit * 3, 32),
        slot_candidates=slot_candidates,
    )
    for witness_state in witness_states:
        for slot_index, values in witness_state.items():
            candidates = slot_candidates.setdefault(slot_index, [original_slot_values(slots[slot_index])])
            if values not in candidates:
                candidates.append(values)

    seed_states.extend(dict(state) for state in witness_states)

    beam: list[BeamCandidate] = list(seed_beam)
    for seed in seed_states[2:]:
        actual, error = evaluate_state(
            query_sql=original_sql,
            slots=slots,
            state=seed,
            runner=runner,
            cache=evaluation_cache,
        )
        relaxed_support, _ = evaluate_relaxed_support(
            query_sql=original_sql,
            slots=slots,
            state=seed,
            runner=runner,
            relax_after_index=-1,
            cache=relaxed_support_cache,
        )
        beam.append(
            BeamCandidate(
                state=seed,
                actual=actual,
                error=error,
                relaxed_support=relaxed_support,
            )
        )

    def score(item: BeamCandidate) -> tuple[int, float, int, float, float, int]:
        return (
            1 if item.actual > 0 and item.error is None else 0,
            math.log10(item.actual + 1.0),
            1 if item.relaxed_support > 0 and item.error is None else 0,
            math.log10(item.relaxed_support + 1.0),
            state_similarity_score(slots, item.state),
            -state_change_count(slots, item.state),
        )

    def dedupe(items: list[BeamCandidate]) -> list[BeamCandidate]:
        best_by_state: dict[tuple[tuple[int, tuple[str, ...]], ...], BeamCandidate] = {}
        for item in items:
            state_key = tuple(sorted(item.state.items()))
            current = best_by_state.get(state_key)
            if current is None or score(item) > score(current):
                best_by_state[state_key] = item
        return list(best_by_state.values())

    beam = sorted(dedupe(beam), key=score, reverse=True)[:beam_width]
    if accept_witness_hit:
        witness_hits = [item for item in beam if item.actual > 0 and item.error is None]
        if witness_hits:
            best_hit = sorted(witness_hits, key=score, reverse=True)[0]
            if best_hit.state:
                notes.append(f"{query_id}: accepted coherent joined-row witness without exhaustive literal search")
            return best_hit.state, best_hit.actual, notes

    for slot in slots:
        expanded: list[BeamCandidate] = []
        for item in beam:
            state = item.state
            for candidate in slot_candidates.get(slot.index, [original_slot_values(slot)]):
                new_state = dict(state)
                if candidate == original_slot_values(slot):
                    new_state.pop(slot.index, None)
                else:
                    new_state[slot.index] = candidate
                actual, error = evaluate_state(
                    query_sql=original_sql,
                    slots=slots,
                    state=new_state,
                    runner=runner,
                    cache=evaluation_cache,
                )
                relaxed_support, _ = evaluate_relaxed_support(
                    query_sql=original_sql,
                    slots=slots,
                    state=new_state,
                    runner=runner,
                    relax_after_index=slot.index,
                    cache=relaxed_support_cache,
                )
                expanded.append(
                    BeamCandidate(
                        state=new_state,
                        actual=actual,
                        error=error,
                        relaxed_support=relaxed_support,
                    )
                )
        beam = sorted(dedupe(expanded), key=score, reverse=True)[:beam_width]

    best_candidate = beam[0]
    best_state = best_candidate.state
    best_error = best_candidate.error
    if best_error:
        notes.append(f"{query_id}: best candidate still raised postgres error: {best_error}")
        best_actual = 0
    elif best_candidate.actual == 0:
        relaxed_all_sql = render_sql(
            original_sql,
            slots,
            {},
            slot_overrides={slot.index: "TRUE" for slot in slots},
        )
        try:
            relaxed_count = runner.scalar_int(with_statement_timeout(strip_to_exists(relaxed_all_sql)))
        except Exception:
            relaxed_count = 0
        if relaxed_count == 0:
            notes.append(f"{query_id}: all-literals-relaxed query still returns zero rows (scale-limited)")
        else:
            notes.append(f"{query_id}: query remains zero after search but relaxed literal context has rows (value-limited)")
        best_actual = 0
    else:
        best_sql = render_sql(original_sql, slots, best_state)
        try:
            # Search only needs existence, but benchmark output needs the exact
            # cardinality. Count once for the accepted exact-structure query.
            best_actual = runner.scalar_int(strip_to_count(best_sql) + ";")
        except Exception as exc:
            best_error = str(exc).splitlines()[0][:240]
            notes.append(f"{query_id}: accepted candidate failed final count: {best_error}")
            best_actual = 0

    return best_state, best_actual, notes


def q_error(est: float, act: float) -> float:
    est = max(float(est), 0.5)
    act = max(float(act), 0.5)
    return max(est / act, act / est)


def explain_postgres_query(*, runner: PostgresRunner, query_sql: str) -> QueryPlanStats:
    explain_sql = f"EXPLAIN (ANALYZE, FORMAT JSON) {strip_to_count(query_sql)};"
    payload = runner.json_scalar(explain_sql)
    if not isinstance(payload, list) or not payload:
        raise RuntimeError("Unexpected EXPLAIN payload")
    root = payload[0]["Plan"]
    planning_ms = float(payload[0].get("Planning Time", 0.0))
    execution_ms = float(payload[0].get("Execution Time", 0.0))

    node = root
    while node.get("Node Type", "") in {"Aggregate", "Result", "Sort", "Limit", "Unique", "Gather", "Gather Merge"}:
        children = node.get("Plans", [])
        if not children:
            break
        node = children[0]

    est = float(node.get("Plan Rows", 1))
    loops = max(int(node.get("Actual Loops", 1)), 1)
    act = float(node.get("Actual Rows", 0)) * loops
    bias = "OVER" if est > act else ("UNDER" if est < act else "EXACT")
    if act == 0 and est == 0:
        bias = "EXACT"
    elif act == 0 and est > 0:
        bias = "OVER"

    actual_count = runner.scalar_int(strip_to_count(query_sql) + ";")
    return QueryPlanStats(
        actual_count=actual_count,
        sub_agg_estimate=est,
        sub_agg_actual=act,
        sub_agg_qerror=q_error(est, act),
        sub_agg_bias=bias,
        sub_agg_node=str(node.get("Node Type", "")),
        planning_ms=planning_ms,
        execution_ms=execution_ms,
        total_ms=planning_ms + execution_ms,
    )


def ensure_duckdb_binary(path: Path | None) -> Path | None:
    if path is not None and path.exists():
        return path
    for candidate in DEFAULT_DUCKDB_BIN_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def find_duckdb_path(dataset_dir: Path, explicit_path: Path | None) -> Path | None:
    if explicit_path is not None and explicit_path.exists():
        return explicit_path
    schema_dir = dataset_dir / "imdb_schema" if (dataset_dir / "imdb_schema").exists() else dataset_dir
    candidates = sorted(schema_dir.glob("*.duckdb"))
    return candidates[0] if candidates else None


def dataset_schema_dir(dataset_dir: Path) -> Path:
    candidate = dataset_dir / "imdb_schema"
    return candidate if candidate.exists() else dataset_dir


def copy_original_queries(
    *,
    source_query_dir: Path,
    target_dir: Path,
    include_ids: set[str] | None = None,
) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for source_path in sorted(source_query_dir.glob("*.sql"), key=sort_key):
        if source_path.name in EXCLUDED_QUERY_FILES:
            continue
        if include_ids is not None and source_path.stem not in include_ids:
            continue
        target_path = target_dir / source_path.name
        shutil.copy2(source_path, target_path)
        copied.append(target_path)
    return copied


def write_markdown_report(
    *,
    path: Path,
    dataset_id: str,
    manifest: dict[str, Any],
) -> None:
    summary = manifest["summary"]
    lines = [
        "# JOB Exact V1 Coverage Report",
        "",
        f"- dataset: `{dataset_id}`",
        f"- canonical queries: `{summary['query_count']}`",
        f"- structure guard passed: `{summary['structure_guard_passed']}/{summary['query_count']}`",
        f"- postgres non-zero: `{summary['postgres_non_zero']}/{summary['query_count']}`",
        f"- postgres zero: `{summary['postgres_zero']}/{summary['query_count']}`",
        f"- postgres errors: `{summary['postgres_errors']}`",
        f"- duckdb non-zero: `{summary['duckdb_non_zero']}/{summary['query_count']}`",
        f"- duckdb zero: `{summary['duckdb_zero']}/{summary['query_count']}`",
        f"- duckdb errors: `{summary['duckdb_errors']}`",
        "",
        "## Postgres Q-Error",
        "",
        f"- median sub-aggregate q-error: `{summary['postgres_qerror_median']}`",
        f"- mean sub-aggregate q-error: `{summary['postgres_qerror_mean']}`",
        f"- max sub-aggregate q-error: `{summary['postgres_qerror_max']}`",
        "",
        "## Remaining Zero Queries",
        "",
    ]
    zero_queries = [query for query in manifest["queries"] if int(query["actual_count"]) == 0]
    if not zero_queries:
        lines.append("- none")
    else:
        for query in zero_queries:
            notes = "; ".join(query.get("blocker_notes", [])) or "no blocker note"
            lines.append(f"- `{query['query_id']}`: {notes}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_results(results: list[QueryResult]) -> dict[str, Any]:
    postgres_non_zero = sum(1 for result in results if result.status == "OK" and result.actual_count > 0)
    postgres_zero = sum(1 for result in results if result.status == "OK" and result.actual_count == 0)
    postgres_errors = sum(1 for result in results if result.status == "ERROR")
    duckdb_non_zero = sum(1 for result in results if result.duckdb_status == "OK" and (result.duckdb_actual_count or 0) > 0)
    duckdb_zero = sum(1 for result in results if result.duckdb_status == "OK" and (result.duckdb_actual_count or 0) == 0)
    duckdb_errors = sum(1 for result in results if result.duckdb_status == "ERROR")
    structure_guard_passed = sum(1 for result in results if result.structure_guard_passed)
    qerrors = [result.postgres_plan.sub_agg_qerror for result in results if result.postgres_plan is not None]
    ordered_qerrors = sorted(qerrors)
    return {
        "query_count": len(results),
        "structure_guard_passed": structure_guard_passed,
        "postgres_non_zero": postgres_non_zero,
        "postgres_zero": postgres_zero,
        "postgres_errors": postgres_errors,
        "duckdb_non_zero": duckdb_non_zero,
        "duckdb_zero": duckdb_zero,
        "duckdb_errors": duckdb_errors,
        "postgres_qerror_median": round(ordered_qerrors[len(ordered_qerrors) // 2], 4) if ordered_qerrors else None,
        "postgres_qerror_mean": round(sum(ordered_qerrors) / len(ordered_qerrors), 4) if ordered_qerrors else None,
        "postgres_qerror_max": round(max(ordered_qerrors), 4) if ordered_qerrors else None,
    }


def emit_run_outputs(
    *,
    run_out_dir: Path,
    dataset_id: str,
    source_query_dir: Path,
    dataset_dir: Path,
    schema_dir: Path,
    db_name: str,
    duckdb_path: Path | None,
    manifest_queries: list[dict[str, Any]],
    pg_rows: list[dict[str, Any]],
    duck_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = summarize_results(
        [
            QueryResult(
                query_id=item["query_id"],
                structure_checksum=item["structure_checksum"],
                structure_guard_passed=bool(item["structure_guard_passed"]),
                actual_count=int(item["actual_count"]),
                status=str(item["status"]),
                blocker_notes=list(item.get("blocker_notes", [])),
                changed_slots=int(item.get("changed_slots", 0)),
                postgres_plan=None
                if item.get("postgres_plan") is None
                else QueryPlanStats(
                    actual_count=int(item["actual_count"]),
                    sub_agg_estimate=float(item["postgres_plan"]["sub_agg_estimate"]),
                    sub_agg_actual=float(item["postgres_plan"]["sub_agg_actual"]),
                    sub_agg_qerror=float(item["postgres_plan"]["sub_agg_qerror"]),
                    sub_agg_bias=str(item["postgres_plan"]["sub_agg_bias"]),
                    sub_agg_node=str(item["postgres_plan"]["sub_agg_node"]),
                    planning_ms=float(item["postgres_plan"]["planning_ms"]),
                    execution_ms=float(item["postgres_plan"]["execution_ms"]),
                    total_ms=float(item["postgres_plan"]["total_ms"]),
                ),
                postgres_error=item.get("postgres_error"),
                duckdb_status=str(item.get("duckdb_status", "SKIPPED")),
                duckdb_actual_count=item.get("duckdb_actual_count"),
                duckdb_ms=item.get("duckdb_ms"),
                duckdb_error=item.get("duckdb_error"),
                original_literals=list(item.get("original_literals", [])),
                adapted_literals=list(item.get("adapted_literals", [])),
            )
            for item in manifest_queries
        ]
    )
    duplicate_rows = adapted_duplicate_rows(manifest_queries)
    duplicate_csv = run_out_dir / "adapted_duplicate_sql.csv"
    write_csv(
        duplicate_csv,
        duplicate_rows,
        [
            "query_id",
            "duplicate_group_query_ids",
            "adapted_sql_sha256",
            "original_sql_sha256",
            "original_group_was_already_duplicate",
            "adapted_path",
            "original_path",
        ],
    )
    manifest = {
        "benchmark_id": "job_exact_v1",
        "dataset_id": dataset_id,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_query_dir": str(source_query_dir),
        "dataset_dir": str(dataset_dir),
        "schema_dir": str(schema_dir),
        "db_name": db_name,
        "duckdb_path": None if duckdb_path is None else str(duckdb_path),
        "summary": summary,
        "adapted_duplicate_sql_count": len(duplicate_rows),
        "adapted_duplicate_sql_csv": str(duplicate_csv),
        "queries": manifest_queries,
    }

    write_json(run_out_dir / "manifest.json", manifest)
    write_json(run_out_dir / "postgres_results.json", pg_rows)
    write_json(run_out_dir / "duckdb_results.json", duck_rows)
    write_csv(
        run_out_dir / "postgres_results.csv",
        pg_rows,
        [
            "query",
            "status",
            "actual_count",
            "structure_checksum",
            "structure_guard_passed",
            "changed_slots",
            "sub_agg_estimate",
            "sub_agg_actual",
            "sub_agg_qerror",
            "sub_agg_bias",
            "sub_agg_node",
            "planning_ms",
            "execution_ms",
            "total_ms",
            "error",
        ],
    )
    write_csv(
        run_out_dir / "duckdb_results.csv",
        duck_rows,
        ["query", "status", "actual_count", "ms", "error"],
    )
    write_markdown_report(
        path=run_out_dir / "coverage_report.md",
        dataset_id=dataset_id,
        manifest=manifest,
    )
    return manifest


def slot_manifest_entry(slot: Slot, values: tuple[str, ...]) -> dict[str, Any]:
    return {
        "expr": slot.expr,
        "operator": slot.operator,
        "kind": slot.kind,
        "values": [sql_unquote(value) if value.startswith("'") else value for value in values],
        "sql_values": list(values),
        "arity": slot.arity,
    }


def adapted_duplicate_rows(manifest_queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    original_hashes: dict[str, str] = {}
    for item in manifest_queries:
        adapted_path = Path(str(item.get("adapted_path", "")))
        original_path = Path(str(item.get("original_path", "")))
        if not adapted_path.exists() or not original_path.exists():
            continue
        adapted_hash = normalized_sql_content_hash(adapted_path.read_text(encoding="utf-8"))
        original_hashes[str(item.get("query_id", ""))] = normalized_sql_content_hash(
            original_path.read_text(encoding="utf-8")
        )
        groups.setdefault(adapted_hash, []).append(item)

    rows: list[dict[str, Any]] = []
    for adapted_hash, items in sorted(groups.items()):
        if len(items) <= 1:
            continue
        query_ids = [str(item.get("query_id", "")) for item in items]
        original_group_hashes = {original_hashes.get(query_id, "") for query_id in query_ids}
        for item in items:
            query_id = str(item.get("query_id", ""))
            rows.append(
                {
                    "query_id": query_id,
                    "duplicate_group_query_ids": ",".join(query_ids),
                    "adapted_sql_sha256": adapted_hash,
                    "original_sql_sha256": original_hashes.get(query_id, ""),
                    "original_group_was_already_duplicate": len(original_group_hashes) == 1,
                    "adapted_path": item.get("adapted_path", ""),
                    "original_path": item.get("original_path", ""),
                }
            )
    return rows


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir).resolve()
    source_query_dir = Path(args.source_query_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    historical_dir = Path(args.historical_query_dir).resolve() if args.historical_query_dir else None
    dataset_id = dataset_dir.name
    query_ids = None
    if args.query_ids:
        query_ids = {
            item.strip()
            for item in str(args.query_ids).split(",")
            if item.strip()
        }

    original_out_dir = out_dir / "original_queries"
    adapted_out_dir = out_dir / "adapted_queries" / dataset_id
    run_out_dir = out_dir / "runs" / dataset_id
    original_out_dir.mkdir(parents=True, exist_ok=True)
    adapted_out_dir.mkdir(parents=True, exist_ok=True)
    run_out_dir.mkdir(parents=True, exist_ok=True)

    copied_queries = copy_original_queries(
        source_query_dir=source_query_dir,
        target_dir=original_out_dir,
        include_ids=query_ids,
    )
    if query_ids is None and len(copied_queries) != 113:
        print(f"warning: expected 113 canonical JOB queries, found {len(copied_queries)}")

    runner = PostgresRunner(
        container=args.pg_container,
        db_name=args.db_name,
        user=args.pg_user,
        persistent=bool(args.persistent_pg),
    )
    runner.analyze()

    duckdb_binary = ensure_duckdb_binary(Path(args.duckdb_bin).resolve() if args.duckdb_bin else None)
    duckdb_path = find_duckdb_path(dataset_dir, Path(args.duckdb_path).resolve() if args.duckdb_path else None)
    duckdb_runner = None
    if not args.skip_duckdb and duckdb_binary and duckdb_path:
        duckdb_runner = DuckDBRunner(binary=duckdb_binary, db_path=duckdb_path)

    results: list[QueryResult] = []
    manifest_queries: list[dict[str, Any]] = []
    pg_rows: list[dict[str, Any]] = []
    duck_rows: list[dict[str, Any]] = []

    for index, original_path in enumerate(copied_queries, start=1):
        query_id = original_path.stem
        print(f"[{index}/{len(copied_queries)}] adapting {query_id}", flush=True)
        original_sql = original_path.read_text(encoding="utf-8")
        slots = extract_slots(original_sql)
        historical_state = load_historical_state(
            historical_query_dir=historical_dir,
            query_id=query_id,
            original_sql=original_sql,
            original_slots=slots,
        )
        final_state, actual_count, notes = beam_search_query(
            query_id=query_id,
            original_sql=original_sql,
            slots=slots,
            runner=runner,
            historical_state=historical_state,
            candidate_limit=int(args.candidate_limit),
            beam_width=int(args.beam_width),
            accept_witness_hit=bool(args.accept_witness_hit),
            skip_joint_witness=bool(args.skip_joint_witness),
        )
        adapted_sql = render_sql(original_sql, slots, final_state)
        structure_checksum = structure_fingerprint(original_sql)
        structure_guard_passed = structure_fingerprint(adapted_sql) == structure_checksum
        adapted_path = adapted_out_dir / original_path.name
        adapted_path.write_text(adapted_sql, encoding="utf-8")

        postgres_plan: QueryPlanStats | None = None
        postgres_error: str | None = None
        status = "OK"
        if not args.postgres_count_only:
            try:
                postgres_plan = explain_postgres_query(runner=runner, query_sql=adapted_sql)
                actual_count = postgres_plan.actual_count
            except Exception as exc:
                status = "ERROR"
                postgres_error = str(exc).splitlines()[0][:240]
                notes.append(f"{query_id}: postgres explain failed: {postgres_error}")

        duckdb_status = "SKIPPED"
        duckdb_actual_count: int | None = None
        duckdb_ms: float | None = None
        duckdb_error: str | None = None
        if duckdb_runner is not None:
            try:
                duckdb_actual_count, duckdb_ms = duckdb_runner.count_query(adapted_sql)
                duckdb_status = "OK"
            except Exception as exc:
                duckdb_status = "ERROR"
                duckdb_error = str(exc).splitlines()[0][:240]

        result = QueryResult(
            query_id=query_id,
            structure_checksum=structure_checksum,
            structure_guard_passed=structure_guard_passed,
            actual_count=actual_count,
            status=status,
            blocker_notes=notes,
            changed_slots=state_change_count(slots, final_state),
            postgres_plan=postgres_plan,
            postgres_error=postgres_error,
            duckdb_status=duckdb_status,
            duckdb_actual_count=duckdb_actual_count,
            duckdb_ms=duckdb_ms,
            duckdb_error=duckdb_error,
            original_literals=[slot_manifest_entry(slot, original_slot_values(slot)) for slot in slots],
            adapted_literals=[slot_manifest_entry(slot, final_state.get(slot.index, original_slot_values(slot))) for slot in slots],
        )
        results.append(result)
        print(
            f"[{index}/{len(copied_queries)}] finished {query_id}: "
            f"actual_count={actual_count}, changed_slots={result.changed_slots}, status={status}",
            flush=True,
        )

        manifest_queries.append(
            {
                "query_id": query_id,
                "original_path": str(original_path),
                "adapted_path": str(adapted_path),
                "structure_checksum": structure_checksum,
                "structure_guard_passed": structure_guard_passed,
                "actual_count": actual_count,
                "status": status,
                "changed_slots": result.changed_slots,
                "blocker_notes": notes,
                "postgres_error": postgres_error,
                "duckdb_status": duckdb_status,
                "duckdb_actual_count": duckdb_actual_count,
                "duckdb_ms": duckdb_ms,
                "duckdb_error": duckdb_error,
                "postgres_plan": None
                if postgres_plan is None
                else {
                    "sub_agg_estimate": postgres_plan.sub_agg_estimate,
                    "sub_agg_actual": postgres_plan.sub_agg_actual,
                    "sub_agg_qerror": postgres_plan.sub_agg_qerror,
                    "sub_agg_bias": postgres_plan.sub_agg_bias,
                    "sub_agg_node": postgres_plan.sub_agg_node,
                    "planning_ms": postgres_plan.planning_ms,
                    "execution_ms": postgres_plan.execution_ms,
                    "total_ms": postgres_plan.total_ms,
                },
                "original_literals": result.original_literals,
                "adapted_literals": result.adapted_literals,
            }
        )

        pg_rows.append(
            {
                "query": query_id,
                "status": status,
                "actual_count": actual_count,
                "structure_checksum": structure_checksum,
                "structure_guard_passed": structure_guard_passed,
                "changed_slots": result.changed_slots,
                "sub_agg_estimate": None if postgres_plan is None else round(postgres_plan.sub_agg_estimate, 4),
                "sub_agg_actual": None if postgres_plan is None else round(postgres_plan.sub_agg_actual, 4),
                "sub_agg_qerror": None if postgres_plan is None else round(postgres_plan.sub_agg_qerror, 4),
                "sub_agg_bias": None if postgres_plan is None else postgres_plan.sub_agg_bias,
                "sub_agg_node": None if postgres_plan is None else postgres_plan.sub_agg_node,
                "planning_ms": None if postgres_plan is None else round(postgres_plan.planning_ms, 4),
                "execution_ms": None if postgres_plan is None else round(postgres_plan.execution_ms, 4),
                "total_ms": None if postgres_plan is None else round(postgres_plan.total_ms, 4),
                "error": postgres_error,
            }
        )
        duck_rows.append(
            {
                "query": query_id,
                "status": duckdb_status,
                "actual_count": duckdb_actual_count,
                "ms": None if duckdb_ms is None else round(duckdb_ms, 4),
                "error": duckdb_error,
            }
        )
        emit_run_outputs(
            run_out_dir=run_out_dir,
            dataset_id=dataset_id,
            source_query_dir=source_query_dir,
            dataset_dir=dataset_dir,
            schema_dir=dataset_schema_dir(dataset_dir),
            db_name=args.db_name,
            duckdb_path=duckdb_path,
            manifest_queries=manifest_queries,
            pg_rows=pg_rows,
            duck_rows=duck_rows,
        )

    manifest = emit_run_outputs(
        run_out_dir=run_out_dir,
        dataset_id=dataset_id,
        source_query_dir=source_query_dir,
        dataset_dir=dataset_dir,
        schema_dir=dataset_schema_dir(dataset_dir),
        db_name=args.db_name,
        duckdb_path=duckdb_path,
        manifest_queries=manifest_queries,
        pg_rows=pg_rows,
        duck_rows=duck_rows,
    )
    duplicates = adapted_duplicate_rows(manifest_queries)
    if duplicates:
        duplicate_csv = run_out_dir / "adapted_duplicate_sql.csv"
        if bool(args.fail_on_adapted_duplicates) and any(
            not bool(row.get("original_group_was_already_duplicate"))
            for row in duplicates
        ):
            runner.close()
            raise RuntimeError(
                f"Adapted query corpus contains {len(duplicates)} duplicate rendered SQL rows; "
                f"see {duplicate_csv}"
            )
    runner.close()
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and validate an exact-structure JOB benchmark corpus for a synthetic IMDb dataset."
    )
    parser.add_argument("--source-query-dir", required=True, help="Canonical JOB SQL directory.")
    parser.add_argument("--dataset-dir", required=True, help="Dataset run directory, e.g. test32.")
    parser.add_argument("--db-name", required=True, help="Postgres database name used as the truth path.")
    parser.add_argument("--out-dir", required=True, help="Benchmark corpus root output directory.")
    parser.add_argument(
        "--historical-query-dir",
        default=str(Path(__file__).resolve().parents[2] / "test19" / "job_adapted"),
        help="Optional historical adapted corpus used only as a seed hint source.",
    )
    parser.add_argument("--pg-container", default=DEFAULT_PG_CONTAINER, help="Docker container name for Postgres.")
    parser.add_argument("--pg-user", default=DEFAULT_PG_USER, help="Postgres user inside the container.")
    parser.add_argument("--duckdb-path", default=None, help="Optional explicit DuckDB database file.")
    parser.add_argument("--duckdb-bin", default=None, help="Optional explicit DuckDB CLI binary.")
    parser.add_argument("--beam-width", type=int, default=6, help="Beam width for bounded literal search.")
    parser.add_argument("--candidate-limit", type=int, default=8, help="Per-slot mined candidate limit.")
    parser.add_argument(
        "--query-ids",
        default=None,
        help="Optional comma-separated query IDs for focused probes, e.g. 6a,7a,15a.",
    )
    parser.add_argument(
        "--skip-duckdb",
        action="store_true",
        help="Skip DuckDB execution checks for faster adaptation-debug probes.",
    )
    parser.add_argument(
        "--postgres-count-only",
        action="store_true",
        help="Skip Postgres EXPLAIN and keep only actual COUNT(*) results for faster probes.",
    )
    parser.add_argument(
        "--persistent-pg",
        action="store_true",
        help="Use one persistent psql session instead of one-shot docker exec calls. Faster, but less isolated.",
    )
    parser.add_argument(
        "--accept-witness-hit",
        action="store_true",
        help="Accept the best nonzero coherent joined-row witness before exhaustive beam expansion.",
    )
    parser.add_argument(
        "--skip-joint-witness",
        action="store_true",
        help="Skip the broad relaxed joined-row witness miner and rely on mined slot candidates/seed states.",
    )
    parser.add_argument(
        "--fail-on-adapted-duplicates",
        action="store_true",
        help="Fail after writing outputs if distinct source queries collapse to identical adapted SQL.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    manifest = run_benchmark(args)
    summary = manifest["summary"]
    print(
        json.dumps(
            {
                "dataset_id": manifest["dataset_id"],
                "query_count": summary["query_count"],
                "postgres_non_zero": summary["postgres_non_zero"],
                "postgres_zero": summary["postgres_zero"],
                "postgres_errors": summary["postgres_errors"],
                "duckdb_non_zero": summary["duckdb_non_zero"],
                "duckdb_zero": summary["duckdb_zero"],
                "duckdb_errors": summary["duckdb_errors"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
