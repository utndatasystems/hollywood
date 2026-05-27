"""
Arrow IPC (Feather v2) streaming sink for the Mirage pipeline.

Provides:
  - ArrowSink: batched streaming writer (lz4 compression by default)
  - PA_SCHEMAS: PyArrow schemas for all TABLE_DEFS tables
  - read_table(): reads .arrow (Feather) or .csv with auto-detection
"""
from __future__ import annotations

import os
from typing import Any

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.feather as feather
import pandas as pd

try:
    import polars as pl
except Exception:  # pragma: no cover - optional at runtime
    pl = None

from schema import get_auto_pk_tables


# ═══════════════════════════════════════════════════════════════════════
# PYARROW SCHEMAS  (one per TABLE_DEFS entry)
# ═══════════════════════════════════════════════════════════════════════
#
# Type conventions:
#   *_id          → int64 (nullable for optional FKs)
#   *_usd / *_pct → float64
#   *_date        → string  (ISO date strings)
#   rating* / sentiment → float64
#   integer counts → int32 or int64
#   everything else → string

PA_SCHEMAS: dict[str, pa.Schema] = {

    # ─── Core tables ──────────────────────────────────────────────────
    "movie": pa.schema([
        ("title_id", pa.int64()),
        ("title", pa.string()),
        ("year", pa.int32()),
        ("country", pa.string()),
        ("language", pa.string()),
        ("original_language", pa.string()),
        ("aspect_ratio", pa.string()),
        ("color_format", pa.string()),
        ("genre", pa.string()),
        ("production_tier", pa.string()),
        ("budget_usd", pa.float64()),
        ("box_office_usd", pa.float64()),
        ("runtime_minutes", pa.int32()),
        ("rating", pa.float64()),
        ("num_votes", pa.int64()),
        ("certification", pa.string()),
        ("tagline", pa.string()),
        ("plot_summary", pa.string()),
        ("franchise_id", pa.string()),
        ("installment_no", pa.int32()),
        ("award_campaign_strength", pa.float64()),
        ("seed", pa.int64()),
        ("snapshot_id", pa.string()),
    ]),
    "cast_info": pa.schema([
        ("title_id", pa.int64()),
        ("person_id", pa.int64()),
        ("character_name", pa.string()),
        ("character_description", pa.string()),
        ("billing_order", pa.int32()),
        ("archetype", pa.string()),
        ("screen_time_minutes", pa.float64()),
        ("salary_usd", pa.float64()),
    ]),
    "movie_directors": pa.schema([
        ("title_id", pa.int64()),
        ("director_id", pa.int64()),
    ]),
    "movie_companies": pa.schema([
        ("title_id", pa.int64()),
        ("company_id", pa.int64()),
        ("role", pa.string()),
    ]),
    "movie_keyword": pa.schema([
        ("title_id", pa.int64()),
        ("keyword_id", pa.int64()),
    ]),
    "movie_crew": pa.schema([
        ("crew_id", pa.int64()),
        ("title_id", pa.int64()),
        ("person_id", pa.int64()),
        ("crew_role", pa.string()),
        ("credit_order", pa.int32()),
        ("department", pa.string()),
    ]),

    # ─── Secondary tables (streamable — never read back during generation) ─
    "release_dates": pa.schema([
        ("release_id", pa.int64()),
        ("title_id", pa.int64()),
        ("country", pa.string()),
        ("release_type", pa.string()),
        ("release_date", pa.string()),
    ]),
    "box_office_weekly": pa.schema([
        ("box_week_id", pa.int64()),
        ("title_id", pa.int64()),
        ("week_no", pa.int32()),
        ("week_start_date", pa.string()),
        ("gross_usd_total", pa.float64()),
        ("gross_usd_domestic", pa.float64()),
        ("gross_usd_international", pa.float64()),
    ]),
    "box_office_by_territory": pa.schema([
        ("territory_id", pa.int64()),
        ("title_id", pa.int64()),
        ("territory", pa.string()),
        ("gross_usd", pa.float64()),
        ("opening_weekend_usd", pa.float64()),
        ("share_pct", pa.float64()),
    ]),
    "box_office_daily": pa.schema([
        ("daily_id", pa.int64()),
        ("title_id", pa.int64()),
        ("day_number", pa.int32()),
        ("date", pa.string()),
        ("gross_usd_domestic", pa.float64()),
        ("gross_usd_international", pa.float64()),
        ("gross_usd_total", pa.float64()),
        ("cumulative_usd", pa.float64()),
    ]),
    "reviews": pa.schema([
        ("review_id", pa.int64()),
        ("title_id", pa.int64()),
        ("reviewer_type", pa.string()),
        ("source", pa.string()),
        ("rating_10", pa.float64()),
        ("sentiment", pa.float64()),
        ("review_date", pa.string()),
        ("review_text", pa.string()),
    ]),
    "awards": pa.schema([
        ("award_id", pa.int64()),
        ("title_id", pa.int64()),
        ("award_year", pa.int32()),
        ("ceremony", pa.string()),
        ("category", pa.string()),
        ("outcome", pa.string()),
        ("person_id", pa.int64()),
    ]),
    "locations": pa.schema([
        ("location_id", pa.int64()),
        ("title_id", pa.int64()),
        ("location_order", pa.int32()),
        ("city", pa.string()),
        ("country", pa.string()),
        ("location_type", pa.string()),
    ]),
    "alternate_titles": pa.schema([
        ("title_id", pa.int64()),
        ("language", pa.string()),
        ("alt_title", pa.string()),
    ]),
    "ratings_breakdown": pa.schema([
        ("breakdown_id", pa.int64()),
        ("title_id", pa.int64()),
        ("age_group", pa.string()),
        ("gender", pa.string()),
        ("vote_count", pa.int64()),
        ("avg_rating", pa.float64()),
    ]),
    "movie_links": pa.schema([
        ("title_id", pa.int64()),
        ("linked_title_id", pa.int64()),
        ("link_type", pa.string()),
    ]),

    # ─── Global tables ────────────────────────────────────────────────
    "person_demographics": pa.schema([
        ("person_id", pa.int64()),
        ("birth_date", pa.string()),
        ("death_date", pa.string()),
        ("birth_city", pa.string()),
        ("birth_country", pa.string()),
        ("height_cm", pa.float64()),
    ]),
    "tv_series": pa.schema([
        ("series_id", pa.int64()),
        ("title", pa.string()),
        ("genre", pa.string()),
        ("country", pa.string()),
        ("language", pa.string()),
        ("network", pa.string()),
        ("network_company_id", pa.int64()),
        ("creator_person_id", pa.int64()),
        ("year_start", pa.int32()),
        ("year_end", pa.int32()),
        ("status", pa.string()),
        ("total_seasons", pa.int32()),
        ("overall_rating", pa.float64()),
        ("content_rating", pa.string()),
        ("plot_summary", pa.string()),
    ]),
    "seasons": pa.schema([
        ("season_id", pa.int64()),
        ("series_id", pa.int64()),
        ("season_number", pa.int32()),
        ("year", pa.int32()),
        ("num_episodes", pa.int32()),
        ("avg_rating", pa.float64()),
    ]),
    "episodes": pa.schema([
        ("episode_id", pa.int64()),
        ("season_id", pa.int64()),
        ("series_id", pa.int64()),
        ("episode_number", pa.int32()),
        ("title", pa.string()),
        ("runtime_minutes", pa.int32()),
        ("rating", pa.float64()),
        ("director_person_id", pa.int64()),
        ("air_date", pa.string()),
        ("viewership_millions", pa.float64()),
        ("writer_person_id", pa.int64()),
        ("description", pa.string()),
    ]),
    "company_links": pa.schema([
        ("company_id_1", pa.int64()),
        ("company_id_2", pa.int64()),
        ("link_type", pa.string()),
    ]),
    "user_ratings": pa.schema([
        ("rating_id", pa.int64()),
        ("user_id", pa.int64()),
        ("title_id", pa.int64()),
        ("rating_10", pa.float64()),
        ("rating_date", pa.string()),
    ]),
    "episode_cast": pa.schema([
        ("episode_cast_id", pa.int64()),
        ("episode_id", pa.int64()),
        ("series_id", pa.int64()),
        ("person_id", pa.int64()),
        ("role_type", pa.string()),
        ("credit_order", pa.int32()),
    ]),
    "world_events": pa.schema([
        ("event_id", pa.int64()),
        ("year", pa.int32()),
        ("event_type", pa.string()),
        ("description", pa.string()),
        ("duration_years", pa.int32()),
        ("affected_entity_id", pa.int64()),
        ("affected_entity_type", pa.string()),
        ("parameter_delta_json", pa.string()),
    ]),
    "production_timeline": pa.schema([
        ("timeline_id", pa.int64()),
        ("movie_id", pa.int64()),
        ("phase", pa.string()),
        ("phase_start", pa.string()),
        ("phase_end", pa.string()),
    ]),
    "streaming_windows": pa.schema([
        ("window_id", pa.int64()),
        ("movie_id", pa.int64()),
        ("platform", pa.string()),
        ("window_start", pa.string()),
        ("window_end", pa.string()),
        ("exclusivity", pa.string()),
    ]),
    "person_contracts": pa.schema([
        ("contract_id", pa.int64()),
        ("person_id", pa.int64()),
        ("company_id", pa.int64()),
        ("start_date", pa.string()),
        ("end_date", pa.string()),
        ("salary_band", pa.string()),
        ("contract_type", pa.string()),
    ]),
    "movie_sequence": pa.schema([
        ("franchise_id", pa.string()),
        ("movie_id", pa.int64()),
        ("sequence_no", pa.int32()),
        ("predecessor_movie_id", pa.int64()),
    ]),
    "person_collaborations": pa.schema([
        ("person_a_id", pa.int64()),
        ("person_b_id", pa.int64()),
        ("collaboration_count", pa.int32()),
        ("first_year", pa.int32()),
        ("last_year", pa.int32()),
        ("shared_genres", pa.string()),
    ]),
    "media_links": pa.schema([
        ("link_id", pa.int64()),
        ("source_id", pa.int64()),
        ("source_type", pa.string()),
        ("target_id", pa.int64()),
        ("target_type", pa.string()),
        ("link_type", pa.string()),
        ("reason", pa.string()),
    ]),
    "critic_repairs": pa.schema([
        ("title_id", pa.int64()),
        ("repair_type", pa.string()),
        ("status", pa.string()),
        ("detail", pa.string()),
    ]),
}

AUTO_PK_TABLES = get_auto_pk_tables()

# Tables that are safe to stream (never read back during generation loop).
STREAMABLE_TABLES = frozenset([
    "reviews",
    "box_office_daily",
    "box_office_weekly",
    "box_office_by_territory",
    "ratings_breakdown",
    "locations",
    "movie_links",
    "release_dates",
    "movie_crew",
])

CORE_STREAMABLE_TABLES = frozenset([
    "movie",
    "cast_info",
    "movie_directors",
    "movie_companies",
    "movie_keyword",
])

# Tables generated post-loop that should also stream to disk.
POST_LOOP_STREAMABLE = frozenset([
    "user_ratings",
    "production_timeline",
    "streaming_windows",
    "person_contracts",
    "movie_sequence",
    "person_collaborations",
    "media_links",
    "world_events",
    "critic_repairs",
])


# ═══════════════════════════════════════════════════════════════════════
# ArrowSink — streaming writer
# ═══════════════════════════════════════════════════════════════════════

class ArrowSink:
    """Batched streaming writer using Arrow IPC (Feather v2) format.

    Usage:
        with ArrowSink("reviews.arrow", PA_SCHEMAS["reviews"]) as sink:
            for movie in movies:
                sink.write_rows(review_rows)  # list[dict]
        print(f"Wrote {sink.total_rows} rows")
    """

    def __init__(
        self,
        path: str,
        schema: pa.Schema,
        batch_size: int = 10_000,
        compression: str = "lz4",
        auto_pk_column: str | None = None,
    ):
        self.path = path
        self.schema = schema
        self.batch_size = batch_size
        self.compression = compression
        self.auto_pk_column = auto_pk_column
        self._buffer: list[dict] = []
        self._writer: ipc.RecordBatchFileWriter | None = None
        self._file: pa.OSFile | None = None
        self.total_rows = 0
        self._closed = False
        self._next_pk = 1

    def _coerce_value(self, field: pa.Field, value: Any) -> Any:
        if value is None:
            return None
        ftype = field.type
        if pa.types.is_string(ftype) or pa.types.is_large_string(ftype):
            if isinstance(value, str):
                return value
            if isinstance(value, (bytes, bytearray, memoryview)):
                return bytes(value).decode("utf-8", errors="replace")
            return str(value)
        if pa.types.is_binary(ftype) or pa.types.is_large_binary(ftype):
            if isinstance(value, (bytes, bytearray, memoryview)):
                return bytes(value)
            if isinstance(value, str):
                return value.encode("utf-8")
        if pa.types.is_integer(ftype) and isinstance(value, bool):
            return int(value)
        if pa.types.is_floating(ftype) and isinstance(value, bool):
            return float(value)
        return value

    def _normalize_row(self, row: dict) -> dict:
        normalized = row
        for field in self.schema:
            name = field.name
            if name not in row:
                continue
            value = row[name]
            coerced = self._coerce_value(field, value)
            if coerced is value:
                continue
            if normalized is row:
                normalized = dict(row)
            normalized[name] = coerced
        return normalized

    def _ensure_open(self):
        if self._writer is None:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            self._file = pa.OSFile(self.path, "wb")
            opts = ipc.IpcWriteOptions(compression=self.compression)
            self._writer = ipc.new_file(self._file, self.schema, options=opts)

    def _flush(self):
        if not self._buffer:
            return
        self._ensure_open()
        batch = pa.RecordBatch.from_pylist(self._buffer, schema=self.schema)
        self._writer.write_batch(batch)
        self.total_rows += len(self._buffer)
        self._buffer.clear()

    def _prepare_rows(self, rows: list[dict]) -> list[dict]:
        if not rows:
            return []
        if not self.auto_pk_column:
            return [self._normalize_row(row) for row in rows]
        prepared: list[dict] = []
        for row in rows:
            base_row = self._normalize_row(row)
            if self.auto_pk_column in base_row and base_row[self.auto_pk_column] is not None:
                prepared.append(base_row)
                continue
            new_row = dict(base_row)
            new_row[self.auto_pk_column] = self._next_pk
            self._next_pk += 1
            prepared.append(new_row)
        return prepared

    def write_rows(self, rows: list[dict]):
        """Buffer rows; flush to disk when batch_size is reached."""
        if not rows:
            return
        self._buffer.extend(self._prepare_rows(rows))
        while len(self._buffer) >= self.batch_size:
            # Flush exactly one batch
            to_flush = self._buffer[:self.batch_size]
            self._buffer = self._buffer[self.batch_size:]
            self._ensure_open()
            batch = pa.RecordBatch.from_pylist(to_flush, schema=self.schema)
            self._writer.write_batch(batch)
            self.total_rows += len(to_flush)

    def write_row(self, row: dict):
        """Convenience: buffer a single row."""
        row = self._normalize_row(row)
        if self.auto_pk_column and (self.auto_pk_column not in row or row[self.auto_pk_column] is None):
            row = dict(row)
            row[self.auto_pk_column] = self._next_pk
            self._next_pk += 1
        self._buffer.append(row)
        if len(self._buffer) >= self.batch_size:
            self._flush()

    def close(self) -> int:
        """Flush remaining buffer, close writer. Returns total rows written."""
        if self._closed:
            return self.total_rows
        self._closed = True
        if self._buffer:
            self._flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._file is not None:
            self._file.close()
            self._file = None
        return self.total_rows

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ═══════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════

def df_to_arrow(df: pd.DataFrame, path: str, table_name: str | None = None):
    """Write a pandas DataFrame to Arrow IPC (Feather v2) with lz4 compression.

    If table_name is given and exists in PA_SCHEMAS, enforces the schema.
    Otherwise infers from the DataFrame.
    """
    schema = PA_SCHEMAS.get(table_name) if table_name else None
    if schema is not None:
        # Reorder and filter columns to match schema
        cols = [f.name for f in schema]
        for c in cols:
            if c not in df.columns:
                df[c] = None
        table = pa.Table.from_pandas(df[cols], schema=schema, preserve_index=False)
    else:
        table = pa.Table.from_pandas(df, preserve_index=False)
    feather.write_feather(table, path, compression="lz4")


def make_table_sink(path: str, table_name: str, batch_size: int = 10_000) -> ArrowSink:
    schema = PA_SCHEMAS.get(table_name)
    if schema is None:
        raise KeyError(f"No Arrow schema registered for table '{table_name}'")
    return ArrowSink(
        path,
        schema,
        batch_size=batch_size,
        auto_pk_column=AUTO_PK_TABLES.get(table_name),
    )


def read_table(base_path: str, table_name: str | None = None) -> pd.DataFrame:
    """Read a table from Arrow (.arrow) or CSV (.csv), preferring Arrow.

    base_path: path WITHOUT extension, e.g. "/data/reviews"
               OR path WITH extension — will try both formats.
    """
    # Strip extension if present
    root, ext = os.path.splitext(base_path)
    if ext in (".arrow", ".csv", ".parquet"):
        base_path = root

    # Try Arrow first (fastest)
    arrow_path = base_path + ".arrow"
    if os.path.exists(arrow_path):
        return feather.read_table(arrow_path).to_pandas()

    # Try Parquet (in case of older files)
    parquet_path = base_path + ".parquet"
    if os.path.exists(parquet_path):
        import pyarrow.parquet as pq
        return pq.read_table(parquet_path).to_pandas()

    # Fall back to CSV
    csv_path = base_path + ".csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, low_memory=False)

    # Nothing found
    return pd.DataFrame()


def read_table_polars(base_path: str):
    """Read Arrow/Parquet/CSV into Polars when available, else return None."""
    if pl is None:
        return None

    root, ext = os.path.splitext(base_path)
    if ext in (".arrow", ".csv", ".parquet"):
        base_path = root

    arrow_path = base_path + ".arrow"
    if os.path.exists(arrow_path):
        return pl.read_ipc(arrow_path)

    parquet_path = base_path + ".parquet"
    if os.path.exists(parquet_path):
        return pl.read_parquet(parquet_path)

    csv_path = base_path + ".csv"
    if os.path.exists(csv_path):
        return pl.read_csv(csv_path)

    return None


def read_table_required(base_path: str, table_name: str | None = None) -> pd.DataFrame:
    """Like read_table but raises FileNotFoundError if no file exists."""
    df = read_table(base_path, table_name)
    if df is None or (hasattr(df, '__len__') and len(df) == 0):
        root = os.path.splitext(base_path)[0] if os.path.splitext(base_path)[1] else base_path
        # Check if ANY file exists — empty df from existing file is OK
        for ext in (".arrow", ".parquet", ".csv"):
            if os.path.exists(root + ext):
                return df
        raise FileNotFoundError(
            f"Required source file missing: {root} (tried .arrow, .parquet, .csv)"
        )
    return df
