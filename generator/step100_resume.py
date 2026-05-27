from __future__ import annotations

import json
import os
import random
import shutil
import gc
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from feather_sink import (
    CORE_STREAMABLE_TABLES,
    PA_SCHEMAS,
    POST_LOOP_STREAMABLE,
    STREAMABLE_TABLES,
    df_to_arrow,
    make_table_sink,
    read_table,
)
from graph_runtime import GraphRuntime
from schema import TABLE_DEFS


STEP100_RESUME_DIRNAME = "_step100_resume"
STEP100_RESUME_FORMAT_VERSION = 1

GLOBAL_TABLES: tuple[str, ...] = (
    "company_links",
    "person_demographics",
    "tv_series",
    "seasons",
    "episodes",
    "episode_cast",
)

POST_LOOP_TABLES: tuple[str, ...] = (
    "user_ratings",
    "world_events",
    "production_timeline",
    "streaming_windows",
    "person_contracts",
    "movie_sequence",
    "person_collaborations",
    "media_links",
    "critic_repairs",
)

PER_MOVIE_TABLES: tuple[str, ...] = tuple(
    name
    for name in TABLE_DEFS
    if name not in set(GLOBAL_TABLES) | set(POST_LOOP_TABLES)
)

RESULT_TABLES: tuple[str, ...] = tuple(
    name
    for name in TABLE_DEFS
    if name not in STREAMABLE_TABLES and name not in POST_LOOP_STREAMABLE
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return float(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_ready(item) for item in value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _entity_key_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").astype("Int64").astype(str)


def _merge_checkpoint_entities(
    current: pd.DataFrame | None,
    checkpoint: pd.DataFrame,
    *,
    id_column: str,
) -> pd.DataFrame:
    """Restore evolved checkpoint rows while keeping continuation top-up rows."""
    if current is None or current.empty:
        return checkpoint
    if checkpoint.empty:
        return current
    if id_column not in current.columns or id_column not in checkpoint.columns:
        return checkpoint

    columns = list(checkpoint.columns) + [col for col in current.columns if col not in checkpoint.columns]
    current_aligned = current.copy()
    checkpoint_aligned = checkpoint.copy()
    for col in columns:
        if col not in current_aligned.columns:
            current_aligned[col] = None
        if col not in checkpoint_aligned.columns:
            checkpoint_aligned[col] = None
    current_aligned = current_aligned[columns]
    checkpoint_aligned = checkpoint_aligned[columns]

    checkpoint_keys = set(_entity_key_series(checkpoint_aligned, id_column).tolist())
    current_keys = _entity_key_series(current_aligned, id_column)
    new_rows = current_aligned.loc[~current_keys.isin(checkpoint_keys)].copy()

    merged = pd.concat([checkpoint_aligned, new_rows], ignore_index=True)
    try:
        merged = merged.sort_values(by=id_column, kind="stable").reset_index(drop=True)
    except Exception:
        merged = merged.reset_index(drop=True)
    return merged


def _encode_tree(value: Any) -> Any:
    if isinstance(value, tuple):
        return {"__tuple__": [_encode_tree(item) for item in value]}
    if isinstance(value, frozenset):
        return {"__frozenset__": [_encode_tree(item) for item in value]}
    if isinstance(value, list):
        return [_encode_tree(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _encode_tree(val) for key, val in value.items()}
    return _json_ready(value)


def _decode_tree(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode_tree(item) for item in value]
    if isinstance(value, dict):
        if set(value.keys()) == {"__tuple__"}:
            return tuple(_decode_tree(item) for item in value["__tuple__"])
        if set(value.keys()) == {"__frozenset__"}:
            return frozenset(_decode_tree(item) for item in value["__frozenset__"])
        return {key: _decode_tree(val) for key, val in value.items()}
    return value


def _encode_kv(mapping: dict[Any, Any] | Counter | defaultdict | None) -> list[list[Any]]:
    out: list[list[Any]] = []
    for key, value in dict(mapping or {}).items():
        out.append([_encode_tree(key), _encode_tree(value)])
    out.sort(key=lambda item: json.dumps(item[0], ensure_ascii=True, sort_keys=True))
    return out


def _decode_kv(items: list[list[Any]] | None) -> list[tuple[Any, Any]]:
    out: list[tuple[Any, Any]] = []
    for pair in list(items or []):
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        out.append((_decode_tree(pair[0]), _decode_tree(pair[1])))
    return out


def _restore_counter(items: list[list[Any]] | None, *, key_fn=int, value_fn=int) -> Counter:
    out: Counter = Counter()
    for key, value in _decode_kv(items):
        out[key_fn(key)] = value_fn(value)
    return out


def _restore_defaultdict_list(items: list[list[Any]] | None, *, key_fn=int, value_fn=int) -> defaultdict:
    out: defaultdict = defaultdict(list)
    for key, value in _decode_kv(items):
        out[key_fn(key)] = [value_fn(item) for item in list(value or [])]
    return out


def _restore_defaultdict_int(items: list[list[Any]] | None, *, key_fn=int, value_fn=int) -> defaultdict:
    out: defaultdict = defaultdict(int)
    for key, value in _decode_kv(items):
        out[key_fn(key)] = value_fn(value)
    return out


def _restore_mapping_of_sets(items: list[list[Any]] | None, *, key_fn=int, value_fn=int) -> dict[int, set[int]]:
    out: dict[int, set[int]] = {}
    for key, value in _decode_kv(items):
        out[key_fn(key)] = {value_fn(item) for item in list(value or [])}
    return out


def _encode_numpy_state(state: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "bit_generator": str(state[0]),
        "keys": np.asarray(state[1], dtype=np.uint32).tolist(),
        "pos": int(state[2]),
        "has_gauss": int(state[3]),
        "cached_gaussian": float(state[4]),
    }


def _decode_numpy_state(payload: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(payload.get("bit_generator", "MT19937")),
        np.asarray(list(payload.get("keys", [])), dtype=np.uint32),
        int(payload.get("pos", 0)),
        int(payload.get("has_gauss", 0)),
        float(payload.get("cached_gaussian", 0.0)),
    )


def _rows_to_dataframe(rows: Any, table_name: str) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        return rows.copy()
    df = pd.DataFrame(list(rows or []))
    schema = PA_SCHEMAS.get(table_name)
    if schema is None:
        return df
    for field in schema:
        if field.name not in df.columns:
            df[field.name] = None
    return df[[field.name for field in schema]]


def _runtime_state_payload(
    world,
    *,
    demand_pool: dict[tuple[int, str], float],
    previous_movies_for_links: list[dict[str, Any]],
    evo_stats: dict[str, Any],
) -> dict[str, Any]:
    title_bank_mask = getattr(world, "_title_bank_used_mask", None)
    used_title_indices: list[int] = []
    if isinstance(title_bank_mask, np.ndarray) and title_bank_mask.dtype == bool:
        used_title_indices = np.flatnonzero(title_bank_mask).astype(int).tolist()

    franchises = [_json_ready(dict(item)) for item in list(getattr(world, "franchises", []) or [])]
    movie_franchise_pairs: list[list[int]] = []
    for movie_id, franchise in dict(getattr(world, "movie_franchise_map", {}) or {}).items():
        if not isinstance(franchise, dict):
            continue
        franchise_id = int(franchise.get("franchise_id", 0) or 0)
        if franchise_id <= 0:
            continue
        movie_franchise_pairs.append([int(movie_id), franchise_id])
    movie_franchise_pairs.sort(key=lambda item: item[0])

    return {
        "person_film_count": _encode_kv(getattr(world, "person_film_count", {})),
        "person_recent": _encode_kv(getattr(world, "person_recent", {})),
        "director_recent": _encode_kv(getattr(world, "director_recent", {})),
        "company_recent": _encode_kv(getattr(world, "company_recent", {})),
        "director_film_count": _encode_kv(getattr(world, "director_film_count", {})),
        "company_film_count": _encode_kv(getattr(world, "company_film_count", {})),
        "used_titles": sorted(str(item) for item in set(getattr(world, "used_titles", set()) or set())),
        "title_bank_used_indices": used_title_indices,
        "franchises": franchises,
        "movie_franchise_map": movie_franchise_pairs,
        "person_award_wins": _encode_kv(getattr(world, "person_award_wins", {})),
        "director_recent_outcomes": _encode_kv(getattr(world, "director_recent_outcomes", {})),
        "company_recent_outcomes": _encode_kv(getattr(world, "company_recent_outcomes", {})),
        "genre_recent_outcomes": _encode_kv(getattr(world, "genre_recent_outcomes", {})),
        "active_effects": _json_ready(list(getattr(world, "active_effects", []) or [])),
        "world_events": _json_ready(list(getattr(world, "world_events", []) or [])),
        "genre_weight_overrides": _json_ready(dict(getattr(world, "genre_weight_overrides", {}) or {})),
        "country_weight_overrides": _json_ready(dict(getattr(world, "country_weight_overrides", {}) or {})),
        "award_prestige": _json_ready(dict(getattr(world, "award_prestige", {}) or {})),
        "paused_persons": _encode_kv(getattr(world, "paused_persons", {})),
        "_chemistry_pairs": [_json_ready(list(pair)) for pair in sorted(list(getattr(world, "_chemistry_pairs", set()) or set()))],
        "_yearly_friendship_spawns": _encode_kv(getattr(world, "_yearly_friendship_spawns", {})),
        "_keyword_usage_counts": _encode_kv(getattr(world, "_keyword_usage_counts", {})),
        "_yearly_workload": _encode_kv(getattr(world, "_yearly_workload", {})),
        "_merge_families": _encode_kv({int(key): sorted(int(item) for item in value) for key, value in dict(getattr(world, "_merge_families", {}) or {}).items()}),
        "director_writer_history": _encode_kv({int(key): sorted(int(item) for item in value) for key, value in dict(getattr(world, "director_writer_history", {}) or {}).items()}),
        "_used_char_names_global": sorted(str(item) for item in set(getattr(world, "_used_char_names_global", set()) or set())),
        "_used_tagline_counts": _encode_kv(getattr(world, "_used_tagline_counts", {})),
        "_used_tagline_history": _json_ready(list(getattr(world, "_used_tagline_history", []) or [])),
        "_used_tagline_recent_entries": _encode_tree(list(getattr(world, "_used_tagline_recent_entries", []) or [])),
        "_used_tagline_template_family_counts": _encode_kv(getattr(world, "_used_tagline_template_family_counts", {})),
        "rerank_budget_remaining": int(getattr(world, "rerank_budget_remaining", 0) or 0),
        "keyword_rerank_budget_remaining": int(getattr(world, "keyword_rerank_budget_remaining", 0) or 0),
        "demand_pool": _encode_kv({tuple(key): float(value) for key, value in dict(demand_pool or {}).items()}),
        "previous_movies_for_links": _json_ready(list(previous_movies_for_links or [])),
        "evo_stats": _json_ready(dict(evo_stats or {})),
        "rng_state": _encode_numpy_state(world.rng.get_state()),
        "py_rng_state": _encode_tree(world.py_rng.getstate()),
    }


def _restore_runtime_state(
    world,
    payload: dict[str, Any],
) -> dict[str, Any]:
    world.person_film_count = _restore_counter(payload.get("person_film_count"))
    world.person_recent = _restore_defaultdict_list(payload.get("person_recent"))
    world.director_recent = _restore_defaultdict_list(payload.get("director_recent"))
    world.company_recent = _restore_defaultdict_list(payload.get("company_recent"))
    world.director_film_count = _restore_counter(payload.get("director_film_count"))
    world.company_film_count = _restore_counter(payload.get("company_film_count"))
    world.used_titles = set(str(item) for item in list(payload.get("used_titles", []) or []))

    title_bank = getattr(world, "title_bank", None)
    title_bank_len = len(title_bank) if title_bank is not None else 0
    title_bank_mask = np.zeros(title_bank_len, dtype=bool)
    for idx in list(payload.get("title_bank_used_indices", []) or []):
        try:
            pos = int(idx)
        except Exception:
            continue
        if 0 <= pos < len(title_bank_mask):
            title_bank_mask[pos] = True
    world._title_bank_used_mask = title_bank_mask

    world.franchises = [dict(item) for item in list(payload.get("franchises", []) or [])]
    franchise_by_id = {
        int(item.get("franchise_id", 0) or 0): item
        for item in world.franchises
        if int(item.get("franchise_id", 0) or 0) > 0
    }
    world.movie_franchise_map = {}
    for pair in list(payload.get("movie_franchise_map", []) or []):
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        movie_id = int(pair[0])
        franchise_id = int(pair[1])
        franchise = franchise_by_id.get(franchise_id)
        if franchise is not None:
            world.movie_franchise_map[movie_id] = franchise

    world.person_award_wins = _restore_counter(payload.get("person_award_wins"))
    world.director_recent_outcomes = _restore_defaultdict_list(payload.get("director_recent_outcomes"), value_fn=lambda item: item)
    world.company_recent_outcomes = _restore_defaultdict_list(payload.get("company_recent_outcomes"), value_fn=lambda item: item)
    world.genre_recent_outcomes = _restore_defaultdict_list(payload.get("genre_recent_outcomes"), key_fn=str, value_fn=lambda item: item)
    world.active_effects = list(payload.get("active_effects", []) or [])
    world.world_events = list(payload.get("world_events", []) or [])
    world.genre_weight_overrides = {str(key): float(value) for key, value in dict(payload.get("genre_weight_overrides", {}) or {}).items()}
    world.country_weight_overrides = {str(key): float(value) for key, value in dict(payload.get("country_weight_overrides", {}) or {}).items()}
    world.award_prestige = {str(key): float(value) for key, value in dict(payload.get("award_prestige", {}) or {}).items()}
    world.paused_persons = {int(key): dict(value) for key, value in _decode_kv(payload.get("paused_persons"))}
    world._chemistry_pairs = {tuple(int(part) for part in list(pair or [])) for pair in list(payload.get("_chemistry_pairs", []) or []) if len(list(pair or [])) == 2}
    world._yearly_friendship_spawns = {int(key): int(value) for key, value in _decode_kv(payload.get("_yearly_friendship_spawns"))}
    world._keyword_usage_counts = _restore_defaultdict_int(payload.get("_keyword_usage_counts"))
    world._yearly_workload = defaultdict(int)
    for key, value in _decode_kv(payload.get("_yearly_workload")):
        if isinstance(key, tuple) and len(key) == 2:
            world._yearly_workload[(int(key[0]), int(key[1]))] = int(value)
    world._merge_families = _restore_mapping_of_sets(payload.get("_merge_families"))
    world.director_writer_history = _restore_mapping_of_sets(payload.get("director_writer_history"))
    world._used_char_names_global = set(str(item) for item in list(payload.get("_used_char_names_global", []) or []))
    world._used_tagline_counts = {str(key): int(value) for key, value in _decode_kv(payload.get("_used_tagline_counts"))}
    world._used_tagline_history = [str(item) for item in list(payload.get("_used_tagline_history", []) or [])]
    world._used_tagline_recent_entries = list(_decode_tree(payload.get("_used_tagline_recent_entries", [])) or [])
    world._used_tagline_template_family_counts = {str(key): int(value) for key, value in _decode_kv(payload.get("_used_tagline_template_family_counts"))}
    world.rerank_budget_remaining = int(payload.get("rerank_budget_remaining", getattr(world, "rerank_budget_remaining", 0)) or 0)
    world.keyword_rerank_budget_remaining = int(payload.get("keyword_rerank_budget_remaining", getattr(world, "keyword_rerank_budget_remaining", 0)) or 0)

    np_state = payload.get("rng_state")
    if isinstance(np_state, dict):
        world.rng = np.random.RandomState()
        world.rng.set_state(_decode_numpy_state(np_state))
    py_state = payload.get("py_rng_state")
    if py_state is not None:
        world.py_rng = random.Random()
        world.py_rng.setstate(_decode_tree(py_state))

    world._build_lookup_dicts()
    world._build_person_role_views()
    world._year_cache = {}
    world._prewarm_year_cache()

    demand_pool: dict[tuple[int, str], float] = {}
    for key, value in _decode_kv(payload.get("demand_pool")):
        if isinstance(key, tuple) and len(key) == 2:
            demand_pool[(int(key[0]), str(key[1]))] = float(value)
    previous_movies_for_links = list(payload.get("previous_movies_for_links", []) or [])
    evo_stats = dict(payload.get("evo_stats", {}) or {})
    return {
        "demand_pool": demand_pool,
        "previous_movies_for_links": previous_movies_for_links,
        "evo_stats": evo_stats,
    }


def _restore_graph_checkpoint(world, graph_path: Path) -> None:
    if not graph_path.exists():
        return
    # Windows keeps NumPy mmap file handles open until all references are gone.
    # world.load() may have opened graph/runtime/*.npy before Step100 restore
    # recompiles the checkpoint graph into the same runtime directory.
    for attr, empty_value in (
        ("graph", None),
        ("edge_graph", None),
        ("affinity_index", None),
        ("edge_weights", {}),
        ("_friend_adj_all", {}),
        ("_rival_adj_all", {}),
    ):
        try:
            setattr(world, attr, empty_value)
        except Exception:
            pass
    gc.collect()
    graph = GraphRuntime.from_rows(
        world.base_dir,
        world._iter_edge_rows_from_arrow(graph_path),
        name_resolver=world._graph_name_resolver,
    )
    world.graph = graph
    world.edge_graph = graph
    world.affinity_index = graph.affinity_index
    world.edge_weights = graph.edge_weights
    world._friend_adj_all = graph.friend_adjacency
    world._rival_adj_all = graph.rival_adjacency


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


class Step100ResumeManager:
    # Step-100 resume guarantees continuation from the last committed year
    # boundary. It is intentionally not a promise of byte-identical replay
    # versus an uninterrupted run, especially when later stages depend on
    # LLM-driven or otherwise nondeterministic behavior.
    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.resume_dir = self.base_dir / STEP100_RESUME_DIRNAME
        self.manifest_path = self.resume_dir / "manifest.json"
        self.plan_path = self.resume_dir / "movie_plan.jsonl"
        self.globals_dir = self.resume_dir / "globals"
        self.shards_dir = self.resume_dir / "shards"
        self.inflight_dir = self.resume_dir / "inflight"
        self.spool_dir = self.resume_dir / "spool"
        self.checkpoint_dir = self.resume_dir / "checkpoint" / "latest"
        self.manifest: dict[str, Any] | None = None
        self.is_resuming = False

    def _load_manifest(self) -> dict[str, Any] | None:
        if not self.manifest_path.exists():
            return None
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _write_manifest(self) -> None:
        if self.manifest is None:
            return
        self.resume_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(self.manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    def _fingerprint_for(self, run_payload: dict[str, Any]) -> str:
        serialized = json.dumps(_json_ready(run_payload), ensure_ascii=True, sort_keys=True)
        return __import__("hashlib").sha256(serialized.encode("utf-8")).hexdigest()

    def prepare(self, *, run_payload: dict[str, Any], resume: bool, reset: bool, extend: bool = False) -> None:
        existing = self._load_manifest()
        if reset and self.resume_dir.exists():
            shutil.rmtree(self.resume_dir)
            existing = None

        fingerprint = self._fingerprint_for(run_payload)
        if extend:
            if reset:
                raise RuntimeError("Step 100 extension cannot be combined with reset.")
            if existing is None:
                raise RuntimeError(
                    "No step 100 resume workspace exists. Start a base run before using --extend-step100."
                )
            produced = int(existing.get("produced_movie_count", 0) or 0)
            if produced <= 0:
                raise RuntimeError("Cannot extend Step 100 before at least one year has been checkpointed.")
            target_movies = int(run_payload.get("n_movies", 0) or 0)
            previous_movies = int(existing.get("movie_count", 0) or 0)
            if target_movies <= previous_movies:
                raise RuntimeError(
                    f"Step 100 extension target must exceed the existing target "
                    f"({target_movies} <= {previous_movies})."
                )
            extension_meta = dict(existing.get("extension", {}) or {})
            extension_meta.setdefault("source_fingerprint", str(existing.get("fingerprint", "")))
            extension_meta.setdefault("source_movie_count", previous_movies)
            extension_meta.setdefault("source_end_year", existing.get("end_year"))
            extension_meta["target_movie_count"] = target_movies
            extension_meta["target_end_year"] = run_payload.get("end_year")
            existing["fingerprint"] = fingerprint
            existing["movie_count"] = target_movies
            existing["benchmark_mode"] = bool(run_payload.get("benchmark_mode", False))
            existing["run_options"] = _json_ready(run_payload)
            existing["extension"] = _json_ready(extension_meta)
            self.resume_dir.mkdir(parents=True, exist_ok=True)
            self.manifest = existing
            self.is_resuming = True
            self._write_manifest()
            return

        if resume:
            if existing is None:
                raise RuntimeError(
                    "No step 100 resume workspace exists. Start a fresh run first or omit --resume-step100."
                )
            if str(existing.get("fingerprint", "")) != fingerprint:
                raise RuntimeError(
                    "Saved step 100 resume workspace is incompatible with the current parameters. "
                    "Use --reset-step100-resume to start over."
                )
            if str(existing.get("status", "")) == "complete":
                raise RuntimeError(
                    "The saved step 100 resume workspace is already complete. Omit --resume-step100 or use --reset-step100-resume."
                )
            self.resume_dir.mkdir(parents=True, exist_ok=True)
            self.manifest = existing
            self.is_resuming = True
            return

        if existing is not None and str(existing.get("status", "")) != "complete":
            raise RuntimeError(
                "Incomplete step 100 resume artifacts already exist in this run directory. "
                "Use --resume-step100 to continue or --reset-step100-resume to discard them."
            )
        if existing is not None and self.resume_dir.exists():
            shutil.rmtree(self.resume_dir)

        self.resume_dir.mkdir(parents=True, exist_ok=True)
        self.globals_dir.mkdir(parents=True, exist_ok=True)
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self.inflight_dir.mkdir(parents=True, exist_ok=True)
        self.spool_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = {
            "format_version": STEP100_RESUME_FORMAT_VERSION,
            "fingerprint": fingerprint,
            "status": "fresh",
            "last_completed_year": None,
            "last_completed_sequence_index": None,
            "movie_count": int(run_payload.get("n_movies", 0) or 0),
            "produced_movie_count": 0,
            "start_year": run_payload.get("start_year"),
            "end_year": run_payload.get("end_year"),
            "benchmark_mode": bool(run_payload.get("benchmark_mode", False)),
            "run_options": _json_ready(run_payload),
        }
        self._write_manifest()
        self.is_resuming = False

    def mark_running(self) -> None:
        if self.manifest is None:
            return
        self.manifest["status"] = "running"
        self.manifest.pop("last_error", None)
        self._write_manifest()

    def mark_interrupted(self, reason: str | None = None) -> None:
        if self.manifest is None:
            return
        self.manifest["status"] = "interrupted"
        if reason:
            self.manifest["last_error"] = str(reason)
        self._write_manifest()

    def mark_complete(self) -> None:
        if self.manifest is None:
            return
        self.manifest["status"] = "complete"
        self.manifest.pop("last_error", None)
        self._write_manifest()

    def discard_inflight(self) -> None:
        if self.inflight_dir.exists():
            shutil.rmtree(self.inflight_dir)
        self.inflight_dir.mkdir(parents=True, exist_ok=True)
        if self.spool_dir.exists():
            shutil.rmtree(self.spool_dir)
        self.spool_dir.mkdir(parents=True, exist_ok=True)

    def save_globals(self, global_tables: dict[str, Any]) -> dict[str, pd.DataFrame]:
        self.globals_dir.mkdir(parents=True, exist_ok=True)
        stored: dict[str, pd.DataFrame] = {}
        for table_name in GLOBAL_TABLES:
            if table_name not in global_tables:
                continue
            df = _rows_to_dataframe(global_tables[table_name], table_name)
            path = self.globals_dir / f"{table_name}.arrow"
            df_to_arrow(df, str(path), table_name=table_name)
            stored[table_name] = df
        return stored

    def load_globals(self) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for table_name in GLOBAL_TABLES:
            path = self.globals_dir / f"{table_name}.arrow"
            if not path.exists():
                continue
            out[table_name] = read_table(str(path), table_name)
        return out

    def save_plan(self, plan_records: list[dict[str, Any]], *, year_min: int | None, year_max: int | None) -> None:
        self.resume_dir.mkdir(parents=True, exist_ok=True)
        with self.plan_path.open("w", encoding="utf-8") as handle:
            for seq_idx, record in enumerate(plan_records):
                concept = dict(record.get("concept", {}) or {})
                title_assignment = dict(record.get("title_assignment", {}) or {})
                franchise = concept.pop("franchise", None)
                concept.pop("_world", None)
                concept.pop("_rerank_candidates", None)
                concept["franchise_id"] = int(franchise.get("franchise_id", 0) or 0) if isinstance(franchise, dict) else None
                payload = {
                    "seq_idx": int(seq_idx),
                    "movie_id": int(record.get("movie_id", 0) or 0),
                    "year": int(record.get("year", concept.get("year", 0)) or 0),
                    "concept": _json_ready(concept),
                    "title_assignment": _json_ready(title_assignment),
                }
                handle.write(json.dumps(payload, ensure_ascii=False))
                handle.write("\n")
        if self.manifest is not None:
            self.manifest["start_year"] = int(year_min) if year_min is not None else None
            self.manifest["end_year"] = int(year_max) if year_max is not None else None
            self._write_manifest()

    def load_plan(self, world) -> list[dict[str, Any]]:
        if not self.plan_path.exists():
            raise RuntimeError("Step 100 resume plan is missing. Use --reset-step100-resume to start over.")
        franchise_by_id = {
            int(item.get("franchise_id", 0) or 0): item
            for item in list(getattr(world, "franchises", []) or [])
            if int(item.get("franchise_id", 0) or 0) > 0
        }
        out: list[dict[str, Any]] = []
        with self.plan_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                payload = json.loads(raw)
                concept = dict(payload.get("concept", {}) or {})
                franchise_id = concept.pop("franchise_id", None)
                concept["franchise"] = franchise_by_id.get(int(franchise_id)) if franchise_id not in (None, "") else None
                concept["_world"] = world
                out.append(
                    {
                        "seq_idx": int(payload.get("seq_idx", len(out))),
                        "movie_id": int(payload.get("movie_id", 0) or 0),
                        "year": int(payload.get("year", concept.get("year", 0)) or 0),
                        "concept": concept,
                        "title_assignment": dict(payload.get("title_assignment", {}) or {}),
                    }
                )
        return out

    def start_index(self) -> int:
        if self.manifest is None:
            return 0
        raw = self.manifest.get("last_completed_sequence_index")
        if raw is None:
            return 0
        return int(raw) + 1

    def restore_graph(self, world, *, preserve_current_graph: bool = False) -> None:
        graph_path = self.checkpoint_dir / "graph_temporal_history.arrow"
        live_manifest_path = self.checkpoint_dir / "graph_live_manifest.json"
        if graph_path.exists() and not preserve_current_graph:
            _restore_graph_checkpoint(world, graph_path)
        elif graph_path.exists() and preserve_current_graph:
            print(
                "  [Continuation] Preserving rebuilt graph with top-up entities; "
                "applying Step100 live graph history only.",
                flush=True,
            )
        graph = getattr(world, "graph", None)
        if graph is not None and live_manifest_path.exists() and hasattr(graph, "restore_live_history"):
            graph.restore_live_history(live_manifest_path)
        elif preserve_current_graph and live_manifest_path.exists():
            print(
                "  [Continuation] WARNING: live graph history exists, but the current graph "
                "cannot restore it; leaving rebuilt graph unchanged.",
                flush=True,
            )

    def restore_checkpoint(
        self,
        world,
        *,
        restore_graph: bool = True,
        merge_current_entities: bool = False,
    ) -> dict[str, Any]:
        self.discard_inflight()
        persons_path = self.checkpoint_dir / "persons.arrow"
        companies_path = self.checkpoint_dir / "companies.arrow"
        runtime_path = self.checkpoint_dir / "runtime_state.json"

        if persons_path.exists():
            checkpoint_persons = read_table(str(persons_path), "persons_enriched")
            if merge_current_entities:
                world.persons = _merge_checkpoint_entities(
                    getattr(world, "persons", None),
                    checkpoint_persons,
                    id_column="person_id",
                )
            else:
                world.persons = checkpoint_persons
        if companies_path.exists():
            checkpoint_companies = read_table(str(companies_path), "companies_enriched")
            if merge_current_entities:
                world.companies = _merge_checkpoint_entities(
                    getattr(world, "companies", None),
                    checkpoint_companies,
                    id_column="company_id",
                )
            else:
                world.companies = checkpoint_companies

        extra_state = {
            "demand_pool": {},
            "previous_movies_for_links": [],
            "evo_stats": {"year_program_ops": 0, "years_evolved": 0, "triggered_events": 0},
        }
        if runtime_path.exists():
            payload = json.loads(runtime_path.read_text(encoding="utf-8"))
            extra_state = _restore_runtime_state(world, payload)
        if restore_graph:
            self.restore_graph(world)
        return extra_state

    def spool_year_tables(
        self,
        *,
        year: int,
        year_tables: dict[str, list[dict[str, Any]]],
    ) -> int:
        year_dir = self.spool_dir / f"year={int(year)}"
        year_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(year_dir.glob("chunk=*"))
        chunk_dir = year_dir / f"chunk={len(existing) + 1:06d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        rows_written = 0
        for table_name in PER_MOVIE_TABLES:
            rows = year_tables.get(table_name)
            if not rows:
                continue
            df = _rows_to_dataframe(rows, table_name)
            df_to_arrow(df, str(chunk_dir / f"{table_name}.arrow"), table_name=table_name)
            rows_written += int(len(rows))
            rows.clear()

        if rows_written <= 0:
            shutil.rmtree(chunk_dir, ignore_errors=True)
        return rows_written

    def _spooled_chunk_paths(self, year: int, table_name: str) -> list[Path]:
        year_dir = self.spool_dir / f"year={int(year)}"
        if not year_dir.exists():
            return []
        out: list[Path] = []
        for chunk_dir in sorted(year_dir.glob("chunk=*")):
            path = chunk_dir / f"{table_name}.arrow"
            if path.exists():
                out.append(path)
        return out

    def _write_year_table(
        self,
        *,
        table_name: str,
        year: int,
        rows: list[dict[str, Any]],
        inflight_year_dir: Path,
    ) -> None:
        chunk_paths = self._spooled_chunk_paths(year, table_name)
        if not chunk_paths:
            if not rows:
                return
            df = _rows_to_dataframe(rows, table_name)
            df_to_arrow(df, str(inflight_year_dir / f"{table_name}.arrow"), table_name=table_name)
            return

        sink = make_table_sink(str(inflight_year_dir / f"{table_name}.arrow"), table_name)
        for chunk_path in chunk_paths:
            df = read_table(str(chunk_path), table_name)
            if df.empty:
                continue
            chunk_rows = df.astype(object).where(pd.notna(df), None).to_dict("records")
            for start in range(0, len(chunk_rows), 10_000):
                sink.write_rows(chunk_rows[start : start + 10_000])
        if rows:
            for start in range(0, len(rows), 10_000):
                sink.write_rows(rows[start : start + 10_000])
        total_rows = sink.close()
        if total_rows <= 0:
            _safe_unlink(inflight_year_dir / f"{table_name}.arrow")

    def commit_year(
        self,
        *,
        year: int,
        seq_idx: int,
        year_tables: dict[str, list[dict[str, Any]]],
        world,
        demand_pool: dict[tuple[int, str], float],
        previous_movies_for_links: list[dict[str, Any]],
        evo_stats: dict[str, Any],
    ) -> None:
        inflight_year_dir = self.inflight_dir / f"year={int(year)}"
        if inflight_year_dir.exists():
            shutil.rmtree(inflight_year_dir)
        inflight_year_dir.mkdir(parents=True, exist_ok=True)

        for table_name in PER_MOVIE_TABLES:
            rows = list(year_tables.get(table_name, []) or [])
            self._write_year_table(
                table_name=table_name,
                year=int(year),
                rows=rows,
                inflight_year_dir=inflight_year_dir,
            )

        tmp_persons = self.checkpoint_dir / "persons.arrow.tmp"
        tmp_companies = self.checkpoint_dir / "companies.arrow.tmp"
        tmp_runtime = self.checkpoint_dir / "runtime_state.json.tmp"
        tmp_graph = self.checkpoint_dir / "graph_temporal_history.arrow.tmp"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        df_to_arrow(world.persons, str(tmp_persons))
        df_to_arrow(world.companies, str(tmp_companies))
        tmp_runtime.write_text(
            json.dumps(
                _runtime_state_payload(
                    world,
                    demand_pool=demand_pool,
                    previous_movies_for_links=previous_movies_for_links,
                    evo_stats=evo_stats,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        graph = getattr(world, "graph", None)
        graph_checkpoint_mode = str(os.getenv("DATA_SYS_STEP100_GRAPH_CHECKPOINT_MODE", "full") or "full").strip().lower()
        live_manifest_tmp = self.checkpoint_dir / "graph_live_manifest.json.tmp"
        if graph is not None and graph_checkpoint_mode in {"delta", "live", "light"}:
            if hasattr(graph, "flush_year"):
                graph.flush_year(int(year))
            live_manifest = getattr(graph, "_live_manifest", None)
            if isinstance(live_manifest, dict):
                live_manifest_tmp.write_text(json.dumps(_json_ready(live_manifest), ensure_ascii=False, indent=2), encoding="utf-8")
        elif graph is not None and hasattr(graph, "export_temporal_history"):
            graph.export_temporal_history(tmp_graph)

        os.replace(tmp_persons, self.checkpoint_dir / "persons.arrow")
        os.replace(tmp_companies, self.checkpoint_dir / "companies.arrow")
        os.replace(tmp_runtime, self.checkpoint_dir / "runtime_state.json")
        if tmp_graph.exists():
            os.replace(tmp_graph, self.checkpoint_dir / "graph_temporal_history.arrow")
            _safe_unlink(self.checkpoint_dir / "graph_live_manifest.json")
        elif live_manifest_tmp.exists():
            os.replace(live_manifest_tmp, self.checkpoint_dir / "graph_live_manifest.json")

        for table_name in PER_MOVIE_TABLES:
            src = inflight_year_dir / f"{table_name}.arrow"
            if not src.exists():
                continue
            dest_dir = self.shards_dir / table_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            os.replace(src, dest_dir / f"year={int(year)}.arrow")
        shutil.rmtree(inflight_year_dir, ignore_errors=True)
        shutil.rmtree(self.spool_dir / f"year={int(year)}", ignore_errors=True)

        if self.manifest is not None:
            self.manifest["status"] = "running"
            self.manifest["last_completed_year"] = int(year)
            self.manifest["last_completed_sequence_index"] = int(seq_idx)
            self.manifest["produced_movie_count"] = int(seq_idx) + 1
            self._write_manifest()

    def _remove_canonical_preloop_outputs(self) -> None:
        for table_name in set(PER_MOVIE_TABLES) | set(GLOBAL_TABLES):
            _safe_unlink(self.base_dir / f"{table_name}.arrow")
            _safe_unlink(self.base_dir / f"{table_name}.csv")

    def merge_preloop_outputs(self) -> None:
        self._remove_canonical_preloop_outputs()

        for table_name in GLOBAL_TABLES:
            src = self.globals_dir / f"{table_name}.arrow"
            if not src.exists():
                continue
            shutil.copy2(src, self.base_dir / f"{table_name}.arrow")

        for table_name in PER_MOVIE_TABLES:
            shard_dir = self.shards_dir / table_name
            shard_paths = sorted(shard_dir.glob("year=*.arrow"))
            if not shard_paths:
                continue
            sink = make_table_sink(str(self.base_dir / f"{table_name}.arrow"), table_name)
            for shard_path in shard_paths:
                df = read_table(str(shard_path), table_name)
                if df.empty:
                    continue
                rows = df.astype(object).where(pd.notna(df), None).to_dict("records")
                for start in range(0, len(rows), 10_000):
                    sink.write_rows(rows[start : start + 10_000])
            sink.close()

    def build_result_frames(self) -> dict[str, pd.DataFrame]:
        result: dict[str, pd.DataFrame] = {}
        for table_name in RESULT_TABLES:
            path = self.base_dir / f"{table_name}.arrow"
            if path.exists():
                result[table_name] = read_table(str(path), table_name)
        return result

    def incomplete_workspace_exists(self) -> bool:
        manifest = self._load_manifest()
        return manifest is not None and str(manifest.get("status", "")) != "complete"
