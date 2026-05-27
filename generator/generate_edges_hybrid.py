"""
Mirage graph builder -- generate_edges_hybrid.py
===============================================
Hybrid graph architecture:
Consumes latent variables (from generate_latent_vars_api.py) and
builds edges PROCEDURALLY using cosine similarity, genre overlap,
and feature-based rules.

ALL edges are:
- Symmetric by construction (no asymmetric friendships)
- Deterministic (seeded RNG)
- Deduplicated (impossible to have duplicates)
- Typed across 3 categories: person<->person, person<->company, company<->company

Also runs Louvain community detection on the friendship subgraph.

Usage:
    python generate_edges_hybrid.py --base-dir .
"""
import json
import math
import os
import sys
import csv
import hashlib
import argparse
import time
from pathlib import Path
from collections import defaultdict, Counter
from itertools import chain
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from bootstrap_artifacts import audit_artifact_usage, audit_fallback_hit, current_mode, load_modeling_priors_artifact, prior_float_from_section, prior_section
from contracts import (
    GENRES, CAREER_STAGES, STYLE_TAGS, EDGE_TYPES,
    RELATIONSHIP_TARGETS, load_json_batch,
)
from pipeline_runtime import year_bounds_from_env
from policy_runtime import infer_market, modeling_priors_path, resolve_company_strategy, safe_load_json, world_policy_path
from utils import canonical_company_genre_vector, project_genres_to_company_basis

BASE_DIR = Path(__file__).parent
ENTITY_DIR = BASE_DIR / "entities"
GRAPH_DIR = BASE_DIR / "graph"

SEED = 42
SCALABLE_PERSON_THRESHOLD = 100_000
SCALABLE_COMPANY_THRESHOLD = 10_000
GRAPH_TIMING_FILENAME = "graph_build_timing.json"
WORLD_POLICY: dict[str, Any] = {}
MODELING_PRIORS_PAYLOAD: dict[str, Any] = {}
EDGE_PRIORS_SECTION: dict[str, Any] = {}
LAST_PP_BUILD_STATS: dict[str, Any] = {}
LAST_CALIBRATION_STATS: dict[str, Any] = {}


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _timing_row(phase: str, elapsed_sec: float, **extra: Any) -> dict[str, Any]:
    row: dict[str, Any] = {"phase": str(phase), "elapsed_sec": round(float(elapsed_sec), 4)}
    for key, value in extra.items():
        if value is not None:
            row[str(key)] = value
    return row


def _write_graph_timing(base_dir: str | Path, payload: dict[str, Any]) -> Path:
    graph_dir = Path(base_dir) / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    out_path = graph_dir / GRAPH_TIMING_FILENAME
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def _edge_export_int(value: Any) -> int | None:
    if value in (None, "", "nan"):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _edge_export_row(edge: dict[str, Any]) -> dict[str, Any]:
    return {
        "src_id": _edge_export_int(edge.get("src_id")),
        "dst_id": _edge_export_int(edge.get("dst_id")),
        "src_name": str(edge.get("src_name", "") or ""),
        "dst_name": str(edge.get("dst_name", "") or ""),
        "edge_type": str(edge.get("edge_type", "") or ""),
        "sign": str(edge.get("sign", "") or ""),
        "weight": float(edge.get("weight", 0.0) or 0.0),
        "source_kind": str(edge.get("source_kind", "") or ""),
        "reason": str(edge.get("reason", "") or ""),
        "valid_from": _edge_export_int(edge.get("valid_from")),
        "valid_to": _edge_export_int(edge.get("valid_to")),
    }


class _ArrowEdgeBatchWriter:
    def __init__(self, path: Path, label: str, *, batch_size: int = 50_000):
        import pyarrow as pa
        import pyarrow.ipc as ipc

        self.path = Path(path)
        self.label = str(label)
        self.batch_size = max(1, int(batch_size))
        self.count = 0
        self._closed = False
        self._batch: list[dict[str, Any]] = []
        self._pa = pa
        self._schema = pa.schema(
            [
                ("src_id", pa.int64()),
                ("dst_id", pa.int64()),
                ("src_name", pa.string()),
                ("dst_name", pa.string()),
                ("edge_type", pa.string()),
                ("sign", pa.string()),
                ("weight", pa.float32()),
                ("source_kind", pa.string()),
                ("reason", pa.string()),
                ("valid_from", pa.int32()),
                ("valid_to", pa.int32()),
            ]
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._sink = pa.OSFile(str(self.path), "wb")
        self._writer = ipc.new_file(
            self._sink,
            self._schema,
            options=ipc.IpcWriteOptions(compression="lz4"),
        )

    def add(self, edge: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError(f"Arrow writer for {self.path} is already closed")
        self._batch.append(_edge_export_row(edge))
        self.count += 1
        if len(self._batch) >= self.batch_size:
            self._flush()

    def _flush(self) -> None:
        if not self._batch:
            return
        self._writer.write(self._pa.Table.from_pylist(self._batch, schema=self._schema))
        self._batch.clear()

    def close(self) -> None:
        if self._closed:
            return
        self._flush()
        self._writer.close()
        self._sink.close()
        self._closed = True
        if self.count <= 0:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            return
        print(
            f"  {self.label}: {self.count:,} edges -> {self.path.name} "
            f"({self.path.stat().st_size / 1024 / 1024:.1f} MB)"
        )


def _write_edge_arrow(rows: list[dict[str, Any]], path: Path, label: str) -> int:
    writer = _ArrowEdgeBatchWriter(path, label)
    try:
        for edge in rows:
            writer.add(edge)
    finally:
        writer.close()
    return int(writer.count)


class _InstrumentedEdgeStream:
    def __init__(
        self,
        rows,
        *,
        edge_counter: Counter[str] | None = None,
        sink: _ArrowEdgeBatchWriter | None = None,
    ):
        self._rows = iter(rows)
        self.edge_counter = edge_counter
        self.sink = sink
        self.count = 0
        self.elapsed_sec = 0.0

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, Any]:
        started = time.perf_counter()
        row = next(self._rows)
        self.elapsed_sec += time.perf_counter() - started
        if self.edge_counter is not None:
            self.edge_counter[str(row.get("edge_type", "unknown"))] += 1
        if self.sink is not None:
            self.sink.add(row)
        self.count += 1
        return row

    def close(self) -> None:
        close_fn = getattr(self._rows, "close", None)
        if callable(close_fn):
            close_fn()


def _year_defaults() -> tuple[int, int, int, int]:
    start_year, end_year = year_bounds_from_env(1950, 2025)
    return int(start_year), int(end_year), int(end_year + 35), int(end_year + 40)


def _policy_genre_strength(world_policy: dict[str, Any], genre: str) -> float:
    if not isinstance(world_policy, dict):
        return 0.0
    buckets = world_policy.get("year_buckets", [])
    if not isinstance(buckets, list) or not buckets:
        return 0.0
    values = []
    for bucket in buckets:
        if not isinstance(bucket, dict):
            continue
        try:
            values.append(float(bucket.get("genre_bias", {}).get(genre, 0.0)))
        except Exception:
            continue
    return float(sum(values) / max(1, len(values))) if values else 0.0


def _edge_prior_float(key: str, default: float, *, lo: float | None = None, hi: float | None = None) -> float:
    row = prior_section(MODELING_PRIORS_PAYLOAD, "edge_priors")
    if key not in row:
        if current_mode() == "research":
            audit_fallback_hit(
                "edge_priors",
                f"missing:{key}",
                detail=f"edge_priors.{key} is required in research mode",
                mode="research",
            )
        return prior_float_from_section(MODELING_PRIORS_PAYLOAD, "edge_priors", key, default, lo=lo, hi=hi)
    try:
        value = float(row.get(key))
    except Exception:
        if current_mode() == "research":
            audit_fallback_hit(
                "edge_priors",
                f"invalid:{key}",
                detail=f"edge_priors.{key} must be numeric in research mode",
                mode="research",
            )
        value = float(default)
    if lo is not None:
        value = max(float(lo), value)
    if hi is not None:
        value = min(float(hi), value)
    return float(value)


_DEFAULT_PERSON_PERSON_CLASSIFICATION = {
    "controversy_high_threshold": 0.70,
    "controversy_gap_avoid_threshold": 0.40,
    "controversy_avoid_weight_floor": 0.10,
    "controversy_friendship_weight_base": 0.25,
    "controversy_friendship_score_weight": 0.55,
    "controversy_friendship_jitter": 0.08,
    "mentorship_style_threshold": 0.65,
    "mentorship_stage_gap_threshold": 2.0,
    "mentorship_weight_base": 0.28,
    "mentorship_style_weight": 0.54,
    "mentorship_jitter": 0.06,
    "rivalry_style_max": 0.30,
    "rivalry_genre_min": 0.35,
    "rivalry_stage_gap_max": 1.0,
    "rivalry_weight_base": 0.25,
    "rivalry_genre_weight": 0.50,
    "rivalry_style_distance_weight": 0.15,
    "rivalry_jitter": 0.06,
    "friendship_style_threshold": 0.70,
    "friendship_probability_threshold": 0.35,
    "friendship_style_soft_threshold": 0.55,
    "friendship_weight_base": 0.28,
    "friendship_score_weight": 0.55,
    "friendship_jitter": 0.08,
    "weight_min": 0.10,
    "weight_max": 0.88,
}

_DEFAULT_PERSON_COMPANY_GENERATION = {
    "genre_supplement_size": 25,
    "controversy_blacklist_person_threshold": 0.60,
    "controversy_blacklist_company_threshold": 0.30,
    "blacklist_weight_base": 0.50,
    "blacklist_weight_controversy_scale": 0.30,
    "event_franchise_micro_budget_penalty_boost": 1.08,
    "market_fit_boost": 1.10,
}

_DEFAULT_COMPANY_COMPANY_GENERATION = {
    "strategy_match_boost": 1.14,
    "market_match_boost": 1.10,
    "rival_overlap_threshold": 0.50,
    "rival_tier_threshold": 0.50,
    "rival_weight_scale": 1.00,
    "rival_weight_policy_cap": 1.20,
    "coproduction_overlap_threshold": 0.30,
    "coproduction_tier_max": 0.30,
    "coproduction_weight_scale": 0.70,
    "coproduction_policy_cap": 1.20,
}

_DEFAULT_SERENDIPITOUS_EDGES = {
    "stage_probabilities": {"legend": 0.65, "veteran": 0.35, "prime": 0.15, "rising": 0.05, "retired": 0.05},
    "stage_max_new_edges": {"legend": 3, "veteran": 2, "prime": 1, "rising": 1, "retired": 1},
    "candidate_multiplier": 20,
    "weight_min": 0.30,
    "weight_max": 0.55,
}

_DEFAULT_TRIADIC_CLOSURE = {
    "stage_probabilities": {"legend": 0.25, "veteran": 0.12, "prime": 0.05, "rising": 0.0, "retired": 0.03},
    "extra_cap": 8,
    "weight_min": 0.38,
    "weight_max": 0.70,
}

_DEFAULT_CALIBRATION = {
    "relationship_targets": dict(RELATIONSHIP_TARGETS),
    "candidate_sample_k": 1000,
    "director_actor_supplement": 200,
    "upsert_weight_min": 0.10,
    "upsert_weight_max": 0.95,
    "friendship_score_weights": {"style": 0.55, "genre": 0.35, "stage": 0.10},
    "rivalry_score_weights": {"genre": 0.60, "style_distance": 0.20, "stage": 0.20},
    "preferential_attachment_log_weight": 1.0,
    "best_friend_weight_base": 0.35,
    "best_friend_weight_score_scale": 0.48,
    "best_friend_weight_noise": 0.05,
    "rival_weight_base": 0.45,
    "rival_weight_score_scale": 0.45,
    "rival_weight_noise": 0.05,
    "director_stage_target_base": 0.80,
    "director_stage_target_span": 0.40,
    "director_preferred_score_weights": {"style": 0.65, "genre": 0.25, "stage": 0.10},
    "director_preferred_weight_base": 0.55,
    "director_preferred_weight_score_scale": 0.40,
    "director_preferred_weight_noise": 0.05,
    "director_avoid_score_weights": {"style_distance": 0.50, "genre_distance": 0.20, "controversy": 0.30},
    "director_avoid_weight_base": 0.55,
    "director_avoid_weight_score_scale": 0.35,
    "director_avoid_weight_noise": 0.05,
    "bf_same_community_weight_floor": 0.85,
    "bf_same_community_weight_boost": 0.05,
}


def _edge_prior_block(key: str) -> dict[str, Any]:
    value = EDGE_PRIORS_SECTION.get(str(key), {})
    if (not isinstance(value, dict) or not value) and current_mode() == "research":
        audit_fallback_hit(
            "edge_priors",
            f"missing:{key}",
            detail=f"edge_priors.{key} must be a non-empty object in research mode",
            mode="research",
        )
    return value if isinstance(value, dict) else {}


def _edge_block_float(block: dict[str, Any], key: str, default: float, *, lo: float | None = None, hi: float | None = None) -> float:
    if key not in block and current_mode() == "research":
        audit_fallback_hit(
            "edge_priors",
            f"missing:{key}",
            detail=f"edge prior field {key} is required in research mode",
            mode="research",
        )
    try:
        value = float(block.get(key, default))
    except Exception:
        if current_mode() == "research":
            audit_fallback_hit(
                "edge_priors",
                f"invalid:{key}",
                detail=f"edge prior field {key} must be numeric in research mode",
                mode="research",
            )
        value = float(default)
    if lo is not None:
        value = max(float(lo), value)
    if hi is not None:
        value = min(float(hi), value)
    return float(value)


def _edge_block_int(block: dict[str, Any], key: str, default: int, *, lo: int | None = None, hi: int | None = None) -> int:
    if key not in block and current_mode() == "research":
        audit_fallback_hit(
            "edge_priors",
            f"missing:{key}",
            detail=f"edge prior field {key} is required in research mode",
            mode="research",
        )
    try:
        value = int(round(float(block.get(key, default))))
    except Exception:
        if current_mode() == "research":
            audit_fallback_hit(
                "edge_priors",
                f"invalid:{key}",
                detail=f"edge prior field {key} must be integer-like in research mode",
                mode="research",
            )
        value = int(default)
    if lo is not None:
        value = max(int(lo), value)
    if hi is not None:
        value = min(int(hi), value)
    return int(value)


def _edge_block_str_float_map(block: dict[str, Any], key: str, default: dict[str, float], *, lo: float | None = None, hi: float | None = None) -> dict[str, float]:
    raw = block.get(key, {})
    if (not isinstance(raw, dict) or not raw) and current_mode() == "research":
        audit_fallback_hit(
            "edge_priors",
            f"missing:{key}",
            detail=f"edge prior map {key} is required in research mode",
            mode="research",
        )
    out = {str(k): float(v) for k, v in default.items()}
    if isinstance(raw, dict):
        if current_mode() == "research":
            missing = [str(k) for k in default.keys() if k not in raw]
            if missing:
                audit_fallback_hit(
                    "edge_priors",
                    f"missing:{key}",
                    detail=f"edge prior map {key} is missing keys: {', '.join(missing)}",
                    mode="research",
                )
        for k, v in raw.items():
            try:
                value = float(v)
            except Exception:
                if current_mode() == "research":
                    audit_fallback_hit(
                        "edge_priors",
                        f"invalid:{key}.{k}",
                        detail=f"edge prior map {key}.{k} must be numeric in research mode",
                        mode="research",
                    )
                continue
            if lo is not None:
                value = max(float(lo), value)
            if hi is not None:
                value = min(float(hi), value)
            out[str(k)] = float(value)
    return out


def _edge_block_str_int_map(block: dict[str, Any], key: str, default: dict[str, int], *, lo: int | None = None, hi: int | None = None) -> dict[str, int]:
    raw = block.get(key, {})
    if not isinstance(raw, dict):
        try:
            scalar_value = int(round(float(raw)))
        except Exception:
            scalar_value = None
        if scalar_value is not None:
            if lo is not None:
                scalar_value = max(int(lo), scalar_value)
            if hi is not None:
                scalar_value = min(int(hi), scalar_value)
            # Accept legacy authored scalars by expanding them into the
            # per-stage map that the runtime consumes.
            return {
                str(stage): min(int(default_value), int(scalar_value)) if scalar_value > 0 else 0
                for stage, default_value in default.items()
            }
    if (not isinstance(raw, dict) or not raw) and current_mode() == "research":
        audit_fallback_hit(
            "edge_priors",
            f"missing:{key}",
            detail=f"edge prior map {key} is required in research mode",
            mode="research",
        )
    out = {str(k): int(v) for k, v in default.items()}
    if isinstance(raw, dict):
        if current_mode() == "research":
            missing = [str(k) for k in default.keys() if k not in raw]
            if missing:
                audit_fallback_hit(
                    "edge_priors",
                    f"missing:{key}",
                    detail=f"edge prior map {key} is missing keys: {', '.join(missing)}",
                    mode="research",
                )
        for k, v in raw.items():
            try:
                value = int(round(float(v)))
            except Exception:
                if current_mode() == "research":
                    audit_fallback_hit(
                        "edge_priors",
                        f"invalid:{key}.{k}",
                        detail=f"edge prior map {key}.{k} must be integer-like in research mode",
                        mode="research",
                    )
                continue
            if lo is not None:
                value = max(int(lo), value)
            if hi is not None:
                value = min(int(hi), value)
            out[str(k)] = int(value)
    return out


def _edge_block_weight_map(block: dict[str, Any], key: str, default: dict[str, float]) -> dict[str, float]:
    weights = _edge_block_str_float_map(block, key, default, lo=0.0, hi=10.0)
    total = sum(max(0.0, float(v)) for v in weights.values())
    if total <= 1e-9:
        if current_mode() == "research":
            audit_fallback_hit(
                "edge_priors",
                f"invalid:{key}",
                detail=f"edge prior weight map {key} must sum to a positive value in research mode",
                mode="research",
            )
        return dict(default)
    return {str(k): max(0.0, float(v)) / total for k, v in weights.items()}


def _person_degree_cap_config(stage: str) -> tuple[float, float, int, int]:
    default_mu, default_sigma, default_lo, default_hi = _CAP_PARAMS.get(stage, (10, 5, 1, 30))
    degree_caps = _edge_prior_block("person_person_degree_caps")
    row = degree_caps.get(str(stage), {}) if isinstance(degree_caps, dict) else {}
    if not isinstance(row, dict):
        row = {}
    mu = _edge_block_float(row, "mean", default_mu, lo=0.0, hi=500.0)
    sigma = _edge_block_float(row, "std", default_sigma, lo=0.0, hi=200.0)
    lo = _edge_block_int(row, "min", default_lo, lo=0, hi=1000)
    hi = _edge_block_int(row, "max", default_hi, lo=lo, hi=5000)
    return float(mu), float(sigma), int(lo), int(hi)


def _pp_classification_config() -> dict[str, float]:
    block = _edge_prior_block("person_person_classification")
    return {
        key: _edge_block_float(block, key, default, hi=1.5)
        for key, default in _DEFAULT_PERSON_PERSON_CLASSIFICATION.items()
    }


def _pc_generation_config() -> dict[str, float]:
    block = _edge_prior_block("person_company_generation")
    config: dict[str, float] = {}
    for key, default in _DEFAULT_PERSON_COMPANY_GENERATION.items():
        if isinstance(default, int):
            config[key] = float(_edge_block_int(block, key, default, lo=0, hi=5000))
        else:
            config[key] = _edge_block_float(block, key, default, lo=0.0, hi=5.0)
    return config


def _cc_generation_config() -> dict[str, float]:
    block = _edge_prior_block("company_company_generation")
    return {
        key: _edge_block_float(block, key, default, lo=0.0, hi=5.0)
        for key, default in _DEFAULT_COMPANY_COMPANY_GENERATION.items()
    }


def _serendipitous_config() -> dict[str, Any]:
    block = _edge_prior_block("serendipitous_edges")
    return {
        "stage_probabilities": _edge_block_str_float_map(
            block, "stage_probabilities", _DEFAULT_SERENDIPITOUS_EDGES["stage_probabilities"], lo=0.0, hi=1.0
        ),
        "stage_max_new_edges": _edge_block_str_int_map(
            block, "stage_max_new_edges", _DEFAULT_SERENDIPITOUS_EDGES["stage_max_new_edges"], lo=0, hi=20
        ),
        "candidate_multiplier": _edge_block_int(
            block, "candidate_multiplier", int(_DEFAULT_SERENDIPITOUS_EDGES["candidate_multiplier"]), lo=1, hi=200
        ),
        "weight_min": _edge_block_float(block, "weight_min", _DEFAULT_SERENDIPITOUS_EDGES["weight_min"], lo=0.0, hi=1.0),
        "weight_max": _edge_block_float(block, "weight_max", _DEFAULT_SERENDIPITOUS_EDGES["weight_max"], lo=0.0, hi=1.0),
    }


def _triadic_closure_config() -> dict[str, Any]:
    block = _edge_prior_block("triadic_closure")
    return {
        "stage_probabilities": _edge_block_str_float_map(
            block, "stage_probabilities", _DEFAULT_TRIADIC_CLOSURE["stage_probabilities"], lo=0.0, hi=1.0
        ),
        "extra_cap": _edge_block_int(block, "extra_cap", int(_DEFAULT_TRIADIC_CLOSURE["extra_cap"]), lo=0, hi=100),
        "weight_min": _edge_block_float(block, "weight_min", _DEFAULT_TRIADIC_CLOSURE["weight_min"], lo=0.0, hi=1.0),
        "weight_max": _edge_block_float(block, "weight_max", _DEFAULT_TRIADIC_CLOSURE["weight_max"], lo=0.0, hi=1.0),
    }


def _calibration_config() -> dict[str, Any]:
    block = _edge_prior_block("calibration")
    return {
        "relationship_targets": _edge_block_str_float_map(
            block, "relationship_targets", _DEFAULT_CALIBRATION["relationship_targets"], lo=0.0, hi=20.0
        ),
        "candidate_sample_k": _edge_block_int(block, "candidate_sample_k", _DEFAULT_CALIBRATION["candidate_sample_k"], lo=16, hi=20000),
        "director_actor_supplement": _edge_block_int(
            block, "director_actor_supplement", _DEFAULT_CALIBRATION["director_actor_supplement"], lo=0, hi=5000
        ),
        "upsert_weight_min": _edge_block_float(block, "upsert_weight_min", _DEFAULT_CALIBRATION["upsert_weight_min"], lo=0.0, hi=1.0),
        "upsert_weight_max": _edge_block_float(block, "upsert_weight_max", _DEFAULT_CALIBRATION["upsert_weight_max"], lo=0.0, hi=1.0),
        "friendship_score_weights": _edge_block_weight_map(block, "friendship_score_weights", _DEFAULT_CALIBRATION["friendship_score_weights"]),
        "rivalry_score_weights": _edge_block_weight_map(block, "rivalry_score_weights", _DEFAULT_CALIBRATION["rivalry_score_weights"]),
        "preferential_attachment_log_weight": _edge_block_float(
            block, "preferential_attachment_log_weight", _DEFAULT_CALIBRATION["preferential_attachment_log_weight"], lo=0.0, hi=5.0
        ),
        "best_friend_weight_base": _edge_block_float(block, "best_friend_weight_base", _DEFAULT_CALIBRATION["best_friend_weight_base"], lo=0.0, hi=1.0),
        "best_friend_weight_score_scale": _edge_block_float(
            block, "best_friend_weight_score_scale", _DEFAULT_CALIBRATION["best_friend_weight_score_scale"], lo=0.0, hi=2.0
        ),
        "best_friend_weight_noise": _edge_block_float(block, "best_friend_weight_noise", _DEFAULT_CALIBRATION["best_friend_weight_noise"], lo=0.0, hi=0.5),
        "rival_weight_base": _edge_block_float(block, "rival_weight_base", _DEFAULT_CALIBRATION["rival_weight_base"], lo=0.0, hi=1.0),
        "rival_weight_score_scale": _edge_block_float(block, "rival_weight_score_scale", _DEFAULT_CALIBRATION["rival_weight_score_scale"], lo=0.0, hi=2.0),
        "rival_weight_noise": _edge_block_float(block, "rival_weight_noise", _DEFAULT_CALIBRATION["rival_weight_noise"], lo=0.0, hi=0.5),
        "director_stage_target_base": _edge_block_float(
            block, "director_stage_target_base", _DEFAULT_CALIBRATION["director_stage_target_base"], lo=0.0, hi=5.0
        ),
        "director_stage_target_span": _edge_block_float(
            block, "director_stage_target_span", _DEFAULT_CALIBRATION["director_stage_target_span"], lo=0.0, hi=5.0
        ),
        "director_preferred_score_weights": _edge_block_weight_map(
            block, "director_preferred_score_weights", _DEFAULT_CALIBRATION["director_preferred_score_weights"]
        ),
        "director_preferred_weight_base": _edge_block_float(
            block, "director_preferred_weight_base", _DEFAULT_CALIBRATION["director_preferred_weight_base"], lo=0.0, hi=1.0
        ),
        "director_preferred_weight_score_scale": _edge_block_float(
            block, "director_preferred_weight_score_scale", _DEFAULT_CALIBRATION["director_preferred_weight_score_scale"], lo=0.0, hi=2.0
        ),
        "director_preferred_weight_noise": _edge_block_float(
            block, "director_preferred_weight_noise", _DEFAULT_CALIBRATION["director_preferred_weight_noise"], lo=0.0, hi=0.5
        ),
        "director_avoid_score_weights": _edge_block_weight_map(
            block, "director_avoid_score_weights", _DEFAULT_CALIBRATION["director_avoid_score_weights"]
        ),
        "director_avoid_weight_base": _edge_block_float(
            block, "director_avoid_weight_base", _DEFAULT_CALIBRATION["director_avoid_weight_base"], lo=0.0, hi=1.0
        ),
        "director_avoid_weight_score_scale": _edge_block_float(
            block, "director_avoid_weight_score_scale", _DEFAULT_CALIBRATION["director_avoid_weight_score_scale"], lo=0.0, hi=2.0
        ),
        "director_avoid_weight_noise": _edge_block_float(
            block, "director_avoid_weight_noise", _DEFAULT_CALIBRATION["director_avoid_weight_noise"], lo=0.0, hi=0.5
        ),
        "bf_same_community_weight_floor": _edge_block_float(
            block, "bf_same_community_weight_floor", _DEFAULT_CALIBRATION["bf_same_community_weight_floor"], lo=0.0, hi=1.0
        ),
        "bf_same_community_weight_boost": _edge_block_float(
            block, "bf_same_community_weight_boost", _DEFAULT_CALIBRATION["bf_same_community_weight_boost"], lo=0.0, hi=0.5
        ),
    }


# ═══════════════════════════════════════════════════════════════════════
# MATH UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-10:
        return 0.0
    return float(np.dot(a, b) / denom)


def dot_overlap(a, b):
    """Weighted dot-product overlap (elements in [0,1])."""
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    return float(np.sum(a * b))


def sigmoid(x, scale=1.0, bias=0.0):
    """Sigmoid function."""
    return 1.0 / (1.0 + math.exp(-scale * (x - bias)))


def genre_str_to_vector(genre_str, genre_list=GENRES):
    """Convert genre string(s) to a binary vector."""
    vec = [0.0] * len(genre_list)
    if isinstance(genre_str, list):
        genres = genre_str
    elif isinstance(genre_str, str):
        genres = [g.strip() for g in genre_str.replace("|", ";").split(";")]
    else:
        genres = []
    for g in genres:
        for i, canonical in enumerate(genre_list):
            if g.lower() == canonical.lower():
                vec[i] = 1.0
                break
    return vec


# ═══════════════════════════════════════════════════════════════════════
# EDGE GENERATION
# ═══════════════════════════════════════════════════════════════════════

# Stochastic per-person degree caps (seeded by person_id for reproducibility).
# Two-mechanism design:
#   1. Stochastic cap -> within-stage variance (not all legends at exactly 80)
#   2. Adaptive threshold -> cluster structure (similar people connect first)
#
# Rising N(2, 3) [0, 8]:  ~30% get cap=0 (true unknowns, never connect pre-assembly)
# Prime  N(10, 6) [1, 30]: some prime actors quite peripheral
# Veteran N(22,10) [3, 60]: established but varied
# Legend N(40,18) [5,100]: reclusive niche legends at 15, generalist stars at 80
# Retired N(14, 8) [1, 35]:
_CAP_PARAMS = {
    "rising":  (2,  3,  0,   8),
    "prime":   (10, 6,  1,  30),
    "veteran": (22, 10, 3,  60),
    "legend":  (40, 18, 5, 100),
    "retired": (14, 8,  1,  35),
}
_CAP_SEED_OFFSET = 9173  # offset to avoid colliding with main RNG seeds

# V15 scaling parameters for the two-phase P-P edge generator.
# CROSS_GENRE_K:    how many cross-genre candidates each person samples in Phase 2.
#                   Higher -> more inter-community bridges but slower.
# CROSS_GENRE_BUMP: extra threshold added on top of the adaptive base for cross-genre
#                   pairs. 0.10 means only very similar cross-genre pairs connect,
#                   preserving dense within-community / sparse between-community structure.
_CROSS_GENRE_K    = 150
_CROSS_GENRE_BUMP = 0.10

# P-C scaling: random supplement companies evaluated per person beyond genre match.
_PC_GENRE_SUPPLEMENT = 25

_UNDIRECTED_EDGE_TYPES = {
    "friendship",
    "rivalry",
    "co_production",
    "market_rival",
    "collaboration",
    "chemistry",
}


def _primary_genre(p) -> str:
    """Extract primary (first) genre from a person's genre_affinity field."""
    ga = p.get("genre_affinity", [])
    if isinstance(ga, str):
        parts = [g.strip() for g in ga.replace(";", ",").split(",") if g.strip()]
        return parts[0].lower() if parts else "__other__"
    if isinstance(ga, (list, tuple)) and ga:
        return str(ga[0]).strip().lower()
    return "__other__"


def _person_stochastic_cap(pid: int, stage: str) -> int:
    """Reproducible per-person degree cap sampled from stage distribution."""
    mu, sigma, lo, hi = _person_degree_cap_config(stage)
    rng_local = np.random.RandomState((int(pid) + _CAP_SEED_OFFSET) % (2**31))
    cap = int(round(rng_local.normal(mu, sigma)))
    return max(lo, min(hi, cap))


def generate_person_person_edges(persons, latent_map, rng, max_edges_per_person=None, world_policy: dict[str, Any] | None = None):
    """Generate person<->person edges from latent variables.

    Edge types: friendship, rivalry, mentorship, avoid.

    V20 SCALING FIX -- Vectorized batch scoring replaces per-pair Python loop.
    ─────────────────────────────────────────────────────────────────────────
    Phase 1  (Within-genre, dense communities):
      For each primary-genre bucket, compute ALL pairwise scores via NumPy
      matrix operations (one BLAS matmul per bucket). Extract passing pairs,
      then greedily enforce degree caps in score-descending order.
      Complexity: O(B × Bk² / BLAS) — ~200× faster than per-pair Python.

    Phase 2  (Cross-genre, sparse bridges):
      Each person i samples _CROSS_GENRE_K=150 candidates j>i from OTHER
      genre buckets.  Threshold raised by _CROSS_GENRE_BUMP=0.10.
      Still uses per-pair scoring (only 30M evals, Python-tolerable at scale).

    Pair guarantee: every (i,j) is evaluated AT MOST ONCE.
      Phase 1 covers all intra-genre pairs with i<j (upper triangle).
      Phase 2 enforces j>i on cross-genre candidates -> zero overlap.

    All original mechanisms preserved unchanged:
      - _person_stochastic_cap degree caps
      - Adaptive threshold:  0.58 / (1 + 0.12 * deg^0.65)
      - Edge type classification (friendship/rivalry/mentorship/avoid)
      - Weight jitter formula
    """
    global LAST_PP_BUILD_STATS
    build_started = time.perf_counter()
    edges = []
    n = len(persons)
    edge_count = Counter()
    stage_ord = {"rising": 1, "prime": 2, "veteran": 3, "legend": 4, "retired": 5}

    person_by_id = {p["person_id"]: p for p in persons}
    person_cap   = {
        p["person_id"]: _person_stochastic_cap(
            p["person_id"], p.get("career_stage", "prime")
        )
        for p in persons
    }

    # Pre-compute per-person feature arrays as NumPy matrices
    pid_list   = [p["person_id"] for p in persons]
    lv_list    = [latent_map.get(p["person_id"]) for p in persons]
    has_lv     = np.array([lv is not None for lv in lv_list], dtype=bool)
    csv_arr    = np.array(
        [lv.get("creative_style_vector", [0]*8) if lv else [0]*8 for lv in lv_list],
        dtype=np.float32,
    )  # (n, 8)
    # Pre-normalize CSV vectors for cosine similarity via matmul
    csv_norms  = np.linalg.norm(csv_arr, axis=1, keepdims=True)
    csv_normed = csv_arr / np.maximum(csv_norms, 1e-10)

    risk_arr   = np.array(
        [lv.get("risk_tolerance", 0.5) if lv else 0.5 for lv in lv_list],
        dtype=np.float32,
    )  # (n,)
    stage_arr  = np.array(
        [stage_ord.get(p.get("career_stage", "prime"), 2) for p in persons],
        dtype=np.float32,
    )  # (n,)
    ctrv_arr   = np.array(
        [lv.get("controversy_score", 0.0) if lv else 0.0 for lv in lv_list],
        dtype=np.float32,
    )  # (n,)
    gnr_arr    = np.array(
        [genre_str_to_vector(p.get("genre_affinity", [])) for p in persons],
        dtype=np.float32,
    )  # (n, 28)
    gnr_sums   = gnr_arr.sum(axis=1)  # (n,) -- per-person genre count

    # Genre bucket construction (primary genre only)
    pg_list = [_primary_genre(persons[i]) for i in range(n)]
    genre_buckets: dict = defaultdict(list)
    for idx, pg in enumerate(pg_list):
        genre_buckets[pg].append(idx)
    bucket_sizes = [len(bucket) for bucket in genre_buckets.values()]
    max_bucket_size = max(bucket_sizes) if bucket_sizes else 0
    avg_bucket_size = round(float(sum(bucket_sizes) / max(1, len(bucket_sizes))), 2) if bucket_sizes else 0.0
    cross_genre_pools: dict[str, np.ndarray] = {}
    for pg in genre_buckets.keys():
        other_bucket_arrays = [
            np.array(bucket, dtype=np.int64)
            for other_pg, bucket in genre_buckets.items()
            if other_pg != pg and bucket
        ]
        if other_bucket_arrays:
            cross_genre_pools[pg] = np.sort(np.concatenate(other_bucket_arrays))
        else:
            cross_genre_pools[pg] = np.array([], dtype=np.int64)

    # Degree cap array for fast indexing
    cap_arr = np.array([person_cap[pid] for pid in pid_list], dtype=np.int32)
    pp_style_weight = _edge_prior_float("person_person_style_weight", 0.40, lo=0.0, hi=1.0)
    pp_genre_weight = _edge_prior_float("person_person_genre_weight", 0.25, lo=0.0, hi=1.0)
    pp_risk_weight = _edge_prior_float("person_person_risk_weight", 0.15, lo=0.0, hi=1.0)
    pp_stage_weight = _edge_prior_float("person_person_stage_weight", 0.10, lo=0.0, hi=1.0)
    pp_noise_weight = _edge_prior_float("person_person_noise_weight", 0.10, lo=0.0, hi=1.0)
    pp_policy_weight = _edge_prior_float("person_person_policy_weight", 0.03, lo=0.0, hi=0.5)
    pp_logistic_scale = _edge_prior_float("person_person_logistic_scale", 8.0, lo=1.0, hi=20.0)
    pp_logistic_bias = _edge_prior_float("person_person_logistic_bias", 0.55, lo=0.0, hi=1.0)
    pp_base_threshold = _edge_prior_float("person_person_base_threshold", 0.58, lo=0.0, hi=1.0)
    pp_degree_decay = _edge_prior_float("person_person_degree_decay", 0.12, lo=0.0, hi=1.0)
    pp_degree_power = _edge_prior_float("person_person_degree_power", 0.65, lo=0.1, hi=2.0)
    cross_genre_sample_multiplier = _edge_prior_float("cross_genre_candidate_multiplier", 3.0, lo=1.0, hi=20.0)
    pp_class = _pp_classification_config()

    # ── Edge type classification (applied to individual passing pairs) ────────
    def _classify_and_add(i_global, j_global, style_sim_v, genre_jaccard,
                          stage_diff, p_edge_val, threshold_extra=0.0):
        """Classify edge type and append to edges list. Returns True if added."""
        pid_a, pid_b = pid_list[i_global], pid_list[j_global]
        controversy_a = float(ctrv_arr[i_global])
        controversy_b = float(ctrv_arr[j_global])

        src_id, dst_id = pid_a, pid_b
        if controversy_a > pp_class["controversy_high_threshold"] or controversy_b > pp_class["controversy_high_threshold"]:
            if abs(controversy_a - controversy_b) > pp_class["controversy_gap_avoid_threshold"]:
                etype, sign = "avoid", "-"
                weight = round(min(controversy_a, controversy_b) + pp_class["controversy_avoid_weight_floor"], 2)
            else:
                etype, sign = "friendship", "+"
                base = (
                    pp_class["controversy_friendship_weight_base"]
                    + pp_class["controversy_friendship_score_weight"] * (p_edge_val + style_sim_v) / 2.0
                )
                jitter = pp_class["controversy_friendship_jitter"]
                weight = round(base + rng.uniform(-jitter, jitter), 2)
        elif style_sim_v > pp_class["mentorship_style_threshold"] and stage_diff >= pp_class["mentorship_stage_gap_threshold"]:
            etype, sign = "mentorship", "+"
            base = pp_class["mentorship_weight_base"] + pp_class["mentorship_style_weight"] * style_sim_v
            jitter = pp_class["mentorship_jitter"]
            weight = round(base + rng.uniform(-jitter, jitter), 2)
            if stage_arr[i_global] < stage_arr[j_global]:
                src_id, dst_id = pid_b, pid_a
        elif (
            style_sim_v < pp_class["rivalry_style_max"]
            and genre_jaccard > pp_class["rivalry_genre_min"]
            and stage_diff <= pp_class["rivalry_stage_gap_max"]
        ):
            etype, sign = "rivalry", "-"
            base = (
                pp_class["rivalry_weight_base"]
                + pp_class["rivalry_genre_weight"] * genre_jaccard
                + pp_class["rivalry_style_distance_weight"] * max(0.0, pp_class["rivalry_style_max"] - style_sim_v)
            )
            jitter = pp_class["rivalry_jitter"]
            weight = round(base + rng.uniform(-jitter, jitter), 2)
        elif (
            style_sim_v > pp_class["friendship_style_threshold"]
            or (p_edge_val > pp_class["friendship_probability_threshold"] and style_sim_v > pp_class["friendship_style_soft_threshold"])
        ):
            etype, sign = "friendship", "+"
            base = pp_class["friendship_weight_base"] + pp_class["friendship_score_weight"] * (p_edge_val + style_sim_v) / 2.0
            jitter = pp_class["friendship_jitter"]
            weight = round(base + rng.uniform(-jitter, jitter), 2)
        else:
            return False

        weight = max(pp_class["weight_min"], min(pp_class["weight_max"], weight))
        xg = " [cross-genre]" if threshold_extra > 0 else ""
        edges.append({
            "src_id":      src_id,
            "dst_id":      dst_id,
            "src_name":    person_by_id.get(src_id, {}).get("name", "?"),
            "dst_name":    person_by_id.get(dst_id, {}).get("name", "?"),
            "edge_type":   etype,
            "sign":        sign,
            "weight":      weight,
            "source_kind": "latent_hybrid",
            "reason":      f"style={style_sim_v:.2f} jaccard={genre_jaccard:.2f} "
                           f"stage_gap={int(stage_diff)}{xg}",
        })
        edge_count[pid_a] += 1
        edge_count[pid_b] += 1
        return True

    # ── Phase 1: Within-genre scoring with bounded-memory blocks ────────────
    # The original full-bucket matrix path exploded RAM on large genre buckets.
    # We now score each bucket in square blocks and only retain passing pairs.
    phase1_started = time.perf_counter()
    phase1_pairs = 0
    phase1_candidates = 0
    phase1_block_size = 1024
    for pg, bucket_idxs in genre_buckets.items():
        bucket_size = len(bucket_idxs)
        if bucket_size < 2:
            continue

        phase1_pairs += bucket_size * (bucket_size - 1) // 2
        bidxs_all = np.array(bucket_idxs, dtype=np.int64)
        lv_mask = has_lv[bidxs_all]
        bidxs = bidxs_all[lv_mask]
        if bidxs.size < 2:
            continue

        csv_b = csv_normed[bidxs]
        gnr_b = gnr_arr[bidxs]
        gsums_b = gnr_sums[bidxs]
        risk_b = risk_arr[bidxs]
        stage_b = stage_arr[bidxs]
        genre_bonus_b = np.array(
            [_policy_genre_strength(world_policy or {}, pg_list[int(idx)]) for idx in bidxs],
            dtype=np.float32,
        )

        for left_start in range(0, len(bidxs), phase1_block_size):
            left_stop = min(left_start + phase1_block_size, len(bidxs))
            left_idx = bidxs[left_start:left_stop]
            csv_left = csv_b[left_start:left_stop]
            gnr_left = gnr_b[left_start:left_stop]
            risk_left = risk_b[left_start:left_stop]
            stage_left = stage_b[left_start:left_stop]
            bonus_left = genre_bonus_b[left_start:left_stop]
            sums_left = gsums_b[left_start:left_stop]

            for right_start in range(left_start, len(bidxs), phase1_block_size):
                right_stop = min(right_start + phase1_block_size, len(bidxs))
                right_idx = bidxs[right_start:right_stop]
                csv_right = csv_b[right_start:right_stop]
                gnr_right = gnr_b[right_start:right_stop]
                risk_right = risk_b[right_start:right_stop]
                stage_right = stage_b[right_start:right_stop]
                bonus_right = genre_bonus_b[right_start:right_stop]
                sums_right = gsums_b[right_start:right_stop]

                style_block = np.clip(csv_left @ csv_right.T, 0.0, 1.0)
                overlap_block = gnr_left @ gnr_right.T
                union_block = sums_left[:, None] + sums_right[None, :] - overlap_block
                jaccard_block = overlap_block / np.maximum(union_block, 1.0)
                risk_block = 1.0 - np.abs(risk_left[:, None] - risk_right[None, :])
                stage_block = np.abs(stage_left[:, None] - stage_right[None, :])
                noise_block = rng.rand(left_stop - left_start, right_stop - right_start).astype(np.float32)

                raw_score_block = (
                    pp_style_weight * style_block +
                    pp_genre_weight * jaccard_block +
                    pp_risk_weight * risk_block +
                    pp_stage_weight * (1.0 - stage_block / 4.0) +
                    pp_noise_weight * noise_block +
                    pp_policy_weight * ((bonus_left[:, None] + bonus_right[None, :]) / 2.0)
                )
                p_edge_block = 1.0 / (
                    1.0 + np.exp(np.clip(-pp_logistic_scale * (raw_score_block - pp_logistic_bias), -30, 30))
                )
                pass_mask = p_edge_block >= pp_base_threshold
                if left_start == right_start:
                    pass_mask &= np.triu(np.ones(pass_mask.shape, dtype=bool), k=1)

                ai_local, bi_local = np.nonzero(pass_mask)
                if ai_local.size == 0:
                    continue

                phase1_candidates += int(ai_local.size)
                scores = p_edge_block[ai_local, bi_local].astype(np.float32, copy=False)
                order = np.argsort(-scores, kind="mergesort")

                for k in order:
                    i_g = int(left_idx[int(ai_local[k])])
                    j_g = int(right_idx[int(bi_local[k])])
                    pid_a = pid_list[i_g]
                    pid_b = pid_list[j_g]

                    if edge_count[pid_a] >= person_cap[pid_a]:
                        continue
                    if edge_count[pid_b] >= person_cap[pid_b]:
                        continue

                    thresh = max(
                        pp_base_threshold / (1.0 + pp_degree_decay * (edge_count[pid_a] ** pp_degree_power)),
                        pp_base_threshold / (1.0 + pp_degree_decay * (edge_count[pid_b] ** pp_degree_power)),
                    )
                    score_val = float(scores[k])
                    if score_val < thresh:
                        continue

                    _classify_and_add(
                        i_g, j_g,
                        style_sim_v=float(style_block[int(ai_local[k]), int(bi_local[k])]),
                        genre_jaccard=float(jaccard_block[int(ai_local[k]), int(bi_local[k])]),
                        stage_diff=float(stage_block[int(ai_local[k]), int(bi_local[k])]),
                        p_edge_val=score_val,
                        threshold_extra=0.0,
                    )

    phase1_elapsed = time.perf_counter() - phase1_started
    print(f"  Phase 1 (within-genre, {len(genre_buckets)} buckets, vectorized): "
          f"{phase1_pairs:,} pairs, {phase1_candidates:,} candidates -> {len(edges)} edges in {phase1_elapsed:.2f}s")

    # ── Phase 2: Cross-genre bridges (sparse inter-community connections) ─────
    # j > i ensures each cross-genre pair is evaluated exactly once.
    # Raised threshold keeps bridges rare -- only truly similar cross-genre
    # people (generalist legends) will connect here.
    # Still per-pair Python loop (30M evals at 200K — tolerable).
    phase2_started = time.perf_counter()
    p2_before = len(edges)
    phase2_candidate_pairs = 0
    for i in range(n):
        if not has_lv[i]:
            continue
        if edge_count[pid_list[i]] >= person_cap[pid_list[i]]:
            continue
        cross_pool = cross_genre_pools.get(pg_list[i])
        if cross_pool is None or cross_pool.size == 0:
            continue
        start_at = int(np.searchsorted(cross_pool, i + 1, side="left"))
        eligible = cross_pool[start_at:]
        if eligible.size == 0:
            continue
        n_sample = min(int(round(_CROSS_GENRE_K * cross_genre_sample_multiplier)), int(eligible.size))
        sampled = rng.choice(eligible, size=n_sample, replace=False)
        candidates: list[tuple[float, int, float, float, float]] = []
        for j in sampled.tolist():
            pid_a, pid_b = pid_list[i], pid_list[j]
            if not has_lv[j]:
                continue
            if (edge_count[pid_a] >= person_cap[pid_a] or
                    edge_count[pid_b] >= person_cap[pid_b]):
                continue

            style_sim_v = float(np.dot(csv_normed[i], csv_normed[j]))
            ovlp = float(np.dot(gnr_arr[i], gnr_arr[j]))
            union_v = max(gnr_sums[i] + gnr_sums[j] - ovlp, 1.0)
            genre_jac = ovlp / union_v
            risk_m = 1.0 - abs(float(risk_arr[i]) - float(risk_arr[j]))
            stage_d = abs(float(stage_arr[i]) - float(stage_arr[j]))

            raw_s = (
                pp_style_weight * style_sim_v +
                pp_genre_weight * genre_jac +
                pp_risk_weight * risk_m +
                pp_stage_weight * (1.0 - stage_d / 4.0) +
                pp_noise_weight * rng.rand()
            )
            p_edge_v = sigmoid(raw_s, scale=pp_logistic_scale, bias=pp_logistic_bias)
            candidates.append((float(p_edge_v), int(j), style_sim_v, genre_jac, stage_d))

        if not candidates:
            continue
        candidates.sort(key=lambda item: item[0], reverse=True)
        phase2_candidate_pairs += len(candidates)
        for p_edge_v, j, style_sim_v, genre_jac, stage_d in candidates[:_CROSS_GENRE_K]:
            pid_a, pid_b = pid_list[i], pid_list[j]
            if (edge_count[pid_a] >= person_cap[pid_a] or
                    edge_count[pid_b] >= person_cap[pid_b]):
                continue

            thresh = max(
                pp_base_threshold / (1.0 + pp_degree_decay * (edge_count[pid_a] ** pp_degree_power)),
                pp_base_threshold / (1.0 + pp_degree_decay * (edge_count[pid_b] ** pp_degree_power)),
            ) + _CROSS_GENRE_BUMP

            if p_edge_v < thresh:
                continue

            _classify_and_add(
                i, j,
                style_sim_v=style_sim_v,
                genre_jaccard=genre_jac,
                stage_diff=stage_d,
                p_edge_val=p_edge_v,
                threshold_extra=_CROSS_GENRE_BUMP,
            )

    phase2_elapsed = time.perf_counter() - phase2_started
    total_elapsed = time.perf_counter() - build_started
    print(f"  Phase 2 (cross-genre, K={_CROSS_GENRE_K}): "
          f"{len(edges) - p2_before} additional edges from {phase2_candidate_pairs:,} scored pairs in {phase2_elapsed:.2f}s")
    print(f"  Person-person build total: {total_elapsed:.2f}s "
          f"(max bucket {max_bucket_size:,}, avg bucket {avg_bucket_size})")
    LAST_PP_BUILD_STATS = {
        "people": int(n),
        "genre_bucket_count": int(len(genre_buckets)),
        "max_bucket_size": int(max_bucket_size),
        "avg_bucket_size": float(avg_bucket_size),
        "phase1_block_size": int(phase1_block_size),
        "phase1_pairs": int(phase1_pairs),
        "phase1_candidates": int(phase1_candidates),
        "phase1_edges": int(p2_before),
        "phase1_elapsed_sec": round(float(phase1_elapsed), 4),
        "phase2_candidate_pairs": int(phase2_candidate_pairs),
        "phase2_added_edges": int(len(edges) - p2_before),
        "phase2_elapsed_sec": round(float(phase2_elapsed), 4),
        "total_elapsed_sec": round(float(total_elapsed), 4),
    }
    return edges





def iter_person_company_edges(persons, companies, person_latent, company_latent, rng, world_policy: dict[str, Any] | None = None):
    """Yield person<->company edges from latent variables with inline validity windows."""
    p_latent_map = {lv["person_id"]: lv for lv in person_latent}
    c_latent_map = {lv["company_id"]: lv for lv in company_latent}
    company_by_id = {c["company_id"]: c for c in companies}
    pc_risk_weight = _edge_prior_float("person_company_risk_weight", 0.30, lo=0.0, hi=1.0)
    pc_budget_weight = _edge_prior_float("person_company_budget_weight", 0.30, lo=0.0, hi=1.0)
    pc_genre_weight = _edge_prior_float("person_company_genre_weight", 0.30, lo=0.0, hi=1.0)
    pc_noise_weight = _edge_prior_float("person_company_noise_weight", 0.10, lo=0.0, hi=1.0)
    pc_blacklist_threshold = _edge_prior_float("person_company_blacklist_threshold", 0.30, lo=0.0, hi=1.0)
    pc_brand_fit_threshold = _edge_prior_float("person_company_brand_fit_threshold", 0.60, lo=0.0, hi=1.0)
    pc_config = _pc_generation_config()
    start_year, end_year = year_bounds_from_env(1950, 2025)
    person_ranges = _build_person_ranges(persons)
    company_ranges = _build_company_ranges(companies)
    default_person = (start_year, end_year + 35)
    default_company = (max(1900, start_year - 5), end_year + 40)

    # Pre-index companies by specialty_genres (lowercase)
    companies_by_genre: dict = defaultdict(list)
    for c in companies:
        specialty_genres = c.get("specialty_genres", [])
        if isinstance(specialty_genres, str):
            specialty_genres = [g.strip() for g in specialty_genres.split(",") if g.strip()]
        for g in specialty_genres:
            companies_by_genre[g.lower()].append(c)

    n_companies = len(companies)

    for p in persons:
        pid = p["person_id"]
        plv = p_latent_map.get(pid)
        if plv is None:
            continue

        p_risk = plv.get("risk_tolerance", 0.5)
        p_controversy = plv.get("controversy_score", 0.0)
        p_budget_pref = plv.get("budget_band_pref", [0.5] * 5)
        p_genre = project_genres_to_company_basis(p.get("genre_affinity", []))

        # Candidate companies: genre-overlapping + random supplement
        candidate_cids: set = set()
        person_genres = p.get("genre_affinity", [])
        if isinstance(person_genres, str):
            person_genres = [g.strip() for g in person_genres.split(",") if g.strip()]
        for g in person_genres:
            for c in companies_by_genre.get(g.lower(), []):
                candidate_cids.add(c["company_id"])
        # Random supplement -- ensures cross-genre brand_fit edges can exist
        supp_n = min(int(round(pc_config["genre_supplement_size"])), n_companies)
        if supp_n > 0:
            for idx in rng.choice(n_companies, size=supp_n, replace=False):
                candidate_cids.add(companies[int(idx)]["company_id"])

        for cid in sorted(candidate_cids):
            c = company_by_id.get(int(cid))
            if c is None:
                continue
            if cid == pid:              # self-loop guard
                continue
            clv = c_latent_map.get(cid)
            if clv is None:
                continue

            c_risk = clv.get("risk_appetite", 0.5)
            c_controversy_tol = clv.get("controversy_tolerance", 0.5)
            c_budget_focus = clv.get("budget_tier_focus", [0.2] * 5)
            c_genre = canonical_company_genre_vector(clv.get("genre_portfolio", [0.083] * 12))

            risk_match = 1.0 - abs(p_risk - c_risk)
            budget_overlap = dot_overlap(p_budget_pref, c_budget_focus)
            genre_overlap = dot_overlap(p_genre, c_genre)
            policy_mult = 1.0
            if isinstance(world_policy, dict) and world_policy:
                company_strategy = resolve_company_strategy(world_policy, cid)
                if company_strategy == "event_franchise" and max(p_budget_pref) == p_budget_pref[0]:
                    policy_mult *= float(pc_config["event_franchise_micro_budget_penalty_boost"])
                company_market = infer_market(c.get("country"))
                person_markets = p.get("market_fit", [])
                if isinstance(person_markets, str):
                    person_markets = [m.strip() for m in person_markets.split(",") if m.strip()]
                if company_market in person_markets:
                    policy_mult *= float(pc_config["market_fit_boost"])

            raw_score = (
                pc_risk_weight * risk_match +
                pc_budget_weight * budget_overlap +
                pc_genre_weight * genre_overlap +
                pc_noise_weight * rng.rand()
            ) * policy_mult

            # Blacklist: controversial person + intolerant company
            if (
                p_controversy > pc_config["controversy_blacklist_person_threshold"]
                and c_controversy_tol < pc_config["controversy_blacklist_company_threshold"]
            ):
                if raw_score > pc_blacklist_threshold:
                    edge = {
                        "src_id":      cid,
                        "dst_id":      pid,
                        "src_name":    c.get("name", "?"),
                        "dst_name":    p.get("name", "?"),
                        "edge_type":   "blacklist",
                        "sign":        "-",
                        "weight":      round(
                            pc_config["blacklist_weight_base"] + pc_config["blacklist_weight_controversy_scale"] * p_controversy,
                            2,
                        ),
                        "source_kind": "latent_hybrid",
                        "reason":      f"controversy={p_controversy:.2f} vs tolerance={c_controversy_tol:.2f}",
                    }
                    _apply_validity_window(
                        edge,
                        company_ranges.get(int(cid), default_company),
                        person_ranges.get(int(pid), default_person),
                    )
                    yield edge
                continue

            # Brand fit: high affinity (raised threshold: 0.55 -> 0.60 to avoid edge explosion)
            if raw_score > pc_brand_fit_threshold:
                edge = {
                    "src_id":      cid,
                    "dst_id":      pid,
                    "src_name":    c.get("name", "?"),
                    "dst_name":    p.get("name", "?"),
                    "edge_type":   "brand_fit",
                    "sign":        "+",
                    "weight":      round(raw_score, 2),
                    "source_kind": "latent_hybrid",
                    "reason":      f"risk={risk_match:.2f} budget={budget_overlap:.2f} genre={genre_overlap:.2f}",
                }
                _apply_validity_window(
                    edge,
                    company_ranges.get(int(cid), default_company),
                    person_ranges.get(int(pid), default_person),
                )
                yield edge


def generate_person_company_edges(persons, companies, person_latent, company_latent, rng, world_policy: dict[str, Any] | None = None):
    return list(
        iter_person_company_edges(
            persons,
            companies,
            person_latent,
            company_latent,
            rng,
            world_policy=world_policy,
        )
    )



def iter_company_company_edges(companies, company_latent, rng, world_policy: dict[str, Any] | None = None):
    """Yield company<->company edges from latent variables with inline validity windows."""
    c_latent_map = {lv["company_id"]: lv for lv in company_latent}
    cc_config = _cc_generation_config()
    start_year, end_year = year_bounds_from_env(1950, 2025)
    company_ranges = _build_company_ranges(companies)
    default_company = (max(1900, start_year - 5), end_year + 40)

    n = len(companies)
    for i in range(n):
        c_a = companies[i]
        cid_a = c_a["company_id"]
        clv_a = c_latent_map.get(cid_a)
        if clv_a is None:
            continue

        genre_a = canonical_company_genre_vector(clv_a.get("genre_portfolio", [0.083] * 12))
        tier_a = clv_a.get("budget_tier_focus", [0.2]*5)

        for j in range(i + 1, n):
            c_b = companies[j]
            cid_b = c_b["company_id"]
            clv_b = c_latent_map.get(cid_b)
            if clv_b is None:
                continue

            genre_b = canonical_company_genre_vector(clv_b.get("genre_portfolio", [0.083] * 12))
            tier_b = clv_b.get("budget_tier_focus", [0.2]*5)

            genre_overlap = dot_overlap(genre_a, genre_b)
            tier_overlap = dot_overlap(tier_a, tier_b)
            policy_mult = 1.0
            if isinstance(world_policy, dict) and world_policy:
                strategy_a = resolve_company_strategy(world_policy, cid_a)
                strategy_b = resolve_company_strategy(world_policy, cid_b)
                if strategy_a == strategy_b:
                    policy_mult *= cc_config["strategy_match_boost"]
                market_a = infer_market(c_a.get("country"))
                market_b = infer_market(c_b.get("country"))
                if market_a == market_b:
                    policy_mult *= cc_config["market_match_boost"]

            # Same genre + same tier = market rivals
            if (
                genre_overlap * policy_mult > cc_config["rival_overlap_threshold"]
                and tier_overlap > cc_config["rival_tier_threshold"]
            ):
                edge = {
                    "src_id": cid_a,
                    "dst_id": cid_b,
                    "src_name": c_a.get("name", "?"),
                    "dst_name": c_b.get("name", "?"),
                    "edge_type": "market_rival",
                    "sign": "-",
                    "weight": round(
                        genre_overlap
                        * tier_overlap
                        * cc_config["rival_weight_scale"]
                        * min(cc_config["rival_weight_policy_cap"], policy_mult),
                        2,
                    ),
                    "source_kind": "latent_hybrid",
                    "reason": f"genre_overlap={genre_overlap:.2f} tier_overlap={tier_overlap:.2f}",
                }
                _apply_validity_window(
                    edge,
                    company_ranges.get(int(cid_a), default_company),
                    company_ranges.get(int(cid_b), default_company),
                )
                yield edge
            # Complementary tiers + overlapping genres = co-production candidates
            elif (
                genre_overlap * policy_mult > cc_config["coproduction_overlap_threshold"]
                and tier_overlap < cc_config["coproduction_tier_max"]
            ):
                edge = {
                    "src_id": cid_a,
                    "dst_id": cid_b,
                    "src_name": c_a.get("name", "?"),
                    "dst_name": c_b.get("name", "?"),
                    "edge_type": "co_production",
                    "sign": "+",
                    "weight": round(
                        genre_overlap
                        * cc_config["coproduction_weight_scale"]
                        * min(cc_config["coproduction_policy_cap"], policy_mult),
                        2,
                    ),
                    "source_kind": "latent_hybrid",
                    "reason": f"genre_shared={genre_overlap:.2f} complementary_tiers",
                }
                _apply_validity_window(
                    edge,
                    company_ranges.get(int(cid_a), default_company),
                    company_ranges.get(int(cid_b), default_company),
                )
                yield edge


def generate_company_company_edges(companies, company_latent, rng, world_policy: dict[str, Any] | None = None):
    return list(iter_company_company_edges(companies, company_latent, rng, world_policy=world_policy))


# ═══════════════════════════════════════════════════════════════════════
# SOCIAL GRAPH ENRICHMENT
# ═══════════════════════════════════════════════════════════════════════

def _add_serendipitous_edges(pp_edges: list, persons: list, latent_map: dict, rng) -> list:
    """Phase 3: Random cross-community 'serendipitous' friendship edges.

    Real social networks have unexpected connections that procedural similarity
    scoring can never produce (same film school, shared agent, random festival
    encounter). These edges punch wormholes through community walls at random
    spots, making the graph topology genuinely unpredictable.

    Stage-gated injection probability:
      legend:  65% chance of 1-3 serendipitous edges
      veteran: 35% chance of 1-2
      prime:   15% chance of 1
      rising:   5% chance of 1

    Tagged with source_kind='serendipitous' so queries can distinguish them.
    Weight range 0.30-0.55 (moderate -- strong enough to matter, weak enough
    to not dominate over within-community ties).
    """
    config = _serendipitous_config()
    seren_prob = config["stage_probabilities"]
    seren_n = config["stage_max_new_edges"]
    weight_min = min(config["weight_min"], config["weight_max"])
    weight_max = max(config["weight_min"], config["weight_max"])
    candidate_multiplier = max(1, int(config["candidate_multiplier"]))
    pg_list = [_primary_genre(p) for p in persons]

    # Existing pair set for dedup
    existing = {(min(e["src_id"], e["dst_id"]), max(e["src_id"], e["dst_id"]))
                for e in pp_edges}

    name_map = {p["person_id"]: p.get("name", "?") for p in persons}
    n = len(persons)
    added = 0

    for i, p in enumerate(persons):
        pid   = p["person_id"]
        stage = p.get("career_stage", "prime")
        prob = float(seren_prob.get(stage, seren_prob.get("prime", 0.05)))
        if rng.rand() >= prob:
            continue
        max_edges = max(1, int(seren_n.get(stage, seren_n.get("prime", 1))))
        n_edges = rng.randint(1, max_edges + 1)

        my_genre = pg_list[i]
        # Sample candidates from different genre + randomly scattered nationality
        n_sample = min(n_edges * candidate_multiplier, n - 1)
        raw = rng.choice(n, size=n_sample, replace=False)
        cross = [j for j in raw if j != i and pg_list[j] != my_genre]

        for j in cross[:n_edges]:
            pid_b = persons[j]["person_id"]
            pair  = (min(pid, pid_b), max(pid, pid_b))
            if pair in existing:
                continue
            existing.add(pair)
            w = round(weight_min + max(0.0, weight_max - weight_min) * rng.rand(), 2)
            pp_edges.append({
                "src_id":      min(pid, pid_b),
                "dst_id":      max(pid, pid_b),
                "src_name":    name_map.get(min(pid, pid_b), "?"),
                "dst_name":    name_map.get(max(pid, pid_b), "?"),
                "edge_type":   "friendship",
                "sign":        "+",
                "weight":      w,
                "source_kind": "serendipitous",
                "reason":      f"serendipitous cross-genre bridge",
            })
            added += 1

    print(f"  Serendipitous edges added: {added}")
    return pp_edges


def _add_triadic_closure(pp_edges: list, persons: list, rng) -> list:
    """Triadic closure: friend-of-friend becomes friend.

    If A--B and B--C are friendship edges, close the open triad A--C with a
    career-stage-gated probability. Creates tight friend GROUPS (high clustering
    coefficient) rather than just pairwise links.

    Stage-gated closure probability per open triad:
      legend:  25%  -- very tight A-list circles (realistic)
      veteran: 12%  -- established but more open
      prime:    5%  -- still building network
      rising:   0%  -- open network, not yet embedded

    Only applied to actors/directors (not all persons) to keep it focused.
    Respects existing per-person degree caps to avoid runaway degree growth.
    """
    config = _triadic_closure_config()
    closure_prob = config["stage_probabilities"]
    cap_extra = int(config["extra_cap"])
    weight_min = min(config["weight_min"], config["weight_max"])
    weight_max = max(config["weight_min"], config["weight_max"])

    stage_map = {p["person_id"]: p.get("career_stage", "prime") for p in persons}
    name_map  = {p["person_id"]: p.get("name", "?")             for p in persons}

    # Build friendship adjacency (only friendship sign=+)
    friends: dict = defaultdict(set)
    for e in pp_edges:
        if e.get("edge_type") == "friendship" and e.get("sign") == "+":
            a, b = e["src_id"], e["dst_id"]
            friends[a].add(b)
            friends[b].add(a)

    existing = {(min(e["src_id"], e["dst_id"]), max(e["src_id"], e["dst_id"]))
                for e in pp_edges}
    closure_count: dict = defaultdict(int)

    new_edges = []
    # Use sorted node list for determinism
    for pid_b in sorted(friends.keys()):
        flist = list(friends[pid_b])
        for ai in range(len(flist)):
            pid_a = flist[ai]
            for pid_c in flist[ai + 1:]:
                pair = (min(pid_a, pid_c), max(pid_a, pid_c))
                if pair in existing:
                    continue
                # Apply closure prob of the HIGHER-status node (the hub)
                highest_stage = max(
                    closure_prob.get(stage_map.get(pid_a, "prime"), closure_prob.get("prime", 0.05)),
                    closure_prob.get(stage_map.get(pid_b, "prime"), closure_prob.get("prime", 0.05)),
                    closure_prob.get(stage_map.get(pid_c, "prime"), closure_prob.get("prime", 0.05)),
                )
                if rng.rand() >= highest_stage:
                    continue
                if closure_count[pid_a] >= cap_extra or closure_count[pid_c] >= cap_extra:
                    continue
                existing.add(pair)
                closure_count[pid_a] += 1
                closure_count[pid_c] += 1
                w = round(weight_min + max(0.0, weight_max - weight_min) * rng.rand(), 2)
                new_edges.append({
                    "src_id":      min(pid_a, pid_c),
                    "dst_id":      max(pid_a, pid_c),
                    "src_name":    name_map.get(min(pid_a, pid_c), "?"),
                    "dst_name":    name_map.get(max(pid_a, pid_c), "?"),
                    "edge_type":   "friendship",
                    "sign":        "+",
                    "weight":      w,
                    "source_kind": "triadic_closure",
                    "reason":      f"mutual friend {name_map.get(pid_b, '?')}",
                })

    print(f"  Triadic closure edges added: {len(new_edges)}")
    return pp_edges + new_edges


# ═══════════════════════════════════════════════════════════════════════
# COMMUNITY DETECTION (Louvain-inspired label propagation)
# ═══════════════════════════════════════════════════════════════════════

def detect_communities_louvain(edges, min_community_size=3, max_iterations=100, seed=42,
                               max_community_fraction=0.25):
    """Weighted label propagation on friendship subgraph.

    D10 fix: max_community_fraction=0.25 (default) -- any community that
    exceeds 25% of all nodes is forcibly split. This prevents the 80%%
    mega-community that appeared in V12 where label propagation converged
    to a single dominant community. No manual resolution parameter needed.

    Returns: {node_id: community_id}
    """
    rng = np.random.RandomState(seed)

    # Build adjacency from friendship edges only
    adj = defaultdict(list)  # node_id -> [(neighbor_id, weight)]
    nodes = set()

    for e in edges:
        if e.get("edge_type") == "friendship" and e.get("sign") == "+":
            src, dst = e["src_id"], e["dst_id"]
            w = e.get("weight", 0.5)
            adj[src].append((dst, w))
            adj[dst].append((src, w))
            nodes.add(src)
            nodes.add(dst)

    if not nodes:
        return {}

    # Initialize: each node is its own community
    labels = {n: n for n in nodes}
    node_list = list(nodes)

    for iteration in range(max_iterations):
        rng.shuffle(node_list)
        changed = 0

        for node in node_list:
            if not adj[node]:
                continue

            # Weighted vote from neighbors
            votes = defaultdict(float)
            for neighbor, weight in adj[node]:
                votes[labels[neighbor]] += weight

            if votes:
                best_label = max(votes, key=votes.get)
                if labels[node] != best_label:
                    labels[node] = best_label
                    changed += 1

        if changed == 0:
            break

    # D10: Forcibly split communities that exceed max_community_fraction
    n_nodes = len(nodes)
    max_size = max(int(max_community_fraction * n_nodes), min_community_size + 1)
    next_label = max(labels.values()) + 1

    for _split_pass in range(5):  # up to 5 rounds of splitting
        community_members = defaultdict(list)
        for node, label in labels.items():
            community_members[label].append(node)

        oversized = {k: v for k, v in community_members.items() if len(v) > max_size}
        if not oversized:
            break

        for label, members in oversized.items():
            # Sort by internal connectivity (least-connected first -> new community seed)
            internal_w = {}
            for node in members:
                w_sum = sum(w for nb, w in adj[node] if labels[nb] == label)
                internal_w[node] = w_sum
            sorted_members = sorted(members, key=lambda n: internal_w[n])

            # Move the bottom half to a new label
            split_size = len(members) // 2
            for node in sorted_members[:split_size]:
                labels[node] = next_label
            next_label += 1

    # Renumber communities and merge tiny ones
    community_members = defaultdict(list)
    for node, label in labels.items():
        community_members[label].append(node)

    # Find tiny communities and merge into nearest large one
    large_comms = {k: v for k, v in community_members.items() if len(v) >= min_community_size}
    tiny_comms = {k: v for k, v in community_members.items() if len(v) < min_community_size}

    for tiny_label, tiny_nodes in tiny_comms.items():
        # Find nearest large community by total edge weight
        best_large = None
        best_weight = -1
        for node in tiny_nodes:
            for neighbor, w in adj[node]:
                n_label = labels[neighbor]
                if n_label in large_comms and w > best_weight:
                    best_large = n_label
                    best_weight = w

        if best_large is not None:
            for node in tiny_nodes:
                labels[node] = best_large
        # else: keep as orphan community

    # Final renumbering
    unique_labels = sorted(set(labels.values()))
    label_map = {old: new + 1 for new, old in enumerate(unique_labels)}

    return {node: label_map[label] for node, label in labels.items()}



# ═══════════════════════════════════════════════════════════════════════
# TEMPORAL DIMENSION
# ═══════════════════════════════════════════════════════════════════════

def add_temporal_edges(edges, persons):
    """Add valid_from/valid_to to edges based on career timelines.

    v10: Uses actual debut_year/retirement_year when available (set by
    _assign_career_timelines which is YEAR_RANGE-aware). Falls back to
    stage-based estimates for persons missing timeline data.
    """
    person_map = {}
    start_year, end_year = year_bounds_from_env(1950, 2025)
    veteran_start = start_year + max(5, (end_year - start_year) // 6)
    prime_start = start_year + max(10, (end_year - start_year) // 3)
    rising_start = max(start_year, end_year - max(12, (end_year - start_year) // 4))
    extended_end = end_year + 10
    active_end = end_year + 35
    for p in persons:
        pid = p.get("person_id")
        if pid is None:
            continue
        # Prefer actual timeline data from person record
        debut = p.get("debut_year")
        retire = p.get("retirement_year")
        if debut is not None and retire is not None:
            person_map[pid] = (int(debut), int(retire))
        else:
            if current_mode() == "research":
                audit_fallback_hit(
                    "generate_edges_hybrid.person_validity",
                    f"missing:timeline:{pid}",
                    detail="person validity fallback from career stage is not allowed in research mode",
                    mode="research",
                )
            # Fallback: approximate from career stage
            stage = p.get("career_stage", "prime")
            if stage == "legend":
                person_map[pid] = (start_year, active_end)
            elif stage == "veteran":
                person_map[pid] = (veteran_start, end_year + 5)
            elif stage == "prime":
                person_map[pid] = (prime_start, active_end)
            elif stage == "rising":
                person_map[pid] = (rising_start, end_year + 25)

            else:  # retired
                person_map[pid] = (start_year, max(start_year, end_year - 10))

    for e in edges:
        src_range = person_map.get(e["src_id"], (start_year, extended_end + 40))
        dst_range = person_map.get(e["dst_id"], (start_year, extended_end + 40))
        # Overlap: relationship valid when both are active
        valid_from = max(src_range[0], dst_range[0])
        valid_to = min(src_range[1], dst_range[1])
        if valid_to < valid_from:
            valid_to = valid_from  # minimal validity
        e["valid_from"] = valid_from
        e["valid_to"] = valid_to

    return edges


# ═══════════════════════════════════════════════════════════════════════
# RELATIONSHIP CALIBRATION (v9+)
# ═══════════════════════════════════════════════════════════════════════

def _role_set(p) -> set:
    roles = p.get("roles", [])
    if roles is None:
        return set()
    if isinstance(roles, str):
        parts = [r.strip().lower() for r in roles.split(",") if r.strip()]
        return set(parts)
    if isinstance(roles, (list, tuple)):
        return set(str(r).strip().lower() for r in roles if str(r).strip())
    return set()


def _stage_index(p) -> int:
    st = p.get("career_stage", "prime")
    try:
        return CAREER_STAGES.index(st)
    except Exception:
        return CAREER_STAGES.index("prime")


def _genre_set(p) -> set:
    gs = p.get("main_genres")
    if gs is None:
        gs = p.get("genre_affinity")
    if gs is None:
        return set()
    if isinstance(gs, str):
        return set(x.strip().lower() for x in gs.split(",") if x.strip())
    if isinstance(gs, (list, tuple)):
        return set(str(x).strip().lower() for x in gs if str(x).strip())
    return set()


def _style_vec(latent_map: dict, pid: int) -> list:
    lv = latent_map.get(pid) or {}
    v = lv.get("creative_style_vector")
    if isinstance(v, list) and len(v) >= 8:
        return [float(x) for x in v[:8]]
    return [0.0] * 8


def _safe01(x, default=0.5) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    if v != v:
        return float(default)
    return max(0.0, min(1.0, v))




def _coerce_year(raw, default: int) -> int:
    try:
        y = int(raw)
        if y <= 0:
            raise ValueError
        return y
    except Exception:
        return int(default)


def _build_person_ranges(persons: list[dict]) -> dict[int, tuple[int, int]]:
    ranges: dict[int, tuple[int, int]] = {}
    start_year, end_year = year_bounds_from_env(1950, 2025)
    for p in persons:
        try:
            pid = int(p.get("person_id"))
        except Exception:
            continue
        debut = _coerce_year(p.get("debut_year"), start_year)
        retire = _coerce_year(p.get("retirement_year"), end_year + 35)
        if retire < debut:
            retire = debut
        ranges[pid] = (debut, retire)
    return ranges


def _build_company_ranges(companies: list[dict]) -> dict[int, tuple[int, int]]:
    ranges: dict[int, tuple[int, int]] = {}
    start_year, end_year = year_bounds_from_env(1950, 2025)
    for c in companies:
        try:
            cid = int(c.get("company_id"))
        except Exception:
            continue
        founded = _coerce_year(c.get("founded_year"), max(1900, start_year - 5))
        defunct = _coerce_year(c.get("defunct_year"), end_year + 40)
        if defunct < founded:
            defunct = founded
        ranges[cid] = (founded, defunct)
    return ranges


def _apply_validity_window(edge: dict, src_range: tuple[int, int], dst_range: tuple[int, int]):
    vf = max(int(src_range[0]), int(dst_range[0]))
    vt = min(int(src_range[1]), int(dst_range[1]))
    if vt < vf:
        vt = vf
    edge["valid_from"] = int(vf)
    edge["valid_to"] = int(vt)


def add_temporal_to_non_person_edges(
    pc_edges: list[dict],
    cc_edges: list[dict],
    persons: list[dict],
    companies: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Ensure person-company/company-company edges also have non-null validity windows."""
    person_ranges = _build_person_ranges(persons)
    company_ranges = _build_company_ranges(companies)

    start_year, end_year = year_bounds_from_env(1950, 2025)
    default_person = (start_year, end_year + 35)
    default_company = (max(1900, start_year - 5), end_year + 40)

    for e in pc_edges:
        try:
            src = int(e.get("src_id"))
            dst = int(e.get("dst_id"))
        except Exception:
            continue
        src_person = person_ranges.get(src)
        dst_person = person_ranges.get(dst)
        src_company = company_ranges.get(src)
        dst_company = company_ranges.get(dst)

        person_range = src_person or dst_person or default_person
        company_range = src_company or dst_company or default_company
        _apply_validity_window(e, person_range, company_range)

    for e in cc_edges:
        try:
            src = int(e.get("src_id"))
            dst = int(e.get("dst_id"))
        except Exception:
            continue
        src_range = company_ranges.get(src, default_company)
        dst_range = company_ranges.get(dst, default_company)
        _apply_validity_window(e, src_range, dst_range)

    return pc_edges, cc_edges


def dedupe_edges(edges: list[dict]) -> tuple[list[dict], int]:
    """Deduplicate edges deterministically, preserving strongest weight per key."""
    best: dict[tuple[str, str, int, int], dict] = {}
    replaced = 0

    for raw in edges:
        e = dict(raw)
        try:
            src = int(e.get("src_id"))
            dst = int(e.get("dst_id"))
        except Exception:
            continue
        if src == dst:
            continue

        et = str(e.get("edge_type", "")).strip().lower()
        sign = str(e.get("sign", "+")).strip()

        if et in _UNDIRECTED_EDGE_TYPES and src > dst:
            src, dst = dst, src
            sn = e.get("src_name")
            dn = e.get("dst_name")
            e["src_name"] = dn
            e["dst_name"] = sn

        e["src_id"] = src
        e["dst_id"] = dst

        try:
            w = float(e.get("weight", 0.0))
        except Exception:
            w = 0.0
        e["weight"] = round(max(0.0, min(1.0, w)), 4)

        start_year, end_year = year_bounds_from_env(1950, 2025)
        vf = _coerce_year(e.get("valid_from"), start_year)
        vt = _coerce_year(e.get("valid_to"), end_year + 40)
        if vt < vf:
            vt = vf
        e["valid_from"] = vf
        e["valid_to"] = vt

        key = (et, sign, src, dst)
        old = best.get(key)
        if old is None:
            best[key] = e
            continue

        if float(e.get("weight", 0.0)) > float(old.get("weight", 0.0)):
            best[key] = e
            replaced += 1
            continue

        old_vf = _coerce_year(old.get("valid_from"), start_year)
        old_vt = _coerce_year(old.get("valid_to"), end_year + 40)
        overlap_from = max(old_vf, vf)
        overlap_to = min(old_vt, vt)
        if overlap_to < overlap_from:
            overlap_to = overlap_from
        old["valid_from"] = overlap_from
        old["valid_to"] = overlap_to
        replaced += 1

    return list(best.values()), replaced


def complete_community_assignments(communities: dict[int, int], persons: list[dict]) -> tuple[dict[int, int], int]:
    """Assign a deterministic community to every person, including isolates."""
    all_pids = []
    for p in persons:
        try:
            all_pids.append(int(p.get("person_id")))
        except Exception:
            continue

    out = {int(pid): int(comm) for pid, comm in communities.items()}
    existing = sorted(set(out.values()))
    added = 0

    if not existing:
        n = max(8, min(64, int(math.sqrt(max(1, len(all_pids))))))
        for pid in all_pids:
            bucket = int(hashlib.md5(f"community|{pid}".encode("utf-8")).hexdigest(), 16) % n
            out[pid] = bucket + 1
            added += 1
        return out, added

    for pid in all_pids:
        if pid in out:
            continue
        h = int(hashlib.md5(f"community|{pid}".encode("utf-8")).hexdigest(), 16)
        out[pid] = existing[h % len(existing)]
        added += 1

    return out, added

def calibrate_relationship_targets(pp_edges: list, persons: list, latent_map: dict, rng,
                                   targets: dict = RELATIONSHIP_TARGETS,
                                   seed: int = SEED) -> list:
    """Post-process person<->person edges to match coarse relationship targets.

    Targets come from contracts.RELATIONSHIP_TARGETS.

    Implemented (approx.):
      - best_friend_rate: fraction of actors with >=1 friendship (actor<->actor)
      - rival_rate: fraction of actors with >=1 rivalry (actor<->actor)
      - bf_same_community_rate: fraction of actor best-friend links within community
      - director_preferred_actors: ~N outgoing mentorship edges per director (-> actors)
      - director_avoided_actors: ~N outgoing avoid edges per director (-> actors)

    Notes:
      - Deterministic by using the seeded RNG.
      - Calibration operates on the *static* graph; activation remains probabilistic
        at movie-assembly time (generate_movies.py).
    """

    global LAST_CALIBRATION_STATS
    LAST_CALIBRATION_STATS = {}

    if not persons:
        return pp_edges

    cal = _calibration_config()
    effective_targets = dict(targets or {})
    for key, value in cal["relationship_targets"].items():
        effective_targets[str(key)] = value

    # ─── Precompute metadata ─────────────────────────────────────────
    name_map = {int(p["person_id"]): p.get("name", "") for p in persons}
    roles_map = {int(p["person_id"]): _role_set(p) for p in persons}
    stage_map = {int(p["person_id"]): _stage_index(p) for p in persons}
    genres_map = {int(p["person_id"]): _genre_set(p) for p in persons}
    style_map = {int(p["person_id"]): _style_vec(latent_map, int(p["person_id"])) for p in persons}

    actor_ids = [pid for pid, rs in roles_map.items() if "actor" in rs]
    actor_id_set = set(actor_ids)
    director_ids = [pid for pid, rs in roles_map.items() if "director" in rs]

    if not actor_ids:
        return pp_edges

    # Edge index for updates (avoid duplicates)
    def _edge_key(et: str, sign: str, src: int, dst: int):
        # Normalize ALL types to min/max so (A->B) and (B->A) are the same key.
        # Mentorship direction is stored in src/dst, but for dedup purposes
        # we only want one edge per (pair, type) regardless of which way it points.
        a, b = (min(src, dst), max(src, dst))
        return (et, sign, a, b)

    edge_pos = {}
    for i, e in enumerate(pp_edges):
        try:
            k = _edge_key(e.get("edge_type"), e.get("sign"), int(e.get("src_id")), int(e.get("dst_id")))
        except Exception:
            continue
        if k not in edge_pos:
            edge_pos[k] = i
        else:
            old_i = edge_pos[k]
            if float(e.get("weight", 0)) > float(pp_edges[old_i].get("weight", 0)):
                edge_pos[k] = i

    def _upsert_edge(et: str, sign: str, src: int, dst: int, weight: float, reason: str):
        w = float(max(cal["upsert_weight_min"], min(cal["upsert_weight_max"], weight)))
        k = _edge_key(et, sign, src, dst)
        if k in edge_pos:
            e = pp_edges[edge_pos[k]]
            if w > float(e.get("weight", 0.0)):
                e["weight"] = w
                e["reason"] = reason
                e["source_kind"] = "latent_hybrid"
            return

        if et in {"friendship", "rivalry"}:
            src2, dst2 = (min(src, dst), max(src, dst))
        else:
            src2, dst2 = int(src), int(dst)

        new_e = {
            "src_id": int(src2),
            "dst_id": int(dst2),
            "src_name": name_map.get(int(src2), ""),
            "dst_name": name_map.get(int(dst2), ""),
            "edge_type": et,
            "sign": sign,
            "weight": w,
            "source_kind": "latent_hybrid",
            "reason": reason,
        }
        pp_edges.append(new_e)
        edge_pos[k] = len(pp_edges) - 1

    # Build adjacency for actor<->actor friendships/rivalries
    friends = defaultdict(dict)
    rivals_adj = defaultdict(dict)

    for e in pp_edges:
        et = e.get("edge_type")
        if et not in {"friendship", "rivalry"}:
            continue
        a = int(e.get("src_id"))
        b = int(e.get("dst_id"))
        if a not in actor_id_set or b not in actor_id_set:
            continue
        w = float(e.get("weight", 0.0))
        if et == "friendship" and e.get("sign") == "+":
            if w > friends[a].get(b, 0.0):
                friends[a][b] = w
                friends[b][a] = w
        elif et == "rivalry":
            if w > rivals_adj[a].get(b, 0.0):
                rivals_adj[a][b] = w
                rivals_adj[b][a] = w

    actor_ids_arr = np.array(actor_ids, dtype=int)
    stage_denom = max(1, len(CAREER_STAGES) - 1)

    def jaccard(sa: set, sb: set) -> float:
        if not sa and not sb:
            return 0.0
        inter = len(sa & sb)
        uni = len(sa | sb)
        return inter / uni if uni else 0.0

    def style_sim(a: int, b: int) -> float:
        return 0.5 * (cosine_similarity(style_map[a], style_map[b]) + 1.0)

    def stage_sim(a: int, b: int) -> float:
        return 1.0 - abs(stage_map[a] - stage_map[b]) / stage_denom

    def friendship_score(a: int, b: int) -> float:
        friend_weights = cal["friendship_score_weights"]
        return (
            friend_weights["style"] * style_sim(a, b)
            + friend_weights["genre"] * jaccard(genres_map[a], genres_map[b])
            + friend_weights["stage"] * stage_sim(a, b)
        )

    def best_friend_map(communities: dict) -> tuple[dict[int, tuple[int, float]], float]:
        bf: dict[int, tuple[int, float]] = {}
        total = 0
        within = 0
        for pid in actor_ids:
            neigh = [(nbr, w) for nbr, w in friends.get(pid, {}).items() if nbr in actor_id_set]
            if not neigh:
                continue
            nbr, w = max(neigh, key=lambda item: item[1])
            bf[pid] = (nbr, w)
            total += 1
            if communities.get(pid) is not None and communities.get(pid) == communities.get(nbr):
                within += 1
        ratio = (within / total) if total else 1.0
        return bf, ratio

    target_bf = float(effective_targets.get("best_friend_rate", 0.0))
    want_bf = int(round(target_bf * len(actor_ids)))
    have_bf = sum(1 for pid in actor_ids if friends.get(pid))
    target_r = float(effective_targets.get("rival_rate", 0.0))
    want_r = int(round(target_r * len(actor_ids)))
    have_r = sum(1 for pid in actor_ids if rivals_adj.get(pid))
    target_same = float(effective_targets.get("bf_same_community_rate", 0.0))

    calibration_stats: dict[str, Any] = {
        "actors": int(len(actor_ids)),
        "directors": int(len(director_ids)),
        "existing_friend_coverage": round(have_bf / max(1, len(actor_ids)), 4),
        "target_friend_coverage": round(target_bf, 4),
        "existing_rival_coverage": round(have_r / max(1, len(actor_ids)), 4),
        "target_rival_coverage": round(target_r, 4),
        "same_community_target_min": round(target_same, 4),
        "friendship_topup_needed": bool(have_bf < want_bf),
        "rivalry_topup_needed": bool(have_r < want_r),
    }

    communities: dict[int, int] | None = None
    bf_map: dict[int, tuple[int, float]] = {}
    ratio = 1.0
    if target_same > 0 and len(actor_ids) >= 5:
        detect_started = time.perf_counter()
        communities = detect_communities_louvain(pp_edges, seed=seed)
        calibration_stats["same_community_detect_elapsed_sec"] = round(float(time.perf_counter() - detect_started), 4)
        bf_map, ratio = best_friend_map(communities)
        calibration_stats["same_community_ratio_before"] = round(float(ratio), 4)

    if have_bf < want_bf or have_r < want_r:
        actor_id_list = list(actor_ids)
        n_actors = len(actor_id_list)
        actor_idx_map = {pid: i for i, pid in enumerate(actor_id_list)}

        style_dim = max(len(v) for v in style_map.values()) if style_map else 8
        style_mat = np.zeros((n_actors, style_dim), dtype=np.float32)
        for i, pid in enumerate(actor_id_list):
            sv = style_map.get(pid, [])
            for j in range(min(len(sv), style_dim)):
                style_mat[i, j] = float(sv[j])
        norms = np.linalg.norm(style_mat, axis=1, keepdims=True)
        norms = np.where(norms > 1e-9, norms, 1.0)
        style_normed = style_mat / norms

        all_genres = sorted({g for gs in genres_map.values() for g in gs})
        genre_to_idx = {g: i for i, g in enumerate(all_genres)}
        genre_mat = np.zeros((n_actors, len(all_genres)), dtype=np.float32)
        for i, pid in enumerate(actor_id_list):
            for g in genres_map.get(pid, set()):
                gi = genre_to_idx.get(g)
                if gi is not None:
                    genre_mat[i, gi] = 1.0
        genre_sums = genre_mat.sum(axis=1)
        actor_stage_arr = np.array([stage_map.get(pid, 2) for pid in actor_id_list], dtype=np.float32)

        actor_genre_index: dict[str, list[int]] = defaultdict(list)
        for pid in actor_id_list:
            for g in genres_map.get(pid, set()):
                actor_genre_index[g].append(pid)
        candidate_sample_k = int(cal["candidate_sample_k"])

        def _sample_candidates(pid: int, exclude: set[int]) -> np.ndarray:
            cands: set[int] = set()
            for g in genres_map.get(pid, set()):
                cands.update(actor_genre_index.get(g, []))
            n_supp = min(candidate_sample_k, n_actors)
            for idx in rng.choice(n_actors, size=n_supp, replace=False):
                cands.add(actor_id_list[int(idx)])
            cands.discard(pid)
            cands -= exclude
            return np.array([actor_idx_map[c] for c in cands if c in actor_idx_map], dtype=int)

        def _vectorized_friendship_scores(pid_idx: int, cand_idxs: np.ndarray) -> np.ndarray:
            friend_weights = cal["friendship_score_weights"]
            style_scores = 0.5 * (style_normed[cand_idxs] @ style_normed[pid_idx] + 1.0)
            inter = genre_mat[cand_idxs] @ genre_mat[pid_idx]
            union = genre_sums[cand_idxs] + genre_sums[pid_idx] - inter
            union = np.where(union > 0, union, 1.0)
            genre_scores = inter / union
            stage_scores = 1.0 - np.abs(actor_stage_arr[cand_idxs] - actor_stage_arr[pid_idx]) / stage_denom
            return (
                friend_weights["style"] * style_scores
                + friend_weights["genre"] * genre_scores
                + friend_weights["stage"] * stage_scores
            )

        def _vectorized_rivalry_scores(pid_idx: int, cand_idxs: np.ndarray) -> np.ndarray:
            rival_weights = cal["rivalry_score_weights"]
            style_scores = 0.5 * (style_normed[cand_idxs] @ style_normed[pid_idx] + 1.0)
            inter = genre_mat[cand_idxs] @ genre_mat[pid_idx]
            union = genre_sums[cand_idxs] + genre_sums[pid_idx] - inter
            union = np.where(union > 0, union, 1.0)
            genre_scores = inter / union
            stage_scores = 1.0 - np.abs(actor_stage_arr[cand_idxs] - actor_stage_arr[pid_idx]) / stage_denom
            return (
                rival_weights["genre"] * genre_scores
                + rival_weights["style_distance"] * (1.0 - style_scores)
                + rival_weights["stage"] * stage_scores
            )

        if have_bf < want_bf:
            missing = [pid for pid in actor_ids if not friends.get(pid)]
            rng.shuffle(missing)
            added = 0
            for pid in missing:
                if have_bf + added >= want_bf:
                    break
                exclude = set(friends.get(pid, {}).keys()) | set(rivals_adj.get(pid, {}).keys()) | {pid}
                cand_idxs = _sample_candidates(pid, exclude)
                if len(cand_idxs) == 0:
                    continue
                pid_idx = actor_idx_map[pid]
                scores = _vectorized_friendship_scores(pid_idx, cand_idxs)
                degrees = np.array([len(friends.get(actor_id_list[ci], {})) for ci in cand_idxs], dtype=float)
                scores *= (1.0 + cal["preferential_attachment_log_weight"] * np.log1p(degrees))
                best_k = int(np.argmax(scores))
                best_sc = float(scores[best_k])
                best = actor_id_list[int(cand_idxs[best_k])]
                w = (
                    cal["best_friend_weight_base"]
                    + cal["best_friend_weight_score_scale"] * best_sc
                    + cal["best_friend_weight_noise"] * rng.rand()
                )
                _upsert_edge(
                    "friendship", "+", pid, best, w,
                    reason=f"CAL(best_friend_rate): friend_score={best_sc:.3f}",
                )
                friends[pid][best] = w
                friends[best][pid] = w
                added += 1
            print(f"[Calibrate] Added {added} friendships to reach best_friend_rate")
            calibration_stats["friendship_edges_added"] = int(added)

        if have_r < want_r:
            missing = [pid for pid in actor_ids if not rivals_adj.get(pid)]
            rng.shuffle(missing)
            added = 0
            for pid in missing:
                if have_r + added >= want_r:
                    break
                exclude = set(rivals_adj.get(pid, {}).keys()) | set(friends.get(pid, {}).keys()) | {pid}
                cand_idxs = _sample_candidates(pid, exclude)
                if len(cand_idxs) == 0:
                    continue
                pid_idx = actor_idx_map[pid]
                scores = _vectorized_rivalry_scores(pid_idx, cand_idxs)
                best_k = int(np.argmax(scores))
                best_sc = float(scores[best_k])
                best = actor_id_list[int(cand_idxs[best_k])]
                w = (
                    cal["rival_weight_base"]
                    + cal["rival_weight_score_scale"] * best_sc
                    + cal["rival_weight_noise"] * rng.rand()
                )
                _upsert_edge(
                    "rivalry", "-", pid, best, w,
                    reason=f"CAL(rival_rate): rival_score={best_sc:.3f}",
                )
                rivals_adj[pid][best] = w
                rivals_adj[best][pid] = w
                added += 1
            print(f"[Calibrate] Added {added} rivalries to reach rival_rate")
            calibration_stats["rivalry_edges_added"] = int(added)

        if target_same > 0 and len(actor_ids) >= 5:
            detect_started = time.perf_counter()
            communities = detect_communities_louvain(pp_edges, seed=seed)
            calibration_stats["same_community_redetect_elapsed_sec"] = round(float(time.perf_counter() - detect_started), 4)
            bf_map, ratio = best_friend_map(communities)

    # ─── Calibrate: director preferred/avoided actors ─────────────────
    # V15 FIX: pre-index actors by genre so each director only scores
    # genre-overlapping actors + DIRECTOR_ACTOR_SUPPLEMENT random ones.
    # Old: O(d x a) = 2,833 x 14,473 = 41M pairs
    # New: O(d x avg_match) ~= 2,833 x ~630 = ~1.8M pairs (23x speedup)
    DIRECTOR_ACTOR_SUPPLEMENT = int(cal["director_actor_supplement"])
    DIRECTOR_SPARSE_CANDIDATE_CAP = 256

    actor_genre_index: dict = defaultdict(list)
    for aid in actor_ids:
        for g in genres_map.get(aid, set()):
            actor_genre_index[g].append(aid)

    def _director_candidate_actors(did: int) -> list:
        """Genre-overlapping actors for director did + random supplement."""
        cands: set = set()
        for g in genres_map.get(did, set()):
            cands.update(actor_genre_index.get(g, []))
        # Random supplement -- prevents complete genre blindness
        n_supp = min(DIRECTOR_ACTOR_SUPPLEMENT, len(actor_ids))
        for aid in rng.choice(actor_ids_arr, size=n_supp, replace=False).tolist():
            cands.add(int(aid))
        cands.discard(did)
        return list(cands)

    # V18-SCALE: Pre-build outgoing edge index instead of scanning all edges per director
    _outgoing_index: dict = defaultdict(lambda: defaultdict(set))
    for e in pp_edges:
        et = e.get("edge_type")
        if et in {"mentorship", "avoid"}:
            src = int(e.get("src_id", 0))
            dst = int(e.get("dst_id", 0))
            if dst in actor_id_set:
                _outgoing_index[et][src].add(dst)

    pref_target_raw = max(0.0, float(effective_targets.get("director_preferred_actors", 0.0)))
    avoid_target_raw = max(0.0, float(effective_targets.get("director_avoided_actors", 0.0)))
    calibration_stats["director_preferred_target_raw"] = round(pref_target_raw, 4)
    calibration_stats["director_avoid_target_raw"] = round(avoid_target_raw, 4)
    calibration_stats["director_target_mode"] = (
        "count"
        if pref_target_raw >= 1.0 or avoid_target_raw >= 1.0
        else "probability"
    )

    actor_contro = {pid: _safe01(latent_map.get(pid, {}).get("controversy_score"), 0.15) for pid in actor_ids}

    pref_added_total = 0
    avoid_added_total = 0
    pref_directors_activated = 0
    avoid_directors_activated = 0

    def _resolve_director_target(raw_target: float, stage_target_scale: float) -> tuple[int, float]:
        scaled_target = max(0.0, float(raw_target) * float(stage_target_scale))
        if raw_target >= 1.0:
            return max(0, int(round(scaled_target))), scaled_target
        probability = min(1.0, scaled_target)
        return (1 if rng.rand() < probability else 0), probability

    if pref_target_raw > 0 or avoid_target_raw > 0:
        for did in director_ids:
            senior = stage_map.get(did, CAREER_STAGES.index("prime"))
            senior_norm = senior / max(1, len(CAREER_STAGES) - 1)
            stage_target_scale = cal["director_stage_target_base"] + cal["director_stage_target_span"] * senior_norm
            pref_target, _pref_target_effective = _resolve_director_target(pref_target_raw, stage_target_scale)
            avoid_target, _avoid_target_effective = _resolve_director_target(avoid_target_raw, stage_target_scale)
            if pref_target > 0:
                pref_directors_activated += 1
            if avoid_target > 0:
                avoid_directors_activated += 1
            if pref_target <= 0 and avoid_target <= 0:
                continue

            prefs = _outgoing_index["mentorship"][did].copy() if pref_target > 0 else set()
            avoids = _outgoing_index["avoid"][did].copy() if avoid_target > 0 else set()
            pref_need = max(0, pref_target - len(prefs))
            avoid_need = max(0, avoid_target - len(avoids))
            if pref_need <= 0 and avoid_need <= 0:
                continue
            cand_actors = _director_candidate_actors(did)
            if (pref_target_raw < 1.0 or avoid_target_raw < 1.0) and len(cand_actors) > DIRECTOR_SPARSE_CANDIDATE_CAP:
                keep_idx = rng.choice(len(cand_actors), size=DIRECTOR_SPARSE_CANDIDATE_CAP, replace=False)
                cand_actors = [cand_actors[int(idx)] for idx in keep_idx]

            pref_weights = cal["director_preferred_score_weights"]
            if pref_need > 0:
                scored = []
                for aid in cand_actors:
                    if aid in prefs or aid in avoids:
                        continue
                    sc = (
                        pref_weights["style"] * style_sim(did, aid)
                        + pref_weights["genre"] * jaccard(genres_map[did], genres_map[aid])
                        + pref_weights["stage"] * stage_sim(did, aid)
                    )
                    scored.append((sc, aid))
                scored.sort(reverse=True)
                for sc, aid in scored[:pref_need]:
                    w = (
                        cal["director_preferred_weight_base"]
                        + cal["director_preferred_weight_score_scale"] * sc
                        + cal["director_preferred_weight_noise"] * rng.rand()
                    )
                    _upsert_edge("mentorship", "+", did, aid, w,
                                 reason=f"CAL(director_pref): score={sc:.3f}")
                    prefs.add(aid)
                    pref_added_total += 1

            avoid_weights = cal["director_avoid_score_weights"]
            if avoid_need > 0:
                d_rep = _safe01(latent_map.get(did, {}).get("public_reputation"), 0.4)
                scored = []
                for aid in cand_actors:
                    if aid in avoids or aid in prefs:
                        continue
                    sc = (
                        avoid_weights["style_distance"] * (1.0 - style_sim(did, aid))
                        + avoid_weights["genre_distance"] * (1.0 - jaccard(genres_map[did], genres_map[aid]))
                        + avoid_weights["controversy"] * (actor_contro[aid] * d_rep)
                    )
                    scored.append((sc, aid))
                scored.sort(reverse=True)
                for sc, aid in scored[:avoid_need]:
                    w = (
                        cal["director_avoid_weight_base"]
                        + cal["director_avoid_weight_score_scale"] * sc
                        + cal["director_avoid_weight_noise"] * rng.rand()
                    )
                    _upsert_edge("avoid", "-", did, aid, w,
                                 reason=f"CAL(director_avoid): score={sc:.3f}")
                    avoids.add(aid)
                    avoid_added_total += 1
        calibration_stats["director_preferred_directors_activated"] = int(pref_directors_activated)
        calibration_stats["director_avoid_directors_activated"] = int(avoid_directors_activated)
        calibration_stats["director_preferred_edges_added"] = int(pref_added_total)
        calibration_stats["director_avoid_edges_added"] = int(avoid_added_total)
    if target_same > 0 and len(actor_ids) >= 5:
        same_comm_iterations = 0
        for _ in range(3):
            same_comm_iterations += 1
            if ratio >= target_same or not bf_map or communities is None:
                break

            comm_members = defaultdict(list)
            for pid in actor_ids:
                c = communities.get(pid)
                if c is not None:
                    comm_members[c].append(pid)

            offenders = [
                pid
                for pid, (nbr, _) in bf_map.items()
                if communities.get(pid) is not None and communities.get(nbr) != communities.get(pid)
            ]
            rng.shuffle(offenders)

            target_fix = int(round((target_same - ratio) * len(bf_map)))
            fixed = 0

            for pid in offenders:
                if fixed >= target_fix:
                    break
                c = communities.get(pid)
                pool = [x for x in comm_members.get(c, []) if x != pid]
                if not pool:
                    continue

                best = None
                best_sc = -1.0
                for cand in pool:
                    if cand in rivals_adj.get(pid, {}):
                        continue
                    sc = friendship_score(pid, cand)
                    if sc > best_sc:
                        best_sc = sc
                        best = cand

                if best is None:
                    continue

                cur_best_w = bf_map[pid][1]
                w = min(
                    cal["upsert_weight_max"],
                    max(cur_best_w + cal["bf_same_community_weight_boost"], cal["bf_same_community_weight_floor"]),
                )
                _upsert_edge(
                    "friendship", "+", pid, best, w,
                    reason=f"CAL(bf_same_comm): boost_to={w:.2f}",
                )
                friends[pid][best] = w
                friends[best][pid] = w
                fixed += 1

            if fixed == 0:
                break

            detect_started = time.perf_counter()
            communities = detect_communities_louvain(pp_edges, seed=seed)
            calibration_stats["same_community_last_redetect_elapsed_sec"] = round(
                float(time.perf_counter() - detect_started),
                4,
            )
            bf_map, ratio = best_friend_map(communities)
        calibration_stats["same_community_iterations"] = int(same_comm_iterations)
        calibration_stats["same_community_ratio_after"] = round(float(ratio), 4)
        status = "satisfied" if ratio >= target_same else "below minimum"
        print(
            f"[Calibrate] bf_same_community_rate={ratio:.3f} "
            f"(minimum target {target_same:.2f}; {status})"
        )

    LAST_CALIBRATION_STATS = calibration_stats

    return pp_edges


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def _resolve_paths(base_dir: str | Path | None = None) -> tuple[Path, Path, Path]:
    root = Path(base_dir).resolve() if base_dir is not None else Path(__file__).parent.resolve()
    return root, root / "entities", root / "graph"


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate Mirage hybrid graph artifacts.")
    parser.add_argument(
        "--base-dir",
        default=str(Path(__file__).parent),
        help="Dataset root containing entities/ and graph/ directories. Defaults to this script's folder.",
    )
    parser.add_argument(
        "--force-legacy",
        action="store_true",
        help="Force the original list-materializing graph builder even on large datasets.",
    )
    parser.add_argument(
        "--force-scalable",
        action="store_true",
        help="Force the scalable direct-to-runtime graph compiler even below the default size thresholds.",
    )
    parser.add_argument(
        "--use-world-policy",
        action="store_true",
        help="Use world_policy.json to bias company and market edge generation.",
    )
    parser.add_argument(
        "--skip-diagnostic-cold-edges",
        action="store_true",
        help="Skip precomputed P-C/C-C diagnostic cold streams; runtime keeps on-demand scoring.",
    )
    return parser.parse_args()


def main(
    base_dir: str | Path | None = None,
    *,
    force_legacy: bool = False,
    force_scalable: bool = False,
    use_world_policy: bool = False,
    skip_diagnostic_cold_edges: bool = False,
):
    global BASE_DIR, ENTITY_DIR, GRAPH_DIR, MODELING_PRIORS_PAYLOAD, EDGE_PRIORS_SECTION, _CROSS_GENRE_K, _CROSS_GENRE_BUMP
    BASE_DIR, ENTITY_DIR, GRAPH_DIR = _resolve_paths(base_dir)
    rng = np.random.RandomState(SEED)
    os.makedirs(GRAPH_DIR, exist_ok=True)
    MODELING_PRIORS_PAYLOAD = load_modeling_priors_artifact(BASE_DIR) or {}
    EDGE_PRIORS_SECTION = prior_section(MODELING_PRIORS_PAYLOAD, "edge_priors")
    audit_artifact_usage("modeling_priors.json", modeling_priors_path(BASE_DIR), sections=["edge_priors"])
    if current_mode() == "research" and not EDGE_PRIORS_SECTION:
        audit_fallback_hit(
            "edge_priors",
            "missing:section",
            detail="modeling_priors missing edge_priors for hybrid edge generation in research mode",
            mode="research",
        )
    _CROSS_GENRE_K = int(round(_edge_prior_float("cross_genre_candidate_k", _CROSS_GENRE_K, lo=16, hi=2000)))
    _CROSS_GENRE_BUMP = _edge_prior_float("cross_genre_threshold_bump", _CROSS_GENRE_BUMP, lo=0.0, hi=0.5)
    world_policy = safe_load_json(world_policy_path(BASE_DIR), default={}) if use_world_policy else {}
    run_started = time.perf_counter()
    timing_rows: list[dict[str, Any]] = []

    def _record_phase(phase: str, started_at: float, **extra: Any) -> None:
        elapsed = time.perf_counter() - started_at
        timing_rows.append(_timing_row(phase, elapsed, **extra))
        print(f"  [timing] {phase}: {elapsed:.2f}s")

    # Load data
    # V15: ALL persons are fully LLM-generated -- no is_extra split needed.
    # The old core/extras distinction (v9-v14) is dead; every person in
    # persons.json has a full bio, latent vars, and graph representation.
    persons = load_json_batch(ENTITY_DIR / "persons.json")
    graph_persons = persons
    # Keep stable IDs; only backfill if missing (shouldn't happen in v15)
    for i, p in enumerate(graph_persons):
        if "person_id" not in p:
            p["person_id"] = i + 1

    companies = load_json_batch(ENTITY_DIR / "companies.json")
    for i, c in enumerate(companies):
        if "company_id" not in c:
            c["company_id"] = i + 1

    print(f"Loaded {len(graph_persons)} persons (all LLM-generated), {len(companies)} companies")

    # Load latent variables
    person_latent_path = ENTITY_DIR / "persons_latent.json"
    company_latent_path = ENTITY_DIR / "companies_latent.json"

    if not person_latent_path.exists():
        print("ERROR: persons_latent.json not found. Run generate_latent_vars_api.py first.")
        return

    person_latent = json.loads(person_latent_path.read_text(encoding="utf-8"))
    p_latent_map = {lv["person_id"]: lv for lv in person_latent}
    print(f"Loaded {len(person_latent)} person latent vars")

    company_latent = []
    if company_latent_path.exists():
        company_latent = json.loads(company_latent_path.read_text(encoding="utf-8"))
        print(f"Loaded {len(company_latent)} company latent vars")
    else:
        print("No company latent vars found -- skipping company edges")
    _record_phase(
        "load_inputs",
        run_started,
        persons=int(len(graph_persons)),
        companies=int(len(companies)),
        person_latents=int(len(person_latent)),
        company_latents=int(len(company_latent)),
    )

    use_scalable = bool(force_scalable) or (
        not force_legacy and (
            len(graph_persons) >= SCALABLE_PERSON_THRESHOLD or len(companies) >= SCALABLE_COMPANY_THRESHOLD
        )
    )

    if use_scalable:
        from scalable_edge_builder import build_scalable_runtime_graph

        if force_scalable:
            print(
                f"Forcing scalable graph compiler for dataset "
                f"({len(graph_persons):,} persons, {len(companies):,} companies)."
            )
        else:
            print(
                f"Detected large dataset ({len(graph_persons):,} persons, {len(companies):,} companies). "
                "Using scalable direct-to-runtime graph compiler."
            )
        scalable_started = time.perf_counter()
        summary = build_scalable_runtime_graph(
            base_dir=BASE_DIR,
            persons=graph_persons,
            person_latent=person_latent,
            companies=companies,
            company_latent=company_latent,
            seed=SEED,
            world_policy=world_policy,
        )
        _record_phase("scalable_runtime_build", scalable_started, **summary)
        timing_path = _write_graph_timing(
            BASE_DIR,
            {
                "builder_mode": "scalable",
                "seed": int(SEED),
                "people": int(len(graph_persons)),
                "companies": int(len(companies)),
                "phases": timing_rows,
                "total_elapsed_sec": round(float(time.perf_counter() - run_started), 4),
                "summary": summary,
            },
        )
        print(f"\n{'='*60}")
        print("  HYBRID GRAPH COMPLETE (SCALABLE)")
        print(f"{'='*60}")
        print(f"  Communities:         {summary['communities']:,}")
        print(f"  History rows:        {summary['history_rows']:,}")
        print(f"  Cold P-C rows:       {summary['cold_cp_count']:,}")
        print(f"  Cold C-C rows:       {summary['cold_cc_count']:,}")
        print(f"  Hot edge types:      {summary['hot_counts']}")
        print(f"  Runtime manifest:    {GRAPH_DIR / 'runtime_manifest.json'}")
        print(f"  Timing report:       {timing_path}")
        return

    # ═══ Generate person<->person edges ═════════════════════════════════
    # Sort persons: legends/veterans first so they bootstrap high degree before
    # rising actors are encountered. With adaptive threshold, legend vs legend
    # connects freely (thresh~=0.21). Rising vs rising (processed last, still
    # deg=0) faces thresh=0.58 and average pairs (p_edge~=0.41) don't connect.
    _STAGE_PRIORITY = {"legend": 0, "veteran": 1, "prime": 2, "retired": 3, "rising": 4}
    graph_persons.sort(key=lambda p: _STAGE_PRIORITY.get(p.get("career_stage", "prime"), 2))
    stage_counts = Counter(p.get("career_stage", "prime") for p in graph_persons)
    print(f"  Stage order: {dict(stage_counts)}")

    print("\nGenerating person<->person edges...")
    phase_started = time.perf_counter()
    pp_edges = generate_person_person_edges(graph_persons, p_latent_map, rng, world_policy=world_policy)
    _record_phase(
        "generate_person_person_edges",
        phase_started,
        edge_count=int(len(pp_edges)),
        details=dict(LAST_PP_BUILD_STATS),
    )
    pp_types = Counter(e["edge_type"] for e in pp_edges)
    print(f"  Person<->Person: {len(pp_edges)} edges | {dict(pp_types)}")

    # ─── Calibration: match coarse relationship targets ─────────────
    print("Calibrating relationships to targets...")
    phase_started = time.perf_counter()
    pp_edges = calibrate_relationship_targets(pp_edges, graph_persons, p_latent_map, rng)
    _record_phase(
        "calibrate_relationship_targets",
        phase_started,
        edge_count=int(len(pp_edges)),
        details=dict(LAST_CALIBRATION_STATS),
    )
    pp_types = Counter(e["edge_type"] for e in pp_edges)
    print(f"  Person<->Person (calibrated): {len(pp_edges)} edges | {dict(pp_types)}")

    # ─── Social graph enrichment: serendipitous + triadic closure ───
    print("Adding serendipitous cross-community edges...")
    phase_started = time.perf_counter()
    pp_edges = _add_serendipitous_edges(pp_edges, graph_persons, p_latent_map, rng)
    _record_phase("add_serendipitous_edges", phase_started, edge_count=int(len(pp_edges)))

    print("Running triadic closure (stage-gated)...")
    phase_started = time.perf_counter()
    pp_edges = _add_triadic_closure(pp_edges, graph_persons, rng)
    _record_phase("add_triadic_closure", phase_started, edge_count=int(len(pp_edges)))

    pp_types = Counter(e["edge_type"] for e in pp_edges)
    print(f"  Person<->Person (enriched): {len(pp_edges)} edges | {dict(pp_types)}")


    # Add temporal dimension to person<->person edges BEFORE merge
    print("Adding temporal dimension...")
    phase_started = time.perf_counter()
    pp_edges = add_temporal_edges(pp_edges, graph_persons)
    _record_phase("add_temporal_dimension", phase_started)

    # ─── Main graph: P-P edges only ─────────────────────────────────────
    # P-C and C-C edges are computed ON-DEMAND at assembly time from latent
    # variables.  Storing them was O(persons × companies) = 380M+ edges at
    # 450K persons → 130+ GB RAM.  Not viable.
    phase_started = time.perf_counter()
    pp_edges, n_dedup = dedupe_edges(pp_edges)
    _record_phase("dedupe_person_person_edges", phase_started, edge_count=int(len(pp_edges)), merges=int(n_dedup))

    # ═══ Community detection ══════════════════════════════════════════
    print("Running community detection...")
    phase_started = time.perf_counter()
    communities = detect_communities_louvain(pp_edges, seed=SEED)
    communities, n_completed = complete_community_assignments(communities, graph_persons)
    n_communities = len(set(communities.values())) if communities else 0
    _record_phase(
        "community_detection",
        phase_started,
        community_count=int(n_communities),
        assigned_people=int(len(communities)),
        filled_isolates=int(n_completed),
    )
    print(f"  Detected {n_communities} communities across {len(communities)} nodes")
    if n_completed > 0:
        print(f"  Filled {n_completed} isolated persons with deterministic community fallback")

    # ═══ Save outputs ═════════════════════════════════════════════════
    import pyarrow as pa
    import pyarrow.ipc as ipc

    # Primary graph: P-P edges only (loaded at runtime)
    phase_started = time.perf_counter()
    _write_edge_arrow(pp_edges, GRAPH_DIR / "edge_graph.arrow", "P-P (runtime)")

    # Communities — Arrow IPC
    comm_rows = [{"person_id": pid, "community": comm}
                 for pid, comm in sorted(communities.items())]
    comm_arrow_path = GRAPH_DIR / "communities.arrow"
    comm_table = pa.Table.from_pylist(
        comm_rows,
        schema=pa.schema([("person_id", pa.int64()), ("community", pa.int64())]),
    )
    with pa.OSFile(str(comm_arrow_path), "wb") as f:
        writer = ipc.new_file(f, comm_table.schema,
                              options=ipc.IpcWriteOptions(compression="lz4"))
        writer.write_table(comm_table)
        writer.close()
    print(f"Saved {len(communities)} community assignments -> {comm_arrow_path}")

    # Backward-compatible CSV output (P-P only — matches Arrow primary)
    keys = ["src_id", "dst_id", "src_name", "dst_name", "edge_type", "sign",
            "weight", "source_kind", "reason", "valid_from", "valid_to"]
    edge_csv_path = GRAPH_DIR / "edge_graph.csv"
    with open(edge_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(pp_edges)

    comm_csv_path = GRAPH_DIR / "communities.csv"
    with open(comm_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["person_id", "community"])
        writer.writeheader()
        for pid, comm in sorted(communities.items()):
            writer.writerow({"person_id": pid, "community": comm})

    # Edge JSON for debugging (P-P only)
    with open(GRAPH_DIR / "edges_hybrid.json", "w", encoding="utf-8") as f:
        json.dump(pp_edges, f, indent=2, ensure_ascii=False)

    _record_phase(
        "save_graph_artifacts",
        phase_started,
        runtime_edges=int(len(pp_edges)),
    )

    # Canonical split runtime artifacts: hot arrays, cold arrays, and Arrow history.
    from graph_runtime import GraphRuntime

    pc_types: Counter[str] = Counter()
    cc_types: Counter[str] = Counter()
    pc_sink: _ArrowEdgeBatchWriter | None = None
    cc_sink: _ArrowEdgeBatchWriter | None = None
    pc_stream: _InstrumentedEdgeStream | None = None
    cc_stream: _InstrumentedEdgeStream | None = None
    row_iter = iter(pp_edges)
    write_cold_diagnostics = bool(company_latent) and not (
        bool(skip_diagnostic_cold_edges) or _env_flag("DATA_SYS_SKIP_DIAGNOSTIC_COLD_EDGES", False)
    )
    if write_cold_diagnostics:
        print("Streaming person<->company edges (diagnostic + runtime cold arrays)...")
        pc_sink = _ArrowEdgeBatchWriter(GRAPH_DIR / "edges_pc_diagnostic.arrow", "P-C (diagnostic)")
        pc_stream = _InstrumentedEdgeStream(
            iter_person_company_edges(
                graph_persons,
                companies,
                person_latent,
                company_latent,
                rng,
                world_policy=world_policy,
            ),
            edge_counter=pc_types,
            sink=pc_sink,
        )
        print("Streaming company<->company edges (diagnostic + runtime cold arrays)...")
        cc_sink = _ArrowEdgeBatchWriter(GRAPH_DIR / "edges_cc_diagnostic.arrow", "C-C (diagnostic)")
        cc_stream = _InstrumentedEdgeStream(
            iter_company_company_edges(companies, company_latent, rng, world_policy=world_policy),
            edge_counter=cc_types,
            sink=cc_sink,
        )
        row_iter = chain(pp_edges, pc_stream, cc_stream)
    elif company_latent:
        print("Skipping precomputed P-C/C-C diagnostic cold edges; runtime uses on-demand company/person scoring.")

    phase_started = time.perf_counter()
    try:
        GraphRuntime.compile_runtime_graph(
            BASE_DIR,
            row_iter=row_iter,
            source_label="generate_edges_hybrid",
        )
    finally:
        if pc_stream is not None:
            pc_stream.close()
        if cc_stream is not None:
            cc_stream.close()
        if pc_sink is not None:
            pc_sink.close()
        if cc_sink is not None:
            cc_sink.close()
    pc_edge_count = int(pc_stream.count) if pc_stream is not None else 0
    cc_edge_count = int(cc_stream.count) if cc_stream is not None else 0
    _record_phase(
        "compile_runtime_graph",
        phase_started,
        total_input_edges=int(len(pp_edges) + pc_edge_count + cc_edge_count),
        diagnostic_pc_edges=pc_edge_count,
        diagnostic_cc_edges=cc_edge_count,
        skipped_diagnostic_cold_edges=bool(company_latent) and not write_cold_diagnostics,
        pc_stream_elapsed_sec=round(float(pc_stream.elapsed_sec), 4) if pc_stream is not None else 0.0,
        cc_stream_elapsed_sec=round(float(cc_stream.elapsed_sec), 4) if cc_stream is not None else 0.0,
    )
    print(f"Saved split graph runtime -> {GRAPH_DIR / 'runtime_manifest.json'}")

    # ═══ Summary ══════════════════════════════════════════════════════
    pp_types = Counter(e["edge_type"] for e in pp_edges)
    print(f"\n{'='*60}")
    print(f"  HYBRID GRAPH COMPLETE")
    print(f"{'='*60}")
    print(f"  Runtime graph (P-P): {len(pp_edges):,} edges")
    print(f"  Diagnostic P-C:      {pc_edge_count:,} edges (not loaded at runtime)")
    if pc_types:
        print(f"  P-C edge types:      {dict(pc_types)}")
    print(f"  Diagnostic C-C:      {cc_edge_count:,} edges (not loaded at runtime)")
    if cc_types:
        print(f"  C-C edge types:      {dict(cc_types)}")
    print(f"  Dedupe merges:       {n_dedup}")
    print(f"  Communities:         {n_communities}")
    print(f"  P-P edge types:      {dict(pp_types)}")
    print(f"  Source:              100% latent_hybrid (deterministic, seeded)")
    print(f"  Cost:                $0.00 (pure procedural)")
    print(f"  NOTE: P-C/C-C affinity computed on-demand at assembly time")
    timing_path = _write_graph_timing(
        BASE_DIR,
        {
            "builder_mode": "legacy",
            "seed": int(SEED),
            "people": int(len(graph_persons)),
            "companies": int(len(companies)),
            "phases": timing_rows,
            "total_elapsed_sec": round(float(time.perf_counter() - run_started), 4),
            "output_summary": {
                "runtime_graph_edges": int(len(pp_edges)),
                "diagnostic_pc_edges": int(pc_edge_count),
                "diagnostic_cc_edges": int(cc_edge_count),
                "dedupe_merges": int(n_dedup),
                "community_count": int(n_communities),
                "pp_edge_types": dict(pp_types),
                "pc_edge_types": dict(pc_types),
                "cc_edge_types": dict(cc_types),
            },
        },
    )
    print(f"  Timing report:       {timing_path}")


if __name__ == "__main__":
    args = _parse_args()
    if args.force_legacy and args.force_scalable:
        raise SystemExit("Choose at most one of --force-legacy and --force-scalable")
    main(
        args.base_dir,
        force_legacy=args.force_legacy,
        force_scalable=args.force_scalable,
        use_world_policy=bool(args.use_world_policy),
        skip_diagnostic_cold_edges=bool(args.skip_diagnostic_cold_edges),
    )
