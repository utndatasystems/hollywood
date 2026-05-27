from __future__ import annotations

import csv
import hashlib
import json
import shutil
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

from bootstrap_artifacts import (
    audit_artifact_usage,
    audit_fallback_hit,
    current_mode,
    load_modeling_priors_artifact,
    prior_section,
)
from contracts import GENRES
from graph_runtime import GraphRuntime
from policy_runtime import modeling_priors_path, resolve_company_strategy
from pipeline_runtime import year_bounds_from_env
from utils import canonical_company_genre_vector, project_genres_to_company_basis

YEAR_LO, YEAR_HI = year_bounds_from_env(1950, 2025)
_ACTIVE_YEAR_LO = int(YEAR_LO)
_ACTIVE_YEAR_HI = int(YEAR_HI)
_ACTIVE_EDGE_END_YEAR = int(YEAR_HI + 10)
STAGE_ORDER = {"rising": 1, "prime": 2, "veteran": 3, "legend": 4, "retired": 1}
DEFAULT_STAGE_PRIORITY = {"legend": 0, "veteran": 1, "prime": 2, "retired": 3, "rising": 4}
TIER_ORDER = {"Global": 0, "Major": 1, "Mid-Budget": 2, "Indie": 3, "Micro": 4}
DEFAULT_COLLAB_STYLE_CODES = {"ensemble": 0, "selective": 1, "auteur": 2}
MODELING_PRIORS_PAYLOAD: dict[str, Any] = {}
EDGE_PRIORS_SECTION: dict[str, Any] = {}
SCALABLE_EDGE_PRIORS_SECTION: dict[str, Any] = {}
_DEFAULT_PERSON_CAP_PARAMS = {
    "rising": {"mean": 2.0, "std": 3.0, "min": 0, "max": 8},
    "prime": {"mean": 10.0, "std": 6.0, "min": 1, "max": 30},
    "veteran": {"mean": 22.0, "std": 10.0, "min": 3, "max": 60},
    "legend": {"mean": 40.0, "std": 18.0, "min": 5, "max": 100},
    "retired": {"mean": 14.0, "std": 8.0, "min": 1, "max": 35},
}
GENRE_TO_IDX = {genre.lower(): idx for idx, genre in enumerate(GENRES)}
EMPTY_I64 = np.array([], dtype=np.int64)
HOT_EDGE_TYPES = (
    "friendship",
    "rivalry",
    "mentorship",
    "avoid",
    "former_collaborator",
    "collaboration",
    "clique",
)
UNDIRECTED_HOT_TYPES = {"friendship", "rivalry", "former_collaborator", "collaboration", "clique"}
NEGATIVE_EDGE_TYPES = {"rivalry", "avoid", "blacklist", "market_rival"}


def _active_year_lo() -> int:
    return int(_ACTIVE_YEAR_LO)


def _active_year_hi() -> int:
    return int(_ACTIVE_YEAR_HI)


def _active_edge_end_year() -> int:
    return int(_ACTIVE_EDGE_END_YEAR)


def _scalable_prior_dict(key: str) -> dict[str, Any]:
    value = SCALABLE_EDGE_PRIORS_SECTION.get(str(key), {})
    return value if isinstance(value, dict) else {}


def _scalable_float(key: str, default: float, *, lo: float | None = None, hi: float | None = None) -> float:
    value = _scalable_prior_dict("profile_overrides").get(key)
    if value is None and current_mode() == "research":
        audit_fallback_hit(
            "scalable_edge_priors",
            f"missing:{key}",
            detail=f"scalable_edge_priors.profile_overrides.{key} is required in research mode",
            mode="research",
        )
    try:
        out = float(default if value is None else value)
    except Exception:
        out = float(default)
    if lo is not None:
        out = max(float(lo), out)
    if hi is not None:
        out = min(float(hi), out)
    return float(out)


def _scalable_map(key: str, default: dict[str, float | int], *, cast=float) -> dict[str, float | int]:
    value = _scalable_prior_dict(key)
    if not value:
        if current_mode() == "research":
            audit_fallback_hit(
                "scalable_edge_priors",
                f"missing:{key}",
                detail=f"scalable_edge_priors.{key} is required in research mode",
                mode="research",
            )
        return dict(default)
    out: dict[str, float | int] = {}
    for item_key, item_value in value.items():
        try:
            out[str(item_key)] = cast(item_value)
        except Exception:
            continue
    return out or dict(default)


def _stage_priority_value(stage: str) -> int:
    mapping = _scalable_map("stage_priority", DEFAULT_STAGE_PRIORITY, cast=int)
    if str(stage) not in mapping and current_mode() == "research":
        audit_fallback_hit(
            "scalable_edge_priors",
            f"missing:stage_priority.{stage}",
            detail=f"scalable_edge_priors.stage_priority missing required stage {stage}",
            mode="research",
        )
    return int(mapping.get(str(stage), DEFAULT_STAGE_PRIORITY.get(str(stage), 2)))


def _collab_style_codes() -> dict[str, int]:
    mapping = {str(k): int(v) for k, v in _scalable_map("collaboration_style_codes", DEFAULT_COLLAB_STYLE_CODES, cast=int).items()}
    if current_mode() == "research":
        missing = [str(style) for style in DEFAULT_COLLAB_STYLE_CODES.keys() if str(style) not in mapping]
        if missing:
            audit_fallback_hit(
                "scalable_edge_priors",
                "missing:collaboration_style_codes",
                detail=f"scalable_edge_priors.collaboration_style_codes missing keys: {', '.join(missing)}",
                mode="research",
            )
    return mapping


def _person_cap_params(stage: str) -> tuple[float, float, int, int]:
    caps = _scalable_prior_dict("person_degree_caps")
    if not caps and current_mode() != "research":
        caps = EDGE_PRIORS_SECTION.get("person_person_degree_caps", {}) if isinstance(EDGE_PRIORS_SECTION, dict) else {}
    row = caps.get(str(stage), {}) if isinstance(caps, dict) else {}
    if not isinstance(row, dict) or not row:
        if current_mode() == "research":
            audit_fallback_hit(
                "scalable_edge_priors",
                "missing:person_degree_caps",
                detail=f"missing scalable degree-cap config for stage {stage}",
                mode="research",
            )
        row = {}
    default = _DEFAULT_PERSON_CAP_PARAMS.get(stage, _DEFAULT_PERSON_CAP_PARAMS["prime"])
    try:
        mu = float(row.get("mean", default["mean"]))
    except Exception:
        mu = float(default["mean"])
    try:
        sigma = float(row.get("std", default["std"]))
    except Exception:
        sigma = float(default["std"])
    try:
        lo = int(round(float(row.get("min", default["min"]))))
    except Exception:
        lo = int(default["min"])
    try:
        hi = int(round(float(row.get("max", default["max"]))))
    except Exception:
        hi = int(default["max"])
    return mu, sigma, lo, hi


def _load_scalable_prior_state(base_dir: str | Path) -> None:
    global MODELING_PRIORS_PAYLOAD, EDGE_PRIORS_SECTION, SCALABLE_EDGE_PRIORS_SECTION
    global _ACTIVE_YEAR_LO, _ACTIVE_YEAR_HI, _ACTIVE_EDGE_END_YEAR

    MODELING_PRIORS_PAYLOAD = load_modeling_priors_artifact(base_dir, mode=current_mode()) or {}
    EDGE_PRIORS_SECTION = prior_section(MODELING_PRIORS_PAYLOAD, "edge_priors")
    SCALABLE_EDGE_PRIORS_SECTION = prior_section(MODELING_PRIORS_PAYLOAD, "scalable_edge_priors")
    if current_mode() == "research" and not SCALABLE_EDGE_PRIORS_SECTION:
        audit_fallback_hit(
            "scalable_edge_priors",
            "missing:section",
            detail="modeling_priors missing scalable_edge_priors for scalable runtime graph construction",
            mode="research",
        )
    audit_artifact_usage(
        "modeling_priors.json",
        modeling_priors_path(base_dir),
        sections=["edge_priors", "scalable_edge_priors"],
    )
    start_year, end_year = year_bounds_from_env(1950, 2025)
    valid_span = _scalable_prior_dict("valid_year_span")
    if current_mode() == "research" and (not isinstance(valid_span, dict) or "start_year" not in valid_span or "end_year" not in valid_span):
        audit_fallback_hit(
            "scalable_edge_priors",
            "missing:valid_year_span",
            detail="scalable_edge_priors.valid_year_span must provide start_year and end_year in research mode",
            mode="research",
        )
    try:
        _ACTIVE_YEAR_LO = int(valid_span.get("start_year", start_year))
    except Exception:
        _ACTIVE_YEAR_LO = int(start_year)
    try:
        _ACTIVE_YEAR_HI = int(valid_span.get("end_year", end_year))
    except Exception:
        _ACTIVE_YEAR_HI = int(end_year)
    offsets = _scalable_prior_dict("year_validity_offsets")
    if current_mode() == "research" and (not isinstance(offsets, dict) or "person_edge_end_offset" not in offsets or "company_edge_end_offset" not in offsets):
        audit_fallback_hit(
            "scalable_edge_priors",
            "missing:year_validity_offsets",
            detail="scalable_edge_priors.year_validity_offsets must provide person_edge_end_offset and company_edge_end_offset in research mode",
            mode="research",
        )
    try:
        person_offset = int(round(float(offsets.get("person_edge_end_offset", 10))))
    except Exception:
        person_offset = 10
    _ACTIVE_EDGE_END_YEAR = int(_ACTIVE_YEAR_HI + max(0, person_offset))


def _parse_roles(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {chunk.strip().lower() for chunk in value.replace(";", ",").split(",") if chunk.strip()}
    if isinstance(value, (list, tuple)):
        return {str(chunk).strip().lower() for chunk in value if str(chunk).strip()}
    return {str(value).strip().lower()} if str(value).strip() else set()


def _parse_genres(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [chunk.strip() for chunk in value.replace("|", ",").replace(";", ",").split(",") if chunk.strip()]
    if isinstance(value, (list, tuple)):
        return [str(chunk).strip() for chunk in value if str(chunk).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _genre_vector(value: Any) -> np.ndarray:
    vec = np.zeros(len(GENRES), dtype=np.float32)
    for genre in _parse_genres(value):
        idx = GENRE_TO_IDX.get(genre.lower())
        if idx is not None:
            vec[idx] = 1.0
    return vec


def _primary_genre(value: Any) -> str:
    genres = _parse_genres(value)
    return genres[0] if genres else "Drama"


def _person_stochastic_cap(pid: int, stage: str) -> int:
    mu, sigma, lo, hi = _person_cap_params(str(stage))
    seed = (int(pid) + 9173) % (2**31)
    rng = np.random.RandomState(seed)
    cap = int(round(rng.normal(mu, sigma)))
    return max(lo, min(hi, cap))


def _safe_year(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(fallback)


def _safe_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(fallback)


def _safe_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(fallback)


def _sample_from_pool(pool: np.ndarray, size: int, rng: np.random.RandomState, exclude_idx: int | None = None) -> np.ndarray:
    if size <= 0 or getattr(pool, "size", 0) == 0:
        return EMPTY_I64
    arr = np.asarray(pool, dtype=np.int64)
    if exclude_idx is not None:
        arr = arr[arr != int(exclude_idx)]
    if arr.size == 0:
        return EMPTY_I64
    if arr.size <= size:
        return np.array(arr, dtype=np.int64, copy=True)
    picks = rng.choice(arr.size, size=size, replace=False)
    return np.array(arr[picks], dtype=np.int64, copy=False)


def _sample_values(values: list[int], size: int, rng: np.random.RandomState, exclude_value: int | None = None) -> np.ndarray:
    if not values or size <= 0:
        return EMPTY_I64
    arr = np.array(values, dtype=np.int64)
    if exclude_value is not None:
        arr = arr[arr != int(exclude_value)]
    if arr.size == 0:
        return EMPTY_I64
    if arr.size <= size:
        return arr
    return np.array(arr[rng.choice(arr.size, size=size, replace=False)], dtype=np.int64, copy=False)


def _sampled_union(
    rng: np.random.RandomState,
    *,
    exclude_idx: int | None = None,
    specs: list[tuple[np.ndarray, int]],
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for pool, size in specs:
        if size <= 0 or getattr(pool, "size", 0) == 0:
            continue
        picked = _sample_from_pool(pool, size=size, rng=rng, exclude_idx=exclude_idx)
        if picked.size:
            parts.append(picked)
    if not parts:
        return EMPTY_I64
    if len(parts) == 1:
        return parts[0]
    return np.unique(np.concatenate(parts).astype(np.int64, copy=False))


def _top_k_desc(scores: np.ndarray, k: int) -> np.ndarray:
    if scores.size == 0 or k <= 0:
        return EMPTY_I64
    if scores.size <= k:
        return np.argsort(-scores)
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


def _stable_bucket_id(value: Any, modulo: int, *, fallback: int = 1) -> int:
    if modulo <= 0:
        return int(fallback)
    text = str(value or "").strip().lower()
    if not text:
        return int(fallback)
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % int(modulo) + 1


def _reset_graph_outputs(base_dir: str | Path) -> None:
    graph_dir = Path(base_dir) / "graph"
    if not graph_dir.exists():
        return
    for folder in ("runtime", "history"):
        shutil.rmtree(graph_dir / folder, ignore_errors=True)
    for filename in (
        "runtime_manifest.json",
        "communities.arrow",
        "communities.csv",
        "scalable_build_status.json",
        "graph_quality_summary.json",
    ):
        path = graph_dir / filename
        if path.exists():
            path.unlink()


def _write_build_status(base_dir: str | Path, phase: str, **extra: Any) -> None:
    graph_dir = Path(base_dir) / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    payload = {"phase": str(phase)}
    payload.update(extra)
    (graph_dir / "scalable_build_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass(slots=True)
class PersonFeatures:
    ids: np.ndarray
    idx_by_pid: np.ndarray
    style: np.ndarray
    risk: np.ndarray
    controversy: np.ndarray
    reputation: np.ndarray
    volatility: np.ndarray
    ambition: np.ndarray
    genre: np.ndarray
    genre_sums: np.ndarray
    primary_genre: np.ndarray
    stage_name: np.ndarray
    stage_ord: np.ndarray
    cap: np.ndarray
    debut: np.ndarray
    retire: np.ndarray
    community: np.ndarray
    agency_id: np.ndarray
    market_bucket: np.ndarray
    collaboration_style: np.ndarray
    is_actor: np.ndarray
    is_director: np.ndarray
    is_writer: np.ndarray
    is_creative: np.ndarray
    budget_pref: np.ndarray
    pid_to_idx: dict[int, int]
    genre_members: dict[str, np.ndarray]
    community_members: dict[int, np.ndarray]
    actor_indices: np.ndarray
    director_indices: np.ndarray
    writer_indices: np.ndarray
    creative_indices: np.ndarray


@dataclass(slots=True)
class CompanyFeatures:
    ids: np.ndarray
    genre: np.ndarray
    tier: np.ndarray
    risk: np.ndarray
    controversy_tolerance: np.ndarray
    primary_genre: np.ndarray
    clique_id: np.ndarray
    market_bucket: np.ndarray
    tier_rank: np.ndarray
    genre_members: dict[str, np.ndarray]
    clique_members: dict[int, np.ndarray]
    market_members: dict[int, np.ndarray]


@dataclass(slots=True)
class HotGraphPools:
    all_indices: np.ndarray
    actor_indices: np.ndarray
    director_indices: np.ndarray
    writer_indices: np.ndarray
    creative_indices: np.ndarray
    competitive_indices: np.ndarray
    stage_members: dict[str, np.ndarray]
    genre_stage_members: dict[tuple[str, str], np.ndarray]
    agency_members: dict[int, np.ndarray]
    agency_genre_members: dict[tuple[int, str], np.ndarray]
    community_genre_members: dict[tuple[int, str], np.ndarray]
    microcluster_members: dict[tuple[int, str, str], np.ndarray]
    market_genre_members: dict[tuple[int, str], np.ndarray]


@dataclass(slots=True)
class GraphBuildProfile:
    friendship_ratio: float
    rivalry_ratio: float
    rivalry_actor_coverage: float
    mentorship_ratio: float
    avoid_ratio: float
    former_ratio: float
    collaboration_ratio: float
    clique_ratio: float
    closure_scale: float
    bridge_anchor_ratio: float
    friendship_threshold: float
    rivalry_threshold: float
    mentorship_threshold: float
    avoid_threshold: float
    former_threshold: float
    collaboration_threshold: float
    clique_threshold: float
    company_genre_sample_ratio: float
    company_clique_sample_ratio: float
    company_random_sample_ratio: float
    brand_fit_ratio_by_stage: dict[str, float]
    employment_ratio_by_stage: dict[str, float]
    brand_fit_threshold: float
    blacklist_threshold: float
    cc_genre_sample_ratio: float
    cc_clique_sample_ratio: float
    cc_market_sample_ratio: float
    cc_random_sample_ratio: float
    cc_rival_pick_ratio: float
    cc_copro_pick_ratio: float
    cc_subsidiary_pick_ratio: float


class EdgeBuffer:
    def __init__(self, edge_type: str):
        self.edge_type = str(edge_type)
        self.src = array("I")
        self.dst = array("I")
        self.weight = array("f")
        self.valid_from = array("i")
        self.valid_to = array("i")

    def add(self, src_id: int, dst_id: int, weight: float, valid_from: int, valid_to: int) -> None:
        self.src.append(int(src_id))
        self.dst.append(int(dst_id))
        self.weight.append(float(max(0.05, min(0.98, weight))))
        self.valid_from.append(int(valid_from))
        self.valid_to.append(int(max(valid_from, valid_to)))

    @property
    def count(self) -> int:
        return len(self.src)

    def iter_batches(self, persons: PersonFeatures, chunk_size: int = 250_000) -> Iterator[dict[str, Any]]:
        count = self.count
        if count <= 0:
            return
        src = np.frombuffer(self.src, dtype=np.uint32)
        dst = np.frombuffer(self.dst, dtype=np.uint32)
        weight = np.frombuffer(self.weight, dtype=np.float32)
        valid_from = np.frombuffer(self.valid_from, dtype=np.int32)
        valid_to = np.frombuffer(self.valid_to, dtype=np.int32)
        idx_lookup = persons.idx_by_pid
        sign = "-" if self.edge_type in {"rivalry", "avoid"} else "+"
        for start in range(0, count, chunk_size):
            stop = min(count, start + chunk_size)
            src_chunk = np.array(src[start:stop], dtype=np.uint32, copy=True)
            yield {
                "edge_type": self.edge_type,
                "sign": sign,
                "src": src_chunk,
                "dst": np.array(dst[start:stop], dtype=np.uint32, copy=True),
                "weight": np.array(weight[start:stop], dtype=np.float32, copy=True),
                "valid_from": np.array(valid_from[start:stop], dtype=np.int32, copy=True),
                "valid_to": np.array(valid_to[start:stop], dtype=np.int32, copy=True),
                "src_type": "person",
                "dst_type": "person",
                "source_kind": "latent_hybrid_v3",
                "source_batch": "generate_edges_hybrid_scalable",
                "reason": "",
                "community_id": persons.community[idx_lookup[src_chunk]].astype(np.int32, copy=False),
            }


class HotGraphState:
    def __init__(self, person_ids: np.ndarray, communities: np.ndarray):
        self.person_ids = np.asarray(person_ids, dtype=np.uint32)
        self.communities = np.asarray(communities, dtype=np.int32)
        self.max_pid = int(self.person_ids.max()) if self.person_ids.size else 0
        self.community_by_pid = np.zeros(self.max_pid + 1, dtype=np.int32)
        if self.person_ids.size:
            self.community_by_pid[self.person_ids.astype(np.int64, copy=False)] = self.communities
        self.seen: dict[str, set[int]] = {edge_type: set() for edge_type in HOT_EDGE_TYPES}
        self.degree = np.zeros(self.max_pid + 1, dtype=np.int32)
        self.family_degree: dict[str, np.ndarray] = {edge_type: np.zeros(self.max_pid + 1, dtype=np.int32) for edge_type in HOT_EDGE_TYPES}
        self.family_out: dict[str, np.ndarray] = {
            edge_type: np.zeros(self.max_pid + 1, dtype=np.int32)
            for edge_type in ("mentorship", "avoid")
        }
        self.adjacency: dict[str, dict[int, list[int]]] = {
            edge_type: defaultdict(list)
            for edge_type in ("friendship", "former_collaborator", "collaboration", "clique")
        }
        self.counts: Counter[str] = Counter()
        self.cross_community_friendships = 0

    def has_edge(self, edge_type: str, src_id: int, dst_id: int) -> bool:
        if edge_type in UNDIRECTED_HOT_TYPES:
            a = min(int(src_id), int(dst_id))
            b = max(int(src_id), int(dst_id))
        else:
            a = int(src_id)
            b = int(dst_id)
        return ((a << 32) | b) in self.seen[edge_type]

    def add(self, buffer: EdgeBuffer, edge_type: str, src_id: int, dst_id: int, weight: float, valid_from: int, valid_to: int) -> bool:
        src = int(src_id)
        dst = int(dst_id)
        if src <= 0 or dst <= 0 or src == dst:
            return False
        if edge_type in UNDIRECTED_HOT_TYPES:
            src, dst = (src, dst) if src <= dst else (dst, src)
        key = (src << 32) | dst
        if key in self.seen[edge_type]:
            return False
        self.seen[edge_type].add(key)
        buffer.add(src, dst, weight, valid_from, valid_to)
        self.counts[edge_type] += 1
        self.degree[src] += 1
        self.degree[dst] += 1
        self.family_degree[edge_type][src] += 1
        self.family_degree[edge_type][dst] += 1
        if edge_type in self.family_out:
            self.family_out[edge_type][src] += 1
        if edge_type in self.adjacency:
            self.adjacency[edge_type][src].append(dst)
            self.adjacency[edge_type][dst].append(src)
        if edge_type == "friendship":
            src_comm = int(self.community_by_pid[src]) if src < len(self.community_by_pid) else 0
            dst_comm = int(self.community_by_pid[dst]) if dst < len(self.community_by_pid) else 0
            if src_comm != dst_comm:
                self.cross_community_friendships += 1
        return True


def _build_person_features(persons: list[dict[str, Any]], latent_map: Mapping[int, Mapping[str, Any]]) -> PersonFeatures:
    ids = np.array([int(p.get("person_id", idx + 1)) for idx, p in enumerate(persons)], dtype=np.uint32)
    pid_to_idx = {int(pid): idx for idx, pid in enumerate(ids.tolist())}
    idx_by_pid = np.full(int(ids.max()) + 1 if ids.size else 1, -1, dtype=np.int32)
    if ids.size:
        idx_by_pid[ids.astype(np.int64, copy=False)] = np.arange(len(ids), dtype=np.int32)

    style = np.zeros((len(persons), 8), dtype=np.float32)
    risk = np.full(len(persons), 0.5, dtype=np.float32)
    controversy = np.zeros(len(persons), dtype=np.float32)
    reputation = np.full(len(persons), 0.5, dtype=np.float32)
    volatility = np.full(len(persons), 0.35, dtype=np.float32)
    ambition = np.full(len(persons), 0.5, dtype=np.float32)
    genre = np.zeros((len(persons), len(GENRES)), dtype=np.float32)
    primary_genre = np.empty(len(persons), dtype=object)
    stage_name = np.empty(len(persons), dtype=object)
    stage_ord = np.zeros(len(persons), dtype=np.int16)
    cap = np.zeros(len(persons), dtype=np.int16)
    debut = np.zeros(len(persons), dtype=np.int32)
    retire = np.zeros(len(persons), dtype=np.int32)
    community = np.zeros(len(persons), dtype=np.int32)
    agency_id = np.zeros(len(persons), dtype=np.int32)
    market_bucket = np.zeros(len(persons), dtype=np.int16)
    collaboration_style = np.zeros(len(persons), dtype=np.int8)
    is_actor = np.zeros(len(persons), dtype=bool)
    is_director = np.zeros(len(persons), dtype=bool)
    is_writer = np.zeros(len(persons), dtype=bool)
    is_creative = np.zeros(len(persons), dtype=bool)
    budget_pref = np.zeros((len(persons), 5), dtype=np.float32)
    genre_members: dict[str, list[int]] = defaultdict(list)
    community_members: dict[int, list[int]] = defaultdict(list)
    agency_mod = max(16, min(1024, max(32, len(persons) // 400 or 32)))

    for idx, person in enumerate(persons):
        pid = int(ids[idx])
        latent = latent_map.get(pid, {})
        style_vec = list(latent.get("creative_style_vector", [0.0] * 8))
        style[idx, : min(8, len(style_vec))] = np.asarray(style_vec[:8], dtype=np.float32)
        risk[idx] = _safe_float(latent.get("risk_tolerance"), 0.5)
        controversy[idx] = _safe_float(latent.get("controversy_score"), 0.0)
        reputation[idx] = _safe_float(latent.get("public_reputation"), 0.5)
        volatility[idx] = _safe_float(latent.get("volatility"), 0.35)
        ambition[idx] = _safe_float(latent.get("artistic_ambition"), 0.5)
        genre[idx] = _genre_vector(person.get("genre_affinity"))
        primary = _primary_genre(person.get("genre_affinity"))
        primary_genre[idx] = primary
        stage = str(person.get("career_stage", "prime") or "prime").lower()
        stage_name[idx] = stage
        stage_ord[idx] = int(STAGE_ORDER.get(stage, 2))
        cap[idx] = int(_person_stochastic_cap(pid, stage))
        debut[idx] = _safe_year(person.get("debut_year"), _active_year_lo() + (pid % 25))
        retire[idx] = max(debut[idx], _safe_year(person.get("retirement_year"), _active_edge_end_year()))
        roles = _parse_roles(person.get("roles"))
        is_actor[idx] = "actor" in roles
        is_director[idx] = "director" in roles
        is_writer[idx] = "writer" in roles or "screenwriter" in roles
        is_creative[idx] = any(role in roles for role in {"actor", "director", "writer", "screenwriter", "producer", "cinematographer", "editor", "composer"}) or bool(roles)
        pref = list(latent.get("budget_band_pref", [0.2] * 5))
        budget_pref[idx, : min(5, len(pref))] = np.asarray(pref[:5], dtype=np.float32)
        market_bucket[idx] = _stable_bucket_id(person.get("nationality") or person.get("country") or "USA", 12)
        raw_agency = person.get("agency_id")
        if raw_agency is not None:
            agency_id[idx] = max(1, _safe_int(raw_agency, 1))
        else:
            agency_name = str(person.get("agency") or "").strip()
            agency_id[idx] = _stable_bucket_id(agency_name or f"{primary}|{pid}", agency_mod, fallback=(pid % agency_mod) + 1)
        collab_style = str(latent.get("collaboration_style", "ensemble") or "ensemble").lower()
        collaboration_style[idx] = int(_collab_style_codes().get(collab_style, 0))
        top_style = int(np.argmax(style[idx])) if style.shape[1] else 0
        genre_idx = GENRE_TO_IDX.get(primary.lower(), 0)
        community_id = int(genre_idx * 64 + top_style * 8 + (int(market_bucket[idx]) % 4) * 2 + (int(agency_id[idx]) % 2) + 1)
        community[idx] = community_id
        genre_members[primary.lower()].append(idx)
        community_members[community_id].append(idx)

    norms = np.linalg.norm(style, axis=1, keepdims=True)
    style = style / np.maximum(norms, 1e-8)
    genre_sums = genre.sum(axis=1)
    return PersonFeatures(
        ids=ids,
        idx_by_pid=idx_by_pid,
        style=style,
        risk=risk,
        controversy=controversy,
        reputation=reputation,
        volatility=volatility,
        ambition=ambition,
        genre=genre,
        genre_sums=genre_sums,
        primary_genre=primary_genre,
        stage_name=stage_name,
        stage_ord=stage_ord,
        cap=cap,
        debut=debut,
        retire=retire,
        community=community,
        agency_id=agency_id,
        market_bucket=market_bucket,
        collaboration_style=collaboration_style,
        is_actor=is_actor,
        is_director=is_director,
        is_writer=is_writer,
        is_creative=is_creative,
        budget_pref=budget_pref,
        pid_to_idx=pid_to_idx,
        genre_members={key: np.array(value, dtype=np.int64) for key, value in genre_members.items()},
        community_members={key: np.array(value, dtype=np.int64) for key, value in community_members.items()},
        actor_indices=np.flatnonzero(is_actor).astype(np.int64),
        director_indices=np.flatnonzero(is_director).astype(np.int64),
        writer_indices=np.flatnonzero(is_writer).astype(np.int64),
        creative_indices=np.flatnonzero(is_creative).astype(np.int64),
    )


def _build_company_features(companies: list[dict[str, Any]], latent_rows: list[dict[str, Any]]) -> CompanyFeatures:
    latent_map = {int(row.get("company_id")): row for row in latent_rows}
    ids = np.array([int(c.get("company_id", idx + 1)) for idx, c in enumerate(companies)], dtype=np.uint32)
    genre = np.zeros((len(companies), 12), dtype=np.float32)
    tier = np.zeros((len(companies), 5), dtype=np.float32)
    risk = np.full(len(companies), 0.5, dtype=np.float32)
    controversy_tolerance = np.full(len(companies), 0.5, dtype=np.float32)
    primary_genre = np.empty(len(companies), dtype=object)
    clique_id = np.zeros(len(companies), dtype=np.int32)
    market_bucket = np.zeros(len(companies), dtype=np.int16)
    tier_rank = np.full(len(companies), 2, dtype=np.int8)
    genre_members: dict[str, list[int]] = defaultdict(list)
    clique_members: dict[int, list[int]] = defaultdict(list)
    market_members: dict[int, list[int]] = defaultdict(list)
    clique_mod = max(8, min(512, max(16, len(companies) // 40 or 16)))

    for idx, company in enumerate(companies):
        cid = int(ids[idx])
        latent = latent_map.get(cid, {})
        genre[idx] = np.asarray(
            canonical_company_genre_vector(latent.get("genre_portfolio", company.get("specialty_genres", []))),
            dtype=np.float32,
        )
        tier_focus = list(latent.get("budget_tier_focus", [0.2] * 5))
        tier[idx, : min(5, len(tier_focus))] = np.asarray(tier_focus[:5], dtype=np.float32)
        risk[idx] = _safe_float(latent.get("risk_appetite"), 0.5)
        controversy_tolerance[idx] = _safe_float(latent.get("controversy_tolerance"), 0.5)
        genres = _parse_genres(company.get("specialty_genres"))
        primary = genres[0] if genres else GENRES[idx % len(GENRES)]
        primary_genre[idx] = primary
        market_bucket[idx] = _stable_bucket_id(company.get("country") or "USA", 12)
        raw_clique = company.get("clique_id")
        if raw_clique is not None:
            clique_id[idx] = max(1, _safe_int(raw_clique, 1))
        else:
            clique_id[idx] = _stable_bucket_id(company.get("clique") or f"{company.get('tier','Mid-Budget')}|{primary}", clique_mod, fallback=(idx % clique_mod) + 1)
        tier_rank[idx] = int(TIER_ORDER.get(str(company.get("tier", "Mid-Budget")), 2))
        genre_members[primary.lower()].append(idx)
        clique_members[int(clique_id[idx])].append(idx)
        market_members[int(market_bucket[idx])].append(idx)

    return CompanyFeatures(
        ids=ids,
        genre=genre,
        tier=tier,
        risk=risk,
        controversy_tolerance=controversy_tolerance,
        primary_genre=primary_genre,
        clique_id=clique_id,
        market_bucket=market_bucket,
        tier_rank=tier_rank,
        genre_members={key: np.array(value, dtype=np.int64) for key, value in genre_members.items()},
        clique_members={key: np.array(value, dtype=np.int64) for key, value in clique_members.items()},
        market_members={key: np.array(value, dtype=np.int64) for key, value in market_members.items()},
    )


def _build_hot_pools(persons: PersonFeatures) -> HotGraphPools:
    stage_members: dict[str, list[int]] = defaultdict(list)
    genre_stage_members: dict[tuple[str, str], list[int]] = defaultdict(list)
    agency_members: dict[int, list[int]] = defaultdict(list)
    agency_genre_members: dict[tuple[int, str], list[int]] = defaultdict(list)
    community_genre_members: dict[tuple[int, str], list[int]] = defaultdict(list)
    microcluster_members: dict[tuple[int, str, str], list[int]] = defaultdict(list)
    market_genre_members: dict[tuple[int, str], list[int]] = defaultdict(list)

    for idx in range(len(persons.ids)):
        stage = str(persons.stage_name[idx])
        genre_key = str(persons.primary_genre[idx]).lower()
        agency = int(persons.agency_id[idx])
        community = int(persons.community[idx])
        market = int(persons.market_bucket[idx])
        stage_members[stage].append(idx)
        genre_stage_members[(genre_key, stage)].append(idx)
        agency_members[agency].append(idx)
        agency_genre_members[(agency, genre_key)].append(idx)
        community_genre_members[(community, genre_key)].append(idx)
        microcluster_members[(community, genre_key, stage)].append(idx)
        market_genre_members[(market, genre_key)].append(idx)

    competitive_mask = persons.is_actor | persons.is_director | persons.is_writer
    return HotGraphPools(
        all_indices=np.arange(len(persons.ids), dtype=np.int64),
        actor_indices=np.asarray(persons.actor_indices, dtype=np.int64),
        director_indices=np.asarray(persons.director_indices, dtype=np.int64),
        writer_indices=np.asarray(persons.writer_indices, dtype=np.int64),
        creative_indices=np.asarray(persons.creative_indices, dtype=np.int64),
        competitive_indices=np.flatnonzero(competitive_mask).astype(np.int64),
        stage_members={key: np.array(value, dtype=np.int64) for key, value in stage_members.items()},
        genre_stage_members={key: np.array(value, dtype=np.int64) for key, value in genre_stage_members.items()},
        agency_members={key: np.array(value, dtype=np.int64) for key, value in agency_members.items()},
        agency_genre_members={key: np.array(value, dtype=np.int64) for key, value in agency_genre_members.items()},
        community_genre_members={key: np.array(value, dtype=np.int64) for key, value in community_genre_members.items()},
        microcluster_members={key: np.array(value, dtype=np.int64) for key, value in microcluster_members.items()},
        market_genre_members={key: np.array(value, dtype=np.int64) for key, value in market_genre_members.items()},
    )


def _valid_overlap(persons: PersonFeatures, idx_a: int, idx_b: int) -> tuple[int, int]:
    valid_from = max(int(persons.debut[idx_a]), int(persons.debut[idx_b]))
    valid_to = min(int(persons.retire[idx_a]), int(persons.retire[idx_b]))
    if valid_to < valid_from:
        valid_to = valid_from
    return valid_from, valid_to


def _pair_metrics(persons: PersonFeatures, idx: int, candidates: np.ndarray) -> dict[str, np.ndarray]:
    if candidates.size == 0:
        zeros = np.zeros(0, dtype=np.float32)
        return {name: zeros for name in ("style", "genre_jaccard", "stage_sim", "risk_match", "same_community", "same_agency", "same_market", "rep_match", "volatility", "controversy_gap", "ambition_match")}
    style = np.clip(persons.style[candidates] @ persons.style[idx], 0.0, 1.0)
    overlap = persons.genre[candidates] @ persons.genre[idx]
    union = persons.genre_sums[candidates] + persons.genre_sums[idx] - overlap
    genre_jaccard = overlap / np.maximum(union, 1.0)
    stage_sim = 1.0 - np.abs(persons.stage_ord[candidates].astype(np.float32) - float(persons.stage_ord[idx])) / 4.0
    risk_match = 1.0 - np.abs(persons.risk[candidates] - persons.risk[idx])
    same_community = (persons.community[candidates] == persons.community[idx]).astype(np.float32)
    same_agency = (persons.agency_id[candidates] == persons.agency_id[idx]).astype(np.float32)
    same_market = (persons.market_bucket[candidates] == persons.market_bucket[idx]).astype(np.float32)
    rep_match = 1.0 - np.abs(persons.reputation[candidates] - persons.reputation[idx])
    volatility = (persons.volatility[candidates] + persons.volatility[idx]) / 2.0
    controversy_gap = np.abs(persons.controversy[candidates] - persons.controversy[idx])
    ambition_match = 1.0 - np.abs(persons.ambition[candidates] - persons.ambition[idx])
    return {
        "style": style.astype(np.float32, copy=False),
        "genre_jaccard": genre_jaccard.astype(np.float32, copy=False),
        "stage_sim": stage_sim.astype(np.float32, copy=False),
        "risk_match": risk_match.astype(np.float32, copy=False),
        "same_community": same_community,
        "same_agency": same_agency,
        "same_market": same_market,
        "rep_match": rep_match.astype(np.float32, copy=False),
        "volatility": volatility.astype(np.float32, copy=False),
        "controversy_gap": controversy_gap.astype(np.float32, copy=False),
        "ambition_match": ambition_match.astype(np.float32, copy=False),
    }


def _pid_values_to_indices(persons: PersonFeatures, values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return EMPTY_I64
    idx = persons.idx_by_pid[np.asarray(values, dtype=np.int64)]
    idx = idx[idx >= 0]
    if idx.size == 0:
        return EMPTY_I64
    return idx.astype(np.int64, copy=False)


def _status_progress(
    phase: str,
    position: int,
    total: int,
    progress_hook: Callable[[str, int, int], None] | None,
    *,
    stride: int = 25_000,
) -> None:
    if progress_hook is None:
        return
    if position == 1 or position % stride == 0 or position == total:
        progress_hook(phase, position, total)


def _scaled_sample_size(pool_size: int, ratio: float, *, minimum: int = 1, maximum: int | None = None) -> int:
    if pool_size <= 0 or ratio <= 0.0:
        return 0
    approx = int(round(np.sqrt(float(pool_size)) * float(ratio) + np.log1p(float(pool_size)) * 0.75))
    size = max(int(minimum), approx)
    size = min(size, int(pool_size))
    if maximum is not None:
        size = min(size, int(maximum))
    return max(0, int(size))


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def _build_graph_profile(persons: PersonFeatures, companies: CompanyFeatures, pools: HotGraphPools) -> GraphBuildProfile:
    people_n = max(1, len(persons.ids))
    actor_n = max(1, len(persons.actor_indices))
    director_n = max(1, len(persons.director_indices))
    creative_n = max(1, len(persons.creative_indices))
    company_n = max(1, len(companies.ids))
    actor_share = actor_n / people_n
    director_share = director_n / people_n
    creative_share = creative_n / people_n
    community_count = max(1, len(persons.community_members))
    microcluster_sizes = np.array([len(v) for v in pools.microcluster_members.values()], dtype=np.float32) if pools.microcluster_members else np.array([1.0], dtype=np.float32)
    market_cluster_sizes = np.array([len(v) for v in pools.market_genre_members.values()], dtype=np.float32) if pools.market_genre_members else np.array([1.0], dtype=np.float32)
    clique_sizes = np.array([len(v) for v in companies.clique_members.values()], dtype=np.float32) if companies.clique_members else np.array([1.0], dtype=np.float32)
    controversy_tail = float(np.mean(persons.controversy >= np.quantile(persons.controversy, 0.85))) if len(persons.controversy) else 0.0
    ensemble_code = _collab_style_codes().get("ensemble", 0)
    ensemble_share = float(np.mean(persons.collaboration_style == ensemble_code)) if len(persons.collaboration_style) else 0.0
    average_cap = float(np.mean(persons.cap[persons.cap > 0])) if np.any(persons.cap > 0) else 1.0
    community_density = float(np.median(microcluster_sizes) / max(1.0, np.sqrt(float(people_n))))
    competition_density = float(np.median(market_cluster_sizes) / max(1.0, np.sqrt(float(actor_n))))
    clique_density = float(np.median(clique_sizes) / max(1.0, np.sqrt(float(company_n))))
    companies_per_person = company_n / people_n

    friendship_ratio = _clamp(
        _scalable_float("friendship_ratio_base", 0.16)
        + _scalable_float("friendship_ratio_creative_share_weight", 0.10) * creative_share
        + _scalable_float("friendship_ratio_community_density_weight", 0.18) * community_density,
        _scalable_float("friendship_ratio_min", 0.18),
        _scalable_float("friendship_ratio_max", 0.34),
    )
    rivalry_ratio = _clamp(
        _scalable_float("rivalry_ratio_base", 0.045)
        + _scalable_float("rivalry_ratio_competition_density_weight", 0.12) * competition_density
        + _scalable_float("rivalry_ratio_actor_share_weight", 0.05) * actor_share,
        _scalable_float("rivalry_ratio_min", 0.05),
        _scalable_float("rivalry_ratio_max", 0.16),
    )
    rivalry_actor_coverage = _clamp(
        _scalable_float("rivalry_actor_coverage_base", 0.38)
        + _scalable_float("rivalry_actor_coverage_competition_weight", 0.45) * competition_density
        + _scalable_float("rivalry_actor_coverage_controversy_weight", 0.08) * controversy_tail,
        _scalable_float("rivalry_actor_coverage_min", 0.45),
        _scalable_float("rivalry_actor_coverage_max", 0.85),
    )
    mentorship_ratio = _clamp(
        _scalable_float("mentorship_ratio_base", 0.05)
        + _scalable_float("mentorship_ratio_director_share_weight", 0.18) * (director_n / creative_n)
        + _scalable_float("mentorship_ratio_cap_weight", 0.05) * (average_cap / 20.0),
        _scalable_float("mentorship_ratio_min", 0.06),
        _scalable_float("mentorship_ratio_max", 0.16),
    )
    avoid_ratio = _clamp(
        _scalable_float("avoid_ratio_base", 0.01)
        + _scalable_float("avoid_ratio_controversy_weight", 0.10) * controversy_tail
        + _scalable_float("avoid_ratio_competition_weight", 0.03) * competition_density,
        _scalable_float("avoid_ratio_min", 0.012),
        _scalable_float("avoid_ratio_max", 0.07),
    )
    former_ratio = _clamp(
        _scalable_float("former_ratio_base", 0.04)
        + _scalable_float("former_ratio_community_density_weight", 0.12) * community_density
        + _scalable_float("former_ratio_creative_share_weight", 0.05) * creative_share,
        _scalable_float("former_ratio_min", 0.05),
        _scalable_float("former_ratio_max", 0.16),
    )
    collaboration_ratio = _clamp(
        _scalable_float("collaboration_ratio_base", 0.035)
        + _scalable_float("collaboration_ratio_ensemble_weight", 0.08) * ensemble_share
        + _scalable_float("collaboration_ratio_community_density_weight", 0.06) * community_density,
        _scalable_float("collaboration_ratio_min", 0.04),
        _scalable_float("collaboration_ratio_max", 0.14),
    )
    clique_ratio = _clamp(
        _scalable_float("clique_ratio_base", 0.022)
        + _scalable_float("clique_ratio_community_density_weight", 0.12) * community_density,
        _scalable_float("clique_ratio_min", 0.025),
        _scalable_float("clique_ratio_max", 0.09),
    )
    closure_scale = _clamp(
        _scalable_float("closure_scale_base", 0.70)
        + _scalable_float("closure_scale_community_density_weight", 0.80) * community_density,
        _scalable_float("closure_scale_min", 0.70),
        _scalable_float("closure_scale_max", 1.50),
    )
    bridge_anchor_ratio = _clamp(
        _scalable_float("bridge_anchor_ratio_base", 0.015)
        + _scalable_float("bridge_anchor_ratio_community_count_weight", 0.030) * (community_count / max(1.0, np.sqrt(float(people_n)))),
        _scalable_float("bridge_anchor_ratio_min", 0.015),
        _scalable_float("bridge_anchor_ratio_max", 0.060),
    )

    brand_base = _clamp(
        _scalable_float("brand_fit_ratio_base", 0.18)
        + _scalable_float("brand_fit_ratio_companies_per_person_weight", 0.65) * companies_per_person,
        _scalable_float("brand_fit_ratio_base_min", 0.20),
        _scalable_float("brand_fit_ratio_base_max", 0.90),
    )
    employment_base = _clamp(
        _scalable_float("employment_ratio_base", 0.009)
        + _scalable_float("employment_ratio_companies_per_person_weight", 0.060) * companies_per_person,
        _scalable_float("employment_ratio_base_min", 0.012),
        _scalable_float("employment_ratio_base_max", 0.060),
    )
    default_brand_fit_ratio_by_stage = {
        "legend": _clamp(brand_base * 1.35, 0.18, 0.95),
        "veteran": _clamp(brand_base * 1.10, 0.16, 0.85),
        "prime": _clamp(brand_base * 0.82, 0.12, 0.72),
        "rising": _clamp(brand_base * 0.40, 0.05, 0.32),
        "retired": _clamp(brand_base * 0.60, 0.08, 0.48),
    }
    default_employment_ratio_by_stage = {
        "legend": _clamp(employment_base * 1.30, 0.010, 0.080),
        "veteran": _clamp(employment_base * 1.10, 0.010, 0.070),
        "prime": _clamp(employment_base * 0.95, 0.008, 0.060),
        "rising": _clamp(employment_base * 0.70, 0.006, 0.045),
        "retired": _clamp(employment_base * 0.60, 0.005, 0.040),
    }
    brand_fit_ratio_by_stage = {
        str(stage): float(value)
        for stage, value in _scalable_map("brand_fit_ratio_by_stage", default_brand_fit_ratio_by_stage, cast=float).items()
    }
    employment_ratio_by_stage = {
        str(stage): float(value)
        for stage, value in _scalable_map("employment_ratio_by_stage", default_employment_ratio_by_stage, cast=float).items()
    }

    return GraphBuildProfile(
        friendship_ratio=friendship_ratio,
        rivalry_ratio=rivalry_ratio,
        rivalry_actor_coverage=rivalry_actor_coverage,
        mentorship_ratio=mentorship_ratio,
        avoid_ratio=avoid_ratio,
        former_ratio=former_ratio,
        collaboration_ratio=collaboration_ratio,
        clique_ratio=clique_ratio,
        closure_scale=closure_scale,
        bridge_anchor_ratio=bridge_anchor_ratio,
        friendship_threshold=_clamp(
            _scalable_float("friendship_threshold_base", 0.34) + _scalable_float("friendship_threshold_inverse_community_weight", 0.06) * (1.0 - community_density),
            _scalable_float("friendship_threshold_min", 0.30),
            _scalable_float("friendship_threshold_max", 0.46),
        ),
        rivalry_threshold=_clamp(
            _scalable_float("rivalry_threshold_base", 0.30) + _scalable_float("rivalry_threshold_inverse_competition_weight", 0.08) * (1.0 - competition_density),
            _scalable_float("rivalry_threshold_min", 0.28),
            _scalable_float("rivalry_threshold_max", 0.44),
        ),
        mentorship_threshold=_clamp(
            _scalable_float("mentorship_threshold_base", 0.34) + _scalable_float("mentorship_threshold_inverse_community_weight", 0.04) * (1.0 - community_density),
            _scalable_float("mentorship_threshold_min", 0.30),
            _scalable_float("mentorship_threshold_max", 0.42),
        ),
        avoid_threshold=_clamp(
            _scalable_float("avoid_threshold_base", 0.40) - _scalable_float("avoid_threshold_controversy_weight", 0.10) * controversy_tail,
            _scalable_float("avoid_threshold_min", 0.30),
            _scalable_float("avoid_threshold_max", 0.42),
        ),
        former_threshold=_clamp(
            _scalable_float("former_threshold_base", 0.30) + _scalable_float("former_threshold_inverse_community_weight", 0.08) * (1.0 - community_density),
            _scalable_float("former_threshold_min", 0.26),
            _scalable_float("former_threshold_max", 0.40),
        ),
        collaboration_threshold=_clamp(
            _scalable_float("collaboration_threshold_base", 0.25) + _scalable_float("collaboration_threshold_inverse_ensemble_weight", 0.08) * (1.0 - ensemble_share),
            _scalable_float("collaboration_threshold_min", 0.22),
            _scalable_float("collaboration_threshold_max", 0.38),
        ),
        clique_threshold=_clamp(
            _scalable_float("clique_threshold_base", 0.34) + _scalable_float("clique_threshold_inverse_community_weight", 0.06) * (1.0 - community_density),
            _scalable_float("clique_threshold_min", 0.30),
            _scalable_float("clique_threshold_max", 0.44),
        ),
        company_genre_sample_ratio=_clamp(
            _scalable_float("company_genre_sample_ratio_base", 1.9) + _scalable_float("company_genre_sample_ratio_companies_per_person_weight", 2.0) * companies_per_person,
            _scalable_float("company_genre_sample_ratio_min", 1.8),
            _scalable_float("company_genre_sample_ratio_max", 6.0),
        ),
        company_clique_sample_ratio=_clamp(
            _scalable_float("company_clique_sample_ratio_base", 0.8) + _scalable_float("company_clique_sample_ratio_clique_density_weight", 1.2) * clique_density,
            _scalable_float("company_clique_sample_ratio_min", 0.8),
            _scalable_float("company_clique_sample_ratio_max", 3.0),
        ),
        company_random_sample_ratio=_clamp(
            _scalable_float("company_random_sample_ratio_base", 0.7) + _scalable_float("company_random_sample_ratio_companies_per_person_weight", 1.0) * companies_per_person,
            _scalable_float("company_random_sample_ratio_min", 0.7),
            _scalable_float("company_random_sample_ratio_max", 2.5),
        ),
        brand_fit_ratio_by_stage=brand_fit_ratio_by_stage,
        employment_ratio_by_stage=employment_ratio_by_stage,
        brand_fit_threshold=_clamp(
            _scalable_float("brand_fit_threshold_base", 0.26) + _scalable_float("brand_fit_threshold_inverse_company_density_weight", 0.08) * (1.0 - companies_per_person * 10.0),
            _scalable_float("brand_fit_threshold_min", 0.20),
            _scalable_float("brand_fit_threshold_max", 0.34),
        ),
        blacklist_threshold=_clamp(
            _scalable_float("blacklist_threshold_base", 0.42) - _scalable_float("blacklist_threshold_controversy_weight", 0.08) * controversy_tail,
            _scalable_float("blacklist_threshold_min", 0.30),
            _scalable_float("blacklist_threshold_max", 0.42),
        ),
        cc_genre_sample_ratio=_clamp(
            _scalable_float("cc_genre_sample_ratio_base", 2.4) + _scalable_float("cc_genre_sample_ratio_clique_density_weight", 2.6) * clique_density,
            _scalable_float("cc_genre_sample_ratio_min", 2.0),
            _scalable_float("cc_genre_sample_ratio_max", 6.0),
        ),
        cc_clique_sample_ratio=_clamp(
            _scalable_float("cc_clique_sample_ratio_base", 1.4) + _scalable_float("cc_clique_sample_ratio_clique_density_weight", 1.4) * clique_density,
            _scalable_float("cc_clique_sample_ratio_min", 1.4),
            _scalable_float("cc_clique_sample_ratio_max", 4.0),
        ),
        cc_market_sample_ratio=_clamp(
            _scalable_float("cc_market_sample_ratio_base", 1.4) + _scalable_float("cc_market_sample_ratio_clique_density_weight", 1.8) * clique_density,
            _scalable_float("cc_market_sample_ratio_min", 1.4),
            _scalable_float("cc_market_sample_ratio_max", 4.5),
        ),
        cc_random_sample_ratio=_clamp(
            _scalable_float("cc_random_sample_ratio_base", 0.8) + _scalable_float("cc_random_sample_ratio_clique_density_weight", 0.4) * clique_density,
            _scalable_float("cc_random_sample_ratio_min", 0.8),
            _scalable_float("cc_random_sample_ratio_max", 2.0),
        ),
        cc_rival_pick_ratio=_clamp(
            _scalable_float("cc_rival_pick_ratio_base", 0.020) + _scalable_float("cc_rival_pick_ratio_clique_density_weight", 0.045) * clique_density,
            _scalable_float("cc_rival_pick_ratio_min", 0.018),
            _scalable_float("cc_rival_pick_ratio_max", 0.060),
        ),
        cc_copro_pick_ratio=_clamp(
            _scalable_float("cc_copro_pick_ratio_base", 0.014) + _scalable_float("cc_copro_pick_ratio_clique_density_weight", 0.040) * clique_density,
            _scalable_float("cc_copro_pick_ratio_min", 0.014),
            _scalable_float("cc_copro_pick_ratio_max", 0.050),
        ),
        cc_subsidiary_pick_ratio=_clamp(
            _scalable_float("cc_subsidiary_pick_ratio_base", 0.06) + _scalable_float("cc_subsidiary_pick_ratio_clique_density_weight", 0.10) * clique_density,
            _scalable_float("cc_subsidiary_pick_ratio_min", 0.05),
            _scalable_float("cc_subsidiary_pick_ratio_max", 0.20),
        ),
    )


def _build_friendship_edges(
    persons: PersonFeatures,
    pools: HotGraphPools,
    profile: GraphBuildProfile,
    state: HotGraphState,
    rng: np.random.RandomState,
    progress_hook: Callable[[str, int, int], None] | None = None,
) -> EdgeBuffer:
    buffer = EdgeBuffer("friendship")
    order = np.argsort(
        np.array(
            [_stage_priority_value(str(stage)) * 1000 - int(cap) for stage, cap in zip(persons.stage_name, persons.cap)],
            dtype=np.int32,
        ),
        kind="mergesort",
    )
    total = len(order)
    for pos, idx in enumerate(order.tolist(), start=1):
        _status_progress("friendship_building", pos, total, progress_hook)
        pid = int(persons.ids[idx])
        target = int(
            round(
                float(persons.cap[idx])
                * (
                    profile.friendship_ratio
                    + 0.04 * float(persons.is_actor[idx])
                    + 0.05 * float(persons.is_director[idx])
                    + 0.03 * float(persons.stage_ord[idx])
                    + 0.04 * float(persons.reputation[idx])
                )
            )
        )
        target = max(0, min(int(persons.cap[idx]), int(target)))
        if target <= 0 or int(state.family_degree["friendship"][pid]) >= target:
            continue
        genre_key = str(persons.primary_genre[idx]).lower()
        specs = [
            (
                pools.microcluster_members.get((int(persons.community[idx]), genre_key, str(persons.stage_name[idx])), EMPTY_I64),
                _scaled_sample_size(len(pools.microcluster_members.get((int(persons.community[idx]), genre_key, str(persons.stage_name[idx])), EMPTY_I64)), 1.8, minimum=4, maximum=18),
            ),
            (
                pools.community_genre_members.get((int(persons.community[idx]), genre_key), EMPTY_I64),
                _scaled_sample_size(len(pools.community_genre_members.get((int(persons.community[idx]), genre_key), EMPTY_I64)), 2.0, minimum=6, maximum=24),
            ),
            (
                pools.agency_genre_members.get((int(persons.agency_id[idx]), genre_key), EMPTY_I64),
                _scaled_sample_size(len(pools.agency_genre_members.get((int(persons.agency_id[idx]), genre_key), EMPTY_I64)), 1.7, minimum=4, maximum=20),
            ),
            (
                pools.genre_stage_members.get((genre_key, str(persons.stage_name[idx])), EMPTY_I64),
                _scaled_sample_size(len(pools.genre_stage_members.get((genre_key, str(persons.stage_name[idx])), EMPTY_I64)), 1.9, minimum=5, maximum=22),
            ),
            (
                pools.market_genre_members.get((int(persons.market_bucket[idx]), genre_key), EMPTY_I64),
                _scaled_sample_size(len(pools.market_genre_members.get((int(persons.market_bucket[idx]), genre_key), EMPTY_I64)), 1.5, minimum=4, maximum=16),
            ),
        ]
        if int(state.family_degree["friendship"][pid]) == 0:
            specs.append((pools.all_indices, _scaled_sample_size(len(pools.all_indices), 0.3, minimum=4, maximum=10)))
        candidates = _sampled_union(rng, exclude_idx=idx, specs=specs)
        if candidates.size == 0:
            continue
        candidate_pids = persons.ids[candidates].astype(np.int64, copy=False)
        mask = state.degree[candidate_pids] < persons.cap[candidates]
        if not np.any(mask):
            continue
        candidates = candidates[mask]
        metrics = _pair_metrics(persons, idx, candidates)
        score = (
            0.38 * metrics["style"]
            + 0.22 * metrics["genre_jaccard"]
            + 0.12 * metrics["stage_sim"]
            + 0.08 * metrics["risk_match"]
            + 0.08 * metrics["same_community"]
            + 0.06 * metrics["same_agency"]
            + 0.03 * metrics["same_market"]
            + 0.03 * metrics["rep_match"]
            + 0.05 * rng.rand(len(candidates)).astype(np.float32)
        )
        score = score - (1.0 - metrics["same_community"]) * 0.08 + metrics["same_agency"] * 0.03
        needed = target - int(state.family_degree["friendship"][pid])
        for pick in _top_k_desc(score, min(len(score), max(needed * 6, 8))).tolist():
            if int(state.family_degree["friendship"][pid]) >= target:
                break
            cand = int(candidates[pick])
            other_pid = int(persons.ids[cand])
            if state.has_edge("friendship", pid, other_pid) or state.has_edge("rivalry", pid, other_pid):
                continue
            if int(state.degree[other_pid]) >= int(persons.cap[cand]) or float(score[pick]) < profile.friendship_threshold:
                continue
            valid_from, valid_to = _valid_overlap(persons, idx, cand)
            state.add(buffer, "friendship", pid, other_pid, 0.28 + 0.58 * float(score[pick]), valid_from, valid_to)

    for pid, neighbors in list(state.adjacency["friendship"].items()):
        if len(neighbors) < 2:
            continue
        idx = int(persons.idx_by_pid[int(pid)])
        if idx < 0:
            continue
        chosen = _sample_values(neighbors, size=min(_scaled_sample_size(len(neighbors), 0.9, minimum=2, maximum=6), len(neighbors)), rng=rng)
        prob = _clamp((0.04 + 0.03 * float(persons.stage_ord[idx])) * profile.closure_scale, 0.02, 0.30)
        for left_pos in range(len(chosen)):
            left_pid = int(chosen[left_pos])
            left_idx = int(persons.idx_by_pid[left_pid])
            for right_pid in chosen[left_pos + 1 :]:
                right_pid = int(right_pid)
                if rng.rand() >= prob or state.has_edge("friendship", left_pid, right_pid) or state.has_edge("rivalry", left_pid, right_pid):
                    continue
                right_idx = int(persons.idx_by_pid[right_pid])
                if left_idx < 0 or right_idx < 0:
                    continue
                if int(state.degree[left_pid]) >= int(persons.cap[left_idx]) or int(state.degree[right_pid]) >= int(persons.cap[right_idx]):
                    continue
                valid_from, valid_to = _valid_overlap(persons, left_idx, right_idx)
                state.add(buffer, "friendship", left_pid, right_pid, 0.44 + 0.18 * rng.rand(), valid_from, valid_to)

    bridge_indices = persons.actor_indices if persons.actor_indices.size else pools.creative_indices
    bridge_sample = _sample_from_pool(
        bridge_indices,
        size=_scaled_sample_size(len(bridge_indices), profile.bridge_anchor_ratio * 10.0, minimum=8, maximum=max(16, len(bridge_indices))),
        rng=rng,
    )
    for idx in bridge_sample.tolist():
        pid = int(persons.ids[idx])
        genre_key = str(persons.primary_genre[idx]).lower()
        specs = [
            (
                pools.market_genre_members.get((int(persons.market_bucket[idx]), genre_key), EMPTY_I64),
                _scaled_sample_size(len(pools.market_genre_members.get((int(persons.market_bucket[idx]), genre_key), EMPTY_I64)), 1.8, minimum=6, maximum=18),
            ),
            (
                pools.agency_members.get(int(persons.agency_id[idx]), EMPTY_I64),
                _scaled_sample_size(len(pools.agency_members.get(int(persons.agency_id[idx]), EMPTY_I64)), 1.2, minimum=3, maximum=10),
            ),
            (pools.all_indices, _scaled_sample_size(len(pools.all_indices), 0.25, minimum=3, maximum=8)),
        ]
        candidates = _sampled_union(rng, exclude_idx=idx, specs=specs)
        if candidates.size == 0:
            continue
        candidates = candidates[persons.community[candidates] != persons.community[idx]]
        if candidates.size == 0:
            continue
        metrics = _pair_metrics(persons, idx, candidates)
        score = 0.45 * metrics["style"] + 0.22 * metrics["genre_jaccard"] + 0.15 * metrics["same_market"] + 0.10 * metrics["rep_match"] + 0.08 * rng.rand(len(candidates)).astype(np.float32)
        for pick in _top_k_desc(score, min(len(score), 6)).tolist():
            cand = int(candidates[pick])
            other_pid = int(persons.ids[cand])
            if state.has_edge("friendship", pid, other_pid) or state.has_edge("rivalry", pid, other_pid):
                continue
            if float(score[pick]) < max(profile.friendship_threshold + 0.04, 0.42):
                continue
            valid_from, valid_to = _valid_overlap(persons, idx, cand)
            if state.add(buffer, "friendship", pid, other_pid, 0.34 + 0.35 * float(score[pick]), valid_from, valid_to):
                break
    return buffer


def _build_rivalry_edges(
    persons: PersonFeatures,
    pools: HotGraphPools,
    profile: GraphBuildProfile,
    state: HotGraphState,
    rng: np.random.RandomState,
    progress_hook: Callable[[str, int, int], None] | None = None,
) -> EdgeBuffer:
    buffer = EdgeBuffer("rivalry")
    competitive = np.asarray(pools.competitive_indices, dtype=np.int64)
    actor_reputation_median = float(np.median(persons.reputation[persons.actor_indices])) if len(persons.actor_indices) else 0.5
    total = len(competitive)
    for pos, idx in enumerate(competitive.tolist(), start=1):
        _status_progress("rivalry_building", pos, total, progress_hook, stride=10_000)
        pid = int(persons.ids[idx])
        target = int(
            round(
                float(persons.cap[idx])
                * (
                    profile.rivalry_ratio
                    + 0.04 * float(persons.is_actor[idx])
                    + 0.03 * float(persons.is_director[idx])
                    + 0.04 * float(persons.volatility[idx])
                    + 0.03 * float(persons.reputation[idx])
                )
            )
        )
        if persons.is_actor[idx]:
            target = max(target, 1 if persons.reputation[idx] > actor_reputation_median else 0)
        if int(state.family_degree["rivalry"][pid]) >= target:
            continue
        genre_key = str(persons.primary_genre[idx]).lower()
        stage = str(persons.stage_name[idx])
        stage_pool = _sample_from_pool(pools.genre_stage_members.get((genre_key, stage), EMPTY_I64), size=_scaled_sample_size(len(pools.genre_stage_members.get((genre_key, stage), EMPTY_I64)), 1.8, minimum=6, maximum=28), rng=rng, exclude_idx=idx)
        market_pool = _sample_from_pool(pools.market_genre_members.get((int(persons.market_bucket[idx]), genre_key), EMPTY_I64), size=_scaled_sample_size(len(pools.market_genre_members.get((int(persons.market_bucket[idx]), genre_key), EMPTY_I64)), 2.0, minimum=8, maximum=32), rng=rng, exclude_idx=idx)
        community_pool = _sample_from_pool(pools.community_genre_members.get((int(persons.community[idx]), genre_key), EMPTY_I64), size=_scaled_sample_size(len(pools.community_genre_members.get((int(persons.community[idx]), genre_key), EMPTY_I64)), 1.5, minimum=5, maximum=20), rng=rng, exclude_idx=idx)
        candidates = _sampled_union(rng, exclude_idx=idx, specs=[(stage_pool, len(stage_pool)), (market_pool, len(market_pool)), (community_pool, len(community_pool)), (competitive, 8)])
        if candidates.size == 0:
            continue
        metrics = _pair_metrics(persons, idx, candidates)
        same_agency_penalty = (persons.agency_id[candidates] == persons.agency_id[idx]).astype(np.float32) * 0.10
        score = (
            0.34 * metrics["genre_jaccard"]
            + 0.20 * (1.0 - metrics["style"])
            + 0.16 * metrics["stage_sim"]
            + 0.12 * metrics["same_market"]
            + 0.07 * metrics["volatility"]
            + 0.06 * metrics["controversy_gap"]
            + 0.05 * metrics["rep_match"]
            + 0.04 * rng.rand(len(candidates)).astype(np.float32)
            - same_agency_penalty
        )
        needed = target - int(state.family_degree["rivalry"][pid])
        for pick in _top_k_desc(score, min(len(score), max(needed * 6, 8))).tolist():
            cand = int(candidates[pick])
            other_pid = int(persons.ids[cand])
            if state.has_edge("rivalry", pid, other_pid) or state.has_edge("friendship", pid, other_pid) or state.has_edge("clique", pid, other_pid):
                continue
            if float(score[pick]) < profile.rivalry_threshold:
                continue
            valid_from, valid_to = _valid_overlap(persons, idx, cand)
            if state.add(buffer, "rivalry", pid, other_pid, 0.42 + 0.38 * float(score[pick]), valid_from, valid_to):
                if int(state.family_degree["rivalry"][pid]) >= target:
                    break

    actor_pids = persons.ids[persons.actor_indices].astype(np.int64, copy=False)
    target_have = int(round(actor_pids.size * profile.rivalry_actor_coverage))
    current_have = int(np.count_nonzero(state.family_degree["rivalry"][actor_pids] > 0))
    if current_have < target_have:
        missing = persons.actor_indices[state.family_degree["rivalry"][actor_pids] == 0].copy()
        rng.shuffle(missing)
        for idx in missing.tolist():
            if current_have >= target_have:
                break
            pid = int(persons.ids[idx])
            genre_key = str(persons.primary_genre[idx]).lower()
            candidates = _sampled_union(
                rng,
                exclude_idx=idx,
                specs=[
                    (pools.market_genre_members.get((int(persons.market_bucket[idx]), genre_key), EMPTY_I64), _scaled_sample_size(len(pools.market_genre_members.get((int(persons.market_bucket[idx]), genre_key), EMPTY_I64)), 1.8, minimum=8, maximum=28)),
                    (pools.genre_stage_members.get((genre_key, str(persons.stage_name[idx])), EMPTY_I64), _scaled_sample_size(len(pools.genre_stage_members.get((genre_key, str(persons.stage_name[idx])), EMPTY_I64)), 1.6, minimum=6, maximum=22)),
                    (persons.actor_indices, _scaled_sample_size(len(persons.actor_indices), 0.45, minimum=6, maximum=14)),
                ],
            )
            if candidates.size == 0:
                continue
            metrics = _pair_metrics(persons, idx, candidates)
            score = 0.42 * metrics["genre_jaccard"] + 0.22 * (1.0 - metrics["style"]) + 0.18 * metrics["stage_sim"] + 0.10 * metrics["same_market"] + 0.08 * rng.rand(len(candidates)).astype(np.float32)
            for pick in _top_k_desc(score, min(len(score), 10)).tolist():
                cand = int(candidates[pick])
                other_pid = int(persons.ids[cand])
                if state.has_edge("friendship", pid, other_pid) or state.has_edge("rivalry", pid, other_pid):
                    continue
                if float(score[pick]) < max(profile.rivalry_threshold + 0.02, 0.32):
                    continue
                valid_from, valid_to = _valid_overlap(persons, idx, cand)
                if state.add(buffer, "rivalry", pid, other_pid, 0.48 + 0.28 * float(score[pick]), valid_from, valid_to):
                    current_have += 1
                    break
    return buffer


def _build_mentorship_edges(
    persons: PersonFeatures,
    pools: HotGraphPools,
    profile: GraphBuildProfile,
    state: HotGraphState,
    rng: np.random.RandomState,
    progress_hook: Callable[[str, int, int], None] | None = None,
) -> EdgeBuffer:
    buffer = EdgeBuffer("mentorship")
    mentor_candidates = np.flatnonzero((persons.stage_ord >= 3) & persons.is_creative | persons.is_director).astype(np.int64)
    total = len(mentor_candidates)
    for pos, idx in enumerate(mentor_candidates.tolist(), start=1):
        _status_progress("mentorship_building", pos, total, progress_hook, stride=5_000)
        mentor_id = int(persons.ids[idx])
        stage = str(persons.stage_name[idx])
        target = int(
            round(
                float(persons.cap[idx])
                * (
                    profile.mentorship_ratio
                    + 0.05 * float(persons.is_director[idx])
                    + 0.03 * float(persons.stage_ord[idx] >= STAGE_ORDER["veteran"])
                )
            )
        )
        if persons.is_director[idx] or persons.stage_ord[idx] >= STAGE_ORDER["veteran"]:
            target = max(target, 1)
        if int(state.family_out["mentorship"][mentor_id]) >= target:
            continue
        genre_key = str(persons.primary_genre[idx]).lower()
        base_actor_pool = persons.actor_indices if persons.is_director[idx] else pools.creative_indices
        candidates = _sampled_union(
            rng,
            exclude_idx=idx,
            specs=[
                (pools.community_genre_members.get((int(persons.community[idx]), genre_key), EMPTY_I64), _scaled_sample_size(len(pools.community_genre_members.get((int(persons.community[idx]), genre_key), EMPTY_I64)), 1.8, minimum=6, maximum=24)),
                (pools.agency_genre_members.get((int(persons.agency_id[idx]), genre_key), EMPTY_I64), _scaled_sample_size(len(pools.agency_genre_members.get((int(persons.agency_id[idx]), genre_key), EMPTY_I64)), 1.6, minimum=5, maximum=20)),
                (pools.market_genre_members.get((int(persons.market_bucket[idx]), genre_key), EMPTY_I64), _scaled_sample_size(len(pools.market_genre_members.get((int(persons.market_bucket[idx]), genre_key), EMPTY_I64)), 1.2, minimum=4, maximum=16)),
                (base_actor_pool, _scaled_sample_size(len(base_actor_pool), 0.35, minimum=6, maximum=16)),
            ],
        )
        if candidates.size == 0:
            continue
        stage_gap = persons.stage_ord[idx] - persons.stage_ord[candidates]
        candidates = candidates[(stage_gap >= 1) & (persons.stage_ord[candidates] <= STAGE_ORDER["prime"])]
        if candidates.size == 0:
            continue
        metrics = _pair_metrics(persons, idx, candidates)
        score = (
            0.36 * metrics["style"]
            + 0.24 * metrics["genre_jaccard"]
            + 0.16 * metrics["same_agency"]
            + 0.10 * metrics["same_community"]
            + 0.10 * persons.ambition[candidates]
            + 0.04 * rng.rand(len(candidates)).astype(np.float32)
        )
        for pick in _top_k_desc(score, min(len(score), max(target * 5, 10))).tolist():
            cand = int(candidates[pick])
            mentee_id = int(persons.ids[cand])
            if state.has_edge("mentorship", mentor_id, mentee_id) or state.has_edge("avoid", mentor_id, mentee_id):
                continue
            if float(score[pick]) < profile.mentorship_threshold:
                continue
            valid_from, valid_to = _valid_overlap(persons, idx, cand)
            if state.add(buffer, "mentorship", mentor_id, mentee_id, 0.44 + 0.38 * float(score[pick]), valid_from, valid_to):
                if int(state.family_out["mentorship"][mentor_id]) >= target:
                    break
    return buffer


def _build_avoid_edges(
    persons: PersonFeatures,
    pools: HotGraphPools,
    profile: GraphBuildProfile,
    state: HotGraphState,
    rng: np.random.RandomState,
    progress_hook: Callable[[str, int, int], None] | None = None,
) -> EdgeBuffer:
    buffer = EdgeBuffer("avoid")
    director_indices = np.asarray(pools.director_indices, dtype=np.int64)
    director_controversy_median = float(np.median(persons.controversy[persons.director_indices])) if len(persons.director_indices) else 0.5
    total = len(director_indices)
    for pos, idx in enumerate(director_indices.tolist(), start=1):
        _status_progress("avoid_building", pos, total, progress_hook, stride=2_000)
        did = int(persons.ids[idx])
        target = int(
            round(
                float(persons.cap[idx])
                * (
                    profile.avoid_ratio
                    + 0.05 * float(persons.is_director[idx])
                    + 0.10 * float(persons.controversy[idx])
                )
            )
        )
        target = max(target, 1 if persons.controversy[idx] > director_controversy_median else 0)
        if int(state.family_out["avoid"][did]) >= target:
            continue
        genre_key = str(persons.primary_genre[idx]).lower()
        candidates = _sampled_union(
            rng,
            exclude_idx=idx,
            specs=[
                (pools.market_genre_members.get((int(persons.market_bucket[idx]), genre_key), EMPTY_I64), _scaled_sample_size(len(pools.market_genre_members.get((int(persons.market_bucket[idx]), genre_key), EMPTY_I64)), 1.9, minimum=8, maximum=28)),
                (pools.community_genre_members.get((int(persons.community[idx]), genre_key), EMPTY_I64), _scaled_sample_size(len(pools.community_genre_members.get((int(persons.community[idx]), genre_key), EMPTY_I64)), 1.3, minimum=5, maximum=18)),
                (persons.actor_indices, _scaled_sample_size(len(persons.actor_indices), 0.35, minimum=6, maximum=16)),
            ],
        )
        if candidates.size == 0:
            continue
        candidates = candidates[persons.is_actor[candidates]]
        if candidates.size == 0:
            continue
        metrics = _pair_metrics(persons, idx, candidates)
        score = (
            0.30 * persons.controversy[candidates]
            + 0.22 * (1.0 - metrics["style"])
            + 0.16 * (1.0 - metrics["genre_jaccard"])
            + 0.12 * metrics["same_market"]
            + 0.10 * metrics["volatility"]
            + 0.06 * (1.0 - metrics["same_agency"])
            + 0.04 * rng.rand(len(candidates)).astype(np.float32)
        )
        for pick in _top_k_desc(score, min(len(score), max(target * 6, 12))).tolist():
            cand = int(candidates[pick])
            aid = int(persons.ids[cand])
            if state.has_edge("avoid", did, aid) or state.has_edge("mentorship", did, aid):
                continue
            if float(score[pick]) < profile.avoid_threshold:
                continue
            valid_from, valid_to = _valid_overlap(persons, idx, cand)
            if state.add(buffer, "avoid", did, aid, 0.50 + 0.30 * float(score[pick]), valid_from, valid_to):
                if int(state.family_out["avoid"][did]) >= target:
                    break

    controversy_people = np.flatnonzero((persons.controversy > 0.65) & persons.is_creative).astype(np.int64)
    for idx in controversy_people.tolist():
        target_id = int(persons.ids[idx])
        genre_key = str(persons.primary_genre[idx]).lower()
        candidates = _sampled_union(
            rng,
            exclude_idx=idx,
            specs=[
                (pools.market_genre_members.get((int(persons.market_bucket[idx]), genre_key), EMPTY_I64), _scaled_sample_size(len(pools.market_genre_members.get((int(persons.market_bucket[idx]), genre_key), EMPTY_I64)), 1.0, minimum=4, maximum=14)),
                (pools.director_indices, _scaled_sample_size(len(pools.director_indices), 0.25, minimum=4, maximum=10)),
                (pools.competitive_indices, _scaled_sample_size(len(pools.competitive_indices), 0.20, minimum=4, maximum=10)),
            ],
        )
        if candidates.size == 0:
            continue
        metrics = _pair_metrics(persons, idx, candidates)
        score = 0.34 * persons.controversy[idx] + 0.22 * (1.0 - metrics["style"]) + 0.18 * metrics["genre_jaccard"] + 0.14 * (1.0 - metrics["same_agency"]) + 0.12 * rng.rand(len(candidates)).astype(np.float32)
        for pick in _top_k_desc(score, min(len(score), 4)).tolist():
            cand = int(candidates[pick])
            src_id = int(persons.ids[cand])
            if src_id == target_id or state.has_edge("avoid", src_id, target_id):
                continue
            if float(score[pick]) < max(profile.avoid_threshold + 0.06, 0.34):
                continue
            valid_from, valid_to = _valid_overlap(persons, idx, cand)
            if state.add(buffer, "avoid", src_id, target_id, 0.54 + 0.24 * float(score[pick]), valid_from, valid_to):
                break
    return buffer


def _build_former_collaborator_edges(
    persons: PersonFeatures,
    pools: HotGraphPools,
    profile: GraphBuildProfile,
    state: HotGraphState,
    rng: np.random.RandomState,
    progress_hook: Callable[[str, int, int], None] | None = None,
) -> EdgeBuffer:
    buffer = EdgeBuffer("former_collaborator")
    creative = np.asarray(pools.creative_indices, dtype=np.int64)
    total = len(creative)
    for pos, idx in enumerate(creative.tolist(), start=1):
        _status_progress("former_collaborator_building", pos, total, progress_hook, stride=10_000)
        pid = int(persons.ids[idx])
        target = int(
            round(
                float(persons.cap[idx])
                * (
                    profile.former_ratio
                    + 0.03 * float(persons.is_director[idx])
                    + 0.03 * float(persons.reputation[idx])
                )
            )
        )
        if int(state.family_degree["former_collaborator"][pid]) >= target:
            continue
        genre_key = str(persons.primary_genre[idx]).lower()
        candidates = _sampled_union(
            rng,
            exclude_idx=idx,
            specs=[
                (pools.agency_genre_members.get((int(persons.agency_id[idx]), genre_key), EMPTY_I64), _scaled_sample_size(len(pools.agency_genre_members.get((int(persons.agency_id[idx]), genre_key), EMPTY_I64)), 1.4, minimum=4, maximum=16)),
                (pools.community_genre_members.get((int(persons.community[idx]), genre_key), EMPTY_I64), _scaled_sample_size(len(pools.community_genre_members.get((int(persons.community[idx]), genre_key), EMPTY_I64)), 1.4, minimum=4, maximum=16)),
                (pools.market_genre_members.get((int(persons.market_bucket[idx]), genre_key), EMPTY_I64), _scaled_sample_size(len(pools.market_genre_members.get((int(persons.market_bucket[idx]), genre_key), EMPTY_I64)), 1.0, minimum=3, maximum=12)),
            ],
        )
        if candidates.size == 0:
            continue
        candidates = candidates[persons.is_creative[candidates]]
        if candidates.size == 0:
            continue
        metrics = _pair_metrics(persons, idx, candidates)
        score = (
            0.30 * metrics["style"]
            + 0.24 * metrics["genre_jaccard"]
            + 0.18 * metrics["same_agency"]
            + 0.12 * metrics["same_community"]
            + 0.08 * metrics["stage_sim"]
            + 0.08 * rng.rand(len(candidates)).astype(np.float32)
        )
        for pick in _top_k_desc(score, min(len(score), max(target * 4, 8))).tolist():
            cand = int(candidates[pick])
            other_pid = int(persons.ids[cand])
            if state.has_edge("former_collaborator", pid, other_pid) or state.has_edge("rivalry", pid, other_pid):
                continue
            if float(score[pick]) < profile.former_threshold:
                continue
            valid_from, valid_to = _valid_overlap(persons, idx, cand)
            if state.add(buffer, "former_collaborator", pid, other_pid, 0.22 + 0.35 * float(score[pick]), valid_from, valid_to):
                if int(state.family_degree["former_collaborator"][pid]) >= target:
                    break
    return buffer


def _build_collaboration_edges(
    persons: PersonFeatures,
    pools: HotGraphPools,
    profile: GraphBuildProfile,
    state: HotGraphState,
    rng: np.random.RandomState,
    progress_hook: Callable[[str, int, int], None] | None = None,
) -> EdgeBuffer:
    buffer = EdgeBuffer("collaboration")
    creative = np.asarray(pools.creative_indices, dtype=np.int64)
    total = len(creative)
    for pos, idx in enumerate(creative.tolist(), start=1):
        _status_progress("collaboration_building", pos, total, progress_hook, stride=10_000)
        pid = int(persons.ids[idx])
        target = int(
            round(
                float(persons.cap[idx])
                * (
                    profile.collaboration_ratio
                    + 0.04 * float(persons.collaboration_style[idx] == _collab_style_codes().get("ensemble", 0))
                    + 0.02 * float(persons.is_director[idx])
                )
            )
        )
        if int(state.family_degree["collaboration"][pid]) >= target:
            continue
        genre_key = str(persons.primary_genre[idx]).lower()
        former_neighbors = _pid_values_to_indices(persons, _sample_values(state.adjacency["former_collaborator"].get(pid, []), size=6, rng=rng, exclude_value=pid))
        friend_neighbors = _pid_values_to_indices(persons, _sample_values(state.adjacency["friendship"].get(pid, []), size=5, rng=rng, exclude_value=pid))
        clique_neighbors = _pid_values_to_indices(persons, _sample_values(state.adjacency["clique"].get(pid, []), size=4, rng=rng, exclude_value=pid))
        candidates = _sampled_union(
            rng,
            exclude_idx=idx,
            specs=[
                (former_neighbors, len(former_neighbors)),
                (friend_neighbors, len(friend_neighbors)),
                (clique_neighbors, len(clique_neighbors)),
                (pools.agency_genre_members.get((int(persons.agency_id[idx]), genre_key), EMPTY_I64), _scaled_sample_size(len(pools.agency_genre_members.get((int(persons.agency_id[idx]), genre_key), EMPTY_I64)), 1.0, minimum=3, maximum=12)),
            ],
        )
        if candidates.size == 0:
            continue
        candidates = candidates[persons.is_creative[candidates]]
        if candidates.size == 0:
            continue
        metrics = _pair_metrics(persons, idx, candidates)
        has_former = np.array([1.0 if state.has_edge("former_collaborator", pid, int(persons.ids[cand])) else 0.0 for cand in candidates], dtype=np.float32)
        has_friend = np.array([1.0 if state.has_edge("friendship", pid, int(persons.ids[cand])) else 0.0 for cand in candidates], dtype=np.float32)
        score = (
            0.24 * metrics["style"]
            + 0.22 * metrics["genre_jaccard"]
            + 0.18 * has_former
            + 0.16 * has_friend
            + 0.10 * metrics["same_agency"]
            + 0.05 * metrics["same_community"]
            + 0.05 * rng.rand(len(candidates)).astype(np.float32)
        )
        for pick in _top_k_desc(score, min(len(score), max(target * 4, 8))).tolist():
            cand = int(candidates[pick])
            other_pid = int(persons.ids[cand])
            if state.has_edge("collaboration", pid, other_pid) or state.has_edge("rivalry", pid, other_pid):
                continue
            if float(score[pick]) < profile.collaboration_threshold:
                continue
            valid_from, valid_to = _valid_overlap(persons, idx, cand)
            if state.add(buffer, "collaboration", pid, other_pid, 0.24 + 0.40 * float(score[pick]), valid_from, valid_to):
                if int(state.family_degree["collaboration"][pid]) >= target:
                    break
    return buffer


def _build_clique_edges(
    persons: PersonFeatures,
    pools: HotGraphPools,
    profile: GraphBuildProfile,
    state: HotGraphState,
    rng: np.random.RandomState,
    progress_hook: Callable[[str, int, int], None] | None = None,
) -> EdgeBuffer:
    buffer = EdgeBuffer("clique")
    clusters = list(pools.microcluster_members.items())
    total = len(clusters)
    for pos, (_cluster_key, members) in enumerate(clusters, start=1):
        _status_progress("clique_building", pos, total, progress_hook, stride=2_000)
        if len(members) < 3:
            continue
        anchors = _sample_from_pool(members, size=_scaled_sample_size(len(members), profile.clique_ratio * 8.0, minimum=2, maximum=max(2, len(members))), rng=rng)
        for idx in anchors.tolist():
            pid = int(persons.ids[idx])
            clique_target = int(round(float(persons.cap[idx]) * profile.clique_ratio))
            clique_target = max(clique_target, 1 if len(members) >= 4 else 0)
            if int(state.family_degree["clique"][pid]) >= clique_target:
                continue
            candidates = _sample_from_pool(members, size=_scaled_sample_size(len(members), profile.clique_ratio * 10.0, minimum=3, maximum=min(len(members), 12)), rng=rng, exclude_idx=idx)
            if candidates.size == 0:
                continue
            metrics = _pair_metrics(persons, idx, candidates)
            score = (
                0.34 * metrics["style"]
                + 0.22 * metrics["same_agency"]
                + 0.18 * metrics["same_community"]
                + 0.12 * metrics["genre_jaccard"]
                + 0.08 * metrics["stage_sim"]
                + 0.06 * rng.rand(len(candidates)).astype(np.float32)
            )
            for pick in _top_k_desc(score, min(len(score), 4)).tolist():
                cand = int(candidates[pick])
                other_pid = int(persons.ids[cand])
                if state.has_edge("clique", pid, other_pid) or state.has_edge("rivalry", pid, other_pid):
                    continue
                if float(score[pick]) < profile.clique_threshold:
                    continue
                valid_from, valid_to = _valid_overlap(persons, idx, cand)
                if state.add(buffer, "clique", pid, other_pid, 0.50 + 0.26 * float(score[pick]), valid_from, valid_to):
                    if int(state.family_degree["clique"][pid]) >= clique_target:
                        break
    return buffer


def _yield_person_company_batches(
    persons: PersonFeatures,
    companies: CompanyFeatures,
    profile: GraphBuildProfile,
    rng: np.random.RandomState,
    *,
    world_policy: Mapping[str, Any] | None = None,
    stats: dict[str, Any],
    batch_size: int = 100_000,
) -> Iterator[dict[str, Any]]:
    buffers: dict[str, dict[str, list[np.ndarray]]] = {
        edge_type: {"src": [], "dst": [], "weight": [], "valid_from": [], "valid_to": []}
        for edge_type in ("brand_fit", "employment", "blacklist")
    }
    queued_counts = {edge_type: 0 for edge_type in buffers}
    stats.setdefault("counts", Counter())
    genre_basis_cache: dict[str, np.ndarray] = {}
    valid_from_years = np.maximum(_active_year_lo(), persons.debut.astype(np.int32, copy=False))
    valid_to_years = np.maximum(
        persons.debut.astype(np.int32, copy=False),
        np.minimum(persons.retire.astype(np.int32, copy=False), _active_edge_end_year()),
    )
    all_company_indices = np.arange(len(companies.ids), dtype=np.int64)
    clique_mod = max(1, len(companies.clique_members))
    company_strategy = np.asarray(
        [resolve_company_strategy(world_policy, int(cid), fallback="genre_lab") for cid in companies.ids.tolist()],
        dtype=object,
    )

    def _flush(edge_type: str) -> dict[str, Any] | None:
        chunk = buffers[edge_type]
        if not chunk["src"]:
            return None
        src = np.concatenate(chunk["src"]).astype(np.uint32, copy=False)
        dst = np.concatenate(chunk["dst"]).astype(np.uint32, copy=False)
        weight = np.concatenate(chunk["weight"]).astype(np.float32, copy=False)
        valid_from = np.concatenate(chunk["valid_from"]).astype(np.int32, copy=False)
        valid_to = np.concatenate(chunk["valid_to"]).astype(np.int32, copy=False)
        for key in chunk:
            chunk[key] = []
        queued_counts[edge_type] = 0
        stats["counts"][edge_type] += int(len(src))
        return {
            "edge_type": edge_type,
            "sign": "-" if edge_type == "blacklist" else "+",
            "src": src,
            "dst": dst,
            "weight": weight,
            "valid_from": valid_from,
            "valid_to": valid_to,
            "src_type": "company",
            "dst_type": "person",
            "source_kind": "latent_hybrid_v3",
            "source_batch": "generate_edges_hybrid_scalable",
            "reason": "",
        }

    for idx in range(len(persons.ids)):
        primary = str(persons.primary_genre[idx]).lower()
        parts = []
        if primary in companies.genre_members:
            parts.append(_sample_from_pool(companies.genre_members[primary], size=_scaled_sample_size(len(companies.genre_members[primary]), profile.company_genre_sample_ratio, minimum=24, maximum=192), rng=rng))
        clique_key = int((persons.agency_id[idx] % clique_mod) + 1)
        clique_candidates = companies.clique_members.get(clique_key, EMPTY_I64)
        if getattr(clique_candidates, "size", 0):
            parts.append(_sample_from_pool(clique_candidates, size=_scaled_sample_size(len(clique_candidates), profile.company_clique_sample_ratio, minimum=6, maximum=48), rng=rng))
        parts.append(_sample_from_pool(all_company_indices, size=_scaled_sample_size(len(all_company_indices), profile.company_random_sample_ratio, minimum=6, maximum=32), rng=rng))
        candidates = np.unique(np.concatenate(parts).astype(np.int64)) if parts else EMPTY_I64
        if candidates.size == 0:
            continue
        p_genre = genre_basis_cache.get(primary)
        if p_genre is None:
            p_genre = np.asarray(project_genres_to_company_basis([persons.primary_genre[idx]]), dtype=np.float32)
            genre_basis_cache[primary] = p_genre
        genre_overlap = companies.genre[candidates] @ p_genre
        tier_overlap = companies.tier[candidates] @ persons.budget_pref[idx]
        risk_match = 1.0 - np.abs(companies.risk[candidates] - persons.risk[idx])
        controversy_fit = 1.0 - np.clip(persons.controversy[idx] - companies.controversy_tolerance[candidates], 0.0, 1.0)
        fit = 0.44 * genre_overlap + 0.24 * tier_overlap + 0.16 * risk_match + 0.16 * controversy_fit
        primary_genre = str(persons.primary_genre[idx])
        if primary_genre in {"Action", "Sci-Fi", "Fantasy", "Animation"}:
            fit += 0.05 * (company_strategy[candidates] == "event_franchise").astype(np.float32)
        elif primary_genre in {"Drama", "Romance", "Mystery"}:
            fit += 0.05 * (company_strategy[candidates] == "prestige_drama").astype(np.float32)
        stage = str(persons.stage_name[idx])
        brand_cap = min(len(candidates), max(1, int(round(len(candidates) * profile.brand_fit_ratio_by_stage.get(stage, profile.brand_fit_ratio_by_stage["prime"])))))
        order = np.argsort(-fit)
        brand_pick = order[:brand_cap]
        brand_pick = brand_pick[fit[brand_pick] >= profile.brand_fit_threshold]
        if brand_pick.size:
            brand_idx = candidates[brand_pick]
            count = len(brand_idx)
            buffers["brand_fit"]["src"].append(companies.ids[brand_idx])
            buffers["brand_fit"]["dst"].append(np.full(count, int(persons.ids[idx]), dtype=np.uint32))
            buffers["brand_fit"]["weight"].append(np.clip(0.20 + 0.58 * fit[brand_pick], 0.14, 0.94).astype(np.float32))
            buffers["brand_fit"]["valid_from"].append(np.full(count, int(valid_from_years[idx]), dtype=np.int32))
            buffers["brand_fit"]["valid_to"].append(np.full(count, int(valid_to_years[idx]), dtype=np.int32))
            queued_counts["brand_fit"] += count
        employ_cap = min(len(candidates), max(1, int(round(len(candidates) * profile.employment_ratio_by_stage.get(stage, profile.employment_ratio_by_stage["prime"])))))
        if employ_cap > 0:
            employ_scores = fit + 0.12 * tier_overlap + 0.05 * (companies.market_bucket[candidates] == persons.market_bucket[idx]).astype(np.float32)
            employ_pick = np.argsort(-employ_scores)[:employ_cap]
            employ_idx = candidates[employ_pick]
            count = len(employ_idx)
            buffers["employment"]["src"].append(companies.ids[employ_idx])
            buffers["employment"]["dst"].append(np.full(count, int(persons.ids[idx]), dtype=np.uint32))
            buffers["employment"]["weight"].append(np.clip(0.40 + 0.40 * employ_scores[employ_pick], 0.22, 0.96).astype(np.float32))
            buffers["employment"]["valid_from"].append(np.full(count, int(valid_from_years[idx]), dtype=np.int32))
            buffers["employment"]["valid_to"].append(np.full(count, int(valid_to_years[idx]), dtype=np.int32))
            queued_counts["employment"] += count
        if persons.controversy[idx] > np.quantile(persons.controversy, 0.60):
            blacklist_scores = 0.50 * persons.controversy[idx] + 0.24 * (1.0 - controversy_fit) + 0.16 * (1.0 - risk_match) + 0.10 * (companies.market_bucket[candidates] == persons.market_bucket[idx]).astype(np.float32)
            blacklist_pick = np.flatnonzero(blacklist_scores > profile.blacklist_threshold)
            if blacklist_pick.size:
                blacklist_pick = blacklist_pick[np.argsort(-blacklist_scores[blacklist_pick])[: min(6, len(blacklist_pick))]]
                blk_idx = candidates[blacklist_pick]
                count = len(blk_idx)
                buffers["blacklist"]["src"].append(companies.ids[blk_idx])
                buffers["blacklist"]["dst"].append(np.full(count, int(persons.ids[idx]), dtype=np.uint32))
                buffers["blacklist"]["weight"].append(np.clip(0.42 + 0.34 * blacklist_scores[blacklist_pick], 0.24, 0.95).astype(np.float32))
                buffers["blacklist"]["valid_from"].append(np.full(count, int(valid_from_years[idx]), dtype=np.int32))
                buffers["blacklist"]["valid_to"].append(np.full(count, int(valid_to_years[idx]), dtype=np.int32))
                queued_counts["blacklist"] += count
        for edge_type in ("brand_fit", "employment", "blacklist"):
            if queued_counts[edge_type] >= batch_size:
                batch = _flush(edge_type)
                if batch is not None:
                    yield batch
    for edge_type in ("brand_fit", "employment", "blacklist"):
        batch = _flush(edge_type)
        if batch is not None:
            yield batch


def _yield_company_company_batches(
    companies: CompanyFeatures,
    profile: GraphBuildProfile,
    rng: np.random.RandomState,
    *,
    world_policy: Mapping[str, Any] | None = None,
    stats: dict[str, Any],
    batch_size: int = 60_000,
) -> Iterator[dict[str, Any]]:
    buffers: dict[str, dict[str, list[np.ndarray]]] = {
        edge_type: {"src": [], "dst": [], "weight": [], "valid_from": [], "valid_to": []}
        for edge_type in ("co_production", "market_rival", "subsidiary")
    }
    queued_counts = {edge_type: 0 for edge_type in buffers}
    stats.setdefault("counts", Counter())
    stats.setdefault("coverage", defaultdict(set))
    all_company_indices = np.arange(len(companies.ids), dtype=np.int64)
    company_strategy = np.asarray(
        [resolve_company_strategy(world_policy, int(cid), fallback="genre_lab") for cid in companies.ids.tolist()],
        dtype=object,
    )

    def _flush(edge_type: str) -> dict[str, Any] | None:
        chunk = buffers[edge_type]
        if not chunk["src"]:
            return None
        src = np.concatenate(chunk["src"]).astype(np.uint32, copy=False)
        dst = np.concatenate(chunk["dst"]).astype(np.uint32, copy=False)
        weight = np.concatenate(chunk["weight"]).astype(np.float32, copy=False)
        valid_from = np.concatenate(chunk["valid_from"]).astype(np.int32, copy=False)
        valid_to = np.concatenate(chunk["valid_to"]).astype(np.int32, copy=False)
        for key in chunk:
            chunk[key] = []
        queued_counts[edge_type] = 0
        stats["counts"][edge_type] += int(len(src))
        stats["coverage"][edge_type].update(int(v) for v in src.tolist())
        stats["coverage"][edge_type].update(int(v) for v in dst.tolist())
        return {
            "edge_type": edge_type,
            "sign": "-" if edge_type == "market_rival" else "+",
            "src": src,
            "dst": dst,
            "weight": weight,
            "valid_from": valid_from,
            "valid_to": valid_to,
            "src_type": "company",
            "dst_type": "company",
            "source_kind": "latent_hybrid_v3",
            "source_batch": "generate_edges_hybrid_scalable",
            "reason": "",
        }

    for idx in range(len(companies.ids)):
        primary = str(companies.primary_genre[idx]).lower()
        clique_pool = companies.clique_members.get(int(companies.clique_id[idx]), EMPTY_I64)
        market_pool = companies.market_members.get(int(companies.market_bucket[idx]), EMPTY_I64)
        candidates = _sampled_union(
            rng,
            exclude_idx=idx,
            specs=[
                (companies.genre_members.get(primary, EMPTY_I64), _scaled_sample_size(len(companies.genre_members.get(primary, EMPTY_I64)), profile.cc_genre_sample_ratio, minimum=24, maximum=144)),
                (clique_pool, _scaled_sample_size(len(clique_pool), profile.cc_clique_sample_ratio, minimum=8, maximum=48)),
                (market_pool, _scaled_sample_size(len(market_pool), profile.cc_market_sample_ratio, minimum=8, maximum=48)),
                (all_company_indices, _scaled_sample_size(len(all_company_indices), profile.cc_random_sample_ratio, minimum=6, maximum=24)),
            ],
        )
        candidates = candidates[candidates > idx]
        if candidates.size == 0:
            continue
        genre_overlap = companies.genre[candidates] @ companies.genre[idx]
        tier_gap = np.abs(companies.tier_rank[candidates].astype(np.float32) - float(companies.tier_rank[idx]))
        tier_sim = 1.0 - tier_gap / 4.0
        same_market = (companies.market_bucket[candidates] == companies.market_bucket[idx]).astype(np.float32)
        same_clique = (companies.clique_id[candidates] == companies.clique_id[idx]).astype(np.float32)
        risk_match = 1.0 - np.abs(companies.risk[candidates] - companies.risk[idx])
        same_strategy = (company_strategy[candidates] == company_strategy[idx]).astype(np.float32)

        rival_score = 0.36 * genre_overlap + 0.22 * tier_sim + 0.16 * same_market + 0.10 * same_strategy + 0.08 * (1.0 - same_clique * 0.6) + 0.08 * risk_match
        rival_pick = _top_k_desc(rival_score, min(len(rival_score), max(1, int(round(len(candidates) * profile.cc_rival_pick_ratio)))))
        rival_pick = rival_pick[rival_score[rival_pick] >= max(profile.rivalry_threshold, 0.30)]
        if rival_pick.size == 0 and rival_score.size:
            rival_pick = _top_k_desc(rival_score, 1)
        if rival_pick.size:
            chosen = candidates[rival_pick]
            count = len(chosen)
            buffers["market_rival"]["src"].append(np.full(count, int(companies.ids[idx]), dtype=np.uint32))
            buffers["market_rival"]["dst"].append(companies.ids[chosen])
            buffers["market_rival"]["weight"].append(np.clip(0.22 + 0.54 * rival_score[rival_pick], 0.16, 0.94).astype(np.float32))
            buffers["market_rival"]["valid_from"].append(np.full(count, _active_year_lo(), dtype=np.int32))
            buffers["market_rival"]["valid_to"].append(np.full(count, _active_edge_end_year(), dtype=np.int32))
            queued_counts["market_rival"] += count

        complementary_tier = 1.0 - np.abs(tier_gap - 1.0) / 4.0
        cross_market_bonus = 1.0 - same_market * 0.5
        copro_score = 0.36 * genre_overlap + 0.18 * complementary_tier + 0.14 * cross_market_bonus + 0.12 * same_strategy + 0.10 * (1.0 - same_clique * 0.4) + 0.10 * risk_match
        copro_pick = _top_k_desc(copro_score, min(len(copro_score), max(1, int(round(len(candidates) * profile.cc_copro_pick_ratio)))))
        copro_pick = copro_pick[copro_score[copro_pick] >= max(profile.collaboration_threshold, 0.22)]
        if copro_pick.size == 0 and copro_score.size:
            copro_pick = _top_k_desc(copro_score, 1)
        if copro_pick.size:
            chosen = candidates[copro_pick]
            count = len(chosen)
            buffers["co_production"]["src"].append(np.full(count, int(companies.ids[idx]), dtype=np.uint32))
            buffers["co_production"]["dst"].append(companies.ids[chosen])
            buffers["co_production"]["weight"].append(np.clip(0.18 + 0.48 * copro_score[copro_pick], 0.12, 0.90).astype(np.float32))
            buffers["co_production"]["valid_from"].append(np.full(count, _active_year_lo(), dtype=np.int32))
            buffers["co_production"]["valid_to"].append(np.full(count, _active_edge_end_year(), dtype=np.int32))
            queued_counts["co_production"] += count

        for edge_type in ("co_production", "market_rival"):
            if queued_counts[edge_type] >= batch_size:
                batch = _flush(edge_type)
                if batch is not None:
                    yield batch

    for clique_id, members in companies.clique_members.items():
        if len(members) < 3:
            continue
        ordered = np.array(sorted(members.tolist(), key=lambda q: (int(companies.tier_rank[int(q)]), int(companies.ids[int(q)]))), dtype=np.int64)
        parent_idx = int(ordered[0])
        children = ordered[1 : 1 + max(1, int(round(len(ordered) * profile.cc_subsidiary_pick_ratio)))]
        if children.size == 0:
            continue
        src = np.full(len(children), int(companies.ids[parent_idx]), dtype=np.uint32)
        dst = companies.ids[children].astype(np.uint32, copy=False)
        buffers["subsidiary"]["src"].append(src)
        buffers["subsidiary"]["dst"].append(dst)
        buffers["subsidiary"]["weight"].append(np.full(len(children), 0.82, dtype=np.float32))
        buffers["subsidiary"]["valid_from"].append(np.full(len(children), _active_year_lo(), dtype=np.int32))
        buffers["subsidiary"]["valid_to"].append(np.full(len(children), _active_edge_end_year(), dtype=np.int32))
        queued_counts["subsidiary"] += int(len(children))
        if queued_counts["subsidiary"] >= batch_size:
            batch = _flush("subsidiary")
            if batch is not None:
                yield batch

    for edge_type in ("co_production", "market_rival", "subsidiary"):
        batch = _flush(edge_type)
        if batch is not None:
            yield batch


def _write_communities(base_dir: str | Path, persons: PersonFeatures) -> None:
    graph_dir = Path(base_dir) / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    person_ids = persons.ids.astype(np.int32, copy=False)
    communities = persons.community.astype(np.int32, copy=False)
    table = pa.Table.from_pydict({"person_id": person_ids, "community": communities})
    with pa.OSFile(str(graph_dir / "communities.arrow"), "wb") as sink:
        with ipc.new_file(sink, table.schema, options=ipc.IpcWriteOptions(compression="lz4")) as writer:
            writer.write(table)
    with (graph_dir / "communities.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["person_id", "community"])
        writer.writerows(zip(person_ids.tolist(), communities.tolist()))


def _write_graph_quality_summary(
    base_dir: str | Path,
    *,
    persons: PersonFeatures,
    companies: CompanyFeatures,
    state: HotGraphState,
    hot_counts: Mapping[str, int],
    cp_counts: Mapping[str, int],
    cc_counts: Mapping[str, int],
    cc_coverage: Mapping[str, set[int]],
) -> None:
    graph_dir = Path(base_dir) / "graph"
    counts_by_family = {
        "friendship": int(hot_counts.get("friendship", 0)),
        "rivalry": int(hot_counts.get("rivalry", 0)),
        "mentorship": int(hot_counts.get("mentorship", 0)),
        "avoid": int(hot_counts.get("avoid", 0)),
        "former_collaborator": int(hot_counts.get("former_collaborator", 0)),
        "collaboration": int(hot_counts.get("collaboration", 0)),
        "clique": int(hot_counts.get("clique", 0)),
        "chemistry": 0,
        "brand_fit": int(cp_counts.get("brand_fit", 0)),
        "employment": int(cp_counts.get("employment", 0)),
        "blacklist": int(cp_counts.get("blacklist", 0)),
        "co_production": int(cc_counts.get("co_production", 0)),
        "market_rival": int(cc_counts.get("market_rival", 0)),
        "subsidiary": int(cc_counts.get("subsidiary", 0)),
    }
    positive_total = sum(count for edge_type, count in counts_by_family.items() if edge_type not in NEGATIVE_EDGE_TYPES)
    negative_total = sum(count for edge_type, count in counts_by_family.items() if edge_type in NEGATIVE_EDGE_TYPES)
    actor_pids = persons.ids[persons.actor_indices].astype(np.int64, copy=False)
    degree_percentiles: dict[str, dict[str, float]] = {}
    for edge_type in ("friendship", "rivalry", "mentorship", "avoid", "former_collaborator", "collaboration", "clique"):
        degree = state.family_degree[edge_type][persons.ids.astype(np.int64, copy=False)]
        degree = degree[degree > 0]
        if degree.size == 0:
            degree_percentiles[edge_type] = {"p50": 0.0, "p75": 0.0, "p90": 0.0, "p99": 0.0}
        else:
            degree_percentiles[edge_type] = {
                "p50": float(np.percentile(degree, 50)),
                "p75": float(np.percentile(degree, 75)),
                "p90": float(np.percentile(degree, 90)),
                "p99": float(np.percentile(degree, 99)),
            }
    payload = {
        "edge_counts_by_family": counts_by_family,
        "positive_negative_balance": {
            "positive_edges": int(positive_total),
            "negative_edges": int(negative_total),
            "negative_share": float(negative_total / max(1, positive_total + negative_total)),
        },
        "actor_coverage": {
            "friendship": float(np.count_nonzero(state.family_degree["friendship"][actor_pids] > 0) / max(1, len(actor_pids))),
            "rivalry": float(np.count_nonzero(state.family_degree["rivalry"][actor_pids] > 0) / max(1, len(actor_pids))),
            "mentorship": float(np.count_nonzero(state.family_degree["mentorship"][actor_pids] > 0) / max(1, len(actor_pids))),
        },
        "company_coverage": {
            edge_type: {
                "count": int(len(cc_coverage.get(edge_type, set()))),
                "fraction": float(len(cc_coverage.get(edge_type, set())) / max(1, len(companies.ids))),
            }
            for edge_type in ("co_production", "market_rival", "subsidiary")
        },
        "hot_degree_percentiles": degree_percentiles,
        "community_count": int(len(set(persons.community.tolist()))),
        "cross_community_bridge_rate": float(state.cross_community_friendships / max(1, hot_counts.get("friendship", 0))),
    }
    (graph_dir / "graph_quality_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_scalable_runtime_graph(
    base_dir: str | Path,
    *,
    persons: list[dict[str, Any]],
    companies: list[dict[str, Any]],
    person_latent: list[dict[str, Any]],
    company_latent: list[dict[str, Any]],
    world_policy: Mapping[str, Any] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    _load_scalable_prior_state(base_dir)
    _reset_graph_outputs(base_dir)
    _write_build_status(base_dir, "initializing", people=len(persons), companies=len(companies), seed=int(seed))
    rng = np.random.RandomState(seed)
    latent_map = {int(row.get("person_id")): row for row in person_latent}
    person_features = _build_person_features(persons, latent_map)
    company_features = _build_company_features(companies, company_latent)
    hot_pools = _build_hot_pools(person_features)
    profile = _build_graph_profile(person_features, company_features, hot_pools)
    state = HotGraphState(person_features.ids, person_features.community)

    _write_communities(base_dir, person_features)
    _write_build_status(base_dir, "communities_written", communities=int(len(set(person_features.community.tolist()))))

    def _progress(phase: str, current: int, total: int) -> None:
        _write_build_status(base_dir, phase, current=int(current), total=int(total), hot_counts=dict(state.counts))

    cp_stats: dict[str, Any] = {"counts": Counter()}
    cc_stats: dict[str, Any] = {"counts": Counter(), "coverage": defaultdict(set)}

    def _iter_batches() -> Iterator[dict[str, Any]]:
        family_builders = [
            ("friendship_built", _build_friendship_edges),
            ("rivalry_built", _build_rivalry_edges),
            ("mentorship_built", _build_mentorship_edges),
            ("avoid_built", _build_avoid_edges),
            ("former_collaborator_built", _build_former_collaborator_edges),
            ("collaboration_built", _build_collaboration_edges),
            ("clique_built", _build_clique_edges),
        ]
        for phase_name, builder in family_builders:
            buffer = builder(person_features, hot_pools, profile, state, rng, progress_hook=_progress)
            for batch in buffer.iter_batches(person_features):
                yield batch
            _write_build_status(base_dir, phase_name, hot_counts=dict(state.counts))

        _write_build_status(base_dir, "company_person_started", hot_counts=dict(state.counts))
        yield from _yield_person_company_batches(person_features, company_features, profile, rng, world_policy=world_policy, stats=cp_stats)

        _write_build_status(
            base_dir,
            "company_company_started",
            hot_counts=dict(state.counts),
            cold_cp_counts=dict(cp_stats["counts"]),
        )
        yield from _yield_company_company_batches(company_features, profile, rng, world_policy=world_policy, stats=cc_stats)

    _write_build_status(base_dir, "runtime_compile_started")
    GraphRuntime.compile_runtime_graph_batches(base_dir, batch_iter=_iter_batches(), source_label="generate_edges_hybrid_scalable")
    manifest = json.loads((Path(base_dir) / "graph" / "runtime_manifest.json").read_text(encoding="utf-8"))
    hot_counts = {edge_type: int(state.counts.get(edge_type, 0)) for edge_type in HOT_EDGE_TYPES}
    _write_graph_quality_summary(
        base_dir,
        persons=person_features,
        companies=company_features,
        state=state,
        hot_counts=hot_counts,
        cp_counts=cp_stats["counts"],
        cc_counts=cc_stats["counts"],
        cc_coverage=cc_stats["coverage"],
    )
    _write_build_status(
        base_dir,
        "completed",
        history_rows=int(manifest.get("history_rows", 0)),
        cold_cp_count=int(manifest.get("cold_cp_count", 0)),
        cold_cc_count=int(manifest.get("cold_cc_count", 0)),
        hot_counts=hot_counts,
        cold_cp_counts=dict(cp_stats["counts"]),
        cold_cc_counts=dict(cc_stats["counts"]),
    )
    return {
        "hot_counts": hot_counts,
        "cold_cp_counts": dict(cp_stats["counts"]),
        "cold_cc_counts": dict(cc_stats["counts"]),
        "communities": int(len(set(person_features.community.tolist()))),
        "people": int(len(person_features.ids)),
        "companies": int(len(company_features.ids)),
        "history_rows": int(manifest.get("history_rows", 0)),
        "cold_cp_count": int(manifest.get("cold_cp_count", 0)),
        "cold_cc_count": int(manifest.get("cold_cc_count", 0)),
    }
