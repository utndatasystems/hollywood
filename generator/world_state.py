from __future__ import annotations

"""
Rewritten world_state.py
========================

This module owns all persistent in-memory state used by movie assembly:
entities, graph-derived affinity lookups, latent-variable caches, hidden
confounders, career timelines, franchise slots, and year-filtered actor caches.

Design goals
------------
- keep the public API stable for assembly.py / generate_movies.py /
  temporal_evolution_api.py / financials.py
- make load-time behaviour deterministic and easier to reason about
- keep hot-path lookups O(1) after load()
- provide sane fallbacks when optional artifacts are missing
"""

import json
import hashlib
import logging
import os
import random
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import sys
sys.path.insert(0, os.path.dirname(__file__))

from pipeline_runtime import resolve_workspace, year_bounds_from_env
from contracts import (
    AWARD_CAMPAIGN_GENRES,
    BUDGET_RANGES,
    CAST_SIZE_RANGES,
    CERTIFICATIONS,
    CERT_DISTS,
    COUNTRIES,
    COUNTRY_LANGUAGE,
    COUNTRY_WEIGHTS,
    CREW_DEPARTMENTS,
    DECADE_WEIGHTS,
    DIRECTOR_STYLES,
    ENTITY_COUNTS,
    FRANCHISE_CONFIG,
    GENRES,
    GENRE_WEIGHTS,
    N_AGENCIES,
    N_COMPANY_CLIQUES,
    PRODUCTION_TIERS,
    SNAPSHOT_CONFIG,
    STYLE_TAGS,
    TIER_WEIGHTS,
    YEAR_RANGE,
    ARCHETYPES,
    generate_compositional_title,
)
from utils import (
    _safe_float,
    _clip01,
    canonical_company_genre_vector,
    normalize_weights,
    project_genres_to_company_basis,
)
from text_polish import (
    contains_placeholder_syntax,
    looks_like_weak_tagline,
    looks_like_weak_title,
    sanitize_tagline,
    sanitize_title,
)
from graph_runtime import GraphRuntime
from policy_runtime import (
    build_default_franchise_bibles,
    concept_packs_path,
    decision_log_path,
    decision_log_path_for_run,
    ensure_support_dirs,
    enrich_keyword_dataframe,
    franchise_bibles_path,
    index_concept_packs,
    index_franchise_bibles,
    index_year_slate_plan,
    keyword_motif_bank_path,
    llm_usage_log_path,
    modeling_priors_path,
    movie_progress_log_path,
    movie_progress_log_path_for_run,
    prompt_calibration_log_path,
    resolve_company_strategy,
    safe_int,
    safe_load_json,
    world_policy_path,
    year_slate_plan_path,
)

log = logging.getLogger(__name__)


# ============================================================================
# Module-level helpers
# ============================================================================

# C1-FIX: _clip01 now imported from utils.py


def _split_multi_value(raw: Any) -> List[str]:
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str):
        return [t.strip() for t in raw.replace(";", ",").split(",") if t.strip()]
    return []


def _stable_unit_interval(*parts: Any) -> float:
    key = "||".join(str(p) for p in parts).encode("utf-8", errors="ignore")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "big") / float((1 << 64) - 1)


def _normalize_distribution(values: List[float]) -> List[float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return []
    arr = np.clip(arr, 1e-9, None)
    s = float(arr.sum())
    if s <= 0:
        return [1.0 / len(arr)] * len(arr)
    return (arr / s).tolist()


def _parse_tag_set(raw: str) -> frozenset[str]:
    raw = str(raw or "").lower().replace("[", "").replace("]", "").replace("'", "")
    return frozenset(t.strip() for t in raw.replace(";", ",").split(",") if t.strip())


def _normalize_vec(vec: Any, dim: int) -> np.ndarray:
    a = np.asarray(vec, dtype=np.float32)
    if a.ndim != 1 or len(a) != dim:
        a = np.full(dim, 0.5, dtype=np.float32)
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-10 else a


def _cosine_sim_cached(a: Any, b: Any) -> float:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    if aa.shape != bb.shape:
        return 0.0
    return float(max(0.0, min(1.0, np.dot(aa, bb))))


# ============================================================================
# Latent fallback getters
# ============================================================================

def get_person_latent(world: "WorldState", person_id: int) -> Dict[str, Any]:
    pid = int(person_id)
    lv = world.person_latent.get(pid)
    if isinstance(lv, dict):
        return lv

    pop = _safe_float(world.person_pop_weight.get(pid, 0.1), default=0.1)
    return {
        "person_id": pid,
        "creative_style_vector": [0.0] * 8,
        "risk_tolerance": 0.5,
        "collaboration_style": "ensemble",
        "controversy_score": 0.15,
        "public_reputation": float(np.clip(pop, 0.0, 1.0)),
        "budget_band_pref": [0.5] * 5,
        "artistic_ambition": 0.5,
        "volatility": 0.4,
    }


def get_company_latent(world: "WorldState", company_id: int) -> Dict[str, Any]:
    cid = int(company_id)
    lv = world.company_latent.get(cid)
    if isinstance(lv, dict):
        return lv

    pop = _safe_float(world.company_pop_weight.get(cid, 0.5), default=0.5)
    tier_to_focus = {
        "Global": [0.60, 0.25, 0.10, 0.03, 0.02],
        "Major": [0.20, 0.50, 0.20, 0.07, 0.03],
        "Mid-Budget": [0.05, 0.20, 0.50, 0.20, 0.05],
        "Indie": [0.02, 0.05, 0.20, 0.55, 0.18],
        "Micro": [0.01, 0.02, 0.10, 0.35, 0.52],
    }
    tier_str = getattr(world, "_company_tier_map", {}).get(cid, "Mid-Budget")
    return {
        "company_id": cid,
        "risk_appetite": 0.5,
        "prestige_score": float(np.clip(pop, 0.0, 1.0)),
        "genre_portfolio": [1.0 / 12] * 12,
        "budget_tier_focus": tier_to_focus.get(tier_str, tier_to_focus["Mid-Budget"]),
        "controversy_tolerance": 0.5,
        "market_trend_sensitivity": 0.5,
    }


# ============================================================================
# Similarity
# ============================================================================

def latent_similarity(world: "WorldState", pid_a: int, pid_b: int) -> float:
    ca = world._person_sim_cache.get(int(pid_a))
    cb = world._person_sim_cache.get(int(pid_b))
    if ca is None or cb is None:
        return 0.0

    csv_a = ca.get("csv_normed")
    csv_b = cb.get("csv_normed")
    csv_sim = float(max(0.0, min(1.0, np.dot(csv_a, csv_b)))) if csv_a is not None and csv_b is not None else _cosine_sim_cached(
        ca.get("creative_style_vector", [0.0] * 8),
        cb.get("creative_style_vector", [0.0] * 8),
    )

    ga_a, ga_b = ca["genre_set"], cb["genre_set"]
    genre_sim = len(ga_a & ga_b) / max(1, len(ga_a | ga_b))

    st_a, st_b = ca["style_set"], cb["style_set"]
    style_sim = len(st_a & st_b) / max(1, len(st_a | st_b))

    lat_dist = (abs(ca["risk_tolerance"] - cb["risk_tolerance"]) + abs(ca["artistic_ambition"] - cb["artistic_ambition"])) / 2.0
    scalar_sim = max(0.0, 1.0 - lat_dist)

    bbp_a = ca.get("bbp_normed")
    bbp_b = cb.get("bbp_normed")
    bbp_sim = float(max(0.0, min(1.0, np.dot(bbp_a, bbp_b)))) if bbp_a is not None and bbp_b is not None else _cosine_sim_cached(
        ca.get("budget_band_pref", [0.5] * 5),
        cb.get("budget_band_pref", [0.5] * 5),
    )

    collab_match = 1.0 if ca.get("collaboration_style") == cb.get("collaboration_style") else 0.0

    return float(
        0.35 * csv_sim
        + 0.20 * genre_sim
        + 0.15 * style_sim
        + 0.15 * scalar_sim
        + 0.10 * bbp_sim
        + 0.05 * collab_match
    )


# Popcount lookup table for 16-bit integers (used by vectorized Jaccard)
_POPCOUNT16 = np.array([bin(i).count('1') for i in range(65536)], dtype=np.int8)

def _popcount32(arr: np.ndarray) -> np.ndarray:
    """Vectorized popcount for uint32 arrays via 16-bit table lookup."""
    a = arr.astype(np.uint32)
    lo = (a & 0xFFFF).astype(np.uint16)
    hi = ((a >> 16) & 0xFFFF).astype(np.uint16)
    return _POPCOUNT16[lo].astype(np.int32) + _POPCOUNT16[hi].astype(np.int32)


def latent_similarity_batch(
    world: "WorldState",
    cand_latent_indices: np.ndarray,
    cast_latent_indices: np.ndarray,
) -> np.ndarray:
    """Vectorized full 6-component similarity: candidates (m) vs cast (k).

    Returns shape (m,) — max similarity of each candidate across all cast members.
    Uses pre-built arrays: CSV, BBP (matrix multiply), genre/style (bit-vector
    Jaccard via popcount), risk/ambition (arithmetic), collab (int comparison).
    """
    m = len(cand_latent_indices)
    k = len(cast_latent_indices)
    if m == 0 or k == 0:
        return np.zeros(m, dtype=np.float32)

    # 1. CSV similarity (weight 0.35) — matrix multiply
    cand_csv = world._latent_csv_normed[cand_latent_indices]   # (m, 8)
    cast_csv = world._latent_csv_normed[cast_latent_indices]   # (k, 8)
    csv_sim = np.clip(cand_csv @ cast_csv.T, 0.0, 1.0)        # (m, k)

    # 2. BBP similarity (weight 0.10) — matrix multiply
    cand_bbp = world._latent_bbp_normed[cand_latent_indices]   # (m, 5)
    cast_bbp = world._latent_bbp_normed[cast_latent_indices]   # (k, 5)
    bbp_sim = np.clip(cand_bbp @ cast_bbp.T, 0.0, 1.0)        # (m, k)

    # 3. Genre Jaccard (weight 0.20) — bit-vector popcount
    cand_gbits = world._latent_genre_bits[cand_latent_indices] # (m,) uint32
    cast_gbits = world._latent_genre_bits[cast_latent_indices] # (k,) uint32
    # Broadcast: (m,1) op (1,k) -> (m,k)
    g_inter = _popcount32((cand_gbits[:, None] & cast_gbits[None, :]).ravel()).reshape(m, k)
    g_union = _popcount32((cand_gbits[:, None] | cast_gbits[None, :]).ravel()).reshape(m, k)
    genre_sim = np.divide(
        g_inter.astype(np.float32),
        g_union.astype(np.float32),
        out=np.zeros_like(g_inter, dtype=np.float32),
        where=g_union > 0,
    )

    # 4. Style Jaccard (weight 0.15) — bit-vector popcount
    cand_sbits = world._latent_style_bits[cand_latent_indices] # (m,) uint32
    cast_sbits = world._latent_style_bits[cast_latent_indices] # (k,) uint32
    s_inter = _popcount32((cand_sbits[:, None] & cast_sbits[None, :]).ravel()).reshape(m, k)
    s_union = _popcount32((cand_sbits[:, None] | cast_sbits[None, :]).ravel()).reshape(m, k)
    style_sim = np.divide(
        s_inter.astype(np.float32),
        s_union.astype(np.float32),
        out=np.zeros_like(s_inter, dtype=np.float32),
        where=s_union > 0,
    )

    # 5. Scalar similarity (weight 0.15) — risk + ambition distance
    cand_risk = world._latent_risk[cand_latent_indices]        # (m,)
    cast_risk = world._latent_risk[cast_latent_indices]        # (k,)
    cand_amb  = world._latent_ambition[cand_latent_indices]    # (m,)
    cast_amb  = world._latent_ambition[cast_latent_indices]    # (k,)
    lat_dist = (np.abs(cand_risk[:, None] - cast_risk[None, :]) +
                np.abs(cand_amb[:, None] - cast_amb[None, :])) / 2.0
    scalar_sim = np.clip(1.0 - lat_dist, 0.0, 1.0)            # (m, k)

    # 6. Collab match (weight 0.05) — integer comparison
    cand_collab = world._latent_collab_code[cand_latent_indices]  # (m,) int8
    cast_collab = world._latent_collab_code[cast_latent_indices]  # (k,) int8
    collab_sim = (cand_collab[:, None] == cast_collab[None, :]).astype(np.float32)  # (m, k)

    # Weighted combination
    total = (0.35 * csv_sim + 0.20 * genre_sim + 0.15 * style_sim +
             0.15 * scalar_sim + 0.10 * bbp_sim + 0.05 * collab_sim)  # (m, k)

    return total.max(axis=1).astype(np.float32)  # (m,)


# ============================================================================
# WorldState
# ============================================================================

class WorldState:
    """Persistent state container used by generation, evolution, and post-processing."""

    def __init__(self, base_dir: str, seed: int = 42, config_path: str | None = None, workspace: Any | None = None):
        self.base_dir = Path(base_dir)
        self.rng = np.random.RandomState(seed)
        self.py_rng = random.Random(seed)
        self.seed = int(seed)
        self.config_path = config_path
        default_config = self.base_dir / "v18_config.json"
        self.workspace = workspace or resolve_workspace(
            script_dir=self.base_dir,
            data_dir=self.base_dir,
            output_dir=self.base_dir,
            config_path=config_path or (str(default_config) if default_config.exists() else None),
        )

        # Core entities / graph
        self.persons: Optional[pd.DataFrame] = None
        self.person_roles: Optional[pd.DataFrame] = None
        self.actors: Optional[pd.DataFrame] = None
        self.directors: Optional[pd.DataFrame] = None
        self.crew_pools: Dict[str, pd.DataFrame] = {}
        self.companies: Optional[pd.DataFrame] = None
        self.keywords: Optional[pd.DataFrame] = None
        self.title_bank: Optional[pd.DataFrame] = None
        self.character_bank: Optional[pd.DataFrame] = None
        self.world_policy: dict[str, Any] = {}
        self.modeling_priors_payload: dict[str, Any] = {}
        self.concept_packs_payload: dict[str, Any] = {}
        self.concept_packs: list[dict[str, Any]] = []
        self.concept_packs_index: dict[str, dict[str, list[dict[str, Any]]]] = {}
        self.edge_graph = None
        self.graph: GraphRuntime | None = None
        self.affinity_index: Optional[dict] = None
        self.communities: Dict[int, int] = {}

        # Confounders / structural assignments
        self.person_agency: Dict[int, int] = {}
        self.company_clique: Dict[int, int] = {}
        self.company_family: Dict[int, set] = {}
        self._merge_families: Dict[int, set] = {}

        # Latents and lookups
        self.person_latent: Dict[int, dict] = {}
        self.company_latent: Dict[int, dict] = {}
        self.person_pop_weight: Dict[int, float] = {}
        self.company_pop_weight: Dict[int, float] = {}
        self.company_financial_profile: Dict[int, dict] = {}
        self._company_tier_map: Dict[int, str] = {}
        self._person_sim_cache: Dict[int, dict] = {}
        self._latent_pid_to_idx: Dict[int, int] = {}
        self._latent_csv_normed: Optional[np.ndarray] = None
        self._latent_bbp_normed: Optional[np.ndarray] = None
        self._latent_public_reputation: Optional[np.ndarray] = None
        self._latent_controversy: Optional[np.ndarray] = None
        self._latent_volatility: Optional[np.ndarray] = None
        self._latent_collab: Optional[np.ndarray] = None
        self._latent_avoid_genres: Dict[int, set] = {}
        self.edge_weights: Dict[tuple, float] = {}

        # Node-keyed edge adjacency (built once at load time)
        self._friend_adj_all: Dict[int, list] = {}
        self._rival_adj_all: Dict[int, list] = {}

        # Hot-path caches
        self._year_cache: Dict[int, pd.DataFrame] = {}
        self._company_by_tier_genre: Dict[tuple, set] = {}
        self._yearly_workload = Counter()
        self._used_char_names_global = set()
        self._crew_fallback_warned = set()
        self._title_bank_used_mask: Optional[np.ndarray] = None
        self._title_bank_all_idx: np.ndarray = np.zeros(0, dtype=np.int32)
        self._title_bank_genre_idx: Dict[str, np.ndarray] = {}

        # Runtime generation state
        self.person_award_wins: Dict[int, int] = {}
        self.director_quality_offset: Dict[int, float] = {}
        self.person_film_count = Counter()
        self.person_recent = defaultdict(list)
        self.director_recent = defaultdict(list)
        self.company_recent = defaultdict(list)
        self.director_writer_history: Dict[int, set] = {}
        self.director_film_count = Counter()
        self.company_film_count = Counter()
        self.used_titles = set()
        self.franchises: List[dict] = []
        self.movie_franchise_map: Dict[int, dict] = {}

        # Event / temporal state
        self.active_effects: List[dict] = []
        self.world_events: List[dict] = []
        self.genre_weight_overrides: Dict[str, float] = {}
        self.country_weight_overrides: Dict[str, float] = {}
        self.award_prestige: Dict[str, float] = {}
        self.paused_persons: Dict[int, dict] = {}

        # Financial momentum state
        self.director_recent_outcomes = defaultdict(list)
        self.company_recent_outcomes = defaultdict(list)
        self.genre_recent_outcomes = defaultdict(list)
        self._financial_regime_cache: Dict[int, dict] = {}

        # Misc temporal-spawn state
        self._chemistry_pairs = set()
        self._yearly_friendship_spawns: Dict[int, int] = {}
        self.enable_llm_world_policy = False
        self.enable_llm_concept_packs = False
        self.enable_llm_year_slates = False
        self.enable_llm_keyword_motifs = False
        self.enable_llm_rerank = False
        self.enable_llm_keyword_rerank = False
        self.rerank_budget_movies = 0
        self.rerank_budget_remaining = 0
        self.keyword_rerank_budget_movies = 0
        self.keyword_rerank_budget_remaining = 0
        run_id = str(os.getenv("DATA_SYS_RUN_ID", "") or "").strip()
        self.decision_log_path = str(decision_log_path_for_run(self.base_dir, run_id) if run_id else decision_log_path(self.base_dir))
        self.decision_log_latest_path = str(decision_log_path(self.base_dir))
        self.movie_progress_log_path = str(movie_progress_log_path_for_run(self.base_dir, run_id) if run_id else movie_progress_log_path(self.base_dir))
        self.movie_progress_log_latest_path = str(movie_progress_log_path(self.base_dir))
        self.llm_usage_log_path = str(llm_usage_log_path(self.base_dir))
        self.prompt_calibration_log_path = str(prompt_calibration_log_path(self.base_dir))

        # Diagnostic-only memory audit
        self._memory_audit = None
        self._memory_audit_ready = False

    def _get_memory_audit(self):
        if self._memory_audit_ready:
            return self._memory_audit
        self._memory_audit_ready = True
        try:
            from memory_probe import get_env_audit_recorder
        except Exception:
            self._memory_audit = None
            return None
        self._memory_audit = get_env_audit_recorder(default_experiment="world-load")
        return self._memory_audit

    def get_selection_year_state(self, year: int):
        from assembly import _get_selection_year_state

        return _get_selection_year_state(self, int(year))

    def _memory_audit_snapshot(
        self,
        phase: str,
        specs: Sequence[tuple],
        *,
        note: str = "",
        metadata: Mapping[str, Any] | None = None,
        sample_kind: str = "checkpoint",
    ) -> None:
        audit = self._get_memory_audit()
        if audit is None:
            return
        try:
            from memory_probe import audit_target
        except Exception:
            return
        targets = []
        for spec in specs:
            if hasattr(spec, "name") and hasattr(spec, "category"):
                targets.append(spec)
                continue
            if len(spec) < 3:
                continue
            name = spec[0]
            obj = spec[1]
            category = spec[2]
            target_note = spec[3] if len(spec) > 3 else ""
            overlap_group = spec[4] if len(spec) > 4 else ""
            measure = spec[5] if len(spec) > 5 else True
            targets.append(audit_target(name, obj, category, target_note, overlap_group, measure=measure))
        audit.record_snapshot(
            phase,
            targets,
            note=note,
            metadata=metadata,
            sample_kind=sample_kind,
        )

    @staticmethod
    def _empty_affinity_index() -> dict[str, Any]:
        return {
            "friendships": {},
            "rivalries": {},
            "director_prefs": defaultdict(list),
            "director_avoids": defaultdict(list),
            "company_affinity": {},
            "company_rivalry": {},
            "person_company_affinity": defaultdict(list),
        }

    def _prepare_title_bank_cache(self) -> None:
        if self.title_bank is None:
            self.title_bank = pd.DataFrame(columns=["title", "tagline", "genre_hint"])
        if len(self.title_bank) == 0:
            self._title_bank_used_mask = np.zeros(0, dtype=bool)
            self._title_bank_all_idx = np.zeros(0, dtype=np.int32)
            self._title_bank_genre_idx = {}
            return

        work = self.title_bank.copy()
        if "title" not in work.columns:
            work["title"] = ""
        if "tagline" not in work.columns:
            work["tagline"] = ""
        if "genre_hint" not in work.columns:
            work["genre_hint"] = ""

        raw_titles = work["title"].fillna("").astype(str).tolist()
        raw_taglines = work["tagline"].fillna("").astype(str).tolist()
        clean_titles = [sanitize_title(value) for value in raw_titles]
        clean_taglines = [
            sanitize_tagline(tagline, title=title)
            for tagline, title in zip(raw_taglines, clean_titles)
        ]
        title_ok_static = np.fromiter(
            (
                bool(title)
                and not contains_placeholder_syntax(title)
                and not looks_like_weak_title(title)
                for title in clean_titles
            ),
            dtype=bool,
            count=len(clean_titles),
        )
        tagline_ok_static = np.fromiter(
            (
                bool(tagline)
                and not contains_placeholder_syntax(tagline)
                and not looks_like_weak_tagline(tagline, title=title)
                for title, tagline in zip(clean_titles, clean_taglines)
            ),
            dtype=bool,
            count=len(clean_titles),
        )
        title_word_count = np.fromiter(
            (len(str(title).split()) for title in clean_titles),
            dtype=np.int16,
            count=len(clean_titles),
        )
        title_has_pulp_punct = np.fromiter(
            ((":" in str(title)) or ("-" in str(title)) for title in clean_titles),
            dtype=bool,
            count=len(clean_titles),
        )
        genre_canon = work["genre_hint"].fillna("").astype(str).str.strip().str.lower().to_numpy(dtype=object)

        work["_tb_row_idx"] = np.arange(len(work), dtype=np.int32)
        work["_title_clean"] = clean_titles
        work["_tagline_clean"] = clean_taglines
        work["_title_ok_static"] = title_ok_static
        work["_tagline_ok_static"] = tagline_ok_static
        work["_research_ok_static"] = title_ok_static & tagline_ok_static
        work["_nonresearch_tagline_weight"] = np.where(tagline_ok_static, 1.18, 0.72)
        work["_title_word_count"] = title_word_count
        work["_title_has_pulp_punct"] = title_has_pulp_punct
        work["_genre_hint_canon"] = genre_canon

        self.title_bank = work
        self._title_bank_used_mask = np.zeros(len(work), dtype=bool)
        self._title_bank_all_idx = work["_tb_row_idx"].to_numpy(dtype=np.int32, copy=False)
        self._title_bank_genre_idx = {}
        for genre in pd.unique(work["_genre_hint_canon"]):
            key = str(genre or "").strip().lower()
            if not key:
                continue
            self._title_bank_genre_idx[key] = work.loc[
                work["_genre_hint_canon"] == genre,
                "_tb_row_idx",
            ].to_numpy(dtype=np.int32, copy=True)

    def available_title_bank_indices(self, genre_hint: str | None = None) -> np.ndarray:
        if self.title_bank is None or len(self.title_bank) == 0:
            return np.zeros(0, dtype=np.int32)
        if self._title_bank_used_mask is None or len(self._title_bank_used_mask) != len(self.title_bank):
            self._prepare_title_bank_cache()
        used_mask = self._title_bank_used_mask
        if used_mask is None:
            return np.zeros(0, dtype=np.int32)
        genre_key = str(genre_hint or "").strip().lower()
        base = self._title_bank_genre_idx.get(genre_key, np.zeros(0, dtype=np.int32)) if genre_key else self._title_bank_all_idx
        if base.size == 0 and genre_key:
            base = self._title_bank_all_idx
        if base.size == 0:
            return np.zeros(0, dtype=np.int32)
        return base[~used_mask[base]]

    def mark_title_used(self, title: object, *, row_idx: int | None = None) -> None:
        clean = sanitize_title(title)
        if clean:
            self.used_titles.add(clean)
        if row_idx is None:
            return
        if self._title_bank_used_mask is None or row_idx < 0 or row_idx >= len(self._title_bank_used_mask):
            return
        self._title_bank_used_mask[int(row_idx)] = True

    # ------------------------------------------------------------------
    # Public load
    # ------------------------------------------------------------------

    def load(self):
        edir = self.base_dir / "entities"
        if not edir.exists():
            raise FileNotFoundError(f"Entities directory not found: {edir}")
        ensure_support_dirs(self.base_dir)

        self.persons = self._load_csv_with_id(edir / "person.csv", "person_id")
        self._assign_career_timelines()
        self._assign_pop_weights()
        self._memory_audit_snapshot(
            "world_load_persons_ready",
            [
                ("persons", self.persons, "entities", "Primary persons table after pop/career enrichment"),
            ],
            metadata={"person_rows": len(self.persons) if self.persons is not None else 0},
        )
        self.person_roles = self._load_person_roles(edir)
        self._build_person_role_views()
        role_specs = [
            ("persons", self.persons, "entities", "Shared base table for role-derived views"),
            ("person_roles", self.person_roles, "entities"),
            ("actors", self.actors, "role_views", "Subset copy derived from persons", "persons_role_views"),
            ("directors", self.directors, "role_views", "Subset copy derived from persons", "persons_role_views"),
            ("crew_pools", self.crew_pools, "role_views", "Dictionary of crew-specific DataFrame copies", "persons_role_views"),
        ]
        for crew_role in CREW_DEPARTMENTS:
            pool = self.crew_pools.get(crew_role)
            if pool is not None and len(pool) > 0:
                role_specs.append(
                    (
                        f"crew_pool_{crew_role}",
                        pool,
                        "role_views",
                        f"Role-specific DataFrame copy for {crew_role}",
                        "persons_role_views",
                    )
                )
        self._memory_audit_snapshot(
            "world_load_role_views_ready",
            role_specs,
            metadata={
                "actor_rows": len(self.actors) if self.actors is not None else 0,
                "director_rows": len(self.directors) if self.directors is not None else 0,
            },
        )

        self.companies = self._load_csv_with_id(edir / "company.csv", "company_id")
        self.companies["pop_weight"] = pd.to_numeric(self.companies.get("pop_weight", 0.5), errors="coerce").fillna(0.5)
        self._assign_company_pop_weights()

        self.keywords = self._load_csv_with_id(edir / "keyword.csv", "keyword_id")

        tb_path = edir / "title_bank.csv"
        self.title_bank = pd.read_csv(tb_path) if tb_path.exists() else pd.DataFrame(columns=["title", "tagline", "genre_hint"])
        self._prepare_title_bank_cache()

        cb_path = edir / "character_bank.csv"
        if not cb_path.exists():
            raise FileNotFoundError(f"character_bank.csv not found at {cb_path}")
        self.character_bank = pd.read_csv(cb_path)
        self.world_policy = safe_load_json(world_policy_path(self.base_dir), default={}) or {}
        self.modeling_priors_payload = safe_load_json(modeling_priors_path(self.base_dir), default={}) or {}
        self.concept_packs_payload = safe_load_json(concept_packs_path(self.base_dir), default={}) or {}
        self.year_slate_plan = safe_load_json(year_slate_plan_path(self.base_dir), default={}) or {}
        self.keyword_motif_bank = safe_load_json(keyword_motif_bank_path(self.base_dir), default={}) or {}
        self.franchise_bibles_payload = safe_load_json(franchise_bibles_path(self.base_dir), default={}) or {}
        if not isinstance(self.world_policy, dict):
            self.world_policy = {}
        if not isinstance(self.concept_packs_payload, dict):
            self.concept_packs_payload = {}
        if not isinstance(self.year_slate_plan, dict):
            self.year_slate_plan = {}
        if not isinstance(self.keyword_motif_bank, dict):
            self.keyword_motif_bank = {}
        if not isinstance(self.franchise_bibles_payload, dict):
            self.franchise_bibles_payload = {}
        raw_packs = self.concept_packs_payload.get("packs", [])
        self.concept_packs = list(raw_packs) if isinstance(raw_packs, list) else []
        self.concept_packs_index = index_concept_packs(self.concept_packs_payload)
        self.year_slate_index = index_year_slate_plan(self.year_slate_plan)
        self.franchise_bibles_index = index_franchise_bibles(self.franchise_bibles_payload)
        self.keywords = enrich_keyword_dataframe(self.keywords, self.keyword_motif_bank)
        self._memory_audit_snapshot(
            "world_load_core_entities_ready",
            [
                ("companies", self.companies, "entities"),
                ("keywords", self.keywords, "entities"),
                ("title_bank", self.title_bank, "entities"),
                ("character_bank", self.character_bank, "entities"),
            ],
            metadata={
                "company_rows": len(self.companies) if self.companies is not None else 0,
                "keyword_rows": len(self.keywords) if self.keywords is not None else 0,
                "world_policy_buckets": len(self.world_policy.get("year_buckets", [])) if isinstance(self.world_policy, dict) else 0,
                "concept_pack_count": len(self.concept_packs),
                "year_slate_count": len(self.year_slate_plan.get("slates", [])) if isinstance(self.year_slate_plan, dict) else 0,
                "keyword_motif_count": len(self.keyword_motif_bank.get("motifs", [])) if isinstance(self.keyword_motif_bank, dict) else 0,
            },
        )

        self._load_edge_graph()
        try:
            from memory_probe import graph_audit_targets
            edge_graph_specs = graph_audit_targets(self.graph)
        except Exception:
            edge_graph_specs = []
        self._memory_audit_snapshot(
            "world_load_edge_graph_ready",
            edge_graph_specs,
            metadata={"edge_rows": int(getattr(self.graph, "history_row_count", 0) or 0)},
        )
        self._build_edge_adjacency()
        self._memory_audit_snapshot(
            "world_load_edge_adjacency_ready",
            edge_graph_specs,
            metadata={
                "friend_nodes": len(self._friend_adj_all),
                "rival_nodes": len(self._rival_adj_all),
            },
        )
        self._load_latents(edir)
        self._memory_audit_snapshot(
            "world_load_latents_ready",
            [
                ("person_latent", self.person_latent, "latent_state", "Person latent dictionary loaded from JSON", "latent"),
                ("company_latent", self.company_latent, "latent_state", "Company latent dictionary loaded from JSON", "latent"),
            ],
            metadata={
                "person_latent_rows": len(self.person_latent),
                "company_latent_rows": len(self.company_latent),
            },
        )
        self._load_company_financial_profiles(edir)
        self._build_lookup_dicts()
        self._init_director_quality_offsets()
        self._load_communities()
        self._assign_agencies()
        self._assign_company_cliques()
        self._setup_franchises()
        self._apply_franchise_bibles()
        self._build_person_sim_cache()
        self._memory_audit_snapshot(
            "world_load_person_sim_cache_ready",
            [
                ("person_latent", self.person_latent, "latent_state", "Person latent dictionary loaded from JSON", "latent"),
                ("person_sim_cache", self._person_sim_cache, "latent_cache", "Similarity cache derived from latent/person tables", "latent"),
                ("latent_pid_to_idx", self._latent_pid_to_idx, "latent_dense", "Person id to dense latent index map", "latent"),
                ("latent_csv_normed", self._latent_csv_normed, "latent_dense", "Dense creative-style matrix", "latent"),
                ("latent_bbp_normed", self._latent_bbp_normed, "latent_dense", "Dense budget-band preference matrix", "latent"),
                ("latent_public_reputation", self._latent_public_reputation, "latent_dense", "Dense public reputation vector", "latent"),
                ("latent_controversy", self._latent_controversy, "latent_dense", "Dense controversy vector", "latent"),
                ("latent_volatility", self._latent_volatility, "latent_dense", "Dense volatility vector", "latent"),
                ("latent_collab", self._latent_collab, "latent_dense", "Dense collaboration-style labels", "latent"),
                ("latent_avoid_genres", self._latent_avoid_genres, "latent_dense", "Sparse avoid-genres map", "latent"),
                ("latent_genre_bits", self._latent_genre_bits, "latent_dense", "Dense genre bitset array", "latent"),
                ("latent_style_bits", self._latent_style_bits, "latent_dense", "Dense style bitset array", "latent"),
                ("latent_risk", self._latent_risk, "latent_dense", "Dense risk vector", "latent"),
                ("latent_ambition", self._latent_ambition, "latent_dense", "Dense ambition vector", "latent"),
                ("latent_collab_code", self._latent_collab_code, "latent_dense", "Dense collaboration code vector", "latent"),
            ],
            metadata={"sim_cache_rows": len(self._person_sim_cache)},
        )
        self._build_pc_scoring_arrays()   # On-demand P-C/C-C scoring
        self._memory_audit_snapshot(
            "world_load_pc_scoring_ready",
            [
                ("pc_p_idx", getattr(self, "_pc_p_idx", None), "pc_scoring", "Person id to scoring-array index map", "latent_pc"),
                ("pc_c_idx", getattr(self, "_pc_c_idx", None), "pc_scoring", "Company id to scoring-array index map", "latent_pc"),
                ("pc_p_risk", getattr(self, "_pc_p_risk", None), "pc_scoring", "Dense person risk vector for company matching", "latent_pc"),
                ("pc_p_controversy", getattr(self, "_pc_p_controversy", None), "pc_scoring", "Dense person controversy vector for company matching", "latent_pc"),
                ("pc_p_budget", getattr(self, "_pc_p_budget", None), "pc_scoring", "Dense person budget preference matrix", "latent_pc"),
                ("pc_p_genre", getattr(self, "_pc_p_genre", None), "pc_scoring", "Dense person genre projection matrix", "latent_pc"),
                ("pc_c_risk", getattr(self, "_pc_c_risk", None), "pc_scoring", "Dense company risk vector", "latent_pc"),
                ("pc_c_prestige", getattr(self, "_pc_c_prestige", None), "pc_scoring", "Dense company prestige vector", "latent_pc"),
                ("pc_c_genre", getattr(self, "_pc_c_genre", None), "pc_scoring", "Dense company genre matrix", "latent_pc"),
                ("pc_c_budget", getattr(self, "_pc_c_budget", None), "pc_scoring", "Dense company budget matrix", "latent_pc"),
                ("pc_c_controversy_tol", getattr(self, "_pc_c_controversy_tol", None), "pc_scoring", "Dense company controversy tolerance vector", "latent_pc"),
            ],
        )
        self._prewarm_year_cache()
        self._memory_audit_snapshot(
            "world_load_year_cache_ready",
            [
                ("year_cache", self._year_cache, "year_cache", "Boolean year masks over actor pool", "year_cache"),
                ("director_recent_outcomes", self.director_recent_outcomes, "runtime_state", "Director momentum history", "runtime"),
                ("company_recent_outcomes", self.company_recent_outcomes, "runtime_state", "Company momentum history", "runtime"),
            ],
            metadata={"year_cache_entries": len(self._year_cache)},
        )

        print(
            f"World loaded: {len(self.persons)} persons, {len(self.companies)} companies, "
            f"{len(self.keywords)} keywords, {len(self.title_bank)} titles, {len(self.character_bank)} characters, "
            f"{len(self.world_policy.get('year_buckets', [])) if isinstance(self.world_policy, dict) else 0} policy buckets, "
            f"{len(self.concept_packs)} concept packs"
        )

    # ------------------------------------------------------------------
    # Load helpers
    # ------------------------------------------------------------------

    def _load_csv_with_id(self, path: Path, id_col: str) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        if id_col not in df.columns:
            df[id_col] = np.arange(1, len(df) + 1, dtype=int)
        else:
            df[id_col] = pd.to_numeric(df[id_col], errors="coerce").fillna(0).astype(int)
            if (df[id_col] <= 0).any() or df[id_col].duplicated().any():
                print(f"  WARNING: invalid {id_col} values in {path.name} -- reassigning sequential IDs")
                df[id_col] = np.arange(1, len(df) + 1, dtype=int)
        return df

    def _load_person_roles(self, edir: Path) -> pd.DataFrame:
        path = edir / "person_roles.csv"
        if path.exists():
            df = pd.read_csv(path)
            if "person_id" in df.columns:
                df["person_id"] = pd.to_numeric(df["person_id"], errors="coerce").fillna(0).astype(int)
            return df

        assert self.persons is not None
        pids = self.persons["person_id"].astype(int).tolist()
        roles_raw = self.persons["roles"].fillna("actor").astype(str).str.lower().tolist() if "roles" in self.persons.columns else ["actor"] * len(self.persons)
        rows = []
        for pid, raw in zip(pids, roles_raw):
            rows.append({"person_id": pid, "role_type": "actor"})
            if "director" in raw:
                rows.append({"person_id": pid, "role_type": "director"})
            for crew_role in CREW_DEPARTMENTS:
                if crew_role.lower() in raw:
                    rows.append({"person_id": pid, "role_type": crew_role})
        return pd.DataFrame(rows)

    def _build_person_role_views(self):
        assert self.persons is not None and self.person_roles is not None
        role_to_ids: Dict[str, set] = {}
        for role, g in self.person_roles.groupby("role_type"):
            role_to_ids[str(role)] = set(g["person_id"].astype(int).tolist())

        actor_ids = role_to_ids.get("actor", set())
        director_ids = role_to_ids.get("director", set())
        self.actors = self.persons[self.persons["person_id"].isin(actor_ids)].copy()
        self.directors = self.persons[self.persons["person_id"].isin(director_ids)].copy()

        self.crew_pools = {}
        for crew_role in CREW_DEPARTMENTS:
            ids = role_to_ids.get(crew_role, set())
            self.crew_pools[crew_role] = self.persons[self.persons["person_id"].isin(ids)].copy() if ids else pd.DataFrame(columns=self.persons.columns)

        self.writers = self.crew_pools.get("writer", pd.DataFrame())
        self.cinematographers = self.crew_pools.get("cinematographer", pd.DataFrame())
        self.editors = self.crew_pools.get("editor", pd.DataFrame())
        self.composers = self.crew_pools.get("composer", pd.DataFrame())

        crew_counts = ", ".join(f"{len(self.crew_pools[r])} {r}s" for r in CREW_DEPARTMENTS if len(self.crew_pools[r]) > 0)
        print(f"Loaded {len(self.persons)} persons ({len(self.actors)} actors, {len(self.directors)} directors, {crew_counts or 'no crew pools'})")
        print(f"Loaded {len(self.companies) if self.companies is not None else 0} companies")
        print(f"Loaded {len(self.keywords) if self.keywords is not None else 0} keywords")
        print(f"Loaded {len(self.title_bank) if self.title_bank is not None else 0} curated titles")
        print(f"Loaded {len(self.character_bank) if self.character_bank is not None else 0} character names")

    def _iter_edge_rows_from_arrow(self, path: Path):
        import pyarrow.ipc as ipc

        reader = ipc.open_file(str(path))
        for batch_idx in range(reader.num_record_batches):
            batch = reader.get_batch(batch_idx)
            for row in batch.to_pylist():
                yield row

    def _iter_edge_rows_from_csv(self, path: Path):
        try:
            import pyarrow.csv as pa_csv

            reader = pa_csv.open_csv(
                str(path),
                read_options=pa_csv.ReadOptions(block_size=1 << 20),
            )
            for batch in reader:
                for row in batch.to_pylist():
                    yield row
            return
        except Exception:
            pass

        import csv

        with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
            for row in csv.DictReader(fh):
                yield row

    def _normalize_loaded_edge(self, row: dict[str, Any]) -> dict[str, Any]:
        def _norm_int(value: Any, default: int = 0) -> int:
            try:
                num = int(float(value))
            except Exception:
                return int(default)
            return num

        def _norm_year(value: Any):
            if value in (None, "", "nan"):
                return None
            try:
                return int(float(value))
            except Exception:
                return None

        def _norm_bool(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            if value is None:
                return False
            if isinstance(value, (int, float)):
                return bool(value)
            text = str(value).strip().lower()
            return text in {"1", "true", "yes", "y"}

        raw_weight = row.get("raw_weight")
        if raw_weight not in (None, "", "nan"):
            try:
                raw_weight = float(raw_weight)
            except Exception:
                raw_weight = None
        else:
            raw_weight = None

        weight = pd.to_numeric(row.get("weight"), errors="coerce")
        if pd.isna(weight):
            weight = 0.0

        return {
            "src_id": _norm_int(row.get("src_id")),
            "dst_id": _norm_int(row.get("dst_id")),
            "src_name": str(row.get("src_name", "") or ""),
            "dst_name": str(row.get("dst_name", "") or ""),
            "src_type": str(row.get("src_type", "person") or "person"),
            "dst_type": str(row.get("dst_type", "person") or "person"),
            "edge_type": str(row.get("edge_type", "") or ""),
            "sign": str(row.get("sign", "+") or "+"),
            "weight": float(weight),
            "raw_weight": raw_weight,
            "reason": str(row.get("reason", "") or ""),
            "source_batch": row.get("source_batch"),
            "source_kind": str(row.get("source_kind", "") or ""),
            "valid_from": _norm_year(row.get("valid_from")),
            "valid_to": _norm_year(row.get("valid_to")),
            "community_id": _norm_year(row.get("community_id")),
            "_scd2_retired": _norm_bool(row.get("_scd2_retired", False)),
            "_overridden_by_rivalry": _norm_bool(row.get("_overridden_by_rivalry", False)),
        }

    def _graph_name_resolver(self, entity_id: int, entity_type: str) -> str:
        entity_id = int(entity_id)
        if entity_type == "company":
            if self.companies is not None and {"company_id", "name"}.issubset(self.companies.columns):
                matches = self.companies.loc[self.companies["company_id"].astype(int) == entity_id, "name"]
                if len(matches) > 0:
                    return str(matches.iloc[0])
            return ""
        if self.persons is not None and {"person_id", "name"}.issubset(self.persons.columns):
            matches = self.persons.loc[self.persons["person_id"].astype(int) == entity_id, "name"]
            if len(matches) > 0:
                return str(matches.iloc[0])
        return ""

    def _load_edge_graph(self):
        arrow_path = self.base_dir / "graph" / "edge_graph.arrow"
        csv_path = self.base_dir / "graph" / "edge_graph.csv"

        # Try Arrow IPC first (5× faster, ~3× smaller), fall back to CSV
        source = None
        row_iter = None
        manifest_path = self.base_dir / "graph" / "runtime_manifest.json"
        if not manifest_path.exists():
            if arrow_path.exists():
                try:
                    row_iter = self._iter_edge_rows_from_arrow(arrow_path)
                    source = "arrow"
                except Exception as exc:
                    print(f"  WARNING: failed to stream edge_graph.arrow ({exc}), falling back to CSV")
                    row_iter = None
            if row_iter is None and csv_path.exists():
                row_iter = self._iter_edge_rows_from_csv(csv_path)
                source = "csv"

        if row_iter is None and not manifest_path.exists():
            self.graph = None
            self.edge_graph = None
            self.affinity_index = self._empty_affinity_index()
            self.edge_weights = {}
            self._friend_adj_all = {}
            self._rival_adj_all = {}
            return

        graph = GraphRuntime.load_or_compile(
            self.base_dir,
            row_iter=row_iter,
            source_label=source,
            normalize_row=self._normalize_loaded_edge,
            name_resolver=self._graph_name_resolver,
        )
        self.graph = graph
        self.edge_graph = graph
        self.affinity_index = graph.affinity_index
        self.edge_weights = graph.edge_weights
        self._friend_adj_all = graph.friend_adjacency
        self._rival_adj_all = graph.rival_adjacency
        print(
            f"  Loaded graph runtime: {graph.history_row_count:,} history rows, "
            f"{sum(int(meta.get('count', 0)) for meta in graph.manifest.get('hot_types', {}).values()):,} hot active rows, "
            f"{int(graph.manifest.get('cold_cp_count', 0)) + int(graph.manifest.get('cold_cc_count', 0)):,} cold active rows"
        )

    def _load_latents(self, edir: Path):
        self.person_latent = self._load_person_latent_json(edir / "persons_latent.json")
        self.company_latent = self._load_company_latent_json(edir / "companies_latent.json")

    def _load_person_latent_json(self, path: Path) -> Dict[int, dict]:
        if not path.exists():
            warnings.warn(
                "persons_latent.json not found. Person latent variables will use fallback values.",
                UserWarning,
                stacklevel=2,
            )
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        out: Dict[int, dict] = {}
        for lv in data:
            if not isinstance(lv, dict):
                continue
            try:
                pid = int(lv.get("person_id", 0))
            except Exception:
                continue
            if pid <= 0:
                continue
            item = dict(lv)
            item["person_id"] = pid
            for k, d in (("public_reputation", 0.5), ("controversy_score", 0.15), ("risk_tolerance", 0.5), ("artistic_ambition", 0.5), ("volatility", 0.4)):
                item[k] = _clip01(item.get(k, d), d)
            out[pid] = item
        print(f"Loaded {len(out)} person latent vars")
        return out

    def _load_company_latent_json(self, path: Path) -> Dict[int, dict]:
        if not path.exists():
            warnings.warn(
                "companies_latent.json not found. Company latent variables will use fallback values.",
                UserWarning,
                stacklevel=2,
            )
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        out: Dict[int, dict] = {}
        for lv in data:
            if not isinstance(lv, dict):
                continue
            try:
                cid = int(lv.get("company_id", 0))
            except Exception:
                continue
            if cid <= 0:
                continue
            item = dict(lv)
            item["company_id"] = cid
            for k, d in (("risk_appetite", 0.5), ("prestige_score", 0.5), ("controversy_tolerance", 0.5), ("market_trend_sensitivity", 0.5)):
                item[k] = _clip01(item.get(k, d), d)
            out[cid] = item
        print(f"Loaded {len(out)} company latent vars")
        return out

    def _load_company_financial_profiles(self, edir: Path):
        path = edir / "company_financial_profile.csv"
        if not path.exists():
            self.company_financial_profile = {}
            return
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"WARNING: could not load company_financial_profile.csv: {exc}")
            self.company_financial_profile = {}
            return

        if "company_id" not in df.columns:
            self.company_financial_profile = {}
            return
        df["company_id"] = pd.to_numeric(df["company_id"], errors="coerce").fillna(0).astype(int)
        self.company_financial_profile = {
            int(row["company_id"]): {k: row[k] for k in df.columns if k != "company_id"}
            for _, row in df.iterrows()
            if int(row["company_id"]) > 0
        }
        print(f"Loaded {len(self.company_financial_profile)} company financial profiles")

    def _build_lookup_dicts(self):
        assert self.persons is not None and self.companies is not None
        self.person_pop_weight = dict(zip(self.persons["person_id"].astype(int), self.persons["pop_weight"].astype(float)))
        self.company_pop_weight = dict(zip(self.companies["company_id"].astype(int), self.companies["pop_weight"].astype(float)))
        self._company_tier_map = dict(zip(self.companies["company_id"].astype(int), self.companies.get("tier", pd.Series(["Mid-Budget"] * len(self.companies))).astype(str)))

    def _load_communities(self):
        arrow_path = self.base_dir / "graph" / "communities.arrow"
        csv_path = self.base_dir / "graph" / "communities.csv"
        all_pids = set(self.persons["person_id"].astype(int).tolist()) if self.persons is not None else set()
        communities: Dict[int, int] = {}

        # Try Arrow IPC first, fall back to CSV
        df = None
        if arrow_path.exists():
            try:
                import pyarrow.ipc as ipc
                reader = ipc.open_file(str(arrow_path))
                df = reader.read_pandas()
            except Exception:
                df = None
        if df is None and csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
            except Exception as exc:
                print(f"WARNING: failed to load communities: {exc}")

        if df is not None and {"person_id", "community"}.issubset(df.columns):
            df = df[["person_id", "community"]].copy()
            df["person_id"] = pd.to_numeric(df["person_id"], errors="coerce").fillna(0).astype(int)
            df["community"] = pd.to_numeric(df["community"], errors="coerce").fillna(0).astype(int)
            df = df[(df["person_id"] > 0) & (df["community"] > 0)]
            communities = dict(zip(df["person_id"], df["community"]))

        if not communities:
            n = max(8, min(64, int(np.sqrt(max(1, len(all_pids))))))
            for pid in all_pids:
                h = int(hashlib.md5(f"community|{pid}".encode("utf-8")).hexdigest(), 16)
                communities[int(pid)] = int(h % n) + 1
        else:
            existing = sorted(set(communities.values()))
            for pid in sorted(all_pids):
                if pid not in communities:
                    h = int(hashlib.md5(f"community|{pid}".encode("utf-8")).hexdigest(), 16)
                    communities[pid] = existing[h % len(existing)]

        self.communities = communities
        print(f"Loaded {len(self.communities)} community assignments")

    # ------------------------------------------------------------------
    # Structural assignments
    # ------------------------------------------------------------------

    def _assign_agencies(self):
        assert self.actors is not None
        self.person_agency = {}
        actor_ids = set(self.actors["person_id"].astype(int).tolist())

        entities_dir = self.base_dir / "entities"
        agencies_path = entities_dir / "agencies.json"
        persons_json_path = entities_dir / "persons.json"

        agency_name_to_id: Dict[str, int] = {}
        if agencies_path.exists():
            try:
                agencies = json.loads(agencies_path.read_text(encoding="utf-8"))
                if isinstance(agencies, list):
                    for i, a in enumerate(agencies, start=1):
                        if not isinstance(a, dict):
                            continue
                        name = str(a.get("name", "")).strip().lower()
                        if name and name not in agency_name_to_id:
                            agency_name_to_id[name] = i
            except Exception as exc:
                print(f"WARNING: failed to parse agencies.json: {exc}")

        assigned_from_generated = 0
        parse_generated_agencies = os.getenv("DATA_SYS_PARSE_PERSON_AGENCIES_JSON", "").strip().lower() in {"1", "true", "yes", "y"}
        if persons_json_path.exists() and parse_generated_agencies:
            try:
                persons_json = json.loads(persons_json_path.read_text(encoding="utf-8"))
                if isinstance(persons_json, list):
                    for row in persons_json:
                        if not isinstance(row, dict):
                            continue
                        try:
                            pid = int(row.get("person_id", 0) or 0)
                        except Exception:
                            pid = 0
                        if pid <= 0 or pid not in actor_ids:
                            continue
                        aid = None
                        raw_aid = row.get("agency_id")
                        if raw_aid is not None:
                            try:
                                q = int(raw_aid)
                                if q > 0:
                                    aid = q
                            except Exception:
                                pass
                        if aid is None:
                            aname = str(row.get("agency", "")).strip().lower()
                            if aname:
                                aid = agency_name_to_id.get(aname)
                                if aid is None and not agency_name_to_id:
                                    agency_name_to_id[aname] = len(agency_name_to_id) + 1
                                    aid = agency_name_to_id[aname]
                        if aid is not None and int(aid) > 0:
                            self.person_agency[pid] = int(aid)
                            assigned_from_generated += 1
            except Exception as exc:
                print(f"WARNING: failed to parse persons.json agency assignments: {exc}")
        elif persons_json_path.exists():
            size_mb = persons_json_path.stat().st_size / (1024 * 1024)
            print(
                f"Skipped optional persons.json agency parse ({size_mb:.1f} MB); "
                "using deterministic actor agency assignment",
                flush=True,
            )

        n_agencies = max(int(N_AGENCIES), len(agency_name_to_id), max(self.person_agency.values(), default=0), 1)
        missing = [pid for pid in sorted(actor_ids) if pid not in self.person_agency]
        if missing:
            assert self.actors is not None
            print(f"Assigning deterministic agencies for {len(missing)} actors...", flush=True)
            actor_meta = {
                int(row.person_id): (
                    str(getattr(row, "style_tags", "") or ""),
                    str(getattr(row, "nationality", "") or ""),
                )
                for row in self.actors[["person_id", "style_tags", "nationality"]].itertuples(index=False)
            }
            for idx, pid in enumerate(missing, start=1):
                tags, nat = actor_meta.get(int(pid), ("", ""))
                key = f"agency|{pid}|{tags}|{nat}"
                bucket = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % n_agencies
                self.person_agency[pid] = bucket + 1
                if idx % 50000 == 0:
                    print(f"  Assigned deterministic agencies: {idx}/{len(missing)}", flush=True)

        sizes = sorted(Counter(self.person_agency.values()).values(), reverse=True)[:10]
        print(f"Assigned {len(self.person_agency)} actors to {n_agencies} agencies (generated={assigned_from_generated}, sizes={sizes})")

    def _assign_company_cliques(self):
        assert self.companies is not None
        self.company_clique = {}
        entities_dir = self.base_dir / "entities"
        cliques_path = entities_dir / "cliques.json"
        companies_json_path = entities_dir / "companies.json"

        clique_name_to_id: Dict[str, int] = {}
        clique_ids: List[int] = []
        if cliques_path.exists():
            try:
                cliques = json.loads(cliques_path.read_text(encoding="utf-8"))
                if isinstance(cliques, list):
                    for i, c in enumerate(cliques, start=1):
                        if not isinstance(c, dict):
                            continue
                        try:
                            cid = int(c.get("clique_id", i))
                        except Exception:
                            cid = i
                        cid = max(cid, 1)
                        name = str(c.get("name", "")).strip().lower()
                        if name:
                            clique_name_to_id[name] = cid
                        clique_ids.append(cid)
            except Exception as exc:
                print(f"WARNING: failed to parse cliques.json: {exc}")

        n_cliques = max(max(clique_ids) if clique_ids else 0, int(N_COMPANY_CLIQUES), 1)
        assigned_from_generated = 0
        if companies_json_path.exists():
            try:
                companies_json = json.loads(companies_json_path.read_text(encoding="utf-8"))
                if isinstance(companies_json, list):
                    for row in companies_json:
                        if not isinstance(row, dict):
                            continue
                        try:
                            company_id = int(row.get("company_id", 0) or 0)
                        except Exception:
                            continue
                        if company_id <= 0:
                            continue
                        target_id = None
                        raw_id = row.get("clique_id")
                        raw_name = row.get("clique")
                        if raw_id is not None:
                            try:
                                q = int(raw_id)
                                if q > 0:
                                    target_id = q
                            except Exception:
                                pass
                        if target_id is None and raw_name is not None:
                            if isinstance(raw_name, (int, float)):
                                try:
                                    q = int(raw_name)
                                    if q > 0:
                                        target_id = q
                                except Exception:
                                    pass
                            else:
                                target_id = clique_name_to_id.get(str(raw_name).strip().lower())
                        if target_id is not None and target_id > 0:
                            self.company_clique[company_id] = int(target_id)
                            assigned_from_generated += 1
            except Exception as exc:
                print(f"WARNING: failed to parse companies.json clique assignments: {exc}")

        tier_order = {"Global": 0, "Major": 1, "Mid-Budget": 2, "Indie": 3, "Micro": 4}
        for row in self.companies.itertuples(index=False):
            cid = int(getattr(row, "company_id"))
            if cid in self.company_clique:
                continue
            tier = str(getattr(row, "tier", "Mid-Budget"))
            specialty = str(getattr(row, "specialty_genres", ""))
            tier_idx = tier_order.get(tier, 2)
            first_genre = specialty.split(";")[0].strip() if specialty else "Drama"
            try:
                genre_idx = GENRES.index(first_genre)
            except ValueError:
                genre_idx = int(hashlib.md5(first_genre.encode("utf-8")).hexdigest(), 16) % max(1, len(GENRES))
            clique = (tier_idx * 7 + genre_idx) % n_cliques + 1
            self.company_clique[cid] = int(clique)

        sizes = sorted(Counter(self.company_clique.values()).values(), reverse=True)[:10]
        print(f"Assigned {len(self.company_clique)} companies to {n_cliques} cliques (generated={assigned_from_generated}, sizes={sizes})")

    # ------------------------------------------------------------------
    # Pop weights / careers / franchises
    # ------------------------------------------------------------------

    def _assign_career_timelines(self):
        assert self.persons is not None
        n = len(self.persons)
        stage = self.persons.get("career_stage", pd.Series(["prime"] * n)).fillna("prime").astype(str)

        yr_lo, yr_hi = year_bounds_from_env(1950, 2025)

        debut, peak_s, peak_e, retire, yearly_max = [], [], [], [], []
        for cs in stage.tolist():
            cs = str(cs).lower()
            # B1-FIX: always use active pipeline year bounds
            # when YEAR_RANGE is None). The old code had a separate branch
            # for YEAR_RANGE=None with hardcoded decades that was never
            # reached by the intended YEAR_RANGE-aware logic.
            if cs == "legend":
                d = self.rng.randint(yr_lo - 35, yr_lo - 15)
                ps = d + self.rng.randint(5, 10)
                pe = ps + self.rng.randint(5, 8)
                r = max(yr_hi + self.rng.randint(0, 8), pe + 2)
                ym = self.rng.randint(5, 9)
            elif cs == "veteran":
                d = self.rng.randint(yr_lo - 25, yr_lo - 8)
                ps = d + self.rng.randint(5, 10)
                pe = ps + self.rng.randint(4, 7)
                r = max(yr_hi + self.rng.randint(0, 10), pe + 3)
                ym = self.rng.randint(3, 6)
            elif cs == "prime":
                d = self.rng.randint(yr_lo - 15, yr_lo - 2)
                ps = d + self.rng.randint(3, 7)
                pe = ps + self.rng.randint(3, 6)
                r = max(yr_hi + self.rng.randint(5, 15), pe + 5)
                ym = self.rng.randint(4, 7)
            elif cs == "rising":
                d = self.rng.randint(yr_lo - 5, yr_hi - 1)
                ps = d + self.rng.randint(2, 5)
                pe = ps + self.rng.randint(3, 6)
                r = 2100
                ym = self.rng.randint(2, 5)
            else:
                if self.rng.random_sample() < 0.7:
                    d = self.rng.randint(yr_lo - 30, yr_lo - 10)
                    ps = d + self.rng.randint(5, 10)
                    pe = ps + self.rng.randint(3, 6)
                    r = self.rng.randint(yr_lo - 3, yr_lo + 1)
                else:
                    d = self.rng.randint(yr_lo - 40, yr_lo - 20)
                    ps = d + self.rng.randint(5, 10)
                    pe = ps + self.rng.randint(3, 6)
                    r = pe + self.rng.randint(0, 5)
                ym = self.rng.randint(1, 2)

            debut.append(int(d))
            peak_s.append(int(ps))
            peak_e.append(int(pe))
            retire.append(int(min(r, 2100)))
            yearly_max.append(int(ym))

        generated = {
            "debut_year": debut,
            "peak_start": peak_s,
            "peak_end": peak_e,
            "retirement_year": retire,
            "yearly_max": yearly_max,
        }
        for column, values in generated.items():
            generated_series = pd.Series(values, index=self.persons.index).astype(int)
            if column in self.persons.columns:
                existing = pd.to_numeric(self.persons[column], errors="coerce")
                self.persons[column] = existing.fillna(generated_series).astype(int)
            else:
                self.persons[column] = generated_series

    def _assign_pop_weights(self):
        assert self.persons is not None
        n = len(self.persons)
        if n <= 0:
            self.persons["pop_weight"] = []
            return

        def _gini(x: np.ndarray) -> float:
            arr = np.asarray(x, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return 0.0
            arr = np.clip(arr, 1e-12, None)
            arr = np.sort(arr)
            idx = np.arange(1, arr.size + 1, dtype=float)
            return float((2.0 * np.sum(idx * arr) - (arr.size + 1.0) * np.sum(arr)) / (arr.size * np.sum(arr)))

        raw = None
        if "pop_weight" in self.persons.columns:
            pw = pd.to_numeric(self.persons["pop_weight"], errors="coerce")
            if pw.notna().sum() > n * 0.6:
                raw = pw.fillna(float(pw.median()) if pw.notna().any() else 0.1).values.astype(float)
        if raw is None:
            raw = (self.rng.pareto(1.45, size=n) + 1.0).astype(float)

        stage_col = self.persons.get("career_stage", pd.Series(["prime"] * n)).fillna("prime").astype(str).str.lower().values
        stage_mult_map = {"legend": 2.6, "prime": 1.55, "veteran": 1.25, "rising": 0.85, "retired": 0.35}
        stage_mult = np.array([stage_mult_map.get(s, 1.0) for s in stage_col], dtype=float)

        signal = np.clip(raw * stage_mult, 1e-12, None)
        signal = signal * (self.rng.pareto(1.20, size=n) + 1.0)
        order = np.argsort(-signal)
        stage_norm = stage_mult / max(float(stage_mult.max()), 1e-12)

        # A1-FIX: scale Gini target with dataset size so top-k concentration
        # stays realistic. log10 scaling: N=500→0.62, N=5000→0.55, N=24000→0.50
        target_mid = float(np.clip(0.72 - 0.025 * np.log10(max(n, 100)), 0.45, 0.65))
        target_lo = target_mid - 0.04
        target_hi = target_mid + 0.04

        def _weights_for_alpha(alpha: float) -> np.ndarray:
            ranks = np.arange(1, n + 1, dtype=float)
            base = 1.0 / np.power(ranks, alpha)
            w = np.zeros(n, dtype=float)
            w[order] = base
            w = w * (0.70 + 0.30 * stage_norm)
            w = w / max(float(w.max()), 1e-12)
            return np.clip(1e-6 + (1.0 - 1e-6) * w, 1e-6, 1.0)

        lo_a, hi_a = 0.15, 3.50
        best_w = _weights_for_alpha(1.0)
        best_gap = abs(_gini(best_w) - target_mid)
        # A8-FIX: add tolerance-based early exit instead of always running 36 iterations
        for _ in range(36):
            mid = (lo_a + hi_a) / 2.0
            cand = _weights_for_alpha(mid)
            g = _gini(cand)
            gap = abs(g - target_mid)
            if gap < best_gap:
                best_gap = gap
                best_w = cand
            if target_lo <= g <= target_hi:
                best_w = cand
                break
            elif g < target_lo:
                lo_a = mid
            else:
                hi_a = mid
            # A8-FIX: exit early if converged (gap < 0.001)
            if best_gap < 0.001:
                break

        self.persons["pop_weight"] = best_w
        print(f"  pop_weight: calibrated Gini={_gini(best_w):.3f} (target {target_lo:.2f}-{target_hi:.2f})")

    def _assign_company_pop_weights(self):
        assert self.companies is not None
        n = len(self.companies)
        if n <= 0:
            self.companies["pop_weight"] = []
            return

        # A2-FIX: sort companies by tier before weight assignment so that
        # Global/Major companies always outweigh Indie/Micro companies.
        # Within each tier, jitter provides natural variation.
        _TIER_ORDER = {"Global": 0, "Major": 1, "A-List": 1, "A": 1,
                       "Mid-Budget": 2, "Mid": 2, "Indie": 3,
                       "Micro": 4, "Micro-Budget": 4}
        # Weight bands per tier: (low, high)
        _TIER_BANDS = {
            0: (0.82, 0.99),   # Global
            1: (0.55, 0.84),   # Major / A-List
            2: (0.20, 0.54),   # Mid-Budget
            3: (0.08, 0.22),   # Indie
            4: (0.01, 0.09),   # Micro
        }

        tiers = self.companies["tier"].astype(str).tolist() if "tier" in self.companies.columns else ["Mid"] * n
        tier_codes = np.array([_TIER_ORDER.get(t, 3) for t in tiers], dtype=int)

        weights = np.zeros(n, dtype=float)
        for i in range(n):
            lo, hi = _TIER_BANDS.get(int(tier_codes[i]), (0.08, 0.22))
            weights[i] = self.rng.uniform(lo, hi)

        self.companies["pop_weight"] = weights

    def _init_director_quality_offsets(self):
        assert self.directors is not None
        self.director_quality_offset = {}
        for did in self.directors["person_id"].astype(int).tolist():
            lv = get_person_latent(self, did)
            rep = _safe_float(lv.get("public_reputation"), 0.4)
            base = (rep - 0.5) * 0.9
            noise = float(self.rng.normal(0, 0.55))
            self.director_quality_offset[did] = float(np.clip(base + noise, -1.6, 1.6))

    def _setup_franchises(self):
        cfg = FRANCHISE_CONFIG
        total_movies = ENTITY_COUNTS.get("movies")
        if total_movies in (None, "", 0):
            total_movies = len(self.title_bank) if self.title_bank is not None and len(self.title_bank) > 0 else ENTITY_COUNTS.get("title_bank")
        try:
            total_movies = max(1, int(total_movies))
        except Exception:
            total_movies = max(1, int(ENTITY_COUNTS.get("title_bank", 1) or 1))
        scale = total_movies / 7500.0
        base_lo, base_hi = cfg["count_range"]
        lo = max(base_lo, int(base_lo * scale))
        hi = max(base_hi, int(base_hi * scale))
        n_franchises = self.rng.randint(lo, hi + 1)
        target_frac = self.rng.uniform(*cfg["target_pct_of_total"])
        target_franchise_movies = int(total_movies * target_frac)

        # A3-FIX: assign each franchise a start_year in the configured range
        # and pre-space installments 2-3 years apart so they cluster temporally.
        yr_lo, yr_hi = year_bounds_from_env(1950, 2025)
        span = max(1, yr_hi - yr_lo)

        self.franchises = []
        remaining = target_franchise_movies
        for i in range(n_franchises):
            if remaining <= 0:
                break
            n_movies = self.rng.randint(*cfg["movies_per_franchise"])
            n_movies = min(n_movies, remaining)

            # A3-FIX: franchise start_year leaves room for installments
            max_start = yr_hi - max(1, n_movies - 1) * 2
            start_year = int(self.rng.randint(yr_lo, max(yr_lo + 1, max_start + 1)))
            # Pre-compute installment years with 2-3 year gaps
            inst_years = [start_year]
            for j in range(1, n_movies):
                gap = self.rng.randint(2, 4)  # 2-3 years between installments
                next_yr = min(inst_years[-1] + gap, yr_hi)
                inst_years.append(next_yr)

            self.franchises.append(
                {
                    "franchise_id": i + 1,
                    "name": f"Franchise_{i+1}",
                    "n_movies": int(n_movies),
                    "genre": self.py_rng.choice(GENRES),
                    "tier": self.py_rng.choice(PRODUCTION_TIERS[:3]),
                    "movies_generated": 0,
                    "cast_pool": [],
                    "director_id": None,
                    "company_ids": [],
                    "installment_years": inst_years,  # A3: pre-assigned years
                }
            )
            remaining -= n_movies

        total_franchise_movies = sum(f["n_movies"] for f in self.franchises)

        # A3-FIX: Build year→movie_id bucket for temporal locality.
        # Movies per year follows a roughly uniform distribution across the range.
        movies_per_year = max(1, total_movies // max(1, span))
        year_buckets: dict[int, list[int]] = {}
        for mid in range(1, total_movies + 1):
            approx_year = yr_lo + (mid - 1) // movies_per_year
            approx_year = min(approx_year, yr_hi)
            year_buckets.setdefault(approx_year, []).append(mid)

        # Assign franchise movies from appropriate year buckets
        self.movie_franchise_map = {}
        used_ids: set[int] = set()

        for franchise in self.franchises:
            for inst_idx in range(franchise["n_movies"]):
                target_year = franchise["installment_years"][inst_idx]
                # Search target year ±2 for available IDs
                candidates = []
                for dy in range(6):  # search radius: 0, ±1, ±2, ±3
                    for sign in ([0] if dy == 0 else [-1, 1]):
                        y = target_year + sign * dy
                        for mid in year_buckets.get(y, []):
                            if mid not in used_ids:
                                candidates.append(mid)
                    if candidates:
                        break
                if candidates:
                    chosen = candidates[int(self.rng.randint(0, len(candidates)))]
                    used_ids.add(chosen)
                    self.movie_franchise_map[chosen] = franchise

        print(f"Setup {len(self.franchises)} franchises ({total_franchise_movies} movies, {100.0 * total_franchise_movies / max(1, total_movies):.1f}%)")

    def _apply_franchise_bibles(self):
        payload = self.franchise_bibles_payload if isinstance(getattr(self, "franchise_bibles_payload", None), dict) else {}
        if not payload.get("bibles") and self.franchises:
            payload = build_default_franchise_bibles(self.franchises)
        self.franchise_bibles_payload = payload if isinstance(payload, dict) else {}
        self.franchise_bibles_index = index_franchise_bibles(self.franchise_bibles_payload)
        if not self.franchises:
            return
        for franchise in self.franchises:
            if not isinstance(franchise, dict):
                continue
            bible = self.franchise_bibles_index.get(safe_int(franchise.get("franchise_id"), 0))
            if bible:
                franchise["franchise_bible"] = dict(bible)

    # ------------------------------------------------------------------
    # Graph-derived lookup helpers
    # ------------------------------------------------------------------

    def _populate_edge_weights(self):
        if self.graph is not None:
            self.edge_weights = self.graph.edge_weights
            return
        aff = self.affinity_index or {}
        ew: Dict[tuple, float] = {}
        friendships = aff.get("friendships", {})
        for key, val in friendships.items():
            if isinstance(key, tuple) and len(key) == 2:
                ew[key] = float(val.get("weight", 0.5) if isinstance(val, dict) else (val or 0.5))
        rivalries = aff.get("rivalries", {})
        if isinstance(rivalries, dict):
            for key, val in rivalries.items():
                if isinstance(key, tuple) and len(key) == 2:
                    ew[key] = float(val.get("weight", 0.8) if isinstance(val, dict) else (val or 0.8))
        self.edge_weights = ew
        if ew:
            print(f"  Populated edge_weights: {len(ew)} entries")

    # ------------------------------------------------------------------
    # On-demand P-C / C-C scoring (replaces precomputed edges)
    # ------------------------------------------------------------------

    def _build_pc_scoring_arrays(self):
        """Pre-build vectorized arrays from latent variables for fast
        on-demand person-company affinity scoring.

        Called once at load time.  Replaces the 380M+ precomputed P-C edge
        list that would consume 130+ GB RAM at 450K persons.
        """
        if not self.person_latent or not self.company_latent:
            self._pc_ready = False
            return

        # Person arrays
        p_pids = sorted(self.person_latent.keys())
        self._pc_p_idx = {pid: i for i, pid in enumerate(p_pids)}
        n_p = len(p_pids)
        self._pc_p_risk = np.zeros(n_p, dtype=np.float32)
        self._pc_p_controversy = np.zeros(n_p, dtype=np.float32)
        self._pc_p_budget = np.zeros((n_p, 5), dtype=np.float32)
        self._pc_p_genre = np.zeros((n_p, 12), dtype=np.float32)
        person_genre_affinity: Dict[int, Any] = {}
        if self.persons is not None and {"person_id", "genre_affinity"}.issubset(self.persons.columns):
            for row in self.persons[["person_id", "genre_affinity"]].itertuples(index=False):
                person_genre_affinity[int(row.person_id)] = row.genre_affinity

        for i, pid in enumerate(p_pids):
            lv = self.person_latent[pid]
            self._pc_p_risk[i] = float(lv.get("risk_tolerance", 0.5))
            self._pc_p_controversy[i] = float(lv.get("controversy_score", 0.0))
            bp = lv.get("budget_band_pref", [0.5] * 5)
            if isinstance(bp, list):
                for j in range(min(5, len(bp))):
                    self._pc_p_budget[i, j] = float(bp[j])
            self._pc_p_genre[i] = project_genres_to_company_basis(person_genre_affinity.get(pid))

        # Company arrays
        c_cids = sorted(self.company_latent.keys())
        self._pc_c_idx = {cid: i for i, cid in enumerate(c_cids)}
        n_c = len(c_cids)
        self._pc_c_risk = np.zeros(n_c, dtype=np.float32)
        self._pc_c_prestige = np.zeros(n_c, dtype=np.float32)
        self._pc_c_controversy_tol = np.zeros(n_c, dtype=np.float32)
        self._pc_c_budget = np.zeros((n_c, 5), dtype=np.float32)
        self._pc_c_genre = np.zeros((n_c, 12), dtype=np.float32)

        for i, cid in enumerate(c_cids):
            lv = self.company_latent[cid]
            self._pc_c_risk[i] = float(lv.get("risk_appetite", 0.5))
            self._pc_c_prestige[i] = float(lv.get("prestige_score", 0.5))
            self._pc_c_controversy_tol[i] = float(lv.get("controversy_tolerance", 0.5))
            bf = lv.get("budget_tier_focus", [0.2] * 5)
            if isinstance(bf, list):
                for j in range(min(5, len(bf))):
                    self._pc_c_budget[i, j] = float(bf[j])
            self._pc_c_genre[i] = canonical_company_genre_vector(
                lv.get("genre_portfolio", [0.083] * 12)
            )

        self._pc_ready = True
        ram_kb = (n_p * (1 + 1 + 5 + 12) + n_c * (1 + 1 + 5 + 12)) * 4 / 1024
        print(f"  Built P-C scoring arrays: {n_p:,} persons × {n_c:,} companies ({ram_kb:.0f} KB)")

    def compute_pc_affinity_batch(self, pids: np.ndarray, cid_set: set) -> np.ndarray:
        """Compute brand-fit scores for pids × candidate companies.

        Returns ndarray of shape (len(pids),) with the max affinity score
        each person has with ANY company in cid_set.  Used as a multiplier
        in pick_cast / pick_director.

        Formula matches generate_edges_hybrid.generate_person_company_edges:
          raw = 0.30 * risk_match + 0.30 * budget_overlap + 0.30 * genre_overlap
        (noise term omitted for deterministic runtime scoring)
        """
        if not getattr(self, "_pc_ready", False) or not cid_set:
            return np.ones(len(pids), dtype=float)

        # Map company IDs to indices
        cids_list = [int(cid) for cid in cid_set if cid in self._pc_c_idx]
        c_idxs = np.array([self._pc_c_idx[cid] for cid in cids_list], dtype=int)
        if len(c_idxs) == 0:
            return np.ones(len(pids), dtype=float)

        p_idxs = np.array([self._pc_p_idx.get(int(pid), -1) for pid in pids], dtype=int)
        valid_mask = p_idxs >= 0
        if not valid_mask.any():
            return np.ones(len(pids), dtype=float)

        c_risk = self._pc_c_risk[c_idxs]
        c_budget = self._pc_c_budget[c_idxs]
        c_genre = self._pc_c_genre[c_idxs]

        vp = p_idxs[valid_mask]
        risk_match = 1.0 - np.abs(self._pc_p_risk[vp][:, None] - c_risk[None, :])
        budget_overlap = self._pc_p_budget[vp] @ c_budget.T
        genre_overlap = self._pc_p_genre[vp] @ c_genre.T
        raw = 0.30 * risk_match + 0.35 * budget_overlap + 0.35 * genre_overlap
        if getattr(self, "enable_llm_world_policy", False) and getattr(self, "world_policy", None):
            strategy_map = getattr(self, "world_policy", {}).get("company_strategy_assignments", {})
            if isinstance(strategy_map, dict):
                event_mask = np.array([strategy_map.get(str(cid)) == "event_franchise" for cid in cids_list], dtype=bool)
                if event_mask.any():
                    raw[:, event_mask] *= 1.05

        out = np.ones(len(pids), dtype=float)
        out[valid_mask] = np.where(np.max(raw, axis=1) > 0.60, 1.8, 1.0)
        return out

    def is_blacklisted(self, pid: int, cid: int) -> bool:
        """Check if person is blacklisted by company (controversy/tolerance)."""
        if not getattr(self, "_pc_ready", False):
            return False
        p_idx = self._pc_p_idx.get(int(pid))
        c_idx = self._pc_c_idx.get(int(cid))
        if p_idx is None or c_idx is None:
            return False
        return (self._pc_p_controversy[p_idx] > 0.6 and
                self._pc_c_controversy_tol[c_idx] < 0.3)

    def compute_cc_affinity(self, cid_a: int, cid_b: int) -> float:
        """On-demand company-company co-production affinity score."""
        if not getattr(self, "_pc_ready", False):
            return 0.0
        a_idx = self._pc_c_idx.get(int(cid_a))
        b_idx = self._pc_c_idx.get(int(cid_b))
        if a_idx is None or b_idx is None:
            return 0.0
        genre_sim = float(np.dot(self._pc_c_genre[a_idx], self._pc_c_genre[b_idx]))
        tier_sim = float(np.dot(self._pc_c_budget[a_idx], self._pc_c_budget[b_idx]))
        risk_match = 1.0 - abs(float(self._pc_c_risk[a_idx]) - float(self._pc_c_risk[b_idx]))
        score = 0.40 * genre_sim + 0.35 * tier_sim + 0.25 * risk_match
        if getattr(self, "enable_llm_world_policy", False) and getattr(self, "world_policy", None):
            if resolve_company_strategy(self.world_policy, cid_a) == resolve_company_strategy(self.world_policy, cid_b):
                score *= 1.12
        return score

    def compute_cc_rivalry(self, cid_a: int, cid_b: int) -> float:
        """On-demand company-company market rivalry score."""
        if not getattr(self, "_pc_ready", False):
            return 0.0
        a_idx = self._pc_c_idx.get(int(cid_a))
        b_idx = self._pc_c_idx.get(int(cid_b))
        if a_idx is None or b_idx is None:
            return 0.0
        genre_sim = float(np.dot(self._pc_c_genre[a_idx], self._pc_c_genre[b_idx]))
        tier_sim = float(np.dot(self._pc_c_budget[a_idx], self._pc_c_budget[b_idx]))
        raw = 0.50 * genre_sim + 0.50 * tier_sim
        if getattr(self, "enable_llm_world_policy", False) and getattr(self, "world_policy", None):
            if resolve_company_strategy(self.world_policy, cid_a) == resolve_company_strategy(self.world_policy, cid_b):
                raw *= 1.10
        return raw if raw > 0.70 else 0.0

    def _build_edge_adjacency(self):
        if self.graph is not None:
            self._friend_adj_all = self.graph.friend_adjacency
            self._rival_adj_all = self.graph.rival_adjacency
            self._adjacency_dirty = False
            return
        aff = self.affinity_index or {}
        friend_adj: Dict[int, list] = defaultdict(list)
        rival_adj: Dict[int, list] = defaultdict(list)

        friendships = aff.get("friendships", {})
        if isinstance(friendships, dict):
            for (a, b), entry in friendships.items():
                if isinstance(entry, dict):
                    w = float(entry.get("weight", 0.0) or 0.0)
                    vf = entry.get("valid_from")
                    vt = entry.get("valid_to")
                else:
                    w = float(entry) if entry else 0.0
                    vf = vt = None
                if w > 0:
                    friend_adj[int(a)].append((int(b), w, vf, vt))
                    friend_adj[int(b)].append((int(a), w, vf, vt))

        rivalries = aff.get("rivalries", {})
        if isinstance(rivalries, dict):
            for (a, b), entry in rivalries.items():
                if isinstance(entry, dict):
                    w = float(entry.get("weight", 0.9) or 0.9)
                    vf = entry.get("valid_from")
                    vt = entry.get("valid_to")
                else:
                    w = 0.9
                    vf = vt = None
                rival_adj[int(a)].append((int(b), w, vf, vt))
                rival_adj[int(b)].append((int(a), w, vf, vt))

        self._friend_adj_all = dict(friend_adj)
        self._rival_adj_all = dict(rival_adj)
        self._adjacency_dirty = False

    def _mark_adjacency_dirty(self):
        """P5-FIX: flag adjacency for lazy rebuild instead of rebuilding immediately."""
        self._adjacency_dirty = True

    def _remove_adj_edge(self, bucket_name: str, src_id: int, dst_id: int) -> None:
        adj = getattr(self, bucket_name, None)
        if not isinstance(adj, dict):
            return
        src = int(src_id)
        dst = int(dst_id)
        for node, other in ((src, dst), (dst, src)):
            rows = adj.get(node)
            if not rows:
                continue
            kept = [entry for entry in rows if int(entry[0]) != other]
            if kept:
                adj[node] = kept
            else:
                adj.pop(node, None)

    def _upsert_adj_edge(self, bucket_name: str, src_id: int, dst_id: int, payload: dict, default_weight: float) -> None:
        adj = getattr(self, bucket_name, None)
        if not isinstance(adj, dict):
            return
        src = int(src_id)
        dst = int(dst_id)
        self._remove_adj_edge(bucket_name, src, dst)
        weight = float(payload.get("weight", default_weight) or default_weight)
        valid_from = payload.get("valid_from")
        valid_to = payload.get("valid_to")
        adj.setdefault(src, []).append((dst, weight, valid_from, valid_to))
        adj.setdefault(dst, []).append((src, weight, valid_from, valid_to))

    def _refresh_edge_adjacency_delta(
        self,
        expired_ops: Sequence[tuple[int, int, str]],
        added_or_updated: Sequence[tuple[int, int, str]],
    ) -> None:
        if self.graph is not None:
            self._friend_adj_all = self.graph.friend_adjacency
            self._rival_adj_all = self.graph.rival_adjacency
            self._adjacency_dirty = False
            return
        if getattr(self, "_adjacency_dirty", False):
            self._build_edge_adjacency()
            return
        if not isinstance(getattr(self, "_friend_adj_all", None), dict) or not isinstance(getattr(self, "_rival_adj_all", None), dict):
            self._build_edge_adjacency()
            return

        expired_seen: set[tuple[int, int, str]] = set()
        for src_id, dst_id, edge_type in expired_ops:
            key = (min(int(src_id), int(dst_id)), max(int(src_id), int(dst_id)), str(edge_type))
            if key in expired_seen:
                continue
            expired_seen.add(key)
            if edge_type == "friendship":
                self._remove_adj_edge("_friend_adj_all", src_id, dst_id)
            elif edge_type == "rivalry":
                self._remove_adj_edge("_rival_adj_all", src_id, dst_id)

        aff = self.affinity_index or {}
        friendships = aff.get("friendships", {})
        rivalries = aff.get("rivalries", {})
        added_seen: set[tuple[int, int, str]] = set()
        for src_id, dst_id, edge_type in added_or_updated:
            key = (min(int(src_id), int(dst_id)), max(int(src_id), int(dst_id)), str(edge_type))
            if key in added_seen:
                continue
            added_seen.add(key)
            if edge_type == "friendship":
                payload = friendships.get((key[0], key[1]))
                if isinstance(payload, dict):
                    self._upsert_adj_edge("_friend_adj_all", key[0], key[1], payload, default_weight=0.0)
            elif edge_type == "rivalry":
                payload = rivalries.get((key[0], key[1]))
                if isinstance(payload, dict):
                    self._upsert_adj_edge("_rival_adj_all", key[0], key[1], payload, default_weight=0.9)

        self._adjacency_dirty = False

    def _ensure_edge_adjacency(self):
        """P5-FIX: rebuild adjacency only when dirty (lazy evaluation)."""
        if getattr(self, "_adjacency_dirty", True):
            self._build_edge_adjacency()

    def _build_person_sim_cache(self):
        assert self.persons is not None
        pids = self.persons["person_id"].astype(int).values
        ga_col = self.persons["genre_affinity"].fillna("").astype(str).values if "genre_affinity" in self.persons.columns else np.full(len(self.persons), "")
        st_col = self.persons["style_tags"].fillna("").astype(str).values if "style_tags" in self.persons.columns else np.full(len(self.persons), "")

        n = len(pids)
        csv_all = np.zeros((n, 8), dtype=np.float32)
        bbp_all = np.zeros((n, 5), dtype=np.float32)
        controversy_all = np.full(n, 0.15, dtype=np.float32)
        volatility_all = np.full(n, 0.4, dtype=np.float32)
        public_rep_all = np.full(n, 0.5, dtype=np.float32)
        collab_all = np.empty(n, dtype=object)
        avoid_genres_sparse: Dict[int, set] = {}
        cache: Dict[int, dict] = {}

        # A3-FIX: bit-vector arrays built in the SAME loop as cache (was two passes)
        from contracts import GENRES as _GENRES, STYLE_TAGS as _STYLE_TAGS
        genre_to_bit = {g.lower(): (1 << i) for i, g in enumerate(_GENRES)}
        style_to_bit = {s.lower(): (1 << i) for i, s in enumerate(_STYLE_TAGS)}
        genre_bits = np.zeros(n, dtype=np.uint32)
        style_bits = np.zeros(n, dtype=np.uint32)
        risk_arr = np.full(n, 0.5, dtype=np.float32)
        ambition_arr = np.full(n, 0.5, dtype=np.float32)
        collab_labels = {"solo": 0, "ensemble": 1, "chameleon": 2, "mentorship": 3}
        collab_code_arr = np.full(n, 2, dtype=np.int8)  # default chameleon

        for i, pid in enumerate(pids):
            lv = get_person_latent(self, int(pid))
            csv_normed = _normalize_vec(lv.get("creative_style_vector", [0.5] * 8), 8)
            bbp_normed = _normalize_vec(lv.get("budget_band_pref", [0.5] * 5), 5)
            ga_parsed = _parse_tag_set(ga_col[i])
            st_parsed = _parse_tag_set(st_col[i])
            risk_val = float(lv.get("risk_tolerance", 0.5))
            ambition_val = float(lv.get("artistic_ambition", 0.5))
            collab_style = str(lv.get("collaboration_style", "chameleon"))
            cache[int(pid)] = {
                "genre_set": ga_parsed,
                "genre_affinities": list(ga_parsed),
                "style_set": st_parsed,
                "risk_tolerance": risk_val,
                "artistic_ambition": ambition_val,
                "csv_normed": csv_normed,
                "bbp_normed": bbp_normed,
                "controversy_score": float(lv.get("controversy_score", 0.15)),
                "public_reputation": float(lv.get("public_reputation", 0.5)),
                "collaboration_style": collab_style,
            }
            csv_all[i] = csv_normed
            bbp_all[i] = bbp_normed
            controversy_all[i] = float(lv.get("controversy_score", 0.15))
            volatility_all[i] = float(lv.get("volatility", 0.4))
            public_rep_all[i] = float(lv.get("public_reputation", 0.5))
            collab_all[i] = collab_style
            ag = lv.get("avoid_genres")
            if isinstance(ag, list) and ag:
                avoid_genres_sparse[int(pid)] = set(ag)

            # A3-FIX: build bit-vectors in the same loop
            for g in ga_parsed:
                genre_bits[i] |= genre_to_bit.get(str(g).lower(), 0)
            for s in st_parsed:
                style_bits[i] |= style_to_bit.get(str(s).lower(), 0)
            risk_arr[i] = risk_val
            ambition_arr[i] = ambition_val
            collab_code_arr[i] = collab_labels.get(collab_style, 2)

        self._person_sim_cache = cache
        self._latent_pid_to_idx = {int(pid): i for i, pid in enumerate(pids)}
        self._latent_csv_normed = csv_all
        self._latent_bbp_normed = bbp_all
        self._latent_public_reputation = public_rep_all
        self._latent_controversy = controversy_all
        self._latent_volatility = volatility_all
        self._latent_collab = collab_all
        self._latent_avoid_genres = avoid_genres_sparse

        self._latent_genre_bits = genre_bits
        self._latent_style_bits = style_bits
        self._latent_risk = risk_arr
        self._latent_ambition = ambition_arr
        self._latent_collab_code = collab_code_arr
        print(f"  Built person similarity cache ({len(cache)} entries, {int(genre_bits.astype(bool).sum())} with genre bits, {int(style_bits.astype(bool).sum())} with style bits)")

    def _prewarm_year_cache(self):
        assert self.actors is not None
        actors = self.actors
        if actors.empty:
            self._year_cache = {}
            return

        has_debut = "debut_year" in actors.columns
        has_retire = "retirement_year" in actors.columns
        yr_lo, yr_hi = year_bounds_from_env(1950, 2025)
        if not YEAR_RANGE and has_debut:
            yr_lo = max(yr_lo, int(pd.to_numeric(actors["debut_year"], errors="coerce").fillna(yr_lo).min()))
            yr_hi = min(
                yr_hi,
                int(pd.to_numeric(actors["retirement_year"], errors="coerce").fillna(yr_hi).max()),
            ) if has_retire else yr_hi

        # P2-FIX: pre-compute lowered string columns ONCE on master DataFrame
        # instead of per-year on each copy.
        if "genre_affinity" in actors.columns:
            actors["_ga_lower"] = actors["genre_affinity"].fillna("").astype(str).str.lower()
        if "style_tags" in actors.columns:
            actors["_st_lower"] = actors["style_tags"].fillna("").astype(str).str.lower()

        # P2-FIX: store boolean masks instead of full DataFrame copies.
        # Each mask is ~N bytes vs ~N×500 bytes per copy.
        # Cache access: `self.actors[self._year_cache[year]]`
        self._year_cache = {}
        debut_arr = pd.to_numeric(actors["debut_year"], errors="coerce").fillna(yr_lo).astype(int).to_numpy() if has_debut else None
        retire_arr = pd.to_numeric(actors["retirement_year"], errors="coerce").fillna(yr_hi + 40).astype(int).to_numpy() if has_retire else None

        full_mask = np.ones(len(actors), dtype=bool)
        for year in range(int(yr_lo), int(yr_hi) + 1):
            if debut_arr is not None and retire_arr is not None:
                mask = (debut_arr <= year) & (retire_arr >= year)
                if mask.sum() < 10:
                    mask = full_mask
            else:
                mask = full_mask
            self._year_cache[year] = mask

        print(f"  Pre-warmed year cache for {len(self._year_cache)} years ({yr_lo}-{yr_hi})")
