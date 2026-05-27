from __future__ import annotations

"""Temporal evolution core for the synthetic film-industry world.

This rewrite keeps the public API compatible with the existing pipeline while
changing two important things:

1. Procedural evolution is driven by exposure aggregates and regime drift,
   rather than a large number of tiny hand-written local tweaks.
2. LLM evolution no longer asks the model to enumerate hundreds of raw edge ops.
   Instead it asks for a compact *strategy plan* that is expanded deterministically
   into validated patch operations.

Compatibility contract
----------------------
The following names are intentionally preserved because other modules import them
 directly:

- PatchApplyReport
- _clamp / _safe_int / _safe_str
- _person_exists / _company_exists / _find_person_row
- apply_world_patches
- procedural_year_step
- llm_year_step

Design notes
------------
- Entities are never deleted.
- Temporal edges use SCD2 versioning: updates expire the old edge version and
  append a new active version.
- Mutations touching person rows invalidate the year cache, because assembly.py
  stores year-level *copies* of actor rows.
- genre_weight_overrides are treated as *additive deltas* because that is what
  sample_movie_concept() currently consumes.
"""

import json
import hashlib
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from llm_provider import get_llm_client


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def _clamp(x: Any, lo: float, hi: float) -> float:
    try:
        v = float(x)
    except Exception:
        v = float(lo)
    if v != v:
        v = float(lo)
    return float(max(lo, min(hi, v)))


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _safe_str(x: Any, default: str = "") -> str:
    try:
        return str(x)
    except Exception:
        return default


def _now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _stable_hash(s: str) -> int:
    digest = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        pass

    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                continue
    raise ValueError("Could not parse JSON from model output")


# ---------------------------------------------------------------------------
# Patch schema / constants
# ---------------------------------------------------------------------------

ALLOWED_PERSON_LATENT_FIELDS = {
    "risk_tolerance",
    "controversy_score",
    "public_reputation",
    "artistic_ambition",
    "volatility",
}

ALLOWED_COMPANY_LATENT_FIELDS = {
    "risk_appetite",
    "prestige_score",
    "controversy_tolerance",
    "market_trend_sensitivity",
}

ALLOWED_EDGE_TYPES = {
    "friendship",
    "rivalry",
    "mentorship",
    "avoid",
    "collaboration",
    "chemistry",
    "preferred",
    "blacklist",
    "co_production",
    "market_rival",
    "brand_fit",
    "employment",
    "exclusive_deal",
    "subsidiary",
}

_EDGE_ENTITY_TYPES: Dict[str, Tuple[str, str]] = {
    # person ↔ person
    "friendship": ("person", "person"),
    "rivalry": ("person", "person"),
    "mentorship": ("person", "person"),
    "avoid": ("person", "person"),
    "collaboration": ("person", "person"),
    "chemistry": ("person", "person"),
    "preferred": ("person", "person"),
    # company ↔ company
    "co_production": ("company", "company"),
    "market_rival": ("company", "company"),
    "subsidiary": ("company", "company"),
    # company → person
    "blacklist": ("company", "person"),
    "brand_fit": ("company", "person"),
    "employment": ("company", "person"),
    "exclusive_deal": ("company", "person"),
}

_UNDIRECTED_EDGE_TYPES = {
    "friendship",
    "rivalry",
    "avoid",
    "collaboration",
    "chemistry",
    "co_production",
    "market_rival",
}

_NEGATIVE_EDGE_TYPES = {"rivalry", "avoid", "blacklist", "market_rival"}

ALLOWED_CAREER_STAGES = ["rising", "prime", "veteran", "legend", "retired"]
_CAREER_STAGE_ORDER = {s: i for i, s in enumerate(ALLOWED_CAREER_STAGES)}

COMPANY_TIERS = ["Micro", "Indie", "Mid-Budget", "Major", "Global"]
_COMPANY_TIER_ORDER = {t: i for i, t in enumerate(COMPANY_TIERS)}

_BASE_OP_BUDGETS = {
    "retire_person": 24,
    "career_stage_transition": 40,
    "dissolve_company": 12,
    "company_tier_transition": 22,
    "genre_trend_shift": 20,
}


@dataclass
class PatchApplyReport:
    applied: int = 0
    skipped: int = 0
    errors: int = 0
    messages: List[str] = field(default_factory=list)
    modified_person_ids: List[int] = field(default_factory=list)

    def log(self, msg: str) -> None:
        self.messages.append(str(msg))


# ---------------------------------------------------------------------------
# World/cache helpers
# ---------------------------------------------------------------------------

def _ensure_temporal_fields(world) -> None:
    if not hasattr(world, "temporal_state"):
        world.temporal_state = {
            "regime_score": 0.0,
            "regime_label": "neutral",
            "year_summaries": {},
            "llm_history": [],
            "relationship_pressure": {},
        }
    if not hasattr(world, "genre_weight_overrides"):
        world.genre_weight_overrides = {}
    if not hasattr(world, "country_weight_overrides"):
        world.country_weight_overrides = {}


def _invalidate_year_cache(world) -> None:
    cache = getattr(world, "_year_cache", None)
    if isinstance(cache, dict):
        cache.clear()


def _sync_company_tier_cache(world, company_id: int, new_tier: str) -> None:
    tier_map = getattr(world, "_company_tier_map", None)
    if isinstance(tier_map, dict):
        tier_map[int(company_id)] = str(new_tier)


def _person_exists(world, person_id: int) -> bool:
    # B2-FIX: cache the set on world to avoid O(N) rebuild per call
    _cache = getattr(world, "_person_id_set_cache", None)
    if _cache is None:
        try:
            if world.persons is not None and "person_id" in world.persons.columns:
                _cache = set(world.persons["person_id"].astype(int).tolist())
            else:
                _cache = set(int(k) for k in getattr(world, "person_latent", {}).keys())
        except Exception:
            _cache = set(int(k) for k in getattr(world, "person_latent", {}).keys())
        world._person_id_set_cache = _cache
    return int(person_id) in _cache


def _company_exists(world, company_id: int) -> bool:
    # B2-FIX: cache the set on world to avoid O(N) rebuild per call
    _cache = getattr(world, "_company_id_set_cache", None)
    if _cache is None:
        try:
            if world.companies is not None and "company_id" in world.companies.columns:
                _cache = set(world.companies["company_id"].astype(int).tolist())
            else:
                _cache = set(int(k) for k in getattr(world, "company_latent", {}).keys())
        except Exception:
            _cache = set(int(k) for k in getattr(world, "company_latent", {}).keys())
        world._company_id_set_cache = _cache
    return int(company_id) in _cache


def _find_person_row(world, person_id: int):
    if world.persons is None:
        return None
    df = world.persons
    try:
        # B3-FIX: return a single Series (or None), not a potentially-empty DataFrame.
        matches = df[df["person_id"].astype(int) == int(person_id)]
        if matches.empty:
            return None
        return matches.iloc[0]
    except Exception:
        return None


def _sync_sim_cache_for_persons(world, pids: Sequence[int]) -> None:
    """Rebuild person latent caches after latent changes.

    assembly.py uses both `_person_sim_cache` and batch latent arrays.  If we
    mutate person latent values but do not refresh these caches, later casting
    decisions still use stale values.
    """
    try:
        from world_state import get_person_latent, _normalize_vec
    except Exception:
        return

    sim_cache = getattr(world, "_person_sim_cache", None)
    pid_to_idx = getattr(world, "_latent_pid_to_idx", None)
    if not isinstance(sim_cache, dict):
        return

    for pid in pids:
        try:
            pid = int(pid)
            entry = sim_cache.get(pid)
            if not isinstance(entry, dict):
                continue
            lv = get_person_latent(world, pid)
            csv_normed = _normalize_vec(lv.get("creative_style_vector", [0.5] * 8), 8)
            bbp_normed = _normalize_vec(lv.get("budget_band_pref", [0.5] * 5), 5)
            entry["risk_tolerance"] = float(lv.get("risk_tolerance", 0.5))
            entry["artistic_ambition"] = float(lv.get("artistic_ambition", 0.5))
            entry["csv_normed"] = csv_normed
            entry["bbp_normed"] = bbp_normed
            entry["controversy_score"] = float(lv.get("controversy_score", 0.15))
            entry["public_reputation"] = float(lv.get("public_reputation", 0.5))
            entry["collaboration_style"] = str(lv.get("collaboration_style", "chameleon"))
            if isinstance(pid_to_idx, dict) and pid in pid_to_idx:
                idx = pid_to_idx[pid]
                if hasattr(world, "_latent_csv_normed"):
                    world._latent_csv_normed[idx] = csv_normed
                if hasattr(world, "_latent_bbp_normed"):
                    world._latent_bbp_normed[idx] = bbp_normed
                if hasattr(world, "_latent_controversy"):
                    world._latent_controversy[idx] = float(lv.get("controversy_score", 0.15))
                if hasattr(world, "_latent_volatility"):
                    world._latent_volatility[idx] = float(lv.get("volatility", 0.4))
                if hasattr(world, "_latent_collab"):
                    world._latent_collab[idx] = str(lv.get("collaboration_style", "chameleon"))
                # Phase 2 vectorized arrays — must stay in sync
                if hasattr(world, "_latent_risk"):
                    world._latent_risk[idx] = float(entry.get("risk_tolerance", 0.5))
                if hasattr(world, "_latent_ambition"):
                    world._latent_ambition[idx] = float(entry.get("artistic_ambition", 0.5))
                if hasattr(world, "_latent_collab_code"):
                    _collab_map = {"solo": 0, "ensemble": 1, "chameleon": 2, "mentorship": 3}
                    world._latent_collab_code[idx] = _collab_map.get(
                        str(lv.get("collaboration_style", "chameleon")), 2)
                if hasattr(world, "_latent_genre_bits"):
                    try:
                        from contracts import GENRES as _GENRES
                        _g2b = {g.lower(): (1 << i) for i, g in enumerate(_GENRES)}
                        bits = 0
                        for g in entry.get("genre_set", frozenset()):
                            bits |= _g2b.get(str(g).lower(), 0)
                        world._latent_genre_bits[idx] = bits
                    except ImportError:
                        pass
                if hasattr(world, "_latent_style_bits"):
                    try:
                        from contracts import STYLE_TAGS as _STYLE_TAGS
                        _s2b = {s.lower(): (1 << i) for i, s in enumerate(_STYLE_TAGS)}
                        bits = 0
                        for s in entry.get("style_set", frozenset()):
                            bits |= _s2b.get(str(s).lower(), 0)
                        world._latent_style_bits[idx] = bits
                    except ImportError:
                        pass
        except Exception:
            continue


# ---------------------------------------------------------------------------
# Edge helpers
# ---------------------------------------------------------------------------

def _edge_key(src_id: int, dst_id: int, edge_type: str) -> Tuple[int, int, str]:
    if edge_type in _UNDIRECTED_EDGE_TYPES:
        a, b = min(int(src_id), int(dst_id)), max(int(src_id), int(dst_id))
        return (a, b, edge_type)
    return (int(src_id), int(dst_id), edge_type)


def _edge_is_active(edge: Dict[str, Any], year: int | None = None) -> bool:
    if edge.get("_scd2_retired", False):
        return False
    if year is None:
        return True
    vf = edge.get("valid_from")
    vt = edge.get("valid_to")
    if vf is not None and _safe_int(vf, year) > year:
        return False
    if vt is not None and _safe_int(vt, year) < year:
        return False
    return True


def _build_edge_lookup(edges: Sequence[Dict[str, Any]], year: int | None = None) -> Dict[Tuple[int, int, str], Tuple[int, float]]:
    lookup: Dict[Tuple[int, int, str], Tuple[int, float]] = {}
    for idx, edge in enumerate(edges):
        try:
            edge_type = _safe_str(edge.get("edge_type"), "").strip()
            if not edge_type:
                continue
            key = _edge_key(_safe_int(edge.get("src_id")), _safe_int(edge.get("dst_id")), edge_type)
            weight = float(edge.get("weight", 0.0) or 0.0)
            active = _edge_is_active(edge, year)
            existing = lookup.get(key)
            if existing is None:
                lookup[key] = (idx, weight if active else -1.0)
            else:
                prev_idx, prev_weight = existing
                if active and weight >= prev_weight:
                    lookup[key] = (idx, weight)
                elif prev_weight < 0 and active:
                    lookup[key] = (idx, weight)
                else:
                    lookup[key] = (prev_idx, prev_weight)
        except Exception:
            continue
    return lookup


def _find_edge_index(
    edges: Sequence[Dict[str, Any]],
    src_id: int,
    dst_id: int,
    edge_type: str,
    lookup: Optional[Dict[Tuple[int, int, str], Tuple[int, float]]] = None,
) -> Optional[int]:
    key = _edge_key(src_id, dst_id, edge_type)
    if lookup is not None:
        entry = lookup.get(key)
        return entry[0] if entry is not None else None
    best_idx = None
    best_weight = -1.0
    for idx, edge in enumerate(edges):
        try:
            if _safe_str(edge.get("edge_type"), "") != edge_type:
                continue
            if _edge_key(_safe_int(edge.get("src_id")), _safe_int(edge.get("dst_id")), edge_type) != key:
                continue
            weight = float(edge.get("weight", 0.0) or 0.0)
            if weight > best_weight:
                best_weight = weight
                best_idx = idx
        except Exception:
            continue
    return best_idx


def _version_edge(
    edges: List[Dict[str, Any]],
    idx: int,
    year: int,
    edge_lookup: Optional[Dict[Tuple[int, int, str], Tuple[int, float]]] = None,
    **changes: Any,
) -> int:
    # B4-FIX: copy only canonical fields, not the entire old dict which
    # accumulates stale keys (reason, source_kind, …) across SCD2 versions.
    _CANONICAL_EDGE_KEYS = {
        "src_id", "dst_id", "edge_type", "weight", "sign",
        "valid_from", "valid_to", "community_id",
    }
    old = edges[idx]
    old["valid_to"] = int(year)
    old["_scd2_retired"] = True

    new_edge = {k: v for k, v in old.items() if k in _CANONICAL_EDGE_KEYS}
    new_edge["valid_from"] = int(year)
    new_edge["valid_to"] = None
    for key, value in changes.items():
        new_edge[key] = value

    edges.append(new_edge)
    new_idx = len(edges) - 1

    if edge_lookup is not None:
        key = _edge_key(_safe_int(new_edge.get("src_id")), _safe_int(new_edge.get("dst_id")), _safe_str(new_edge.get("edge_type"), ""))
        edge_lookup[key] = (new_idx, float(new_edge.get("weight", 0.0) or 0.0))
    return new_idx


def _incremental_index_refresh(world, expired_ops: Sequence[Tuple[int, int, str]], added_or_updated: Sequence[Tuple[int, int, str]]) -> None:
    graph = getattr(world, "graph", None)
    if graph is not None:
        try:
            world.affinity_index = graph.affinity_index
            world.edge_weights = graph.edge_weights
            if hasattr(world, "_refresh_edge_adjacency_delta"):
                world._refresh_edge_adjacency_delta(expired_ops, added_or_updated)
        except Exception:
            pass
        return
    edge_graph = getattr(world, "edge_graph", None)
    affinity_index = getattr(world, "affinity_index", None)
    if edge_graph is None or affinity_index is None:
        return
    try:
        for src, dst, edge_type in expired_ops:
            edge_graph.index_expire_edge(affinity_index, src, dst, edge_type)
        lookup = _build_edge_lookup(edge_graph.edges)
        for src, dst, edge_type in added_or_updated:
            idx = _find_edge_index(edge_graph.edges, src, dst, edge_type, lookup=lookup)
            if idx is not None:
                edge_graph.index_add_edge(affinity_index, edge_graph.edges[idx])
        if (expired_ops or added_or_updated) and hasattr(world, "_refresh_edge_adjacency_delta"):
            world._refresh_edge_adjacency_delta(expired_ops, added_or_updated)
    except Exception:
        try:
            world.affinity_index = edge_graph.build_affinity_index()
            if hasattr(world, "_mark_adjacency_dirty"):
                world._mark_adjacency_dirty()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Patch application
# ---------------------------------------------------------------------------

def _scaled_op_budgets(n_persons: int = 10000, n_companies: int = 1000) -> Dict[str, int]:
    return {
        "retire_person": max(_BASE_OP_BUDGETS["retire_person"], n_persons // 8000),
        "career_stage_transition": max(_BASE_OP_BUDGETS["career_stage_transition"], n_persons // 500),
        "dissolve_company": max(_BASE_OP_BUDGETS["dissolve_company"], n_companies // 500),
        "company_tier_transition": max(_BASE_OP_BUDGETS["company_tier_transition"], n_companies // 100),
        "genre_trend_shift": max(_BASE_OP_BUDGETS["genre_trend_shift"], n_persons // 2000),
    }


def apply_world_patches(
    world,
    patches: List[Dict[str, Any]],
    from_year: int,
    to_year: int,
    max_abs_latent_delta: float = 0.22,
    max_ops: int = 120,
) -> PatchApplyReport:
    rep = PatchApplyReport()
    _ensure_temporal_fields(world)

    if not isinstance(patches, list):
        rep.errors += 1
        rep.log("patches must be a list")
        return rep
    if not patches:
        return rep

    n_persons = len(world.persons) if getattr(world, "persons", None) is not None else 10000
    n_companies = len(world.companies) if getattr(world, "companies", None) is not None else 1000
    op_budgets = _scaled_op_budgets(n_persons, n_companies)

    if len(patches) > max_ops:
        rep.log(f"Too many ops ({len(patches)}). Truncating to {max_ops}.")
        patches = patches[:max_ops]

    # B2-FIX: invalidate cached id-sets so _person_exists / _company_exists
    # pick up any mutations made by this patch run.
    world._person_id_set_cache = None
    world._company_id_set_cache = None

    person_id_set: set[int] = set()
    person_row_idx: Dict[int, int] = {}
    company_id_set: set[int] = set()
    company_row_idx: Dict[int, int] = {}
    person_col_idx: Dict[str, int] = {}
    company_col_idx: Dict[str, int] = {}

    if getattr(world, "persons", None) is not None and "person_id" in world.persons.columns:
        pids = world.persons["person_id"].astype(int)
        person_id_set = set(pids.tolist())
        person_row_idx = {int(pid): i for i, pid in enumerate(pids)}
        person_col_idx = {c: i for i, c in enumerate(world.persons.columns)}
    else:
        person_id_set = set(int(k) for k in getattr(world, "person_latent", {}).keys())

    if getattr(world, "companies", None) is not None and "company_id" in world.companies.columns:
        cids = world.companies["company_id"].astype(int)
        company_id_set = set(cids.tolist())
        company_row_idx = {int(cid): i for i, cid in enumerate(cids)}
        company_col_idx = {c: i for i, c in enumerate(world.companies.columns)}
    else:
        company_id_set = set(int(k) for k in getattr(world, "company_latent", {}).keys())

    graph = getattr(world, "graph", None)

    op_counts: Dict[str, int] = defaultdict(int)
    invalidate_actor_cache = False
    expired_ops: List[Tuple[int, int, str]] = []
    add_or_update_ops: List[Tuple[int, int, str]] = []

    for op in patches:
        try:
            if not isinstance(op, dict):
                rep.skipped += 1
                rep.log("Skipped non-dict op")
                continue
            kind = _safe_str(op.get("op"), "").strip()
            if not kind:
                rep.skipped += 1
                rep.log("Skipped op with empty 'op' field")
                continue
            if kind in op_budgets and op_counts[kind] >= op_budgets[kind]:
                rep.skipped += 1
                rep.log(f"Budget exceeded for {kind} ({op_counts[kind]}/{op_budgets[kind]})")
                continue
            op_counts[kind] += 1

            if kind == "person_latent_delta":
                pid = _safe_int(op.get("person_id"))
                if pid not in person_id_set:
                    rep.skipped += 1
                    rep.log(f"person_latent_delta: unknown person_id={pid}")
                    continue
                delta = op.get("delta") or {}
                if not isinstance(delta, dict):
                    rep.skipped += 1
                    rep.log(f"person_latent_delta: delta not a dict for person_id={pid}")
                    continue
                lv = world.person_latent.get(pid)
                if not isinstance(lv, dict):
                    lv = {"person_id": pid}
                    world.person_latent[pid] = lv
                for field, raw_delta in delta.items():
                    if field not in ALLOWED_PERSON_LATENT_FIELDS:
                        continue
                    dv = _clamp(raw_delta, -max_abs_latent_delta, max_abs_latent_delta)
                    old = _clamp(lv.get(field, 0.5), 0.0, 1.0)
                    lv[field] = _clamp(old + dv, 0.0, 1.0)
                rep.applied += 1
                rep.modified_person_ids.append(pid)
                continue

            if kind == "company_latent_delta":
                cid = _safe_int(op.get("company_id"))
                if cid not in company_id_set:
                    rep.skipped += 1
                    rep.log(f"company_latent_delta: unknown company_id={cid}")
                    continue
                delta = op.get("delta") or {}
                if not isinstance(delta, dict):
                    rep.skipped += 1
                    rep.log(f"company_latent_delta: delta not a dict for company_id={cid}")
                    continue
                lv = world.company_latent.get(cid)
                if not isinstance(lv, dict):
                    lv = {"company_id": cid}
                    world.company_latent[cid] = lv
                for field, raw_delta in delta.items():
                    if field not in ALLOWED_COMPANY_LATENT_FIELDS:
                        continue
                    dv = _clamp(raw_delta, -max_abs_latent_delta, max_abs_latent_delta)
                    old = _clamp(lv.get(field, 0.5), 0.0, 1.0)
                    lv[field] = _clamp(old + dv, 0.0, 1.0)
                rep.applied += 1
                continue

            if kind == "retire_person":
                pid = _safe_int(op.get("person_id"))
                year = max(from_year, min(to_year, _safe_int(op.get("year"), to_year)))
                row_idx = person_row_idx.get(pid)
                if pid not in person_id_set or row_idx is None:
                    rep.skipped += 1
                    rep.log(f"retire_person: unknown person_id={pid}")
                    continue
                debut = _safe_int(world.persons.iat[row_idx, person_col_idx.get("debut_year", 0)] if "debut_year" in person_col_idx else from_year - 10, from_year - 10)
                if year < debut:
                    rep.skipped += 1
                    rep.log(f"retire_person: invalid year {year} < debut_year {debut} for person_id={pid}")
                    continue
                # V18 guard: minimum career length before retirement
                min_career_years = 8
                if year - debut < min_career_years:
                    rep.skipped += 1
                    rep.log(f"retire_person: person_id={pid} career too short ({year - debut}yr < {min_career_years}yr minimum)")
                    continue
                # V18 guard: don't retire 'rising' actors regardless of years
                if "career_stage" in person_col_idx:
                    cur_stage = str(world.persons.iat[row_idx, person_col_idx["career_stage"]]).strip()
                    if cur_stage == "rising":
                        rep.skipped += 1
                        rep.log(f"retire_person: person_id={pid} is still 'rising' -- skipped")
                        continue
                old_ret = _safe_int(world.persons.iat[row_idx, person_col_idx.get("retirement_year", 0)] if "retirement_year" in person_col_idx else 2100, 2100)
                new_ret = min(old_ret, year)
                if "retirement_year" in person_col_idx:
                    world.persons.iat[row_idx, person_col_idx["retirement_year"]] = int(new_ret)
                if "career_stage" in person_col_idx:
                    world.persons.iat[row_idx, person_col_idx["career_stage"]] = "retired"
                invalidate_actor_cache = True
                rep.applied += 1
                continue

            if kind == "set_yearly_max":
                pid = _safe_int(op.get("person_id"))
                ym = max(1, min(12, _safe_int(op.get("yearly_max"), 3)))
                row_idx = person_row_idx.get(pid)
                if pid not in person_id_set or row_idx is None or "yearly_max" not in person_col_idx:
                    rep.skipped += 1
                    rep.log(f"set_yearly_max: unknown or unsupported person_id={pid}")
                    continue
                world.persons.iat[row_idx, person_col_idx["yearly_max"]] = int(ym)
                invalidate_actor_cache = True
                rep.applied += 1
                continue

            if kind == "career_stage_transition":
                pid = _safe_int(op.get("person_id"))
                new_stage = _safe_str(op.get("new_stage"), "").strip()
                row_idx = person_row_idx.get(pid)
                if pid not in person_id_set or row_idx is None:
                    rep.skipped += 1
                    rep.log(f"career_stage_transition: unknown person_id={pid}")
                    continue
                if new_stage not in _CAREER_STAGE_ORDER:
                    rep.skipped += 1
                    rep.log(f"career_stage_transition: invalid stage '{new_stage}'")
                    continue
                old_stage = _safe_str(world.persons.iat[row_idx, person_col_idx.get("career_stage", 0)] if "career_stage" in person_col_idx else "rising", "rising")
                if _CAREER_STAGE_ORDER.get(new_stage, 0) <= _CAREER_STAGE_ORDER.get(old_stage, 0):
                    rep.skipped += 1
                    rep.log(f"career_stage_transition: non-forward transition {old_stage}->{new_stage}")
                    continue
                if "career_stage" in person_col_idx:
                    world.persons.iat[row_idx, person_col_idx["career_stage"]] = new_stage
                invalidate_actor_cache = True
                rep.applied += 1
                continue

            if kind == "genre_trend_shift":
                genre = _safe_str(op.get("genre"), "").strip()
                delta = _clamp(op.get("delta", 0.0), -0.05, 0.05)
                if not genre:
                    rep.skipped += 1
                    rep.log("genre_trend_shift: empty genre")
                    continue
                # V18: validate genre exists in contracts
                if not hasattr(world, 'genre_weight_overrides'):
                    world.genre_weight_overrides = {}
                try:
                    from contracts import GENRE_WEIGHTS
                    if genre not in GENRE_WEIGHTS:
                        rep.skipped += 1
                        rep.log(f"genre_trend_shift: unknown genre '{genre}'")
                        continue
                except ImportError:
                    pass
                old = float(getattr(world, "genre_weight_overrides", {}).get(genre, 0.0))
                world.genre_weight_overrides[genre] = _clamp(old + delta, -0.10, 0.10)
                rep.applied += 1
                continue

            if kind == "company_tier_transition":
                cid = _safe_int(op.get("company_id"))
                new_tier = _safe_str(op.get("new_tier"), "").strip()
                row_idx = company_row_idx.get(cid)
                if cid not in company_id_set or row_idx is None:
                    rep.skipped += 1
                    rep.log(f"company_tier_transition: unknown company_id={cid}")
                    continue
                if new_tier not in _COMPANY_TIER_ORDER:
                    rep.skipped += 1
                    rep.log(f"company_tier_transition: invalid tier '{new_tier}'")
                    continue
                old_tier = _safe_str(world.companies.iat[row_idx, company_col_idx.get("tier", 0)] if "tier" in company_col_idx else "Indie", "Indie")
                if abs(_COMPANY_TIER_ORDER[new_tier] - _COMPANY_TIER_ORDER.get(old_tier, 1)) > 1:
                    rep.skipped += 1
                    rep.log(f"company_tier_transition: jump too large {old_tier}->{new_tier}")
                    continue
                if "tier" in company_col_idx:
                    world.companies.iat[row_idx, company_col_idx["tier"]] = new_tier
                _sync_company_tier_cache(world, cid, new_tier)
                rep.applied += 1
                continue

            if kind == "dissolve_company":
                cid = _safe_int(op.get("company_id"))
                year = max(from_year, min(to_year, _safe_int(op.get("year"), to_year)))
                row_idx = company_row_idx.get(cid)
                if cid not in company_id_set or row_idx is None:
                    rep.skipped += 1
                    rep.log(f"dissolve_company: unknown company_id={cid}")
                    continue
                founded = _safe_int(world.companies.iat[row_idx, company_col_idx.get("founded_year", 0)] if "founded_year" in company_col_idx else from_year - 20, from_year - 20)
                if year < founded:
                    rep.skipped += 1
                    rep.log(f"dissolve_company: invalid year {year} < founded_year {founded}")
                    continue
                if "defunct_year" in company_col_idx:
                    old = world.companies.iat[row_idx, company_col_idx["defunct_year"]]
                    if old is None or (isinstance(old, float) and old != old):
                        world.companies.iat[row_idx, company_col_idx["defunct_year"]] = int(year)
                    else:
                        world.companies.iat[row_idx, company_col_idx["defunct_year"]] = min(int(old), int(year))
                rep.applied += 1
                continue

            if kind in {"edge_add", "edge_update", "edge_expire"}:
                if graph is None:
                    rep.skipped += 1
                    rep.log(f"{kind}: world has no graph runtime")
                    continue
                edge_type = _safe_str(op.get("edge_type"), "").strip()
                if edge_type not in ALLOWED_EDGE_TYPES:
                    rep.skipped += 1
                    rep.log(f"{kind}: invalid edge_type={edge_type}")
                    continue
                src = _safe_int(op.get("src_id"))
                dst = _safe_int(op.get("dst_id"))
                src_type, dst_type = _EDGE_ENTITY_TYPES.get(edge_type, ("person", "person"))
                src_ok = src in (company_id_set if src_type == "company" else person_id_set)
                dst_ok = dst in (company_id_set if dst_type == "company" else person_id_set)
                if not src_ok or not dst_ok:
                    rep.skipped += 1
                    rep.log(f"{kind}: unknown src/dst ({src}, {dst})")
                    continue
                if kind == "edge_add":
                    if graph.has_active_edge(edge_type, src, dst, from_year):
                        rep.skipped += 1
                        rep.log(f"edge_add: edge already exists for {edge_type} ({src}, {dst})")
                        continue
                    weight = _clamp(op.get("weight", 0.4), 0.0, 1.0)
                    sign = _safe_str(op.get("sign"), "")
                    if sign not in {"+", "-"}:
                        sign = "-" if edge_type in _NEGATIVE_EDGE_TYPES else "+"
                    valid_from = _safe_int(op.get("valid_from"), from_year)
                    valid_to_raw = op.get("valid_to")
                    valid_to = None if valid_to_raw is None else max(valid_from, _safe_int(valid_to_raw, valid_from))
                    new_edge = {
                        "src_id": int(src),
                        "dst_id": int(dst),
                        "src_type": src_type,
                        "dst_type": dst_type,
                        "edge_type": edge_type,
                        "sign": sign,
                        "weight": float(weight),
                        "valid_from": int(valid_from),
                        "valid_to": valid_to,
                        "reason": _safe_str(op.get("reason"), ""),
                        "source_kind": _safe_str(op.get("source_kind"), "temporal"),
                    }
                    if graph.add_edge(
                        edge_type,
                        int(src),
                        int(dst),
                        float(weight),
                        int(valid_from),
                        sign=sign,
                        reason=_safe_str(op.get("reason"), ""),
                        source_kind=_safe_str(op.get("source_kind"), "temporal"),
                    ):
                        add_or_update_ops.append((src, dst, edge_type))
                        rep.applied += 1
                    else:
                        rep.skipped += 1
                        rep.log(f"edge_add: graph runtime rejected duplicate for {edge_type} ({src}, {dst})")
                    continue

                if not graph.has_active_edge(edge_type, src, dst, from_year):
                    rep.skipped += 1
                    rep.log(f"{kind}: edge not found for {edge_type} ({src}, {dst})")
                    continue

                if kind == "edge_update":
                    new_weight = graph.get_active_edge_weight(edge_type, src, dst, from_year)
                    if "weight" in op:
                        new_weight = _clamp(op.get("weight"), 0.0, 1.0)
                    if "delta_weight" in op:
                        new_weight = _clamp(new_weight + float(_clamp(op.get("delta_weight"), -0.35, 0.35)), 0.0, 1.0)
                    if graph.update_edge(edge_type, src, dst, from_year, weight=float(new_weight), reason=_safe_str(op.get("reason"), "")):
                        expired_ops.append((src, dst, edge_type))
                        add_or_update_ops.append((src, dst, edge_type))
                        rep.applied += 1
                    else:
                        rep.skipped += 1
                        rep.log(f"edge_update: graph runtime failed for {edge_type} ({src}, {dst})")
                    continue

                year = max(from_year, min(to_year, _safe_int(op.get("year"), to_year)))
                if graph.expire_edge(edge_type, src, dst, int(year), reason=_safe_str(op.get("reason"), "")):
                    expired_ops.append((src, dst, edge_type))
                    rep.applied += 1
                else:
                    rep.skipped += 1
                    rep.log(f"edge_expire: graph runtime failed for {edge_type} ({src}, {dst})")
                continue

            rep.skipped += 1
            rep.log(f"Unknown op type: {kind}")

        except Exception as exc:
            rep.errors += 1
            rep.log(f"Error applying op {op}: {exc}")

    if rep.modified_person_ids:
        _sync_sim_cache_for_persons(world, rep.modified_person_ids)
    if invalidate_actor_cache:
        _invalidate_year_cache(world)
    if expired_ops or add_or_update_ops:
        _incremental_index_refresh(world, expired_ops, add_or_update_ops)
    return rep


# ---------------------------------------------------------------------------
# Exposure / regime modelling
# ---------------------------------------------------------------------------

def _pair_key(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a < b else (b, a)


@dataclass
class ExposureContext:
    person_perf: Dict[int, List[Tuple[float, float]]] = field(default_factory=lambda: defaultdict(list))
    company_perf: Dict[int, List[Tuple[float, float]]] = field(default_factory=lambda: defaultdict(list))
    co_counts: Counter = field(default_factory=Counter)
    competition_counts: Counter = field(default_factory=Counter)
    director_actor_counts: Counter = field(default_factory=Counter)
    company_person_counts: Counter = field(default_factory=Counter)
    company_pair_counts: Counter = field(default_factory=Counter)
    active_person_ids: set[int] = field(default_factory=set)
    active_company_ids: set[int] = field(default_factory=set)
    genre_counts: Counter = field(default_factory=Counter)
    country_counts: Counter = field(default_factory=Counter)
    avg_rating: float = 6.0
    avg_perf: float = 1.0
    n_movies: int = 0


def _build_exposure_context(year_bucket: Sequence[Dict[str, Any]]) -> ExposureContext:
    ctx = ExposureContext()
    ratings: List[float] = []
    perfs: List[float] = []
    for movie in year_bucket:
        rating = float(movie.get("rating", 6.0) or 6.0)
        perf = float(movie.get("performance_ratio", 1.0) or 1.0)
        ratings.append(rating)
        perfs.append(perf)
        cast_ids = [int(pid) for pid in (movie.get("cast_ids") or []) if pid]
        company_ids = [int(cid) for cid in (movie.get("company_ids") or []) if cid]
        director_id = movie.get("director_id")
        genre = _safe_str(movie.get("genre"), "").strip()
        country = _safe_str(movie.get("country"), "").strip()
        if genre:
            ctx.genre_counts[genre] += 1
        if country:
            ctx.country_counts[country] += 1
        ctx.n_movies += 1

        for pid in cast_ids:
            ctx.person_perf[pid].append((rating, perf))
            ctx.active_person_ids.add(pid)
        if director_id is not None:
            did = int(director_id)
            ctx.person_perf[did].append((rating, perf))
            ctx.active_person_ids.add(did)
            for pid in cast_ids[: min(5, len(cast_ids))]:
                ctx.director_actor_counts[(did, pid)] += 1

        for cid in company_ids:
            ctx.company_perf[cid].append((rating, perf))
            ctx.active_company_ids.add(cid)
            for pid in cast_ids[: min(4, len(cast_ids))]:
                ctx.company_person_counts[(cid, pid)] += 1

        for pair in combinations(cast_ids[: min(8, len(cast_ids))], 2):
            ctx.co_counts[_pair_key(int(pair[0]), int(pair[1]))] += 1
        for pair in movie.get("competition_pairs") or []:
            try:
                a, b = int(pair[0]), int(pair[1])
                if a != b:
                    ctx.competition_counts[_pair_key(a, b)] += 1
            except Exception:
                continue
        for pair in combinations(company_ids[: min(3, len(company_ids))], 2):
            ctx.company_pair_counts[_pair_key(int(pair[0]), int(pair[1]))] += 1

    if ratings:
        ctx.avg_rating = float(np.mean(ratings))
    if perfs:
        ctx.avg_perf = float(np.mean(perfs))
    return ctx


def _summarize_regime(world, from_year: int, ctx: ExposureContext) -> Dict[str, Any]:
    _ensure_temporal_fields(world)
    state = world.temporal_state
    prev_score = float(state.get("regime_score", 0.0))
    quality_signal = np.tanh((ctx.avg_rating - 6.4) / 1.4)
    profit_signal = np.tanh((ctx.avg_perf - 1.0) / 0.9)
    size_signal = np.tanh((ctx.n_movies - 120.0) / 120.0)
    new_score = float(np.clip(0.72 * prev_score + 0.45 * quality_signal + 0.35 * profit_signal + 0.10 * size_signal, -1.0, 1.0))
    if new_score >= 0.35:
        label = "boom"
    elif new_score <= -0.35:
        label = "stress"
    else:
        label = "neutral"
    summary = {
        "year": int(from_year),
        "regime_score": round(new_score, 4),
        "regime_label": label,
        "avg_rating": round(ctx.avg_rating, 4),
        "avg_perf": round(ctx.avg_perf, 4),
        "n_movies": int(ctx.n_movies),
        "top_genres": ctx.genre_counts.most_common(6),
        "top_countries": ctx.country_counts.most_common(6),
    }
    state["regime_score"] = new_score
    state["regime_label"] = label
    state["year_summaries"][int(from_year)] = summary
    return summary


def _company_tier_for(world, company_id: int) -> str:
    # A9-FIX: lazily build _company_tier_map if missing, instead of
    # falling back to an expensive per-row DataFrame filter.
    tier_map = getattr(world, "_company_tier_map", None)
    if tier_map is None and getattr(world, "companies", None) is not None and "company_id" in world.companies.columns:
        tier_col = world.companies.get("tier", pd.Series(["Mid-Budget"] * len(world.companies))).fillna("Mid-Budget").astype(str)
        world._company_tier_map = dict(zip(world.companies["company_id"].astype(int), tier_col))
        tier_map = world._company_tier_map
    if isinstance(tier_map, dict) and company_id in tier_map:
        return _safe_str(tier_map[company_id], "Mid-Budget")
    return "Mid-Budget"


def _person_stage_for(world, person_id: int) -> str:
    if getattr(world, "persons", None) is None or "person_id" not in world.persons.columns:
        return "prime"
    row = world.persons[world.persons["person_id"].astype(int) == int(person_id)]
    if len(row) == 0:
        return "prime"
    return _safe_str(row.iloc[0].get("career_stage"), "prime")


def _build_person_stage_cache(world) -> Dict[int, str]:
    """Build {person_id: career_stage} dict for O(1) lookups."""
    if getattr(world, "persons", None) is None or "person_id" not in world.persons.columns:
        return {}
    pids = world.persons["person_id"].astype(int).values
    stages = world.persons["career_stage"].fillna("prime").astype(str).values if "career_stage" in world.persons.columns else ["prime"] * len(pids)
    return {int(pid): str(s) for pid, s in zip(pids, stages)}


def _top_abs_items(signal_map: Dict[Any, float], limit: int) -> List[Tuple[Any, float]]:
    return sorted(signal_map.items(), key=lambda kv: abs(kv[1]), reverse=True)[:limit]


def _ensure_edge_arrays(edge_graph) -> None:
    """Lazily build/rebuild NumPy arrays for vectorized edge filtering.

    Caches `_arr_valid_from`, `_arr_valid_to`, `_arr_retired` on the edge_graph
    object.  Rebuilds only when the edge list has grown since last build.
    """
    edges = edge_graph.edges
    n = len(edges)
    cached_n = getattr(edge_graph, "_arr_n", 0)
    if cached_n == n and hasattr(edge_graph, "_arr_valid_from"):
        return  # already up-to-date



    # P6-FIX: batch extraction via list comprehension (C-speed) instead of
    # per-element Python loop with dict.get() calls.
    vf_raw = [e.get("valid_from") for e in edges]
    vt_raw = [e.get("valid_to") for e in edges]
    ret_raw = [bool(e.get("_scd2_retired", False)) for e in edges]

    vf = np.array([float(v) if v is not None else -1e9 for v in vf_raw], dtype=np.float64)
    vt = np.array([float(v) if v is not None else 1e9 for v in vt_raw], dtype=np.float64)
    retired = np.array(ret_raw, dtype=bool)
    edge_graph._arr_valid_from = vf
    edge_graph._arr_valid_to = vt
    edge_graph._arr_retired = retired
    edge_graph._arr_n = n


def _background_edge_sample(edges: Sequence[Dict[str, Any]], from_year: int, sample_size: int, seed: int, edge_graph=None) -> List[int]:
    """Sample active edges using vectorized NumPy filtering.

    If `edge_graph` is provided and has cached arrays, uses O(N) NumPy boolean
    masking instead of O(N) Python dict access — ~20× faster on 7M edges.
    Falls back to Python scan if arrays aren't available.
    """
    if not edges or sample_size <= 0:
        return []

    if edge_graph is not None and hasattr(edge_graph, "_arr_valid_from"):
        vf = edge_graph._arr_valid_from
        vt = edge_graph._arr_valid_to
        retired = edge_graph._arr_retired
        # Vectorized active check: not retired AND valid_from <= year AND valid_to >= year
        n = len(vf)
        if n < len(edges):
            # A4-FIX: extend arrays to cover new edges instead of silently
            # ignoring them. Triggers a full rebuild to stay consistent.
            _ensure_edge_arrays.__wrapped__(edge_graph) if hasattr(_ensure_edge_arrays, '__wrapped__') else None
            # Simplest fix: force a full rebuild by resetting cached_n
            edge_graph._arr_n = 0
            _ensure_edge_arrays(edge_graph)
            vf = edge_graph._arr_valid_from
            vt = edge_graph._arr_valid_to
            retired = edge_graph._arr_retired
            mask = (~retired) & (vf <= from_year) & (vt >= from_year)
        else:
            mask = (~retired) & (vf <= from_year) & (vt >= from_year)
        active_indices = np.flatnonzero(mask)
        if len(active_indices) <= sample_size:
            return active_indices.tolist()
        rng_np = np.random.RandomState(seed & 0x7FFFFFFF)
        chosen = rng_np.choice(active_indices, size=sample_size, replace=False)
        return chosen.tolist()

    # Fallback: Python scan
    active_indices = [i for i, e in enumerate(edges) if _edge_is_active(e, from_year)]
    if len(active_indices) <= sample_size:
        return active_indices
    rng = random.Random(seed)
    return rng.sample(active_indices, sample_size)


def _emit_edge_program_patches(
    world,
    ctx: ExposureContext,
    from_year: int,
    to_year: int,
    rng: random.Random,
    regime_label: str = "neutral",
    budget_hint: int | None = None,
) -> List[Dict[str, Any]]:
    """Translate exposure aggregates into concrete edge mutations.

    The goal here is not maximum local cleverness.  The goal is to create many
    *coherent* changes driven by the current activity graph and the yearly regime.

    Scaling:
    - .most_common() caps scale with counter size (1/3 of total, min=base cap)
    - Latent similarity for co-star pairs is batched via latent_similarity_batch
    - Background churn targets ~1.5% of edges per year
    - Regime multipliers: boom amplifies positive edges, stress amplifies negative
    """
    patches: List[Dict[str, Any]] = []
    stage_cache = _build_person_stage_cache(world)

    # Regime multipliers
    if regime_label == "boom":
        friend_weight_mult = 1.3
        rivalry_weight_mult = 0.85
        expiry_prob_mult = 0.5
    elif regime_label == "stress":
        friend_weight_mult = 0.85
        rivalry_weight_mult = 1.2
        expiry_prob_mult = 1.5
    else:
        friend_weight_mult = 1.0
        rivalry_weight_mult = 1.0
        expiry_prob_mult = 1.0

    # 1) Repeated co-stars -> friendship/collaboration strengthening or creation.
    #    Batched latent similarity instead of per-pair calls.
    budget_hint = None if budget_hint is None else max(128, int(budget_hint))
    costar_cap = max(1800, len(ctx.co_counts) // 3)
    if budget_hint is not None:
        costar_cap = min(costar_cap, max(240, budget_hint // 3))
    costar_pairs = ctx.co_counts.most_common(costar_cap)

    if costar_pairs:
        # Collect all unique person IDs for batch similarity
        all_a_ids = []
        all_b_ids = []
        pair_counts = []
        for (a, b), count in costar_pairs:
            all_a_ids.append(a)
            all_b_ids.append(b)
            pair_counts.append(count)

        # Batch similarity computation
        pid_to_idx = getattr(world, "_latent_pid_to_idx", None)
        sims = np.zeros(len(costar_pairs), dtype=np.float32)
        if pid_to_idx is not None and hasattr(world, "_latent_csv_normed"):
            try:
                from world_state import latent_similarity_batch
                # Build index arrays for all pairs
                a_li = np.array([pid_to_idx.get(a, -1) for a in all_a_ids], dtype=int)
                b_li = np.array([pid_to_idx.get(b, -1) for b in all_b_ids], dtype=int)
                valid = (a_li >= 0) & (b_li >= 0)
                if valid.any():
                    valid_a = a_li[valid]
                    valid_b = b_li[valid]
                    # A1-FIX: true vectorized pairwise similarity.
                    # latent_similarity_batch returns an (m × k) matrix;
                    # for 1:1 pairs we extract the diagonal in chunks.
                    chunk_size = 500
                    batch_sims = np.zeros(int(valid.sum()), dtype=np.float32)
                    for ci in range(0, len(valid_a), chunk_size):
                        ca = valid_a[ci:ci+chunk_size]
                        cb = valid_b[ci:ci+chunk_size]
                        sim_mat = latent_similarity_batch(world, ca, cb)
                        # Extract diagonal: sim_mat[i, i] gives pairwise sim for (ca[i], cb[i])
                        diag = np.diag(sim_mat) if sim_mat.ndim == 2 else sim_mat
                        batch_sims[ci:ci+len(diag)] = diag
                    sims[valid] = batch_sims
            except Exception:
                pass

        for i, ((a, b), count) in enumerate(costar_pairs):
            sim = float(sims[i])
            base_weight = 0.22 + 0.20 * min(1.0, sim)
            if count >= 2:
                base_weight += 0.08
            base_weight *= friend_weight_mult  # Regime amplification
            if sim >= 0.60 or count >= 3:
                patches.append({
                    "op": "edge_add",
                    "edge_type": "friendship",
                    "src_id": a,
                    "dst_id": b,
                    "weight": _clamp(base_weight, 0.18, 0.85),
                    "valid_from": from_year,
                    "reason": f"repeated co-starring x{count}",
                    "source_kind": "procedural_costar",
                })
                if count >= 3:
                    patches.append({
                        "op": "edge_add",
                        "edge_type": "collaboration",
                        "src_id": a,
                        "dst_id": b,
                        "weight": _clamp(base_weight + 0.08, 0.20, 0.92),
                        "valid_from": from_year,
                        "reason": f"high-repeat collaboration x{count}",
                        "source_kind": "procedural_costar",
                    })

    # 2) Competition produces rivalry (scaled cap, regime multiplier).
    rivalry_cap = max(900, len(ctx.competition_counts) // 3)
    if budget_hint is not None:
        rivalry_cap = min(rivalry_cap, max(160, budget_hint // 5))
    for (a, b), count in ctx.competition_counts.most_common(rivalry_cap):
        base_w = (0.24 + 0.10 * count) * rivalry_weight_mult
        patches.append({
            "op": "edge_add",
            "edge_type": "rivalry",
            "src_id": a,
            "dst_id": b,
            "weight": _clamp(base_w, 0.25, 0.90),
            "valid_from": from_year,
            "reason": f"role competition x{count}",
            "source_kind": "procedural_competition",
        })

    # 3) Veteran/legend directors repeatedly working with a rising actor -> mentorship.
    #    Uses stage_cache for O(1) lookups instead of O(N) DataFrame scans.
    mentorship_cap = max(900, len(ctx.director_actor_counts) // 3)
    if budget_hint is not None:
        mentorship_cap = min(mentorship_cap, max(160, budget_hint // 6))
    for (director_id, actor_id), count in ctx.director_actor_counts.most_common(mentorship_cap):
        stage = stage_cache.get(int(director_id), "prime")
        if stage not in {"veteran", "legend"}:
            continue
        actor_stage = stage_cache.get(int(actor_id), "prime")
        if actor_stage == "retired":
            continue
        if count >= 2 or actor_stage == "rising":
            patches.append({
                "op": "edge_add",
                "edge_type": "mentorship",
                "src_id": director_id,
                "dst_id": actor_id,
                "weight": _clamp(0.28 + 0.07 * count, 0.30, 0.88),
                "valid_from": from_year,
                "reason": f"repeated guidance x{count}",
                "source_kind": "procedural_mentorship",
            })

    # 4) Company-person fit and company-company co-production ties (scaled caps).
    company_person_cap = max(1100, len(ctx.company_person_counts) // 3)
    if budget_hint is not None:
        company_person_cap = min(company_person_cap, max(180, budget_hint // 4))
    for (company_id, person_id), count in ctx.company_person_counts.most_common(company_person_cap):
        patches.append({
            "op": "edge_add",
            "edge_type": "brand_fit",
            "src_id": company_id,
            "dst_id": person_id,
            "weight": _clamp(0.18 + 0.08 * count, 0.20, 0.86),
            "valid_from": from_year,
            "reason": f"repeated company/person exposure x{count}",
            "source_kind": "procedural_company_person",
        })
        if count >= 2:
            patches.append({
                "op": "edge_add",
                "edge_type": "employment",
                "src_id": company_id,
                "dst_id": person_id,
                "weight": _clamp(0.12 + 0.06 * count, 0.18, 0.75),
                "valid_from": from_year,
                "reason": f"quasi-recurring employment x{count}",
                "source_kind": "procedural_company_person",
            })

    company_pair_cap = max(600, len(ctx.company_pair_counts) // 3)
    if budget_hint is not None:
        company_pair_cap = min(company_pair_cap, max(120, budget_hint // 7))
    for (a, b), count in ctx.company_pair_counts.most_common(company_pair_cap):
        patches.append({
            "op": "edge_add",
            "edge_type": "co_production",
            "src_id": a,
            "dst_id": b,
            "weight": _clamp(0.25 + 0.09 * count, 0.25, 0.90),
            "valid_from": from_year,
            "reason": f"shared slate x{count}",
            "source_kind": "procedural_company_pair",
        })

    # 5) Background churn — scaled to ~1.5% of edges per year.
    graph = getattr(world, "graph", None)
    if graph is not None:
        hot_count = sum(int(meta.get("count", 0)) for meta in getattr(graph, "manifest", {}).get("hot_types", {}).values())
        cold_count = int(getattr(graph, "manifest", {}).get("cold_cp_count", 0)) + int(getattr(graph, "manifest", {}).get("cold_cc_count", 0))
        n_edges = hot_count + cold_count
        hot_sample_size = max(240, min(18_000, max(1, hot_count) // 30 + len(ctx.active_person_ids)))
        cold_sample_size = max(64, min(2_048, max(1, cold_count) // 4096))
        if budget_hint is not None:
            hot_sample_size = min(hot_sample_size, max(192, budget_hint // 2))
            cold_sample_size = min(cold_sample_size, max(64, budget_hint // 12))
        sampled_edges = graph.sample_hot_edges(from_year, hot_sample_size, seed=_stable_hash(f"hot_edge_sample|{from_year}|{to_year}"))
        sampled_edges.extend(graph._sample_cold_edges(from_year, cold_sample_size, seed=_stable_hash(f"cold_edge_sample|{from_year}|{to_year}")))
        for edge in sampled_edges:
            edge_type = _safe_str(edge.get("edge_type"), "")
            src = _safe_int(edge.get("src_id"))
            dst = _safe_int(edge.get("dst_id"))
            weight = float(edge.get("weight", 0.0) or 0.0)
            touch_active = src in ctx.active_person_ids or src in ctx.active_company_ids or dst in ctx.active_person_ids or dst in ctx.active_company_ids
            if edge_type in {"friendship", "collaboration", "chemistry", "co_production", "brand_fit", "employment", "mentorship"}:
                if touch_active:
                    delta = rng.uniform(-0.03, 0.08)
                    patches.append({
                        "op": "edge_update",
                        "edge_type": edge_type,
                        "src_id": src,
                        "dst_id": dst,
                        "delta_weight": delta,
                        "reason": f"background drift (active)",
                    })
                elif weight < 0.20 or rng.random() < 0.35 * expiry_prob_mult:
                    patches.append({
                        "op": "edge_expire",
                        "edge_type": edge_type,
                        "src_id": src,
                        "dst_id": dst,
                        "year": from_year,
                        "reason": "background expiry after inactivity",
                    })
            elif edge_type in {"rivalry", "avoid", "market_rival", "blacklist"}:
                delta = rng.uniform(-0.05, 0.06) if touch_active else rng.uniform(-0.08, 0.02)
                delta *= rivalry_weight_mult  # Regime amplification
                patches.append({
                    "op": "edge_update",
                    "edge_type": edge_type,
                    "src_id": src,
                    "dst_id": dst,
                    "delta_weight": delta,
                    "reason": "background negative-edge drift",
                })
    return patches


def _amplify_genre_cascade(
    world, genre: str, delta: float, from_year: int, rng: random.Random, budget_hint: int | None = None,
) -> List[Dict[str, Any]]:
    """When a genre trend shifts, propagate to people who work in that genre.

    For each person with `genre` in their genre_set:
    - public_reputation gets a small nudge in the direction of the genre shift
    - Edges between same-genre people get strengthened (if positive delta)
    """
    patches: List[Dict[str, Any]] = []
    sim_cache = getattr(world, "_person_sim_cache", None)
    if not isinstance(sim_cache, dict) or abs(delta) < 0.01:
        return patches

    genre_lower = genre.lower().strip()
    affected_pids = []
    for pid, entry in sim_cache.items():
        gs = entry.get("genre_set", frozenset())
        if any(str(g).lower() == genre_lower for g in gs):
            affected_pids.append(int(pid))

    if not affected_pids:
        return patches

    # Scale: affect up to 1/4 of matching people, minimum 20
    n_affect = max(20, min(len(affected_pids), len(affected_pids) // 4 + 10))
    if budget_hint is not None:
        n_affect = min(n_affect, max(12, int(budget_hint // 3)))
    rng.shuffle(affected_pids)
    for pid in affected_pids[:n_affect]:
        rep_nudge = _clamp(delta * 0.4, -0.06, 0.06)  # ~40% of genre delta
        if abs(rep_nudge) >= 0.01:
            patches.append({
                "op": "person_latent_delta",
                "person_id": pid,
                "delta": {"public_reputation": rep_nudge},
            })

    # Strengthen edges between affected people (if positive delta = genre is hot)
    if delta > 0.02 and len(affected_pids) >= 4:
        sample = affected_pids[:min(80, len(affected_pids))]
        edge_cap = min(40, len(sample) - 1)
        if budget_hint is not None:
            edge_cap = min(edge_cap, max(12, int(budget_hint // 4)))
        for i in range(edge_cap):
            a, b = sample[i], sample[(i + 1) % len(sample)]
            if a == b:
                continue
            patches.append({
                "op": "edge_add",
                "edge_type": "collaboration",
                "src_id": min(a, b),
                "dst_id": max(a, b),
                "weight": _clamp(0.22 + abs(delta) * 2.0, 0.20, 0.65),
                "valid_from": from_year,
                "reason": f"genre cascade: {genre} trending",
                "source_kind": "procedural_genre_cascade",
            })
    return patches


def _amplify_regional_wave(
    world, ctx: ExposureContext, from_year: int, rng: random.Random, budget_hint: int | None = None,
) -> List[Dict[str, Any]]:
    """Countries with disproportionate movie share get a regional amplification wave.

    Strengthens same-country edges and bumps reputation for top performers.
    """
    patches: List[Dict[str, Any]] = []
    if not ctx.country_counts or ctx.n_movies < 10:
        return patches

    total = max(1, ctx.n_movies)
    # Find countries with > 15% share (disproportionate)
    hot_countries = [
        (country, count / total)
        for country, count in ctx.country_counts.most_common(6)
        if count / total > 0.15
    ]
    if not hot_countries:
        return patches

    persons_df = getattr(world, "persons", None)
    if persons_df is None or "nationality" not in persons_df.columns:
        return patches

    # Pre-build pid → nationality dict (O(N_persons) once, then O(1) lookups)
    pid_to_nat = {}
    try:
        _p_ids = persons_df["person_id"].astype(int).values
        _p_nats = persons_df["nationality"].fillna("").astype(str).values
        for _i in range(len(_p_ids)):
            pid_to_nat[int(_p_ids[_i])] = _p_nats[_i].strip().lower()
    except Exception:
        return patches

    for country, share in hot_countries:
        excess = share - 0.10  # How much above baseline
        wave_strength = _clamp(excess * 0.5, 0.01, 0.08)
        country_lower = country.lower()

        # Find people from this country who are active — O(1) per person
        country_pids = [
            int(pid) for pid in ctx.active_person_ids
            if pid_to_nat.get(int(pid), "") == country_lower
        ]

        if not country_pids:
            continue

        # Bump reputation for top performers from that country
        n_bump = max(5, min(len(country_pids), len(country_pids) // 3))
        if budget_hint is not None:
            n_bump = min(n_bump, max(6, int(budget_hint // 6)))
        rng.shuffle(country_pids)
        for pid in country_pids[:n_bump]:
            patches.append({
                "op": "person_latent_delta",
                "person_id": pid,
                "delta": {"public_reputation": wave_strength},
            })

        # Strengthen same-country edges
        if len(country_pids) >= 4:
            sample = country_pids[:min(60, len(country_pids))]
            edge_cap = min(30, len(sample) - 1)
            if budget_hint is not None:
                edge_cap = min(edge_cap, max(8, int(budget_hint // 8)))
            for i in range(edge_cap):
                a, b = sample[i], sample[(i + 1) % len(sample)]
                if a == b:
                    continue
                patches.append({
                    "op": "edge_add",
                    "edge_type": "collaboration",
                    "src_id": min(a, b),
                    "dst_id": max(a, b),
                    "weight": _clamp(0.20 + wave_strength * 3.0, 0.18, 0.55),
                    "valid_from": from_year,
                    "reason": f"regional wave: {country} surge",
                    "source_kind": "procedural_regional_wave",
                })
    return patches


def procedural_year_step(world, from_year: int, to_year: int, year_bucket: List[Dict[str, Any]]) -> PatchApplyReport:
    """Deterministic yearly evolution driven by exposure aggregates.

    This is intentionally more global than the previous implementation.  It does
    not try to make one-off movie-local changes only.  Instead it computes yearly
    signals, updates macro regime state, then emits a bounded but broad batch of
    patches affecting people, companies, genre priors and typed relations.
    """
    rep = PatchApplyReport()
    _ensure_temporal_fields(world)
    if not year_bucket:
        rep.log("procedural_year_step: empty year bucket")
        return rep

    rng = random.Random(_stable_hash(f"proc|{from_year}|{to_year}|{len(year_bucket)}"))
    ctx = _build_exposure_context(year_bucket)
    summary = _summarize_regime(world, from_year, ctx)

    patches: List[Dict[str, Any]] = []

    # 1) Genre drift: push current hot genres up and cooler ones gently down.
    total_movies = max(1, ctx.n_movies)
    genre_signal: Dict[str, float] = {}
    for genre, count in ctx.genre_counts.items():
        share = count / total_movies
        genre_signal[genre] = float((share - 0.08) * 0.18)
    for genre, signal in _top_abs_items(genre_signal, limit=10):
        if abs(signal) < 0.01:
            continue
        patches.append({
            "op": "genre_trend_shift",
            "genre": genre,
            "delta": _clamp(signal, -0.04, 0.05),
        })

    # 2) Person latent drift from yearly performance.
    person_signal: Dict[int, Dict[str, float]] = {}
    for pid, vals in ctx.person_perf.items():
        if not vals:
            continue
        avg_rating = float(np.mean([v[0] for v in vals]))
        avg_perf = float(np.mean([v[1] for v in vals]))
        exposure = len(vals)
        rep_delta = _clamp((avg_rating - 6.3) * 0.035 + (avg_perf - 1.0) * 0.03, -0.12, 0.12)
        ambition_delta = _clamp((avg_rating - 6.0) * 0.02 + 0.01 * math.log1p(exposure), -0.08, 0.10)
        controversy_delta = _clamp((0.65 - avg_rating) * 0.03 + max(0.0, 0.55 - avg_perf) * 0.05, -0.06, 0.12)
        volatility_delta = _clamp((avg_perf - 1.0) * 0.02 + (rng.random() - 0.5) * 0.03, -0.05, 0.05)
        signal = max(abs(rep_delta), abs(ambition_delta), abs(controversy_delta), abs(volatility_delta))
        if signal < 0.015:
            continue
        person_signal[pid] = {
            "public_reputation": rep_delta,
            "artistic_ambition": ambition_delta,
            "controversy_score": controversy_delta,
            "volatility": volatility_delta,
            "signal": signal,
        }
    for pid, deltas in sorted(person_signal.items(), key=lambda kv: kv[1]["signal"], reverse=True)[: max(160, len(person_signal) // 8 or 1)]:
        delta_payload = {k: v for k, v in deltas.items() if k in ALLOWED_PERSON_LATENT_FIELDS and abs(v) >= 0.01}
        if delta_payload:
            patches.append({"op": "person_latent_delta", "person_id": pid, "delta": delta_payload})

    # 3) Career stage movement and retirement hazard.
    if getattr(world, "persons", None) is not None:
        person_df = world.persons
        pid_to_row = {int(pid): idx for idx, pid in enumerate(person_df["person_id"].astype(int).tolist())}
        for pid in list(ctx.active_person_ids)[:]:
            row_idx = pid_to_row.get(int(pid))
            if row_idx is None:
                continue
            row = person_df.iloc[row_idx]
            stage = _safe_str(row.get("career_stage"), "prime")
            debut = _safe_int(row.get("debut_year"), from_year - 10)
            career_len = from_year - debut
            vals = ctx.person_perf.get(int(pid), [])
            if vals:
                avg_rating = float(np.mean([v[0] for v in vals]))
                avg_perf = float(np.mean([v[1] for v in vals]))
            else:
                avg_rating, avg_perf = 6.0, 1.0
            if stage == "rising" and career_len >= 4 and (avg_rating >= 6.8 or avg_perf >= 1.2):
                patches.append({"op": "career_stage_transition", "person_id": int(pid), "new_stage": "prime"})
            elif stage == "prime" and career_len >= 11 and (avg_rating >= 7.0 or avg_perf >= 1.15):
                patches.append({"op": "career_stage_transition", "person_id": int(pid), "new_stage": "veteran"})
            elif stage == "veteran" and career_len >= 22 and avg_rating >= 6.7:
                patches.append({"op": "career_stage_transition", "person_id": int(pid), "new_stage": "legend"})
            elif stage in {"legend", "veteran"} and career_len >= 34 and avg_perf < 0.85 and rng.random() < 0.08:
                patches.append({"op": "retire_person", "person_id": int(pid), "year": to_year})

    # 4) Company latent drift and one-step tier transitions.
    company_signal: Dict[int, Tuple[float, float, float]] = {}
    for cid, vals in ctx.company_perf.items():
        avg_rating = float(np.mean([v[0] for v in vals])) if vals else 6.0
        avg_perf = float(np.mean([v[1] for v in vals])) if vals else 1.0
        prestige = _clamp((avg_rating - 6.2) * 0.04, -0.08, 0.10)
        risk = _clamp((avg_perf - 1.0) * 0.05, -0.10, 0.10)
        trend = _clamp((summary["regime_score"] * 0.05) + (avg_perf - 1.0) * 0.03, -0.08, 0.08)
        company_signal[cid] = (prestige, risk, trend)
    for cid, _abs_score in _top_abs_items({cid: max(abs(prestige), abs(risk), abs(trend)) for cid, (prestige, risk, trend) in company_signal.items()}, limit=max(80, len(company_signal) // 5 or 1)):
        p, r, t = company_signal[cid]
        delta_payload = {
            "prestige_score": p,
            "risk_appetite": r,
            "market_trend_sensitivity": t,
        }
        patches.append({"op": "company_latent_delta", "company_id": cid, "delta": delta_payload})

        tier = _company_tier_for(world, cid)
        vals = ctx.company_perf.get(cid, [])
        avg_perf = float(np.mean([v[1] for v in vals])) if vals else 1.0
        avg_rating = float(np.mean([v[0] for v in vals])) if vals else 6.0
        tier_idx = _COMPANY_TIER_ORDER.get(tier, 2)
        if avg_perf >= 1.55 and avg_rating >= 6.8 and tier_idx < len(COMPANY_TIERS) - 1:
            patches.append({"op": "company_tier_transition", "company_id": cid, "new_tier": COMPANY_TIERS[tier_idx + 1]})
        elif avg_perf < 0.70 and avg_rating < 5.8 and tier_idx > 0:
            patches.append({"op": "company_tier_transition", "company_id": cid, "new_tier": COMPANY_TIERS[tier_idx - 1]})
            # A5-FIX: dissolution for lowest-tier companies (was impossible:
            # tested tier_idx == 0 inside an elif tier_idx > 0 block)
            if tier_idx == 1 and rng.random() < 0.12:
                patches.append({"op": "dissolve_company", "company_id": cid, "year": to_year})

    # 5) Relationship evolution from exposure + sampled background churn.
    regime_label = summary.get("regime_label", "neutral")
    patches.extend(_emit_edge_program_patches(world, ctx, from_year, to_year, rng, regime_label=regime_label))

    # 6) Genre cascade: genre shifts propagate to affected people.
    for patch in patches[:]:
        if patch.get("op") == "genre_trend_shift":
            genre = patch.get("genre", "")
            delta = float(patch.get("delta", 0.0))
            patches.extend(_amplify_genre_cascade(world, genre, delta, from_year, rng))

    # 7) Country/regional wave amplification.
    patches.extend(_amplify_regional_wave(world, ctx, from_year, rng))

    # Scale max ops with graph size — scaled up for larger patch volumes.
    graph = getattr(world, "graph", None)
    if graph is not None:
        n_edges = (
            sum(int(meta.get("count", 0)) for meta in getattr(graph, "manifest", {}).get("hot_types", {}).values())
            + int(getattr(graph, "manifest", {}).get("cold_cp_count", 0))
            + int(getattr(graph, "manifest", {}).get("cold_cc_count", 0))
        )
    else:
        n_edges = len(getattr(getattr(world, "edge_graph", None), "edges", []) or [])
    max_ops = max(800, min(250_000, len(patches) + n_edges // 200 + ctx.n_movies * 8))
    apply_rep = apply_world_patches(world, patches, from_year=from_year, to_year=to_year, max_abs_latent_delta=0.18, max_ops=max_ops)
    rep.applied += apply_rep.applied
    rep.skipped += apply_rep.skipped
    rep.errors += apply_rep.errors
    rep.messages.extend(apply_rep.messages)

    rep.log(
        "procedural_year_step: "
        f"year={from_year}->{to_year} regime={regime_label} score={summary['regime_score']:.3f} "
        f"movies={ctx.n_movies} co_pairs={len(ctx.co_counts)} company_pairs={len(ctx.company_pair_counts)} "
        f"patches_emitted={len(patches)} applied={apply_rep.applied}"
    )
    return rep


# ---------------------------------------------------------------------------
# LLM evolution: strategy-plan prompt + deterministic expansion
# ---------------------------------------------------------------------------

def _select_sample_people(
    world,
    year_bucket: Sequence[Dict[str, Any]],
    *,
    max_people: int = 180,
    seed: int = 0,
    strategy: str = "mixed",
) -> List[int]:
    """Sample people with real batch diversity.

    The old implementation was effectively deterministic for the same year bucket.
    This one varies by strategy and seed so multiple batches cover distinct parts
    of the world instead of re-querying the same local neighbourhood.
    """
    rng = random.Random(seed)
    if getattr(world, "persons", None) is None:
        return []
    df = world.persons.copy()
    df["person_id"] = df["person_id"].astype(int)

    active_counts: Counter = Counter()
    for movie in year_bucket:
        director_id = movie.get("director_id")
        if director_id is not None:
            active_counts[int(director_id)] += 2
        for pid in movie.get("cast_ids") or []:
            active_counts[int(pid)] += 1

    def _top_ids(frame, by: str, n: int) -> List[int]:
        if by not in frame.columns:
            return []
        return frame.sort_values(by, ascending=False)["person_id"].astype(int).head(n).tolist()

    active_ids = [pid for pid, _ in active_counts.most_common(max_people)]
    elite_ids = _top_ids(df, "pop_weight", max_people)

    if strategy == "active":
        pool = active_ids
    elif strategy == "elite":
        pool = elite_ids
    elif strategy == "volatile":
        if all(c in df.columns for c in ("person_id",)):
            vol_score = []
            for pid in df["person_id"].astype(int).tolist():
                lv = getattr(world, "person_latent", {}).get(pid, {})
                score = float(lv.get("controversy_score", 0.15)) + float(lv.get("volatility", 0.4))
                vol_score.append((pid, score))
            pool = [pid for pid, _ in sorted(vol_score, key=lambda kv: kv[1], reverse=True)[: max_people * 2]]
        else:
            pool = active_ids
    elif strategy == "bridges":
        communities = getattr(world, "communities", {}) or {}
        by_comm: Dict[int, List[int]] = defaultdict(list)
        for pid in df["person_id"].astype(int).tolist():
            by_comm[int(communities.get(pid, -1))].append(pid)
        pool = []
        for _, members in sorted(by_comm.items(), key=lambda kv: len(kv[1])):
            rng.shuffle(members)
            pool.extend(members[: min(8, len(members))])
        pool.extend(active_ids[: max_people // 2])
    else:
        pool = active_ids[: max_people] + elite_ids[: max_people]  # mixed

    # Add a random-but-seeded tail for diversity.
    remaining = [int(pid) for pid in df["person_id"].astype(int).tolist() if int(pid) not in set(pool)]
    rng.shuffle(remaining)
    merged = list(dict.fromkeys(pool + remaining))
    return merged[:max_people]


def _select_sample_companies(
    world,
    year_bucket: Sequence[Dict[str, Any]],
    *,
    max_companies: int = 80,
    seed: int = 0,
    strategy: str = "mixed",
) -> List[int]:
    rng = random.Random(seed)
    if getattr(world, "companies", None) is None:
        return []
    df = world.companies.copy()
    df["company_id"] = df["company_id"].astype(int)

    active_counts: Counter = Counter()
    for movie in year_bucket:
        for cid in movie.get("company_ids") or []:
            active_counts[int(cid)] += 1
    active_ids = [cid for cid, _ in active_counts.most_common(max_companies)]
    elite_ids = df.sort_values("pop_weight", ascending=False)["company_id"].astype(int).head(max_companies).tolist() if "pop_weight" in df.columns else active_ids

    if strategy == "active":
        pool = active_ids
    elif strategy == "elite":
        pool = elite_ids
    elif strategy == "diverse_tiers" and "tier" in df.columns:
        pool = []
        for tier in COMPANY_TIERS:
            ids = df[df["tier"].astype(str) == tier]["company_id"].astype(int).tolist()
            rng.shuffle(ids)
            pool.extend(ids[: max(3, max_companies // 10)])
        pool.extend(active_ids[: max_companies // 2])
    else:
        pool = active_ids[:max_companies] + elite_ids[:max_companies]

    remaining = [int(cid) for cid in df["company_id"].astype(int).tolist() if int(cid) not in set(pool)]
    rng.shuffle(remaining)
    merged = list(dict.fromkeys(pool + remaining))
    return merged[:max_companies]


def _edges_for_entities(world, people: Sequence[int], companies: Sequence[int], year: int, limit: int = 260) -> List[Dict[str, Any]]:
    graph = getattr(world, "graph", None)
    if graph is None:
        return []
    return graph.sample_edges_for_entities(people, companies, year, limit=limit)


def build_llm_evolution_prompt(
    world,
    from_year: int,
    to_year: int,
    year_bucket: List[Dict[str, Any]],
    *,
    seed: int,
    strategy: str,
) -> str:
    ctx = _build_exposure_context(year_bucket)
    people = _select_sample_people(world, year_bucket, max_people=180, seed=seed, strategy=strategy)
    companies = _select_sample_companies(world, year_bucket, max_companies=80, seed=seed ^ 7919, strategy=("diverse_tiers" if strategy == "bridges" else strategy))
    edges = _edges_for_entities(world, people, companies, from_year)

    people_rows = []
    if getattr(world, "persons", None) is not None:
        pdf = world.persons.set_index("person_id", drop=False)
        for pid in people:
            if pid not in pdf.index:
                continue
            row = pdf.loc[pid]
            lv = getattr(world, "person_latent", {}).get(int(pid), {})
            people_rows.append({
                "person_id": int(pid),
                "name": row.get("name"),
                "career_stage": row.get("career_stage"),
                "genre_affinity": row.get("genre_affinity"),
                "style_tags": row.get("style_tags"),
                "pop_weight": float(row.get("pop_weight", 0.1) or 0.1),
                "public_reputation": float(lv.get("public_reputation", 0.5)),
                "controversy_score": float(lv.get("controversy_score", 0.15)),
                "volatility": float(lv.get("volatility", 0.4)),
            })

    company_rows = []
    if getattr(world, "companies", None) is not None:
        cdf = world.companies.set_index("company_id", drop=False)
        for cid in companies:
            if cid not in cdf.index:
                continue
            row = cdf.loc[cid]
            lv = getattr(world, "company_latent", {}).get(int(cid), {})
            company_rows.append({
                "company_id": int(cid),
                "name": row.get("name"),
                "tier": row.get("tier"),
                "specialty_genres": row.get("specialty_genres"),
                "pop_weight": float(row.get("pop_weight", 0.3) or 0.3),
                "prestige_score": float(lv.get("prestige_score", 0.5)),
                "risk_appetite": float(lv.get("risk_appetite", 0.5)),
            })

    prompt_schema = {
        "narrative": "1-2 sentence summary",
        "macro_shifts": {
            "genre_deltas": [{"genre": "Drama", "delta": 0.03}],
            "company_tier_moves": [{"company_id": 1, "new_tier": "Major"}],
            "retirements": [{"person_id": 2, "year": to_year}],
            "dissolutions": [{"company_id": 7, "year": to_year}],
        },
        "person_shocks": [
            {
                "person_id": 10,
                "reputation_delta": 0.07,
                "controversy_delta": -0.03,
                "ambition_delta": 0.04,
                "volatility_delta": 0.01,
                "career_stage": "veteran",
            }
        ],
        "relationship_programs": [
            {
                "edge_type": "friendship",
                "mode": "create",
                "anchor_ids": [10, 11],
                "scope": "active_collaborators",
                "target_count": 12,
                "weight": 0.42,
                "delta_weight": 0.12,
                "reason": "brief reason"
            }
        ]
    }

    return f"""
You are planning one year of temporal evolution for a synthetic film-industry world.

Return ONLY valid JSON. No markdown. No prose before or after the JSON.

Your job is NOT to enumerate hundreds of raw graph ops.
Your job IS to propose a compact yearly strategy plan that can be expanded
programmatically.

Return JSON with exactly this top-level shape:
{json.dumps(prompt_schema, ensure_ascii=False)}

Rules:
- Only reference person_id and company_id values present in the payload.
- Keep all deltas modest. The engine will expand them globally.
- Prefer broad, plausible shifts over quirky one-off rewiring.
- relationship_programs should describe *patterns* to apply, not hand-crafted pairs.
- Valid relationship scopes: active_collaborators, same_community, cross_community,
  actor_company, company_pairs, competitors.
- Valid modes: create, strengthen, expire.
- Valid edge_type: friendship, rivalry, mentorship, collaboration, avoid,
  chemistry, brand_fit, employment, co_production, market_rival.
- target_count should be in [4, 30].
- person_shocks should be 4-20 entries total.
- relationship_programs should be 3-10 entries total.
- macro_shifts genre deltas must stay in [-0.05, 0.05].

Context:
- evolve from year {from_year} to {to_year}
- batch strategy: {strategy}
- movies this year: {ctx.n_movies}
- avg_rating: {ctx.avg_rating:.3f}
- avg_box_office_over_budget: {ctx.avg_perf:.3f}
- top genres: {ctx.genre_counts.most_common(6)}
- active sampled people: {json.dumps(people_rows, ensure_ascii=False)}
- active sampled companies: {json.dumps(company_rows, ensure_ascii=False)}
- sampled active edges: {json.dumps(edges, ensure_ascii=False)}

Output JSON now.
""".strip()


def _iter_candidate_pairs_for_program(
    world,
    ctx: ExposureContext,
    program: Dict[str, Any],
    rng: random.Random,
) -> List[Tuple[int, int]]:
    edge_type = _safe_str(program.get("edge_type"), "friendship")
    scope = _safe_str(program.get("scope"), "active_collaborators")
    anchors = [int(x) for x in (program.get("anchor_ids") or []) if x]
    pairs: List[Tuple[int, int]] = []

    if scope == "active_collaborators":
        for anchor in anchors:
            for (a, b), _ in ctx.co_counts.most_common(500):
                if anchor in (a, b):
                    pairs.append((a, b))
            for (a, b), _ in ctx.company_pair_counts.most_common(200):
                if edge_type in {"co_production", "market_rival"} and anchor in (a, b):
                    pairs.append((a, b))
    elif scope == "competitors":
        for anchor in anchors:
            for (a, b), _ in ctx.competition_counts.most_common(600):
                if anchor in (a, b):
                    pairs.append((a, b))
    elif scope == "same_community":
        communities = getattr(world, "communities", {}) or {}
        people = list(ctx.active_person_ids)
        for anchor in anchors:
            comm = communities.get(anchor)
            if comm is None:
                continue
            cohort = [pid for pid in people if communities.get(pid) == comm and pid != anchor]
            rng.shuffle(cohort)
            for pid in cohort[:40]:
                pairs.append(_pair_key(anchor, pid))
    elif scope == "cross_community":
        communities = getattr(world, "communities", {}) or {}
        people = list(ctx.active_person_ids)
        for anchor in anchors:
            comm = communities.get(anchor)
            cohort = [pid for pid in people if pid != anchor and communities.get(pid) != comm]
            rng.shuffle(cohort)
            for pid in cohort[:40]:
                pairs.append(_pair_key(anchor, pid))
    elif scope == "actor_company":
        for anchor in anchors:
            for (cid, pid), _ in ctx.company_person_counts.most_common(700):
                if anchor in (cid, pid):
                    pairs.append((cid, pid))
    elif scope == "company_pairs":
        for anchor in anchors:
            for (a, b), _ in ctx.company_pair_counts.most_common(400):
                if anchor in (a, b):
                    pairs.append((a, b))

    deduped = list(dict.fromkeys(pairs))
    rng.shuffle(deduped)
    return deduped


def _expand_relationship_programs(
    world,
    ctx: ExposureContext,
    programs: Sequence[Dict[str, Any]],
    from_year: int,
    to_year: int,
    batch_seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(batch_seed)
    patches: List[Dict[str, Any]] = []
    for idx, program in enumerate(programs):
        if not isinstance(program, dict):
            continue
        edge_type = _safe_str(program.get("edge_type"), "").strip()
        mode = _safe_str(program.get("mode"), "create").strip()
        if edge_type not in ALLOWED_EDGE_TYPES or mode not in {"create", "strengthen", "expire"}:
            continue
        target_count = max(1, min(30, _safe_int(program.get("target_count"), 8)))
        create_weight = _clamp(program.get("weight", 0.35), 0.08, 0.95)
        delta_weight = _clamp(program.get("delta_weight", 0.10), -0.30, 0.30)
        reason = _safe_str(program.get("reason"), f"llm program {idx}")
        for pair in _iter_candidate_pairs_for_program(world, ctx, program, rng)[:target_count]:
            src, dst = int(pair[0]), int(pair[1])
            if mode == "create":
                patches.append({
                    "op": "edge_add",
                    "edge_type": edge_type,
                    "src_id": src,
                    "dst_id": dst,
                    "weight": create_weight,
                    "valid_from": from_year,
                    "reason": reason,
                    "source_kind": "llm_program",
                })
            elif mode == "strengthen":
                patches.append({
                    "op": "edge_update",
                    "edge_type": edge_type,
                    "src_id": src,
                    "dst_id": dst,
                    "delta_weight": abs(delta_weight),
                    "reason": reason,
                })
            else:
                patches.append({
                    "op": "edge_expire",
                    "edge_type": edge_type,
                    "src_id": src,
                    "dst_id": dst,
                    "year": from_year,
                    "reason": reason,
                })
    return patches


def _patches_from_strategy_plan(
    world,
    ctx: ExposureContext,
    parsed: Dict[str, Any],
    from_year: int,
    to_year: int,
    *,
    batch_seed: int,
) -> List[Dict[str, Any]]:
    patches: List[Dict[str, Any]] = []
    macro = parsed.get("macro_shifts") or {}
    if isinstance(macro, dict):
        for row in macro.get("genre_deltas") or []:
            if isinstance(row, dict):
                patches.append({
                    "op": "genre_trend_shift",
                    "genre": _safe_str(row.get("genre"), "").strip(),
                    "delta": _clamp(row.get("delta", 0.0), -0.05, 0.05),
                })
        for row in macro.get("company_tier_moves") or []:
            if isinstance(row, dict):
                patches.append({
                    "op": "company_tier_transition",
                    "company_id": _safe_int(row.get("company_id")),
                    "new_tier": _safe_str(row.get("new_tier"), "").strip(),
                })
        for row in macro.get("retirements") or []:
            if isinstance(row, dict):
                patches.append({
                    "op": "retire_person",
                    "person_id": _safe_int(row.get("person_id")),
                    "year": max(from_year, min(to_year, _safe_int(row.get("year"), to_year))),
                })
        for row in macro.get("dissolutions") or []:
            if isinstance(row, dict):
                patches.append({
                    "op": "dissolve_company",
                    "company_id": _safe_int(row.get("company_id")),
                    "year": max(from_year, min(to_year, _safe_int(row.get("year"), to_year))),
                })

    for row in parsed.get("person_shocks") or []:
        if not isinstance(row, dict):
            continue
        pid = _safe_int(row.get("person_id"))
        delta = {
            "public_reputation": _clamp(row.get("reputation_delta", 0.0), -0.15, 0.15),
            "controversy_score": _clamp(row.get("controversy_delta", 0.0), -0.15, 0.15),
            "artistic_ambition": _clamp(row.get("ambition_delta", 0.0), -0.12, 0.12),
            "volatility": _clamp(row.get("volatility_delta", 0.0), -0.10, 0.10),
        }
        if any(abs(v) >= 0.01 for v in delta.values()):
            patches.append({"op": "person_latent_delta", "person_id": pid, "delta": delta})
        new_stage = _safe_str(row.get("career_stage"), "").strip()
        if new_stage:
            patches.append({"op": "career_stage_transition", "person_id": pid, "new_stage": new_stage})

    patches.extend(_expand_relationship_programs(world, ctx, parsed.get("relationship_programs") or [], from_year, to_year, batch_seed=batch_seed))
    return patches


def llm_year_step(
    world,
    from_year: int,
    to_year: int,
    year_bucket: List[Dict[str, Any]],
    model: Optional[str] = None,
    log_dir: Optional[str] = None,
    api_key_env: str = "GEMINI_API_KEY",  # kept for signature compat, unused
) -> PatchApplyReport:
    rep = PatchApplyReport()
    _ensure_temporal_fields(world)

    if not year_bucket:
        rep.log("llm_year_step: empty year bucket")
        return rep

    try:
        llm = get_llm_client()
    except Exception as exc:
        rep.skipped += 1
        rep.log(f"LLM client unavailable: {exc}")
        return rep

    if model is None:
        try:
            from contracts import MODEL_TIERS
            model = MODEL_TIERS.get("temporal_evolution", None)
        except ImportError:
            pass

    ctx = _build_exposure_context(year_bucket)
    n_persons = len(world.persons) if getattr(world, "persons", None) is not None else 10000
    graph = getattr(world, "graph", None)
    if graph is not None:
        n_edges = (
            sum(int(meta.get("count", 0)) for meta in getattr(graph, "manifest", {}).get("hot_types", {}).values())
            + int(getattr(graph, "manifest", {}).get("cold_cp_count", 0))
            + int(getattr(graph, "manifest", {}).get("cold_cc_count", 0))
        )
    else:
        n_edges = len(getattr(getattr(world, "edge_graph", None), "edges", []) or [])
    n_batches = max(3, min(8, n_persons // 10000 + 3))
    strategies = ["active", "elite", "volatile", "bridges", "mixed", "active", "elite", "bridges"]

    batch_reports = []
    for batch_idx in range(n_batches):
        strategy = strategies[batch_idx % len(strategies)]
        batch_seed = _stable_hash(f"llm|{from_year}|{to_year}|{batch_idx}|{strategy}|{ctx.n_movies}")
        prompt = build_llm_evolution_prompt(world, from_year, to_year, year_bucket, seed=batch_seed, strategy=strategy)
        raw_text = ""
        parsed = None
        try:
            response = llm.generate(
                prompt,
                model=model,
                json_mode=True,
                temperature=0.60 + 0.05 * (batch_idx % 3),
                timeout_sec=80,
                max_attempts=5,
                on_retry=lambda attempt, total, exc, sleep_for: rep.log(
                    f"LLM batch {batch_idx} retry {attempt}/{total} in {sleep_for:.1f}s: {exc}"
                ),
            )
            raw_text = response.text.strip()
            parsed = _safe_json_loads(raw_text)
        except Exception as exc:
            rep.errors += 1
            rep.log(f"LLM batch {batch_idx} failed: {exc}")
            continue

        if not isinstance(parsed, dict):
            rep.skipped += 1
            rep.log(f"LLM batch {batch_idx}: parsed output is not a dict")
            continue

        patches = _patches_from_strategy_plan(world, ctx, parsed, from_year, to_year, batch_seed=batch_seed)
        max_ops = max(180, min(2400, 120 + n_edges // 2000 + len(patches)))
        apply_rep = apply_world_patches(world, patches, from_year=from_year, to_year=to_year, max_abs_latent_delta=0.18, max_ops=max_ops)
        rep.applied += apply_rep.applied
        rep.skipped += apply_rep.skipped
        rep.errors += apply_rep.errors
        rep.messages.extend(apply_rep.messages)

        # V18 compat: route LLMMasterClass actions (career_pause, boost_person,
        # change_genre_affinity) that aren't handled by apply_world_patches.
        try:
            from llm_master import LLMMasterClass, _LEGACY_OPS
            _person_shocks = parsed.get("person_shocks") or []
            llm_actions = []
            for shock in _person_shocks:
                if not isinstance(shock, dict):
                    continue
                # Check for LLMMasterClass-specific fields
                if shock.get("career_pause_years"):
                    llm_actions.append({"action": "career_pause", "params": {
                        "person_id": _safe_int(shock.get("person_id")),
                        "duration_years": _safe_int(shock.get("career_pause_years"), 2),
                    }})
                if shock.get("boost_multiplier"):
                    llm_actions.append({"action": "boost_person", "params": {
                        "person_id": _safe_int(shock.get("person_id")),
                        "multiplier": _clamp(shock.get("boost_multiplier", 1.5), 1.0, 3.0),
                        "duration_years": _safe_int(shock.get("boost_duration_years"), 2),
                    }})
                if shock.get("genre_affinity_change"):
                    ga = shock["genre_affinity_change"]
                    if isinstance(ga, dict):
                        for genre, delta in ga.items():
                            llm_actions.append({"action": "change_genre_affinity", "params": {
                                "person_id": _safe_int(shock.get("person_id")),
                                "genre": str(genre),
                                "delta": float(_clamp(delta, -0.5, 0.5)),
                            }})
            if llm_actions:
                master = LLMMasterClass(world)
                action_rep = master.execute({"actions": llm_actions}, year=from_year)
                rep.applied += action_rep.applied
                rep.skipped += action_rep.skipped
                rep.errors += action_rep.errors
                rep.messages.extend(action_rep.skipped_reasons)
        except ImportError:
            pass  # LLMMasterClass not available, skip extended actions
        narrative = _safe_str(parsed.get("narrative"), "").strip()
        batch_reports.append({
            "batch_idx": batch_idx,
            "strategy": strategy,
            "patches_emitted": len(patches),
            "applied": apply_rep.applied,
            "skipped": apply_rep.skipped,
            "errors": apply_rep.errors,
            "narrative": narrative,
        })

        if log_dir:
            try:
                Path(log_dir).mkdir(parents=True, exist_ok=True)
                stem = Path(log_dir) / f"evolution_{from_year}_{batch_idx}_{strategy}_{_now_ts()}"
                stem.with_suffix(".prompt.txt").write_text(prompt, encoding="utf-8")
                stem.with_suffix(".raw.json").write_text(raw_text, encoding="utf-8")
                stem.with_suffix(".report.json").write_text(json.dumps(batch_reports[-1], ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as exc:
                rep.log(f"Failed to write LLM evolution logs for batch {batch_idx}: {exc}")

    world.temporal_state["llm_history"].append({
        "year": int(from_year),
        "to_year": int(to_year),
        "batches": batch_reports,
    })

    rep.log(f"llm_year_step: year={from_year}->{to_year} batches={len(batch_reports)} applied={rep.applied} errors={rep.errors}")
    return rep


# ---------------------------------------------------------------------------
# Unified active-path compatibility shims
# ---------------------------------------------------------------------------


def procedural_year_step(world, from_year: int, to_year: int, year_bucket: List[Dict[str, Any]]) -> PatchApplyReport:  # type: ignore[no-redef]
    """Compatibility shim to the unified yearly planner.

    The active Mirage runtime now evolves years only through `year_planner`.
    We keep these public names for compatibility, but they no longer run the
    legacy high-overhead execution path above.
    """

    from year_planner import procedural_year_step as _active_procedural_year_step

    return _active_procedural_year_step(world, from_year, to_year, year_bucket)


def llm_year_step(  # type: ignore[no-redef]
    world,
    from_year: int,
    to_year: int,
    year_bucket: List[Dict[str, Any]],
    model: Optional[str] = None,
    log_dir: Optional[str] = None,
    api_key_env: str = "GEMINI_API_KEY",
) -> PatchApplyReport:
    """Compatibility shim to the unified yearly planner."""

    from year_planner import llm_year_step as _active_llm_year_step

    return _active_llm_year_step(world, from_year, to_year, year_bucket, model=model, log_dir=log_dir, api_key_env=api_key_env)
