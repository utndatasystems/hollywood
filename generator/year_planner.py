from __future__ import annotations

import json
import random
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from big_history_events import roll_events
from graph_runtime import COLD_CC_TYPES, COLD_CP_TYPES, HOT_DIRECTED_TYPES, HOT_EDGE_TYPES, HOT_UNDIRECTED_TYPES
from llm_provider import get_llm_client
from temporal_evolution_api import (
    ALLOWED_COMPANY_LATENT_FIELDS,
    ALLOWED_EDGE_TYPES,
    ALLOWED_PERSON_LATENT_FIELDS,
    COMPANY_TIERS,
    ExposureContext,
    PatchApplyReport,
    _CAREER_STAGE_ORDER,
    _COMPANY_TIER_ORDER,
    _EDGE_ENTITY_TYPES,
    _amplify_genre_cascade,
    _amplify_regional_wave,
    _build_exposure_context,
    _build_person_stage_cache,
    _clamp,
    _company_tier_for,
    _emit_edge_program_patches,
    _ensure_temporal_fields,
    _invalidate_year_cache,
    _safe_int,
    _safe_json_loads,
    _safe_str,
    _stable_hash,
    _summarize_regime,
)

POSITIVE_EDGE_TYPES = {
    "friendship",
    "collaboration",
    "chemistry",
    "clique",
    "former_collaborator",
    "mentorship",
    "brand_fit",
    "employment",
    "exclusive_deal",
    "co_production",
    "subsidiary",
}
NEGATIVE_EDGE_TYPES = {"rivalry", "avoid", "blacklist", "market_rival"}
SOCIAL_EDGE_TYPES = HOT_EDGE_TYPES
COMPANY_EDGE_TYPES = COLD_CP_TYPES | COLD_CC_TYPES
REACH_WEIGHTS = {"local": 0.90, "cohort": 1.0, "regional": 1.15, "industry": 1.30}
NEGATIVE_SHARE_BANDS = {
    "boom": (0.15, 0.30),
    "neutral": (0.20, 0.35),
    "stress": (0.30, 0.45),
}
MOTIF_EVENT_HINTS = {
    "scandal_fallout": {"scandal", "award_controversy"},
    "regional_migration_wave": {"country_emergence"},
    "company_competition_shift": {"studio_merger", "market_crash", "tech_disruption"},
    "blacklist_cascade": {"scandal", "market_crash"},
    "prestige_alliance": {"genre_boom", "streaming_revolution"},
}
MOTIF_BLUEPRINTS: dict[str, tuple[tuple[str, str, float], ...]] = {
    "friendship_clustering": (
        ("friendship", "create", 0.52),
        ("collaboration", "strengthen", 0.28),
        ("clique", "create", 0.20),
    ),
    "rivalry_wave": (("rivalry", "create", 0.72), ("avoid", "create", 0.28)),
    "mentor_tree": (("mentorship", "create", 0.72), ("friendship", "create", 0.28)),
    "director_camp": (
        ("collaboration", "strengthen", 0.45),
        ("mentorship", "create", 0.30),
        ("former_collaborator", "create", 0.25),
    ),
    "recruiting_wave": (("brand_fit", "create", 0.68), ("employment", "create", 0.32)),
    "blacklist_cascade": (
        ("blacklist", "create", 0.48),
        ("avoid", "create", 0.22),
        ("employment", "expire", 0.30),
    ),
    "prestige_alliance": (
        ("co_production", "create", 0.48),
        ("brand_fit", "strengthen", 0.22),
        ("friendship", "create", 0.30),
    ),
    "bridge_building": (
        ("friendship", "create", 0.50),
        ("collaboration", "create", 0.25),
        ("co_production", "create", 0.25),
    ),
    "company_competition_shift": (
        ("market_rival", "create", 0.48),
        ("co_production", "expire", 0.20),
        ("brand_fit", "weaken", 0.32),
    ),
    "regional_migration_wave": (
        ("friendship", "create", 0.32),
        ("collaboration", "create", 0.22),
        ("brand_fit", "create", 0.26),
        ("co_production", "create", 0.20),
    ),
    "scandal_fallout": (
        ("avoid", "create", 0.40),
        ("rivalry", "create", 0.25),
        ("blacklist", "create", 0.20),
        ("employment", "expire", 0.15),
    ),
}


def _speed_scope(
    speed_audit,
    name: str,
    *,
    category: str = "",
    units: int = 0,
    metadata: dict[str, Any] | None = None,
    note: str = "",
):
    if speed_audit is None:
        return nullcontext(None)
    return speed_audit.track(name, category=category, units=units, metadata=metadata, note=note)


def _planner_priors(world):
    return getattr(getattr(getattr(world, "workspace", None), "config", None), "priors", None)


def _planner_prior_float(world, name: str, default: float, lo: float | None = None, hi: float | None = None) -> float:
    priors = _planner_priors(world)
    try:
        value = float(getattr(priors, name, default)) if priors is not None else float(default)
    except Exception:
        value = float(default)
    if lo is not None:
        value = max(float(lo), value)
    if hi is not None:
        value = min(float(hi), value)
    return float(value)


@dataclass(slots=True)
class RelationshipProgram:
    motif: str
    edge_type: str
    mode: str = "create"
    target_cohort: str = "active_people"
    graph_scope: str = "local_frontier"
    anchors_people: list[int] = field(default_factory=list)
    anchors_companies: list[int] = field(default_factory=list)
    intensity: float = 0.45
    reach: str = "cohort"
    target_size: int = 0
    fanout: int = 4
    max_depth: int = 1
    weight_mean: float = 0.35
    weight_spread: float = 0.08
    delta_weight: float = 0.10
    decay: float = 0.20
    reciprocity_rate: float = 0.0
    closure_rate: float = 0.0
    cross_community_bias: float = 0.25
    novelty_bias: float = 0.30
    durability_years: int = 3
    reason: str = ""


@dataclass(slots=True)
class GraphScaleProfile:
    hot_total: int
    cold_cp_total: int
    cold_cc_total: int
    hot_family_counts: dict[str, int]
    n_movies: int
    active_people: int
    active_companies: int
    regime_label: str
    triggered_event_count: int
    co_count: int
    competition_count: int
    director_actor_count: int
    company_person_count: int
    company_pair_count: int
    hot_genre_count: int
    hot_country_count: int

    @property
    def total_edges(self) -> int:
        return int(self.hot_total + self.cold_cp_total + self.cold_cc_total)


@dataclass(slots=True)
class YearEdgeBudget:
    social_program_budget: int
    company_program_budget: int
    background_budget: int
    cascade_budget: int

    @property
    def total_budget(self) -> int:
        return int(self.social_program_budget + self.company_program_budget + self.background_budget + self.cascade_budget)


@dataclass(slots=True)
class ProgramAllocation:
    program: RelationshipProgram
    targeted_ops: int
    family_budgets: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class YearEvolutionContext:
    from_year: int
    to_year: int
    exposure: ExposureContext
    summary: dict[str, Any]
    triggered_events: list[dict[str, Any]]
    sampled_people: list[int]
    sampled_companies: list[int]
    sampled_edges: list[dict[str, Any]]
    top_people: list[int]
    top_companies: list[int]
    execution_cache: "YearExecutionCache | None" = None
    scale_profile: GraphScaleProfile | None = None
    edge_budget: YearEdgeBudget | None = None


@dataclass(slots=True)
class YearExecutionCache:
    coappearance_pairs: list[tuple[int, int]] = field(default_factory=list)
    competition_pairs: list[tuple[int, int]] = field(default_factory=list)
    director_actor_pairs: list[tuple[int, int]] = field(default_factory=list)
    company_person_pairs: list[tuple[int, int]] = field(default_factory=list)
    company_company_pairs: list[tuple[int, int]] = field(default_factory=list)
    top_similarity_pairs: list[tuple[int, int]] = field(default_factory=list)
    cross_community_bridge_pairs: list[tuple[int, int]] = field(default_factory=list)
    cohort_people: dict[str, list[int]] = field(default_factory=dict)
    cohort_companies: dict[str, list[int]] = field(default_factory=dict)
    anchor_frontiers: dict[tuple[Any, ...], list[int]] = field(default_factory=dict)
    genre_hotspot_people: dict[str, list[int]] = field(default_factory=dict)
    country_hotspot_people: dict[str, list[int]] = field(default_factory=dict)
    active_edge_samples_by_family: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    stale_edge_samples_by_family: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    person_row_idx: dict[int, int] = field(default_factory=dict)
    person_col_idx: dict[str, int] = field(default_factory=dict)
    company_row_idx: dict[int, int] = field(default_factory=dict)
    company_col_idx: dict[str, int] = field(default_factory=dict)
    person_country: dict[int, str] = field(default_factory=dict)
    person_genres: dict[int, tuple[str, ...]] = field(default_factory=dict)
    company_country: dict[int, str] = field(default_factory=dict)
    company_genres: dict[int, tuple[str, ...]] = field(default_factory=dict)
    communities_by_person: dict[int, int] = field(default_factory=dict)
    person_stage_cache: dict[int, str] = field(default_factory=dict)


@dataclass(slots=True)
class YearPlan:
    market_regime_moves: list[dict[str, Any]] = field(default_factory=list)
    person_shocks: list[dict[str, Any]] = field(default_factory=list)
    company_moves: list[dict[str, Any]] = field(default_factory=list)
    relationship_programs: list[RelationshipProgram] = field(default_factory=list)
    world_event_narrative: str = ""
    planner_source: str = "deterministic"
    raw_text: str = ""
    parse_error: str | None = None


@dataclass(slots=True)
class YearProgram:
    person_latent_deltas: dict[int, dict[str, float]] = field(default_factory=dict)
    company_latent_deltas: dict[int, dict[str, float]] = field(default_factory=dict)
    career_stage_updates: dict[int, str] = field(default_factory=dict)
    retirements: list[tuple[int, int]] = field(default_factory=list)
    company_tier_updates: dict[int, str] = field(default_factory=dict)
    dissolutions: list[tuple[int, int]] = field(default_factory=list)
    genre_deltas: dict[str, float] = field(default_factory=dict)
    country_multipliers: dict[str, float] = field(default_factory=dict)
    edge_ops: list[dict[str, Any]] = field(default_factory=list)
    world_events: list[dict[str, Any]] = field(default_factory=list)
    telemetry: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class YearApplyReport:
    applied: int = 0
    skipped: int = 0
    errors: int = 0
    messages: list[str] = field(default_factory=list)
    triggered_events: list[str] = field(default_factory=list)
    planner_source: str = "deterministic"
    edge_applied_by_family: dict[str, int] = field(default_factory=dict)
    edge_applied_by_mode: dict[str, int] = field(default_factory=dict)
    invalidated_actor_cache: bool = False
    invalidated_company_cache: bool = False

    def log(self, msg: str) -> None:
        self.messages.append(str(msg))


def _rank_people(world, ctx: ExposureContext, limit: int = 72) -> list[int]:
    scores: list[tuple[int, float]] = []
    for pid, vals in ctx.person_perf.items():
        if not vals:
            continue
        avg_rating = float(np.mean([v[0] for v in vals]))
        avg_perf = float(np.mean([v[1] for v in vals]))
        exposure = float(len(vals))
        lv = getattr(world, "person_latent", {}).get(int(pid), {})
        volatility = float(lv.get("volatility", 0.35) or 0.35)
        controversy = float(lv.get("controversy_score", 0.15) or 0.15)
        score = exposure * 0.55 + avg_rating * 2.5 + avg_perf * 3.0 + volatility * 4.0 + controversy * 3.0
        scores.append((int(pid), score))
    scores.sort(key=lambda item: item[1], reverse=True)
    return [pid for pid, _ in scores[:limit]]


def _rank_companies(world, ctx: ExposureContext, limit: int = 32) -> list[int]:
    scores: list[tuple[int, float]] = []
    for cid, vals in ctx.company_perf.items():
        avg_rating = float(np.mean([v[0] for v in vals])) if vals else 6.0
        avg_perf = float(np.mean([v[1] for v in vals])) if vals else 1.0
        lv = getattr(world, "company_latent", {}).get(int(cid), {})
        prestige = float(lv.get("prestige_score", 0.5) or 0.5)
        risk = float(lv.get("risk_appetite", 0.5) or 0.5)
        score = len(vals) * 0.70 + avg_rating * 1.5 + avg_perf * 2.5 + prestige * 3.0 + risk
        scores.append((int(cid), score))
    scores.sort(key=lambda item: item[1], reverse=True)
    return [cid for cid, _ in scores[:limit]]


def _sample_people_for_prompt(world, ranked_people: Sequence[int], seed: int, limit: int = 90) -> list[int]:
    ranked = list(dict.fromkeys(int(pid) for pid in ranked_people if pid))
    if getattr(world, "persons", None) is None or "person_id" not in world.persons.columns:
        return ranked[:limit]
    rng = random.Random(seed)
    all_people = world.persons["person_id"].astype(int).tolist()
    remaining = [pid for pid in all_people if pid not in set(ranked)]
    rng.shuffle(remaining)
    return (ranked[: limit // 2] + remaining[:limit])[:limit]


def _sample_companies_for_prompt(world, ranked_companies: Sequence[int], seed: int, limit: int = 36) -> list[int]:
    ranked = list(dict.fromkeys(int(cid) for cid in ranked_companies if cid))
    if getattr(world, "companies", None) is None or "company_id" not in world.companies.columns:
        return ranked[:limit]
    rng = random.Random(seed ^ 7919)
    all_companies = world.companies["company_id"].astype(int).tolist()
    remaining = [cid for cid in all_companies if cid not in set(ranked)]
    rng.shuffle(remaining)
    return (ranked[: limit // 2] + remaining[:limit])[:limit]


def _sample_edges_for_prompt(world, people: Sequence[int], companies: Sequence[int], year: int, limit: int = 120) -> list[dict[str, Any]]:
    graph = getattr(world, "graph", None)
    if graph is None:
        return []
    return graph.sample_edges_for_entities(people, companies, year, limit=limit)


def _normalize_multi(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, list):
        return tuple(str(item).strip() for item in raw if str(item).strip())
    if isinstance(raw, str):
        return tuple(item.strip() for item in raw.replace(";", ",").split(",") if item.strip())
    return ()


def _build_year_execution_cache(world, exposure: ExposureContext) -> YearExecutionCache:
    cache = YearExecutionCache(
        coappearance_pairs=[pair for pair, _ in exposure.co_counts.most_common(4096)],
        competition_pairs=[pair for pair, _ in exposure.competition_counts.most_common(3072)],
        director_actor_pairs=[pair for pair, _ in exposure.director_actor_counts.most_common(3072)],
        company_person_pairs=[pair for pair, _ in exposure.company_person_counts.most_common(4096)],
        company_company_pairs=[pair for pair, _ in exposure.company_pair_counts.most_common(2048)],
        person_stage_cache=_build_person_stage_cache(world),
        communities_by_person={int(pid): int(cid) for pid, cid in (getattr(world, "communities", {}) or {}).items()},
    )
    if getattr(world, "persons", None) is not None and "person_id" in world.persons.columns:
        pids = world.persons["person_id"].astype(int)
        cache.person_row_idx = {int(pid): i for i, pid in enumerate(pids)}
        cache.person_col_idx = {c: i for i, c in enumerate(world.persons.columns)}
        if "nationality" in world.persons.columns:
            cache.person_country = {int(pid): _safe_str(nat, "").strip().lower() for pid, nat in zip(world.persons["person_id"].astype(int), world.persons["nationality"])}
        if "genre_affinity" in world.persons.columns:
            cache.person_genres = {int(pid): _normalize_multi(genre) for pid, genre in zip(world.persons["person_id"].astype(int), world.persons["genre_affinity"])}
    if getattr(world, "companies", None) is not None and "company_id" in world.companies.columns:
        cids = world.companies["company_id"].astype(int)
        cache.company_row_idx = {int(cid): i for i, cid in enumerate(cids)}
        cache.company_col_idx = {c: i for i, c in enumerate(world.companies.columns)}
        if "country" in world.companies.columns:
            cache.company_country = {int(cid): _safe_str(country, "").strip().lower() for cid, country in zip(world.companies["company_id"].astype(int), world.companies["country"])}
        if "specialty_genres" in world.companies.columns:
            cache.company_genres = {int(cid): _normalize_multi(genres) for cid, genres in zip(world.companies["company_id"].astype(int), world.companies["specialty_genres"])}
    return cache


def build_year_context(world, from_year: int, to_year: int, year_bucket: Sequence[dict[str, Any]]) -> YearEvolutionContext:
    exposure = _build_exposure_context(year_bucket)
    summary = _summarize_regime(world, from_year, exposure)
    rng = random.Random(_stable_hash(f"year_ctx|{from_year}|{to_year}|{exposure.n_movies}"))
    triggered_events = roll_events(from_year, rng)
    ranked_people = _rank_people(world, exposure, limit=72)
    ranked_companies = _rank_companies(world, exposure, limit=32)
    sampled_people = _sample_people_for_prompt(world, ranked_people, seed=_stable_hash(f"year_people|{from_year}"))
    sampled_companies = _sample_companies_for_prompt(world, ranked_companies, seed=_stable_hash(f"year_companies|{from_year}"))
    sampled_edges = _sample_edges_for_prompt(world, sampled_people, sampled_companies, from_year, limit=120)
    return YearEvolutionContext(
        from_year=int(from_year),
        to_year=int(to_year),
        exposure=exposure,
        summary=summary,
        triggered_events=triggered_events,
        sampled_people=sampled_people,
        sampled_companies=sampled_companies,
        sampled_edges=sampled_edges,
        top_people=ranked_people,
        top_companies=ranked_companies,
        execution_cache=_build_year_execution_cache(world, exposure),
    )


def _compact_people_payload(world, people: Sequence[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if getattr(world, "persons", None) is None:
        return rows
    pdf = world.persons.set_index("person_id", drop=False)
    for pid in people:
        if pid not in pdf.index:
            continue
        row = pdf.loc[pid]
        lv = getattr(world, "person_latent", {}).get(int(pid), {})
        rows.append(
            {
                "person_id": int(pid),
                "name": row.get("name"),
                "career_stage": row.get("career_stage"),
                "genre_affinity": row.get("genre_affinity"),
                "style_tags": row.get("style_tags"),
                "pop_weight": round(float(row.get("pop_weight", 0.1) or 0.1), 4),
                "public_reputation": round(float(lv.get("public_reputation", 0.5) or 0.5), 4),
                "controversy_score": round(float(lv.get("controversy_score", 0.15) or 0.15), 4),
                "volatility": round(float(lv.get("volatility", 0.35) or 0.35), 4),
            }
        )
    return rows


def _compact_company_payload(world, companies: Sequence[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if getattr(world, "companies", None) is None:
        return rows
    cdf = world.companies.set_index("company_id", drop=False)
    for cid in companies:
        if cid not in cdf.index:
            continue
        row = cdf.loc[cid]
        lv = getattr(world, "company_latent", {}).get(int(cid), {})
        rows.append(
            {
                "company_id": int(cid),
                "name": row.get("name"),
                "tier": row.get("tier"),
                "specialty_genres": row.get("specialty_genres"),
                "pop_weight": round(float(row.get("pop_weight", 0.2) or 0.2), 4),
                "prestige_score": round(float(lv.get("prestige_score", 0.5) or 0.5), 4),
                "risk_appetite": round(float(lv.get("risk_appetite", 0.5) or 0.5), 4),
            }
        )
    return rows


def _build_year_prompt(world, ctx: YearEvolutionContext) -> str:
    schema = {
        "world_event_narrative": "1-2 sentences",
        "market_regime_moves": [{"kind": "genre_delta", "genre": "Drama", "delta": 0.03}],
        "person_shocks": [{"person_id": 10, "reputation_delta": 0.06, "controversy_delta": -0.03}],
        "company_moves": [{"company_id": 7, "kind": "tier_transition", "new_tier": "Major"}],
        "relationship_programs": [
            {
                "motif": "bridge_building",
                "edge_type": "friendship",
                "mode": "create",
                "target_cohort": "active_people",
                "graph_scope": "cross_community",
                "anchors_people": [10, 11],
                "anchors_companies": [],
                "intensity": 0.68,
                "reach": "regional",
                "fanout": 5,
                "max_depth": 2,
                "weight_mean": 0.36,
                "weight_spread": 0.08,
                "delta_weight": 0.10,
                "cross_community_bias": 0.85,
                "novelty_bias": 0.55,
                "durability_years": 3,
                "reason": "brief reason",
            }
        ],
    }
    payload = {
        "from_year": ctx.from_year,
        "to_year": ctx.to_year,
        "summary": ctx.summary,
        "triggered_events": [{"type": item.get("type"), "description": item.get("description")} for item in ctx.triggered_events],
        "sample_people": _compact_people_payload(world, ctx.sampled_people[:80]),
        "sample_companies": _compact_company_payload(world, ctx.sampled_companies[:30]),
        "sample_edges": ctx.sampled_edges[:120],
    }
    return (
        "You are planning one year of synthetic film-industry evolution.\n\n"
        "Return ONLY valid JSON. Do not emit markdown.\n\n"
        "Important:\n"
        "- Return 6 to 12 relationship programs.\n"
        "- Do not choose raw edge counts; use intensity and reach instead.\n"
        "- Keep plans rich, plausible, and diverse across social, negative, and company motifs.\n"
        "- Use anchor entities and cohort-level programs to create ripple effects.\n"
        "- Prefer multiple distinct motifs over one giant noisy wave.\n\n"
        f"Return JSON matching this shape:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
        "Valid motif families include: scandal_fallout, friendship_clustering, rivalry_wave, mentor_tree, director_camp, recruiting_wave, blacklist_cascade, prestige_alliance, bridge_building, company_competition_shift, regional_migration_wave.\n\n"
        f"World payload:\n{json.dumps(payload, ensure_ascii=False)}"
    )


def _normalize_relationship_program(row: Mapping[str, Any]) -> RelationshipProgram | None:
    if not isinstance(row, Mapping):
        return None
    edge_type = _safe_str(row.get("edge_type"), "").strip()
    if edge_type not in ALLOWED_EDGE_TYPES:
        return None
    mode = _safe_str(row.get("mode"), "create").strip().lower()
    if mode not in {"create", "strengthen", "expire", "weaken"}:
        mode = "create"
    legacy_target_size = max(0, min(36, _safe_int(row.get("target_size"), 0)))
    raw_intensity = row.get("intensity", None)
    intensity = (
        _clamp(raw_intensity, 0.0, 1.0)
        if raw_intensity not in (None, "", "nan")
        else _clamp(legacy_target_size / 36.0 if legacy_target_size else 0.45, 0.15, 0.75)
    )
    reach = _safe_str(row.get("reach"), "cohort").strip().lower()
    if reach not in REACH_WEIGHTS:
        reach = "cohort"
    return RelationshipProgram(
        motif=_safe_str(row.get("motif"), edge_type).strip() or edge_type,
        edge_type=edge_type,
        mode=mode,
        target_cohort=_safe_str(row.get("target_cohort"), "active_people").strip() or "active_people",
        graph_scope=_safe_str(row.get("graph_scope"), "local_frontier").strip() or "local_frontier",
        anchors_people=[int(x) for x in (row.get("anchors_people") or []) if x],
        anchors_companies=[int(cid) for cid in (row.get("anchors_companies") or []) if cid],
        intensity=float(intensity),
        reach=reach,
        target_size=int(legacy_target_size),
        fanout=max(1, min(10, _safe_int(row.get("fanout"), 4))),
        max_depth=max(1, min(3, _safe_int(row.get("max_depth"), 1))),
        weight_mean=_clamp(row.get("weight_mean", row.get("weight", 0.35)), 0.05, 0.95),
        weight_spread=_clamp(row.get("weight_spread", 0.08), 0.01, 0.25),
        delta_weight=_clamp(row.get("delta_weight", 0.10), -0.35, 0.35),
        decay=_clamp(row.get("decay", 0.20), 0.0, 0.80),
        reciprocity_rate=_clamp(row.get("reciprocity_rate", 0.0), 0.0, 1.0),
        closure_rate=_clamp(row.get("closure_rate", 0.0), 0.0, 0.70),
        cross_community_bias=_clamp(row.get("cross_community_bias", 0.25), 0.0, 2.0),
        novelty_bias=_clamp(row.get("novelty_bias", 0.30), 0.0, 1.0),
        durability_years=max(1, min(8, _safe_int(row.get("durability_years"), 3))),
        reason=_safe_str(row.get("reason"), "").strip(),
    )


def _append_unique_program(target: list[RelationshipProgram], program: RelationshipProgram) -> None:
    key = (program.motif, program.edge_type, program.mode)
    existing = {(item.motif, item.edge_type, item.mode) for item in target}
    if key not in existing:
        target.append(program)


def _build_deterministic_plan(world, ctx: YearEvolutionContext) -> YearPlan:
    exposure = ctx.exposure
    summary = ctx.summary
    rng = random.Random(_stable_hash(f"det_plan|{ctx.from_year}|{ctx.to_year}|{exposure.n_movies}"))
    relationship_programs: list[RelationshipProgram] = []
    anchors = [pid for pid in ctx.top_people[:8] if pid in exposure.active_person_ids]
    company_anchors = [cid for cid in ctx.top_companies[:6] if cid in exposure.active_company_ids]

    if exposure.co_counts:
        _append_unique_program(
            relationship_programs,
            RelationshipProgram(
                motif="friendship_clustering",
                edge_type="friendship",
                mode="create",
                target_cohort="active_people",
                graph_scope="local_frontier",
                anchors_people=anchors[:5],
                intensity=0.78,
                reach="cohort",
                fanout=6,
                max_depth=2,
                weight_mean=0.34,
                weight_spread=0.07,
                novelty_bias=0.22,
                durability_years=4,
                reason="repeat collaboration pressure",
            ),
        )
        _append_unique_program(
            relationship_programs,
            RelationshipProgram(
                motif="director_camp",
                edge_type="collaboration",
                mode="strengthen",
                target_cohort="active_people",
                graph_scope="director_camp",
                anchors_people=anchors[:4],
                intensity=0.62,
                reach="cohort",
                fanout=5,
                max_depth=2,
                weight_mean=0.28,
                weight_spread=0.05,
                delta_weight=0.09,
                novelty_bias=0.16,
                durability_years=3,
                reason="stable creative camps",
            ),
        )
        _append_unique_program(
            relationship_programs,
            RelationshipProgram(
                motif="bridge_building",
                edge_type="friendship",
                mode="create",
                target_cohort="cross_market_people",
                graph_scope="cross_community",
                anchors_people=anchors[:4],
                intensity=0.52,
                reach="regional",
                fanout=4,
                max_depth=2,
                cross_community_bias=0.95,
                novelty_bias=0.55,
                durability_years=4,
                reason="community bridging",
            ),
        )

    if exposure.competition_counts:
        _append_unique_program(
            relationship_programs,
            RelationshipProgram(
                motif="rivalry_wave",
                edge_type="rivalry",
                mode="create",
                target_cohort="volatile_people",
                graph_scope="competition_frontier",
                anchors_people=ctx.top_people[:6],
                intensity=0.66 if summary.get("regime_label") != "boom" else 0.52,
                reach="cohort",
                fanout=4,
                max_depth=1,
                weight_mean=0.42,
                weight_spread=0.09,
                novelty_bias=0.28,
                durability_years=3,
                reason="competitive pressure",
            ),
        )

    if exposure.director_actor_counts:
        _append_unique_program(
            relationship_programs,
            RelationshipProgram(
                motif="mentor_tree",
                edge_type="mentorship",
                mode="create",
                target_cohort="rising_people",
                graph_scope="director_camp",
                anchors_people=ctx.top_people[:4],
                intensity=0.58,
                reach="cohort",
                fanout=4,
                max_depth=2,
                weight_mean=0.39,
                weight_spread=0.06,
                durability_years=5,
                reason="career guidance",
            ),
        )

    if exposure.company_person_counts:
        _append_unique_program(
            relationship_programs,
            RelationshipProgram(
                motif="recruiting_wave",
                edge_type="brand_fit",
                mode="create",
                target_cohort="top_companies_to_active_people",
                graph_scope="company_frontier",
                anchors_companies=company_anchors[:5],
                intensity=0.70,
                reach="cohort",
                fanout=5,
                max_depth=1,
                weight_mean=0.31,
                weight_spread=0.07,
                novelty_bias=0.34,
                durability_years=3,
                reason="talent courting",
            ),
        )

    if exposure.company_pair_counts:
        _append_unique_program(
            relationship_programs,
            RelationshipProgram(
                motif="prestige_alliance",
                edge_type="co_production",
                mode="create",
                target_cohort="top_companies",
                graph_scope="company_pairs",
                anchors_companies=company_anchors[:4],
                intensity=0.52,
                reach="industry",
                fanout=3,
                max_depth=1,
                weight_mean=0.36,
                weight_spread=0.06,
                novelty_bias=0.24,
                durability_years=4,
                reason="slate alignment",
            ),
        )

    for event in ctx.triggered_events:
        etype = _safe_str(event.get("type"), "unknown")
        if etype == "scandal":
            _append_unique_program(
                relationship_programs,
                RelationshipProgram(
                    motif="scandal_fallout",
                    edge_type="avoid",
                    mode="create",
                    target_cohort="controversial_people",
                    graph_scope="cross_community",
                    anchors_people=ctx.top_people[:4],
                    intensity=0.78,
                    reach="regional",
                    fanout=5,
                    max_depth=2,
                    weight_mean=0.48,
                    weight_spread=0.10,
                    closure_rate=0.18,
                    cross_community_bias=0.90,
                    novelty_bias=0.48,
                    durability_years=3,
                    reason="public scandal fallout",
                ),
            )
            _append_unique_program(
                relationship_programs,
                RelationshipProgram(
                    motif="blacklist_cascade",
                    edge_type="blacklist",
                    mode="create",
                    target_cohort="controversial_people",
                    graph_scope="company_frontier",
                    anchors_companies=company_anchors[:3],
                    anchors_people=ctx.top_people[:3],
                    intensity=0.60,
                    reach="industry",
                    fanout=4,
                    max_depth=2,
                    weight_mean=0.42,
                    weight_spread=0.09,
                    novelty_bias=0.45,
                    durability_years=3,
                    reason="contagious industry recoil",
                ),
            )
        elif etype == "studio_merger":
            _append_unique_program(
                relationship_programs,
                RelationshipProgram(
                    motif="company_competition_shift",
                    edge_type="market_rival",
                    mode="create",
                    target_cohort="top_companies",
                    graph_scope="company_pairs",
                    anchors_companies=company_anchors[:5],
                    intensity=0.62,
                    reach="industry",
                    fanout=4,
                    max_depth=1,
                    weight_mean=0.35,
                    weight_spread=0.06,
                    novelty_bias=0.18,
                    durability_years=3,
                    reason="market consolidation response",
                ),
            )
        elif etype == "country_emergence":
            _append_unique_program(
                relationship_programs,
                RelationshipProgram(
                    motif="regional_migration_wave",
                    edge_type="friendship",
                    mode="create",
                    target_cohort="cross_market_people",
                    graph_scope="cross_community",
                    anchors_people=ctx.top_people[:6],
                    anchors_companies=company_anchors[:4],
                    intensity=0.66,
                    reach="regional",
                    fanout=5,
                    max_depth=2,
                    weight_mean=0.33,
                    weight_spread=0.08,
                    cross_community_bias=1.10,
                    novelty_bias=0.55,
                    durability_years=4,
                    reason="regional breakout",
                ),
            )

    market_moves: list[dict[str, Any]] = []
    total_movies = max(1, exposure.n_movies)
    for genre, count in exposure.genre_counts.most_common(4):
        share = count / total_movies
        delta = _clamp((share - 0.08) * 0.16, -0.04, 0.05)
        if abs(delta) >= 0.01:
            market_moves.append({"kind": "genre_delta", "genre": genre, "delta": delta})
    if ctx.triggered_events:
        first = ctx.triggered_events[0]
        if _safe_str(first.get("type"), "") == "country_emergence" and exposure.country_counts:
            country, _count = exposure.country_counts.most_common(1)[0]
            market_moves.append({"kind": "country_multiplier", "country": country, "multiplier": 1.15})

    person_shocks: list[dict[str, Any]] = []
    for pid in ctx.top_people[:12]:
        vals = exposure.person_perf.get(int(pid), [])
        if not vals:
            continue
        avg_rating = float(np.mean([v[0] for v in vals]))
        avg_perf = float(np.mean([v[1] for v in vals]))
        volatility = float(getattr(world, "person_latent", {}).get(int(pid), {}).get("volatility", 0.35) or 0.35)
        shock = {
            "person_id": int(pid),
            "reputation_delta": _clamp((avg_rating - 6.3) * 0.03, -0.10, 0.10),
            "controversy_delta": _clamp((0.62 - avg_rating) * 0.04 + (volatility - 0.35) * 0.04, -0.08, 0.10),
            "ambition_delta": _clamp((avg_perf - 1.0) * 0.03 + rng.uniform(-0.01, 0.02), -0.08, 0.10),
            "volatility_delta": _clamp((avg_perf - 1.0) * 0.02 + rng.uniform(-0.02, 0.02), -0.06, 0.06),
        }
        if any(abs(float(v)) >= 0.01 for k, v in shock.items() if k != "person_id"):
            person_shocks.append(shock)

    company_moves: list[dict[str, Any]] = []
    for cid in ctx.top_companies[:8]:
        tier = _company_tier_for(world, int(cid))
        vals = exposure.company_perf.get(int(cid), [])
        avg_perf = float(np.mean([v[1] for v in vals])) if vals else 1.0
        avg_rating = float(np.mean([v[0] for v in vals])) if vals else 6.0
        tier_idx = _COMPANY_TIER_ORDER.get(tier, 2)
        if avg_perf >= 1.45 and avg_rating >= 6.8 and tier_idx < len(COMPANY_TIERS) - 1:
            company_moves.append({"company_id": int(cid), "kind": "tier_transition", "new_tier": COMPANY_TIERS[tier_idx + 1]})
        elif avg_perf < 0.72 and avg_rating < 5.9 and tier_idx > 0:
            company_moves.append({"company_id": int(cid), "kind": "tier_transition", "new_tier": COMPANY_TIERS[tier_idx - 1]})

    narrative_bits = [_safe_str(item.get("type"), "shift").replace("_", " ") for item in ctx.triggered_events]
    if not narrative_bits:
        narrative_bits = [summary.get("regime_label", "neutral market")]
    narrative = f"{ctx.from_year} saw {' and '.join(narrative_bits[:2])}, reshaping alliances, reputations, and production momentum."
    return YearPlan(
        market_regime_moves=market_moves,
        person_shocks=person_shocks,
        company_moves=company_moves,
        relationship_programs=relationship_programs,
        world_event_narrative=narrative,
        planner_source="deterministic",
    )


def _program_channel_family(program: RelationshipProgram) -> str:
    if program.edge_type in NEGATIVE_EDGE_TYPES:
        return "negative_social" if program.edge_type in SOCIAL_EDGE_TYPES else "negative_company"
    if program.edge_type in SOCIAL_EDGE_TYPES:
        return "positive_social"
    return "company"


def _finalize_relationship_programs(programs: Sequence[RelationshipProgram], fallback_programs: Sequence[RelationshipProgram]) -> list[RelationshipProgram]:
    merged: list[RelationshipProgram] = []
    by_key: set[tuple[str, str, str]] = set()
    per_family: Counter[str] = Counter()
    for source in (programs, fallback_programs):
        for program in source:
            key = (program.motif, program.edge_type, program.mode)
            if key in by_key:
                continue
            family = _program_channel_family(program)
            if source is programs and per_family[family] >= 4:
                continue
            merged.append(program)
            by_key.add(key)
            per_family[family] += 1
            if len(merged) >= 12:
                break
        if len(merged) >= 12:
            break

    required_checks = [
        lambda items: any(p.edge_type in POSITIVE_EDGE_TYPES and p.edge_type in SOCIAL_EDGE_TYPES for p in items),
        lambda items: any(p.edge_type in NEGATIVE_EDGE_TYPES and p.edge_type in SOCIAL_EDGE_TYPES for p in items),
        lambda items: any(p.edge_type in COMPANY_EDGE_TYPES for p in items),
    ]
    for fallback in fallback_programs:
        if len(merged) >= 12:
            break
        key = (fallback.motif, fallback.edge_type, fallback.mode)
        if key in by_key:
            continue
        if all(check(merged) for check in required_checks) and len(merged) >= 6:
            break
        merged.append(fallback)
        by_key.add(key)

    for fallback in fallback_programs:
        if len(merged) >= 6:
            break
        key = (fallback.motif, fallback.edge_type, fallback.mode)
        if key in by_key:
            continue
        merged.append(fallback)
        by_key.add(key)

    return sorted(merged, key=lambda item: (item.intensity, REACH_WEIGHTS.get(item.reach, 1.0)), reverse=True)[:12]


def _apply_temporal_correlation_profile(world, programs: Sequence[RelationshipProgram]) -> list[RelationshipProgram]:
    novelty_scale = _planner_prior_float(world, "temporal_novelty_scale", 1.0, lo=0.40, hi=1.20)
    cross_scale = _planner_prior_float(world, "temporal_cross_community_scale", 1.0, lo=0.40, hi=1.20)
    tuned: list[RelationshipProgram] = []
    for program in programs:
        tuned.append(
            replace(
                program,
                novelty_bias=_clamp(program.novelty_bias * novelty_scale, 0.0, 1.0),
                cross_community_bias=_clamp(program.cross_community_bias * cross_scale, 0.0, 2.0),
            )
        )
    return tuned


def plan_year(
    world,
    ctx: YearEvolutionContext,
    *,
    use_llm: bool,
    model: str | None = None,
    log_dir: str | None = None,
) -> YearPlan:
    fallback = _build_deterministic_plan(world, ctx)
    fallback.relationship_programs = _apply_temporal_correlation_profile(world, fallback.relationship_programs)
    if not use_llm:
        return fallback
    try:
        llm = get_llm_client()
    except Exception:
        return fallback
    prompt = _build_year_prompt(world, ctx)
    raw_text = ""
    try:
        response = llm.generate(prompt, model=model, json_mode=True, temperature=0.68, timeout_sec=85, max_attempts=4)
        raw_text = response.text.strip()
        parsed = _safe_json_loads(raw_text)
    except Exception as exc:
        fallback.raw_text = raw_text
        fallback.parse_error = str(exc)
        return fallback

    if not isinstance(parsed, dict):
        fallback.raw_text = raw_text
        fallback.parse_error = "planner_output_not_dict"
        return fallback

    llm_programs = [
        program
        for row in (parsed.get("relationship_programs") or [])
        if (program := _normalize_relationship_program(row)) is not None
    ]
    plan = YearPlan(
        market_regime_moves=[row for row in (parsed.get("market_regime_moves") or []) if isinstance(row, dict)],
        person_shocks=[row for row in (parsed.get("person_shocks") or []) if isinstance(row, dict)],
        company_moves=[row for row in (parsed.get("company_moves") or []) if isinstance(row, dict)],
        relationship_programs=_apply_temporal_correlation_profile(
            world,
            _finalize_relationship_programs(llm_programs, fallback.relationship_programs),
        ),
        world_event_narrative=_safe_str(parsed.get("world_event_narrative"), "").strip() or fallback.world_event_narrative,
        planner_source="llm",
        raw_text=raw_text,
    )
    if log_dir:
        base = Path(log_dir)
        base.mkdir(parents=True, exist_ok=True)
        stem = base / f"year_plan_{ctx.from_year}_{time.strftime('%Y%m%d_%H%M%S')}"
        stem.with_suffix(".prompt.txt").write_text(prompt, encoding="utf-8")
        stem.with_suffix(".raw.json").write_text(raw_text, encoding="utf-8")
    return plan


def _build_graph_scale_profile(world, ctx: YearEvolutionContext) -> GraphScaleProfile:
    graph = getattr(world, "graph", None)
    manifest = getattr(graph, "manifest", {}) if graph is not None else {}
    hot_family_counts = {
        edge_type: int((manifest.get("hot_types", {}) or {}).get(edge_type, {}).get("count", 0))
        for edge_type in sorted(HOT_EDGE_TYPES)
    }
    total_movies = max(1, ctx.exposure.n_movies)
    hot_genre_count = sum(1 for _genre, count in ctx.exposure.genre_counts.items() if float(count) / float(total_movies) >= 0.08)
    hot_country_count = sum(1 for _country, count in ctx.exposure.country_counts.items() if float(count) / float(total_movies) >= 0.15)
    return GraphScaleProfile(
        hot_total=sum(hot_family_counts.values()),
        cold_cp_total=int(manifest.get("cold_cp_count", 0)),
        cold_cc_total=int(manifest.get("cold_cc_count", 0)),
        hot_family_counts=hot_family_counts,
        n_movies=int(ctx.exposure.n_movies),
        active_people=len(ctx.exposure.active_person_ids),
        active_companies=len(ctx.exposure.active_company_ids),
        regime_label=_safe_str(ctx.summary.get("regime_label"), "neutral"),
        triggered_event_count=len(ctx.triggered_events),
        co_count=len(ctx.exposure.co_counts),
        competition_count=len(ctx.exposure.competition_counts),
        director_actor_count=len(ctx.exposure.director_actor_counts),
        company_person_count=len(ctx.exposure.company_person_counts),
        company_pair_count=len(ctx.exposure.company_pair_counts),
        hot_genre_count=max(1, hot_genre_count),
        hot_country_count=max(1, hot_country_count),
    )


def _build_year_edge_budget(profile: GraphScaleProfile) -> YearEdgeBudget:
    social_program_budget = int(
        np.clip(
            round(
                0.0060 * profile.hot_total
                + 0.20 * min(profile.co_count, 120_000)
                + 0.14 * min(profile.competition_count, 80_000)
                + 14 * profile.n_movies
            ),
            4_000,
            120_000,
        )
    )
    company_program_budget = int(
        np.clip(
            round(
                0.00055 * profile.cold_cp_total
                + 0.00150 * profile.cold_cc_total
                + 0.010 * min(profile.company_person_count, 250_000)
                + 0.016 * min(profile.company_pair_count, 120_000)
            ),
            2_000,
            80_000,
        )
    )
    background_budget = int(
        np.clip(
            round(
                0.0040 * profile.hot_total
                + 0.00018 * profile.cold_cp_total
                + 0.00080 * profile.cold_cc_total
            ),
            3_000,
            100_000,
        )
    )
    cascade_budget = int(
        np.clip(
            round(
                0.0015 * profile.hot_total
                + 900 * profile.triggered_event_count
                + 350 * profile.hot_genre_count
                + 500 * profile.hot_country_count
            ),
            1_500,
            40_000,
        )
    )
    return YearEdgeBudget(
        social_program_budget=social_program_budget,
        company_program_budget=company_program_budget,
        background_budget=background_budget,
        cascade_budget=cascade_budget,
    )


def _program_target_limit(program: RelationshipProgram, targeted_ops: int) -> tuple[int, int, int]:
    core = max(1, int(round(targeted_ops * 0.50)))
    frontier = max(1, int(round(targeted_ops * 0.30)))
    novelty = max(1, targeted_ops - core - frontier)
    return core, frontier, novelty


def _ensure_similarity_reservoir(world, ctx: YearEvolutionContext) -> None:
    cache = ctx.execution_cache
    if cache is None or cache.top_similarity_pairs:
        return
    from world_state import latent_similarity

    candidates = list(
        dict.fromkeys(
            ctx.top_people[:72]
            + _pick_people_by_cohort(world, ctx, "active_people", limit=96, rng=random.Random(_stable_hash(f"sim_pool|{ctx.from_year}")))
        )
    )
    scored: list[tuple[int, int, float]] = []
    communities = cache.communities_by_person
    for a, b in combinations(candidates[:128], 2):
        sim = float(latent_similarity(world, int(a), int(b)))
        if sim < 0.50:
            continue
        shared_genres = len(set(cache.person_genres.get(int(a), ())) & set(cache.person_genres.get(int(b), ())))
        same_comm = 1.0 if communities.get(int(a), -1) == communities.get(int(b), -2) else 0.0
        score = sim + 0.04 * shared_genres + 0.03 * same_comm
        scored.append((int(a), int(b), float(score)))
    scored.sort(key=lambda item: item[2], reverse=True)
    cache.top_similarity_pairs = [(a, b) for a, b, _ in scored[:2048]]
    cache.cross_community_bridge_pairs = [
        (a, b)
        for a, b, _score in scored
        if communities.get(int(a), -1) != communities.get(int(b), -1)
    ][:1024]


def _ensure_hotspot_people(ctx: YearEvolutionContext) -> None:
    cache = ctx.execution_cache
    if cache is None:
        return
    if not cache.genre_hotspot_people and cache.person_genres:
        for genre, _count in ctx.exposure.genre_counts.most_common(5):
            genre_lower = _safe_str(genre, "").strip().lower()
            cache.genre_hotspot_people[genre_lower] = [
                int(pid)
                for pid in ctx.exposure.active_person_ids
                if genre_lower in {item.lower() for item in cache.person_genres.get(int(pid), ())}
            ][:2048]
    if not cache.country_hotspot_people and cache.person_country:
        for country, _count in ctx.exposure.country_counts.most_common(5):
            country_lower = _safe_str(country, "").strip().lower()
            cache.country_hotspot_people[country_lower] = [
                int(pid)
                for pid in ctx.exposure.active_person_ids
                if cache.person_country.get(int(pid), "") == country_lower
            ][:2048]


def _ensure_edge_samples(world, ctx: YearEvolutionContext, profile: GraphScaleProfile, budget: YearEdgeBudget) -> None:
    cache = ctx.execution_cache
    graph = getattr(world, "graph", None)
    if cache is None or graph is None or cache.active_edge_samples_by_family:
        return
    total_sample = max(48, min(512, int(round(np.sqrt(max(1, budget.background_budget)) * 2.5))))
    for edge_type in sorted(HOT_EDGE_TYPES | COLD_CP_TYPES | COLD_CC_TYPES):
        count_hint = 0
        if edge_type in HOT_EDGE_TYPES:
            count_hint = profile.hot_family_counts.get(edge_type, 0)
        elif edge_type in COLD_CP_TYPES:
            count_hint = max(1, profile.cold_cp_total // max(1, len(COLD_CP_TYPES)))
        else:
            count_hint = max(1, profile.cold_cc_total // max(1, len(COLD_CC_TYPES)))
        if count_hint <= 0:
            continue
        sample_size = min(total_sample, max(32, int(round(np.sqrt(max(1, count_hint)) * 1.5))))
        rows = graph.sample_active_edges([edge_type], ctx.from_year, sample_size, seed=_stable_hash(f"edge_sample|{ctx.from_year}|{edge_type}"))
        cache.active_edge_samples_by_family[edge_type] = rows
        stale: list[dict[str, Any]] = []
        for row in rows:
            valid_from = _safe_int(row.get("valid_from"), ctx.from_year)
            weight = float(row.get("weight", 0.0) or 0.0)
            age = max(0, ctx.from_year - int(valid_from))
            if age >= 3 or weight <= 0.28:
                stale.append(row)
        cache.stale_edge_samples_by_family[edge_type] = stale


def _ensure_year_reservoirs(world, ctx: YearEvolutionContext, profile: GraphScaleProfile, budget: YearEdgeBudget) -> None:
    _ensure_similarity_reservoir(world, ctx)
    _ensure_hotspot_people(ctx)
    _ensure_edge_samples(world, ctx, profile, budget)


def _pick_people_by_cohort(world, ctx: YearEvolutionContext, cohort: str, limit: int, rng: random.Random) -> list[int]:
    cache = ctx.execution_cache
    if cache is not None and cohort in cache.cohort_people:
        cached = list(cache.cohort_people[cohort])
        rng.shuffle(cached[max(0, limit // 2) :])
        return cached[:limit]

    exposure = ctx.exposure
    active_people = list(exposure.active_person_ids)
    if cohort == "rising_people":
        stage_cache = cache.person_stage_cache if cache is not None else _build_person_stage_cache(world)
        pool = [pid for pid in ctx.top_people if stage_cache.get(int(pid), "prime") == "rising"] or ctx.top_people
    elif cohort == "volatile_people":
        scores = []
        for pid in active_people:
            lv = getattr(world, "person_latent", {}).get(int(pid), {})
            score = float(lv.get("volatility", 0.35) or 0.35) + float(lv.get("controversy_score", 0.15) or 0.15)
            scores.append((int(pid), score))
        scores.sort(key=lambda item: item[1], reverse=True)
        pool = [pid for pid, _ in scores]
    elif cohort == "controversial_people":
        scores = []
        for pid in active_people:
            lv = getattr(world, "person_latent", {}).get(int(pid), {})
            score = float(lv.get("controversy_score", 0.15) or 0.15)
            scores.append((int(pid), score))
        scores.sort(key=lambda item: item[1], reverse=True)
        pool = [pid for pid, _ in scores]
    elif cohort == "cross_market_people":
        pool = list(ctx.top_people)
        if cache is not None and cache.person_country:
            ranked = sorted(
                active_people,
                key=lambda pid: (cache.person_country.get(int(pid), ""), -1 if int(pid) in ctx.top_people else 0, int(pid)),
            )
            pool.extend(ranked)
        else:
            pool.extend(active_people)
    else:
        pool = list(ctx.top_people) + active_people

    deduped = list(dict.fromkeys(int(pid) for pid in pool if int(pid) > 0))
    if cache is not None:
        cache.cohort_people[cohort] = list(deduped)
    rng.shuffle(deduped[max(0, limit // 2) :])
    return deduped[:limit]


def _pick_companies_by_cohort(ctx: YearEvolutionContext, cohort: str, limit: int, rng: random.Random) -> list[int]:
    cache = ctx.execution_cache
    if cache is not None and cohort in cache.cohort_companies:
        cached = list(cache.cohort_companies[cohort])
        rng.shuffle(cached[max(0, limit // 2) :])
        return cached[:limit]

    active_companies = list(ctx.exposure.active_company_ids)
    pool = list(ctx.top_companies) + active_companies
    deduped = list(dict.fromkeys(int(cid) for cid in pool if int(cid) > 0))
    if cache is not None:
        cache.cohort_companies[cohort] = list(deduped)
    rng.shuffle(deduped[max(0, limit // 2) :])
    return deduped[:limit]


def _frontier_people(world, anchors: Sequence[int], year: int, limit: int, cross_bias: float, rng: random.Random) -> list[int]:
    if limit <= 0:
        return []
    anchor_key = tuple(sorted(int(anchor) for anchor in anchors if anchor))
    cross_bucket = int(round(max(0.0, float(cross_bias or 0.0)) * 100.0))
    key = (anchor_key, int(limit), cross_bucket)
    cache = getattr(world, "_year_execution_cache_current", None)
    if cache is not None and key in cache.anchor_frontiers:
        cached = list(cache.anchor_frontiers[key])
        rng.shuffle(cached)
        return cached[:limit]
    graph = getattr(world, "graph", None)
    collected: list[int] = []
    anchor_set = {int(anchor) for anchor in anchors if anchor}
    if graph is not None and anchors:
        if hasattr(graph, "sample_frontier"):
            frontier_edges = graph.sample_frontier(
                anchors,
                ["friendship", "rivalry", "collaboration", "former_collaborator", "clique", "mentorship"],
                year,
                per_anchor_limit=max(8, min(64, limit)),
                seed=_stable_hash(f"frontier|{year}|{anchor_key}|{cross_bucket}"),
            )
            for row in frontier_edges:
                src_id = int(row.get("src_id", 0))
                dst_id = int(row.get("dst_id", 0))
                if src_id in anchor_set and dst_id not in anchor_set:
                    collected.append(dst_id)
                elif dst_id in anchor_set and src_id not in anchor_set:
                    collected.append(src_id)
        else:
            for anchor in anchors:
                for neighbor, *_rest in getattr(graph, "iter_friend_neighbors", lambda *_args, **_kwargs: [])(int(anchor), year):
                    if int(neighbor) not in anchor_set:
                        collected.append(int(neighbor))
                for neighbor, *_rest in getattr(graph, "iter_rival_neighbors", lambda *_args, **_kwargs: [])(int(anchor), year):
                    if int(neighbor) not in anchor_set:
                        collected.append(int(neighbor))
    if cache is not None and cross_bias > 0:
        cross_pairs = list(cache.cross_community_bridge_pairs)
        rng.shuffle(cross_pairs)
        cross_target = min(len(cross_pairs), max(1, int(round(limit * min(2.0, cross_bias)))))
        for a, b in cross_pairs[:cross_target]:
            if int(a) in anchor_set and int(b) not in anchor_set:
                collected.append(int(b))
            elif int(b) in anchor_set and int(a) not in anchor_set:
                collected.append(int(a))
            else:
                collected.extend([int(a), int(b)])
    if cross_bias > 0 and len(collected) < limit:
        communities = getattr(world, "communities", {}) or {}
        if communities:
            anchor_communities = {communities.get(int(anchor)) for anchor in anchor_set}
            cross_candidates = [
                int(pid)
                for pid, comm in communities.items()
                if int(pid) not in anchor_set and comm not in anchor_communities
            ]
            rng.shuffle(cross_candidates)
            cross_target = max(1, int(round(limit * min(1.5, float(cross_bias)))))
            collected.extend(cross_candidates[:cross_target])
    deduped = [pid for pid in dict.fromkeys(collected) if pid not in anchor_set]
    if cache is not None:
        cache.anchor_frontiers[key] = list(deduped)
    rng.shuffle(deduped)
    return deduped[:limit]


def _build_year_scale(world, ctx: YearEvolutionContext) -> tuple[GraphScaleProfile, YearEdgeBudget]:
    profile = _build_graph_scale_profile(world, ctx)
    budget = _build_year_edge_budget(profile)
    ctx.scale_profile = profile
    ctx.edge_budget = budget
    return profile, budget


def _regime_multiplier(profile: GraphScaleProfile, *, family: str, action: str) -> float:
    if profile.regime_label == "boom":
        if family == "negative":
            return 0.85
        if action == "expire":
            return 0.80
        return 1.20
    if profile.regime_label == "stress":
        if family == "negative":
            return 1.20
        if action == "expire":
            return 1.15
        return 0.90
    return 1.0


def _program_event_bonus(program: RelationshipProgram, triggered_events: Sequence[dict[str, Any]]) -> float:
    event_types = {_safe_str(item.get("type"), "") for item in triggered_events}
    motif_events = MOTIF_EVENT_HINTS.get(program.motif, set())
    return 1.25 if event_types & motif_events else 1.0


def _blueprint_for_program(program: RelationshipProgram) -> list[tuple[str, str, float]]:
    blueprint = list(MOTIF_BLUEPRINTS.get(program.motif, ((program.edge_type, program.mode, 1.0),)))
    if all(edge_type != program.edge_type for edge_type, _mode, _share in blueprint):
        blueprint.insert(0, (program.edge_type, program.mode, 0.25))
    total = sum(max(0.0, share) for _edge_type, _mode, share in blueprint) or 1.0
    scaled: list[tuple[str, str, float]] = []
    for edge_type, mode, share in blueprint:
        share_weight = float(share) / float(total)
        if edge_type == program.edge_type:
            share_weight += 0.10
        scaled.append((edge_type, mode, share_weight))
    total_scaled = sum(weight for _edge_type, _mode, weight in scaled) or 1.0
    return [(edge_type, mode, weight / total_scaled) for edge_type, mode, weight in scaled]


def _allocate_program_budgets(
    programs: Sequence[RelationshipProgram],
    profile: GraphScaleProfile,
    budget: YearEdgeBudget,
    triggered_events: Sequence[dict[str, Any]],
) -> list[ProgramAllocation]:
    if not programs:
        return []
    total_targeted = int(budget.social_program_budget + budget.company_program_budget)
    scores: list[float] = []
    for program in programs:
        score = 0.60 + 1.80 * float(program.intensity) + (REACH_WEIGHTS.get(program.reach, 1.0) - 1.0) + (_program_event_bonus(program, triggered_events) - 1.0)
        if program.edge_type in NEGATIVE_EDGE_TYPES:
            score *= _regime_multiplier(profile, family="negative", action=program.mode)
        elif program.edge_type in POSITIVE_EDGE_TYPES:
            score *= _regime_multiplier(profile, family="positive", action=program.mode)
        scores.append(max(0.05, score))
    score_sum = sum(scores) or 1.0
    raw_allocations = [max(1, int(round(total_targeted * score / score_sum))) for score in scores]
    max_share = max(1, int(round(total_targeted * 0.30)))
    raw_allocations = [min(max_share, value) for value in raw_allocations]
    allocated_sum = sum(raw_allocations)
    if allocated_sum > total_targeted:
        diff = allocated_sum - total_targeted
        for idx in np.argsort(raw_allocations)[::-1].tolist():
            if diff <= 0:
                break
            trim = min(diff, max(0, raw_allocations[idx] - 1))
            raw_allocations[idx] -= trim
            diff -= trim
    elif allocated_sum < total_targeted:
        diff = total_targeted - allocated_sum
        order = np.argsort(scores)[::-1].tolist()
        cursor = 0
        while diff > 0 and order:
            idx = int(order[cursor % len(order)])
            if raw_allocations[idx] < max_share:
                raw_allocations[idx] += 1
                diff -= 1
            cursor += 1

    allocations: list[ProgramAllocation] = []
    for program, targeted_ops in zip(programs, raw_allocations):
        family_budgets: dict[str, int] = {}
        blueprint = _blueprint_for_program(program)
        remainder = int(targeted_ops)
        for idx, (edge_type, _mode, share) in enumerate(blueprint):
            family_budget = remainder if idx == len(blueprint) - 1 else max(1, int(round(targeted_ops * share)))
            family_budgets[edge_type] = family_budgets.get(edge_type, 0) + int(family_budget)
            remainder -= int(family_budget)
        allocations.append(ProgramAllocation(program=program, targeted_ops=int(targeted_ops), family_budgets=family_budgets))
    return allocations


def _program_people_pool(world, ctx: YearEvolutionContext, program: RelationshipProgram, targeted_ops: int, rng: random.Random) -> list[int]:
    reach_factor = REACH_WEIGHTS.get(program.reach, 1.0)
    limit = int(max(64, min(960, targeted_ops // 6 * reach_factor + 64)))
    pool = _pick_people_by_cohort(world, ctx, program.target_cohort, limit=limit, rng=rng)
    frontier = _frontier_people(world, program.anchors_people, ctx.from_year, limit=max(24, min(256, targeted_ops // 10 + 24)), cross_bias=program.cross_community_bias, rng=rng)
    return list(dict.fromkeys(int(pid) for pid in list(program.anchors_people) + frontier + pool if int(pid) > 0))


def _program_company_pool(ctx: YearEvolutionContext, program: RelationshipProgram, targeted_ops: int, rng: random.Random) -> list[int]:
    reach_factor = REACH_WEIGHTS.get(program.reach, 1.0)
    limit = int(max(16, min(256, targeted_ops // 10 * reach_factor + 24)))
    return list(dict.fromkeys(int(cid) for cid in list(program.anchors_companies) + _pick_companies_by_cohort(ctx, program.target_cohort, limit=limit, rng=rng) if int(cid) > 0))


def _pair_similarity_hint(world, a: int, b: int) -> float:
    try:
        from world_state import latent_similarity

        return float(latent_similarity(world, int(a), int(b)))
    except Exception:
        return 0.0


def _core_pairs_for_family(world, ctx: YearEvolutionContext, program: RelationshipProgram, edge_type: str) -> list[tuple[int, int]]:
    cache = ctx.execution_cache
    if cache is None:
        return []
    if edge_type in {"friendship", "collaboration", "clique", "former_collaborator", "chemistry"}:
        return list(cache.coappearance_pairs) + list(cache.top_similarity_pairs)
    if edge_type == "mentorship":
        return list(cache.director_actor_pairs)
    if edge_type in {"rivalry", "avoid"}:
        return list(cache.competition_pairs) + list(cache.cross_community_bridge_pairs)
    if edge_type in COLD_CP_TYPES:
        return list(cache.company_person_pairs)
    if edge_type in COLD_CC_TYPES:
        return list(cache.company_company_pairs)
    return []


def _frontier_pairs_for_family(world, ctx: YearEvolutionContext, program: RelationshipProgram, edge_type: str, people_pool: Sequence[int], company_pool: Sequence[int]) -> list[tuple[int, int]]:
    graph = getattr(world, "graph", None)
    pairs: list[tuple[int, int]] = []
    if edge_type in HOT_EDGE_TYPES:
        if graph is None:
            return pairs
        rows = graph.sample_frontier(
            program.anchors_people,
            [edge_type],
            ctx.from_year,
            per_anchor_limit=max(8, min(64, program.fanout * 8)),
            seed=_stable_hash(f"frontier_pairs|{ctx.from_year}|{program.motif}|{edge_type}"),
        )
        for row in rows:
            pairs.append((int(row.get("src_id", 0)), int(row.get("dst_id", 0))))
        return pairs
    if edge_type in COLD_CC_TYPES and graph is not None:
        rows = graph.sample_company_pairs(
            list(program.anchors_companies) + list(company_pool),
            [edge_type],
            ctx.from_year,
            sample_size=max(8, min(64, program.fanout * 8)),
            seed=_stable_hash(f"company_pairs|{ctx.from_year}|{program.motif}|{edge_type}"),
        )
        for row in rows:
            pairs.append((int(row.get("src_id", 0)), int(row.get("dst_id", 0))))
        return pairs
    if edge_type in COLD_CP_TYPES:
        for company_id in (list(program.anchors_companies) + list(company_pool))[:24]:
            for person_id in people_pool[:64]:
                pairs.append((int(company_id), int(person_id)))
        return pairs
    return pairs


def _novelty_pairs_for_family(ctx: YearEvolutionContext, program: RelationshipProgram, edge_type: str, people_pool: Sequence[int], company_pool: Sequence[int], rng: random.Random) -> list[tuple[int, int]]:
    cache = ctx.execution_cache
    pairs: list[tuple[int, int]] = []
    if edge_type in HOT_EDGE_TYPES:
        if cache is not None and cache.cross_community_bridge_pairs and program.cross_community_bias > 0.2:
            pairs.extend(cache.cross_community_bridge_pairs[: max(32, len(people_pool) // 2)])
        communities = cache.communities_by_person if cache is not None else {}
        limit = min(256, max(48, len(people_pool)))
        attempts = 0
        while len(pairs) < limit and attempts < limit * 6 and len(people_pool) >= 2:
            a = int(people_pool[int(rng.randrange(len(people_pool)))])
            b = int(people_pool[int(rng.randrange(len(people_pool)))])
            if a == b:
                attempts += 1
                continue
            if program.cross_community_bias > 0.7 and communities and communities.get(a, -1) == communities.get(b, -1) and rng.random() < 0.65:
                attempts += 1
                continue
            pairs.append((a, b))
            attempts += 1
        return pairs
    if edge_type in COLD_CP_TYPES:
        limit = min(512, max(64, len(company_pool) * 4))
        attempts = 0
        while len(pairs) < limit and attempts < limit * 8 and company_pool and people_pool:
            company_id = int(company_pool[int(rng.randrange(len(company_pool)))])
            person_id = int(people_pool[int(rng.randrange(len(people_pool)))])
            pairs.append((company_id, person_id))
            attempts += 1
        return pairs
    if edge_type in COLD_CC_TYPES:
        limit = min(256, max(32, len(company_pool) * 3))
        attempts = 0
        while len(pairs) < limit and attempts < limit * 8 and len(company_pool) >= 2:
            left = int(company_pool[int(rng.randrange(len(company_pool)))])
            right = int(company_pool[int(rng.randrange(len(company_pool)))])
            if left == right:
                attempts += 1
                continue
            pairs.append((left, right))
            attempts += 1
        return pairs
    return pairs


def _orient_pair(
    world,
    ctx: YearEvolutionContext,
    program: RelationshipProgram,
    edge_type: str,
    left: int,
    right: int,
) -> tuple[int, int] | None:
    a = int(left)
    b = int(right)
    if a <= 0 or b <= 0:
        return None
    if edge_type in HOT_UNDIRECTED_TYPES or edge_type in COLD_CC_TYPES:
        if a == b:
            return None
        return (a, b) if a <= b else (b, a)

    cache = ctx.execution_cache
    company_idx = cache.company_row_idx if cache is not None else {}
    if edge_type in COLD_CP_TYPES:
        left_is_company = int(a) in company_idx
        right_is_company = int(b) in company_idx
        if left_is_company and not right_is_company:
            return (a, b)
        if right_is_company and not left_is_company:
            return (b, a)
        if int(a) in program.anchors_companies and int(b) not in program.anchors_companies:
            return (a, b)
        if int(b) in program.anchors_companies and int(a) not in program.anchors_companies:
            return (b, a)
        return (a, b)

    if a == b:
        return None
    if edge_type == "mentorship":
        stage_cache = cache.person_stage_cache if cache is not None else _build_person_stage_cache(world)
        left_stage = _CAREER_STAGE_ORDER.get(stage_cache.get(int(a), "prime"), _CAREER_STAGE_ORDER.get("prime", 2))
        right_stage = _CAREER_STAGE_ORDER.get(stage_cache.get(int(b), "prime"), _CAREER_STAGE_ORDER.get("prime", 2))
        if left_stage > right_stage:
            return (a, b)
        if right_stage > left_stage:
            return (b, a)
        if int(a) in program.anchors_people and int(b) not in program.anchors_people:
            return (a, b)
        if int(b) in program.anchors_people and int(a) not in program.anchors_people:
            return (b, a)
        return (a, b)
    if edge_type == "avoid":
        if int(a) in program.anchors_people and int(b) not in program.anchors_people:
            return (a, b)
        if int(b) in program.anchors_people and int(a) not in program.anchors_people:
            return (b, a)
        left_lv = getattr(world, "person_latent", {}).get(int(a), {})
        right_lv = getattr(world, "person_latent", {}).get(int(b), {})
        left_score = float(left_lv.get("controversy_score", 0.15) or 0.15) + float(left_lv.get("volatility", 0.35) or 0.35)
        right_score = float(right_lv.get("controversy_score", 0.15) or 0.15) + float(right_lv.get("volatility", 0.35) or 0.35)
        return (a, b) if left_score >= right_score else (b, a)
    return (a, b)


def _build_edge_op(
    world,
    ctx: YearEvolutionContext,
    program: RelationshipProgram,
    *,
    edge_type: str,
    mode: str,
    src_id: int,
    dst_id: int,
    rng: random.Random,
    channel: str,
) -> dict[str, Any] | None:
    if edge_type not in ALLOWED_EDGE_TYPES:
        return None
    if edge_type == "chemistry":
        co_key = (min(int(src_id), int(dst_id)), max(int(src_id), int(dst_id)))
        if ctx.exposure.co_counts.get(co_key, 0) < 2:
            return None
        if _pair_similarity_hint(world, int(src_id), int(dst_id)) < 0.68:
            return None

    graph = getattr(world, "graph", None)
    payload = graph.get_active_payload(edge_type, int(src_id), int(dst_id), ctx.from_year) if graph is not None else None
    profile = ctx.scale_profile
    family_name = "negative" if edge_type in NEGATIVE_EDGE_TYPES else "positive"
    base_weight = float(program.weight_mean) + float(rng.uniform(-program.weight_spread, program.weight_spread))
    if edge_type in HOT_EDGE_TYPES and src_id and dst_id:
        base_weight += 0.10 * (_pair_similarity_hint(world, int(src_id), int(dst_id)) - 0.5)
    if profile is not None:
        base_weight *= _regime_multiplier(profile, family=family_name, action=mode)
    base_weight = float(_clamp(base_weight, 0.05, 0.96))
    delta_weight = float(max(0.02, abs(program.delta_weight) + abs(rng.uniform(-0.04, 0.04))))
    if profile is not None:
        delta_weight *= _regime_multiplier(profile, family=family_name, action="update")
    delta_weight = float(_clamp(delta_weight, 0.02, 0.45))

    if mode == "expire":
        if payload is None:
            return None
        return {
            "mode": "expire",
            "edge_type": edge_type,
            "src_id": int(src_id),
            "dst_id": int(dst_id),
            "year": int(ctx.from_year),
            "reason": program.reason or f"{program.motif}:{channel}",
            "source_kind": f"year_plan_{program.motif}",
            "program_key": f"{program.motif}|{edge_type}|{channel}",
            "program_motif": program.motif,
            "channel": channel,
        }

    if mode == "weaken":
        if payload is None:
            return None
        return {
            "mode": "update",
            "edge_type": edge_type,
            "src_id": int(src_id),
            "dst_id": int(dst_id),
            "delta_weight": -float(delta_weight),
            "reason": program.reason or f"{program.motif}:{channel}",
            "source_kind": f"year_plan_{program.motif}",
            "program_key": f"{program.motif}|{edge_type}|{channel}",
            "program_motif": program.motif,
            "channel": channel,
        }

    if payload is not None and mode in {"create", "strengthen", "update"}:
        return {
            "mode": "update",
            "edge_type": edge_type,
            "src_id": int(src_id),
            "dst_id": int(dst_id),
            "delta_weight": float(delta_weight),
            "reason": program.reason or f"{program.motif}:{channel}",
            "source_kind": f"year_plan_{program.motif}",
            "program_key": f"{program.motif}|{edge_type}|{channel}",
            "program_motif": program.motif,
            "channel": channel,
        }

    valid_to = int(ctx.to_year + max(1, int(program.durability_years)) - 1)
    return {
        "mode": "add",
        "edge_type": edge_type,
        "src_id": int(src_id),
        "dst_id": int(dst_id),
        "weight": float(base_weight),
        "valid_from": int(ctx.from_year),
        "valid_to": int(valid_to),
        "reason": program.reason or f"{program.motif}:{channel}",
        "source_kind": f"year_plan_{program.motif}",
        "program_key": f"{program.motif}|{edge_type}|{channel}",
        "program_motif": program.motif,
        "channel": channel,
    }


def _emit_ops_from_pairs(
    world,
    ctx: YearEvolutionContext,
    program: RelationshipProgram,
    *,
    edge_type: str,
    mode: str,
    pairs: Sequence[tuple[int, int]],
    limit: int,
    channel: str,
    rng: random.Random,
    seen: set[tuple[str, str, int, int]],
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    ops: list[dict[str, Any]] = []
    for left, right in pairs:
        oriented = _orient_pair(world, ctx, program, edge_type, int(left), int(right))
        if oriented is None:
            continue
        src_id, dst_id = oriented
        dedupe_key = (str(mode), str(edge_type), int(src_id), int(dst_id))
        if dedupe_key in seen:
            continue
        op = _build_edge_op(world, ctx, program, edge_type=edge_type, mode=mode, src_id=int(src_id), dst_id=int(dst_id), rng=rng, channel=channel)
        if op is None:
            continue
        seen.add(dedupe_key)
        ops.append(op)
        if len(ops) >= int(limit):
            break
    return ops


def _expand_relationship_program(world, ctx: YearEvolutionContext, allocation: ProgramAllocation, rng: random.Random) -> list[dict[str, Any]]:
    program = allocation.program
    people_pool = _program_people_pool(world, ctx, program, allocation.targeted_ops, rng)
    company_pool = _program_company_pool(ctx, program, allocation.targeted_ops, rng)
    people_allowed = {int(pid) for pid in people_pool}
    company_allowed = {int(cid) for cid in company_pool}
    seen: set[tuple[str, str, int, int]] = set()
    ops: list[dict[str, Any]] = []
    family_counts: dict[str, int] = defaultdict(int)

    for edge_type, mode_share, _share in _blueprint_for_program(program):
        family_budget = int(allocation.family_budgets.get(edge_type, 0))
        if family_budget <= 0:
            continue
        core_limit, frontier_limit, novelty_limit = _program_target_limit(program, family_budget)
        core_pairs = _core_pairs_for_family(world, ctx, program, edge_type)
        frontier_pairs = _frontier_pairs_for_family(world, ctx, program, edge_type, people_pool, company_pool)
        novelty_pairs = _novelty_pairs_for_family(ctx, program, edge_type, people_pool, company_pool, rng)

        def _filter_pairs(raw_pairs: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
            if edge_type in HOT_EDGE_TYPES:
                filtered = [(int(a), int(b)) for a, b in raw_pairs if int(a) in people_allowed and int(b) in people_allowed]
            elif edge_type in COLD_CP_TYPES:
                filtered = []
                for a, b in raw_pairs:
                    ia = int(a)
                    ib = int(b)
                    if ia in company_allowed and ib in people_allowed:
                        filtered.append((ia, ib))
                    elif ib in company_allowed and ia in people_allowed:
                        filtered.append((ib, ia))
            else:
                filtered = [(int(a), int(b)) for a, b in raw_pairs if int(a) in company_allowed and int(b) in company_allowed]
            return filtered or [(int(a), int(b)) for a, b in raw_pairs]

        core_pairs = _filter_pairs(core_pairs)
        frontier_pairs = _filter_pairs(frontier_pairs)
        novelty_pairs = _filter_pairs(novelty_pairs)
        rng.shuffle(core_pairs)
        rng.shuffle(frontier_pairs)
        rng.shuffle(novelty_pairs)

        emitted = _emit_ops_from_pairs(
            world,
            ctx,
            program,
            edge_type=edge_type,
            mode=mode_share,
            pairs=core_pairs,
            limit=core_limit,
            channel="core",
            rng=rng,
            seen=seen,
        )
        ops.extend(emitted)
        family_counts[edge_type] += len(emitted)
        remaining = max(0, family_budget - family_counts[edge_type])
        if remaining > 0:
            emitted = _emit_ops_from_pairs(
                world,
                ctx,
                program,
                edge_type=edge_type,
                mode=mode_share,
                pairs=frontier_pairs,
                limit=min(frontier_limit, remaining),
                channel="frontier",
                rng=rng,
                seen=seen,
            )
            ops.extend(emitted)
            family_counts[edge_type] += len(emitted)
        remaining = max(0, family_budget - family_counts[edge_type])
        if remaining > 0:
            emitted = _emit_ops_from_pairs(
                world,
                ctx,
                program,
                edge_type=edge_type,
                mode=mode_share,
                pairs=novelty_pairs,
                limit=min(novelty_limit + remaining, family_budget),
                channel="novelty",
                rng=rng,
                seen=seen,
            )
            ops.extend(emitted)
            family_counts[edge_type] += len(emitted)
    return ops


def _accumulate_person_delta(target: dict[int, dict[str, float]], person_id: int, field_name: str, delta: float) -> None:
    if field_name not in ALLOWED_PERSON_LATENT_FIELDS:
        return
    payload = target.setdefault(int(person_id), {})
    payload[field_name] = float(_clamp(payload.get(field_name, 0.0) + float(delta), -0.18, 0.18))


def _convert_legacy_patch(
    compiled: YearProgram,
    patch: Mapping[str, Any],
    *,
    from_year: int,
    to_year: int,
) -> dict[str, Any] | None:
    op = _safe_str(patch.get("op"), "").strip()
    if op == "edge_add":
        return {
            "mode": "add",
            "edge_type": _safe_str(patch.get("edge_type"), ""),
            "src_id": _safe_int(patch.get("src_id")),
            "dst_id": _safe_int(patch.get("dst_id")),
            "weight": float(_clamp(patch.get("weight", 0.25), 0.02, 0.96)),
            "valid_from": _safe_int(patch.get("valid_from"), from_year),
            "valid_to": _safe_int(patch.get("valid_to"), to_year + 2),
            "reason": _safe_str(patch.get("reason"), ""),
            "source_kind": _safe_str(patch.get("source_kind"), "legacy_port"),
            "program_key": f"legacy|{_safe_str(patch.get('source_kind'), 'legacy')}",
            "program_motif": _safe_str(patch.get("source_kind"), "legacy"),
            "channel": "legacy",
        }
    if op == "edge_update":
        return {
            "mode": "update",
            "edge_type": _safe_str(patch.get("edge_type"), ""),
            "src_id": _safe_int(patch.get("src_id")),
            "dst_id": _safe_int(patch.get("dst_id")),
            "delta_weight": float(_clamp(patch.get("delta_weight", 0.0), -0.45, 0.45)),
            "reason": _safe_str(patch.get("reason"), ""),
            "source_kind": _safe_str(patch.get("source_kind"), "legacy_port"),
            "program_key": f"legacy|{_safe_str(patch.get('source_kind'), 'legacy')}",
            "program_motif": _safe_str(patch.get("source_kind"), "legacy"),
            "channel": "legacy",
        }
    if op == "edge_expire":
        return {
            "mode": "expire",
            "edge_type": _safe_str(patch.get("edge_type"), ""),
            "src_id": _safe_int(patch.get("src_id")),
            "dst_id": _safe_int(patch.get("dst_id")),
            "year": _safe_int(patch.get("year"), from_year),
            "reason": _safe_str(patch.get("reason"), ""),
            "source_kind": _safe_str(patch.get("source_kind"), "legacy_port"),
            "program_key": f"legacy|{_safe_str(patch.get('source_kind'), 'legacy')}",
            "program_motif": _safe_str(patch.get("source_kind"), "legacy"),
            "channel": "legacy",
        }
    if op == "person_latent_delta":
        person_id = _safe_int(patch.get("person_id"))
        delta = patch.get("delta") or {}
        if isinstance(delta, Mapping):
            for field_name, value in delta.items():
                _accumulate_person_delta(compiled.person_latent_deltas, int(person_id), str(field_name), float(value))
        return None
    if op == "company_latent_delta":
        company_id = _safe_int(patch.get("company_id"))
        delta = patch.get("delta") or {}
        if isinstance(delta, Mapping):
            payload = compiled.company_latent_deltas.setdefault(int(company_id), {})
            for field_name, value in delta.items():
                if str(field_name) in ALLOWED_COMPANY_LATENT_FIELDS:
                    payload[str(field_name)] = float(_clamp(payload.get(str(field_name), 0.0) + float(value), -0.18, 0.18))
        return None
    if op == "career_stage_transition":
        person_id = _safe_int(patch.get("person_id"))
        stage = _safe_str(patch.get("new_stage"), "").strip()
        if stage:
            compiled.career_stage_updates[int(person_id)] = stage
        return None
    if op == "company_tier_transition":
        company_id = _safe_int(patch.get("company_id"))
        tier = _safe_str(patch.get("new_tier"), "").strip()
        if tier:
            compiled.company_tier_updates[int(company_id)] = tier
        return None
    if op == "retire_person":
        person_id = _safe_int(patch.get("person_id"))
        year = _safe_int(patch.get("year"), to_year)
        if person_id > 0:
            compiled.retirements.append((int(person_id), int(year)))
        return None
    if op == "dissolve_company":
        company_id = _safe_int(patch.get("company_id"))
        year = _safe_int(patch.get("year"), to_year)
        if company_id > 0:
            compiled.dissolutions.append((int(company_id), int(year)))
        return None
    if op == "genre_trend_shift":
        genre = _safe_str(patch.get("genre"), "").strip()
        if genre:
            compiled.genre_deltas[genre] = float(_clamp(compiled.genre_deltas.get(genre, 0.0) + float(patch.get("delta", 0.0)), -0.20, 0.20))
        return None
    if op == "adjust_country_weight":
        country = _safe_str(patch.get("country"), "").strip()
        delta = float(patch.get("delta", 0.0) or 0.0)
        if country:
            compiled.country_multipliers[country] = float(_clamp(compiled.country_multipliers.get(country, 1.0) + delta, 0.70, 1.40))
        return None
    return None


def _background_candidate_ops(
    world,
    ctx: YearEvolutionContext,
    compiled: YearProgram,
    profile: GraphScaleProfile,
    budget: YearEdgeBudget,
    rng: random.Random,
) -> list[dict[str, Any]]:
    legacy_patches = _emit_edge_program_patches(
        world,
        ctx.exposure,
        ctx.from_year,
        ctx.to_year,
        rng,
        regime_label=profile.regime_label,
        budget_hint=max(256, int(budget.background_budget)),
    )
    candidate_ops: list[dict[str, Any]] = []
    for patch in legacy_patches:
        converted = _convert_legacy_patch(compiled, patch, from_year=ctx.from_year, to_year=ctx.to_year)
        if converted is not None:
            candidate_ops.append(converted)

    cache = ctx.execution_cache
    if cache is not None:
        for edge_type, rows in cache.active_edge_samples_by_family.items():
            for row in rows:
                src_id = _safe_int(row.get("src_id"))
                dst_id = _safe_int(row.get("dst_id"))
                weight = float(row.get("weight", 0.0) or 0.0)
                if src_id <= 0 or dst_id <= 0:
                    continue
                touch_active = src_id in ctx.exposure.active_person_ids or src_id in ctx.exposure.active_company_ids or dst_id in ctx.exposure.active_person_ids or dst_id in ctx.exposure.active_company_ids
                if edge_type in NEGATIVE_EDGE_TYPES:
                    delta = rng.uniform(-0.08, 0.10 if touch_active else 0.04)
                else:
                    delta = rng.uniform(-0.04, 0.08 if touch_active else 0.03)
                candidate_ops.append(
                    {
                        "mode": "update",
                        "edge_type": edge_type,
                        "src_id": int(src_id),
                        "dst_id": int(dst_id),
                        "delta_weight": float(_clamp(delta, -0.25, 0.25)),
                        "reason": "background sampled drift",
                        "source_kind": "background_drift",
                        "program_key": f"background|{edge_type}",
                        "program_motif": "background_drift",
                        "channel": "background",
                    }
                )
                if weight <= 0.22 and rng.random() < (0.20 if edge_type in NEGATIVE_EDGE_TYPES else 0.14):
                    candidate_ops.append(
                        {
                            "mode": "expire",
                            "edge_type": edge_type,
                            "src_id": int(src_id),
                            "dst_id": int(dst_id),
                            "year": int(ctx.from_year),
                            "reason": "background stale expiry",
                            "source_kind": "background_drift",
                            "program_key": f"background|{edge_type}",
                            "program_motif": "background_drift",
                            "channel": "background",
                        }
                    )
        for edge_type, rows in cache.stale_edge_samples_by_family.items():
            for row in rows[: max(16, min(192, budget.background_budget // 40))]:
                src_id = _safe_int(row.get("src_id"))
                dst_id = _safe_int(row.get("dst_id"))
                if src_id <= 0 or dst_id <= 0:
                    continue
                candidate_ops.append(
                    {
                        "mode": "expire",
                        "edge_type": edge_type,
                        "src_id": int(src_id),
                        "dst_id": int(dst_id),
                        "year": int(ctx.from_year),
                        "reason": "background stale edge trim",
                        "source_kind": "background_drift",
                        "program_key": f"background|stale|{edge_type}",
                        "program_motif": "background_drift",
                        "channel": "background",
                    }
                )
    return candidate_ops


def _cascade_candidate_ops(
    world,
    ctx: YearEvolutionContext,
    compiled: YearProgram,
    profile: GraphScaleProfile,
    budget: YearEdgeBudget,
    rng: random.Random,
) -> list[dict[str, Any]]:
    candidate_ops: list[dict[str, Any]] = []
    genre_signals = list(compiled.genre_deltas.items())
    if not genre_signals:
        total_movies = max(1, ctx.exposure.n_movies)
        for genre, count in ctx.exposure.genre_counts.most_common(4):
            signal = float(_clamp((float(count) / float(total_movies) - 0.08) * 0.18, -0.05, 0.05))
            if abs(signal) >= 0.01:
                genre_signals.append((str(genre), signal))

    for genre, delta in genre_signals[:4]:
        for patch in _amplify_genre_cascade(
            world,
            str(genre),
            float(delta),
            ctx.from_year,
            rng,
            budget_hint=max(96, int(budget.cascade_budget // max(1, len(genre_signals[:4]) + 1))),
        ):
            converted = _convert_legacy_patch(compiled, patch, from_year=ctx.from_year, to_year=ctx.to_year)
            if converted is not None:
                converted["program_key"] = f"cascade|genre|{genre}"
                converted["program_motif"] = "genre_cascade"
                converted["channel"] = "cascade"
                candidate_ops.append(converted)
    for patch in _amplify_regional_wave(
        world,
        ctx.exposure,
        ctx.from_year,
        rng,
        budget_hint=max(128, int(budget.cascade_budget // 2)),
    ):
        converted = _convert_legacy_patch(compiled, patch, from_year=ctx.from_year, to_year=ctx.to_year)
        if converted is not None:
            converted["program_key"] = "cascade|regional_wave"
            converted["program_motif"] = "regional_wave"
            converted["channel"] = "cascade"
            candidate_ops.append(converted)

    cache = ctx.execution_cache
    if cache is None:
        return candidate_ops

    for genre, people in list(cache.genre_hotspot_people.items())[:4]:
        hot_people = [int(pid) for pid in people[:256]]
        if len(hot_people) < 2:
            continue
        tmp_program = RelationshipProgram(
            motif="genre_cascade",
            edge_type="collaboration",
            mode="create",
            target_cohort="active_people",
            graph_scope="genre_hotspot",
            anchors_people=hot_people[:8],
            intensity=0.60,
            reach="regional",
            fanout=6,
            weight_mean=0.32,
            weight_spread=0.07,
            delta_weight=0.10,
            cross_community_bias=0.35,
            novelty_bias=0.25,
            durability_years=3,
            reason=f"genre cascade {genre}",
        )
        pairs = [(int(a), int(b)) for a, b in cache.top_similarity_pairs if int(a) in hot_people and int(b) in hot_people]
        if len(pairs) < 96:
            for idx in range(min(len(hot_people) - 1, 192)):
                pairs.append((int(hot_people[idx]), int(hot_people[(idx + 1) % len(hot_people)])))
        candidate_ops.extend(
            _emit_ops_from_pairs(
                world,
                ctx,
                tmp_program,
                edge_type="collaboration",
                mode="create",
                pairs=pairs,
                limit=min(1200, max(120, budget.cascade_budget // 8)),
                channel="cascade",
                rng=rng,
                seen=set(),
            )
        )
        candidate_ops.extend(
            _emit_ops_from_pairs(
                world,
                ctx,
                tmp_program,
                edge_type="friendship",
                mode="create",
                pairs=pairs,
                limit=min(900, max(90, budget.cascade_budget // 10)),
                channel="cascade",
                rng=rng,
                seen=set(),
            )
        )

    for country, people in list(cache.country_hotspot_people.items())[:4]:
        hot_people = [int(pid) for pid in people[:256]]
        if len(hot_people) < 2:
            continue
        tmp_program = RelationshipProgram(
            motif="regional_migration_wave",
            edge_type="friendship",
            mode="create",
            target_cohort="cross_market_people",
            graph_scope="country_hotspot",
            anchors_people=hot_people[:8],
            intensity=0.58,
            reach="regional",
            fanout=5,
            weight_mean=0.30,
            weight_spread=0.08,
            delta_weight=0.09,
            cross_community_bias=0.55,
            novelty_bias=0.40,
            durability_years=3,
            reason=f"regional wave {country}",
        )
        pairs = [(int(a), int(b)) for a, b in cache.cross_community_bridge_pairs if int(a) in hot_people or int(b) in hot_people]
        if len(pairs) < 64:
            for idx in range(min(len(hot_people) - 1, 128)):
                pairs.append((int(hot_people[idx]), int(hot_people[(idx + 3) % len(hot_people)])))
        candidate_ops.extend(
            _emit_ops_from_pairs(
                world,
                ctx,
                tmp_program,
                edge_type="friendship",
                mode="create",
                pairs=pairs,
                limit=min(900, max(90, budget.cascade_budget // 10)),
                channel="cascade",
                rng=rng,
                seen=set(),
            )
        )

    if cache.company_company_pairs and ctx.top_companies:
        tmp_program = RelationshipProgram(
            motif="company_competition_shift",
            edge_type="market_rival",
            mode="create",
            target_cohort="active_companies",
            graph_scope="company_pairs",
            anchors_companies=ctx.top_companies[:6],
            intensity=0.62,
            reach="industry",
            fanout=5,
            weight_mean=0.34,
            weight_spread=0.08,
            delta_weight=0.10,
            durability_years=4,
            reason="company market competition reshuffle",
        )
        company_pairs = list(cache.company_company_pairs)
        candidate_ops.extend(
            _emit_ops_from_pairs(
                world,
                ctx,
                tmp_program,
                edge_type="market_rival",
                mode="create",
                pairs=company_pairs,
                limit=min(1200, max(120, budget.cascade_budget // 7)),
                channel="cascade",
                rng=rng,
                seen=set(),
            )
        )
    return candidate_ops


def _pick_budgeted_ops(ops: Sequence[dict[str, Any]], budget: int, rng: random.Random) -> list[dict[str, Any]]:
    if budget <= 0:
        return []
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, int]] = set()
    for op in ops:
        key = (str(op.get("mode", "")), str(op.get("edge_type", "")), int(op.get("src_id", 0)), int(op.get("dst_id", 0)))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(op)
    if len(deduped) <= int(budget):
        return deduped

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for op in deduped:
        by_family[str(op.get("edge_type", ""))].append(op)
    for family_ops in by_family.values():
        rng.shuffle(family_ops)

    family_names = sorted(by_family.keys(), key=lambda name: len(by_family[name]), reverse=True)
    quotas: dict[str, int] = {}
    remaining = int(budget)
    total_count = max(1, sum(len(by_family[name]) for name in family_names))
    for idx, name in enumerate(family_names):
        desired = remaining if idx == len(family_names) - 1 else max(1, int(round(budget * len(by_family[name]) / total_count)))
        quotas[name] = min(len(by_family[name]), desired)
        remaining -= quotas[name]
        total_count -= len(by_family[name])
    if remaining > 0 and family_names:
        cursor = 0
        while remaining > 0:
            name = family_names[cursor % len(family_names)]
            if quotas[name] < len(by_family[name]):
                quotas[name] += 1
                remaining -= 1
            cursor += 1
            if cursor > len(family_names) * 8:
                break

    chosen: list[dict[str, Any]] = []
    for name in family_names:
        chosen.extend(by_family[name][: quotas[name]])
    return chosen[: int(budget)]


def _edge_family_counts(ops: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for op in ops:
        counts[str(op.get("edge_type", ""))] += 1
    return dict(counts)


def _negative_share(ops: Sequence[dict[str, Any]]) -> float:
    if not ops:
        return 0.0
    negative = sum(1 for op in ops if str(op.get("edge_type", "")) in NEGATIVE_EDGE_TYPES)
    return float(negative) / float(len(ops))


def _rebalance_edge_ops(ops: Sequence[dict[str, Any]], profile: GraphScaleProfile, rng: random.Random) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, int]] = set()
    by_family_all: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_program_all: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for op in ops:
        key = (str(op.get("mode", "")), str(op.get("edge_type", "")), int(op.get("src_id", 0)), int(op.get("dst_id", 0)))
        if key in seen:
            continue
        seen.add(key)
        op_copy = dict(op)
        deduped.append(op_copy)
        by_family_all[str(op_copy.get("edge_type", ""))].append(op_copy)
        by_program_all[str(op_copy.get("program_key", ""))].append(op_copy)

    if not deduped:
        return []

    total = len(deduped)
    family_cap = max(1, int(round(total * 0.45)))
    program_cap = max(1, int(round(total * 0.30)))
    lo, hi = NEGATIVE_SHARE_BANDS.get(profile.regime_label, NEGATIVE_SHARE_BANDS["neutral"])
    target_neg_min = int(round(total * lo))
    target_neg_max = int(round(total * hi))

    family_order = sorted(by_family_all.keys(), key=lambda name: (name in NEGATIVE_EDGE_TYPES, len(by_family_all[name])), reverse=True)
    selected: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    program_counts: Counter[str] = Counter()

    def _can_take(op: dict[str, Any]) -> bool:
        family = str(op.get("edge_type", ""))
        program_key = str(op.get("program_key", ""))
        return family_counts[family] < family_cap and (not program_key or program_counts[program_key] < program_cap)

    seeded_families = 0
    for family in family_order:
        if seeded_families >= 4:
            break
        for op in by_family_all[family]:
            if _can_take(op):
                selected.append(op)
                family_counts[family] += 1
                if str(op.get("program_key", "")):
                    program_counts[str(op.get("program_key", ""))] += 1
                seeded_families += 1
                break

    negatives = [op for op in deduped if str(op.get("edge_type", "")) in NEGATIVE_EDGE_TYPES]
    positives = [op for op in deduped if str(op.get("edge_type", "")) not in NEGATIVE_EDGE_TYPES]
    rng.shuffle(negatives)
    rng.shuffle(positives)

    current_negative = sum(1 for op in selected if str(op.get("edge_type", "")) in NEGATIVE_EDGE_TYPES)
    for op in negatives:
        if len(selected) >= total or current_negative >= target_neg_min:
            break
        if not _can_take(op):
            continue
        selected.append(op)
        family_counts[str(op.get("edge_type", ""))] += 1
        if str(op.get("program_key", "")):
            program_counts[str(op.get("program_key", ""))] += 1
        current_negative += 1

    combined = positives + negatives
    for op in combined:
        if len(selected) >= total:
            break
        is_negative = str(op.get("edge_type", "")) in NEGATIVE_EDGE_TYPES
        if is_negative and current_negative >= target_neg_max and len(selected) >= target_neg_min:
            continue
        if not _can_take(op):
            continue
        selected.append(op)
        family_counts[str(op.get("edge_type", ""))] += 1
        if str(op.get("program_key", "")):
            program_counts[str(op.get("program_key", ""))] += 1
        if is_negative:
            current_negative += 1

    return selected


def _compile_program_telemetry(
    ctx: YearEvolutionContext,
    plan: YearPlan,
    targeted_ops: Sequence[dict[str, Any]],
    background_ops: Sequence[dict[str, Any]],
    cascade_ops: Sequence[dict[str, Any]],
    final_ops: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    family_counts = _edge_family_counts(final_ops)
    mode_counts = Counter(str(op.get("mode", "")) for op in final_ops)
    touched_people: set[int] = set()
    touched_companies: set[int] = set()
    cache = ctx.execution_cache
    company_idx = cache.company_row_idx if cache is not None else {}
    for op in final_ops:
        src_id = int(op.get("src_id", 0))
        dst_id = int(op.get("dst_id", 0))
        if src_id in company_idx:
            touched_companies.add(src_id)
        elif src_id > 0:
            touched_people.add(src_id)
        if dst_id in company_idx:
            touched_companies.add(dst_id)
        elif dst_id > 0:
            touched_people.add(dst_id)
    top_motifs = Counter(program.motif for program in plan.relationship_programs)
    return {
        "from_year": int(ctx.from_year),
        "to_year": int(ctx.to_year),
        "regime_label": _safe_str(ctx.summary.get("regime_label"), "neutral"),
        "triggered_events": [_safe_str(item.get("type"), "unknown") for item in ctx.triggered_events],
        "planner_source": plan.planner_source,
        "budget_totals": {
            "social_program_budget": int(ctx.edge_budget.social_program_budget if ctx.edge_budget else 0),
            "company_program_budget": int(ctx.edge_budget.company_program_budget if ctx.edge_budget else 0),
            "background_budget": int(ctx.edge_budget.background_budget if ctx.edge_budget else 0),
            "cascade_budget": int(ctx.edge_budget.cascade_budget if ctx.edge_budget else 0),
        },
        "layer_counts": {
            "targeted": int(len(targeted_ops)),
            "background": int(len(background_ops)),
            "cascade": int(len(cascade_ops)),
            "final": int(len(final_ops)),
        },
        "ops_by_family": family_counts,
        "ops_by_mode": {mode: int(count) for mode, count in mode_counts.items()},
        "negative_share": float(_negative_share(final_ops)),
        "touched_entities": {"people": int(len(touched_people)), "companies": int(len(touched_companies))},
        "top_motifs": top_motifs.most_common(8),
        "anchors": {
            "people": [int(pid) for program in plan.relationship_programs for pid in program.anchors_people[:3]][:24],
            "companies": [int(cid) for program in plan.relationship_programs for cid in program.anchors_companies[:3]][:24],
        },
    }


def compile_year_program(
    world,
    ctx: YearEvolutionContext,
    plan: YearPlan,
    *,
    speed_audit=None,
) -> YearProgram:
    rng = random.Random(_stable_hash(f"compile|{ctx.from_year}|{ctx.to_year}|{plan.planner_source}|{len(plan.relationship_programs)}"))
    compiled = YearProgram()

    with _speed_scope(
        speed_audit,
        "year.budget_build",
        category="year_boundary",
        units=len(plan.relationship_programs) or 1,
        metadata={"from_year": int(ctx.from_year), "to_year": int(ctx.to_year)},
    ):
        profile, budget = _build_year_scale(world, ctx)

    with _speed_scope(
        speed_audit,
        "year.reservoir_build",
        category="year_boundary",
        units=profile.total_edges or 1,
        metadata={"from_year": int(ctx.from_year), "to_year": int(ctx.to_year)},
    ):
        _ensure_year_reservoirs(world, ctx, profile, budget)
        setattr(world, "_year_execution_cache_current", ctx.execution_cache)

    for move in plan.market_regime_moves:
        kind = _safe_str(move.get("kind"), "").strip()
        if kind == "genre_delta":
            genre = _safe_str(move.get("genre"), "").strip()
            if genre:
                compiled.genre_deltas[genre] = float(_clamp(compiled.genre_deltas.get(genre, 0.0) + float(move.get("delta", 0.0)), -0.20, 0.20))
        elif kind in {"country_multiplier", "country_weight"}:
            country = _safe_str(move.get("country"), "").strip()
            multiplier = float(move.get("multiplier", move.get("delta", 0.0)) or 0.0)
            if country:
                if abs(multiplier) <= 0.4:
                    multiplier = 1.0 + multiplier
                compiled.country_multipliers[country] = float(_clamp(multiplier, 0.70, 1.40))

    for shock in plan.person_shocks:
        person_id = _safe_int(shock.get("person_id"))
        if person_id <= 0:
            continue
        if shock.get("reputation_delta") not in (None, "", "nan"):
            _accumulate_person_delta(compiled.person_latent_deltas, int(person_id), "public_reputation", float(shock.get("reputation_delta", 0.0)))
        if shock.get("controversy_delta") not in (None, "", "nan"):
            _accumulate_person_delta(compiled.person_latent_deltas, int(person_id), "controversy_score", float(shock.get("controversy_delta", 0.0)))
        if shock.get("ambition_delta") not in (None, "", "nan"):
            _accumulate_person_delta(compiled.person_latent_deltas, int(person_id), "artistic_ambition", float(shock.get("ambition_delta", 0.0)))
        if shock.get("volatility_delta") not in (None, "", "nan"):
            _accumulate_person_delta(compiled.person_latent_deltas, int(person_id), "volatility", float(shock.get("volatility_delta", 0.0)))
        stage = _safe_str(shock.get("career_stage"), "").strip()
        if stage:
            compiled.career_stage_updates[int(person_id)] = stage

    for move in plan.company_moves:
        company_id = _safe_int(move.get("company_id"))
        if company_id <= 0:
            continue
        kind = _safe_str(move.get("kind"), "").strip()
        if kind == "tier_transition":
            new_tier = _safe_str(move.get("new_tier"), "").strip()
            if new_tier:
                compiled.company_tier_updates[int(company_id)] = new_tier
        elif kind == "dissolution":
            compiled.dissolutions.append((int(company_id), int(ctx.to_year)))
        elif kind == "latent_delta":
            payload = compiled.company_latent_deltas.setdefault(int(company_id), {})
            for field_name, value in (move.get("delta") or {}).items():
                if str(field_name) in ALLOWED_COMPANY_LATENT_FIELDS:
                    payload[str(field_name)] = float(_clamp(payload.get(str(field_name), 0.0) + float(value), -0.18, 0.18))

    exposure = ctx.exposure
    total_movies = max(1, exposure.n_movies)
    for genre, count in exposure.genre_counts.most_common(8):
        signal = float(_clamp((float(count) / float(total_movies) - 0.08) * 0.18, -0.05, 0.05))
        if abs(signal) >= 0.01:
            compiled.genre_deltas[str(genre)] = float(_clamp(compiled.genre_deltas.get(str(genre), 0.0) + signal, -0.20, 0.20))

    for pid in ctx.top_people[:18]:
        vals = exposure.person_perf.get(int(pid), [])
        if not vals:
            continue
        avg_rating = float(np.mean([v[0] for v in vals]))
        avg_perf = float(np.mean([v[1] for v in vals]))
        _accumulate_person_delta(compiled.person_latent_deltas, int(pid), "public_reputation", (avg_rating - 6.25) * 0.015)
        _accumulate_person_delta(compiled.person_latent_deltas, int(pid), "artistic_ambition", (avg_perf - 1.0) * 0.02)
        _accumulate_person_delta(compiled.person_latent_deltas, int(pid), "controversy_score", max(0.0, 0.60 - avg_rating) * 0.01)

    for cid in ctx.top_companies[:12]:
        vals = exposure.company_perf.get(int(cid), [])
        avg_rating = float(np.mean([v[0] for v in vals])) if vals else 6.0
        avg_perf = float(np.mean([v[1] for v in vals])) if vals else 1.0
        payload = compiled.company_latent_deltas.setdefault(int(cid), {})
        payload["prestige_score"] = float(_clamp(payload.get("prestige_score", 0.0) + (avg_rating - 6.2) * 0.02, -0.18, 0.18))
        payload["risk_appetite"] = float(_clamp(payload.get("risk_appetite", 0.0) + (avg_perf - 1.0) * 0.03, -0.18, 0.18))
        if avg_perf < 0.82 and _company_tier_for(world, int(cid)) in COMPANY_TIERS:
            current_tier = _company_tier_for(world, int(cid))
            current_idx = _COMPANY_TIER_ORDER.get(current_tier, 0)
            if current_idx > 0:
                compiled.company_tier_updates.setdefault(int(cid), COMPANY_TIERS[current_idx - 1])

    allocations = _allocate_program_budgets(plan.relationship_programs, profile, budget, ctx.triggered_events)

    with _speed_scope(
        speed_audit,
        "year.program_targeted",
        category="year_boundary",
        units=sum(int(item.targeted_ops) for item in allocations) or 1,
        metadata={"from_year": int(ctx.from_year), "to_year": int(ctx.to_year)},
    ):
        targeted_ops: list[dict[str, Any]] = []
        for allocation in allocations:
            local_rng = random.Random(_stable_hash(f"targeted|{ctx.from_year}|{allocation.program.motif}|{allocation.targeted_ops}"))
            targeted_ops.extend(_expand_relationship_program(world, ctx, allocation, local_rng))
        targeted_ops = _pick_budgeted_ops(targeted_ops, budget.social_program_budget + budget.company_program_budget, rng)

    with _speed_scope(
        speed_audit,
        "year.background_drift",
        category="year_boundary",
        units=budget.background_budget,
        metadata={"from_year": int(ctx.from_year), "to_year": int(ctx.to_year)},
    ):
        background_ops = _background_candidate_ops(world, ctx, compiled, profile, budget, rng)
        background_ops = _pick_budgeted_ops(background_ops, budget.background_budget, rng)

    with _speed_scope(
        speed_audit,
        "year.cascade_apply",
        category="year_boundary",
        units=budget.cascade_budget,
        metadata={"from_year": int(ctx.from_year), "to_year": int(ctx.to_year)},
    ):
        cascade_ops = _cascade_candidate_ops(world, ctx, compiled, profile, budget, rng)
        cascade_ops = _pick_budgeted_ops(cascade_ops, budget.cascade_budget, rng)

    with _speed_scope(
        speed_audit,
        "year.balance_finalize",
        category="year_boundary",
        units=len(targeted_ops) + len(background_ops) + len(cascade_ops),
        metadata={"from_year": int(ctx.from_year), "to_year": int(ctx.to_year)},
    ):
        compiled.edge_ops = _rebalance_edge_ops(targeted_ops + background_ops + cascade_ops, profile, rng)

    narrative = plan.world_event_narrative.strip()
    if narrative:
        compiled.world_events.append(
            {
                "event_type": "yearly_macro_plan",
                "description": narrative,
                "duration_years": int(ctx.triggered_events[0].get("default_duration", 0) or 0) if ctx.triggered_events else 0,
                "year": int(ctx.from_year),
            }
        )
    for event in ctx.triggered_events:
        compiled.world_events.append(
            {
                "event_type": _safe_str(event.get("type"), "macro_event"),
                "description": _safe_str(event.get("description"), "").strip() or f"{ctx.from_year} macro event",
                "duration_years": int(event.get("default_duration", 0) or 0),
                "year": int(ctx.from_year),
            }
        )
    compiled.telemetry = _compile_program_telemetry(ctx, plan, targeted_ops, background_ops, cascade_ops, compiled.edge_ops)
    return compiled


def _invalidate_selection_caches(
    world,
    *,
    invalidate_actor_cache: bool = False,
    invalidate_company_cache: bool = False,
) -> None:
    attrs = [
        "_selection_year_state_cache",
        "_director_year_edge_cache",
        "_director_pc_affinity_cache",
        "_pid_to_career_stage",
    ]
    if invalidate_actor_cache:
        attrs.extend(
            [
                "_cast_year_cache",
                "_crew_year_pool_cache",
            ]
        )
    if invalidate_company_cache:
        attrs.append("_company_by_tier_genre")

    for attr in attrs:
        cache = getattr(world, attr, None)
        if isinstance(cache, dict):
            cache.clear()
        elif cache is not None:
            setattr(world, attr, None)


def apply_year_program(world, compiled: YearProgram, *, from_year: int, to_year: int, ctx: YearEvolutionContext | None = None) -> YearApplyReport:
    rep = YearApplyReport()
    _ensure_temporal_fields(world)

    execution_cache = ctx.execution_cache if ctx is not None else None
    person_row_idx: dict[int, int] = execution_cache.person_row_idx if execution_cache is not None else {}
    person_col_idx: dict[str, int] = execution_cache.person_col_idx if execution_cache is not None else {}
    company_row_idx: dict[int, int] = execution_cache.company_row_idx if execution_cache is not None else {}
    company_col_idx: dict[str, int] = execution_cache.company_col_idx if execution_cache is not None else {}
    if not person_row_idx and getattr(world, "persons", None) is not None and "person_id" in world.persons.columns:
        pids = world.persons["person_id"].astype(int)
        person_row_idx = {int(pid): i for i, pid in enumerate(pids)}
        person_col_idx = {c: i for i, c in enumerate(world.persons.columns)}
    if not company_row_idx and getattr(world, "companies", None) is not None and "company_id" in world.companies.columns:
        cids = world.companies["company_id"].astype(int)
        company_row_idx = {int(cid): i for i, cid in enumerate(cids)}
        company_col_idx = {c: i for i, c in enumerate(world.companies.columns)}

    invalidate_actor_cache = False
    invalidate_company_cache = False

    for genre, delta in compiled.genre_deltas.items():
        old = float(getattr(world, "genre_weight_overrides", {}).get(genre, 0.0))
        world.genre_weight_overrides[genre] = _clamp(old + delta, -0.20, 0.20)
        rep.applied += 1
    for country, multiplier in compiled.country_multipliers.items():
        world.country_weight_overrides[country] = float(multiplier)
        rep.applied += 1

    for pid, delta in compiled.person_latent_deltas.items():
        lv = getattr(world, "person_latent", {}).get(int(pid))
        if not isinstance(lv, dict):
            lv = {"person_id": int(pid)}
            world.person_latent[int(pid)] = lv
        for field_name, value in delta.items():
            if field_name not in ALLOWED_PERSON_LATENT_FIELDS:
                continue
            old = _clamp(lv.get(field_name, 0.5), 0.0, 1.0)
            lv[field_name] = _clamp(old + value, 0.0, 1.0)
        rep.applied += 1

    for cid, delta in compiled.company_latent_deltas.items():
        lv = getattr(world, "company_latent", {}).get(int(cid))
        if not isinstance(lv, dict):
            lv = {"company_id": int(cid)}
            world.company_latent[int(cid)] = lv
        for field_name, value in delta.items():
            if field_name not in ALLOWED_COMPANY_LATENT_FIELDS:
                continue
            old = _clamp(lv.get(field_name, 0.5), 0.0, 1.0)
            lv[field_name] = _clamp(old + value, 0.0, 1.0)
        rep.applied += 1

    for pid, new_stage in compiled.career_stage_updates.items():
        row_idx = person_row_idx.get(int(pid))
        if row_idx is None or "career_stage" not in person_col_idx:
            rep.skipped += 1
            continue
        world.persons.iat[row_idx, person_col_idx["career_stage"]] = str(new_stage)
        invalidate_actor_cache = True
        rep.applied += 1

    for pid, year in compiled.retirements:
        row_idx = person_row_idx.get(int(pid))
        if row_idx is None:
            rep.skipped += 1
            continue
        if "retirement_year" in person_col_idx:
            old = world.persons.iat[row_idx, person_col_idx["retirement_year"]]
            current = 2100 if old is None or (isinstance(old, float) and old != old) else int(old)
            world.persons.iat[row_idx, person_col_idx["retirement_year"]] = min(current, int(year))
        if "career_stage" in person_col_idx:
            world.persons.iat[row_idx, person_col_idx["career_stage"]] = "retired"
        invalidate_actor_cache = True
        rep.applied += 1

    for cid, new_tier in compiled.company_tier_updates.items():
        row_idx = company_row_idx.get(int(cid))
        if row_idx is None or "tier" not in company_col_idx:
            rep.skipped += 1
            continue
        world.companies.iat[row_idx, company_col_idx["tier"]] = str(new_tier)
        invalidate_company_cache = True
        rep.applied += 1

    for cid, year in compiled.dissolutions:
        row_idx = company_row_idx.get(int(cid))
        if row_idx is None or "defunct_year" not in company_col_idx:
            rep.skipped += 1
            continue
        old = world.companies.iat[row_idx, company_col_idx["defunct_year"]]
        current = None if old is None or (isinstance(old, float) and old != old) else int(old)
        world.companies.iat[row_idx, company_col_idx["defunct_year"]] = int(year) if current is None else min(current, int(year))
        invalidate_company_cache = True
        rep.applied += 1

    graph = getattr(world, "graph", None)
    if graph is None:
        rep.skipped += len(compiled.edge_ops)
    elif compiled.edge_ops:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for op in compiled.edge_ops:
            grouped.setdefault((str(op.get("mode", "")), str(op.get("edge_type", ""))), []).append(op)
        for (mode, edge_type), ops in grouped.items():
            applied, skipped, errors, messages = graph.apply_edge_batch(ops, default_from_year=int(from_year))
            rep.applied += int(applied)
            rep.skipped += int(skipped)
            rep.errors += int(errors)
            rep.edge_applied_by_family[str(edge_type)] = rep.edge_applied_by_family.get(str(edge_type), 0) + int(applied)
            rep.edge_applied_by_mode[str(mode)] = rep.edge_applied_by_mode.get(str(mode), 0) + int(applied)
            for msg in messages:
                rep.log(msg)

    for item in compiled.world_events:
        description = _safe_str(item.get("description"), "").strip()
        if not description:
            continue
        world.world_events.append(
            {
                "event_id": len(world.world_events) + 1,
                "year": int(item.get("year", from_year)),
                "event_type": _safe_str(item.get("event_type"), "macro_event"),
                "description": description,
                "duration_years": int(item.get("duration_years", 0) or 0),
                "affected_entity_id": None,
                "affected_entity_type": None,
                "parameter_delta_json": None,
            }
        )
        rep.applied += 1

    if invalidate_actor_cache:
        _invalidate_year_cache(world)
    _invalidate_selection_caches(
        world,
        invalidate_actor_cache=invalidate_actor_cache,
        invalidate_company_cache=invalidate_company_cache,
    )
    rep.invalidated_actor_cache = bool(invalidate_actor_cache)
    rep.invalidated_company_cache = bool(invalidate_company_cache)
    return rep


def _write_year_summary(world, ctx: YearEvolutionContext, plan: YearPlan, compiled: YearProgram, rep: YearApplyReport) -> None:
    try:
        out_dir = Path(getattr(world, "base_dir", ".")) / "graph" / "temporal_patches"
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = dict(compiled.telemetry)
        summary["applied_report"] = {
            "applied": int(rep.applied),
            "skipped": int(rep.skipped),
            "errors": int(rep.errors),
            "edge_applied_by_family": {k: int(v) for k, v in rep.edge_applied_by_family.items()},
            "edge_applied_by_mode": {k: int(v) for k, v in rep.edge_applied_by_mode.items()},
        }
        summary["planner"] = {
            "source": plan.planner_source,
            "parse_error": str(getattr(plan, "parse_error", "") or ""),
            "raw_text_chars": len(str(getattr(plan, "raw_text", "") or "")),
            "relationship_programs": [
                {
                    "motif": program.motif,
                    "edge_type": program.edge_type,
                    "mode": program.mode,
                    "intensity": float(program.intensity),
                    "reach": program.reach,
                    "anchors_people": [int(pid) for pid in program.anchors_people[:8]],
                    "anchors_companies": [int(cid) for cid in program.anchors_companies[:8]],
                    "reason": program.reason,
                }
                for program in plan.relationship_programs
            ],
        }
        path = out_dir / f"year_summary_{int(ctx.from_year)}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        return


def evolve_year(
    world,
    *,
    from_year: int,
    to_year: int,
    year_bucket: Sequence[dict[str, Any]],
    enable_llm: bool,
    model: str | None = None,
    log_dir: str | None = None,
    speed_audit=None,
) -> YearApplyReport:
    if not year_bucket:
        rep = YearApplyReport()
        rep.log("evolve_year: empty year bucket")
        return rep

    with _speed_scope(
        speed_audit,
        "year.context_build",
        category="year_boundary",
        units=len(year_bucket),
        metadata={"from_year": int(from_year), "to_year": int(to_year)},
    ):
        ctx = build_year_context(world, int(from_year), int(to_year), year_bucket)

    with _speed_scope(
        speed_audit,
        "year.plan_llm",
        category="year_boundary",
        units=len(ctx.triggered_events) or 1,
        metadata={"from_year": int(from_year), "to_year": int(to_year), "model": model or "", "enable_llm": bool(enable_llm)},
    ):
        plan = plan_year(world, ctx, use_llm=bool(enable_llm), model=model, log_dir=log_dir)

    with _speed_scope(
        speed_audit,
        "year.program_compile",
        category="year_boundary",
        units=len(plan.relationship_programs),
        metadata={"from_year": int(from_year), "to_year": int(to_year), "planner_source": plan.planner_source},
    ):
        compiled = compile_year_program(world, ctx, plan, speed_audit=speed_audit)

    with _speed_scope(
        speed_audit,
        "year.program_apply",
        category="year_boundary",
        units=len(compiled.edge_ops),
        metadata={"from_year": int(from_year), "to_year": int(to_year), "planner_source": plan.planner_source},
    ):
        rep = apply_year_program(world, compiled, from_year=int(from_year), to_year=int(to_year), ctx=ctx)

    _write_year_summary(world, ctx, plan, compiled, rep)
    setattr(world, "_year_execution_cache_current", None)
    rep.triggered_events = [_safe_str(item.get("type"), "unknown") for item in ctx.triggered_events]
    rep.planner_source = plan.planner_source
    return rep


def procedural_year_step(world, from_year: int, to_year: int, year_bucket: list[dict[str, Any]]) -> PatchApplyReport:
    rep = evolve_year(world, from_year=from_year, to_year=to_year, year_bucket=year_bucket, enable_llm=False)
    return PatchApplyReport(applied=rep.applied, skipped=rep.skipped, errors=rep.errors, messages=rep.messages)


def llm_year_step(
    world,
    from_year: int,
    to_year: int,
    year_bucket: list[dict[str, Any]],
    model: str | None = None,
    log_dir: str | None = None,
    api_key_env: str = "GEMINI_API_KEY",
) -> PatchApplyReport:
    _ = api_key_env
    rep = evolve_year(world, from_year=from_year, to_year=to_year, year_bucket=year_bucket, enable_llm=True, model=model, log_dir=log_dir)
    return PatchApplyReport(applied=rep.applied, skipped=rep.skipped, errors=rep.errors, messages=rep.messages)
