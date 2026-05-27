from __future__ import annotations

"""Big history events for the synthetic film-industry world.

This rewrite keeps the public surface compatible with the rest of the pipeline,
but fixes the main structural problems from the legacy version:

- event triggering is deterministic per year when no RNG is supplied
- year-range gating is explicit and enforced centrally
- prompt construction and execution are split into clean helpers
- fallback behaviour is meaningful and always logs a world event
- malformed LLM output does not silently poison the pipeline
- logging/reporting are stable and easier to inspect
"""

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from bootstrap_artifacts import audit_artifact_usage, audit_fallback_hit, current_mode, load_modeling_priors_artifact, prior_section
from llm_provider import get_llm_client
from pipeline_runtime import year_bounds_from_env
from policy_runtime import modeling_priors_path

try:
    from contracts import COUNTRIES, GENRES, MODEL_TIERS
except Exception:  # pragma: no cover
    COUNTRIES = []
    GENRES = []
    MODEL_TIERS = {}
from llm_master import ActionReport, LLMMasterClass


_HISTORY_PRIORS_SENTINEL = object()
_HISTORY_PRIORS_CACHE: dict[str, Any] | object = _HISTORY_PRIORS_SENTINEL


# ---------------------------------------------------------------------------
# Event registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EventSpec:
    event_type: str
    description: str
    prob: float
    year_range: tuple[float, float]
    default_duration: int
    suggested_actions: tuple[str, ...]

    def active_in_year(self, year: int) -> bool:
        lo, hi = _event_year_range(self.year_range)
        return lo <= int(year) <= hi


def _event_year_range(year_range: tuple[float, float]) -> tuple[int, int]:
    src_lo, src_hi = year_range
    dst_lo, dst_hi = year_bounds_from_env(1950, 2025)
    if dst_hi <= dst_lo:
        return int(dst_lo), int(dst_hi)

    try:
        src_lo_f = float(src_lo)
        src_hi_f = float(src_hi)
    except Exception:
        src_lo_f = float(dst_lo)
        src_hi_f = float(dst_hi)

    if max(abs(src_lo_f), abs(src_hi_f)) <= 1.0:
        lo_frac = max(0.0, min(1.0, src_lo_f))
        hi_frac = max(lo_frac, min(1.0, src_hi_f))
        dst_span = max(1, int(dst_hi) - int(dst_lo))
        mapped_lo = int(round(dst_lo + lo_frac * dst_span))
        mapped_hi = int(round(dst_lo + hi_frac * dst_span))
        return int(max(dst_lo, min(dst_hi, mapped_lo))), int(max(mapped_lo, min(dst_hi, mapped_hi)))

    src_span = max(1, int(src_hi) - int(src_lo))
    dst_span = max(1, int(dst_hi) - int(dst_lo))
    template_lo, template_hi = _event_template_bounds()
    template_span = max(1, template_hi - template_lo)
    lo_frac = (int(src_lo) - template_lo) / float(template_span)
    hi_frac = (int(src_hi) - template_lo) / float(template_span)
    mapped_lo = int(round(dst_lo + lo_frac * dst_span))
    mapped_hi = int(round(dst_lo + hi_frac * dst_span))
    mapped_lo = max(dst_lo, min(dst_hi, mapped_lo))
    mapped_hi = max(mapped_lo, min(dst_hi, mapped_hi))
    if mapped_hi - mapped_lo < max(1, src_span // 4):
        mapped_hi = min(dst_hi, mapped_lo + max(1, src_span // 4))
    return int(mapped_lo), int(mapped_hi)


BIG_EVENT_SPECS: tuple[EventSpec, ...] = (
    EventSpec(
        event_type="market_crash",
        description=(
            "A financial shock contracts studio risk appetite. Big-budget slates "
            "thin out, weaker companies struggle, and prestige-indie production rises."
        ),
        prob=0.05,
        year_range=(1970, 2035),
        default_duration=2,
        suggested_actions=(
            "dissolve_company for weak studios",
            "adjust_genre_weight toward Drama or Thriller",
            "career_pause for overexposed stars",
            "create_world_event",
        ),
    ),
    EventSpec(
        event_type="streaming_revolution",
        description=(
            "A distribution platform shift changes release economics and content shape. "
            "Firms that adapt gain reach while theatrical purists lose leverage."
        ),
        prob=0.06,
        year_range=(2005, 2035),
        default_duration=0,
        suggested_actions=(
            "adjust_genre_weight for streaming-friendly genres",
            "adjust_company_specialty",
            "company_tier_transition",
            "create_world_event",
        ),
    ),
    EventSpec(
        event_type="scandal",
        description=(
            "A public scandal damages a prominent career and ripples through the "
            "surrounding professional network."
        ),
        prob=0.08,
        year_range=(1970, 2035),
        default_duration=3,
        suggested_actions=(
            "career_pause or retire_person",
            "cascade_edge_change",
            "person_latent_delta",
            "boost_person for replacements",
            "create_world_event",
        ),
    ),
    EventSpec(
        event_type="genre_boom",
        description=(
            "A genre surges in cultural and commercial relevance, pulling talent and "
            "capital toward it for several years."
        ),
        prob=0.10,
        year_range=(1970, 2035),
        default_duration=3,
        suggested_actions=(
            "adjust_genre_weight",
            "change_genre_affinity",
            "change_style_tags",
            "boost_person",
            "create_world_event",
        ),
    ),
    EventSpec(
        event_type="country_emergence",
        description=(
            "A country's film industry gains global momentum through breakout works, "
            "new export strength, and clustered talent visibility."
        ),
        prob=0.04,
        year_range=(1985, 2035),
        default_duration=5,
        suggested_actions=(
            "adjust_country_weight",
            "boost_person",
            "change_director_specialty",
            "create_world_event",
        ),
    ),
    EventSpec(
        event_type="award_controversy",
        description=(
            "An awards-season controversy changes the prestige equilibrium around a "
            "major ceremony and affects the careers tied to it."
        ),
        prob=0.06,
        year_range=(1975, 2035),
        default_duration=2,
        suggested_actions=(
            "award_prestige_shift",
            "person_latent_delta",
            "edge_add or edge_expire",
            "create_world_event",
        ),
    ),
    EventSpec(
        event_type="studio_merger",
        description=(
            "A merger consolidates catalogs, capital, and influence, altering the "
            "production network and company hierarchy."
        ),
        prob=0.05,
        year_range=(1980, 2035),
        default_duration=0,
        suggested_actions=(
            "merge_companies",
            "adjust_company_specialty",
            "company_tier_transition",
            "create_world_event",
        ),
    ),
    EventSpec(
        event_type="tech_disruption",
        description=(
            "A technological shift changes production capability and genre economics, "
            "rewarding early adopters and exposing laggards."
        ),
        prob=0.04,
        year_range=(1990, 2035),
        default_duration=3,
        suggested_actions=(
            "adjust_genre_weight",
            "career_stage_transition",
            "career_pause",
            "company_tier_transition",
            "create_world_event",
        ),
    ),
)

_DEFAULT_HISTORY_EVENT_PRIORS: dict[str, Any] = {
    "event_specs": [
        {
            "event_type": spec.event_type,
            "description": spec.description,
            "prob": spec.prob,
            "year_range": [spec.year_range[0], spec.year_range[1]],
            "default_duration": spec.default_duration,
            "suggested_actions": list(spec.suggested_actions),
        }
        for spec in BIG_EVENT_SPECS
    ]
}


def _history_priors_payload() -> dict[str, Any]:
    global _HISTORY_PRIORS_CACHE
    if _HISTORY_PRIORS_CACHE is _HISTORY_PRIORS_SENTINEL:
        try:
            payload = load_modeling_priors_artifact(os.path.dirname(__file__))
        except Exception:
            payload = {}
        _HISTORY_PRIORS_CACHE = payload if isinstance(payload, dict) else {}
    return _HISTORY_PRIORS_CACHE if isinstance(_HISTORY_PRIORS_CACHE, dict) else {}


def _history_priors_block() -> dict[str, Any]:
    block = prior_section(_history_priors_payload(), "history_event_priors")
    if current_mode() == "research" and (not isinstance(block, dict) or not block):
        audit_fallback_hit(
            "history_event_priors",
            "missing:section",
            detail="modeling_priors missing history_event_priors for big-history events in research mode",
            mode="research",
        )
    audit_artifact_usage("modeling_priors.json", modeling_priors_path(os.path.dirname(__file__)), sections=["history_event_priors"])
    return block if isinstance(block, dict) else {}


def _coerce_event_specs(raw_specs: Any) -> tuple[EventSpec, ...]:
    if not isinstance(raw_specs, list):
        return ()
    out: list[EventSpec] = []
    for row in raw_specs:
        if not isinstance(row, dict):
            continue
        event_type = str(row.get("event_type", "")).strip()
        description = str(row.get("description", "")).strip()
        if not event_type or not description:
            continue
        try:
            prob = float(row.get("prob", 0.0))
        except Exception:
            prob = 0.0
        prob = max(0.0, min(1.0, prob))
        raw_range = row.get("year_range_fraction")
        if isinstance(raw_range, (int, float)):
            frac = max(0.0, min(1.0, float(raw_range)))
            raw_range = (0.0, frac)
        elif isinstance(raw_range, str):
            try:
                frac = max(0.0, min(1.0, float(raw_range.strip())))
            except Exception:
                frac = None
            raw_range = (0.0, frac) if frac is not None else raw_range
        if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
            raw_range = row.get("year_range")
        if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
            continue
        try:
            year_range = (float(raw_range[0]), float(raw_range[1]))
        except Exception:
            continue
        try:
            default_duration = int(row.get("default_duration", 0) or 0)
        except Exception:
            default_duration = 0
        suggested_actions_raw = row.get("suggested_actions", [])
        suggested_actions = tuple(str(x).strip() for x in suggested_actions_raw if str(x).strip()) if isinstance(suggested_actions_raw, list) else tuple()
        out.append(
            EventSpec(
                event_type=event_type,
                description=description,
                prob=prob,
                year_range=year_range,
                default_duration=default_duration,
                suggested_actions=suggested_actions,
            )
        )
    return tuple(out)


def _active_event_specs() -> tuple[EventSpec, ...]:
    block = _history_priors_block()
    specs = _coerce_event_specs(block.get("event_specs"))
    if not specs and current_mode() == "research":
        audit_fallback_hit(
            "history_event_priors",
            "missing:event_specs",
            detail="history_event_priors.event_specs is required in research mode",
            mode="research",
        )
    return specs or BIG_EVENT_SPECS


def _event_template_bounds() -> tuple[int, int]:
    specs = _active_event_specs()
    absolute_ranges = [
        (int(round(spec.year_range[0])), int(round(spec.year_range[1])))
        for spec in specs
        if max(abs(float(spec.year_range[0])), abs(float(spec.year_range[1]))) > 1.0
    ]
    if not absolute_ranges:
        return (
            min(int(spec.year_range[0]) for spec in BIG_EVENT_SPECS),
            max(int(spec.year_range[1]) for spec in BIG_EVENT_SPECS),
        )
    return (
        min(lo for lo, _ in absolute_ranges),
        max(hi for _, hi in absolute_ranges),
    )


def _big_event_types() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for spec in _active_event_specs():
        rows.append(
            {
                "type": spec.event_type,
                "description": spec.description,
                "prob": spec.prob,
                "year_range": _event_year_range(spec.year_range),
                "default_duration": spec.default_duration,
                "suggested_actions": list(spec.suggested_actions),
            }
        )
    return rows


BIG_EVENT_TYPES: List[Dict[str, Any]] = [
    {
        "type": spec.event_type,
        "description": spec.description,
        "prob": spec.prob,
        "year_range": _event_year_range(spec.year_range),
        "default_duration": spec.default_duration,
        "suggested_actions": list(spec.suggested_actions),
    }
    for spec in BIG_EVENT_SPECS
]

_EVENT_BY_TYPE = {spec.event_type: spec for spec in BIG_EVENT_SPECS}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _year_rng(world, year: int) -> random.Random:
    seed = int(getattr(world, "seed", 42))
    return random.Random((seed * 1_000_003) ^ (int(year) * 9_973))


def _safe_json_loads(text: str) -> Any:
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass

    starts = [text.find("{"), text.find("[")]
    starts = [s for s in starts if s != -1]
    if not starts:
        return None
    start = min(starts)
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        s = text.find(open_ch)
        e = text.rfind(close_ch)
        if s != -1 and e != -1 and e > s:
            try:
                return json.loads(text[s : e + 1])
            except Exception:
                continue
    return None


def _json_default(obj: Any) -> Any:
    try:
        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _write_log_file(path: Path, content: Any, is_json: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_json:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False, indent=2, default=_json_default)
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(content))


def _top_counts(values: Iterable[str], limit: int = 5) -> List[tuple[str, int]]:
    counts: Dict[str, int] = {}
    for value in values:
        key = str(value or "").strip()
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]


def _sample_people(world, year: int, limit: int = 0) -> List[Dict[str, Any]]:
    """Sample active people for BHE prompt. Limit defaults to scaled value."""
    if getattr(world, "persons", None) is None or len(world.persons) == 0:
        return []
    df = world.persons
    n_persons = len(df)
    if limit <= 0:
        limit = min(100, max(40, n_persons // 200))
    active = df
    year_lo, year_hi = year_bounds_from_env(1950, 2025)
    if {"debut_year", "retirement_year"}.issubset(df.columns):
        active = df[
            (df["debut_year"].fillna(year_lo).astype(int) <= int(year))
            & (df["retirement_year"].fillna(year_hi + 40).astype(float) >= float(year))
        ]
        if len(active) == 0:
            active = df

    sample_size = min(int(limit), len(active))
    if sample_size <= 0:
        return []

    try:
        chosen = active.sample(sample_size, random_state=int(year) + 17)
    except Exception:
        chosen = active.head(sample_size)

    rows: List[Dict[str, Any]] = []
    for _, row in chosen.iterrows():
        pid = int(row.get("person_id", 0) or 0)
        lv = getattr(world, "person_latent", {}).get(pid, {}) or {}
        rows.append(
            {
                "person_id": pid,
                "name": row.get("name"),
                "career_stage": row.get("career_stage"),
                "genre_affinity": row.get("genre_affinity"),
                "style_tags": row.get("style_tags"),
                "pop_weight": round(float(row.get("pop_weight", 0.1) or 0.1), 4),
                "public_reputation": round(float(lv.get("public_reputation", 0.5) or 0.5), 4),
                "controversy_score": round(float(lv.get("controversy_score", 0.15) or 0.15), 4),
            }
        )
    return rows


def _sample_companies(world, year: int, limit: int = 25) -> List[Dict[str, Any]]:
    if getattr(world, "companies", None) is None or len(world.companies) == 0:
        return []
    df = world.companies
    active = df
    if "defunct_year" in df.columns:
        active = df[df["defunct_year"].isna() | (df["defunct_year"].fillna(2100).astype(float) > float(year))]
        if len(active) == 0:
            active = df

    sample_size = min(int(limit), len(active))
    if sample_size <= 0:
        return []

    try:
        chosen = active.sample(sample_size, random_state=int(year) + 29)
    except Exception:
        chosen = active.head(sample_size)

    rows: List[Dict[str, Any]] = []
    for _, row in chosen.iterrows():
        cid = int(row.get("company_id", 0) or 0)
        rows.append(
            {
                "company_id": cid,
                "name": row.get("name"),
                "tier": row.get("tier"),
                "specialty_genres": row.get("specialty_genres"),
                "pop_weight": round(float(row.get("pop_weight", 0.2) or 0.2), 4),
            }
        )
    return rows


def _recent_world_events(world, year: int, horizon: int = 3, limit: int = 5) -> List[Dict[str, Any]]:
    events = []
    for event in getattr(world, "world_events", []) or []:
        ey = int(event.get("year", 0) or 0)
        if ey >= int(year) - int(horizon):
            events.append(
                {
                    "year": ey,
                    "event_type": event.get("event_type"),
                    "description": event.get("description"),
                }
            )
    return events[-limit:]


def _build_world_summary(world, year: int, year_bucket: List[Dict[str, Any]]) -> Dict[str, Any]:
    avg_rating = 6.0
    avg_perf = 1.0
    if year_bucket:
        ratings = [float(m.get("rating", 6.0) or 6.0) for m in year_bucket]
        perfs = [
            float(m.get("performance_ratio", float(m.get("box_office_usd", 0) or 0.0) / max(1.0, float(m.get("budget_usd", 1) or 1.0))))
            for m in year_bucket
        ]
        avg_rating = sum(ratings) / max(1, len(ratings))
        avg_perf = sum(perfs) / max(1, len(perfs))

    top_genres = _top_counts((m.get("genre", "") for m in year_bucket), limit=5)
    top_countries = _top_counts((m.get("country", "") for m in year_bucket), limit=5)

    return {
        "current_year": int(year),
        "movies_produced_this_year": int(len(year_bucket)),
        "avg_rating": round(float(avg_rating), 3),
        "avg_boxoffice_over_budget": round(float(avg_perf), 3),
        "top_genres_this_year": top_genres,
        "top_countries_this_year": top_countries,
        "active_genre_weight_overrides": dict(getattr(world, "genre_weight_overrides", {}) or {}),
        "active_country_weight_overrides": dict(getattr(world, "country_weight_overrides", {}) or {}),
        "sample_persons": _sample_people(world, year, limit=40),
        "sample_companies": _sample_companies(world, year, limit=25),
        "recent_world_events": _recent_world_events(world, year, horizon=3, limit=5),
    }


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------


_ACTION_MENU = """
AVAILABLE ACTIONS (use exact action names):

Person-level:
  career_pause        {person_id, duration_years:1-5}
  boost_person        {person_id, multiplier:1.0-5.0, duration_years:1-5}
  change_genre_affinity {person_id, genre, delta}
  change_style_tags   {person_id, add_tags:[], remove_tags:[]}
  change_director_specialty {person_id, genres:[]}
  retire_person       {person_id, year}
  career_stage_transition {person_id, new_stage}
  person_latent_delta {person_id, delta:{field:value}}

Edge-level:
  cascade_edge_change {person_id, delta:-0.5..0.5, n:1-20}
  edge_add            {edge_type, src_id, dst_id, sign, weight:0-1, valid_from}
  edge_update         {edge_type, src_id, dst_id, delta_weight:-0.2..0.2}
  edge_expire         {edge_type, src_id, dst_id, year}

Company-level:
  merge_companies     {company_a_id, company_b_id}
  adjust_company_specialty {company_id, genres:[]}
  dissolve_company    {company_id, year}
  company_tier_transition {company_id, new_tier}

Market-level:
  adjust_genre_weight    {genre, multiplier:0.1-5.0, duration_years}
  adjust_country_weight  {country, multiplier:0.1-10.0, duration_years}
  award_prestige_shift   {ceremony, delta:-0.5..0.5}
  genre_trend_shift      {genre, delta:-0.05..0.05}

Required:
  create_world_event  {event_type, narrative, duration_years,
                       affected_entity_id?, affected_entity_type?}
""".strip()


def build_big_event_prompt(
    world,
    year: int,
    triggered_events: Sequence[Dict[str, Any]],
    year_bucket: List[Dict[str, Any]],
    *,
    target_n_actions: int = 20,
) -> str:
    summary = _build_world_summary(world, year, year_bucket)
    event_block = "\n\n".join(
        (
            f"EVENT: {event['type'].upper()}\n"
            f"Description: {event['description']}\n"
            f"Suggested actions: {'; '.join(event.get('suggested_actions', []))}"
        )
        for event in triggered_events
    )
    n_persons = len(world.persons) if getattr(world, "persons", None) is not None else "unknown"

    return (
        f"You are a world-simulation agent for a synthetic film industry database.\n\n"
        f"Year {int(year)} has triggered the following macro event(s):\n\n"
        f"{event_block}\n\n"
        f"Current world state:\n"
        f"{json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default)}\n\n"
        f"{_ACTION_MENU}\n\n"
        f"RULES:\n"
        f"- Only reference person_id and company_id values visible in the world-state sample.\n"
        f"- You MUST include at least one create_world_event action.\n"
        f"- Produce EXACTLY {target_n_actions} actions. The world has {n_persons} people — scale your impact accordingly.\n"
        f"- Macro events should create visible ripple effects proportional to population size.\n"
        f"- Prefer bounded, realistic interventions over random noise.\n"
        f"- Duration-based actions auto-restore. Permanent actions do not.\n"
        f"- If an event affects a genre or country, use adjust_genre_weight or adjust_country_weight.\n"
        f"- If an event affects reputation or controversy, use person_latent_delta and/or cascade_edge_change.\n\n"
        f"Return ONLY valid JSON with this exact shape:\n"
        f"{{\n"
        f"  \"narrative\": \"1-2 sentence summary of what happened in {int(year)}\",\n"
        f"  \"actions\": [\n"
        f"    {{\"action\": \"action_name\", \"params\": {{...}}}},\n"
        f"    ...\n"
        f"  ]\n"
        f"}}\n"
    )


# ---------------------------------------------------------------------------
# Event rolling + fallback
# ---------------------------------------------------------------------------


def roll_events(year: int, rng: random.Random) -> List[Dict[str, Any]]:
    year = int(year)
    triggered: List[Dict[str, Any]] = []
    for spec in _active_event_specs():
        if not spec.active_in_year(year):
            continue
        if rng.random() < float(spec.prob):
            triggered.append(
                {
                    "type": spec.event_type,
                    "description": spec.description,
                    "prob": spec.prob,
                    "year_range": _event_year_range(spec.year_range),
                    "default_duration": spec.default_duration,
                    "suggested_actions": list(spec.suggested_actions),
                }
            )
    return triggered


def _fallback_actions_for_event(world, year: int, event: Dict[str, Any]) -> List[Dict[str, Any]]:
    event_type = str(event.get("type", "unknown"))
    duration = int(event.get("default_duration", 0) or 0)
    actions: List[Dict[str, Any]] = [
        {
            "action": "create_world_event",
            "params": {
                "event_type": event_type,
                "narrative": f"[fallback] {event.get('description', event_type)}",
                "duration_years": duration,
            },
        }
    ]

    # Keep fallback effects conservative but nontrivial.
    if event_type == "genre_boom" and GENRES:
        genre = GENRES[year % len(GENRES)]
        actions.append(
            {
                "action": "adjust_genre_weight",
                "params": {"genre": genre, "multiplier": 1.8, "duration_years": max(1, duration or 2)},
            }
        )
    elif event_type == "country_emergence" and COUNTRIES:
        country = COUNTRIES[year % len(COUNTRIES)]
        actions.append(
            {
                "action": "adjust_country_weight",
                "params": {"country": country, "multiplier": 2.0, "duration_years": max(2, duration or 3)},
            }
        )
    elif event_type == "award_controversy":
        actions.append(
            {
                "action": "award_prestige_shift",
                "params": {"ceremony": "Synthetic Awards", "delta": -0.2},
            }
        )
    elif event_type == "streaming_revolution" and GENRES:
        genre = "Drama" if "Drama" in GENRES else GENRES[0]
        actions.append(
            {
                "action": "adjust_genre_weight",
                "params": {"genre": genre, "multiplier": 1.6, "duration_years": 3},
            }
        )

    return actions


def _run_fallback(world, year: int, triggered: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    master = LLMMasterClass(world)
    applied_total = 0
    skipped_total = 0
    skipped_reasons: List[str] = []

    for event in triggered:
        rep = master.execute({"actions": _fallback_actions_for_event(world, year, event)}, year=year)
        applied_total += int(rep.applied)
        skipped_total += int(rep.skipped)
        skipped_reasons.extend(list(rep.skipped_reasons))

    return {
        "year": int(year),
        "events": [str(e.get("type", "unknown")) for e in triggered],
        "applied": applied_total,
        "skipped": skipped_total,
        "skipped_reasons": skipped_reasons[:10],
        "note": "fallback",
    }


# ---------------------------------------------------------------------------
# LLM execution
# ---------------------------------------------------------------------------


def _resolve_llm():
    try:
        return get_llm_client()
    except Exception:
        return None


def _call_llm(llm, model: str, prompt: str):
    response = llm.generate(
        prompt,
        model=model,
        json_mode=True,
        temperature=0.75,
        timeout_sec=70,
        max_attempts=5,
        on_retry=lambda attempt, total, exc, sleep_for: print(
            f"  [big_history_events] retry {attempt}/{total} in {sleep_for:.1f}s: {exc}"
        ),
    )
    raw_text = response.text.strip()
    parsed = _safe_json_loads(raw_text)
    return raw_text, parsed


def _ensure_world_event(master: LLMMasterClass, year: int, triggered: Sequence[Dict[str, Any]], payload: Dict[str, Any]) -> None:
    actions = payload.get("actions") or []
    action_names = {str(a.get("action", "")) for a in actions if isinstance(a, dict)}
    if "create_world_event" in action_names:
        return

    first = triggered[0] if triggered else {"type": "unknown", "default_duration": 0}
    auto_params = {
        "event_type": str(first.get("type", "unknown")),
        "narrative": str(payload.get("narrative") or f"[auto] {first.get('type', 'unknown')} in {year}"),
        "duration_years": int(first.get("default_duration", 0) or 0),
    }
    rep = ActionReport()
    master._create_world_event(auto_params, year=year, rep=rep)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_and_apply_events(
    world,
    year: int,
    year_bucket: List[Dict[str, Any]],
    llm_client=None,
    rng: Optional[random.Random] = None,
    model: Optional[str] = None,
    log_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Roll big-history events, optionally call the LLM, and apply actions.

    This function is safe to call every year. It first restores expired
    duration-based effects through LLMMasterClass.apply_expiry(), then rolls the
    event dice, then either calls the LLM or uses a deterministic fallback.
    """
    year = int(year)
    master = LLMMasterClass(world)
    master.apply_expiry(current_year=year)

    if rng is None:
        rng = _year_rng(world, year)

    triggered = roll_events(year, rng)
    if not triggered:
        return {"year": year, "events": [], "applied": 0, "skipped": 0}

    event_types = [str(e.get("type", "unknown")) for e in triggered]
    print(f"  [big_history_events] Year {year}: triggered -> {event_types}")

    llm = _resolve_llm()
    if model is None:
        model = str(MODEL_TIERS.get("big_history_events") or MODEL_TIERS.get("temporal_evolution") or None)

    # Sample a specific target action count — scaled to population, randomized
    n_persons = len(world.persons) if getattr(world, "persons", None) is not None else 5000
    action_lo = max(12, n_persons // 1500 + 8)
    action_hi = max(action_lo + 5, n_persons // 400 + 15)
    target_n_actions = rng.randint(action_lo, action_hi)

    prompt = build_big_event_prompt(world, year, triggered, year_bucket, target_n_actions=target_n_actions)

    # Fallback path if no llm.
    if llm is None:
        if current_mode() == "research":
            audit_fallback_hit(
                "big_history_events",
                "llm_unavailable",
                detail="deterministic big-history fallback is disabled in research mode",
                mode="research",
            )
        result = _run_fallback(world, year, triggered)
        if log_dir:
            stem = Path(log_dir) / f"big_event_{year}_fallback.json"
            _write_log_file(stem, {"prompt": prompt, "result": result}, is_json=True)
        return result

    raw_text = ""
    parsed: Any = None
    error_text: Optional[str] = None
    try:
        raw_text, parsed = _call_llm(llm, str(model), prompt)
    except Exception as exc:
        error_text = str(exc)
        print(f"  [big_history_events] LLM failed: {exc}")

    if not isinstance(parsed, dict):
        if current_mode() == "research":
            audit_fallback_hit(
                "big_history_events",
                "invalid_llm_response",
                detail="deterministic big-history fallback after invalid LLM response is disabled in research mode",
                mode="research",
            )
        result = _run_fallback(world, year, triggered)
        if error_text:
            result["error"] = error_text
        if log_dir:
            base = Path(log_dir)
            _write_log_file(base / f"big_event_{year}_prompt.txt", prompt, is_json=False)
            _write_log_file(base / f"big_event_{year}_raw.txt", raw_text or error_text or "", is_json=False)
            _write_log_file(base / f"big_event_{year}_fallback.json", result, is_json=True)
        return result

    rep = master.execute(parsed, year=year)
    _ensure_world_event(master, year, triggered, parsed)

    result = {
        "year": year,
        "events": event_types,
        "applied": int(rep.applied),
        "skipped": int(rep.skipped),
        "skipped_reasons": list(rep.skipped_reasons[:10]),
    }

    if log_dir:
        base = Path(log_dir)
        ts = time.strftime("%Y%m%d_%H%M%S")
        _write_log_file(base / f"big_event_{year}_{ts}_prompt.txt", prompt, is_json=False)
        _write_log_file(base / f"big_event_{year}_{ts}_raw.txt", raw_text, is_json=False)
        _write_log_file(base / f"big_event_{year}_{ts}_report.json", result, is_json=True)

    print(
        f"  [big_history_events] Year {year} complete: applied={result['applied']}, "
        f"skipped={result['skipped']}"
    )
    return result
