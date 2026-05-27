#!/usr/bin/env python3
"""Keyword generator for Mirage.

Research mode consumes `keyword_seed_bank.json`.
Debug mode keeps the older fixed-pool fallback.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, deque
from pathlib import Path

BASE_DIR = Path(__file__).parent
ENTITY_DIR = BASE_DIR / "entities"

sys.path.insert(0, str(BASE_DIR))

from bootstrap_artifacts import (
    audit_artifact_usage,
    audit_fallback_hit,
    current_mode,
    load_keyword_seed_bank,
    load_modeling_priors_artifact,
    prior_section,
    require_payload_value,
)
from contracts import GENRES, GENRE_WEIGHTS
from policy_runtime import keyword_seed_bank_path, modeling_priors_path


DEBUG_GENERIC_KEYWORDS = [
    "revenge", "survival", "justice", "deception", "identity", "transformation", "fate", "legacy",
    "honor", "freedom", "power", "greed", "trust", "loyalty", "obsession", "regret", "hope", "truth",
]
DEBUG_UNIVERSAL_QUALIFIERS = [
    "dark", "epic", "secret", "lost", "final", "ancient", "modern", "urban", "neon", "fractured",
]
DEBUG_UNIVERSAL_CONTEXTS = [
    "conflict", "journey", "mystery", "pursuit", "alliance", "rivalry", "crisis", "mission", "reckoning",
]

DEFAULT_SELECTION_BUCKET_TARGETS = {
    "exact_anchor": 0.40,
    "related_support": 0.22,
    "story_specific": 0.30,
    "generic": 0.08,
}


def _debug_bank() -> dict:
    return {
        "universal_qualifiers": DEBUG_UNIVERSAL_QUALIFIERS,
        "universal_contexts": DEBUG_UNIVERSAL_CONTEXTS,
        "generic_themes": DEBUG_GENERIC_KEYWORDS,
        "genres": [
            {
                "genre": genre,
                "seeds": [f"{genre.lower().replace(' ', '-')}-seed-{i}" for i in range(1, 25)],
                "qualifiers": [f"{genre.lower().replace(' ', '-')}-tone-{i}" for i in range(1, 9)],
                "contexts": [f"{genre.lower().replace(' ', '-')}-context-{i}" for i in range(1, 9)],
                "tone_tokens": [f"{genre.lower().replace(' ', '-')}-mood-{i}" for i in range(1, 5)],
                "exclusion_hints": [],
            }
            for genre in GENRES
        ],
    }


def _clean_text_list(raw: object) -> list[str]:
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for value in raw:
        item = str(value or "").strip()
        key = item.casefold()
        if not item or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _normalise_genre_weight_map(raw: dict[str, float] | None = None) -> dict[str, float]:
    merged = {str(genre): max(0.001, float(GENRE_WEIGHTS.get(genre, 0.001) or 0.001)) for genre in GENRES}
    if isinstance(raw, dict):
        canon = {str(genre).strip().lower(): str(genre) for genre in GENRES}
        for key, value in raw.items():
            try:
                genre = canon.get(str(key).strip().lower())
                if genre is None:
                    continue
                merged[genre] = max(0.001, float(value))
            except Exception:
                continue
    total = float(sum(merged.values())) or 1.0
    return {genre: float(weight) / total for genre, weight in merged.items()}


def _genre_target_weights(base_dir: Path, mode: str) -> dict[str, float]:
    priors_path = modeling_priors_path(base_dir)
    priors = load_modeling_priors_artifact(base_dir, mode=mode) or {}
    keyword_priors = prior_section(priors, "keyword_generation")
    raw_keyword_weights = keyword_priors.get("genre_target_weights")
    if isinstance(raw_keyword_weights, dict) and raw_keyword_weights:
        audit_artifact_usage("modeling_priors.json", priors_path, sections=["keyword_generation.genre_target_weights"])
        return _normalise_genre_weight_map(raw_keyword_weights)
    title_priors = prior_section(priors, "title_generation")
    for key in ("genre_base_weights", "genre_weights", "genre_prevalence"):
        raw = title_priors.get(key)
        if isinstance(raw, dict) and raw:
            audit_artifact_usage("modeling_priors.json", priors_path, sections=[f"title_generation.{key}"])
            return _normalise_genre_weight_map(raw)
    if mode == "research":
        audit_fallback_hit(
            "modeling_priors.json",
            "keyword_generation.genre_target_weights_missing",
            detail="research-mode keyword generation requires keyword_generation.genre_target_weights",
            mode=mode,
        )
    return _normalise_genre_weight_map(dict(GENRE_WEIGHTS))


def _keyword_generation_config(base_dir: Path, mode: str) -> dict[str, float]:
    priors_path = modeling_priors_path(base_dir)
    priors = load_modeling_priors_artifact(base_dir, mode=mode) or {}
    section = prior_section(priors, "keyword_generation")

    def _float_value(key: str, default: float, *, lo: float | None = None, hi: float | None = None) -> float:
        raw = section.get(key)
        if raw is None and mode == "research":
            audit_fallback_hit(
                "modeling_priors.json",
                f"keyword_generation.{key}_missing",
                detail=f"research-mode keyword generation requires keyword_generation.{key}",
                mode=mode,
            )
        try:
            value = float(default if raw is None else raw)
        except Exception:
            value = float(default)
        if lo is not None:
            value = max(float(lo), value)
        if hi is not None:
            value = min(float(hi), value)
        return float(value)

    raw_bucket_targets = section.get("selection_bucket_targets")
    bucket_targets = dict(DEFAULT_SELECTION_BUCKET_TARGETS)
    if isinstance(raw_bucket_targets, dict):
        for key in ("exact_anchor", "related_support", "story_specific", "generic"):
            try:
                if key in raw_bucket_targets:
                    bucket_targets[key] = max(0.0, float(raw_bucket_targets[key]))
            except Exception:
                continue
    elif mode == "research":
        audit_fallback_hit(
            "modeling_priors.json",
            "keyword_generation.selection_bucket_targets_missing",
            detail="research-mode keyword generation requires keyword_generation.selection_bucket_targets",
            mode=mode,
        )
    total_bucket = float(sum(bucket_targets.values())) or 1.0
    bucket_targets = {key: float(value) / total_bucket for key, value in bucket_targets.items()}

    audit_artifact_usage(
        "modeling_priors.json",
        priors_path,
        sections=[
            "keyword_generation.genre_target_weights",
            "keyword_generation.generic_budget_ratio",
            "keyword_generation.min_specific_story_share",
            "keyword_generation.selection_bucket_targets",
        ],
    )
    return {
        "generic_budget_ratio": _float_value("generic_budget_ratio", 0.06, lo=0.0, hi=0.18),
        "min_specific_story_share": _float_value("min_specific_story_share", 0.72, lo=0.45, hi=0.95),
        "selection_bucket_targets": bucket_targets,
    }


def _keyword_runtime_requirements(base_dir: Path, mode: str) -> dict[str, int]:
    priors_path = modeling_priors_path(base_dir)
    priors = load_modeling_priors_artifact(base_dir, mode=mode) or {}
    selection = prior_section(priors, "selection_weights")
    keyword_selection = selection.get("keyword_selection") if isinstance(selection.get("keyword_selection"), dict) else {}
    exact_map = keyword_selection.get("exact_topic_min_count_by_tier")
    primary_related_map = keyword_selection.get("primary_plus_related_min_count_by_tier")

    def _max_required(raw: object, field_name: str, default: int) -> int:
        if not isinstance(raw, dict):
            if mode == "research":
                audit_fallback_hit(
                    "modeling_priors.json",
                    f"selection_weights.keyword_selection.{field_name}_missing",
                    detail=f"research-mode keyword generation requires selection_weights.keyword_selection.{field_name}",
                    mode=mode,
                )
            return int(default)
        values: list[int] = []
        for value in raw.values():
            try:
                values.append(max(0, int(round(float(value)))))
            except Exception:
                continue
        return max(values) if values else int(default)

    audit_artifact_usage(
        "modeling_priors.json",
        priors_path,
        sections=[
            "selection_weights.keyword_selection.exact_topic_min_count_by_tier",
            "selection_weights.keyword_selection.primary_plus_related_min_count_by_tier",
        ],
    )
    exact_floor = max(1, _max_required(exact_map, "exact_topic_min_count_by_tier", 1))
    primary_related_floor = max(exact_floor, _max_required(primary_related_map, "primary_plus_related_min_count_by_tier", exact_floor))
    return {
        "exact_topic_floor": int(exact_floor),
        "primary_plus_related_floor": int(primary_related_floor),
    }


def _allocate_weighted_genre_targets(target: int, weights: dict[str, float], *, floor_per_genre: int = 0) -> dict[str, int]:
    targets = {str(genre): 0 for genre in GENRES}
    if int(target) <= 0:
        return targets

    floor_slots = min(int(target), max(0, int(floor_per_genre)) * len(GENRES))
    floor_index = 0
    while floor_slots > 0:
        genre = str(GENRES[floor_index % len(GENRES)])
        targets[genre] += 1
        floor_slots -= 1
        floor_index += 1

    remaining = max(0, int(target) - sum(targets.values()))
    if remaining <= 0:
        return targets

    raw_extra = {
        str(genre): float(weights.get(str(genre), 0.0) or 0.0) * float(remaining)
        for genre in GENRES
    }
    extra_floor = {genre: int(raw_extra[genre]) for genre in GENRES}
    for genre, value in extra_floor.items():
        targets[str(genre)] += int(value)
    leftover = max(0, int(target) - sum(targets.values()))
    ranked = sorted(
        GENRES,
        key=lambda genre: (raw_extra[str(genre)] - extra_floor[str(genre)], float(weights.get(str(genre), 0.0))),
        reverse=True,
    )
    for genre in ranked[:leftover]:
        targets[str(genre)] += 1
    return targets


def _allocate_related_targets_with_floor(
    total: int,
    weights: dict[str, float],
    *,
    exact_targets: dict[str, int],
    primary_plus_related_floor: int,
) -> dict[str, int]:
    targets = {str(genre): 0 for genre in GENRES}
    total = max(0, int(total))
    base_required = {
        str(genre): max(0, int(primary_plus_related_floor) - int(exact_targets.get(str(genre), 0) or 0))
        for genre in GENRES
    }
    required_total = int(sum(base_required.values()))
    if required_total > total:
        # Caller decides whether to fail or expand target; keep the floor evidence visible.
        for genre, count in base_required.items():
            targets[str(genre)] = int(count)
        return targets
    for genre, count in base_required.items():
        targets[str(genre)] = int(count)
    remaining = max(0, total - required_total)
    if remaining <= 0:
        return targets
    raw_extra = {
        str(genre): float(weights.get(str(genre), 0.0) or 0.0) * float(remaining)
        for genre in GENRES
    }
    extra_floor = {genre: int(raw_extra[genre]) for genre in GENRES}
    for genre, value in extra_floor.items():
        targets[str(genre)] += int(value)
    leftover = max(0, total - sum(targets.values()))
    ranked = sorted(
        GENRES,
        key=lambda genre: (raw_extra[str(genre)] - extra_floor[str(genre)], float(weights.get(str(genre), 0.0) or 0.0)),
        reverse=True,
    )
    for genre in ranked[:leftover]:
        targets[str(genre)] += 1
    return targets


def _bucket_counts(target: int, bucket_targets: dict[str, float], *, generic_budget_ratio: float) -> dict[str, int]:
    total = max(0, int(target))
    generic_cap = max(0, min(total, int(round(total * float(generic_budget_ratio)))))
    generic_target = min(generic_cap, int(round(total * float(bucket_targets.get("generic", 0.0) or 0.0))))
    non_generic = max(0, total - generic_target)
    non_generic_weights = {
        "exact_anchor": max(0.0, float(bucket_targets.get("exact_anchor", 0.0) or 0.0)),
        "related_support": max(0.0, float(bucket_targets.get("related_support", 0.0) or 0.0)),
        "story_specific": max(0.0, float(bucket_targets.get("story_specific", 0.0) or 0.0)),
    }
    total_non_generic_weight = float(sum(non_generic_weights.values())) or 1.0
    raw = {
        key: (float(value) / total_non_generic_weight) * float(non_generic)
        for key, value in non_generic_weights.items()
    }
    counts = {key: int(raw[key]) for key in raw}
    allocated = int(sum(counts.values()))
    leftovers = sorted(
        raw.keys(),
        key=lambda key: (raw[key] - counts[key], raw[key]),
        reverse=True,
    )
    for key in leftovers[: max(0, non_generic - allocated)]:
        counts[key] += 1
    counts["generic"] = int(generic_target)
    return counts


def _dedupe_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip()
        key = item.casefold()
        if not item or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _bucket_candidates(
    row: dict[str, object],
    universal_qualifiers: list[str],
    universal_contexts: list[str],
    bucket: str,
) -> list[str]:
    seeds = _clean_text_list(row.get("seeds"))
    qualifiers = _clean_text_list(row.get("qualifiers")) + list(universal_qualifiers)
    contexts = _clean_text_list(row.get("contexts")) + list(universal_contexts)
    tone_tokens = _clean_text_list(row.get("tone_tokens"))
    if bucket == "exact_anchor":
        return _dedupe_keep_order(seeds + [f"{tone}-{seed}" for tone in tone_tokens[:4] for seed in seeds[:6]])
    if bucket == "related_support":
        return _dedupe_keep_order(
            [f"{qualifier}-{seed}" for qualifier in qualifiers[:10] for seed in seeds[:10]]
            + [f"{seed}-{context}" for seed in seeds[:8] for context in contexts[:8]]
            + [f"{tone}-{seed}" for tone in tone_tokens[:6] for seed in seeds[:8]]
        )
    if bucket == "story_specific":
        return _dedupe_keep_order(
            [f"{qualifier}-{context}-{seed}" for qualifier in qualifiers[:8] for context in contexts[:8] for seed in seeds[:8]]
            + [f"{seed}-{context}-{qualifier}" for seed in seeds[:6] for context in contexts[:6] for qualifier in qualifiers[:6]]
            + [f"{tone}-{context}-{seed}" for tone in tone_tokens[:6] for context in contexts[:6] for seed in seeds[:6]]
            + [f"{qualifier}-{tone}-{seed}" for qualifier in qualifiers[:6] for tone in tone_tokens[:5] for seed in seeds[:6]]
        )
    return []


def _keyword_metadata(topic_genre: str, selection_bucket: str) -> dict[str, object]:
    bucket = str(selection_bucket or "").strip() or "story_specific"
    topic = str(topic_genre or "").strip()
    if bucket == "exact_anchor":
        return {
            "motif_family": "genre",
            "specificity_tier": 1,
            "scope_hint": "global",
            "franchise_affinity": 0.02,
            "cooccurrence_cluster": f"{topic.lower().replace(' ', '_')}::genre" if topic else "generic::genre",
            "recurrence_strength": 0.72,
        }
    if bucket == "related_support":
        return {
            "motif_family": "subgenre",
            "specificity_tier": 2,
            "scope_hint": "global",
            "franchise_affinity": 0.05,
            "cooccurrence_cluster": f"{topic.lower().replace(' ', '_')}::subgenre" if topic else "generic::subgenre",
            "recurrence_strength": 0.55,
        }
    if bucket == "generic":
        return {
            "motif_family": "tone",
            "specificity_tier": 1,
            "scope_hint": "global",
            "franchise_affinity": 0.03,
            "cooccurrence_cluster": "generic::tone",
            "recurrence_strength": 0.46,
        }
    return {
        "motif_family": "setting",
        "specificity_tier": 3,
        "scope_hint": "concept_pack",
        "franchise_affinity": 0.12,
        "cooccurrence_cluster": f"{topic.lower().replace(' ', '_')}::setting" if topic else "generic::setting",
        "recurrence_strength": 0.34,
    }


def _bank(base_dir: Path, mode: str) -> dict:
    artifact_path = keyword_seed_bank_path(base_dir)
    payload = load_keyword_seed_bank(base_dir, mode=mode)
    if not isinstance(payload, dict):
        audit_fallback_hit("keyword_generation", "debug_keyword_seed_bank_missing", detail="using built-in keyword seed bank", mode=mode)
        return _debug_bank()

    universal_qualifiers = _clean_text_list(
        require_payload_value(
            payload,
            "universal_qualifiers",
            artifact_label="keyword_seed_bank.json",
            artifact_path=artifact_path,
            mode=mode,
            validator=lambda value: isinstance(value, list) and len(value) > 0,
        )
        or []
    )
    universal_contexts = _clean_text_list(
        require_payload_value(
            payload,
            "universal_contexts",
            artifact_label="keyword_seed_bank.json",
            artifact_path=artifact_path,
            mode=mode,
            validator=lambda value: isinstance(value, list) and len(value) > 0,
        )
        or []
    )
    generic_themes = _clean_text_list(
        require_payload_value(
            payload,
            "generic_themes",
            artifact_label="keyword_seed_bank.json",
            artifact_path=artifact_path,
            mode=mode,
            validator=lambda value: isinstance(value, list) and len(value) > 0,
        )
        or []
    )
    raw_genres = payload.get("genres")
    if not isinstance(raw_genres, list) or not raw_genres:
        raw_genres = payload.get("genre_data")
    if not isinstance(raw_genres, list) or not raw_genres:
        audit_fallback_hit(
            "keyword_seed_bank.json",
            "genre_rows_missing",
            detail="keyword seed bank must provide genres or genre_data rows",
            mode=mode,
        )
        raw_genres = []
    genre_rows: list[dict] = []
    seen_genres: set[str] = set()
    for idx, raw in enumerate(raw_genres, start=1):
        if not isinstance(raw, dict):
            if mode == "research":
                audit_fallback_hit("keyword_seed_bank.json", "invalid_genre_row", detail=f"genre row {idx} must be an object", mode=mode)
            continue
        genre = str(raw.get("genre", "") or "").strip()
        seeds = _clean_text_list(raw.get("seeds"))
        qualifiers = _clean_text_list(raw.get("qualifiers"))
        contexts = _clean_text_list(raw.get("contexts"))
        tone_tokens = _clean_text_list(raw.get("tone_tokens"))
        if not genre or not seeds:
            if mode == "research":
                audit_fallback_hit(
                    "keyword_seed_bank.json",
                    "genre_seed_inventory_missing",
                    detail=f"{genre or f'genre row {idx}'} missing seeds",
                    mode=mode,
                )
            continue
        if not qualifiers and mode == "research":
            audit_fallback_hit("keyword_seed_bank.json", "genre_qualifiers_missing", detail=f"{genre} missing qualifiers", mode=mode)
        if not contexts and mode == "research":
            audit_fallback_hit("keyword_seed_bank.json", "genre_contexts_missing", detail=f"{genre} missing contexts", mode=mode)
        seen_genres.add(genre)
        genre_rows.append(
            {
                "genre": genre,
                "seeds": seeds,
                "qualifiers": qualifiers,
                "contexts": contexts,
                "tone_tokens": tone_tokens,
                "exclusion_hints": _clean_text_list(raw.get("exclusion_hints")),
            }
        )
    if mode == "research":
        missing_genres = [str(genre) for genre in GENRES if str(genre) not in seen_genres]
        if missing_genres:
            audit_fallback_hit(
                "keyword_seed_bank.json",
                "genre_coverage_missing",
                detail=f"keyword seed bank missing benchmark genres: {', '.join(missing_genres)}",
                mode=mode,
            )
    if not genre_rows:
        audit_fallback_hit("keyword_generation", "debug_keyword_seed_rows_empty", detail="using built-in keyword genre rows", mode=mode)
        return _debug_bank()
    audit_artifact_usage(
        "keyword_seed_bank.json",
        artifact_path,
        sections=["universal_qualifiers", "universal_contexts", "generic_themes", "genre_data"],
    )
    return {
        "universal_qualifiers": universal_qualifiers,
        "universal_contexts": universal_contexts,
        "generic_themes": generic_themes,
        "genres": genre_rows,
    }


def _append_keyword(
    keywords: list[dict],
    used: set[str],
    keyword: str,
    topic_genre: str,
    rng: random.Random,
    *,
    low: float = 0.2,
    high: float = 0.8,
    origin: str = "specific_story",
    selection_bucket: str = "story_specific",
) -> bool:
    key = str(keyword).strip()
    if not key or key in used:
        return False
    used.add(key)
    metadata = _keyword_metadata(str(topic_genre or ""), str(selection_bucket or "story_specific"))
    keywords.append(
        {
            "keyword_id": len(keywords) + 1,
            "keyword": key,
            "topic_genre": topic_genre,
            "pop_weight": round(rng.uniform(low, high), 3),
            "selection_bucket": str(selection_bucket or "story_specific"),
            "motif_family": str(metadata["motif_family"]),
            "specificity_tier": int(metadata["specificity_tier"]),
            "scope_hint": str(metadata["scope_hint"]),
            "franchise_affinity": float(metadata["franchise_affinity"]),
            "cooccurrence_cluster": str(metadata["cooccurrence_cluster"]),
            "recurrence_strength": float(metadata["recurrence_strength"]),
            "_origin": str(origin or "specific_story"),
        }
    )
    return True


def _genre_expansion_candidates(row: dict, universal_qualifiers: list[str], universal_contexts: list[str], rng: random.Random) -> deque[str]:
    seeds = [str(x).strip() for x in row.get("seeds", []) if str(x).strip()]
    qualifiers = [str(x).strip() for x in row.get("qualifiers", []) if str(x).strip()] + list(universal_qualifiers)
    contexts = [str(x).strip() for x in row.get("contexts", []) if str(x).strip()] + list(universal_contexts)
    tone_tokens = [str(x).strip() for x in row.get("tone_tokens", []) if str(x).strip()]

    rng.shuffle(seeds)
    rng.shuffle(qualifiers)
    rng.shuffle(contexts)
    rng.shuffle(tone_tokens)

    qualifier_cap = min(len(qualifiers), 32)
    context_cap = min(len(contexts), 28)
    tone_cap = min(len(tone_tokens), 20)
    pair_qualifier_cap = min(len(qualifiers), 18)
    pair_context_cap = min(len(contexts), 14)
    pair_tone_cap = min(len(tone_tokens), 16)

    candidates: list[str] = []
    for seed in seeds:
        for qualifier in qualifiers[:qualifier_cap]:
            candidates.append(f"{qualifier}-{seed}")
        for context in contexts[:context_cap]:
            candidates.append(f"{seed}-{context}")
        for tone in tone_tokens[:tone_cap]:
            candidates.append(f"{tone}-{seed}")
        for qualifier in qualifiers[:pair_qualifier_cap]:
            for context in contexts[:pair_context_cap]:
                candidates.append(f"{qualifier}-{context}-{seed}")
        for tone in tone_tokens[:pair_tone_cap]:
            for context in contexts[:pair_context_cap]:
                candidates.append(f"{tone}-{context}-{seed}")
        for qualifier in qualifiers[:pair_qualifier_cap]:
            for tone in tone_tokens[:min(pair_tone_cap, 12)]:
                candidates.append(f"{qualifier}-{tone}-{seed}")
    rng.shuffle(candidates)
    return deque(candidates)


def generate_keywords(target: int, *, seed: int, base_dir: Path, mode: str) -> list[dict]:
    rng = random.Random(seed)
    bank = _bank(base_dir, mode)
    keyword_cfg = _keyword_generation_config(base_dir, mode)
    keywords: list[dict] = []
    used: set[str] = set()

    genre_rows = {
        str(row.get("genre")): row
        for row in bank.get("genres", [])
        if isinstance(row, dict) and str(row.get("genre", "")).strip()
    }
    universal_qualifiers = [str(x).strip() for x in bank.get("universal_qualifiers", []) if str(x).strip()]
    universal_contexts = [str(x).strip() for x in bank.get("universal_contexts", []) if str(x).strip()]
    generic_themes = [str(x).strip() for x in bank.get("generic_themes", []) if str(x).strip()]
    genre_weights = _genre_target_weights(base_dir, mode)
    bucket_targets = dict(keyword_cfg["selection_bucket_targets"])
    runtime_requirements = _keyword_runtime_requirements(base_dir, mode)
    exact_topic_floor = int(runtime_requirements["exact_topic_floor"])
    primary_plus_related_floor = int(runtime_requirements["primary_plus_related_floor"])
    bucket_counts = _bucket_counts(
        int(target),
        bucket_targets,
        generic_budget_ratio=float(keyword_cfg["generic_budget_ratio"]),
    )
    min_exact_total = int(exact_topic_floor * len(GENRES))
    min_primary_related_total = int(primary_plus_related_floor * len(GENRES))

    bucket_counts["exact_anchor"] = max(int(bucket_counts.get("exact_anchor", 0) or 0), min_exact_total)
    bucket_counts["related_support"] = max(
        int(bucket_counts.get("related_support", 0) or 0),
        max(0, min_primary_related_total - int(bucket_counts["exact_anchor"])),
    )
    minimum_total_required = int(bucket_counts["exact_anchor"] + bucket_counts["related_support"] + int(bucket_counts.get("story_specific", 0) or 0) + int(bucket_counts.get("generic", 0) or 0))
    if minimum_total_required > int(target):
        if mode == "research":
            target = int(minimum_total_required)
        else:
            bucket_counts["story_specific"] = max(0, int(bucket_counts.get("story_specific", 0) or 0) - (minimum_total_required - int(target)))

    exact_targets = _allocate_weighted_genre_targets(int(bucket_counts.get("exact_anchor", 0) or 0), genre_weights, floor_per_genre=exact_topic_floor)
    related_targets = _allocate_related_targets_with_floor(
        int(bucket_counts.get("related_support", 0) or 0),
        genre_weights,
        exact_targets=exact_targets,
        primary_plus_related_floor=primary_plus_related_floor,
    )
    story_targets = _allocate_weighted_genre_targets(int(bucket_counts.get("story_specific", 0) or 0), genre_weights, floor_per_genre=0)

    if mode == "research":
        exact_shortfall = [str(genre) for genre in GENRES if int(exact_targets.get(str(genre), 0) or 0) < exact_topic_floor]
        if exact_shortfall:
            audit_fallback_hit(
                "keyword_generation",
                "exact_anchor_floor_unsatisfied",
                detail=f"planned exact-anchor allocation below floor {exact_topic_floor} for genres: {', '.join(exact_shortfall)}",
                mode=mode,
            )
        primary_related_shortfall = [
            str(genre)
            for genre in GENRES
            if int(exact_targets.get(str(genre), 0) or 0) + int(related_targets.get(str(genre), 0) or 0) < primary_plus_related_floor
        ]
        if primary_related_shortfall:
            audit_fallback_hit(
                "keyword_generation",
                "primary_related_floor_unsatisfied",
                detail=f"planned exact+related allocation below floor {primary_plus_related_floor} for genres: {', '.join(primary_related_shortfall)}",
                mode=mode,
            )

    current_bucket_genre_counts: Counter[tuple[str, str]] = Counter()
    current_bucket_counts: Counter[str] = Counter()
    for bucket_name, per_genre_targets in (
        ("exact_anchor", exact_targets),
        ("related_support", related_targets),
        ("story_specific", story_targets),
    ):
        for genre in GENRES:
            row = genre_rows.get(str(genre), {})
            candidates = _bucket_candidates(row, universal_qualifiers, universal_contexts, bucket_name)
            rng.shuffle(candidates)
            target_count = int(per_genre_targets.get(str(genre), 0) or 0)
            if target_count <= 0:
                continue
            for kw in candidates:
                if len(keywords) >= target or current_bucket_counts[bucket_name] >= int(bucket_counts.get(bucket_name, 0) or 0):
                    break
                if current_bucket_genre_counts[(bucket_name, str(genre))] >= target_count:
                    break
                if _append_keyword(
                    keywords,
                    used,
                    kw,
                    str(genre),
                    rng,
                    low=0.28 if bucket_name == "exact_anchor" else 0.22,
                    high=1.0 if bucket_name == "exact_anchor" else 0.86,
                    origin=f"bucket_{bucket_name}",
                    selection_bucket=bucket_name,
                ):
                    current_bucket_genre_counts[(bucket_name, str(genre))] += 1
                    current_bucket_counts[bucket_name] += 1

    # If the first pass leaves any bucket under target, expand the candidate pool for the
    # remaining unmet genre/bucket combinations instead of silently relying on debug overflow.
    for bucket_name, per_genre_targets in (
        ("exact_anchor", exact_targets),
        ("related_support", related_targets),
        ("story_specific", story_targets),
    ):
        bucket_limit = int(bucket_counts.get(bucket_name, 0) or 0)
        if bucket_limit <= 0:
            continue
        for genre in GENRES:
            if len(keywords) >= target or current_bucket_counts[bucket_name] >= bucket_limit:
                break
            target_count = int(per_genre_targets.get(str(genre), 0) or 0)
            current_count = int(current_bucket_genre_counts[(bucket_name, str(genre))] or 0)
            if target_count <= current_count:
                continue
            row = genre_rows.get(str(genre), {})
            if not row:
                continue
            expansion = _genre_expansion_candidates(row, universal_qualifiers, universal_contexts, rng)
            while expansion and len(keywords) < target and current_bucket_counts[bucket_name] < bucket_limit:
                if current_bucket_genre_counts[(bucket_name, str(genre))] >= target_count:
                    break
                kw = expansion.popleft()
                if _append_keyword(
                    keywords,
                    used,
                    kw,
                    str(genre),
                    rng,
                    low=0.28 if bucket_name == "exact_anchor" else 0.22,
                    high=1.0 if bucket_name == "exact_anchor" else 0.86,
                    origin=f"expanded_{bucket_name}",
                    selection_bucket=bucket_name,
                ):
                    current_bucket_genre_counts[(bucket_name, str(genre))] += 1
                    current_bucket_counts[bucket_name] += 1

    generic_budget = int(bucket_counts.get("generic", 0) or 0)
    if generic_budget > 0:
        generic_pool = _dedupe_keep_order(
            list(generic_themes)
            + [f"{qualifier}-{theme}" for qualifier in universal_qualifiers[:16] for theme in generic_themes[:20]]
            + [f"{theme}-{context}" for theme in generic_themes[:20] for context in universal_contexts[:12]]
            + [
                f"{qualifier}-{theme}-{context}"
                for qualifier in universal_qualifiers[:10]
                for theme in generic_themes[:12]
                for context in universal_contexts[:8]
            ]
        )
        rng.shuffle(generic_pool)
        for kw in generic_pool:
            if len(keywords) >= target or current_bucket_counts["generic"] >= generic_budget:
                break
            if _append_keyword(
                keywords,
                used,
                kw,
                "",
                rng,
                low=0.12,
                high=0.48,
                origin="generic_theme",
                selection_bucket="generic",
            ):
                current_bucket_counts["generic"] += 1

    overflow = 1
    while len(keywords) < target and mode != "research":
        genre = GENRES[(overflow - 1) % len(GENRES)]
        qualifier = universal_qualifiers[(overflow - 1) % len(universal_qualifiers)] if universal_qualifiers else "synthetic"
        context = universal_contexts[(overflow * 3 - 1) % len(universal_contexts)] if universal_contexts else "signal"
        kw = f"{qualifier}-{genre.lower().replace(' ', '-')}-{context}-{overflow:04d}"
        _append_keyword(keywords, used, kw, genre, rng, origin="debug_overflow", selection_bucket="story_specific")
        overflow += 1

    if len(keywords) < target and mode == "research":
        audit_fallback_hit(
            "keyword_generation",
            "keyword_seed_capacity_exhausted",
            detail=f"keyword bank produced only {len(keywords)} / {target} unique keywords without overflow fallback",
            mode=mode,
        )
    if mode == "research" and int(target) >= len(GENRES):
        covered = {
            str(row.get("topic_genre", "") or "").strip()
            for row in keywords
            if str(row.get("selection_bucket", "") or "").strip() == "exact_anchor"
        }
        missing = [str(genre) for genre in GENRES if str(genre) not in covered]
        if missing:
            audit_fallback_hit(
                "keyword_generation",
                "keyword_exact_anchor_coverage_missing",
                detail=f"generated keyword bank missing exact-anchor coverage for benchmark genres: {', '.join(missing)}",
                mode=mode,
            )
    if mode == "research":
        exact_per_genre = Counter(
            str(row.get("topic_genre", "") or "").strip()
            for row in keywords
            if str(row.get("selection_bucket", "") or "").strip() == "exact_anchor"
            and str(row.get("topic_genre", "") or "").strip()
        )
        related_per_genre = Counter(
            str(row.get("topic_genre", "") or "").strip()
            for row in keywords
            if str(row.get("selection_bucket", "") or "").strip() == "related_support"
            and str(row.get("topic_genre", "") or "").strip()
        )
        empty_genres = [str(genre) for genre in GENRES if int(exact_per_genre.get(str(genre), 0)) <= 0]
        if empty_genres:
            audit_fallback_hit(
                "keyword_generation",
                "exact_anchor_quota_missing",
                detail=f"generated keyword bank left benchmark genres without exact anchors: {', '.join(empty_genres)}",
                mode=mode,
            )
        exact_floor_shortfall = [
            str(genre)
            for genre in GENRES
            if int(exact_per_genre.get(str(genre), 0)) < exact_topic_floor
        ]
        if exact_floor_shortfall:
            audit_fallback_hit(
                "keyword_generation",
                "exact_anchor_floor_realized_shortfall",
                detail=f"generated keyword bank exact-anchor floor {exact_topic_floor} not met for genres: {', '.join(exact_floor_shortfall)}",
                mode=mode,
            )
        primary_related_floor_shortfall = [
            str(genre)
            for genre in GENRES
            if int(exact_per_genre.get(str(genre), 0)) + int(related_per_genre.get(str(genre), 0)) < primary_plus_related_floor
        ]
        if primary_related_floor_shortfall:
            audit_fallback_hit(
                "keyword_generation",
                "primary_related_floor_realized_shortfall",
                detail=(
                    f"generated keyword bank exact+related floor {primary_plus_related_floor} "
                    f"not met for genres: {', '.join(primary_related_floor_shortfall)}"
                ),
                mode=mode,
            )
        specific_count = sum(
            1
            for row in keywords
            if str(row.get("selection_bucket", "") or "").strip() in {"exact_anchor", "related_support", "story_specific"}
        )
        min_specific = int(round(len(keywords) * float(keyword_cfg["min_specific_story_share"])))
        if specific_count < min_specific:
            audit_fallback_hit(
                "keyword_generation",
                "specific_story_share_too_low",
                detail=f"generated keyword bank has only {specific_count} specific entries; need at least {min_specific}",
                mode=mode,
            )
        invalid_generic = [
            str(row.get("keyword", "") or "").strip()
            for row in keywords
            if str(row.get("selection_bucket", "") or "").strip() == "generic"
            and str(row.get("topic_genre", "") or "").strip()
        ]
        if invalid_generic:
            audit_fallback_hit(
                "keyword_generation",
                "generic_bucket_topic_genre_leak",
                detail=f"generic keyword rows must have blank topic_genre; examples: {', '.join(invalid_generic[:5])}",
                mode=mode,
            )

    for row in keywords:
        row.pop("_origin", None)

    return keywords


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword generator for Mirage.")
    parser.add_argument("--base-dir", default=str(BASE_DIR))
    parser.add_argument("--target", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("research", "debug"), default=current_mode())
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    out_path = Path(args.out).resolve() if args.out else (base_dir / "entities" / "keywords.json")

    print("=" * 60)
    print("  KEYWORD GENERATOR")
    print("=" * 60)
    print(f"  Target:  {args.target}")
    print(f"  Mode:    {args.mode}")
    print("=" * 60)

    keywords = generate_keywords(int(args.target), seed=int(args.seed), base_dir=base_dir, mode=str(args.mode))
    genres = Counter(k["topic_genre"] for k in keywords)
    print(f"  Generated {len(keywords)} keywords")
    print(f"  Genre distribution: {dict(genres.most_common(10))}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(keywords, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
