"""Top up title_bank.csv for Mirage.

Research mode uses `title_grammar_bank.json` and `temporal_regime_plan.json`.
Debug mode falls back to the older deterministic historical path.
"""
from __future__ import annotations

import argparse
import re
from string import Formatter
from pathlib import Path

import numpy as np
import pandas as pd

from bootstrap_artifacts import (
    audit_artifact_usage,
    audit_fallback_hit,
    current_mode,
    load_modeling_priors_artifact,
    load_temporal_regime_plan,
    load_title_grammar_bank,
    require_payload_value,
    prior_section,
    year_weight_map,
)
from contracts import DECADE_WEIGHTS, GENRES, GENRE_WEIGHTS, generate_compositional_title
from policy_runtime import modeling_priors_path, temporal_regime_plan_path, title_grammar_bank_path
from text_polish import (
    clean_display_text,
    contains_placeholder_syntax,
    looks_like_weak_tagline,
    looks_like_weak_title,
    sanitize_tagline,
    sanitize_title,
    tagline_is_near_duplicate,
    tagline_signature,
)


_PRESTIGE_GENRES = {
    "Biography",
    "Documentary",
    "Drama",
    "Film-Noir",
    "History",
    "Music",
    "Musical",
    "War",
}
_FRANCHISE_GENRES = {
    "Action",
    "Adventure",
    "Animation",
    "Disaster",
    "Family",
    "Fantasy",
    "Martial Arts",
    "Sci-Fi",
    "Superhero",
    "Thriller",
}
_EXPERIMENTAL_GENRES = {
    "Animation",
    "Documentary",
    "Experimental",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Mystery",
    "Sci-Fi",
}
_LOW_COST_GENRES = {
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Experimental",
    "Film-Noir",
    "History",
    "Horror",
    "Mystery",
    "Romance",
    "Short",
}

_TITLE_RENDER_VOCAB_FIELDS = {
    "adjective": "adjectives",
    "noun": "nouns",
    "abstract": "abstract_nouns",
    "location": "locations",
    "celestial": "celestial_words",
    "technology": "technology_words",
    "mythic": "mythic_words",
    "action": "action_words",
    "franchise_affix": "franchise_affixes",
}

_RENDER_ALIAS_MAP = {
    "adjectives": "adjective",
    "nouns": "noun",
    "abstract_noun": "abstract",
    "abstract_nouns": "abstract",
    "location_word": "location",
    "locations": "location",
    "celestial_word": "celestial",
    "celestial_words": "celestial",
    "technology_word": "technology",
    "technology_words": "technology",
    "mythic_word": "mythic",
    "mythic_words": "mythic",
    "action_word": "action",
    "action_words": "action",
    "plural_noun": "noun",
    "plural_nouns": "noun",
    "setting": "location",
    "franchise_suffix": "franchise_affix",
    "franchise_prefix": "franchise_affix",
    "franchise_affixes": "franchise_affix",
}

_MIN_TAGLINE_TEMPLATES_PER_GENRE = 12
_MAX_TITLE_BANK_TAGLINE_REUSE = 1
_MAX_TAGLINE_TEMPLATE_FAMILY_REUSE = 1
_TRANSITIVE_TAGLINE_ACTIONS = {
    "avenge",
    "betray",
    "conquer",
    "destroy",
    "fight",
    "hunt",
    "kill",
    "protect",
    "save",
    "strike",
}


def _canonical_genre_map() -> dict[str, str]:
    return {str(genre).strip().lower(): str(genre) for genre in GENRES}


def _normalise_genre_weight_map(weights: dict[str, float] | None) -> dict[str, float]:
    merged = {str(genre): max(0.001, float(GENRE_WEIGHTS.get(genre, 0.001) or 0.001)) for genre in GENRES}
    if isinstance(weights, dict):
        canon = _canonical_genre_map()
        for key, value in weights.items():
            try:
                genre = canon.get(str(key).strip().lower())
                if genre is None:
                    continue
                merged[genre] = max(0.001, float(value))
            except Exception:
                continue
    total = float(sum(merged.values())) or 1.0
    return {genre: float(weight) / total for genre, weight in merged.items()}


def _genre_list(raw: object, fallback: set[str]) -> set[str]:
    canon = _canonical_genre_map()
    out: set[str] = set()
    if isinstance(raw, (list, tuple, set)):
        for value in raw:
            genre = canon.get(str(value).strip().lower())
            if genre:
                out.add(genre)
    return out or set(fallback)


def _phase_for_year(temporal: dict | None, year: int) -> dict:
    if not isinstance(temporal, dict):
        return {}
    phases = temporal.get("phases")
    if not isinstance(phases, list):
        return {}
    for row in phases:
        if not isinstance(row, dict):
            continue
        try:
            start = int(row.get("start_year"))
            end = int(row.get("end_year"))
        except Exception:
            continue
        if start <= int(year) <= end:
            return row
    return {}


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


def _template_signature(text: object) -> str:
    value = clean_display_text(text).lower()
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _tagline_template_family_signature(text: object) -> str:
    value = _template_signature(text)
    if not value:
        return ""
    value = re.sub(r"\{[^}]+\}", "{slot}", value)
    value = re.sub(r"\b(?:the|a|an)\b", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _dedupe_template_list(raw: object) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in _clean_text_list(raw):
        key = _template_signature(value)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _template_fields(grammar: dict, template: str) -> tuple[str, ...]:
    cache = grammar.setdefault("_template_field_cache", {})
    cached = cache.get(template)
    if isinstance(cached, tuple):
        return cached
    fields = tuple(field_name for _, field_name, _, _ in Formatter().parse(str(template or "")) if field_name)
    cache[template] = fields
    return fields


def _prepare_render_grammar(grammar: dict) -> dict:
    if not isinstance(grammar, dict):
        return {}
    if grammar.get("_render_cache_ready"):
        return grammar
    prepared = dict(grammar)
    prepared["_render_vocab"] = {
        key: tuple(_clean_text_list(prepared.get(source_key, [])))
        for key, source_key in _TITLE_RENDER_VOCAB_FIELDS.items()
    }
    extra_placeholder_values: dict[str, tuple[str, ...]] = {}
    for raw_key, raw_values in dict(prepared.get("tagline_placeholder_values", {}) or {}).items():
        key = str(raw_key).strip()
        values = tuple(_clean_text_list(raw_values))
        if not key or not values:
            continue
        extra_placeholder_values[key] = values
        extra_placeholder_values.setdefault(key.casefold(), values)
    prepared["_extra_placeholder_values"] = extra_placeholder_values
    prepared["_template_field_cache"] = {}

    ordered_tagline_templates: dict[str, tuple[str, ...]] = {}
    for genre, raw_values in dict(prepared.get("tagline_templates", {}) or {}).items():
        values = tuple(
            sorted(
                _dedupe_template_list(raw_values),
                key=lambda template: (
                    template.count("{") > 1,
                    template.count("{"),
                    -(1 if re.search(r"[.!?]", str(template)) else 0),
                    -(1 if len(str(template).split()) >= 5 else 0),
                    _template_signature(template),
                ),
            )
        )
        if values:
            ordered_tagline_templates[str(genre)] = values
    prepared["_ordered_tagline_templates"] = ordered_tagline_templates
    prepared["_render_cache_ready"] = True
    return prepared


def _validate_tagline_templates(grammar: dict, *, mode: str) -> None:
    tagline_templates = grammar.get("tagline_templates", {})
    if not isinstance(tagline_templates, dict):
        return
    cross_genre: dict[str, set[str]] = {}
    for genre, raw_values in tagline_templates.items():
        genre_key = str(genre)
        if genre_key == "default":
            continue
        for value in _dedupe_template_list(raw_values):
            if contains_placeholder_syntax(value) and "[" in str(value):
                audit_fallback_hit(
                    "title_grammar_bank.json",
                    "tagline_template_square_bracket_placeholder",
                    detail=f"{genre_key} template uses square-bracket placeholder: {value}",
                    mode=mode,
                )
            key = _template_signature(value)
            if key:
                cross_genre.setdefault(key, set()).add(genre_key)
    if mode == "research":
        for genre in GENRES:
            values = _dedupe_template_list(tagline_templates.get(genre, []))
            if len(values) < _MIN_TAGLINE_TEMPLATES_PER_GENRE:
                audit_fallback_hit(
                    "title_grammar_bank.json",
                    "tagline_template_inventory_shallow",
                    detail=f"{genre} has only {len(values)} tagline templates; need at least {_MIN_TAGLINE_TEMPLATES_PER_GENRE}",
                    mode=mode,
                )
        duplicate_rows = sorted(
            [
                (template, sorted(genres))
                for template, genres in cross_genre.items()
                if len(genres) > 1
            ],
            key=lambda item: (-len(item[1]), item[0]),
        )
        for template, genres in duplicate_rows[:20]:
            audit_fallback_hit(
                "title_grammar_bank.json",
                "tagline_template_cross_genre_duplicate",
                detail=f"template '{template}' reused across genres: {', '.join(genres)}",
                mode=mode,
            )


def _load_research_grammar(base_dir: Path, mode: str) -> dict:
    artifact_path = title_grammar_bank_path(base_dir)
    payload = load_title_grammar_bank(base_dir, mode=mode)
    if not isinstance(payload, dict):
        audit_fallback_hit("title_generation", "debug_title_grammar_missing", detail="using built-in title grammar", mode=mode)
        return _default_grammar()
    required_lists = (
        "adjectives",
        "nouns",
        "abstract_nouns",
        "locations",
        "celestial_words",
        "technology_words",
        "mythic_words",
        "action_words",
        "franchise_affixes",
    )
    normalized = {}
    for key in required_lists:
        values = _clean_text_list(
            require_payload_value(
                payload,
                key,
                artifact_label="title_grammar_bank.json",
                artifact_path=artifact_path,
                mode=mode,
                validator=lambda value: isinstance(value, list) and len(value) > 0,
            )
            or []
        )
        normalized[key] = values or _default_grammar()[key]
    genre_templates = require_payload_value(
        payload,
        "genre_templates",
        artifact_label="title_grammar_bank.json",
        artifact_path=artifact_path,
        mode=mode,
        validator=lambda value: isinstance(value, dict) and len(value) > 0,
    )
    tagline_templates = require_payload_value(
        payload,
        "tagline_templates",
        artifact_label="title_grammar_bank.json",
        artifact_path=artifact_path,
        mode=mode,
        validator=lambda value: isinstance(value, dict) and len(value) > 0,
    )
    normalized["genre_templates"] = {str(k): _clean_text_list(v) for k, v in dict(genre_templates or {}).items() if _clean_text_list(v)}
    normalized["tagline_templates"] = {str(k): _dedupe_template_list(v) for k, v in dict(tagline_templates or {}).items() if _dedupe_template_list(v)}
    allowed_placeholders = payload.get("allowed_tagline_placeholders", [])
    normalized["allowed_tagline_placeholders"] = [
        str(item).strip()
        for item in list(allowed_placeholders or [])
        if str(item).strip()
    ]
    render_constraints = payload.get("tagline_render_constraints", {})
    if isinstance(render_constraints, dict):
        normalized["tagline_render_constraints"] = dict(render_constraints)
    else:
        normalized["tagline_render_constraints"] = {}
    extra_placeholder_values = payload.get("tagline_placeholder_values", {})
    if isinstance(extra_placeholder_values, dict):
        normalized["tagline_placeholder_values"] = {
            str(k): _clean_text_list(v)
            for k, v in extra_placeholder_values.items()
            if _clean_text_list(v)
        }
    else:
        normalized["tagline_placeholder_values"] = {}
    if mode == "research":
        for genre in GENRES:
            if genre not in normalized["genre_templates"] and "default" not in normalized["genre_templates"]:
                audit_fallback_hit("title_grammar_bank.json", "genre_template_missing", detail=f"{genre} missing genre template", mode=mode)
            if genre not in normalized["tagline_templates"] and "default" not in normalized["tagline_templates"]:
                audit_fallback_hit("title_grammar_bank.json", "tagline_template_missing", detail=f"{genre} missing tagline template", mode=mode)
    _validate_tagline_templates(normalized, mode=mode)
    audit_artifact_usage(
        "title_grammar_bank.json",
        artifact_path,
        sections=[*required_lists, "genre_templates", "tagline_templates", "tagline_placeholder_values", "allowed_tagline_placeholders", "tagline_render_constraints"],
    )
    return _prepare_render_grammar(normalized)


def _base_title_genre_weights(base_dir: Path, mode: str) -> tuple[dict[str, float], dict]:
    priors_path = modeling_priors_path(base_dir)
    priors = load_modeling_priors_artifact(base_dir, mode=mode) or {}
    title_priors = prior_section(priors, "title_generation")
    for key in ("genre_base_weights", "genre_weights", "genre_prevalence"):
        raw = title_priors.get(key)
        if isinstance(raw, dict) and raw:
            audit_artifact_usage("modeling_priors.json", priors_path, sections=[f"title_generation.{key}"])
            return _normalise_genre_weight_map(raw), title_priors
    if mode == "research":
        audit_fallback_hit(
            "modeling_priors.json",
            "title_generation.genre_base_weights_missing",
            detail="research-mode title generation requires title_generation.genre_base_weights",
            mode=mode,
        )
    return _normalise_genre_weight_map(dict(GENRE_WEIGHTS)), title_priors


def _genre_probability_vector(
    base_weights: dict[str, float],
    *,
    temporal: dict | None,
    title_priors: dict | None,
    year: int,
) -> np.ndarray:
    weights = dict(base_weights)
    phase = _phase_for_year(temporal, year)
    title_priors = title_priors if isinstance(title_priors, dict) else {}

    prestige_genres = _genre_list(title_priors.get("prestige_genres"), _PRESTIGE_GENRES)
    franchise_genres = _genre_list(title_priors.get("franchise_genres"), _FRANCHISE_GENRES)
    experimental_genres = _genre_list(title_priors.get("experimental_genres"), _EXPERIMENTAL_GENRES)
    low_cost_genres = _genre_list(title_priors.get("low_cost_genres"), _LOW_COST_GENRES)

    try:
        prestige_bias = max(0.0, min(1.0, float(phase.get("prestige_bias", 0.5))))
    except Exception:
        prestige_bias = 0.5
    try:
        franchise_pressure = max(0.0, min(1.0, float(phase.get("franchise_pressure", 0.5))))
    except Exception:
        franchise_pressure = 0.5
    try:
        experimentation_bias = max(0.0, min(1.0, float(phase.get("experimentation_bias", 0.5))))
    except Exception:
        experimentation_bias = 0.5
    try:
        economic_heat = max(0.0, min(1.0, float(phase.get("economic_heat", 0.5))))
    except Exception:
        economic_heat = 0.5

    prestige_delta = prestige_bias - 0.5
    franchise_delta = franchise_pressure - 0.5
    experimental_delta = experimentation_bias - 0.5
    heat_delta = economic_heat - 0.5

    for genre in list(weights.keys()):
        factor = 1.0
        if genre in prestige_genres:
            factor *= 1.0 + 0.95 * prestige_delta
        elif genre in franchise_genres:
            factor *= 1.0 - 0.25 * prestige_delta
        if genre in franchise_genres:
            factor *= 1.0 + 1.10 * franchise_delta
        elif genre in prestige_genres:
            factor *= 1.0 - 0.20 * franchise_delta
        if genre in experimental_genres:
            factor *= 1.0 + 1.15 * experimental_delta
        elif genre in franchise_genres:
            factor *= 1.0 - 0.20 * experimental_delta
        if genre in low_cost_genres:
            factor *= 1.0 - 0.30 * heat_delta
        elif genre in franchise_genres:
            factor *= 1.0 + 0.35 * heat_delta
        weights[genre] = max(0.001, float(weights[genre]) * factor)

    total = float(sum(weights.values())) or 1.0
    return np.array([float(weights.get(genre, 0.001)) / total for genre in GENRES], dtype=float)


def _year_sample_debug(rng: np.random.RandomState) -> int:
    decades = sorted(DECADE_WEIGHTS.keys())
    probs = np.array([DECADE_WEIGHTS[d] for d in decades], dtype=float)
    probs = probs / probs.sum()
    decade = int(rng.choice(decades, p=probs))
    return decade + int(rng.randint(0, 10))


def _allocate_years_from_weights(existing_years: pd.Series, target_count: int, weights: dict[int, float], rng: np.random.RandomState) -> list[int]:
    if not weights:
        return []
    years = sorted(weights.keys())
    total_weight = float(sum(weights.values())) or 1.0
    raw_targets = {year: (float(weights[year]) / total_weight) * float(target_count) for year in years}
    desired = {year: int(np.floor(value)) for year, value in raw_targets.items()}
    remainder = max(0, int(target_count) - sum(desired.values()))
    ranked = sorted(years, key=lambda year: raw_targets[year] - desired[year], reverse=True)
    for year in ranked[:remainder]:
        desired[year] += 1

    existing_counts = (
        pd.to_numeric(existing_years, errors="coerce")
        .dropna()
        .astype(int)
        .value_counts()
        .to_dict()
    )
    planned: list[int] = []
    for year in years:
        deficit = max(0, int(desired[year]) - int(existing_counts.get(year, 0)))
        planned.extend([year] * deficit)
    rng.shuffle(planned)
    return planned


def _desired_year_counts_from_weights(target_count: int, weights: dict[int, float]) -> dict[int, int]:
    if not weights:
        return {}
    years = sorted(int(year) for year in weights.keys())
    total_weight = float(sum(weights.values())) or 1.0
    raw_targets = {year: (float(weights[year]) / total_weight) * float(target_count) for year in years}
    desired = {year: int(np.floor(value)) for year, value in raw_targets.items()}
    remainder = max(0, int(target_count) - sum(desired.values()))
    ranked = sorted(years, key=lambda year: raw_targets[year] - desired[year], reverse=True)
    for year in ranked[:remainder]:
        desired[year] += 1
    return desired


def _desired_year_counts_uniform(target_count: int, start_year: int, end_year: int) -> dict[int, int]:
    years = list(range(int(start_year), int(end_year) + 1))
    if not years:
        return {}
    desired = {year: target_count // len(years) for year in years}
    for year in years[: target_count % len(years)]:
        desired[year] += 1
    return desired


def _sanitize_existing_title_bank(
    df: pd.DataFrame,
    *,
    desired_counts: dict[int, int],
    start_year: int,
    end_year: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    stats = {
        "dropped_out_of_range": 0,
        "trimmed_surplus": 0,
        "dropped_duplicate_titles": 0,
        "dropped_duplicate_taglines": 0,
        "dropped_placeholder_rows": 0,
        "dropped_weak_rows": 0,
    }
    if df.empty:
        return df.copy(), stats

    work = df.copy()
    if "title" in work.columns:
        work["title"] = work["title"].map(sanitize_title)
    if {"title", "tagline"}.issubset(work.columns):
        work["tagline"] = work.apply(
            lambda row: sanitize_tagline(row.get("tagline", ""), title=row.get("title", "")),
            axis=1,
        )
        weak_mask = work.apply(
            lambda row: looks_like_weak_title(row.get("title", ""))
            or looks_like_weak_tagline(row.get("tagline", ""), title=row.get("title", "")),
            axis=1,
        )
        stats["dropped_weak_rows"] = int(weak_mask.sum())
        work = work.loc[~weak_mask].copy()
    if {"title", "tagline"}.issubset(work.columns):
        placeholder_mask = work["title"].astype(str).map(contains_placeholder_syntax) | work["tagline"].astype(str).map(contains_placeholder_syntax)
        stats["dropped_placeholder_rows"] = int(placeholder_mask.sum())
        work = work.loc[~placeholder_mask].copy()
    if "title" in work.columns:
        before = len(work)
        work = work.drop_duplicates(subset=["title"], keep="first").reset_index(drop=True)
        stats["dropped_duplicate_titles"] = max(0, before - len(work))
    if "tagline" in work.columns:
        before = len(work)
        work = work.drop_duplicates(subset=["tagline"], keep="first").reset_index(drop=True)
        stats["dropped_duplicate_taglines"] = max(0, before - len(work))

    if "year" not in work.columns:
        return work.iloc[0:0].copy(), stats

    year_values = pd.to_numeric(work["year"], errors="coerce")
    in_range_mask = year_values.between(int(start_year), int(end_year), inclusive="both")
    stats["dropped_out_of_range"] = int((~in_range_mask).sum())
    work = work.loc[in_range_mask].copy()
    if work.empty:
        return work.reset_index(drop=True), stats

    year_values = pd.to_numeric(work["year"], errors="coerce").fillna(int(start_year)).astype(int)
    rng = np.random.RandomState(seed)
    kept_parts: list[pd.DataFrame] = []
    for year in sorted(int(y) for y in desired_counts.keys()):
        year_rows = work.loc[year_values == int(year)]
        keep_n = max(0, int(desired_counts.get(int(year), 0)))
        if year_rows.empty or keep_n <= 0:
            continue
        if len(year_rows) > keep_n:
            chosen = rng.permutation(year_rows.index.to_numpy())[:keep_n]
            year_rows = year_rows.loc[sorted(chosen)]
        kept_parts.append(year_rows)

    if kept_parts:
        trimmed = pd.concat(kept_parts, ignore_index=False).sort_index().reset_index(drop=True)
    else:
        trimmed = work.iloc[0:0].copy()

    stats["trimmed_surplus"] = max(0, len(work) - len(trimmed))
    return trimmed, stats


def _default_grammar() -> dict:
    return _prepare_render_grammar({
        "adjectives": ["Broken", "Silent", "Crimson", "Neon", "Golden", "Ancient", "Hidden", "Final"],
        "nouns": ["Edge", "Cipher", "Raven", "Threshold", "Fortune", "Mirror", "Passage", "Archive"],
        "abstract_nouns": ["Signal", "Accord", "Legacy", "Directive", "Labyrinth", "Nexus"],
        "locations": ["Harbor", "Frontier", "Summit", "Midnight", "Skyline", "Fortress"],
        "celestial_words": ["Aurora", "Zenith", "Eclipse", "Orbit"],
        "technology_words": ["Protocol", "Matrix", "Circuit", "Code"],
        "mythic_words": ["Oracle", "Phoenix", "Titan", "Avalon"],
        "action_words": ["Rising", "Unbound", "Reclaimed", "Awakened"],
        "franchise_affixes": ["Origins", "Legacy", "Chronicles", "Returns"],
        "genre_templates": {genre: ["{adjective} {noun}", "{noun}: {abstract}", "{celestial} {noun}", "{adjective} {noun} {action}"] for genre in GENRES},
        "tagline_templates": {genre: ["Everything changes tonight.", "No one gets out unchanged."] for genre in GENRES},
    })


def _render_title(rng: np.random.RandomState, grammar: dict, genre: str, *, mode: str) -> str:
    templates = grammar.get("genre_templates", {}).get(genre) or grammar.get("genre_templates", {}).get("default")
    if not templates:
        audit_fallback_hit("title_generation", "genre_template_missing", detail=f"missing title template for {genre}", mode=mode)
        templates = ["{adjective} {noun}"]
    template = str(templates[int(rng.randint(0, len(templates)))])
    vocab = grammar.get("_render_vocab") or {
        key: tuple(_clean_text_list(grammar.get(source_key, [])))
        for key, source_key in _TITLE_RENDER_VOCAB_FIELDS.items()
    }
    substitutions = {}
    for field_name in _template_fields(grammar, template):
        vocab_key = _RENDER_ALIAS_MAP.get(field_name, field_name)
        values = tuple(str(x).strip() for x in vocab.get(vocab_key, ()) if str(x).strip())
        if not values:
            audit_fallback_hit(
                "title_generation",
                "template_vocabulary_missing",
                detail=f"title template field {field_name} had no backing vocabulary",
                mode=mode,
            )
            substitutions[field_name] = field_name.replace("_", " ").title()
        else:
            substitutions[field_name] = values[int(rng.randint(0, len(values)))]
    return sanitize_title(_clean_rendered_text(template.format(**substitutions)))


def _estimate_title_capacity_for_genre(grammar: dict, genre: str) -> int:
    templates = grammar.get("genre_templates", {}).get(genre) or grammar.get("genre_templates", {}).get("default") or []
    vocab = grammar.get("_render_vocab") or {
        key: tuple(_clean_text_list(grammar.get(source_key, [])))
        for key, source_key in _TITLE_RENDER_VOCAB_FIELDS.items()
    }
    capacity = 0
    for template in templates:
        combos = 1
        fields = _template_fields(grammar, str(template))
        if not fields:
            capacity += 1
            continue
        for field_name in dict.fromkeys(fields):
            vocab_key = _RENDER_ALIAS_MAP.get(field_name, field_name)
            combos *= max(1, len(tuple(v for v in vocab.get(vocab_key, ()) if str(v).strip())))
        capacity += int(combos)
    return int(capacity)


def _validate_title_capacity_for_target(
    grammar: dict,
    *,
    target_count: int,
    base_genre_weights: dict[str, float],
    temporal: dict | None,
    title_priors: dict,
    start_year: int | None,
    end_year: int | None,
) -> None:
    """Fail early if a title grammar cannot support the requested scale.

    Without this guard a 200k lab run could spend time on earlier steps and then
    fail deep inside title generation once low-capacity genres exhaust their
    unique combinations.
    """
    target_count = int(target_count)
    if target_count <= 0:
        return
    year_lo = int(start_year if start_year is not None else 1950)
    year_hi = int(end_year if end_year is not None else 2025)
    years = list(range(year_lo, year_hi + 1))
    if not years:
        return
    if temporal:
        year_weights = year_weight_map(temporal, start_year=year_lo, end_year=year_hi)
    else:
        year_weights = {year: 1.0 for year in years}
    desired_by_year = _desired_year_counts_from_weights(target_count, year_weights)
    expected_by_genre = {str(genre): 0.0 for genre in GENRES}
    for year, year_count in desired_by_year.items():
        probs = _genre_probability_vector(
            base_genre_weights,
            temporal=temporal,
            title_priors=title_priors,
            year=int(year),
        )
        for genre, prob in zip(GENRES, probs, strict=False):
            expected_by_genre[str(genre)] += float(year_count) * float(prob)

    capacity_by_genre = {str(genre): _estimate_title_capacity_for_genre(grammar, str(genre)) for genre in GENRES}
    bad = []
    for genre in GENRES:
        expected = float(expected_by_genre.get(str(genre), 0.0))
        required = max(64, int(np.ceil(expected * 1.25)) + 8)
        capacity = int(capacity_by_genre.get(str(genre), 0))
        if capacity < required:
            bad.append((str(genre), capacity, required, int(round(expected))))
    total_capacity = int(sum(capacity_by_genre.values()))
    total_required = int(np.ceil(target_count * 1.25))
    if total_capacity < total_required:
        bad.append(("TOTAL", total_capacity, total_required, target_count))
    if bad:
        preview = "; ".join(
            f"{genre}: capacity={capacity}, required~{required}, expected={expected}"
            for genre, capacity, required, expected in bad[:10]
        )
        raise RuntimeError(
            "title_grammar_bank.json does not have enough unique title capacity "
            f"for target_count={target_count}. Regenerate the title grammar with "
            f"larger vocabulary/template pools. Problem genres: {preview}"
        )


def _tagline_action_candidates(template: str, values: list[str]) -> list[str]:
    clean = [clean_display_text(value).strip().lower() for value in values if clean_display_text(value).strip()]
    if not clean:
        return []
    template_low = str(template or "").strip().lower()
    if any(token in template_low for token in ("them all", "the enemy", "must ", "will ")) or re.search(r"\b(?:kill|save|protect|avenge)\b", template_low):
        filtered = [value for value in clean if value in _TRANSITIVE_TAGLINE_ACTIONS]
        if filtered:
            return filtered
    return clean


def _render_tagline_template(rng: np.random.RandomState, grammar: dict, template: str, *, mode: str) -> str:
    extra_placeholder_values = grammar.get("_extra_placeholder_values") or {}
    vocab = grammar.get("_render_vocab") or {
        key: tuple(_clean_text_list(grammar.get(source_key, [])))
        for key, source_key in _TITLE_RENDER_VOCAB_FIELDS.items()
    }
    substitutions: dict[str, str] = {}
    for field_name in _template_fields(grammar, template):
        vocab_key = _RENDER_ALIAS_MAP.get(field_name, field_name)
        extra_values = (
            extra_placeholder_values.get(field_name)
            or extra_placeholder_values.get(str(field_name).casefold())
            or extra_placeholder_values.get(vocab_key)
            or extra_placeholder_values.get(str(vocab_key).casefold())
            or []
        )
        values = tuple(str(x).strip() for x in (extra_values or vocab.get(vocab_key, ())) if str(x).strip())
        if extra_values:
            candidates = [clean_display_text(value).strip() for value in values if clean_display_text(value).strip()]
        elif vocab_key == "action":
            candidates = _tagline_action_candidates(template, list(values))
        elif vocab_key == "location":
            candidates = [clean_display_text(value).strip() for value in values if clean_display_text(value).strip()]
        else:
            candidates = [clean_display_text(value).strip().lower() for value in values if clean_display_text(value).strip()]
        candidates = [value for value in candidates if value]
        if not candidates:
            audit_fallback_hit(
                "title_generation",
                "tagline_template_vocabulary_missing",
                detail=f"tagline template field {field_name} had no backing vocabulary",
                mode=mode,
            )
            substitutions[field_name] = field_name.replace("_", " ").lower()
            continue
        substitutions[field_name] = candidates[int(rng.randint(0, len(candidates)))]
    return sanitize_tagline(_clean_rendered_text(str(template).format(**substitutions)))


_RENDER_PLACEHOLDER_PATTERNS = (
    r"\babstract nouns\b",
    r"\baction words\b",
    r"\bmythic words\b",
    r"\btechnology words\b",
    r"\bcelestial words\b",
    r"\bfranchise affixes\b",
)


def _is_usable_rendered_text(text: object) -> bool:
    value = clean_display_text(text)
    if not value:
        return False
    if contains_placeholder_syntax(text):
        return False
    low = value.lower()
    for pattern in _RENDER_PLACEHOLDER_PATTERNS:
        if re.search(pattern, low):
            return False
    return True


def _clean_rendered_text(text: object) -> str:
    return clean_display_text(text)


def _tagline(
    grammar: dict,
    genre: str,
    rng: np.random.RandomState,
    *,
    mode: str,
    avoid: list[str] | None = None,
) -> tuple[str, str]:
    ordered_templates = grammar.get("_ordered_tagline_templates", {})
    templates = ordered_templates.get(genre) or ordered_templates.get("default") or grammar.get("tagline_templates", {}).get(genre) or grammar.get("tagline_templates", {}).get("default")
    if not templates:
        audit_fallback_hit("title_generation", "tagline_template_missing", detail=f"missing tagline template for {genre}", mode=mode)
        templates = ["Everything changes tonight.", "No one gets out unchanged."]
    template_count = len(templates)
    start_idx = int(rng.randint(0, template_count)) if template_count > 1 else 0
    for offset in range(template_count):
        template = str(templates[(start_idx + offset) % template_count])
        if "{" not in template:
            candidate = sanitize_tagline(template)
        else:
            candidate = _render_tagline_template(rng, grammar, template, mode=mode)
        if candidate and not looks_like_weak_tagline(candidate):
            if avoid and tagline_is_near_duplicate(candidate, avoid, threshold=0.92):
                continue
            return candidate, _tagline_template_family_signature(template)
    template = str(templates[start_idx]) if template_count else str(templates[0])
    if "{" not in template:
        return sanitize_tagline(template), _tagline_template_family_signature(template)
    return _render_tagline_template(rng, grammar, template, mode=mode), _tagline_template_family_signature(template)


def _tagline_constraints(grammar: dict) -> dict[str, int | bool]:
    raw = grammar.get("tagline_render_constraints")
    if not isinstance(raw, dict):
        raw = {}
    return {
        "min_words": max(1, int(raw.get("min_words", 4) or 4)),
        "max_words": max(3, int(raw.get("max_words", 10) or 10)),
        "max_placeholder_count": max(0, int(raw.get("max_placeholder_count", 1) or 1)),
        "forbid_square_brackets": bool(raw.get("forbid_square_brackets", True)),
        "allow_unresolved_placeholders": bool(raw.get("allow_unresolved_placeholders", False)),
    }


def _is_valid_materialized_tagline(text: object, *, title: object | None, grammar: dict) -> bool:
    value = sanitize_tagline(text, title=title)
    if not value or contains_placeholder_syntax(value):
        return False
    if looks_like_weak_tagline(value, title=title):
        return False
    constraints = _tagline_constraints(grammar)
    words = len(value.split())
    if words < int(constraints["min_words"]) or words > int(constraints["max_words"]):
        return False
    return True


def _fast_title_specific_tagline(title: str, tagline_counts: dict[str, int]) -> tuple[str, str] | None:
    title_text = sanitize_title(title)
    if not title_text:
        return None
    templates = (
        "{title} changes everything.",
        "No one leaves {title} unchanged.",
        "The truth behind {title} comes due.",
        "Every secret in {title} has a price.",
        "{title} leaves no one untouched.",
    )
    for template in templates:
        candidate = sanitize_tagline(template.format(title=title_text), title=title_text)
        tagline_sig = tagline_signature(candidate)
        if not candidate or not tagline_sig:
            continue
        if contains_placeholder_syntax(candidate):
            continue
        if looks_like_weak_tagline(candidate, title=title_text):
            continue
        if tagline_counts.get(tagline_sig, 0) >= _MAX_TITLE_BANK_TAGLINE_REUSE:
            continue
        return candidate, "fast_title_specific"
    return None


def _materialize_tagline_for_title(
    *,
    grammar: dict,
    genre: str,
    rng: np.random.RandomState,
    mode: str,
    title: str,
    tagline_history: list[str],
    tagline_counts: dict[str, int],
    tagline_template_family_counts: dict[str, int],
    tagline_template_family_cap: int,
    fast_taglines: bool = False,
) -> tuple[str, str]:
    templates = list(
        grammar.get("tagline_templates", {}).get(genre)
        or grammar.get("tagline_templates", {}).get("default")
        or []
    )
    if fast_taglines:
        fast_tagline = _fast_title_specific_tagline(title, tagline_counts)
        if fast_tagline is not None:
            return fast_tagline
    family_cap = max(_MAX_TAGLINE_TEMPLATE_FAMILY_REUSE, int(tagline_template_family_cap))
    if fast_taglines:
        family_cap = max(family_cap, int(sum(tagline_counts.values()) + 1))
    passes = [
        {"attempts": 8 if fast_taglines else 48, "near_window": 0 if fast_taglines else 24, "near_threshold": 0.93, "family_cap": family_cap},
        {"attempts": 16 if fast_taglines else 96, "near_window": 0 if fast_taglines else 12, "near_threshold": 0.975, "family_cap": max(family_cap * 2, 2)},
    ]
    best_candidate = ""
    best_family_sig = ""
    best_score: tuple[float, int, int] | None = None

    for cfg in passes:
        for _ in range(int(cfg["attempts"])):
            rendered_tagline, template_family_sig = _tagline(
                grammar,
                genre,
                rng,
                mode=mode,
                avoid=None if fast_taglines else tagline_history[-40:],
            )
            candidate_tagline = sanitize_tagline(rendered_tagline, title=title)
            tagline_sig = tagline_signature(candidate_tagline)
            if not (
                _is_valid_materialized_tagline(candidate_tagline, title=title, grammar=grammar)
                and tagline_sig
                and tagline_counts.get(tagline_sig, 0) < _MAX_TITLE_BANK_TAGLINE_REUSE
            ):
                continue
            family_count = int(tagline_template_family_counts.get(template_family_sig, 0))
            if family_count >= int(cfg["family_cap"]):
                continue
            if int(cfg["near_window"]) > 0 and tagline_is_near_duplicate(candidate_tagline, tagline_history[-int(cfg["near_window"]):], threshold=float(cfg["near_threshold"])):
                continue
            return candidate_tagline, template_family_sig

            # unreachable, kept for clarity

    # If the strict/relaxed passes both fail, keep the best valid non-exact-duplicate
    # candidate rather than crashing on random exhaustion.
    for _ in range(96):
        rendered_tagline, template_family_sig = _tagline(
            grammar,
            genre,
            rng,
            mode=mode,
            avoid=None,
        )
        candidate_tagline = sanitize_tagline(rendered_tagline, title=title)
        tagline_sig = tagline_signature(candidate_tagline)
        if not (
            _is_valid_materialized_tagline(candidate_tagline, title=title, grammar=grammar)
            and tagline_sig
            and tagline_counts.get(tagline_sig, 0) < _MAX_TITLE_BANK_TAGLINE_REUSE
        ):
            continue
        family_count = int(tagline_template_family_counts.get(template_family_sig, 0))
        near_penalty = 0 if fast_taglines else int(tagline_is_near_duplicate(candidate_tagline, tagline_history[-24:], threshold=0.93))
        score = (float(near_penalty), family_count, len(candidate_tagline.split()))
        if best_score is None or score < best_score:
            best_score = score
            best_candidate = candidate_tagline
            best_family_sig = template_family_sig

    if best_candidate:
        return best_candidate, best_family_sig

    # Final deterministic sweep: walk the actual templates directly and accept the
    # best placeholder-free materialization even if the stricter family/near-dup
    # heuristics are exhausted. This keeps research mode fail-closed on malformed
    # text, but avoids random exhaustion for otherwise valid genre templates.
    for template in templates:
        template = str(template or "").strip()
        if not template:
            continue
        family_sig = _tagline_template_family_signature(template)
        render_attempts = 4 if "{" in template else 1
        for _ in range(render_attempts):
            candidate_tagline = sanitize_tagline(
                _render_tagline_template(rng, grammar, template, mode=mode) if "{" in template else template,
                title=title,
            )
            tagline_sig = tagline_signature(candidate_tagline)
            if not candidate_tagline or not tagline_sig:
                continue
            if contains_placeholder_syntax(candidate_tagline):
                continue
            word_count = len(candidate_tagline.split())
            if word_count < 3 or word_count > 12:
                continue
            if tagline_counts.get(tagline_sig, 0) >= _MAX_TITLE_BANK_TAGLINE_REUSE:
                continue
            return candidate_tagline, family_sig
    for raw in (
        f"{title} changes everything.",
        f"No one leaves {title} unchanged.",
        f"The truth behind {title} comes due.",
        f"Every secret in {title} has a price.",
    ):
        candidate_tagline = sanitize_tagline(raw, title=title)
        tagline_sig = tagline_signature(candidate_tagline)
        if not candidate_tagline or not tagline_sig:
            continue
        if contains_placeholder_syntax(candidate_tagline):
            continue
        if looks_like_weak_tagline(candidate_tagline, title=title):
            continue
        if tagline_counts.get(tagline_sig, 0) >= _MAX_TITLE_BANK_TAGLINE_REUSE:
            continue
        return candidate_tagline, "title_specific_fallback"
    raise RuntimeError(f"Unable to materialize strong placeholder-free tagline for genre={genre} title={title}")


def topup(base_dir: Path, target_count: int, seed: int, start_year: int | None = None, end_year: int | None = None, mode: str = "research") -> None:
    import os

    os.makedirs(base_dir / "entities", exist_ok=True)
    fast_taglines = str(os.environ.get("DATA_SYS_FAST_TITLE_TAGLINES", "")).strip().lower() in {"1", "true", "yes", "on"}
    tb_path = base_dir / "entities" / "title_bank.csv"

    if tb_path.exists():
        df = pd.read_csv(tb_path, low_memory=False)
        cur = len(df)
        used = set(df["title"].astype(str).tolist()) if "title" in df.columns else set()
    else:
        df = pd.DataFrame(columns=["title", "tagline", "genre_hint", "year", "award_contender"])
        cur = 0
        used = set()
    tagline_counts: dict[str, int] = {}
    tagline_history: list[str] = []
    tagline_template_family_counts: dict[str, int] = {}
    if "tagline" in df.columns:
        for raw in df["tagline"].astype(str).tolist():
            clean = sanitize_tagline(raw)
            sig = tagline_signature(clean)
            if not sig:
                continue
            tagline_counts[sig] = tagline_counts.get(sig, 0) + 1
            tagline_history.append(clean)

    if cur >= target_count and start_year is None and end_year is None:
        print(f"title_bank already has {cur} rows (target {target_count})")
        return

    rng = np.random.RandomState(seed)
    rows = []
    if (start_year is None) != (end_year is None):
        raise ValueError("start_year and end_year must be provided together")

    if mode == "research":
        grammar = _load_research_grammar(base_dir, mode)
        temporal = load_temporal_regime_plan(base_dir, mode=mode)
        audit_artifact_usage("temporal_regime_plan.json", temporal_regime_plan_path(base_dir), sections=["year_weights", "phases"])
        base_genre_weights, title_priors = _base_title_genre_weights(base_dir, mode)
        _validate_title_capacity_for_target(
            grammar,
            target_count=int(target_count),
            base_genre_weights=base_genre_weights,
            temporal=temporal,
            title_priors=title_priors,
            start_year=start_year,
            end_year=end_year,
        )
    else:
        grammar = _default_grammar()
        temporal = None
        base_genre_weights, title_priors = _normalise_genre_weight_map(dict(GENRE_WEIGHTS)), {}

    if start_year is not None and end_year is not None:
        if mode == "research":
            weights = year_weight_map(temporal, start_year=int(start_year), end_year=int(end_year))
            desired_counts = _desired_year_counts_from_weights(int(target_count), weights)
            df, sanitize_stats = _sanitize_existing_title_bank(
                df,
                desired_counts=desired_counts,
                start_year=int(start_year),
                end_year=int(end_year),
                seed=int(seed),
            )
            cur = len(df)
            used = set(df["title"].astype(str).tolist()) if "title" in df.columns else set()
            if any(int(v) > 0 for v in sanitize_stats.values()):
                print(
                    "Sanitized existing title_bank: "
                    f"dropped_out_of_range={sanitize_stats['dropped_out_of_range']}, "
                    f"trimmed_surplus={sanitize_stats['trimmed_surplus']}, "
                    f"dropped_duplicate_titles={sanitize_stats['dropped_duplicate_titles']}, "
                    f"dropped_duplicate_taglines={sanitize_stats['dropped_duplicate_taglines']}, "
                    f"dropped_placeholder_rows={sanitize_stats['dropped_placeholder_rows']}, "
                    f"dropped_weak_rows={sanitize_stats['dropped_weak_rows']}"
                )
            planned_years = _allocate_years_from_weights(
                df["year"] if "year" in df.columns else pd.Series(dtype=float),
                int(target_count),
                weights,
                rng,
            )
        else:
            desired = _desired_year_counts_uniform(int(target_count), int(start_year), int(end_year))
            df, sanitize_stats = _sanitize_existing_title_bank(
                df,
                desired_counts=desired,
                start_year=int(start_year),
                end_year=int(end_year),
                seed=int(seed),
            )
            cur = len(df)
            used = set(df["title"].astype(str).tolist()) if "title" in df.columns else set()
            if any(int(v) > 0 for v in sanitize_stats.values()):
                print(
                    "Sanitized existing title_bank: "
                    f"dropped_out_of_range={sanitize_stats['dropped_out_of_range']}, "
                    f"trimmed_surplus={sanitize_stats['trimmed_surplus']}, "
                    f"dropped_duplicate_titles={sanitize_stats['dropped_duplicate_titles']}, "
                    f"dropped_duplicate_taglines={sanitize_stats['dropped_duplicate_taglines']}, "
                    f"dropped_placeholder_rows={sanitize_stats['dropped_placeholder_rows']}, "
                    f"dropped_weak_rows={sanitize_stats['dropped_weak_rows']}"
                )
            years = list(range(int(start_year), int(end_year) + 1))
            existing_counts = (
                pd.to_numeric(df["year"] if "year" in df.columns else pd.Series(dtype=float), errors="coerce")
                .dropna()
                .astype(int)
                .value_counts()
                .to_dict()
            )
            planned_years = []
            for year in years:
                deficit = max(0, int(desired[year]) - int(existing_counts.get(year, 0)))
                planned_years.extend([year] * deficit)
            rng.shuffle(planned_years)
        needed = len(planned_years)
    else:
        needed = max(0, target_count - cur)
        planned_years = []

    if needed <= 0:
        print(f"title_bank already satisfies target/distribution ({cur} rows)")
        return

    tagline_template_family_cap = max(_MAX_TAGLINE_TEMPLATE_FAMILY_REUSE, 2)
    if mode == "research":
        template_families = {
            _tagline_template_family_signature(template)
            for templates in (grammar.get("tagline_templates") or {}).values()
            for template in (templates or [])
            if _tagline_template_family_signature(template)
        }
        average_uses = int(np.ceil(float(target_count) / float(max(1, len(template_families)))))
        tagline_template_family_cap = max(64, average_uses * 4)
        if fast_taglines:
            tagline_template_family_cap = max(tagline_template_family_cap, int(target_count))
        print(
            "Using scaled tagline template family cap: "
            f"{tagline_template_family_cap} across {len(template_families)} template families"
            f" (fast_taglines={fast_taglines})",
            flush=True,
        )

    for idx in range(needed):
        year = planned_years[idx] if planned_years else _year_sample_debug(rng)
        genre_probs = _genre_probability_vector(
            base_genre_weights,
            temporal=temporal,
            title_priors=title_priors,
            year=int(year),
        )
        genre = str(rng.choice(GENRES, p=genre_probs))
        if mode == "research":
            title = None
            for _ in range(96):
                candidate = _render_title(rng, grammar, genre, mode=mode)
                if (
                    candidate
                    and candidate not in used
                    and _is_usable_rendered_text(candidate)
                    and not looks_like_weak_title(candidate)
                ):
                    title = candidate
                    break
            if title is None:
                raise RuntimeError("Unable to generate unique title from title grammar bank without fallback dedupe")
        else:
            title = sanitize_title(generate_compositional_title(rng, used))
        used.add(title)
        if mode == "research":
            tagline, tagline_template_family = _materialize_tagline_for_title(
                grammar=grammar,
                genre=genre,
                rng=rng,
                mode=mode,
                title=title,
                tagline_history=tagline_history,
                tagline_counts=tagline_counts,
                tagline_template_family_counts=tagline_template_family_counts,
                tagline_template_family_cap=tagline_template_family_cap,
                fast_taglines=fast_taglines,
            )
        else:
            tagline = sanitize_tagline(
                _tagline(grammar, genre, rng, mode=mode, avoid=tagline_history[-40:])[0],
                title=title,
            )
            tagline_template_family = ""
        tagline_sig = tagline_signature(tagline)
        if tagline_sig:
            tagline_counts[tagline_sig] = tagline_counts.get(tagline_sig, 0) + 1
            tagline_history.append(tagline)
        if tagline_template_family:
            tagline_template_family_counts[tagline_template_family] = tagline_template_family_counts.get(tagline_template_family, 0) + 1
        rows.append(
            {
                "title": title,
                "tagline": tagline,
                "genre_hint": genre,
                "year": year,
                "award_contender": bool(rng.rand() < 0.08),
            }
        )
        progress_interval = 1000 if fast_taglines else 10000
        if (idx + 1) % progress_interval == 0 or (idx + 1) == needed:
            print(f"Generated {idx + 1:,} / {needed:,} title-bank rows", flush=True)

    add_df = pd.DataFrame(rows)
    for col in df.columns:
        if col not in add_df.columns:
            add_df[col] = None
    for col in add_df.columns:
        if col not in df.columns:
            df[col] = None

    out = pd.concat([df[df.columns], add_df[df.columns]], ignore_index=True)
    if mode == "research" and {"title", "tagline"}.issubset(out.columns):
        placeholder_mask = out["title"].astype(str).map(contains_placeholder_syntax) | out["tagline"].astype(str).map(contains_placeholder_syntax)
        if bool(placeholder_mask.any()):
            bad = out.loc[placeholder_mask, ["title", "tagline"]].head(5).to_dict(orient="records")
            raise RuntimeError(f"title_bank.csv still contains placeholder syntax after materialization: {bad}")
    out.to_csv(tb_path, index=False)
    print(f"Added {needed} titles to {tb_path}")
    print(f"New size: {len(out)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Title bank top-up for Mirage.")
    parser.add_argument("--base-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--target-count", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260305)
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--mode", choices=("research", "debug"), default=current_mode())
    args = parser.parse_args()

    topup(
        Path(args.base_dir).resolve(),
        int(args.target_count),
        int(args.seed),
        start_year=args.start_year,
        end_year=args.end_year,
        mode=str(args.mode),
    )


if __name__ == "__main__":
    main()
