"""
V13 Pipeline -- generate_movies.py (orchestrator)
==================================================
Main assembly loop + CLI entry point.
Imports all components from the modular split:
  utils, world_state, financials, assembly, secondary_tables.

No business logic lives here -- only orchestration & I/O.
"""
import pandas as pd
import numpy as np
import gc
import hashlib
import json
import os, sys
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.ipc as ipc

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

sys.path.insert(0, os.path.dirname(__file__))

# === Contracts (constants & config) ====================================
from contracts import (
    SNAPSHOT_CONFIG, ENTITY_COUNTS, FRANCHISE_CONFIG,
)

# === Modular imports ==================================================
from utils import normalize_weights, _safe_float, TONE_STYLE_HINTS
from world_state import WorldState, get_person_latent, get_company_latent, latent_similarity
from llm_provider import get_llm_client, safe_json_parse
from financials import compute_financials, edge_is_active, record_financial_outcome
from assembly import (
    sample_movie_concept, pick_director, pick_co_director, pick_companies,
    pick_cast, pick_title, pick_keywords, pick_crew,
    _generate_tagline, _register_tagline_use, _tagline_reuse_score,
    _canonical_genre_label,
)
from policy_runtime import append_jsonl, keyword_rerank_budget_for_movies, rerank_budget_for_movies
from secondary_tables import (
    generate_release_dates, generate_box_office_weekly,
    generate_box_office_daily, generate_box_office_by_territory,
    generate_reviews, generate_awards, generate_locations,
    generate_alternate_titles, generate_ratings_breakdown,
    generate_movie_links, generate_company_links,
    generate_person_demographics, generate_tv_series,
    generate_user_ratings, generate_episode_cast,
    generate_media_links,
)
from schema import (
    TABLE_DEFS, build_secondary_generators, get_auto_pk_tables,
)
from year_planner import evolve_year
from generation_critic import run_post_generation_critic
from feather_sink import (
    PA_SCHEMAS,
    POST_LOOP_STREAMABLE,
    STREAMABLE_TABLES, df_to_arrow, make_table_sink, read_table,
)
from pipeline_runtime import resolve_workspace
from step100_resume import (
    GLOBAL_TABLES,
    PER_MOVIE_TABLES,
    Step100ResumeManager,
)
from continuation import (
    append_extension_plan,
    load_continuation_summary,
    validate_extension_request,
)
from continuation_lifecycle import apply_lifecycle_to_world
from text_polish import looks_like_weak_tagline, looks_like_weak_title, sanitize_tagline, sanitize_title

YEARLY_SNAPSHOT_ENABLED = bool(SNAPSHOT_CONFIG.get("write_yearly_snapshots", False))


def _yearly_snapshots_enabled(world: WorldState | None) -> bool:
    runtime_cfg = getattr(getattr(getattr(world, "workspace", None), "config", None), "runtime", None)
    override = getattr(runtime_cfg, "write_yearly_snapshots", None)
    if override is None:
        override = getattr(runtime_cfg, "enable_yearly_snapshots", None)
    if override is None:
        return YEARLY_SNAPSHOT_ENABLED
    return bool(override)


def _character_description_for_cast(archetype: object, genre: object, billing_order: int) -> str:
    archetype_text = " ".join(str(archetype or "").replace("_", " ").split()).strip()
    genre_text = str(genre or "drama").strip().lower() or "drama"
    slot_phrase = {
        1: "central figure in the film's main conflict",
        2: "major counterpart driving the film's emotional stakes",
        3: "key supporting presence shaping the ensemble dynamic",
    }.get(int(billing_order or 0), "supporting presence within the wider ensemble")
    if archetype_text:
        return f"{archetype_text}; {slot_phrase} in the {genre_text} storyline."
    return f"{slot_phrase.capitalize()} in the {genre_text} storyline."


def _movie_progress_log_path(base_dir: str) -> str:
    env_path = str(os.getenv("DATA_SYS_MOVIE_PROGRESS_LOG", "") or "").strip()
    if env_path:
        return env_path
    return os.path.join(base_dir, "decision_logs", "movie_generation_progress.jsonl")


def _log_movie_progress(base_dir: str, payload: dict[str, Any]) -> None:
    try:
        path = _movie_progress_log_path(base_dir)
        append_jsonl(path, payload)
        latest_path = str(os.getenv("DATA_SYS_MOVIE_PROGRESS_LOG_LATEST", "") or "").strip()
        if latest_path and latest_path != path:
            append_jsonl(latest_path, payload)
    except Exception:
        pass


def _write_csv_mirror(df: pd.DataFrame, path: str, *, label: str = "") -> None:
    """Keep root CSV mirrors synchronized with the authoritative in-memory tables."""
    try:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False, encoding="utf-8")
        if label:
            print(f"Saved {out_path} ({label}, {len(df):,} rows)")
    except Exception as exc:
        print(f"  CSV mirror warning for {path}: {exc}", flush=True)


def _empty_table_frame(table_name: str) -> pd.DataFrame:
    schema = PA_SCHEMAS.get(table_name)
    if schema is None:
        return pd.DataFrame()
    return pd.DataFrame({field.name: pd.Series(dtype=object) for field in schema})


def _write_empty_arrow(base_dir: str, table_name: str) -> None:
    schema = PA_SCHEMAS.get(table_name)
    if schema is None:
        return
    path = os.path.join(base_dir, f"{table_name}.arrow")
    df_to_arrow(_empty_table_frame(table_name), path, table_name=table_name)


def _append_decision_log(world: WorldState, payload: dict[str, Any]) -> None:
    path = getattr(world, "decision_log_path", None)
    if path:
        append_jsonl(path, payload)
    latest_path = getattr(world, "decision_log_latest_path", None)
    if latest_path and str(latest_path) != str(path):
        append_jsonl(latest_path, payload)


def _apply_company_links_to_world(world: WorldState, company_links: pd.DataFrame | list[dict[str, Any]] | None) -> None:
    world.company_family = {}
    if not hasattr(world, "_merge_families") or world._merge_families is None:
        world._merge_families = {}
    records: list[dict[str, Any]]
    if isinstance(company_links, pd.DataFrame):
        if company_links.empty:
            records = []
        else:
            records = company_links.to_dict("records")
    else:
        records = list(company_links or [])
    for link in records:
        c1 = int(link.get("company_id_1", 0) or 0)
        c2 = int(link.get("company_id_2", 0) or 0)
        if c1 <= 0 or c2 <= 0:
            continue
        world.company_family.setdefault(c1, set()).add(c2)
        world.company_family.setdefault(c2, set()).add(c1)
    for cid, related in world.company_family.items():
        world._merge_families.setdefault(int(cid), set()).update(int(item) for item in related)


def _scaled_tv_series_target(n_movies: int, world: WorldState | None = None) -> int:
    """V17: Dynamic TV series count based on movie count.

    Uses sqrt scaling for a natural sub-linear relationship:
    - 100 movies -> ~60 series
    - 1000 movies -> ~190 series
    - 10000 movies -> ~600 series
    - 100000 movies -> ~6000 series
    """
    base = int(ENTITY_COUNTS.get("tv_series", 150))
    runtime_cfg = getattr(getattr(getattr(world, 'workspace', None), 'config', None), 'runtime', None)
    floor = int(getattr(runtime_cfg, 'tv_series_floor_small', 8))
    sqrt_scale = float(getattr(runtime_cfg, 'tv_series_sqrt_scale', 6.0))
    large_ratio = float(getattr(runtime_cfg, 'tv_series_large_ratio', 0.06))
    max_series = int(getattr(runtime_cfg, 'tv_series_max', 8000))
    sqrt_target = int(round(max(float(floor), sqrt_scale * np.sqrt(max(1, n_movies)))))
    if n_movies <= base:
        return max(floor, min(base, sqrt_target))
    scaled = max(base, sqrt_target, int(round(max(0.0, large_ratio) * max(1, n_movies))))
    if max_series > 0:
        scaled = min(max_series, scaled)
    return max(base, scaled)


def _get_memory_audit_recorder(default_experiment: str = "pipeline-checkpoints"):
    try:
        from memory_probe import get_env_audit_recorder
    except Exception:
        return None
    return get_env_audit_recorder(default_experiment=default_experiment)


def _get_speed_audit_recorder(base_dir: str | None = None, default_experiment: str = "pipeline-speed"):
    try:
        from speed_probe import get_env_speed_recorder
    except Exception:
        return None
    return get_env_speed_recorder(default_experiment=default_experiment, base_dir=base_dir)


def _speed_scope(speed_audit, name: str, *, category: str = "", units: int = 0, metadata: dict[str, Any] | None = None, note: str = ""):
    if speed_audit is None:
        return nullcontext(None)
    return speed_audit.track(name, category=category, units=units, metadata=metadata, note=note)


def _rerank_prompt(batch: list[dict[str, Any]]) -> str:
    return (
        "You are choosing the strongest reusable movie concept candidate for a synthetic IMDb-style dataset.\n"
        "Return JSON only as an array with objects shaped like "
        '{"movie_id": 1, "selected_pack_id": "pack_0001", "selected_index": 0}.\n'
        "Choose the candidate that feels most coherent, specific, and non-generic for its year, genre, tier, country, and market.\n"
        "Avoid mode collapse: if one candidate is already heavily reused and another is nearly as coherent, prefer the less overused option.\n"
        "Do not keep selecting the same pack_id across unrelated movies unless it is clearly the best fit.\n\n"
        f"Candidates:\n{json.dumps(batch, ensure_ascii=True)}"
    )


def _rerank_batch(world: WorldState, items: list[dict[str, Any]], llm_model: str | None) -> dict[int, tuple[int, str]]:
    if not items:
        return {}
    client = get_llm_client()
    response = client.generate(
        _rerank_prompt(items),
        model=llm_model,
        json_mode=True,
        temperature=0.2,
        max_tokens=3072,
        timeout_sec=90.0,
        max_attempts=4,
    )
    parsed = safe_json_parse(response.text)
    result: dict[int, tuple[int, str]] = {}
    if not isinstance(parsed, list):
        return result
    for row in parsed:
        if not isinstance(row, dict):
            continue
        movie_id = int(row.get("movie_id", 0) or 0)
        raw_selected_index = row.get("selected_index", -1)
        selected_index = int(raw_selected_index) if raw_selected_index is not None else -1
        selected_pack_id = str(row.get("selected_pack_id", "") or "")
        if movie_id > 0:
            result[movie_id] = (selected_index, selected_pack_id)
    return result


def _apply_llm_concept_rerank(
    world: WorldState,
    year_list: list[tuple[int, int, dict[str, Any]]],
    *,
    llm_model: str | None = None,
) -> int:
    if not getattr(world, "enable_llm_rerank", False):
        return 0
    budget = int(getattr(world, "rerank_budget_remaining", 0) or 0)
    if budget <= 0:
        return 0

    candidates = []
    for idx, (_year, movie_id, concept) in enumerate(year_list):
        options = concept.get("_rerank_candidates", []) if isinstance(concept, dict) else []
        confidence = float(concept.get("selection_confidence", 1.0)) if isinstance(concept, dict) else 1.0
        if len(options) < 2 or confidence >= 0.58:
            continue
        compact_candidates = []
        for pos, option in enumerate(options[:6]):
            option_pack_id = str(option.get("concept_pack_id", ""))
            option_country = str(option.get("country", ""))
            option_bucket = str(option.get("year_bucket", ""))
            compact_candidates.append(
                {
                    "index": pos,
                    "pack_id": option_pack_id,
                    "genre": str(option.get("genre", "")),
                    "tier": str(option.get("tier", "")),
                    "country": option_country,
                    "market": str(option.get("market", "")),
                    "tone": str(option.get("tone", "")),
                    "premise": str((option.get("concept_pack") or {}).get("premise_archetype", "")),
                    "conflict": str((option.get("concept_pack") or {}).get("conflict_pattern", "")),
                    "strategy": str(option.get("policy_targets", {}).get("company_strategy_tag", "")),
                    "season": str(option.get("policy_targets", {}).get("release_season_bias", "")),
                    "selection_confidence": round(float(option.get("selection_confidence", 0.0) or 0.0), 4),
                    "usage_count": int(getattr(world, "_concept_pack_usage_counts", {}).get(option_pack_id, 0)),
                    "country_usage": int(getattr(world, "_concept_country_usage_counts", {}).get(option_country, 0)),
                    "bucket_country_usage": int(
                        getattr(world, "_concept_bucket_country_usage_counts", {}).get((option_bucket, option_country), 0)
                    ),
                }
            )
        candidates.append(
            {
                "index_in_year_list": idx,
                "movie_id": int(movie_id),
                "confidence": round(confidence, 4),
                "current_pack_id": str(concept.get("concept_pack_id", "")),
                "candidates": compact_candidates,
            }
        )

    candidates.sort(key=lambda row: (row["confidence"], row["movie_id"]))
    candidates = candidates[:budget]
    if not candidates:
        return 0

    changed = 0
    batch_size = 12
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        try:
            picks = _rerank_batch(world, batch, llm_model)
        except Exception as exc:
            print(f"  LLM rerank skipped for batch {start // batch_size + 1}: {exc}")
            continue
        for item in batch:
            movie_id = int(item["movie_id"])
            selection = picks.get(movie_id)
            if selection is None:
                continue
            selected_index, selected_pack_id = selection
            year_idx = int(item["index_in_year_list"])
            year, _, concept = year_list[year_idx]
            options = concept.get("_rerank_candidates", [])
            if not (0 <= selected_index < len(options)):
                if selected_pack_id:
                    for option_idx, option in enumerate(options):
                        if str(option.get("concept_pack_id", "")) == selected_pack_id:
                            selected_index = option_idx
                            break
            if not (0 <= selected_index < len(options)):
                continue
            chosen = dict(options[selected_index])
            previous_pack_id = str(concept.get("concept_pack_id", ""))
            previous_country = str(concept.get("country", ""))
            previous_bucket = str(concept.get("year_bucket", ""))
            chosen["selection_mode"] = "concept_pack_reranked"
            chosen["selection_confidence"] = max(float(chosen.get("selection_confidence", 0.0) or 0.0), float(concept.get("selection_confidence", 0.0) or 0.0))
            year_list[year_idx] = (int(year), movie_id, chosen)
            chosen_pack_id = str(chosen.get("concept_pack_id", ""))
            chosen_country = str(chosen.get("country", ""))
            chosen_bucket = str(chosen.get("year_bucket", ""))
            pack_counts = getattr(world, "_concept_pack_usage_counts", None)
            country_counts = getattr(world, "_concept_country_usage_counts", None)
            bucket_country_counts = getattr(world, "_concept_bucket_country_usage_counts", None)
            if pack_counts is not None and previous_pack_id and previous_pack_id != chosen_pack_id:
                pack_counts[previous_pack_id] = max(0, int(pack_counts.get(previous_pack_id, 0)) - 1)
            if country_counts is not None and previous_country and previous_country != chosen_country:
                country_counts[previous_country] = max(0, int(country_counts.get(previous_country, 0)) - 1)
            if bucket_country_counts is not None and previous_bucket and previous_country and (
                previous_bucket != chosen_bucket or previous_country != chosen_country
            ):
                key = (previous_bucket, previous_country)
                bucket_country_counts[key] = max(0, int(bucket_country_counts.get(key, 0)) - 1)
            if pack_counts is not None and chosen_pack_id:
                pack_counts[chosen_pack_id] += 1
            if country_counts is not None and chosen_country:
                country_counts[chosen_country] += 1
            if bucket_country_counts is not None and chosen_bucket and chosen_country:
                bucket_country_counts[(chosen_bucket, chosen_country)] += 1
            changed += 1
            _append_decision_log(
                world,
                {
                    "stage": "llm_concept_rerank",
                    "movie_id": movie_id,
                    "year": int(year),
                    "previous_pack_id": str(concept.get("concept_pack_id", "")),
                    "chosen_pack_id": str(chosen.get("concept_pack_id", "")),
                    "previous_confidence": float(concept.get("selection_confidence", 0.0) or 0.0),
                    "candidate_count": len(options),
                    "previous_country": previous_country,
                    "chosen_country": chosen_country,
                },
            )
    world.rerank_budget_remaining = max(0, budget - changed)
    return changed


def _keyword_rerank_prompt(batch: list[dict[str, Any]]) -> str:
    return (
        "You are refining keyword bundles for synthetic IMDb-style movie rows.\n"
        "Return JSON only as an array of objects shaped like "
        '{"movie_id": 1, "selected_keyword_ids": [10, 11, 12, 13]}.\n'
        "Select a coherent, non-generic bundle that reflects genre, year, company flavor, planning motifs, and franchise continuity when present.\n"
        "Rules:\n"
        "- Most selected keywords should match the movie genre or a close genre family.\n"
        "- Reject contradictory off-genre keywords even if they are popular.\n"
        "- Prefer concrete setting, event, object, profession, relationship, subgenre, and franchise keywords over abstract labels.\n"
        "- If the movie is in a franchise, keep a small recurring core but allow installment-specific drift.\n"
        "- Choose only from the candidate keyword ids provided for each movie.\n"
        "Prefer 4-8 keywords per movie.\n\n"
        f"Items:\n{json.dumps(batch, ensure_ascii=True)}"
    )


def _rerank_keyword_batch(world: WorldState, items: list[dict[str, Any]], llm_model: str | None) -> dict[int, list[int]]:
    client = get_llm_client()
    response = client.generate(
        _keyword_rerank_prompt(items),
        model=llm_model,
        json_mode=True,
        temperature=0.15,
        max_tokens=3072,
        timeout_sec=90.0,
        max_attempts=4,
    )
    parsed = safe_json_parse(response.text)
    raw_rows = parsed if isinstance(parsed, list) else parsed.get("items", []) if isinstance(parsed, dict) else []
    out: dict[int, list[int]] = {}
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        movie_id = int(row.get("movie_id", 0) or 0)
        selected = row.get("selected_keyword_ids", [])
        if movie_id <= 0 or not isinstance(selected, list):
            continue
        cleaned = []
        seen = set()
        for value in selected:
            try:
                kid = int(value)
            except Exception:
                continue
            if kid > 0 and kid not in seen:
                seen.add(kid)
                cleaned.append(kid)
        if cleaned:
            out[movie_id] = cleaned
    return out


def _maybe_refine_keywords(
    world: WorldState,
    concept: dict[str, Any],
    keyword_ids: list[int],
    *,
    llm_model: str | None = None,
) -> list[int]:
    if not getattr(world, "enable_llm_keyword_rerank", False):
        return keyword_ids
    budget = int(getattr(world, "keyword_rerank_budget_remaining", 0) or 0)
    if budget <= 0:
        return keyword_ids
    confidence = float(concept.get("_keyword_confidence", 1.0) or 1.0)
    candidate_rows = list(concept.get("_keyword_candidates", []) or [])
    selection_summary = dict(concept.get("_keyword_selection_summary", {}) or {})
    selected_count = max(1, int(selection_summary.get("selected_count", len(keyword_ids) or 1)))
    generic_rate = float(selection_summary.get("generic_count", 0)) / float(selected_count)
    off_genre_rate = float(selection_summary.get("off_genre_count", 0)) / float(selected_count)
    genre_match_rate = float(selection_summary.get("genre_match_count", 0)) / float(selected_count)
    specificity_avg = float(selection_summary.get("specificity_avg", 0.0) or 0.0)
    franchise_core_missing = bool(selection_summary.get("franchise_core_missing", False))
    rerank_reason = None
    if generic_rate > 0.40:
        rerank_reason = "too_generic"
    elif off_genre_rate > 0.28:
        rerank_reason = "off_genre_bundle"
    elif genre_match_rate < 0.55:
        rerank_reason = "weak_genre_alignment"
    elif specificity_avg < 2.15 and generic_rate > 0.22:
        rerank_reason = "low_specificity"
    elif franchise_core_missing:
        rerank_reason = "missing_franchise_core"
    elif confidence < 0.03 and (off_genre_rate > 0.18 or generic_rate > 0.18 or specificity_avg < 2.35):
        rerank_reason = "low_confidence"
    if rerank_reason is None or len(candidate_rows) < 6:
        return keyword_ids

    item = {
        "movie_id": int(concept.get("movie_id", 0) or 0),
        "genre": str(concept.get("genre", "")),
        "tier": str(concept.get("tier", "")),
        "year": int(concept.get("year", 0) or 0),
        "country": str(concept.get("country", "")),
        "market": str(concept.get("market", "")),
        "franchise_id": int((concept.get("franchise") or {}).get("franchise_id", 0) or 0),
        "current_keyword_ids": [int(k) for k in keyword_ids],
        "layers": dict(concept.get("_keyword_layers", {}) or {}),
        "selection_summary": selection_summary,
        "rerank_reason": rerank_reason,
        "candidates": candidate_rows[:12],
    }
    try:
        picks = _rerank_keyword_batch(world, [item], llm_model)
    except Exception as exc:
        print(f"  Keyword rerank skipped for movie {item['movie_id']}: {exc}")
        return keyword_ids
    selected = picks.get(int(item["movie_id"]))
    if not selected:
        return keyword_ids
    allowed_ids = {int(row.get("keyword_id", 0) or 0) for row in candidate_rows}
    allowed_ids.update(int(k) for k in keyword_ids)
    selected = [int(k) for k in selected if int(k) in allowed_ids]
    if not selected:
        return keyword_ids
    if len(selected) > 8:
        selected = selected[:8]
    if len(selected) < 4:
        for row in candidate_rows:
            kid = int(row.get("keyword_id", 0) or 0)
            if kid > 0 and kid not in selected:
                selected.append(kid)
            if len(selected) >= 4:
                break
    world.keyword_rerank_budget_remaining = max(0, budget - 1)
    _append_decision_log(
        world,
        {
            "stage": "llm_keyword_rerank",
            "movie_id": int(item["movie_id"]),
            "previous_keyword_ids": [int(k) for k in keyword_ids],
            "chosen_keyword_ids": [int(k) for k in selected],
            "previous_confidence": confidence,
            "rerank_reason": rerank_reason,
            "selection_summary": selection_summary,
        },
    )
    return [int(k) for k in selected]


def _ensure_exact_topic_keyword_support(
    world: WorldState,
    concept: dict[str, Any],
    keyword_ids: list[int],
) -> list[int]:
    """Keep LLM keyword reranking from violating the structural JOB signal."""
    if not keyword_ids:
        return keyword_ids
    kw = getattr(world, "keywords", None)
    if kw is None or len(kw) == 0 or "keyword_id" not in kw.columns or "topic_genre" not in kw.columns:
        return keyword_ids

    genre = _canonical_genre_label(str(concept.get("genre", "")))
    if not genre:
        return keyword_ids

    if not hasattr(world, "_keyword_exact_topic_cache"):
        cache: dict[int, dict[str, Any]] = {}
        for row in kw.to_dict("records"):
            try:
                kid = int(row.get("keyword_id", 0) or 0)
            except Exception:
                continue
            if kid <= 0:
                continue
            cache[kid] = {
                "topic": _canonical_genre_label(str(row.get("topic_genre", ""))),
                "bucket": str(row.get("selection_bucket", "") or "").strip().lower(),
                "pop_weight": _safe_float(row.get("pop_weight", 1.0), 1.0),
            }
        world._keyword_exact_topic_cache = cache

    meta = getattr(world, "_keyword_exact_topic_cache", {})
    selected = [int(kid) for kid in keyword_ids if int(kid) in meta]
    if any(meta.get(kid, {}).get("topic") == genre for kid in selected):
        return keyword_ids

    usage_counts = getattr(world, "_keyword_usage_counts", {}) or {}
    selected_set = set(selected)
    candidates = [
        (
            1 if data.get("bucket") == "exact_anchor" else 0,
            float(data.get("pop_weight", 1.0)) / (1.0 + 0.05 * float(usage_counts.get(kid, 0) or 0)),
            -int(usage_counts.get(kid, 0) or 0),
            int(kid),
        )
        for kid, data in meta.items()
        if data.get("topic") == genre
        and data.get("bucket") != "generic"
        and int(kid) not in selected_set
    ]
    if not candidates:
        return keyword_ids

    replacement_kid = int(max(candidates)[3])
    replace_pos = 0
    for pos, kid in enumerate(selected):
        data = meta.get(kid, {})
        if data.get("topic") != genre and data.get("bucket") in {"generic", ""}:
            replace_pos = pos
            break
    else:
        for pos, kid in enumerate(selected):
            if meta.get(kid, {}).get("topic") != genre:
                replace_pos = pos
                break

    repaired = list(keyword_ids)
    repaired[replace_pos] = replacement_kid
    _append_decision_log(
        world,
        {
            "stage": "keyword_exact_topic_repair",
            "movie_id": int(concept.get("movie_id", 0) or 0),
            "genre": genre,
            "previous_keyword_ids": [int(k) for k in keyword_ids],
            "chosen_keyword_ids": [int(k) for k in repaired],
            "inserted_keyword_id": int(replacement_kid),
        },
    )
    return [int(k) for k in repaired]


def _record_pipeline_memory_snapshot(
    audit,
    phase: str,
    world: WorldState,
    *,
    accum: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    note: str = "",
    sample_kind: str = "checkpoint",
    extra_specs: list[tuple] | None = None,
) -> None:
    if audit is None:
        return
    try:
        from memory_probe import audit_target, graph_audit_targets
    except Exception:
        return

    specs = graph_audit_targets(getattr(world, "graph", getattr(world, "edge_graph", None))) + [
        audit_target("person_latent", getattr(world, "person_latent", None), "latent_state", "Latent dictionary backing person-level reasoning", "latent"),
        audit_target("company_latent", getattr(world, "company_latent", None), "latent_state", "Latent dictionary backing company-level reasoning", "latent"),
        audit_target("person_sim_cache", getattr(world, "_person_sim_cache", None), "latent_cache", "Similarity cache derived from person latents and tags", "latent"),
        audit_target("latent_csv_normed", getattr(world, "_latent_csv_normed", None), "latent_dense", "Dense style matrix", "latent"),
        audit_target("latent_bbp_normed", getattr(world, "_latent_bbp_normed", None), "latent_dense", "Dense budget preference matrix", "latent"),
        audit_target("latent_genre_bits", getattr(world, "_latent_genre_bits", None), "latent_dense", "Dense genre bitset array", "latent"),
        audit_target("latent_style_bits", getattr(world, "_latent_style_bits", None), "latent_dense", "Dense style bitset array", "latent"),
        audit_target("latent_risk", getattr(world, "_latent_risk", None), "latent_dense", "Dense risk vector", "latent"),
        audit_target("latent_ambition", getattr(world, "_latent_ambition", None), "latent_dense", "Dense ambition vector", "latent"),
        audit_target("latent_collab_code", getattr(world, "_latent_collab_code", None), "latent_dense", "Dense collaboration style vector", "latent"),
        audit_target("crew_pools", getattr(world, "crew_pools", None), "role_views", "Crew DataFrame copies derived from persons", "persons_role_views"),
        audit_target("actors", getattr(world, "actors", None), "role_views", "Actor role DataFrame copy", "persons_role_views"),
        audit_target("directors", getattr(world, "directors", None), "role_views", "Director role DataFrame copy", "persons_role_views"),
        audit_target("year_cache", getattr(world, "_year_cache", None), "year_cache", "Actor year-mask cache", "year_cache"),
        audit_target("director_recent_outcomes", getattr(world, "director_recent_outcomes", None), "runtime_state", "Director momentum state", "runtime"),
        audit_target("company_recent_outcomes", getattr(world, "company_recent_outcomes", None), "runtime_state", "Company momentum state", "runtime"),
    ]
    if accum is not None:
        if "company_links" in accum:
            specs.append(audit_target("accum_company_links", accum.get("company_links"), "global_tables", "Company links global table held in memory", "global_tables"))
        if "person_demographics" in accum:
            specs.append(audit_target("accum_person_demographics", accum.get("person_demographics"), "global_tables", "Person demographics global table held in memory", "global_tables"))
        if "tv_series" in accum:
            specs.append(audit_target("accum_tv_series", accum.get("tv_series"), "global_tables", "TV series global table held in memory", "global_tables"))
        if "seasons" in accum:
            specs.append(audit_target("accum_seasons", accum.get("seasons"), "global_tables", "TV seasons global table held in memory", "global_tables"))
        if "episodes" in accum:
            specs.append(audit_target("accum_episodes", accum.get("episodes"), "global_tables", "Episodes global table held in memory", "global_tables"))
        if "episode_cast" in accum:
            specs.append(audit_target("accum_episode_cast", accum.get("episode_cast"), "global_tables", "Episode cast rows held in memory", "global_tables"))
    for spec in extra_specs or []:
        if len(spec) < 3:
            continue
        spec_name = spec[0]
        spec_obj = spec[1]
        spec_category = spec[2]
        spec_note = spec[3] if len(spec) > 3 else ""
        overlap_group = spec[4] if len(spec) > 4 else ""
        specs.append(audit_target(spec_name, spec_obj, spec_category, spec_note, overlap_group))
    audit.record_snapshot(
        phase,
        specs,
        note=note,
        metadata=metadata,
        sample_kind=sample_kind,
    )


# -----
# MAIN ASSEMBLY LOOP
# -

def assemble_movies(world: WorldState, n_movies: int = 5000,
                    enable_llm_evolution: bool = False,
                    llm_model: str = None,
                    evolution_log_dir: str = None,
                    checkpoint_dir: str = None,
                    speed_audit = None,
                    resume_manager: Step100ResumeManager | None = None,
                    extend_step100: bool = False,
                    extension_start_year: int | None = None,
                    extension_end_year: int | None = None) -> dict:
    """Assemble N movies with temporal evolution.

    Movies are pre-sampled by year and assembled chronologically.
    At each year boundary, procedural_year_step() nudges the world state
    (reputation, edges). If enable_llm_evolution is True, llm_year_step()
    is also called for richer graph mutations.
    """
    if resume_manager is None:
        resume_manager = Step100ResumeManager(world.base_dir)
    global_tables: dict[str, pd.DataFrame] = {}
    year_buffers: dict[str, list[dict[str, Any]]] = {name: [] for name in PER_MOVIE_TABLES}
    spool_rows_limit = max(0, int(os.getenv("DATA_SYS_STEP100_SPOOL_ROWS", "0") or 0))
    spool_buffered_rows = 0
    memory_audit = _get_memory_audit_recorder()
    if speed_audit is None:
        speed_audit = _get_speed_audit_recorder(str(world.base_dir), default_experiment="pipeline-speed")

    base_dir = str(world.base_dir)
    yearly_snapshots_enabled = _yearly_snapshots_enabled(world)

    def _maybe_spool_year_buffers(reason: str) -> None:
        nonlocal spool_buffered_rows
        if spool_rows_limit <= 0 or current_year is None or spool_buffered_rows < spool_rows_limit:
            return
        written = resume_manager.spool_year_tables(
            year=int(current_year),
            year_tables=year_buffers,
        )
        if written > 0:
            print(
                f"  [Spool] Year {current_year}: wrote {written:,} buffered rows to disk "
                f"({reason})",
                flush=True,
            )
        spool_buffered_rows = 0
        gc.collect()

    def _trim_generation_caches_after_year(completed_year: int) -> None:
        for attr in ("_selection_year_state_cache", "_cast_year_cache", "_crew_year_pool_cache"):
            cache = getattr(world, attr, None)
            if not isinstance(cache, dict):
                continue
            for key in list(cache.keys()):
                try:
                    cache_year = int(key[0] if isinstance(key, tuple) else key)
                except Exception:
                    continue
                if cache_year <= int(completed_year):
                    cache.pop(key, None)

    def write_rows(table_name: str, rows: list[dict]) -> None:
        nonlocal spool_buffered_rows
        if not rows:
            return
        if table_name in GLOBAL_TABLES:
            current_df = global_tables.get(table_name)
            incoming_df = pd.DataFrame(rows)
            global_tables[table_name] = incoming_df if current_df is None else pd.concat([current_df, incoming_df], ignore_index=True)
            return
        year_buffers.setdefault(table_name, []).extend(rows)
        spool_buffered_rows += int(len(rows))
        _maybe_spool_year_buffers(table_name)

    def write_row(table_name: str, row: dict) -> None:
        write_rows(table_name, [row])

    # Internal state for movie_links generator
    previous_movies_for_links: list[dict[str, Any]] = []

    def _keyword_id_by_text(term: str) -> int | None:
        cache = getattr(world, "_keyword_id_by_text_cache", None)
        if cache is None:
            cache = {}
            kw = getattr(world, "keywords", None)
            if kw is not None and len(kw) > 0 and "keyword" in kw.columns:
                kid_col = kw["keyword_id"].astype(int) if "keyword_id" in kw.columns else pd.Series(np.arange(1, len(kw) + 1, dtype=int))
                for kid, text in zip(kid_col, kw["keyword"].fillna("").astype(str)):
                    norm = text.strip().lower()
                    if norm and norm not in cache:
                        cache[norm] = int(kid)
            world._keyword_id_by_text_cache = cache
        return cache.get(str(term).strip().lower())

    def _ensure_structural_keyword_anchors(keyword_ids: list[int], concept: dict, installment: int | None) -> list[int]:
        """Add semantically justified rare anchors that JOB-style joins rely on.

        These are not query-answer rows; they make the existing movie semantics
        visible through exact IMDb keywords for sequels, animation, and violent
        genre motifs. That improves benchmark coverage while preserving the
        generator's topic/correlation structure.
        """
        chosen = [int(kid) for kid in keyword_ids]
        chosen_set = set(chosen)
        genre = str(concept.get("genre", ""))
        tier = str(concept.get("tier", "Mid"))
        year_i = int(concept.get("year", 0) or 0)
        h = ((int(concept.get("movie_id", 0) or 0) + 23) * 2654435761) & 0xFFFFFFFF
        terms: list[str] = []
        if installment and int(installment) > 1:
            terms.append("sequel")
        if genre == "Animation":
            terms.append("computer-animation")
            if tier in {"Epic", "A", "Mid"} and year_i >= 1995:
                terms.append("computer-animated-movie")
        if genre in {"Horror", "Thriller", "Crime"}:
            pool = ["murder", "violence", "blood", "death", "gore", "hospital"]
            start = h % len(pool)
            terms.extend(pool[start:start + 2] if start + 2 <= len(pool) else pool[start:] + pool[:(start + 2) % len(pool)])
            if genre == "Horror" and (h & 3) == 0:
                terms.append("female-nudity")
        if genre == "Action":
            action_pool = ["hero", "martial-arts", "hand-to-hand-combat", "violence"]
            if (h % 100) < 35:
                terms.extend(action_pool[:2])
            elif (h % 100) < 55:
                terms.append(action_pool[2])
        if genre in {"Adventure", "Fantasy"} and (h % 100) < 18:
            terms.append("hero")

        added = 0
        for term in terms:
            kid = _keyword_id_by_text(term)
            if kid is None or kid in chosen_set:
                continue
            chosen.append(int(kid))
            chosen_set.add(int(kid))
            added += 1
            if hasattr(world, "_keyword_usage_counts"):
                world._keyword_usage_counts[int(kid)] += 1
            if added >= 3:
                break
        return chosen

    # Runtime profiles are explicit: by default the benchmark-candidate path
    # keeps rich tables enabled, but skipped tables still need stable schemas.
    runtime_cfg = getattr(getattr(getattr(world, "workspace", None), "config", None), "runtime", None)
    disabled_secondary_tables = {
        str(name).strip()
        for name in (getattr(runtime_cfg, "disabled_secondary_tables", None) or [])
        if str(name).strip()
    }
    disabled_global_tables = {
        str(name).strip()
        for name in (getattr(runtime_cfg, "disabled_global_tables", None) or [])
        if str(name).strip()
    }
    disabled_post_loop_tables = {
        str(name).strip()
        for name in (getattr(runtime_cfg, "disabled_post_loop_tables", None) or [])
        if str(name).strip()
    }

    if resume_manager.is_resuming:
        global_tables = resume_manager.load_globals()
        _apply_company_links_to_world(world, global_tables.get("company_links"))
        if extend_step100 and "person_demographics" not in disabled_global_tables:
            existing_demo = global_tables.get("person_demographics")
            if not isinstance(existing_demo, pd.DataFrame):
                existing_demo = pd.DataFrame()
            persons_df = world.persons if isinstance(getattr(world, "persons", None), pd.DataFrame) else pd.DataFrame()
            if not persons_df.empty and "person_id" in persons_df.columns:
                existing_ids: set[int] = set()
                if not existing_demo.empty and "person_id" in existing_demo.columns:
                    existing_ids = {
                        int(value)
                        for value in pd.to_numeric(existing_demo["person_id"], errors="coerce").dropna().astype(int).tolist()
                    }
                person_ids = pd.to_numeric(persons_df["person_id"], errors="coerce")
                missing_persons = persons_df.loc[~person_ids.isin(existing_ids)].copy()
                if not missing_persons.empty:
                    added_demo = pd.DataFrame(generate_person_demographics(missing_persons, world.rng))
                    for col in existing_demo.columns:
                        if col not in added_demo.columns:
                            added_demo[col] = None
                    for col in added_demo.columns:
                        if col not in existing_demo.columns:
                            existing_demo[col] = None
                    ordered_cols = list(existing_demo.columns) if len(existing_demo.columns) else list(added_demo.columns)
                    global_tables["person_demographics"] = pd.concat(
                        [existing_demo[ordered_cols], added_demo[ordered_cols]],
                        ignore_index=True,
                    )
                    resume_manager.save_globals(global_tables)
                    print(
                        f"  [Continuation] Added {len(added_demo):,} person_demographics rows "
                        f"for topped-up persons",
                        flush=True,
                    )
    else:
        # Global tables (generated once, not per-movie)
        print("  Global setup: company_links...", flush=True)
        with _speed_scope(speed_audit, "global.company_links", category="global_tables") as _sp:
            if "company_links" in disabled_global_tables:
                company_links = []
                global_tables["company_links"] = _empty_table_frame("company_links")
            else:
                company_links = generate_company_links(
                    world.companies if hasattr(world, 'companies') else None,
                    world.rng
                )
                global_tables["company_links"] = pd.DataFrame(company_links)
            if _sp is not None:
                _sp.units = len(company_links)
        print(f"  Global setup: company_links done ({len(company_links)} rows)", flush=True)
        _apply_company_links_to_world(world, global_tables.get("company_links"))

        print("  Global setup: person_demographics...", flush=True)
        with _speed_scope(speed_audit, "global.person_demographics", category="global_tables") as _sp:
            if "person_demographics" in disabled_global_tables:
                person_demographics = []
                global_tables["person_demographics"] = _empty_table_frame("person_demographics")
            else:
                person_demographics = generate_person_demographics(world.persons, world.rng)
                global_tables["person_demographics"] = pd.DataFrame(person_demographics)
            if _sp is not None:
                _sp.units = len(person_demographics)
        print(f"  Global setup: person_demographics done ({len(person_demographics)} rows)", flush=True)

        # TV series hierarchy (series -> seasons -> episodes)
        tv_table_names = {"tv_series", "seasons", "episodes"}
        print("  Global setup: tv_series_bundle...", flush=True)
        with _speed_scope(speed_audit, "global.tv_series_bundle", category="global_tables") as _sp:
            if tv_table_names.issubset(disabled_global_tables):
                tv_data = {"tv_series": [], "seasons": [], "episodes": []}
            else:
                tv_data = generate_tv_series(
                    world.persons, world.companies, world.rng,
                    n_series=_scaled_tv_series_target(n_movies, world)
                )
            if _sp is not None:
                _sp.units = len(tv_data.get("tv_series", []))
        print(
            f"  Global setup: tv_series_bundle done "
            f"({len(tv_data.get('tv_series', []))} series, {len(tv_data.get('episodes', []))} episodes)",
            flush=True,
        )
        for _table_name in ("tv_series", "seasons", "episodes"):
            if _table_name in disabled_global_tables:
                global_tables[_table_name] = _empty_table_frame(_table_name)
            else:
                global_tables[_table_name] = pd.DataFrame(tv_data[_table_name])
        print("  Global setup: episode_cast...", flush=True)
        with _speed_scope(speed_audit, "global.episode_cast", category="global_tables") as _sp:
            if "episode_cast" in disabled_global_tables or "episodes" in disabled_global_tables:
                episode_cast = []
                global_tables["episode_cast"] = _empty_table_frame("episode_cast")
            else:
                episode_cast = generate_episode_cast(tv_data, world.actors, world.rng)
                global_tables["episode_cast"] = pd.DataFrame(episode_cast)
            if _sp is not None:
                _sp.units = len(episode_cast)
        print(f"  Global setup: episode_cast done ({len(episode_cast)} rows)", flush=True)
        resume_manager.save_globals(global_tables)

    print(f"  TV series: {len(global_tables.get('tv_series', pd.DataFrame()))} series, "
          f"{len(global_tables.get('seasons', pd.DataFrame()))} seasons, "
          f"{len(global_tables.get('episodes', pd.DataFrame()))} episodes, "
          f"{len(global_tables.get('episode_cast', pd.DataFrame()))} episode cast rows", flush=True)
    _record_pipeline_memory_snapshot(
        memory_audit,
        "pipeline_global_tables_ready",
        world,
        accum=global_tables,
        metadata={
            "n_movies_requested": int(n_movies),
            "tv_series_rows": len(global_tables.get("tv_series", pd.DataFrame())),
            "episode_rows": len(global_tables.get("episodes", pd.DataFrame())),
            "episode_cast_rows": len(global_tables.get("episode_cast", pd.DataFrame())),
        },
        note="Snapshot after one-time global tables are materialized in memory.",
    )

    # NOTE: user_ratings is generated AFTER the movie loop (needs movie rows)
    # See post-processing section below

    # Build secondary generator registry. Runtime profiles can suppress
    # benchmark-irrelevant fan-out tables (for example large review text).
    with _speed_scope(speed_audit, "global.build_secondary_generators", category="setup"):
        secondary_generators = build_secondary_generators(disabled_tables=disabled_secondary_tables)
    if disabled_global_tables:
        print(
            "  Runtime profile disabled global tables: "
            + ", ".join(sorted(disabled_global_tables)),
            flush=True,
        )
    if disabled_secondary_tables:
        print(
            "  Runtime profile disabled secondary tables: "
            + ", ".join(sorted(disabled_secondary_tables)),
            flush=True,
        )
    if disabled_post_loop_tables:
        print(
            "  Runtime profile disabled post-loop tables: "
            + ", ".join(sorted(disabled_post_loop_tables)),
            flush=True,
        )
    # v16: no hard cap by title bank size.
    # Curated titles are used first; overflow titles are generated compositionally.

    progress_log_path = _movie_progress_log_path(base_dir)
    progress_interval = 10 if n_movies <= 200 else 50 if n_movies <= 5000 else 500
    progress_verbose = bool(n_movies <= 300)
    print(f"\nAssembling {n_movies} movies (temporal mode)...", flush=True)
    _log_movie_progress(
        base_dir,
        {
            "event": "step100_start",
            "n_movies": int(n_movies),
            "progress_interval": int(progress_interval),
            "llm_model": str(llm_model or ""),
            "enable_llm_critic": bool(enable_llm_critic),
            "enable_llm_evolution": bool(enable_llm_evolution),
            "progress_log_path": progress_log_path,
            "timestamp": time.time(),
        },
    )

    if resume_manager.is_resuming:
        planned_resume_start_index = int(resume_manager.start_index())
        restored_state = resume_manager.restore_checkpoint(
            world,
            restore_graph=False,
            merge_current_entities=bool(extend_step100),
        )
        previous_movies_for_links = list(restored_state.get("previous_movies_for_links", []) or [])
        evo_stats = dict(restored_state.get("evo_stats", {}) or {})
        demand_pool = dict(restored_state.get("demand_pool", {}) or {})
        if extend_step100:
            lifecycle_counts = apply_lifecycle_to_world(
                world,
                world.base_dir,
                extension_start_year=extension_start_year,
                extension_end_year=extension_end_year,
            )
            if lifecycle_counts.get("applied_person_updates") or lifecycle_counts.get("applied_company_updates"):
                print(
                    "  [Continuation] Replayed lifecycle after checkpoint restore: "
                    f"persons={lifecycle_counts.get('applied_person_updates', 0):,}, "
                    f"companies={lifecycle_counts.get('applied_company_updates', 0):,}",
                    flush=True,
                )
        plan_records = resume_manager.load_plan(world)
        if planned_resume_start_index < len(plan_records) or extend_step100:
            resume_manager.restore_graph(world, preserve_current_graph=bool(extend_step100))
        if extend_step100 and len(plan_records) < int(n_movies):
            ext_start = int(extension_start_year) if extension_start_year is not None else int(resume_manager.manifest.get("last_completed_year", 0) or 0) + 1
            ext_end = int(extension_end_year) if extension_end_year is not None else ext_start
            extension_result = append_extension_plan(
                world=world,
                resume_manager=resume_manager,
                existing_plan=plan_records,
                target_movie_count=int(n_movies),
                extension_start_year=ext_start,
                extension_end_year=ext_end,
                sample_movie_concept_fn=sample_movie_concept,
                rerank_fn=_apply_llm_concept_rerank,
                llm_model=llm_model,
            )
            print(
                "  [Continuation] Appended "
                f"{extension_result.appended:,} movies to plan "
                f"(seq {extension_result.first_seq_idx}-{extension_result.last_seq_idx}, "
                f"movie_id {extension_result.first_movie_id}-{extension_result.last_movie_id}, "
                f"years {extension_result.year_min}-{extension_result.year_max}, "
                f"title_bank={extension_result.title_assignments:,}, "
                f"compositional={extension_result.compositional_title_movies:,}, "
                f"reranked={extension_result.reranked:,})",
                flush=True,
            )
            plan_records = resume_manager.load_plan(world)
            if int(resume_manager.start_index()) >= len(plan_records):
                raise RuntimeError("Continuation plan append did not create any work to resume.")
        resume_manager.mark_running()
        reranked = 0
    else:
        # === Pre-assign titles and use their LLM-generated years ==========
        # Each title in the bank has a year assigned by the LLM.
        # We shuffle the bank, assign one title per movie, and use that
        # title's year so every movie keeps its LLM-assigned time period.
        title_assignments = {}  # movie_id -> {title, tagline, year, genre_hint}
        with _speed_scope(speed_audit, "setup.title_assignment", category="setup") as _sp:
            if len(world.title_bank) > 0 and 'year' in world.title_bank.columns:
                tb = world.title_bank.sample(frac=1, random_state=world.rng.randint(0, 2**31)).reset_index(drop=True)
                assigned_titles: set[str] = set()
                mid = 1
                for i in range(len(tb)):
                    if mid > int(n_movies):
                        break
                    row = tb.iloc[i]
                    clean_title = str(row.get("_title_clean", row.get("title", "")) or "")
                    if not clean_title or clean_title in assigned_titles:
                        continue
                    assigned_titles.add(clean_title)
                    clean_tagline = str(row.get("_tagline_clean", row.get("tagline", "")) or "")
                    title_assignments[mid] = {
                        "title": clean_title,
                        "tagline": clean_tagline,
                        "year": int(row["year"]) if pd.notna(row.get("year")) else 2020,
                        "genre_hint": str(row.get("genre_hint", "")) if pd.notna(row.get("genre_hint")) else "",
                        # D23: propagate award_contender flag from title bank
                        "award_contender": bool(row.get("award_contender", False))
                            if not (isinstance(row.get("award_contender"), float)
                                    and row.get("award_contender") != row.get("award_contender")) else False,
                        "_tb_row_idx": int(row.get("_tb_row_idx", i)),
                        "_title_ok_static": bool(row.get("_title_ok_static", bool(clean_title))),
                        "_tagline_ok_static": bool(row.get("_tagline_ok_static", bool(clean_tagline))),
                    }
                    mid += 1
                if _sp is not None:
                    _sp.units = len(title_assignments)
                print(f"  Pre-assigned {len(title_assignments)} titles from bank (years {tb['year'].min()}-{tb['year'].max()})", flush=True)
            else:
                print(f"  WARNING: No title bank years found, falling back to YEAR_RANGE", flush=True)

        # === D5 (v13): Fix franchise chronological ordering ==================
        # _setup_franchises() assigns movie IDs to franchise slots randomly (no
        # knowledge of years). Now that title_assignments gives us each movie's
        # year, we can sort each franchise's movie IDs by year so installment
        # numbers match chronological release order.
        if world.movie_franchise_map:
            _franchise_scope = _speed_scope(speed_audit, "setup.franchise_reorder", category="setup")
        else:
            _franchise_scope = nullcontext(None)
        with _franchise_scope as _sp:
            if not world.movie_franchise_map:
                pass
            else:
                _all_years = [ta["year"] for ta in title_assignments.values()]
                max_year = max(_all_years) if _all_years else 2025

                franchise_movie_ids: dict = {}
                for mid, franchise in world.movie_franchise_map.items():
                    fid = franchise["franchise_id"]
                    franchise_movie_ids.setdefault(fid, []).append(mid)

                new_franchise_map = {}
                for fid, mids in franchise_movie_ids.items():
                    franchise = world.movie_franchise_map[mids[0]]
                    mids_sorted = sorted(
                        mids,
                        key=lambda m: title_assignments[m]["year"] if m in title_assignments else 9999
                    )
                    for rank, m in enumerate(mids_sorted, start=1):
                        new_franchise_map[m] = franchise
                    if len(mids_sorted) > 1:
                        prev_year = (title_assignments[mids_sorted[0]]["year"]
                                     if mids_sorted[0] in title_assignments else None)
                        for m in mids_sorted[1:]:
                            if m in title_assignments and prev_year is not None:
                                y = title_assignments[m]["year"]
                                if y <= prev_year:
                                    title_assignments[m] = dict(title_assignments[m])
                                    title_assignments[m]["year"] = min(prev_year + 2, max_year)
                                prev_year = title_assignments[m]["year"]
                                if prev_year >= max_year:
                                    prev_year = max_year

                world.movie_franchise_map = new_franchise_map
                n_fmovies = len(new_franchise_map)
                if _sp is not None:
                    _sp.units = n_fmovies
                print(f"  D5: Franchise chronological ordering applied ({n_fmovies} franchise movie slots, max_year={max_year})", flush=True)

        # === Pre-sample concepts and sort chronologically ====================
        year_list = []
        with _speed_scope(speed_audit, "setup.sample_movie_concepts", category="setup", units=n_movies) as _sp:
            for mid in range(1, n_movies + 1):
                forced_year = title_assignments[mid]["year"] if mid in title_assignments else None
                concept = sample_movie_concept(world, mid, forced_year=forced_year, title_assignment=title_assignments.get(mid))
                year_list.append((concept["year"], mid, concept))
            reranked = _apply_llm_concept_rerank(world, year_list, llm_model=llm_model)
            year_list.sort(key=lambda x: (x[0], x[1]))
            if _sp is not None:
                _sp.units = len(year_list)
        if getattr(world, "enable_llm_rerank", False):
            print(f"  LLM concept rerank applied to {reranked} low-confidence movies", flush=True)
        plan_records = [
            {
                "seq_idx": int(seq_idx),
                "movie_id": int(mid),
                "year": int(year),
                "concept": concept,
                "title_assignment": dict(title_assignments.get(mid, {}) or {}),
            }
            for seq_idx, (year, mid, concept) in enumerate(year_list)
        ]
        resume_manager.save_plan(
            plan_records,
            year_min=int(year_list[0][0]) if year_list else None,
            year_max=int(year_list[-1][0]) if year_list else None,
        )
        resume_manager.mark_running()
        demand_pool = {}
        evo_stats = {"year_program_ops": 0, "years_evolved": 0, "triggered_events": 0}

    if not plan_records:
        raise RuntimeError("Step 100 movie plan is empty")

    print(f"  Year range: {plan_records[0]['year']}-{plan_records[-1]['year']}", flush=True)
    _log_movie_progress(
        base_dir,
        {
            "event": "step100_titles_ready",
            "n_movies": int(n_movies),
            "title_assignments": int(sum(1 for row in plan_records if row.get("title_assignment"))),
            "year_min": int(plan_records[0]["year"]) if plan_records else None,
            "year_max": int(plan_records[-1]["year"]) if plan_records else None,
            "resume_mode": bool(resume_manager.is_resuming),
            "resume_start_index": int(resume_manager.start_index()),
            "timestamp": time.time(),
        },
    )

    resume_start_index = int(resume_manager.start_index())
    current_year = None
    year_bucket = []  # movies produced in current_year (for evolution)
    plateau_recorded = False

    def evolve_completed_year(completed_year: int, bucket: list[dict[str, Any]]) -> None:
        if not bucket:
            return
        if enable_llm_evolution:
            print(f"  [Year {completed_year}->{completed_year+1}] Running unified yearly planner...", flush=True)
        rep = evolve_year(
            world,
            from_year=completed_year,
            to_year=completed_year + 1,
            year_bucket=bucket,
            enable_llm=enable_llm_evolution,
            model=llm_model,
            log_dir=evolution_log_dir,
            speed_audit=speed_audit,
        )
        evo_stats["year_program_ops"] += rep.applied
        evo_stats["years_evolved"] += 1
        evo_stats["triggered_events"] += len(rep.triggered_events)
        print(
            f"  [Year {completed_year}->{completed_year+1}] Planner({rep.planner_source}): "
            f"{rep.applied} ops, {rep.skipped} skipped, {rep.errors} errors, "
            f"events={rep.triggered_events or []}",
            flush=True,
        )
        if rep.messages:
            for log_line in rep.messages[:3]:
                print(f"    Planner log: {log_line}", flush=True)
        graph = getattr(world, "graph", None)
        if graph is not None and hasattr(graph, "flush_year"):
            with _speed_scope(
                speed_audit,
                "year.graph_flush",
                category="year_boundary",
                metadata={"year": int(completed_year)},
            ):
                graph.flush_year(int(completed_year))

    # Market competition demand carries across committed-year resumes.
    DEPLETION_BY_TIER = {"Epic": 0.25, "A": 0.15, "Mid": 0.08, "Indie": 0.04, "Micro": 0.02}

    for record in plan_records[resume_start_index:]:
        seq_idx = int(record.get("seq_idx", 0))
        year = int(record.get("year", 0))
        mid = int(record.get("movie_id", 0))
        concept = dict(record.get("concept", {}) or {})
        title_assignment = dict(record.get("title_assignment", {}) or {})
        movie_started_at = time.perf_counter()
        def _stage_mark(stage_name: str, *, status: str = "done", extra: dict[str, Any] | None = None) -> None:
            payload = {
                "event": "movie_stage",
                "seq_idx": int(seq_idx + 1),
                "movie_id": int(mid),
                "year": int(year),
                "genre": str(concept.get("genre", "")),
                "tier": str(concept.get("tier", "")),
                "country": str(concept.get("country", "")),
                "stage": str(stage_name),
                "status": str(status),
                "elapsed_sec": round(float(time.perf_counter() - movie_started_at), 4),
                "timestamp": time.time(),
            }
            if extra:
                payload.update(extra)
            _log_movie_progress(base_dir, payload)
        # === Year boundary: evolve world state ========================
        if current_year is not None and year != current_year:
            # Run evolution for the completed year
            if year_bucket:
                evolve_completed_year(int(current_year), year_bucket)

            if memory_audit is not None:
                process_row = memory_audit.record_process(
                    f"pipeline_year_{current_year}_sample",
                    sample_kind="process_sample",
                    note="Year-boundary process envelope sample after temporal evolution.",
                    metadata={
                        "year_completed": int(current_year),
                        "years_evolved": int(evo_stats["years_evolved"]),
                        "movie_rows_buffered": int(seq_idx),
                    },
                )
                milestone = evo_stats["years_evolved"] == 1 or evo_stats["years_evolved"] % 5 == 0
                if milestone:
                    _record_pipeline_memory_snapshot(
                        memory_audit,
                        f"pipeline_year_{current_year}_checkpoint",
                        world,
                        accum=global_tables,
                        metadata={
                            "year_completed": int(current_year),
                            "years_evolved": int(evo_stats["years_evolved"]),
                            "movie_rows_buffered": int(seq_idx),
                        },
                        note="Pipeline milestone snapshot after temporal evolution.",
                    )
                if process_row.get("plateau_hit") and not plateau_recorded:
                    plateau_recorded = True
                    _record_pipeline_memory_snapshot(
                        memory_audit,
                        f"pipeline_year_{current_year}_plateau",
                        world,
                        accum=global_tables,
                        metadata={
                            "year_completed": int(current_year),
                            "years_evolved": int(evo_stats["years_evolved"]),
                        },
                        note="Plateau-triggered snapshot after six consecutive process samples stayed within a 5% private-memory band.",
                        sample_kind="plateau",
                    )

            try:
                resume_manager.commit_year(
                    year=int(current_year),
                    seq_idx=int(seq_idx - 1),
                    year_tables=year_buffers,
                    world=world,
                    demand_pool=demand_pool,
                    previous_movies_for_links=previous_movies_for_links,
                    evo_stats=evo_stats,
                )
                print(f"  [Checkpoint] {seq_idx:,} movies saved after year {current_year}", flush=True)
            except Exception as _ce:
                print(f"  [Checkpoint] Warning: {_ce}", flush=True)

            # === TEMPORAL SNAPSHOT: year-level CSV exports for time-series queries ==
            # Writes 3 lightweight CSVs per year to snapshots/{year}/:
            #   edges_active.csv   - all edges with valid_from<=year and valid_to>=year (or null)
            #   persons_state.csv  - person_id, pop_weight, career_stage at this year
            #   companies_state.csv - company_id, tier, pop_weight at this year
            # These allow SQL queries like:
            #   "Show me the friendship graph in 1998"
            #   "How did career stages shift from 1990 to 2010?"
            #   "Which companies changed tier between 1995 and 2005?"
            _final_year = int(plan_records[-1]["year"]) if plan_records else 2025
            _is_snapshot_year = (current_year % 10 == 0) or (current_year == _final_year)
            if checkpoint_dir and yearly_snapshots_enabled and _is_snapshot_year:
                try:
                    _snap_dir = os.path.join(checkpoint_dir, "snapshots", str(current_year))
                    os.makedirs(_snap_dir, exist_ok=True)

                    # 1. Active edges at this year
                    graph = getattr(world, "graph", None)
                    if graph is not None:
                        _n_edges = graph.materialize_legacy_csv(
                            os.path.join(_snap_dir, "edges_active.csv"),
                            current_year,
                        )
                        _n_all = 0
                    elif getattr(world, "edge_graph", None) is not None:
                        _active_edges = []
                        _all_edges = []
                        for _e in world.edge_graph.edges:
                            try:
                                _vf = _e.get("valid_from")
                                _vt = _e.get("valid_to")
                                _edge_row = {
                                    "src_id": _e.get("src_id"),
                                    "dst_id": _e.get("dst_id"),
                                    "edge_type": _e.get("edge_type"),
                                    "sign": _e.get("sign"),
                                    "weight": round(float(_e.get("weight", 0.0) or 0.0), 4),
                                    "source_kind": _e.get("source_kind", ""),
                                    "valid_from": _vf,
                                    "valid_to": _vt,
                                }
                                # V17: ALL edges go into edges_all.csv (full temporal history)
                                _all_edges.append(_edge_row)
                                # Active-only filter for edges_active.csv
                                _vf_ok = (_vf is None or int(_vf) <= current_year)
                                _vt_ok = (_vt is None or int(_vt) >= current_year)
                                if _vf_ok and _vt_ok:
                                    _active_edges.append(_edge_row)
                            except Exception:
                                continue
                        if _active_edges:
                            pd.DataFrame(_active_edges).to_csv(
                                os.path.join(_snap_dir, "edges_active.csv"), index=False
                            )
                        # V17: Save complete edge history (active + expired) for temporal studies.
                        # This preserves edges that existed in the past but have since expired,
                        # enabling queries like "show all friendships that ended before 2005".
                        if _all_edges:
                            pd.DataFrame(_all_edges).to_csv(
                                os.path.join(_snap_dir, "edges_all.csv"), index=False
                            )

                    # 2. Person state
                    if world.persons is not None:
                        _cols = ["person_id", "name", "career_stage", "pop_weight",
                                 "debut_year", "retirement_year", "peak_start", "peak_end"]
                        _pcols = [c for c in _cols if c in world.persons.columns]
                        world.persons[_pcols].to_csv(
                            os.path.join(_snap_dir, "persons_state.csv"), index=False
                        )

                    # 3. Company state
                    if world.companies is not None:
                        _ccols_want = ["company_id", "name", "tier", "pop_weight",
                                       "founded_year", "defunct_year"]
                        _ccols = [c for c in _ccols_want if c in world.companies.columns]
                        world.companies[_ccols].to_csv(
                            os.path.join(_snap_dir, "companies_state.csv"), index=False
                        )

                    _n_edges = _n_edges if "_n_edges" in locals() else (len(_active_edges) if "_active_edges" in locals() else 0)
                    _n_all = _n_all if "_n_all" in locals() else (len(_all_edges) if "_all_edges" in locals() else 0)
                    print(f"  [Snapshot] Year {current_year}: "
                          f"{_n_edges} active / {_n_all} total edges / "
                          f"{len(world.persons) if world.persons is not None else 0} persons / "
                          f"{len(world.companies) if world.companies is not None else 0} companies", flush=True)
                except Exception as _se:
                    print(f"  [Snapshot] Warning: {_se}", flush=True)

            year_bucket = []
            year_buffers = {name: [] for name in PER_MOVIE_TABLES}
            spool_buffered_rows = 0
            _trim_generation_caches_after_year(int(current_year))
            gc.collect()
            # G4-FIX: Carry 50% of demand depletion into the next year.
            # Real market saturation is sticky -- if Horror was saturated in 2019,
            # studios reduce Horror output in 2020 too. A hard reset was unrealistic.
            demand_pool = {k: 0.5 + 0.5 * v for k, v in demand_pool.items()}

            # E3-FIX: index_add_edge() / index_expire_edge() keep the affinity index
            # current incrementally (O(1) per edge). No full rebuild needed here.
            # _pending_affinity_rebuild flag retained for safety but never set.
            pass  # no-op: index stays current via incremental updates

        current_year = year

        if progress_verbose or (seq_idx + 1) % progress_interval == 0 or seq_idx == 0:
            used_actors = len(world.person_film_count)
            print(
                f"  Movie {seq_idx + 1}/{n_movies} (movie_id {mid}, year {year})... "
                f"({used_actors} actors, {evo_stats['years_evolved']} years evolved)",
                flush=True,
            )
            _log_movie_progress(
                base_dir,
                {
                    "event": "movie_started",
                    "seq_idx": int(seq_idx + 1),
                    "movie_id": int(mid),
                    "year": int(year),
                    "genre": str(concept.get("genre", "")),
                    "tier": str(concept.get("tier", "")),
                    "country": str(concept.get("country", "")),
                    "timestamp": time.time(),
                },
            )

        # Step 2: Director (+ optional co-director, D12)
        with _speed_scope(speed_audit, "movie.pick_director", category="movie_selection", metadata={"movie_id": int(mid), "year": int(year)}):
            director_id = pick_director(world, concept)
        _stage_mark("pick_director")
        with _speed_scope(speed_audit, "movie.pick_co_director", category="movie_selection", metadata={"movie_id": int(mid), "year": int(year)}):
            co_director_id = pick_co_director(world, concept, director_id) if director_id else None
        _stage_mark("pick_co_director")

        # Step 3: Companies
        with _speed_scope(speed_audit, "movie.pick_companies", category="movie_selection", metadata={"movie_id": int(mid), "year": int(year)}):
            companies = pick_companies(world, concept, director_id)
        _stage_mark("pick_companies", extra={"company_count": int(len(companies))})
        # Step 4: Cast (shortlist + rescore + retry)
        with _speed_scope(speed_audit, "movie.pick_cast", category="movie_selection", metadata={"movie_id": int(mid), "year": int(year)}):
            cast, competition_pairs = pick_cast(world, concept, director_id)
        _stage_mark("pick_cast", extra={"cast_count": int(len(cast))})

        # Step 4b: Crew (below-the-line)
        with _speed_scope(speed_audit, "movie.pick_crew", category="movie_selection", metadata={"movie_id": int(mid), "year": int(year)}):
            crew_rows = pick_crew(world, concept, director_id, cast)
        _stage_mark("pick_crew", extra={"crew_count": int(len(crew_rows))})

        # Step 5: Title (use pre-assigned from bank, or pick dynamically)
        if title_assignment:
            ta = title_assignment
            title = str(ta.get("title", "") or "")
            row_idx = int(ta.get("_tb_row_idx", -1))
            preassigned_title_ok = bool(ta.get("_title_ok_static", bool(title))) and title not in world.used_titles
            if preassigned_title_ok:
                raw_tagline = ta["tagline"] if ta["tagline"] and ta["tagline"] != "nan" else ""
                tagline = str(raw_tagline or "")
                if (
                    not bool(ta.get("_tagline_ok_static", bool(tagline)))
                    or _tagline_reuse_score(world, tagline) > 0.0
                ):
                    tagline = _generate_tagline(str(concept.get("genre", "")), world, title=title)
                award_contender = ta.get("award_contender", False)  # D23
                world.mark_title_used(title, row_idx=row_idx)
                _register_tagline_use(world, tagline)
            else:
                with _speed_scope(speed_audit, "movie.pick_title_fallback", category="movie_selection", metadata={"movie_id": int(mid), "year": int(year)}):
                    title, tagline, award_contender = pick_title(world, concept)
        else:
            with _speed_scope(speed_audit, "movie.pick_title", category="movie_selection", metadata={"movie_id": int(mid), "year": int(year)}):
                title, tagline, award_contender = pick_title(world, concept)  # D23: unpack 3-tuple
        _stage_mark("pick_title")

        # Guard against empty/null titles (pick_title compositional fallback can return "")
        if not title or title == "nan":
            title = f"Untitled-{mid}"

        # Franchise tracking must happen before financials and keyword
        # selection, because both components read concept["installment"].
        franchise = concept.get("franchise")
        fid = None
        inst = None
        if franchise:
            fid = franchise["franchise_id"]
            franchise["movies_generated"] += 1
            inst = franchise["movies_generated"]
            concept["installment"] = inst
            if inst == 1:
                franchise["director_id"] = director_id
                franchise["company_ids"] = [c["company_id"] for c in companies]
                franchise["cast_pool"] = [c["person_id"] for c in cast]

        # Step 6: Financials (latent-driven correlated model)
        genre = concept["genre"]
        demand_key = (year, genre)
        remaining_demand = demand_pool.get(demand_key, 1.0)
        with _speed_scope(speed_audit, "movie.compute_financials", category="movie_selection", metadata={"movie_id": int(mid), "year": int(year)}):
            fin = compute_financials(world, concept, cast, director_id, companies,
                                     demand_factor=remaining_demand)
        _stage_mark("compute_financials")
        # Deplete demand pool after this movie
        depletion = DEPLETION_BY_TIER.get(concept["tier"], 0.08)
        demand_pool[demand_key] = max(0.1, remaining_demand * (1.0 - depletion))

        # Step 7: Keywords (D25: pass company_ids for cluster routing)
        with _speed_scope(speed_audit, "movie.pick_keywords", category="movie_selection", metadata={"movie_id": int(mid), "year": int(year)}):
            kw_ids = pick_keywords(world, concept,
                                   company_ids=[c["company_id"] for c in companies])
            _stage_mark("pick_keywords_inner", extra={"keyword_count": int(len(kw_ids))})
        _stage_mark("pick_keywords", extra={"keyword_count": int(len(kw_ids))})
        if getattr(world, "enable_llm_keyword_rerank", False):
            kw_ids = _maybe_refine_keywords(world, concept, kw_ids, llm_model=llm_model)
            kw_ids = _ensure_exact_topic_keyword_support(world, concept, kw_ids)
            _stage_mark("keyword_rerank", extra={"keyword_count": int(len(kw_ids))})
        kw_ids = _ensure_structural_keyword_anchors(kw_ids, concept, inst)
        _stage_mark("keyword_structural_anchors", extra={"keyword_count": int(len(kw_ids))})

        # === Build rows ==============================================

        # G2-FIX: Formula-generated plot_summary -- previously always "".
        # Non-empty values are essential for text analysis tests (review CV, keyword density).
        # This is a light deterministic description; LLM enrichment can overwrite it later.
        _tier_adj = {"Epic": "sweeping", "A": "acclaimed", "Mid": "gripping",
                     "Indie": "intimate", "Micro": "raw"}.get(concept["tier"], "compelling")
        _genre = concept["genre"]
        _country = concept["country"]
        _year = concept["year"]
        _franchise_note = f" (installment {inst} of the franchise)" if inst and inst > 1 else ""
        plot_summary = (
            f"A {_tier_adj} {_genre} film from {_country} ({_year}){_franchise_note}. "
            f"{'Rated ' + fin['certification'] + '.' if fin.get('certification') else ''}"
        ).strip()

        # A1: language ISO code (from full language name)
        _LANG_CODE = {
            "English": "en", "French": "fr", "Spanish": "es", "German": "de",
            "Italian": "it", "Japanese": "ja", "Korean": "ko", "Chinese": "zh",
            "Hindi": "hi", "Portuguese": "pt", "Arabic": "ar", "Russian": "ru",
            "Turkish": "tr", "Polish": "pl", "Dutch": "nl",
        }
        original_language = _LANG_CODE.get(concept.get("language", "English"), "en")

        # A1: aspect ratio (genre + era weighted)
        _yr = concept["year"]
        if _yr < 1960:
            aspect_ratio = world.rng.choice(["1.33:1", "1.37:1"], p=[0.60, 0.40])
        elif _yr < 1980:
            aspect_ratio = world.rng.choice(["1.33:1", "1.85:1", "2.39:1"], p=[0.15, 0.55, 0.30])
        elif concept["genre"] in ("Action", "Sci-Fi", "Fantasy") or concept["tier"] == "Epic":
            aspect_ratio = world.rng.choice(["1.85:1", "2.39:1"], p=[0.35, 0.65])
        else:
            aspect_ratio = world.rng.choice(["1.85:1", "2.39:1", "1.78:1"], p=[0.65, 0.25, 0.10])

        # A1: color format (era-based)
        if _yr < 1966:
            color_format = world.rng.choice(["B&W", "Color"], p=[0.75, 0.25])
        elif _yr < 1975:
            color_format = world.rng.choice(["B&W", "Color", "Colorized"], p=[0.20, 0.78, 0.02])
        else:
            color_format = "Color"

        movie_row = {
            "title_id": mid,
            "title": title,
            "year": concept["year"],
            "country": concept["country"],
            "language": concept["language"],
            "original_language": original_language,
            "aspect_ratio": aspect_ratio,
            "color_format": color_format,
            "genre": concept["genre"],
            "production_tier": concept["tier"],
            "budget_usd": fin["budget_usd"],
            "box_office_usd": fin["box_office_usd"],
            "runtime_minutes": fin["runtime_minutes"],
            "rating": fin["rating"],
            "num_votes": fin["num_votes"],
            "certification": fin["certification"],
            "tagline": tagline,
            "plot_summary": plot_summary,
            "franchise_id": str(fid) if fid is not None else None,
            "installment_no": inst,
            "award_campaign_strength": fin.get("award_campaign_strength", 0.0),
            "seed": world.seed,
            "snapshot_id": str(SNAPSHOT_CONFIG["snapshot_id"]),
        }

        with _speed_scope(speed_audit, "movie.write_movie_row", category="movie_writes", units=1):
            write_row("movie", movie_row)
        _stage_mark("write_movie_row")

        # === Secondary tables (auto-wired via registry) ===============
        # Compute base_release_date for downstream generators
        # Generate release dates first (needed by others)
        with _speed_scope(speed_audit, "secondary.release_dates", category="secondary_tables", metadata={"movie_id": int(mid), "year": int(year)}) as _sp:
            if "release_dates" in disabled_secondary_tables:
                _rd_rows = []
            else:
                _rd_rows = generate_release_dates(concept, mid, world.rng)
            if _sp is not None:
                _sp.units = len(_rd_rows)
        base_release_date = f"{int(concept['year']):04d}-01-01"
        if _rd_rows:
            write_rows("release_dates", _rd_rows)
            base_release_date = next(
                (r.get("release_date") for r in _rd_rows if r.get("release_type") == "Theatrical"),
                base_release_date
            )
        _stage_mark("release_dates", extra={"row_count": int(len(_rd_rows))})

        # Build context dict for all secondary generators
        movie_context = {
            "mid": mid,
            "concept": concept,
            "fin": fin,
            "director_id": director_id,
            "cast": cast,
            "crew_rows": crew_rows,
            "title": title,
            "rng": world.rng,
            "world": world,
            "base_release_date": base_release_date,
            "previous_movies_for_links": previous_movies_for_links,
            "award_contender": award_contender,  # D23: boosts award nomination probability
        }

        # D29: Generate daily first so weekly can derive from it (eliminates 98.1% mismatch)
        daily_table_enabled = "box_office_daily" not in disabled_secondary_tables
        weekly_table_enabled = "box_office_weekly" not in disabled_secondary_tables
        with _speed_scope(speed_audit, "secondary.box_office_daily", category="secondary_tables", metadata={"movie_id": int(mid), "year": int(year)}) as _sp:
            if daily_table_enabled or weekly_table_enabled:
                _daily_rows = generate_box_office_daily(
                    title_id=mid,
                    total_box_office_usd=fin["box_office_usd"],
                    base_release_date=base_release_date,
                    rng=world.rng,
                )
            else:
                _daily_rows = []
            if _sp is not None:
                _sp.units = len(_daily_rows)
        if daily_table_enabled and _daily_rows:
            write_rows("box_office_daily", _daily_rows)
        movie_context["_daily_rows"] = _daily_rows  # weekly will derive from these
        _stage_mark("box_office_daily", extra={"row_count": int(len(_daily_rows))})

        # Run all secondary generators (except release_dates and box_office_daily, already done)
        award_rows = []
        for gen in secondary_generators:
            if gen.table_name in ("release_dates", "box_office_daily"):
                continue
            kwargs = gen.build_args(movie_context)
            with _speed_scope(
                speed_audit,
                f"secondary.{gen.table_name}",
                category="secondary_tables",
                metadata={"movie_id": int(mid), "year": int(year)},
            ) as _sp:
                rows = gen.generate_fn(**kwargs)
                if _sp is not None:
                    _sp.units = len(rows or [])
            if rows:
                write_rows(gen.table_name, rows)
                if gen.table_name == "awards":
                    award_rows = rows
                if gen.post_hook:
                    gen.post_hook(rows, world)
            _stage_mark(f"secondary:{gen.table_name}", extra={"row_count": int(len(rows or []))})

        # V17: Feed the financial momentum system so subsequent movies
        # benefit from lagged performance memory (regime/momentum integration).
        with _speed_scope(speed_audit, "movie.record_financial_outcome", category="movie_updates", metadata={"movie_id": int(mid), "year": int(year)}):
            record_financial_outcome(
                world,
                concept,
                fin,
                director_id=director_id,
                companies=companies,
                award_rows=award_rows,
            )
        _stage_mark("record_financial_outcome")

        previous_movies_for_links.append({
            "title_id": mid,
            "genre": concept["genre"],
            "year": int(concept["year"]),
            "country": concept.get("country"),
            "tier": concept.get("tier"),
            "franchise_id": fid,
            "installment": inst,
        })

        # Track for year-boundary evolution
        year_bucket.append({
            "movie_id": mid,
            "director_id": director_id,
            "cast_ids": [c["person_id"] for c in cast],
            "company_ids": [c["company_id"] for c in companies],
            "genre": concept["genre"],
            "tier": concept["tier"],
            "rating": fin["rating"],
            "budget_usd": fin["budget_usd"],
            "box_office_usd": fin["box_office_usd"],
            "performance_ratio": float(fin.get("performance_ratio", float(fin["box_office_usd"]) / max(1.0, float(fin["budget_usd"])))),
            "market_regime_score": float(fin.get("market_regime_score", 0.0)),
            "company_momentum": float(fin.get("company_momentum", 0.0)),
            "director_momentum": float(fin.get("director_momentum", 0.0)),
            "genre_heat": float(fin.get("genre_heat", 0.0)),
            "slate_pressure": float(fin.get("slate_pressure", 1.0)),
            "competition_pairs": list(competition_pairs),
        })

        # === Dynamic edge spawning =================================
        graph = getattr(world, "graph", None)
        if graph is not None:
            _graph_spawn_scope = _speed_scope(speed_audit, "movie.graph_spawn", category="graph_updates", metadata={"movie_id": int(mid), "year": int(year)})
        else:
            _graph_spawn_scope = nullcontext(None)
        if graph is not None:
            with _graph_spawn_scope:
                spawn_rng = np.random.RandomState(
                    int(hashlib.blake2b(f"spawn|{world.seed}|{mid}|{concept['year']}".encode(), digest_size=4).hexdigest(), 16))
                cast_pids = [c["person_id"] for c in cast]
                spawned = 0

                if not hasattr(world, '_yearly_friendship_spawns'):
                    world._yearly_friendship_spawns = {}  # type: ignore[attr-defined]
                yr_spawns = world._yearly_friendship_spawns.get(year, 0)
                # V18-SCALE: All caps proportional to graph size
                _n_persons = len(world.persons) if world.persons is not None else 10000
                _per_movie_cap = max(10, _n_persons // 2000)   # 20 @ 40K, 100 @ 200K
                _yearly_cap    = max(500, n_movies // 2)        # scales with movie count

                # Actor-actor: new friendships from co-starring
                for ci in range(len(cast_pids)):
                    if spawned >= _per_movie_cap or yr_spawns >= _yearly_cap:
                        break
                    for cj in range(ci + 1, len(cast_pids)):
                        if spawned >= _per_movie_cap:
                            break
                        a, b = int(cast_pids[ci]), int(cast_pids[cj])
                        key = (min(a, b), max(a, b))
                        if graph.has_active_edge("friendship", key[0], key[1], year):
                            continue
                        sim = latent_similarity(world, a, b)
                        spawn_prob = 0.15 * sim  # V18-SCALE: tripled from 0.05
                        if spawn_rng.random() < spawn_prob:
                            # B07 fix: varied reason text so duplicate rate < 30%
                            _friendship_reasons = [
                                f"Co-starred in {title!r} ({concept['year']}); style sim={sim:.2f}",
                                f"Worked together on film {mid} in {concept['year']}; creative compatibility={sim:.2f}",
                                f"Developed on-set rapport during {concept['genre']} production ({concept['year']})",
                                f"Recurring collaboration since {concept['year']} ({concept['tier']} tier)",
                                f"Chemistry discovered filming {title!r}; latent sim={sim:.2f}",
                            ]
                            _reason = _friendship_reasons[int(spawn_rng.randint(0, len(_friendship_reasons)))]
                            _new_edge = {
                                "src_id": key[0], "dst_id": key[1],
                                "src_type": "person", "dst_type": "person",
                                "edge_type": "friendship", "sign": "+",
                                "weight": round(0.15 + 0.10 * sim, 3),
                                "reason": _reason,
                                "source_kind": "spawned_costar",
                                "valid_from": int(concept["year"]),
                                "valid_to": None,
                            }
                            graph.add_edge(
                                "friendship",
                                key[0],
                                key[1],
                                float(_new_edge["weight"]),
                                int(concept["year"]),
                                sign="+",
                                reason=_reason,
                                source_kind="spawned_costar",
                            )
                            spawned += 1
                            yr_spawns += 1

                # Director-actor: new mentorship edges
                if director_id and spawned < _per_movie_cap:
                    for pid in cast_pids[:5]:  # V18-SCALE: top-5 billed (was top-3)
                        if spawned >= _per_movie_cap:
                            break
                        existing_pref_ids = {int(actor_id) for actor_id, _weight, _vf, _vt in graph.get_director_prefs(int(director_id), year)}
                        if int(pid) in existing_pref_ids:
                            continue
                        if spawn_rng.random() < 0.20:  # V18-SCALE: raised from 0.08
                            # B07 fix: varied mentorship reason text
                            _mentor_reasons = [
                                f"Director guided actor on {title!r} ({concept['year']}); mentorship developed",
                                f"Intensive collaboration on {concept['genre']} film {mid}; director took mentorial role",
                                f"Actor first major role under this director ({concept['year']})",
                                f"Creative mentorship formed during {concept['tier']}-tier production in {concept['year']}",
                            ]
                            _mreason = _mentor_reasons[int(spawn_rng.randint(0, len(_mentor_reasons)))]
                            _new_edge = {
                                "src_id": int(director_id), "dst_id": int(pid),
                                "src_type": "person", "dst_type": "person",
                                "edge_type": "mentorship", "sign": "+",
                                "weight": 0.30,
                                "reason": _mreason,
                                "source_kind": "spawned_mentorship",
                                "valid_from": int(concept["year"]),
                                "valid_to": None,
                            }
                            graph.add_edge(
                                "mentorship",
                                int(director_id),
                                int(pid),
                                0.30,
                                int(concept["year"]),
                                sign="+",
                                reason=_mreason,
                                source_kind="spawned_mentorship",
                            )
                            spawned += 1

                # == Chemistry edges: emerge from successful co-starring (temporal) ==
                # Chemistry is DISCOVERED by a hit film, not pre-existing.
                # Only spawn if this movie clears both quality AND commercial thresholds.
                # valid_from = release year -> feeds back into casting via D19 weight boost.
                rating_val = float(fin.get("rating", 0.0))
                bo_val = float(fin.get("box_office_usd", 0.0))
                budget_val = float(fin.get("budget_usd", 1.0))
                is_successful_film = (rating_val >= 7.0 and bo_val >= budget_val * 0.8)  # V18-SCALE: relaxed thresholds

                if is_successful_film and len(cast_pids) >= 2:
                    if not hasattr(world, "_chemistry_pairs"):
                        world._chemistry_pairs = set()  # type: ignore[attr-defined]
                    chemistry_spawned = 0
                    for ci in range(len(cast_pids)):
                        for cj in range(ci + 1, len(cast_pids)):
                            _chem_cap = max(5, len(cast_pids) // 4)  # V18-SCALE: proportional to cast
                            if chemistry_spawned >= _chem_cap:
                                break
                            a, b = int(cast_pids[ci]), int(cast_pids[cj])
                            pair_key = (min(a, b), max(a, b))
                            if pair_key in world._chemistry_pairs:
                                continue  # already have chemistry from earlier film
                            sim = latent_similarity(world, a, b)
                            if sim >= 0.50:  # V18-SCALE: lowered from 0.70
                                _new_edge = {
                                    "src_id": pair_key[0], "dst_id": pair_key[1],
                                    "src_type": "person", "dst_type": "person",
                                    "edge_type": "chemistry", "sign": "+",
                                    "weight": round(0.50 + 0.45 * sim, 3),  # 0.84-0.95 range
                                    "source_kind": "latent_hybrid",
                                    "reason": f"co-star sim={sim:.2f} rating={rating_val:.1f} film_id={mid}",
                                    "valid_from": int(concept["year"]),
                                    "valid_to": None,
                                }
                                graph.add_edge(
                                    "chemistry",
                                    pair_key[0],
                                    pair_key[1],
                                    float(_new_edge["weight"]),
                                    int(concept["year"]),
                                    sign="+",
                                    reason=str(_new_edge["reason"]),
                                    source_kind="latent_hybrid",
                                )
                                world._chemistry_pairs.add(pair_key)
                                chemistry_spawned += 1
                                spawned += 1
                                yr_spawns += 1
                    if chemistry_spawned > 0:
                        evo_stats["chemistry_edges"] = evo_stats.get("chemistry_edges", 0) + chemistry_spawned

                # V18-FIX #14: Removed dead _pending_affinity_rebuild flag.
                # Edges are now incrementally indexed via index_add_edge (fix #12).
                if spawned > 0:
                    evo_stats["spawned_edges"] = evo_stats.get("spawned_edges", 0) + spawned
                    world._yearly_friendship_spawns[year] = yr_spawns
            _stage_mark("graph_spawn", extra={"spawned_edges": int(spawned)})

        # Cast (A1: screen_time_minutes + salary_usd)
        _SCREEN_TIME = {1: (65, 100), 2: (35, 65), 3: (20, 40)}
        # V18-FIX #15: Extended salary fractions for deep billing orders.
        # Previously only 1-3 were covered; billing 4-44+ all got identical
        # default (0.001, 0.005), creating unnaturally flat salary distributions.
        _SALARY_FRAC = {
            1: (0.03, 0.10),   # Lead star
            2: (0.01, 0.04),   # Second lead
            3: (0.005, 0.02),  # Third billing
            4: (0.003, 0.012), # Featured supporting
            5: (0.002, 0.008), # Supporting
            6: (0.001, 0.005), # Ensemble supporting
            7: (0.001, 0.004), # Ensemble supporting
            8: (0.0008, 0.003),# Minor supporting
        }
        with _speed_scope(speed_audit, "movie.write_cast_info", category="movie_writes", units=len(cast)):
            for c in cast:
                bo = c["billing_order"]
                st_lo, st_hi = _SCREEN_TIME.get(bo, (5, 20))
                sl_lo, sl_hi = _SALARY_FRAC.get(bo, (0.001, 0.005))
                write_row("cast_info", {
                    "title_id": mid,
                    "person_id": c["person_id"],
                    "character_name": c["character_name"],
                    "character_description": _character_description_for_cast(
                        c.get("archetype", ""),
                        concept.get("genre", ""),
                        bo,
                    ),
                    "billing_order": bo,
                    "archetype": c["archetype"],
                    "screen_time_minutes": int(world.rng.randint(st_lo, st_hi + 1)),
                    "salary_usd": int(fin["budget_usd"] * world.rng.uniform(sl_lo, sl_hi)),
                })
        _stage_mark("write_cast_info", extra={"row_count": int(len(cast))})

        # Crew (A1: department from pick_crew) -- streamed to sink
        _crew_batch = []
        for cr in crew_rows:
            _crew_batch.append({
                "title_id": mid,
                "person_id": int(cr["person_id"]),
                "crew_role": cr["crew_role"],
                "credit_order": int(cr.get("credit_order", 1)),
                "department": str(cr.get("department", "Production")),
            })
        if _crew_batch:
            with _speed_scope(speed_audit, "movie.write_movie_crew", category="movie_writes", units=len(_crew_batch)):
                write_rows("movie_crew", _crew_batch)
        _stage_mark("write_movie_crew", extra={"row_count": int(len(_crew_batch))})

        # Director(s)
        with _speed_scope(speed_audit, "movie.write_directors", category="movie_writes", units=1 + int(bool(co_director_id))):
            write_row("movie_directors", {
                "title_id": mid,
                "director_id": director_id,
            })
            if co_director_id:  # D12: co-director row
                write_row("movie_directors", {
                    "title_id": mid,
                    "director_id": co_director_id,
                })
        _stage_mark("write_directors", extra={"row_count": int(1 + int(bool(co_director_id)))})

        # Companies
        with _speed_scope(speed_audit, "movie.write_companies", category="movie_writes", units=len(companies)):
            for c in companies:
                write_row("movie_companies", {
                    "title_id": mid,
                    "company_id": c["company_id"],
                    "role": c["role"],
                })
        _stage_mark("write_companies", extra={"row_count": int(len(companies))})

        # Keywords
        with _speed_scope(speed_audit, "movie.write_keywords", category="movie_writes", units=len(kw_ids)):
            for kid in kw_ids:
                write_row("movie_keyword", {
                    "title_id": mid,
                    "keyword_id": kid,
                })
        _stage_mark("write_keywords", extra={"row_count": int(len(kw_ids))})
        if progress_verbose or (seq_idx + 1) % progress_interval == 0 or (seq_idx + 1) == n_movies:
            print(
                f"    completed movie_id {mid} in {time.perf_counter() - movie_started_at:.1f}s",
                flush=True,
            )
            _log_movie_progress(
                base_dir,
                {
                    "event": "movie_completed",
                    "seq_idx": int(seq_idx + 1),
                    "movie_id": int(mid),
                    "year": int(year),
                    "genre": str(concept.get("genre", "")),
                    "tier": str(concept.get("tier", "")),
                    "country": str(concept.get("country", "")),
                    "elapsed_sec": round(float(time.perf_counter() - movie_started_at), 4),
                    "timestamp": time.time(),
                },
            )

    # ─── Final year-boundary evolution for the last year ───────────────
    if year_bucket and current_year is not None:
        evolve_completed_year(int(current_year), year_bucket)
        if memory_audit is not None:
            process_row = memory_audit.record_process(
                f"pipeline_year_{current_year}_final_sample",
                sample_kind="process_sample",
                note="Final year-boundary process envelope sample after temporal evolution.",
                metadata={
                    "year_completed": int(current_year),
                    "years_evolved": int(evo_stats["years_evolved"]),
                },
            )
            milestone = evo_stats["years_evolved"] == 1 or evo_stats["years_evolved"] % 5 == 0
            if milestone:
                _record_pipeline_memory_snapshot(
                    memory_audit,
                    f"pipeline_year_{current_year}_final_checkpoint",
                    world,
                    accum=global_tables,
                    metadata={
                        "year_completed": int(current_year),
                        "years_evolved": int(evo_stats["years_evolved"]),
                    },
                    note="Final milestone snapshot after temporal evolution.",
                )
            if process_row.get("plateau_hit") and not plateau_recorded:
                plateau_recorded = True
                _record_pipeline_memory_snapshot(
                    memory_audit,
                    f"pipeline_year_{current_year}_final_plateau",
                    world,
                    accum=global_tables,
                    metadata={
                        "year_completed": int(current_year),
                        "years_evolved": int(evo_stats["years_evolved"]),
                    },
                    note="Final plateau-triggered snapshot after six consecutive process samples stayed within a 5% private-memory band.",
                    sample_kind="plateau",
                )
        resume_manager.commit_year(
            year=int(current_year),
            seq_idx=int(plan_records[-1]["seq_idx"]),
            year_tables=year_buffers,
            world=world,
            demand_pool=demand_pool,
            previous_movies_for_links=previous_movies_for_links,
            evo_stats=evo_stats,
        )

    # ─── Post-loop global tables (need completed movie rows) ───────────
    _log_movie_progress(base_dir, {"event": "post_loop_start", "timestamp": time.time()})
    with _speed_scope(speed_audit, "post_loop.merge_preloop_outputs", category="post_loop"):
        resume_manager.merge_preloop_outputs()
    for _disabled_table in sorted(disabled_global_tables | disabled_secondary_tables):
        _write_empty_arrow(base_dir, _disabled_table)

    with _speed_scope(speed_audit, "post_loop.readback_core_tables", category="post_loop") as _sp:
        movies_df = read_table(os.path.join(base_dir, "movie"), "movie")
        cast_df = read_table(os.path.join(base_dir, "cast_info"), "cast_info")
        movie_companies_df = read_table(os.path.join(base_dir, "movie_companies"), "movie_companies")
        if _sp is not None:
            _sp.units = len(movies_df) + len(cast_df) + len(movie_companies_df)

    ur_count = 0
    if "user_ratings" in disabled_post_loop_tables:
        print("  User ratings: skipped by runtime profile")
    else:
        ur_sink = make_table_sink(os.path.join(base_dir, "user_ratings.arrow"), "user_ratings")
        with _speed_scope(speed_audit, "post_loop.user_ratings", category="post_loop", units=len(movies_df)) as _sp:
            ur_count = generate_user_ratings(movies_df, world.rng, sink=ur_sink)
            if _sp is not None:
                _sp.units = int(ur_count or 0)
        ur_sink.close()
        print(f"  User ratings: {ur_count:,} ratings (streamed to {ur_sink.path})")

    # V14: Flush world_events (populated by LLMMasterClass during Big History Events)
    _world_events = list(getattr(world, "world_events", []))
    if "world_events" in disabled_post_loop_tables:
        print("  World events: skipped by runtime profile")
    elif _world_events:
        world_event_sink = make_table_sink(os.path.join(base_dir, "world_events.arrow"), "world_events")
        with _speed_scope(speed_audit, "post_loop.world_events_flush", category="post_loop", units=len(_world_events)):
            world_event_sink.write_rows(_world_events)
        world_event_sink.close()
        print(f"  World events: {len(_world_events)} events logged")

    # V14 A2/A3: Interval + cross-entity tables (post-loop; need full movie/cast data)
    # These are generated and immediately streamed to Arrow sinks.
    from secondary_tables import (
        generate_production_timeline, generate_streaming_windows,
        generate_person_contracts, generate_movie_sequence,
        generate_person_collaborations,
    )
    tv_series_media = global_tables.get("tv_series", pd.DataFrame())
    episode_cast_media = global_tables.get("episode_cast", pd.DataFrame())
    if isinstance(tv_series_media, pd.DataFrame):
        tv_series_media = tv_series_media.astype(object).where(pd.notna(tv_series_media), None).to_dict("records")
    if isinstance(episode_cast_media, pd.DataFrame):
        episode_cast_media = episode_cast_media.astype(object).where(pd.notna(episode_cast_media), None).to_dict("records")
    post_loop_counts: dict[str, int] = {}
    for tname, fn, args in [
        ("production_timeline", generate_production_timeline, (movies_df, world.rng)),
        ("streaming_windows", generate_streaming_windows, (movies_df, world.rng)),
        ("person_contracts", generate_person_contracts, (world.persons, world.companies, cast_df, world.rng)),
        ("movie_sequence", generate_movie_sequence, (movies_df,)),
        ("person_collaborations", generate_person_collaborations, (cast_df, movies_df)),
        (
            "media_links",
            generate_media_links,
            (
                movies_df,
                tv_series_media,
                cast_df,
                movie_companies_df,
                episode_cast_media,
                world.rng,
            ),
        ),
    ]:
        if tname in disabled_post_loop_tables:
            post_loop_counts[tname] = 0
            continue
        sink = make_table_sink(os.path.join(base_dir, f"{tname}.arrow"), tname)
        with _speed_scope(speed_audit, f"post_loop.{tname}", category="post_loop") as _sp:
            post_loop_counts[tname] = int(fn(*args, sink=sink) or 0)
            if _sp is not None:
                _sp.units = int(post_loop_counts[tname])
        sink.close()
    print(
        f"  A2/A3: {post_loop_counts.get('production_timeline', 0)} timeline phases, "
        f"{post_loop_counts.get('streaming_windows', 0)} streaming windows, "
        f"{post_loop_counts.get('person_contracts', 0)} contracts, "
        f"{post_loop_counts.get('movie_sequence', 0)} seq links, "
        f"{post_loop_counts.get('person_collaborations', 0)} collab pairs, "
        f"{post_loop_counts.get('media_links', 0)} media links"
    )
    for _disabled_table in sorted(disabled_post_loop_tables):
        _write_empty_arrow(base_dir, _disabled_table)
    _record_pipeline_memory_snapshot(
        memory_audit,
        "pipeline_post_loop_tables_ready",
        world,
        accum=global_tables,
        metadata={
            "movie_rows": len(movies_df),
            "cast_rows": len(cast_df),
            "movie_company_rows": len(movie_companies_df),
            "user_ratings_rows": int(ur_count),
            "world_event_rows": len(_world_events),
        },
        note="Snapshot after post-loop table generation and core streamed-table readback.",
        extra_specs=[
            ("movies_df_readback", movies_df, "post_loop_tables", "Movie readback DataFrame used for post-loop generators", "post_loop"),
            ("cast_df_readback", cast_df, "post_loop_tables", "Cast readback DataFrame used for post-loop generators", "post_loop"),
            ("movie_companies_df_readback", movie_companies_df, "post_loop_tables", "Movie-company readback DataFrame used for post-loop generators", "post_loop"),
        ],
    )

    # === Build output DataFrames (registry-driven) ===================
    with _speed_scope(speed_audit, "post_loop.build_result_frames", category="post_loop"):
        result = resume_manager.build_result_frames()

    # === Post-assembly: clamp birth_year in person_demographics ======
    # Demographics are generated BEFORE movies (L90), so birth_year is
    # based on debut_year from persons_enriched -- which may not match
    # the actual movie years a person gets assigned to during assembly.
    # Fix: find each person's earliest/latest movie year from the actual
    # cast_info + movie_directors tables, then clamp birth_year so that
    # age is always in [15, 100] at every assigned movie.
    demo_df = result.get("person_demographics")
    movie_df = result.get("movie")
    md_df = result.get("movie_directors")
    ci_df = result.get("cast_info")
    if demo_df is not None and len(demo_df) > 0 and movie_df is not None and len(movie_df) > 0:
        year_by_movie = movie_df.set_index("title_id")["year"]

        # V18-SCALE: Vectorized earliest/latest movie year computation
        # replaces iterrows() on 1.6M+ rows
        person_years = []
        if ci_df is not None and len(ci_df) > 0:
            ci_years = ci_df[["person_id", "title_id"]].copy()
            ci_years["year"] = ci_years["title_id"].map(year_by_movie)
            ci_years = ci_years.dropna(subset=["year"])
            ci_years["person_id"] = ci_years["person_id"].astype(int)
            ci_years["year"] = ci_years["year"].astype(int)
            person_years.append(ci_years[["person_id", "year"]])
        if md_df is not None and len(md_df) > 0:
            md_years = md_df[["director_id", "title_id"]].copy()
            md_years["year"] = md_years["title_id"].map(year_by_movie)
            md_years = md_years.dropna(subset=["year"])
            md_years.rename(columns={"director_id": "person_id"}, inplace=True)
            md_years["person_id"] = md_years["person_id"].astype(int)
            md_years["year"] = md_years["year"].astype(int)
            person_years.append(md_years[["person_id", "year"]])

        if person_years:
            all_py = pd.concat(person_years, ignore_index=True)
            year_bounds = all_py.groupby("person_id")["year"].agg(["min", "max"])
            year_bounds.columns = ["earliest", "latest"]

            # Vectorized clamping
            if "birth_date" in demo_df.columns:
                with _speed_scope(speed_audit, "post_loop.clamp_birth_years", category="post_loop"):
                    demo_df = demo_df.copy()
                    demo_df["_birth_year"] = pd.to_datetime(demo_df["birth_date"], errors="coerce").dt.year
                    demo_df["_pid"] = demo_df["person_id"].astype(int)
                    demo_df = demo_df.merge(year_bounds, left_on="_pid", right_index=True, how="left")

                    has_bounds = demo_df["earliest"].notna() & demo_df["_birth_year"].notna()
                    if has_bounds.any():
                        by = demo_df.loc[has_bounds, "_birth_year"].values.astype(int)
                        max_by = demo_df.loc[has_bounds, "earliest"].values.astype(int) - 15
                        min_by = demo_df.loc[has_bounds, "latest"].values.astype(int) - 100
                        # Fix tiny windows
                        tiny = min_by > max_by
                        min_by[tiny] = max_by[tiny] - 5
                        new_by = np.clip(by, min_by, max_by)
                        changed = new_by != by

                        if changed.any():
                            # Reconstruct birth_date strings for clamped rows
                            c_idx = demo_df.index[has_bounds][changed]
                            old_dates = demo_df.loc[c_idx, "birth_date"].astype(str)
                            new_years = new_by[changed]
                            new_dates = [
                                f"{yr:04d}{d[4:]}" if isinstance(d, str) and len(d) >= 10 else f"{yr:04d}-06-15"
                                for yr, d in zip(new_years, old_dates)
                            ]
                            demo_df.loc[c_idx, "birth_date"] = new_dates
                            print(f"  Birth-year clamped for {int(changed.sum())} persons (post-assembly age fix)")

                    demo_df.drop(columns=["_birth_year", "_pid", "earliest", "latest"], inplace=True, errors="ignore")
                    result["person_demographics"] = demo_df

    # Auto-PK assignment for tables that need it
    with _speed_scope(speed_audit, "post_loop.auto_pk_assignment", category="post_loop"):
        for table_name, pk_col in get_auto_pk_tables().items():
            df = result.get(table_name)
            if df is not None and len(df) > 0 and pk_col not in df.columns:
                df.insert(0, pk_col, range(1, len(df) + 1))

    # Stats
    used_actors = len(world.person_film_count)
    movie_df = result.get("movie")
    n_movies = len(movie_df) if movie_df is not None else 0
    print(f"\n=== Assembly complete ===")
    print(f"  Movies:     {n_movies}")
    for tname in TABLE_DEFS:
        if tname == "movie":
            continue
        if tname in result:
            n = len(result[tname])
        elif os.path.exists(os.path.join(base_dir, f"{tname}.arrow")):
            n = len(read_table(os.path.join(base_dir, tname), tname))
        else:
            n = 0
        if n > 0:
            avg = f" (avg {n/n_movies:.1f}/movie)" if n_movies > 0 else ""
            src = " [streamed]" if tname in (STREAMABLE_TABLES | POST_LOOP_STREAMABLE) else ""
            print(f"  {tname:20s}: {n:>8}{avg}{src}")
    print(f"  Actors used: {used_actors}/{len(world.actors)} ({used_actors/len(world.actors)*100:.1f}%)")
    print(
        f"  Evolution: {evo_stats['years_evolved']} years, "
        f"{evo_stats['year_program_ops']} yearly ops, "
        f"{evo_stats['triggered_events']} triggered macro events, "
        f"{evo_stats.get('spawned_edges', 0)} spawned edges"
    )

    freq = Counter(world.person_film_count)
    top = world.person_film_count.most_common(5)
    print(f"  Top actors: {top}")

    return result


# -----------------------------------------------------------------------
# FLAT TABLE BUILDER
# -----------------------------------------------------------------------

def build_flat_table(result: dict, world: 'WorldState') -> pd.DataFrame:
    """Build the unnormalized deliverable A quickly (vectorized string assembly)."""
    movies_df = result["movie"]
    if movies_df is None or len(movies_df) == 0:
        return pd.DataFrame(columns=[
            "title_id", "title", "year", "genre", "country", "director",
            "actors", "companies", "budget_usd", "box_office_usd",
            "rating", "description", "keywords",
        ])

    cast_df = result["cast_info"]
    mc_df = result["movie_companies"]
    mk_df = result["movie_keyword"]
    md_df = result["movie_directors"]

    person_name = dict(zip(world.persons["person_id"], world.persons["name"]))
    company_name = dict(zip(world.companies["company_id"], world.companies["name"]))
    keyword_text = dict(zip(
        world.keywords["keyword_id"],
        world.keywords.get("keyword", world.keywords.get("name", world.keywords.iloc[:, 1]))
    ))
    director_name = dict(zip(world.directors["person_id"], world.directors["name"]))

    # V17: Vectorized string assembly using .map() + .agg() instead of per-row loops
    cast_by_mid = {}
    if len(cast_df) > 0 and "title_id" in cast_df.columns:
        cast_work = cast_df.copy()
        cast_work["person_label"] = cast_work["person_id"].map(person_name).fillna("?")
        if "archetype" in cast_work.columns:
            cast_work["actor_label"] = cast_work["person_label"] + " (" + cast_work["archetype"].fillna("").astype(str) + ")"
        else:
            cast_work["actor_label"] = cast_work["person_label"]
        sort_cols = [c for c in ["title_id", "billing_order"] if c in cast_work.columns]
        if sort_cols:
            cast_work = cast_work.sort_values(sort_cols)
        cast_by_mid = cast_work.groupby("title_id")["actor_label"].agg("; ".join).to_dict()

    comp_by_mid = {}
    if len(mc_df) > 0 and "title_id" in mc_df.columns:
        comp_work = mc_df.copy()
        comp_work["company_label"] = comp_work["company_id"].map(company_name).fillna("?")
        if "role" in comp_work.columns:
            comp_work["company_label"] = comp_work["company_label"] + " (" + comp_work["role"].fillna("").astype(str) + ")"
        comp_by_mid = comp_work.groupby("title_id")["company_label"].agg("; ".join).to_dict()

    dir_by_mid = {}
    if len(md_df) > 0 and {"title_id", "director_id"}.issubset(md_df.columns):
        dir_first = md_df.groupby("title_id")["director_id"].first()
        dir_by_mid = dir_first.map(lambda pid: director_name.get(pid, "?")).to_dict()

    kw_by_mid = {}
    if len(mk_df) > 0 and {"title_id", "keyword_id"}.issubset(mk_df.columns):
        kw_work = mk_df.copy()
        kw_work["keyword_label"] = kw_work["keyword_id"].map(keyword_text).fillna("?").astype(str)
        kw_by_mid = kw_work.groupby("title_id")["keyword_label"].agg(", ".join).to_dict()

    rows = []
    append_row = rows.append
    for row in movies_df.itertuples(index=False):
        mid = int(getattr(row, "title_id"))
        append_row({
            "title_id": mid,
            "title": getattr(row, "title", ""),
            "year": getattr(row, "year", None),
            "genre": getattr(row, "genre", ""),
            "country": getattr(row, "country", ""),
            "director": dir_by_mid.get(mid, "?"),
            "actors": cast_by_mid.get(mid, ""),
            "companies": comp_by_mid.get(mid, ""),
            "budget_usd": getattr(row, "budget_usd", None),
            "box_office_usd": getattr(row, "box_office_usd", None),
            "rating": getattr(row, "rating", None),
            "description": getattr(row, "plot_summary", ""),
            "keywords": kw_by_mid.get(mid, ""),
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------

def _count_available_titles(base_dir: str) -> int:
    """Choose default movie count with overflow support.

    Uses title bank size as a floor and ENTITY_COUNTS['movies'] as the target.
    If target is None (auto-detect), generates exactly one movie per curated title.
    If target exceeds curated titles, generation falls back to compositional titles.
    """
    import json as _json
    edir = Path(base_dir) / "entities"

    configured_target = ENTITY_COUNTS.get("movies")
    if configured_target is not None:
        configured_target = int(configured_target)

    # Try CSV first (post-conversion)
    csv_path = edir / "title_bank.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        curated = int(len(df))
        if configured_target is None:
            print(f"  Auto-detect: will generate {curated} movies (1 per curated title)")
            return curated
        if configured_target > curated:
            print(f"  Title bank has {curated} curated titles; will generate {configured_target - curated} compositional overflow titles.")
        return max(curated, configured_target)

    # Try JSON (pre-conversion)
    json_path = edir / "movie_titlebank.json"
    if json_path.exists():
        data = _json.loads(json_path.read_text(encoding="utf-8"))
        curated = int(len(data))
        if configured_target is None:
            print(f"  Auto-detect: will generate {curated} movies (1 per curated title)")
            return curated
        if configured_target > curated:
            print(f"  Title bank has {curated} curated titles; will generate {configured_target - curated} compositional overflow titles.")
        return max(curated, configured_target)

    # Fallback
    fallback = configured_target or 5000
    print(f"  WARNING: No title bank found -- using fallback {fallback}")
    return fallback


def main(base_dir: str = None, n_movies: int = None,
         enable_llm_evolution: bool = True,
         enable_llm_critic: bool = True,
         enable_llm_world_policy: bool = True,
         enable_llm_concept_packs: bool = True,
         enable_llm_year_slates: bool = True,
         enable_llm_keyword_motifs: bool = True,
         enable_llm_rerank: bool = True,
         enable_llm_keyword_rerank: bool = True,
         rerank_budget_movies: int | None = None,
         keyword_rerank_budget_movies: int | None = None,
         llm_model: str = None,
         llm_critic_model: str | None = None,
         evolution_log_dir: str = None,
         critic_log_dir: str | None = None,
          start_year: int | None = None,
          end_year: int | None = None,
          benchmark_mode: bool = False,
          resume_step100: bool = False,
          reset_step100_resume: bool = False,
          extend_step100: bool = False):
    """Run full movie assembly pipeline."""
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    # Dynamic movie count: use title bank size unless explicitly overridden.
    # When a year window is requested, defer auto-detection until after the
    # title bank has been filtered to the requested range.
    if n_movies is None and start_year is None and end_year is None:
        n_movies = _count_available_titles(base_dir)
        print(f"  Auto-detected {n_movies} titles in title bank")

    continuation_summary = None
    continuation_append_count = None
    if extend_step100:
        if n_movies is None:
            raise ValueError("--extend-step100 requires an explicit cumulative --n_movies target")
        continuation_summary = load_continuation_summary(base_dir)
        continuation_append_count = validate_extension_request(
            continuation_summary,
            target_movie_count=int(n_movies),
        )
        print(
            "  Continuation mode: "
            f"existing_plan={continuation_summary.plan_count:,}, "
            f"produced={continuation_summary.produced_movie_count:,}, "
            f"last_year={continuation_summary.last_completed_year}, "
            f"append_target={continuation_append_count:,}",
            flush=True,
        )

    config_path = os.getenv("DATA_SYS_PIPELINE_CONFIG") or os.getenv("DATA_SYS_V17_CONFIG")
    if config_path and not os.path.exists(config_path):
        config_path = None
    if config_path is None:
        config_path = os.path.join(base_dir, "v18_config.json")
        if not os.path.exists(config_path):
            config_path = None
    workspace = resolve_workspace(
        script_dir=base_dir,
        data_dir=base_dir,
        output_dir=base_dir,
        config_path=config_path,
    )
    speed_audit = _get_speed_audit_recorder(base_dir, default_experiment="pipeline-speed")

    world = WorldState(
        base_dir,
        seed=SNAPSHOT_CONFIG["seed"],
        config_path=config_path,
        workspace=workspace,
    )
    if start_year is not None and end_year is not None:
        os.environ["DATA_SYS_START_YEAR"] = str(int(start_year))
        os.environ["DATA_SYS_END_YEAR"] = str(int(end_year))
    with _speed_scope(speed_audit, "main.world_load", category="main"):
        world.load()
    if start_year is not None or end_year is not None:
        if start_year is None or end_year is None:
            raise ValueError("start_year and end_year must be provided together")
        if getattr(world, "title_bank", None) is None or len(world.title_bank) == 0:
            raise RuntimeError("Title bank is missing; cannot enforce requested year range")
        if "year" not in world.title_bank.columns:
            raise RuntimeError("Title bank has no year column; cannot enforce requested year range")
        tb_years = pd.to_numeric(world.title_bank["year"], errors="coerce")
        filtered_tb = world.title_bank.loc[
            tb_years.between(int(start_year), int(end_year), inclusive="both")
        ].copy()
        filtered_tb = filtered_tb.reset_index(drop=True)
        dropped = int(len(world.title_bank) - len(filtered_tb))
        if dropped > 0:
            print(
                f"  Filtered title bank to requested range {int(start_year)}-{int(end_year)} "
                f"(dropped {dropped} out-of-range rows)",
                flush=True,
            )
        continuation_resume = bool(extend_step100 or resume_step100)
        if filtered_tb.empty and not continuation_resume:
            raise RuntimeError(
                f"No titles remain in title bank for requested range {int(start_year)}-{int(end_year)}"
            )
        if n_movies is None:
            n_movies = int(len(filtered_tb))
        required_title_rows = int(n_movies)
        if extend_step100 and continuation_append_count is not None:
            required_title_rows = int(continuation_append_count)
        if int(len(filtered_tb)) < int(required_title_rows) and not continuation_resume:
            raise RuntimeError(
                f"Title bank has only {len(filtered_tb)} in-range titles for {int(start_year)}-{int(end_year)}, "
                f"but {int(required_title_rows)} movies were requested"
            )
        if int(len(filtered_tb)) < int(required_title_rows) and continuation_resume:
            print(
                f"  [Continuation] Title bank has {len(filtered_tb):,} rows for extension range "
                f"{int(start_year)}-{int(end_year)}; {int(required_title_rows):,} extension movies requested. "
                "Missing titles will use compositional fallback.",
                flush=True,
            )
        world.title_bank = filtered_tb
        world._prepare_title_bank_cache()
    if n_movies is None:
        n_movies = _count_available_titles(base_dir)
        print(f"  Auto-detected {n_movies} titles in title bank", flush=True)

    # Ensure dependent components (e.g., franchise planner) see the requested scale
    ENTITY_COUNTS["movies"] = int(n_movies)
    world.enable_llm_world_policy = bool(enable_llm_world_policy)
    world.enable_llm_concept_packs = bool(enable_llm_concept_packs)
    world.enable_llm_year_slates = bool(enable_llm_year_slates)
    world.enable_llm_keyword_motifs = bool(enable_llm_keyword_motifs)
    world.enable_llm_rerank = bool(enable_llm_rerank)
    world.enable_llm_keyword_rerank = bool(enable_llm_keyword_rerank)
    world.target_movie_count = int(n_movies)
    world.rerank_budget_movies = rerank_budget_for_movies(int(n_movies), rerank_budget_movies)
    world.rerank_budget_remaining = int(world.rerank_budget_movies)
    world.keyword_rerank_budget_movies = keyword_rerank_budget_for_movies(int(n_movies), keyword_rerank_budget_movies)
    world.keyword_rerank_budget_remaining = int(world.keyword_rerank_budget_movies)
    if reset_step100_resume and (resume_step100 or extend_step100):
        raise ValueError("Use reset-step100-resume by itself; it cannot be combined with resume or extension")
    resume_manager = Step100ResumeManager(base_dir)
    resume_manager.prepare(
        run_payload={
            "seed": int(getattr(world, "seed", SNAPSHOT_CONFIG.get("seed", 42))),
            "n_movies": int(n_movies),
            "start_year": int(start_year) if start_year is not None else None,
            "end_year": int(end_year) if end_year is not None else None,
            "benchmark_mode": bool(benchmark_mode),
            "enable_llm_evolution": bool(enable_llm_evolution),
            "enable_llm_critic": bool(enable_llm_critic),
            "enable_llm_world_policy": bool(enable_llm_world_policy),
            "enable_llm_concept_packs": bool(enable_llm_concept_packs),
            "enable_llm_year_slates": bool(enable_llm_year_slates),
            "enable_llm_keyword_motifs": bool(enable_llm_keyword_motifs),
            "enable_llm_rerank": bool(enable_llm_rerank),
            "enable_llm_keyword_rerank": bool(enable_llm_keyword_rerank),
            "entity_counts": {
                "persons": int(len(world.persons) if world.persons is not None else 0),
                "companies": int(len(world.companies) if world.companies is not None else 0),
                "keywords": int(len(world.keywords) if world.keywords is not None else 0),
                "titles": int(len(world.title_bank) if world.title_bank is not None else 0),
                "characters": int(len(world.character_bank) if world.character_bank is not None else 0),
            },
        },
        resume=bool(resume_step100 or extend_step100),
        reset=bool(reset_step100_resume),
        extend=bool(extend_step100),
    )
    _record_pipeline_memory_snapshot(
        _get_memory_audit_recorder(),
        "pipeline_post_world_load",
        world,
        metadata={
            "person_rows": len(world.persons) if world.persons is not None else 0,
            "company_rows": len(world.companies) if world.companies is not None else 0,
        },
        note="Pipeline snapshot immediately after world.load().",
    )

    # Save enriched persons with pop_weight for verify.py
    with _speed_scope(speed_audit, "main.save_persons_enriched_pre", category="main"):
        df_to_arrow(world.persons, os.path.join(base_dir, "persons_enriched.arrow"))

    # V15 FIX: Save enriched companies so defunct_year mutations from llm_master.py
    # (e.g. merge_companies marking company B defunct) are persisted to disk.
    with _speed_scope(speed_audit, "main.save_companies_enriched_pre", category="main"):
        df_to_arrow(world.companies, os.path.join(base_dir, "companies_enriched.arrow"))

    # Default evolution log dir
    if evolution_log_dir is None and enable_llm_evolution:
        evolution_log_dir = os.path.join(base_dir, "graph", "temporal_patches")
    if critic_log_dir is None and enable_llm_critic:
        critic_log_dir = os.path.join(base_dir, "critic")

    # Keep durable movie checkpoints even in benchmark mode so long runs can
    # be resumed after host/laptop interruptions. Benchmark mode still skips
    # heavyweight derivative exports and yearly snapshots via runtime config.
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    try:
        with _speed_scope(speed_audit, "main.assemble_movies", category="main", units=int(n_movies or 0)):
            result = assemble_movies(world, n_movies,
                                     enable_llm_evolution=enable_llm_evolution,
                                     llm_model=llm_model,
                                     evolution_log_dir=evolution_log_dir,
                                     checkpoint_dir=checkpoint_dir,
                                     speed_audit=speed_audit,
                                     resume_manager=resume_manager,
                                     extend_step100=bool(extend_step100),
                                     extension_start_year=start_year,
                                     extension_end_year=end_year)
    except BaseException as exc:
        try:
            resume_manager.mark_interrupted(str(exc))
        except Exception:
            pass
        raise

    # V19-FIX: Re-save enriched data AFTER assembly.  Temporal evolution
    # (procedural_year_step, llm_year_step) mutates world.persons pop_weight,
    # career_stage, retirement_year etc. and world.companies defunct_year
    # during the loop.  The pre-assembly save captured the initial state;
    # this captures the post-evolution final state.
    with _speed_scope(speed_audit, "main.save_persons_enriched_post", category="main"):
        df_to_arrow(world.persons, os.path.join(base_dir, "persons_enriched.arrow"))
    with _speed_scope(speed_audit, "main.save_companies_enriched_post", category="main"):
        df_to_arrow(world.companies, os.path.join(base_dir, "companies_enriched.arrow"))

    # V17: Post-generation LLM critic -- samples riskiest movies, proposes
    # bounded safe repairs (plot summaries, taglines, keywords, alt titles).
    critic_report = {"status": "disabled", "applied": 0, "sampled_titles": []}
    if enable_llm_critic:
        _log_movie_progress(base_dir, {"event": "critic_start", "timestamp": time.time()})
        try:
            with _speed_scope(speed_audit, "main.post_generation_critic", category="main"):
                result, critic_report = run_post_generation_critic(
                    result,
                    world,
                    enabled=True,
                    model=llm_critic_model,
                    log_dir=critic_log_dir,
                )
        except Exception as critic_exc:
            critic_report = {
                "status": "error_nonfatal",
                "reason": str(critic_exc),
                "applied": 0,
                "sampled_titles": [],
                "cache_hit": False,
                "non_fatal": True,
            }
            print(f"  Post-generation critic warning: {critic_exc}", flush=True)
        print(
            f"  Post-generation critic: {critic_report.get('status', 'unknown')} "
            f"(applied={critic_report.get('applied', 0)}, sampled={len(critic_report.get('sampled_titles', []))}, "
            f"cache_hit={critic_report.get('cache_hit', False)})",
            flush=True,
        )
        _log_movie_progress(
            base_dir,
            {
                "event": "critic_done",
                "status": str(critic_report.get("status", "unknown")),
                "applied": int(critic_report.get("applied", 0) or 0),
                "sampled": int(len(critic_report.get("sampled_titles", []) or [])),
                "non_fatal": bool(critic_report.get("non_fatal", False)),
                "timestamp": time.time(),
            },
        )
        try:
            import json as _json_cr
            with open(os.path.join(base_dir, "critic_report.json"), "w", encoding="utf-8") as _f:
                _json_cr.dump(critic_report, _f, ensure_ascii=False, indent=2)
        except Exception as _critic_exc:
            print(f"  Critic report warning: {_critic_exc}", flush=True)

    # Save in-memory tables as Arrow IPC (.arrow)
    with _speed_scope(speed_audit, "main.save_result_tables", category="main", units=sum(len(df) for df in result.values())):
        for name, df in result.items():
            path = os.path.join(base_dir, f"{name}.arrow")
            df_to_arrow(df, path, table_name=name)
            print(f"Saved {path} ({len(df):,} rows)")
    csv_mirror_tables = {
        "movie",
        "movie_directors",
        "movie_companies",
        "cast_info",
        "movie_keyword",
        "movie_crew",
        "awards",
        "reviews",
        "alternate_titles",
        "ratings_breakdown",
    }
    with _speed_scope(speed_audit, "main.sync_csv_mirrors", category="exports"):
        for name, df in result.items():
            if name not in csv_mirror_tables:
                continue
            _write_csv_mirror(df, os.path.join(base_dir, f"{name}.csv"), label="csv mirror")
    # Streamed tables were already written by their sinks during assembly.
    # List them for completeness.
    for tname in STREAMABLE_TABLES | POST_LOOP_STREAMABLE:
        arrow_path = os.path.join(base_dir, f"{tname}.arrow")
        if os.path.exists(arrow_path):
            sz_mb = os.path.getsize(arrow_path) / 1e6
            print(f"Saved {arrow_path} (streamed, {sz_mb:.1f} MB)")

    if benchmark_mode:
        print("Benchmark mode: skipped flat/analysis/temporal-edge derivative exports")
    else:
        # Save flat table (Deliverable A)
        with _speed_scope(speed_audit, "main.build_flat_table", category="exports"):
            flat = build_flat_table(result, world)
        with _speed_scope(speed_audit, "main.save_flat_table", category="exports", units=len(flat)):
            df_to_arrow(flat, os.path.join(base_dir, "movies_flat.arrow"))
            print(f"Saved movies_flat.arrow (Deliverable A, {len(flat)} rows x {len(flat.columns)} cols)")
            _write_csv_mirror(flat, os.path.join(base_dir, "movies_flat.csv"), label="csv mirror")

        # Save analysis table (Deliverable B = A + debug cols)
        with _speed_scope(speed_audit, "main.save_analysis_table", category="exports", units=len(flat)):
            analysis = flat.copy()
            movie_df = result["movie"]
            for col in ["title_id", "production_tier", "runtime_minutes", "certification",
                         "num_votes", "franchise_id", "installment_no", "seed", "snapshot_id"]:
                if col in movie_df.columns:
                    analysis[col] = movie_df[col].values
            df_to_arrow(analysis, os.path.join(base_dir, "movies_analysis.arrow"))
            print(f"Saved movies_analysis.arrow (Deliverable B)")
            _write_csv_mirror(analysis, os.path.join(base_dir, "movies_analysis.csv"), label="csv mirror")

        # Save complete temporal edge graph (all versions, including expired SCD2 rows).
        # This is the definitive output for temporal graph analysis.
        graph = getattr(world, "graph", None)
        if graph is not None:
            movie_year_values = result.get("movie", pd.DataFrame()).get("year", pd.Series(dtype=float))
            movie_years = pd.to_numeric(movie_year_values, errors="coerce")
            final_year = int(movie_years.max()) if len(movie_years) > 0 and pd.notna(movie_years.max()) else 2025
            if hasattr(graph, "flush_all"):
                with _speed_scope(speed_audit, "main.graph_flush_all", category="exports"):
                    graph.flush_all()
            with _speed_scope(speed_audit, "main.export_temporal_history", category="exports"):
                temporal_count = graph.export_temporal_history(os.path.join(base_dir, "edges_temporal.arrow"))
            with _speed_scope(speed_audit, "main.export_final_active", category="exports"):
                final_count = graph.export_final_active(os.path.join(base_dir, "edges_final.arrow"), final_year)
            with _speed_scope(speed_audit, "main.materialize_legacy_edge_csv", category="exports"):
                graph.materialize_legacy_csv(os.path.join(base_dir, "graph", "edge_graph.csv"), final_year)
            print(f"Saved edges_temporal.arrow (full SCD2 history, {temporal_count} rows)")
            print(f"Saved edges_final.arrow (active edges at final year, {final_count} rows)")

    if speed_audit is not None:
        try:
            speed_audit.finalize(
                {
                    "base_dir": str(base_dir),
                    "benchmark_mode": bool(benchmark_mode),
                    "n_movies": int(n_movies),
                    "llm_evolution_enabled": bool(enable_llm_evolution),
                    "llm_critic_enabled": bool(enable_llm_critic),
                    "llm_world_policy_enabled": bool(enable_llm_world_policy),
                    "llm_concept_packs_enabled": bool(enable_llm_concept_packs),
                    "llm_year_slates_enabled": bool(enable_llm_year_slates),
                    "llm_keyword_motifs_enabled": bool(enable_llm_keyword_motifs),
                    "llm_rerank_enabled": bool(enable_llm_rerank),
                    "llm_keyword_rerank_enabled": bool(enable_llm_keyword_rerank),
                    "llm_rerank_budget_movies": int(world.rerank_budget_movies),
                    "llm_keyword_rerank_budget_movies": int(world.keyword_rerank_budget_movies),
                }
            )
        except Exception as exc:
            print(f"Speed audit warning: {exc}")

    _log_movie_progress(
        base_dir,
        {
            "event": "step100_complete",
            "movie_rows": int(len(result.get("movie", []))) if isinstance(result.get("movie"), pd.DataFrame) else 0,
            "timestamp": time.time(),
        },
    )
    resume_manager.mark_complete()

    return result, world


if __name__ == "__main__":
    import argparse
    from model_defaults import model_for_role

    parser = argparse.ArgumentParser(description="Generate Mirage movies and relational tables.")
    parser.add_argument("--base_dir", default=None, help="Directory containing entities/ and graph/")
    parser.add_argument("--n_movies", type=int, default=None,
                        help="Number of movies to generate")
    parser.add_argument("--enable_llm_evolution", action="store_true", default=None,
                        help="Enable LLM-based world evolution at year boundaries")
    parser.add_argument("--disable_llm_evolution", action="store_true", default=None,
                        help="Disable LLM-based world evolution at year boundaries")
    parser.add_argument("--llm_model", default=model_for_role("temporal_evolution"),
                        help="LLM model for planning, evolution, and refinement")
    parser.add_argument("--enable_llm_critic", action="store_true", default=None,
                        help="Enable the bounded post-generation LLM critic/repair pass")
    parser.add_argument("--disable_llm_critic", action="store_true", default=None,
                        help="Disable the post-generation LLM critic/repair pass")
    parser.add_argument("--enable_llm_world_policy", action="store_true", default=None,
                        help="Use world_policy.json to bias correlated movie generation")
    parser.add_argument("--disable_llm_world_policy", action="store_true", default=None,
                        help="Disable world_policy.json usage during movie generation")
    parser.add_argument("--enable_llm_concept_packs", action="store_true", default=None,
                        help="Use concept_packs.json to drive concept-first movie generation")
    parser.add_argument("--disable_llm_concept_packs", action="store_true", default=None,
                        help="Disable concept_packs.json usage during movie generation")
    parser.add_argument("--enable_llm_year_slates", action="store_true", default=None,
                        help="Use year_slate_plan.json during movie generation")
    parser.add_argument("--disable_llm_year_slates", action="store_true", default=None,
                        help="Disable year_slate_plan.json usage during movie generation")
    parser.add_argument("--enable_llm_keyword_motifs", action="store_true", default=None,
                        help="Use keyword_motif_bank.json during keyword selection")
    parser.add_argument("--disable_llm_keyword_motifs", action="store_true", default=None,
                        help="Disable keyword_motif_bank.json usage during keyword selection")
    parser.add_argument("--enable_llm_rerank", action="store_true", default=None,
                        help="Rerank low-confidence concept-pack choices with a capped LLM pass")
    parser.add_argument("--disable_llm_rerank", action="store_true", default=None,
                        help="Disable LLM concept reranking")
    parser.add_argument("--enable_llm_keyword_rerank", action="store_true", default=None,
                        help="Refine weak keyword bundles with a capped LLM pass")
    parser.add_argument("--disable_llm_keyword_rerank", action="store_true", default=None,
                        help="Disable LLM keyword refinement/reranking")
    parser.add_argument("--rerank_budget_movies", type=int, default=None,
                        help="Override the maximum number of movies eligible for concept reranking")
    parser.add_argument("--keyword_rerank_budget_movies", type=int, default=None,
                        help="Override the maximum number of movies eligible for keyword refinement")
    parser.add_argument("--llm_critic_model", default=None,
                        help="Model override for the post-generation LLM critic")
    parser.add_argument("--evolution_log_dir", default=None,
                        help="Directory for evolution patch logs (default: graph/temporal_patches/)")
    parser.add_argument("--critic_log_dir", default=None,
                        help="Directory for post-generation critic logs (default: critic/)")
    parser.add_argument("--start_year", type=int, default=None,
                        help="Restrict movie generation to title-bank rows at or after this year")
    parser.add_argument("--end_year", type=int, default=None,
                        help="Restrict movie generation to title-bank rows at or before this year")
    parser.add_argument("--benchmark-mode", action="store_true", default=False,
                        help="Skip derivative flat/analysis/temporal-edge exports for cleaner performance benchmarking")
    parser.add_argument("--resume-step100", action="store_true", default=False,
                        help="Resume step 100 from the last fully committed year boundary in _step100_resume/. This continues the run, but does not guarantee byte-identical replay.")
    parser.add_argument("--extend-step100", action="store_true", default=False,
                        help="Append a new extension plan to an existing Step 100 workspace. --n_movies is the cumulative target; --start_year/--end_year describe the extension window.")
    parser.add_argument("--reset-step100-resume", action="store_true", default=False,
                        help="Discard prior _step100_resume artifacts and start a fresh step 100 run")
    args = parser.parse_args()

    enable_llm_evolution = True
    if getattr(args, 'enable_llm_evolution', None) is True:
        enable_llm_evolution = True
    if getattr(args, 'disable_llm_evolution', False):
        enable_llm_evolution = False
    enable_llm_critic = True
    if getattr(args, 'enable_llm_critic', None) is True:
        enable_llm_critic = True
    if getattr(args, 'disable_llm_critic', False):
        enable_llm_critic = False
    enable_llm_world_policy = False if getattr(args, 'disable_llm_world_policy', False) else True
    if getattr(args, 'enable_llm_world_policy', None) is True:
        enable_llm_world_policy = True
    enable_llm_concept_packs = False if getattr(args, 'disable_llm_concept_packs', False) else True
    if getattr(args, 'enable_llm_concept_packs', None) is True:
        enable_llm_concept_packs = True
    enable_llm_year_slates = False if getattr(args, 'disable_llm_year_slates', False) else True
    if getattr(args, 'enable_llm_year_slates', None) is True:
        enable_llm_year_slates = True
    enable_llm_keyword_motifs = False if getattr(args, 'disable_llm_keyword_motifs', False) else True
    if getattr(args, 'enable_llm_keyword_motifs', None) is True:
        enable_llm_keyword_motifs = True
    enable_llm_rerank = False if getattr(args, 'disable_llm_rerank', False) else True
    if getattr(args, 'enable_llm_rerank', None) is True:
        enable_llm_rerank = True
    enable_llm_keyword_rerank = False if getattr(args, 'disable_llm_keyword_rerank', False) else True
    if getattr(args, 'enable_llm_keyword_rerank', None) is True:
        enable_llm_keyword_rerank = True


    main(base_dir=args.base_dir, n_movies=args.n_movies,
         enable_llm_evolution=enable_llm_evolution,
         enable_llm_critic=enable_llm_critic,
         enable_llm_world_policy=enable_llm_world_policy,
         enable_llm_concept_packs=enable_llm_concept_packs,
         enable_llm_year_slates=enable_llm_year_slates,
         enable_llm_keyword_motifs=enable_llm_keyword_motifs,
         enable_llm_rerank=enable_llm_rerank,
         enable_llm_keyword_rerank=enable_llm_keyword_rerank,
         rerank_budget_movies=args.rerank_budget_movies,
         keyword_rerank_budget_movies=args.keyword_rerank_budget_movies,
         llm_model=args.llm_model,
         llm_critic_model=args.llm_critic_model,
         evolution_log_dir=args.evolution_log_dir,
         critic_log_dir=args.critic_log_dir,
         start_year=args.start_year,
         end_year=args.end_year,
         benchmark_mode=bool(args.benchmark_mode),
         resume_step100=bool(args.resume_step100),
         reset_step100_resume=bool(args.reset_step100_resume),
         extend_step100=bool(args.extend_step100))
