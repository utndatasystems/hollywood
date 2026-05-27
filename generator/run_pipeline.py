"""
Mirage end-to-end pipeline runner.

Design goals:
  - fresh-machine bootstrap from procedural entities + LLM enrichment
  - modern Arrow-aware completion checks
  - explicit year-span control via title-bank generation
  - one command works for both smoke tests and large runs
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from bootstrap_artifacts import (
    assert_no_recorded_fallbacks,
    clear_step_fallback_hits,
    initialize_research_audit,
    record_pipeline_step,
)
from contracts import ENTITY_COUNTS, GENRES, GENRE_WEIGHTS
from feather_sink import read_table
from model_defaults import model_for_role
from policy_runtime import (
    character_identity_bank_path,
    comparison_report_path,
    company_lexicon_path,
    concept_packs_path,
    decision_log_dir,
    decision_log_path,
    decision_log_path_for_run,
    franchise_bibles_path,
    identity_bank_path,
    keyword_seed_bank_path,
    keyword_motif_bank_path,
    llm_usage_log_path,
    llm_usage_log_path_for_run,
    modeling_priors_path,
    movie_progress_log_path,
    movie_progress_log_path_for_run,
    prompt_calibration_log_path,
    research_audit_path,
    research_audit_path_for_run,
    temporal_regime_plan_path,
    title_grammar_bank_path,
    world_policy_path,
    year_slate_plan_path,
)
from run_profiles import apply_run_profile
from text_polish import contains_placeholder_syntax, looks_like_weak_tagline, looks_like_weak_title, sanitize_tagline

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_FILE = BASE_DIR / ".pipeline_checkpoint.json"

ROOT_ARROW_TABLES = [
    "movie",
    "cast_info",
    "movie_directors",
    "movie_companies",
    "movie_keyword",
    "movie_crew",
    "release_dates",
    "box_office_weekly",
    "box_office_by_territory",
    "box_office_daily",
    "reviews",
    "awards",
    "locations",
    "alternate_titles",
    "ratings_breakdown",
    "movie_links",
    "person_demographics",
    "tv_series",
    "seasons",
    "episodes",
    "episode_cast",
    "company_links",
    "user_ratings",
    "production_timeline",
    "media_links",
    "person_contracts",
    "world_events",
    "movies_flat",
    "movies_analysis",
    "persons_enriched",
    "companies_enriched",
    "edges_temporal",
    "edges_final",
]

ROOT_COMPAT_FILES = [
    "movie.csv",
    "movies_flat.csv",
    "movies_analysis.csv",
    "tv_series.csv",
    "seasons.csv",
    "episodes.csv",
    "episode_cast.csv",
    "critic_report.json",
]

PIPELINE_TIMING_FILENAME = "pipeline_timing.jsonl"
IMDB_EXPORT_DIRNAME = "imdb_schema"

ENTITY_GENERATED_FILES = [
    "persons.json",
    "companies.json",
    "keywords.json",
    "persons_latent.json",
    "companies_latent.json",
    "character_bank.csv",
    "title_bank.csv",
    "person.csv",
    "person_roles.csv",
    "company.csv",
    "keyword.csv",
    "company_financial_profile.csv",
]

ENTITY_GENERATED_PATTERNS = [
    "latent_batch*_raw.txt",
    "person_enrich*_raw.txt",
]

ROOT_RESEARCH_ARTIFACTS = [
    identity_bank_path(BASE_DIR),
    character_identity_bank_path(BASE_DIR),
    company_lexicon_path(BASE_DIR),
    keyword_seed_bank_path(BASE_DIR),
    title_grammar_bank_path(BASE_DIR),
    temporal_regime_plan_path(BASE_DIR),
    modeling_priors_path(BASE_DIR),
]


def _load_table(name: str, table_name: str | None = None) -> pd.DataFrame:
    return read_table(str(BASE_DIR / name), table_name)


def _nonempty_text_fraction(series: pd.Series) -> float:
    if series is None or len(series) == 0:
        return 0.0
    values = series.fillna("").astype(str).str.strip()
    return float((values != "").mean())


def _word_count_like(text: object) -> int:
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'/-]*", str(text or "")))


def _looks_like_placeholder_plot(text: object) -> bool:
    value = " ".join(str(text or "").split()).strip()
    low = value.lower()
    if not value:
        return True
    if _word_count_like(value) < 18:
        return True
    if "synthetic" in low:
        return True
    if low.startswith(("a ", "an ")) and " film from " in low and re.search(r"\(\d{4}\)", low):
        return True
    if re.match(r"^(a|an)\s+[a-z-]+\s+.+?\s+film from\s+.+?\(\d{4}\)", low):
        return True
    if "rated " in low and _word_count_like(value) < 28:
        return True
    return False


def _is_good_plot(text: object) -> bool:
    value = " ".join(str(text or "").split()).strip()
    return bool(value) and not _looks_like_placeholder_plot(value)


def _is_good_series_summary(text: object) -> bool:
    value = " ".join(str(text or "").split()).strip()
    if not value:
        return False
    if "synthetic" in value.lower():
        return False
    return _word_count_like(value) >= 45


def _is_good_episode_description(text: object) -> bool:
    value = " ".join(str(text or "").split()).strip()
    if not value:
        return False
    if "synthetic" in value.lower():
        return False
    return _word_count_like(value) >= 14


def _count_csv_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return max(sum(1 for _ in handle) - 1, 0)
    except Exception:
        return None


def _check_json_list_exact_count(path: Path, expected_count: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(payload, list) and len(payload) == int(expected_count)


def _check_csv_exact_count(path: Path, expected_count: int) -> bool:
    row_count = _count_csv_rows(path)
    return row_count is not None and int(row_count) == int(expected_count)


def _check_persons_enriched(base_dir: Path, expected_count: int | None = None) -> bool:
    path = base_dir / "entities" / "persons.json"
    if not path.exists():
        return False
    try:
        persons = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(persons, list) or not persons:
        return False
    if expected_count is not None and len(persons) != int(expected_count):
        return False
    ok = 0
    for person in persons:
        if (
            str(person.get("bio", "") or "").strip()
            and bool(person.get("style_tags"))
            and bool(person.get("genre_affinity"))
        ):
            ok += 1
    return ok == len(persons)


def _check_latent(
    base_dir: Path,
    *,
    expected_persons: int | None = None,
    expected_companies: int | None = None,
) -> bool:
    persons_path = base_dir / "entities" / "persons.json"
    companies_path = base_dir / "entities" / "companies.json"
    persons_latent_path = base_dir / "entities" / "persons_latent.json"
    companies_latent_path = base_dir / "entities" / "companies_latent.json"

    if not all(path.exists() for path in (persons_path, companies_path, persons_latent_path, companies_latent_path)):
        return False
    try:
        persons = json.loads(persons_path.read_text(encoding="utf-8"))
        companies = json.loads(companies_path.read_text(encoding="utf-8"))
        person_latent = json.loads(persons_latent_path.read_text(encoding="utf-8"))
        company_latent = json.loads(companies_latent_path.read_text(encoding="utf-8"))
        if expected_persons is not None and len(persons) != int(expected_persons):
            return False
        if expected_companies is not None and len(companies) != int(expected_companies):
            return False
        return len(person_latent) >= len(persons) and len(company_latent) >= len(companies)
    except Exception:
        return False


def _check_edges(base_dir: Path, *, expected_persons: int | None = None) -> bool:
    graph_dir = base_dir / "graph"
    if not ((graph_dir / "runtime_manifest.json").exists() or (graph_dir / "edge_graph.csv").exists()):
        return False
    if expected_persons is None:
        return True
    communities_csv = graph_dir / "communities.csv"
    if communities_csv.exists():
        row_count = _count_csv_rows(communities_csv)
        if row_count is not None and int(row_count) >= int(expected_persons):
            return True
    communities_arrow = graph_dir / "communities.arrow"
    if communities_arrow.exists():
        try:
            import pyarrow as pa
            import pyarrow.ipc as ipc

            with pa.memory_map(str(communities_arrow), "r") as source:
                table = ipc.open_file(source).read_all()
            return int(table.num_rows) >= int(expected_persons)
        except Exception:
            return False
    return False


def _check_entities_csv(
    base_dir: Path,
    *,
    expected_persons: int | None = None,
    expected_companies: int | None = None,
    expected_keywords: int | None = None,
) -> bool:
    edir = base_dir / "entities"
    required = ("person.csv", "person_roles.csv", "company.csv", "keyword.csv")
    if not all((edir / name).exists() for name in required):
        return False
    if expected_persons is not None and not _check_csv_exact_count(edir / "person.csv", expected_persons):
        return False
    if expected_companies is not None and not _check_csv_exact_count(edir / "company.csv", expected_companies):
        return False
    if expected_keywords is not None and not _check_csv_exact_count(edir / "keyword.csv", expected_keywords):
        return False
    person_roles_count = _count_csv_rows(edir / "person_roles.csv")
    return person_roles_count is not None and int(person_roles_count) > 0


def _check_company_financial_profiles(base_dir: Path, expected_companies: int | None = None) -> bool:
    entities_dir = base_dir / "entities"
    profile_path = entities_dir / "company_financial_profile.csv"
    company_path = entities_dir / "company.csv"
    if not profile_path.exists() or not company_path.exists():
        return False
    try:
        profiles = pd.read_csv(profile_path, low_memory=False)
        companies = pd.read_csv(company_path, low_memory=False)
    except Exception:
        return False
    if profiles.empty or companies.empty:
        return False
    if "company_id" not in profiles.columns or "company_id" not in companies.columns:
        return False
    company_count = int(companies["company_id"].nunique())
    if expected_companies is not None and company_count != int(expected_companies):
        return False
    return int(profiles["company_id"].nunique()) >= company_count


def _check_title_bank(
    base_dir: Path,
    target_count: int,
    start_year: int | None = None,
    end_year: int | None = None,
    *,
    allow_prior_years: bool = False,
) -> bool:
    path = base_dir / "entities" / "title_bank.csv"
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return False
    if df.empty:
        return False
    if len(df) != int(target_count):
        return False
    if "title" in df.columns and int(df["title"].astype(str).nunique()) != len(df):
        return False
    if {"title", "tagline"}.issubset(df.columns):
        titles = df["title"].fillna("").astype(str).tolist()
        taglines = [
            sanitize_tagline(tagline, title=title)
            for title, tagline in zip(titles, df["tagline"].fillna("").astype(str).tolist(), strict=False)
        ]
        if any(not title or contains_placeholder_syntax(title) or looks_like_weak_title(title) for title in titles):
            return False
        if any(
            not tagline
            or contains_placeholder_syntax(tagline)
            or looks_like_weak_tagline(tagline, title=title)
            for title, tagline in zip(titles, taglines, strict=False)
        ):
            return False
        if len(set(taglines)) != len(taglines):
            return False
    if start_year is not None or end_year is not None:
        if start_year is None or end_year is None or "year" not in df.columns:
            return False
        years = pd.to_numeric(df["year"], errors="coerce")
        if years.isna().any():
            return False
        in_range = years.between(int(start_year), int(end_year), inclusive="both")
        if allow_prior_years:
            if not bool(in_range.any()):
                return False
        elif not bool(in_range.all()):
            return False
    return True


def _check_movies(base_dir: Path, *, expected_movies: int | None = None) -> bool:
    resume_manifest = base_dir / "_step100_resume" / "manifest.json"
    if resume_manifest.exists():
        try:
            payload = json.loads(resume_manifest.read_text(encoding="utf-8"))
        except Exception:
            return False
        if str(payload.get("status", "")) != "complete":
            return False
        if expected_movies is not None:
            if int(payload.get("movie_count", 0) or 0) != int(expected_movies):
                return False
            if int(payload.get("produced_movie_count", 0) or 0) < int(expected_movies):
                return False
    # Step 100 has two valid output shapes: the archival/full shape and the
    # benchmark-candidate shape. Benchmark mode intentionally skips derivative
    # flat/analysis/edge exports, so completion should be judged by the
    # canonical relational tables consumed by export/signoff.
    required_paths = [
        base_dir / "movie.arrow",
        base_dir / "cast_info.arrow",
        base_dir / "movie_directors.arrow",
        base_dir / "movie_companies.arrow",
        base_dir / "movie_keyword.arrow",
        base_dir / "release_dates.arrow",
        base_dir / "awards.arrow",
        base_dir / "alternate_titles.arrow",
        base_dir / "ratings_breakdown.arrow",
        base_dir / "movie_links.arrow",
        base_dir / "persons_enriched.arrow",
        base_dir / "companies_enriched.arrow",
    ]
    if not all(path.exists() for path in required_paths):
        return False
    try:
        movies = read_table(str(base_dir / "movie"), "movie")
        cast = read_table(str(base_dir / "cast_info"), "cast_info")
        movie_keyword = read_table(str(base_dir / "movie_keyword"), "movie_keyword")
    except Exception:
        return False
    if movies.empty or cast.empty or movie_keyword.empty:
        return False
    if expected_movies is not None and len(movies) < int(expected_movies):
        return False
    return True


def _check_keyword_genres(base_dir: Path) -> bool:
    path = base_dir / "entities" / "keyword.csv"
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return False
    required_columns = {"topic_genre", "pop_weight", "selection_bucket"}
    if df.empty or any(column not in df.columns for column in required_columns):
        return False
    if df["pop_weight"].isna().any() or df["selection_bucket"].isna().any():
        return False
    allowed_buckets = {"exact_anchor", "related_support", "story_specific", "generic"}
    bucket_values = {str(value).strip() for value in df["selection_bucket"].fillna("").astype(str).tolist() if str(value).strip()}
    if any(value not in allowed_buckets for value in bucket_values):
        return False
    generic_mask = df["selection_bucket"].fillna("").astype(str).str.strip().eq("generic")
    if df.loc[generic_mask, "topic_genre"].fillna("").astype(str).str.strip().ne("").any():
        return False
    non_generic = df.loc[~generic_mask].copy()
    if non_generic.empty:
        return False
    if non_generic["topic_genre"].fillna("").astype(str).str.strip().eq("").any():
        return False
    if len(df) >= len(GENRES):
        exact_anchor = df.loc[df["selection_bucket"].fillna("").astype(str).str.strip().eq("exact_anchor")]
        covered = {str(value).strip() for value in exact_anchor["topic_genre"].fillna("").astype(str).tolist() if str(value).strip()}
        if any(str(genre) not in covered for genre in GENRES):
            return False
    return True


def _check_world_policy(base_dir: Path, *, start_year: int | None = None, end_year: int | None = None) -> bool:
    path = world_policy_path(base_dir)
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    ok = (
        isinstance(payload, dict)
        and isinstance(payload.get("year_buckets"), list)
        and isinstance(payload.get("company_strategies"), list)
    )
    if not ok:
        return False
    if start_year is not None and int(payload.get("start_year", start_year)) != int(start_year):
        return False
    if end_year is not None and int(payload.get("end_year", end_year)) != int(end_year):
        return False
    return True


def _check_concept_packs(base_dir: Path) -> bool:
    path = concept_packs_path(base_dir)
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    packs = payload.get("packs") if isinstance(payload, dict) else None
    return isinstance(packs, list) and len(packs) > 0


def _check_year_slate_plan(base_dir: Path, *, start_year: int | None = None, end_year: int | None = None) -> bool:
    path = year_slate_plan_path(base_dir)
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    slates = payload.get("slates") if isinstance(payload, dict) else None
    if not isinstance(slates, list) or len(slates) <= 0:
        return False
    if start_year is not None:
        try:
            if min(int(row.get("start_year", start_year)) for row in slates if isinstance(row, dict)) != int(start_year):
                return False
        except Exception:
            return False
    if end_year is not None:
        try:
            if max(int(row.get("end_year", end_year)) for row in slates if isinstance(row, dict)) != int(end_year):
                return False
        except Exception:
            return False
    return True


def _check_keyword_motif_bank(base_dir: Path) -> bool:
    path = keyword_motif_bank_path(base_dir)
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    motifs = payload.get("motifs") if isinstance(payload, dict) else None
    return isinstance(motifs, list) and len(motifs) > 0


def _check_franchise_bibles(base_dir: Path) -> bool:
    path = franchise_bibles_path(base_dir)
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    bibles = payload.get("bibles") if isinstance(payload, dict) else None
    return isinstance(bibles, list) and len(bibles) > 0


def _check_plots(base_dir: Path) -> bool:
    movies = read_table(str(base_dir / "movie"), "movie")
    return (
        not movies.empty
        and "plot_summary" in movies.columns
        and bool(movies["plot_summary"].fillna("").map(_is_good_plot).all())
    )


def _check_tv_summaries(base_dir: Path) -> bool:
    series = read_table(str(base_dir / "tv_series"), "tv_series")
    episodes = read_table(str(base_dir / "episodes"), "episodes")
    if series.empty or episodes.empty:
        return False
    if "plot_summary" not in series.columns or "description" not in episodes.columns:
        return False
    return (
        bool(series["plot_summary"].fillna("").map(_is_good_series_summary).all())
        and bool(episodes["description"].fillna("").map(_is_good_episode_description).all())
    )


def _check_imdb_export(base_dir: Path) -> bool:
    export_dir = base_dir / IMDB_EXPORT_DIRNAME
    required = [
        export_dir / "title.csv",
        export_dir / "name.csv",
        export_dir / "cast_info.csv",
        export_dir / "movie_info.csv",
        export_dir / "export_manifest.json",
        export_dir / "imdb.duckdb",
    ]
    if not all(path.exists() for path in required):
        return False
    if (export_dir / "imdb.duckdb").stat().st_size <= 0:
        return False
    try:
        title = pd.read_csv(export_dir / "title.csv", nrows=5)
        name = pd.read_csv(export_dir / "name.csv", nrows=5)
        cast_info = pd.read_csv(export_dir / "cast_info.csv", nrows=5)
    except Exception:
        return False
    return list(title.columns[:3]) == ["id", "title", "imdb_index"] and not name.empty and not cast_info.empty


def _check_json_artifact(path: Path, required_keys: tuple[str, ...] = ()) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    return all(key in payload for key in required_keys)


def _load_json_object(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _check_identity_bank(base_dir: Path) -> bool:
    return _check_json_artifact(identity_bank_path(base_dir), ("families", "defaults"))


def _check_character_identity_bank(base_dir: Path) -> bool:
    return _check_json_artifact(character_identity_bank_path(base_dir), ("alias_templates", "solo_monikers"))


def _check_company_lexicon(base_dir: Path) -> bool:
    return _check_json_artifact(company_lexicon_path(base_dir), ("prefixes", "suffixes", "templates"))


def _check_keyword_seed_bank(base_dir: Path) -> bool:
    return _check_json_artifact(keyword_seed_bank_path(base_dir), ("genres", "universal_qualifiers"))


def _check_title_grammar_bank(
    base_dir: Path,
    *,
    target_count: int | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
) -> bool:
    path = title_grammar_bank_path(base_dir)
    if not _check_json_artifact(path, ("genre_templates", "tagline_templates")):
        return False
    if target_count is None:
        return True
    try:
        from topup_title_bank import _prepare_render_grammar, _validate_title_capacity_for_target

        grammar = _prepare_render_grammar(_load_json_object(path))
        temporal = _load_json_object(temporal_regime_plan_path(base_dir))
        priors = _load_json_object(modeling_priors_path(base_dir))
        title_priors = priors.get("title_generation", {})
        title_priors = title_priors if isinstance(title_priors, dict) else {}
        base_weights = {
            str(genre): float(GENRE_WEIGHTS.get(str(genre), 0.001) or 0.001)
            for genre in GENRES
        }
        for key in ("genre_base_weights", "genre_weights", "genre_prevalence"):
            raw = title_priors.get(key)
            if isinstance(raw, dict) and raw:
                base_weights.update({str(k): float(v) for k, v in raw.items() if str(k) in GENRES})
                break
        _validate_title_capacity_for_target(
            grammar,
            target_count=int(target_count),
            base_genre_weights=base_weights,
            temporal=temporal,
            title_priors=title_priors,
            start_year=start_year,
            end_year=end_year,
        )
    except Exception:
        return False
    return True


def _check_temporal_regime_plan(base_dir: Path, *, start_year: int | None = None, end_year: int | None = None) -> bool:
    path = temporal_regime_plan_path(base_dir)
    if not _check_json_artifact(path, ("year_weights", "phases")):
        return False
    if start_year is None and end_year is None:
        return True
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if start_year is not None and int(payload.get("start_year", start_year)) != int(start_year):
        return False
    if end_year is not None and int(payload.get("end_year", end_year)) != int(end_year):
        return False
    return True


def _check_modeling_priors(base_dir: Path) -> bool:
    return _check_json_artifact(
        modeling_priors_path(base_dir),
        ("person_generation", "company_generation", "edge_priors"),
    )


def _record_timing(step: dict, elapsed: float, ok: bool) -> None:
    path = BASE_DIR / PIPELINE_TIMING_FILENAME
    row = {
        "step_id": int(step["id"]),
        "step_name": str(step["name"]),
        "ok": bool(ok),
        "elapsed_sec": round(float(elapsed), 4),
        "timestamp": datetime.now().isoformat(),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True))
        handle.write("\n")


def _load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        try:
            return json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"completed_steps": [], "started_at": None}


def _save_checkpoint(state: dict) -> None:
    CHECKPOINT_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _clear_checkpoint() -> None:
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()


def _sanitize_run_id(value: str) -> str:
    safe = []
    for ch in str(value or ""):
        safe.append(ch if ch.isalnum() or ch in "._-" else "_")
    out = "".join(safe).strip("._-")
    return out or "run"


def _derive_run_id(started_at: str | None) -> str:
    raw = str(started_at or "").strip()
    if raw:
        try:
            dt = datetime.fromisoformat(raw)
            return dt.strftime("%Y%m%d_%H%M%S")
        except Exception:
            return _sanitize_run_id(raw)
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _reset_generated_outputs(*, preserve_entities: bool = False, preserve_artifacts: bool = False) -> None:
    for dirname in ("graph", "checkpoints", "critic", IMDB_EXPORT_DIRNAME, "_step100_resume"):
        target = BASE_DIR / dirname
        if target.exists():
            shutil.rmtree(target)

    for name in ROOT_ARROW_TABLES:
        path = BASE_DIR / f"{name}.arrow"
        if path.exists():
            path.unlink()
    for name in ROOT_COMPAT_FILES:
        path = BASE_DIR / name
        if path.exists():
            path.unlink()

    entities_dir = BASE_DIR / "entities"
    if entities_dir.exists() and not preserve_entities:
        for name in ENTITY_GENERATED_FILES:
            path = entities_dir / name
            if path.exists():
                path.unlink()
        for pattern in ENTITY_GENERATED_PATTERNS:
            for path in entities_dir.glob(pattern):
                if path.is_file():
                    path.unlink()

    artifact_extra_paths = (
        world_policy_path(BASE_DIR),
        year_slate_plan_path(BASE_DIR),
        keyword_motif_bank_path(BASE_DIR),
        concept_packs_path(BASE_DIR),
        franchise_bibles_path(BASE_DIR),
    )
    runtime_extra_paths = (
        comparison_report_path(BASE_DIR),
        llm_usage_log_path(BASE_DIR),
        research_audit_path(BASE_DIR),
        prompt_calibration_log_path(BASE_DIR),
        BASE_DIR / PIPELINE_TIMING_FILENAME,
    )
    for extra_path in (() if preserve_artifacts else artifact_extra_paths):
        if extra_path.exists():
            extra_path.unlink()
    for extra_path in runtime_extra_paths:
        if extra_path.exists():
            extra_path.unlink()

    if not preserve_artifacts:
        for extra_path in ROOT_RESEARCH_ARTIFACTS:
            if extra_path.exists():
                extra_path.unlink()

    decision_dir = decision_log_dir(BASE_DIR)
    if decision_dir.exists():
        for path in decision_dir.glob("*_research_mode_audit.json"):
            if path.is_file():
                path.unlink()

    _clear_checkpoint()


def _auto_counts(n_movies: int) -> dict[str, int]:
    movies = max(1, int(n_movies))
    if movies <= 250:
        return {
            "n_titles": movies,
            "n_persons": max(450, int(round(movies * 4.0))),
            "n_companies": max(60, int(round(movies * 0.25))),
            "n_keywords": max(260, int(round(movies * 2.6))),
            "n_characters": max(500, int(round(movies * 5.0))),
        }
    return {
        "n_titles": movies,
        "n_persons": max(1200, int(round(movies * 3.2))),
        "n_companies": max(120, int(round(movies * 0.065))),
        "n_keywords": max(350, int(round(movies * 0.14))),
        "n_characters": max(1500, int(round(movies * 4.0))),
    }


def _resolve_targets(args: argparse.Namespace) -> None:
    auto = _auto_counts(args.n_movies)
    args.n_titles = int(args.n_titles if args.n_titles is not None else auto["n_titles"])
    args.n_persons = int(args.n_persons if args.n_persons is not None else auto["n_persons"])
    args.n_companies = int(args.n_companies if args.n_companies is not None else auto["n_companies"])
    args.n_keywords = int(args.n_keywords if args.n_keywords is not None else auto["n_keywords"])
    args.n_characters = int(args.n_characters if args.n_characters is not None else auto["n_characters"])

    if args.start_year is not None and args.end_year is None:
        args.end_year = args.start_year
    if args.end_year is not None and args.start_year is None:
        args.start_year = args.end_year
    if args.start_year is None and args.end_year is None:
        args.start_year = 1950
        args.end_year = 2025
    if args.start_year is not None and args.end_year is not None and args.end_year < args.start_year:
        raise ValueError("end-year must be >= start-year")


def _resolve_toggle(args: argparse.Namespace, enable_name: str, disable_name: str, *, default: bool) -> bool:
    enabled = bool(default)
    if getattr(args, enable_name, None) is True:
        enabled = True
    if getattr(args, disable_name, None) is True:
        enabled = False
    setattr(args, enable_name, enabled)
    return enabled


def _resolve_llm_flags(args: argparse.Namespace) -> None:
    _resolve_toggle(args, "enable_llm_evolution", "disable_llm_evolution", default=True)
    _resolve_toggle(args, "enable_llm_critic", "disable_llm_critic", default=True)
    _resolve_toggle(args, "enable_llm_world_policy", "disable_llm_world_policy", default=True)
    _resolve_toggle(args, "enable_llm_concept_packs", "disable_llm_concept_packs", default=True)
    _resolve_toggle(args, "enable_llm_year_slates", "disable_llm_year_slates", default=True)
    _resolve_toggle(args, "enable_llm_keyword_motifs", "disable_llm_keyword_motifs", default=True)
    _resolve_toggle(args, "enable_llm_rerank", "disable_llm_rerank", default=True)
    _resolve_toggle(args, "enable_llm_keyword_rerank", "disable_llm_keyword_rerank", default=True)


def _clean_model_name(value: object) -> str | None:
    model = str(value or "").strip()
    return model or None


def _resolve_step_model(args: argparse.Namespace, *, step_id: int, script: str) -> str | None:
    general_model = _clean_model_name(getattr(args, "model", None))
    bootstrap_model = _clean_model_name(getattr(args, "bootstrap_model", None))
    planning_model = _clean_model_name(getattr(args, "planning_model", None))
    bulk_model = _clean_model_name(getattr(args, "bulk_artifact_model", None)) or model_for_role("artifact_bulk")

    if script == "generate_bootstrap_artifacts_api.py":
        return bootstrap_model or general_model
    if step_id == 72:
        return planning_model or general_model
    if step_id in {74, 95, 96, 97, 98}:
        return bulk_model
    return general_model


def _step_model_args(
    args: argparse.Namespace,
    *,
    step_id: int,
    script: str,
    option: str = "--model",
) -> list[str]:
    model = _resolve_step_model(args, step_id=step_id, script=script)
    return [option, model] if model else []


def _build_steps(args: argparse.Namespace) -> list[dict]:
    extend = bool(getattr(args, "extend_step100", False))

    def _continuation_entity_args(kind: str, target: int) -> list[str]:
        return (
            [
                "--base-dir",
                str(BASE_DIR),
                "--kind",
                kind,
                "--target-count",
                str(int(target)),
                "--seed",
                str(args.seed),
                "--mode",
                str(args.mode),
            ]
            + (
                ["--extension-start-year", str(args.start_year), "--extension-end-year", str(args.end_year)]
                if args.start_year is not None and args.end_year is not None
                else []
            )
            + ["--company-lifecycle-policy", str(args.company_lifecycle_policy)]
        )

    return [
        {
            "id": 4,
            "name": "Generate Modeling Priors",
            "script": "generate_bootstrap_artifacts_api.py",
            "args": [
                "--artifact", "modeling_priors",
                "--base-dir", str(BASE_DIR),
                "--mode", str(args.mode),
                "--n-movies", str(args.n_movies),
                "--n-persons", str(args.n_persons),
                "--n-companies", str(args.n_companies),
                "--n-keywords", str(args.n_keywords),
                "--n-titles", str(args.n_titles),
                "--start-year", str(args.start_year),
                "--end-year", str(args.end_year),
            ] + _step_model_args(args, step_id=4, script="generate_bootstrap_artifacts_api.py"),
            "check": _check_modeling_priors,
            "requires_api": bool(args.mode == "research"),
            "enabled": bool(args.mode == "research"),
            "description": "Generate the reusable numeric priors/control-plane artifact for research mode.",
        },
        {
            "id": 8,
            "name": "Generate Identity Bank",
            "script": "generate_bootstrap_artifacts_api.py",
            "args": [
                "--artifact", "identity_bank",
                "--base-dir", str(BASE_DIR),
                "--mode", str(args.mode),
                "--n-movies", str(args.n_movies),
                "--n-persons", str(args.n_persons),
                "--n-companies", str(args.n_companies),
                "--n-keywords", str(args.n_keywords),
                "--n-titles", str(args.n_titles),
                "--start-year", str(args.start_year),
                "--end-year", str(args.end_year),
            ] + _step_model_args(args, step_id=8, script="generate_bootstrap_artifacts_api.py"),
            "check": _check_identity_bank,
            "requires_api": bool(args.mode == "research"),
            "enabled": bool(args.mode == "research"),
            "description": "Generate the LLM-authored reusable human identity bank for person synthesis.",
        },
        {
            "id": 10,
            "name": "Top Up Persons" if extend else "Generate Persons (Procedural)",
            "script": "prepare_continuation_entities.py" if extend else "generate_persons_procedural.py",
            "args": _continuation_entity_args("persons", int(args.n_persons)) if extend else ["--base-dir", str(BASE_DIR), "--target", str(args.n_persons), "--seed", str(args.seed), "--mode", str(args.mode)],
            "check": lambda bd: _check_json_list_exact_count(bd / "entities" / "persons.json", int(args.n_persons)),
            "requires_api": False,
            "rerun_on_extend": True,
            "description": "Append missing people for a continuation run while preserving existing IDs." if extend else "Generate people from the research identity bank or the debug procedural fallback.",
        },
        {
            "id": 18,
            "name": "Generate Company Lexicon",
            "script": "generate_bootstrap_artifacts_api.py",
            "args": [
                "--artifact", "company_lexicon",
                "--base-dir", str(BASE_DIR),
                "--mode", str(args.mode),
                "--n-movies", str(args.n_movies),
                "--n-persons", str(args.n_persons),
                "--n-companies", str(args.n_companies),
                "--n-keywords", str(args.n_keywords),
                "--n-titles", str(args.n_titles),
                "--start-year", str(args.start_year),
                "--end-year", str(args.end_year),
            ] + _step_model_args(args, step_id=18, script="generate_bootstrap_artifacts_api.py"),
            "check": _check_company_lexicon,
            "requires_api": bool(args.mode == "research"),
            "enabled": bool(args.mode == "research"),
            "description": "Generate the LLM-authored reusable company naming lexicon.",
        },
        {
            "id": 20,
            "name": "Top Up Companies" if extend else "Generate Companies (Procedural)",
            "script": "prepare_continuation_entities.py" if extend else "generate_companies_procedural.py",
            "args": _continuation_entity_args("companies", int(args.n_companies)) if extend else ["--base-dir", str(BASE_DIR), "--target", str(args.n_companies), "--seed", str(args.seed), "--mode", str(args.mode)],
            "check": lambda bd: _check_json_list_exact_count(bd / "entities" / "companies.json", int(args.n_companies)),
            "requires_api": False,
            "rerun_on_extend": True,
            "description": "Append missing companies for a continuation run while preserving existing IDs." if extend else "Generate companies from the research lexicon or the debug procedural fallback.",
        },
        {
            "id": 28,
            "name": "Generate Keyword Seed Bank",
            "script": "generate_bootstrap_artifacts_api.py",
            "args": [
                "--artifact", "keyword_seed_bank",
                "--base-dir", str(BASE_DIR),
                "--mode", str(args.mode),
                "--n-movies", str(args.n_movies),
                "--n-persons", str(args.n_persons),
                "--n-companies", str(args.n_companies),
                "--n-keywords", str(args.n_keywords),
                "--n-titles", str(args.n_titles),
                "--start-year", str(args.start_year),
                "--end-year", str(args.end_year),
            ] + _step_model_args(args, step_id=28, script="generate_bootstrap_artifacts_api.py"),
            "check": _check_keyword_seed_bank,
            "requires_api": bool(args.mode == "research"),
            "enabled": bool(args.mode == "research"),
            "description": "Generate the LLM-authored reusable keyword seed bank.",
        },
        {
            "id": 30,
            "name": "Top Up Keywords" if extend else "Generate Keywords (Procedural)",
            "script": "prepare_continuation_entities.py" if extend else "generate_keywords_procedural.py",
            "args": _continuation_entity_args("keywords", int(args.n_keywords)) if extend else ["--base-dir", str(BASE_DIR), "--target", str(args.n_keywords), "--seed", str(args.seed), "--mode", str(args.mode)],
            "check": lambda bd: _check_json_list_exact_count(bd / "entities" / "keywords.json", int(args.n_keywords)),
            "requires_api": False,
            "rerun_on_extend": True,
            "description": "Append missing keywords for a continuation run while preserving existing IDs." if extend else "Generate keywords from the research seed bank or the debug procedural fallback.",
        },
        {
            "id": 40,
            "name": "Enrich Persons (LLM)",
            "script": "enrich_persons_api.py",
            "args": ["--base-dir", str(BASE_DIR), "--auto", "--mode", str(args.mode)]
            + _step_model_args(args, step_id=40, script="enrich_persons_api.py"),
            "check": lambda bd: _check_persons_enriched(bd, expected_count=int(args.n_persons)),
            "requires_api": True,
            "rerun_on_extend": True,
            "description": "LLM-enrich procedural persons with bios, styles, and genre affinities.",
        },
        {
            "id": 48,
            "name": "Generate Character Identity Bank",
            "script": "generate_bootstrap_artifacts_api.py",
            "args": [
                "--artifact", "character_identity_bank",
                "--base-dir", str(BASE_DIR),
                "--mode", str(args.mode),
                "--n-movies", str(args.n_movies),
                "--n-persons", str(args.n_persons),
                "--n-companies", str(args.n_companies),
                "--n-keywords", str(args.n_keywords),
                "--n-titles", str(args.n_titles),
                "--start-year", str(args.start_year),
                "--end-year", str(args.end_year),
            ] + _step_model_args(args, step_id=48, script="generate_bootstrap_artifacts_api.py"),
            "check": _check_character_identity_bank,
            "requires_api": bool(args.mode == "research"),
            "enabled": bool(args.mode == "research"),
            "description": "Generate the LLM-authored reusable character identity bank.",
        },
        {
            "id": 50,
            "name": "Top Up Character Bank" if extend else "Generate Character Bank",
            "script": "prepare_continuation_entities.py" if extend else "generate_character_bank.py",
            "args": _continuation_entity_args("characters", int(args.n_characters)) if extend else ["--base-dir", str(BASE_DIR), "--target", str(args.n_characters), "--seed", str(args.seed), "--mode", str(args.mode)],
            "check": lambda bd: _check_csv_exact_count(bd / "entities" / "character_bank.csv", int(args.n_characters)),
            "requires_api": False,
            "rerun_on_extend": True,
            "description": "Append missing character names for a continuation run." if extend else "Generate character names from the research bank or the debug procedural fallback.",
        },
        {
            "id": 56,
            "name": "Generate Temporal Regime Plan",
            "script": "generate_bootstrap_artifacts_api.py",
            "args": [
                "--artifact", "temporal_regime_plan",
                "--base-dir", str(BASE_DIR),
                "--mode", str(args.mode),
                "--n-movies", str(args.n_movies),
                "--n-persons", str(args.n_persons),
                "--n-companies", str(args.n_companies),
                "--n-keywords", str(args.n_keywords),
                "--n-titles", str(args.n_titles),
                "--start-year", str(args.start_year),
                "--end-year", str(args.end_year),
            ] + _step_model_args(args, step_id=56, script="generate_bootstrap_artifacts_api.py"),
            "check": lambda bd: _check_temporal_regime_plan(bd, start_year=args.start_year, end_year=args.end_year),
            "requires_api": bool(args.mode == "research"),
            "enabled": bool(args.mode == "research"),
            "rerun_on_extend": True,
            "description": "Generate the reusable temporal regime plan for arbitrary year spans.",
        },
        {
            "id": 58,
            "name": "Generate Title Grammar Bank",
            "script": "generate_bootstrap_artifacts_api.py",
            "args": [
                "--artifact", "title_grammar_bank",
                "--base-dir", str(BASE_DIR),
                "--mode", str(args.mode),
                "--n-movies", str(args.n_movies),
                "--n-persons", str(args.n_persons),
                "--n-companies", str(args.n_companies),
                "--n-keywords", str(args.n_keywords),
                "--n-titles", str(args.n_titles),
                "--start-year", str(args.start_year),
                "--end-year", str(args.end_year),
            ] + _step_model_args(args, step_id=58, script="generate_bootstrap_artifacts_api.py"),
            "check": lambda bd: _check_title_grammar_bank(
                bd,
                target_count=int(args.n_titles),
                start_year=args.start_year,
                end_year=args.end_year,
            ),
            "requires_api": bool(args.mode == "research"),
            "enabled": bool(args.mode == "research"),
            "rerun_on_extend": True,
            "description": "Generate the reusable title grammar and tagline bank.",
        },
        {
            "id": 60,
            "name": "Top Up Title Bank" if extend else "Generate Title Bank",
            "script": "prepare_continuation_entities.py" if extend else "topup_title_bank.py",
            "args": _continuation_entity_args("titles", int(args.n_titles)) if extend else (
                ["--base-dir", str(BASE_DIR), "--target-count", str(args.n_titles), "--seed", str(args.seed), "--mode", str(args.mode)]
                + (["--start-year", str(args.start_year), "--end-year", str(args.end_year)] if args.start_year is not None and args.end_year is not None else [])
            ),
            "check": lambda bd: _check_title_bank(
                bd,
                target_count=int(args.n_titles),
                start_year=args.start_year,
                end_year=args.end_year,
                allow_prior_years=extend,
            ),
            "requires_api": False,
            "rerun_on_extend": True,
            "description": "Append missing future-range titles while preserving existing title rows." if extend else "Generate titles and distribute them across the requested year span.",
        },
        {
            "id": 70,
            "name": "Generate Latent Variables",
            "script": "generate_latent_vars_api.py",
            "args": ["--auto", "--mode", str(args.mode)]
            + _step_model_args(args, step_id=70, script="generate_latent_vars_api.py"),
            "check": lambda bd: _check_latent(
                bd,
                expected_persons=int(args.n_persons),
                expected_companies=int(args.n_companies),
            ),
            "requires_api": True,
            "rerun_on_extend": True,
            "description": "LLM-generate latent variables for persons and companies.",
        },
        {
            "id": 72,
            "name": "Generate World Policy",
            "script": "generate_world_policy_api.py",
            "args": ["--base-dir", str(BASE_DIR), "--auto"]
            + _step_model_args(args, step_id=72, script="generate_world_policy_api.py"),
            "check": lambda bd: _check_world_policy(bd, start_year=args.start_year, end_year=args.end_year),
            "requires_api": True,
            "enabled": bool(args.enable_llm_world_policy),
            "rerun_on_extend": True,
            "description": "Generate structured world-policy biases for correlated movie and graph generation.",
        },
        {
            "id": 74,
            "name": "Generate Year Slate Plan",
            "script": "generate_year_slate_plan_api.py",
            "args": ["--base-dir", str(BASE_DIR), "--auto"]
            + _step_model_args(args, step_id=74, script="generate_year_slate_plan_api.py"),
            "check": lambda bd: _check_year_slate_plan(bd, start_year=args.start_year, end_year=args.end_year),
            "requires_api": True,
            "enabled": bool(args.enable_llm_year_slates),
            "rerun_on_extend": True,
            "description": "Generate reusable year-bucket x market x tier slate plans for drift, release pressure, and sequel appetite.",
        },
        {
            "id": 80,
            "name": "Generate Edge Graph",
            "script": "generate_edges_hybrid.py",
            "args": (
                ["--base-dir", str(BASE_DIR)]
                + (["--force-legacy"] if getattr(args, "force_legacy_graph", False) else [])
                + (["--force-scalable"] if getattr(args, "force_scalable_graph", False) else [])
                + (["--use-world-policy"] if args.enable_llm_world_policy else [])
                + (["--skip-diagnostic-cold-edges"] if getattr(args, "skip_diagnostic_cold_edges", False) else [])
            ),
            "check": lambda bd: _check_edges(bd, expected_persons=int(args.n_persons)),
            "requires_api": False,
            "rerun_on_extend": True,
            "description": "Build the initial graph runtime from entities and latents.",
        },
        {
            "id": 90,
            "name": "Convert Entities to CSV",
            "script": "entities_to_csv.py",
            "args": [str(BASE_DIR), "convert"],
            "check": lambda bd: _check_entities_csv(
                bd,
                expected_persons=int(args.n_persons),
                expected_companies=int(args.n_companies),
                expected_keywords=int(args.n_keywords),
            ),
            "requires_api": False,
            "rerun_on_extend": True,
            "description": "Convert JSON entities into the CSV/Arrow inputs used by assembly.",
        },
        {
            "id": 92,
            "name": "Generate Company Financial Profiles",
            "script": "generate_company_financial_profiles.py",
            "args": ["--base-dir", str(BASE_DIR), "--mode", str(args.mode)],
            "check": lambda bd: _check_company_financial_profiles(bd, expected_companies=int(args.n_companies)),
            "requires_api": False,
            "rerun_on_extend": True,
            "description": "Generate baseline company finance profiles used during movie generation.",
        },
        {
            "id": 95,
            "name": "Enrich Keyword Genres",
            "script": "generate_keyword_genres.py",
            "args": ["--base-dir", str(BASE_DIR), "--auto"]
            + _step_model_args(args, step_id=95, script="generate_keyword_genres.py"),
            "check": _check_keyword_genres,
            "requires_api": True,
            "rerun_on_extend": True,
            "description": "Fill any missing keyword genre metadata.",
        },
        {
            "id": 97,
            "name": "Generate Concept Packs",
            "script": "generate_concept_packs_api.py",
            "args": ["--base-dir", str(BASE_DIR), "--auto", "--n-movies", str(args.n_movies)]
            + _step_model_args(args, step_id=97, script="generate_concept_packs_api.py"),
            "check": _check_concept_packs,
            "requires_api": True,
            "enabled": bool(args.enable_llm_concept_packs),
            "force_on_extend": True,
            "description": "Generate reusable LLM concept packs for year-bucket x genre x tier x market slots.",
        },
        {
            "id": 96,
            "name": "Generate Keyword Motif Bank",
            "script": "generate_keyword_motif_bank_api.py",
            "args": ["--base-dir", str(BASE_DIR), "--auto"]
            + _step_model_args(args, step_id=96, script="generate_keyword_motif_bank_api.py"),
            "check": _check_keyword_motif_bank,
            "requires_api": True,
            "enabled": bool(args.enable_llm_keyword_motifs),
            "force_on_extend": True,
            "description": "Generate hierarchical keyword motif metadata and enrich keyword.csv with motif families and recurrence signals.",
        },
        {
            "id": 98,
            "name": "Generate Franchise Bibles",
            "script": "generate_franchise_bibles_api.py",
            "args": ["--base-dir", str(BASE_DIR), "--n-movies", str(args.n_movies)]
            + _step_model_args(args, step_id=98, script="generate_franchise_bibles_api.py"),
            "check": _check_franchise_bibles,
            "requires_api": True,
            "enabled": bool(args.enable_llm_world_policy or args.enable_llm_concept_packs or args.enable_llm_keyword_motifs),
            "force_on_extend": True,
            "description": "Generate franchise continuity bibles for the sequel slots that will materialize in this run.",
        },
        {
            "id": 100,
            "name": "Generate Movies",
            "script": "generate_movies.py",
            "args": (
                ["--base_dir", str(BASE_DIR), "--n_movies", str(args.n_movies)]
                + (["--start_year", str(args.start_year)] if args.start_year is not None else [])
                + (["--end_year", str(args.end_year)] if args.end_year is not None else [])
                + (["--benchmark-mode"] if args.benchmark_mode else [])
                + (["--resume-step100"] if args.resume_step100 else [])
                + (["--extend-step100"] if args.extend_step100 else [])
                + (["--reset-step100-resume"] if args.reset_step100_resume else [])
                + ([] if args.enable_llm_evolution else ["--disable_llm_evolution"])
                + ([] if args.enable_llm_critic else ["--disable_llm_critic"])
                + ([] if args.enable_llm_world_policy else ["--disable_llm_world_policy"])
                + ([] if args.enable_llm_concept_packs else ["--disable_llm_concept_packs"])
                + ([] if args.enable_llm_year_slates else ["--disable_llm_year_slates"])
                + ([] if args.enable_llm_keyword_motifs else ["--disable_llm_keyword_motifs"])
                + ([] if args.enable_llm_rerank else ["--disable_llm_rerank"])
                + ([] if args.enable_llm_keyword_rerank else ["--disable_llm_keyword_rerank"])
                + (["--rerank_budget_movies", str(args.rerank_budget_movies)] if args.rerank_budget_movies is not None else [])
                + (["--keyword_rerank_budget_movies", str(args.keyword_rerank_budget_movies)] if args.keyword_rerank_budget_movies is not None else [])
                + _step_model_args(args, step_id=100, script="generate_movies.py", option="--llm_model")
            ),
            "check": lambda bd: _check_movies(bd, expected_movies=int(args.n_movies)),
            "requires_api": bool(
                args.enable_llm_evolution
                or args.enable_llm_critic
                or args.enable_llm_rerank
                or args.enable_llm_keyword_rerank
            ),
            "rerun_on_extend": True,
            "description": "Run full movie assembly and export all primary tables.",
        },
        {
            "id": 110,
            "name": "Generate Plot Summaries",
            "script": "generate_plot_summaries_api.py",
            "args": ["--base-dir", str(BASE_DIR), "--auto"]
            + _step_model_args(args, step_id=110, script="generate_plot_summaries_api.py"),
            "check": _check_plots,
            "requires_api": True,
            "description": "Generate plot summaries for the assembled movie table.",
        },
        {
            "id": 120,
            "name": "Generate TV Summaries",
            "script": "generate_tv_summaries.py",
            "args": ["--base-dir", str(BASE_DIR)]
            + _step_model_args(args, step_id=120, script="generate_tv_summaries.py", option="--tier1-model")
            + _step_model_args(args, step_id=120, script="generate_tv_summaries.py", option="--tier2-model"),
            "check": _check_tv_summaries,
            "requires_api": True,
            "description": "Generate TV series and episode summaries for the TV tables.",
        },
        {
            "id": 130,
            "name": "Export IMDb Schema",
            "script": "export_imdb_schema.py",
            "args": [
                "--base-dir", str(BASE_DIR),
                "--out-dir", str(BASE_DIR / IMDB_EXPORT_DIRNAME),
                "--strict-job",
                "--no-include-extras",
                "--build-duckdb",
                "--duckdb-out", str(BASE_DIR / IMDB_EXPORT_DIRNAME / "imdb.duckdb"),
                "--overwrite-duckdb",
            ],
            "check": _check_imdb_export,
            "requires_api": False,
            "description": "Convert the finished dataset into the strict JOB/IMDb CSV schema bundle and matching DuckDB file.",
        },
    ]


def _run_step(step: dict, args: argparse.Namespace) -> bool:
    script_path = BASE_DIR / step["script"]
    if not script_path.exists():
        print(f"  ERROR: missing script: {script_path.name}")
        return False

    cmd = [sys.executable, str(script_path)] + list(step["args"])
    print(f"\n{'=' * 72}")
    print(f"STEP {step['id']}: {step['name']}")
    print(step["description"])
    print("CMD:", " ".join(cmd))
    print(f"{'=' * 72}")

    t0 = time.time()
    try:
        env = os.environ.copy()
        env["DATA_SYS_RUN_ID"] = str(getattr(args, "run_id", ""))
        env["DATA_SYS_DECISION_LOG"] = str(decision_log_path_for_run(BASE_DIR, str(getattr(args, "run_id", "run"))))
        env["DATA_SYS_DECISION_LOG_LATEST"] = str(decision_log_path(BASE_DIR))
        env["DATA_SYS_MOVIE_PROGRESS_LOG"] = str(movie_progress_log_path_for_run(BASE_DIR, str(getattr(args, "run_id", "run"))))
        env["DATA_SYS_MOVIE_PROGRESS_LOG_LATEST"] = str(movie_progress_log_path(BASE_DIR))
        env["DATA_SYS_LLM_USAGE_LOG"] = str(llm_usage_log_path_for_run(BASE_DIR, str(getattr(args, "run_id", "run"))))
        env["DATA_SYS_LLM_USAGE_LOG_LATEST"] = str(llm_usage_log_path(BASE_DIR))
        env["DATA_SYS_RESEARCH_AUDIT"] = str(research_audit_path_for_run(BASE_DIR, str(getattr(args, "run_id", "run"))))
        env["DATA_SYS_RESEARCH_AUDIT_LATEST"] = str(research_audit_path(BASE_DIR))
        env["DATA_SYS_STEP_ID"] = str(step["id"])
        env["DATA_SYS_STEP_NAME"] = str(step["name"])
        env["DATA_SYS_PIPELINE_MODE"] = str(args.mode)
        env["DATA_SYS_START_YEAR"] = str(args.start_year)
        env["DATA_SYS_END_YEAR"] = str(args.end_year)
        os.environ["DATA_SYS_STEP_ID"] = str(step["id"])
        os.environ["DATA_SYS_STEP_NAME"] = str(step["name"])
        clear_step_fallback_hits(step["id"])
        result = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            timeout=args.timeout,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {args.timeout}s")
        record_pipeline_step(step["id"], step["name"], status="timeout", ok=False, enabled=True, detail=f"timeout after {args.timeout}s")
        return False
    except Exception as exc:
        print(f"  ERROR: {exc}")
        record_pipeline_step(step["id"], step["name"], status="error", ok=False, enabled=True, detail=str(exc))
        return False

    elapsed = time.time() - t0
    if result.returncode != 0:
        _record_timing(step, elapsed, False)
        record_pipeline_step(
            step["id"],
            step["name"],
            status="failed",
            ok=False,
            enabled=True,
            elapsed_seconds=elapsed,
            detail=f"exit_code={result.returncode}",
        )
        print(f"  FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
        return False

    _record_timing(step, elapsed, True)
    record_pipeline_step(step["id"], step["name"], status="ok", ok=True, enabled=True, elapsed_seconds=elapsed)
    assert_no_recorded_fallbacks(args.mode)
    print(f"  OK ({elapsed:.1f}s)")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Mirage synthetic IMDb/JOB pipeline runner")
    parser.add_argument("--profile", default=None, help="Load defaults from local_run_profiles/<profile>.env")
    parser.add_argument("--profile-file", default=None, help="Load defaults from an explicit KEY=VALUE profile file")
    parser.add_argument("--n-movies", type=int, default=None, dest="n_movies", help="Number of movies to generate")
    parser.add_argument("--start-year", type=int, default=None, help="Start year for title-bank distribution")
    parser.add_argument("--end-year", type=int, default=None, help="End year for title-bank distribution")
    parser.add_argument("--n-persons", type=int, default=None, dest="n_persons")
    parser.add_argument("--n-companies", type=int, default=None, dest="n_companies")
    parser.add_argument("--n-keywords", type=int, default=None, dest="n_keywords")
    parser.add_argument("--n-characters", type=int, default=None, dest="n_characters")
    parser.add_argument("--n-titles", type=int, default=None, dest="n_titles")
    parser.add_argument(
        "--model",
        default=None,
        help="Override the general LLM model for latent/movie/text steps. Heavy bulk artifact steps keep their safer routing unless you set the dedicated flags below.",
    )
    parser.add_argument(
        "--bootstrap-model",
        default=None,
        help="Override the model used by bootstrap artifact steps (4/8/18/28/48/56/58).",
    )
    parser.add_argument(
        "--planning-model",
        default=None,
        help="Override the model used by the high-level world-policy planning step (72).",
    )
    parser.add_argument(
        "--bulk-artifact-model",
        default=model_for_role("artifact_bulk"),
        help="Override the model used by heavy fan-out artifact steps (74/95/96/97/98).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-llm-evolution", action="store_true", default=None)
    parser.add_argument("--disable-llm-evolution", action="store_true", default=None)
    parser.add_argument("--enable-llm-critic", action="store_true", default=None)
    parser.add_argument("--disable-llm-critic", action="store_true", default=None)
    parser.add_argument("--enable-llm-world-policy", action="store_true", default=None)
    parser.add_argument("--disable-llm-world-policy", action="store_true", default=None)
    parser.add_argument("--enable-llm-concept-packs", action="store_true", default=None)
    parser.add_argument("--disable-llm-concept-packs", action="store_true", default=None)
    parser.add_argument("--enable-llm-year-slates", action="store_true", default=None)
    parser.add_argument("--disable-llm-year-slates", action="store_true", default=None)
    parser.add_argument("--enable-llm-keyword-motifs", action="store_true", default=None)
    parser.add_argument("--disable-llm-keyword-motifs", action="store_true", default=None)
    parser.add_argument("--enable-llm-rerank", action="store_true", default=None)
    parser.add_argument("--disable-llm-rerank", action="store_true", default=None)
    parser.add_argument("--enable-llm-keyword-rerank", action="store_true", default=None)
    parser.add_argument("--disable-llm-keyword-rerank", action="store_true", default=None)
    parser.add_argument("--force-scalable-graph", action="store_true", help="Force the scalable graph compiler for step 80.")
    parser.add_argument("--force-legacy-graph", action="store_true", help="Force the legacy in-memory graph builder for step 80.")
    parser.add_argument(
        "--skip-diagnostic-cold-edges",
        action="store_true",
        help="Skip precomputed P-C/C-C diagnostic cold graph streams for benchmark/lab runs.",
    )
    parser.add_argument(
        "--benchmark-mode",
        action="store_true",
        help="Pass benchmark-first mode into step 100 so derivative exports are skipped.",
    )
    parser.add_argument("--rerank-budget-movies", type=int, default=None)
    parser.add_argument("--keyword-rerank-budget-movies", type=int, default=None)
    parser.add_argument(
        "--resume-step100",
        action="store_true",
        help="Explicitly resume step 100 from the last completed year boundary using _step100_resume/. This guarantees continuation, not byte-identical replay.",
    )
    parser.add_argument(
        "--extend-step100",
        action="store_true",
        help="Append a new extension plan to an existing Step100 workspace. --n-movies is cumulative; --start-year/--end-year describe the extension window.",
    )
    parser.add_argument(
        "--company-lifecycle-policy",
        choices=("balanced", "preserve", "new-era"),
        default="balanced",
        help="Balanced company turnover policy for --extend-step100 entity top-up.",
    )
    parser.add_argument(
        "--reset-step100-resume",
        action="store_true",
        help="Discard prior _step100_resume artifacts before starting a fresh step 100 run.",
    )
    parser.add_argument("--from-step", type=int, default=4)
    parser.add_argument("--until-step", type=int, default=None)
    parser.add_argument("--only", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="Delete generated outputs before running")
    parser.add_argument(
        "--fresh-preserve-entities",
        action="store_true",
        help="Delete generated outputs but keep the current entities/ checkpoint for seeded reruns.",
    )
    parser.add_argument("--clear-checkpoint", action="store_true")
    parser.add_argument("--timeout", type=int, default=None, help="Per-step timeout in seconds")
    parser.add_argument("--mode", choices=("research", "debug"), default="research")
    args = parser.parse_args()

    loaded_profile = apply_run_profile(args, BASE_DIR)
    if args.n_movies is None:
        parser.error("--n-movies is required unless --profile/--profile-file supplies N_MOVIES")

    if args.fresh_preserve_entities:
        args.fresh = True

    if args.reset_step100_resume and (args.resume_step100 or args.extend_step100):
        raise ValueError("Use --reset-step100-resume by itself; it cannot be combined with resume or extension")

    if args.force_scalable_graph and args.force_legacy_graph:
        raise ValueError("Choose at most one of --force-scalable-graph and --force-legacy-graph")

    _resolve_targets(args)
    _resolve_llm_flags(args)

    print("=" * 72)
    print("MIRAGE PIPELINE RUNNER")
    print("=" * 72)
    print(f"Base dir:      {BASE_DIR}")
    if loaded_profile is not None:
        print(f"Profile:       {loaded_profile.name} ({loaded_profile.path})")
    print(f"Movies:        {args.n_movies}")
    print(f"Year span:     {args.start_year if args.start_year is not None else '(default)'}"
          f"{' -> ' + str(args.end_year) if args.end_year is not None else ''}")
    print(f"Persons:       {args.n_persons}")
    print(f"Companies:     {args.n_companies}")
    print(f"Keywords:      {args.n_keywords}")
    print(f"Characters:    {args.n_characters}")
    print(f"Titles:        {args.n_titles}")
    print(f"Mode:          {args.mode}")
    print(f"Model:         {args.model or '(provider default)'}")
    print(f"LLM evolution: {args.enable_llm_evolution}")
    print(f"LLM critic:    {args.enable_llm_critic}")
    print(f"World policy:  {args.enable_llm_world_policy}")
    print(f"Year slates:   {args.enable_llm_year_slates}")
    print(f"Concept packs: {args.enable_llm_concept_packs}")
    print(f"Keyword bank:  {args.enable_llm_keyword_motifs}")
    print(f"Benchmark:     {args.benchmark_mode}")
    print(f"LLM rerank:    {args.enable_llm_rerank}")
    print(f"Keyword rerank:{args.enable_llm_keyword_rerank}")
    print(f"Force scalable graph: {args.force_scalable_graph}")
    print(f"Force legacy graph:   {args.force_legacy_graph}")
    print(f"Step100 extension:    {args.extend_step100}")
    print(f"Fresh preserve entities: {args.fresh_preserve_entities}")
    print(f"Rerank budget: {args.rerank_budget_movies if args.rerank_budget_movies is not None else '(auto)'}")
    print(f"KW rerank bdg: {args.keyword_rerank_budget_movies if args.keyword_rerank_budget_movies is not None else '(auto)'}")
    print("=" * 72)

    if args.fresh and args.fresh_preserve_entities:
        print("Resetting generated outputs while preserving entities checkpoint...")
        _reset_generated_outputs(preserve_entities=True, preserve_artifacts=True)
    elif args.fresh:
        print("Resetting generated outputs...")
        _reset_generated_outputs()
    elif args.clear_checkpoint:
        _clear_checkpoint()

    state = _load_checkpoint()
    completed = set(state.get("completed_steps", []))
    if state.get("started_at") is None:
        state["started_at"] = datetime.now().isoformat()
    state["run_id"] = _sanitize_run_id(str(state.get("run_id") or _derive_run_id(state.get("started_at"))))
    args.run_id = str(state["run_id"])
    _save_checkpoint(state)
    print(f"Active run id: {args.run_id}")

    os.environ["DATA_SYS_RUN_ID"] = str(args.run_id)
    os.environ["DATA_SYS_PIPELINE_MODE"] = str(args.mode)
    os.environ["DATA_SYS_START_YEAR"] = str(args.start_year)
    os.environ["DATA_SYS_END_YEAR"] = str(args.end_year)
    os.environ["DATA_SYS_RESEARCH_AUDIT"] = str(research_audit_path_for_run(BASE_DIR, args.run_id))
    os.environ["DATA_SYS_RESEARCH_AUDIT_LATEST"] = str(research_audit_path(BASE_DIR))
    initialize_research_audit(
        run_id=args.run_id,
        mode=str(args.mode),
        start_year=args.start_year,
        end_year=args.end_year,
        model_override=args.model,
        metadata={
            "n_movies": int(args.n_movies),
            "n_persons": int(args.n_persons),
            "n_companies": int(args.n_companies),
            "n_keywords": int(args.n_keywords),
            "n_characters": int(args.n_characters),
            "n_titles": int(args.n_titles),
            "fresh_requested": bool(args.fresh),
            "fresh_preserve_entities_requested": bool(args.fresh_preserve_entities),
            "extend_step100_requested": bool(args.extend_step100),
            "clear_checkpoint_requested": bool(args.clear_checkpoint),
            "general_model_override": _clean_model_name(args.model),
            "bootstrap_model_override": _clean_model_name(args.bootstrap_model),
            "planning_model_override": _clean_model_name(args.planning_model),
            "bulk_artifact_model_override": _clean_model_name(args.bulk_artifact_model),
            "profile": loaded_profile.name if loaded_profile is not None else None,
            "profile_path": str(loaded_profile.path) if loaded_profile is not None else None,
            "from_step": int(args.from_step),
            "until_step": int(args.until_step) if args.until_step is not None else None,
            "only_step": int(args.only) if args.only is not None else None,
            "signoff_run_candidate": bool(args.fresh and args.from_step <= 4 and args.only is None),
            "llm_critic_enabled": bool(args.enable_llm_critic),
            "llm_evolution_enabled": bool(args.enable_llm_evolution),
            "llm_world_policy_enabled": bool(args.enable_llm_world_policy),
            "llm_concept_packs_enabled": bool(args.enable_llm_concept_packs),
            "llm_year_slates_enabled": bool(args.enable_llm_year_slates),
            "llm_keyword_motifs_enabled": bool(args.enable_llm_keyword_motifs),
            "llm_rerank_enabled": bool(args.enable_llm_rerank),
            "llm_keyword_rerank_enabled": bool(args.enable_llm_keyword_rerank),
        },
    )

    steps = sorted(_build_steps(args), key=lambda step: step["id"])
    if args.only is not None:
        steps = [step for step in steps if step["id"] == args.only]
    else:
        steps = [step for step in steps if step["id"] >= args.from_step]
    if args.until_step is not None:
        steps = [step for step in steps if step["id"] <= args.until_step]

    succeeded = 0
    skipped = 0
    failed = 0
    t0 = time.time()

    for step in steps:
        sid = step["id"]
        extension_sensitive = bool(args.extend_step100 and (step.get("rerun_on_extend") or step.get("force_on_extend")))
        force_this_step = bool(args.force or (args.extend_step100 and step.get("force_on_extend")))
        if step.get("enabled") is False:
            print(f"\nStep {sid} ({step['name']}): feature disabled, skipping")
            record_pipeline_step(sid, step["name"], status="disabled", ok=True, skipped=True, enabled=False, detail="feature disabled")
            skipped += 1
            continue
        if sid in completed and not args.force and not extension_sensitive:
            print(f"\nStep {sid} ({step['name']}): checkpoint says complete, skipping")
            clear_step_fallback_hits(sid)
            record_pipeline_step(sid, step["name"], status="checkpoint_skip", ok=True, skipped=True, enabled=True, detail="checkpoint marked complete")
            skipped += 1
            continue

        if not force_this_step and step["check"](BASE_DIR):
            print(f"\nStep {sid} ({step['name']}): outputs already present, skipping")
            clear_step_fallback_hits(sid)
            completed.add(sid)
            state["completed_steps"] = sorted(completed)
            _save_checkpoint(state)
            record_pipeline_step(sid, step["name"], status="existing_outputs_skip", ok=True, skipped=True, enabled=True, detail="outputs already present")
            skipped += 1
            continue

        ok = _run_step(step, args)
        if not ok:
            failed += 1
            print(f"\nPipeline halted at step {sid}. Resume with --from-step {sid}.")
            break

        completed.add(sid)
        state["completed_steps"] = sorted(completed)
        _save_checkpoint(state)
        succeeded += 1

    elapsed = time.time() - t0
    if failed == 0:
        assert_no_recorded_fallbacks(args.mode)
    print(f"\n{'=' * 72}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 72}")
    print(f"Succeeded: {succeeded}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")
    print(f"Time:      {elapsed:.1f}s")
    print(f"{'=' * 72}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
