from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ContinuationSummary:
    base_dir: Path
    resume_dir: Path
    manifest_path: Path
    plan_path: Path
    status: str
    movie_count: int
    produced_movie_count: int
    last_completed_year: int | None
    last_completed_sequence_index: int | None
    plan_count: int
    max_movie_id: int
    plan_year_min: int | None
    plan_year_max: int | None

    def additional_movie_count(self, target_movie_count: int) -> int:
        return max(0, int(target_movie_count) - int(self.plan_count))


@dataclass(frozen=True)
class ExtensionPlanResult:
    appended: int
    total_plan_count: int
    first_seq_idx: int | None
    last_seq_idx: int | None
    first_movie_id: int | None
    last_movie_id: int | None
    year_min: int | None
    year_max: int | None
    title_assignments: int
    compositional_title_movies: int
    reranked: int


def _safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return payload


def _iter_plan_records(plan_path: Path) -> list[dict[str, Any]]:
    if not plan_path.exists():
        return []
    out: list[dict[str, Any]] = []
    with plan_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            if isinstance(payload, dict):
                out.append(payload)
    return out


def load_continuation_summary(base_dir: str | Path) -> ContinuationSummary:
    base = Path(base_dir)
    resume_dir = base / "_step100_resume"
    manifest_path = resume_dir / "manifest.json"
    plan_path = resume_dir / "movie_plan.jsonl"
    if not manifest_path.exists():
        raise RuntimeError(f"No Step100 manifest found at {manifest_path}")

    manifest = _load_json(manifest_path)
    plan_records = _iter_plan_records(plan_path)
    years: list[int] = []
    movie_ids: list[int] = []
    for record in plan_records:
        year = _safe_int(record.get("year"))
        movie_id = _safe_int(record.get("movie_id"))
        if year is not None:
            years.append(year)
        if movie_id is not None:
            movie_ids.append(movie_id)

    return ContinuationSummary(
        base_dir=base,
        resume_dir=resume_dir,
        manifest_path=manifest_path,
        plan_path=plan_path,
        status=str(manifest.get("status", "")),
        movie_count=int(manifest.get("movie_count", 0) or 0),
        produced_movie_count=int(manifest.get("produced_movie_count", 0) or 0),
        last_completed_year=_safe_int(manifest.get("last_completed_year")),
        last_completed_sequence_index=_safe_int(manifest.get("last_completed_sequence_index")),
        plan_count=len(plan_records),
        max_movie_id=max(movie_ids) if movie_ids else 0,
        plan_year_min=min(years) if years else None,
        plan_year_max=max(years) if years else None,
    )


def validate_extension_request(summary: ContinuationSummary, *, target_movie_count: int) -> int:
    target = int(target_movie_count)
    if summary.plan_count <= 0:
        raise RuntimeError("Cannot extend Step100: existing movie_plan.jsonl is empty or missing.")
    if summary.produced_movie_count <= 0:
        raise RuntimeError("Cannot extend Step100: no completed movies are recorded in the manifest.")
    if target <= summary.plan_count:
        raise RuntimeError(
            f"Continuation target must exceed existing plan count "
            f"({target} <= {summary.plan_count})."
        )
    return target - summary.plan_count


def _clean_title_row(row: pd.Series, row_idx: int) -> dict[str, Any] | None:
    title = str(row.get("_title_clean", row.get("title", "")) or "").strip()
    if not title or title.lower() == "nan":
        return None
    tagline = str(row.get("_tagline_clean", row.get("tagline", "")) or "").strip()
    if tagline.lower() == "nan":
        tagline = ""
    year = _safe_int(row.get("year"))
    if year is None:
        return None
    return {
        "title": title,
        "tagline": tagline,
        "year": int(year),
        "genre_hint": str(row.get("genre_hint", "") or ""),
        "award_contender": bool(row.get("award_contender", False))
        if not (isinstance(row.get("award_contender", None), float) and pd.isna(row.get("award_contender")))
        else False,
        "_tb_row_idx": int(row.get("_tb_row_idx", row_idx) or row_idx),
        "_title_ok_static": bool(row.get("_title_ok_static", True)),
        "_tagline_ok_static": bool(row.get("_tagline_ok_static", bool(tagline))),
    }


def _available_title_assignments(
    world: Any,
    *,
    extension_start_year: int,
    extension_end_year: int,
    limit: int,
) -> list[dict[str, Any]]:
    title_bank = getattr(world, "title_bank", None)
    if not isinstance(title_bank, pd.DataFrame) or title_bank.empty or "year" not in title_bank.columns:
        return []

    tb = title_bank.copy()
    years = pd.to_numeric(tb["year"], errors="coerce")
    tb = tb.loc[years.between(int(extension_start_year), int(extension_end_year), inclusive="both")].copy()
    if tb.empty:
        return []
    tb = tb.sample(frac=1, random_state=int(getattr(world, "seed", 42)) + 43_001).reset_index(drop=True)

    used_titles = {str(item).casefold() for item in set(getattr(world, "used_titles", set()) or set())}
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row_idx, row in tb.iterrows():
        assignment = _clean_title_row(row, int(row_idx))
        if assignment is None:
            continue
        key = str(assignment["title"]).casefold()
        if key in used_titles or key in seen:
            continue
        seen.add(key)
        out.append(assignment)
        if len(out) >= int(limit):
            break
    return out


def _balanced_years(count: int, start_year: int, end_year: int, rng: np.random.RandomState) -> list[int]:
    years = list(range(int(start_year), int(end_year) + 1))
    if not years:
        raise RuntimeError("Continuation year range is empty.")
    base = int(count) // len(years)
    rem = int(count) % len(years)
    out: list[int] = []
    for idx, year in enumerate(years):
        out.extend([int(year)] * (base + (1 if idx < rem else 0)))
    rng.shuffle(out)
    return out


def append_extension_plan(
    *,
    world: Any,
    resume_manager: Any,
    existing_plan: list[dict[str, Any]],
    target_movie_count: int,
    extension_start_year: int,
    extension_end_year: int,
    sample_movie_concept_fn: Callable[..., dict[str, Any]],
    rerank_fn: Callable[..., int] | None = None,
    llm_model: str | None = None,
) -> ExtensionPlanResult:
    existing_count = len(existing_plan)
    append_count = int(target_movie_count) - int(existing_count)
    if append_count <= 0:
        return ExtensionPlanResult(
            appended=0,
            total_plan_count=existing_count,
            first_seq_idx=None,
            last_seq_idx=None,
            first_movie_id=None,
            last_movie_id=None,
            year_min=None,
            year_max=None,
            title_assignments=0,
            compositional_title_movies=0,
            reranked=0,
        )

    if int(extension_start_year) > int(extension_end_year):
        raise RuntimeError(
            f"Invalid continuation year range: {extension_start_year}>{extension_end_year}"
        )

    max_seq_idx = max((_safe_int(row.get("seq_idx"), -1) or -1) for row in existing_plan) if existing_plan else -1
    start_seq_idx = max(existing_count, int(max_seq_idx) + 1)
    max_movie_id = max((_safe_int(row.get("movie_id"), 0) or 0) for row in existing_plan) if existing_plan else 0

    title_assignments = _available_title_assignments(
        world,
        extension_start_year=int(extension_start_year),
        extension_end_year=int(extension_end_year),
        limit=append_count,
    )
    overflow_count = max(0, append_count - len(title_assignments))
    overflow_years = _balanced_years(
        overflow_count,
        int(extension_start_year),
        int(extension_end_year),
        getattr(world, "rng", np.random.RandomState(int(getattr(world, "seed", 42)))),
    )

    year_list: list[tuple[int, int, dict[str, Any], dict[str, Any]]] = []
    for local_idx in range(append_count):
        movie_id = int(max_movie_id) + local_idx + 1
        title_assignment: dict[str, Any] = {}
        if local_idx < len(title_assignments):
            title_assignment = dict(title_assignments[local_idx])
            forced_year = int(title_assignment["year"])
        else:
            forced_year = int(overflow_years[local_idx - len(title_assignments)])
        concept = sample_movie_concept_fn(
            world,
            movie_id,
            forced_year=int(forced_year),
            title_assignment=title_assignment or None,
        )
        year_list.append((int(concept.get("year", forced_year)), movie_id, concept, title_assignment))

    reranked = 0
    if rerank_fn is not None and bool(getattr(world, "enable_llm_rerank", False)):
        rerank_items = [(year, movie_id, concept) for year, movie_id, concept, _ta in year_list]
        reranked = int(rerank_fn(world, rerank_items, llm_model=llm_model) or 0)
        concept_by_movie_id = {int(movie_id): concept for year, movie_id, concept in rerank_items}
        year_list = [
            (int(concept_by_movie_id[int(movie_id)].get("year", year)), movie_id, concept_by_movie_id[int(movie_id)], title_assignment)
            for year, movie_id, _concept, title_assignment in year_list
        ]

    year_list.sort(key=lambda item: (int(item[0]), int(item[1])))

    appended_records: list[dict[str, Any]] = []
    for offset, (year, movie_id, concept, title_assignment) in enumerate(year_list):
        appended_records.append(
            {
                "seq_idx": int(start_seq_idx + offset),
                "movie_id": int(movie_id),
                "year": int(year),
                "concept": concept,
                "title_assignment": dict(title_assignment or {}),
            }
        )

    combined_plan = list(existing_plan) + appended_records
    years = [int(row["year"]) for row in combined_plan if _safe_int(row.get("year")) is not None]
    resume_manager.save_plan(
        combined_plan,
        year_min=min(years) if years else None,
        year_max=max(years) if years else None,
    )

    return ExtensionPlanResult(
        appended=len(appended_records),
        total_plan_count=len(combined_plan),
        first_seq_idx=int(appended_records[0]["seq_idx"]) if appended_records else None,
        last_seq_idx=int(appended_records[-1]["seq_idx"]) if appended_records else None,
        first_movie_id=int(appended_records[0]["movie_id"]) if appended_records else None,
        last_movie_id=int(appended_records[-1]["movie_id"]) if appended_records else None,
        year_min=min((int(row["year"]) for row in appended_records), default=None),
        year_max=max((int(row["year"]) for row in appended_records), default=None),
        title_assignments=len(title_assignments),
        compositional_title_movies=overflow_count,
        reranked=reranked,
    )
