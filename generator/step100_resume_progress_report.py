#!/usr/bin/env python3
"""Progress and partial-health report for resumable step-100 runs.

This reads only durable `_step100_resume/` artifacts: manifest, committed yearly
shards, planner summaries, and the JSONL progress log. It is safe to run while
`generate_movies.py` is still appending the current in-flight year.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

try:
    import pyarrow as pa
    import pyarrow.ipc as ipc
except Exception as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "step100_resume_progress_report.py requires pyarrow. "
        "Run ./setup_linux_env.sh first."
    ) from exc


def pct(count: float, total: float) -> float:
    return round(100.0 * count / total, 4) if total else 0.0


def mean(values: Iterable[float]) -> float | None:
    vals = list(values)
    return round(float(statistics.mean(vals)), 4) if vals else None


def median(values: Iterable[float]) -> float | None:
    vals = list(values)
    return round(float(statistics.median(vals)), 4) if vals else None


def gini(values: Iterable[int | float]) -> float:
    vals = sorted(float(v) for v in values)
    n = len(vals)
    if not vals:
        return 0.0
    total = sum(vals)
    if total <= 0:
        return 0.0
    return round((2.0 * sum((i + 1) * v for i, v in enumerate(vals)) / (n * total)) - ((n + 1) / n), 4)


def top(counter: Counter, n: int = 10) -> list[dict[str, int]]:
    return [{"value": str(k), "count": int(v)} for k, v in counter.most_common(n)]


def read_arrow_rows(path: Path) -> list[dict[str, Any]]:
    with pa.memory_map(str(path), "r") as source:
        try:
            table = ipc.open_file(source).read_all()
        except pa.ArrowInvalid:
            source.seek(0)
            table = ipc.open_stream(source).read_all()
    return table.to_pylist()


def read_shards(resume_dir: Path, table_name: str) -> list[dict[str, Any]]:
    table_dir = resume_dir / "shards" / table_name
    if not table_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    for shard in sorted(table_dir.glob("year=*.arrow")):
        rows.extend(read_arrow_rows(shard))
    return rows


def find_progress_log(base_dir: Path, explicit: Path | None) -> Path | None:
    if explicit:
        return explicit
    logs = sorted((base_dir / "decision_logs").glob("*_movie_generation_progress.jsonl"))
    return logs[-1] if logs else None


def read_progress(path: Path | None, target_movies: int | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {"log_path": str(path) if path else None, "completed_movie_events": 0}
    completed_events = 0
    unique_seq_indices: set[int] = set()
    first: dict[str, Any] | None = None
    last: dict[str, Any] | None = None
    stage_counter: Counter[str] = Counter()
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except Exception:
                continue
            if event.get("event") != "movie_stage":
                continue
            stage_counter[str(event.get("stage") or "")] += 1
            if event.get("stage") == "write_keywords" and event.get("status") == "done":
                completed_events += 1
                first = first or event
                last = event
                try:
                    unique_seq_indices.add(int(event.get("seq_idx")))
                except Exception:
                    pass
    latest_seq_progress = None
    if last and last.get("seq_idx") is not None:
        try:
            latest_seq_progress = int(last.get("seq_idx")) + 1
        except Exception:
            latest_seq_progress = None
    duplicate_done_events = max(0, completed_events - len(unique_seq_indices))
    wall_sec_per_movie = None
    eta_hours = None
    progress_for_eta = latest_seq_progress if latest_seq_progress is not None else len(unique_seq_indices)
    if first and last and completed_events > 1:
        span = float(last.get("timestamp", 0.0)) - float(first.get("timestamp", 0.0))
        if span > 0:
            wall_sec_per_movie = round(span / (completed_events - 1), 4)
            if target_movies:
                eta_hours = round(max(target_movies - progress_for_eta, 0) * wall_sec_per_movie / 3600.0, 3)
    return {
        "log_path": str(path),
        "log_mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
        "completed_movie_events": completed_events,
        "unique_completed_seq_indices": len(unique_seq_indices),
        "duplicate_done_events": duplicate_done_events,
        "latest_seq_progress": latest_seq_progress,
        "last_movie_event": last,
        "wall_sec_per_movie": wall_sec_per_movie,
        "eta_hours_for_target": eta_hours,
        "stage_event_counts": dict(stage_counter),
    }


def read_planner_summary(base_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted((base_dir / "graph" / "temporal_patches").glob("year_summary_*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        planner = data.get("planner") or {}
        rows.append({
            "year": data.get("from_year"),
            "source": planner.get("source"),
            "parse_error": planner.get("parse_error") or "",
            "file": path.name,
        })
    sources = Counter(row["source"] for row in rows)
    fallbacks = [row for row in rows if row["source"] != "llm"]
    return {
        "year_summaries": len(rows),
        "source_counts": dict(sources),
        "fallback_years": fallbacks,
        "last_years": rows[-10:],
    }


def build_report(base_dir: Path, progress_log: Path | None = None) -> dict[str, Any]:
    resume_dir = base_dir / "_step100_resume"
    manifest_path = resume_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"No step-100 resume manifest found at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())

    movies = read_shards(resume_dir, "movie")
    casts = read_shards(resume_dir, "cast_info")
    directors = read_shards(resume_dir, "movie_directors")
    companies = read_shards(resume_dir, "movie_companies")
    keywords = read_shards(resume_dir, "movie_keyword")
    awards = read_shards(resume_dir, "awards")

    movie_ids = {row.get("title_id") for row in movies}
    title_counts = Counter(str(row.get("title") or "").strip() for row in movies)
    tagline_counts = Counter(str(row.get("tagline") or "").strip() for row in movies if str(row.get("tagline") or "").strip())
    years = Counter(row.get("year") for row in movies)
    genres = Counter(row.get("genre") for row in movies)
    tiers = Counter(row.get("production_tier") for row in movies)
    countries = Counter(row.get("country") for row in movies)

    cast_by_movie: Counter[Any] = Counter()
    actor_movies: Counter[Any] = Counter()
    blank_character_descriptions = 0
    for row in casts:
        cast_by_movie[row.get("title_id")] += 1
        actor_movies[row.get("person_id")] += 1
        if not str(row.get("character_description") or "").strip():
            blank_character_descriptions += 1

    directors_by_movie: Counter[Any] = Counter()
    director_movies: Counter[Any] = Counter()
    for row in directors:
        directors_by_movie[row.get("title_id")] += 1
        director_movies[row.get("director_id")] += 1

    companies_by_movie: Counter[Any] = Counter()
    company_movies: Counter[Any] = Counter()
    company_roles: Counter[str] = Counter()
    for row in companies:
        companies_by_movie[row.get("title_id")] += 1
        company_movies[row.get("company_id")] += 1
        company_roles[str(row.get("role") or "")] += 1

    keywords_by_movie: Counter[Any] = Counter()
    keyword_ids: Counter[Any] = Counter()
    for row in keywords:
        keywords_by_movie[row.get("title_id")] += 1
        keyword_ids[row.get("keyword_id")] += 1

    award_movies: Counter[Any] = Counter()
    award_outcomes: Counter[str] = Counter()
    for row in awards:
        award_movies[row.get("title_id")] += 1
        award_outcomes[str(row.get("outcome") or "").strip().lower()] += 1
    wins = sum(count for outcome, count in award_outcomes.items() if outcome in {"win", "won", "winner"})

    target_movies = int(manifest.get("movie_count") or 0)
    progress = read_progress(find_progress_log(base_dir, progress_log), target_movies)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_dir": str(base_dir),
        "resume_dir": str(resume_dir),
        "manifest": {
            "status": manifest.get("status"),
            "last_completed_year": manifest.get("last_completed_year"),
            "last_completed_sequence_index": manifest.get("last_completed_sequence_index"),
            "movie_count": target_movies,
            "start_year": manifest.get("start_year"),
            "end_year": manifest.get("end_year"),
        },
        "progress_log": progress,
        "planner": read_planner_summary(base_dir),
        "committed_shards": {
            "movie_rows": len(movies),
            "movie_completion_pct": pct(len(movies), target_movies),
            "year_min": min(years) if years else None,
            "year_max": max(years) if years else None,
            "unique_title_ids": len(movie_ids),
            "unique_titles": len(title_counts),
            "duplicate_title_rows": sum(c for c in title_counts.values() if c > 1),
            "unique_nonblank_taglines": len(tagline_counts),
            "duplicate_tagline_rows": sum(c for c in tagline_counts.values() if c > 1),
            "top_genres": top(genres, 12),
            "tiers": dict(tiers),
            "top_countries": top(countries, 12),
        },
        "cast": {
            "rows": len(casts),
            "movies_with_cast": len(cast_by_movie),
            "unique_people": len(actor_movies),
            "reuse_ratio": round(len(casts) / max(len(actor_movies), 1), 4),
            "per_movie_mean": mean(cast_by_movie.values()),
            "per_movie_median": median(cast_by_movie.values()),
            "top_actor_movies": top(actor_movies, 10),
            "blank_character_description_rows": blank_character_descriptions,
            "blank_character_description_pct": pct(blank_character_descriptions, len(casts)),
        },
        "directors": {
            "rows": len(directors),
            "movies_with_director": len(directors_by_movie),
            "unique_directors": len(director_movies),
            "top_director_movies": top(director_movies, 10),
        },
        "companies": {
            "rows": len(companies),
            "movies_with_company": len(companies_by_movie),
            "unique_companies": len(company_movies),
            "company_gini": gini(company_movies.values()),
            "roles": dict(company_roles),
            "top_company_movies": top(company_movies, 10),
        },
        "keywords": {
            "rows": len(keywords),
            "movies_with_keywords": len(keywords_by_movie),
            "unique_keywords": len(keyword_ids),
            "per_movie_mean": mean(keywords_by_movie.values()),
            "per_movie_median": median(keywords_by_movie.values()),
        },
        "awards": {
            "rows": len(awards),
            "movies_with_awards": len(award_movies),
            "movies_with_awards_pct": pct(len(award_movies), len(movies)),
            "outcomes": dict(award_outcomes),
            "win_pct": pct(wins, len(awards)),
        },
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    manifest = report["manifest"]
    progress = report["progress_log"]
    committed = report["committed_shards"]
    lines = [
        "# Step 100 Resume Progress",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Status: `{manifest.get('status')}`",
        f"- Target movies: `{manifest.get('movie_count')}`",
        f"- Latest sequence progress: `{progress.get('latest_seq_progress')}`",
        f"- Raw completed movie events: `{progress.get('completed_movie_events')}`",
        f"- Duplicate completed events: `{progress.get('duplicate_done_events')}`",
        f"- Committed shard movies: `{committed.get('movie_rows')}` ({committed.get('movie_completion_pct')}%)",
        f"- Last committed year: `{manifest.get('last_completed_year')}`",
        f"- Last committed sequence index: `{manifest.get('last_completed_sequence_index')}`",
        f"- Estimated remaining hours: `{progress.get('eta_hours_for_target')}`",
        "",
        "## Partial Health",
        "",
        f"- Unique titles: `{committed.get('unique_titles')}`",
        f"- Duplicate title rows: `{committed.get('duplicate_title_rows')}`",
        f"- Duplicate tagline rows: `{committed.get('duplicate_tagline_rows')}`",
        f"- Cast unique people: `{report['cast'].get('unique_people')}`",
        f"- Cast reuse ratio: `{report['cast'].get('reuse_ratio')}`",
        f"- Top actor movies: `{report['cast'].get('top_actor_movies', [{}])[0].get('count') if report['cast'].get('top_actor_movies') else 0}`",
        f"- Unique directors: `{report['directors'].get('unique_directors')}`",
        f"- Top director movies: `{report['directors'].get('top_director_movies', [{}])[0].get('count') if report['directors'].get('top_director_movies') else 0}`",
        f"- Company Gini: `{report['companies'].get('company_gini')}`",
        f"- Awards movie coverage: `{report['awards'].get('movies_with_awards_pct')}%`",
        f"- Awards win rate: `{report['awards'].get('win_pct')}%`",
        "",
        "## Planner",
        "",
        f"- Year summaries: `{report['planner'].get('year_summaries')}`",
        f"- Source counts: `{report['planner'].get('source_counts')}`",
        f"- Fallback years: `{report['planner'].get('fallback_years')}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Report progress and partial health for a resumable step-100 run.")
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    parser.add_argument("--progress-log", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    report = build_report(base_dir, args.progress_log)
    out_json = args.out_json or (base_dir / "reports" / "step100_resume_progress.json")
    out_md = args.out_md or (base_dir / "reports" / "step100_resume_progress.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, out_md)

    progress = report["progress_log"]
    manifest = report["manifest"]
    print(
        "Step100 progress: "
        f"{progress.get('latest_seq_progress')}/{manifest.get('movie_count')} latest-seq, "
        f"{report['committed_shards'].get('movie_rows')} committed, "
        f"last year {manifest.get('last_completed_year')}, "
        f"ETA {progress.get('eta_hours_for_target')}h"
    )
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
