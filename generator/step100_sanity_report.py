#!/usr/bin/env python3
"""Dependency-free sanity report for step-100 movie-generation outputs.

This intentionally reads the CSV mirrors so it can run from WSL even when the
Windows pandas/pyarrow environment is unavailable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable


PLACEHOLDER_RE = re.compile(
    r"\b(TBD|placeholder|lorem|summary pending|to be generated|a gripping tale)\b|\{\{|\}\}|\[.*\]",
    re.IGNORECASE,
)
GENERIC_STEP100_SUMMARY_RE = re.compile(
    r"^A\s+\w+\s+[A-Za-z-]+\s+film\s+from\s+.+\(\d{4}\)\.\s+Rated\s+",
    re.IGNORECASE,
)


def _rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def _canonical(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").strip().lower()).strip()


def _pct(count: int | float, total: int | float) -> float:
    return round((100.0 * float(count) / float(total)), 4) if total else 0.0


def _median(values: list[int | float]) -> float | None:
    return round(float(statistics.median(values)), 4) if values else None


def _mean(values: list[int | float]) -> float | None:
    return round(float(statistics.mean(values)), 4) if values else None


def _gini(values: Iterable[int | float]) -> float:
    vals = sorted(float(v) for v in values)
    n = len(vals)
    if n == 0:
        return 0.0
    total = sum(vals)
    if total <= 0:
        return 0.0
    return round((2 * sum((i + 1) * v for i, v in enumerate(vals)) / (n * total)) - ((n + 1) / n), 4)


def _top(counter: Counter, n: int = 10) -> list[dict[str, int]]:
    return [{"value": str(key), "count": int(count)} for key, count in counter.most_common(n)]


def build_report(base_dir: Path, *, require_enriched_summaries: bool = False) -> dict:
    movie_rows = list(_rows(base_dir / "movie.csv"))
    movie_count = len(movie_rows)
    movie_by_id = {int(row["title_id"]): row for row in movie_rows}
    title_ids = list(movie_by_id.keys())
    title_counts = Counter(row.get("title", "") for row in movie_rows)
    duplicate_titles = {title: count for title, count in title_counts.items() if count > 1}
    taglines = [row.get("tagline", "") for row in movie_rows]
    tagline_counts = Counter(t.strip() for t in taglines if t.strip())
    duplicated_tagline_rows = sum(count for count in tagline_counts.values() if count > 1)
    plots = [row.get("plot_summary", "") for row in movie_rows]
    plot_placeholders = sum(1 for text in plots if not text.strip() or PLACEHOLDER_RE.search(text))
    generic_step100_summaries = sum(1 for text in plots if GENERIC_STEP100_SUMMARY_RE.search(text.strip()))
    tagline_placeholders = sum(1 for text in taglines if not text.strip() or PLACEHOLDER_RE.search(text))

    ratings = [float(row["rating"]) for row in movie_rows if row.get("rating")]
    votes = [int(float(row["num_votes"])) for row in movie_rows if row.get("num_votes")]
    years = Counter(int(row["year"]) for row in movie_rows if row.get("year"))
    genres = Counter(row.get("genre", "") for row in movie_rows)
    tiers = Counter(row.get("production_tier", "") for row in movie_rows)
    countries = Counter(row.get("country", "") for row in movie_rows)

    cast_rows = 0
    cast_by_movie: Counter[int] = Counter()
    actor_movies: Counter[int] = Counter()
    blank_character_description = 0
    for row in _rows(base_dir / "cast_info.csv"):
        cast_rows += 1
        title_id = int(row["title_id"])
        person_id = int(row["person_id"])
        cast_by_movie[title_id] += 1
        actor_movies[person_id] += 1
        if not row.get("character_description", "").strip():
            blank_character_description += 1

    director_rows = 0
    directors_by_movie: Counter[int] = Counter()
    director_movies: Counter[int] = Counter()
    for row in _rows(base_dir / "movie_directors.csv"):
        director_rows += 1
        title_id = int(row["title_id"])
        director_id = int(row["director_id"])
        directors_by_movie[title_id] += 1
        director_movies[director_id] += 1

    company_rows = 0
    companies_by_movie: Counter[int] = Counter()
    company_movies: Counter[int] = Counter()
    company_roles: Counter[str] = Counter()
    for row in _rows(base_dir / "movie_companies.csv"):
        company_rows += 1
        title_id = int(row["title_id"])
        company_id = int(row["company_id"])
        companies_by_movie[title_id] += 1
        company_movies[company_id] += 1
        company_roles[row.get("role", "")] += 1

    keyword_topic: dict[int, str] = {}
    keyword_bucket: dict[int, str] = {}
    for row in _rows(base_dir / "entities" / "keyword.csv"):
        keyword_id = int(row["keyword_id"])
        keyword_topic[keyword_id] = _canonical(row.get("topic_genre", ""))
        keyword_bucket[keyword_id] = row.get("selection_bucket", "")

    keyword_rows = 0
    keywords_by_movie: Counter[int] = Counter()
    exact_topic_by_movie: Counter[int] = Counter()
    keyword_buckets: Counter[str] = Counter()
    for row in _rows(base_dir / "movie_keyword.csv"):
        keyword_rows += 1
        title_id = int(row["title_id"])
        keyword_id = int(row["keyword_id"])
        keywords_by_movie[title_id] += 1
        keyword_buckets[keyword_bucket.get(keyword_id, "")] += 1
        movie_genre = _canonical(movie_by_id.get(title_id, {}).get("genre", ""))
        if keyword_topic.get(keyword_id) and keyword_topic.get(keyword_id) == movie_genre:
            exact_topic_by_movie[title_id] += 1
    zero_exact_topic_ids = [title_id for title_id in title_ids if exact_topic_by_movie[title_id] <= 0]

    award_rows = 0
    award_movies: Counter[int] = Counter()
    award_outcomes: Counter[str] = Counter()
    award_ceremonies: Counter[str] = Counter()
    for row in _rows(base_dir / "awards.csv"):
        award_rows += 1
        title_id = int(row["title_id"])
        award_movies[title_id] += 1
        award_outcomes[row.get("outcome", "").strip().lower()] += 1
        award_ceremonies[row.get("ceremony", "")] += 1
    wins = sum(count for outcome, count in award_outcomes.items() if outcome in {"won", "win", "winner"})

    cast_counts = list(cast_by_movie.values())
    director_counts = list(directors_by_movie.values())
    company_counts = list(companies_by_movie.values())
    keyword_counts = list(keywords_by_movie.values())

    gates = {
        "movie_count_20k": movie_count == 20000,
        "unique_title_ids": len(set(title_ids)) == movie_count,
        "unique_titles": len(duplicate_titles) == 0,
        "plot_placeholders_zero": plot_placeholders == 0,
        "tagline_placeholders_zero": tagline_placeholders == 0,
        "zero_exact_topic_keywords": len(zero_exact_topic_ids) == 0,
        "awards_movie_rate_10_to_35_pct": 10.0 <= _pct(len(award_movies), movie_count) <= 35.0,
        "award_wins_no_more_than_35_pct": _pct(wins, award_rows) <= 35.0 if award_rows else True,
        "cast_reuse_ratio_le_15": (cast_rows / max(1, len(actor_movies))) <= 15.0,
        "unique_cast_people_ge_14000": len(actor_movies) >= 14000,
        "top_actor_le_250": (actor_movies.most_common(1)[0][1] if actor_movies else 0) <= 250,
        "unique_directors_ge_1000": len(director_movies) >= 1000,
        "top_director_le_120": (director_movies.most_common(1)[0][1] if director_movies else 0) <= 120,
        "company_gini_ge_045": _gini(company_movies.values()) >= 0.45,
        "blank_character_description_zero": blank_character_description == 0,
    }
    if require_enriched_summaries:
        gates["generic_step100_plot_summaries_zero"] = generic_step100_summaries == 0

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_dir": str(base_dir),
        "movie": {
            "rows": movie_count,
            "unique_title_ids": len(set(title_ids)),
            "unique_titles": len(title_counts),
            "duplicate_titles": duplicate_titles,
            "year_min": min(years) if years else None,
            "year_max": max(years) if years else None,
            "top_years": _top(years, 8),
            "top_genres": _top(genres, 12),
            "tiers": dict(tiers),
            "top_countries": _top(countries, 12),
            "rating_median": _median(ratings),
            "rating_mean": _mean(ratings),
            "votes_median": _median(votes),
            "votes_mean": _mean(votes),
            "plot_placeholder_rows": plot_placeholders,
            "generic_step100_summary_rows": generic_step100_summaries,
            "tagline_placeholder_rows": tagline_placeholders,
            "unique_nonblank_taglines": len(tagline_counts),
            "duplicated_tagline_rows": duplicated_tagline_rows,
            "duplicated_tagline_rate_pct": _pct(duplicated_tagline_rows, len(taglines)),
            "top_repeated_taglines": _top(tagline_counts, 8),
        },
        "cast": {
            "rows": cast_rows,
            "movies_with_cast": len(cast_by_movie),
            "unique_people": len(actor_movies),
            "reuse_ratio": round(cast_rows / max(1, len(actor_movies)), 4),
            "per_movie_median": _median(cast_counts),
            "per_movie_mean": _mean(cast_counts),
            "per_movie_max": max(cast_counts) if cast_counts else 0,
            "top_actor_movies": _top(actor_movies, 10),
            "blank_character_description_rows": blank_character_description,
            "blank_character_description_rate_pct": _pct(blank_character_description, cast_rows),
        },
        "directors": {
            "rows": director_rows,
            "movies_with_director": len(directors_by_movie),
            "unique_directors": len(director_movies),
            "per_movie_median": _median(director_counts),
            "per_movie_max": max(director_counts) if director_counts else 0,
            "top_director_movies": _top(director_movies, 10),
        },
        "companies": {
            "rows": company_rows,
            "movies_with_company": len(companies_by_movie),
            "unique_companies": len(company_movies),
            "per_movie_median": _median(company_counts),
            "per_movie_mean": _mean(company_counts),
            "per_movie_max": max(company_counts) if company_counts else 0,
            "gini": _gini(company_movies.values()),
            "top_company_movies": _top(company_movies, 10),
            "roles": dict(company_roles),
        },
        "keywords": {
            "rows": keyword_rows,
            "movies_with_keywords": len(keywords_by_movie),
            "per_movie_median": _median(keyword_counts),
            "per_movie_mean": _mean(keyword_counts),
            "per_movie_max": max(keyword_counts) if keyword_counts else 0,
            "zero_exact_topic_movies": len(zero_exact_topic_ids),
            "zero_exact_topic_rate_pct": _pct(len(zero_exact_topic_ids), movie_count),
            "zero_exact_topic_title_ids": zero_exact_topic_ids[:50],
            "bucket_counts": dict(keyword_buckets),
        },
        "awards": {
            "rows": award_rows,
            "movies_with_awards": len(award_movies),
            "movies_with_awards_rate_pct": _pct(len(award_movies), movie_count),
            "wins": wins,
            "win_rate_pct": _pct(wins, award_rows),
            "outcomes": dict(award_outcomes),
            "top_ceremonies": _top(award_ceremonies, 10),
        },
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "summary_policy": "strict_enriched" if require_enriched_summaries else "benchmark_first_deferred",
    }


def write_markdown(report: dict, path: Path) -> None:
    failed = report["failed_gates"]
    movie = report["movie"]
    cast = report["cast"]
    directors = report["directors"]
    companies = report["companies"]
    keywords = report["keywords"]
    awards = report["awards"]
    lines = [
        "# Step 100 Sanity Report",
        "",
        f"- Dataset: `{report['dataset_dir']}`",
        f"- Generated: `{report['generated_at']}`",
        f"- Summary policy: `{report.get('summary_policy', 'unknown')}`",
        f"- Overall: {'PASS' if not failed else 'NEEDS WORK'}",
        f"- Failed gates: `{', '.join(failed) if failed else 'none'}`",
        "",
        "## Core Counts",
        f"- Movies: `{movie['rows']}` rows, `{movie['unique_titles']}` unique titles",
        f"- Cast: `{cast['rows']}` rows, `{cast['unique_people']}` unique people, reuse ratio `{cast['reuse_ratio']}`",
        f"- Directors: `{directors['unique_directors']}` unique, top director `{directors['top_director_movies'][:1]}`",
        f"- Companies: `{companies['unique_companies']}` unique, Gini `{companies['gini']}`",
        f"- Keywords: `{keywords['rows']}` rows, zero exact-topic movies `{keywords['zero_exact_topic_movies']}`",
        f"- Awards: `{awards['rows']}` rows, movies with awards `{awards['movies_with_awards_rate_pct']}%`, win rate `{awards['win_rate_pct']}%`",
        "",
        "## Notable Details",
        f"- Generic step-100 plot summaries: `{movie['generic_step100_summary_rows']}`",
        f"- Duplicate titles: `{movie['duplicate_titles']}`",
        f"- Top actors: `{cast['top_actor_movies'][:5]}`",
        f"- Top repeated taglines: `{movie['top_repeated_taglines'][:5]}`",
        f"- Zero exact-topic title ids: `{keywords['zero_exact_topic_title_ids']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dependency-free step-100 sanity stats.")
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--require-enriched-summaries",
        action="store_true",
        help="Treat generic step-100 plot summaries as a failed gate. Use after step 110, not for benchmark-first step 100.",
    )
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    out_dir = (args.out_dir or (base_dir / "reports")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(base_dir, require_enriched_summaries=bool(args.require_enriched_summaries))
    json_path = out_dir / "step100_sanity_report.json"
    md_path = out_dir / "step100_sanity_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Failed gates: {', '.join(report['failed_gates']) if report['failed_gates'] else 'none'}")


if __name__ == "__main__":
    main()
