from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import pandas as pd

from validate_imdb_schema import validate as validate_imdb_schema


DEFAULT_EXPECTED_MOVIE_ROWS = 10_000
DEFAULT_MAX_PLACEHOLDER_PLOT_RATE = 0.05
DEFAULT_AWARD_MOVIE_RATE_MIN = 0.10
DEFAULT_AWARD_MOVIE_RATE_MAX = 0.35
DEFAULT_MAX_AWARD_WIN_SHARE = 0.35
DEFAULT_MIN_UNIQUE_CAST_PEOPLE = 7_000
DEFAULT_MIN_UNIQUE_DIRECTORS = 1_000
DEFAULT_MAX_TOP_ACTOR_MOVIES = 250
DEFAULT_MAX_TOP_DIRECTOR_MOVIES = 120
DEFAULT_MIN_TV_SERIES_VALID_RATE = 1.00
DEFAULT_MIN_TV_EPISODE_VALID_RATE = 0.98
DEFAULT_MAX_GENERIC_EPISODE_TITLE_RATE = 0.05
DEFAULT_MAX_JOB_EMPTY_RATE = 0.15
DEFAULT_SCALE_SENSITIVE_THRESHOLD_ROWS = 10_000


@dataclass(frozen=True)
class SignoffThresholds:
    max_placeholder_plot_rate: float = DEFAULT_MAX_PLACEHOLDER_PLOT_RATE
    award_movie_rate_min: float = DEFAULT_AWARD_MOVIE_RATE_MIN
    award_movie_rate_max: float = DEFAULT_AWARD_MOVIE_RATE_MAX
    max_award_win_share: float = DEFAULT_MAX_AWARD_WIN_SHARE
    min_unique_cast_people: int = DEFAULT_MIN_UNIQUE_CAST_PEOPLE
    min_unique_directors: int = DEFAULT_MIN_UNIQUE_DIRECTORS
    max_top_actor_movies: int = DEFAULT_MAX_TOP_ACTOR_MOVIES
    max_top_director_movies: int = DEFAULT_MAX_TOP_DIRECTOR_MOVIES
    min_tv_series_valid_rate: float = DEFAULT_MIN_TV_SERIES_VALID_RATE
    min_tv_episode_valid_rate: float = DEFAULT_MIN_TV_EPISODE_VALID_RATE
    max_generic_episode_title_rate: float = DEFAULT_MAX_GENERIC_EPISODE_TITLE_RATE
    max_job_empty_rate: float = DEFAULT_MAX_JOB_EMPTY_RATE


def _word_count(text: object) -> int:
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'/-]*", str(text or "")))


def _normalise_text(text: object) -> str:
    return " ".join(str(text or "").split()).strip()


def _looks_like_placeholder_plot(text: object) -> bool:
    value = _normalise_text(text)
    low = value.lower()
    if not value:
        return True
    if _word_count(value) < 18:
        return True
    if "synthetic" in low:
        return True
    if low.startswith(("a ", "an ")) and " film from " in low and re.search(r"\(\d{4}\)", low):
        return True
    if re.match(r"^(a|an)\s+[a-z-]+\s+.+?\s+film from\s+.+?\(\d{4}\)", low):
        return True
    if "rated " in low and _word_count(value) < 28:
        return True
    return False


def _is_valid_series_summary(text: object) -> bool:
    value = _normalise_text(text)
    return bool(value) and "synthetic" not in value.lower() and _word_count(value) >= 45


def _is_valid_episode_description(text: object) -> bool:
    value = _normalise_text(text)
    return bool(value) and "synthetic" not in value.lower() and _word_count(value) >= 14


def _is_generic_episode_title(text: object) -> bool:
    return bool(re.match(r"^Episode\s+\d+$", _normalise_text(text), flags=re.IGNORECASE))


def _parse_boolish(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(csv_path.open("r", encoding="utf-8", newline="")))


def _summarize_result_rows(rows: list[dict[str, str]]) -> dict[str, object]:
    total = len(rows)
    ok = sum(1 for row in rows if str(row.get("status", "")).upper() == "OK")
    empty = sum(1 for row in rows if str(row.get("status", "")).upper() == "EMPTY")
    error = sum(1 for row in rows if str(row.get("status", "")).upper() == "ERROR")
    nonzero = sum(1 for row in rows if int(float(row.get("actual_count", "0") or 0)) > 0)
    qerrors: list[float] = []
    for row in rows:
        raw = str(row.get("sub_agg_qerror", "") or "").strip()
        if not raw:
            continue
        try:
            qerrors.append(float(raw))
        except Exception:
            continue
    summary = {
        "queries": total,
        "ok": ok,
        "empty": empty,
        "error": error,
        "nonzero": nonzero,
        "empty_rate": round(empty / total, 4) if total else 0.0,
        "error_examples": [row for row in rows if str(row.get("status", "")).upper() == "ERROR"][:5],
    }
    if qerrors:
        summary["median_qerror"] = round(median(qerrors), 4)
        summary["mean_qerror"] = round(sum(qerrors) / len(qerrors), 4)
        summary["max_qerror"] = round(max(qerrors), 4)
    return summary


def _load_legacy_job_summary(csv_path: Path) -> dict[str, object]:
    if not csv_path.exists():
        return {"exists": False}
    rows = _read_csv_rows(csv_path)
    total = len(rows)
    ok_rows = [row for row in rows if row.get("status") != "ERROR"]
    empty_rows = [row for row in rows if row.get("status") == "EMPTY"]
    error_rows = [row for row in rows if row.get("status") == "ERROR"]
    return {
        "exists": True,
        "source": "legacy_duckdb_csv",
        "path": str(csv_path),
        "queries": total,
        "ok": len(ok_rows),
        "empty": len(empty_rows),
        "error": len(error_rows),
        "empty_rate": round(len(empty_rows) / total, 4) if total else 0.0,
        "error_examples": error_rows[:5],
    }


def _load_exact_job_summary(benchmark_run_dir: Path) -> dict[str, object]:
    manifest_path = benchmark_run_dir / "manifest.json"
    postgres_path = benchmark_run_dir / "postgres_results.csv"
    duckdb_path = benchmark_run_dir / "duckdb_results.csv"
    if not manifest_path.exists() or not postgres_path.exists() or not duckdb_path.exists():
        return {"exists": False}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_queries = manifest.get("queries", [])
    postgres_rows = _read_csv_rows(postgres_path)
    duckdb_rows = _read_csv_rows(duckdb_path)
    return {
        "exists": True,
        "source": "exact_job_v1",
        "path": str(benchmark_run_dir),
        "dataset_id": manifest.get("dataset_id", benchmark_run_dir.name),
        "canonical_queries": int(manifest.get("canonical_query_count", len(manifest_queries))),
        "structure_guard_passed": sum(
            1 for row in manifest_queries if _parse_boolish(row.get("structure_guard_passed"))
        ),
        "postgres": _summarize_result_rows(postgres_rows),
        "duckdb": _summarize_result_rows(duckdb_rows),
    }


def _detect_default_benchmark_run_dir(base_dir: Path) -> Path | None:
    root_dir = Path(__file__).resolve().parent.parent
    candidate = root_dir / "benchmark" / "job_exact_v1" / "runs" / base_dir.name
    return candidate if candidate.exists() else None


def _load_job_summary(base_dir: Path, benchmark_run_dir: Path | None) -> dict[str, object]:
    effective_run_dir = benchmark_run_dir or _detect_default_benchmark_run_dir(base_dir)
    if effective_run_dir is not None:
        exact_summary = _load_exact_job_summary(effective_run_dir)
        if exact_summary.get("exists"):
            return exact_summary
    return _load_legacy_job_summary(base_dir / "duckdb_job_timing_results.csv")


def _load_generation_errors(base_dir: Path) -> dict[str, int]:
    progress_path = base_dir / "decision_logs" / "movie_generation_progress.jsonl"
    if not progress_path.exists():
        return {"exists": False, "error_events": 0}

    error_events = 0
    with progress_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except Exception:
                continue
            event_name = str(event.get("event", "") or "").strip().lower()
            if "error" in event_name or event_name in {"movie_failed", "step100_failed"}:
                error_events += 1
    return {"exists": True, "error_events": error_events}


def _check(name: str, ok: bool, actual, expected: str) -> dict[str, object]:
    return {
        "name": name,
        "ok": bool(ok),
        "actual": actual,
        "expected": expected,
    }


def _read_optional_table(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, low_memory=False)
    arrow_path = path.with_suffix(".arrow")
    if arrow_path.exists():
        import pyarrow.feather as feather

        return feather.read_table(str(arrow_path)).to_pandas()
    return pd.DataFrame()


def generate_signoff(
    base_dir: Path,
    *,
    expected_movie_rows: int | None = DEFAULT_EXPECTED_MOVIE_ROWS,
    thresholds: SignoffThresholds | None = None,
    benchmark_run_dir: Path | None = None,
) -> tuple[dict[str, object], str]:
    thresholds = thresholds or SignoffThresholds()
    movie = pd.read_csv(base_dir / "movie.csv", low_memory=False)
    cast_info = pd.read_csv(base_dir / "cast_info.csv", low_memory=False)
    movie_directors = pd.read_csv(base_dir / "movie_directors.csv", low_memory=False)
    awards = pd.read_csv(base_dir / "awards.csv", low_memory=False)
    tv_series = _read_optional_table(base_dir / "tv_series.csv")
    episodes = _read_optional_table(base_dir / "episodes.csv")

    export_dir = base_dir / "imdb_schema"
    export_validation = {"exists": export_dir.exists(), "ok": False, "error": None}
    if export_dir.exists():
        try:
            validate_imdb_schema(export_dir)
            export_validation["ok"] = True
        except Exception as exc:
            export_validation["error"] = str(exc)

    tagline_counter = Counter(movie["tagline"].fillna("").astype(str).tolist())
    actor_counter = Counter(cast_info["person_id"].dropna().astype(int).tolist())
    director_counter = Counter(movie_directors["director_id"].dropna().astype(int).tolist())
    award_movie_counter = Counter(awards["title_id"].dropna().astype(int).tolist())
    award_outcome_counter = Counter(awards["outcome"].fillna("").astype(str).tolist())

    plot_placeholder_count = int(movie["plot_summary"].fillna("").map(_looks_like_placeholder_plot).sum())
    series_valid_count = int(tv_series["plot_summary"].fillna("").map(_is_valid_series_summary).sum()) if "plot_summary" in tv_series.columns else 0
    episode_valid_count = int(episodes["description"].fillna("").map(_is_valid_episode_description).sum()) if "description" in episodes.columns else 0
    generic_episode_titles = int(episodes["title"].fillna("").map(_is_generic_episode_title).sum()) if "title" in episodes.columns else 0

    job_summary = _load_job_summary(base_dir, benchmark_run_dir)
    generation_errors = _load_generation_errors(base_dir)

    movie_row_count = int(len(movie))
    unique_titles = int(movie["title"].fillna("").astype(str).nunique())
    placeholder_plot_rate = round(plot_placeholder_count / max(len(movie), 1), 4)
    award_movie_rate = round(len(award_movie_counter) / max(len(movie), 1), 4)
    award_win_share = round(award_outcome_counter.get("Won", 0) / max(len(awards), 1), 4) if len(awards) else 0.0
    series_valid_rate = round(series_valid_count / max(len(tv_series), 1), 4) if len(tv_series) else 0.0
    episode_valid_rate = round(episode_valid_count / max(len(episodes), 1), 4) if len(episodes) else 0.0
    generic_episode_title_rate = round(generic_episode_titles / max(len(episodes), 1), 4) if len(episodes) else 0.0
    top_actor_movie_count = int(actor_counter.most_common(1)[0][1]) if actor_counter else 0
    top_director_movie_count = int(director_counter.most_common(1)[0][1]) if director_counter else 0
    unique_cast_people = len(actor_counter)
    unique_directors = len(director_counter)
    tv_present = len(tv_series) > 0 or len(episodes) > 0
    scale_sensitive_checks_enabled = (
        expected_movie_rows is not None and expected_movie_rows >= DEFAULT_SCALE_SENSITIVE_THRESHOLD_ROWS
    )

    report = {
        "base_dir": str(base_dir),
        "expected_movie_rows": expected_movie_rows,
        "benchmark_run_dir": str(benchmark_run_dir) if benchmark_run_dir is not None else None,
        "scale_sensitive_checks_enabled": scale_sensitive_checks_enabled,
        "movie": {
            "row_count": movie_row_count,
            "unique_titles": unique_titles,
            "placeholder_plot_count": plot_placeholder_count,
            "placeholder_plot_rate": placeholder_plot_rate,
            "tagline_unique_count": len(tagline_counter),
            "tagline_duplicate_rows": int(sum(v - 1 for v in tagline_counter.values() if v > 1)),
            "top_tagline_duplicates": tagline_counter.most_common(10),
        },
        "talent": {
            "unique_cast_people": unique_cast_people,
            "unique_directors": unique_directors,
            "top_actor_movie_count": top_actor_movie_count,
            "top_director_movie_count": top_director_movie_count,
            "top_actors": actor_counter.most_common(10),
            "top_directors": director_counter.most_common(10),
            "blank_character_description_rows": int(cast_info["character_description"].fillna("").astype(str).str.strip().eq("").sum())
            if "character_description" in cast_info.columns
            else int(len(cast_info)),
        },
        "awards": {
            "rows": int(len(awards)),
            "movies_with_awards": len(award_movie_counter),
            "movie_rate": award_movie_rate,
            "wins": int(award_outcome_counter.get("Won", 0)),
            "nominations": int(award_outcome_counter.get("Nominated", 0)),
            "win_share": award_win_share,
        },
        "tv": {
            "series_rows": int(len(tv_series)),
            "series_valid_count": series_valid_count,
            "series_valid_rate": series_valid_rate,
            "episode_rows": int(len(episodes)),
            "episode_valid_count": episode_valid_count,
            "episode_valid_rate": episode_valid_rate,
            "generic_episode_title_count": generic_episode_titles,
            "generic_episode_title_rate": generic_episode_title_rate,
        },
        "export": export_validation,
        "job_benchmark": job_summary,
        "generation": generation_errors,
    }

    checks: list[dict[str, object]] = []
    if expected_movie_rows is not None:
        checks.extend(
            [
                _check("movie_rows", movie_row_count == expected_movie_rows, movie_row_count, f"== {expected_movie_rows}"),
                _check("unique_titles", unique_titles == expected_movie_rows, unique_titles, f"== {expected_movie_rows}"),
            ]
        )
    checks.extend(
        [
            _check("generation_errors", generation_errors.get("error_events", 0) == 0, generation_errors.get("error_events", 0), "== 0"),
            _check(
                "plot_placeholder_rate",
                placeholder_plot_rate <= thresholds.max_placeholder_plot_rate,
                placeholder_plot_rate,
                f"<= {thresholds.max_placeholder_plot_rate:.2f}",
            ),
            _check(
                "awards_movie_rate",
                thresholds.award_movie_rate_min <= award_movie_rate <= thresholds.award_movie_rate_max,
                award_movie_rate,
                f"between {thresholds.award_movie_rate_min:.2f} and {thresholds.award_movie_rate_max:.2f}",
            ),
            _check(
                "awards_win_share",
                award_win_share <= thresholds.max_award_win_share,
                award_win_share,
                f"<= {thresholds.max_award_win_share:.2f}",
            ),
            _check("export_validation", export_validation.get("ok", False), export_validation.get("ok", False), "== True"),
        ]
    )
    if scale_sensitive_checks_enabled:
        checks.extend(
            [
                _check("unique_cast_people", unique_cast_people >= thresholds.min_unique_cast_people, unique_cast_people, f">= {thresholds.min_unique_cast_people}"),
                _check("unique_directors", unique_directors >= thresholds.min_unique_directors, unique_directors, f">= {thresholds.min_unique_directors}"),
                _check("top_actor_movie_count", top_actor_movie_count <= thresholds.max_top_actor_movies, top_actor_movie_count, f"<= {thresholds.max_top_actor_movies}"),
                _check("top_director_movie_count", top_director_movie_count <= thresholds.max_top_director_movies, top_director_movie_count, f"<= {thresholds.max_top_director_movies}"),
            ]
        )
    if tv_present:
        checks.extend(
            [
                _check("tv_series_valid_rate", series_valid_rate >= thresholds.min_tv_series_valid_rate, series_valid_rate, f">= {thresholds.min_tv_series_valid_rate:.2f}"),
                _check("tv_episode_valid_rate", episode_valid_rate >= thresholds.min_tv_episode_valid_rate, episode_valid_rate, f">= {thresholds.min_tv_episode_valid_rate:.2f}"),
                _check("generic_episode_title_rate", generic_episode_title_rate < thresholds.max_generic_episode_title_rate, generic_episode_title_rate, f"< {thresholds.max_generic_episode_title_rate:.2f}"),
            ]
        )
    if job_summary.get("exists"):
        if job_summary.get("source") == "exact_job_v1":
            postgres_summary = job_summary.get("postgres", {})
            duckdb_summary = job_summary.get("duckdb", {})
            checks.extend(
                [
                    _check(
                        "job_structure_guard",
                        int(job_summary.get("structure_guard_passed", 0)) == int(job_summary.get("canonical_queries", 0)),
                        int(job_summary.get("structure_guard_passed", 0)),
                        f"== {int(job_summary.get('canonical_queries', 0))}",
                    ),
                    _check("job_postgres_errors", int(postgres_summary.get("error", 0)) == 0, int(postgres_summary.get("error", 0)), "== 0"),
                    _check(
                        "job_postgres_empty_rate",
                        float(postgres_summary.get("empty_rate", 0.0)) <= thresholds.max_job_empty_rate,
                        float(postgres_summary.get("empty_rate", 0.0)),
                        f"<= {thresholds.max_job_empty_rate:.2f}",
                    ),
                    _check("job_duckdb_errors", int(duckdb_summary.get("error", 0)) == 0, int(duckdb_summary.get("error", 0)), "== 0"),
                ]
            )
        else:
            checks.extend(
                [
                    _check("job_errors", int(job_summary.get("error", 0)) == 0, int(job_summary.get("error", 0)), "== 0"),
                    _check(
                        "job_empty_rate",
                        float(job_summary.get("empty_rate", 0.0)) <= thresholds.max_job_empty_rate,
                        float(job_summary.get("empty_rate", 0.0)),
                        f"<= {thresholds.max_job_empty_rate:.2f}",
                    ),
                ]
            )
    report["acceptance"] = {
        "all_passed": all(check["ok"] for check in checks),
        "checks": checks,
        "failed_checks": [check["name"] for check in checks if not check["ok"]],
    }
    report["polish_findings"] = [
        {
            "name": "tagline_duplication",
            "needs_cleanup": report["movie"]["tagline_duplicate_rows"] > 0,
            "value": report["movie"]["tagline_duplicate_rows"],
            "note": "Non-blocking for benchmark execution, but should be cleaned before archival publish-ready signoff.",
        },
        {
            "name": "blank_character_descriptions",
            "needs_cleanup": report["talent"]["blank_character_description_rows"] > 0,
            "value": report["talent"]["blank_character_description_rows"],
            "note": "Non-blocking for JOB/CE benchmark validation, but incomplete for final publication polish.",
        },
    ]

    lines = [
        f"# {base_dir.name} Signoff Report",
        "",
        f"- Movies: `{report['movie']['row_count']}` rows, `{report['movie']['unique_titles']}` unique titles",
        f"- Expected movie rows for this signoff: `{report['expected_movie_rows'] if report['expected_movie_rows'] is not None else 'not enforced'}`",
        f"- Scale-sensitive talent gates enabled: `{'yes' if report['scale_sensitive_checks_enabled'] else 'no'}`",
        f"- Generation errors in step 100 log: `{report['generation']['error_events']}`",
        f"- Plot placeholders currently present: `{report['movie']['placeholder_plot_count']}` ({report['movie']['placeholder_plot_rate']:.2%})",
        f"- Tagline duplicate rows: `{report['movie']['tagline_duplicate_rows']}`; unique taglines: `{report['movie']['tagline_unique_count']}`",
        f"- Unique cast people: `{report['talent']['unique_cast_people']}`; unique directors: `{report['talent']['unique_directors']}`",
        f"- Top actor movie count: `{report['talent']['top_actor_movie_count']}`; top director movie count: `{report['talent']['top_director_movie_count']}`",
        f"- Blank character descriptions: `{report['talent']['blank_character_description_rows']}`",
        f"- Awards: `{report['awards']['rows']}` rows across `{report['awards']['movies_with_awards']}` movies ({report['awards']['movie_rate']:.2%}); win share `{report['awards']['win_share']:.2%}`",
        f"- TV summaries valid: `{report['tv']['series_valid_count']}/{report['tv']['series_rows']}` series, `{report['tv']['episode_valid_count']}/{report['tv']['episode_rows']}` episodes" + (" (skipped: no TV rows present)" if not tv_present else ""),
        f"- Generic episode titles: `{report['tv']['generic_episode_title_count']}` ({report['tv']['generic_episode_title_rate']:.2%})",
        f"- Export validation: `{'OK' if report['export']['ok'] else 'FAILED' if report['export']['exists'] else 'MISSING'}`",
        f"- JOB benchmark: `{'present' if report['job_benchmark'].get('exists') else 'missing'}`" + (f" via `{report['job_benchmark'].get('source')}`" if report['job_benchmark'].get('exists') else ""),
        f"- Acceptance gates: `{'PASS' if report['acceptance']['all_passed'] else 'FAIL'}`",
    ]
    if report["job_benchmark"].get("exists"):
        if report["job_benchmark"].get("source") == "exact_job_v1":
            pg = report["job_benchmark"]["postgres"]
            duck = report["job_benchmark"]["duckdb"]
            lines.append(
                f"- Exact JOB Postgres: `{pg['ok']}` OK, `{pg['empty']}` empty, `{pg['error']}` errors, `{pg['nonzero']}` non-zero"
            )
            lines.append(
                f"- Exact JOB DuckDB: `{duck['ok']}` OK, `{duck['empty']}` empty, `{duck['error']}` errors, `{duck['nonzero']}` non-zero"
            )
            if "median_qerror" in pg:
                lines.append(
                    f"- Exact JOB Postgres q-error: median `{pg['median_qerror']}`, mean `{pg['mean_qerror']}`, max `{pg['max_qerror']}`"
                )
        else:
            lines.append(
                f"- JOB results: `{report['job_benchmark']['ok']}` OK, `{report['job_benchmark']['empty']}` empty, `{report['job_benchmark']['error']}` errors"
            )
    if report["acceptance"]["failed_checks"]:
        lines.append(f"- Failed checks: `{', '.join(report['acceptance']['failed_checks'])}`")
    pending_polish = [item["name"] for item in report["polish_findings"] if item["needs_cleanup"]]
    if pending_polish:
        lines.append(f"- Non-blocking polish still pending: `{', '.join(pending_polish)}`")
    markdown = "\n".join(lines) + "\n"
    return report, markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a publish-readiness signoff report for a completed run directory.")
    parser.add_argument("--base-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--expected-movie-rows", type=int, default=DEFAULT_EXPECTED_MOVIE_ROWS)
    parser.add_argument("--benchmark-run-dir", default=None)
    parser.add_argument("--max-placeholder-plot-rate", type=float, default=DEFAULT_MAX_PLACEHOLDER_PLOT_RATE)
    parser.add_argument("--award-movie-rate-min", type=float, default=DEFAULT_AWARD_MOVIE_RATE_MIN)
    parser.add_argument("--award-movie-rate-max", type=float, default=DEFAULT_AWARD_MOVIE_RATE_MAX)
    parser.add_argument("--max-award-win-share", type=float, default=DEFAULT_MAX_AWARD_WIN_SHARE)
    parser.add_argument("--min-unique-cast-people", type=int, default=DEFAULT_MIN_UNIQUE_CAST_PEOPLE)
    parser.add_argument("--min-unique-directors", type=int, default=DEFAULT_MIN_UNIQUE_DIRECTORS)
    parser.add_argument("--max-top-actor-movies", type=int, default=DEFAULT_MAX_TOP_ACTOR_MOVIES)
    parser.add_argument("--max-top-director-movies", type=int, default=DEFAULT_MAX_TOP_DIRECTOR_MOVIES)
    parser.add_argument("--min-tv-series-valid-rate", type=float, default=DEFAULT_MIN_TV_SERIES_VALID_RATE)
    parser.add_argument("--min-tv-episode-valid-rate", type=float, default=DEFAULT_MIN_TV_EPISODE_VALID_RATE)
    parser.add_argument("--max-generic-episode-title-rate", type=float, default=DEFAULT_MAX_GENERIC_EPISODE_TITLE_RATE)
    parser.add_argument("--max-job-empty-rate", type=float, default=DEFAULT_MAX_JOB_EMPTY_RATE)
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    benchmark_run_dir = Path(args.benchmark_run_dir).resolve() if args.benchmark_run_dir else None
    thresholds = SignoffThresholds(
        max_placeholder_plot_rate=args.max_placeholder_plot_rate,
        award_movie_rate_min=args.award_movie_rate_min,
        award_movie_rate_max=args.award_movie_rate_max,
        max_award_win_share=args.max_award_win_share,
        min_unique_cast_people=args.min_unique_cast_people,
        min_unique_directors=args.min_unique_directors,
        max_top_actor_movies=args.max_top_actor_movies,
        max_top_director_movies=args.max_top_director_movies,
        min_tv_series_valid_rate=args.min_tv_series_valid_rate,
        min_tv_episode_valid_rate=args.min_tv_episode_valid_rate,
        max_generic_episode_title_rate=args.max_generic_episode_title_rate,
        max_job_empty_rate=args.max_job_empty_rate,
    )
    report, markdown = generate_signoff(
        base_dir,
        expected_movie_rows=args.expected_movie_rows,
        thresholds=thresholds,
        benchmark_run_dir=benchmark_run_dir,
    )
    json_path = base_dir / "signoff_report.json"
    md_path = base_dir / "signoff_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
