#!/usr/bin/env python3
"""Fast structural feasibility checks before an expensive step-100 run.

The goal is to catch impossible configurations before a lab or laptop spends
many hours generating movies.  This script intentionally reads only CSV/JSON
metadata and has no pandas/pyarrow dependency.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable


ACTOR_ROLE_NAMES = {"actor", "voice_actor", "performer"}
VALID_CAREER_STAGES = {"rising", "prime", "veteran", "legend", "retired"}
TIMELINE_FIELDS = ("debut_year", "peak_start", "peak_end", "retirement_year", "yearly_max")


def _rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="ignore") as handle:
        yield from csv.DictReader(handle)


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def _safe_load_json(path: Path) -> object | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _actor_ids_from_roles(base_dir: Path) -> set[int]:
    role_path = base_dir / "entities" / "person_roles.csv"
    if not role_path.exists():
        return set()
    actor_ids: set[int] = set()
    for row in _rows(role_path):
        try:
            person_id = int(float(row.get("person_id") or 0))
        except Exception:
            continue
        role = str(row.get("role_type") or row.get("role") or "").strip().lower()
        if role in ACTOR_ROLE_NAMES:
            actor_ids.add(person_id)
    return actor_ids


def _person_count(base_dir: Path) -> int:
    person_csv = base_dir / "entities" / "person.csv"
    if person_csv.exists():
        return _count_csv_rows(person_csv)
    persons_json = _safe_load_json(base_dir / "entities" / "persons.json")
    return len(persons_json) if isinstance(persons_json, list) else 0


def _timeline_stats(base_dir: Path) -> dict[str, int]:
    persons_json = _safe_load_json(base_dir / "entities" / "persons.json")
    stats = {
        "person_json_rows": 0,
        "missing_timeline_rows": 0,
        "invalid_stage_rows": 0,
        "invalid_timeline_order_rows": 0,
    }
    if not isinstance(persons_json, list):
        return stats
    stats["person_json_rows"] = len(persons_json)
    for row in persons_json:
        if not isinstance(row, dict):
            stats["missing_timeline_rows"] += 1
            continue
        stage = str(row.get("career_stage") or "").strip().lower()
        if stage not in VALID_CAREER_STAGES:
            stats["invalid_stage_rows"] += 1
        if any(row.get(field) is None for field in TIMELINE_FIELDS):
            stats["missing_timeline_rows"] += 1
            continue
        try:
            debut = int(float(row.get("debut_year")))
            peak_start = int(float(row.get("peak_start")))
            peak_end = int(float(row.get("peak_end")))
            retire = int(float(row.get("retirement_year")))
            yearly_max = int(float(row.get("yearly_max")))
        except Exception:
            stats["missing_timeline_rows"] += 1
            continue
        if not (debut <= peak_start <= peak_end <= retire and yearly_max > 0):
            stats["invalid_timeline_order_rows"] += 1
    return stats


def _current_cast_stats(base_dir: Path) -> dict[str, float | int] | None:
    cast_path = base_dir / "cast_info.csv"
    if not cast_path.exists():
        return None
    cast_rows = 0
    movies: set[int] = set()
    actors: set[int] = set()
    top_actor = Counter()
    for row in _rows(cast_path):
        cast_rows += 1
        try:
            title_id = int(float(row.get("title_id") or 0))
            person_id = int(float(row.get("person_id") or 0))
        except Exception:
            continue
        movies.add(title_id)
        actors.add(person_id)
        top_actor[person_id] += 1
    if not movies:
        return None
    return {
        "observed_movies": len(movies),
        "observed_cast_rows": cast_rows,
        "observed_unique_cast_people": len(actors),
        "observed_cast_per_movie": round(cast_rows / max(1, len(movies)), 4),
        "observed_reuse_ratio": round(cast_rows / max(1, len(actors)), 4),
        "observed_top_actor_movies": int(top_actor.most_common(1)[0][1]) if top_actor else 0,
    }


def build_report(
    base_dir: Path,
    *,
    n_movies: int,
    target_reuse_ratio: float,
    expected_cast_per_movie: float | None,
    min_unique_cast_people: int | None,
) -> dict:
    base_dir = base_dir.resolve()
    person_count = _person_count(base_dir)
    actor_ids = _actor_ids_from_roles(base_dir)
    actor_count = len(actor_ids)
    actor_share = (actor_count / person_count) if person_count else 0.0
    observed = _current_cast_stats(base_dir)
    timeline_stats = _timeline_stats(base_dir)

    if expected_cast_per_movie is None:
        if observed and observed.get("observed_cast_per_movie"):
            expected_cast_per_movie = float(observed["observed_cast_per_movie"])
        else:
            expected_cast_per_movie = 16.0

    estimated_cast_rows = float(n_movies) * float(expected_cast_per_movie)
    max_feasible_reuse_ratio = estimated_cast_rows / max(1, actor_count)
    required_actor_count = int(math.ceil(estimated_cast_rows / max(0.1, target_reuse_ratio)))
    requested_min_unique = int(min_unique_cast_people or math.ceil(n_movies * 0.7))
    required_actor_count = max(required_actor_count, requested_min_unique)
    required_person_count_at_current_mix = (
        int(math.ceil(required_actor_count / actor_share)) if actor_share > 0 else required_actor_count
    )

    title_bank_rows = _count_csv_rows(base_dir / "entities" / "title_bank.csv")
    character_rows = _count_csv_rows(base_dir / "entities" / "character_bank.csv")
    keyword_rows = _count_csv_rows(base_dir / "entities" / "keyword.csv")
    company_rows = _count_csv_rows(base_dir / "entities" / "company.csv")

    checks = {
        "entities_present": person_count > 0 and actor_count > 0,
        "actor_pool_can_meet_reuse_ratio": max_feasible_reuse_ratio <= target_reuse_ratio,
        "actor_pool_can_meet_min_unique": actor_count >= requested_min_unique,
        "title_bank_large_enough": title_bank_rows >= n_movies,
        "character_bank_reasonable": character_rows >= int(math.ceil(estimated_cast_rows * 0.75)),
        "keywords_present": keyword_rows > 0,
        "companies_present": company_rows > 0,
        "person_timelines_complete": timeline_stats["person_json_rows"] == person_count and timeline_stats["missing_timeline_rows"] == 0,
        "person_stages_valid": timeline_stats["invalid_stage_rows"] == 0,
        "person_timeline_order_valid": timeline_stats["invalid_timeline_order_rows"] == 0,
    }
    failed = [name for name, ok in checks.items() if not ok]

    return {
        "dataset_dir": str(base_dir),
        "n_movies": int(n_movies),
        "target_reuse_ratio": float(target_reuse_ratio),
        "expected_cast_per_movie": round(float(expected_cast_per_movie), 4),
        "estimated_cast_rows": int(round(estimated_cast_rows)),
        "person_count": int(person_count),
        "actor_role_people": int(actor_count),
        "actor_share_of_person_pool": round(actor_share, 4),
        "max_feasible_reuse_ratio": round(max_feasible_reuse_ratio, 4),
        "requested_min_unique_cast_people": int(requested_min_unique),
        "required_actor_count": int(required_actor_count),
        "required_person_count_at_current_mix": int(required_person_count_at_current_mix),
        "title_bank_rows": int(title_bank_rows),
        "character_bank_rows": int(character_rows),
        "keyword_rows": int(keyword_rows),
        "company_rows": int(company_rows),
        "observed_cast_stats": observed,
        "timeline_stats": timeline_stats,
        "checks": checks,
        "failed_checks": failed,
        "recommendations": _recommendations(
            failed,
            required_actor_count=required_actor_count,
            required_person_count_at_current_mix=required_person_count_at_current_mix,
            n_movies=n_movies,
            estimated_cast_rows=int(round(estimated_cast_rows)),
        ),
    }


def _recommendations(
    failed: list[str],
    *,
    required_actor_count: int,
    required_person_count_at_current_mix: int,
    n_movies: int,
    estimated_cast_rows: int,
) -> list[str]:
    recommendations: list[str] = []
    if "actor_pool_can_meet_reuse_ratio" in failed or "actor_pool_can_meet_min_unique" in failed:
        recommendations.append(
            "Increase the seeded person pool before step 100. "
            f"This configuration needs at least {required_actor_count} actor-role people; "
            f"at the current role mix that is about {required_person_count_at_current_mix} total people."
        )
    if "title_bank_large_enough" in failed:
        recommendations.append(f"Generate or merge at least {n_movies} in-range title-bank rows.")
    if "character_bank_reasonable" in failed:
        recommendations.append(f"Generate roughly {estimated_cast_rows} character-bank rows for this cast fan-out.")
    if "person_timelines_complete" in failed or "person_stages_valid" in failed or "person_timeline_order_valid" in failed:
        recommendations.append("Repair entities/persons.json career stages and timeline fields before graph generation, e.g. run repair_person_timelines.py or regenerate seeded entities.")
    if not recommendations:
        recommendations.append("Configuration is structurally feasible for the checked gates.")
    return recommendations


def _write_markdown(report: dict, path: Path) -> None:
    failed = report["failed_checks"]
    lines = [
        "# Step 100 Structural Preflight",
        "",
        f"- Dataset: `{report['dataset_dir']}`",
        f"- Target movies: `{report['n_movies']}`",
        f"- Overall: {'PASS' if not failed else 'BLOCKED'}",
        f"- Failed checks: `{', '.join(failed) if failed else 'none'}`",
        "",
        "## Actor Feasibility",
        f"- Person rows: `{report['person_count']}`",
        f"- Actor-role people: `{report['actor_role_people']}`",
        f"- Expected cast rows: `{report['estimated_cast_rows']}`",
        f"- Max feasible reuse ratio: `{report['max_feasible_reuse_ratio']}`",
        f"- Required actor-role people: `{report['required_actor_count']}`",
        f"- Required total people at current role mix: `{report['required_person_count_at_current_mix']}`",
        "",
        "## Entity Banks",
        f"- Title bank rows: `{report['title_bank_rows']}`",
        f"- Character bank rows: `{report['character_bank_rows']}`",
        f"- Keyword rows: `{report['keyword_rows']}`",
        f"- Company rows: `{report['company_rows']}`",
        "",
        "## Person Timelines",
        f"- Person JSON rows: `{report['timeline_stats']['person_json_rows']}`",
        f"- Missing timeline rows: `{report['timeline_stats']['missing_timeline_rows']}`",
        f"- Invalid career-stage rows: `{report['timeline_stats']['invalid_stage_rows']}`",
        f"- Invalid timeline-order rows: `{report['timeline_stats']['invalid_timeline_order_rows']}`",
        "",
        "## Recommendations",
    ]
    lines.extend(f"- {item}" for item in report["recommendations"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether a step-100 configuration is structurally feasible.")
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    parser.add_argument("--n-movies", type=int, required=True)
    parser.add_argument("--target-reuse-ratio", type=float, default=15.0)
    parser.add_argument("--expected-cast-per-movie", type=float, default=None)
    parser.add_argument("--min-unique-cast-people", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--warn-only", action="store_true", help="Print failures but exit 0.")
    args = parser.parse_args()

    report = build_report(
        args.base_dir,
        n_movies=int(args.n_movies),
        target_reuse_ratio=float(args.target_reuse_ratio),
        expected_cast_per_movie=args.expected_cast_per_movie,
        min_unique_cast_people=args.min_unique_cast_people,
    )
    out_dir = (args.out_dir or (args.base_dir / "reports")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "step100_structural_preflight.json"
    md_path = out_dir / "step100_structural_preflight.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(report, md_path)

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    if report["failed_checks"]:
        print(f"Failed checks: {', '.join(report['failed_checks'])}")
        for item in report["recommendations"]:
            print(f"  - {item}")
        if not args.warn_only:
            raise SystemExit(2)
        print("Structural preflight completed with warnings because --warn-only was set.")
        return
    print("Structural preflight passed.")


if __name__ == "__main__":
    main()
