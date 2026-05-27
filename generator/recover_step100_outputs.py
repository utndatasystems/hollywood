from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bootstrap_artifacts import audit_critic_report
from feather_sink import df_to_arrow, read_table
from generation_critic import _persist_critic_logs, _precritic_structural_gate
from secondary_tables import generate_alternate_titles, generate_awards
from world_state import WorldState
from assembly import pick_keywords


BASE_DIR = Path(__file__).resolve().parent


def _stable_seed(*parts: Any) -> int:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.blake2b(raw, digest_size=8).hexdigest()[:8], 16)


def _stable_rng(*parts: Any) -> np.random.RandomState:
    return np.random.RandomState(_stable_seed(*parts))


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def _backup_existing(paths: list[Path], backup_dir: Path) -> list[str]:
    backed_up: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup_dir / path.name)
        backed_up.append(path.name)
    return backed_up


def _canonical_genre(text: object) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    lowered = value.lower()
    alias_map = {
        "film noir": "Film-Noir",
        "film-noir": "Film-Noir",
        "sci fi": "Sci-Fi",
        "sci-fi": "Sci-Fi",
        "scifi": "Sci-Fi",
        "super hero": "Superhero",
    }
    return alias_map.get(lowered, value)


def _build_minimal_concept(movie_row: pd.Series) -> dict[str, Any]:
    return {
        "movie_id": int(movie_row["title_id"]),
        "genre": str(movie_row.get("genre", "") or ""),
        "year": int(movie_row.get("year", 0) or 0),
        "country": str(movie_row.get("country", "") or ""),
        "tier": str(movie_row.get("production_tier", "Mid") or "Mid"),
        "language": str(movie_row.get("language", "") or ""),
        "policy_targets": {},
    }


def _group_rows(df: pd.DataFrame, key: str) -> dict[int, list[dict[str, Any]]]:
    if df is None or len(df) == 0 or key not in df.columns:
        return {}
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in df.to_dict(orient="records"):
        try:
            out[int(row[key])].append(row)
        except Exception:
            continue
    return out


def _load_table(base_dir: Path, name: str) -> pd.DataFrame:
    return read_table(str(base_dir / name), name)


def _repair_zero_exact_topic_support(
    *,
    world: WorldState,
    movie_df: pd.DataFrame,
    movie_keyword_df: pd.DataFrame,
    movie_companies_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    work = {
        "movie": movie_df,
        "movie_keyword": movie_keyword_df,
    }
    gate = _precritic_structural_gate(work, world)
    title_ids = [int(v) for v in gate.get("zero_exact_topic_support_ids", []) or []]
    if not title_ids:
        return movie_keyword_df, []

    world.movie_progress_log_path = None
    world.movie_progress_log_latest_path = None
    usage_counts = Counter(int(v) for v in movie_keyword_df.get("keyword_id", pd.Series(dtype=int)).tolist())
    world._keyword_usage_counts = usage_counts

    company_map: dict[int, list[int]] = defaultdict(list)
    if len(movie_companies_df) > 0 and {"title_id", "company_id"}.issubset(movie_companies_df.columns):
        for row in movie_companies_df[["title_id", "company_id"]].itertuples(index=False):
            company_map[int(row.title_id)].append(int(row.company_id))

    repaired_df = movie_keyword_df.copy()
    repairs: list[dict[str, Any]] = []
    for title_id in title_ids:
        movie_rows = movie_df.loc[movie_df["title_id"] == int(title_id)]
        if len(movie_rows) == 0:
            continue
        movie_row = movie_rows.iloc[0]
        current_mask = repaired_df["title_id"].astype(int) == int(title_id)
        current_keyword_ids = repaired_df.loc[current_mask, "keyword_id"].astype(int).tolist()
        current_count = max(1, len(current_keyword_ids))
        for keyword_id in current_keyword_ids:
            usage_counts[int(keyword_id)] -= 1
            if usage_counts[int(keyword_id)] <= 0:
                usage_counts.pop(int(keyword_id), None)

        concept = _build_minimal_concept(movie_row)
        company_ids = company_map.get(int(title_id), [])
        picked = pick_keywords(world, concept, n=current_count, company_ids=company_ids)
        if not picked:
            for keyword_id in current_keyword_ids:
                usage_counts[int(keyword_id)] += 1
            continue

        repaired_df = repaired_df.loc[~current_mask].copy()
        new_rows = pd.DataFrame(
            [{"title_id": int(title_id), "keyword_id": int(keyword_id)} for keyword_id in picked]
        )
        repaired_df = pd.concat([repaired_df, new_rows], ignore_index=True)
        for keyword_id in picked:
            usage_counts[int(keyword_id)] += 1

        repairs.append(
            {
                "title_id": int(title_id),
                "old_keyword_ids": [int(v) for v in current_keyword_ids],
                "new_keyword_ids": [int(v) for v in picked],
                "genre": str(movie_row.get("genre", "") or ""),
            }
        )

    repaired_df["title_id"] = repaired_df["title_id"].astype(int)
    repaired_df["keyword_id"] = repaired_df["keyword_id"].astype(int)
    repaired_df = repaired_df.sort_values(["title_id", "keyword_id"], kind="stable").reset_index(drop=True)
    return repaired_df, repairs


def _regenerate_awards(
    *,
    movie_df: pd.DataFrame,
    movie_directors_df: pd.DataFrame,
    cast_df: pd.DataFrame,
    crew_df: pd.DataFrame,
    world: WorldState,
) -> pd.DataFrame:
    cast_groups = _group_rows(cast_df.sort_values(["title_id", "billing_order"], kind="stable"), "title_id")
    crew_groups = _group_rows(crew_df, "title_id")
    director_groups = _group_rows(movie_directors_df, "title_id")

    world.person_award_wins = Counter()
    rows: list[dict[str, Any]] = []
    movies_sorted = movie_df.sort_values(["year", "title_id"], kind="stable")
    for movie_row in movies_sorted.to_dict(orient="records"):
        title_id = int(movie_row["title_id"])
        director_rows = director_groups.get(title_id, [])
        director_id = int(director_rows[0]["director_id"]) if director_rows else None
        cast_rows = cast_groups.get(title_id, [])
        crew_rows = crew_groups.get(title_id, [])
        award_rows = generate_awards(
            title_id=title_id,
            year=int(movie_row.get("year", 0) or 0),
            rating=float(movie_row.get("rating", 0.0) or 0.0),
            tier=str(movie_row.get("production_tier", "Mid") or "Mid"),
            director_id=director_id,
            cast=cast_rows,
            crew_rows=crew_rows,
            rng=_stable_rng("recovery-awards", world.seed, title_id),
            award_campaign=float(movie_row.get("award_campaign_strength", 0.0) or 0.0),
            world=world,
        )
        rows.extend(award_rows)
        for award_row in award_rows:
            if str(award_row.get("outcome", "")) == "Won" and award_row.get("person_id") not in (None, "", 0):
                world.person_award_wins[int(award_row["person_id"])] += 1
    return pd.DataFrame(rows)


def _regenerate_alternate_titles(movie_df: pd.DataFrame, world: WorldState) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for movie_row in movie_df.sort_values(["title_id"], kind="stable").to_dict(orient="records"):
        rows.extend(
            generate_alternate_titles(
                title_id=int(movie_row["title_id"]),
                title=str(movie_row.get("title", "") or ""),
                language=str(movie_row.get("language", "") or ""),
                rng=_stable_rng("recovery-alt-titles", world.seed, int(movie_row["title_id"])),
            )
        )
    return pd.DataFrame(rows)


def _materialize_csv_if_present(base_dir: Path, table_name: str, written: list[str]) -> None:
    df = _load_table(base_dir, table_name)
    if len(df) == 0:
        return
    _write_csv(df, base_dir / f"{table_name}.csv")
    written.append(f"{table_name}.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover root outputs after a post-step-100 critic/save failure.")
    parser.add_argument("--base-dir", default=str(BASE_DIR), help="Run directory to recover.")
    parser.add_argument(
        "--run-full-critic",
        action="store_true",
        help="Run the full post-generation critic after structural recovery. Off by default to avoid extra token spend.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    backup_dir = base_dir / "_dev" / "recovery_backups" / datetime.now().strftime("%Y%m%d_%H%M%S")

    movie_df = _load_table(base_dir, "movie")
    cast_df = _load_table(base_dir, "cast_info")
    movie_directors_df = _load_table(base_dir, "movie_directors")
    movie_companies_df = _load_table(base_dir, "movie_companies")
    movie_keyword_df = _load_table(base_dir, "movie_keyword")
    movie_crew_df = _load_table(base_dir, "movie_crew")

    if len(movie_df) == 0:
        raise FileNotFoundError(f"movie.arrow/movie.csv not found in {base_dir}")

    print(f"Loaded recovery source tables from {base_dir}")
    print(f"  movie rows          : {len(movie_df):,}")
    print(f"  cast rows           : {len(cast_df):,}")
    print(f"  movie_directors rows: {len(movie_directors_df):,}")
    print(f"  movie_companies rows: {len(movie_companies_df):,}")
    print(f"  movie_keyword rows  : {len(movie_keyword_df):,}")
    print(f"  movie_crew rows     : {len(movie_crew_df):,}")

    world = WorldState(str(base_dir), seed=42)
    world.load()

    repaired_keyword_df, keyword_repairs = _repair_zero_exact_topic_support(
        world=world,
        movie_df=movie_df,
        movie_keyword_df=movie_keyword_df,
        movie_companies_df=movie_companies_df,
    )

    files_to_backup = [
        base_dir / "movie_keyword.arrow",
        base_dir / "movie_keyword.csv",
        base_dir / "critic_report.json",
        base_dir / "critic" / "post_generation_critic_report.json",
    ]
    backed_up = _backup_existing(files_to_backup, backup_dir)

    if keyword_repairs:
        df_to_arrow(repaired_keyword_df, str(base_dir / "movie_keyword.arrow"), table_name="movie_keyword")
        _write_csv(repaired_keyword_df, base_dir / "movie_keyword.csv")
        movie_keyword_df = repaired_keyword_df
        print(f"Repaired exact-topic keyword support for {len(keyword_repairs)} title(s)")
    else:
        _write_csv(movie_keyword_df, base_dir / "movie_keyword.csv")

    awards_df = _regenerate_awards(
        movie_df=movie_df,
        movie_directors_df=movie_directors_df,
        cast_df=cast_df,
        crew_df=movie_crew_df,
        world=world,
    )
    alternate_titles_df = _regenerate_alternate_titles(movie_df, world)

    df_to_arrow(awards_df, str(base_dir / "awards.arrow"), table_name="awards")
    _write_csv(awards_df, base_dir / "awards.csv")
    df_to_arrow(alternate_titles_df, str(base_dir / "alternate_titles.arrow"), table_name="alternate_titles")
    _write_csv(alternate_titles_df, base_dir / "alternate_titles.csv")

    written_csvs: list[str] = ["movie_keyword.csv", "awards.csv", "alternate_titles.csv"]
    _write_csv(movie_df, base_dir / "movie.csv")
    written_csvs.append("movie.csv")
    _write_csv(cast_df, base_dir / "cast_info.csv")
    written_csvs.append("cast_info.csv")
    _write_csv(movie_directors_df, base_dir / "movie_directors.csv")
    written_csvs.append("movie_directors.csv")
    _write_csv(movie_companies_df, base_dir / "movie_companies.csv")
    written_csvs.append("movie_companies.csv")

    for table_name in [
        "movie_crew",
        "release_dates",
        "movie_links",
        "person_demographics",
        "tv_series",
        "seasons",
        "episodes",
        "ratings_breakdown",
        "persons_enriched",
        "companies_enriched",
    ]:
        _materialize_csv_if_present(base_dir, table_name, written_csvs)

    critic_result = {
        "movie": movie_df,
        "movie_keyword": movie_keyword_df,
        "alternate_titles": alternate_titles_df,
    }
    if args.run_full_critic:
        from generation_critic import run_post_generation_critic

        critic_result, critic_report = run_post_generation_critic(
            critic_result,
            world,
            enabled=True,
            log_dir=str(base_dir / "critic"),
        )
    else:
        structural_gate = _precritic_structural_gate(critic_result, world)
        placeholder_title_ids = [int(v) for v in structural_gate.get("placeholder_title_ids", []) or []]
        zero_exact_ids = [int(v) for v in structural_gate.get("zero_exact_topic_support_ids", []) or []]
        if placeholder_title_ids or zero_exact_ids:
            critic_report = {
                "status": "structural_failure",
                "reason": (
                    f"placeholder taglines in title_ids={placeholder_title_ids[:8]}; "
                    f"zero exact-topic keyword support in title_ids={zero_exact_ids[:8]}"
                ).strip("; "),
                "applied": 0,
                "sampled_titles": [],
                "placeholder_title_ids": placeholder_title_ids,
                "zero_exact_topic_support_ids": zero_exact_ids,
                "non_fatal": True,
                "llm_skipped": True,
            }
        else:
            critic_report = {
                "status": "structural_gate_passed",
                "applied": 0,
                "sampled_titles": [],
                "placeholder_title_ids": [],
                "zero_exact_topic_support_ids": [],
                "non_fatal": True,
                "llm_skipped": True,
            }
        _persist_critic_logs(str(base_dir / "critic"), "", "", critic_report)
        audit_critic_report(critic_report)

    with open(base_dir / "critic_report.json", "w", encoding="utf-8") as fh:
        json.dump(critic_report, fh, ensure_ascii=False, indent=2)

    recovery_report = {
        "status": "ok",
        "base_dir": str(base_dir),
        "backed_up_files": backed_up,
        "keyword_repairs": keyword_repairs,
        "awards_rows": int(len(awards_df)),
        "alternate_titles_rows": int(len(alternate_titles_df)),
        "critic_status": str(critic_report.get("status", "")),
        "critic_non_fatal": bool(critic_report.get("non_fatal", False)),
        "written_csvs": written_csvs,
    }
    with open(base_dir / "recovery_report.json", "w", encoding="utf-8") as fh:
        json.dump(recovery_report, fh, ensure_ascii=False, indent=2)

    print("Recovery complete")
    print(f"  backup dir          : {backup_dir if backed_up else '(none needed)'}")
    print(f"  keyword repairs     : {len(keyword_repairs)}")
    print(f"  awards rows         : {len(awards_df):,}")
    print(f"  alternate titles    : {len(alternate_titles_df):,}")
    print(f"  critic status       : {critic_report.get('status', 'unknown')}")


if __name__ == "__main__":
    main()
