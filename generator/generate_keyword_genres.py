"""
Generate keyword-to-genre assignments using LLM.
==================================================
Reads keyword.csv, identifies keywords with missing or default topic_genre
and pop_weight, and uses the LLM to assign proper genre associations.

Usage:
    python generate_keyword_genres.py --auto
    python generate_keyword_genres.py --model <model-name>
    python generate_keyword_genres.py --force   # re-assign ALL keywords
"""
import os
import sys
import json
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env")

try:
    from contracts import GENRES
except ImportError:
    GENRES = ["Action", "Drama", "Comedy", "Horror", "Sci-Fi", "Thriller",
              "Romance", "Animation", "Documentary", "Fantasy", "Crime",
              "Mystery", "Western", "Film-Noir", "Sport", "Family"]

from llm_provider import get_llm_client
from model_defaults import model_for_role

# ═══════════════════════════════════════════════════════════════════════
# Default model
# ═══════════════════════════════════════════════════════════════════════
MODEL = model_for_role("entity_gen")

BATCH_SIZE = 80  # keywords per LLM call

PRICING = {
    "gemini-3.1-flash-lite": {"in": 0.02, "out": 0.08},
    "gemini-2.5-flash": {"in": 0.15, "out": 0.60},
}


def _build_prompt(keywords: list[str], genres: list[str]) -> str:
    """Build LLM prompt for keyword→genre assignment."""
    return f"""You are a film-industry metadata expert.

For each keyword below, assign:
1. "topic_genre": the single most relevant genre from this list: {json.dumps(genres)}
2. "pop_weight": a float 0.005-0.12 indicating how commonly this keyword appears in movies.
   - Very common keywords (love, death, murder, police): 0.06-0.12
   - Moderately common (hospital, robot, time-travel): 0.02-0.05
   - Niche keywords (zeppelin, tundra): 0.005-0.015

Return ONLY a JSON array. No markdown. No explanation.
Each element: {{"keyword": "...", "topic_genre": "...", "pop_weight": 0.5}}

Keywords:
{json.dumps(keywords)}

Output JSON array now."""


def assign_genres_batch(
    llm, model: str, keywords: list[str], genres: list[str]
) -> list[dict]:
    """Call LLM to assign genre + pop_weight for a batch of keywords."""
    prompt = _build_prompt(keywords, genres)

    response = llm.generate(
        prompt,
        model=model,
        json_mode=True,
        temperature=0.3,
        timeout_sec=60,
        max_attempts=5,
    )

    raw = response.text.strip()
    parsed = json.loads(raw)

    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON array, got {type(parsed)}")

    # Validate and clean
    valid_genres = set(genres)
    results = []
    for item in parsed:
        kw = item.get("keyword", "")
        tg = item.get("topic_genre", "Drama")
        pw = item.get("pop_weight", 0.5)

        if tg not in valid_genres:
            tg = "Drama"  # safe fallback
        pw = max(0.005, min(0.12, float(pw)))

        results.append({"keyword": kw, "topic_genre": tg, "pop_weight": pw})

    return results


ALLOWED_SELECTION_BUCKETS = {"exact_anchor", "related_support", "story_specific", "generic"}


def _repair_selection_buckets(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Ensure keyword.csv has research-mode selection buckets.

    Legacy keyword pools may already have topic_genre/pop_weight but predate the
    slot/bucket-aware keyword selector.  We derive buckets from the authored
    global proportions while preserving per-genre coverage: high-popularity rows
    become exact anchors, the next band supports related-genre picks, and the
    long tail stays story-specific.  Rows without a topic genre are treated as
    generic.
    """
    if "selection_bucket" not in df.columns:
        df["selection_bucket"] = ""

    old = df["selection_bucket"].fillna("").astype(str).str.strip().str.lower()
    topic = df.get("topic_genre", pd.Series([""] * len(df))).fillna("").astype(str).str.strip()
    pop = pd.to_numeric(df.get("pop_weight", pd.Series([0.0] * len(df))), errors="coerce").fillna(0.0)

    assigned = pd.Series(["story_specific"] * len(df), index=df.index, dtype=object)
    generic_mask = topic.eq("")
    assigned.loc[generic_mask] = "generic"

    for genre in GENRES:
        genre_mask = topic.eq(str(genre))
        idxs = list(df.loc[genre_mask].index)
        if not idxs:
            continue
        idxs.sort(key=lambda idx: (-float(pop.loc[idx]), str(df.at[idx, "keyword"]) if "keyword" in df.columns else str(idx)))
        n_rows = len(idxs)
        exact_n = max(1, int(math.ceil(n_rows * 0.20)))
        related_n = int(math.ceil(n_rows * 0.30))
        exact_cut = min(n_rows, exact_n)
        related_cut = min(n_rows, exact_cut + related_n)
        assigned.loc[idxs[:exact_cut]] = "exact_anchor"
        assigned.loc[idxs[exact_cut:related_cut]] = "related_support"
        assigned.loc[idxs[related_cut:]] = "story_specific"

    invalid_existing = ~old.isin(ALLOWED_SELECTION_BUCKETS)
    missing_existing = old.eq("")
    exact_coverage_missing = []
    if "topic_genre" in df.columns:
        current_exact = set(topic.loc[old.eq("exact_anchor")].tolist())
        for genre in GENRES:
            if genre not in current_exact and bool(topic.eq(str(genre)).any()):
                exact_coverage_missing.append(str(genre))

    if invalid_existing.any() or missing_existing.any() or exact_coverage_missing:
        df["selection_bucket"] = assigned
    else:
        df["selection_bucket"] = old

    changed = int((old != df["selection_bucket"].fillna("").astype(str).str.strip().str.lower()).sum())
    return df, changed


def _sync_keywords_json(edir: Path, df: pd.DataFrame, all_results: dict | None = None) -> None:
    json_path = edir / "keywords.json"
    if not json_path.exists():
        return
    try:
        kw_json = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(kw_json, list):
            return
        rows_by_keyword = {}
        if "keyword" in df.columns:
            for _, row in df.iterrows():
                rows_by_keyword[str(row.get("keyword", ""))] = row
        json_updated = 0
        for item in kw_json:
            if not isinstance(item, dict):
                continue
            kw_name = str(item.get("keyword", item.get("name", "")))
            if all_results and kw_name in all_results:
                r = all_results[kw_name]
                item["topic_genre"] = r["topic_genre"]
                item["pop_weight"] = r["pop_weight"]
                json_updated += 1
            row = rows_by_keyword.get(kw_name)
            if row is not None and "selection_bucket" in df.columns:
                item["selection_bucket"] = str(row.get("selection_bucket", "story_specific") or "story_specific")
                json_updated += 1
        json_path.write_text(json.dumps(kw_json, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  Synced keyword schema to {json_path} ({json_updated} field updates)")
    except Exception as exc:
        print(f"  Warning: could not update keywords.json: {exc}")


def main():
    parser = argparse.ArgumentParser(
        description="Assign topic_genre and pop_weight to keywords using LLM"
    )
    parser.add_argument("--auto", action="store_true", help="Skip prompts")
    parser.add_argument("--model", default=None, help=f"Override model (default: {MODEL})")
    parser.add_argument("--force", action="store_true",
                        help="Re-assign ALL keywords, not just missing ones")
    parser.add_argument("--base-dir", default=None, help="Base directory")
    args = parser.parse_args()

    base_dir = Path(args.base_dir) if args.base_dir else BASE_DIR
    edir = base_dir / "entities"
    csv_path = edir / "keyword.csv"

    if not csv_path.exists():
        print(f"  keyword.csv not found at {csv_path}")
        sys.exit(1)

    model = args.model or MODEL

    # Load keywords
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} keywords from {csv_path}")
    for column in ("topic_genre", "pop_weight"):
        if column not in df.columns:
            df[column] = np.nan
    df, bucket_updates = _repair_selection_buckets(df)

    # Find keywords needing assignment
    if args.force:
        needs_assignment = df.index.tolist()
    else:
        mask = df["topic_genre"].isna() | df["pop_weight"].isna()
        needs_assignment = df[mask].index.tolist()

    if not needs_assignment:
        if bucket_updates:
            df.to_csv(csv_path, index=False)
            print(f"  Added/repaired selection_bucket for {bucket_updates} keywords in {csv_path}")
            _sync_keywords_json(edir, df)
        else:
            print("  All keywords already have topic_genre, pop_weight, and selection_bucket. Nothing to do.")
        return

    print(f"  {len(needs_assignment)} keywords need genre assignment")

    llm = get_llm_client()

    # Process in batches
    keywords_to_process = df.loc[needs_assignment, "keyword"].tolist()
    n_batches = math.ceil(len(keywords_to_process) / BATCH_SIZE)
    total_in, total_out = 0, 0

    print(f"  Processing {len(keywords_to_process)} keywords in {n_batches} batches "
          f"using {model}")

    all_results = {}
    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(keywords_to_process))
        batch_kws = keywords_to_process[start:end]

        print(f"  Batch {batch_idx + 1}/{n_batches} ({len(batch_kws)} keywords)...",
              end="", flush=True)

        try:
            results = assign_genres_batch(llm, model, batch_kws, GENRES)
            # Map results back by keyword name
            for r in results:
                all_results[r["keyword"]] = r
            print(f" OK ({len(results)} assigned)")
        except Exception as exc:
            print(f" ERROR: {exc}")
            # Fallback: assign Drama + 0.3 for failed batch
            for kw in batch_kws:
                if kw not in all_results:
                    all_results[kw] = {
                        "keyword": kw,
                        "topic_genre": "Drama",
                        "pop_weight": 0.3,
                    }

    # Apply results back to DataFrame
    updated = 0
    for idx in needs_assignment:
        kw_name = df.at[idx, "keyword"]
        if kw_name in all_results:
            r = all_results[kw_name]
            df.at[idx, "topic_genre"] = r["topic_genre"]
            df.at[idx, "pop_weight"] = r["pop_weight"]
            updated += 1

    df, bucket_updates_after = _repair_selection_buckets(df)

    # Save
    df.to_csv(csv_path, index=False)
    print(f"  Updated {updated} keywords and {bucket_updates_after} selection buckets in {csv_path}")
    _sync_keywords_json(edir, df, all_results)

    # Summary
    remaining = df["topic_genre"].isna().sum()
    print(f"\n  Summary: {updated} assigned, {remaining} still missing")
    if remaining > 0:
        print(f"  WARNING: {remaining} keywords still have no topic_genre")


if __name__ == "__main__":
    main()
