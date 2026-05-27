from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

from contracts import ENTITY_COUNTS, SNAPSHOT_CONFIG
from llm_provider import get_llm_client, safe_json_parse
from model_defaults import model_for_role
from policy_runtime import (
    franchise_bibles_path,
    normalize_franchise_bibles,
    safe_load_json,
    write_json,
)
from world_state import WorldState

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = model_for_role("entity_gen")
BATCH_SIZE = 18


def _is_local_like(model: str | None = None) -> bool:
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    model_name = str(model or "").strip().lower()
    return provider in {"local", "ollama", "vllm", "openai", "tgi", "litellm"} or "qwen" in model_name


def _franchise_batch_size(model: str | None = None) -> int:
    return 4 if _is_local_like(model) else BATCH_SIZE


def _extract_bible_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("bibles", "rows", "items", "results", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        if payload.get("franchise_id"):
            return [payload]
    return []


def _build_prompt(batch: list[dict[str, Any]], context: dict[str, Any]) -> str:
    return (
        "You are generating franchise continuity bibles for a synthetic film database.\n"
        "Return JSON only.\n\n"
        "Requirements:\n"
        '1. Top-level key must be "bibles".\n'
        "2. Return one bible per franchise.\n"
        "3. Each bible must include: franchise_id, franchise_name, genre, tier, installments,\n"
        "   continuity_anchors, recurring_motifs, keyword_families, title_style, subtitle_tokens,\n"
        "   release_season_bias, company_strategy_tag, cast_chemistry_target,\n"
        "   carryover_director_bias, carryover_cast_bias.\n"
        "3b. Never keep placeholder names like Franchise_1. Derive a reusable franchise name from the seed titles and taglines.\n"
        "4. Keep lists short and machine-friendly.\n"
        "5. carryover_director_bias and carryover_cast_bias must be 0..1.\n\n"
        f"Context:\n{json.dumps(context, ensure_ascii=True)}\n\n"
        f"Franchises:\n{json.dumps(batch, ensure_ascii=True)}\n"
    )


def _load_world(base_dir: Path, n_movies: int) -> WorldState:
    ENTITY_COUNTS["movies"] = int(n_movies)
    world = WorldState(
        str(base_dir),
        seed=SNAPSHOT_CONFIG["seed"],
        config_path=None,
        workspace=None,
    )
    world.load()
    return world


def _safe_int(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except Exception:
        return None


def _load_title_bank_rows(base_dir: Path) -> list[dict[str, Any]]:
    path = base_dir / "entities" / "title_bank.csv"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not isinstance(row, dict):
                    continue
                rows.append(
                    {
                        "title": str(row.get("title") or "").strip(),
                        "tagline": str(row.get("tagline") or "").strip(),
                        "genre_hint": str(row.get("genre_hint") or "").strip(),
                        "year": _safe_int(row.get("year")),
                    }
                )
    except Exception:
        return []
    return rows


def _build_title_hints(
    franchises: list[dict[str, Any]],
    title_bank_rows: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    by_genre: dict[str, list[dict[str, Any]]] = {}
    for row in title_bank_rows:
        genre_hint = str(row.get("genre_hint") or "").strip()
        by_genre.setdefault(genre_hint, []).append(row)

    hints: dict[int, list[dict[str, Any]]] = {}
    for franchise in franchises:
        franchise_id = int(franchise.get("franchise_id", 0) or 0)
        genre = str(franchise.get("genre") or "").strip()
        years = [_safe_int(value) for value in franchise.get("installment_years", []) or []]
        target_years = [value for value in years if value is not None]
        candidates = by_genre.get(genre, []) or title_bank_rows
        scored: list[tuple[tuple[int, int, str], dict[str, Any]]] = []
        for row in candidates:
            title = str(row.get("title") or "").strip()
            if not title:
                continue
            row_year = _safe_int(row.get("year"))
            same_genre_penalty = 0 if str(row.get("genre_hint") or "").strip() == genre else 1
            if row_year is None or not target_years:
                year_distance = 999
            else:
                year_distance = min(abs(row_year - year) for year in target_years)
            scored.append(((same_genre_penalty, year_distance, title.lower()), row))
        scored.sort(key=lambda item: item[0])
        selected: list[dict[str, Any]] = []
        seen_titles: set[str] = set()
        for _sort_key, row in scored:
            title = str(row.get("title") or "").strip()
            lowered = title.lower()
            if lowered in seen_titles:
                continue
            seen_titles.add(lowered)
            selected.append(
                {
                    "title": title,
                    "tagline": str(row.get("tagline") or "").strip(),
                    "year": _safe_int(row.get("year")),
                    "genre_hint": str(row.get("genre_hint") or "").strip(),
                }
            )
            if len(selected) >= 6:
                break
        hints[franchise_id] = selected
    return hints


def generate_franchise_bibles(base_dir: Path, *, n_movies: int, model: str | None = None) -> dict[str, Any]:
    world = _load_world(base_dir, n_movies)
    context = {
        "world_policy": safe_load_json(base_dir / "world_policy.json", default={}),
        "year_slate_count": len((safe_load_json(base_dir / "year_slate_plan.json", default={}) or {}).get("slates", [])),
        "concept_pack_count": len((safe_load_json(base_dir / "concept_packs.json", default={}) or {}).get("packs", [])),
    }
    franchises = [
        {
            "franchise_id": int(row.get("franchise_id", 0) or 0),
            "name": str(row.get("name", "")),
            "genre": str(row.get("genre", "")),
            "tier": str(row.get("tier", "")),
            "n_movies": int(row.get("n_movies", 0) or 0),
            "installment_years": list(row.get("installment_years", [])),
        }
        for row in getattr(world, "franchises", [])
        if isinstance(row, dict)
    ]
    title_hints = _build_title_hints(franchises, _load_title_bank_rows(base_dir))
    parsed_rows: list[dict[str, Any]] = []
    client = None
    try:
        client = get_llm_client()
    except Exception:
        client = None
    batch_size = _franchise_batch_size(model)
    n_batches = math.ceil(len(franchises) / batch_size) if franchises else 0
    for batch_idx in range(n_batches):
        batch = []
        for row in franchises[batch_idx * batch_size : (batch_idx + 1) * batch_size]:
            franchise_id = int(row.get("franchise_id", 0) or 0)
            hint_rows = title_hints.get(franchise_id, [])
            batch.append(
                {
                    **row,
                    "seed_titles": [str(item.get("title") or "") for item in hint_rows[:4] if str(item.get("title") or "").strip()],
                    "seed_taglines": [str(item.get("tagline") or "") for item in hint_rows[:3] if str(item.get("tagline") or "").strip()],
                }
            )
        if client is None:
            continue
        try:
            response = client.generate(
                _build_prompt(batch, context),
                model=model or DEFAULT_MODEL,
                json_mode=True,
                temperature=0.2,
                max_tokens=4096,
                timeout_sec=90.0,
                max_attempts=4,
            )
            parsed = safe_json_parse(response.text)
            parsed_rows.extend(_extract_bible_rows(parsed))
        except Exception as exc:
            print(f"  Franchise bible batch {batch_idx + 1}/{max(1, n_batches)} fallback: {exc}")
    payload = normalize_franchise_bibles({"bibles": parsed_rows}, franchises=franchises, title_hints=title_hints)
    write_json(franchise_bibles_path(base_dir), payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate franchise continuity bibles for Mirage.")
    parser.add_argument("--base-dir", default=str(BASE_DIR))
    parser.add_argument("--n-movies", type=int, required=True)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    payload = generate_franchise_bibles(base_dir, n_movies=int(args.n_movies), model=args.model)
    print(
        "  Saved franchise bibles:",
        franchise_bibles_path(base_dir),
        f"({len(payload.get('bibles', []))} bibles)",
    )


if __name__ == "__main__":
    main()
