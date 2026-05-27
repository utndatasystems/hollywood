from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from contracts import COUNTRIES, GENRES, PRODUCTION_TIERS
from llm_provider import get_llm_client, safe_json_parse
from model_defaults import model_for_role
from policy_runtime import (
    build_concept_pack_slots,
    concept_packs_path,
    keyword_motif_bank_path,
    normalize_concept_packs,
    safe_load_json,
    world_policy_path,
    write_json,
    year_slate_plan_path,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = model_for_role("entity_gen")
BATCH_SIZE = 18


def _is_local_like(model: str | None = None) -> bool:
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    model_name = str(model or "").strip().lower()
    return provider in {"local", "ollama", "vllm", "openai", "tgi", "litellm"} or "qwen" in model_name


def _concept_batch_size(model: str | None = None) -> int:
    return 6 if _is_local_like(model) else 8


def _top_weighted_items(raw: Any, *, limit: int) -> list[dict[str, float]]:
    if not isinstance(raw, dict):
        return []
    rows: list[tuple[str, float]] = []
    for key, value in raw.items():
        name = str(key).strip()
        if not name:
            continue
        try:
            weight = float(value)
        except Exception:
            continue
        rows.append((name, weight))
    rows.sort(key=lambda item: item[1], reverse=True)
    return [{"name": name, "weight": round(weight, 4)} for name, weight in rows[:limit]]


def _compact_company_strategies(world_policy: dict[str, Any]) -> list[dict[str, Any]]:
    strategies = world_policy.get("company_strategies", []) if isinstance(world_policy, dict) else []
    compact: list[dict[str, Any]] = []
    for row in strategies:
        if not isinstance(row, dict):
            continue
        compact.append(
            {
                "strategy_tag": str(row.get("strategy_tag") or "").strip(),
                "label": str(row.get("label") or "").strip(),
                "genre_focus": list(row.get("genre_focus") or [])[:3] if isinstance(row.get("genre_focus"), list) else [],
                "tier_bias": str(row.get("tier_bias") or "").strip(),
            }
        )
    return [row for row in compact if row.get("strategy_tag")]


def _extract_pack_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("packs", "rows", "items", "results", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        if payload.get("pack_id"):
            return [payload]
    return []


def _request_pack_batch(
    client: Any,
    policy: dict[str, Any],
    year_slate_plan: dict[str, Any],
    keyword_motif_bank: dict[str, Any],
    slots: list[dict[str, Any]],
    keyword_hints: dict[str, list[str]],
    company_hints: dict[str, Counter[str]],
    *,
    model: str | None = None,
    depth: int = 0,
) -> list[dict[str, Any]]:
    try:
        response = client.generate(
            _build_prompt(policy, year_slate_plan, keyword_motif_bank, slots, keyword_hints, company_hints),
            model=model or DEFAULT_MODEL,
            json_mode=True,
            temperature=0.25,
            max_tokens=3072,
            timeout_sec=60.0,
            max_attempts=2,
        )
        parsed = safe_json_parse(response.text)
        packs = _extract_pack_rows(parsed)
    except Exception:
        if len(slots) <= 2 or depth >= 3:
            raise
        mid = max(1, len(slots) // 2)
        left = _request_pack_batch(client, policy, year_slate_plan, keyword_motif_bank, slots[:mid], keyword_hints, company_hints, model=model, depth=depth + 1)
        right = _request_pack_batch(client, policy, year_slate_plan, keyword_motif_bank, slots[mid:], keyword_hints, company_hints, model=model, depth=depth + 1)
        return left + right

    if len(packs) >= max(1, int(0.6 * len(slots))):
        return packs
    if len(slots) <= 3 or depth >= 2:
        return packs
    mid = max(1, len(slots) // 2)
    left = _request_pack_batch(client, policy, year_slate_plan, keyword_motif_bank, slots[:mid], keyword_hints, company_hints, model=model, depth=depth + 1)
    right = _request_pack_batch(client, policy, year_slate_plan, keyword_motif_bank, slots[mid:], keyword_hints, company_hints, model=model, depth=depth + 1)
    return left + right


def _read_keywords(base_dir: Path) -> dict[str, list[str]]:
    csv_path = base_dir / "entities" / "keyword.csv"
    if not csv_path.exists():
        return {}
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        return {}
    by_genre: dict[str, list[str]] = {}
    if "topic_genre" not in df.columns or "keyword" not in df.columns:
        return by_genre
    for genre, group in df.groupby("topic_genre"):
        items = group.sort_values("pop_weight", ascending=False) if "pop_weight" in group.columns else group
        by_genre[str(genre)] = [str(value) for value in items["keyword"].head(6).tolist()]
    return by_genre


def _read_company_hints(base_dir: Path) -> dict[str, Counter[str]]:
    json_rows = safe_load_json(base_dir / "entities" / "companies.json", default=[])
    counters = {"country": Counter(), "genre": Counter(), "tier": Counter()}
    if not isinstance(json_rows, list):
        return counters
    for row in json_rows:
        if not isinstance(row, dict):
            continue
        counters["country"][str(row.get("country", "USA"))] += 1
        counters["tier"][str(row.get("tier", "Mid-Budget"))] += 1
        raw = row.get("specialty_genres", [])
        if isinstance(raw, str):
            parts = [part.strip() for part in raw.replace("|", ",").replace(";", ",").split(",") if part.strip()]
        elif isinstance(raw, list):
            parts = [str(part).strip() for part in raw if str(part).strip()]
        else:
            parts = []
        for part in parts:
            counters["genre"][part] += 1
    return counters


def _build_prompt(
    world_policy: dict[str, Any],
    year_slate_plan: dict[str, Any],
    keyword_motif_bank: dict[str, Any],
    slots: list[dict[str, Any]],
    keyword_hints: dict[str, list[str]],
    company_hints: dict[str, Counter[str]],
) -> str:
    bucket_ids = {str(slot.get("bucket_id", "")) for slot in slots}
    markets = {str(slot.get("market", "")) for slot in slots}
    genres_in_batch = {str(slot.get("genre", "")) for slot in slots}
    compact_policy = {
        "start_year": world_policy.get("start_year"),
        "end_year": world_policy.get("end_year"),
        "year_buckets": [
            {
                "bucket_id": str(row.get("bucket_id", "")),
                "start_year": row.get("start_year"),
                "end_year": row.get("end_year"),
                "top_genres": _top_weighted_items(row.get("genre_bias", {}), limit=5),
                "top_countries": _top_weighted_items(row.get("country_bias", {}), limit=4),
                "market_bias": _top_weighted_items(row.get("market_bias", {}), limit=4),
                "franchise_pressure": row.get("franchise_pressure", 0.0),
                "sequel_pressure": row.get("sequel_pressure", 0.0),
            }
            for row in world_policy.get("year_buckets", [])
            if isinstance(row, dict) and str(row.get("bucket_id", "")) in bucket_ids
        ][:6],
        "company_strategies": _compact_company_strategies(world_policy),
    }
    compact_slates = {
        "slates": [
            {
                "slate_id": str(row.get("slate_id", "")),
                "bucket_id": str(row.get("bucket_id", "")),
                "market": str(row.get("market", "")),
                "tier": str(row.get("tier", "")),
                "trending_subgenres": list(row.get("trending_subgenres", []))[:4] if isinstance(row.get("trending_subgenres"), list) else [],
                "priority_motifs": list(row.get("priority_motifs", []))[:4] if isinstance(row.get("priority_motifs"), list) else [],
                "release_pressure": row.get("release_pressure", 0.0),
                "sequel_appetite": row.get("sequel_appetite", 0.0),
                "novelty_target": row.get("novelty_target", 0.0),
                "company_strategy_bias": list(row.get("company_strategy_bias", []))[:3] if isinstance(row.get("company_strategy_bias"), list) else [],
            }
            for row in (year_slate_plan.get("slates", []) if isinstance(year_slate_plan, dict) else [])
            if isinstance(row, dict)
            and str(row.get("bucket_id", "")) in bucket_ids
            and str(row.get("market", "")) in markets
        ][:16],
    }
    compact_motifs = {
        "motifs": [
            {
                "keyword": str(row.get("keyword", "")),
                "topic_genre": str(row.get("topic_genre", "")),
                "motif_family": str(row.get("motif_family", "")),
                "scope_hint": str(row.get("scope_hint", "")),
                "recurrence_strength": row.get("recurrence_strength", 0.0),
            }
            for row in (keyword_motif_bank.get("motifs", []) if isinstance(keyword_motif_bank, dict) else [])
            if isinstance(row, dict) and str(row.get("topic_genre", "")) in genres_in_batch
        ][:24],
    }
    compact_hints = {
        "keywords_by_genre": {genre: keyword_hints.get(genre, [])[:4] for genre in sorted(genres_in_batch)[:6]},
        "top_company_countries": dict(company_hints["country"].most_common(8)),
        "top_company_genres": dict(company_hints["genre"].most_common(8)),
    }
    return (
        "You are generating reusable concept packs for a synthetic film database.\n"
        "Return JSON only.\n\n"
        "Requirements:\n"
        '1. Top-level key must be "packs".\n'
        "2. Create one pack for each requested slot.\n"
        "3. Every pack must include: pack_id, bucket_id, start_year, end_year, genre, tier, country, market,\n"
        "   premise_archetype, conflict_pattern, relationship_motif, ensemble_shape, tone_intensity,\n"
        "   keyword_seed_cluster, title_style, tagline_style, company_strategy_tag, cast_chemistry_target,\n"
        "   franchise_eligible, release_season_bias.\n"
        "4. Keep keyword_seed_cluster short and structured.\n"
        "5. Use company_strategy_tag values that exist in the provided policy.\n\n"
        "6. Keep the pack faithful to the requested slot's genre, tier, country, market, and year bucket.\n"
        "7. Avoid generic reuse across unrelated genres; a War pack should not read like Crime unless the slot says Crime.\n\n"
        "8. Respect slot-level country diversity; if requested slots span many countries, keep that spread instead of collapsing to a few familiar markets.\n\n"
        f"World policy summary:\n{json.dumps(compact_policy, ensure_ascii=True)}\n\n"
        f"Year slate summary:\n{json.dumps(compact_slates, ensure_ascii=True)}\n\n"
        f"Keyword motif summary:\n{json.dumps(compact_motifs, ensure_ascii=True)}\n\n"
        f"Keyword and company hints:\n{json.dumps(compact_hints, ensure_ascii=True)}\n\n"
        f"Requested slots:\n{json.dumps(slots, ensure_ascii=True)}\n"
    )


def _llm_slots(slots: list[dict[str, Any]], *, model: str | None = None) -> list[dict[str, Any]]:
    if _is_local_like(model):
        return list(slots)
    target_cap = min(256, len(slots))
    if len(slots) <= target_cap:
        return list(slots)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for slot in slots:
        key = (str(slot.get("bucket_id", "")), str(slot.get("market", "")))
        grouped[key].append(slot)
    selected: list[dict[str, Any]] = []
    keys = sorted(grouped)
    while len(selected) < target_cap:
        progressed = False
        for key in keys:
            rows = grouped.get(key, [])
            if not rows:
                continue
            selected.append(rows.pop(0))
            progressed = True
            if len(selected) >= target_cap:
                break
        if not progressed:
            break
    return selected


def generate_concept_packs(base_dir: Path, *, model: str | None = None, n_movies: int = 5000) -> dict[str, Any]:
    policy = safe_load_json(world_policy_path(base_dir), default={})
    if not isinstance(policy, dict):
        raise FileNotFoundError("world_policy.json must exist before generating concept packs")
    year_slate_plan = safe_load_json(year_slate_plan_path(base_dir), default={}) or {}
    keyword_motif_bank = safe_load_json(keyword_motif_bank_path(base_dir), default={}) or {}
    slots = build_concept_pack_slots(policy, genres=GENRES, tiers=PRODUCTION_TIERS, countries=COUNTRIES, n_movies=n_movies)
    keyword_hints = _read_keywords(base_dir)
    company_hints = _read_company_hints(base_dir)
    parsed_rows: list[dict[str, Any]] = []
    try:
        client = get_llm_client()
    except Exception as exc:
        print(f"  Concept pack LLM unavailable, using fallback: {exc}")
        client = None

    if client is not None:
        batch_size = _concept_batch_size(model)
        llm_slots = _llm_slots(slots, model=model)
        n_batches = (len(llm_slots) + batch_size - 1) // batch_size if llm_slots else 0
        for batch_idx in range(n_batches):
            batch_slots = llm_slots[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            try:
                parsed_rows.extend(
                    _request_pack_batch(
                        client,
                        policy,
                        year_slate_plan,
                        keyword_motif_bank,
                        batch_slots,
                        keyword_hints,
                        company_hints,
                        model=model,
                    )
                )
            except Exception as exc:
                print(f"  Concept pack batch {batch_idx + 1}/{max(1, n_batches)} fallback: {exc}")

    packs = normalize_concept_packs(
        {"packs": parsed_rows},
        world_policy=policy,
        genres=GENRES,
        tiers=PRODUCTION_TIERS,
        countries=COUNTRIES,
        n_movies=n_movies,
    )
    write_json(concept_packs_path(base_dir), packs)
    return packs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reusable Mirage concept packs.")
    parser.add_argument("--base-dir", default=str(BASE_DIR))
    parser.add_argument("--auto", action="store_true", help="Accepted for pipeline parity.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--n-movies", type=int, default=5000)
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    packs = generate_concept_packs(base_dir, model=args.model, n_movies=int(args.n_movies))
    print(
        "  Saved concept packs:",
        concept_packs_path(base_dir),
        f"({len(packs.get('packs', []))} packs)",
    )


if __name__ == "__main__":
    main()
