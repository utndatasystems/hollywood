from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from contracts import GENRES, PRODUCTION_TIERS
from llm_provider import get_llm_client, safe_json_parse
from model_defaults import model_for_role
from policy_runtime import (
    normalize_year_slate_plan,
    safe_load_json,
    world_policy_path,
    write_json,
    year_slate_plan_path,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = model_for_role("entity_gen")


def _is_local_like(model: str | None = None) -> bool:
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    model_name = str(model or "").strip().lower()
    return provider in {"local", "ollama", "vllm", "openai", "tgi", "litellm"} or "qwen" in model_name


def _batch_size_for_year_slates(model: str | None = None) -> int:
    return 12 if _is_local_like(model) else 8


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


def _extract_slate_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("slates", "rows", "items", "results", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        if payload.get("slate_id"):
            return [payload]
    return []


def _request_slate_rows(
    client: Any,
    world_policy: dict[str, Any],
    bucket: dict[str, Any],
    requested_rows: list[dict[str, Any]],
    *,
    model: str | None = None,
    depth: int = 0,
) -> list[dict[str, Any]]:
    try:
        response = client.generate(
            _build_prompt(world_policy, bucket, requested_rows),
            model=model or DEFAULT_MODEL,
            json_mode=True,
            temperature=0.2,
            max_tokens=3072,
            timeout_sec=60.0,
            max_attempts=2,
        )
        parsed = safe_json_parse(response.text)
        rows = _extract_slate_rows(parsed)
    except Exception:
        if len(requested_rows) <= 2 or depth >= 3:
            raise
        mid = max(1, len(requested_rows) // 2)
        left = _request_slate_rows(client, world_policy, bucket, requested_rows[:mid], model=model, depth=depth + 1)
        right = _request_slate_rows(client, world_policy, bucket, requested_rows[mid:], model=model, depth=depth + 1)
        return left + right

    if len(rows) >= max(1, int(0.6 * len(requested_rows))):
        return rows
    if len(requested_rows) <= 4 or depth >= 2:
        return rows
    mid = max(1, len(requested_rows) // 2)
    left = _request_slate_rows(client, world_policy, bucket, requested_rows[:mid], model=model, depth=depth + 1)
    right = _request_slate_rows(client, world_policy, bucket, requested_rows[mid:], model=model, depth=depth + 1)
    return left + right


def _requested_rows_for_bucket(world_policy: dict[str, Any], bucket: dict[str, Any]) -> list[dict[str, Any]]:
    country_market_map = world_policy.get("country_market_map", {}) if isinstance(world_policy, dict) else {}
    markets = sorted({str(value) for value in country_market_map.values() if str(value).strip()} | {"Global"})
    bucket_id = str(bucket.get("bucket_id", ""))
    start_year = int(bucket.get("start_year", world_policy.get("start_year", 1975)))
    end_year = int(bucket.get("end_year", world_policy.get("end_year", start_year)))
    return [
        {
            "slate_id": f"{bucket_id}|{market}|{tier}",
            "bucket_id": bucket_id,
            "start_year": start_year,
            "end_year": end_year,
            "market": market,
            "tier": str(tier),
        }
        for market in markets
        for tier in PRODUCTION_TIERS
    ]


def _llm_requested_rows_for_bucket(requested_rows: list[dict[str, Any]], *, model: str | None = None) -> list[dict[str, Any]]:
    if _is_local_like(model):
        return list(requested_rows)

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    major_markets = {"Global", "North America", "Europe", "Asia"}

    def add_rows(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            slate_id = str(row.get("slate_id") or "")
            if not slate_id or slate_id in seen:
                continue
            selected.append(row)
            seen.add(slate_id)

    add_rows([row for row in requested_rows if str(row.get("market")) == "Global"])
    add_rows([row for row in requested_rows if str(row.get("tier")) == "Mid" and str(row.get("market")) in major_markets])
    add_rows([row for row in requested_rows if str(row.get("tier")) == "A" and str(row.get("market")) in {"North America", "Europe", "Asia"}])
    add_rows([row for row in requested_rows if str(row.get("tier")) == "Indie" and str(row.get("market")) not in major_markets])
    add_rows(requested_rows)
    return selected[:10]


def _build_prompt(world_policy: dict[str, Any], bucket: dict[str, Any], requested_rows: list[dict[str, Any]]) -> str:
    compact_policy = {
        "start_year": world_policy.get("start_year"),
        "end_year": world_policy.get("end_year"),
        "markets": sorted({str(value).strip() for value in (world_policy.get("country_market_map", {}) or {}).values() if str(value).strip()} | {"Global"}),
        "company_strategies": _compact_company_strategies(world_policy),
    }
    bucket_summary = {
        "bucket_id": bucket.get("bucket_id"),
        "start_year": bucket.get("start_year"),
        "end_year": bucket.get("end_year"),
        "top_genres": _top_weighted_items(bucket.get("genre_bias", {}), limit=8),
        "top_countries": _top_weighted_items(bucket.get("country_bias", {}), limit=8),
        "market_bias": _top_weighted_items(bucket.get("market_bias", {}), limit=6),
        "franchise_pressure": bucket.get("franchise_pressure", 0.0),
        "sequel_pressure": bucket.get("sequel_pressure", 0.0),
    }
    return (
        "You are generating reusable year-slate planning artifacts for a synthetic film database.\n"
        "Return JSON only.\n\n"
        "Requirements:\n"
        '1. Top-level key must be "slates".\n'
        "2. Create one row for each requested bucket_id x market x tier slot.\n"
        "3. Each slate must include: slate_id, bucket_id, start_year, end_year, market, tier,\n"
        "   trending_subgenres, priority_motifs, motif_drift, release_pressure, release_season_bias,\n"
        "   sequel_appetite, novelty_target, company_strategy_bias.\n"
        "4. Keep lists short and machine-friendly.\n"
        "5. release_pressure, sequel_appetite, novelty_target must be 0..1 floats.\n\n"
        "6. These rows are cross-genre industry-planning rows, not single-movie or single-genre rows.\n"
        "7. Avoid repeating the same crime-heavy motifs for every market and tier unless the bucket summary truly supports it.\n"
        "8. Use short semantic tags like market_cycle, discovery_impulse, ensemble_pressure, analogue_texture.\n\n"
        f"World policy summary:\n{json.dumps(compact_policy, ensure_ascii=True)}\n\n"
        f"Year bucket summary:\n{json.dumps(bucket_summary, ensure_ascii=True)}\n\n"
        f"Requested rows:\n{json.dumps(requested_rows, ensure_ascii=True)}\n"
    )


def generate_year_slate_plan(base_dir: Path, *, model: str | None = None) -> dict[str, Any]:
    world_policy = safe_load_json(world_policy_path(base_dir), default={})
    if not isinstance(world_policy, dict):
        raise FileNotFoundError("world_policy.json must exist before generating year_slate_plan.json")

    parsed_rows: list[dict[str, Any]] = []
    try:
        client = get_llm_client()
    except Exception as exc:
        print(f"  Year slate LLM unavailable, using fallback: {exc}")
        client = None

    year_buckets = [row for row in world_policy.get("year_buckets", []) if isinstance(row, dict)]
    consecutive_failures = 0
    for bucket_idx, bucket in enumerate(year_buckets, start=1):
        requested_rows = _requested_rows_for_bucket(world_policy, bucket)
        if client is None:
            continue
        llm_rows = _llm_requested_rows_for_bucket(requested_rows, model=model)
        chunk_size = _batch_size_for_year_slates(model)
        for offset in range(0, len(llm_rows), chunk_size):
            chunk = llm_rows[offset : offset + chunk_size]
            try:
                parsed_rows.extend(_request_slate_rows(client, world_policy, bucket, chunk, model=model))
                consecutive_failures = 0
            except Exception as exc:
                consecutive_failures += 1
                print(f"  Year slate bucket {bucket_idx}/{max(1, len(year_buckets))} chunk fallback: {exc}")
                if consecutive_failures >= 4 and not _is_local_like(model):
                    print("  Year slate provider unstable, finishing remaining rows via deterministic fallback.")
                    client = None
                    break

    plan = normalize_year_slate_plan(
        {"slates": parsed_rows},
        world_policy=world_policy,
        genres=GENRES,
        tiers=PRODUCTION_TIERS,
    )
    write_json(year_slate_plan_path(base_dir), plan)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reusable Mirage year-slate planning artifacts.")
    parser.add_argument("--base-dir", default=str(BASE_DIR))
    parser.add_argument("--auto", action="store_true", help="Accepted for pipeline parity.")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    plan = generate_year_slate_plan(base_dir, model=args.model)
    print(
        "  Saved year slate plan:",
        year_slate_plan_path(base_dir),
        f"({len(plan.get('slates', []))} slates)",
    )


if __name__ == "__main__":
    main()
