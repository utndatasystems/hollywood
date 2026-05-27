from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from bootstrap_artifacts import load_temporal_regime_plan
from contracts import COUNTRIES, GENRES, PRODUCTION_TIERS
from llm_provider import get_llm_client, safe_json_parse
from model_defaults import model_for_role
from policy_runtime import (
    normalize_world_policy,
    resolve_year_bounds,
    safe_load_json,
    world_policy_path,
    write_json,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = model_for_role("entity_gen")


def _read_json_rows(path: Path) -> list[dict[str, Any]]:
    data = safe_load_json(path, default=[])
    return list(data) if isinstance(data, list) else []


def _split_values(raw: Any) -> list[str]:
    if isinstance(raw, str):
        return [part.strip() for part in raw.replace("|", ",").replace(";", ",").split(",") if part.strip()]
    if isinstance(raw, (list, tuple)):
        return [str(part).strip() for part in raw if str(part).strip()]
    return []


def _top_counts(rows: list[dict[str, Any]], key: str, *, top_n: int = 10) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        for value in _split_values(row.get(key)):
            counter[value] += 1
    return dict(counter.most_common(top_n))


def _mean_latent(rows: list[dict[str, Any]], key: str, *, default: float = 0.5) -> float:
    values = []
    for row in rows:
        try:
            values.append(float(row.get(key, default)))
        except Exception:
            continue
    if not values:
        return float(default)
    return round(float(sum(values)) / max(1, len(values)), 4)


def _load_years(base_dir: Path) -> tuple[int, int]:
    temporal = load_temporal_regime_plan(base_dir, mode="debug")
    if isinstance(temporal, dict):
        try:
            return int(temporal.get("start_year", 1950)), int(temporal.get("end_year", 2025))
        except Exception:
            pass
    tb_path = base_dir / "entities" / "title_bank.csv"
    if not tb_path.exists():
        return 1950, 2025
    try:
        df = pd.read_csv(tb_path, low_memory=False)
    except Exception:
        return 1950, 2025
    return resolve_year_bounds(years=df.get("year", []), fallback_start=1950, fallback_end=2025)


def _build_summary(base_dir: Path) -> dict[str, Any]:
    entities_dir = base_dir / "entities"
    persons = _read_json_rows(entities_dir / "persons.json")
    companies = _read_json_rows(entities_dir / "companies.json")
    person_latent = _read_json_rows(entities_dir / "persons_latent.json")
    company_latent = _read_json_rows(entities_dir / "companies_latent.json")
    start_year, end_year = _load_years(base_dir)
    return {
        "years": {"start_year": start_year, "end_year": end_year},
        "persons": {
            "count": len(persons),
            "career_stages": dict(Counter(str(row.get("career_stage", "prime")) for row in persons).most_common()),
            "roles": _top_counts(persons, "roles", top_n=12),
            "genre_affinity": _top_counts(persons, "genre_affinity", top_n=10),
            "market_fit": _top_counts(persons, "market_fit", top_n=8),
        },
        "companies": {
            "count": len(companies),
            "tiers": dict(Counter(str(row.get("tier", "Mid-Budget")) for row in companies).most_common()),
            "countries": dict(Counter(str(row.get("country", "USA")) for row in companies).most_common(12)),
            "specialty_genres": _top_counts(companies, "specialty_genres", top_n=10),
        },
        "latents": {
            "person_risk_mean": _mean_latent(person_latent, "risk_tolerance"),
            "person_ambition_mean": _mean_latent(person_latent, "artistic_ambition"),
            "person_controversy_mean": _mean_latent(person_latent, "controversy_score", default=0.15),
            "company_risk_mean": _mean_latent(company_latent, "risk_appetite"),
            "company_prestige_mean": _mean_latent(company_latent, "prestige_score"),
            "company_market_sensitivity_mean": _mean_latent(company_latent, "market_trend_sensitivity"),
        },
        "genres": list(GENRES),
        "countries_full": list(COUNTRIES),
        "tiers_full": list(PRODUCTION_TIERS),
        "persons_rows": persons,
        "companies_rows": companies,
    }


def _build_prompt(summary: dict[str, Any]) -> str:
    compact_summary = {
        "years": summary["years"],
        "persons": summary["persons"],
        "companies": summary["companies"],
        "latents": summary["latents"],
    }
    return (
        "You are designing a compact world policy for an IMDb-style synthetic film database.\n"
        "Return JSON only.\n\n"
        "Requirements:\n"
        "1. Keep the response compact and structured.\n"
        "2. Produce numeric biases, not prose.\n"
        "3. Create reusable policy, not movie-by-movie plans.\n"
        "3b. Respect the provided year span exactly, even when it is future-only.\n"
        "4. Include these top-level keys:\n"
        '   "country_market_map", "year_buckets", "company_strategies", "company_strategy_assignments", "talent_boost_rules", "compatibility".\n'
        "5. Each year bucket needs: bucket_id, start_year, end_year, genre_bias, country_bias, market_bias, franchise_pressure, sequel_pressure.\n"
        "6. company_strategies should be a small reusable catalog with strategy_tag, label, genre_focus, tier_bias, title_style, cast_chemistry_target.\n"
        "7. company_strategy_assignments should map company ids to those strategy tags.\n"
        "8. compatibility should provide weight maps for director, company, cast, title, keywords.\n\n"
        f"World summary:\n{json.dumps(compact_summary, ensure_ascii=True)}\n"
    )


def _extract_world_policy_payload(payload: Any) -> dict[str, Any] | None:
    recognized = {
        "country_market_map",
        "year_buckets",
        "company_strategies",
        "company_strategy_assignments",
        "talent_boost_rules",
        "compatibility",
    }
    if isinstance(payload, dict):
        if any(key in payload for key in recognized):
            return payload
        for wrapper in ("world_policy", "policy", "data", "result"):
            inner = payload.get(wrapper)
            if isinstance(inner, dict) and any(key in inner for key in recognized):
                return inner
    if isinstance(payload, list):
        for item in payload:
            extracted = _extract_world_policy_payload(item)
            if extracted is not None:
                return extracted
    return None


def generate_world_policy(base_dir: Path, *, model: str | None = None) -> dict[str, Any]:
    summary = _build_summary(base_dir)
    start_year = int(summary["years"]["start_year"])
    end_year = int(summary["years"]["end_year"])
    parsed = None

    try:
        client = get_llm_client()
        response = client.generate(
            _build_prompt(summary),
            model=model or DEFAULT_MODEL,
            json_mode=True,
            temperature=0.3,
            max_tokens=4096,
            timeout_sec=90.0,
            max_attempts=4,
        )
        parsed = _extract_world_policy_payload(safe_json_parse(response.text))
    except Exception as exc:
        print(f"  World policy LLM fallback: {exc}")

    policy = normalize_world_policy(
        parsed,
        start_year=start_year,
        end_year=end_year,
        genres=GENRES,
        countries=COUNTRIES,
        tiers=PRODUCTION_TIERS,
        company_rows=summary["companies_rows"],
        person_rows=summary["persons_rows"],
    )
    write_json(world_policy_path(base_dir), policy)
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate structured Mirage world policy.")
    parser.add_argument("--base-dir", default=str(BASE_DIR))
    parser.add_argument("--auto", action="store_true", help="Accepted for pipeline parity.")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    policy = generate_world_policy(base_dir, model=args.model)
    print(
        "  Saved world policy:",
        world_policy_path(base_dir),
        f"({len(policy.get('year_buckets', []))} buckets, {len(policy.get('company_strategies', []))} strategies)",
    )


if __name__ == "__main__":
    main()
