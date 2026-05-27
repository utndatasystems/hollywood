#!/usr/bin/env python3
"""Company generator for Mirage.

Research mode consumes `company_lexicon.json` plus `modeling_priors.json`.
Debug mode keeps the older fixed-vocabulary fallback.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent
ENTITY_DIR = BASE_DIR / "entities"

sys.path.insert(0, str(BASE_DIR))

from bootstrap_artifacts import (
    audit_artifact_usage,
    audit_fallback_hit,
    current_mode,
    load_company_lexicon,
    load_modeling_priors_artifact,
    require_payload_value,
)
from contracts import COMPANY_COUNTRY_WEIGHTS, COUNTRIES, DIRECTOR_STYLES, GENRES, STYLE_TAGS
from policy_runtime import company_lexicon_path, modeling_priors_path


DEBUG_PREFIXES = [
    "Apex", "Nova", "Zenith", "Eclipse", "Prism", "Titan", "Nebula", "Solaris", "Vortex", "Atlas",
    "Phoenix", "Onyx", "Stellar", "Crimson", "Azure", "Polar", "Emerald", "Silver", "Golden", "Iron",
    "Horizon", "Summit", "Pinnacle", "Vertex", "Nexus", "Meridian", "Beacon", "Voyager", "Frontier",
    "Mirage", "Oasis", "Lotus", "Jade", "Dragon", "Tiger", "Falcon", "Raven", "Wolf", "Lion",
    "Thunder", "Lightning", "Blaze", "Storm", "Aurora", "Quantum", "Vector", "Matrix", "Cipher", "Neon",
]
DEBUG_SUFFIXES = [
    "Studios", "Pictures", "Films", "Entertainment", "Media", "Productions", "Cinema",
    "Motion Pictures", "Releasing", "International", "Worldwide", "Creative", "Arts",
    "Works", "Group", "Company", "Collective",
]
DEBUG_TEMPLATES = [
    "{prefix} {suffix}",
    "{prefix} {abstract} {suffix}",
    "{geo} {prefix} {suffix}",
    "{motion} {abstract} {suffix}",
]
DEBUG_TIER_WEIGHTS = {"Global": 0.05, "Major": 0.15, "Mid-Budget": 0.35, "Indie": 0.30, "Micro": 0.15}

_COMPANY_TEMPLATE_ALIAS_MAP = {
    "local_word": "geo",
    "local_words": "geo",
    "geographic_word": "geo",
    "geographic_words": "geo",
    "abstract_word": "abstract",
    "abstract_words": "abstract",
    "motion_word": "motion",
    "motion_words": "motion",
    "mythic_word": "mythic",
    "mythic_words": "mythic",
    "material_word": "material",
    "material_words": "material",
}


def _normalize_weights(raw: object, fallback: dict[str, float]) -> dict[str, float]:
    if not isinstance(raw, dict):
        return dict(fallback)
    out: dict[str, float] = {}
    for key, value in raw.items():
        try:
            score = float(value)
        except Exception:
            continue
        if score > 0:
            out[str(key)] = score
    return out or dict(fallback)


def _clean_text_list(raw: object) -> list[str]:
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for value in raw:
        item = str(value or "").strip()
        key = item.casefold()
        if not item or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _canonicalize_company_template(template: str) -> str:
    raw = str(template or "").strip()
    if not raw:
        return ""

    def _replace(match: re.Match[str]) -> str:
        key = str(match.group(1) or "").strip()
        canonical = _COMPANY_TEMPLATE_ALIAS_MAP.get(key, key)
        return "{" + canonical + "}"

    return re.sub(r"\{([^{}]+)\}", _replace, raw)


def _lexicon(base_dir: Path, mode: str) -> dict:
    artifact_path = company_lexicon_path(base_dir)
    payload = load_company_lexicon(base_dir, mode=mode)
    if not isinstance(payload, dict):
        audit_fallback_hit("company_generation", "debug_company_lexicon_missing", detail="using built-in company lexicon", mode=mode)
        return {
            "prefixes": DEBUG_PREFIXES,
            "suffixes": DEBUG_SUFFIXES,
            "abstract_nouns": ["Vision", "Signal", "Archive", "Meridian", "Compass", "Canvas", "Legacy", "Bridge"],
            "material_words": ["Silver", "Golden", "Ivory", "Cobalt", "Amber"],
            "geographic_words": ["Harbor", "Coast", "Skyline", "Frontier", "Valley", "Summit"],
            "motion_words": ["Voyage", "Drift", "Ascent", "Pulse", "Orbit", "Transit"],
            "mythic_words": ["Phoenix", "Atlas", "Oracle", "Titan", "Mirage", "Avalon"],
            "templates": DEBUG_TEMPLATES,
            "tier_styles": {},
            "country_style_bias": {},
        }
    required_lists = {
        "prefixes": _clean_text_list(require_payload_value(payload, "prefixes", artifact_label="company_lexicon.json", artifact_path=artifact_path, mode=mode, validator=lambda value: isinstance(value, list) and len(value) > 0) or []),
        "suffixes": _clean_text_list(require_payload_value(payload, "suffixes", artifact_label="company_lexicon.json", artifact_path=artifact_path, mode=mode, validator=lambda value: isinstance(value, list) and len(value) > 0) or []),
        "abstract_nouns": _clean_text_list(require_payload_value(payload, "abstract_nouns", artifact_label="company_lexicon.json", artifact_path=artifact_path, mode=mode, validator=lambda value: isinstance(value, list) and len(value) > 0) or []),
        "material_words": _clean_text_list(require_payload_value(payload, "material_words", artifact_label="company_lexicon.json", artifact_path=artifact_path, mode=mode, validator=lambda value: isinstance(value, list) and len(value) > 0) or []),
        "geographic_words": _clean_text_list(require_payload_value(payload, "geographic_words", artifact_label="company_lexicon.json", artifact_path=artifact_path, mode=mode, validator=lambda value: isinstance(value, list) and len(value) > 0) or []),
        "motion_words": _clean_text_list(require_payload_value(payload, "motion_words", artifact_label="company_lexicon.json", artifact_path=artifact_path, mode=mode, validator=lambda value: isinstance(value, list) and len(value) > 0) or []),
        "mythic_words": _clean_text_list(require_payload_value(payload, "mythic_words", artifact_label="company_lexicon.json", artifact_path=artifact_path, mode=mode, validator=lambda value: isinstance(value, list) and len(value) > 0) or []),
        "templates": [
            tpl for tpl in (
                _canonicalize_company_template(item)
                for item in _clean_text_list(
                    require_payload_value(
                        payload,
                        "templates",
                        artifact_label="company_lexicon.json",
                        artifact_path=artifact_path,
                        mode=mode,
                        validator=lambda value: isinstance(value, list) and len(value) > 0,
                    ) or []
                )
            )
            if tpl
        ],
    }
    audit_artifact_usage("company_lexicon.json", artifact_path, sections=list(required_lists.keys()))
    return {
        **required_lists,
        "tier_styles": payload.get("tier_styles", {}) if isinstance(payload.get("tier_styles"), dict) else {},
        "country_style_bias": payload.get("country_style_bias", {}) if isinstance(payload.get("country_style_bias"), dict) else {},
    }


def _tier_weights(base_dir: Path, mode: str) -> dict[str, float]:
    artifact_path = modeling_priors_path(base_dir)
    priors = load_modeling_priors_artifact(base_dir, mode=mode)
    company_gen = priors.get("company_generation", {}) if isinstance(priors, dict) else {}
    raw = require_payload_value(
        company_gen,
        "tier_weights",
        artifact_label="modeling_priors.json",
        artifact_path=artifact_path,
        mode=mode,
        validator=lambda value: isinstance(value, dict) and len(value) > 0,
        detail="company_generation.tier_weights is required for research-mode company synthesis",
    )
    if raw is None:
        audit_fallback_hit("company_generation", "debug_company_tier_weights", detail="using built-in company tier weights", mode=mode)
        return dict(DEBUG_TIER_WEIGHTS)
    audit_artifact_usage("modeling_priors.json", artifact_path, sections=["company_generation.tier_weights"])
    return _normalize_weights(raw, DEBUG_TIER_WEIGHTS)


def _render_company_name(rng: random.Random, lexicon: dict) -> str:
    template = _canonicalize_company_template(rng.choice(list(lexicon["templates"])))
    pools = {
        "prefix": list(lexicon["prefixes"]),
        "suffix": list(lexicon["suffixes"]),
        "abstract": list(lexicon["abstract_nouns"]),
        "geo": list(lexicon["geographic_words"]),
        "motion": list(lexicon["motion_words"]),
        "mythic": list(lexicon["mythic_words"]),
        "material": list(lexicon["material_words"]),
    }
    pools["local_word"] = list(pools["geo"])
    substitutions = {key: rng.choice(values) for key, values in pools.items()}
    name = template.format(**substitutions)
    return " ".join(str(name).split()).strip()


def generate_companies(target: int, *, seed: int, base_dir: Path, mode: str) -> list[dict]:
    rng = random.Random(seed)
    companies: list[dict] = []
    used_names: set[str] = set()
    lexicon = _lexicon(base_dir, mode)
    tier_weights = _tier_weights(base_dir, mode)

    all_countries = [country for country in COUNTRIES if str(country).strip()]
    country_weights = [float(COMPANY_COUNTRY_WEIGHTS.get(country, 0.0005)) for country in all_countries]
    tier_list = list(tier_weights.keys())
    tier_w = list(tier_weights.values())

    t0 = time.time()
    for i in range(target):
        name = None
        for _ in range(320):
            candidate = _render_company_name(rng, lexicon)
            key = candidate.casefold()
            if candidate and key not in used_names:
                used_names.add(key)
                name = candidate
                break
        if name is None:
            raise RuntimeError(f"Unable to synthesize unique company name without numeric dedupe at row {i + 1}")

        tier = rng.choices(tier_list, weights=tier_w, k=1)[0]
        country = rng.choices(all_countries, weights=country_weights, k=1)[0]
        n_genres = rng.choices([1, 2, 3], weights=[0.3, 0.5, 0.2], k=1)[0]
        specialty = rng.sample(GENRES, min(n_genres, len(GENRES)))

        companies.append(
            {
                "company_id": i + 1,
                "name": name,
                "country": country,
                "description": f"A {tier.lower()} production company specializing in {', '.join(g.lower() for g in specialty)} films.",
                "specialty_genres": specialty,
                "tier": tier,
                "preferred_actor_styles": rng.sample(STYLE_TAGS, min(rng.randint(1, 3), len(STYLE_TAGS))),
                "preferred_director_styles": rng.sample(DIRECTOR_STYLES, min(rng.randint(1, 3), len(DIRECTOR_STYLES))),
            }
        )

    elapsed = time.time() - t0
    print(f"  Generated {len(companies)} companies in {elapsed:.1f}s")
    return companies


def main() -> None:
    parser = argparse.ArgumentParser(description="Company generator for Mirage.")
    parser.add_argument("--base-dir", default=str(BASE_DIR))
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("research", "debug"), default=current_mode())
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    out_path = Path(args.out).resolve() if args.out else (base_dir / "entities" / "companies.json")

    print("=" * 60)
    print("  COMPANY GENERATOR")
    print("=" * 60)
    print(f"  Target:  {args.target}")
    print(f"  Mode:    {args.mode}")
    print("=" * 60)

    companies = generate_companies(int(args.target), seed=int(args.seed), base_dir=base_dir, mode=str(args.mode))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(companies, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
