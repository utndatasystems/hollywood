#!/usr/bin/env python3
"""Person generator for Mirage.

Research mode consumes `identity_bank.json` plus `modeling_priors.json`.
Debug mode falls back to the older procedural name-bank path.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).parent
ENTITY_DIR = BASE_DIR / "entities"

sys.path.insert(0, str(BASE_DIR))

from bootstrap_artifacts import (
    audit_artifact_usage,
    audit_fallback_hit,
    current_mode,
    load_identity_bank,
    load_modeling_priors_artifact,
    require_payload_value,
)
from contracts import MARKETS, ROLE_TYPES
from name_banks import NAME_BANKS
from pipeline_runtime import year_bounds_from_env
from policy_runtime import identity_bank_path, modeling_priors_path


DEBUG_ROLE_DISTRIBUTION = {
    "actor": 0.60,
    "director": 0.06,
    "actor,director": 0.04,
    "producer": 0.06,
    "writer": 0.05,
    "writer,director": 0.02,
    "cinematographer": 0.05,
    "editor": 0.05,
    "composer": 0.04,
    "production_designer": 0.03,
}
DEBUG_STAGE_WEIGHTS = {"rising": 0.30, "prime": 0.35, "veteran": 0.20, "legend": 0.10, "retired": 0.05}
DEBUG_GENDER_WEIGHTS = {"M": 0.45, "F": 0.45, "NB": 0.10}

REGION_MARKETS = {
    "north_america": ["North America", "Global"],
    "europe": ["Europe", "Global"],
    "east_asia": ["Asia", "Global"],
    "south_asia": ["Asia", "Regional"],
    "southeast_asia": ["Asia", "Regional"],
    "middle_east": ["Middle East", "Regional"],
    "africa": ["Africa", "Regional"],
    "latin_america": ["Latin America", "South America", "Regional"],
    "oceania": ["Oceania", "Regional"],
}

SUFFIXES = ["Jr.", "Sr.", "II", "III", "IV"]
DEFAULT_SUFFIX_PROB = 0.03
DEFAULT_DOUBLE_SURNAME_PROB = 0.08
DEFAULT_MIDDLE_NAME_PROB = 0.15
DEFAULT_MIDDLE_INITIAL_PROB = 0.10

ROLE_ALIASES = {
    "actor": "actor",
    "performer": "actor",
    "voice_actor": "actor",
    "director": "director",
    "writer": "writer",
    "screenwriter": "writer",
    "producer": "producer",
    "cinematographer": "cinematographer",
    "director_of_photography": "cinematographer",
    "editor": "editor",
    "composer": "composer",
    "production_designer": "production_designer",
    "costume_designer": "costume_designer",
    "makeup_artist": "makeup_artist",
    "visual_effects": "vfx_supervisor",
    "vfx": "vfx_supervisor",
    "vfx_supervisor": "vfx_supervisor",
    "sound_engineer": "sound_designer",
    "sound_designer": "sound_designer",
    "casting_director": "casting_director",
    "stunt_coordinator": "stunt_coordinator",
}


def _normalize_role_token(raw: object) -> str | None:
    key = str(raw or "").strip().lower()
    if not key:
        return None
    key = re.sub(r"[\s-]+", "_", key)
    key = ROLE_ALIASES.get(key, key)
    return key if key in ROLE_TYPES else None


def _roles_from_combo(raw: object) -> list[str]:
    roles: list[str] = []
    for part in re.split(r"[,/|+&]+", str(raw or "")):
        role = _normalize_role_token(part)
        if role and role not in roles:
            roles.append(role)
    return roles or ["actor"]


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


def _weighted_choice(rng: random.Random, items_weights: dict[str, float]) -> str:
    items = list(items_weights.keys())
    weights = list(items_weights.values())
    return rng.choices(items, weights=weights, k=1)[0]


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


def _debug_families() -> list[dict]:
    out: list[dict] = []
    for row in NAME_BANKS:
        out.append(
            {
                "nationality": row.get("nationality"),
                "region": row.get("region", "europe"),
                "weight": float(row.get("weight", 1.0)),
                "first_m": list(row.get("first_m", [])),
                "first_f": list(row.get("first_f", [])),
                "first_nb": list(row.get("first_nb", [])),
                "surnames": list(row.get("surnames", [])),
                "connectors": [],
                "double_surname_probability": DEFAULT_DOUBLE_SURNAME_PROB,
                "middle_name_probability": DEFAULT_MIDDLE_NAME_PROB,
                "middle_initial_probability": DEFAULT_MIDDLE_INITIAL_PROB,
                "suffix_probability": DEFAULT_SUFFIX_PROB,
                "market_bias": REGION_MARKETS.get(str(row.get("region", "europe")), ["Regional"]),
            }
        )
    return out


def _identity_artifact_path(base_dir: Path) -> Path:
    return identity_bank_path(base_dir)


def _priors_artifact_path(base_dir: Path) -> Path:
    return modeling_priors_path(base_dir)


def _artifact_families(base_dir: Path, mode: str) -> list[dict]:
    artifact_path = _identity_artifact_path(base_dir)
    payload = load_identity_bank(base_dir, mode=mode)
    raw_families = require_payload_value(
        payload,
        "families",
        artifact_label="identity_bank.json",
        artifact_path=artifact_path,
        mode=mode,
        validator=lambda value: isinstance(value, list) and len(value) > 0,
        detail="identity_bank.json must provide non-empty families",
    )
    if raw_families is None:
        audit_fallback_hit("person_generation", "debug_identity_bank_families", detail="using procedural name-bank families", mode=mode)
        return _debug_families()
    audit_artifact_usage("identity_bank.json", artifact_path, sections=["families"])

    prepared: list[dict] = []
    for idx, raw in enumerate(raw_families, start=1):
        if not isinstance(raw, dict):
            if mode == "research":
                audit_fallback_hit("identity_bank.json", "invalid_family_row", detail=f"family[{idx}] must be an object", mode=mode)
            continue
        family = {
            "nationality": str(raw.get("nationality", "") or "").strip(),
            "region": str(raw.get("region", "") or "").strip() or "Europe",
            "weight": float(raw.get("weight", 1.0) or 1.0),
            "first_m": _clean_text_list(raw.get("first_m")),
            "first_f": _clean_text_list(raw.get("first_f")),
            "first_nb": _clean_text_list(raw.get("first_nb")),
            "surnames": _clean_text_list(raw.get("surnames")),
            "connectors": _clean_text_list(raw.get("connectors")),
            "market_bias": [market for market in _clean_text_list(raw.get("market_bias")) if market in MARKETS],
        }
        for key, default in (
            ("double_surname_probability", DEFAULT_DOUBLE_SURNAME_PROB),
            ("middle_name_probability", DEFAULT_MIDDLE_NAME_PROB),
            ("middle_initial_probability", DEFAULT_MIDDLE_INITIAL_PROB),
            ("suffix_probability", DEFAULT_SUFFIX_PROB),
        ):
            try:
                family[key] = float(raw.get(key))
            except Exception:
                if mode == "research":
                    audit_fallback_hit(
                        "identity_bank.json",
                        "family_probability_missing",
                        detail=f"{family['nationality'] or f'family[{idx}]'} missing {key}",
                        mode=mode,
                    )
                family[key] = float(default)
        if not family["nationality"] or not family["surnames"] or not (family["first_m"] or family["first_f"] or family["first_nb"]):
            if mode == "research":
                audit_fallback_hit(
                    "identity_bank.json",
                    "family_name_inventory_missing",
                    detail=f"{family['nationality'] or f'family[{idx}]'} missing usable first/surname pools",
                    mode=mode,
                )
            continue
        if not family["market_bias"]:
            if mode == "research":
                audit_fallback_hit(
                    "identity_bank.json",
                    "family_market_bias_missing",
                    detail=f"{family['nationality'] or f'family[{idx}]'} missing market_bias",
                    mode=mode,
                )
            family["market_bias"] = REGION_MARKETS.get(family["region"].casefold(), ["Regional"])
        prepared.append(family)

    if prepared:
        return prepared
    audit_fallback_hit("person_generation", "debug_identity_bank_empty", detail="identity bank families normalized to empty; using debug fallback", mode=mode)
    return _debug_families()


def _map_gender_weights_from_priors(raw: object) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    key_map = {
        "m": "M",
        "male": "M",
        "f": "F",
        "female": "F",
        "nb": "NB",
        "non_binary": "NB",
        "nonbinary": "NB",
    }
    out: dict[str, float] = {}
    for key, value in raw.items():
        try:
            score = float(value)
        except Exception:
            continue
        mapped = key_map.get(str(key).strip().lower())
        if mapped and score > 0:
            out[mapped] = out.get(mapped, 0.0) + score
    return out


def _map_stage_weights_from_priors(raw: object) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    key_map = {
        "debut": "rising",
        "emerging": "rising",
        "rising": "rising",
        "established": "prime",
        "prime": "prime",
        "veteran": "veteran",
        "legend": "legend",
        "retired": "retired",
    }
    out: dict[str, float] = {}
    for key, value in raw.items():
        try:
            score = float(value)
        except Exception:
            continue
        mapped = key_map.get(str(key).strip().lower())
        if mapped and score > 0:
            out[mapped] = out.get(mapped, 0.0) + score
    return out


def _identity_defaults(base_dir: Path, mode: str) -> dict[str, Any]:
    artifact_path = _identity_artifact_path(base_dir)
    payload = load_identity_bank(base_dir, mode=mode)
    defaults = require_payload_value(
        payload,
        "defaults",
        artifact_label="identity_bank.json",
        artifact_path=artifact_path,
        mode=mode,
        validator=lambda value: isinstance(value, dict) and len(value) > 0,
        detail="identity_bank.json must provide defaults for research-mode person synthesis",
    )
    if defaults is None:
        audit_fallback_hit("person_generation", "debug_identity_defaults", detail="using debug identity defaults", mode=mode)
        return {
            "role_distribution": dict(DEBUG_ROLE_DISTRIBUTION),
            "stage_weights": dict(DEBUG_STAGE_WEIGHTS),
            "gender_weights": dict(DEBUG_GENDER_WEIGHTS),
        }
    audit_artifact_usage("identity_bank.json", artifact_path, sections=["defaults"])
    return defaults


def _generation_defaults(base_dir: Path, mode: str) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    identity_defaults = _identity_defaults(base_dir, mode)
    priors_path = _priors_artifact_path(base_dir)
    priors = load_modeling_priors_artifact(base_dir, mode=mode)
    person_generation = {}
    if isinstance(priors, dict):
        raw_section = priors.get("person_generation")
        if isinstance(raw_section, dict) and raw_section:
            person_generation = raw_section
            audit_artifact_usage("modeling_priors.json", priors_path, sections=["person_generation"])
    role_distribution = _normalize_weights(
        person_generation.get("role_distribution", identity_defaults.get("role_distribution")),
        DEBUG_ROLE_DISTRIBUTION,
    )
    stage_source = person_generation.get("stage_weights")
    mapped_stage_weights = _normalize_weights(
        stage_source if isinstance(stage_source, dict) else _map_stage_weights_from_priors(person_generation.get("career_stage_weights")),
        _normalize_weights(identity_defaults.get("stage_weights"), DEBUG_STAGE_WEIGHTS),
    )
    gender_source = person_generation.get("gender_weights")
    mapped_gender_weights = _normalize_weights(
        gender_source if isinstance(gender_source, dict) else _map_gender_weights_from_priors(person_generation.get("gender_ratio")),
        _normalize_weights(identity_defaults.get("gender_weights"), DEBUG_GENDER_WEIGHTS),
    )
    if mode == "research":
        if not role_distribution:
            audit_fallback_hit("person_generation", "role_distribution_missing", detail="role distribution missing from identity_bank defaults / person_generation priors", mode=mode)
        if not mapped_stage_weights:
            audit_fallback_hit("person_generation", "stage_weights_missing", detail="stage weights missing from identity_bank defaults / person_generation priors", mode=mode)
        if not mapped_gender_weights:
            audit_fallback_hit("person_generation", "gender_weights_missing", detail="gender weights missing from identity_bank defaults / person_generation priors", mode=mode)
    return role_distribution, mapped_stage_weights, mapped_gender_weights


def generate_name(rng: random.Random, family: dict, gender: str, used_names: set[str]) -> str | None:
    first_m = list(family["first_m"])
    first_f = list(family["first_f"])
    first_nb = list(family["first_nb"])
    surnames = list(family["surnames"])
    connectors = list(family["connectors"])

    if gender == "M":
        first_pool = first_m or first_f or first_nb
    elif gender == "F":
        first_pool = first_f or first_m or first_nb
    else:
        first_pool = first_nb or first_f or first_m
    if not first_pool or not surnames:
        return None

    double_prob = float(family["double_surname_probability"])
    middle_name_prob = float(family["middle_name_probability"])
    middle_initial_prob = float(family["middle_initial_probability"])
    suffix_prob = float(family["suffix_probability"])

    for _ in range(240):
        first = rng.choice(first_pool)
        last = rng.choice(surnames)

        if rng.random() < double_prob and len(surnames) > 1:
            second_last = rng.choice(surnames)
            while second_last == last and len(surnames) > 1:
                second_last = rng.choice(surnames)
            if connectors and rng.random() < 0.35:
                last = f"{last} {rng.choice(connectors)} {second_last}"
            else:
                separator = " " if rng.random() < 0.6 else "-"
                last = f"{last}{separator}{second_last}"

        middle = ""
        r = rng.random()
        if r < middle_name_prob:
            middle_pick = rng.choice(first_pool)
            if middle_pick != first:
                middle = f" {middle_pick}"
        elif r < middle_name_prob + middle_initial_prob:
            middle = f" {rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}."

        suffix = f" {rng.choice(SUFFIXES)}" if rng.random() < suffix_prob else ""
        full_name = f"{first}{middle} {last}{suffix}".strip()
        key = full_name.casefold()
        if key not in used_names:
            used_names.add(key)
            return full_name
    return None


def assign_career_timelines(persons: list[dict], *, seed: int) -> None:
    rng = random.Random(int(seed) + 100_003)
    yr_lo, yr_hi = year_bounds_from_env(1950, 2025)
    for person in persons:
        stage = str(person.get("career_stage", "prime") or "prime").strip().lower()
        if stage == "legend":
            debut = rng.randint(yr_lo - 35, yr_lo - 15)
            peak_start = debut + rng.randint(5, 10)
            peak_end = peak_start + rng.randint(5, 8)
            retire = max(yr_hi + rng.randint(0, 8), peak_end + 2)
            yearly_max = rng.randint(5, 9)
        elif stage == "veteran":
            debut = rng.randint(yr_lo - 25, yr_lo - 8)
            peak_start = debut + rng.randint(5, 10)
            peak_end = peak_start + rng.randint(4, 7)
            retire = max(yr_hi + rng.randint(0, 10), peak_end + 3)
            yearly_max = rng.randint(3, 6)
        elif stage == "prime":
            debut = rng.randint(yr_lo - 15, yr_lo - 2)
            peak_start = debut + rng.randint(3, 7)
            peak_end = peak_start + rng.randint(3, 6)
            retire = max(yr_hi + rng.randint(5, 15), peak_end + 5)
            yearly_max = rng.randint(4, 7)
        elif stage == "rising":
            debut = rng.randint(yr_lo - 5, yr_hi - 1)
            peak_start = debut + rng.randint(2, 5)
            peak_end = peak_start + rng.randint(3, 6)
            retire = 2100
            yearly_max = rng.randint(2, 5)
        else:
            # Retired people should mostly peak before the benchmark window,
            # but their timeline still has to be chronologically valid.
            if rng.random() < 0.7:
                debut = rng.randint(yr_lo - 50, yr_lo - 22)
                peak_start = debut + rng.randint(6, 12)
                peak_end = peak_start + rng.randint(3, 6)
                retire_hint = rng.randint(yr_lo - 8, yr_lo + 2)
                retire = max(peak_end + rng.randint(0, 3), retire_hint)
            else:
                debut = rng.randint(yr_lo - 45, yr_lo - 18)
                peak_start = debut + rng.randint(5, 10)
                peak_end = peak_start + rng.randint(3, 6)
                retire = peak_end + rng.randint(0, 5)
            yearly_max = rng.randint(1, 2)
        person["debut_year"] = int(debut)
        person["peak_start"] = int(peak_start)
        person["peak_end"] = int(peak_end)
        person["retirement_year"] = int(min(retire, 2100))
        person["yearly_max"] = int(yearly_max)


def generate_persons(target: int, *, seed: int, base_dir: Path, mode: str) -> list[dict]:
    rng = random.Random(seed)
    used_names: set[str] = set()
    persons: list[dict] = []

    families = _artifact_families(base_dir, mode)
    if not families:
        raise RuntimeError("No identity families available for person generation")

    nat_weights = {
        str(row.get("nationality")): float(row.get("weight", 1.0) or 1.0)
        for row in families
        if str(row.get("nationality", "")).strip()
    }
    nat_index = {str(row.get("nationality")): row for row in families if str(row.get("nationality", "")).strip()}
    nat_list = list(nat_weights.keys())
    nat_w = list(nat_weights.values())
    role_distribution, stage_weights, gender_weights = _generation_defaults(base_dir, mode)

    t0 = time.time()
    for i in range(target):
        nationality = rng.choices(nat_list, weights=nat_w, k=1)[0]
        family = nat_index[nationality]
        gender = _weighted_choice(rng, gender_weights)
        role_combo = _weighted_choice(rng, role_distribution)
        roles = _roles_from_combo(role_combo)
        career_stage = _weighted_choice(rng, stage_weights)

        name = generate_name(rng, family, gender, used_names)
        if name is None:
            for _ in range(16):
                nationality = rng.choices(nat_list, weights=nat_w, k=1)[0]
                family = nat_index[nationality]
                name = generate_name(rng, family, gender, used_names)
                if name is not None:
                    break
        if name is None:
            raise RuntimeError(f"Unable to generate unique person name after exhausting identity bank at row {i + 1}")

        market_fit = [str(x).strip() for x in family.get("market_bias", []) if str(x).strip()]
        if not market_fit:
            market_fit = REGION_MARKETS.get(str(family.get("region", "europe")).casefold(), ["Regional"])
        if rng.random() < 0.15 and "Global" not in market_fit:
            market_fit = list(market_fit) + ["Global"]
        market_fit = [market for market in market_fit if market in MARKETS]

        persons.append(
            {
                "person_id": i + 1,
                "name": name,
                "nationality": nationality,
                "gender": gender,
                "bio": "",
                "style_tags": [],
                "genre_affinity": [],
                "roles": roles,
                "career_stage": career_stage,
                "market_fit": list(dict.fromkeys(market_fit)),
            }
        )

        if (i + 1) % 50000 == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1:>8,} / {target:,} ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"\n  Generated {len(persons):,} unique persons in {elapsed:.1f}s")
    assign_career_timelines(persons, seed=seed)
    return persons


def main() -> None:
    parser = argparse.ArgumentParser(description="Person generator for Mirage.")
    parser.add_argument("--base-dir", default=str(BASE_DIR))
    parser.add_argument("--target", type=int, default=400_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("research", "debug"), default=current_mode())
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    out_path = Path(args.out).resolve() if args.out else (base_dir / "entities" / "persons.json")

    print("=" * 60)
    print("  PERSON GENERATOR")
    print("=" * 60)
    print(f"  Target:  {args.target:,}")
    print(f"  Seed:    {args.seed}")
    print(f"  Mode:    {args.mode}")
    print("=" * 60)

    persons = generate_persons(int(args.target), seed=int(args.seed), base_dir=base_dir, mode=str(args.mode))

    genders = Counter(p["gender"] for p in persons)
    stages = Counter(p["career_stage"] for p in persons)
    roles = Counter()
    for person in persons:
        for role in person["roles"]:
            roles[role] += 1
    nats = Counter(p["nationality"] for p in persons)

    print(f"\n  Gender:  {dict(genders.most_common())}")
    print(f"  Stages:  {dict(stages.most_common())}")
    print(f"  Roles:   {dict(roles.most_common(10))}")
    print(f"  Top nats: {dict(nats.most_common(10))}")
    print(f"  Unique names: {len(set(p['name'] for p in persons)):,}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(persons, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
