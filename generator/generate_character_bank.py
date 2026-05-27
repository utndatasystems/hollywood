#!/usr/bin/env python3
"""Character bank generator for Mirage.

Research mode consumes `character_identity_bank.json` and `identity_bank.json`.
Debug mode keeps the older procedural fallback with fixed moniker pools.
"""
from __future__ import annotations

import argparse
import csv
import random
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
    load_character_identity_bank,
    load_identity_bank,
    load_modeling_priors_artifact,
    require_payload_value,
)
from name_banks import NAME_BANKS
from policy_runtime import character_identity_bank_path, identity_bank_path, modeling_priors_path
from text_polish import sanitize_character_name, strip_leading_article


DEBUG_ARCHETYPES = {
    "Lead Hero": 0.12,
    "Lead Villain": 0.10,
    "Love Interest": 0.06,
    "Sidekick": 0.06,
    "Mentor": 0.07,
    "Supporting": 0.18,
    "Extra": 0.13,
    "Authority Figure": 0.07,
    "Henchman": 0.06,
    "Comic Relief": 0.05,
    "Victim": 0.05,
    "Mysterious Stranger": 0.05,
}
DEBUG_PAYLOAD = {
    "archetype_weights": DEBUG_ARCHETYPES,
    "title_prefixes_m": ["Dr.", "Prof.", "Capt.", "Det.", "Agent", "Judge", "Chief", "King", "Lord"],
    "title_prefixes_f": ["Dr.", "Prof.", "Capt.", "Det.", "Agent", "Judge", "Chief", "Queen", "Lady"],
    "title_prefixes_nb": ["Dr.", "Prof.", "Capt.", "Det.", "Agent", "Judge", "Chief"],
    "quote_nicknames": ["The Ghost", "The Fox", "The Blade", "The Raven", "The Oracle", "The Sentinel"],
    "solo_monikers": ["The Collector", "The Architect", "The Warden", "The Harbinger", "The Regent", "The Oracle"],
    "codename_adjectives": ["Silent", "Iron", "Crimson", "Midnight", "Broken", "Golden", "Neon"],
    "codename_nouns": ["Viper", "Falcon", "Cipher", "Ghost", "Torch", "Wraith", "Atlas"],
    "mythic_epithets": ["Stormborn", "Wayfarer", "Ashen", "Radiant", "Veilwalker"],
    "role_epithets": ["Fixer", "Broker", "Handler", "Watcher", "Operator", "Judge"],
    "alias_templates": [
        "{title} {surname}",
        "{first} '{nickname}' {surname}",
        "The {moniker}",
        "{codename_adj} {codename_noun}",
        "{title} {first} {surname}",
    ],
    "nonliteral_share_by_archetype": {
        "Lead Villain": 0.28,
        "Mysterious Stranger": 0.32,
        "Mentor": 0.18,
        "Comic Relief": 0.12,
    },
    "human_name_mix_by_archetype": {
        "Lead Hero": 0.90,
        "Lead Villain": 0.65,
        "Love Interest": 0.96,
        "Supporting": 0.88,
        "Mysterious Stranger": 0.55,
    },
}

NAME_SUFFIXES = ["Jr.", "Sr.", "II", "III", "IV"]


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


def _weights(raw: object) -> dict[str, float]:
    if not isinstance(raw, dict):
        return dict(DEBUG_ARCHETYPES)
    out: dict[str, float] = {}
    for key, value in raw.items():
        try:
            score = float(value)
        except Exception:
            continue
        if score > 0:
            out[str(key)] = score
    return out or dict(DEBUG_ARCHETYPES)


def _character_payload(base_dir: Path, mode: str) -> dict:
    artifact_path = character_identity_bank_path(base_dir)
    payload = load_character_identity_bank(base_dir, mode=mode)
    if not isinstance(payload, dict):
        audit_fallback_hit("character_generation", "debug_character_identity_bank_missing", detail="using built-in character payload", mode=mode)
        return dict(DEBUG_PAYLOAD)
    required_list_keys = (
        "title_prefixes_m",
        "title_prefixes_f",
        "title_prefixes_nb",
        "quote_nicknames",
        "solo_monikers",
        "codename_adjectives",
        "codename_nouns",
        "mythic_epithets",
        "role_epithets",
        "alias_templates",
    )
    normalized: dict[str, Any] = {}
    normalized["archetype_weights"] = require_payload_value(
        payload,
        "archetype_weights",
        artifact_label="character_identity_bank.json",
        artifact_path=artifact_path,
        mode=mode,
        validator=lambda value: isinstance(value, dict) and len(value) > 0,
    ) or dict(DEBUG_ARCHETYPES)
    for key in required_list_keys:
        values = _clean_text_list(
            require_payload_value(
                payload,
                key,
                artifact_label="character_identity_bank.json",
                artifact_path=artifact_path,
                mode=mode,
                validator=lambda value: isinstance(value, list) and len(value) > 0,
            )
            or []
        )
        normalized[key] = values or list(DEBUG_PAYLOAD[key])
    for key in ("nonliteral_share_by_archetype", "human_name_mix_by_archetype"):
        raw = require_payload_value(
            payload,
            key,
            artifact_label="character_identity_bank.json",
            artifact_path=artifact_path,
            mode=mode,
            validator=lambda value: isinstance(value, dict) and len(value) > 0,
            detail=f"character identity bank must provide {key}",
        )
        normalized[key] = dict(raw or {})
    audit_artifact_usage(
        "character_identity_bank.json",
        artifact_path,
        sections=[
            "archetype_weights",
            *required_list_keys,
            "nonliteral_share_by_archetype",
            "human_name_mix_by_archetype",
        ],
    )
    return normalized


def _identity_families(base_dir: Path, mode: str) -> list[dict]:
    artifact_path = identity_bank_path(base_dir)
    payload = load_identity_bank(base_dir, mode=mode)
    raw_families = require_payload_value(
        payload,
        "families",
        artifact_label="identity_bank.json",
        artifact_path=artifact_path,
        mode=mode,
        validator=lambda value: isinstance(value, list) and len(value) > 0,
    )
    if not isinstance(raw_families, list):
        audit_fallback_hit("character_generation", "debug_identity_bank_families", detail="using NAME_BANKS character family fallback", mode=mode)
        return [
            {
                "first_m": list(row.get("first_m", [])),
                "first_f": list(row.get("first_f", [])),
                "first_nb": list(row.get("first_nb", [])),
                "surnames": list(row.get("surnames", [])),
            }
            for row in NAME_BANKS
        ]
    prepared = [
        {
            "first_m": _clean_text_list(row.get("first_m")),
            "first_f": _clean_text_list(row.get("first_f")),
            "first_nb": _clean_text_list(row.get("first_nb")),
            "surnames": _clean_text_list(row.get("surnames")),
        }
        for row in raw_families
        if isinstance(row, dict)
    ]
    prepared = [row for row in prepared if row["surnames"] and (row["first_m"] or row["first_f"] or row["first_nb"])]
    if prepared:
        audit_artifact_usage("identity_bank.json", artifact_path, sections=["families"])
        return prepared
    audit_fallback_hit("character_generation", "debug_identity_bank_empty", detail="identity bank lacked usable human families for character names", mode=mode)
    return [
        {
            "first_m": list(row.get("first_m", [])),
            "first_f": list(row.get("first_f", [])),
            "first_nb": list(row.get("first_nb", [])),
            "surnames": list(row.get("surnames", [])),
        }
        for row in NAME_BANKS
    ]


def _human_name(rng: random.Random, family: dict, gender: str, *, distinctive: bool = False) -> str | None:
    first_m = [str(x).strip() for x in family.get("first_m", []) if str(x).strip()]
    first_f = [str(x).strip() for x in family.get("first_f", []) if str(x).strip()]
    first_nb = [str(x).strip() for x in family.get("first_nb", []) if str(x).strip()]
    surnames = [str(x).strip() for x in family.get("surnames", []) if str(x).strip()]
    if gender == "M":
        first_pool = first_m or first_f or first_nb
    elif gender == "F":
        first_pool = first_f or first_m or first_nb
    else:
        first_pool = first_nb or first_f or first_m
    if not first_pool or not surnames:
        return None
    first = rng.choice(first_pool)
    surname = rng.choice(surnames)
    if (distinctive or rng.random() < 0.12) and len(surnames) > 1:
        second = rng.choice(surnames)
        if second != surname:
            connector = rng.choice([str(x).strip() for x in family.get("connectors", []) if str(x).strip()] or ["-"])
            joiner = f" {connector} " if connector != "-" and rng.random() < 0.35 else "-"
            surname = f"{surname}{joiner}{second}"
    middle = ""
    if distinctive or rng.random() < 0.38:
        if rng.random() < 0.35 and len(first_pool) > 1:
            picked = rng.choice(first_pool)
            if picked != first:
                middle = f" {picked}"
        if not middle:
            middle = f" {rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}."
    suffix = ""
    if distinctive or rng.random() < 0.06:
        suffix = f" {rng.choice(NAME_SUFFIXES)}"
    return f"{first}{middle} {surname}{suffix}".strip()


def _render_alias(rng: random.Random, payload: dict, family: dict, gender: str) -> str:
    titles = payload["title_prefixes_m" if gender == "M" else "title_prefixes_f" if gender == "F" else "title_prefixes_nb"]
    first = _human_name(rng, family, gender)
    if not first:
        return ""
    if " " in first:
        first_name, surname = first.split(" ", 1)
    else:
        first_name, surname = first, ""
    template = rng.choice(list(payload["alias_templates"]))
    nickname = rng.choice(list(payload["quote_nicknames"]))
    moniker = strip_leading_article(rng.choice(list(payload["solo_monikers"])))
    codename_adj = rng.choice(list(payload["codename_adjectives"]))
    codename_noun = rng.choice(list(payload["codename_nouns"]))
    title = rng.choice(list(titles))
    substitutions = {
        "title": title,
        "title_prefix": title,
        "first": first_name,
        "human_first": first_name,
        "first_name": first_name,
        "given_name": first_name,
        "human_first_name": first_name,
        "surname": surname,
        "human_last": surname,
        "human_surname": surname,
        "last_name": surname,
        "family_name": surname,
        "human_last_name": surname,
        "full_name": f"{first_name} {surname}".strip(),
        "human_name": f"{first_name} {surname}".strip(),
        "nickname": nickname,
        "quote_nickname": nickname,
        "moniker": moniker,
        "solo_moniker": moniker,
        "codename_adj": codename_adj,
        "codename_adjective": codename_adj,
        "codename_noun": codename_noun,
        "mythic_epithet": rng.choice(list(payload["mythic_epithets"])),
        "role_epithet": rng.choice(list(payload["role_epithets"])),
    }
    return sanitize_character_name(template.format(**substitutions).strip())


def _character_priors(base_dir: Path, mode: str) -> dict[str, Any]:
    artifact_path = modeling_priors_path(base_dir)
    payload = load_modeling_priors_artifact(base_dir, mode=mode)
    section = payload.get("character_generation", {}) if isinstance(payload, dict) and isinstance(payload.get("character_generation"), dict) else {}
    if section:
        audit_artifact_usage("modeling_priors.json", artifact_path, sections=["character_generation"])
    return section


def _human_name_mix(payload: dict, priors: dict[str, Any], archetype: str, mode: str) -> tuple[float, dict[str, float]]:
    raw = (payload.get("human_name_mix_by_archetype", {}) or {}).get(archetype)
    if raw is None:
        if mode == "research":
            audit_fallback_hit(
                "character_identity_bank.json",
                "human_name_mix_missing_for_archetype",
                detail=f"{archetype} missing human_name_mix_by_archetype entry",
                mode=mode,
            )
        alias_frequency = float(priors.get("alias_frequency", 0.12) or 0.12)
        scalar = max(0.05, min(0.98, 1.0 - alias_frequency))
        return scalar, {"full_name": 1.0}
    if isinstance(raw, dict):
        weights = _weights(raw)
        total = sum(float(v) for v in weights.values())
        return float(min(1.0, max(0.0, total))), weights
    try:
        scalar = float(raw)
    except Exception:
        scalar = 0.8
    return float(min(1.0, max(0.0, scalar))), {"full_name": 1.0}


def _render_human_variant(
    rng: random.Random,
    family: dict,
    gender: str,
    payload: dict,
    variant_weights: dict[str, float],
) -> str:
    full = _human_name(rng, family, gender)
    if not full:
        return ""
    if " " in full:
        first_name, surname = full.split(" ", 1)
    else:
        first_name, surname = full, "Vale"
    variant = rng.choices(list(variant_weights.keys()), weights=list(variant_weights.values()), k=1)[0]
    if variant == "first_only":
        return first_name
    if variant == "title_last":
        titles = payload.get("title_prefixes_m" if gender == "M" else "title_prefixes_f" if gender == "F" else "title_prefixes_nb", [])
        title = rng.choice([str(x) for x in titles if str(x).strip()] or ["Agent"])
        return f"{title} {surname}".strip()
    return full


def _fallback_unique_human_name(
    rng: random.Random,
    families: list[dict],
    gender: str,
    used_names: set[str],
) -> str | None:
    for _ in range(1000):
        family = rng.choice(families)
        candidate = sanitize_character_name(_human_name(rng, family, gender, distinctive=True) or "")
        if not candidate:
            continue
        key = candidate.casefold()
        if key not in used_names:
            used_names.add(key)
            return candidate
    return None


def generate_characters(target: int, *, seed: int, base_dir: Path, mode: str) -> list[dict]:
    rng = random.Random(seed)
    used_names: set[str] = set()
    characters: list[dict] = []
    payload = _character_payload(base_dir, mode)
    families = _identity_families(base_dir, mode)
    priors = _character_priors(base_dir, mode)

    archetypes = _weights(payload.get("archetype_weights"))
    arch_list = list(archetypes.keys())
    arch_weights = list(archetypes.values())

    t0 = time.time()
    for i in range(target):
        archetype = rng.choices(arch_list, weights=arch_weights, k=1)[0]
        gender = rng.choices(["M", "F", "NB"], weights=[0.50, 0.42, 0.08], k=1)[0]
        human_share, human_variants = _human_name_mix(payload, priors, archetype, mode)
        raw_nonliteral = payload.get("nonliteral_share_by_archetype", {}).get(archetype)
        if raw_nonliteral is None:
            if mode == "research":
                audit_fallback_hit(
                    "character_identity_bank.json",
                    "nonliteral_share_missing_for_archetype",
                    detail=f"{archetype} missing nonliteral_share_by_archetype entry",
                    mode=mode,
                )
            raw_nonliteral = priors.get("nonliteral_name_share", 0.05)
        nonliteral_share = float(raw_nonliteral)

        name = None
        for _ in range(240):
            family = rng.choice(families)
            if rng.random() < nonliteral_share:
                candidate = strip_leading_article(rng.choice(list(payload["solo_monikers"])))
            elif rng.random() < human_share:
                candidate = _render_human_variant(rng, family, gender, payload, human_variants)
            else:
                candidate = _render_alias(rng, payload, family, gender)
            candidate = sanitize_character_name(candidate)
            if not candidate:
                continue
            key = candidate.casefold()
            if key not in used_names:
                used_names.add(key)
                name = candidate
                break
        if name is None:
            name = _fallback_unique_human_name(rng, families, gender, used_names)
        if name is None:
            raise RuntimeError(f"Unable to synthesize unique character name at row {i + 1}")

        characters.append({"character_name": name, "archetype": archetype})

        if (i + 1) % 100000 == 0:
            print(f"  {i + 1:>10,} / {target:,} ({time.time() - t0:.1f}s)")

    elapsed = time.time() - t0
    print(f"\n  Generated {len(characters):,} unique characters in {elapsed:.1f}s")
    return characters


def main() -> None:
    parser = argparse.ArgumentParser(description="Character bank generator for Mirage.")
    parser.add_argument("--base-dir", default=str(BASE_DIR))
    parser.add_argument("--target", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("research", "debug"), default=current_mode())
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    out_path = Path(args.out).resolve() if args.out else (base_dir / "entities" / "character_bank.csv")

    print("=" * 60)
    print("  CHARACTER BANK GENERATOR")
    print("=" * 60)
    print(f"  Target:  {args.target:,}")
    print(f"  Mode:    {args.mode}")
    print("=" * 60)

    characters = generate_characters(int(args.target), seed=int(args.seed), base_dir=base_dir, mode=str(args.mode))
    archetypes = Counter(c["archetype"] for c in characters)
    print("\n  Archetypes:")
    for label, count in archetypes.most_common():
        print(f"    {label:<22} {count:>8,} ({count / len(characters) * 100:.1f}%)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["character_name", "archetype"])
        writer.writeheader()
        writer.writerows(characters)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
