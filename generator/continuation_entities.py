from __future__ import annotations

import csv
import json
import math
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from contracts import GENRES, GENRE_WEIGHTS, generate_compositional_title
from continuation_lifecycle import (
    plan_company_lifecycle,
    plan_person_survivors,
    update_lifecycle_artifact,
)
from text_polish import (
    contains_placeholder_syntax,
    looks_like_weak_tagline,
    looks_like_weak_title,
    sanitize_tagline,
    tagline_signature,
)


MANIFEST_NAME = "continuation_entity_manifest.json"


@dataclass(frozen=True)
class EntityTopupResult:
    kind: str
    before: int
    after: int
    target: int
    added: int
    duplicate_candidates: int = 0
    survivor_extensions: int = 0
    lifecycle_updates: int = 0
    company_dissolutions: int = 0
    new_founded: int = 0
    new_defunct: int = 0
    path: str | None = None

    @property
    def satisfied(self) -> bool:
        return int(self.after) >= int(self.target)


@contextmanager
def _temporary_year_env(start_year: int | None, end_year: int | None) -> Iterable[None]:
    old_start = os.environ.get("DATA_SYS_START_YEAR")
    old_end = os.environ.get("DATA_SYS_END_YEAR")
    try:
        if start_year is not None:
            os.environ["DATA_SYS_START_YEAR"] = str(int(start_year))
        if end_year is not None:
            os.environ["DATA_SYS_END_YEAR"] = str(int(end_year))
        yield
    finally:
        if old_start is None:
            os.environ.pop("DATA_SYS_START_YEAR", None)
        else:
            os.environ["DATA_SYS_START_YEAR"] = old_start
        if old_end is None:
            os.environ.pop("DATA_SYS_END_YEAR", None)
        else:
            os.environ["DATA_SYS_END_YEAR"] = old_end


def _clear_keyword_capacity_fallback_hits(base_dir: Path) -> int:
    """Remove expected audit hits from caught keyword capacity probes only."""
    step_id = str(os.environ.get("DATA_SYS_STEP_ID", "") or "").strip()
    if not step_id:
        return 0

    audit_paths: list[Path] = []
    for key in ("DATA_SYS_RESEARCH_AUDIT", "DATA_SYS_RESEARCH_AUDIT_LATEST"):
        value = str(os.environ.get(key, "") or "").strip()
        if value:
            audit_paths.append(Path(value))
    try:
        from policy_runtime import research_audit_path

        audit_paths.append(research_audit_path(base_dir))
    except Exception:
        pass

    seen: set[str] = set()
    removed_total = 0
    for audit_path in audit_paths:
        key = str(audit_path.resolve()) if audit_path.exists() else str(audit_path)
        if key in seen or not audit_path.exists():
            continue
        seen.add(key)
        try:
            payload = json.loads(audit_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue

        def is_expected(row: Any) -> bool:
            if not isinstance(row, dict):
                return False
            if str(row.get("step_id", "") or "").strip() != step_id:
                return False
            haystack = " ".join(
                str(row.get(field, "") or "")
                for field in ("component", "reason", "detail")
            )
            return "keyword_seed_capacity_exhausted" in haystack

        root_hits = payload.get("fallback_hits")
        root_removed = 0
        if isinstance(root_hits, list):
            kept = []
            for row in root_hits:
                if is_expected(row):
                    root_removed += 1
                else:
                    kept.append(row)
            payload["fallback_hits"] = kept

        steps = payload.get("steps")
        if isinstance(steps, dict):
            bucket = steps.get(step_id)
            if isinstance(bucket, dict):
                step_hits = bucket.get("fallback_hits")
                if isinstance(step_hits, list):
                    bucket["fallback_hits"] = [row for row in step_hits if not is_expected(row)]

        if root_removed:
            payload["fallback_hit_count"] = max(0, int(payload.get("fallback_hit_count", 0) or 0) - root_removed)
            try:
                audit_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                continue
            removed_total += root_removed
    return removed_total


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected JSON list at {path}")
    return [row for row in payload if isinstance(row, dict)]


def _write_json_list(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _max_int_id(rows: Iterable[dict[str, Any]], key: str) -> int:
    out = 0
    for row in rows:
        try:
            out = max(out, int(row.get(key, 0) or 0))
        except Exception:
            continue
    return out


def _clean_key(value: object) -> str:
    return " ".join(str(value or "").split()).casefold()


def _person_name_variant(base_name: object, used_names: set[str], *, seed: int) -> str | None:
    base = " ".join(str(base_name or "").strip().split()) or "Mirage Performer"
    suffixes = (
        "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
        "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII",
    )
    for suffix in suffixes:
        candidate = f"{base} {suffix}"
        if _clean_key(candidate) not in used_names:
            return candidate
    for idx in range(1, 5000):
        candidate = f"{base} {chr(65 + ((seed + idx) % 26))}. {idx}"
        if _clean_key(candidate) not in used_names:
            return candidate
    return None


def _company_name_variant(base_name: object, used_names: set[str], *, seed: int) -> str | None:
    base = " ".join(str(base_name or "").strip().split()) or "Mirage Pictures"
    suffixes = (
        "Studios", "Pictures", "Films", "Productions", "Media", "Entertainment",
        "Collective", "Works", "International", "Ventures", "House", "Group",
        "Factory", "Unit", "Labs", "Releasing",
    )
    for suffix in suffixes:
        candidate = f"{base} {suffix}"
        if _clean_key(candidate) not in used_names:
            return candidate
    for idx in range(1, 20000):
        candidate = f"{base} {2050 + ((seed + idx) % 50)}-{idx}"
        if _clean_key(candidate) not in used_names:
            return candidate
    return None


def _keyword_variant(source: dict[str, Any], used: set[str], *, seed: int) -> dict[str, Any] | None:
    base = str(source.get("keyword", source.get("name", "")) or "").strip()
    if not base:
        return None
    modifiers = (
        "post-2050", "neo", "late-century", "orbital", "synthetic", "virtual",
        "climate-era", "diaspora", "algorithmic", "memory", "quantum", "frontier",
        "augmented", "autonomous", "deep-space", "networked",
    )
    contexts = (
        "reckoning", "legacy", "alliance", "crisis", "migration", "signal",
        "archive", "uprising", "protocol", "frontier", "afterimage", "compact",
    )
    for idx in range(1, 5000):
        modifier = modifiers[(seed + idx) % len(modifiers)]
        context = contexts[(seed * 3 + idx) % len(contexts)]
        candidate = f"{modifier}-{base}-{context}"
        if _clean_key(candidate) not in used:
            row = dict(source)
            row["keyword"] = candidate
            row["selection_bucket"] = str(row.get("selection_bucket") or "story_specific")
            row["_origin"] = "continuation_future_variant"
            return row
    return None


def _candidate_batch_size(remaining: int, *, minimum: int = 256) -> int:
    return max(int(minimum), int(math.ceil(float(remaining) * 1.18)) + 64)


def topup_persons(
    base_dir: Path,
    *,
    target: int,
    seed: int,
    mode: str,
    extension_start_year: int | None = None,
    extension_end_year: int | None = None,
    survivor_share: float = 0.015,
) -> EntityTopupResult:
    from generate_persons_procedural import generate_persons

    path = base_dir / "entities" / "persons.json"
    persons = _load_json_list(path)
    before = len(persons)
    survivor_plan = plan_person_survivors(
        persons,
        seed=seed,
        extension_start_year=extension_start_year,
        extension_end_year=extension_end_year,
        survivor_share=survivor_share,
    )
    persons = survivor_plan.rows
    survivor_extensions = int(survivor_plan.stats.get("survivor_extensions", 0) or 0)
    if survivor_plan.updates:
        update_lifecycle_artifact(
            base_dir,
            extension_start_year=extension_start_year,
            extension_end_year=extension_end_year,
            person_updates=survivor_plan.updates,
            stats=survivor_plan.stats,
        )
    if before >= int(target):
        _write_json_list(path, persons)
        return EntityTopupResult(
            "persons",
            before,
            len(persons),
            int(target),
            0,
            survivor_extensions=survivor_extensions,
            lifecycle_updates=len(survivor_plan.updates),
            path=str(path),
        )

    needed = int(target) - before
    used_names = {_clean_key(row.get("name")) for row in persons if _clean_key(row.get("name"))}
    next_id = _max_int_id(persons, "person_id") + 1
    added: list[dict[str, Any]] = []
    duplicate_candidates = 0

    with _temporary_year_env(extension_start_year, extension_end_year):
        for attempt in range(12):
            remaining = needed - len(added)
            if remaining <= 0:
                break
            batch_target = _candidate_batch_size(remaining, minimum=min(5000, max(256, remaining)))
            batch_seed = int(seed) + 910_000 + attempt * 10_007 + before
            batch = generate_persons(batch_target, seed=batch_seed, base_dir=base_dir, mode=mode)
            for row in batch:
                key = _clean_key(row.get("name"))
                if not key or key in used_names:
                    duplicate_candidates += 1
                    continue
                row = dict(row)
                row["person_id"] = int(next_id)
                row["bio"] = ""
                row["style_tags"] = []
                row["genre_affinity"] = []
                used_names.add(key)
                next_id += 1
                added.append(row)
                if len(added) >= needed:
                    break

    if len(added) < needed:
        # At very large scales the finite LLM-authored name bank can run out of
        # unique combinations by a tiny margin. Fill only the residual shortfall
        # with deterministic IMDb-style disambiguated variants so the target
        # cardinality stays exact without changing the normal diversity path.
        for attempt in range(6):
            remaining = needed - len(added)
            if remaining <= 0:
                break
            batch_target = max(512, remaining * 6)
            batch_seed = int(seed) + 1_930_000 + attempt * 17_017 + before
            with _temporary_year_env(extension_start_year, extension_end_year):
                batch = generate_persons(batch_target, seed=batch_seed, base_dir=base_dir, mode=mode)
            for row in batch:
                row = dict(row)
                key = _clean_key(row.get("name"))
                if not key:
                    duplicate_candidates += 1
                    continue
                if key in used_names:
                    duplicate_candidates += 1
                    variant = _person_name_variant(row.get("name"), used_names, seed=int(next_id))
                    if not variant:
                        continue
                    row["name"] = variant
                    key = _clean_key(variant)
                row["person_id"] = int(next_id)
                row["bio"] = ""
                row["style_tags"] = []
                row["genre_affinity"] = []
                used_names.add(key)
                next_id += 1
                added.append(row)
                if len(added) >= needed:
                    break

    if len(added) < needed:
        raise RuntimeError(f"Could only top up persons by {len(added)} rows; needed {needed}")
    persons.extend(added)
    _write_json_list(path, persons)
    return EntityTopupResult(
        "persons",
        before,
        len(persons),
        int(target),
        len(added),
        duplicate_candidates=duplicate_candidates,
        survivor_extensions=survivor_extensions,
        lifecycle_updates=len(survivor_plan.updates),
        path=str(path),
    )


def topup_companies(
    base_dir: Path,
    *,
    target: int,
    seed: int,
    mode: str,
    extension_start_year: int | None = None,
    extension_end_year: int | None = None,
    company_lifecycle_policy: str = "balanced",
) -> EntityTopupResult:
    from generate_companies_procedural import generate_companies

    path = base_dir / "entities" / "companies.json"
    companies = _load_json_list(path)
    before = len(companies)
    if before >= int(target):
        lifecycle_plan = plan_company_lifecycle(
            companies,
            base_dir=base_dir,
            seed=seed,
            extension_start_year=extension_start_year,
            extension_end_year=extension_end_year,
            company_lifecycle_policy=company_lifecycle_policy,
            new_company_ids=[],
        )
        if lifecycle_plan.updates:
            companies = lifecycle_plan.rows
            _write_json_list(path, companies)
            update_lifecycle_artifact(
                base_dir,
                extension_start_year=extension_start_year,
                extension_end_year=extension_end_year,
                company_lifecycle_policy=company_lifecycle_policy,
                company_updates=lifecycle_plan.updates,
                stats=lifecycle_plan.stats,
            )
        return EntityTopupResult(
            "companies",
            before,
            len(companies),
            int(target),
            0,
            lifecycle_updates=len(lifecycle_plan.updates),
            company_dissolutions=int(lifecycle_plan.stats.get("company_dissolutions", 0) or 0),
            new_founded=int(lifecycle_plan.stats.get("new_founded", 0) or 0),
            new_defunct=int(lifecycle_plan.stats.get("new_defunct", 0) or 0),
            path=str(path),
        )

    needed = int(target) - before
    used_names = {_clean_key(row.get("name")) for row in companies if _clean_key(row.get("name"))}
    next_id = _max_int_id(companies, "company_id") + 1
    added: list[dict[str, Any]] = []
    new_company_ids: list[int] = []
    duplicate_candidates = 0

    for attempt in range(12):
        remaining = needed - len(added)
        if remaining <= 0:
            break
        batch_target = _candidate_batch_size(remaining, minimum=min(1000, max(128, remaining)))
        batch_seed = int(seed) + 1_210_000 + attempt * 7_919 + before
        batch = generate_companies(batch_target, seed=batch_seed, base_dir=base_dir, mode=mode)
        for row in batch:
            key = _clean_key(row.get("name"))
            if not key or key in used_names:
                duplicate_candidates += 1
                continue
            row = dict(row)
            row["company_id"] = int(next_id)
            new_company_ids.append(int(next_id))
            used_names.add(key)
            next_id += 1
            added.append(row)
            if len(added) >= needed:
                break

    if len(added) < needed:
        for attempt in range(8):
            remaining = needed - len(added)
            if remaining <= 0:
                break
            batch_target = max(512, remaining * 8)
            batch_seed = int(seed) + 2_230_000 + attempt * 19_019 + before
            batch = generate_companies(batch_target, seed=batch_seed, base_dir=base_dir, mode=mode)
            for row in batch:
                row = dict(row)
                key = _clean_key(row.get("name"))
                if not key:
                    duplicate_candidates += 1
                    continue
                if key in used_names:
                    duplicate_candidates += 1
                    variant = _company_name_variant(row.get("name"), used_names, seed=int(next_id))
                    if not variant:
                        continue
                    row["name"] = variant
                    key = _clean_key(variant)
                row["company_id"] = int(next_id)
                new_company_ids.append(int(next_id))
                used_names.add(key)
                next_id += 1
                added.append(row)
                if len(added) >= needed:
                    break

    if len(added) < needed:
        raise RuntimeError(f"Could only top up companies by {len(added)} rows; needed {needed}")
    companies.extend(added)
    lifecycle_plan = plan_company_lifecycle(
        companies,
        base_dir=base_dir,
        seed=seed,
        extension_start_year=extension_start_year,
        extension_end_year=extension_end_year,
        company_lifecycle_policy=company_lifecycle_policy,
        new_company_ids=new_company_ids,
    )
    companies = lifecycle_plan.rows
    _write_json_list(path, companies)
    if lifecycle_plan.updates:
        update_lifecycle_artifact(
            base_dir,
            extension_start_year=extension_start_year,
            extension_end_year=extension_end_year,
            company_lifecycle_policy=company_lifecycle_policy,
            company_updates=lifecycle_plan.updates,
            stats=lifecycle_plan.stats,
        )
    return EntityTopupResult(
        "companies",
        before,
        len(companies),
        int(target),
        len(added),
        duplicate_candidates=duplicate_candidates,
        lifecycle_updates=len(lifecycle_plan.updates),
        company_dissolutions=int(lifecycle_plan.stats.get("company_dissolutions", 0) or 0),
        new_founded=int(lifecycle_plan.stats.get("new_founded", 0) or 0),
        new_defunct=int(lifecycle_plan.stats.get("new_defunct", 0) or 0),
        path=str(path),
    )


def topup_keywords(
    base_dir: Path,
    *,
    target: int,
    seed: int,
    mode: str,
) -> EntityTopupResult:
    from generate_keywords_procedural import generate_keywords

    path = base_dir / "entities" / "keywords.json"
    keywords = _load_json_list(path)
    before = len(keywords)
    if before >= int(target):
        _clear_keyword_capacity_fallback_hits(base_dir)
        return EntityTopupResult("keywords", before, before, int(target), 0, path=str(path))

    needed = int(target) - before
    used = {_clean_key(row.get("keyword", row.get("name"))) for row in keywords if _clean_key(row.get("keyword", row.get("name")))}
    next_id = _max_int_id(keywords, "keyword_id") + 1
    added: list[dict[str, Any]] = []
    duplicate_candidates = 0
    capacity_probe_failures = 0

    for attempt in range(8):
        remaining = needed - len(added)
        if remaining <= 0:
            break
        batch_target = _candidate_batch_size(remaining, minimum=min(2500, max(256, remaining)))
        batch_seed = int(seed) + attempt * 4_001
        try:
            batch = generate_keywords(batch_target, seed=batch_seed, base_dir=base_dir, mode=mode)
        except Exception as exc:
            if "keyword_seed_capacity_exhausted" not in str(exc):
                raise
            capacity_probe_failures += 1
            duplicate_candidates += 1
            continue
        for row in batch:
            key = _clean_key(row.get("keyword", row.get("name")))
            if not key or key in used:
                duplicate_candidates += 1
                continue
            row = dict(row)
            row["keyword_id"] = int(next_id)
            used.add(key)
            next_id += 1
            added.append(row)
            if len(added) >= needed:
                break

    if len(added) < needed:
        rng = np.random.default_rng(int(seed) + before + 6_100_000)
        source_rows = list(keywords) + list(added)
        for _ in range(max(needed * 4, 1024)):
            if len(added) >= needed or not source_rows:
                break
            source = dict(source_rows[int(rng.integers(0, len(source_rows)))])
            row = _keyword_variant(source, used, seed=int(next_id))
            if not row:
                duplicate_candidates += 1
                continue
            key = _clean_key(row.get("keyword", row.get("name")))
            if not key or key in used:
                duplicate_candidates += 1
                continue
            row["keyword_id"] = int(next_id)
            used.add(key)
            next_id += 1
            added.append(row)

    if len(added) < needed:
        raise RuntimeError(f"Could only top up keywords by {len(added)} rows; needed {needed}")
    keywords.extend(added)
    _write_json_list(path, keywords)
    if capacity_probe_failures:
        _clear_keyword_capacity_fallback_hits(base_dir)
    return EntityTopupResult("keywords", before, len(keywords), int(target), len(added), duplicate_candidates=duplicate_candidates, path=str(path))


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        count = sum(1 for _ in handle)
    return max(0, count - 1)


def _read_character_names(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = _clean_key(row.get("character_name"))
            if key:
                out.add(key)
    return out


def topup_characters(
    base_dir: Path,
    *,
    target: int,
    seed: int,
    mode: str,
) -> EntityTopupResult:
    from generate_character_bank import generate_characters

    path = base_dir / "entities" / "character_bank.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    before = _count_csv_rows(path)
    if before >= int(target):
        return EntityTopupResult("characters", before, before, int(target), 0, path=str(path))

    needed = int(target) - before
    used = _read_character_names(path)
    added_rows: list[dict[str, Any]] = []
    duplicate_candidates = 0

    max_attempts = 32
    for attempt in range(max_attempts):
        remaining = needed - len(added_rows)
        if remaining <= 0:
            break
        if attempt < 12:
            minimum = min(20000, max(1000, remaining))
        else:
            # At very large continuation scale the existing bank can collide
            # with most small batches. Keep generating from the same rich
            # character model, but use wider late batches so the tail does not
            # fail a long run because of a few thousand duplicates.
            minimum = max(20000, min(120000, remaining * 4))
        batch_target = _candidate_batch_size(remaining, minimum=minimum)
        batch_seed = int(seed) + 1_510_000 + attempt * 6_151 + before
        batch = generate_characters(batch_target, seed=batch_seed, base_dir=base_dir, mode=mode)
        for row in batch:
            key = _clean_key(row.get("character_name"))
            if not key or key in used:
                duplicate_candidates += 1
                continue
            used.add(key)
            added_rows.append({"character_name": row.get("character_name", ""), "archetype": row.get("archetype", "Supporting")})
            if len(added_rows) >= needed:
                break

    if len(added_rows) < needed:
        raise RuntimeError(f"Could only top up characters by {len(added_rows)} rows; needed {needed}")

    file_exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["character_name", "archetype"])
        if not file_exists or before == 0:
            writer.writeheader()
        writer.writerows(added_rows)

    after = before + len(added_rows)
    return EntityTopupResult("characters", before, after, int(target), len(added_rows), duplicate_candidates=duplicate_candidates, path=str(path))


def _planned_years_for_extension(
    *,
    needed: int,
    start_year: int,
    end_year: int,
    seed: int,
    mode: str,
    temporal: dict[str, Any] | None,
) -> list[int]:
    rng = np.random.RandomState(int(seed) + 31_337)
    if needed <= 0:
        return []
    if mode == "research" and isinstance(temporal, dict):
        from bootstrap_artifacts import year_weight_map
        from topup_title_bank import _desired_year_counts_from_weights

        weights = year_weight_map(temporal, start_year=int(start_year), end_year=int(end_year))
        counts = _desired_year_counts_from_weights(int(needed), weights)
    else:
        years = list(range(int(start_year), int(end_year) + 1))
        base = int(needed) // len(years)
        rem = int(needed) % len(years)
        counts = {year: base + (1 if idx < rem else 0) for idx, year in enumerate(years)}
    out: list[int] = []
    for year, count in counts.items():
        out.extend([int(year)] * int(count))
    rng.shuffle(out)
    return out


def topup_titles(
    base_dir: Path,
    *,
    target: int,
    seed: int,
    mode: str,
    extension_start_year: int | None,
    extension_end_year: int | None,
) -> EntityTopupResult:
    from bootstrap_artifacts import load_temporal_regime_plan
    from topup_title_bank import (
        _base_title_genre_weights,
        _default_grammar,
        _genre_probability_vector,
        _is_usable_rendered_text,
        _load_research_grammar,
        _materialize_tagline_for_title,
        _normalise_genre_weight_map,
        _render_title,
        _tagline_template_family_signature,
        _validate_title_capacity_for_target,
    )

    if extension_start_year is None or extension_end_year is None:
        raise RuntimeError("Title continuation requires --extension-start-year and --extension-end-year")

    path = base_dir / "entities" / "title_bank.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        df = pd.read_csv(path, low_memory=False)
    else:
        df = pd.DataFrame(columns=["title", "tagline", "genre_hint", "year", "award_contender"])
    before = int(len(df))
    if before >= int(target):
        return EntityTopupResult("titles", before, before, int(target), 0, path=str(path))

    needed = int(target) - before
    rng = np.random.RandomState(int(seed) + 41_011)
    fast_taglines = str(os.environ.get("DATA_SYS_FAST_TITLE_TAGLINES", "")).strip().lower() in {"1", "true", "yes", "on"}

    used_titles = set(df["title"].fillna("").astype(str).tolist()) if "title" in df.columns else set()
    existing_title_keys = {_clean_key(title) for title in used_titles if _clean_key(title)}
    used_title_keys = set(existing_title_keys)
    tagline_counts: dict[str, int] = {}
    tagline_history: list[str] = []
    if "tagline" in df.columns:
        for raw in df["tagline"].fillna("").astype(str).tolist():
            clean = sanitize_tagline(raw)
            sig = tagline_signature(clean)
            if not sig:
                continue
            tagline_counts[sig] = tagline_counts.get(sig, 0) + 1
            tagline_history.append(clean)

    if mode == "research":
        grammar = _load_research_grammar(base_dir, mode)
        temporal = load_temporal_regime_plan(base_dir, mode=mode)
        base_genre_weights, title_priors = _base_title_genre_weights(base_dir, mode)
        _validate_title_capacity_for_target(
            grammar,
            target_count=int(needed),
            base_genre_weights=base_genre_weights,
            temporal=temporal,
            title_priors=title_priors,
            start_year=int(extension_start_year),
            end_year=int(extension_end_year),
        )
    else:
        grammar = _default_grammar()
        temporal = None
        base_genre_weights, title_priors = _normalise_genre_weight_map(dict(GENRE_WEIGHTS)), {}

    planned_years = _planned_years_for_extension(
        needed=needed,
        start_year=int(extension_start_year),
        end_year=int(extension_end_year),
        seed=seed,
        mode=mode,
        temporal=temporal if isinstance(temporal, dict) else None,
    )

    tagline_template_family_counts: dict[str, int] = {}
    tagline_template_family_cap = 2
    if mode == "research":
        template_families = {
            _tagline_template_family_signature(template)
            for templates in (grammar.get("tagline_templates") or {}).values()
            for template in (templates or [])
            if _tagline_template_family_signature(template)
        }
        average_uses = int(np.ceil(float(max(1, needed)) / float(max(1, len(template_families)))))
        tagline_template_family_cap = max(64, average_uses * 4)
        if fast_taglines:
            tagline_template_family_cap = max(tagline_template_family_cap, int(target))

    rows: list[dict[str, Any]] = []
    duplicate_candidates = 0
    for idx, year in enumerate(planned_years):
        genre_probs = _genre_probability_vector(
            base_genre_weights,
            temporal=temporal if isinstance(temporal, dict) else None,
            title_priors=title_priors,
            year=int(year),
        )
        genre = str(rng.choice(GENRES, p=genre_probs))
        if mode == "research":
            title = None
            for _ in range(128):
                candidate = _render_title(rng, grammar, genre, mode=mode)
                candidate_key = _clean_key(candidate)
                if (
                    candidate
                    and candidate_key
                    and candidate_key not in used_title_keys
                    and _is_usable_rendered_text(candidate)
                    and not looks_like_weak_title(candidate)
                ):
                    title = candidate
                    break
                duplicate_candidates += 1
            if title is None:
                raise RuntimeError(f"Unable to generate unique continuation title for year={year} genre={genre}")
            tagline, tagline_template_family = _materialize_tagline_for_title(
                grammar=grammar,
                genre=genre,
                rng=rng,
                mode=mode,
                title=title,
                tagline_history=tagline_history,
                tagline_counts=tagline_counts,
                tagline_template_family_counts=tagline_template_family_counts,
                tagline_template_family_cap=tagline_template_family_cap,
                fast_taglines=fast_taglines,
            )
        else:
            title = None
            for _ in range(512):
                candidate = generate_compositional_title(rng, used_titles)
                candidate_key = _clean_key(candidate)
                if candidate and candidate_key and candidate_key not in used_title_keys:
                    title = candidate
                    break
                duplicate_candidates += 1
            if title is None:
                raise RuntimeError(f"Unable to generate unique continuation title for year={year} genre={genre}")
            tagline = sanitize_tagline(f"{title} changes everything.", title=title)
            tagline_template_family = ""

        if not title or contains_placeholder_syntax(title) or looks_like_weak_title(title):
            raise RuntimeError(f"Generated invalid continuation title: {title!r}")
        if not tagline or contains_placeholder_syntax(tagline) or looks_like_weak_tagline(tagline, title=title):
            raise RuntimeError(f"Generated invalid continuation tagline for {title!r}: {tagline!r}")

        used_titles.add(title)
        used_title_keys.add(_clean_key(title))
        sig = tagline_signature(tagline)
        if sig:
            tagline_counts[sig] = tagline_counts.get(sig, 0) + 1
            tagline_history.append(tagline)
        if tagline_template_family:
            tagline_template_family_counts[tagline_template_family] = tagline_template_family_counts.get(tagline_template_family, 0) + 1
        rows.append(
            {
                "title": title,
                "tagline": tagline,
                "genre_hint": genre,
                "year": int(year),
                "award_contender": bool(rng.rand() < 0.08),
            }
        )
        if (idx + 1) % 10000 == 0 or (idx + 1) == needed:
            print(f"Generated {idx + 1:,} / {needed:,} continuation title rows", flush=True)

    add_df = pd.DataFrame(rows)
    for col in df.columns:
        if col not in add_df.columns:
            add_df[col] = None
    for col in add_df.columns:
        if col not in df.columns:
            df[col] = None
    out = pd.concat([df[df.columns], add_df[df.columns]], ignore_index=True)
    added_title_keys = [_clean_key(title) for title in add_df["title"].fillna("").astype(str).tolist()]
    if len(set(added_title_keys)) != len(added_title_keys):
        raise RuntimeError("Continuation title top-up generated duplicate extension titles")
    if any(key in existing_title_keys for key in added_title_keys):
        raise RuntimeError("Continuation title top-up generated a title that already exists")
    out.to_csv(path, index=False)
    return EntityTopupResult("titles", before, len(out), int(target), len(rows), duplicate_candidates=duplicate_candidates, path=str(path))


def write_manifest(base_dir: Path, results: list[EntityTopupResult], *, metadata: dict[str, Any] | None = None) -> Path:
    path = base_dir / "entities" / MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    previous: dict[str, Any] = {}
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            previous = {}
    history = previous.get("history")
    if not isinstance(history, list):
        history = []
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
        "results": [asdict(result) for result in results],
    }
    history.append(entry)
    payload = {"latest": entry, "history": history[-50:]}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
