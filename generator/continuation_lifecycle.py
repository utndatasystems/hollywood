from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


LIFECYCLE_ARTIFACT_NAME = "continuation_lifecycle.json"
VALID_COMPANY_LIFECYCLE_POLICIES = {"balanced", "preserve", "new-era"}

_POLICY_RATES = {
    "preserve": {"old_dissolve": 0.03, "new_defunct": 0.03, "founding_mode": 0.12},
    "balanced": {"old_dissolve": 0.12, "new_defunct": 0.07, "founding_mode": 0.18},
    "new-era": {"old_dissolve": 0.24, "new_defunct": 0.12, "founding_mode": 0.35},
}

_DISSOLUTION_WEIGHT_BY_TIER = {
    "Global": 0.03,
    "Major": 0.08,
    "Mid-Budget": 1.00,
    "Indie": 2.20,
    "Micro": 3.20,
}


@dataclass(frozen=True)
class LifecyclePlanResult:
    rows: list[dict[str, Any]]
    updates: list[dict[str, Any]]
    stats: dict[str, int]


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _artifact_path(base_dir: Path) -> Path:
    return Path(base_dir) / "entities" / LIFECYCLE_ARTIFACT_NAME


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return float(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _coerce_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    text = str(value).strip()
    if not text:
        return default
    try:
        return int(float(text))
    except Exception:
        return default


def _is_missing_year(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    return str(value).strip() == ""


def _stable_unit(seed: int, *parts: Any) -> float:
    payload = "|".join([str(int(seed))] + [str(part) for part in parts])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(0xFFFFFFFFFFFF)


def _normalize_tier(value: Any) -> str:
    text = str(value or "").strip()
    low = text.casefold()
    if "global" in low:
        return "Global"
    if "major" in low:
        return "Major"
    if "micro" in low:
        return "Micro"
    if "indie" in low or "independent" in low:
        return "Indie"
    if "mid" in low:
        return "Mid-Budget"
    return text or "Mid-Budget"


def _row_id(row: dict[str, Any], key: str) -> int | None:
    value = _coerce_int(row.get(key), None)
    return value if value is not None and value > 0 else None


def _row_pop_weight(row: dict[str, Any]) -> float:
    try:
        value = float(row.get("pop_weight", 0.5) or 0.5)
    except Exception:
        value = 0.5
    if not math.isfinite(value):
        value = 0.5
    return max(0.01, min(10.0, value))


def _weighted_choice_without_replacement(
    rng: np.random.RandomState,
    items: list[int],
    weights: list[float],
    take: int,
) -> set[int]:
    if take <= 0 or not items:
        return set()
    take = min(int(take), len(items))
    weights_arr = np.asarray(weights, dtype=float)
    if not np.isfinite(weights_arr).all() or float(weights_arr.sum()) <= 0:
        weights_arr = np.ones(len(items), dtype=float)
    weights_arr = weights_arr / float(weights_arr.sum())
    chosen = rng.choice(len(items), size=take, replace=False, p=weights_arr)
    return {int(items[int(idx)]) for idx in chosen}


def _infer_source_years(base_dir: Path, extension_start_year: int | None) -> tuple[int, int]:
    ext_start = int(extension_start_year) if extension_start_year is not None else 2026
    source_start = ext_start - 51
    source_end = ext_start - 1
    manifest_path = Path(base_dir) / "_step100_resume" / "manifest.json"
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        if isinstance(payload, dict):
            source_start = _coerce_int(payload.get("start_year"), source_start) or source_start
            extension = payload.get("extension", {}) if isinstance(payload.get("extension"), dict) else {}
            source_end = (
                _coerce_int(extension.get("source_end_year"), None)
                or _coerce_int(extension.get("source_end"), None)
                or _coerce_int(payload.get("end_year"), source_end)
                or source_end
            )
            if source_end >= ext_start:
                source_end = ext_start - 1
    return int(source_start), int(source_end)


def _load_company_usage_counts(base_dir: Path) -> dict[int, int]:
    runtime_path = Path(base_dir) / "_step100_resume" / "checkpoint" / "latest" / "runtime_state.json"
    if not runtime_path.exists():
        return {}
    try:
        payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    raw = payload.get("company_film_count")
    if not isinstance(raw, list):
        return {}
    out: dict[int, int] = {}
    for item in raw:
        if not isinstance(item, list) or len(item) != 2:
            continue
        cid = _coerce_int(item[0], None)
        count = _coerce_int(item[1], 0)
        if cid is not None and cid > 0:
            out[int(cid)] = int(count or 0)
    return out


def load_lifecycle_artifact(base_dir: Path) -> dict[str, Any] | None:
    path = _artifact_path(base_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def update_lifecycle_artifact(
    base_dir: Path,
    *,
    extension_start_year: int | None,
    extension_end_year: int | None,
    company_lifecycle_policy: str | None = None,
    person_updates: list[dict[str, Any]] | None = None,
    company_updates: list[dict[str, Any]] | None = None,
    stats: dict[str, int] | None = None,
) -> Path:
    path = _artifact_path(base_dir)
    existing = load_lifecycle_artifact(base_dir) or {}
    same_window = (
        _coerce_int(existing.get("extension_start_year"), extension_start_year) == extension_start_year
        and _coerce_int(existing.get("extension_end_year"), extension_end_year) == extension_end_year
    )
    artifact: dict[str, Any] = {
        "version": 1,
        "updated_at": _now_iso(),
        "extension_start_year": extension_start_year,
        "extension_end_year": extension_end_year,
        "company_lifecycle_policy": company_lifecycle_policy
        or existing.get("company_lifecycle_policy")
        or "balanced",
        "person_updates": list(existing.get("person_updates", []) or []) if same_window else [],
        "company_updates": list(existing.get("company_updates", []) or []) if same_window else [],
        "stats": dict(existing.get("stats", {}) or {}) if same_window else {},
    }
    if person_updates is not None:
        if person_updates or not artifact["person_updates"]:
            artifact["person_updates"] = list(person_updates)
    if company_updates is not None:
        if company_updates or not artifact["company_updates"]:
            artifact["company_updates"] = list(company_updates)
    if stats:
        artifact["stats"].update({str(k): int(v) for k, v in stats.items()})
    artifact["stats"]["person_lifecycle_updates"] = len(artifact["person_updates"])
    artifact["stats"]["company_lifecycle_updates"] = len(artifact["company_updates"])
    artifact["stats"]["lifecycle_updates"] = (
        len(artifact["person_updates"]) + len(artifact["company_updates"])
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_json_ready(artifact), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def plan_person_survivors(
    persons: list[dict[str, Any]],
    *,
    seed: int,
    extension_start_year: int | None,
    extension_end_year: int | None,
    survivor_share: float,
) -> LifecyclePlanResult:
    if extension_start_year is None or extension_end_year is None or survivor_share <= 0:
        return LifecyclePlanResult(persons, [], {"survivor_extensions": 0})

    ext_start = int(extension_start_year)
    ext_end = int(extension_end_year)
    rng = np.random.RandomState(int(seed) + 87_003)
    eligible: list[dict[str, Any]] = []
    for person in persons:
        stage = str(person.get("career_stage", "") or "").casefold()
        if stage not in {"prime", "veteran", "legend", "rising"}:
            continue
        retire = _coerce_int(person.get("retirement_year"), None)
        if retire is None:
            continue
        if ext_start - 8 <= retire < ext_start:
            eligible.append(person)

    if not eligible:
        return LifecyclePlanResult(persons, [], {"survivor_extensions": 0})

    take = min(len(eligible), max(1, int(round(len(persons) * float(survivor_share)))))
    chosen = rng.choice(len(eligible), size=take, replace=False)
    updates: list[dict[str, Any]] = []
    for idx in chosen:
        person = eligible[int(idx)]
        pid = _row_id(person, "person_id")
        if pid is None:
            continue
        old_retire = _coerce_int(person.get("retirement_year"), ext_start - 1) or (ext_start - 1)
        new_retire = max(old_retire + 1, ext_end + int(rng.randint(0, 9)))
        new_retire = min(ext_end + 10, new_retire)
        person["retirement_year"] = int(new_retire)
        updates.append(
            {
                "person_id": int(pid),
                "retirement_year": int(new_retire),
                "old_retirement_year": int(old_retire),
                "reason": "near_boundary_survivor_extension",
            }
        )

    return LifecyclePlanResult(
        persons,
        updates,
        {"survivor_extensions": len(updates), "person_lifecycle_updates": len(updates)},
    )


def _old_company_founded_year(
    *,
    seed: int,
    tier: str,
    source_start_year: int,
    company_id: int,
) -> int:
    if tier in {"Global", "Major"}:
        lo = max(1880, int(source_start_year) - 55)
        hi = max(lo, int(source_start_year) - 4)
    else:
        lo = max(1900, int(source_start_year) - 25)
        hi = max(lo, int(source_start_year) - 1)
    offset = int(_stable_unit(int(seed), "old-founded", company_id) * (hi - lo + 1))
    return int(min(hi, lo + offset))


def _new_company_founded_year(
    rng: np.random.RandomState,
    *,
    extension_start_year: int,
    extension_end_year: int,
    policy: str,
) -> int:
    if int(extension_start_year) >= int(extension_end_year):
        return int(extension_start_year)
    span = max(1, int(extension_end_year) - int(extension_start_year))
    mode_fraction = _POLICY_RATES[policy]["founding_mode"]
    mode = int(extension_start_year) + max(1, int(round(span * mode_fraction)))
    value = rng.triangular(int(extension_start_year), mode, int(extension_end_year))
    return int(max(extension_start_year, min(extension_end_year, round(value))))


def plan_company_lifecycle(
    companies: list[dict[str, Any]],
    *,
    base_dir: Path,
    seed: int,
    extension_start_year: int | None,
    extension_end_year: int | None,
    company_lifecycle_policy: str = "balanced",
    new_company_ids: Iterable[int] | None = None,
) -> LifecyclePlanResult:
    if extension_start_year is None or extension_end_year is None:
        return LifecyclePlanResult(companies, [], {"company_lifecycle_updates": 0})
    policy = str(company_lifecycle_policy or "balanced")
    if policy not in VALID_COMPANY_LIFECYCLE_POLICIES:
        raise ValueError(f"Unknown company lifecycle policy: {policy}")

    ext_start = int(extension_start_year)
    ext_end = int(extension_end_year)
    source_start, source_end = _infer_source_years(base_dir, ext_start)
    rng = np.random.RandomState(int(seed) + 431_221)
    new_ids = {int(cid) for cid in list(new_company_ids or [])}
    rows_by_id = {cid: row for row in companies if (cid := _row_id(row, "company_id")) is not None}
    usage_counts = _load_company_usage_counts(base_dir)
    active_old_ids: list[int] = []
    old_ids: list[int] = []
    for cid, row in rows_by_id.items():
        if cid in new_ids:
            continue
        old_ids.append(cid)
        defunct = _coerce_int(row.get("defunct_year"), None)
        if defunct is None or defunct >= ext_start:
            active_old_ids.append(cid)

    positive_usage = sorted(
        ((cid, count) for cid, count in usage_counts.items() if cid in active_old_ids and int(count) > 0),
        key=lambda item: item[1],
        reverse=True,
    )
    high_usage_take = max(1, int(round(len(positive_usage) * 0.20))) if positive_usage else 0
    high_usage_ids = {cid for cid, _ in positive_usage[:high_usage_take]}

    core_old_ids: set[int] = set()
    non_core_active: list[int] = []
    for cid in active_old_ids:
        row = rows_by_id[cid]
        tier = _normalize_tier(row.get("tier"))
        if tier in {"Global", "Major"} or cid in high_usage_ids:
            core_old_ids.add(cid)
        else:
            non_core_active.append(cid)

    dissolve_target = int(round(len(non_core_active) * float(_POLICY_RATES[policy]["old_dissolve"])))
    dissolve_weights: list[float] = []
    for cid in non_core_active:
        row = rows_by_id[cid]
        tier = _normalize_tier(row.get("tier"))
        usage = float(usage_counts.get(cid, 0) or 0)
        pop = _row_pop_weight(row)
        weight = _DISSOLUTION_WEIGHT_BY_TIER.get(tier, 1.0) / ((1.0 + math.log1p(usage)) * (0.35 + pop))
        dissolve_weights.append(max(0.001, weight))
    dissolved_old_ids = _weighted_choice_without_replacement(
        rng,
        non_core_active,
        dissolve_weights,
        dissolve_target,
    )

    new_rows = [rows_by_id[cid] for cid in sorted(new_ids) if cid in rows_by_id]
    new_defunct_target = int(round(len(new_rows) * float(_POLICY_RATES[policy]["new_defunct"])))
    new_defunct_weights = [
        _DISSOLUTION_WEIGHT_BY_TIER.get(_normalize_tier(row.get("tier")), 1.0)
        for row in new_rows
    ]
    new_defunct_ids = _weighted_choice_without_replacement(
        rng,
        [_row_id(row, "company_id") or 0 for row in new_rows],
        new_defunct_weights,
        new_defunct_target,
    )

    dissolve_lo = min(ext_end, ext_start + 4)
    dissolve_hi = max(dissolve_lo, ext_end - 5)
    updates: list[dict[str, Any]] = []
    old_backfilled = 0
    old_core_preserved = 0
    company_dissolutions = 0
    new_founded = 0
    new_defunct = 0

    for cid in sorted(old_ids):
        row = rows_by_id[cid]
        tier = _normalize_tier(row.get("tier"))
        founded_old = _coerce_int(row.get("founded_year"), None)
        if founded_old is None:
            founded = _old_company_founded_year(seed=seed, tier=tier, source_start_year=source_start, company_id=cid)
            row["founded_year"] = int(founded)
            old_backfilled += 1
        else:
            founded = int(founded_old)
            row["founded_year"] = int(founded)

        defunct_old = _coerce_int(row.get("defunct_year"), None)
        reason = "old_preserved"
        defunct_value: int | None = defunct_old
        if cid in core_old_ids:
            defunct_value = None
            row["defunct_year"] = None
            old_core_preserved += 1
            reason = "old_core_preserved"
        elif cid in dissolved_old_ids:
            if dissolve_lo >= dissolve_hi:
                value = int(dissolve_lo)
            else:
                value = int(round(rng.triangular(dissolve_lo, dissolve_lo + (dissolve_hi - dissolve_lo) * 0.55, dissolve_hi)))
            defunct_value = max(int(founded), min(dissolve_hi, value))
            row["defunct_year"] = int(defunct_value)
            company_dissolutions += 1
            reason = "old_dissolved"
        elif defunct_old is None:
            row["defunct_year"] = None
        else:
            row["defunct_year"] = int(max(defunct_old, founded))
            defunct_value = int(row["defunct_year"])

        updates.append(
            {
                "company_id": int(cid),
                "founded_year": int(row["founded_year"]),
                "defunct_year": defunct_value,
                "old_defunct_year": defunct_old,
                "reason": reason,
            }
        )

    for row in new_rows:
        cid = _row_id(row, "company_id")
        if cid is None:
            continue
        founded = _new_company_founded_year(
            rng,
            extension_start_year=ext_start,
            extension_end_year=ext_end,
            policy=policy,
        )
        row["founded_year"] = int(founded)
        new_founded += 1
        defunct: int | None = None
        reason = "new_company_lifecycle"
        if cid in new_defunct_ids:
            defunct = int(founded + rng.randint(4, 26))
            if defunct <= ext_end + 10:
                defunct = max(int(founded), int(defunct))
                row["defunct_year"] = int(defunct)
                new_defunct += 1
                reason = "new_company_finite_lifecycle"
            else:
                defunct = None
                row["defunct_year"] = None
        else:
            row["defunct_year"] = None
        updates.append(
            {
                "company_id": int(cid),
                "founded_year": int(founded),
                "defunct_year": defunct,
                "old_defunct_year": None,
                "reason": reason,
            }
        )

    stats = {
        "source_start_year": int(source_start),
        "source_end_year": int(source_end),
        "old_companies": len(old_ids),
        "old_active_companies": len(active_old_ids),
        "old_core_preserved": old_core_preserved,
        "old_founded_backfilled": old_backfilled,
        "company_dissolutions": company_dissolutions,
        "new_founded": new_founded,
        "new_defunct": new_defunct,
        "company_lifecycle_updates": len(updates),
    }
    return LifecyclePlanResult(companies, updates, stats)


def _id_index(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    if frame is None or frame.empty or column not in frame.columns:
        return {}
    keys = pd.to_numeric(frame[column], errors="coerce")
    out: dict[str, Any] = {}
    for idx, value in keys.items():
        if pd.isna(value):
            continue
        out[str(int(value))] = idx
    return out


def _apply_dataframe_updates(
    frame: pd.DataFrame | None,
    *,
    id_column: str,
    updates: list[dict[str, Any]],
    fields: tuple[str, ...],
) -> tuple[pd.DataFrame | None, int]:
    if frame is None or frame.empty or not updates:
        return frame, 0
    if id_column not in frame.columns:
        return frame, 0
    out = frame.copy()
    index = _id_index(out, id_column)
    changed = 0
    for update in updates:
        entity_id = _coerce_int(update.get(id_column), None)
        if entity_id is None:
            continue
        idx = index.get(str(int(entity_id)))
        if idx is None:
            continue
        touched = False
        for field in fields:
            if field not in update:
                continue
            if field not in out.columns:
                out[field] = None
            value = update.get(field)
            if value is None or _is_missing_year(value):
                out.at[idx, field] = None
            else:
                out.at[idx, field] = int(value)
            touched = True
        if touched:
            changed += 1
    return out, changed


def refresh_world_after_lifecycle(world: Any) -> None:
    for attr in ("_year_cache", "_cast_year_cache", "_crew_year_pool_cache", "_company_by_tier_genre"):
        if hasattr(world, attr):
            try:
                setattr(world, attr, {})
            except Exception:
                pass
    if hasattr(world, "_build_lookup_dicts") and getattr(world, "persons", None) is not None and getattr(world, "companies", None) is not None:
        try:
            world._build_lookup_dicts()
        except Exception:
            pass
    if hasattr(world, "_build_person_role_views") and getattr(world, "person_roles", None) is not None and getattr(world, "persons", None) is not None:
        try:
            world._build_person_role_views()
        except Exception:
            pass
    if hasattr(world, "_prewarm_year_cache") and getattr(world, "actors", None) is not None:
        try:
            world._prewarm_year_cache()
        except Exception:
            pass


def apply_lifecycle_to_world(
    world: Any,
    base_dir: Path,
    *,
    extension_start_year: int | None = None,
    extension_end_year: int | None = None,
) -> dict[str, int]:
    artifact = load_lifecycle_artifact(base_dir)
    if not artifact:
        return {"applied_person_updates": 0, "applied_company_updates": 0}
    if extension_start_year is not None:
        artifact_start = _coerce_int(artifact.get("extension_start_year"), None)
        if artifact_start is not None and artifact_start != int(extension_start_year):
            raise RuntimeError(
                f"Lifecycle artifact start year {artifact_start} does not match extension start year {extension_start_year}."
            )
    if extension_end_year is not None:
        artifact_end = _coerce_int(artifact.get("extension_end_year"), None)
        if artifact_end is not None and artifact_end != int(extension_end_year):
            raise RuntimeError(
                f"Lifecycle artifact end year {artifact_end} does not match extension end year {extension_end_year}."
            )

    person_updates = [row for row in list(artifact.get("person_updates", []) or []) if isinstance(row, dict)]
    company_updates = [row for row in list(artifact.get("company_updates", []) or []) if isinstance(row, dict)]
    world.persons, person_count = _apply_dataframe_updates(
        getattr(world, "persons", None),
        id_column="person_id",
        updates=person_updates,
        fields=("retirement_year",),
    )
    world.companies, company_count = _apply_dataframe_updates(
        getattr(world, "companies", None),
        id_column="company_id",
        updates=company_updates,
        fields=("founded_year", "defunct_year"),
    )
    if person_count or company_count:
        refresh_world_after_lifecycle(world)
    return {
        "applied_person_updates": int(person_count),
        "applied_company_updates": int(company_count),
    }
