from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from pipeline_runtime import pipeline_mode, year_bounds_from_env
from policy_runtime import (
    character_identity_bank_path,
    company_lexicon_path,
    identity_bank_path,
    keyword_seed_bank_path,
    modeling_priors_path,
    research_audit_path,
    temporal_regime_plan_path,
    title_grammar_bank_path,
)


class ArtifactRequiredError(RuntimeError):
    pass


class ResearchFallbackError(RuntimeError):
    pass


_AUDIT_PRIMARY_ENV = "DATA_SYS_RESEARCH_AUDIT"
_AUDIT_LATEST_ENV = "DATA_SYS_RESEARCH_AUDIT_LATEST"
_MISSING = object()


def current_mode(explicit_mode: str | None = None) -> str:
    mode = str(explicit_mode or os.getenv("DATA_SYS_PIPELINE_MODE") or pipeline_mode()).strip().lower()
    return "debug" if mode == "debug" else "research"


def debug_mode(explicit_mode: str | None = None) -> bool:
    return current_mode(explicit_mode) == "debug"


def dedupe_strings(values: Iterable[object]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        item = str(value or "").strip()
        key = item.casefold()
        if not item or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def load_json_artifact(path: str | Path) -> dict[str, Any] | None:
    target = Path(path)
    if not target.exists():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _audit_paths() -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for raw in (os.getenv(_AUDIT_PRIMARY_ENV), os.getenv(_AUDIT_LATEST_ENV)):
        if not raw:
            continue
        text = str(raw).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(Path(text))
    if not out:
        fallback = research_audit_path(Path(__file__).resolve().parent)
        out.append(fallback)
    return out


def _audit_load() -> dict[str, Any]:
    for path in _audit_paths():
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _audit_write(payload: dict[str, Any]) -> None:
    for path in _audit_paths():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            continue


def _normalize_step_bucket(audit: dict[str, Any]) -> dict[str, Any]:
    steps = audit.get("steps")
    if not isinstance(steps, dict):
        steps = {}
        audit["steps"] = steps
    step_id = str(os.getenv("DATA_SYS_STEP_ID", "") or "").strip() or "unknown"
    bucket = steps.get(step_id)
    if not isinstance(bucket, dict):
        bucket = {"step_id": step_id}
        steps[step_id] = bucket
    bucket.setdefault("step_name", str(os.getenv("DATA_SYS_STEP_NAME", "") or "").strip())
    bucket.setdefault("artifacts", {})
    bucket.setdefault("fallback_hits", [])
    bucket.setdefault("notes", [])
    return bucket


def _artifact_sha256(path: str | Path) -> str:
    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _touch_audit_root() -> dict[str, Any]:
    audit = _audit_load()
    audit.setdefault("run_id", str(os.getenv("DATA_SYS_RUN_ID", "") or "").strip())
    audit.setdefault("mode", current_mode())
    audit.setdefault("started_at", datetime.now().isoformat())
    audit["updated_at"] = datetime.now().isoformat()
    audit.setdefault("fallback_hit_count", 0)
    audit.setdefault("fallback_hits", [])
    audit.setdefault("artifacts", {})
    audit.setdefault("critic", {})
    audit.setdefault("steps", {})
    return audit


def audit_note(message: str) -> None:
    audit = _touch_audit_root()
    bucket = _normalize_step_bucket(audit)
    notes = bucket.get("notes")
    if not isinstance(notes, list):
        notes = []
        bucket["notes"] = notes
    notes.append({"timestamp": datetime.now().isoformat(), "message": str(message)})
    _audit_write(audit)


def audit_artifact_usage(
    label: str,
    path: str | Path,
    *,
    sections: Iterable[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    target = Path(path)
    if not target.exists():
        return
    audit = _touch_audit_root()
    artifacts = audit.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
        audit["artifacts"] = artifacts
    step_bucket = _normalize_step_bucket(audit)
    step_artifacts = step_bucket.get("artifacts")
    if not isinstance(step_artifacts, dict):
        step_artifacts = {}
        step_bucket["artifacts"] = step_artifacts
    section_values = dedupe_strings(sections or [])
    artifact_row = artifacts.get(str(label))
    if not isinstance(artifact_row, dict):
        artifact_row = {
            "label": str(label),
            "path": str(target),
            "sha256": _artifact_sha256(target),
            "consumed_steps": [],
            "sections_used": [],
        }
        artifacts[str(label)] = artifact_row
    else:
        artifact_row["path"] = str(target)
        artifact_row.setdefault("sha256", _artifact_sha256(target))
        artifact_row.setdefault("consumed_steps", [])
        artifact_row.setdefault("sections_used", [])
    consumed_steps = dedupe_strings(list(artifact_row.get("consumed_steps", [])) + [str(step_bucket.get("step_id", ""))])
    artifact_row["consumed_steps"] = consumed_steps
    artifact_row["sections_used"] = dedupe_strings(list(artifact_row.get("sections_used", [])) + section_values)
    if isinstance(extra, dict) and extra:
        artifact_row.update(extra)
    step_row = step_artifacts.get(str(label))
    if not isinstance(step_row, dict):
        step_row = {"path": str(target)}
    step_row["sha256"] = str(artifact_row.get("sha256", ""))
    step_row["sections_used"] = section_values
    if isinstance(extra, dict) and extra:
        step_row.update(extra)
    step_artifacts[str(label)] = step_row
    _audit_write(audit)


def audit_fallback_hit(
    component: str,
    reason: str,
    *,
    detail: str | None = None,
    mode: str | None = None,
) -> None:
    audit = _touch_audit_root()
    audit["fallback_hit_count"] = int(audit.get("fallback_hit_count", 0) or 0) + 1
    row = {
        "timestamp": datetime.now().isoformat(),
        "step_id": str(os.getenv("DATA_SYS_STEP_ID", "") or "").strip(),
        "step_name": str(os.getenv("DATA_SYS_STEP_NAME", "") or "").strip(),
        "component": str(component),
        "reason": str(reason),
    }
    if detail:
        row["detail"] = str(detail)
    fallback_hits = audit.get("fallback_hits")
    if not isinstance(fallback_hits, list):
        fallback_hits = []
        audit["fallback_hits"] = fallback_hits
    fallback_hits.append(row)
    bucket = _normalize_step_bucket(audit)
    step_hits = bucket.get("fallback_hits")
    if not isinstance(step_hits, list):
        step_hits = []
        bucket["fallback_hits"] = step_hits
    step_hits.append(row)
    _audit_write(audit)
    if current_mode(mode) == "research":
        raise ResearchFallbackError(f"{component} used authored fallback/default in research mode: {reason}")


def audit_step_status(*, ok: bool | None = None, enabled: bool | None = None, skipped: bool | None = None) -> None:
    audit = _touch_audit_root()
    bucket = _normalize_step_bucket(audit)
    if ok is not None:
        bucket["ok"] = bool(ok)
    if enabled is not None:
        bucket["enabled"] = bool(enabled)
    if skipped is not None:
        bucket["skipped"] = bool(skipped)
    _audit_write(audit)


def audit_critic_report(report: dict[str, Any] | None) -> None:
    audit = _touch_audit_root()
    critic = audit.get("critic")
    if not isinstance(critic, dict):
        critic = {}
        audit["critic"] = critic
    payload = report if isinstance(report, dict) else {}
    critic["enabled"] = True
    critic["status"] = str(payload.get("status", "") or critic.get("status", ""))
    critic["applied"] = int(payload.get("applied", 0) or 0)
    critic["skipped"] = int(payload.get("skipped", 0) or 0)
    critic["actions_proposed"] = int(payload.get("actions_proposed", 0) or 0)
    critic["sampled_titles"] = list(payload.get("sampled_titles", []) or [])
    critic["flagged_titles"] = list(payload.get("flagged_titles", []) or [])
    critic["cache_hit"] = bool(payload.get("cache_hit", False))
    llm_rewrite_actions = 0
    deterministic_sanitation_actions = 0
    repair_types = {}
    repair_reasons = {}
    duplicate_tagline_rewrite_actions = 0
    repairs = payload.get("repairs")
    if isinstance(repairs, list):
        for row in repairs:
            if not isinstance(row, dict):
                continue
            repair_type = str(row.get("repair_type", "") or "").strip()
            if not repair_type:
                continue
            repair_types[repair_type] = int(repair_types.get(repair_type, 0)) + 1
            repair_reason = str(row.get("repair_reason", "") or "").strip()
            if repair_reason:
                repair_reasons[repair_reason] = int(repair_reasons.get(repair_reason, 0)) + 1
            if repair_type in {"rewrite_plot_summary", "rewrite_tagline"}:
                llm_rewrite_actions += 1
            else:
                deterministic_sanitation_actions += 1
            if repair_type == "rewrite_tagline" and repair_reason == "duplicate_tagline_cluster":
                duplicate_tagline_rewrite_actions += 1
    elif isinstance(payload.get("repair_types"), dict):
        for key, value in payload.get("repair_types", {}).items():
            try:
                repair_types[str(key)] = int(value)
            except Exception:
                continue
        if isinstance(payload.get("repair_reasons"), dict):
            for key, value in payload.get("repair_reasons", {}).items():
                try:
                    repair_reasons[str(key)] = int(value)
                except Exception:
                    continue
        llm_rewrite_actions = int(payload.get("llm_rewrite_actions", 0) or 0)
        deterministic_sanitation_actions = int(payload.get("deterministic_sanitation_actions", 0) or 0)
        duplicate_tagline_rewrite_actions = int(payload.get("duplicate_tagline_rewrite_actions", 0) or 0)
    critic["llm_rewrite_actions"] = int(llm_rewrite_actions)
    critic["deterministic_sanitation_actions"] = int(deterministic_sanitation_actions)
    critic["repair_types"] = repair_types
    critic["repair_reasons"] = repair_reasons
    critic["duplicate_tagline_rewrite_actions"] = int(duplicate_tagline_rewrite_actions)
    _audit_write(audit)


def audit_payload_section(
    artifact_label: str,
    artifact_path: str | Path,
    section_name: str,
) -> None:
    audit_artifact_usage(artifact_label, artifact_path, sections=[section_name])


def audit_manifest() -> dict[str, Any]:
    return _audit_load()


def audit_fallback_hit_count() -> int:
    audit = _audit_load()
    return int(audit.get("fallback_hit_count", 0) or 0)


def initialize_research_audit(
    *,
    run_id: str,
    mode: str,
    start_year: int | None = None,
    end_year: int | None = None,
    model_override: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    audit = _touch_audit_root()
    audit["run_id"] = str(run_id or audit.get("run_id", "") or "").strip()
    audit["mode"] = current_mode(mode)
    audit.setdefault("started_at", datetime.now().isoformat())
    audit["updated_at"] = datetime.now().isoformat()
    audit["year_span"] = {
        "start_year": int(start_year) if start_year is not None else None,
        "end_year": int(end_year) if end_year is not None else None,
    }
    pipeline = audit.get("pipeline")
    if not isinstance(pipeline, dict):
        pipeline = {}
        audit["pipeline"] = pipeline
    pipeline["mode"] = audit["mode"]
    pipeline["model_override"] = str(model_override or "").strip() or None
    if isinstance(metadata, dict) and metadata:
        pipeline.update(metadata)
    _audit_write(audit)
    return audit


def record_pipeline_step(
    step_id: int | str,
    step_name: str,
    *,
    status: str,
    ok: bool | None = None,
    skipped: bool | None = None,
    enabled: bool | None = None,
    elapsed_seconds: float | None = None,
    detail: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    audit = _touch_audit_root()
    steps = audit.get("steps")
    if not isinstance(steps, dict):
        steps = {}
        audit["steps"] = steps
    sid = str(step_id)
    bucket = steps.get(sid)
    if not isinstance(bucket, dict):
        bucket = {"step_id": sid}
        steps[sid] = bucket
    bucket["step_name"] = str(step_name or "").strip()
    bucket["status"] = str(status or "").strip() or bucket.get("status", "")
    bucket["updated_at"] = datetime.now().isoformat()
    bucket.setdefault("artifacts", {})
    bucket.setdefault("fallback_hits", [])
    bucket.setdefault("notes", [])
    if ok is not None:
        bucket["ok"] = bool(ok)
    if skipped is not None:
        bucket["skipped"] = bool(skipped)
    if enabled is not None:
        bucket["enabled"] = bool(enabled)
    if elapsed_seconds is not None:
        bucket["elapsed_seconds"] = round(float(elapsed_seconds), 3)
    if detail:
        notes = bucket.get("notes")
        if not isinstance(notes, list):
            notes = []
            bucket["notes"] = notes
        notes.append({"timestamp": datetime.now().isoformat(), "message": str(detail)})
    if isinstance(extra, dict) and extra:
        bucket.update(extra)
    _audit_write(audit)


def clear_step_fallback_hits(step_id: int | str) -> None:
    audit = _touch_audit_root()
    steps = audit.get("steps")
    if not isinstance(steps, dict):
        return
    sid = str(step_id)
    bucket = steps.get(sid)
    if not isinstance(bucket, dict):
        return
    previous = bucket.get("fallback_hits")
    if not isinstance(previous, list) or not previous:
        return
    removed = 0
    root_hits = audit.get("fallback_hits")
    if not isinstance(root_hits, list):
        root_hits = []
        audit["fallback_hits"] = root_hits
    step_entries = []
    for row in previous:
        if isinstance(row, dict):
            step_entries.append(
                (
                    str(row.get("timestamp", "")),
                    str(row.get("component", "")),
                    str(row.get("reason", "")),
                    str(row.get("detail", "")),
                )
            )
    if step_entries:
        remaining: list[Any] = []
        for row in root_hits:
            if not isinstance(row, dict):
                remaining.append(row)
                continue
            signature = (
                str(row.get("timestamp", "")),
                str(row.get("component", "")),
                str(row.get("reason", "")),
                str(row.get("detail", "")),
            )
            if signature in step_entries:
                removed += 1
                continue
            remaining.append(row)
        audit["fallback_hits"] = remaining
    else:
        removed = len(previous)
    bucket["fallback_hits"] = []
    audit["fallback_hit_count"] = max(0, int(audit.get("fallback_hit_count", 0) or 0) - max(removed, len(previous)))
    _audit_write(audit)


def assert_no_recorded_fallbacks(mode: str | None = None) -> None:
    if current_mode(mode) != "research":
        return
    count = audit_fallback_hit_count()
    if count > 0:
        raise ResearchFallbackError(f"research mode recorded {count} fallback/default hits")


def payload_at(payload: Any, dotted_path: str, default: Any = _MISSING) -> Any:
    node = payload
    for part in str(dotted_path).split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
            continue
        return default
    return node


def require_payload_value(
    payload: dict[str, Any] | None,
    dotted_path: str,
    *,
    artifact_label: str,
    artifact_path: str | Path,
    mode: str | None = None,
    validator: callable | None = None,
    detail: str | None = None,
) -> Any:
    value = payload_at(payload or {}, dotted_path)
    ok = value is not _MISSING
    if ok and validator is not None:
        try:
            ok = bool(validator(value))
        except Exception:
            ok = False
    if not ok:
        reason = f"missing_or_invalid:{dotted_path}"
        audit_fallback_hit(artifact_label, reason, detail=detail or f"{artifact_label} requires {dotted_path}", mode=mode)
        if current_mode(mode) == "research":
            raise ArtifactRequiredError(f"{artifact_label} missing required field {dotted_path}")
        return None
    audit_payload_section(artifact_label, artifact_path, dotted_path)
    return value


def require_artifact(
    path: str | Path,
    *,
    label: str,
    mode: str | None = None,
) -> dict[str, Any] | None:
    payload = load_json_artifact(path)
    if payload is not None:
        return payload
    if current_mode(mode) == "research":
        raise ArtifactRequiredError(f"{label} is required in research mode but missing at {Path(path)}")
    return None


def load_identity_bank(base_dir: str | Path, mode: str | None = None) -> dict[str, Any] | None:
    return require_artifact(identity_bank_path(base_dir), label="identity_bank.json", mode=mode)


def load_character_identity_bank(base_dir: str | Path, mode: str | None = None) -> dict[str, Any] | None:
    return require_artifact(character_identity_bank_path(base_dir), label="character_identity_bank.json", mode=mode)


def load_company_lexicon(base_dir: str | Path, mode: str | None = None) -> dict[str, Any] | None:
    return require_artifact(company_lexicon_path(base_dir), label="company_lexicon.json", mode=mode)


def load_keyword_seed_bank(base_dir: str | Path, mode: str | None = None) -> dict[str, Any] | None:
    return require_artifact(keyword_seed_bank_path(base_dir), label="keyword_seed_bank.json", mode=mode)


def load_title_grammar_bank(base_dir: str | Path, mode: str | None = None) -> dict[str, Any] | None:
    return require_artifact(title_grammar_bank_path(base_dir), label="title_grammar_bank.json", mode=mode)


def load_temporal_regime_plan(base_dir: str | Path, mode: str | None = None) -> dict[str, Any] | None:
    return require_artifact(temporal_regime_plan_path(base_dir), label="temporal_regime_plan.json", mode=mode)


def load_modeling_priors_artifact(base_dir: str | Path, mode: str | None = None) -> dict[str, Any] | None:
    return require_artifact(modeling_priors_path(base_dir), label="modeling_priors.json", mode=mode)


def modeling_priors_sections(payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, dict):
        return {}
    return {
        str(key): value
        for key, value in payload.items()
        if isinstance(value, dict)
    }


def flatten_modeling_priors(payload: dict[str, Any] | None) -> dict[str, Any]:
    flat: dict[str, Any] = {}

    def _walk(node: dict[str, Any]) -> None:
        for key, value in node.items():
            if key == "meta":
                continue
            if isinstance(value, dict):
                _walk(value)
            else:
                flat.setdefault(str(key), value)

    if isinstance(payload, dict):
        _walk(payload)
    return flat


def prior_section(payload: dict[str, Any] | None, section: str) -> dict[str, Any]:
    sections = modeling_priors_sections(payload)
    row = sections.get(str(section), {})
    return row if isinstance(row, dict) else {}


def prior_float_from_section(
    payload: dict[str, Any] | None,
    section: str,
    key: str,
    default: float,
    *,
    lo: float | None = None,
    hi: float | None = None,
) -> float:
    value = prior_section(payload, section).get(key, default)
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if lo is not None:
        out = max(float(lo), out)
    if hi is not None:
        out = min(float(hi), out)
    return float(out)


def effective_year_bounds(
    base_dir: str | Path | None = None,
    *,
    fallback_start: int = 1950,
    fallback_end: int = 2025,
    mode: str | None = None,
) -> tuple[int, int]:
    start_year, end_year = year_bounds_from_env(fallback_start=fallback_start, fallback_end=fallback_end)
    if base_dir is not None:
        temporal = load_temporal_regime_plan(base_dir, mode="debug" if debug_mode(mode) else mode)
        if temporal:
            try:
                start_year = int(temporal.get("start_year", start_year))
                end_year = int(temporal.get("end_year", end_year))
            except Exception:
                pass
    if end_year < start_year:
        start_year, end_year = end_year, start_year
    return int(start_year), int(end_year)


def year_weight_map(
    temporal_plan: dict[str, Any] | None,
    *,
    start_year: int,
    end_year: int,
) -> dict[int, float]:
    years = list(range(int(start_year), int(end_year) + 1))
    if not years:
        return {}
    if not temporal_plan:
        return {year: 1.0 for year in years}
    raw_weights = temporal_plan.get("year_weights")
    weights: dict[int, float] = {}
    if isinstance(raw_weights, list):
        for row in raw_weights:
            if not isinstance(row, dict):
                continue
            try:
                year = int(row.get("year"))
                weight = float(row.get("weight", 0.0))
            except Exception:
                continue
            if start_year <= year <= end_year and weight > 0:
                weights[year] = weight
    if not weights:
        curve = temporal_plan.get("curve")
        phases = temporal_plan.get("phases")
        if isinstance(curve, dict):
            peaks = [float(v) for v in curve.get("phase_bias", {}).values() if isinstance(v, (int, float))]
            base = max(0.1, float(curve.get("base_weight", 1.0)))
            for idx, year in enumerate(years):
                frac = 0.0 if len(years) == 1 else idx / float(len(years) - 1)
                weights[year] = base + 0.25 * math.sin(2.0 * math.pi * frac) + 0.15 * math.sin(5.0 * math.pi * frac)
            if peaks:
                peak_boost = max(peaks)
                for year in years:
                    weights[year] = max(0.05, weights[year] + 0.08 * peak_boost)
        elif isinstance(phases, list) and phases:
            for idx, year in enumerate(years):
                frac = 0.0 if len(years) == 1 else idx / float(len(years) - 1)
                phase_idx = min(len(phases) - 1, int(frac * len(phases)))
                phase = phases[phase_idx] if isinstance(phases[phase_idx], dict) else {}
                weights[year] = max(0.05, float(phase.get("movie_density", 1.0)))
    if not weights:
        weights = {year: 1.0 for year in years}
    for year in years:
        weights.setdefault(year, min(weights.values()) if weights else 1.0)
    return {int(year): max(0.01, float(weight)) for year, weight in weights.items()}
