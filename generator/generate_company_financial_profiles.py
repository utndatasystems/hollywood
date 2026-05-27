"""
Generate baseline company financial profiles for movie generation.

This is a fresh-bootstrap generator, not a post-run analytics pass.
It derives stable finance signals from company tier, specialties, and
optionally company latent variables so WorldState/financials.py can use
them during movie generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import re

import numpy as np
import pandas as pd

from bootstrap_artifacts import (
    audit_artifact_usage,
    audit_fallback_hit,
    load_modeling_priors_artifact,
    require_payload_value,
)
from policy_runtime import modeling_priors_path

BASE_DIR = Path(__file__).resolve().parent
ENTITY_DIR = BASE_DIR / "entities"


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _stable_uniform(*parts: object) -> float:
    raw = "|".join(str(part) for part in parts).encode("utf-8", errors="ignore")
    digest = hashlib.blake2b(raw, digest_size=8).digest()
    return int.from_bytes(digest, "big") / float((1 << 64) - 1)


def _safe_float(value: object, default: float) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _load_company_latent_map(entities_dir: Path) -> dict[int, dict]:
    path = entities_dir / "companies_latent.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, list):
        return {}
    out: dict[int, dict] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        try:
            cid = int(row.get("company_id", 0))
        except Exception:
            continue
        if cid > 0:
            out[cid] = row
    return out


DEBUG_COMPANY_PROFILE_TIERS = {
    "Global": {"capital": 0.88, "margin": 0.24, "debt": 0.44, "slate": 0.88, "buffer": 0.74, "growth": 0.44, "eff": 0.70},
    "Major": {"capital": 0.76, "margin": 0.21, "debt": 0.47, "slate": 0.74, "buffer": 0.62, "growth": 0.48, "eff": 0.63},
    "Mid-Budget": {"capital": 0.58, "margin": 0.18, "debt": 0.43, "slate": 0.54, "buffer": 0.50, "growth": 0.54, "eff": 0.56},
    "Indie": {"capital": 0.40, "margin": 0.15, "debt": 0.35, "slate": 0.34, "buffer": 0.40, "growth": 0.58, "eff": 0.51},
    "Micro": {"capital": 0.26, "margin": 0.11, "debt": 0.29, "slate": 0.22, "buffer": 0.30, "growth": 0.52, "eff": 0.45},
}

DEBUG_COMPANY_PROFILE_COEFFICIENTS = {
    "budget_focus_weights": [0.18, 0.34, 0.56, 0.78, 1.00],
    "capital_prestige_weight": 0.18,
    "capital_focus_weight": 0.12,
    "capital_focus_penalty_weight": 0.05,
    "margin_prestige_weight": 0.08,
    "margin_risk_discipline_weight": 0.05,
    "margin_controversy_discipline_weight": 0.03,
    "debt_inverse_capital_weight": 0.12,
    "debt_risk_weight": 0.10,
    "debt_margin_penalty_weight": 0.04,
    "debt_focus_penalty_weight": 0.03,
    "slate_focus_weight": 0.18,
    "slate_trend_weight": 0.10,
    "slate_prestige_weight": 0.08,
    "slate_focus_penalty_weight": 0.06,
    "buffer_capital_weight": 0.18,
    "buffer_margin_weight": 0.10,
    "buffer_debt_penalty_weight": 0.12,
    "buffer_risk_penalty_weight": 0.08,
    "growth_trend_weight": 0.14,
    "growth_risk_weight": 0.08,
    "growth_prestige_weight": 0.06,
    "growth_focus_penalty_weight": 0.05,
    "eff_margin_weight": 0.10,
    "eff_prestige_weight": 0.08,
    "eff_low_debt_weight": 0.06,
    "eff_focus_weight": 0.05,
    "eff_focus_penalty_weight": 0.03,
    "epsilon_span": 0.08,
    "margin_epsilon_scale": 0.50,
    "debt_epsilon_scale": 0.40,
    "capital_min": 0.12,
    "capital_max": 0.98,
    "margin_min": 0.04,
    "margin_max": 0.45,
    "debt_min": 0.08,
    "debt_max": 0.88,
    "slate_min": 0.08,
    "slate_max": 0.99,
    "buffer_min": 0.05,
    "buffer_max": 0.95,
    "growth_min": 0.04,
    "growth_max": 0.96,
    "eff_min": 0.08,
    "eff_max": 0.95,
}


def _company_profile_priors(base_dir: Path, mode: str) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    artifact_path = modeling_priors_path(base_dir)
    payload = load_modeling_priors_artifact(base_dir, mode=mode)
    financial_priors = payload.get("financial_priors", {}) if isinstance(payload, dict) else {}
    raw_tiers = require_payload_value(
        financial_priors,
        "company_profile_tiers",
        artifact_label="modeling_priors.json",
        artifact_path=artifact_path,
        mode=mode,
        validator=lambda value: isinstance(value, dict) and len(value) > 0,
        detail="financial_priors.company_profile_tiers is required for research-mode company finance profiles",
    )
    raw_coeffs = require_payload_value(
        financial_priors,
        "company_profile_coefficients",
        artifact_label="modeling_priors.json",
        artifact_path=artifact_path,
        mode=mode,
        validator=lambda value: isinstance(value, dict) and len(value) > 0,
        detail="financial_priors.company_profile_coefficients is required for research-mode company finance profiles",
    )
    if raw_tiers is None or raw_coeffs is None:
        audit_fallback_hit("company_financial_profiles", "debug_company_profile_priors", detail="using built-in company profile finance priors", mode=mode)
        return dict(DEBUG_COMPANY_PROFILE_TIERS), dict(DEBUG_COMPANY_PROFILE_COEFFICIENTS)
    normalized_tiers: dict[str, dict[str, float]] = {}
    for tier_name, values in raw_tiers.items():
        if not isinstance(values, dict):
            continue
        row: dict[str, float] = {}
        for key in ("capital", "margin", "debt", "slate", "buffer", "growth", "eff"):
            try:
                row[key] = float(values[key])
            except Exception:
                row[key] = float(DEBUG_COMPANY_PROFILE_TIERS.get(str(tier_name), DEBUG_COMPANY_PROFILE_TIERS["Mid-Budget"]).get(key, 0.5))
        normalized_tiers[str(tier_name)] = row
    if mode == "research":
        missing_tiers = [tier for tier in DEBUG_COMPANY_PROFILE_TIERS if tier not in normalized_tiers]
        if missing_tiers:
            audit_fallback_hit(
                "modeling_priors.json",
                "company_profile_tiers_missing_required_company_tiers",
                detail=", ".join(missing_tiers),
                mode=mode,
            )
    coeffs = dict(DEBUG_COMPANY_PROFILE_COEFFICIENTS)
    for key, value in dict(raw_coeffs).items():
        if key == "budget_focus_weights" and isinstance(value, list) and len(value) == 5:
            try:
                coeffs[key] = [float(v) for v in value]
            except Exception:
                continue
            continue
        try:
            coeffs[str(key)] = float(value)
        except Exception:
            continue
    audit_artifact_usage(
        "modeling_priors.json",
        artifact_path,
        sections=["financial_priors.company_profile_tiers", "financial_priors.company_profile_coefficients"],
    )
    return normalized_tiers or dict(DEBUG_COMPANY_PROFILE_TIERS), coeffs


def synthesize_company_financial_profiles(base_dir: Path, *, mode: str = "research") -> Path:
    entities_dir = base_dir / "entities"
    company_csv = entities_dir / "company.csv"
    if not company_csv.exists():
        raise FileNotFoundError(f"company.csv not found at {company_csv}")

    cdf = pd.read_csv(company_csv, low_memory=False)
    if cdf.empty or "company_id" not in cdf.columns:
        raise RuntimeError("company.csv is empty or missing company_id")

    cdf["company_id"] = pd.to_numeric(cdf["company_id"], errors="coerce").fillna(0).astype(int)
    cdf = cdf[cdf["company_id"] > 0].copy()
    if cdf.empty:
        raise RuntimeError("company.csv contains no valid company rows")

    latent_map = _load_company_latent_map(entities_dir)

    tier_prior, coeffs = _company_profile_priors(base_dir, str(mode))
    budget_weights = np.asarray(coeffs.get("budget_focus_weights", DEBUG_COMPANY_PROFILE_COEFFICIENTS["budget_focus_weights"]), dtype=float)

    rows: list[dict] = []
    for rec in cdf.itertuples(index=False):
        cid = int(getattr(rec, "company_id"))
        tier = str(getattr(rec, "tier", "Mid-Budget") or "Mid-Budget")
        base = tier_prior.get(tier, tier_prior["Mid-Budget"])

        spec_raw = str(getattr(rec, "specialty_genres", "") or "")
        spec_cnt = max(1, len([chunk for chunk in re.split(r"[;,|]", spec_raw) if str(chunk).strip()]))
        focus_pen = min(0.25, 0.05 * max(0, spec_cnt - 3))

        latent = latent_map.get(cid, {})
        prestige = _safe_float(latent.get("prestige_score"), _safe_float(getattr(rec, "pop_weight", 0.50), 0.50))
        risk_appetite = _safe_float(latent.get("risk_appetite"), 0.50)
        controversy_tol = _safe_float(latent.get("controversy_tolerance"), 0.50)
        trend = _safe_float(latent.get("market_trend_sensitivity"), 0.50)
        budget_focus = latent.get("budget_tier_focus", [0.5] * 5)
        if not isinstance(budget_focus, list):
            budget_focus = [0.5] * 5
        if len(budget_focus) != 5:
            budget_focus = (list(budget_focus) + [0.5] * 5)[:5]
        budget_focus_arr = np.clip(np.asarray(budget_focus, dtype=float), 0.0, 1.0)
        focus_strength = float(np.dot(budget_focus_arr, budget_weights) / float(budget_weights.sum()))

        eps = (_stable_uniform("finance-profile", cid) - 0.5) * float(coeffs.get("epsilon_span", 0.08))

        capital = np.clip(
            base["capital"]
            + float(coeffs.get("capital_prestige_weight", 0.18)) * prestige
            + float(coeffs.get("capital_focus_weight", 0.12)) * focus_strength
            - float(coeffs.get("capital_focus_penalty_weight", 0.05)) * focus_pen
            + eps,
            float(coeffs.get("capital_min", 0.12)),
            float(coeffs.get("capital_max", 0.98)),
        )
        margin = np.clip(
            base["margin"]
            + float(coeffs.get("margin_prestige_weight", 0.08)) * prestige
            + float(coeffs.get("margin_risk_discipline_weight", 0.05)) * (1.0 - risk_appetite)
            + float(coeffs.get("margin_controversy_discipline_weight", 0.03)) * (1.0 - controversy_tol)
            + eps * float(coeffs.get("margin_epsilon_scale", 0.50)),
            float(coeffs.get("margin_min", 0.04)),
            float(coeffs.get("margin_max", 0.45)),
        )
        debt = np.clip(
            base["debt"]
            + float(coeffs.get("debt_inverse_capital_weight", 0.12)) * (1.0 - capital)
            + float(coeffs.get("debt_risk_weight", 0.10)) * risk_appetite
            - float(coeffs.get("debt_margin_penalty_weight", 0.04)) * margin
            + float(coeffs.get("debt_focus_penalty_weight", 0.03)) * focus_pen
            + eps * float(coeffs.get("debt_epsilon_scale", 0.40)),
            float(coeffs.get("debt_min", 0.08)),
            float(coeffs.get("debt_max", 0.88)),
        )
        slate = np.clip(
            base["slate"]
            + float(coeffs.get("slate_focus_weight", 0.18)) * focus_strength
            + float(coeffs.get("slate_trend_weight", 0.10)) * trend
            + float(coeffs.get("slate_prestige_weight", 0.08)) * prestige
            - float(coeffs.get("slate_focus_penalty_weight", 0.06)) * focus_pen
            + eps,
            float(coeffs.get("slate_min", 0.08)),
            float(coeffs.get("slate_max", 0.99)),
        )
        risk_buffer = np.clip(
            base["buffer"]
            + float(coeffs.get("buffer_capital_weight", 0.18)) * capital
            + float(coeffs.get("buffer_margin_weight", 0.10)) * margin
            - float(coeffs.get("buffer_debt_penalty_weight", 0.12)) * debt
            - float(coeffs.get("buffer_risk_penalty_weight", 0.08)) * risk_appetite
            + eps,
            float(coeffs.get("buffer_min", 0.05)),
            float(coeffs.get("buffer_max", 0.95)),
        )
        growth = np.clip(
            base["growth"]
            + float(coeffs.get("growth_trend_weight", 0.14)) * trend
            + float(coeffs.get("growth_risk_weight", 0.08)) * risk_appetite
            + float(coeffs.get("growth_prestige_weight", 0.06)) * prestige
            - float(coeffs.get("growth_focus_penalty_weight", 0.05)) * focus_pen
            + eps,
            float(coeffs.get("growth_min", 0.04)),
            float(coeffs.get("growth_max", 0.96)),
        )
        efficiency = np.clip(
            base["eff"]
            + float(coeffs.get("eff_margin_weight", 0.10)) * margin
            + float(coeffs.get("eff_prestige_weight", 0.08)) * prestige
            + float(coeffs.get("eff_low_debt_weight", 0.06)) * (1.0 - debt)
            + float(coeffs.get("eff_focus_weight", 0.05)) * focus_strength
            - float(coeffs.get("eff_focus_penalty_weight", 0.03)) * focus_pen
            + eps,
            float(coeffs.get("eff_min", 0.08)),
            float(coeffs.get("eff_max", 0.95)),
        )

        rows.append({
            "company_id": cid,
            "capital_score": round(float(capital), 4),
            "operating_margin": round(float(margin), 4),
            "debt_ratio": round(float(debt), 4),
            "slate_capacity": round(float(slate), 4),
            "risk_buffer": round(float(risk_buffer), 4),
            "growth_bias": round(float(growth), 4),
            "revenue_efficiency": round(float(efficiency), 4),
            "profile_bucket": tier,
            "updated_at": _now_utc(),
        })

    out = pd.DataFrame(rows).sort_values("company_id")
    out_path = entities_dir / "company_financial_profile.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved company_financial_profile.csv ({len(out):,} rows)")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate baseline company financial profiles")
    parser.add_argument("--base-dir", default=str(BASE_DIR))
    parser.add_argument("--mode", choices=("research", "debug"), default="research")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    synthesize_company_financial_profiles(base_dir, mode=str(args.mode))


if __name__ == "__main__":
    main()
