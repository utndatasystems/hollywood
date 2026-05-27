"""Central model defaults for Hollywood LLM-backed generation.

The project has a few distinct model roles, but paper-scale unattended runs
should default to the cheapest/most stable automated model unless a caller
explicitly overrides a role.
"""
from __future__ import annotations

import os


DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"
DEFAULT_GEMINI_PRO_MODEL = "gemini-3.1-flash-lite"


DEFAULT_MODEL_TIERS: dict[str, str] = {
    "latent_vars": DEFAULT_GEMINI_MODEL,
    "entity_gen": DEFAULT_GEMINI_MODEL,
    "edge_generation": DEFAULT_GEMINI_MODEL,
    "temporal_evolution": DEFAULT_GEMINI_MODEL,
    "plot_summaries": DEFAULT_GEMINI_MODEL,
    "character_descriptions": DEFAULT_GEMINI_MODEL,
    "artifact_bulk": DEFAULT_GEMINI_MODEL,
    "artifact_mid": DEFAULT_GEMINI_MODEL,
    "artifact_pro": DEFAULT_GEMINI_PRO_MODEL,
}


def _role_env_name(role: str) -> str:
    normalized = str(role or "").upper().replace("-", "_")
    return f"MIRAGE_MODEL_{normalized}"


def model_for_role(role: str, fallback: str | None = None) -> str:
    """Return the configured model for a Hollywood pipeline role.

    Override order:
      1. MIRAGE_MODEL_<ROLE>, for example MIRAGE_MODEL_LATENT_VARS
      2. MIRAGE_DEFAULT_GEMINI_MODEL for non-Pro roles
      3. MIRAGE_DEFAULT_GEMINI_PRO_MODEL for artifact_pro
      4. Built-in defaults above
    """
    role_key = str(role or "").strip()
    role_override = os.getenv(_role_env_name(role_key))
    if role_override:
        return role_override.strip()

    if role_key == "artifact_pro":
        return (
            os.getenv("MIRAGE_DEFAULT_GEMINI_PRO_MODEL")
            or fallback
            or DEFAULT_MODEL_TIERS.get(role_key)
            or DEFAULT_GEMINI_PRO_MODEL
        ).strip()

    return (
        os.getenv("MIRAGE_DEFAULT_GEMINI_MODEL")
        or fallback
        or DEFAULT_MODEL_TIERS.get(role_key)
        or DEFAULT_GEMINI_MODEL
    ).strip()


def model_tiers() -> dict[str, str]:
    """Materialize all role defaults with current environment overrides."""
    return {role: model_for_role(role, default) for role, default in DEFAULT_MODEL_TIERS.items()}
