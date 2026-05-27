"""Run profile loading for Mirage command-line entry points.

Profiles are simple KEY=VALUE files under ``local_run_profiles/``. Keeping the
parser in Python lets ``run_pipeline.py`` run a paper-scale profile directly
without relying on shell-specific wrappers.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


PROFILE_ARG_FIELDS: dict[str, tuple[str, type]] = {
    "N_MOVIES": ("n_movies", int),
    "START_YEAR": ("start_year", int),
    "END_YEAR": ("end_year", int),
    "N_PERSONS": ("n_persons", int),
    "N_COMPANIES": ("n_companies", int),
    "N_KEYWORDS": ("n_keywords", int),
    "N_CHARACTERS": ("n_characters", int),
    "N_TITLES": ("n_titles", int),
    "RERANK_BUDGET_MOVIES": ("rerank_budget_movies", int),
    "KEYWORD_RERANK_BUDGET_MOVIES": ("keyword_rerank_budget_movies", int),
}

PROFILE_ENV_KEYS = {
    "RUN_PROFILE",
    "FAST_TITLE_TAGLINES",
    "SKIP_DIAGNOSTIC_COLD_EDGES",
    "DATA_SYS_STEP100_SPOOL_ROWS",
    "DATA_SYS_STEP100_GRAPH_CHECKPOINT_MODE",
    "DATA_SYS_ACTOR_VIEW_CACHE_MAX",
    "DATA_SYS_CAST_BASE_FOCUS_SIZE_CAP",
    "DATA_SYS_CAST_FOCUS_CAP",
}

PROFILE_ENV_ALIASES = {
    "FAST_TITLE_TAGLINES": "DATA_SYS_FAST_TITLE_TAGLINES",
    "SKIP_DIAGNOSTIC_COLD_EDGES": "DATA_SYS_SKIP_DIAGNOSTIC_COLD_EDGES",
}


@dataclass(frozen=True)
class RunProfile:
    name: str
    path: Path
    values: dict[str, str]


def parse_profile_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid profile line {line_number} in {path}: {raw_line!r}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if not key:
            raise ValueError(f"Invalid empty profile key on line {line_number} in {path}")
        values[key] = value
    return values


def load_run_profile(base_dir: Path, *, profile: str | None = None, profile_file: str | None = None) -> RunProfile | None:
    profile_name = str(profile or "").strip()
    file_name = str(profile_file or "").strip()
    if not profile_name and not file_name:
        return None

    if file_name:
        path = Path(file_name)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
    else:
        path = (base_dir / "local_run_profiles" / f"{profile_name}.env").resolve()

    if not path.exists():
        raise FileNotFoundError(f"Run profile not found: {path}")

    values = parse_profile_file(path)
    name = values.get("RUN_PROFILE") or profile_name or path.stem
    return RunProfile(name=name, path=path, values=values)


def _coerce_profile_value(key: str, value: str, value_type: type) -> object:
    if value_type is int:
        return int(float(value))
    return value_type(value)


def apply_run_profile(args: argparse.Namespace, base_dir: Path) -> RunProfile | None:
    """Apply a profile to missing CLI args and process env.

    Explicit command-line values win over profile values. Profile-only runtime
    tuning keys are exported to the process environment only if not already set.
    """
    profile = load_run_profile(
        base_dir,
        profile=getattr(args, "profile", None),
        profile_file=getattr(args, "profile_file", None),
    )
    if profile is None:
        return None

    for key, (attr, value_type) in PROFILE_ARG_FIELDS.items():
        if key not in profile.values:
            continue
        if getattr(args, attr, None) is None:
            setattr(args, attr, _coerce_profile_value(key, profile.values[key], value_type))

    os.environ.setdefault("RUN_PROFILE", profile.name)
    for key, value in profile.values.items():
        if key.startswith("DATA_SYS_") or key in PROFILE_ENV_KEYS:
            os.environ.setdefault(key, value)
        alias = PROFILE_ENV_ALIASES.get(key)
        if alias:
            os.environ.setdefault(alias, value)

    return profile
