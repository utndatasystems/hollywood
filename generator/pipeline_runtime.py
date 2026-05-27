"""
Stable pipeline runtime configuration and workspace resolution.

This replaces the old `v17_runtime` naming while keeping one-cycle
compatibility aliases for the older imports and environment variables.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from model_defaults import model_for_role


ENV_DATA_DIR = "DATA_SYS_PIPELINE_DATA_DIR"
ENV_OUTPUT_DIR = "DATA_SYS_PIPELINE_OUTPUT_DIR"
ENV_CONFIG = "DATA_SYS_PIPELINE_CONFIG"
ENV_LLM_PROVIDER = "DATA_SYS_PIPELINE_LLM_PROVIDER"
ENV_LLM_CACHE_DIR = "DATA_SYS_PIPELINE_LLM_CACHE_DIR"
ENV_MODE = "DATA_SYS_PIPELINE_MODE"
ENV_START_YEAR = "DATA_SYS_START_YEAR"
ENV_END_YEAR = "DATA_SYS_END_YEAR"
MODELING_PRIORS_FILENAME = "modeling_priors.json"

_LEGACY_ENV_ALIASES = {
    ENV_DATA_DIR: ("DATA_SYS_V17_DATA_DIR",),
    ENV_OUTPUT_DIR: ("DATA_SYS_V17_OUTPUT_DIR",),
    ENV_CONFIG: ("DATA_SYS_V17_CONFIG",),
    ENV_LLM_PROVIDER: ("DATA_SYS_V17_LLM_PROVIDER",),
    ENV_LLM_CACHE_DIR: ("DATA_SYS_V17_LLM_CACHE_DIR",),
}


def _getenv(*names: str) -> str | None:
    for name in names:
        raw = os.getenv(name)
        if raw is not None and str(raw).strip() != "":
            return str(raw)
    return None


def _env_aliases(name: str) -> tuple[str, ...]:
    return (name,) + tuple(_LEGACY_ENV_ALIASES.get(name, ()))


def _as_path(raw: str | os.PathLike[str] | None, default: Path) -> Path:
    if raw is None:
        return default.resolve()
    return Path(raw).expanduser().resolve()


def pipeline_mode(default: str = "research") -> str:
    raw = _getenv(ENV_MODE)
    mode = str(raw or default).strip().lower()
    return "debug" if mode == "debug" else "research"


def year_bounds_from_env(
    fallback_start: int = 1950,
    fallback_end: int = 2025,
) -> tuple[int, int]:
    start_raw = _getenv(ENV_START_YEAR)
    end_raw = _getenv(ENV_END_YEAR)
    try:
        start_year = int(float(start_raw)) if start_raw is not None else int(fallback_start)
    except Exception:
        start_year = int(fallback_start)
    try:
        end_year = int(float(end_raw)) if end_raw is not None else int(fallback_end)
    except Exception:
        end_year = int(fallback_end)
    if end_year < start_year:
        start_year, end_year = end_year, start_year
    return start_year, end_year


@dataclass
class CalibrationTargets:
    cast_gini_min: float = 0.48
    cast_gini_max: float = 0.60
    pop_cast_corr_min: float = 0.32
    edge_giant_component_min: float = 0.84
    best_friend_same_comm_min: float = 0.76
    bridge_ratio_min: float = 0.03
    bridge_ratio_max: float = 0.12
    blockbuster_tail_target: int = 110


@dataclass
class ModelingPriors:
    target_actor_load: float = 6.2
    target_director_load: float = 7.5
    target_company_load: float = 10.5
    role_scarcity_power: float = 1.35
    shortlist_size: int = 96
    cast_base_focus_exploration: float = 0.45
    cast_focus_exploration: float = 0.40
    cast_slot_exploration_empty: float = 0.35
    cast_slot_exploration_filled: float = 0.30
    director_exploration_share: float = 0.30
    company_primary_exploration_share: float = 0.35
    company_secondary_exploration_share: float = 0.40
    crew_exploration_share: float = 0.30
    cast_style_multiplier: float = 2.0
    cast_community_match_multiplier: float = 1.5
    graph_candidate_k: int = 160
    graph_bridge_budget: float = 0.055
    graph_same_block_share: float = 0.78
    graph_closure_budget: float = 0.12
    graph_closure_top_k: int = 12
    graph_director_candidate_k: int = 48
    financial_regime_amplitude: float = 0.18
    financial_slate_pressure: float = 0.075
    financial_momentum_decay: float = 0.68
    financial_recent_horizon: int = 6
    financial_genre_memory_weight: float = 0.11
    temporal_macro_event_budget: int = 18
    temporal_micro_event_budget: int = 48
    temporal_novelty_scale: float = 1.0
    temporal_cross_community_scale: float = 1.0
    critic_sample_size: int = 12
    critic_max_actions: int = 20
    critic_max_repairs_per_title: int = 2


@dataclass
class LLMRoleConfig:
    provider: str = "gemini"
    model: str = field(default_factory=lambda: model_for_role("entity_gen"))
    temperature: float = 0.25
    response_mime_type: str | None = "application/json"
    max_output_tokens: int | None = None


@dataclass
class LLMSettings:
    provider: str = "gemini"
    cache_namespace: str = "pipeline"
    structured: LLMRoleConfig = field(default_factory=LLMRoleConfig)
    creative: LLMRoleConfig = field(
        default_factory=lambda: LLMRoleConfig(
            provider="gemini",
            model=model_for_role("plot_summaries"),
            temperature=0.75,
            response_mime_type="application/json",
        )
    )
    critic: LLMRoleConfig = field(
        default_factory=lambda: LLMRoleConfig(
            provider="gemini",
            model=model_for_role("temporal_evolution"),
            temperature=0.15,
            response_mime_type="application/json",
        )
    )
    artifact_bulk_model: str = field(default_factory=lambda: model_for_role("artifact_bulk"))
    artifact_mid_model: str = field(default_factory=lambda: model_for_role("artifact_mid"))
    artifact_pro_model: str = field(default_factory=lambda: model_for_role("artifact_pro"))


@dataclass
class RuntimeSettings:
    lazy_world_threshold_movies: int = 250
    full_cache_threshold_movies: int = 1000
    write_yearly_snapshots: bool = True
    tv_series_floor_small: int = 8
    tv_series_sqrt_scale: float = 6.0
    tv_series_large_ratio: float = 0.06
    tv_series_max: int = 8000
    contract_background_min_people: int = 180
    contract_background_sqrt_scale: float = 40.0
    contract_background_linear_ratio: float = 0.20
    contract_background_max_people: int = 20000
    media_links_min_total: int = 12
    media_links_target_ratio: float = 2.0
    media_links_per_movie: int = 4
    media_links_bucket_cap: int = 96
    media_links_max_total: int = 0
    disabled_secondary_tables: list[str] = field(default_factory=list)
    disabled_global_tables: list[str] = field(default_factory=list)
    disabled_post_loop_tables: list[str] = field(default_factory=list)


@dataclass
class PipelineConfig:
    version: str = "30"
    data_dir: str | None = None
    output_dir: str | None = None
    llm_provider: str = "gemini"
    llm_cache_dir: str | None = None
    mode: str = "research"
    calibration: CalibrationTargets = field(default_factory=CalibrationTargets)
    priors: ModelingPriors = field(default_factory=ModelingPriors)
    llm: LLMSettings = field(default_factory=LLMSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)


def _merge_dataclass(dc: Any, payload: dict[str, Any]) -> Any:
    for key, value in payload.items():
        if not hasattr(dc, key):
            continue
        current = getattr(dc, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(dc, key, value)
    return dc


def _flatten_priors_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
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


def _merge_modeling_priors_artifact(cfg: PipelineConfig, *roots: Path) -> None:
    for root in roots:
        candidate = Path(root).resolve() / MODELING_PRIORS_FILENAME
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        flat = _flatten_priors_payload(payload if isinstance(payload, dict) else None)
        if not flat:
            continue
        for key, value in flat.items():
            if hasattr(cfg.priors, key):
                try:
                    setattr(cfg.priors, key, type(getattr(cfg.priors, key))(value))
                except Exception:
                    setattr(cfg.priors, key, value)
        break


def load_pipeline_config(config_path: str | os.PathLike[str] | None = None) -> PipelineConfig:
    cfg = PipelineConfig()
    raw_path = config_path or _getenv(*_env_aliases(ENV_CONFIG))
    if raw_path:
        path = Path(raw_path).expanduser().resolve()
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(payload, dict):
                _merge_dataclass(cfg, payload)

    llm_provider = _getenv(*_env_aliases(ENV_LLM_PROVIDER))
    if llm_provider:
        cfg.llm_provider = str(llm_provider)
        cfg.llm.provider = cfg.llm_provider
        cfg.llm.structured.provider = cfg.llm_provider
        cfg.llm.creative.provider = cfg.llm_provider
        cfg.llm.critic.provider = cfg.llm_provider
    llm_cache_dir = _getenv(*_env_aliases(ENV_LLM_CACHE_DIR))
    data_dir = _getenv(*_env_aliases(ENV_DATA_DIR))
    output_dir = _getenv(*_env_aliases(ENV_OUTPUT_DIR))
    if llm_cache_dir:
        cfg.llm_cache_dir = str(llm_cache_dir)
    if data_dir:
        cfg.data_dir = str(data_dir)
    if output_dir:
        cfg.output_dir = str(output_dir)
    cfg.mode = pipeline_mode(default=cfg.mode)
    return cfg


@dataclass
class WorkspacePaths:
    data_dir: Path
    output_dir: Path
    cache_dir: Path
    config: PipelineConfig

    def input_path(self, *parts: str) -> Path:
        rel = Path(*parts)
        out = self.output_dir / rel
        src = self.data_dir / rel
        if out.exists():
            if out.is_file():
                return out
            try:
                if any(out.iterdir()) or not src.exists():
                    return out
            except OSError:
                return out
        return src

    def output_path(self, *parts: str) -> Path:
        path = self.output_dir / Path(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def cache_path(self, *parts: str) -> Path:
        path = self.cache_dir / Path(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for rel in ("entities", "graph", "checkpoints", "_dev"):
            (self.output_dir / rel).mkdir(parents=True, exist_ok=True)


def resolve_workspace(
    *,
    script_dir: str | os.PathLike[str] | None = None,
    data_dir: str | os.PathLike[str] | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    config_path: str | os.PathLike[str] | None = None,
) -> WorkspacePaths:
    default_dir = Path(script_dir or Path(__file__).resolve().parent).resolve()
    cfg = load_pipeline_config(config_path)

    data_root = _as_path(data_dir or cfg.data_dir, default_dir)
    output_root = _as_path(output_dir or cfg.output_dir, default_dir)
    _merge_modeling_priors_artifact(cfg, output_root, data_root, default_dir)
    cache_default = output_root / ".cache" / "llm"
    cache_root = _as_path(cfg.llm_cache_dir, cache_default)

    ws = WorkspacePaths(
        data_dir=data_root,
        output_dir=output_root,
        cache_dir=cache_root,
        config=cfg,
    )
    ws.ensure_dirs()
    return ws


def export_workspace_env(workspace: WorkspacePaths, config_path: str | None = None) -> dict[str, str]:
    env = {
        ENV_DATA_DIR: str(workspace.data_dir),
        ENV_OUTPUT_DIR: str(workspace.output_dir),
        ENV_LLM_PROVIDER: str(workspace.config.llm_provider),
        ENV_LLM_CACHE_DIR: str(workspace.cache_dir),
        ENV_MODE: str(workspace.config.mode),
    }
    for new_name, aliases in _LEGACY_ENV_ALIASES.items():
        if new_name in env:
            for alias in aliases:
                env[alias] = env[new_name]
    if config_path:
        resolved = str(Path(config_path).expanduser().resolve())
        env[ENV_CONFIG] = resolved
        for alias in _LEGACY_ENV_ALIASES.get(ENV_CONFIG, ()):
            env[alias] = resolved
    else:
        existing = _getenv(*_env_aliases(ENV_CONFIG))
        if existing:
            resolved = str(Path(existing).expanduser().resolve())
            env[ENV_CONFIG] = resolved
            for alias in _LEGACY_ENV_ALIASES.get(ENV_CONFIG, ()):
                env[alias] = resolved
    return env


def bootstrap_env_from_argv(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--config")
    parser.add_argument("--llm-provider")
    parser.add_argument("--mode")
    parser.add_argument("--start-year")
    parser.add_argument("--end-year")
    ns, _ = parser.parse_known_args(argv if argv is not None else sys.argv[1:])

    if ns.data_dir:
        resolved = str(Path(ns.data_dir).expanduser().resolve())
        os.environ[ENV_DATA_DIR] = resolved
        os.environ["DATA_SYS_V17_DATA_DIR"] = resolved
    if ns.output_dir:
        resolved = str(Path(ns.output_dir).expanduser().resolve())
        os.environ[ENV_OUTPUT_DIR] = resolved
        os.environ["DATA_SYS_V17_OUTPUT_DIR"] = resolved
    if ns.config:
        resolved = str(Path(ns.config).expanduser().resolve())
        os.environ[ENV_CONFIG] = resolved
        os.environ["DATA_SYS_V17_CONFIG"] = resolved
    if ns.llm_provider:
        os.environ[ENV_LLM_PROVIDER] = str(ns.llm_provider)
        os.environ["DATA_SYS_V17_LLM_PROVIDER"] = str(ns.llm_provider)
    if ns.mode:
        os.environ[ENV_MODE] = str(ns.mode).strip().lower()
    if ns.start_year:
        os.environ[ENV_START_YEAR] = str(ns.start_year)
    if ns.end_year:
        os.environ[ENV_END_YEAR] = str(ns.end_year)


def add_shared_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", default=_getenv(*_env_aliases(ENV_DATA_DIR)))
    parser.add_argument("--output-dir", default=_getenv(*_env_aliases(ENV_OUTPUT_DIR)))
    parser.add_argument("--config", default=_getenv(*_env_aliases(ENV_CONFIG)))
    parser.add_argument("--llm-provider", default=_getenv(*_env_aliases(ENV_LLM_PROVIDER)))
    parser.add_argument("--mode", default=pipeline_mode())


def effective_model_config(workspace: WorkspacePaths) -> dict[str, Any]:
    return {
        "version": workspace.config.version,
        "mode": workspace.config.mode,
        "data_dir": str(workspace.data_dir),
        "output_dir": str(workspace.output_dir),
        "llm_provider": workspace.config.llm_provider,
        "llm_cache_dir": str(workspace.cache_dir),
        "calibration": asdict(workspace.config.calibration),
        "priors": asdict(workspace.config.priors),
        "llm": asdict(workspace.config.llm),
        "runtime": asdict(workspace.config.runtime),
    }
