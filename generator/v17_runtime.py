"""
Compatibility shim for the renamed pipeline runtime module.

Mirage should use `pipeline_runtime`, but older imports still resolve here for
one migration cycle.
"""
from __future__ import annotations

from pipeline_runtime import *  # noqa: F401,F403
from pipeline_runtime import (  # noqa: F401
    CalibrationTargets,
    LLMRoleConfig,
    LLMSettings,
    ModelingPriors,
    PipelineConfig,
    RuntimeSettings,
    WorkspacePaths,
    add_shared_runtime_args,
    bootstrap_env_from_argv,
    effective_model_config,
    export_workspace_env,
    load_pipeline_config,
    pipeline_mode,
    resolve_workspace,
    year_bounds_from_env,
)


V17Config = PipelineConfig
load_v17_config = load_pipeline_config
