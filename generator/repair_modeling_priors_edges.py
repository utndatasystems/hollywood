from __future__ import annotations

import argparse
import json
from pathlib import Path

from bootstrap_artifacts import current_mode
from generate_bootstrap_artifacts_api import (
    _artifact_specs,
    _coerce_modeling_priors_group_payload,
    _deep_merge,
    _generate_json_artifact,
    _modeling_priors_completion_prompt,
    _normalize_modeling_priors_payload,
    _prune_nested_modeling_sections,
    _save_artifact,
    _validate_modeling_priors,
    _validate_modeling_priors_sections,
)
from llm_provider import get_llm_client
from policy_runtime import modeling_priors_path


def _load_current_payload(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("modeling_priors payload must be a JSON object")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair only edge-related modeling priors sections in place")
    parser.add_argument("--base-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--mode", default="research")
    parser.add_argument("--model", default=None)
    parser.add_argument("--n-movies", type=int, default=10000)
    parser.add_argument("--n-persons", type=int, default=32000)
    parser.add_argument("--n-companies", type=int, default=1000)
    parser.add_argument("--n-keywords", type=int, default=1500)
    parser.add_argument("--n-titles", type=int, default=10000)
    parser.add_argument("--start-year", type=int, default=1950)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--thinking-budget", type=int, default=None)
    args = parser.parse_args()

    if current_mode(args.mode) != "research":
        raise RuntimeError("edge-priors repair helper is intended for research mode only")

    base_dir = Path(args.base_dir).resolve()
    path = modeling_priors_path(base_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing modeling_priors artifact at {path}")

    current_payload = _load_current_payload(path)
    sections = ("edge_priors", "scalable_edge_priors")
    working = _prune_nested_modeling_sections(
        _normalize_modeling_priors_payload({name: current_payload.get(name, {}) for name in sections}, sections)
    )

    client = get_llm_client()
    spec = _artifact_specs()["modeling_priors"]
    model = str(args.model or spec["default_model"])
    thinking_budget = int(args.thinking_budget) if args.thinking_budget is not None else None

    last_error: Exception | None = None
    try:
        _validate_modeling_priors_sections(working, sections)
    except Exception as exc:
        last_error = exc

    if last_error is not None:
        for attempt in range(4):
            completion = _generate_json_artifact(
                client=client,
                artifact_name=f"repair_modeling_priors_edges_attempt_{attempt + 1}",
                prompt=_modeling_priors_completion_prompt(args, "edges_repair", sections, last_error, working),
                model=model,
                temperature=0.0,
                max_tokens=max(8000, int(spec["max_tokens"])),
                thinking_budget=thinking_budget,
                base_dir=base_dir,
                coerce=lambda raw, _sections=sections: _coerce_modeling_priors_group_payload(raw, _sections),
                validate=lambda _payload: None,
            )
            working = _prune_nested_modeling_sections(
                _normalize_modeling_priors_payload(_deep_merge(working, completion), sections)
            )
            try:
                _validate_modeling_priors_sections(working, sections)
                last_error = None
                break
            except Exception as exc:
                last_error = exc

    if last_error is not None:
        raise last_error

    merged = dict(current_payload)
    for name in sections:
        merged[name] = _deep_merge(current_payload.get(name, {}), working.get(name, {}))
    merged = _prune_nested_modeling_sections(_normalize_modeling_priors_payload(merged))
    _validate_modeling_priors(merged)
    _save_artifact(path, "modeling_priors", merged, model)
    print(f"Repaired edge-related modeling priors in {path}")


if __name__ == "__main__":
    main()
