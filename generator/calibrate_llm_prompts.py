from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from llm_provider import get_llm_client
from policy_runtime import append_jsonl, prompt_calibration_log_path, safe_load_json
from generate_world_policy_api import _build_prompt as build_world_policy_prompt, _build_summary as build_world_policy_summary
from generate_year_slate_plan_api import _build_prompt as build_year_slate_prompt
from generate_keyword_motif_bank_api import _build_prompt as build_keyword_motif_prompt
from generate_franchise_bibles_api import _build_prompt as build_franchise_bible_prompt
from model_defaults import model_for_role

BASE_DIR = Path(__file__).resolve().parent


def _sample_keyword_units(base_dir: Path, units: int) -> list[dict]:
    csv_path = base_dir / "entities" / "keyword.csv"
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path, low_memory=False).head(max(1, int(units)))
    return df.to_dict(orient="records")


def _sample_franchise_units(base_dir: Path, units: int) -> list[dict]:
    payload = safe_load_json(base_dir / "franchise_bibles.json", default={}) or {}
    if isinstance(payload, dict) and payload.get("bibles"):
        return list(payload.get("bibles", []))[: max(1, int(units))]
    world_policy = safe_load_json(base_dir / "world_policy.json", default={}) or {}
    sample = []
    for idx, bucket in enumerate((world_policy.get("year_buckets", []) if isinstance(world_policy, dict) else [])[: max(1, min(6, units))], start=1):
        sample.append(
            {
                "franchise_id": idx,
                "name": f"Probe Franchise {idx}",
                "genre": next(iter((bucket.get("genre_bias") or {}).keys()), "Action"),
                "tier": "A",
                "n_movies": 3,
                "installment_years": [bucket.get("start_year", 2000), bucket.get("start_year", 2000) + 2, bucket.get("start_year", 2000) + 4],
            }
        )
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Run small live prompt calibration probes for Mirage LLM stages.")
    parser.add_argument("--base-dir", default=str(BASE_DIR))
    parser.add_argument("--model", default=model_for_role("artifact_mid"))
    parser.add_argument("--units", type=int, default=25)
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    client = get_llm_client()
    log_path = prompt_calibration_log_path(base_dir)

    probes = []
    world_summary = build_world_policy_summary(base_dir)
    probes.append(("world_policy", build_world_policy_prompt(world_summary)))

    world_policy = safe_load_json(base_dir / "world_policy.json", default={}) or {}
    probes.append(("year_slates", build_year_slate_prompt(world_policy if isinstance(world_policy, dict) else {})))

    keyword_units = _sample_keyword_units(base_dir, min(100, max(25, args.units)))
    if keyword_units:
        for batch_idx in range(math.ceil(len(keyword_units) / 25)):
            batch = keyword_units[batch_idx * 25 : (batch_idx + 1) * 25]
            probes.append((f"keyword_motifs_{batch_idx + 1}", build_keyword_motif_prompt(batch)))

    franchise_units = _sample_franchise_units(base_dir, min(30, max(5, args.units // 5)))
    if franchise_units:
        probes.append(("franchise_bibles", build_franchise_bible_prompt(franchise_units[:10], {"probe_mode": True})))

    for name, prompt in probes:
        try:
            response = client.generate(
                prompt,
                model=args.model,
                json_mode=True,
                temperature=0.2,
                max_tokens=2048,
                timeout_sec=90.0,
                max_attempts=3,
            )
            append_jsonl(
                log_path,
                {
                    "probe": name,
                    "model": response.model,
                    "prompt_chars": len(prompt),
                    "input_tokens": int(response.input_tokens or 0),
                    "output_tokens": int(response.output_tokens or 0),
                    "cost_usd": float(response.cost_usd or 0.0),
                    "response_preview": response.text[:800],
                },
            )
            print(f"  Probe {name}: OK")
        except Exception as exc:
            append_jsonl(
                log_path,
                {
                    "probe": name,
                    "model": args.model,
                    "prompt_chars": len(prompt),
                    "error": str(exc),
                },
            )
            print(f"  Probe {name}: ERROR {exc}")

    print("  Calibration log:", log_path)


if __name__ == "__main__":
    main()
