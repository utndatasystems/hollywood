#!/usr/bin/env python3
"""Small OpenAI-compatible endpoint smoke check for lab/cluster runs."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from llm_provider import get_llm_client


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify that the configured OpenAI-compatible endpoint can answer JSON prompts.")
    parser.add_argument("--model", default=None, help="Model name registered by the OpenAI-compatible backend.")
    parser.add_argument("--timeout-sec", type=float, default=90.0)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.getenv("DATA_SYS_LOCAL_CHECK_MAX_TOKENS", "768")),
        help="Output-token budget for the provider smoke. Small thinking models may need >=512.",
    )
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    os.environ["LLM_PROVIDER"] = "local"
    if args.model:
        os.environ["LOCAL_LLM_MODEL"] = str(args.model)

    prompt = (
        "Return exactly one compact JSON object with keys ok, label, and note. "
        "Set ok to true, label to \"datasys_lab\", and note to a short phrase. "
        "Do not include markdown."
    )
    t0 = time.time()
    client = get_llm_client(provider="local", force_new=True)
    response = client.generate(
        prompt,
        model=args.model,
        json_mode=True,
        temperature=0.0,
        max_tokens=int(args.max_tokens),
        timeout_sec=float(args.timeout_sec),
        max_attempts=int(args.max_attempts),
        base_delay_sec=1.0,
        max_delay_sec=3.0,
    )
    elapsed = time.time() - t0
    raw_text = str(response.text or "").strip()
    parsed: object
    try:
        parsed = json.loads(raw_text)
    except Exception as exc:
        raise SystemExit(f"Local LLM responded, but not with valid JSON: {exc}\nRaw text: {raw_text[:500]}")

    if not isinstance(parsed, dict) or parsed.get("ok") is not True:
        raise SystemExit(f"Local LLM JSON check failed. Parsed response: {parsed!r}")

    result = {
        "ok": True,
        "provider": "local",
        "model": response.model,
        "base_url": os.getenv("LOCAL_LLM_URL", "http://localhost:8000/v1"),
        "elapsed_sec": round(elapsed, 3),
        "input_tokens": int(response.input_tokens or 0),
        "output_tokens": int(response.output_tokens or 0),
        "parsed": parsed,
    }

    out_json = args.out_json or (args.base_dir / "reports" / "local_llm_check.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Local LLM check passed: model={result['model']} elapsed={result['elapsed_sec']}s")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
