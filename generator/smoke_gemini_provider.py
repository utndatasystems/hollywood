from __future__ import annotations

import argparse
import json

from llm_provider import get_llm_client
from model_defaults import model_for_role


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify that Google Gemini generation is configured.")
    parser.add_argument("--model", default=model_for_role("entity_gen"), help="Gemini model to test.")
    parser.add_argument("--timeout-sec", type=int, default=60)
    parser.add_argument("--max-attempts", type=int, default=2)
    args = parser.parse_args()

    client = get_llm_client(provider="gemini", force_new=True)
    response = client.generate(
        "Return exactly one compact JSON object with ok=true and label='hollywood'. No markdown.",
        model=args.model,
        json_mode=True,
        temperature=0.0,
        max_tokens=128,
        timeout_sec=args.timeout_sec,
        max_attempts=args.max_attempts,
    )
    payload = json.loads(str(response.text or "").strip())
    if not isinstance(payload, dict) or payload.get("ok") is not True or payload.get("label") != "hollywood":
        raise SystemExit(f"Unexpected Gemini smoke response: {payload!r}")
    print(f"Gemini smoke passed: model={response.model} payload={payload}")


if __name__ == "__main__":
    main()
