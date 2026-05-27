from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd

from contracts import GENRES
from llm_provider import get_llm_client, safe_json_parse
from model_defaults import model_for_role
from policy_runtime import (
    enrich_keyword_dataframe,
    keyword_motif_bank_path,
    normalize_keyword_motif_bank,
    safe_load_json,
    write_json,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = model_for_role("entity_gen")
BATCH_SIZE = 32


def _is_local_like(model: str | None = None) -> bool:
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    model_name = str(model or "").strip().lower()
    return provider in {"local", "ollama", "vllm", "openai", "tgi", "litellm"} or "qwen" in model_name


def _motif_batch_size(model: str | None = None) -> int:
    return 24 if _is_local_like(model) else 16


def _load_keyword_rows(base_dir: Path) -> list[dict[str, Any]]:
    csv_path = base_dir / "entities" / "keyword.csv"
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path, low_memory=False)
    if df.empty:
        return []
    return df.to_dict(orient="records")


def _build_prompt(batch: list[dict[str, Any]]) -> str:
    compact = [
        {
            "keyword": str(row.get("keyword", "")),
            "topic_genre": str(row.get("topic_genre", "Drama")),
            "pop_weight": float(row.get("pop_weight", 0.02) or 0.02),
        }
        for row in batch
    ]
    return (
        "You are annotating reusable keyword motifs for a synthetic IMDb-style keyword ontology.\n"
        "Return JSON only.\n\n"
        "Requirements:\n"
        '1. Top-level key must be "motifs".\n'
        "2. Return exactly one motif row per input keyword.\n"
        "3. Each motif must include: keyword, topic_genre, motif_family, specificity_tier,\n"
        "   scope_hint, franchise_affinity, cooccurrence_cluster, recurrence_strength.\n"
        "4. motif_family must be one of: genre, subgenre, setting, object, profession,\n"
        "   relationship, event, place, tone, franchise, sequel_drift.\n"
        "5. specificity_tier is an integer 1..5.\n"
        "6. franchise_affinity and recurrence_strength are floats 0..1.\n"
        "7. scope_hint should be one of: global, year_slate, concept_pack, franchise.\n\n"
        "8. Do not lazily label everything as genre.\n"
        "9. Use subgenre for phrases like cyberpunk, coming-of-age, prison-break, police-procedural.\n"
        "10. Use event for things like battle, escape, wedding, festival, heist.\n"
        "11. Use relationship/profession/setting/object/place/tone whenever the keyword clearly fits.\n"
        "12. Reserve genre only for broad thematic catch-alls.\n\n"
        f"Keywords:\n{json.dumps(compact, ensure_ascii=True)}\n"
    )


def _request_batch(
    client: Any,
    batch: list[dict[str, Any]],
    *,
    model: str | None,
    depth: int = 0,
) -> list[dict[str, Any]]:
    try:
        response = client.generate(
            _build_prompt(batch),
            model=model or DEFAULT_MODEL,
            json_mode=True,
            temperature=0.2,
            max_tokens=2048,
            timeout_sec=60.0,
            max_attempts=2,
        )
        parsed = safe_json_parse(response.text)
        motifs = parsed.get("motifs") if isinstance(parsed, dict) else None
    except Exception:
        if len(batch) <= 8 or depth >= 3:
            raise
        mid = max(1, len(batch) // 2)
        left = _request_batch(client, batch[:mid], model=model, depth=depth + 1)
        right = _request_batch(client, batch[mid:], model=model, depth=depth + 1)
        return left + right

    if isinstance(motifs, list):
        return [row for row in motifs if isinstance(row, dict)]
    if len(batch) <= 12 or depth >= 2:
        return []
    mid = max(1, len(batch) // 2)
    left = _request_batch(client, batch[:mid], model=model, depth=depth + 1)
    right = _request_batch(client, batch[mid:], model=model, depth=depth + 1)
    return left + right


def _annotate_batches(base_dir: Path, keyword_rows: list[dict[str, Any]], *, model: str | None = None) -> dict[str, Any]:
    parsed_rows: list[dict[str, Any]] = []
    if not keyword_rows:
        return {"motifs": []}
    client = None
    try:
        client = get_llm_client()
    except Exception:
        client = None
    batch_size = _motif_batch_size(model)
    n_batches = math.ceil(len(keyword_rows) / batch_size)
    for batch_idx in range(n_batches):
        batch = keyword_rows[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        if client is None:
            continue
        try:
            parsed_rows.extend(_request_batch(client, batch, model=model))
        except Exception as exc:
            print(f"  Keyword motif batch {batch_idx + 1}/{n_batches} fallback: {exc}")
    return {"motifs": parsed_rows}


def _sync_keyword_files(base_dir: Path, bank: dict[str, Any]) -> None:
    csv_path = base_dir / "entities" / "keyword.csv"
    json_path = base_dir / "entities" / "keywords.json"
    if csv_path.exists():
        df = pd.read_csv(csv_path, low_memory=False)
        df = enrich_keyword_dataframe(df, bank)
        df.to_csv(csv_path, index=False)
    if json_path.exists():
        rows = safe_load_json(json_path, default=[])
        if isinstance(rows, list):
            enriched_df = None
            if csv_path.exists():
                enriched_df = pd.read_csv(csv_path, low_memory=False)
                lookup = {
                    str(row.get("keyword", "")).strip().lower(): row
                    for row in enriched_df.to_dict(orient="records")
                }
            else:
                lookup = {}
            for row in rows:
                if not isinstance(row, dict):
                    continue
                key = str(row.get("keyword", row.get("name", ""))).strip().lower()
                if key in lookup:
                    src = lookup[key]
                    for column in (
                        "topic_genre",
                        "pop_weight",
                        "selection_bucket",
                        "motif_family",
                        "specificity_tier",
                        "scope_hint",
                        "franchise_affinity",
                        "cooccurrence_cluster",
                        "recurrence_strength",
                    ):
                        if column in src:
                            if column in {"specificity_tier"}:
                                row[column] = int(src[column])
                            elif column in {"pop_weight", "franchise_affinity", "recurrence_strength"}:
                                row[column] = float(src[column])
                            else:
                                row[column] = str(src[column])
            json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_keyword_motif_bank(base_dir: Path, *, model: str | None = None) -> dict[str, Any]:
    keyword_rows = _load_keyword_rows(base_dir)
    parsed = _annotate_batches(base_dir, keyword_rows, model=model)
    bank = normalize_keyword_motif_bank(
        parsed,
        keyword_rows=keyword_rows,
        genres=GENRES,
    )
    write_json(keyword_motif_bank_path(base_dir), bank)
    _sync_keyword_files(base_dir, bank)
    return bank


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate hierarchical Mirage keyword motif metadata.")
    parser.add_argument("--base-dir", default=str(BASE_DIR))
    parser.add_argument("--auto", action="store_true", help="Accepted for pipeline parity.")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    bank = generate_keyword_motif_bank(base_dir, model=args.model)
    print(
        "  Saved keyword motif bank:",
        keyword_motif_bank_path(base_dir),
        f"({len(bank.get('motifs', []))} motifs)",
    )


if __name__ == "__main__":
    main()
