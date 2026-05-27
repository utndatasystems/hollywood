"""
LLM enrichment for procedurally generated persons.

Reads entities/persons.json produced by generate_persons_procedural.py and fills:
  - bio
  - style_tags
  - genre_affinity
  - market_fit

Names, nationality, gender, roles, and career stage stay procedural.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import shutil
from pathlib import Path

from dotenv import load_dotenv

from contracts import (
    DIRECTOR_STYLES,
    GENRES,
    MARKETS,
    STYLE_TAGS,
    load_json_batch,
    normalize_style_tag,
    save_json_batch,
    validate_batch,
    validate_person,
)
from entities_to_csv import normalize_person_record
from llm_provider import get_llm_client
from model_defaults import model_for_role

def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return max(minimum, int(default))
    try:
        return max(minimum, int(float(str(raw).strip())))
    except Exception:
        return max(minimum, int(default))


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


BASE_DIR = Path(__file__).resolve().parent
ENTITY_DIR = BASE_DIR / "entities"
DEFAULT_MODEL = model_for_role("entity_gen")
DEFAULT_BATCH_SIZE = _env_int("DATA_SYS_PERSON_ENRICH_BATCH_SIZE", 40)
SHARD_DIRNAME = "_person_enrich_shards"


OUTER_RETRIES = _env_int("DATA_SYS_PERSON_ENRICH_OUTER_RETRIES", 3)


def _retry_sleep(attempt_index: int) -> float:
    profile = str(os.getenv("DATA_SYS_OVERLOAD_PROFILE", "default") or "default").strip().lower()
    if profile == "smoke":
        schedule = [1.5, 2.5, 4.0, 6.0]
    else:
        schedule = [2.0, 4.0, 6.0, 8.0]
    idx = min(max(0, int(attempt_index)), len(schedule) - 1)
    return float(schedule[idx])


def _parse_json_response(text: str) -> list[dict]:
    raw = str(text or "").strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:])
        if raw.rstrip().endswith("```"):
            raw = raw.rstrip()[:-3]
    parsed = json.loads(raw)
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict):
        for key in ("people", "persons", "items", "rows", "results", "data"):
            value = parsed.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        if parsed.get("person_id") is not None:
            return [parsed]
    raise ValueError(f"Expected JSON list-compatible payload, got {type(parsed)}")


def _normalize_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        text = value.replace("|", ",").replace(";", ",")
        return [part.strip() for part in text.split(",") if part.strip()]
    return [str(value).strip()]


def _fallback_fields(person: dict) -> dict:
    roles = [str(r).strip() for r in person.get("roles", []) if str(r).strip()]
    primary = roles[0] if roles else "actor"
    stage = str(person.get("career_stage", "prime"))
    nationality = str(person.get("nationality", "American"))
    name = str(person.get("name", f"Person {person.get('person_id', '?')}"))
    market_fit = _normalize_string_list(person.get("market_fit", [])) or ["Regional"]

    if "director" in roles:
        styles = ["atmospheric", "intimate"]
        genres = ["Drama", "Thriller"]
    elif "writer" in roles:
        styles = ["cerebral", "understated"]
        genres = ["Drama", "Mystery"]
    elif "composer" in roles:
        styles = ["lyrical", "theatrical"]
        genres = ["Drama", "Fantasy"]
    elif "cinematographer" in roles:
        styles = ["visual-spectacle", "naturalistic"]
        genres = ["Drama", "Thriller"]
    else:
        styles = ["naturalistic", "magnetic"]
        genres = ["Drama", "Comedy"]

    bio = (
        f"{name} is a {nationality.lower()} {primary} in the {stage} stage of their career, "
        f"known for disciplined collaboration and a clear on-screen identity. Their work tends "
        f"to fit {', '.join(g.lower() for g in genres[:2])} projects with strong ensemble energy."
    )
    return {
        "bio": bio,
        "style_tags": styles,
        "genre_affinity": genres,
        "market_fit": market_fit[:2],
    }


def _sanitize_enrichment(person: dict, patch: dict) -> dict:
    combined_styles = list(STYLE_TAGS) + list(DIRECTOR_STYLES)
    fallback = _fallback_fields(person)

    bio = str(patch.get("bio", "") or "").strip()
    if len(bio) < 40:
        bio = fallback["bio"]

    styles: list[str] = []
    for tag in _normalize_string_list(patch.get("style_tags", [])):
        norm = normalize_style_tag(tag, vocab=combined_styles)
        if norm and norm not in styles:
            styles.append(norm)
    if not styles:
        styles = fallback["style_tags"]
    styles = styles[:4]

    genres: list[str] = []
    for genre in _normalize_string_list(patch.get("genre_affinity", [])):
        for valid in GENRES:
            if genre.strip().lower() == valid.lower() and valid not in genres:
                genres.append(valid)
                break
    if not genres:
        genres = fallback["genre_affinity"]
    genres = genres[:3]

    market_fit: list[str] = []
    for market in _normalize_string_list(patch.get("market_fit", [])):
        for valid in MARKETS:
            if market.strip().lower() == valid.lower() and valid not in market_fit:
                market_fit.append(valid)
                break
    if not market_fit:
        market_fit = _normalize_string_list(person.get("market_fit", [])) or fallback["market_fit"]
    market_fit = market_fit[:2]

    return {
        "bio": bio,
        "style_tags": styles,
        "genre_affinity": genres,
        "market_fit": market_fit,
    }


def _build_prompt(batch: list[dict]) -> str:
    style_vocab = ", ".join(list(STYLE_TAGS) + list(DIRECTOR_STYLES))
    genre_vocab = ", ".join(GENRES)
    market_vocab = ", ".join(MARKETS)

    lines = []
    for person in batch:
        lines.append(
            f"[{person['person_id']}] {person['name']} | nationality: {person['nationality']} | "
            f"gender: {person['gender']} | roles: {', '.join(person.get('roles', []))} | "
            f"career_stage: {person.get('career_stage', 'prime')} | "
            f"current_market_fit: {', '.join(_normalize_string_list(person.get('market_fit', [])))}"
        )

    people_block = "\n".join(lines)
    return f"""You are enriching a synthetic film-industry person roster.

For each person below, keep the existing identity fixed and add four fields:
1. bio: 2-3 vivid Wikipedia-style sentences, 45-95 words total
2. style_tags: 2-4 tags chosen only from this list: {style_vocab}
3. genre_affinity: 1-3 genres chosen only from this list: {genre_vocab}
4. market_fit: 1-2 markets chosen only from this list: {market_vocab}

Important rules:
- Do NOT invent or change names, nationality, gender, roles, or career_stage.
- Match the bio to the person's role mix and career stage.
- Make the bios varied and specific rather than templated.
- Director / writer / composer people can use behind-the-camera style tags when appropriate.
- Output ONLY a JSON array.

People:
{people_block}

Return JSON array of:
{{
  "person_id": 123,
  "bio": "...",
  "style_tags": ["..."],
  "genre_affinity": ["..."],
  "market_fit": ["..."]
}}
"""


def _shard_dir(base_dir: Path) -> Path:
    return base_dir / "entities" / SHARD_DIRNAME


def _load_shard_records(base_dir: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    shard_dir = _shard_dir(base_dir)
    if not shard_dir.exists():
        return out
    for path in sorted(shard_dir.glob("batch_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                pid = int(row.get("person_id"))
            except Exception:
                continue
            if pid > 0 and isinstance(row, dict):
                out[pid] = row
    return out


def _write_shard(base_dir: Path, batch_idx: int, batch_rows: list[dict]) -> None:
    shard_dir = _shard_dir(base_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    path = shard_dir / f"batch_{batch_idx + 1:06d}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in batch_rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _next_shard_offset(base_dir: Path) -> int:
    shard_dir = _shard_dir(base_dir)
    if not shard_dir.exists():
        return 0
    highest = 0
    for path in shard_dir.glob("batch_*.jsonl"):
        try:
            highest = max(highest, int(path.stem.rsplit("_", 1)[1]))
        except Exception:
            continue
    return highest


def enrich_persons(
    base_dir: Path,
    *,
    model: str | None,
    batch_size: int,
    force: bool,
) -> None:
    persons_path = base_dir / "entities" / "persons.json"
    if not persons_path.exists():
        raise FileNotFoundError(f"persons.json not found at {persons_path}")

    if force and _shard_dir(base_dir).exists():
        shutil.rmtree(_shard_dir(base_dir))

    persons = load_json_batch(persons_path)
    by_id = {int(p["person_id"]): p for p in persons if "person_id" in p}
    shard_rows = _load_shard_records(base_dir)
    for pid, row in shard_rows.items():
        if pid in by_id:
            by_id[pid].update(row)
    pending = [
        p for p in persons
        if force
        or not str(p.get("bio", "") or "").strip()
        or not _normalize_string_list(p.get("style_tags", []))
        or not _normalize_string_list(p.get("genre_affinity", []))
    ]

    def _normalize_and_validate(current_persons: list[dict]) -> None:
        normalized = [normalize_person_record(person) for person in current_persons]
        save_json_batch(normalized, persons_path)
        result = validate_batch(normalized, validate_person, "persons")
        if result["invalid"]:
            raise ValueError(
                f"Person enrichment finished with {result['invalid']} invalid records; "
                "inspect persons.json before continuing."
            )
        print(f"Validated {result['valid']} persons")

    print(f"Loaded {len(persons)} persons")
    print(f"Need enrichment: {len(pending)}")
    if not pending:
        print("All persons already enriched.")
        _normalize_and_validate(persons)
        return

    if _env_bool("DATA_SYS_DETERMINISTIC_ENRICH", False):
        print("Deterministic enrichment enabled; using local fallback fields without LLM calls.")
        for person in pending:
            person.update(_sanitize_enrichment(person, {}))
        _normalize_and_validate(persons)
        if _shard_dir(base_dir).exists():
            shutil.rmtree(_shard_dir(base_dir), ignore_errors=True)
        return

    llm = get_llm_client()
    effective_model = model or DEFAULT_MODEL
    total_batches = (len(pending) + batch_size - 1) // batch_size
    shard_offset = _next_shard_offset(base_dir)

    for batch_idx in range(total_batches):
        batch = pending[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        print(f"  Batch {batch_idx + 1}/{total_batches} ({len(batch)} persons)...", flush=True)
        prompt = _build_prompt(batch)

        parsed: list[dict] | None = None
        last_error: Exception | None = None
        for attempt in range(OUTER_RETRIES):
            try:
                response = llm.generate(
                    prompt,
                    model=effective_model,
                    json_mode=True,
                    temperature=0.8,
                    max_tokens=16384,
                    timeout_sec=90,
                    max_attempts=5,
                )
                parsed = _parse_json_response(response.text)
                break
            except Exception as exc:
                last_error = exc
                wait_for = _retry_sleep(attempt)
                print(f"    Retry {attempt + 1}/{OUTER_RETRIES} after error: {exc}")
                if attempt + 1 < OUTER_RETRIES:
                    print(f"    Waiting {wait_for:.1f}s before retry...")
                    time.sleep(wait_for)

        patches_by_id: dict[int, dict] = {}
        if parsed is not None:
            for item in parsed:
                try:
                    pid = int(item.get("person_id"))
                except Exception:
                    continue
                if pid in by_id:
                    patches_by_id[pid] = item
        elif last_error is not None:
            print(f"    Falling back for batch after repeated failure: {last_error}")

        for person in batch:
            pid = int(person["person_id"])
            patch = patches_by_id.get(pid, {})
            enriched = _sanitize_enrichment(person, patch)
            person.update(enriched)

        batch_rows = [normalize_person_record(person) for person in batch]
        _write_shard(base_dir, shard_offset + batch_idx, batch_rows)

    _normalize_and_validate(persons)
    if _shard_dir(base_dir).exists():
        shutil.rmtree(_shard_dir(base_dir), ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-enrich procedural persons")
    parser.add_argument("--base-dir", default=str(BASE_DIR))
    parser.add_argument("--model", default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--auto", action="store_true", help="Accepted for runner compatibility")
    parser.add_argument("--mode", choices=("research", "debug"), default=os.getenv("DATA_SYS_PIPELINE_MODE", "research"))
    args = parser.parse_args()

    load_dotenv(BASE_DIR.parent / ".env")
    enrich_persons(
        Path(args.base_dir).resolve(),
        model=args.model,
        batch_size=max(1, int(args.batch_size)),
        force=bool(args.force),
    )


if __name__ == "__main__":
    main()
