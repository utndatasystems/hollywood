from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_persons_procedural import assign_career_timelines


VALID_STAGES = {"rising", "prime", "veteran", "legend", "retired"}
STAGE_ALIASES = {
    "veterant": "veteran",
    "vetaran": "veteran",
    "legand": "legend",
    "legendary": "legend",
    "priime": "prime",
    "emerging": "rising",
    "established": "prime",
}


def _normalize_stage(value: object) -> str:
    key = str(value or "").strip().lower()
    key = STAGE_ALIASES.get(key, key)
    return key if key in VALID_STAGES else "prime"


def _needs_repair(person: dict) -> bool:
    required = ("debut_year", "peak_start", "peak_end", "retirement_year", "yearly_max")
    if any(person.get(key) is None for key in required):
        return True
    try:
        debut = int(float(person.get("debut_year")))
        peak_start = int(float(person.get("peak_start")))
        peak_end = int(float(person.get("peak_end")))
        retire = int(float(person.get("retirement_year")))
        yearly_max = int(float(person.get("yearly_max")))
    except Exception:
        return True
    return not (debut <= peak_start <= peak_end <= retire and yearly_max > 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill person career timelines in entities/persons.json")
    parser.add_argument("--base-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    persons_path = base_dir / "entities" / "persons.json"
    payload = json.loads(persons_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("persons.json must contain a list")
    persons = [row for row in payload if isinstance(row, dict)]
    if len(persons) != len(payload):
        raise ValueError("persons.json must contain only person objects")

    stage_changed = 0
    for person in persons:
        normalized = _normalize_stage(person.get("career_stage"))
        if person.get("career_stage") != normalized:
            person["career_stage"] = normalized
            stage_changed += 1

    timeline_changed = any(_needs_repair(person) for person in persons)
    if timeline_changed:
        assign_career_timelines(persons, seed=int(args.seed))

    if stage_changed or timeline_changed:
        persons_path.write_text(json.dumps(persons, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Repaired career timelines in {persons_path} (stage fixes={stage_changed}, timeline repair={timeline_changed})")
    else:
        print(f"Career timelines already present in {persons_path}")


if __name__ == "__main__":
    main()
