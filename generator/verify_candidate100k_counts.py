from __future__ import annotations

import csv
import json
from pathlib import Path


def count_csv(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as f:
        return max(0, sum(1 for _ in f) - 1)


def main() -> None:
    base = Path(__file__).resolve().parent / "entities"
    counts = {
        "persons_json": len(json.loads((base / "persons.json").read_text(encoding="utf-8"))),
        "companies_json": len(json.loads((base / "companies.json").read_text(encoding="utf-8"))),
        "keywords_json": len(json.loads((base / "keywords.json").read_text(encoding="utf-8"))),
        "titles_csv": count_csv(base / "title_bank.csv"),
        "characters_csv": count_csv(base / "character_bank.csv"),
        "person_csv": count_csv(base / "person.csv"),
        "company_csv": count_csv(base / "company.csv"),
        "keyword_csv": count_csv(base / "keyword.csv"),
    }
    print(json.dumps(counts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
