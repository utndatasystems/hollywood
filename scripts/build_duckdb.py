from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "generator"))

from build_duckdb_from_imdb_schema import build_duckdb  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a DuckDB database from the strict IMDb CSV tables.")
    parser.add_argument("--csv-dir", default=str(ROOT / "data" / "imdb_csv"))
    parser.add_argument("--out", default=str(ROOT / "data" / "hollywood_200k.duckdb"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out = build_duckdb(Path(args.csv_dir), Path(args.out), overwrite=args.overwrite, core_only=True)
    print(out)


if __name__ == "__main__":
    main()
