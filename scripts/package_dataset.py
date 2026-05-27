from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a ZIP archive for the strict IMDb CSV dataset.")
    parser.add_argument("--csv-dir", default=str(ROOT / "data" / "imdb_csv"))
    parser.add_argument("--out", default=str(ROOT / "data" / "hollywood_200k_imdb_csv.zip"))
    parser.add_argument("--checksums", default=str(ROOT / "data" / "checksums.sha256"))
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir).resolve()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(csv_dir.glob("*.csv")):
            zf.write(path, arcname=f"imdb_csv/{path.name}")

    checksum_lines = []
    metadata = [
        ROOT / "data" / "export_manifest.json",
        ROOT / "data" / "genre_derivation_summary.json",
        ROOT / "data" / "source_export_coverage.csv",
    ]
    for path in sorted([out, ROOT / "data" / "hollywood_200k.duckdb", *metadata, *csv_dir.glob("*.csv")]):
        if path.exists():
            checksum_lines.append(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}")
    Path(args.checksums).write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(out)
    print(args.checksums)


if __name__ == "__main__":
    main()
