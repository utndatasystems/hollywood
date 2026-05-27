from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a generated Hollywood workspace to the public strict IMDb/JOB schema.")
    parser.add_argument("--base-dir", required=True, help="Completed generator workspace.")
    parser.add_argument("--out-dir", required=True, help="Output directory for strict CSV files.")
    parser.add_argument("--company-country-policy", default="imdb-skewed", choices=["imdb-skewed", "preserve"])
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(ROOT / "generator" / "export_imdb_schema.py"),
        "--base-dir",
        args.base_dir,
        "--out-dir",
        args.out_dir,
        "--strict-job",
        "--no-include-extras",
        "--company-country-policy",
        args.company_country_policy,
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
