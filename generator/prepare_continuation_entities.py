from __future__ import annotations

import argparse
from pathlib import Path

from bootstrap_artifacts import current_mode
from continuation_entities import (
    EntityTopupResult,
    topup_characters,
    topup_companies,
    topup_keywords,
    topup_persons,
    topup_titles,
    write_manifest,
)


def _positive_target(value: str) -> int:
    out = int(value)
    if out < 0:
        raise argparse.ArgumentTypeError("target must be non-negative")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Top up existing Mirage entities for a Step100 continuation run."
    )
    parser.add_argument("--base-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument(
        "--kind",
        choices=("all", "persons", "companies", "keywords", "characters", "titles"),
        required=True,
    )
    parser.add_argument("--target-count", type=_positive_target, default=None)
    parser.add_argument("--target-persons", type=_positive_target, default=None)
    parser.add_argument("--target-companies", type=_positive_target, default=None)
    parser.add_argument("--target-keywords", type=_positive_target, default=None)
    parser.add_argument("--target-characters", type=_positive_target, default=None)
    parser.add_argument("--target-titles", type=_positive_target, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=("research", "debug"), default=current_mode())
    parser.add_argument("--extension-start-year", type=int, default=None)
    parser.add_argument("--extension-end-year", type=int, default=None)
    parser.add_argument(
        "--survivor-share",
        type=float,
        default=0.015,
        help="Share of existing people eligible for a small retirement-year extension.",
    )
    parser.add_argument(
        "--company-lifecycle-policy",
        choices=("balanced", "preserve", "new-era"),
        default="balanced",
        help="Company turnover policy used when preparing continuation entities.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    kind = str(args.kind)
    results: list[EntityTopupResult] = []

    def target_for(name: str) -> int:
        explicit = getattr(args, f"target_{name}")
        if explicit is not None:
            return int(explicit)
        if args.target_count is not None:
            return int(args.target_count)
        raise SystemExit(f"--target-count or --target-{name.replace('_', '-')} is required for {name}")

    if kind in {"all", "persons"}:
        results.append(
            topup_persons(
                base_dir,
                target=target_for("persons"),
                seed=int(args.seed),
                mode=str(args.mode),
                extension_start_year=args.extension_start_year,
                extension_end_year=args.extension_end_year,
                survivor_share=max(0.0, float(args.survivor_share)),
            )
        )
    if kind in {"all", "companies"}:
        results.append(
            topup_companies(
                base_dir,
                target=target_for("companies"),
                seed=int(args.seed),
                mode=str(args.mode),
                extension_start_year=args.extension_start_year,
                extension_end_year=args.extension_end_year,
                company_lifecycle_policy=str(args.company_lifecycle_policy),
            )
        )
    if kind in {"all", "keywords"}:
        results.append(
            topup_keywords(
                base_dir,
                target=target_for("keywords"),
                seed=int(args.seed),
                mode=str(args.mode),
            )
        )
    if kind in {"all", "characters"}:
        results.append(
            topup_characters(
                base_dir,
                target=target_for("characters"),
                seed=int(args.seed),
                mode=str(args.mode),
            )
        )
    if kind in {"all", "titles"}:
        results.append(
            topup_titles(
                base_dir,
                target=target_for("titles"),
                seed=int(args.seed),
                mode=str(args.mode),
                extension_start_year=args.extension_start_year,
                extension_end_year=args.extension_end_year,
            )
        )

    manifest_path = write_manifest(
        base_dir,
        results,
        metadata={
            "kind": kind,
            "mode": str(args.mode),
            "seed": int(args.seed),
            "extension_start_year": args.extension_start_year,
            "extension_end_year": args.extension_end_year,
            "company_lifecycle_policy": str(args.company_lifecycle_policy),
        },
    )

    print("=" * 72)
    print("CONTINUATION ENTITY TOP-UP")
    print("=" * 72)
    for result in results:
        print(
            f"{result.kind:<12} before={result.before:,} target={result.target:,} "
            f"added={result.added:,} after={result.after:,}"
        )
        if result.survivor_extensions:
            print(f"{'':<12} survivor_extensions={result.survivor_extensions:,}")
        if result.lifecycle_updates:
            print(f"{'':<12} lifecycle_updates={result.lifecycle_updates:,}")
        if result.company_dissolutions:
            print(f"{'':<12} company_dissolutions={result.company_dissolutions:,}")
        if result.new_founded:
            print(f"{'':<12} new_founded={result.new_founded:,}")
        if result.new_defunct:
            print(f"{'':<12} new_defunct={result.new_defunct:,}")
        if result.duplicate_candidates:
            print(f"{'':<12} duplicate_candidates_skipped={result.duplicate_candidates:,}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
