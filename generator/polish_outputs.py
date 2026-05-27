from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from feather_sink import df_to_arrow, read_table
from text_polish import sanitize_alternate_title, sanitize_character_name, sanitize_tagline, sanitize_title


BASE_DIR = Path(__file__).resolve().parent


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _write_csv_if_changed(df: pd.DataFrame, path: Path, original: pd.DataFrame, counters: dict[str, int]) -> None:
    if len(df) != len(original) or not df.equals(original):
        df.to_csv(path, index=False)
        counters[str(path)] = int(len(df))


def _polish_df(df: pd.DataFrame, mapping: dict[str, str]) -> tuple[pd.DataFrame, int]:
    if df is None or df.empty:
        return df, 0
    work = df.copy()
    changed = 0
    title_series = work["title"].astype(str) if "title" in work.columns else None
    for column, mode in mapping.items():
        if column not in work.columns:
            continue
        before = work[column].fillna("").astype(str)
        if mode == "title":
            after = before.map(sanitize_title)
        elif mode == "tagline":
            after = [
                sanitize_tagline(value, title=title_series.iloc[idx] if title_series is not None else None)
                for idx, value in enumerate(before.tolist())
            ]
            after = pd.Series(after, index=work.index, dtype=object)
        elif mode == "alt_title":
            after = before.map(sanitize_alternate_title)
        elif mode == "character_name":
            after = before.map(sanitize_character_name)
        else:
            continue
        changed += int((before != after).sum())
        work[column] = after
    return work, changed


def _polish_arrow_table(base_dir: Path, stem: str, table_name: str, mapping: dict[str, str], counters: dict[str, int]) -> None:
    df = read_table(str(base_dir / stem), table_name)
    if df.empty:
        return
    polished, changed = _polish_df(df, mapping)
    if changed > 0:
        df_to_arrow(polished, str(base_dir / f"{stem}.arrow"), table_name=table_name)
        csv_path = base_dir / f"{stem}.csv"
        if csv_path.exists():
            polished.to_csv(csv_path, index=False)
        counters[str(base_dir / f"{stem}.arrow")] = int(changed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministically polish structural text artifacts in current outputs.")
    parser.add_argument("--base-dir", default=str(BASE_DIR))
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    counters: dict[str, int] = {}

    csv_targets = [
        (base_dir / "entities" / "title_bank.csv", {"title": "title", "tagline": "tagline"}),
        (base_dir / "entities" / "character_bank.csv", {"character_name": "character_name"}),
        (base_dir / "movie.csv", {"title": "title", "tagline": "tagline"}),
        (base_dir / "movies_flat.csv", {"title": "title"}),
        (base_dir / "movies_analysis.csv", {"title": "title"}),
        (base_dir / "alternate_titles.csv", {"alt_title": "alt_title"}),
        (base_dir / "imdb_schema" / "title.csv", {"title": "title"}),
        (base_dir / "imdb_schema" / "aka_title.csv", {"title": "alt_title"}),
        (base_dir / "imdb_schema" / "char_name.csv", {"name": "character_name"}),
        (base_dir / "imdb_schema" / "extra_cast_info.csv", {"character_name": "character_name"}),
        (base_dir / "imdb_schema" / "movies_flat.csv", {"title": "title"}),
    ]
    for path, mapping in csv_targets:
        original = _read_csv(path)
        if original.empty:
            continue
        polished, changed = _polish_df(original, mapping)
        if changed > 0:
            _write_csv_if_changed(polished, path, original, counters)

    _polish_arrow_table(base_dir, "movie", "movie", {"title": "title", "tagline": "tagline"}, counters)
    _polish_arrow_table(base_dir, "movies_flat", "movies_flat", {"title": "title"}, counters)
    _polish_arrow_table(base_dir, "movies_analysis", "movies_analysis", {"title": "title"}, counters)
    _polish_arrow_table(base_dir, "alternate_titles", "alternate_titles", {"alt_title": "alt_title"}, counters)
    _polish_arrow_table(base_dir, "cast_info", "cast_info", {"character_name": "character_name"}, counters)

    if counters:
        print("Polished outputs:")
        for path, count in sorted(counters.items()):
            print(f"  {path}: {count}")
    else:
        print("No structural text polish changes were needed.")


if __name__ == "__main__":
    main()
