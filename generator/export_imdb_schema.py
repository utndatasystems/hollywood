"""Convert synthetic movie data into JOB/IMDb-compatible CSVs.

This exporter is Arrow-first and can read the current pipeline outputs directly.
It writes the strict 21-table JOB/IMDb core schema plus optional research extras.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from contracts import COMPANY_COUNTRY_WEIGHTS
from imdb_job_contract import JOB_CORE_TABLES, JOB_TABLE_COLUMNS


def _read_csv(path: Path, required: bool = False) -> pd.DataFrame:
    """Read Arrow IPC (.arrow) if available, else CSV. Backward-compatible."""
    arrow_path = path.with_suffix(".arrow")
    if arrow_path.exists():
        import pyarrow.feather as feather
        return feather.read_table(str(arrow_path)).to_pandas()
    if path.exists():
        return pd.read_csv(path, low_memory=False)
    # Try with .csv suffix if path had no extension
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path, low_memory=False)
    if required:
        raise FileNotFoundError(f"Required source file missing: {path}")
    return pd.DataFrame()


def _source_exists(path: Path) -> bool:
    if path.exists():
        return True
    return path.with_suffix(".arrow").exists() or path.with_suffix(".csv").exists()


def _concat_nonempty(frames: Iterable[pd.DataFrame], *, columns: list[str] | None = None) -> pd.DataFrame:
    usable = [
        frame
        for frame in frames
        if frame is not None and not frame.empty and not frame.dropna(axis=0, how="all").empty
    ]
    if not usable:
        return pd.DataFrame(columns=columns or [])
    if len(usable) == 1:
        return usable[0].reset_index(drop=True)
    normalized = [frame.dropna(axis=1, how="all").copy() for frame in usable]
    merged = pd.concat(normalized, ignore_index=True, sort=False)
    if columns:
        for column in columns:
            if column not in merged.columns:
                merged[column] = None
        merged = merged[columns]
    return merged


def _build_movies_analysis_extra(base_dir: Path) -> pd.DataFrame:
    movie_df = _read_csv(base_dir / "movie", required=False)
    flat_df = _read_csv(base_dir / "movies_flat", required=False)
    if movie_df.empty or flat_df.empty or "title_id" not in movie_df.columns or "title_id" not in flat_df.columns:
        return _read_csv(base_dir / "movies_analysis", required=False)
    analysis = flat_df.copy()
    movie_subset = movie_df.copy()
    for col in [
        "title_id",
        "production_tier",
        "runtime_minutes",
        "certification",
        "num_votes",
        "franchise_id",
        "installment_no",
        "seed",
        "snapshot_id",
    ]:
        if col in movie_subset.columns:
            analysis[col] = movie_subset[col].values
    return analysis


def _safe_int(v, default: int = 0) -> int:
    try:
        if pd.isna(v):
            return default
        return int(v)
    except Exception:
        return default


def _maybe_int(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        return int(v)
    except Exception:
        return None


def _soundex(s: str, length: int = 5) -> str:
    """American Soundex encoding (IMDB-style, variable length up to `length`)."""
    s = "".join(c for c in str(s).upper() if c.isalpha())
    if not s:
        return ""
    _MAP = {
        "B": "1", "F": "1", "P": "1", "V": "1",
        "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
        "D": "3", "T": "3",
        "L": "4",
        "M": "5", "N": "5",
        "R": "6",
    }
    result = [s[0]]
    prev = _MAP.get(s[0], "0")
    for ch in s[1:]:
        code = _MAP.get(ch, "0")
        if code != "0" and code != prev:
            result.append(code)
            if len(result) >= length:
                break
        prev = code
    return "".join(result)


def _split_name_for_pcode(full: str) -> tuple[str, str]:
    """Split a person/character-style name into first and surname tokens."""
    cleaned = str(full or "").strip()
    if "," in cleaned:
        last, first = cleaned.split(",", 1)
        return first.strip(), last.strip()
    parts = [part for part in cleaned.replace("-", " ").split() if part]
    if len(parts) >= 2:
        return parts[0], parts[-1]
    return cleaned, cleaned


def _name_pcodes(full: str) -> tuple[str, str, str]:
    """Return IMDb-like complete, normal-first, and surname phonetic codes."""
    first, last = _split_name_for_pcode(full)
    return _soundex(last + first), _soundex(first), _soundex(last, 4)


_IMDB_INDEX_VALUES = [
    "(I)", "(II)", "(III)", "(IV)", "(V)", "(VI)", "(VII)", "(VIII)",
    "(IX)", "(X)", "(XI)", "(XII)", "(XIII)", "(XIV)", "(XV)", "(XVI)",
    "(XVII)", "(XVIII)", "(XIX)", "(XX)", "(XXI)", "(XXII)", "(XXIII)",
    "(XXIV)", "(XXV)", "(XXVI)", "(XXVII)", "(XXVIII)", "(XXIX)", "(XXX)",
    "(XXXI)", "(XXXII)",
]

_COMPANY_SUFFIX_TOKENS = {
    "cinema", "collective", "company", "entertainment", "film", "films", "group",
    "media", "network", "partners", "pictures", "productions", "studio", "studios",
    "works",
}


def _stable_u32(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.md5(text.encode("utf-8", errors="ignore")).digest()
    return int.from_bytes(digest[:4], "big")


def _stable_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _stable_md5(*parts: object) -> str:
    text = "|".join(_stable_text(part) for part in parts)
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


def _imdb_index_value(namespace: str, row_id: object, label: object, *, density: float) -> str | None:
    """Return a deterministic IMDb disambiguation index for some rows.

    IMDb's ``imdb_index`` is not a key. It is a small disambiguator such as
    ``(I)``/``(II)`` used when titles or names collide.  Synthetic names and
    titles are intentionally diverse, so we materialize a controlled ambiguous
    subset; this keeps JOB-Complex disambiguation joins possible without using
    fake ID-style bridges.
    """
    if not str(label or "").strip():
        return None
    h = _stable_u32(namespace, row_id, label)
    if (h / 0xFFFFFFFF) >= density:
        return None
    return _IMDB_INDEX_VALUES[(h >> 8) % len(_IMDB_INDEX_VALUES)]


def _company_core_text(company: str) -> str:
    tokens = [
        token
        for token in re.split(r"[^A-Za-z0-9]+", str(company or ""))
        if token and token.lower() not in {"the", "a", "an", "and", "of"}
    ]
    while len(tokens) > 1 and tokens[-1].lower() in _COMPANY_SUFFIX_TOKENS:
        tokens.pop()
    return " ".join(tokens).strip()


def _company_pcodes(company: str) -> tuple[str, str]:
    """Return approximate IMDb company phonetic codes.

    The original JOB/JOB-Complex workload uses these columns in several
    non-key joins. Leaving them empty makes those queries structurally
    impossible even when literals are relaxed.  Use the company core with
    common legal/production suffixes stripped, so the normal and suffix-free
    forms intentionally agree for names such as "Apex Works" -> "Apex".
    """
    core = _company_core_text(company)
    if not core:
        return "", ""
    code = _soundex(core)
    return code, code


def _keyword_pcode(keyword: str) -> str | None:
    text = re.sub(r"[-_]+", " ", str(keyword or "")).strip()
    if not text:
        return None
    return _soundex(text)


_CANONICAL_GENRES = [
    "Action", "Adventure", "Animation", "Biography", "Comedy", "Crime",
    "Documentary", "Drama", "Family", "Fantasy", "Film-Noir", "History",
    "Horror", "Music", "Musical", "Mystery", "Romance", "Sci-Fi", "Sport",
    "Thriller", "War", "Western", "Short", "Reality-TV",
]

_GENRE_BY_KEY = {genre.lower(): genre for genre in _CANONICAL_GENRES}
_GENRE_ALIASES = {
    "bio": "Biography",
    "biopic": "Biography",
    # Generated-world subgenres are useful provenance, but strict IMDb/JOB
    # genre rows should use canonical IMDb genre labels.
    "superhero": "Action",
    "martial arts": "Action",
    "martial-arts": "Action",
    "disaster": "Action",
    "experimental": "Drama",
    "film noir": "Film-Noir",
    "noir": "Film-Noir",
    "science fiction": "Sci-Fi",
    "sci fi": "Sci-Fi",
    "sports": "Sport",
}


def _first_multi_value(value) -> str | None:
    parts = _split_multi_values(value)
    if not parts:
        return None
    raw = re.sub(r"\s+", " ", parts[0].replace("_", " ")).strip()
    return raw or None


def _split_multi_values(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []
    return [part.strip() for part in re.split(r"\s*[;|]\s*|\s*,\s*", text) if part.strip()]


def _normalize_genre(value) -> str | None:
    raw = _first_multi_value(value)
    if not raw:
        return None
    key = raw.lower()
    if key in _GENRE_ALIASES:
        return _GENRE_ALIASES[key]
    return _GENRE_BY_KEY.get(key)


def _primary_genre_value(value) -> str | None:
    return _normalize_genre(value)


def _id_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _counter_text(counter: Counter[str]) -> str:
    if not counter:
        return ""
    return ";".join(f"{genre}:{count}" for genre, count in sorted(counter.items()))


def _build_movie_genre_derivations(
    movie: pd.DataFrame,
    movie_keyword: pd.DataFrame,
    keywords: pd.DataFrame,
    movie_companies: pd.DataFrame,
    companies: pd.DataFrame,
) -> tuple[dict[int, list[str]], pd.DataFrame, dict]:
    """Derive strict IMDb genre rows from generated, signal-backed metadata.

    The generated world keeps a single primary movie genre, while keywords and
    company specialties carry additional genre evidence. IMDb stores one
    ``movie_info`` row per genre, so this builds a conservative secondary
    genre set without inventing unsupported genres.
    """
    all_movie_ids: set[int] = set()
    primary_by_movie: dict[int, str] = {}
    source_primary_by_movie: dict[int, str] = {}
    primary_alias_counts: Counter[str] = Counter()
    for tid, genre in zip(movie["title_id"], movie.get("genre", pd.Series([""] * len(movie)))):
        mid = _safe_int(tid)
        if not mid:
            continue
        all_movie_ids.add(mid)
        source_primary = _first_multi_value(genre)
        if source_primary:
            source_primary_by_movie[mid] = source_primary
        primary = _primary_genre_value(genre)
        if primary:
            primary_by_movie[mid] = primary
            if source_primary and source_primary != primary:
                primary_alias_counts[f"{source_primary}->{primary}"] += 1

    movie_ids = all_movie_ids
    candidate_scores: dict[int, Counter[str]] = defaultdict(Counter)
    keyword_signals: dict[int, Counter[str]] = defaultdict(Counter)
    company_signals: dict[int, Counter[str]] = defaultdict(Counter)

    keyword_id_col = _id_column(keywords, ["keyword_id", "id"])
    if (
        keyword_id_col
        and "topic_genre" in keywords.columns
        and not keywords.empty
        and not movie_keyword.empty
    ):
        keyword_genre_by_id = {
            _safe_int(kid): _normalize_genre(topic)
            for kid, topic in zip(keywords[keyword_id_col], keywords["topic_genre"])
        }
        mk_movie_col = _id_column(movie_keyword, ["title_id", "movie_id"])
        mk_keyword_col = _id_column(movie_keyword, ["keyword_id", "id"])
        if mk_movie_col and mk_keyword_col:
            for tid_raw, kid_raw in zip(movie_keyword[mk_movie_col], movie_keyword[mk_keyword_col]):
                mid = _safe_int(tid_raw)
                genre = keyword_genre_by_id.get(_safe_int(kid_raw))
                if mid in movie_ids and genre:
                    candidate_scores[mid][genre] += 3.0
                    keyword_signals[mid][genre] += 1

    company_id_col = _id_column(companies, ["company_id", "id"])
    if (
        company_id_col
        and "specialty_genres" in companies.columns
        and not companies.empty
        and not movie_companies.empty
    ):
        company_genres_by_id: dict[int, list[str]] = {}
        for cid_raw, specialty in zip(companies[company_id_col], companies["specialty_genres"]):
            genres = []
            for part in _split_multi_values(specialty):
                genre = _normalize_genre(part)
                if genre and genre not in genres:
                    genres.append(genre)
            company_genres_by_id[_safe_int(cid_raw)] = genres

        mc_movie_col = _id_column(movie_companies, ["title_id", "movie_id"])
        mc_company_col = _id_column(movie_companies, ["company_id"])
        if mc_movie_col and mc_company_col:
            for tid_raw, cid_raw in zip(movie_companies[mc_movie_col], movie_companies[mc_company_col]):
                mid = _safe_int(tid_raw)
                if mid not in movie_ids:
                    continue
                for genre in company_genres_by_id.get(_safe_int(cid_raw), []):
                    candidate_scores[mid][genre] += 1.0
                    company_signals[mid][genre] += 1

    genres_by_movie: dict[int, list[str]] = {}
    audit_rows: list[dict] = []
    total_secondary = 0
    for mid in sorted(movie_ids):
        primary = primary_by_movie.get(mid)
        selected = []
        for genre, score in sorted(
            candidate_scores.get(mid, Counter()).items(),
            key=lambda item: (-item[1], item[0]),
        ):
            if genre == primary:
                continue
            # A single company specialty is weak evidence. Keywords or repeated
            # company agreement are enough to materialize a secondary genre.
            if keyword_signals[mid].get(genre, 0) <= 0 and score < 2.0:
                continue
            selected.append(genre)
            if len(selected) >= 3:
                break
        exported = ([primary] if primary else []) + selected
        if not exported:
            exported = ["Drama"]
        genres_by_movie[mid] = exported
        total_secondary += max(0, len(exported) - 1)
        audit_rows.append({
            "movie_id": mid,
            "source_primary_genre": source_primary_by_movie.get(mid),
            "primary_genre": primary,
            "primary_genre_canonicalized": bool(
                source_primary_by_movie.get(mid) and primary and source_primary_by_movie.get(mid) != primary
            ),
            "exported_genres": ";".join(exported),
            "genre_count": len(exported),
            "secondary_genres": ";".join(selected),
            "keyword_signal_genres": _counter_text(keyword_signals.get(mid, Counter())),
            "company_signal_genres": _counter_text(company_signals.get(mid, Counter())),
            "candidate_scores_json": json.dumps(
                {genre: float(score) for genre, score in sorted(candidate_scores.get(mid, Counter()).items())},
                sort_keys=True,
            ),
        })

    audit = pd.DataFrame(audit_rows)
    summary = {
        "primary_movie_count": int(len(movie_ids)),
        "genre_row_count": int(sum(len(values) for values in genres_by_movie.values())),
        "secondary_genre_row_count": int(total_secondary),
        "movies_with_secondary_genres": int((audit["genre_count"] > 1).sum()) if not audit.empty else 0,
        "movies_without_primary_genre": int(len(movie_ids) - len(primary_by_movie)),
        "keyword_signal_movie_count": int(len(keyword_signals)),
        "company_signal_movie_count": int(len(company_signals)),
        "max_genres_per_movie": int(audit["genre_count"].max()) if not audit.empty else 0,
        "primary_genre_alias_counts": dict(sorted(primary_alias_counts.items())),
    }
    return genres_by_movie, audit, summary


def _source_export_coverage_matrix(company_country_policy: str = "imdb-skewed") -> pd.DataFrame:
    company_country_status = "preserved" if company_country_policy == "preserve" else "derived"
    company_country_note = (
        "Source company country is exported directly."
        if company_country_policy == "preserve"
        else "Strict JOB company_name.country_code is deterministically projected to an IMDb-like company-market distribution; source country is retained in extras and company_country_export_audit.csv."
    )
    rows = [
        ("movie", "title_id/title/year", "title", "preserved", "Canonical JOB title rows."),
        ("movie", "genre", "movie_info(info_type_id=3)", "derived", "Primary genre is exported as a canonical IMDb genre row; generated subgenres such as Superhero/Martial Arts/Disaster/Experimental are canonicalized and kept verbatim in extras."),
        ("movie_keyword + keyword", "keyword.topic_genre", "movie_info(info_type_id=3)", "derived", "Secondary genres are canonicalized and derived only when keyword links provide topic evidence."),
        ("movie_companies + company", "company.specialty_genres", "movie_info(info_type_id=3)", "derived", "Secondary genres use canonicalized repeated company-specialty evidence as a weak signal."),
        ("movie", "country/language/certification/color/runtime/tagline/plot/aspect_ratio", "movie_info", "preserved", "Mapped to canonical IMDb info_type ids where available."),
        ("movie", "rating/num_votes", "movie_info_idx", "preserved", "Mapped to canonical IMDb numeric info_type ids where available."),
        ("movie", "budget/box_office", "movie_info", "preserved", "Mapped to canonical IMDb budget/gross info_type ids used by JOB predicates."),
        ("movie", "production_tier/original_language/franchise/seed/snapshot", "extra_movie.csv", "extra-only", "Synthetic-only or provenance fields are retained outside strict IMDb core."),
        ("locations", "city/country/location_type", "movie_info(info_type_id=18)", "preserved", "Generated filming/location rows map to IMDb locations."),
        ("tv_series", "genre/country/language/content_rating/plot_summary/network", "movie_info + company_name + movie_companies", "derived", "TV-series title rows keep canonical metadata; unmapped generated network names are synthesized as IMDb company rows with source names preserved in extras."),
        ("episodes", "episode_number/runtime/director/writer/description/air_date", "title + movie_info + cast_info", "preserved", "Episode numbering and crew facts map to canonical IMDb columns/rows; release-date country is inherited from the parent series."),
        ("cast_info", "person_id/title_id/character_name/billing_order", "cast_info + char_name", "preserved", "Canonical cast rows and character names."),
        ("cast_info", "character_description/archetype/screen_time_minutes/salary_usd", "extra_cast_info.csv", "extra-only", "Generated role richness is retained outside strict JOB."),
        ("movie_companies", "company_id/title_id/role", "movie_companies", "preserved", "Roles are mapped to IMDb company_type ids."),
        ("movie_companies", "source role/year/media note", "movie_companies.note", "derived", "IMDb-style notes are deterministic derivations."),
        ("company", "name", "company_name.name", "preserved", "Canonical company names."),
        ("company", "country", "company_name.country_code", company_country_status, company_country_note),
        ("company", "specialty_genres/tier/founded_year/defunct_year/style prefs", "extra_company_metadata.csv", "extra-only", "Company lifecycle and preferences are not strict JOB columns."),
        ("person", "name/gender", "name", "preserved", "Canonical person table with phonetic codes."),
        ("person", "career/style/market fields", "extra_person_metadata.csv", "extra-only", "Generated person profile remains inspectable outside strict JOB."),
        ("keyword", "keyword", "keyword", "preserved", "Canonical keyword table with phonetic codes."),
        ("keyword", "topic_genre/bucket/motif/specificity/scope", "extra_keyword_metadata.csv", "extra-only", "Keyword semantics remain available for audits."),
        ("release_dates", "release_date/country/type", "movie_info(info_type_id=16)", "derived", "Converted to IMDb-style release-date strings."),
        ("ratings_breakdown/reviews/awards/timelines", "*", "extra_*.csv", "extra-only", "Research metadata retained when --include-extras is enabled."),
    ]
    return pd.DataFrame(rows, columns=["source_table", "source_field", "export_target", "status", "notes"])


def _country_code(country: str) -> str:
    mapping = {
        "USA": "us", "United States": "us", "UK": "gb", "United Kingdom": "gb", "India": "in",
        "Japan": "jp", "South Korea": "kr", "Korea": "kr", "China": "cn", "France": "fr",
        "Germany": "de", "Italy": "it", "Spain": "es", "Canada": "ca", "Australia": "au",
        "Brazil": "br", "Mexico": "mx", "Russia": "ru", "Sweden": "se", "Denmark": "dk",
        "Norway": "no", "Finland": "fi", "Netherlands": "nl", "Belgium": "be", "Switzerland": "ch",
        "Austria": "at", "Portugal": "pt", "Czech Republic": "cz", "Hungary": "hu", "Romania": "ro",
        "Ukraine": "ua", "Poland": "pl", "Turkey": "tr", "Greece": "gr", "Argentina": "ar",
        "Colombia": "co", "Chile": "cl", "Peru": "pe", "Egypt": "eg", "South Africa": "za",
        "Nigeria": "ng", "Kenya": "ke", "Morocco": "ma", "Thailand": "th", "Indonesia": "id",
        "Philippines": "ph", "Vietnam": "vn", "Malaysia": "my", "Singapore": "sg", "Iran": "ir",
        "Israel": "il", "Saudi Arabia": "sa", "UAE": "ae", "Pakistan": "pk", "Bangladesh": "bd",
        "New Zealand": "nz", "Ireland": "ie", "Hong Kong": "hk", "Taiwan": "tw",
        "Armenia": "am", "Azerbaijan": "az", "Bahrain": "bh", "Belarus": "by", "Benin": "bj",
        "Bolivia": "bo", "Bulgaria": "bg", "Cape Verde": "cv", "Congo": "cg", "Costa Rica": "cr",
        "Croatia": "hr", "Cuba": "cu", "Dominican Republic": "do", "Ecuador": "ec", "El Salvador": "sv",
        "Estonia": "ee", "Ethiopia": "et", "Georgia": "ge", "Ghana": "gh", "Guatemala": "gt",
        "Honduras": "hn", "Iceland": "is", "Iraq": "iq", "Jordan": "jo", "Kazakhstan": "kz",
        "Kuwait": "kw", "Latvia": "lv", "Lebanon": "lb", "Lithuania": "lt", "Luxembourg": "lu",
        "Macedonia": "mk", "Mongolia": "mn", "Nepal": "np", "Oman": "om", "Panama": "pa",
        "Puerto Rico": "pr", "Qatar": "qa", "Senegal": "sn", "Serbia": "rs", "Slovakia": "sk",
        "Slovenia": "si", "Sri Lanka": "lk", "Syria": "sy", "Tanzania": "tz", "Tunisia": "tn",
        "Uruguay": "uy", "Uzbekistan": "uz", "Venezuela": "ve", "Zimbabwe": "zw",
    }
    code = mapping.get(str(country).strip(), "xx")
    return f"[{code}]"  # IMDB uses bracketed format: [us], [de], [jp]


def _stable_unit_interval(*parts: object) -> float:
    digest = hashlib.blake2b(
        "|".join(str(part) for part in parts).encode("utf-8", errors="ignore"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big") / float(2**64 - 1)


def _weighted_country_from_unit(unit: float, weights: dict[str, float]) -> str:
    total = sum(float(value) for value in weights.values() if float(value) > 0)
    if total <= 0:
        return "USA"
    threshold = max(0.0, min(1.0, float(unit))) * total
    running = 0.0
    last_country = "USA"
    for country, weight in sorted(weights.items()):
        w = float(weight)
        if w <= 0:
            continue
        running += w
        last_country = country
        if running >= threshold:
            return country
    return last_country


def _company_country_export(companies: pd.DataFrame, policy: str) -> tuple[pd.Series, pd.DataFrame, dict]:
    """Return exported IMDb company country codes plus an audit table.

    The generator now uses COMPANY_COUNTRY_WEIGHTS for future workspaces, but
    older 200K artifacts contain nearly uniform company countries.  For strict
    JOB exports we therefore make the company-country surface explicit:
    preserve source countries when requested, or project them to an IMDb-like
    production-market distribution while retaining source countries in extras.
    """
    if "country" in companies.columns:
        source_countries = companies["country"].fillna("").astype(str)
    else:
        source_countries = pd.Series(["USA"] * len(companies), index=companies.index)
    company_ids = pd.to_numeric(companies.get("company_id", pd.Series(np.arange(1, len(companies) + 1))), errors="coerce").fillna(0).astype(int)
    names = companies.get("name", pd.Series([""] * len(companies))).fillna("").astype(str)

    if policy == "preserve":
        export_countries = source_countries.where(source_countries.str.strip() != "", "USA")
    elif policy == "imdb-skewed":
        export_countries = pd.Series(
            [
                _weighted_country_from_unit(
                    _stable_unit_interval("company-country", cid, name),
                    COMPANY_COUNTRY_WEIGHTS,
                )
                for cid, name in zip(company_ids, names)
            ],
            index=companies.index,
        )
    else:
        raise ValueError(f"Unknown company country policy: {policy}")

    export_codes = export_countries.map(_country_code)
    audit = pd.DataFrame({
        "company_id": company_ids.astype(int),
        "name": names,
        "source_country": source_countries,
        "export_country": export_countries,
        "export_country_code": export_codes,
        "policy": policy,
        "source_country_preserved": source_countries.fillna("").astype(str).str.strip().str.casefold()
        == export_countries.fillna("").astype(str).str.strip().str.casefold(),
    })
    summary = {
        "policy": policy,
        "source_country_counts_top20": {str(k): int(v) for k, v in source_countries.value_counts().head(20).items()},
        "export_country_code_counts_top20": {str(k): int(v) for k, v in export_codes.value_counts().head(20).items()},
        "source_country_preserved_count": int(audit["source_country_preserved"].sum()),
        "company_count": int(len(audit)),
    }
    return export_codes, audit, summary


def _role_id_from_gender(gender: str) -> int:
    g = str(gender).strip().lower()
    if g in {"f", "female", "woman"}:
        return 2
    return 1


def _crew_role_to_role_id(crew_role: str) -> int:
    """Map pipeline crew_role to real IMDB role_type.id."""
    mapping = {
        "producer": 3,
        "writer": 4,
        "cinematographer": 5,
        "composer": 6,
        "costume_designer": 7,
        # 8 = director (handled separately)
        "editor": 9,
        "production_designer": 11,
        "production designer": 11,
        # 10 = miscellaneous crew (default fallback)
        # 12 = guest
    }
    return mapping.get(str(crew_role).strip().lower(), 10)


def _link_type_id(link_type: str) -> int:
    lt = str(link_type).strip().lower().replace("_", " ")
    mapping = {
        "follows": 1,
        "prequel": 1,
        "followed by": 2,
        "sequel": 2,
        "remake": 3,
        "remake of": 3,
        "remade as": 4,
        "references": 5,
        "referenced in": 6,
        "spiritual successor": 1,
        "spoofs": 7,
        "spoofed in": 8,
        "features": 9,
        "featured in": 10,
        "spin-off from": 11,
        "spin off from": 11,
        "spin-off": 12,
        "spin off": 12,
        "shared universe": 9,
        "version of": 13,
        "similar": 14,
        "similar to": 14,
        "edited into": 15,
        "edited from": 16,
        "alternate language version of": 17,
    }
    return mapping.get(lt, 18)


def _company_type_id(role: str) -> int:
    """Map to real IMDB company_type: 1=distributors, 2=production, 3=sfx, 4=misc."""
    r = str(role).strip().lower()
    if "distrib" in r:
        return 1
    if "effect" in r or "vfx" in r:
        return 3
    if "production" in r or "producer" in r:
        return 2
    return 4


def _write_table(df: pd.DataFrame, table: str, out_dir: Path):
    cols = JOB_TABLE_COLUMNS[table]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]
    df.to_csv(out_dir / f"{table}.csv", index=False)
    print(f"  OK  {table:<16} {len(df):>10,} rows")


def _validate_core_headers(out_dir: Path):
    missing = []
    wrong = []
    for table, cols in JOB_TABLE_COLUMNS.items():
        p = out_dir / f"{table}.csv"
        if not p.exists():
            missing.append(table)
            continue
        got = list(pd.read_csv(p, nrows=0).columns)
        if got != cols:
            wrong.append((table, cols, got))
    if missing or wrong:
        lines = []
        if missing:
            lines.append("Missing JOB tables: " + ", ".join(sorted(missing)))
        for table, exp, got in wrong:
            lines.append(f"Header mismatch for {table}: expected {exp}, got {got}")
        raise RuntimeError("\n".join(lines))


def _make_static_tables() -> Dict[str, pd.DataFrame]:
    # All IDs match real IMDB exactly
    kind_type = pd.DataFrame([
        {"id": 1, "kind": "movie"},
        {"id": 2, "kind": "tv series"},
        {"id": 3, "kind": "tv movie"},
        {"id": 4, "kind": "video movie"},
        {"id": 5, "kind": "tv mini series"},
        {"id": 6, "kind": "video game"},
        {"id": 7, "kind": "episode"},
    ])
    role_type = pd.DataFrame([
        {"id": 1, "role": "actor"},
        {"id": 2, "role": "actress"},
        {"id": 3, "role": "producer"},
        {"id": 4, "role": "writer"},
        {"id": 5, "role": "cinematographer"},
        {"id": 6, "role": "composer"},
        {"id": 7, "role": "costume designer"},
        {"id": 8, "role": "director"},
        {"id": 9, "role": "editor"},
        {"id": 10, "role": "miscellaneous crew"},   # was "guest" -- fixed to real IMDB
        {"id": 11, "role": "production designer"},   # was "miscellaneous" -- fixed to real IMDB
        {"id": 12, "role": "guest"},                 # was missing -- added to match real IMDB
    ])
    company_type = pd.DataFrame([
        {"id": 1, "kind": "distributors"},
        {"id": 2, "kind": "production companies"},
        {"id": 3, "kind": "special effects companies"},
        {"id": 4, "kind": "miscellaneous companies"},
    ])
    link_type = pd.DataFrame([
        {"id": 1, "link": "follows"},
        {"id": 2, "link": "followed by"},
        {"id": 3, "link": "remake of"},
        {"id": 4, "link": "remade as"},
        {"id": 5, "link": "references"},
        {"id": 6, "link": "referenced in"},
        {"id": 7, "link": "spoofs"},
        {"id": 8, "link": "spoofed in"},
        {"id": 9, "link": "features"},
        {"id": 10, "link": "featured in"},
        {"id": 11, "link": "spin off from"},     # real IMDB has no hyphen
        {"id": 12, "link": "spin off"},              # real IMDB has no hyphen
        {"id": 13, "link": "version of"},
        {"id": 14, "link": "similar to"},
        {"id": 15, "link": "edited into"},      # was "edited from" -- fixed to real IMDB
        {"id": 16, "link": "edited from"},       # was "edited into" -- fixed to real IMDB
        {"id": 17, "link": "alternate language version of"},
        {"id": 18, "link": "unknown link"},
    ])
    # IDs 1-113 match the real IMDB info_type table exactly.
    # IDs 200+ are synthetic-only extensions (no collision with real IMDB).
    info_type = pd.DataFrame([
        # ── movie_info (real IMDB IDs) ──
        {"id": 1, "info": "runtimes"},
        {"id": 2, "info": "color info"},        # was 7 -- fixed to real IMDB
        {"id": 3, "info": "genres"},
        {"id": 4, "info": "languages"},
        {"id": 5, "info": "certificates"},       # was 16 -- fixed to real IMDB
        {"id": 7, "info": "tech info"},
        {"id": 8, "info": "countries"},
        {"id": 9, "info": "taglines"},           # was 99 -- fixed to real IMDB
        {"id": 16, "info": "release dates"},     # JOB uses this in 20 queries
        {"id": 17, "info": "trivia"},            # JOB uses this in 2 queries
        {"id": 18, "info": "locations"},
        {"id": 70, "info": "LD aspect ratio"},
        {"id": 98, "info": "plot"},              # was 100 -- real IMDB id=98
        # ── movie_info_idx (real IMDB IDs) ──
        {"id": 99, "info": "votes distribution"},  # real IMDB: unused by us but defines the slot
        {"id": 100, "info": "votes"},             # was 101 -- fixed to real IMDB
        {"id": 101, "info": "rating"},            # was 102 -- fixed to real IMDB
        {"id": 105, "info": "budget"},            # was 103 -- fixed to real IMDB
        {"id": 107, "info": "gross"},             # was 104 ("box office") -- real IMDB id=107
        {"id": 112, "info": "top 250 rank"},     # JOB uses this in 2 queries
        {"id": 113, "info": "bottom 10 rank"},   # JOB uses this in 3 queries
        # ── person_info (real IMDB IDs) ──
        {"id": 19, "info": "mini biography"},     # was 501 ("biography") -- real IMDB id=19
        {"id": 20, "info": "birth notes"},        # was 22 -- fixed to real IMDB
        {"id": 21, "info": "birth date"},         # correct already
        {"id": 22, "info": "height"},             # was 34 -- fixed to real IMDB
        {"id": 23, "info": "death date"},         # was 26 -- fixed to real IMDB
    ])
    comp_cast_type = pd.DataFrame([
        {"id": 1, "kind": "cast"},
        {"id": 2, "kind": "crew"},
        {"id": 3, "kind": "complete"},
        {"id": 4, "kind": "complete+verified"},
    ])
    return {
        "kind_type": kind_type,
        "role_type": role_type,
        "company_type": company_type,
        "link_type": link_type,
        "info_type": info_type,
        "comp_cast_type": comp_cast_type,
    }


def _generate_aka_name(name_df: pd.DataFrame, rng: np.random.RandomState) -> pd.DataFrame:
    rows: List[Dict] = []
    next_id = 1
    for r in name_df.itertuples(index=False):
        pid = int(getattr(r, "id"))
        full = str(getattr(r, "name", "")).strip()
        if not full:
            continue
        if rng.rand() > 0.12:
            continue
        parts = [p for p in full.replace("-", " ").split() if p]
        if len(parts) == 1:
            aliases = [f"{parts[0]} {parts[0]}"]
        else:
            first, last = parts[0], parts[-1]
            aliases = [
                f"{first[0]}. {last}",
                f"{last}, {first}",
                f"{first} {last[0]}",
            ]
        rng.shuffle(aliases)
        take = 1 if rng.rand() < 0.72 else 2
        seen = set()
        for alias in aliases[:take]:
            alias = alias.strip()
            if not alias or alias == full or alias in seen:
                continue
            seen.add(alias)
            pcode_cf, pcode_nf, surname_pcode = _name_pcodes(alias)
            rows.append({
                "id": next_id,
                "person_id": pid,
                "name": alias,
                "imdb_index": None,
                "name_pcode_cf": pcode_cf,
                "name_pcode_nf": pcode_nf,
                "surname_pcode": surname_pcode,
                "md5sum": hashlib.md5(alias.encode("utf-8", errors="ignore")).hexdigest(),
            })
            next_id += 1
    return pd.DataFrame(rows)


def _build_complete_cast(movie_df: pd.DataFrame, cast_df: pd.DataFrame, crew_df: pd.DataFrame, directors_df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.RandomState(777)
    cast_counts = cast_df.groupby("title_id").size().to_dict() if not cast_df.empty else {}
    crew_counts = {}
    if not crew_df.empty and "title_id" in crew_df.columns:
        crew_counts = crew_df.groupby("title_id").size().to_dict()
    dir_counts = directors_df.groupby("title_id").size().to_dict() if not directors_df.empty else {}

    tier_bonus = {
        "Epic": 0.55,
        "A": 0.40,
        "Mid": 0.25,
        "Indie": 0.15,
        "Micro": 0.05,
    }

    rows: List[Dict] = []
    rid = 1
    for m in movie_df.itertuples(index=False):
        mid = int(getattr(m, "title_id"))
        tier = str(getattr(m, "production_tier", "Mid"))
        c_n = int(cast_counts.get(mid, 0))
        crew_n = int(crew_counts.get(mid, 0) + dir_counts.get(mid, 0))

        score = np.log1p(c_n) * 0.25 + np.log1p(crew_n) * 0.18 + tier_bonus.get(tier, 0.2)
        verified_p = float(np.clip(0.12 + 0.22 * score, 0.08, 0.90))

        # Subject 1: cast
        status_id = 4 if rng.rand() < verified_p else 3
        rows.append({"id": rid, "movie_id": mid, "subject_id": 1, "status_id": status_id})
        rid += 1

        # Subject 2: crew (probabilistic-rich; include often but not always)
        if crew_n > 0 or rng.rand() < 0.42:
            crew_verified_p = float(np.clip(verified_p - 0.08 + (0.03 if crew_n > 4 else 0.0), 0.05, 0.85))
            status_id = 4 if rng.rand() < crew_verified_p else 3
            rows.append({"id": rid, "movie_id": mid, "subject_id": 2, "status_id": status_id})
            rid += 1

    return pd.DataFrame(rows)


def convert(
    base_dir: Path,
    out_dir: Path,
    strict_job: bool,
    include_extras: bool,
    company_country_policy: str = "imdb-skewed",
):
    ent = base_dir / "entities"

    movie = _read_csv(base_dir / "movie.csv", required=True)

    # Build movie year lookup for _mc_note
    _movie_year_map = dict(zip(
        pd.to_numeric(movie["title_id"], errors="coerce").fillna(0).astype(int),
        pd.to_numeric(movie.get("year", 0), errors="coerce").fillna(0).astype(int),
    )) if "year" in movie.columns else {}
    _movie_genre_source = movie.get("genre", pd.Series([""] * len(movie))).fillna("").astype(str)
    _movie_genre_export = _movie_genre_source.map(lambda value: _primary_genre_value(value) or str(value))
    _movie_genre_map = dict(zip(
        pd.to_numeric(movie["title_id"], errors="coerce").fillna(0).astype(int),
        _movie_genre_export,
    ))
    _movie_country_map = dict(zip(
        pd.to_numeric(movie["title_id"], errors="coerce").fillna(0).astype(int),
        movie.get("country", pd.Series(["USA"] * len(movie))).fillna("USA").astype(str),
    ))
    _movie_tier_map = dict(zip(
        pd.to_numeric(movie["title_id"], errors="coerce").fillna(0).astype(int),
        movie.get("production_tier", pd.Series(["Mid"] * len(movie))).fillna("Mid").astype(str),
    ))

    def _actor_cast_note(row: pd.Series) -> str:
        """Build IMDB-style cast_info.note with role annotations.

        Real IMDB notes look like: '(voice)', '(uncredited)', '(as John Smith)',
        '(voice: English version)', '(archive footage)'.
        JOB queries filter on these patterns.
        """
        row_idx = int(getattr(row, "name", 0) or 0)
        tid = _safe_int(row.get("title_id", 0))
        genre = str(_movie_genre_map.get(tid, "") or "")
        char_name = str(row.get("_char", "") or "").strip()
        h = ((tid + 17) * 1000003 + (row_idx + 1) * 2654435761) & 0xFFFFFFFF
        frac = (h & 0xFFFF) / 0xFFFF

        # Animation and animated-adjacent genres need exact voice notes, not
        # free-text character descriptions, because JOB filters on equality.
        voice_p = 0.42 if genre == "Animation" else 0.10 if genre in {"Fantasy", "Adventure", "Sci-Fi"} else 0.025
        if frac < voice_p:
            voice_notes = ["(voice)", "(voice)", "(voice: English version)", "(voice) (uncredited)"]
            return voice_notes[(h >> 17) % len(voice_notes)]
        if frac < voice_p + 0.035:
            return "(uncredited)"
        if char_name and char_name not in {"Unknown", "nan", "None"} and frac < voice_p + 0.075:
            return f"(as {char_name})"
        return None

    def _crew_credit_note(role: str, row_idx: int) -> str | None:
        role_key = str(role or "").strip().lower().replace("_", " ")
        h = ((row_idx + 11) * 2654435761) & 0xFFFFFFFF
        if role_key == "writer":
            choices = ["(writer)", "(written by)", "(story)", "(story editor)", "(head writer)"]
            return choices[(h >> 13) % len(choices)]
        if role_key == "producer":
            return "(executive producer)" if (h & 7) == 0 else "(producer)"
        if role_key == "composer":
            return "(composer)"
        if role_key == "cinematographer":
            return "(cinematographer)"
        if role_key == "editor":
            return "(editor)"
        return None

    def _mc_note(role: str, title_id, movie_df) -> str:
        """Build IMDB-style movie_companies.note with year and production type.

        Real IMDB notes look like: '(presents)', '(co-production)', '(2006) (USA) (theatrical)',
        '(2007) (DVD)'. JOB queries filter LIKE '%(co-production)%', '%(200%)%', etc.
        """
        role_str = str(role).strip().lower() if role else ""
        tid = _safe_int(title_id)
        year = _movie_year_map.get(tid, 0)

        parts = []
        h = tid * 2654435761 & 0xFFFFFFFF
        frac = (h & 0xFFFF) / 0xFFFF

        if "production" in role_str:
            # Real IMDb production-company credits are often unnoted. Keep most
            # primary production rows NULL so JOB sequel/company predicates with
            # mc.note IS NULL are representable; reserve notes for actual credit
            # qualifiers such as co-production/presents.
            if "co" in role_str:
                parts.append("(co-production)")
            elif frac < 0.08:
                parts.append("(co-production)")
            elif frac < 0.16:
                parts.append("(presents)")
        elif "distrib" in role_str:
            if year > 0:
                parts.append(f"({year})")
            origin = str(_movie_country_map.get(tid, "USA") or "USA")
            origin_note = f"({origin})" if origin else "(USA)"
            if origin == "USA":
                region_pool = ["(USA)", "(worldwide)", "(USA)", "(worldwide)", "(Canada)", "(UK)"]
            else:
                region_pool = [origin_note, "(worldwide)", "(USA)", origin_note, "(UK)", "(Germany)", "(Japan)"]
            parts.append(region_pool[tid % len(region_pool)])
            media = ["(theatrical)", "(DVD)", "(Blu-ray)", "(TV)", "(video)", "(VHS)"]
            parts.append(media[tid % len(media)])

        return " ".join(parts) if parts else None

    cast = _read_csv(base_dir / "cast_info.csv", required=True)
    movie_companies = _read_csv(base_dir / "movie_companies.csv", required=True)
    movie_keyword = _read_csv(base_dir / "movie_keyword.csv", required=True)
    movie_links = _read_csv(base_dir / "movie_links.csv")
    aka_titles = _read_csv(base_dir / "alternate_titles.csv")
    movie_crew = _read_csv(base_dir / "movie_crew.csv")
    movie_directors = _read_csv(base_dir / "movie_directors.csv")
    person_demo = _read_csv(base_dir / "person_demographics.csv")
    tv_series = _read_csv(base_dir / "tv_series.csv")
    episodes = _read_csv(base_dir / "episodes.csv")
    episode_cast = _read_csv(base_dir / "episode_cast.csv")
    release_dates = _read_csv(base_dir / "release_dates.csv")
    locations = _read_csv(base_dir / "locations.csv")

    persons = _read_csv(base_dir / "persons_enriched.csv")
    if persons.empty:
        persons = _read_csv(ent / "person.csv", required=True)

    companies = _read_csv(base_dir / "companies_enriched.csv")
    if companies.empty:
        companies = _read_csv(ent / "company.csv", required=True)

    keywords = _read_csv(ent / "keyword.csv", required=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    for old_csv in out_dir.glob("*.csv"):
        try:
            old_csv.unlink()
        except PermissionError:
            print(f"  WARN locked CSV not removed: {old_csv.name} (continuing)")
    manifest_path = out_dir / "export_manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()
    # Standardize key IDs.
    if "person_id" not in persons.columns:
        persons["person_id"] = np.arange(1, len(persons) + 1)
    if "company_id" not in companies.columns:
        companies["company_id"] = np.arange(1, len(companies) + 1)

    genres_by_movie, genre_derivation_audit, genre_derivation_summary = _build_movie_genre_derivations(
        movie=movie,
        movie_keyword=movie_keyword,
        keywords=keywords,
        movie_companies=movie_companies,
        companies=companies,
    )
    source_export_coverage = _source_export_coverage_matrix(company_country_policy=company_country_policy)

    static = _make_static_tables()

    # title (movies)
    title_movies = pd.DataFrame({
        "id": movie["title_id"].astype(int),
        "title": movie["title"].astype(str),
        "imdb_index": None,
        "kind_id": 1,
        "production_year": pd.to_numeric(movie.get("year"), errors="coerce").astype("Int64"),
        "imdb_id": movie["title_id"].astype(int),
        "phonetic_code": None,
        "episode_of_id": None,
        "season_nr": None,
        "episode_nr": None,
        "series_years": None,
        "md5sum": None,
    })

    # title (TV series + episodes)
    title_parts = [title_movies]
    max_title_id = int(title_movies["id"].max()) if len(title_movies) > 0 else 0

    # Map series_id -> title.id so episodes can reference via episode_of_id
    series_id_to_title_id = {}
    episode_id_to_title_id = {}
    if not tv_series.empty and "series_id" in tv_series.columns:
        series_title_ids = np.arange(max_title_id + 1, max_title_id + 1 + len(tv_series))
        for sid, tid in zip(tv_series["series_id"].astype(int), series_title_ids):
            series_id_to_title_id[sid] = int(tid)
        year_start = pd.to_numeric(tv_series.get("year_start"), errors="coerce").astype("Int64")
        year_end = pd.to_numeric(tv_series.get("year_end"), errors="coerce").astype("Int64")
        series_years_str = (year_start.astype(object).astype(str) + "-" + year_end.astype(object).fillna("").astype(str)).where(year_start.notna(), None)
        title_tv = pd.DataFrame({
            "id": series_title_ids,
            "title": tv_series["title"].astype(str),
            "imdb_index": None,
            "kind_id": 2,  # tv series
            "production_year": year_start,
            "imdb_id": series_title_ids,
            "phonetic_code": None,
            "episode_of_id": None,
            "season_nr": None,
            "episode_nr": None,
            "series_years": series_years_str,
            "md5sum": None,
        })
        title_parts.append(title_tv)
        max_title_id = int(series_title_ids.max())
        print(f"  +   TV series in title:     {len(title_tv):,} rows (kind_id=2)")

    if not episodes.empty and "episode_id" in episodes.columns:
        ep_title_ids = np.arange(max_title_id + 1, max_title_id + 1 + len(episodes))
        for eid, tid in zip(pd.to_numeric(episodes["episode_id"], errors="coerce").fillna(0).astype(int), ep_title_ids):
            episode_id_to_title_id[int(eid)] = int(tid)
        ep_series_ids = pd.to_numeric(episodes.get("series_id"), errors="coerce").fillna(0).astype(int)
        ep_order = episodes.copy()
        ep_order["_orig_pos"] = np.arange(len(ep_order))
        sort_cols = [
            col
            for col in ["series_id", "air_date", "season_id", "episode_number", "episode_id"]
            if col in ep_order.columns
        ]
        if sort_cols:
            ep_order = ep_order.sort_values(sort_cols, kind="stable")
        ep_abs = (ep_order.groupby("series_id", sort=False).cumcount() + 1).astype(int)
        episode_abs_number = pd.Series(ep_abs.values, index=ep_order["_orig_pos"]).sort_index().astype("Int64")
        episode_number = pd.to_numeric(episodes.get("episode_number", None), errors="coerce").astype("Int64") if "episode_number" in episodes.columns else pd.Series([pd.NA] * len(episodes), dtype="Int64")
        season_number = pd.to_numeric(episodes.get("season_id", None), errors="coerce").astype("Int64") if "season_id" in episodes.columns else pd.Series([pd.NA] * len(episodes), dtype="Int64")
        title_episodes = pd.DataFrame({
            "id": ep_title_ids,
            "title": episodes["title"].astype(str) if "title" in episodes.columns else "Episode",
            "imdb_index": None,
            "kind_id": 7,  # episode (real IMDB)
            "production_year": pd.to_numeric(episodes.get("air_date", "").astype(str).str[:4], errors="coerce").astype("Int64") if "air_date" in episodes.columns else None,
            "imdb_id": ep_title_ids,
            "phonetic_code": None,
            "episode_of_id": ep_series_ids.map(series_id_to_title_id).astype("Int64"),
            "season_nr": season_number,
            "episode_nr": episode_number,
            "series_years": None,
            "md5sum": None,
        })
        # Fix season_nr: derive from seasons table if available
        seasons = _read_csv(base_dir / "seasons.csv")
        if not seasons.empty and "season_id" in seasons.columns and "season_id" in episodes.columns:
            season_map = dict(zip(
                pd.to_numeric(seasons["season_id"], errors="coerce").fillna(0).astype(int),
                pd.to_numeric(seasons["season_number"], errors="coerce").fillna(1).astype(int),
            ))
            ep_season_ids = pd.to_numeric(episodes["season_id"], errors="coerce").fillna(0).astype(int)
            title_episodes["season_nr"] = ep_season_ids.map(season_map).astype("Int64")

        title_parts.append(title_episodes)
        max_title_id = int(ep_title_ids.max())
        print(f"  +   episodes in title:      {len(title_episodes):,} rows (kind_id=7)")

    title = _concat_nonempty(title_parts, columns=JOB_TABLE_COLUMNS["title"])
    if not title.empty:
        title["imdb_index"] = [
            _imdb_index_value(
                "title",
                getattr(row, "id"),
                f"{getattr(row, 'title', '')}|{getattr(row, 'kind_id', '')}|{getattr(row, 'production_year', '')}",
                density=0.14,
            )
            for row in title.itertuples(index=False)
        ]
        title["phonetic_code"] = [_soundex(value) for value in title["title"].astype(str)]
        title["md5sum"] = [
            _stable_md5(
                "title",
                getattr(row, "id"),
                getattr(row, "title", ""),
                getattr(row, "kind_id", ""),
                getattr(row, "production_year", ""),
                getattr(row, "episode_of_id", ""),
                getattr(row, "season_nr", ""),
                getattr(row, "episode_nr", ""),
            )
            for row in title.itertuples(index=False)
        ]
    title_imdb_index_by_id = {
        int(row.id): row.imdb_index
        for row in title.itertuples(index=False)
        if pd.notna(getattr(row, "imdb_index", None)) and str(getattr(row, "imdb_index", "")).strip()
    }

    # name — with phonetic codes (Soundex) matching real IMDB
    # IMDB names are "Last, First" format. We split to compute codes.
    person_names = persons["name"].astype(str)
    pcodes = [_name_pcodes(n) for n in person_names]
    pcode_cf = [item[0] for item in pcodes]  # complete flattened name
    pcode_nf = [item[1] for item in pcodes]  # normal (first) name
    surname_pcode = [item[2] for item in pcodes]  # surname
    name_imdb_index = [
        _imdb_index_value("name", pid, label, density=0.10)
        for pid, label in zip(persons["person_id"].astype(int), person_names)
    ]

    name = pd.DataFrame({
        "id": persons["person_id"].astype(int),
        "name": person_names,
        "imdb_index": name_imdb_index,
        "imdb_id": persons["person_id"].astype(int),
        "gender": (persons["gender"].astype(str).str.strip().str.lower().str[0]
                  .map(lambda g: g if g in ("m", "f") else None)
                  if "gender" in persons.columns else None),
        "name_pcode_cf": pcode_cf,
        "name_pcode_nf": pcode_nf,
        "surname_pcode": surname_pcode,
        "md5sum": [
            _stable_md5("name", pid, value)
            for pid, value in zip(persons["person_id"].astype(int), person_names)
        ],
    })

    # char_name and cast_info
    cast_char = cast.get("character_name", pd.Series(["Unknown"] * len(cast))).fillna("Unknown").astype(str).str.strip()
    cast_char = cast_char.replace("", "Unknown")
    char_unique = pd.Series(sorted(set(cast_char.tolist())))
    char_name = pd.DataFrame({
        "id": np.arange(1, len(char_unique) + 1),
        "name": char_unique,
        "imdb_index": [
            _imdb_index_value("char_name", idx, value, density=0.05)
            for idx, value in enumerate(char_unique, 1)
        ],
        "imdb_id": np.arange(1, len(char_unique) + 1),
        "name_pcode_nf": [_name_pcodes(value)[1] for value in char_unique],
        "surname_pcode": [_name_pcodes(value)[2] for value in char_unique],
        "md5sum": [
            _stable_md5("char_name", idx, value)
            for idx, value in enumerate(char_unique, 1)
        ],
    })
    name_imdb_index_by_id = {
        int(row.id): row.imdb_index
        for row in name.itertuples(index=False)
        if pd.notna(getattr(row, "imdb_index", None)) and str(getattr(row, "imdb_index", "")).strip()
    }
    char_map = dict(zip(char_name["name"], char_name["id"]))
    person_gender = dict(zip(name["id"], name["gender"]))

    cast_work = cast.copy()
    if "billing_order" not in cast_work.columns:
        cast_work["billing_order"] = cast_work.groupby("title_id").cumcount() + 1
    if "character_description" not in cast_work.columns:
        cast_work["character_description"] = cast_work.get("archetype", "")
    cast_work["_char"] = cast_char

    cast_info_actors = pd.DataFrame({
        "id": np.arange(1, len(cast_work) + 1),
        "person_id": pd.to_numeric(cast_work["person_id"], errors="coerce").fillna(0).astype(int),
        "movie_id": pd.to_numeric(cast_work["title_id"], errors="coerce").fillna(0).astype(int),
        "person_role_id": cast_work["_char"].map(char_map).astype("Int64"),
        "note": cast_work.apply(_actor_cast_note, axis=1),
        "nr_order": pd.to_numeric(cast_work["billing_order"], errors="coerce").fillna(0).astype(int),
        "role_id": cast_work["person_id"].map(lambda pid: _role_id_from_gender(person_gender.get(_safe_int(pid), "U"))).astype(int),
    })
    # Preserve a real generated edge for JOB-Complex disambiguation bridges:
    # when the generator casts a person as a character with the same name, let
    # that person share the movie's disambiguation marker.  This keeps query 17
    # possible without inventing disconnected global name/title equality.
    person_name_by_id = {int(row.id): str(row.name) for row in name.itertuples(index=False)}
    char_name_by_id = {int(row.id): str(row.name) for row in char_name.itertuples(index=False)}
    char_name_values = {value.strip() for value in char_name_by_id.values() if value.strip()}
    name_index_promotions: Dict[int, str] = {}
    for row in cast_info_actors.itertuples(index=False):
        pid = _safe_int(getattr(row, "person_id", 0))
        mid = _safe_int(getattr(row, "movie_id", 0))
        title_idx = title_imdb_index_by_id.get(mid)
        if not pid or not title_idx:
            continue
        if person_name_by_id.get(pid, "").strip() not in char_name_values:
            continue
        name_index_promotions.setdefault(pid, title_idx)
    if name_index_promotions:
        name["_promoted_imdb_index"] = name["id"].map(name_index_promotions)
        name["imdb_index"] = name["_promoted_imdb_index"].combine_first(name["imdb_index"])
        name = name.drop(columns=["_promoted_imdb_index"])
        print(f"  +   name/title imdb_index cast bridges: {len(name_index_promotions):,} persons")
    name_imdb_index_by_id = {
        int(row.id): row.imdb_index
        for row in name.itertuples(index=False)
        if pd.notna(getattr(row, "imdb_index", None)) and str(getattr(row, "imdb_index", "")).strip()
    }

    # --- Append directors to cast_info with role_id=8 (real IMDB) ---
    cast_parts = [cast_info_actors]
    next_cast_id = len(cast_info_actors) + 1

    # Episode title rows need their own cast_info rows. Without this, JOB TV
    # queries that join episode titles through cast_info are structurally
    # impossible even when the synthetic TV subsystem generated episode casts.
    if not episode_cast.empty and episode_id_to_title_id and "person_id" in episode_cast.columns:
        ep_cast_work = episode_cast.copy()
        ep_cast_work["movie_id"] = (
            pd.to_numeric(ep_cast_work.get("episode_id"), errors="coerce")
            .fillna(0)
            .astype(int)
            .map(episode_id_to_title_id)
        )
        ep_cast_work = ep_cast_work[ep_cast_work["movie_id"].notna() & (ep_cast_work["movie_id"] > 0)].copy()
        if not ep_cast_work.empty:
            def _episode_cast_note(role_type: str) -> str | None:
                role_key = str(role_type or "").strip().lower()
                if "voice" in role_key:
                    return "(voice)"
                if "guest" in role_key:
                    return "(guest star)"
                return None

            ep_rows = pd.DataFrame({
                "id": np.arange(next_cast_id, next_cast_id + len(ep_cast_work)),
                "person_id": pd.to_numeric(ep_cast_work["person_id"], errors="coerce").fillna(0).astype(int),
                "movie_id": pd.to_numeric(ep_cast_work["movie_id"], errors="coerce").fillna(0).astype(int),
                "person_role_id": None,
                "note": [
                    _episode_cast_note(role)
                    for role in ep_cast_work.get("role_type", pd.Series([""] * len(ep_cast_work))).astype(str)
                ],
                "nr_order": pd.to_numeric(ep_cast_work.get("credit_order", 0), errors="coerce").fillna(0).astype(int),
                "role_id": ep_cast_work["person_id"].map(lambda pid: _role_id_from_gender(person_gender.get(_safe_int(pid), "U"))).astype(int),
            })
            cast_parts.append(ep_rows)
            next_cast_id += len(ep_rows)
            print(f"  +   episode cast_info:      {len(ep_rows):,} rows (kind_id=7 titles)")

    if not episodes.empty and episode_id_to_title_id:
        episode_crew_rows: List[Dict] = []
        for role_col, role_id, note in [
            ("writer_person_id", 4, "(written by)"),
            ("director_person_id", 8, None),
        ]:
            if role_col not in episodes.columns:
                continue
            work = episodes[[role_col, "episode_id"]].copy()
            work["person_id"] = pd.to_numeric(work[role_col], errors="coerce").fillna(0).astype(int)
            work["movie_id"] = (
                pd.to_numeric(work["episode_id"], errors="coerce")
                .fillna(0)
                .astype(int)
                .map(episode_id_to_title_id)
            )
            work = work[(work["person_id"] > 0) & work["movie_id"].notna() & (work["movie_id"] > 0)].copy()
            for row in work.itertuples(index=False):
                episode_crew_rows.append({
                    "id": next_cast_id + len(episode_crew_rows),
                    "person_id": int(getattr(row, "person_id")),
                    "movie_id": int(getattr(row, "movie_id")),
                    "person_role_id": None,
                    "note": note,
                    "nr_order": 1,
                    "role_id": role_id,
                })
        if episode_crew_rows:
            ep_crew_df = pd.DataFrame(episode_crew_rows)
            cast_parts.append(ep_crew_df)
            next_cast_id += len(ep_crew_df)
            print(f"  +   episode crew cast_info: {len(ep_crew_df):,} rows (writer/director)")

    if not tv_series.empty and series_id_to_title_id and "creator_person_id" in tv_series.columns:
        creator_rows: List[Dict] = []
        for row in tv_series.itertuples(index=False):
            sid = _safe_int(getattr(row, "series_id", 0))
            pid = _safe_int(getattr(row, "creator_person_id", 0))
            tid = series_id_to_title_id.get(sid)
            if not tid or not pid:
                continue
            creator_rows.append({
                "id": next_cast_id + len(creator_rows),
                "person_id": int(pid),
                "movie_id": int(tid),
                "person_role_id": None,
                "note": "(creator)",
                "nr_order": 1,
                "role_id": 4,
            })
        if creator_rows:
            creator_df = pd.DataFrame(creator_rows)
            cast_parts.append(creator_df)
            next_cast_id += len(creator_df)
            print(f"  +   TV creators in cast_info: {len(creator_df):,} rows (role_id=4)")

    if not movie_directors.empty and "director_id" in movie_directors.columns:
        director_order = movie_directors.groupby("title_id").cumcount() + 1 if "title_id" in movie_directors.columns else np.arange(1, len(movie_directors) + 1)
        dir_rows = pd.DataFrame({
            "id": np.arange(next_cast_id, next_cast_id + len(movie_directors)),
            "person_id": pd.to_numeric(movie_directors["director_id"], errors="coerce").fillna(0).astype(int),
            "movie_id": pd.to_numeric(movie_directors["title_id"], errors="coerce").fillna(0).astype(int),
            "person_role_id": None,
            "note": None,
            "nr_order": director_order,
            "role_id": 8,  # director in real IMDB
        })
        cast_parts.append(dir_rows)
        next_cast_id += len(movie_directors)
        print(f"  +   directors in cast_info: {len(dir_rows):,} rows (role_id=8)")

    # --- Append crew to cast_info with role_ids 4-11 ---
    if not movie_crew.empty and "person_id" in movie_crew.columns:
        crew_rows = pd.DataFrame({
            "id": np.arange(next_cast_id, next_cast_id + len(movie_crew)),
            "person_id": pd.to_numeric(movie_crew["person_id"], errors="coerce").fillna(0).astype(int),
            "movie_id": pd.to_numeric(movie_crew["title_id"], errors="coerce").fillna(0).astype(int),
            "person_role_id": None,
            "note": [
                _crew_credit_note(role, idx)
                for idx, role in enumerate(movie_crew["crew_role"].astype(str) if "crew_role" in movie_crew.columns else [""] * len(movie_crew))
            ],
            "nr_order": pd.to_numeric(movie_crew.get("credit_order", 0), errors="coerce").fillna(0).astype(int),
            "role_id": movie_crew["crew_role"].map(_crew_role_to_role_id).astype(int) if "crew_role" in movie_crew.columns else 11,
        })
        cast_parts.append(crew_rows)
        next_cast_id += len(movie_crew)
        print(f"  +   crew in cast_info:      {len(crew_rows):,} rows (role_ids 4-11)")

    cast_info = _concat_nonempty(cast_parts, columns=JOB_TABLE_COLUMNS["cast_info"])
    cast_info["id"] = np.arange(1, len(cast_info) + 1)  # re-number sequentially

    # company_name
    company_pcodes = [_company_pcodes(value) for value in companies["name"].astype(str)]
    company_country_codes, company_country_export_audit, company_country_summary = _company_country_export(
        companies,
        policy=company_country_policy,
    )
    company_name = pd.DataFrame({
        "id": companies["company_id"].astype(int),
        "name": companies["name"].astype(str),
        "country_code": company_country_codes,
        "imdb_id": companies["company_id"].astype(int),
        "name_pcode_nf": [item[0] for item in company_pcodes],
        "name_pcode_sf": [item[1] for item in company_pcodes],
        "md5sum": [
            _stable_md5("company_name", cid, value)
            for cid, value in zip(companies["company_id"].astype(int), companies["name"].astype(str))
        ],
    })

    # Some generated TV networks are stored as names without a resolved
    # network_company_id.  Strict IMDb has networks/broadcasters in the
    # company graph, so synthesize deterministic company_name rows for those
    # unmapped networks and link them through movie_companies below.
    network_company_by_series_id: Dict[int, int] = {}
    tv_network_company_audit_rows: List[Dict] = []
    if not tv_series.empty and "series_id" in tv_series.columns:
        existing_company_ids = set(pd.to_numeric(company_name["id"], errors="coerce").dropna().astype(int).tolist())
        company_name_by_key = {
            re.sub(r"\s+", " ", str(getattr(row, "name", "") or "").strip()).casefold(): int(row.id)
            for row in company_name.itertuples(index=False)
            if str(getattr(row, "name", "") or "").strip()
        }
        next_company_id = (max(existing_company_ids) + 1) if existing_company_ids else 1
        synthetic_company_rows: List[Dict] = []

        for row in tv_series.itertuples(index=False):
            sid = _safe_int(getattr(row, "series_id", 0))
            if not sid:
                continue
            tid = series_id_to_title_id.get(sid)
            source_cid = _safe_int(getattr(row, "network_company_id", 0))
            network_name = str(getattr(row, "network", "") or "").strip()
            series_country = str(getattr(row, "country", "") or "").strip()
            export_cid = 0
            resolution = "missing_network"

            if source_cid and source_cid in existing_company_ids:
                export_cid = int(source_cid)
                resolution = "source_network_company_id"
            elif network_name:
                key = re.sub(r"\s+", " ", network_name).casefold()
                existing_id = company_name_by_key.get(key)
                if existing_id:
                    export_cid = int(existing_id)
                    resolution = "matched_existing_company_name"
                else:
                    export_cid = int(next_company_id)
                    next_company_id += 1
                    company_name_by_key[key] = export_cid
                    existing_company_ids.add(export_cid)
                    pcode_nf, pcode_sf = _company_pcodes(network_name)
                    synthetic_company_rows.append({
                        "id": export_cid,
                        "name": network_name,
                        "country_code": _country_code(series_country or "USA"),
                        "imdb_id": export_cid,
                        "name_pcode_nf": pcode_nf,
                        "name_pcode_sf": pcode_sf,
                        "md5sum": _stable_md5("company_name", export_cid, network_name),
                    })
                    resolution = "synthetic_network_company"

            if export_cid:
                network_company_by_series_id[int(sid)] = int(export_cid)
            tv_network_company_audit_rows.append({
                "series_id": int(sid),
                "title_id": int(tid) if tid else None,
                "network": network_name or None,
                "series_country": series_country or None,
                "source_network_company_id": int(source_cid) if source_cid else None,
                "export_company_id": int(export_cid) if export_cid else None,
                "resolution": resolution,
            })

        if synthetic_company_rows:
            company_name = pd.concat([company_name, pd.DataFrame(synthetic_company_rows)], ignore_index=True)
            print(f"  +   synthetic TV network companies: {len(synthetic_company_rows):,} rows")

    tv_network_company_audit = pd.DataFrame(tv_network_company_audit_rows)
    tv_network_company_summary = (
        tv_network_company_audit["resolution"].value_counts(dropna=False).to_dict()
        if not tv_network_company_audit.empty and "resolution" in tv_network_company_audit.columns
        else {}
    )

    # movie_companies
    mc = movie_companies.copy()
    movie_companies_out = pd.DataFrame({
        "id": np.arange(1, len(mc) + 1),
        "movie_id": pd.to_numeric(mc["title_id"], errors="coerce").fillna(0).astype(int),
        "company_id": pd.to_numeric(mc["company_id"], errors="coerce").fillna(0).astype(int),
        "company_type_id": ((mc["role"] if "role" in mc.columns else pd.Series(["production"] * len(mc))).map(_company_type_id).astype(int)),
        "note": mc.apply(
            lambda row: _mc_note(row.get("role", ""), row.get("title_id", 0), movie),
            axis=1,
        ),
    })
    next_mc_id = int(len(movie_companies_out) + 1)

    # Add distributor rows derived from the existing production graph. This is
    # not a JOB-specific repair: IMDb commonly contains both production-company
    # and distributor credits, and JOB predicates rely on those regional notes.
    dist_rows: List[Dict] = []
    if not mc.empty and "title_id" in mc.columns and "company_id" in mc.columns:
        primary_company = (
            mc.sort_values(["title_id", "company_id"])
            .groupby("title_id", as_index=False)
            .first()[["title_id", "company_id"]]
        )
        for idx, row in enumerate(primary_company.itertuples(index=False)):
            tid = _safe_int(getattr(row, "title_id", 0))
            if tid == 0:
                continue
            h = (tid * 1103515245 + 12345) & 0x7FFFFFFF
            tier = str(_movie_tier_map.get(tid, "Mid") or "Mid")
            dist_p = 0.78 if tier in {"Epic", "A"} else 0.58 if tier == "Mid" else 0.42
            if (h % 1000) / 1000.0 > dist_p:
                continue
            dist_rows.append({
                "id": next_mc_id + len(dist_rows),
                "movie_id": tid,
                "company_id": _safe_int(getattr(row, "company_id", 0)),
                "company_type_id": 1,
                "note": _mc_note("Distribution", tid, movie),
            })
    if dist_rows:
        movie_companies_out = pd.concat([movie_companies_out, pd.DataFrame(dist_rows)], ignore_index=True)

    # TV series network/company credits are needed for strict IMDb/JOB queries
    # that join company_name through title rows with kind_id=2.
    tv_company_rows: List[Dict] = []
    if not tv_series.empty and series_id_to_title_id:
        series_company_by_id = {}
        for row in tv_series.itertuples(index=False):
            sid = _safe_int(getattr(row, "series_id", 0))
            cid = network_company_by_series_id.get(sid) or _safe_int(getattr(row, "network_company_id", 0))
            if sid and cid:
                series_company_by_id[int(sid)] = int(cid)
            tid = series_id_to_title_id.get(sid)
            if not tid or not cid:
                continue
            tv_company_rows.append({
                "id": int(len(movie_companies_out) + len(tv_company_rows) + 1),
                "movie_id": int(tid),
                "company_id": int(cid),
                "company_type_id": 2,
                "note": None,
            })
        if not episodes.empty and episode_id_to_title_id and series_company_by_id:
            for row in episodes.itertuples(index=False):
                eid = _safe_int(getattr(row, "episode_id", 0))
                sid = _safe_int(getattr(row, "series_id", 0))
                tid = episode_id_to_title_id.get(eid)
                cid = series_company_by_id.get(sid)
                if not tid or not cid:
                    continue
                year = str(getattr(row, "air_date", "") or "")[:4]
                year_note = f"({year})" if year.isdigit() else ""
                # Production credit: usually unnoted, matching real IMDb.
                tv_company_rows.append({
                    "id": int(len(movie_companies_out) + len(tv_company_rows) + 1),
                    "movie_id": int(tid),
                    "company_id": int(cid),
                    "company_type_id": 2,
                    "note": None,
                })
                # Broadcast/distribution credit: gives TV predicates a real
                # home without polluting production-company null-note patterns.
                tv_company_rows.append({
                    "id": int(len(movie_companies_out) + len(tv_company_rows) + 1),
                    "movie_id": int(tid),
                    "company_id": int(cid),
                    "company_type_id": 1,
                    "note": " ".join(part for part in [year_note, "(TV)"] if part),
                })
    if tv_company_rows:
        movie_companies_out = pd.concat([movie_companies_out, pd.DataFrame(tv_company_rows)], ignore_index=True)

    # keyword, movie_keyword
    stable_structural_keyword_ids = {
        "sequel": 8701,
        "computer-animation": 8702,
        "computer-animated-movie": 8703,
        "murder": 8704,
        "murder-in-title": 8705,
        "violence": 8706,
        "blood": 8707,
        "death": 8708,
        "gore": 8709,
        "hospital": 8710,
        "female-nudity": 8711,
        "hero": 8712,
        "martial-arts": 8713,
        "hand-to-hand-combat": 8714,
        "character-name-in-title": 8715,
        "tv-episode": 8716,
        "pilot": 8717,
        "series-finale": 8718,
    }
    keyword = pd.DataFrame({
        "id": keywords["keyword_id"].astype(int),
        "keyword": keywords["keyword"].astype(str),
        "phonetic_code": [_keyword_pcode(value) for value in keywords["keyword"].astype(str)],
    })
    mk = movie_keyword.copy()

    keyword_id_stability_rows: List[Dict] = []

    def _stabilize_structural_keyword_ids(keyword_df: pd.DataFrame, mk_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Reserve stable IDs for canonical JOB structural keywords.

        The generated keyword table can grow between the 100K and 200K exports.
        Without this normalization, canonical anchors such as ``hero`` or
        ``tv-episode`` receive different numeric IDs across scales, making a
        single SQL workload impossible even though the semantic keyword exists
        in both datasets.  We keep all generated keywords by moving displaced
        rows to fresh IDs and remapping movie_keyword references.
        """

        keyword_df = keyword_df.copy()
        mk_df = mk_df.copy()
        keyword_df["id"] = pd.to_numeric(keyword_df["id"], errors="coerce").fillna(0).astype(int)
        if "keyword_id" in mk_df.columns:
            mk_df["keyword_id"] = pd.to_numeric(mk_df["keyword_id"], errors="coerce").fillna(0).astype(int)

        reserved_ids = set(stable_structural_keyword_ids.values())
        used_ids = {int(value) for value in keyword_df["id"].tolist() if int(value) > 0}
        next_id = max(used_ids | reserved_ids | {0}) + 1
        id_remap: dict[int, int] = {}

        def fresh_id() -> int:
            nonlocal next_id
            while next_id in used_ids or next_id in reserved_ids:
                next_id += 1
            value = next_id
            next_id += 1
            used_ids.add(value)
            return value

        def move_row(row_index: int, reason: str) -> None:
            old_id = int(keyword_df.at[row_index, "id"])
            new_id = fresh_id()
            keyword_df.at[row_index, "id"] = new_id
            id_remap[old_id] = new_id
            keyword_id_stability_rows.append({
                "keyword": str(keyword_df.at[row_index, "keyword"]),
                "old_id": old_id,
                "new_id": new_id,
                "reason": reason,
            })

        keyword_text = keyword_df["keyword"].fillna("").astype(str).str.strip().str.lower()
        for term, target_id in stable_structural_keyword_ids.items():
            term_mask = keyword_text == term
            term_indices = list(keyword_df.index[term_mask])

            occupant_mask = (keyword_df["id"] == target_id) & (~term_mask)
            for occupant_index in list(keyword_df.index[occupant_mask]):
                move_row(int(occupant_index), f"displaced_by_stable_structural_keyword:{term}")

            if term_indices:
                primary_index = int(term_indices[0])
                old_id = int(keyword_df.at[primary_index, "id"])
                if old_id != target_id:
                    keyword_df.at[primary_index, "id"] = int(target_id)
                    id_remap[old_id] = int(target_id)
                    used_ids.discard(old_id)
                    used_ids.add(int(target_id))
                    keyword_id_stability_rows.append({
                        "keyword": term,
                        "old_id": old_id,
                        "new_id": int(target_id),
                        "reason": "canonical_structural_keyword_stable_id",
                    })

                # Preserve duplicate source keyword strings, but move duplicates
                # away from the canonical stable ID if they exist.
                for duplicate_index in term_indices[1:]:
                    move_row(int(duplicate_index), f"duplicate_structural_keyword:{term}")
            else:
                keyword_df.loc[len(keyword_df)] = {
                    "id": int(target_id),
                    "keyword": term,
                    "phonetic_code": _keyword_pcode(term),
                }
                used_ids.add(int(target_id))
                keyword_id_stability_rows.append({
                    "keyword": term,
                    "old_id": "",
                    "new_id": int(target_id),
                    "reason": "inserted_missing_structural_keyword_stable_id",
                })
                keyword_text = keyword_df["keyword"].fillna("").astype(str).str.strip().str.lower()

        if id_remap and "keyword_id" in mk_df.columns:
            mk_df["keyword_id"] = mk_df["keyword_id"].map(lambda value: id_remap.get(int(value), int(value)))

        keyword_df = keyword_df.sort_values("id", kind="stable").reset_index(drop=True)
        return keyword_df, mk_df

    keyword, mk = _stabilize_structural_keyword_ids(keyword, mk)
    keyword_id_by_text = {
        str(row.keyword).strip().lower(): int(row.id)
        for row in keyword.itertuples(index=False)
        if str(row.keyword).strip()
    }

    def _ensure_keyword(term: str) -> int:
        key = str(term or "").strip().lower()
        if not key:
            return 0
        existing = keyword_id_by_text.get(key)
        if existing:
            return int(existing)
        new_id = int(keyword["id"].max()) + 1 if not keyword.empty else 1
        keyword.loc[len(keyword)] = {"id": new_id, "keyword": term, "phonetic_code": _keyword_pcode(term)}
        keyword_id_by_text[key] = new_id
        return new_id

    for canonical_keyword in [
        "sequel",
        "computer-animation",
        "computer-animated-movie",
        "murder",
        "murder-in-title",
        "violence",
        "blood",
        "death",
        "gore",
        "hospital",
        "female-nudity",
        "hero",
        "martial-arts",
        "hand-to-hand-combat",
        "character-name-in-title",
        "tv-episode",
        "pilot",
        "series-finale",
    ]:
        _ensure_keyword(canonical_keyword)

    movie_keyword_out = pd.DataFrame({
        "id": np.arange(1, len(mk) + 1),
        "movie_id": pd.to_numeric(mk["title_id"], errors="coerce").fillna(0).astype(int),
        "keyword_id": pd.to_numeric(mk["keyword_id"], errors="coerce").fillna(0).astype(int),
    })
    existing_keyword_pairs = {
        (int(row.movie_id), int(row.keyword_id))
        for row in movie_keyword_out.itertuples(index=False)
        if _safe_int(getattr(row, "movie_id", 0)) and _safe_int(getattr(row, "keyword_id", 0))
    }

    def _add_keyword_anchor(rows: List[Dict], movie_id: int, term: str) -> None:
        kid = keyword_id_by_text.get(str(term).strip().lower())
        if not kid:
            return
        pair = (int(movie_id), int(kid))
        if pair in existing_keyword_pairs:
            return
        existing_keyword_pairs.add(pair)
        rows.append({"movie_id": int(movie_id), "keyword_id": int(kid)})

    anchor_rows: List[Dict] = []
    linked_movie_ids = set()
    if not movie_links.empty and "title_id" in movie_links.columns:
        link_text = movie_links.get("link_type", pd.Series([""] * len(movie_links))).fillna("").astype(str).str.lower().str.replace("_", " ", regex=False)
        linked_movie_ids = set(pd.to_numeric(movie_links.loc[link_text.isin(["follows", "followed by", "sequel", "spiritual successor"]), "title_id"], errors="coerce").fillna(0).astype(int).tolist())
    for r in movie.itertuples(index=False):
        mid = int(getattr(r, "title_id"))
        genre = str(getattr(r, "genre", "") or "")
        tier = str(getattr(r, "production_tier", "Mid") or "Mid")
        year_i = _safe_int(getattr(r, "year", 0))
        installment = _safe_int(getattr(r, "installment_no", 0))
        h = (mid * 2654435761) & 0xFFFFFFFF
        terms: List[str] = []
        if installment > 1 or mid in linked_movie_ids:
            terms.append("sequel")
        if genre == "Animation":
            terms.append("computer-animation")
            if tier in {"Epic", "A", "Mid"} and year_i >= 1995:
                terms.append("computer-animated-movie")
        if genre in {"Horror", "Thriller", "Crime"}:
            pool = ["murder", "violence", "blood", "death", "gore", "hospital"]
            start = h % len(pool)
            terms.extend(pool[start:start + 2] if start + 2 <= len(pool) else pool[start:] + pool[:(start + 2) % len(pool)])
            if genre == "Horror" and (h & 3) == 0:
                terms.append("female-nudity")
        if genre == "Action":
            if (h % 100) < 35:
                terms.extend(["hero", "martial-arts"])
            elif (h % 100) < 55:
                terms.append("hand-to-hand-combat")
        if genre in {"Adventure", "Fantasy"} and (h % 100) < 18:
            terms.append("hero")
        added_for_movie = 0
        for term in terms:
            before = len(anchor_rows)
            _add_keyword_anchor(anchor_rows, mid, term)
            if len(anchor_rows) > before:
                added_for_movie += 1
            if added_for_movie >= 3:
                break
    if not episodes.empty and episode_id_to_title_id:
        series_genre_by_id = {}
        if not tv_series.empty and "series_id" in tv_series.columns:
            series_genre_by_id = {
                _safe_int(getattr(row, "series_id", 0)): (_primary_genre_value(getattr(row, "genre", None)) or str(getattr(row, "genre", "") or ""))
                for row in tv_series.itertuples(index=False)
            }
        for row in episodes.itertuples(index=False):
            eid = _safe_int(getattr(row, "episode_id", 0))
            sid = _safe_int(getattr(row, "series_id", 0))
            tid = episode_id_to_title_id.get(eid)
            if not tid:
                continue
            ep_no = _safe_int(getattr(row, "episode_number", 0))
            title_text = str(getattr(row, "title", "") or "")
            genre = series_genre_by_id.get(sid, "")
            h = (int(tid) * 1103515245 + int(sid)) & 0xFFFFFFFF
            terms = ["tv-episode"]
            if ep_no == 1:
                terms.append("pilot")
            if genre in {"Crime", "Horror", "Thriller"}:
                pool = ["murder", "violence", "blood", "death", "hospital"]
                terms.append(pool[h % len(pool)])
            if genre in {"Action", "Adventure", "Fantasy"}:
                terms.append("hero" if (h % 3) else "martial-arts")
            # Sparse but explicit: some non-generic episode titles are named
            # after an in-universe character/person/place, a pattern JOB uses.
            if title_text and not title_text.lower().startswith("episode ") and (h % 100) < 18:
                terms.append("character-name-in-title")
            if (h % 250) == 0:
                terms.append("series-finale")
            added_for_episode = 0
            for term in terms:
                before = len(anchor_rows)
                _add_keyword_anchor(anchor_rows, int(tid), term)
                if len(anchor_rows) > before:
                    added_for_episode += 1
                if added_for_episode >= 3:
                    break
    if anchor_rows:
        anchor_df = pd.DataFrame(anchor_rows)
        anchor_df.insert(0, "id", np.arange(len(movie_keyword_out) + 1, len(movie_keyword_out) + 1 + len(anchor_df)))
        movie_keyword_out = pd.concat([movie_keyword_out, anchor_df], ignore_index=True)
        print(f"  +   structural keyword anchors: {len(anchor_df):,} rows")

    # JOB-Complex uses a non-key phonetic bridge between company_name and
    # keyword.  Materialize sparse company-core keyword aliases on movies that
    # already carry the company, so the bridge reflects actual movie/company
    # participation rather than a random global code collision.
    company_core_by_id = {
        int(row.id): _company_core_text(getattr(row, "name", ""))
        for row in company_name.itertuples(index=False)
        if str(getattr(row, "country_code", "")).lower() == "[us]"
    }
    company_keyword_rows: List[Dict] = []
    for core in sorted(set(value for value in company_core_by_id.values() if value)):
        _ensure_keyword(core)
    for row in movie_companies_out.itertuples(index=False):
        mid = _safe_int(getattr(row, "movie_id", 0))
        cid = _safe_int(getattr(row, "company_id", 0))
        core = company_core_by_id.get(cid)
        if not mid or not core:
            continue
        before = len(company_keyword_rows)
        _add_keyword_anchor(company_keyword_rows, mid, core)
        if len(company_keyword_rows) == before:
            continue
    if company_keyword_rows:
        company_anchor_df = pd.DataFrame(company_keyword_rows)
        company_anchor_df.insert(
            0,
            "id",
            np.arange(len(movie_keyword_out) + 1, len(movie_keyword_out) + 1 + len(company_anchor_df)),
        )
        movie_keyword_out = pd.concat([movie_keyword_out, company_anchor_df], ignore_index=True)
        print(f"  +   company phonetic keyword anchors: {len(company_anchor_df):,} rows")

    # movie_link
    ml = movie_links.copy()
    if ml.empty:
        movie_link = pd.DataFrame(columns=JOB_TABLE_COLUMNS["movie_link"])
    else:
        movie_link = pd.DataFrame({
            "id": np.arange(1, len(ml) + 1),
            "movie_id": pd.to_numeric(ml["title_id"], errors="coerce").fillna(0).astype(int),
            "linked_movie_id": pd.to_numeric(ml["linked_title_id"], errors="coerce").fillna(0).astype(int),
            "link_type_id": ml.get("link_type", "unknown").map(_link_type_id).astype(int),
        })
    tv_link_rows: List[Dict] = []
    if not tv_series.empty and series_id_to_title_id:
        last_by_key: Dict[tuple[str, str], int] = {}
        sorted_tv = tv_series.sort_values(["year_start", "series_id"]) if "year_start" in tv_series.columns else tv_series
        for row in sorted_tv.itertuples(index=False):
            sid = _safe_int(getattr(row, "series_id", 0))
            tid = series_id_to_title_id.get(sid)
            if not tid:
                continue
            key = (str(getattr(row, "network", "") or ""), str(getattr(row, "genre", "") or ""))
            prev_tid = last_by_key.get(key)
            if prev_tid and prev_tid != tid:
                h = (int(tid) * 2654435761) & 0xFFFFFFFF
                link_type_id = 1 if (h % 5) else 12  # mostly follows, sometimes spin off
                tv_link_rows.append({
                    "id": int(len(movie_link) + len(tv_link_rows) + 1),
                    "movie_id": int(tid),
                    "linked_movie_id": int(prev_tid),
                    "link_type_id": int(link_type_id),
                })
            last_by_key[key] = int(tid)
    if tv_link_rows:
        movie_link = pd.concat([movie_link, pd.DataFrame(tv_link_rows)], ignore_index=True)

    # aka_title
    at = aka_titles.copy()
    if at.empty:
        aka_title = pd.DataFrame(columns=JOB_TABLE_COLUMNS["aka_title"])
    else:
        at_movie_ids = pd.to_numeric(at["title_id"], errors="coerce").fillna(0).astype(int)
        title_kind_by_id = title.set_index("id")["kind_id"] if not title.empty and "id" in title.columns else pd.Series(dtype="Int64")
        title_year_by_id = title.set_index("id")["production_year"] if not title.empty and "id" in title.columns else pd.Series(dtype="Int64")
        title_episode_of_by_id = title.set_index("id")["episode_of_id"] if not title.empty and "id" in title.columns else pd.Series(dtype="Int64")
        title_season_by_id = title.set_index("id")["season_nr"] if not title.empty and "id" in title.columns else pd.Series(dtype="Int64")
        title_episode_by_id = title.set_index("id")["episode_nr"] if not title.empty and "id" in title.columns else pd.Series(dtype="Int64")
        aka_titles_text = (at["alt_title"] if "alt_title" in at.columns else pd.Series([""] * len(at))).astype(str)
        aka_notes = (at["language"] if "language" in at.columns else pd.Series([""] * len(at))).astype(str)
        aka_title = pd.DataFrame({
            "id": np.arange(1, len(at) + 1),
            "movie_id": at_movie_ids,
            "title": aka_titles_text,
            "imdb_index": at_movie_ids.map(title_imdb_index_by_id),
            "kind_id": at_movie_ids.map(title_kind_by_id).fillna(1).astype("Int64"),
            "production_year": at_movie_ids.map(title_year_by_id).astype("Int64"),
            "phonetic_code": [_soundex(value) for value in aka_titles_text],
            "episode_of_id": at_movie_ids.map(title_episode_of_by_id).astype("Int64"),
            "season_nr": at_movie_ids.map(title_season_by_id).astype("Int64"),
            "episode_nr": at_movie_ids.map(title_episode_by_id).astype("Int64"),
            "note": aka_notes,
            "md5sum": [
                _stable_md5("aka_title", idx, movie_id, alias, note)
                for idx, movie_id, alias, note in zip(np.arange(1, len(at) + 1), at_movie_ids, aka_titles_text, aka_notes)
            ],
        })

    # aka_name
    rng = np.random.RandomState(42)
    aka_name = _generate_aka_name(name, rng)
    if not aka_name.empty and "person_id" in aka_name.columns:
        aka_name["imdb_index"] = pd.to_numeric(aka_name["person_id"], errors="coerce").fillna(0).astype(int).map(name_imdb_index_by_id)

    # movie_info
    mi_rows: List[Dict] = []
    for r in movie.itertuples(index=False):
        mid = int(getattr(r, "title_id"))

        def add(info_type_id: int, value):
            if value is None:
                return
            sval = str(value).strip()
            if not sval or sval.lower() == "nan":
                return
            mi_rows.append({"movie_id": mid, "info_type_id": info_type_id, "info": sval, "note": None})

        for genre_value in genres_by_movie.get(mid, [_normalize_genre(getattr(r, "genre", None)) or "Drama"]):
            add(3, genre_value)
        add(8, getattr(r, "country", None))
        add(4, getattr(r, "language", None))
        add(5, getattr(r, "certification", None))     # real IMDB id=5 (was 16)
        add(2, getattr(r, "color_format", None))       # real IMDB id=2 (was 7)
        runtime = _maybe_int(getattr(r, "runtime_minutes", None))
        if runtime is not None and runtime > 0:
            add(1, f"{runtime} min")
        add(9, getattr(r, "tagline", None))             # real IMDB id=9 (was 99)
        add(98, getattr(r, "plot_summary", None))       # real IMDB id=98 (was 100)
        budget = _maybe_int(getattr(r, "budget_usd", None))
        gross = _maybe_int(getattr(r, "box_office_usd", None))
        if budget is not None:
            add(105, f"${budget:,}")                    # real IMDB id=105 (was 103)
        if gross is not None:
            add(107, f"${gross:,}")                     # real IMDB id=107 (was 104)
        aspect_ratio = getattr(r, "aspect_ratio", None)
        if aspect_ratio is not None and str(aspect_ratio).strip().lower() not in {"", "nan"}:
            aspect_text = re.sub(r"\s*:\s*", " : ", str(aspect_ratio).strip())
            add(70, aspect_text)                         # real IMDB id=70 "LD aspect ratio"

    # ── release dates (id=16): encode from release_dates.csv into movie_info EAV ──
    # Real IMDB format: "USA:5 January 2007", "France:14 February 2007 (Paris premiere)"
    if not release_dates.empty and "title_id" in release_dates.columns:
        _MONTH_NAMES = ["", "January", "February", "March", "April", "May", "June",
                        "July", "August", "September", "October", "November", "December"]
        for rd in release_dates.itertuples(index=False):
            mid = _safe_int(getattr(rd, "title_id", 0))
            country = str(getattr(rd, "country", "USA")).strip()
            date_str = str(getattr(rd, "release_date", "")).strip()
            rel_type = str(getattr(rd, "release_type", "")).strip()
            if not date_str or date_str.lower() == "nan" or mid == 0:
                continue
            # Convert YYYY-MM-DD to "DD Month YYYY" format
            try:
                parts = date_str.split("-")
                y, m, d = int(parts[0]), int(parts[1]) if len(parts) > 1 else 1, int(parts[2]) if len(parts) > 2 else 1
                imdb_date = f"{d} {_MONTH_NAMES[m]} {y}"
            except (ValueError, IndexError):
                imdb_date = date_str
            # Build note: real IMDB has (internet), (premiere), (limited), etc.
            rel_type_l = rel_type.lower()
            if rel_type_l in {"streaming", "digital", "online", "internet"}:
                note = "(internet)"
            elif rel_type and rel_type_l not in ("nan", "none", ""):
                note = f"({rel_type})"
            else:
                # Add some variety matching JOB predicate patterns
                h = (mid * 2654435761) & 0xFFFF
                frac = h / 0xFFFF
                if frac < 0.05:
                    note = "(internet)"
                elif frac < 0.10:
                    note = "(limited)"
                elif frac < 0.14:
                    note = "(premiere)"
                else:
                    note = None
            info_val = f"{country}:{imdb_date}"
            mi_rows.append({"movie_id": mid, "info_type_id": 16, "info": info_val, "note": note})
        print(f"  +   release dates in movie_info: {sum(1 for r in mi_rows if r['info_type_id'] == 16):,} rows")

    if not locations.empty and "title_id" in locations.columns:
        location_added = 0
        for loc in locations.itertuples(index=False):
            mid = _safe_int(getattr(loc, "title_id", 0))
            if not mid:
                continue
            city = str(getattr(loc, "city", "") or "").strip()
            country = str(getattr(loc, "country", "") or "").strip()
            location_type = str(getattr(loc, "location_type", "") or "").strip()
            info = ", ".join(part for part in [city, country] if part and part.lower() != "nan")
            if not info:
                continue
            note = f"({location_type})" if location_type and location_type.lower() not in {"nan", "none"} else None
            mi_rows.append({"movie_id": int(mid), "info_type_id": 18, "info": info, "note": note})
            location_added += 1
        print(f"  +   locations in movie_info: {location_added:,} rows")

    # ── trivia (id=17): stub generation, 1-5 entries per movie ──
    _TRIVIA_TEMPLATES = [
        "The production went through several script rewrites before final approval.",
        "Multiple locations were scouted before the director settled on the final setting.",
        "Several scenes were improvised by the cast during filming.",
        "The original cut was significantly longer than the theatrical release.",
        "The soundtrack was recorded with a live orchestra over three sessions.",
        "Pre-production lasted over six months due to scheduling conflicts.",
        "The director drew inspiration from classic films of the same genre.",
        "Practical effects were used extensively instead of CGI.",
        "The lead actor performed many of their own stunts.",
        "Filming was completed ahead of the original schedule.",
        "The costume department created over 200 unique outfits for the production.",
        "Several real locations were used alongside purpose-built sets.",
        "The film's color palette was carefully chosen to reflect the narrative tone.",
        "Post-production took nearly a year to complete.",
        "The script underwent a table read with the full cast before shooting began.",
    ]
    trivia_rng = np.random.RandomState(717)
    movie_ids_list = movie["title_id"].astype(int).tolist()
    for mid in movie_ids_list:
        n_trivia = trivia_rng.choice([1, 1, 2, 2, 3, 3, 4, 5])
        chosen = trivia_rng.choice(len(_TRIVIA_TEMPLATES), size=min(n_trivia, len(_TRIVIA_TEMPLATES)), replace=False)
        for idx in chosen:
            mi_rows.append({"movie_id": mid, "info_type_id": 17, "info": _TRIVIA_TEMPLATES[idx], "note": None})
    print(f"  +   trivia in movie_info: {sum(1 for r in mi_rows if r['info_type_id'] == 17):,} rows")

    if not tv_series.empty and series_id_to_title_id:
        series_info_added = 0
        for row in tv_series.itertuples(index=False):
            sid = _safe_int(getattr(row, "series_id", 0))
            tid = series_id_to_title_id.get(sid)
            if not tid:
                continue

            def add_series(info_type_id: int, value):
                nonlocal series_info_added
                if value is None:
                    return
                sval = str(value).strip()
                if not sval or sval.lower() == "nan":
                    return
                mi_rows.append({"movie_id": int(tid), "info_type_id": info_type_id, "info": sval, "note": None})
                series_info_added += 1

            add_series(3, _primary_genre_value(getattr(row, "genre", None)))
            add_series(8, getattr(row, "country", None))
            add_series(4, getattr(row, "language", None))
            add_series(5, getattr(row, "content_rating", None))
            add_series(98, getattr(row, "plot_summary", None))
        print(f"  +   TV series metadata in movie_info: {series_info_added:,} rows")

    if not episodes.empty and episode_id_to_title_id:
        series_meta_by_id = {}
        if not tv_series.empty and "series_id" in tv_series.columns:
            for row in tv_series.itertuples(index=False):
                sid = _safe_int(getattr(row, "series_id", 0))
                if not sid:
                    continue
                series_meta_by_id[sid] = {
                    "genre": getattr(row, "genre", None),
                    "country": getattr(row, "country", None),
                    "language": getattr(row, "language", None),
                    "content_rating": getattr(row, "content_rating", None),
                }
        _MONTH_NAMES = ["", "January", "February", "March", "April", "May", "June",
                        "July", "August", "September", "October", "November", "December"]
        episode_info_added = 0
        for row in episodes.itertuples(index=False):
            eid = _safe_int(getattr(row, "episode_id", 0))
            sid = _safe_int(getattr(row, "series_id", 0))
            tid = episode_id_to_title_id.get(eid)
            if not tid:
                continue
            meta = series_meta_by_id.get(sid, {})
            for info_type_id, key in [(3, "genre"), (8, "country"), (4, "language"), (5, "content_rating")]:
                value = meta.get(key)
                if value is None or str(value).strip().lower() in {"", "nan"}:
                    continue
                if info_type_id == 3:
                    value = _primary_genre_value(value)
                    if value is None:
                        continue
                mi_rows.append({"movie_id": int(tid), "info_type_id": info_type_id, "info": str(value).strip(), "note": None})
                episode_info_added += 1
            runtime = _maybe_int(getattr(row, "runtime_minutes", None))
            if runtime is not None and runtime > 0:
                mi_rows.append({"movie_id": int(tid), "info_type_id": 1, "info": f"{runtime} min", "note": None})
                episode_info_added += 1
            description = str(getattr(row, "description", "") or "").strip()
            if description and description.lower() != "nan":
                mi_rows.append({"movie_id": int(tid), "info_type_id": 98, "info": description, "note": None})
                episode_info_added += 1
            air_date = str(getattr(row, "air_date", "") or "").strip()
            if air_date and air_date.lower() != "nan":
                release_country = str(meta.get("country") or "USA").strip()
                if not release_country or release_country.lower() == "nan":
                    release_country = "USA"
                try:
                    parts = air_date.split("-")
                    y = int(parts[0])
                    m = int(parts[1]) if len(parts) > 1 else 1
                    d = int(parts[2]) if len(parts) > 2 else 1
                    release_info = f"{release_country}:{d} {_MONTH_NAMES[m]} {y}"
                except Exception:
                    release_info = f"{release_country}:{air_date}"
                mi_rows.append({"movie_id": int(tid), "info_type_id": 16, "info": release_info, "note": "(TV)"})
                episode_info_added += 1
        print(f"  +   episode metadata in movie_info: {episode_info_added:,} rows")

    movie_info = pd.DataFrame(mi_rows)
    if movie_info.empty:
        movie_info = pd.DataFrame(columns=JOB_TABLE_COLUMNS["movie_info"])
    else:
        genre_mask = movie_info["info_type_id"].eq(3)
        genre_rows = movie_info.loc[genre_mask].drop_duplicates(
            subset=["movie_id", "info_type_id", "info"],
            keep="first",
        )
        movie_info = pd.concat(
            [genre_rows, movie_info.loc[~genre_mask]],
            ignore_index=True,
            sort=False,
        )
        movie_info.insert(0, "id", np.arange(1, len(movie_info) + 1))

    # movie_info_idx
    mix_rows: List[Dict] = []
    for r in movie.itertuples(index=False):
        mid = int(getattr(r, "title_id"))
        rating = getattr(r, "rating", None)
        votes = getattr(r, "num_votes", None)
        if rating is not None and not pd.isna(rating):
            mix_rows.append({"movie_id": mid, "info_type_id": 101, "info": f"{float(rating):.1f}", "note": None})  # real IMDB id=101
        if votes is not None and not pd.isna(votes):
            mix_rows.append({"movie_id": mid, "info_type_id": 100, "info": f"{int(votes)}", "note": None})  # real IMDB id=100

    if not tv_series.empty and series_id_to_title_id:
        for row in tv_series.itertuples(index=False):
            sid = _safe_int(getattr(row, "series_id", 0))
            tid = series_id_to_title_id.get(sid)
            rating = getattr(row, "overall_rating", None)
            if tid and rating is not None and not pd.isna(rating):
                mix_rows.append({"movie_id": int(tid), "info_type_id": 101, "info": f"{float(rating):.1f}", "note": None})
                seasons = max(1, _safe_int(getattr(row, "total_seasons", 1)))
                votes = max(100, int((float(rating) ** 2) * seasons * 120))
                mix_rows.append({"movie_id": int(tid), "info_type_id": 100, "info": str(votes), "note": None})
    if not episodes.empty and "episode_id" in episodes.columns:
        for row in episodes.itertuples(index=False):
            eid = _safe_int(getattr(row, "episode_id", 0))
            tid = episode_id_to_title_id.get(eid)
            rating = getattr(row, "rating", None)
            if tid and rating is not None and not pd.isna(rating):
                mix_rows.append({"movie_id": int(tid), "info_type_id": 101, "info": f"{float(rating):.1f}", "note": None})
                viewership = getattr(row, "viewership_millions", 0.0)
                votes = max(25, int(float(viewership or 0.0) * 1000))
                mix_rows.append({"movie_id": int(tid), "info_type_id": 100, "info": str(votes), "note": None})

    # ── top 250 rank (id=112) and bottom 10 rank (id=113) ──
    # Compute from actual ratings: rank movies, take top 250 and bottom 10
    rated_movies = movie.dropna(subset=["rating"]).copy()
    if not rated_movies.empty:
        rated_movies["_rating_f"] = pd.to_numeric(rated_movies["rating"], errors="coerce")
        rated_movies["_votes_f"] = pd.to_numeric(rated_movies.get("num_votes", 0), errors="coerce").fillna(0)
        # Top 250: highest rated, require minimum votes (top 10% vote threshold)
        vote_threshold = rated_movies["_votes_f"].quantile(0.10)
        eligible = rated_movies[rated_movies["_votes_f"] >= vote_threshold].copy()
        top250 = eligible.nlargest(250, ["_rating_f", "_votes_f"])
        for rank, (_, row) in enumerate(top250.iterrows(), 1):
            mid = int(row["title_id"])
            mix_rows.append({"movie_id": mid, "info_type_id": 112, "info": str(rank), "note": None})
        # Bottom 10: lowest rated
        bottom10 = eligible.nsmallest(10, ["_rating_f", "_votes_f"])
        for rank, (_, row) in enumerate(bottom10.iterrows(), 1):
            mid = int(row["title_id"])
            mix_rows.append({"movie_id": mid, "info_type_id": 113, "info": str(rank), "note": None})
        print(f"  +   top 250 rank in movie_info_idx: {len(top250)} rows")
        print(f"  +   bottom 10 rank in movie_info_idx: {len(bottom10)} rows")

    movie_info_idx = pd.DataFrame(mix_rows)
    if movie_info_idx.empty:
        movie_info_idx = pd.DataFrame(columns=JOB_TABLE_COLUMNS["movie_info_idx"])
    else:
        movie_info_idx.insert(0, "id", np.arange(1, len(movie_info_idx) + 1))

    # person_info
    pi_rows: List[Dict] = []
    demo_by_person = person_demo.set_index("person_id") if not person_demo.empty and "person_id" in person_demo.columns else None

    for r in persons.itertuples(index=False):
        pid = int(getattr(r, "person_id"))

        def padd(info_type_id: int, value, note=None):
            if value is None:
                return
            sval = str(value).strip()
            if not sval or sval.lower() == "nan":
                return
            pi_rows.append({"person_id": pid, "info_type_id": info_type_id, "info": sval, "note": note})

        # mini biography: populate note with contributor names (JOB 7a/7b filter pi.note='Volker Boehm')
        _BIO_CONTRIBUTORS = [
            "Volker Boehm", "Volker Boehm", "Volker Boehm",  # ~30% weight (matches real IMDB)
            "Pedro Borges", "Pedro Borges",
            "Anonymous", "Anonymous",
            "Steve Shelokhonov", "Jon C. Hopwood", "Sam Sharpe",
        ]
        bio_note = _BIO_CONTRIBUTORS[pid % len(_BIO_CONTRIBUTORS)]
        padd(19, getattr(r, "bio", None), note=bio_note)  # real IMDB id=19 "mini biography"
        style_tags = str(getattr(r, "style_tags", "") or "").strip()
        career_stage = str(getattr(r, "career_stage", "") or "").strip()
        trivia_bits = [
            bit for bit in [
                f"Known for {style_tags.replace(';', ', ')}" if style_tags and style_tags.lower() != "nan" else "",
                f"Career stage: {career_stage}" if career_stage and career_stage.lower() != "nan" else "",
            ] if bit
        ]
        if trivia_bits:
            padd(17, "; ".join(trivia_bits))              # real IMDB id=17 "trivia"

        if demo_by_person is not None and pid in demo_by_person.index:
            drow = demo_by_person.loc[pid]
            if isinstance(drow, pd.DataFrame):
                drow = drow.iloc[0]
            padd(21, drow.get("birth_date"))              # real IMDB id=21 (correct)
            birth_note = ", ".join([str(drow.get("birth_city", "")).strip(), str(drow.get("birth_country", "")).strip()]).strip(", ").strip()
            padd(20, birth_note)                          # real IMDB id=20 (was 22)
            padd(23, drow.get("death_date"))              # real IMDB id=23 (was 26)
            h = drow.get("height_cm")
            if h is not None and not pd.isna(h):
                padd(22, f"{float(h):.1f} cm")            # real IMDB id=22 (was 34)

    person_info = pd.DataFrame(pi_rows)
    if person_info.empty:
        person_info = pd.DataFrame(columns=JOB_TABLE_COLUMNS["person_info"])
    else:
        person_info.insert(0, "id", np.arange(1, len(person_info) + 1))

    # complete_cast
    complete_cast = _build_complete_cast(movie, cast, movie_crew, movie_directors)
    if not episode_cast.empty and episode_id_to_title_id and "episode_id" in episode_cast.columns:
        ep_cast_counts = (
            episode_cast.assign(
                _movie_id=pd.to_numeric(episode_cast["episode_id"], errors="coerce")
                .fillna(0)
                .astype(int)
                .map(episode_id_to_title_id)
            )
            .dropna(subset=["_movie_id"])
            .groupby("_movie_id")
            .size()
        )
        ep_cc_rows: List[Dict] = []
        next_cc_id = int(len(complete_cast) + 1)
        for idx, (movie_id, cast_count) in enumerate(ep_cast_counts.items()):
            if _safe_int(movie_id) == 0 or int(cast_count) <= 0:
                continue
            status_id = 4 if int(cast_count) >= 5 else 3
            ep_cc_rows.append({
                "id": next_cc_id + len(ep_cc_rows),
                "movie_id": int(movie_id),
                "subject_id": 1,
                "status_id": status_id,
            })
        if ep_cc_rows:
            complete_cast = pd.concat([complete_cast, pd.DataFrame(ep_cc_rows)], ignore_index=True)
            print(f"  +   episode complete_cast:  {len(ep_cc_rows):,} rows")
    if not episodes.empty and episode_id_to_title_id:
        episode_crew_movie_ids: set[int] = set()
        for role_col in ["writer_person_id", "director_person_id"]:
            if role_col not in episodes.columns:
                continue
            for row in episodes[["episode_id", role_col]].itertuples(index=False):
                pid = _safe_int(getattr(row, role_col, 0))
                eid = _safe_int(getattr(row, "episode_id", 0))
                tid = episode_id_to_title_id.get(eid)
                if pid and tid:
                    episode_crew_movie_ids.add(int(tid))
        if episode_crew_movie_ids:
            existing_cc = {
                (int(row.movie_id), int(row.subject_id))
                for row in complete_cast.itertuples(index=False)
                if _safe_int(getattr(row, "movie_id", 0)) and _safe_int(getattr(row, "subject_id", 0))
            }
            ep_crew_cc_rows = []
            next_cc_id = int(len(complete_cast) + 1)
            for movie_id in sorted(episode_crew_movie_ids):
                if (movie_id, 2) in existing_cc:
                    continue
                ep_crew_cc_rows.append({
                    "id": next_cc_id + len(ep_crew_cc_rows),
                    "movie_id": int(movie_id),
                    "subject_id": 2,
                    "status_id": 4,
                })
            if ep_crew_cc_rows:
                complete_cast = pd.concat([complete_cast, pd.DataFrame(ep_crew_cc_rows)], ignore_index=True)
                print(f"  +   episode crew complete_cast: {len(ep_crew_cc_rows):,} rows")

    # write core tables
    core_map = {
        "title": title,
        "name": name,
        "cast_info": cast_info,
        "char_name": char_name,
        "company_name": company_name,
        "movie_companies": movie_companies_out,
        "movie_keyword": movie_keyword_out,
        "keyword": keyword,
        "movie_link": movie_link,
        "aka_title": aka_title,
        "aka_name": aka_name,
        "movie_info": movie_info,
        "movie_info_idx": movie_info_idx,
        "person_info": person_info,
        "complete_cast": complete_cast,
        **static,
    }

    for table in JOB_CORE_TABLES:
        _write_table(core_map.get(table, pd.DataFrame()), table, out_dir)

    genre_derivation_audit.to_csv(out_dir / "genre_derivation_audit.csv", index=False)
    company_country_export_audit.to_csv(out_dir / "company_country_export_audit.csv", index=False)
    tv_network_company_audit.to_csv(out_dir / "tv_network_company_audit.csv", index=False)
    pd.DataFrame(
        keyword_id_stability_rows,
        columns=["keyword", "old_id", "new_id", "reason"],
    ).to_csv(out_dir / "keyword_id_stability_audit.csv", index=False)
    source_export_coverage.to_csv(out_dir / "source_export_coverage.csv", index=False)
    (out_dir / "genre_derivation_summary.json").write_text(
        json.dumps(genre_derivation_summary, indent=2),
        encoding="utf-8",
    )
    print(f"  OK  audit:genre_derivation_audit.csv {len(genre_derivation_audit):>10,} rows")
    print(f"  OK  audit:company_country_export_audit.csv {len(company_country_export_audit):>10,} rows")
    print(f"  OK  audit:tv_network_company_audit.csv {len(tv_network_company_audit):>10,} rows")
    print(f"  OK  audit:keyword_id_stability_audit.csv {len(keyword_id_stability_rows):>10,} rows")
    print(f"  OK  audit:source_export_coverage.csv  {len(source_export_coverage):>10,} rows")

    if strict_job:
        _validate_core_headers(out_dir)
        from validate_imdb_schema import validate as validate_imdb_schema
        validate_imdb_schema(out_dir)
        print("  JOB strict schema validation: OK")

    if include_extras:
        extras = [
            "movie", "cast_info", "movie_companies", "movie_keyword", "movie_links",
            "movie_crew", "movie_directors", "release_dates", "ratings_breakdown",
            "reviews", "awards", "locations", "world_events", "production_timeline",
            "streaming_windows", "person_contracts", "movie_sequence", "person_collaborations",
            "tv_series", "seasons", "episodes", "episode_cast", "box_office_daily",
            "box_office_weekly", "box_office_by_territory", "person_demographics",
            "company_links", "media_links", "user_ratings",
            "persons_enriched", "companies_enriched", "movies_flat", "movies_analysis",
        ]
        core_csv_names = {f"{t}.csv" for t in JOB_CORE_TABLES}
        for stem in extras:
            dst_name = f"{stem}.csv" if f"{stem}.csv" not in core_csv_names else f"extra_{stem}.csv"
            if stem == "movies_analysis":
                df = _build_movies_analysis_extra(base_dir)
                if not df.empty:
                    df.to_csv(out_dir / dst_name, index=False)
                    print(f"  OK  extra:{dst_name:<24} {len(df):>10,} rows")
                continue
            src = base_dir / f"{stem}.csv"
            if _source_exists(src):
                df = _read_csv(src, required=False)
                df.to_csv(out_dir / dst_name, index=False)
                print(f"  OK  extra:{dst_name:<24} {len(df):>10,} rows")
        metadata_extras = [
            ("extra_keyword_metadata.csv", keywords),
            ("extra_person_metadata.csv", persons),
            ("extra_company_metadata.csv", companies),
            ("extra_cast_metadata.csv", cast),
            ("extra_movie_company_provenance.csv", movie_companies),
        ]
        for dst_name, df in metadata_extras:
            if df is not None and not df.empty:
                df.to_csv(out_dir / dst_name, index=False)
                print(f"  OK  extra:{dst_name:<24} {len(df):>10,} rows")
        edge_src = base_dir / "graph" / "edge_graph.csv"
        if edge_src.exists():
            edf = pd.read_csv(edge_src, low_memory=False)
            edf.to_csv(out_dir / "extra_edges.csv", index=False)
            print(f"  OK  extra:extra_edges.csv{'':<11} {len(edf):>10,} rows")

    manifest = {
        "schema": "job_imdb_core",
        "strict_job": bool(strict_job),
        "include_extras": bool(include_extras),
        "source_dir": str(base_dir),
        "output_dir": str(out_dir),
        "core_tables": {table: int(len(core_map.get(table, pd.DataFrame()))) for table in JOB_CORE_TABLES},
        "genre_derivation": genre_derivation_summary,
        "company_country_export": company_country_summary,
        "tv_network_company_export": tv_network_company_summary,
        "audit_files": {
            "genre_derivation_audit": "genre_derivation_audit.csv",
            "company_country_export_audit": "company_country_export_audit.csv",
            "tv_network_company_audit": "tv_network_company_audit.csv",
            "source_export_coverage": "source_export_coverage.csv",
            "genre_derivation_summary": "genre_derivation_summary.json",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  OK  export_manifest.json")


def main():
    parser = argparse.ArgumentParser(description="Convert dataset to strict JOB/IMDB core schema")
    parser.add_argument("--base-dir", default=str(Path(__file__).resolve().parent),
                        help="Source dataset directory (contains movie.csv, entities/, graph/)")
    parser.add_argument("--out-dir", default=str((Path(__file__).resolve().parent / "imdb_schema").resolve()),
                        help="Output directory for converted CSV tables")
    parser.add_argument("--strict-job", action=argparse.BooleanOptionalAction, default=True,
                        help="Validate and fail if any of the 21 JOB tables/headers are missing or mismatched")
    parser.add_argument("--include-extras", action=argparse.BooleanOptionalAction, default=False,
                        help="Copy non-JOB research tables into output directory")
    parser.add_argument("--company-country-policy", choices=("imdb-skewed", "preserve"), default="imdb-skewed",
                        help="How to export company_name.country_code. imdb-skewed projects legacy uniform companies to IMDb-like production markets; preserve keeps source country codes.")
    parser.add_argument("--build-duckdb", action=argparse.BooleanOptionalAction, default=False,
                        help="After CSV export, also build a DuckDB database from the strict core tables.")
    parser.add_argument("--duckdb-out", default=None,
                        help="DuckDB output path used with --build-duckdb. Defaults to <out-dir>/imdb.duckdb.")
    parser.add_argument("--overwrite-duckdb", action="store_true",
                        help="Overwrite an existing DuckDB output file when --build-duckdb is enabled.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    print(f"Source: {base_dir}")
    print(f"Output: {out_dir}")
    convert(
        base_dir=base_dir,
        out_dir=out_dir,
        strict_job=bool(args.strict_job),
        include_extras=bool(args.include_extras),
        company_country_policy=str(args.company_country_policy),
    )
    if args.build_duckdb:
        from build_duckdb_from_imdb_schema import build_duckdb

        duckdb_out = Path(args.duckdb_out).resolve() if args.duckdb_out else out_dir / "imdb.duckdb"
        built_path = build_duckdb(out_dir, duckdb_out, overwrite=bool(args.overwrite_duckdb), core_only=True)
        print(f"  OK  duckdb: {built_path}")
    print("Done.")


if __name__ == "__main__":
    main()
