from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

WORLD_POLICY_FILENAME = "world_policy.json"
CONCEPT_PACKS_FILENAME = "concept_packs.json"
YEAR_SLATE_PLAN_FILENAME = "year_slate_plan.json"
KEYWORD_MOTIF_BANK_FILENAME = "keyword_motif_bank.json"
FRANCHISE_BIBLES_FILENAME = "franchise_bibles.json"
IDENTITY_BANK_FILENAME = "identity_bank.json"
CHARACTER_IDENTITY_BANK_FILENAME = "character_identity_bank.json"
COMPANY_LEXICON_FILENAME = "company_lexicon.json"
KEYWORD_SEED_BANK_FILENAME = "keyword_seed_bank.json"
TITLE_GRAMMAR_BANK_FILENAME = "title_grammar_bank.json"
TEMPORAL_REGIME_PLAN_FILENAME = "temporal_regime_plan.json"
MODELING_PRIORS_FILENAME = "modeling_priors.json"
DECISION_LOG_DIRNAME = "decision_logs"
DECISION_LOG_FILENAME = "movie_selection_log.jsonl"
MOVIE_PROGRESS_LOG_FILENAME = "movie_generation_progress.jsonl"
LLM_USAGE_LOG_FILENAME = "llm_usage.jsonl"
PROMPT_CALIBRATION_LOG_FILENAME = "llm_prompt_calibration.jsonl"
RESEARCH_AUDIT_FILENAME = "research_mode_audit.json"
COMPARISON_REPORT_FILENAME = "pipeline_comparison_report.json"
DEFAULT_BUCKET_SIZE = 5

DEFAULT_COUNTRY_TO_MARKET = {
    "USA": "North America",
    "Canada": "North America",
    "UK": "Europe",
    "France": "Europe",
    "Germany": "Europe",
    "Italy": "Europe",
    "Spain": "Europe",
    "Sweden": "Europe",
    "Denmark": "Europe",
    "Norway": "Europe",
    "Netherlands": "Europe",
    "Belgium": "Europe",
    "Switzerland": "Europe",
    "Austria": "Europe",
    "Ireland": "Europe",
    "Iceland": "Europe",
    "Czech Republic": "Europe",
    "Slovakia": "Europe",
    "Slovenia": "Europe",
    "Croatia": "Europe",
    "Serbia": "Europe",
    "Romania": "Europe",
    "Bulgaria": "Europe",
    "Estonia": "Europe",
    "Latvia": "Europe",
    "Lithuania": "Europe",
    "Luxembourg": "Europe",
    "Macedonia": "Europe",
    "Belarus": "Europe",
    "Ukraine": "Europe",
    "Russia": "Europe",
    "Armenia": "Europe",
    "Azerbaijan": "Europe",
    "Georgia": "Europe",
    "Poland": "Europe",
    "Portugal": "Europe",
    "Greece": "Europe",
    "India": "Asia",
    "Japan": "Asia",
    "South Korea": "Asia",
    "China": "Asia",
    "Thailand": "Asia",
    "Indonesia": "Asia",
    "Philippines": "Asia",
    "Pakistan": "Asia",
    "Bangladesh": "Asia",
    "Vietnam": "Asia",
    "Malaysia": "Asia",
    "Singapore": "Asia",
    "Taiwan": "Asia",
    "Sri Lanka": "Asia",
    "Nepal": "Asia",
    "Mongolia": "Asia",
    "Kazakhstan": "Asia",
    "Uzbekistan": "Asia",
    "Australia": "Oceania",
    "New Zealand": "Oceania",
    "Nigeria": "Africa",
    "South Africa": "Africa",
    "Kenya": "Africa",
    "Ghana": "Africa",
    "Senegal": "Africa",
    "Tunisia": "Africa",
    "Ethiopia": "Africa",
    "Tanzania": "Africa",
    "Zimbabwe": "Africa",
    "Congo": "Africa",
    "Benin": "Africa",
    "Cape Verde": "Africa",
    "Brazil": "South America",
    "Argentina": "South America",
    "Chile": "South America",
    "Ecuador": "South America",
    "Bolivia": "South America",
    "Uruguay": "South America",
    "Mexico": "Latin America",
    "Colombia": "Latin America",
    "Peru": "Latin America",
    "Cuba": "Latin America",
    "Dominican Republic": "Latin America",
    "Costa Rica": "Latin America",
    "Guatemala": "Latin America",
    "Honduras": "Latin America",
    "Puerto Rico": "Latin America",
    "Turkey": "Middle East",
    "Iran": "Middle East",
    "Egypt": "Middle East",
    "Israel": "Middle East",
    "UAE": "Middle East",
    "Jordan": "Middle East",
    "Iraq": "Middle East",
    "Saudi Arabia": "Middle East",
    "Kuwait": "Middle East",
    "Qatar": "Middle East",
    "Bahrain": "Middle East",
    "Oman": "Middle East",
    "Syria": "Middle East",
}

DEFAULT_STYLE_BY_GENRE = {
    "Action": "kinetic",
    "Animation": "playful",
    "Comedy": "wry",
    "Crime": "hardboiled",
    "Documentary": "observational",
    "Drama": "elegant",
    "Fantasy": "mythic",
    "Horror": "ominous",
    "Mystery": "enigmatic",
    "Romance": "luminous",
    "Sci-Fi": "futurist",
    "Thriller": "urgent",
    "War": "somber",
}

DEFAULT_CONFLICT_BY_GENRE = {
    "Action": "high-stakes pursuit",
    "Animation": "coming-of-age adventure",
    "Comedy": "social misunderstanding cascade",
    "Crime": "ambition-versus-loyalty spiral",
    "Documentary": "institutional uncovering",
    "Drama": "family and status fracture",
    "Fantasy": "realm-balance quest",
    "Horror": "containment failure",
    "Mystery": "layered investigation",
    "Romance": "timing and class friction",
    "Sci-Fi": "technology ethics rupture",
    "Thriller": "trust breakdown under pressure",
    "War": "duty-versus-survival dilemma",
}

DEFAULT_RELATIONSHIP_BY_GENRE = {
    "Action": "mentor-protege",
    "Animation": "found family",
    "Comedy": "mismatched duo",
    "Crime": "partners under suspicion",
    "Documentary": "community portrait",
    "Drama": "parent-child reckoning",
    "Fantasy": "heir and guardian",
    "Horror": "fractured survivors",
    "Mystery": "detective and witness",
    "Romance": "slow-burn pair",
    "Sci-Fi": "crew with conflicting loyalties",
    "Thriller": "allies who may betray each other",
    "War": "unit forged under attrition",
}

DEFAULT_TONE_BY_GENRE = {
    "Action": "charged",
    "Animation": "bright",
    "Comedy": "buoyant",
    "Crime": "gritty",
    "Documentary": "measured",
    "Drama": "grounded",
    "Fantasy": "sweeping",
    "Horror": "claustrophobic",
    "Mystery": "atmospheric",
    "Romance": "warm",
    "Sci-Fi": "cerebral",
    "Thriller": "tense",
    "War": "grim",
}

DEFAULT_SEASON_BY_GENRE = {
    "Action": "summer",
    "Animation": "holiday",
    "Comedy": "spring",
    "Crime": "fall",
    "Documentary": "festival",
    "Drama": "awards",
    "Fantasy": "holiday",
    "Horror": "fall",
    "Mystery": "fall",
    "Romance": "winter",
    "Sci-Fi": "summer",
    "Thriller": "fall",
    "War": "awards",
}

DEFAULT_CHEMISTRY_BY_TIER = {
    "Epic": "star_vehicle",
    "A": "prestige_pairing",
    "Mid": "balanced_ensemble",
    "Indie": "volatile_ensemble",
    "Micro": "intimate_pairing",
}

DEFAULT_SUBGENRES_BY_GENRE = {
    "Action": ["heist", "survival", "espionage"],
    "Animation": ["family_adventure", "coming_of_age", "mythic_quest"],
    "Comedy": ["workplace_farce", "romantic_comedy", "satire"],
    "Crime": ["gangland", "procedural", "conspiracy"],
    "Documentary": ["investigative", "biographical", "social_issue"],
    "Drama": ["family_saga", "prestige_character_study", "social_drama"],
    "Fantasy": ["epic_fantasy", "urban_fantasy", "dark_fable"],
    "Horror": ["supernatural", "folk_horror", "body_horror"],
    "Mystery": ["detective_case", "period_mystery", "neo_noir"],
    "Romance": ["period_romance", "melodrama", "slow_burn"],
    "Sci-Fi": ["space_opera", "cyberpunk", "time_travel"],
    "Thriller": ["political_thriller", "psychological_thriller", "manhunt"],
    "War": ["frontline_drama", "resistance_story", "command_crisis"],
}

_GENERIC_KEYWORD_TOKENS = {
    "love", "friendship", "death", "family", "betrayal", "murder",
    "war", "hero", "villain", "investigation", "journey", "revenge",
    "honor", "courage", "justice", "freedom", "legacy", "regret",
    "ambition", "greed", "loneliness", "sacrifice", "hope",
}
_SETTING_KEYWORDS = {
    "city", "village", "island", "school", "university", "hotel", "hospital",
    "desert", "forest", "mountain", "space", "planet", "ship", "submarine",
}
_PROFESSION_KEYWORDS = {
    "detective", "police", "cop", "lawyer", "doctor", "journalist",
    "scientist", "teacher", "soldier", "pilot", "chef", "musician",
}
_RELATIONSHIP_KEYWORDS = {
    "marriage", "divorce", "friendship", "mother", "father", "daughter",
    "son", "brother", "sister", "couple", "affair", "mentor",
}
_EVENT_KEYWORDS = {
    "wedding", "funeral", "trial", "robbery", "battle", "escape",
    "mystery", "election", "pandemic", "accident", "festival", "investigation",
}
_OBJECT_KEYWORDS = {
    "artifact", "weapon", "sword", "spaceship", "car", "camera",
    "ring", "mask", "journal", "device", "computer", "phone",
}
_PLACE_KEYWORDS = {
    "new york", "london", "paris", "tokyo", "mumbai", "berlin", "los angeles",
    "brooklyn", "harlem", "suburb", "countryside", "palace", "prison",
}
_TONE_KEYWORDS = {
    "dark", "comic", "gritty", "uplifting", "violent", "romantic",
    "suspenseful", "surreal", "melancholic", "hopeful",
}
_FRANCHISE_TOKENS = {"franchise", "universe", "saga", "chronicles", "legacy", "origins"}
_SEQUEL_TOKENS = {"part", "chapter", "returns", "rise", "fall", "resurrection", "ii", "iii", "iv"}


def world_policy_path(base_dir: str | Path) -> Path:
    return Path(base_dir) / WORLD_POLICY_FILENAME


def concept_packs_path(base_dir: str | Path) -> Path:
    return Path(base_dir) / CONCEPT_PACKS_FILENAME


def year_slate_plan_path(base_dir: str | Path) -> Path:
    return Path(base_dir) / YEAR_SLATE_PLAN_FILENAME


def keyword_motif_bank_path(base_dir: str | Path) -> Path:
    return Path(base_dir) / KEYWORD_MOTIF_BANK_FILENAME


def franchise_bibles_path(base_dir: str | Path) -> Path:
    return Path(base_dir) / FRANCHISE_BIBLES_FILENAME


def identity_bank_path(base_dir: str | Path) -> Path:
    return Path(base_dir) / IDENTITY_BANK_FILENAME


def character_identity_bank_path(base_dir: str | Path) -> Path:
    return Path(base_dir) / CHARACTER_IDENTITY_BANK_FILENAME


def company_lexicon_path(base_dir: str | Path) -> Path:
    return Path(base_dir) / COMPANY_LEXICON_FILENAME


def keyword_seed_bank_path(base_dir: str | Path) -> Path:
    return Path(base_dir) / KEYWORD_SEED_BANK_FILENAME


def title_grammar_bank_path(base_dir: str | Path) -> Path:
    return Path(base_dir) / TITLE_GRAMMAR_BANK_FILENAME


def temporal_regime_plan_path(base_dir: str | Path) -> Path:
    return Path(base_dir) / TEMPORAL_REGIME_PLAN_FILENAME


def modeling_priors_path(base_dir: str | Path) -> Path:
    return Path(base_dir) / MODELING_PRIORS_FILENAME


def decision_log_dir(base_dir: str | Path) -> Path:
    return Path(base_dir) / DECISION_LOG_DIRNAME


def decision_log_path(base_dir: str | Path) -> Path:
    return decision_log_dir(base_dir) / DECISION_LOG_FILENAME


def decision_log_path_for_run(base_dir: str | Path, run_id: str) -> Path:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_id or "").strip()).strip("._")
    if not clean:
        clean = "run"
    return decision_log_dir(base_dir) / f"{clean}_{DECISION_LOG_FILENAME}"


def movie_progress_log_path(base_dir: str | Path) -> Path:
    return decision_log_dir(base_dir) / MOVIE_PROGRESS_LOG_FILENAME


def movie_progress_log_path_for_run(base_dir: str | Path, run_id: str) -> Path:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_id or "").strip()).strip("._")
    if not clean:
        clean = "run"
    return decision_log_dir(base_dir) / f"{clean}_{MOVIE_PROGRESS_LOG_FILENAME}"


def llm_usage_log_path(base_dir: str | Path) -> Path:
    return decision_log_dir(base_dir) / LLM_USAGE_LOG_FILENAME


def llm_usage_log_path_for_run(base_dir: str | Path, run_id: str) -> Path:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_id or "").strip()).strip("._")
    if not clean:
        clean = "run"
    return decision_log_dir(base_dir) / f"{clean}_{LLM_USAGE_LOG_FILENAME}"


def research_audit_path(base_dir: str | Path) -> Path:
    return decision_log_dir(base_dir) / RESEARCH_AUDIT_FILENAME


def research_audit_path_for_run(base_dir: str | Path, run_id: str) -> Path:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_id or "").strip()).strip("._")
    if not clean:
        clean = "run"
    return decision_log_dir(base_dir) / f"{clean}_{RESEARCH_AUDIT_FILENAME}"


def prompt_calibration_log_path(base_dir: str | Path) -> Path:
    return decision_log_dir(base_dir) / PROMPT_CALIBRATION_LOG_FILENAME


def comparison_report_path(base_dir: str | Path) -> Path:
    return Path(base_dir) / COMPARISON_REPORT_FILENAME


def ensure_support_dirs(base_dir: str | Path) -> None:
    decision_log_dir(base_dir).mkdir(parents=True, exist_ok=True)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def safe_load_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def append_jsonl(path: str | Path, row: Mapping[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=True, default=str))
        handle.write("\n")


def infer_market(country: str | None) -> str:
    text = str(country or "").strip()
    return DEFAULT_COUNTRY_TO_MARKET.get(text, "Global")


def normalize_counter_weights(
    values: Mapping[str, Any] | Iterable[tuple[str, Any]] | None,
    *,
    allowed: Sequence[str] | None = None,
    floor: float = 0.0,
) -> dict[str, float]:
    if values is None:
        source: dict[str, float] = {}
    elif isinstance(values, Mapping):
        source = {str(k): max(floor, safe_float(v, 0.0)) for k, v in values.items()}
    elif isinstance(values, str):
        source = {str(values): 1.0} if str(values).strip() else {}
    elif isinstance(values, (int, float, bool)):
        source = {}
    else:
        source = {}
        for item in values:
            key = None
            raw_value: Any = 1.0
            if isinstance(item, Mapping):
                key = (
                    item.get("key")
                    or item.get("name")
                    or item.get("genre")
                    or item.get("tier")
                    or item.get("country")
                    or item.get("market")
                    or item.get("label")
                )
                raw_value = item.get("weight", item.get("value", item.get("score", 1.0)))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                key = item[0]
                raw_value = item[1]
            elif isinstance(item, str):
                key = item
                raw_value = 1.0
            if key is None:
                continue
            source[str(key)] = max(floor, safe_float(raw_value, 0.0))
    if allowed is not None:
        source = {str(k): max(floor, safe_float(source.get(str(k), 0.0), 0.0)) for k in allowed}
    total = float(sum(source.values()))
    if total <= 0:
        if not source:
            return {}
        weight = 1.0 / max(1, len(source))
        return {k: weight for k in source}
    return {k: float(v) / total for k, v in source.items()}


def confidence_from_scores(scores: Sequence[float]) -> float:
    vals = [max(0.0, safe_float(v, 0.0)) for v in scores]
    if not vals:
        return 0.0
    ordered = sorted(vals, reverse=True)
    best = ordered[0]
    second = ordered[1] if len(ordered) > 1 else 0.0
    total = max(1e-9, sum(ordered))
    margin = max(0.0, best - second)
    concentration = best / total
    return float(max(0.0, min(1.0, 0.55 * concentration + 0.45 * min(1.0, margin / max(0.05, best)))))


def rerank_budget_for_movies(n_movies: int, explicit_budget: int | None = None) -> int:
    if explicit_budget is not None:
        return max(0, int(explicit_budget))
    estimated = int(round(0.08 * max(0, int(n_movies))))
    return max(100, min(750, estimated))


def keyword_rerank_budget_for_movies(n_movies: int, explicit_budget: int | None = None) -> int:
    if explicit_budget is not None:
        return max(0, int(explicit_budget))
    estimated = int(round(0.15 * max(0, int(n_movies))))
    return max(250, min(1500, estimated))


def year_bucket_id(year: int, start_year: int | None = None, bucket_size: int = DEFAULT_BUCKET_SIZE) -> str:
    y = int(year)
    if start_year is None:
        start = y - ((y - 1) % max(1, int(bucket_size)))
    else:
        start = int(start_year) + ((y - int(start_year)) // max(1, int(bucket_size))) * max(1, int(bucket_size))
    end = start + max(1, int(bucket_size)) - 1
    return f"{start:04d}-{end:04d}"


def iter_year_buckets(start_year: int, end_year: int, bucket_size: int = DEFAULT_BUCKET_SIZE) -> list[dict[str, int | str]]:
    lo = int(start_year)
    hi = int(end_year)
    if hi < lo:
        lo, hi = hi, lo
    buckets: list[dict[str, int | str]] = []
    cursor = lo
    while cursor <= hi:
        bucket_end = min(hi, cursor + max(1, int(bucket_size)) - 1)
        buckets.append(
            {
                "bucket_id": f"{cursor:04d}-{bucket_end:04d}",
                "start_year": int(cursor),
                "end_year": int(bucket_end),
            }
        )
        cursor = bucket_end + 1
    return buckets


def bucket_for_year(world_policy: Mapping[str, Any] | None, year: int) -> dict[str, Any]:
    policy = world_policy or {}
    buckets = policy.get("year_buckets") if isinstance(policy, Mapping) else None
    if isinstance(buckets, list):
        for bucket in buckets:
            if not isinstance(bucket, Mapping):
                continue
            lo = safe_int(bucket.get("start_year"), 0)
            hi = safe_int(bucket.get("end_year"), lo)
            if lo <= int(year) <= hi:
                return dict(bucket)
    bucket_id = year_bucket_id(int(year), safe_int(policy.get("start_year"), int(year)))
    return {
        "bucket_id": bucket_id,
        "start_year": int(year),
        "end_year": int(year),
        "genre_bias": {},
        "country_bias": {},
        "market_bias": {},
        "franchise_pressure": 0.25,
        "sequel_pressure": 0.25,
    }


def resolve_year_bounds(
    *,
    years: Iterable[Any] | None = None,
    fallback_start: int = 1950,
    fallback_end: int = 2025,
) -> tuple[int, int]:
    numeric = []
    iterable = years if years is not None else []
    for value in iterable:
        try:
            numeric.append(int(value))
        except Exception:
            continue
    if not numeric:
        return int(fallback_start), int(fallback_end)
    return int(min(numeric)), int(max(numeric))


def _relative_year_phase(year: int, *, start_year: int, end_year: int) -> str:
    lo = int(start_year)
    hi = int(end_year)
    if hi <= lo:
        return "late"
    frac = (int(year) - lo) / float(max(1, hi - lo))
    if frac < 0.34:
        return "early"
    if frac < 0.68:
        return "mid"
    return "late"


def _top_values_from_rows(rows: Sequence[Mapping[str, Any]] | None, key: str, *, top_n: int = 6) -> list[str]:
    counter: Counter[str] = Counter()
    for row in rows or []:
        raw = row.get(key)
        if isinstance(raw, str):
            parts = [part.strip() for part in raw.replace("|", ",").replace(";", ",").split(",") if part.strip()]
        elif isinstance(raw, (list, tuple)):
            parts = [str(part).strip() for part in raw if str(part).strip()]
        else:
            parts = []
        for part in parts:
            counter[part] += 1
    return [name for name, _count in counter.most_common(top_n)]


def _weighted_counter_from_rows(
    rows: Sequence[Mapping[str, Any]] | None,
    key: str,
    *,
    allowed: Sequence[str] | None = None,
    weight: float = 1.0,
) -> Counter[str]:
    allowed_set = {str(item) for item in (allowed or []) if str(item).strip()} or None
    counter: Counter[str] = Counter()
    for row in rows or []:
        raw = row.get(key)
        if isinstance(raw, str):
            parts = [part.strip() for part in raw.replace("|", ",").replace(";", ",").split(",") if part.strip()]
        elif isinstance(raw, (list, tuple)):
            parts = [str(part).strip() for part in raw if str(part).strip()]
        else:
            parts = []
        for part in parts:
            if allowed_set is not None and part not in allowed_set:
                continue
            counter[part] += float(weight)
    return counter


def _stable_rotation(values: Sequence[str], seed: str) -> list[str]:
    items = [str(value) for value in values if str(value).strip()]
    if not items:
        return []
    offset = sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(seed or ""))) % len(items)
    return items[offset:] + items[:offset]


def _strategy_catalog(genres: Sequence[str]) -> list[dict[str, Any]]:
    genre_pool = [str(genre) for genre in genres if str(genre).strip()]
    if not genre_pool:
        genre_pool = ["Drama", "Action", "Thriller", "Comedy", "Sci-Fi", "Fantasy"]

    def choose(*preferred: str) -> str:
        for candidate in preferred:
            if candidate in genre_pool:
                return candidate
        return genre_pool[0]

    lead = genre_pool[0]
    second = genre_pool[1] if len(genre_pool) > 1 else lead
    third = genre_pool[2] if len(genre_pool) > 2 else second
    return [
        {
            "strategy_tag": "prestige_drama",
            "label": "Prestige drama slate",
            "genre_focus": normalize_counter_weights(
                {
                    choose("Drama"): 5,
                    choose("Romance", "Mystery"): 2,
                    choose("Mystery", "Documentary", third): 1,
                },
                allowed=genre_pool,
            ),
            "tier_bias": normalize_counter_weights({"A": 3, "Mid": 2, "Indie": 2, "Epic": 1, "Micro": 1}),
            "title_style": "elegant",
            "cast_chemistry_target": "prestige_pairing",
        },
        {
            "strategy_tag": "event_franchise",
            "label": "Event-driven franchise builder",
            "genre_focus": normalize_counter_weights(
                {
                    lead: 3,
                    second: 3,
                    choose("Sci-Fi", "Fantasy", "Action"): 3,
                    choose("Fantasy", "Action", third): 3,
                    choose("Action", "Thriller", lead): 3,
                },
                allowed=genre_pool,
            ),
            "tier_bias": normalize_counter_weights({"Epic": 5, "A": 4, "Mid": 1, "Indie": 0.5, "Micro": 0.2}),
            "title_style": "bold",
            "cast_chemistry_target": "star_vehicle",
        },
        {
            "strategy_tag": "genre_lab",
            "label": "Risk-seeking genre lab",
            "genre_focus": normalize_counter_weights(
                {
                    choose("Horror", "Thriller", third): 3,
                    choose("Thriller", "Mystery", second): 3,
                    third: 2,
                    choose("Sci-Fi", "Fantasy", lead): 2,
                },
                allowed=genre_pool,
            ),
            "tier_bias": normalize_counter_weights({"Mid": 3, "Indie": 3, "Micro": 2, "A": 1, "Epic": 0.2}),
            "title_style": "pulp",
            "cast_chemistry_target": "volatile_ensemble",
        },
        {
            "strategy_tag": "audience_broadcaster",
            "label": "Family-friendly wide audience slate",
            "genre_focus": normalize_counter_weights(
                {
                    choose("Comedy", second): 3,
                    choose("Animation", "Fantasy", third): 3,
                    choose("Fantasy", "Animation", lead): 2,
                    choose("Drama", "Romance", lead): 1,
                },
                allowed=genre_pool,
            ),
            "tier_bias": normalize_counter_weights({"A": 2, "Mid": 3, "Indie": 2, "Epic": 1.5, "Micro": 0.5}),
            "title_style": "uplifting",
            "cast_chemistry_target": "balanced_ensemble",
        },
    ]


def _default_bucket_bias(
    bucket: Mapping[str, Any],
    *,
    genres: Sequence[str],
    countries: Sequence[str],
    genre_base: Mapping[str, Any] | None = None,
    country_base: Mapping[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    lo = safe_int(bucket.get("start_year"), 1950)
    hi = safe_int(bucket.get("end_year"), lo)
    phase = _relative_year_phase(
        (lo + hi) // 2,
        start_year=safe_int(bucket.get("range_start_year"), lo),
        end_year=safe_int(bucket.get("range_end_year"), hi),
    )
    genre_seed = normalize_counter_weights(genre_base, allowed=genres) if isinstance(genre_base, Mapping) else {}
    genre_bias = {
        str(genre): max(0.01, safe_float(genre_seed.get(str(genre)), 0.0) or 1.0)
        for genre in genres
    }
    if phase == "early":
        for genre in ("Drama", "Crime", "War", "Mystery", "Romance"):
            if genre in genre_bias:
                genre_bias[genre] += 0.22
    elif phase == "mid":
        for genre in ("Action", "Romance", "Thriller", "Crime", "Comedy"):
            if genre in genre_bias:
                genre_bias[genre] += 0.18
    else:
        for genre in ("Sci-Fi", "Fantasy", "Animation", "Thriller", "Horror"):
            if genre in genre_bias:
                genre_bias[genre] += 0.2

    country_seed = normalize_counter_weights(country_base, allowed=countries) if isinstance(country_base, Mapping) else {}
    country_bias = {
        str(country): max(0.01, safe_float(country_seed.get(str(country)), 0.0) or 1.0)
        for country in countries
    }
    for country in countries:
        market = infer_market(country)
        if phase == "early":
            if market == "Europe":
                country_bias[country] += 0.12
            elif market == "North America":
                country_bias[country] += 0.08
        elif phase == "mid":
            if market == "North America":
                country_bias[country] += 0.12
            elif market == "Asia":
                country_bias[country] += 0.1
            elif market == "Europe":
                country_bias[country] += 0.06
        else:
            if market == "Asia":
                country_bias[country] += 0.14
            elif market in {"Latin America", "South America", "Africa", "Middle East"}:
                country_bias[country] += 0.08
            elif market == "North America":
                country_bias[country] += 0.05

    market_counter: Counter[str] = Counter()
    for country, weight in country_bias.items():
        market_counter[infer_market(country)] += float(weight)
    market_counter["Global"] += 0.65 if phase == "early" else 0.95 if phase == "mid" else 1.15
    market_bias = normalize_counter_weights(market_counter)
    return (
        normalize_counter_weights(genre_bias, allowed=genres),
        normalize_counter_weights(country_bias, allowed=countries),
        market_bias,
    )


def build_default_world_policy(
    *,
    start_year: int,
    end_year: int,
    genres: Sequence[str],
    countries: Sequence[str],
    tiers: Sequence[str],
    company_rows: Sequence[Mapping[str, Any]] | None = None,
    person_rows: Sequence[Mapping[str, Any]] | None = None,
    bucket_size: int = DEFAULT_BUCKET_SIZE,
) -> dict[str, Any]:
    bucket_defs = iter_year_buckets(start_year, end_year, bucket_size=bucket_size)
    company_rows = list(company_rows or [])
    person_rows = list(person_rows or [])

    genre_counter = Counter()
    genre_counter.update(_weighted_counter_from_rows(company_rows, "specialty_genres", allowed=genres, weight=2.0))
    genre_counter.update(_weighted_counter_from_rows(person_rows, "genre_affinity", allowed=genres, weight=1.0))
    if not genre_counter:
        genre_counter.update({str(genre): 1.0 for genre in genres})
    genre_base = normalize_counter_weights(genre_counter, allowed=genres)
    ranked_genres = [genre for genre, _weight in sorted(genre_base.items(), key=lambda item: item[1], reverse=True)]

    country_counter: Counter[str] = Counter()
    for row in company_rows:
        country = str(row.get("country", "")).strip()
        if country in countries:
            country_counter[country] += 2.0
    for row in person_rows:
        country = str(row.get("country", "")).strip()
        if country in countries:
            country_counter[country] += 0.5
    if not country_counter:
        country_counter.update({str(country): 1.0 for country in countries})
    country_base = normalize_counter_weights(country_counter, allowed=countries)

    strategies = _strategy_catalog(ranked_genres)
    company_assignments: dict[str, str] = {}
    for idx, row in enumerate(company_rows):
        cid = safe_int(row.get("company_id"), idx + 1)
        tier = str(row.get("tier", "Mid-Budget"))
        genres_for_company = _top_values_from_rows([row], "specialty_genres", top_n=2)
        if tier in ("Global", "Major"):
            tag = "event_franchise"
        elif any(genre in {"Drama", "Romance"} for genre in genres_for_company):
            tag = "prestige_drama"
        elif any(genre in {"Comedy", "Animation", "Fantasy"} for genre in genres_for_company):
            tag = "audience_broadcaster"
        else:
            tag = "genre_lab"
        company_assignments[str(cid)] = tag

    year_buckets = []
    for bucket in bucket_defs:
        genre_bias, country_bias, market_bias = _default_bucket_bias(
            bucket,
            genres=genres,
            countries=countries,
            genre_base=genre_base,
            country_base=country_base,
        )
        phase = _relative_year_phase(
            safe_int(bucket.get("start_year"), start_year),
            start_year=int(start_year),
            end_year=int(end_year),
        )
        lead_genre = max(genre_bias.items(), key=lambda item: item[1])[0] if genre_bias else "Drama"
        franchise_pressure = 0.16 if phase == "early" else 0.27 if phase == "mid" else 0.38
        sequel_pressure = 0.12 if phase == "early" else 0.22 if phase == "mid" else 0.31
        if lead_genre in {"Action", "Sci-Fi", "Fantasy", "Animation"}:
            franchise_pressure += 0.06
            sequel_pressure += 0.04
        year_buckets.append(
            {
                "bucket_id": str(bucket["bucket_id"]),
                "start_year": int(bucket["start_year"]),
                "end_year": int(bucket["end_year"]),
                "range_start_year": int(start_year),
                "range_end_year": int(end_year),
                "genre_bias": genre_bias,
                "country_bias": country_bias,
                "market_bias": market_bias,
                "franchise_pressure": round(min(1.0, franchise_pressure), 3),
                "sequel_pressure": round(min(1.0, sequel_pressure), 3),
            }
        )

    compatibility = {
        "director": {"genre_match": 0.34, "country_match": 0.14, "style_match": 0.24, "strategy_match": 0.28},
        "company": {"tier_match": 0.28, "genre_match": 0.34, "strategy_match": 0.24, "market_match": 0.14},
        "cast": {"genre_match": 0.26, "chemistry_match": 0.26, "market_match": 0.12, "talent_boost": 0.36},
        "title": {"style_match": 0.46, "genre_match": 0.28, "release_alignment": 0.26},
        "keywords": {"genre_match": 0.42, "seed_cluster_match": 0.36, "strategy_match": 0.22},
    }
    talent_boost_rules = [
        {"rule_id": "prestige_veteran", "career_stage": "veteran", "genre": "Drama", "boost": 1.14},
        {"rule_id": "genre_specialist_horror", "style_tag": "brooding", "genre": "Horror", "boost": 1.12},
        {"rule_id": "blockbuster_charisma", "career_stage": "prime", "genre": "Action", "boost": 1.10},
        {"rule_id": "rising_auteur", "career_stage": "rising", "genre": "Sci-Fi", "boost": 1.08},
    ]

    return {
        "meta": {
            "schema_version": 1,
            "bucket_size": int(bucket_size),
            "start_year": int(start_year),
            "end_year": int(end_year),
            "person_count": len(person_rows or []),
            "company_count": len(company_rows or []),
        },
        "start_year": int(start_year),
        "end_year": int(end_year),
        "country_market_map": {country: infer_market(country) for country in countries},
        "year_buckets": year_buckets,
        "company_strategies": strategies,
        "company_strategy_assignments": company_assignments,
        "talent_boost_rules": talent_boost_rules,
        "compatibility": compatibility,
        "notes": {
            "generator_mode": "fallback",
            "intent": "Structured policy for movie and edge selection.",
        },
    }


def normalize_world_policy(
    payload: Any,
    *,
    start_year: int,
    end_year: int,
    genres: Sequence[str],
    countries: Sequence[str],
    tiers: Sequence[str],
    company_rows: Sequence[Mapping[str, Any]] | None = None,
    person_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    fallback = build_default_world_policy(
        start_year=start_year,
        end_year=end_year,
        genres=genres,
        countries=countries,
        tiers=tiers,
        company_rows=company_rows,
        person_rows=person_rows,
    )
    if not isinstance(payload, Mapping):
        return fallback

    out = dict(fallback)
    out["meta"] = dict(fallback.get("meta", {}))
    if isinstance(payload.get("meta"), Mapping):
        out["meta"].update(dict(payload["meta"]))
    out["notes"] = dict(fallback.get("notes", {}))
    if isinstance(payload.get("notes"), Mapping):
        out["notes"].update(dict(payload["notes"]))

    country_market_map = dict(fallback.get("country_market_map", {}))
    raw_country_market = payload.get("country_market_map")
    if isinstance(raw_country_market, Mapping):
        valid_markets = {str(value) for value in fallback.get("country_market_map", {}).values() if str(value).strip()}
        for country in countries:
            fallback_market = str(country_market_map.get(country, infer_market(country)))
            candidate = str(raw_country_market.get(country, fallback_market)).strip()
            if not candidate or candidate == "Global" or candidate not in valid_markets:
                candidate = fallback_market
            country_market_map[country] = candidate
    out["country_market_map"] = country_market_map

    bucket_by_id = {str(bucket["bucket_id"]): dict(bucket) for bucket in fallback["year_buckets"]}
    raw_buckets = payload.get("year_buckets")
    if isinstance(raw_buckets, list):
        for raw in raw_buckets:
            if not isinstance(raw, Mapping):
                continue
            bucket_id = str(raw.get("bucket_id") or year_bucket_id(safe_int(raw.get("start_year"), start_year), start_year))
            bucket = dict(bucket_by_id.get(bucket_id, {}))
            if not bucket:
                bucket = {
                    "bucket_id": bucket_id,
                    "start_year": safe_int(raw.get("start_year"), start_year),
                    "end_year": safe_int(raw.get("end_year"), end_year),
                }
            bucket["genre_bias"] = normalize_counter_weights(raw.get("genre_bias"), allowed=genres) or bucket.get("genre_bias", {})
            bucket["country_bias"] = normalize_counter_weights(raw.get("country_bias"), allowed=countries) or bucket.get("country_bias", {})
            raw_market_bias = raw.get("market_bias")
            if isinstance(raw_market_bias, Mapping):
                bucket["market_bias"] = normalize_counter_weights(raw_market_bias)
            bucket["franchise_pressure"] = max(0.0, min(1.0, safe_float(raw.get("franchise_pressure"), bucket.get("franchise_pressure", 0.25))))
            bucket["sequel_pressure"] = max(0.0, min(1.0, safe_float(raw.get("sequel_pressure"), bucket.get("sequel_pressure", 0.25))))
            bucket_by_id[bucket_id] = bucket
    out["year_buckets"] = sorted(bucket_by_id.values(), key=lambda bucket: (safe_int(bucket.get("start_year"), start_year), str(bucket.get("bucket_id"))))

    out["company_strategies"] = list(fallback.get("company_strategies", []))
    raw_strategies = payload.get("company_strategies")
    if isinstance(raw_strategies, list):
        cleaned = []
        for raw in raw_strategies:
            if not isinstance(raw, Mapping):
                continue
            strategy_tag = str(raw.get("strategy_tag") or raw.get("id") or "").strip()
            if not strategy_tag:
                continue
            cleaned.append(
                {
                    "strategy_tag": strategy_tag,
                    "label": str(raw.get("label") or strategy_tag.replace("_", " ").title()),
                    "genre_focus": normalize_counter_weights(raw.get("genre_focus"), allowed=genres),
                    "tier_bias": normalize_counter_weights(raw.get("tier_bias"), allowed=tiers),
                    "title_style": str(raw.get("title_style") or "clean"),
                    "cast_chemistry_target": str(raw.get("cast_chemistry_target") or "balanced_ensemble"),
                }
            )
        if cleaned:
            out["company_strategies"] = cleaned

    assignments = dict(fallback.get("company_strategy_assignments", {}))
    raw_assignments = payload.get("company_strategy_assignments")
    valid_tags = {row["strategy_tag"] for row in out["company_strategies"] if isinstance(row, Mapping) and row.get("strategy_tag")}
    if isinstance(raw_assignments, Mapping):
        for key, value in raw_assignments.items():
            tag = str(value)
            if tag in valid_tags:
                assignments[str(key)] = tag
    out["company_strategy_assignments"] = assignments

    raw_rules = payload.get("talent_boost_rules")
    if isinstance(raw_rules, list):
        cleaned_rules = []
        for idx, rule in enumerate(raw_rules):
            if not isinstance(rule, Mapping):
                continue
            cleaned_rules.append(
                {
                    "rule_id": str(rule.get("rule_id") or f"rule_{idx+1}"),
                    "career_stage": str(rule.get("career_stage") or ""),
                    "style_tag": str(rule.get("style_tag") or ""),
                    "genre": str(rule.get("genre") or ""),
                    "boost": max(0.8, min(1.4, safe_float(rule.get("boost"), 1.0))),
                }
            )
        if cleaned_rules:
            out["talent_boost_rules"] = cleaned_rules

    compatibility = dict(fallback.get("compatibility", {}))
    raw_compatibility = payload.get("compatibility")
    if isinstance(raw_compatibility, Mapping):
        for section, base_values in compatibility.items():
            raw_values = raw_compatibility.get(section)
            if isinstance(raw_values, Mapping):
                compatibility[section] = {str(k): max(0.0, safe_float(v, base_values.get(k, 0.0))) for k, v in raw_values.items()}
    out["compatibility"] = compatibility
    out["start_year"] = int(start_year)
    out["end_year"] = int(end_year)
    used_llm_payload = any(
        (
            isinstance(raw_country_market, Mapping) and bool(raw_country_market),
            isinstance(raw_buckets, list) and any(isinstance(row, Mapping) for row in raw_buckets),
            isinstance(raw_strategies, list) and any(isinstance(row, Mapping) for row in raw_strategies),
            isinstance(raw_assignments, Mapping) and bool(raw_assignments),
            isinstance(raw_rules, list) and any(isinstance(row, Mapping) for row in raw_rules),
            isinstance(raw_compatibility, Mapping) and bool(raw_compatibility),
        )
    )
    if used_llm_payload:
        out["meta"]["generator_mode"] = "llm"
        out["notes"]["generator_mode"] = "llm"
    return out


def build_concept_pack_slots(
    world_policy: Mapping[str, Any],
    *,
    genres: Sequence[str],
    tiers: Sequence[str],
    countries: Sequence[str],
    n_movies: int = 5000,
) -> list[dict[str, Any]]:
    bucket_rows = list(world_policy.get("year_buckets", [])) if isinstance(world_policy, Mapping) else []
    if not bucket_rows:
        return []

    def _rotation_seed(text: str) -> int:
        token = str(text or "")
        return sum((idx + 1) * ord(ch) for idx, ch in enumerate(token))

    def _rotate(values: Sequence[str], offset: int) -> list[str]:
        rows = [str(value) for value in values if str(value).strip()]
        if not rows:
            return []
        start = int(offset) % len(rows)
        return rows[start:] + rows[:start]

    country_market_map = world_policy.get("country_market_map", {}) if isinstance(world_policy, Mapping) else {}
    slots: list[dict[str, Any]] = []
    n_buckets = max(1, len(bucket_rows))
    target_cap = max(120, min(960, max(1, int(round(n_movies * 0.12)))))
    per_bucket_cap = max(10, int(round(target_cap / n_buckets)))
    for bucket in bucket_rows:
        if not isinstance(bucket, Mapping):
            continue
        genre_bias = normalize_counter_weights(bucket.get("genre_bias"), allowed=genres)
        sorted_genres = sorted(genre_bias.items(), key=lambda item: item[1], reverse=True)
        top_genres = [genre for genre, _weight in sorted_genres[: min(6, max(3, len(sorted_genres)))]]
        if not top_genres:
            top_genres = list(genres[:4])
        country_bias = normalize_counter_weights(bucket.get("country_bias"), allowed=countries)
        sorted_countries = sorted(country_bias.items(), key=lambda item: item[1], reverse=True)
        start_year = safe_int(bucket.get("start_year"), safe_int(world_policy.get("start_year"), 1975))
        end_year = safe_int(bucket.get("end_year"), safe_int(world_policy.get("end_year"), 2025))
        bucket_id = str(bucket.get("bucket_id"))
        ordered_countries = [str(country) for country, _weight in sorted_countries if str(country).strip()]
        if not ordered_countries:
            ordered_countries = list(countries[:8])
        primary_pool_size = min(len(ordered_countries), 14)
        exploratory_pool_size = min(max(0, len(ordered_countries) - primary_pool_size), 8)
        rotated_primary = _rotate(ordered_countries[:primary_pool_size], _rotation_seed(bucket_id))
        rotated_exploratory = _rotate(
            ordered_countries[primary_pool_size : primary_pool_size + exploratory_pool_size],
            _rotation_seed(f"{bucket_id}:{start_year}:{end_year}"),
        )
        market_groups: dict[str, list[str]] = defaultdict(list)
        for country in rotated_primary + rotated_exploratory + list(countries):
            country_name = str(country)
            if not country_name:
                continue
            market_groups[infer_market(country_name)].append(country_name)
        market_order = _stable_rotation(
            [market for market in ("North America", "Europe", "Asia", "Latin America", "South America", "Africa", "Middle East", "Oceania") if market_groups.get(market)],
            bucket_id,
        )
        top_countries: list[str] = []
        round_idx = 0
        while len(top_countries) < 16:
            progressed = False
            for market in market_order:
                rows = market_groups.get(market, [])
                if round_idx >= len(rows):
                    continue
                country_name = str(rows[round_idx])
                if not country_name or country_name in top_countries:
                    continue
                top_countries.append(country_name)
                progressed = True
                if len(top_countries) >= 16:
                    break
            if not progressed:
                break
            round_idx += 1

        bucket_slots: list[tuple[float, dict[str, Any]]] = []
        tier_weight_map = {"Epic": 1.10, "A": 1.02, "Mid": 1.00, "Indie": 0.98, "Micro": 0.94}
        for genre_idx, genre in enumerate(top_genres):
            genre_weight = safe_float(genre_bias.get(genre), 0.0) or max(0.01, 1.0 / max(1, len(top_genres)))
            for tier in tiers:
                tier_name = str(tier)
                tier_weight = tier_weight_map.get(tier_name, 1.0)
                country_limit = 5 if tier_name == "Epic" else 4 if tier_name in {"A", "Mid"} else 3
                for country_idx, country in enumerate(top_countries[:country_limit]):
                    country_weight = safe_float(country_bias.get(country), 0.0) or max(0.01, 1.0 / max(1, len(top_countries)))
                    score = (genre_weight * 1.8) + (country_weight * 1.2) + tier_weight
                    score -= 0.012 * float(country_idx)
                    score -= 0.02 * float(genre_idx)
                    if country_idx >= 3:
                        score += 0.05
                    if infer_market(country) not in {"North America", "Europe", "Asia"}:
                        score += 0.04
                    bucket_slots.append(
                        (
                            score,
                            {
                                "bucket_id": bucket_id,
                                "start_year": start_year,
                                "end_year": end_year,
                                "genre": str(genre),
                                "tier": tier_name,
                                "country": str(country),
                                "market": str(country_market_map.get(country, infer_market(country))),
                            },
                        )
                    )
        bucket_slots.sort(key=lambda item: item[0], reverse=True)
        country_counts: Counter[str] = Counter()
        market_counts: Counter[str] = Counter()
        selected_bucket_slots: list[dict[str, Any]] = []
        target_unique_countries = min(max(6, min(10, len(top_countries))), per_bucket_cap)
        max_per_country = max(2, int(round(per_bucket_cap * 0.22)))
        available_markets = {
            str(slot.get("market", "Global"))
            for _score, slot in bucket_slots
            if str(slot.get("market", "Global")).strip()
        }
        target_unique_markets = min(max(2, len(available_markets)), 5)
        max_per_market = max(2, int(round(per_bucket_cap * 0.38)))

        for _score, slot in bucket_slots:
            country = str(slot.get("country", ""))
            market = str(slot.get("market", "Global"))
            if not country or country_counts[country] > 0:
                continue
            if market_counts[market] > 0 and len(market_counts) < target_unique_markets:
                continue
            selected_bucket_slots.append(slot)
            country_counts[country] += 1
            market_counts[market] += 1
            if len(country_counts) >= target_unique_countries or len(selected_bucket_slots) >= per_bucket_cap:
                break

        for _score, slot in bucket_slots:
            if len(selected_bucket_slots) >= per_bucket_cap:
                break
            country = str(slot.get("country", ""))
            market = str(slot.get("market", "Global"))
            if not country or country_counts[country] >= max_per_country or market_counts[market] >= max_per_market:
                continue
            slot_key = (
                str(slot.get("genre", "")),
                str(slot.get("tier", "")),
                str(slot.get("country", "")),
            )
            if any(
                (
                    str(existing.get("genre", "")),
                    str(existing.get("tier", "")),
                    str(existing.get("country", "")),
                ) == slot_key
                for existing in selected_bucket_slots
            ):
                continue
            selected_bucket_slots.append(slot)
            country_counts[country] += 1
            market_counts[market] += 1

        for slot in selected_bucket_slots:
            slots.append(slot)
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for slot in slots:
        key = (str(slot.get("bucket_id")), str(slot.get("genre")), str(slot.get("tier")), str(slot.get("country")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(slot)
    return deduped[:target_cap]


def _default_pack(slot: Mapping[str, Any], idx: int) -> dict[str, Any]:
    genre = str(slot.get("genre", "Drama"))
    tier = str(slot.get("tier", "Mid"))
    country = str(slot.get("country", "USA"))
    market = str(slot.get("market") or infer_market(country))
    start_year = safe_int(slot.get("start_year"), 1975)
    end_year = safe_int(slot.get("end_year"), start_year)
    bucket_id = str(slot.get("bucket_id") or f"{start_year:04d}-{end_year:04d}")
    return {
        "pack_id": f"pack_{idx:04d}",
        "bucket_id": bucket_id,
        "start_year": start_year,
        "end_year": end_year,
        "genre": genre,
        "tier": tier,
        "country": country,
        "market": market,
        "premise_archetype": {
            "Action": "escalating pursuit across hostile territory",
            "Animation": "found-family quest through a changing world",
            "Comedy": "status scramble inside a social machine",
            "Crime": "loyalty fracture inside a criminal network",
            "Documentary": "institutional story revealed through access",
            "Drama": "private fracture with public consequences",
            "Fantasy": "succession struggle inside a mythic order",
            "Horror": "containment breach in a closed environment",
            "Mystery": "buried truth reopened by a new witness",
            "Romance": "intimacy tested by timing and ambition",
            "Sci-Fi": "technology rupture reshaping human ties",
            "Thriller": "trust collapse under escalating pressure",
            "War": "survival dilemma inside a failing command chain",
        }.get(genre, f"{genre.lower()} pressure-cooker story"),
        "conflict_pattern": DEFAULT_CONFLICT_BY_GENRE.get(genre, "institutional pressure cooker"),
        "relationship_motif": DEFAULT_RELATIONSHIP_BY_GENRE.get(genre, "team under strain"),
        "ensemble_shape": "expansive_ensemble" if tier in ("Epic", "A") else "focused_ensemble",
        "tone_intensity": DEFAULT_TONE_BY_GENRE.get(genre, "grounded"),
        "keyword_seed_cluster": [
            genre.lower(),
            _default_subgenres_for_genre(genre)[0],
            market.lower().replace(" ", "_"),
            tier.lower(),
        ],
        "title_style": DEFAULT_STYLE_BY_GENRE.get(genre, "clean"),
        "tagline_style": DEFAULT_STYLE_BY_GENRE.get(genre, "clean"),
        "company_strategy_tag": "event_franchise" if tier in ("Epic", "A") else "prestige_drama" if genre in {"Drama", "Romance"} else "genre_lab",
        "cast_chemistry_target": DEFAULT_CHEMISTRY_BY_TIER.get(tier, "balanced_ensemble"),
        "franchise_eligible": bool(tier in ("Epic", "A") or genre in {"Action", "Sci-Fi", "Fantasy", "Animation", "Horror"}),
        "release_season_bias": DEFAULT_SEASON_BY_GENRE.get(genre, "spring"),
    }


def normalize_concept_packs(
    payload: Any,
    *,
    world_policy: Mapping[str, Any],
    genres: Sequence[str],
    tiers: Sequence[str],
    countries: Sequence[str],
    n_movies: int = 5000,
) -> dict[str, Any]:
    slots = build_concept_pack_slots(world_policy, genres=genres, tiers=tiers, countries=countries, n_movies=n_movies)
    slot_lookup = {(slot["bucket_id"], slot["genre"], slot["tier"], slot["country"]): slot for slot in slots}
    valid_tags = {
        str(row.get("strategy_tag"))
        for row in world_policy.get("company_strategies", [])
        if isinstance(row, Mapping) and row.get("strategy_tag")
    }
    raw_packs = payload.get("packs") if isinstance(payload, Mapping) else None
    cleaned_by_slot: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    if isinstance(raw_packs, list):
        for idx, raw in enumerate(raw_packs, start=1):
            if not isinstance(raw, Mapping):
                continue
            start_year = safe_int(raw.get("start_year"), safe_int(world_policy.get("start_year"), 1975))
            end_year = safe_int(raw.get("end_year"), start_year)
            bucket_id = str(raw.get("bucket_id") or year_bucket_id(start_year, safe_int(world_policy.get("start_year"), start_year)))
            genre = str(raw.get("genre") or "Drama")
            tier = str(raw.get("tier") or "Mid")
            country = str(raw.get("country") or (countries[0] if countries else "USA"))
            slot = slot_lookup.get((bucket_id, genre, tier, country), {
                "bucket_id": bucket_id,
                "start_year": start_year,
                "end_year": end_year,
                "genre": genre,
                "tier": tier,
                "country": country,
                "market": infer_market(country),
            })
            pack = _default_pack(slot, idx)
            pack["pack_id"] = str(raw.get("pack_id") or pack["pack_id"])
            for field in (
                "premise_archetype",
                "conflict_pattern",
                "relationship_motif",
                "ensemble_shape",
                "tone_intensity",
                "title_style",
                "tagline_style",
                "cast_chemistry_target",
                "release_season_bias",
            ):
                if raw.get(field):
                    pack[field] = str(raw[field])
            keyword_seed_cluster = raw.get("keyword_seed_cluster")
            if isinstance(keyword_seed_cluster, str):
                pack["keyword_seed_cluster"] = [part.strip() for part in keyword_seed_cluster.replace("|", ",").split(",") if part.strip()]
            elif isinstance(keyword_seed_cluster, list):
                pack["keyword_seed_cluster"] = [str(item).strip() for item in keyword_seed_cluster if str(item).strip()]
            strategy_tag = str(raw.get("company_strategy_tag") or pack["company_strategy_tag"])
            pack["company_strategy_tag"] = strategy_tag if strategy_tag in valid_tags or not valid_tags else next(iter(valid_tags))
            pack["franchise_eligible"] = bool(raw.get("franchise_eligible", pack["franchise_eligible"]))
            cleaned_by_slot[(pack["bucket_id"], pack["genre"], pack["tier"], pack["country"])] = pack
    for idx, slot in enumerate(slots, start=1):
        slot_key = (slot["bucket_id"], slot["genre"], slot["tier"], slot["country"])
        if slot_key not in cleaned_by_slot:
            cleaned_by_slot[slot_key] = _default_pack(slot, idx)
    if not cleaned_by_slot and slots:
        cleaned_by_slot = {
            (slot["bucket_id"], slot["genre"], slot["tier"], slot["country"]): _default_pack(slot, idx + 1)
            for idx, slot in enumerate(slots)
        }
    cleaned = sorted(
        cleaned_by_slot.values(),
        key=lambda row: (
            safe_int(row.get("start_year"), 1975),
            str(row.get("genre", "")),
            str(row.get("tier", "")),
            str(row.get("country", "")),
        ),
    )
    return {
        "meta": {
            "schema_version": 1,
            "count": len(cleaned),
            "generator_mode": "llm" if isinstance(payload, Mapping) and isinstance(raw_packs, list) and raw_packs else "fallback",
        },
        "packs": cleaned,
    }


def index_concept_packs(packs_payload: Mapping[str, Any] | None) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    payload = packs_payload or {}
    for pack in payload.get("packs", []):
        if not isinstance(pack, Mapping):
            continue
        bucket_id = str(pack.get("bucket_id") or year_bucket_id(safe_int(pack.get("start_year"), 1975)))
        grouped[bucket_id]["*"].append(dict(pack))
        grouped[bucket_id][str(pack.get("genre") or "*")].append(dict(pack))
        grouped[bucket_id][f"{pack.get('genre')}|{pack.get('tier')}"].append(dict(pack))
        grouped[bucket_id][f"{pack.get('genre')}|{pack.get('tier')}|{pack.get('market')}"].append(dict(pack))
        grouped[bucket_id][f"{pack.get('genre')}|{pack.get('tier')}|{pack.get('country')}"].append(dict(pack))
    return {bucket: dict(values) for bucket, values in grouped.items()}


def resolve_company_strategy(world_policy: Mapping[str, Any] | None, company_id: int | str | None, fallback: str = "genre_lab") -> str:
    policy = world_policy or {}
    assignments = policy.get("company_strategy_assignments", {}) if isinstance(policy, Mapping) else {}
    tag = None
    if isinstance(assignments, Mapping) and company_id is not None:
        tag = assignments.get(str(company_id))
    if tag:
        return str(tag)
    return str(fallback)


def _normalize_text_list(raw: Any, *, max_items: int | None = None) -> list[str]:
    if isinstance(raw, str):
        items = [part.strip() for part in raw.replace("|", ",").replace(";", ",").split(",") if part.strip()]
    elif isinstance(raw, (list, tuple)):
        items = [str(part).strip() for part in raw if str(part).strip()]
    else:
        items = []
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
        if max_items is not None and len(deduped) >= max_items:
            break
    return deduped


def _default_subgenres_for_genre(genre: str) -> list[str]:
    return list(DEFAULT_SUBGENRES_BY_GENRE.get(str(genre), [str(genre).lower().replace(" ", "_")]))


def _default_priority_motifs(genre: str, market: str, tier: str) -> list[str]:
    base = [
        str(market).lower().replace(" ", "_"),
        "industry_cycle",
    ]
    if tier in {"Epic", "A"}:
        base.append("scale_event")
    elif tier in {"Indie", "Micro"}:
        base.append("character_intimacy")
    else:
        base.append("ensemble_pressure")
    return base


def _default_trending_subgenres(genres: Sequence[str]) -> list[str]:
    out: list[str] = []
    for genre in genres:
        for value in DEFAULT_SUBGENRES_BY_GENRE.get(str(genre), []):
            if value not in out:
                out.append(value)
            if len(out) >= 4:
                return out
    return out or ["character_study", "institutional_thriller", "social_crossroads"]


def _default_priority_motifs_for_bucket(genres: Sequence[str], market: str, tier: str) -> list[str]:
    market_name = str(market).strip() or "Global"
    market_tokens = {
        "Global": ["platform_window", "global_exchange"],
        "North America": ["wide_release", "star_vehicle"],
        "Europe": ["festival_signal", "co_production"],
        "Asia": ["cross_market_momentum", "melodrama_current"],
        "Latin America": ["regional_breakout", "music_pulse"],
        "South America": ["regional_breakout", "social_current"],
        "Africa": ["breakout_voice", "festival_launch"],
        "Middle East": ["diaspora_pull", "prestige_breakout"],
        "Oceania": ["genre_export", "regional_buzz"],
    }.get(market_name, ["regional_breakout", "discovery_impulse"])
    base = [
        market_name.lower().replace(" ", "_"),
        market_tokens[0],
    ]
    genre_labels = [str(genre).lower().replace(" ", "_") for genre in genres[:2] if str(genre).strip()]
    if tier in {"Epic", "A"}:
        base.extend(["scale_event", "franchise_pressure", market_tokens[-1]])
    elif tier in {"Indie", "Micro"}:
        base.extend(["character_intimacy", "discovery_impulse", market_tokens[-1]])
    else:
        base.extend(["ensemble_pressure", "market_balance", market_tokens[-1]])
    for label in genre_labels:
        if label not in base:
            base.append(label)
        if len(base) >= 5:
            break
    return base[:5]


def _default_motif_drift(
    bucket_start_year: int,
    genre: str,
    *,
    range_start_year: int,
    range_end_year: int,
) -> list[str]:
    phase = _relative_year_phase(int(bucket_start_year), start_year=int(range_start_year), end_year=int(range_end_year))
    if phase == "early":
        return ["analogue_texture", "classical_storytelling", f"{str(genre).lower().replace(' ', '_')}_classical"]
    if phase == "mid":
        return ["media_shift", "global_exchange", f"{str(genre).lower().replace(' ', '_')}_globalization"]
    return ["platform_fragmentation", "franchise_acceleration", f"{str(genre).lower().replace(' ', '_')}_platform_era"]


def build_default_year_slate_plan(
    world_policy: Mapping[str, Any],
    *,
    genres: Sequence[str],
    tiers: Sequence[str],
) -> dict[str, Any]:
    policy = world_policy or {}
    year_buckets = list(policy.get("year_buckets", [])) if isinstance(policy, Mapping) else []
    if not year_buckets:
        return {"meta": {"schema_version": 1, "generator_mode": "fallback"}, "slates": []}
    country_market_map = policy.get("country_market_map", {}) if isinstance(policy, Mapping) else {}
    markets = sorted({str(value) for value in country_market_map.values() if str(value).strip()} | {"Global"})
    strategies = [
        str(row.get("strategy_tag"))
        for row in policy.get("company_strategies", [])
        if isinstance(row, Mapping) and row.get("strategy_tag")
    ]
    slates: list[dict[str, Any]] = []
    for bucket in year_buckets:
        if not isinstance(bucket, Mapping):
            continue
        bucket_id = str(bucket.get("bucket_id") or year_bucket_id(safe_int(bucket.get("start_year"), 1975)))
        start_year = safe_int(bucket.get("start_year"), 1975)
        end_year = safe_int(bucket.get("end_year"), start_year)
        genre_bias = normalize_counter_weights(bucket.get("genre_bias"), allowed=genres)
        top_genres = [genre for genre, _weight in sorted(genre_bias.items(), key=lambda item: item[1], reverse=True)[:3]]
        if not top_genres:
            top_genres = list(genres[:3])
        sequel_pressure = max(0.0, min(1.0, safe_float(bucket.get("sequel_pressure"), 0.25)))
        phase = _relative_year_phase(
            start_year,
            start_year=safe_int(policy.get("start_year"), start_year),
            end_year=safe_int(policy.get("end_year"), end_year),
        )
        for market in markets:
            for tier in tiers:
                tier_name = str(tier)
                market_name = str(market)
                release_season_bias = (
                    "summer" if tier_name in {"Epic", "A"} and market_name in {"Global", "North America", "Asia"} else
                    "festival" if market_name in {"Europe", "Africa"} else
                    "awards" if tier_name in {"Indie", "Micro"} and any(g in {"Drama", "Documentary", "Mystery"} for g in top_genres) else
                    DEFAULT_SEASON_BY_GENRE.get(top_genres[0], "fall")
                )
                strategy_bias = _stable_rotation(strategies, f"{bucket_id}|{market_name}|{tier_name}")[:3] if strategies else ["genre_lab"]
                release_pressure = 0.36 + (0.19 if tier_name in {"Epic", "A"} else 0.07)
                if market_name in {"Global", "North America", "Asia"}:
                    release_pressure += 0.05
                if phase == "late":
                    release_pressure += 0.04
                novelty_target = 0.26 + (0.22 if tier_name in {"Indie", "Micro"} else 0.06)
                if market_name in {"Africa", "Latin America", "South America", "Middle East", "Oceania"}:
                    novelty_target += 0.04
                if phase == "early":
                    novelty_target -= 0.03
                slates.append(
                    {
                        "slate_id": f"{bucket_id}|{market}|{tier_name}",
                        "bucket_id": bucket_id,
                        "start_year": start_year,
                        "end_year": end_year,
                        "market": market_name,
                        "tier": tier_name,
                        "trending_subgenres": _default_trending_subgenres(top_genres),
                        "priority_motifs": _default_priority_motifs_for_bucket(top_genres, market_name, tier_name),
                        "motif_drift": _default_motif_drift(
                            start_year,
                            top_genres[0],
                            range_start_year=safe_int(policy.get("start_year"), start_year),
                            range_end_year=safe_int(policy.get("end_year"), end_year),
                        ),
                        "release_pressure": round(min(1.0, release_pressure), 3),
                        "release_season_bias": release_season_bias,
                        "sequel_appetite": round(
                            min(
                                1.0,
                                sequel_pressure
                                + (0.12 if tier_name in {"Epic", "A"} else -0.03)
                                + (0.05 if market_name in {"Global", "North America"} else 0.0),
                            ),
                            3,
                        ),
                        "novelty_target": round(max(0.0, min(1.0, novelty_target)), 3),
                        "company_strategy_bias": strategy_bias,
                    }
                )
    return {
        "meta": {
            "schema_version": 1,
            "count": len(slates),
            "generator_mode": "fallback",
        },
        "slates": slates,
    }


def normalize_year_slate_plan(
    payload: Any,
    *,
    world_policy: Mapping[str, Any],
    genres: Sequence[str],
    tiers: Sequence[str],
) -> dict[str, Any]:
    fallback = build_default_year_slate_plan(world_policy, genres=genres, tiers=tiers)
    out = {"meta": dict(fallback.get("meta", {})), "slates": list(fallback.get("slates", []))}
    raw_slates = payload.get("slates") if isinstance(payload, Mapping) else None
    if not isinstance(raw_slates, list):
        return out
    slate_by_id = {str(row.get("slate_id")): dict(row) for row in out["slates"] if isinstance(row, Mapping)}
    for idx, raw in enumerate(raw_slates, start=1):
        if not isinstance(raw, Mapping):
            continue
        bucket_id = str(raw.get("bucket_id") or raw.get("year_bucket") or "")
        market = str(raw.get("market") or "Global")
        tier = str(raw.get("tier") or "Mid")
        slate_id = str(raw.get("slate_id") or f"{bucket_id}|{market}|{tier}" or f"slate_{idx:04d}")
        slate = dict(slate_by_id.get(slate_id, {}))
        if not slate:
            slate = {
                "slate_id": slate_id,
                "bucket_id": bucket_id,
                "start_year": safe_int(raw.get("start_year"), safe_int(world_policy.get("start_year"), 1975)),
                "end_year": safe_int(raw.get("end_year"), safe_int(world_policy.get("end_year"), 2025)),
                "market": market,
                "tier": tier,
            }
        slate["trending_subgenres"] = _normalize_text_list(raw.get("trending_subgenres"), max_items=5) or slate.get("trending_subgenres", [])
        slate["priority_motifs"] = _normalize_text_list(raw.get("priority_motifs"), max_items=6) or slate.get("priority_motifs", [])
        slate["motif_drift"] = _normalize_text_list(raw.get("motif_drift"), max_items=4) or slate.get("motif_drift", [])
        slate["release_pressure"] = max(0.0, min(1.0, safe_float(raw.get("release_pressure"), slate.get("release_pressure", 0.45))))
        slate["release_season_bias"] = str(raw.get("release_season_bias") or slate.get("release_season_bias") or "fall")
        slate["sequel_appetite"] = max(0.0, min(1.0, safe_float(raw.get("sequel_appetite"), slate.get("sequel_appetite", 0.3))))
        slate["novelty_target"] = max(0.0, min(1.0, safe_float(raw.get("novelty_target"), slate.get("novelty_target", 0.35))))
        slate["company_strategy_bias"] = _normalize_text_list(raw.get("company_strategy_bias"), max_items=4) or slate.get("company_strategy_bias", ["genre_lab"])
        slate_by_id[slate_id] = slate
    out["slates"] = sorted(
        slate_by_id.values(),
        key=lambda row: (
            safe_int(row.get("start_year"), safe_int(world_policy.get("start_year"), 1975)),
            str(row.get("market", "")),
            str(row.get("tier", "")),
        ),
    )
    out["meta"]["count"] = len(out["slates"])
    if isinstance(payload, Mapping) and isinstance(raw_slates, list) and raw_slates:
        out["meta"]["generator_mode"] = "llm"
    return out


def index_year_slate_plan(payload: Mapping[str, Any] | None) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    grouped: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for slate in (payload or {}).get("slates", []):
        if not isinstance(slate, Mapping):
            continue
        bucket_id = str(slate.get("bucket_id") or "")
        market = str(slate.get("market") or "Global")
        tier = str(slate.get("tier") or "Mid")
        grouped[bucket_id][market][tier] = dict(slate)
    return {bucket: {market: dict(tiers) for market, tiers in markets.items()} for bucket, markets in grouped.items()}


def resolve_year_slate(
    year_slate_index: Mapping[str, Any] | None,
    *,
    bucket_id: str,
    market: str,
    tier: str,
) -> dict[str, Any]:
    index = year_slate_index or {}
    bucket = index.get(str(bucket_id), {}) if isinstance(index, Mapping) else {}
    if isinstance(bucket, Mapping):
        for market_key in (str(market), "Global", "*"):
            market_rows = bucket.get(market_key)
            if not isinstance(market_rows, Mapping):
                continue
            for tier_key in (str(tier), "*"):
                slate = market_rows.get(tier_key)
                if isinstance(slate, Mapping):
                    return dict(slate)
    return {}


def _infer_keyword_motif_family(keyword: str, topic_genre: str) -> str:
    text = str(keyword or "").strip().lower()
    tokens = set(re.findall(r"[a-z0-9]+", text))
    if tokens & _FRANCHISE_TOKENS:
        return "franchise"
    if tokens & _SEQUEL_TOKENS or re.search(r"\b(?:ii|iii|iv|part|chapter)\b", text):
        return "sequel_drift"
    if tokens & _PROFESSION_KEYWORDS:
        return "profession"
    if tokens & _RELATIONSHIP_KEYWORDS:
        return "relationship"
    if tokens & _EVENT_KEYWORDS:
        return "event"
    if tokens & _OBJECT_KEYWORDS:
        return "object"
    if tokens & _PLACE_KEYWORDS:
        return "place"
    if tokens & _SETTING_KEYWORDS:
        return "setting"
    if tokens & _TONE_KEYWORDS:
        return "tone"
    if len(tokens) >= 3:
        return "subgenre"
    return "genre" if str(topic_genre or "").strip() else "setting"


def _infer_keyword_specificity(keyword: str, pop_weight: float) -> int:
    tokens = re.findall(r"[a-z0-9]+", str(keyword or "").lower())
    if not tokens:
        return 2
    if set(tokens) & _GENERIC_KEYWORD_TOKENS:
        return 1
    if len(tokens) >= 4:
        return 4
    if len(tokens) == 3:
        return 3
    if float(pop_weight) <= 0.02:
        return 4
    if float(pop_weight) <= 0.05:
        return 3
    return 2


def _infer_keyword_scope(motif_family: str, specificity_tier: int) -> str:
    if motif_family in {"franchise", "sequel_drift"}:
        return "franchise"
    if specificity_tier >= 4:
        return "concept_pack"
    if motif_family in {"tone", "relationship"}:
        return "year_slate"
    return "global"


def _default_keyword_motif_entry(row: Mapping[str, Any], idx: int) -> dict[str, Any]:
    keyword = str(row.get("keyword") or row.get("name") or f"keyword_{idx}")
    topic_genre = str(row.get("topic_genre") or "Drama")
    pop_weight = max(0.001, safe_float(row.get("pop_weight"), 0.02))
    motif_family = _infer_keyword_motif_family(keyword, topic_genre)
    specificity_tier = _infer_keyword_specificity(keyword, pop_weight)
    scope_hint = _infer_keyword_scope(motif_family, specificity_tier)
    cooccurrence_cluster = f"{topic_genre.lower().replace(' ', '_')}::{motif_family}"
    franchise_affinity = 0.85 if motif_family in {"franchise", "sequel_drift"} else (0.55 if "return" in keyword.lower() else 0.08)
    recurrence_strength = min(1.0, 0.15 + pop_weight * 3.5 + (0.20 if motif_family in {"genre", "relationship"} else 0.0))
    return {
        "motif_id": str(row.get("motif_id") or row.get("keyword_id") or f"motif_{idx:04d}"),
        "keyword_id": safe_int(row.get("keyword_id"), idx),
        "keyword": keyword,
        "topic_genre": topic_genre,
        "motif_family": motif_family,
        "specificity_tier": int(max(1, min(5, specificity_tier))),
        "scope_hint": scope_hint,
        "franchise_affinity": round(max(0.0, min(1.0, franchise_affinity)), 4),
        "cooccurrence_cluster": cooccurrence_cluster,
        "recurrence_strength": round(max(0.0, min(1.0, recurrence_strength)), 4),
    }


def build_default_keyword_motif_bank(
    keyword_rows: Sequence[Mapping[str, Any]] | None,
    *,
    genres: Sequence[str],
) -> dict[str, Any]:
    rows = list(keyword_rows or [])
    motifs = [_default_keyword_motif_entry(row, idx + 1) for idx, row in enumerate(rows)]
    clusters = Counter(str(row.get("cooccurrence_cluster", "")) for row in motifs if str(row.get("cooccurrence_cluster", "")).strip())
    return {
        "meta": {
            "schema_version": 1,
            "count": len(motifs),
            "generator_mode": "fallback",
        },
        "motifs": motifs,
        "clusters": [{"cluster_id": name, "size": int(size)} for name, size in clusters.most_common()],
        "genres": list(genres),
    }


def normalize_keyword_motif_bank(
    payload: Any,
    *,
    keyword_rows: Sequence[Mapping[str, Any]] | None,
    genres: Sequence[str],
) -> dict[str, Any]:
    fallback = build_default_keyword_motif_bank(keyword_rows, genres=genres)
    motif_by_keyword = {
        str(row.get("keyword", "")).strip().lower(): dict(row)
        for row in fallback.get("motifs", [])
        if isinstance(row, Mapping) and str(row.get("keyword", "")).strip()
    }
    raw_motifs = payload.get("motifs") if isinstance(payload, Mapping) else None
    if isinstance(raw_motifs, list):
        for idx, raw in enumerate(raw_motifs, start=1):
            if not isinstance(raw, Mapping):
                continue
            key = str(raw.get("keyword", "")).strip().lower()
            if not key:
                continue
            motif = dict(motif_by_keyword.get(key, _default_keyword_motif_entry(raw, idx)))
            motif["motif_id"] = str(raw.get("motif_id") or motif.get("motif_id") or f"motif_{idx:04d}")
            motif["keyword_id"] = safe_int(raw.get("keyword_id"), safe_int(motif.get("keyword_id"), idx))
            motif["keyword"] = str(raw.get("keyword") or motif.get("keyword") or "")
            motif["topic_genre"] = str(raw.get("topic_genre") or motif.get("topic_genre") or "Drama")
            motif["motif_family"] = str(raw.get("motif_family") or motif.get("motif_family") or _infer_keyword_motif_family(motif["keyword"], motif["topic_genre"]))
            motif["specificity_tier"] = int(max(1, min(5, safe_int(raw.get("specificity_tier"), safe_int(motif.get("specificity_tier"), 2)))))
            motif["scope_hint"] = str(raw.get("scope_hint") or motif.get("scope_hint") or _infer_keyword_scope(motif["motif_family"], motif["specificity_tier"]))
            motif["franchise_affinity"] = round(
                max(0.0, min(1.0, safe_float(raw.get("franchise_affinity"), safe_float(motif.get("franchise_affinity"), 0.0)))),
                4,
            )
            motif["cooccurrence_cluster"] = str(raw.get("cooccurrence_cluster") or motif.get("cooccurrence_cluster") or f"{motif['topic_genre'].lower()}::{motif['motif_family']}")
            motif["recurrence_strength"] = round(
                max(0.0, min(1.0, safe_float(raw.get("recurrence_strength"), safe_float(motif.get("recurrence_strength"), 0.25)))),
                4,
            )
            motif_by_keyword[key] = motif
    motifs = sorted(
        motif_by_keyword.values(),
        key=lambda row: (safe_int(row.get("keyword_id"), 0), str(row.get("keyword", ""))),
    )
    clusters = Counter(str(row.get("cooccurrence_cluster", "")) for row in motifs if str(row.get("cooccurrence_cluster", "")).strip())
    return {
        "meta": {
            "schema_version": 1,
            "count": len(motifs),
            "generator_mode": "llm" if isinstance(payload, Mapping) and isinstance(raw_motifs, list) and raw_motifs else "fallback",
        },
        "motifs": motifs,
        "clusters": [{"cluster_id": name, "size": int(size)} for name, size in clusters.most_common()],
        "genres": list(genres),
    }


def keyword_motif_updates(payload: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    updates: dict[str, dict[str, Any]] = {}
    for row in (payload or {}).get("motifs", []):
        if not isinstance(row, Mapping):
            continue
        key = str(row.get("keyword", "")).strip().lower()
        if not key:
            continue
        updates[key] = {
            "motif_family": str(row.get("motif_family") or ""),
            "specificity_tier": safe_int(row.get("specificity_tier"), 2),
            "scope_hint": str(row.get("scope_hint") or ""),
            "franchise_affinity": safe_float(row.get("franchise_affinity"), 0.0),
            "cooccurrence_cluster": str(row.get("cooccurrence_cluster") or ""),
            "recurrence_strength": safe_float(row.get("recurrence_strength"), 0.0),
        }
    return updates


def enrich_keyword_dataframe(df: Any, payload: Mapping[str, Any] | None) -> Any:
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return df
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or "keyword" not in df.columns:
        return df
    updates = keyword_motif_updates(payload)
    if not updates:
        return df
    out = df.copy()
    key_series = out["keyword"].fillna("").astype(str).str.strip().str.lower()
    defaults = {
        "motif_family": "",
        "specificity_tier": 2,
        "scope_hint": "",
        "franchise_affinity": 0.0,
        "cooccurrence_cluster": "",
        "recurrence_strength": 0.0,
    }
    for column, default_value in defaults.items():
        mapping = {key: value.get(column, default_value) for key, value in updates.items()}
        base_series = out[column] if column in out.columns else default_value
        if column in {"specificity_tier"}:
            merged = key_series.map(mapping)
            if column in out.columns:
                out[column] = merged.fillna(out[column]).fillna(default_value).astype(int)
            else:
                out[column] = merged.fillna(default_value).astype(int)
        elif column in {"franchise_affinity", "recurrence_strength"}:
            merged = key_series.map(mapping)
            if column in out.columns:
                out[column] = merged.fillna(out[column]).fillna(default_value).astype(float)
            else:
                out[column] = merged.fillna(default_value).astype(float)
        else:
            merged = key_series.map(mapping)
            if column in out.columns:
                out[column] = merged.fillna(out[column]).fillna(default_value).astype(str)
            else:
                out[column] = merged.fillna(default_value).astype(str)
    return out


_FRANCHISE_NAME_STOPWORDS = {
    "the", "a", "an", "and", "of", "for", "to", "in", "on", "at", "with", "from",
    "part", "chapter", "returns", "return", "legacy", "origins", "rise", "fall",
    "again", "story", "chronicles", "franchise", "movie", "film",
}

_FRANCHISE_SUFFIX_BY_GENRE = {
    "Action": "Protocol",
    "Animation": "Tales",
    "Comedy": "Club",
    "Crime": "Files",
    "Documentary": "Project",
    "Drama": "Legacy",
    "Fantasy": "Chronicle",
    "Horror": "Night",
    "Mystery": "Archive",
    "Romance": "Letters",
    "Sci-Fi": "Frontier",
    "Thriller": "Directive",
    "War": "Command",
}


def _looks_placeholder_franchise_name(name: str) -> bool:
    text = str(name or "").strip()
    if not text:
        return True
    return bool(re.fullmatch(r"franchise[_\s-]*\d+", text, flags=re.IGNORECASE))


def _derive_franchise_name(
    franchise: Mapping[str, Any],
    *,
    title_hints: Mapping[int, Sequence[Mapping[str, Any]]] | None = None,
) -> str:
    franchise_id = safe_int(franchise.get("franchise_id"), 0)
    genre = str(franchise.get("genre") or "Action")
    existing = str(franchise.get("name") or franchise.get("franchise_name") or "").strip()
    if existing and not _looks_placeholder_franchise_name(existing):
        return existing

    hints = list((title_hints or {}).get(franchise_id, []))
    token_counter: Counter[str] = Counter()
    for row in hints:
        title = str(row.get("title") or "")
        tagline = str(row.get("tagline") or "")
        for token in re.findall(r"[A-Za-z][A-Za-z0-9']*", title):
            clean = token.strip("'").lower()
            if len(clean) < 3 or clean in _FRANCHISE_NAME_STOPWORDS:
                continue
            token_counter[clean] += 2
        for token in re.findall(r"[A-Za-z][A-Za-z0-9']*", tagline):
            clean = token.strip("'").lower()
            if len(clean) < 4 or clean in _FRANCHISE_NAME_STOPWORDS:
                continue
            token_counter[clean] += 1

    lead_words = [token.title() for token, _count in token_counter.most_common(2)]
    suffix = _FRANCHISE_SUFFIX_BY_GENRE.get(genre, "Saga")
    if len(lead_words) >= 2:
        return " ".join(lead_words[:2])
    if len(lead_words) == 1:
        return lead_words[0] if lead_words[0].lower() == suffix.lower() else f"{lead_words[0]} {suffix}"
    genre_root = genre.replace("-", " ").replace("_", " ").title()
    return f"{genre_root} {suffix}"


def build_default_franchise_bibles(
    franchises: Sequence[Mapping[str, Any]] | None,
    *,
    title_hints: Mapping[int, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    rows = list(franchises or [])
    bibles: list[dict[str, Any]] = []
    for idx, franchise in enumerate(rows, start=1):
        genre = str(franchise.get("genre") or "Action")
        tier = str(franchise.get("tier") or "A")
        franchise_id = safe_int(franchise.get("franchise_id"), idx)
        name = _derive_franchise_name(franchise, title_hints=title_hints)
        bibles.append(
            {
                "franchise_id": franchise_id,
                "franchise_name": name,
                "genre": genre,
                "tier": tier,
                "installments": safe_int(franchise.get("n_movies"), 2),
                "continuity_anchors": [
                    DEFAULT_CONFLICT_BY_GENRE.get(genre, "recurring institutional pressure"),
                    DEFAULT_RELATIONSHIP_BY_GENRE.get(genre, "returning alliance"),
                    DEFAULT_TONE_BY_GENRE.get(genre, "grounded"),
                ],
                "recurring_motifs": _default_subgenres_for_genre(genre)[:2] + [f"{genre.lower().replace(' ', '_')}_legacy"],
                "keyword_families": [
                    genre.lower().replace(" ", "_"),
                    DEFAULT_STYLE_BY_GENRE.get(genre, "clean"),
                    "returning_character",
                    "installment",
                ],
                "title_style": DEFAULT_STYLE_BY_GENRE.get(genre, "clean"),
                "subtitle_tokens": ["Returns", "Legacy", "Part", "Origins"] if tier in {"Epic", "A"} else ["Aftermath", "Again", "Night Shift"],
                "release_season_bias": DEFAULT_SEASON_BY_GENRE.get(genre, "summer"),
                "company_strategy_tag": "event_franchise" if tier in {"Epic", "A"} else "genre_lab",
                "cast_chemistry_target": DEFAULT_CHEMISTRY_BY_TIER.get(tier, "balanced_ensemble"),
                "carryover_director_bias": 0.82,
                "carryover_cast_bias": 0.66,
            }
        )
    return {
        "meta": {
            "schema_version": 1,
            "count": len(bibles),
            "generator_mode": "fallback",
        },
        "bibles": bibles,
    }


def normalize_franchise_bibles(
    payload: Any,
    *,
    franchises: Sequence[Mapping[str, Any]] | None,
    title_hints: Mapping[int, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    fallback = build_default_franchise_bibles(franchises, title_hints=title_hints)
    franchise_lookup = {
        safe_int(row.get("franchise_id"), idx + 1): dict(row)
        for idx, row in enumerate(franchises or [])
        if isinstance(row, Mapping)
    }
    bible_by_id = {
        safe_int(row.get("franchise_id"), idx + 1): dict(row)
        for idx, row in enumerate(fallback.get("bibles", []))
        if isinstance(row, Mapping)
    }
    raw_bibles = payload.get("bibles") if isinstance(payload, Mapping) else None
    if isinstance(raw_bibles, list):
        for idx, raw in enumerate(raw_bibles, start=1):
            if not isinstance(raw, Mapping):
                continue
            franchise_id = safe_int(raw.get("franchise_id"), idx)
            bible = dict(bible_by_id.get(franchise_id, {}))
            if not bible:
                franchise = franchise_lookup.get(franchise_id, {})
                bible = {
                    "franchise_id": franchise_id,
                    "franchise_name": _derive_franchise_name(franchise or raw, title_hints=title_hints),
                    "genre": str(raw.get("genre") or "Action"),
                    "tier": str(raw.get("tier") or "A"),
                }
            raw_name = str(raw.get("franchise_name") or raw.get("name") or "").strip()
            if raw_name and not _looks_placeholder_franchise_name(raw_name):
                bible["franchise_name"] = raw_name
            elif _looks_placeholder_franchise_name(str(bible.get("franchise_name") or "")):
                bible["franchise_name"] = _derive_franchise_name(
                    franchise_lookup.get(franchise_id, raw),
                    title_hints=title_hints,
                )
            bible["installments"] = safe_int(raw.get("installments"), safe_int(bible.get("installments"), 2))
            bible["continuity_anchors"] = _normalize_text_list(raw.get("continuity_anchors"), max_items=5) or bible.get("continuity_anchors", [])
            bible["recurring_motifs"] = _normalize_text_list(raw.get("recurring_motifs"), max_items=6) or bible.get("recurring_motifs", [])
            bible["keyword_families"] = _normalize_text_list(raw.get("keyword_families"), max_items=8) or bible.get("keyword_families", [])
            bible["title_style"] = str(raw.get("title_style") or bible.get("title_style") or "clean")
            bible["subtitle_tokens"] = _normalize_text_list(raw.get("subtitle_tokens"), max_items=8) or bible.get("subtitle_tokens", [])
            bible["release_season_bias"] = str(raw.get("release_season_bias") or bible.get("release_season_bias") or "summer")
            bible["company_strategy_tag"] = str(raw.get("company_strategy_tag") or bible.get("company_strategy_tag") or "event_franchise")
            bible["cast_chemistry_target"] = str(raw.get("cast_chemistry_target") or bible.get("cast_chemistry_target") or "star_vehicle")
            bible["carryover_director_bias"] = max(0.0, min(1.0, safe_float(raw.get("carryover_director_bias"), safe_float(bible.get("carryover_director_bias"), 0.8))))
            bible["carryover_cast_bias"] = max(0.0, min(1.0, safe_float(raw.get("carryover_cast_bias"), safe_float(bible.get("carryover_cast_bias"), 0.65))))
            bible_by_id[franchise_id] = bible
    bibles = sorted(bible_by_id.values(), key=lambda row: safe_int(row.get("franchise_id"), 0))
    return {
        "meta": {
            "schema_version": 1,
            "count": len(bibles),
            "generator_mode": "llm" if isinstance(payload, Mapping) and isinstance(raw_bibles, list) and raw_bibles else "fallback",
        },
        "bibles": bibles,
    }


def index_franchise_bibles(payload: Mapping[str, Any] | None) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in (payload or {}).get("bibles", []):
        if not isinstance(row, Mapping):
            continue
        out[safe_int(row.get("franchise_id"), 0)] = dict(row)
    return out
