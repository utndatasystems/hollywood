"""
Mirage -- Contracts, Schemas, Vocabularies, Validators
===============================================================
This defines ALL controlled vocabularies, JSON schemas, and validation
functions. Every LLM output must pass through these validators before
entering the pipeline.

v11: Wider cast ranges, style normalization, hybrid graph edge types,
    inflation correlation, company power-law, 2000+ actor pool.
"""
import os
import re
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from model_defaults import model_tiers


# ═══════════════════════════════════════════════════════════════════════
# CONTROLLED VOCABULARIES
# ═══════════════════════════════════════════════════════════════════════

GENRES = [
    "Action", "Adventure", "Animation", "Biography", "Comedy", "Crime",
    "Documentary", "Drama", "Family", "Fantasy", "Film-Noir", "History",
    "Horror", "Music", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Sport", "Thriller", "War", "Western", "Superhero", "Martial Arts",
    "Disaster", "Experimental", "Short", "Reality-TV",
]

# Sub-genres loaded dynamically from genres.json if available
import json as _json
from pathlib import Path as _Path
_genres_path = _Path(__file__).parent / "entities" / "genres.json"
SUB_GENRES = []
SUB_GENRE_MAP = {}  # parent_genre -> [sub_genre, ...]
if _genres_path.exists():
    _taxonomy = _json.load(open(_genres_path, "r", encoding="utf-8"))
    SUB_GENRES = _taxonomy.get("sub_genres", [])
    for _sg in SUB_GENRES:
        SUB_GENRE_MAP.setdefault(_sg["parent_genre"], []).append(_sg["sub_genre"])

# B1-FIX: weights must sum to 1.0. Added Reality-TV=0.01, bumped Experimental to
# 0.01, reduced Drama slightly. Runtime assertion below catches future drift.
GENRE_WEIGHTS = {
    "Action": 0.095, "Adventure": 0.050, "Animation": 0.040, "Biography": 0.020,
    "Comedy": 0.095, "Crime": 0.050, "Documentary": 0.040, "Drama": 0.105,
    "Family": 0.030, "Fantasy": 0.040, "Film-Noir": 0.010, "History": 0.020,
    "Horror": 0.060, "Music": 0.010, "Musical": 0.010, "Mystery": 0.040,
    "Romance": 0.050, "Sci-Fi": 0.060, "Sport": 0.020, "Thriller": 0.060,
    "War": 0.020, "Western": 0.010, "Superhero": 0.030, "Martial Arts": 0.010,
    "Disaster": 0.010, "Experimental": 0.005, "Short": 0.005, "Reality-TV": 0.005,
}
assert abs(sum(GENRE_WEIGHTS.values()) - 1.0) < 0.001, (
    f"GENRE_WEIGHTS must sum to 1.0, got {sum(GENRE_WEIGHTS.values()):.4f}")

# B4-FIX: Full country list covering all 34 unique countries in V12 movie.csv
# plus all nationality home-countries from person.csv. No country merging.
COUNTRIES = [
    # Core markets (high weight)
    "USA", "UK", "France", "India", "South Korea", "Japan", "Germany",
    "Brazil", "Canada", "China", "Italy", "Spain", "Australia",
    "Mexico", "Nigeria", "Sweden", "Argentina", "Thailand", "Iran", "Poland",
    "Colombia", "Denmark", "Cuba", "Greece", "Philippines", "Ireland", "New Zealand",
    "Egypt", "Russia", "Turkey", "Ghana", "Pakistan", "Vietnam", "Indonesia",
    # V12 movie.csv - confirmed unique countries
    "Norway", "Finland", "Netherlands", "Belgium", "Switzerland", "Austria",
    "Portugal", "Czech Republic", "Hungary", "Romania", "Bulgaria", "Croatia",
    "Serbia", "Ukraine", "Hong Kong", "Taiwan", "Malaysia", "Singapore",
    "Bangladesh", "Sri Lanka", "Nepal", "Israel", "Lebanon", "Jordan", "Iraq",
    "Saudi Arabia", "UAE", "Kuwait", "Qatar", "Bahrain", "Oman",
    "Kazakhstan", "Azerbaijan", "Uzbekistan", "Georgia", "Armenia",
    "Chile", "Peru", "Venezuela", "Ecuador", "Bolivia", "Uruguay",
    "Dominican Republic", "Costa Rica", "Guatemala", "Honduras",
    "South Africa", "Kenya", "Morocco", "Tunisia", "Senegal",
    "Zimbabwe", "Tanzania", "Ethiopia", "Congo", "Benin", "Cape Verde",
    "Mongolia", "Iceland", "Syria", "Belarus", "Slovakia", "Slovenia",
    "Estonia", "Latvia", "Lithuania", "Luxembourg", "Macedonia",
    "Puerto Rico",
]

COUNTRY_WEIGHTS = {
    "USA": 0.200, "UK": 0.060, "France": 0.045, "India": 0.055,
    "South Korea": 0.045, "Japan": 0.045, "Germany": 0.035, "Brazil": 0.035,
    "Canada": 0.025, "China": 0.040, "Italy": 0.025, "Spain": 0.025,
    "Australia": 0.025, "Mexico": 0.025, "Nigeria": 0.018, "Sweden": 0.015,
    "Argentina": 0.018, "Thailand": 0.018, "Iran": 0.010, "Poland": 0.018,
    "Colombia": 0.010, "Denmark": 0.010, "Cuba": 0.008, "Greece": 0.008,
    "Philippines": 0.010, "Ireland": 0.010, "New Zealand": 0.008,
    "Egypt": 0.010, "Russia": 0.020, "Turkey": 0.012, "Ghana": 0.005,
    "Pakistan": 0.006, "Vietnam": 0.005, "Indonesia": 0.010,
    "Norway": 0.005, "Finland": 0.005, "Netherlands": 0.007, "Belgium": 0.005,
    "Switzerland": 0.005, "Austria": 0.005, "Portugal": 0.004, "Czech Republic": 0.004,
    "Hungary": 0.004, "Romania": 0.004, "Bulgaria": 0.003, "Croatia": 0.003,
    "Serbia": 0.003, "Ukraine": 0.005, "Hong Kong": 0.008, "Taiwan": 0.006,
    "Malaysia": 0.005, "Singapore": 0.005, "Bangladesh": 0.004, "Sri Lanka": 0.003,
    "Nepal": 0.002, "Israel": 0.005, "Lebanon": 0.003, "Jordan": 0.002,
    "Iraq": 0.002, "Saudi Arabia": 0.005, "UAE": 0.005, "Kazakhstan": 0.003,
    "Chile": 0.005, "Peru": 0.004, "Venezuela": 0.003, "Ecuador": 0.002,
    "Bolivia": 0.002, "Uruguay": 0.002, "South Africa": 0.006, "Kenya": 0.003,
    "Morocco": 0.004, "Tunisia": 0.003, "Senegal": 0.002, "Zimbabwe": 0.002,
    "Iceland": 0.002, "Mongolia": 0.002,
    # Remaining countries all get 0.001 (non-zero, fully represented)
}
# Fill any country not explicitly listed with minimal weight
for _c in COUNTRIES:
    if _c not in COUNTRY_WEIGHTS:
        COUNTRY_WEIGHTS[_c] = 0.001
# Normalize to exactly 1.0
_cw_total = sum(COUNTRY_WEIGHTS.values())
COUNTRY_WEIGHTS = {k: v / _cw_total for k, v in COUNTRY_WEIGHTS.items()}

COMPANY_COUNTRY_WEIGHTS = {country: 0.0005 for country in COUNTRIES}
COMPANY_COUNTRY_WEIGHTS.update({
    # IMDb company predicates are heavily centered on large production markets.
    # Keep a long tail, but make studio/company country much less uniform than
    # person nationality or movie origin.
    "USA": 0.480,
    "UK": 0.080,
    "India": 0.055,
    "Japan": 0.045,
    "France": 0.045,
    "Germany": 0.040,
    "Canada": 0.035,
    "South Korea": 0.030,
    "China": 0.030,
    "Italy": 0.022,
    "Spain": 0.020,
    "Australia": 0.020,
    "Brazil": 0.018,
    "Mexico": 0.018,
    "Russia": 0.012,
    "Netherlands": 0.010,
    "Hong Kong": 0.010,
    "Ireland": 0.008,
    "Sweden": 0.008,
    "Denmark": 0.006,
    "New Zealand": 0.006,
    "Belgium": 0.006,
    "Switzerland": 0.006,
    "Norway": 0.004,
    "Finland": 0.004,
    "Poland": 0.004,
    "Argentina": 0.004,
    "South Africa": 0.004,
})
_ccw_total = sum(COMPANY_COUNTRY_WEIGHTS.values())
COMPANY_COUNTRY_WEIGHTS = {k: v / _ccw_total for k, v in COMPANY_COUNTRY_WEIGHTS.items()}

COUNTRY_LANGUAGE = {
    "USA": "English", "UK": "English", "Australia": "English",
    "Canada": "English", "Nigeria": "English", "Ghana": "English",
    "Ireland": "English", "New Zealand": "English", "South Africa": "English",
    "Kenya": "English", "Zimbabwe": "English", "Singapore": "English",
    "Puerto Rico": "Spanish",
    "France": "French", "Belgium": "French",  # +Dutch, but French-majority in film
    "Switzerland": "French",  # French-majority Swiss cinema
    "India": "Hindi", "Bangladesh": "Bengali", "Nepal": "Nepali",
    "Sri Lanka": "Sinhala", "Pakistan": "Urdu",
    "South Korea": "Korean", "Korea": "Korean",
    "Japan": "Japanese",
    "Germany": "German", "Austria": "German", "Luxembourg": "German",
    "Brazil": "Portuguese", "Portugal": "Portuguese",
    "Italy": "Italian",
    "Spain": "Spanish", "Mexico": "Spanish", "Argentina": "Spanish",
    "Colombia": "Spanish", "Cuba": "Spanish", "Chile": "Spanish",
    "Peru": "Spanish", "Venezuela": "Spanish", "Ecuador": "Spanish",
    "Bolivia": "Spanish", "Uruguay": "Spanish", "Dominican Republic": "Spanish",
    "Costa Rica": "Spanish", "Guatemala": "Spanish", "Honduras": "Spanish",
    "El Salvador": "Spanish", "Panama": "Spanish",
    "Sweden": "Swedish", "Norway": "Norwegian", "Denmark": "Danish",
    "Finland": "Finnish", "Iceland": "Icelandic",
    "Netherlands": "Dutch",
    "China": "Mandarin", "Taiwan": "Mandarin", "Hong Kong": "Cantonese",
    "Thailand": "Thai", "Vietnam": "Vietnamese", "Indonesia": "Indonesian",
    "Philippines": "Filipino", "Malaysia": "Malay",
    "Iran": "Persian", "Iraq": "Arabic", "Jordan": "Arabic",
    "Egypt": "Arabic", "Lebanon": "Arabic", "Saudi Arabia": "Arabic",
    "UAE": "Arabic", "Kuwait": "Arabic", "Qatar": "Arabic",
    "Bahrain": "Arabic", "Oman": "Arabic", "Morocco": "Arabic",
    "Tunisia": "Arabic", "Senegal": "French", "Benin": "French",
    "Poland": "Polish", "Czech Republic": "Czech", "Slovakia": "Slovak",
    "Hungary": "Hungarian", "Romania": "Romanian", "Bulgaria": "Bulgarian",
    "Croatia": "Croatian", "Serbia": "Serbian", "Slovenia": "Slovenian",
    "Ukraine": "Ukrainian", "Belarus": "Belarusian", "Estonia": "Estonian",
    "Latvia": "Latvian", "Lithuania": "Lithuanian", "Macedonia": "Macedonian",
    "Greece": "Greek", "Turkey": "Turkish", "Israel": "Hebrew",
    "Russia": "Russian", "Kazakhstan": "Kazakh", "Azerbaijan": "Azerbaijani",
    "Uzbekistan": "Uzbek", "Armenia": "Armenian", "Georgia": "Georgian",
    "Mongolia": "Mongolian", "Syria": "Arabic",
    "Tanzania": "Swahili", "Ethiopia": "Amharic", "Congo": "French",
    "Cape Verde": "Portuguese",
}

# GPT fix #5: explicit NATIONALITIES (superset of countries + diaspora origins)
NATIONALITIES = [
    "American", "British", "French", "Indian", "South Korean", "Japanese",
    "German", "Brazilian", "Canadian", "Chinese", "Italian", "Spanish",
    "Australian", "Mexican", "Nigerian", "Swedish", "Argentine", "Thai",
    "Iranian", "Polish", "Russian", "Irish", "Scottish", "Dutch",
    "Norwegian", "Danish", "Greek", "Turkish", "Egyptian", "Jamaican",
    "Colombian", "Cuban", "Filipino", "Vietnamese", "Indonesian",
    "South African", "Kenyan", "New Zealander", "Israeli", "Lebanese",
    # 8b additions: nationalities from API-generated persons
    "Ghanaian", "Pakistani", "Ukrainian", "Bulgarian", "Portuguese",
    "Omani", "Saudi", "Belarusian", "Syrian",
    # V12 additions: all nationalities from LLM-generated persons
    "Argentinian", "Armenian", "Azerbaijani", "Bahraini", "Bangladeshi",
    "Belgian", "Beninese", "Bolivian", "British-Indian", "Cape Verdean",
    "Chilean", "Congolese", "Costa Rican", "Croatian", "Czech",
    "Dominican", "Ecuadorian", "Emirati", "Estonian", "Ethiopian",
    "Finnish", "Georgian", "Guatemalan", "Honduran", "Hong Konger",
    "Hungarian", "Icelandic", "Iraqi", "Jordanian", "Kazakh",
    "Kazakhstani", "Korean", "Kuwaiti", "Luxembourgish", "Macedonian",
    "Malaysian", "Mongolian", "Moroccan", "Nepali", "New Zealand",
    "Palestinian", "Peruvian", "Puerto Rican", "Qatari", "Romanian",
    "Saudi Arabian", "Senegalese", "Serbian", "South Sudanese",
    "Sri Lankan", "Swiss", "Tanzanian", "Tibetan", "Tunisian",
    "Uruguayan", "Uzbek", "Venezuelan", "Welsh", "Zimbabwean",
]

# GPT fix #5: explicit MARKETS
MARKETS = ["Local", "Regional", "Europe", "Asia", "North America", "Global",
           "South America", "Africa", "Latin America",
           "Middle East", "Oceania"]  # V12: expanded for LLM-generated entities

COMPANY_TIERS = ["Global", "Major", "Mid-Budget", "Indie", "Micro"]

# v11 P1: Hidden Confounder constants
N_AGENCIES = 8              # Talent agencies that cluster actors -> correlated casting
N_COMPANY_CLIQUES = 5       # Producer cliques with shared "house style" (budget/genre/rating)
AWARD_CAMPAIGN_GENRES = {"Drama", "War", "History", "Romance", "Biography"}  # Prestige genres for Q4 Oscar-bait

PRODUCTION_TIERS = ["Epic", "A", "Mid", "Indie", "Micro"]

TIER_WEIGHTS = {
    "Epic": 0.05, "A": 0.15, "Mid": 0.40, "Indie": 0.30, "Micro": 0.10
}

# Actor style tags
STYLE_TAGS = [
    "physical", "comedic", "cerebral", "stoic", "intense", "improvisational",
    "theatrical", "naturalistic", "chameleon", "deadpan", "explosive",
    "minimalist", "vulnerable", "menacing", "magnetic", "understated",
    "provocative", "acrobatic", "musical", "voice-artist"
]

# Director style tags (GPT fix #6: separate from actor styles)
DIRECTOR_STYLES = [
    "long-take", "handheld", "slow-burn", "visual-spectacle",
    "dialogue-driven", "non-linear", "documentary-style",
    "genre-bending", "atmospheric", "kinetic", "surrealist",
    "neo-noir", "ensemble-driven", "intimate", "epic-scale",
    "practical-effects", "improvised", "formalist", "lyrical"
]

CAREER_STAGES = ["rising", "prime", "veteran", "legend", "retired"]

# ─── Data-driven crew departments ─────────────────────────────────────
# Each entry: role_name -> {count: per-tier headcount, pool_fallback: attr name,
#                           genre_boost: list of genres that boost assignment probability}
# Adding a new crew role = add 1 dict entry here. No code changes needed.
CREW_DEPARTMENTS = {
    "writer":              {"count": {"Epic": 3, "A": 2, "Mid": 2, "Indie": 1, "Micro": 1},
                            "pool_fallback": "directors", "genre_boost": None},
    "cinematographer":     {"count": {"Epic": 1, "A": 1, "Mid": 1, "Indie": 1, "Micro": 1},
                            "pool_fallback": "directors", "genre_boost": None},
    "editor":              {"count": {"Epic": 1, "A": 1, "Mid": 1, "Indie": 1, "Micro": 1},
                            "pool_fallback": "directors", "genre_boost": None},
    "composer":            {"count": {"Epic": 1, "A": 1, "Mid": 1, "Indie": 1, "Micro": 1},
                            "pool_fallback": "directors", "genre_boost": None},
    "producer":            {"count": {"Epic": 3, "A": 2, "Mid": 2, "Indie": 1, "Micro": 1},
                            "pool_fallback": "persons",   "genre_boost": None},
    "production_designer": {"count": {"Epic": 1, "A": 1, "Mid": 1, "Indie": 0, "Micro": 0},
                            "pool_fallback": "persons",   "genre_boost": None},
    "costume_designer":    {"count": {"Epic": 1, "A": 1, "Mid": 0, "Indie": 0, "Micro": 0},
                            "pool_fallback": "persons",   "genre_boost": ["Drama", "Fantasy", "Romance"]},
    "casting_director":    {"count": {"Epic": 1, "A": 1, "Mid": 1, "Indie": 0, "Micro": 0},
                            "pool_fallback": "persons",   "genre_boost": None},
    "sound_designer":      {"count": {"Epic": 1, "A": 1, "Mid": 1, "Indie": 0, "Micro": 0},
                            "pool_fallback": "persons",   "genre_boost": ["Sci-Fi", "Horror", "Action"]},
    "vfx_supervisor":      {"count": {"Epic": 2, "A": 1, "Mid": 0, "Indie": 0, "Micro": 0},
                            "pool_fallback": "persons",   "genre_boost": ["Sci-Fi", "Fantasy", "Action", "Animation"]},
    "stunt_coordinator":   {"count": {"Epic": 1, "A": 1, "Mid": 0, "Indie": 0, "Micro": 0},
                            "pool_fallback": "persons",   "genre_boost": ["Action", "Thriller"]},
    "makeup_artist":       {"count": {"Epic": 1, "A": 1, "Mid": 0, "Indie": 0, "Micro": 0},
                            "pool_fallback": "persons",   "genre_boost": ["Horror", "Fantasy", "Sci-Fi"]},
}

# Auto-derived from departments (actor + director are always present)
ROLE_TYPES = ["actor", "director"] + list(CREW_DEPARTMENTS.keys())

ARCHETYPES = [
    "Lead Hero", "Lead Villain", "Love Interest", "Sidekick",
    "Mentor", "Comic Relief", "Authority Figure", "Henchman",
    "Mysterious Stranger", "Victim", "Supporting", "Extra"
]

CERTIFICATIONS = ["G", "PG", "PG-13", "R", "NR"]

# v11: expanded edge types for heterogeneous graph
EDGE_TYPES_PERSON_PERSON = ["friendship", "rivalry", "mentorship", "avoid", "clique", "former_collaborator", "chemistry"]
EDGE_TYPES_PERSON_COMPANY = ["employment", "blacklist", "exclusive_deal", "brand_fit"]
EDGE_TYPES_COMPANY_COMPANY = ["co_production", "subsidiary", "market_rival"]
EDGE_TYPES = EDGE_TYPES_PERSON_PERSON + EDGE_TYPES_PERSON_COMPANY + EDGE_TYPES_COMPANY_COMPANY
EDGE_SIGNS = ["+", "-"]
# D4-FIX: Added 'procedural_tv' for TV episode co-appearance edges (was missing,
# causing validate_edge() to reject them if re-ingested via standard pipeline).
EDGE_SOURCE_KINDS = [
    "llm", "latent_hybrid", "inferred_tag", "inferred_transitive",
    "procedural", "procedural_tv",
]

# Budget ranges per production tier (USD)
BUDGET_RANGES = {
    "Epic":  (150_000_000, 350_000_000),
    "A":     (40_000_000,  150_000_000),
    "Mid":   (10_000_000,   50_000_000),
    "Indie": (   500_000,   10_000_000),
    "Micro": (    20_000,      500_000),
}

# Cast size ranges per production tier (v11: wider spread for genre->cast correlation)
CAST_SIZE_RANGES = {
    "Epic":  (10, 20),
    "A":     (6, 12),
    "Mid":   (3, 7),
    "Indie": (2, 4),
    "Micro": (1, 2),
}

# Decade distribution targets (%) -- matches LLM-generated title bank
DECADE_WEIGHTS = {
    # Distribution from title_bank.csv: 311+497+782+1112+1586+863 = 5151
    1970: 0.060,   # 311/5151
    1980: 0.096,   # 497/5151
    1990: 0.152,   # 782/5151
    2000: 0.216,   # 1112/5151
    2010: 0.308,   # 1586/5151
    2020: 0.168,   # 863/5151
}

# B2-FIX: All 28 genres now have explicit CERT_DISTS entries.
# Previously 16 genres fell back silently to "Drama" -- e.g. Shorts and
# Reality-TV were getting 45% R ratings, which is clearly wrong.
CERT_DISTS = {
    "Horror":       {"G": 0.00, "PG": 0.00, "PG-13": 0.25, "R": 0.65, "NR": 0.10},
    "Comedy":       {"G": 0.05, "PG": 0.15, "PG-13": 0.50, "R": 0.20, "NR": 0.10},
    "Action":       {"G": 0.00, "PG": 0.10, "PG-13": 0.55, "R": 0.30, "NR": 0.05},
    "Drama":        {"G": 0.02, "PG": 0.08, "PG-13": 0.30, "R": 0.45, "NR": 0.15},
    "Documentary":  {"G": 0.10, "PG": 0.30, "PG-13": 0.25, "R": 0.10, "NR": 0.25},
    "Romance":      {"G": 0.05, "PG": 0.20, "PG-13": 0.45, "R": 0.20, "NR": 0.10},
    "Sci-Fi":       {"G": 0.00, "PG": 0.15, "PG-13": 0.55, "R": 0.25, "NR": 0.05},
    "Thriller":     {"G": 0.00, "PG": 0.05, "PG-13": 0.30, "R": 0.55, "NR": 0.10},
    "Fantasy":      {"G": 0.05, "PG": 0.25, "PG-13": 0.45, "R": 0.15, "NR": 0.10},
    "Mystery":      {"G": 0.00, "PG": 0.10, "PG-13": 0.35, "R": 0.45, "NR": 0.10},
    "Crime":        {"G": 0.00, "PG": 0.05, "PG-13": 0.25, "R": 0.60, "NR": 0.10},
    "Animation":    {"G": 0.20, "PG": 0.40, "PG-13": 0.25, "R": 0.05, "NR": 0.10},
    # Previously missing genres (all were silently using Drama distribution)
    "Adventure":    {"G": 0.05, "PG": 0.35, "PG-13": 0.45, "R": 0.10, "NR": 0.05},
    "War":          {"G": 0.00, "PG": 0.05, "PG-13": 0.20, "R": 0.70, "NR": 0.05},
    "Biography":    {"G": 0.02, "PG": 0.15, "PG-13": 0.35, "R": 0.40, "NR": 0.08},
    "History":      {"G": 0.05, "PG": 0.20, "PG-13": 0.35, "R": 0.35, "NR": 0.05},
    "Sport":        {"G": 0.05, "PG": 0.30, "PG-13": 0.45, "R": 0.15, "NR": 0.05},
    "Superhero":    {"G": 0.00, "PG": 0.10, "PG-13": 0.75, "R": 0.10, "NR": 0.05},
    "Martial Arts": {"G": 0.00, "PG": 0.10, "PG-13": 0.40, "R": 0.45, "NR": 0.05},
    "Disaster":     {"G": 0.00, "PG": 0.10, "PG-13": 0.60, "R": 0.25, "NR": 0.05},
    "Western":      {"G": 0.00, "PG": 0.05, "PG-13": 0.25, "R": 0.65, "NR": 0.05},
    "Music":        {"G": 0.10, "PG": 0.30, "PG-13": 0.35, "R": 0.15, "NR": 0.10},
    "Musical":      {"G": 0.15, "PG": 0.35, "PG-13": 0.35, "R": 0.10, "NR": 0.05},
    "Family":       {"G": 0.40, "PG": 0.45, "PG-13": 0.10, "R": 0.00, "NR": 0.05},
    "Film-Noir":    {"G": 0.00, "PG": 0.05, "PG-13": 0.20, "R": 0.60, "NR": 0.15},
    "Experimental": {"G": 0.05, "PG": 0.10, "PG-13": 0.20, "R": 0.25, "NR": 0.40},
    "Short":        {"G": 0.15, "PG": 0.25, "PG-13": 0.25, "R": 0.15, "NR": 0.20},
    "Reality-TV":   {"G": 0.20, "PG": 0.40, "PG-13": 0.25, "R": 0.05, "NR": 0.10},
}
# Runtime assertion: all 28 genres defined, all distributions sum to 1.0
assert set(CERT_DISTS.keys()) == set(GENRES), (
    f"CERT_DISTS missing genres: {set(GENRES) - set(CERT_DISTS.keys())}")
assert all(
    abs(sum(v.values()) - 1.0) < 0.001
    for v in CERT_DISTS.values()
), "Some CERT_DISTS entries do not sum to 1.0"

# Entity count targets
# V15: ALL persons are fully LLM-generated via generate_persons_llm.py.
# The old core/extras split (persons_core=453 + generate_extras.py procedural stubs)
# is DEAD -- no hollow extras exist anymore. is_extra flag is unused.
ENTITY_COUNTS = {
    "persons_total": 24000,   # V15 target: 4x scale via generate_persons_llm.py
    # Role targets at 24k (from ROLE_DISTRIBUTION in generate_persons_llm.py):
    # actor ~55% = 13200, director ~10% = 2400, writer ~7% = 1680,
    # producer ~8% = 1920, cinematographer ~4% = 960, editor ~3% = 720, composer ~3% = 720
    "companies": 500,
    "keywords": 900,
    "title_bank": 8000,
    "character_bank": 8000,
    "franchises_target_pct": (0.05, 0.09),
    "franchises_count": (50, 75),
    # V19: None = auto-detect from title_bank.csv (generates 1 movie per title).
    # Pass --n_movies to override.
    "movies": None,
    "tv_series": 150,
}

# v12: YEAR_RANGE = None because we use LLM-assigned years from title_bank.csv.
# Falls back to DECADE_WEIGHTS for any movie without a pre-assigned title.
YEAR_RANGE = None

# Relationship distribution targets
RELATIONSHIP_TARGETS = {
    # Keep calibration sparse. These are coverage / activation probabilities,
    # not literal per-director edge counts.
    "best_friend_rate": 0.12,
    "rival_rate": 0.04,
    "bf_same_community_rate": 0.65,
    "director_preferred_actors": 0.18,
    "director_avoided_actors": 0.03,
}


# ═══════════════════════════════════════════════════════════════════════
# JSON SCHEMAS for LLM output validation
# ═══════════════════════════════════════════════════════════════════════

PERSON_SCHEMA_REQUIRED = {
    "name": str,
    "nationality": str,
    "gender": str,
    "bio": str,
    "style_tags": list,
    "genre_affinity": list,
    "career_stage": str,
    "roles": list,
}
PERSON_SCHEMA_OPTIONAL = {
    "person_id": int,
    "market_fit": list,
    "avoid_genres": list,
}

COMPANY_SCHEMA_REQUIRED = {
    "name": str,
    "country": str,
    "description": str,
    "specialty_genres": list,
    "tier": str,
    "preferred_actor_styles": list,   # GPT fix #6: separate from director styles
    "preferred_director_styles": list, # GPT fix #6
}
COMPANY_SCHEMA_OPTIONAL = {
    "company_id": int,
    "founded_year": int,
    "defunct_year": (int, type(None)),
    "avoid_actor_styles": list,
    "avoid_director_styles": list,
}

EDGE_SCHEMA_REQUIRED = {
    "src": str,
    "dst": str,
    "edge_type": str,
    "weight": (int, float),
    "reason": str,
}
EDGE_SCHEMA_OPTIONAL = {
    "sign": str,
    "valid_from": int,    # GPT extra A: temporal validity
    "valid_to": int,      # GPT extra A
    "source_kind": str,   # GPT extra B: provenance
}


# ═══════════════════════════════════════════════════════════════════════
# STYLE TAG NORMALIZATION (v9 Fix #6: fuzzy-match OOV tokens)
# ═══════════════════════════════════════════════════════════════════════

# Manual overrides for common LLM drift patterns.
#
# NOTE: values are ordered *candidates*. The normalizer will pick the first
# candidate that exists in the provided vocabulary.
#
# This lets the same drift token map differently for actor-style vs director-style
# vocabularies (e.g. "gritty" -> "intense" for actors, but "neo-noir" for directors).
_STYLE_TAG_ALIASES: dict[str, tuple[str, ...]] = {
    # ─── Cross-cutting / common drift ───────────────────────────────
    "brooding": ("stoic", "atmospheric"),
    "dark": ("menacing", "neo-noir", "atmospheric"),
    "gritty": ("intense", "neo-noir", "handheld"),
    "emotional": ("vulnerable", "intimate", "lyrical"),
    "powerful": ("explosive", "epic-scale"),
    "subtle": ("understated", "formalist"),
    "dramatic": ("theatrical",),
    "quiet": ("minimalist", "slow-burn"),
    "dynamic": ("kinetic", "explosive"),
    "charming": ("magnetic",),
    "witty": ("comedic",),
    "raw": ("naturalistic", "handheld"),
    "bold": ("provocative", "genre-bending"),
    "agile": ("acrobatic",),
    "soulful": ("vulnerable", "lyrical"),
    "fierce": ("intense",),
    "gentle": ("understated",),
    "edgy": ("provocative", "neo-noir"),
    "versatile": ("chameleon",),

    # ─── Actor-style specific drift ─────────────────────────────────
    "physical-comedy": ("physical", "comedic"),
    "charismatic": ("magnetic",),
    "intellectual": ("cerebral",),
    "introspective": ("minimalist", "understated", "stoic"),
    "reserved": ("stoic", "understated"),
    "subdued": ("understated", "minimalist"),
    "nuanced": ("understated", "minimalist"),
    "energetic": ("explosive", "magnetic"),
    "jovial": ("comedic",),
    "earnest": ("vulnerable", "understated"),
    "relatable": ("naturalistic",),
    "tough": ("stoic", "menacing"),
    "resilient": ("stoic",),
    "athletic": ("acrobatic", "physical"),
    "heroic": ("physical", "magnetic", "stoic"),
    "action-star": ("physical", "explosive", "acrobatic"),
    "passionate": ("intense", "vulnerable"),
    "expressive": ("theatrical", "provocative", "magnetic"),
    "elegant": ("understated", "magnetic"),

    # ─── Director-style specific drift ──────────────────────────────
    "ensemble-focused": ("ensemble-driven",),
    "documentary-realist": ("documentary-style",),
    "social-realist": ("documentary-style",),
    "kinetic-montage": ("kinetic",),
    "action-oriented": ("kinetic",),
    "cgi-heavy": ("visual-spectacle",),
    "magical-realist": ("surrealist", "lyrical"),
    "poetic": ("lyrical",),
    "operatic": ("epic-scale", "lyrical"),
    "minimalist-frame": ("formalist",),
    # "minimalist" is a valid ACTOR tag but not a director tag; map to "formalist" when needed.
    "minimalist": ("minimalist", "formalist"),

    "character-focused": ("intimate", "dialogue-driven"),
    "character-driven": ("intimate", "dialogue-driven"),

    "visually-stylized": ("visual-spectacle", "formalist"),
    "visually-inventive": ("visual-spectacle", "formalist"),
    "visually-rich": ("visual-spectacle",),
    "visually-dynamic": ("visual-spectacle", "kinetic"),
    "visually-striking": ("visual-spectacle",),

    "visionary": ("visual-spectacle", "surrealist"),
    "conceptual": ("formalist", "non-linear"),
    "experimental": ("genre-bending", "non-linear", "surrealist", "formalist"),

    "fast-paced": ("kinetic",),
    "paced": ("kinetic",),
    "tense": ("slow-burn", "atmospheric"),
    "suspenseful": ("slow-burn", "neo-noir", "atmospheric"),
    "psychological": ("slow-burn", "neo-noir"),

    "realistic": ("documentary-style", "handheld"),
    "observational": ("documentary-style", "handheld"),

    "epic": ("epic-scale", "visual-spectacle"),
    "lighthearted": ("comedic", "lyrical"),
    "whimsical": ("lyrical", "surrealist"),
    "romantic": ("lyrical", "intimate"),
}

def normalize_style_tag(tag: str, vocab: list[str] = None) -> str | None:
    """Normalize a style tag to a valid vocab token.

    Steps:
      1) lowercase + strip
      2) normalize separators (spaces/underscores -> '-')
      3) exact match in vocab
      4) alias mapping (vocab-aware: first candidate in vocab wins)
      5) fuzzy match via Levenshtein distance <= 3
      6) return None if no match
    """
    if vocab is None:
        vocab = STYLE_TAGS

    if tag is None:
        return None

    # Normalize token shape
    tag = str(tag).strip().lower()
    tag = re.sub(r"[\s_]+", "-", tag)
    tag = re.sub(r"-+", "-", tag)

    # Exact match
    if tag in vocab:
        return tag

    # Alias mapping (choose first candidate that exists in vocab)
    candidates = _STYLE_TAG_ALIASES.get(tag)
    if candidates:
        for cand in candidates:
            if cand in vocab:
                return cand

    # B3-FIX: Use canonical levenshtein() (defined later in this module),
    # not the now-removed duplicate _lev().
    best, best_dist = None, 999
    for v in vocab:
        d = levenshtein(tag, v)
        if d < best_dist:
            best, best_dist = v, d
    if best_dist <= 3:
        return best

    return None  # drop invalid tag


# B3-FIX: _lev() removed -- was a duplicate of levenshtein() defined below.
# normalize_style_tag() now calls levenshtein() directly.


def normalize_company_styles(company: dict) -> dict:
    """Normalize all style fields in a company record, dropping invalid tags."""
    for field in ("preferred_actor_styles", "avoid_actor_styles"):
        if field in company and isinstance(company[field], list):
            company[field] = [t for t in
                (normalize_style_tag(s, STYLE_TAGS) for s in company[field])
                if t is not None]
    for field in ("preferred_director_styles", "avoid_director_styles"):
        if field in company and isinstance(company[field], list):
            company[field] = [t for t in
                (normalize_style_tag(s, DIRECTOR_STYLES) for s in company[field])
                if t is not None]
    return company


# ═══════════════════════════════════════════════════════════════════════
# NAME NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════

def normalize_name(name: str) -> str:
    """Normalize a name: strip, normalize quotes/hyphens/accents lightly."""
    name = name.strip()
    # Normalize fancy quotes to standard
    name = name.replace("\u2018", "'").replace("\u2019", "'")
    name = name.replace("\u201c", '"').replace("\u201d", '"')
    # Collapse multiple spaces
    name = re.sub(r'\s+', ' ', name)
    return name


def levenshtein(s1: str, s2: str) -> int:
    """Simple Levenshtein distance for fuzzy duplicate detection."""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            ins = prev_row[j + 1] + 1
            rem = curr_row[j] + 1
            sub = prev_row[j] + (c1 != c2)
            curr_row.append(min(ins, rem, sub))
        prev_row = curr_row
    return prev_row[-1]


def find_near_duplicates(names: list[str], threshold: int = 3) -> list[tuple[str,str,int]]:
    """Find pairs of names that are suspiciously similar.

    Uses rapidfuzz for C-optimized edit distance (~100x faster than pure Python).
    Falls back to O(n²) brute-force if rapidfuzz is not installed.
    """
    normalized = [normalize_name(n).lower() for n in names]

    try:
        from rapidfuzz.distance import Levenshtein
        from rapidfuzz import process, fuzz

        dupes = []
        seen = set()
        for i, query in enumerate(normalized):
            # Extract matches with score_cutoff based on threshold
            # rapidfuzz ratio is 0-100; we need edit distance <= threshold
            # Use Levenshtein.distance directly with score_cutoff
            results = process.extract(
                query, normalized[i+1:],
                scorer=Levenshtein.distance,
                score_cutoff=threshold,
                limit=None,  # return all matches
            )
            for match_str, dist, j_offset in results:
                j = i + 1 + j_offset
                pair = (min(i, j), max(i, j))
                if pair not in seen and dist > 0:
                    seen.add(pair)
                    dupes.append((names[i], names[j], int(dist)))
        return dupes

    except ImportError:
        # Fallback: O(n²) brute-force
        dupes = []
        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
                dist = levenshtein(normalized[i], normalized[j])
                if dist <= threshold and dist > 0:
                    dupes.append((names[i], names[j], dist))
        return dupes


# ═══════════════════════════════════════════════════════════════════════
# VALIDATORS
# ═══════════════════════════════════════════════════════════════════════

def validate_record(record: dict, required: dict, optional: dict,
                    vocab_checks: dict = None, label: str = "") -> list[str]:
    """Validate a single JSON record against a schema.

    Returns list of error strings (empty = valid).
    """
    errors = []
    prefix = f"[{label}] " if label else ""

    # Check required fields
    for field, ftype in required.items():
        if field not in record:
            errors.append(f"{prefix}Missing required field: '{field}'")
        elif isinstance(ftype, tuple):
            if not isinstance(record[field], ftype):
                errors.append(f"{prefix}'{field}' must be {ftype}, got {type(record[field])}")
        elif not isinstance(record[field], ftype):
            errors.append(f"{prefix}'{field}' must be {ftype.__name__}, got {type(record[field]).__name__}")

    # Check optional fields if present
    for field, ftype in optional.items():
        if field in record:
            if isinstance(ftype, tuple):
                if not isinstance(record[field], ftype):
                    errors.append(f"{prefix}Optional '{field}' must be {ftype}, got {type(record[field])}")
            elif not isinstance(record[field], ftype):
                errors.append(f"{prefix}Optional '{field}' must be {ftype.__name__}, got {type(record[field]).__name__}")

    # Vocab checks: {field: allowed_values_set}
    if vocab_checks:
        for field, allowed in vocab_checks.items():
            val = record.get(field)
            if val is None:
                continue
            if isinstance(val, list):
                for v in val:
                    if v not in allowed:
                        errors.append(f"{prefix}'{field}' value '{v}' not in vocabulary")
            elif val not in allowed:
                errors.append(f"{prefix}'{field}' value '{val}' not in vocabulary")

    return errors


def validate_person(record: dict) -> list[str]:
    """Validate a person record."""
    return validate_record(
        record, PERSON_SCHEMA_REQUIRED, PERSON_SCHEMA_OPTIONAL,
        vocab_checks={
            "nationality": set(NATIONALITIES),
            "gender": {"M", "F", "NB"},
            "style_tags": set(STYLE_TAGS) | set(DIRECTOR_STYLES),
            "genre_affinity": set(GENRES),
            "career_stage": set(CAREER_STAGES),
            "roles": set(ROLE_TYPES),
            "market_fit": set(MARKETS),
        },
        label=record.get("name", "?")
    )


def validate_company(record: dict) -> list[str]:
    """Validate a company record."""
    return validate_record(
        record, COMPANY_SCHEMA_REQUIRED, COMPANY_SCHEMA_OPTIONAL,
        vocab_checks={
            "country": set(COUNTRIES),
            "specialty_genres": set(GENRES),
            "tier": set(COMPANY_TIERS),
            "preferred_actor_styles": set(STYLE_TAGS),
            "preferred_director_styles": set(DIRECTOR_STYLES),
            "avoid_actor_styles": set(STYLE_TAGS),
            "avoid_director_styles": set(DIRECTOR_STYLES),
        },
        label=record.get("name", "?")
    )


def validate_edge(record: dict) -> list[str]:
    """Validate a graph edge record."""
    return validate_record(
        record, EDGE_SCHEMA_REQUIRED, EDGE_SCHEMA_OPTIONAL,
        vocab_checks={
            "edge_type": set(EDGE_TYPES),
            "sign": set(EDGE_SIGNS),
            "source_kind": set(EDGE_SOURCE_KINDS),
        },
        label=f"{record.get('src','?')}->{record.get('dst','?')}"
    )


def validate_batch(records: list[dict], validator_fn, label: str = "batch") -> dict:
    """Validate a batch of records, return summary.

    Returns: {'valid': int, 'invalid': int, 'errors': list, 'clean': list[dict]}
    """
    valid, invalid = 0, 0
    all_errors = []
    clean = []
    for i, rec in enumerate(records):
        errs = validator_fn(rec)
        if errs:
            invalid += 1
            all_errors.extend([f"  Record {i}: {e}" for e in errs])
        else:
            valid += 1
            # Normalize name if present
            if "name" in rec:
                rec["name"] = normalize_name(rec["name"])
            clean.append(rec)

    print(f"[{label}] Validated {len(records)} records: {valid} valid, {invalid} invalid")
    if all_errors:
        print(f"  First 10 errors:")
        for e in all_errors[:10]:
            print(f"    {e}")

    return {"valid": valid, "invalid": invalid, "errors": all_errors, "clean": clean}


# ═══════════════════════════════════════════════════════════════════════
# COMPOSITIONAL TITLE GENERATOR (GPT fix #2: unbounded titles)
# ═══════════════════════════════════════════════════════════════════════

TITLE_PREFIXES = [
    "The", "A", "Last", "First", "Final", "Dark", "Silent", "Lost",
    "Broken", "Eternal", "Crimson", "Midnight", "Golden", "Savage",
    "Bitter", "Hollow", "Iron", "Burning", "Frozen", "Wicked",
    # V18-SCALE: expanded for 200K+ unique titles
    "Hidden", "Sacred", "Fallen", "Distant", "Shattered", "True",
    "Deep", "Wild", "Bright", "Fading", "Ancient", "Lone",
    "Twisted", "Pale", "Rising", "Secret", "Blind", "Scarlet",
    "Copper", "Phantom", "Neon", "Velvet", "Glass", "Stone",
    "Silver", "Black", "White", "Red", "Dead", "New",
]

TITLE_NOUNS = [
    "Shadow", "Crown", "Edge", "Storm", "Blade", "Throne", "Mirror",
    "Flame", "Dream", "Ghost", "Code", "Signal", "Wolf", "Falcon",
    "Serpent", "Phoenix", "Raven", "Compass", "Cipher", "Horizon",
    "Veil", "Accord", "Witness", "Oracle", "Requiem", "Protocol",
    "Descent", "Ember", "Fracture", "Paradox", "Remedy", "Verdict",
    # V18-SCALE: expanded
    "Haven", "Summit", "Passage", "Crossing", "Fortune", "Silence",
    "Harbor", "Bridge", "Circuit", "Nexus", "Archive", "Catalyst",
    "Legion", "Dominion", "Frontier", "Gambit", "Impulse", "Junction",
    "Labyrinth", "Meridian", "Overture", "Quarantine", "Reckoning",
    "Sanctuary", "Threshold", "Vendetta", "Zenith", "Atlas",
    "Covenant", "Directive", "Echo", "Forge", "Grid",
]

TITLE_MODIFIERS = [
    "of Shadows", "in Red", "Beyond", "Within", "Unchained", "Rising",
    "Reborn", "Unleashed", "Below Zero", "After Dark", "at Dawn",
    "Without Mercy", "in Exile", "on Fire", "Under Glass", "Unbroken",
    # V18-SCALE: expanded
    "of the Fallen", "at Midnight", "in Black", "Above All", "Undone",
    "of Glass", "in Chains", "Untold", "Divided", "in Flames",
    "Unbound", "Forgotten", "Reclaimed", "of Iron", "Between Worlds",
    "After Midnight", "Unseen", "of Gold", "in Silence", "Awakened",
    "from Below", "Without End", "of Blood", "Resurrected",
]

TITLE_PATTERNS = [
    "{prefix} {noun}",
    "{prefix} {noun} {modifier}",
    "{noun} {modifier}",
    "{noun}",
    "{prefix} {noun}: {noun2}",
    # V18-SCALE: additional patterns for more diversity
    "{noun}: {prefix} {noun2}",
    "{prefix} {noun} of {noun2}",
    "{noun} & {noun2}",
]

def generate_compositional_title(rng, used: set) -> str:
    """Generate a unique title from word banks. Fallback from curated title bank."""
    for _ in range(500):  # V18-SCALE: raised from 100
        pattern = rng.choice(TITLE_PATTERNS)
        title = pattern.format(
            prefix=rng.choice(TITLE_PREFIXES),
            noun=rng.choice(TITLE_NOUNS),
            noun2=rng.choice(TITLE_NOUNS),
            modifier=rng.choice(TITLE_MODIFIERS),
        )
        if title not in used:
            return title
    # Digit-suffix fallback before absolute fallback
    base = rng.choice(TITLE_NOUNS)
    for d in range(1000):
        t = f"{base} {len(used) + d}"
        if t not in used:
            return t
    return f"Untitled Film #{len(used) + 1}"


# ═══════════════════════════════════════════════════════════════════════
# PROVISIONAL ANCHOR SCORING (GPT fix #4: anchors before pop_weight)
# ═══════════════════════════════════════════════════════════════════════

def compute_provisional_anchor_score(person: dict) -> float:
    """Score a person for anchor selection before pop_weight assignment.

    Based on: #roles, genre breadth, career_stage, bio richness.
    """
    score = 0.0
    # Role breadth
    roles = person.get("roles", [])
    score += len(roles) * 1.5  # actor-directors score higher
    # Genre breadth
    genres = person.get("genre_affinity", [])
    score += min(len(genres), 4) * 1.0
    # Career stage bonus
    stage_bonus = {"legend": 4, "veteran": 3, "prime": 2, "rising": 1, "retired": 0}
    score += stage_bonus.get(person.get("career_stage", "rising"), 0)
    # Bio richness (rough proxy: character count)
    bio = person.get("bio", "")
    score += min(len(bio) / 50, 3.0)
    # Style tag breadth
    score += min(len(person.get("style_tags", [])), 3) * 0.5
    # Market breadth
    market = person.get("market_fit", [])
    if "Global" in market:
        score += 2.0
    elif len(market) > 1:
        score += 1.0
    return score


def select_anchors(persons: list[dict], n: int = 15) -> list[dict]:
    """Select top-n anchor persons by provisional score."""
    scored = [(compute_provisional_anchor_score(p), p) for p in persons]
    scored.sort(key=lambda x: -x[0])
    return [p for _, p in scored[:n]]


# ═══════════════════════════════════════════════════════════════════════
# FRANCHISE MATH (GPT fix #3: consistent targets)
# ═══════════════════════════════════════════════════════════════════════

FRANCHISE_CONFIG = {
    "count_range": (50, 75),           # number of franchises (scales in _setup_franchises)
    "movies_per_franchise": (2, 10),   # V17: widened from (2,6) -- allows longer chains like MCU/Star Wars
    "target_pct_of_total": (0.05, 0.09),  # 5-9% of all movies are franchise entries
    "cast_retention_rate": 0.70,       # 70% of cast carries over per sequel
    "director_retention_rate": 0.80,   # 80% director retention
    "budget_growth_per_sequel": 1.20,  # 20% budget increase per installment
    "rating_decay_per_sequel": -0.30,  # -0.3 rating per installment
}


# ═══════════════════════════════════════════════════════════════════════
# SNAPSHOT / PROVENANCE
# ═══════════════════════════════════════════════════════════════════════

SNAPSHOT_CONFIG = {
    "snapshot_id": "mirage_v1",
    "generator_version": "2.3.0",
    "seed": 42,
    "write_yearly_snapshots": True,  # V17: ON by default -- never lose temporal data
}

# Tiered model strategy -- different models per pipeline step. Defaults and
# environment overrides live in model_defaults.py so public runs can switch
# model families without editing generator code.
MODEL_TIERS = model_tiers()


# ═══════════════════════════════════════════════════════════════════════
# UTILITY: load/save JSON batches
# ═══════════════════════════════════════════════════════════════════════

def load_json_batch(filepath: str | Path) -> list[dict]:
    """Load a JSON file containing a list of records."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Try to find the list inside common wrapper keys
        for key in ['persons', 'companies', 'edges', 'titles', 'keywords',
                     'characters', 'data', 'records', 'items']:
            if key in data and isinstance(data[key], list):
                return data[key]
    raise ValueError(f"Cannot find list of records in {filepath}")


def save_json_batch(records: list[dict], filepath: str | Path):
    """Save a list of records as JSON with atomic replacement + retry.

    Large incremental writes on Windows can sporadically fail with transient
    OSError variants while reopening the destination file. Write to a sibling
    temp path first, then replace the target with a small retry budget.
    """
    target = Path(filepath)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    last_error: OSError | None = None

    for attempt in range(5):
        try:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, target)
            print(f"Saved {len(records)} records to {target}")
            return
        except OSError as exc:
            last_error = exc
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            if attempt == 4:
                raise
            time.sleep(0.5 * (attempt + 1))

    if last_error is not None:
        raise last_error


# ═══════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Contracts self-test ===")
    print(f"Genres: {len(GENRES)}")
    print(f"Countries: {len(COUNTRIES)}")
    print(f"Nationalities: {len(NATIONALITIES)}")
    print(f"Style tags: {len(STYLE_TAGS)}")
    print(f"Director styles: {len(DIRECTOR_STYLES)}")
    print(f"Markets: {len(MARKETS)}")

    # Validate weights sum to 1
    assert abs(sum(GENRE_WEIGHTS.values()) - 1.0) < 0.01, f"Genre weights sum: {sum(GENRE_WEIGHTS.values())}"
    assert abs(sum(COUNTRY_WEIGHTS.values()) - 1.0) < 0.01, f"Country weights sum: {sum(COUNTRY_WEIGHTS.values())}"
    assert abs(sum(TIER_WEIGHTS.values()) - 1.0) < 0.01, f"Tier weights: {sum(TIER_WEIGHTS.values())}"
    assert abs(sum(DECADE_WEIGHTS.values()) - 1.0) < 0.01, f"Decade weights: {sum(DECADE_WEIGHTS.values())}"
    print("All weight sums validated.")

    # Test person validation
    good_person = {
        "name": "Nikolai Volkov",
        "nationality": "Russian",
        "gender": "M",
        "bio": "Former ballet dancer turned method actor.",
        "style_tags": ["physical", "intense"],
        "genre_affinity": ["Thriller", "Drama"],
        "career_stage": "prime",
        "roles": ["actor"],
    }
    errs = validate_person(good_person)
    assert len(errs) == 0, f"Good person should validate: {errs}"

    bad_person = {"name": "Test"}
    errs = validate_person(bad_person)
    assert len(errs) > 0, "Bad person should fail"
    print(f"Person validation: good=OK, bad={len(errs)} errors")

    # Test near-duplicate detection
    dupes = find_near_duplicates(["Celine Moreau", "Céline Moreau", "Bob Smith"])
    print(f"Near duplicates found: {dupes}")

    # Test title generation
    import random
    rng = random.Random(42)
    used = set()
    titles = [generate_compositional_title(rng, used) for _ in range(20)]
    for t in titles:
        used.add(t)
    print(f"Generated {len(titles)} unique titles: {titles[:5]}...")

    # Test anchor scoring
    score = compute_provisional_anchor_score(good_person)
    print(f"Anchor score for Nikolai: {score:.1f}")

    print("\n=== All self-tests passed ===")
