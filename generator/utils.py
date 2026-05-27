"""
V13 Pipeline -- utils.py
========================
Shared utility functions and constants for movie assembly.
Extracted from generate_movies.py for modular architecture.
"""
import numpy as np
import hashlib

from contracts import GENRES


# ═══════════════════════════════════════════════════════════════════════
# WEIGHT HELPERS
# ═══════════════════════════════════════════════════════════════════════

def normalize_weights(w):
    """Normalize weights to sum to 1, handling zeros and NaN."""
    w = np.asarray(w, dtype=float)
    w = np.where(np.isfinite(w), w, 0.0)  # replace NaN/Inf with 0
    w = np.maximum(w, 0.0)  # clamp negatives
    s = w.sum()
    if s <= 0:
        return np.ones(len(w)) / len(w)
    return w / s


def _safe_float(x, default=0.0):
    try:
        v = float(x)
    except Exception:
        return float(default)
    if v != v:  # NaN
        return float(default)
    return v


# C1-FIX: consolidated from duplicate definitions in financials.py and world_state.py
def _clip01(value, default=0.5):
    """Clip value to [0, 1] with NaN/error safety."""
    try:
        x = float(value)
    except Exception:
        x = float(default)
    if x != x:  # NaN
        x = float(default)
    return max(0.0, min(1.0, x))


def _safe_mean(values, default=0.0):
    """NaN-safe mean over an iterable of numeric values."""
    clean = []
    for value in values:
        try:
            v = float(value)
        except Exception:
            continue
        if v == v:  # not NaN
            clean.append(v)
    if not clean:
        return float(default)
    return float(sum(clean) / len(clean))


# ═══════════════════════════════════════════════════════════════════════
# DETERMINISTIC HASHING
# ═══════════════════════════════════════════════════════════════════════

def stable_uniform_0_1(*parts) -> float:
    """Deterministic uniform[0,1) derived from stringified parts.

    Uses BLAKE2b (not Python hash, which is salted per-process).
    """
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(str(p).encode('utf-8'))
        h.update(b'|')
    return int.from_bytes(h.digest(), 'big') / 2**64


# ═══════════════════════════════════════════════════════════════════════
# LATENT VARIABLE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

# Latent tier order matches generate_latent_vars_api.py prompts:
# [Micro, Indie, Mid, A, Epic]
LATENT_TIER_ORDER = ["Micro", "Indie", "Mid", "A", "Epic"]
TIER_TO_LATENT_IDX = {t: i for i, t in enumerate(LATENT_TIER_ORDER)}

# D2-FIX: extend with common aliases that appear across modules
TIER_TO_LATENT_IDX.update({
    "A-List": 3, "Mid-Budget": 2, "Micro-Budget": 0,
    "Major": 3, "Global": 4,
})


COMPANY_GENRE_BASIS = [
    "Action", "Drama", "Comedy", "Sci-Fi", "Horror", "Romance",
    "Thriller", "Fantasy", "Mystery", "Documentary", "Crime", "Animation",
]
COMPANY_GENRE_INDEX = {genre.lower(): idx for idx, genre in enumerate(COMPANY_GENRE_BASIS)}

_GENRE_PROJECTION = {
    "Action": {"Action": 1.0},
    "Adventure": {"Action": 0.65, "Fantasy": 0.20, "Sci-Fi": 0.15},
    "Animation": {"Animation": 1.0},
    "Biography": {"Drama": 0.65, "Documentary": 0.35},
    "Comedy": {"Comedy": 1.0},
    "Crime": {"Crime": 0.7, "Thriller": 0.3},
    "Documentary": {"Documentary": 1.0},
    "Drama": {"Drama": 1.0},
    "Family": {"Animation": 0.55, "Comedy": 0.20, "Fantasy": 0.15, "Drama": 0.10},
    "Fantasy": {"Fantasy": 1.0},
    "Film-Noir": {"Mystery": 0.45, "Crime": 0.35, "Thriller": 0.20},
    "History": {"Drama": 0.60, "Documentary": 0.40},
    "Horror": {"Horror": 1.0},
    "Music": {"Drama": 0.35, "Documentary": 0.35, "Comedy": 0.15, "Romance": 0.15},
    "Musical": {"Comedy": 0.40, "Romance": 0.20, "Drama": 0.20, "Animation": 0.20},
    "Mystery": {"Mystery": 0.65, "Thriller": 0.35},
    "Romance": {"Romance": 1.0},
    "Sci-Fi": {"Sci-Fi": 1.0},
    "Sport": {"Drama": 0.55, "Documentary": 0.30, "Comedy": 0.15},
    "Thriller": {"Thriller": 1.0},
    "War": {"Action": 0.50, "Drama": 0.30, "Thriller": 0.20},
    "Western": {"Action": 0.55, "Drama": 0.25, "Mystery": 0.20},
    "Superhero": {"Action": 0.70, "Sci-Fi": 0.15, "Fantasy": 0.15},
    "Martial Arts": {"Action": 0.80, "Thriller": 0.20},
    "Disaster": {"Action": 0.60, "Thriller": 0.40},
    "Experimental": {"Documentary": 0.40, "Drama": 0.35, "Mystery": 0.25},
    "Short": {"Drama": 0.35, "Animation": 0.25, "Documentary": 0.20, "Comedy": 0.20},
    "Reality-TV": {"Documentary": 0.75, "Comedy": 0.25},
}


def _split_genre_tokens(raw) -> list[str]:
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str):
        txt = raw.replace("|", ";").replace(",", ";")
        return [token.strip() for token in txt.split(";") if token.strip()]
    return []


def project_genres_to_company_basis(raw) -> np.ndarray:
    """Project free-form/person genre labels into the 12-d company genre basis."""
    vec = np.zeros(len(COMPANY_GENRE_BASIS), dtype=np.float32)
    tokens = _split_genre_tokens(raw)
    if not tokens:
        return vec

    for token in tokens:
        projection = _GENRE_PROJECTION.get(token)
        if projection is None:
            projection = {token: 1.0} if token in GENRES else {}
        for basis_name, weight in projection.items():
            idx = COMPANY_GENRE_INDEX.get(str(basis_name).lower())
            if idx is not None:
                vec[idx] += float(weight)

    total = float(vec.sum())
    if total > 0:
        vec /= total
    return vec


def canonical_company_genre_vector(raw) -> np.ndarray:
    """Normalize company portfolio data onto the shared 12-d genre basis."""
    if isinstance(raw, str):
        projected = project_genres_to_company_basis(raw)
        if float(projected.sum()) > 0:
            return projected
        return np.full(len(COMPANY_GENRE_BASIS), 1.0 / len(COMPANY_GENRE_BASIS), dtype=np.float32)

    if isinstance(raw, (list, tuple)) and raw and any(isinstance(x, str) for x in raw):
        projected = project_genres_to_company_basis(raw)
        if float(projected.sum()) > 0:
            return projected
        return np.full(len(COMPANY_GENRE_BASIS), 1.0 / len(COMPANY_GENRE_BASIS), dtype=np.float32)

    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim != 1:
        return np.full(len(COMPANY_GENRE_BASIS), 1.0 / len(COMPANY_GENRE_BASIS), dtype=np.float32)
    if len(arr) >= len(COMPANY_GENRE_BASIS):
        arr = arr[: len(COMPANY_GENRE_BASIS)]
    else:
        padded = np.zeros(len(COMPANY_GENRE_BASIS), dtype=np.float32)
        padded[: len(arr)] = arr
        arr = padded
    total = float(arr.sum())
    if total <= 0:
        return np.full(len(COMPANY_GENRE_BASIS), 1.0 / len(COMPANY_GENRE_BASIS), dtype=np.float32)
    return arr / total

# D2-FIX: Canonical movie-tier names. Maps any alias → canonical name.
_TIER_ALIASES = {
    "Epic": "Epic", "A": "A", "Mid": "Mid", "Indie": "Indie", "Micro": "Micro",
    "A-List": "A", "Mid-Budget": "Mid", "Micro-Budget": "Micro",
}


def normalize_movie_tier(tier: str) -> str:
    """Map any tier name variant to canonical movie production tier.

    D2-FIX: resolves the inconsistency where ``secondary_tables.py`` used
    ``A-List`` / ``Mid-Budget`` / ``Micro-Budget`` (company-tier names)
    for movie-tier lookups, causing silent fallthrough to defaults.
    """
    return _TIER_ALIASES.get(str(tier).strip(), "Mid")


def style_spectacle_score(lv: dict) -> float:
    """Map creative_style_vector[7] (intimate..spectacle) to [0,1]."""
    vec = lv.get("creative_style_vector")
    if isinstance(vec, list) and len(vec) >= 8:
        try:
            x = float(vec[7])
        except Exception:
            return 0.5
        if x != x:
            return 0.5
        return max(0.0, min(1.0, (x + 1.0) / 2.0))
    return 0.5


# ═══════════════════════════════════════════════════════════════════════
# TONE -> STYLE TAG MAPPING
# ═══════════════════════════════════════════════════════════════════════

# Heuristic mapping from sampled movie "tone" -> style tag hints.
# Used in casting to boost actors whose style tags fit the movie tone.
TONE_STYLE_HINTS = {
    "intense": ["intense", "explosive", "physical", "menacing"],
    "emotional": ["vulnerable", "understated", "theatrical", "lyrical"],
    "light": ["comedic", "magnetic", "improvisational"],
    "cerebral": ["cerebral", "minimalist"],
    "dark": ["menacing", "neo-noir", "slow-burn", "atmospheric"],
    "warm": ["vulnerable", "magnetic", "naturalistic", "lyrical"],
    "suspenseful": ["menacing", "neo-noir", "slow-burn", "atmospheric"],
    "epic": ["epic-scale", "visual-spectacle", "explosive", "physical"],
    "atmospheric": ["atmospheric", "neo-noir", "slow-burn"],
    "observational": ["naturalistic", "documentary-style", "handheld"],
    "gritty": ["intense", "neo-noir", "handheld", "naturalistic"],
    "whimsical": ["comedic", "theatrical", "lyrical", "surrealist"],
    "neutral": [],
}
