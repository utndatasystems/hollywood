"""
Mirage movie assembly -- assembly.py
===================================
Movie component selection: director, cast, companies, crew, title, keywords.

This rewrite keeps the public API stable while cleaning up three chronic problems
from the previous version:

1. Too much orchestration leaked into local selection code.
2. Several scoring passes mixed vectorised blocks with thousands of tiny Python
   loops, so runtime scaled badly as the world grew.
3. Some temporal / event-system signals existed in world state but were not
   consistently consumed here, especially country overrides and temporal edge
   validity.

The module still exports the same core entry points expected by generate_movies:
    - sample_movie_concept
    - pick_director
    - pick_co_director
    - pick_companies
    - pick_cast
    - pick_title
    - pick_keywords
    - pick_crew
"""
from __future__ import annotations

import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from contracts import (
    ARCHETYPES,
    CAST_SIZE_RANGES,
    COUNTRIES,
    COUNTRY_LANGUAGE,
    COUNTRY_WEIGHTS,
    CREW_DEPARTMENTS,
    DECADE_WEIGHTS,
    FRANCHISE_CONFIG,
    GENRES,
    GENRE_WEIGHTS,
    PRODUCTION_TIERS,
    TIER_WEIGHTS,
    YEAR_RANGE,
    generate_compositional_title,
)
from bootstrap_artifacts import (
    audit_artifact_usage,
    audit_fallback_hit,
    current_mode,
    flatten_modeling_priors,
    prior_float_from_section,
    prior_section,
)
from financials import edge_is_active
from utils import (
    TIER_TO_LATENT_IDX,
    TONE_STYLE_HINTS,
    canonical_company_genre_vector,
    normalize_weights,
    project_genres_to_company_basis,
)
from policy_runtime import (
    append_jsonl,
    bucket_for_year,
    confidence_from_scores,
    modeling_priors_path,
    resolve_year_slate,
    resolve_company_strategy,
)
from text_polish import (
    clean_display_text,
    contains_placeholder_syntax,
    looks_like_weak_tagline,
    looks_like_weak_title,
    sanitize_character_name,
    sanitize_tagline,
    sanitize_title,
)
from world_state import WorldState, get_person_latent, latent_similarity_batch
from topup_title_bank import (
    _load_research_grammar as _load_title_research_grammar,
    _materialize_tagline_for_title as _materialize_title_bank_tagline,
    _render_title as _render_title_bank_title,
    _tagline as _render_title_bank_tagline,
)


# ---------------------------------------------------------------------------
# Geography / scale helpers
# ---------------------------------------------------------------------------

_NATIONALITY_TO_COUNTRY: dict[str, str] = {
    "American": "USA", "British": "UK", "French": "France", "German": "Germany",
    "Indian": "India", "Japanese": "Japan", "Chinese": "China", "Korean": "South Korea",
    "Italian": "Italy", "Spanish": "Spain", "Brazilian": "Brazil",
    "Mexican": "Mexico", "Australian": "Australia", "Canadian": "Canada",
    "Russian": "Russia", "Nigerian": "Nigeria", "Swedish": "Sweden",
    "Danish": "Denmark", "Norwegian": "Norway", "Polish": "Poland",
    "Turkish": "Turkey", "Iranian": "Iran", "Argentine": "Argentina",
    "Colombian": "Colombia", "Egyptian": "Egypt", "South African": "South Africa",
    "Thai": "Thailand", "Indonesian": "Indonesia", "Filipino": "Philippines",
    "Pakistani": "Pakistan", "Bangladeshi": "Bangladesh", "Greek": "Greece",
    "Dutch": "Netherlands", "Belgian": "Belgium", "Swiss": "Switzerland",
    "Austrian": "Austria", "Portuguese": "Portugal", "Czech": "Czech Republic",
    "Romanian": "Romania", "Hungarian": "Hungary", "Ukrainian": "Ukraine",
}

_GEO_BOOST_BY_TIER: dict[str, float] = {
    "Epic": 1.5,
    "A": 2.0,
    "Mid": 3.0,
    "Indie": 4.5,
    "Micro": 5.0,
}

_DYNAMIC_CAST_BASE: dict[str, tuple[int, int]] = {
    "Epic": (16, 44),
    "A": (9, 24),
    "Mid": (4, 12),
    "Indie": (2, 7),
    "Micro": (1, 4),
}

_BLOCKBUSTER_GENRES = {
    "Action", "Sci-Fi", "Fantasy", "Superhero", "Adventure", "War",
}


def _to_str_ndarray(values: Any) -> np.ndarray:
    raw_arr = np.asarray(values)
    if raw_arr.dtype.kind in ("U", "S"):
        if raw_arr.ndim == 0:
            return raw_arr.reshape(1).astype(str, copy=False)
        return raw_arr.astype(str, copy=False)
    arr = np.asarray(values, dtype=object)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    flat = arr.reshape(-1)
    normalized: list[str] = []
    for value in flat:
        try:
            if value is None or bool(pd.isna(value)):
                normalized.append("")
                continue
        except Exception:
            pass
        normalized.append(str(value))
    return np.asarray(normalized, dtype=str).reshape(arr.shape)

_MAJOR_HUBS = {
    "USA", "UK", "China", "India", "Japan", "France", "Germany",
    "South Korea", "Australia", "Canada", "Brazil", "Italy", "Spain",
}

_COUNTRY_TO_MARKET = {
    "USA": "North America",
    "Canada": "North America",
    "UK": "Europe",
    "France": "Europe",
    "Germany": "Europe",
    "Italy": "Europe",
    "Spain": "Europe",
    "Sweden": "Europe",
    "India": "Asia",
    "Japan": "Asia",
    "South Korea": "Asia",
    "China": "Asia",
    "Australia": "Oceania",
    "Nigeria": "Africa",
    "Brazil": "South America",
    "Mexico": "Latin America",
    "Argentina": "Latin America",
}

_GENRE_TONE = {
    "Action": "intense",
    "Drama": "emotional",
    "Comedy": "light",
    "Sci-Fi": "cerebral",
    "Horror": "dark",
    "Romance": "warm",
    "Thriller": "suspenseful",
    "Fantasy": "epic",
    "Animation": "whimsical",
    "Documentary": "observational",
    "Crime": "gritty",
    "Mystery": "atmospheric",
    "War": "somber",
}

_GENRE_TIER_DIST = {
    "Action":      np.array([0.15, 0.30, 0.35, 0.15, 0.05]),
    "Sci-Fi":      np.array([0.12, 0.25, 0.35, 0.20, 0.08]),
    "Fantasy":     np.array([0.15, 0.30, 0.30, 0.18, 0.07]),
    "Animation":   np.array([0.10, 0.25, 0.40, 0.20, 0.05]),
    "Drama":       np.array([0.03, 0.12, 0.40, 0.35, 0.10]),
    "Comedy":      np.array([0.02, 0.10, 0.45, 0.35, 0.08]),
    "Thriller":    np.array([0.05, 0.15, 0.40, 0.30, 0.10]),
    "Horror":      np.array([0.02, 0.05, 0.25, 0.45, 0.23]),
    "Romance":     np.array([0.02, 0.08, 0.35, 0.40, 0.15]),
    "Documentary": np.array([0.00, 0.02, 0.15, 0.45, 0.38]),
    "Crime":       np.array([0.03, 0.12, 0.40, 0.35, 0.10]),
    "Mystery":     np.array([0.02, 0.10, 0.38, 0.38, 0.12]),
    "War":         np.array([0.06, 0.18, 0.34, 0.28, 0.14]),
}

_DEFAULT_DIRECTOR_SELECTION = {
    "genre_match_boost": 3.0,
    "geo_boost_scale": 0.60,
    "geo_boost_floor": 1.50,
    "film_count_decay_over_30": 0.70,
    "film_count_decay_over_50": 0.50,
    "film_count_decay_over_80": 0.30,
    "company_multiplier_rescale": 2.0 / 1.8,
    "event_franchise_pop_quantile": 0.75,
    "event_franchise_pop_boost": 1.18,
    "prestige_drama_alignment_threshold": 0.65,
    "prestige_drama_alignment_boost": 1.12,
    "co_director_probability_by_tier": {"Epic": 0.10, "A": 0.07, "Mid": 0.04, "Indie": 0.01, "Micro": 0.0},
}

_DEFAULT_COMPANY_SELECTION = {
    "strategy_match_boost": 1.25,
    "market_bias_base": 0.90,
    "market_bias_scale": 1.20,
    "partner_affinity_scale": 3.0,
    "partner_rivalry_penalty_floor": 0.05,
    "family_boost": 6.0,
}

_DEFAULT_CAST_SELECTION = {
    "geo_boost_by_tier": dict(_GEO_BOOST_BY_TIER),
    "dynamic_cast_base_by_tier": {k: [int(v[0]), int(v[1])] for k, v in _DYNAMIC_CAST_BASE.items()},
    "blockbuster_bonus_major": 8.0,
    "blockbuster_bonus_other": 2.0,
    "franchise_bonus_major": 10.0,
    "franchise_bonus_other": 2.0,
    "epic_tail_prob_base": 0.10,
    "epic_tail_prob_franchise_bonus": 0.08,
    "epic_tail_prob_blockbuster_bonus": 0.05,
    "epic_tail_prob_cap": 0.55,
    "epic_tail_lognorm_mean": 3.75,
    "epic_tail_lognorm_sigma": 0.42,
    "epic_tail_min": 48,
    "a_tail_prob": 0.06,
    "a_tail_min": 30,
    "a_tail_max": 86,
    "unused_actor_boost": 2.10,
    "award_recent_boost": 1.50,
    "franchise_pool_base_boost": 1.70,
    "star_vehicle_slot0_boost": 1.25,
    "prestige_pairing_boost": 1.12,
    "volatile_ensemble_boost": 1.10,
    "balanced_ensemble_boost": 1.08,
    "agency_match_boost": 2.0,
    "gender_novelty_boost": 1.50,
    "nationality_novelty_boost": 1.30,
    "tag_similarity_penalty": 0.50,
}

_DEFAULT_KEYWORD_SELECTION = {
    "count_by_tier": {"Epic": [5, 8], "A": [5, 8], "Mid": [4, 6], "Indie": [3, 5], "Micro": [3, 5]},
    "franchise_min_count": 5.0,
    "exact_genre_boost": 5.25,
    "family_genre_boost": 1.28,
    "family_genre_max_share": 0.35,
    "off_genre_penalty": 0.18,
    "lexical_match_scale": 0.56,
    "lexical_match_cap": 2.5,
    "year_slate_family_boosts": {
        "relationship": {"relationship": 1.18, "tone": 1.08},
        "ensemble": {"relationship": 1.10, "profession": 1.10},
        "event": {"event": 1.16, "setting": 1.08, "object": 1.06},
        "global": {"place": 1.12, "setting": 1.08},
        "analogue": {"tone": 1.05},
        "platform": {"franchise": 1.08, "sequel_drift": 1.05},
    },
    "specificity_tier1_penalty": 0.34,
    "generic_motif_penalty": 0.26,
    "specific_story_boost": 1.08,
    "franchise_scope_boost_base": 1.65,
    "franchise_scope_affinity_scale": 0.55,
    "franchise_family_boost": 1.42,
    "franchise_recurrence_base": 0.98,
    "franchise_recurrence_scale": 0.58,
    "nonfranchise_scope_penalty": 0.24,
    "nonfranchise_family_penalty": 0.34,
    "nonfranchise_affinity_penalty": 0.58,
    "nonfranchise_affinity_threshold": 0.40,
    "high_specificity_novelty_base": 0.92,
    "high_specificity_novelty_scale": 0.50,
    "movie_scope_novelty_base": 0.90,
    "movie_scope_novelty_scale": 0.42,
    "usage_penalty_scale": 0.024,
    "company_exact_boost": 1.38,
    "company_family_boost": 1.05,
    "franchise_core_boost": 4.25,
}

_DEFAULT_EXACT_TOPIC_MIN_COUNT_BY_TIER = {"Epic": 3.0, "A": 3.0, "Mid": 2.0, "Indie": 2.0, "Micro": 1.0}
_DEFAULT_PRIMARY_PLUS_RELATED_MIN_COUNT_BY_TIER = {"Epic": 5.0, "A": 5.0, "Mid": 4.0, "Indie": 3.0, "Micro": 2.0}
_DEFAULT_GENERIC_KEYWORD_CAP_BY_TIER = {"Epic": 1.0, "A": 1.0, "Mid": 1.0, "Indie": 1.0, "Micro": 1.0}
_DEFAULT_OFF_GENRE_CAP_BY_TIER = {"Epic": 1.0, "A": 1.0, "Mid": 1.0, "Indie": 1.0, "Micro": 1.0}
_DEFAULT_KEYWORD_SLOT_MIX_BY_TIER = {
    "Epic": {"exact_anchor": 0.50, "related_support": 0.12, "story_specific": 0.23, "franchise": 0.10, "generic": 0.05},
    "A": {"exact_anchor": 0.50, "related_support": 0.12, "story_specific": 0.23, "franchise": 0.10, "generic": 0.05},
    "Mid": {"exact_anchor": 0.45, "related_support": 0.15, "story_specific": 0.25, "franchise": 0.10, "generic": 0.05},
    "Indie": {"exact_anchor": 0.45, "related_support": 0.15, "story_specific": 0.30, "franchise": 0.05, "generic": 0.05},
    "Micro": {"exact_anchor": 0.40, "related_support": 0.20, "story_specific": 0.35, "franchise": 0.00, "generic": 0.05},
}
_DEFAULT_RELATED_GENRES_BY_GENRE = {
    "Action": ["Adventure", "Thriller", "Superhero", "War"],
    "Adventure": ["Action", "Fantasy", "Family", "Sci-Fi"],
    "Animation": ["Family", "Fantasy", "Comedy", "Adventure"],
    "Biography": ["Drama", "History", "Music", "War"],
    "Comedy": ["Romance", "Family", "Sport", "Animation"],
    "Crime": ["Thriller", "Mystery", "Drama", "Action"],
    "Disaster": ["Action", "Thriller", "Sci-Fi", "Adventure"],
    "Documentary": ["Biography", "History", "Music", "Sport"],
    "Drama": ["Romance", "Biography", "History", "Crime"],
    "Experimental": ["Fantasy", "Sci-Fi", "Drama", "Horror"],
    "Family": ["Animation", "Adventure", "Comedy", "Fantasy"],
    "Fantasy": ["Adventure", "Sci-Fi", "Animation", "Family"],
    "Film-Noir": ["Crime", "Mystery", "Thriller", "Drama"],
    "History": ["Biography", "Drama", "War", "Documentary"],
    "Horror": ["Mystery", "Thriller", "Fantasy", "Sci-Fi"],
    "Martial Arts": ["Action", "Adventure", "Crime", "Thriller"],
    "Music": ["Musical", "Drama", "Biography", "Documentary"],
    "Musical": ["Music", "Comedy", "Romance", "Drama"],
    "Mystery": ["Thriller", "Crime", "Horror", "Film-Noir"],
    "Reality-TV": ["Documentary", "Comedy", "Music", "Sport"],
    "Romance": ["Drama", "Comedy", "Musical", "Biography"],
    "Sci-Fi": ["Fantasy", "Action", "Adventure", "Disaster"],
    "Short": ["Experimental", "Animation", "Comedy", "Drama"],
    "Sport": ["Drama", "Comedy", "Biography", "Documentary"],
    "Superhero": ["Action", "Adventure", "Sci-Fi", "Fantasy"],
    "Thriller": ["Crime", "Mystery", "Action", "Horror"],
    "War": ["Action", "History", "Drama", "Biography"],
    "Western": ["Action", "Drama", "Adventure", "Crime"],
}


# ---------------------------------------------------------------------------
# Character / tagline banks
# ---------------------------------------------------------------------------

_GENRE_CHAR_NAMES = {
    "action": {
        "M": ["Jake Reaper", "Sgt. Stone", "Rex Viper", "Duke Slater", "Marcus Blaze",
              "Hawk Jensen", "Brick Malone", "Logan Cruz", "Axel Storm", "Kane Bishop"],
        "F": ["Maya Cruz", "Elena Black", "Tara Fox", "Nina Cortez", "Sierra Voss",
              "Jordan Steele", "Athena Sharp", "Raven Cole", "Zara Knight", "Jade Fury"],
    },
    "comedy": {
        "M": ["Buddy Feldman", "Phil Bumble", "Benny Chuckles", "Gus Wobble", "Norm Dinkle",
              "Larry Fink", "Doug Peppers", "Ted Crumble", "Milo Pratt", "Chip Wadsworth"],
        "F": ["Liz Trotter", "Margot Fizz", "Diane Pratt", "Sally Sparks", "Patty Loop",
              "Dottie Banks", "Brenda Pluck", "Wendy Quirk", "Greta Bloom", "Faye Nibbles"],
    },
    "drama": {
        "M": ["Thomas Mercer", "William Hale", "James Whitfield", "Michael Carey", "Arthur Webb",
              "Daniel Cross", "Edward Blake", "Henry Thorne", "Robert Ashworth", "Samuel Voss"],
        "F": ["Claire Ashton", "Rebecca Forsythe", "Eleanor Voss", "Isabelle Dunn", "Catherine Pierce",
              "Margaret Hayes", "Vivian Cross", "Helen Marsh", "Grace Whitmore", "Lillian Ford"],
    },
    "sci-fi": {
        "M": ["Commander Kael", "Orion Voss", "Axel Quantum", "Professor Marsh", "Tau Epsilon",
              "Dr. Renn Solaris", "Major Atlas", "Cipher-9", "Zane Helix", "Capt. Holt"],
        "F": ["Dr. Nova Rix", "Zara-7", "Captain Sera Blaine", "Lyra Xenon", "Juno Six",
              "Aria Nebula", "Lt. Kira Voss", "Mira Starfall", "Echo Prime", "Dr. Elise Kepler"],
    },
    "horror": {
        "M": ["The Hollow Man", "Pastor Vex", "Dr. Graves", "The Watcher", "Father Morrow",
              "The Reaper", "Brother Silent", "Mr. Wilt", "The Surgeon", "Deacon Ash"],
        "F": ["Sister Agnes", "Darla Crane", "Mira Blackwood", "Evelyn Shade", "Ruby Thorn",
              "Moira Glass", "The Bride", "Nurse Hallow", "Lily Grave", "Constance Veil"],
    },
    "romance": {
        "M": ["Julien Marchand", "Daniel Hart", "Oliver Reed", "Marcus Cavanaugh", "Leo Ashford",
              "Sebastian Cole", "Ethan Sinclair", "Alexander Frost", "Gabriel Montague", "Theo Fairchild"],
        "F": ["Sophia Belmont", "Lily Fairweather", "Camille Laurent", "Rose Delacroix", "Natalie Summers",
              "Isabelle Chase", "Vivienne Hart", "Clara Beaumont", "Juliana Voss", "Eloise Wren"],
    },
    "thriller": {
        "M": ["Victor Kane", "Dr. Orin West", "Agent Cole Bishop", "Mason Hargrave", "Pierce Ashton",
              "Det. Jack Mercer", "Marcus Holt", "Raymond Cross", "Felix Strand", "Callum Grey"],
        "F": ["Det. Sarah Voss", "Nadia Cipher", "Claire Devlin", "Alina Markova", "Vera Cross",
              "Agent Lena Park", "Dr. Maren Hale", "Irina Kozlov", "Diana Thorn", "Cassandra Wren"],
    },
    "animation": {
        "N": ["Sparky", "Captain Fluffbeard", "Zippy the Fox", "Old Sage Oak", "Ember",
              "Pip Starlight", "Shadow Paw", "Snickers", "Blaze", "Whiskers",
              "Luna Moonwhisker", "Princess Coral", "Queen Blossom", "Petal", "Twinkle",
              "Prince Bramble", "Sir Hoot", "Nutmeg", "Frost", "Sunbeam"],
    },
    "fantasy": {
        "M": ["Lord Aldric", "Theron the Brave", "Grimjaw", "Prince Kael", "Dregan Ironhelm",
              "Sir Gideon", "Warden Kross", "Orin Darkfire", "Bael Stonehelm", "Ragnar Wolfsbane"],
        "F": ["Elara Windborne", "Lady Ashworth", "Sorceress Ilya", "Mira Frostweaver", "Celeste Moonshadow",
              "Queen Seraphina", "Priestess Yara", "Isolde Brightveil", "Runa Starwoven", "Sylve Thornheart"],
    },
    "documentary": {
        "N": ["Narrator", "Subject A", "Expert Witness", "The Director", "Interview Subject",
              "Commentator", "Field Reporter", "Historian", "Survivor", "Scientist",
              "Analyst", "Activist", "Witness", "Researcher", "Advocate"],
    },
}

_GENRE_KEY_MAP = {
    "science fiction": "sci-fi",
    "scifi": "sci-fi",
    "sci fi": "sci-fi",
    "animated": "animation",
    "adventure": "action",
    "war": "action",
    "mystery": "thriller",
    "crime": "thriller",
    "noir": "thriller",
    "musical": "comedy",
    "family": "animation",
    "historical": "drama",
    "period": "drama",
    "western": "action",
    "superhero": "action",
}

_GENRE_TO_ARCHETYPES = {
    "action": ["Lead Hero", "Lead Villain", "Sidekick", "Henchman", "Supporting"],
    "drama": ["Lead Hero", "Mentor", "Supporting", "Love Interest", "Authority Figure"],
    "comedy": ["Comic Relief", "Sidekick", "Love Interest", "Supporting", "Lead Hero"],
    "horror": ["Victim", "Lead Villain", "Lead Hero", "Mysterious Stranger", "Supporting"],
    "sci-fi": ["Lead Hero", "Lead Villain", "Sidekick", "Authority Figure", "Supporting"],
    "fantasy": ["Lead Hero", "Lead Villain", "Mentor", "Mysterious Stranger", "Sidekick"],
    "thriller": ["Lead Hero", "Lead Villain", "Mysterious Stranger", "Authority Figure", "Supporting"],
    "romance": ["Love Interest", "Lead Hero", "Sidekick", "Mentor", "Supporting"],
    "animation": ["Lead Hero", "Sidekick", "Lead Villain", "Comic Relief", "Supporting"],
    "documentary": ["Authority Figure", "Supporting", "Mentor", "Lead Hero"],
    "crime": ["Lead Hero", "Lead Villain", "Henchman", "Authority Figure", "Supporting"],
    "mystery": ["Lead Hero", "Lead Villain", "Mysterious Stranger", "Victim", "Supporting"],
    "war": ["Lead Hero", "Lead Villain", "Henchman", "Mentor", "Supporting"],
}

_TAGLINE_TEMPLATES = {
    "Action": [
        "No rules. No limits. No mercy.",
        "The only way out is through.",
        "When the dust settles, only one will stand.",
        "Payback has a new name.",
        "This time, it's personal.",
        "Some fights can't be won. This one must be.",
        "Heroes aren't born. They're forged.",
        "The clock is ticking.",
    ],
    "Comedy": [
        "Expect the unexpected. Then laugh.",
        "Life's a mess. Might as well enjoy it.",
        "Rules were made to be broken... hilariously.",
        "Some mistakes are worth repeating.",
        "You can't make this stuff up. Actually, we did.",
        "Normal is overrated.",
        "The worst plan ever... might just work.",
        "Chaos has never been this much fun.",
    ],
    "Drama": [
        "Every family has its secrets.",
        "The truth will set you free. Eventually.",
        "Some wounds never heal.",
        "A story that needs to be told.",
        "Behind every silence lies a story.",
        "The hardest battles are fought within.",
        "What we leave behind defines who we are.",
        "Sometimes the only way forward is to look back.",
    ],
    "Sci-Fi": [
        "The future is closer than you think.",
        "Humanity's greatest discovery... is its greatest threat.",
        "Beyond the stars. Beyond reason.",
        "Evolution doesn't ask permission.",
        "The universe doesn't care about your plans.",
        "First contact. Last chance.",
        "In space, the rules are different.",
        "What if everything you knew was designed?",
    ],
    "Horror": [
        "Don't look back.",
        "Some doors should stay closed.",
        "The darkness is listening.",
        "Fear is just the beginning.",
        "They're already inside.",
        "You can't escape what's in your head.",
        "Pray it doesn't find you.",
        "The dead don't rest here.",
    ],
    "Romance": [
        "Love finds a way. It always does.",
        "Two hearts. One chance.",
        "The greatest risk is not falling at all.",
        "Some stories are written in the stars.",
        "Love doesn't follow the rules.",
        "Where there's love, there's a way.",
        "Falling was the easy part.",
        "The heart wants what it wants.",
    ],
    "Thriller": [
        "Trust no one.",
        "The truth is the most dangerous weapon.",
        "Everyone has a breaking point.",
        "Nothing is what it seems.",
        "The game was rigged from the start.",
        "How far would you go to survive?",
        "Secrets have consequences.",
        "The closer you look, the less you see.",
    ],
    "Fantasy": [
        "Legends are born in darkness.",
        "A world beyond imagination.",
        "The prophecy was just the beginning.",
        "Magic always comes with a price.",
        "One realm. One chance. One destiny.",
        "The old magic is awakening.",
        "Kingdoms will fall. Heroes will rise.",
        "Beyond the veil lies another world.",
    ],
    "Animation": [
        "Adventure is just around the corner.",
        "Big dreams come in small packages.",
        "A journey beyond imagination.",
        "Some heroes are unexpected.",
        "Believe in the impossible.",
        "The adventure of a lifetime.",
    ],
    "Documentary": [
        "The story they didn't want told.",
        "Truth is stranger than fiction.",
        "See the world as it really is.",
        "A story that changes everything.",
        "What you don't know can change you.",
        "The untold story. Until now.",
    ],
    "Crime": [
        "In this city, everyone has a price.",
        "Justice has a dark side.",
        "The line between law and chaos.",
        "Every crime tells a story.",
        "Honor among thieves is a myth.",
        "The streets remember everything.",
    ],
    "Mystery": [
        "The answer is hiding in plain sight.",
        "Every clue leads deeper.",
        "Some puzzles are better left unsolved.",
        "The truth is buried. Start digging.",
        "What happened that night?",
        "Not everything can be explained.",
    ],
}

_CREW_DEPT = {
    "writer": "Production",
    "director_of_photography": "Camera",
    "cinematographer": "Camera",
    "editor": "Post-Production",
    "composer": "Sound",
    "sound_designer": "Sound",
    "costume_designer": "Art",
    "production_designer": "Art",
    "visual_effects_supervisor": "VFX",
    "vfx_supervisor": "VFX",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _env_float(name: str, default: float, lo: float | None = None, hi: float | None = None) -> float:
    raw = os.getenv(name)
    if raw is None:
        val = float(default)
    else:
        try:
            val = float(raw)
        except Exception:
            val = float(default)
    if lo is not None:
        val = max(float(lo), val)
    if hi is not None:
        val = min(float(hi), val)
    return float(val)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _workspace_priors(world: WorldState):
    return getattr(getattr(getattr(world, "workspace", None), "config", None), "priors", None)


def _artifact_priors_payload(world: WorldState) -> dict[str, Any]:
    payload = getattr(world, "modeling_priors_payload", {}) or {}
    return payload if isinstance(payload, dict) else {}


def _audit_modeling_priors_usage(world: WorldState, section: str, key: str | None = None) -> None:
    token = f"{section}.{key}" if key else str(section)
    seen = getattr(world, "_audited_modeling_prior_sections", None)
    if not isinstance(seen, set):
        seen = set()
        setattr(world, "_audited_modeling_prior_sections", seen)
    if token in seen:
        return
    seen.add(token)
    try:
        workspace = getattr(world, "workspace", None)
        base_dir = getattr(workspace, "base_dir", None)
        root = Path(base_dir).resolve() if base_dir else Path(__file__).resolve().parent
        audit_artifact_usage(
            "modeling_priors.json",
            modeling_priors_path(root),
            sections=[token],
        )
    except Exception:
        pass


def _prior_float(
    world: WorldState,
    name: str,
    default: float,
    lo: float | None = None,
    hi: float | None = None,
) -> float:
    priors = _workspace_priors(world)
    try:
        value = float(getattr(priors, name, default)) if priors is not None else float(default)
    except Exception:
        value = float(default)
    if priors is None or not hasattr(priors, name):
        flat = flatten_modeling_priors(_artifact_priors_payload(world))
        try:
            value = float(flat.get(name, value))
        except Exception:
            value = float(value)
    if lo is not None:
        value = max(float(lo), value)
    if hi is not None:
        value = min(float(hi), value)
    return float(value)


def _section_prior_float(
    world: WorldState,
    section: str,
    key: str,
    default: float,
    *,
    lo: float | None = None,
    hi: float | None = None,
) -> float:
    _audit_modeling_priors_usage(world, section, key)
    row = prior_section(_artifact_priors_payload(world), section)
    if key not in row:
        if current_mode() == "research":
            audit_fallback_hit(
                f"assembly.{section}",
                f"missing:{key}",
                detail=f"selection prior {section}.{key} is required in research mode",
                mode="research",
            )
        return prior_float_from_section(_artifact_priors_payload(world), section, key, default, lo=lo, hi=hi)
    try:
        value = float(row.get(key))
    except Exception:
        if current_mode() == "research":
            audit_fallback_hit(
                f"assembly.{section}",
                f"invalid:{key}",
                detail=f"selection prior {section}.{key} must be numeric in research mode",
                mode="research",
            )
        value = float(default)
    if lo is not None:
        value = max(float(lo), value)
    if hi is not None:
        value = min(float(hi), value)
    return float(value)


def _section_prior_dict(world: WorldState, section: str, key: str) -> dict[str, Any]:
    _audit_modeling_priors_usage(world, section, key)
    row = prior_section(_artifact_priors_payload(world), section)
    value = row.get(key, {})
    if (not isinstance(value, dict) or not value) and current_mode() == "research":
        audit_fallback_hit(
            f"assembly.{section}",
            f"missing:{key}",
            detail=f"selection prior {section}.{key} must be a non-empty object in research mode",
            mode="research",
        )
    return value if isinstance(value, dict) else {}


def _section_prior_list(world: WorldState, section: str, key: str, *, expected_len: int | None = None) -> list[float] | None:
    _audit_modeling_priors_usage(world, section, key)
    row = prior_section(_artifact_priors_payload(world), section)
    value = row.get(key)
    if not isinstance(value, (list, tuple)):
        if current_mode() == "research":
            audit_fallback_hit(
                f"assembly.{section}",
                f"missing:{key}",
                detail=f"selection prior {section}.{key} must be a list in research mode",
                mode="research",
            )
        return None
    try:
        out = [float(v) for v in value]
    except Exception:
        if current_mode() == "research":
            audit_fallback_hit(
                f"assembly.{section}",
                f"invalid:{key}",
                detail=f"selection prior {section}.{key} must contain numeric values in research mode",
                mode="research",
            )
        return None
    if expected_len is not None and len(out) != int(expected_len):
        if current_mode() == "research":
            audit_fallback_hit(
                f"assembly.{section}",
                f"invalid:{key}",
                detail=f"selection prior {section}.{key} must have length {expected_len} in research mode",
                mode="research",
            )
        return None
    return out


def _section_prior_float_map(
    world: WorldState,
    section: str,
    key: str,
    default: dict[str, float],
    *,
    lo: float | None = None,
    hi: float | None = None,
) -> dict[str, float]:
    raw = _section_prior_dict(world, section, key)
    if current_mode() == "research":
        missing = [str(k) for k in default.keys() if k not in raw]
        if missing:
            audit_fallback_hit(
                f"assembly.{section}",
                f"missing:{key}",
                detail=f"selection prior {section}.{key} is missing keys: {', '.join(missing)}",
                mode="research",
            )
    out = {str(k): float(v) for k, v in default.items()}
    for k, v in raw.items():
        try:
            value = float(v)
        except Exception:
            if current_mode() == "research":
                audit_fallback_hit(
                    f"assembly.{section}",
                    f"invalid:{key}.{k}",
                    detail=f"selection prior {section}.{key}.{k} must be numeric in research mode",
                    mode="research",
                )
            continue
        if lo is not None:
            value = max(float(lo), value)
        if hi is not None:
            value = min(float(hi), value)
        out[str(k)] = float(value)
    return out


def _section_prior_range_map(
    world: WorldState,
    section: str,
    key: str,
    default: dict[str, tuple[int, int] | list[int]],
) -> dict[str, tuple[int, int]]:
    raw = _section_prior_dict(world, section, key)
    if current_mode() == "research":
        missing = [str(k) for k in default.keys() if k not in raw]
        if missing:
            audit_fallback_hit(
                f"assembly.{section}",
                f"missing:{key}",
                detail=f"selection prior {section}.{key} is missing keys: {', '.join(missing)}",
                mode="research",
            )
    out: dict[str, tuple[int, int]] = {}
    for k, v in default.items():
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            out[str(k)] = (int(v[0]), int(v[1]))
    for k, v in raw.items():
        pair: tuple[int, int] | None = None
        if isinstance(v, dict):
            try:
                pair = (int(v.get("min")), int(v.get("max")))
            except Exception:
                pair = None
        elif isinstance(v, (list, tuple)) and len(v) >= 2:
            try:
                pair = (int(v[0]), int(v[1]))
            except Exception:
                pair = None
        elif isinstance(v, (int, float)):
            try:
                span = int(round(float(v)))
                pair = (max(1, span), max(1, span))
            except Exception:
                pair = None
        if pair is None:
            if current_mode() == "research":
                audit_fallback_hit(
                    f"assembly.{section}",
                    f"invalid:{key}.{k}",
                    detail=f"selection prior {section}.{key}.{k} must be a [min, max] pair or {{min,max}} object in research mode",
                    mode="research",
                )
            continue
        lo, hi = pair
        if hi < lo:
            lo, hi = hi, lo
        out[str(k)] = (int(lo), int(hi))
    return out


def _selection_config(world: WorldState, key: str) -> dict[str, Any]:
    return _section_prior_dict(world, "selection_weights", key)


def _keyword_generation_float(
    world: WorldState,
    key: str,
    default: float,
    *,
    lo: float | None = None,
    hi: float | None = None,
) -> float:
    _audit_modeling_priors_usage(world, "keyword_generation", key)
    row = prior_section(_artifact_priors_payload(world), "keyword_generation")
    raw = row.get(key)
    if raw is None and current_mode() == "research":
        audit_fallback_hit(
            "assembly.keyword_generation",
            f"missing:{key}",
            detail=f"keyword_generation.{key} is required in research mode",
            mode="research",
        )
    try:
        value = float(default if raw is None else raw)
    except Exception:
        value = float(default)
    if lo is not None:
        value = max(float(lo), value)
    if hi is not None:
        value = min(float(hi), value)
    return float(value)


def _character_generation_dict(world: WorldState, key: str, default: dict[str, Any]) -> dict[str, Any]:
    _audit_modeling_priors_usage(world, "character_generation", key)
    row = prior_section(_artifact_priors_payload(world), "character_generation")
    value = row.get(key, {})
    if (not isinstance(value, dict) or not value) and current_mode() == "research":
        audit_fallback_hit(
            "assembly.character_generation",
            f"missing:{key}",
            detail=f"character_generation.{key} must be a non-empty object in research mode",
            mode="research",
        )
    if not isinstance(value, dict):
        return dict(default)
    out = dict(default)
    out.update(value)
    return out


def _character_generation_list(world: WorldState, key: str, default: list[str]) -> list[str]:
    _audit_modeling_priors_usage(world, "character_generation", key)
    row = prior_section(_artifact_priors_payload(world), "character_generation")
    value = row.get(key)
    if not isinstance(value, list) or not value:
        if current_mode() == "research":
            audit_fallback_hit(
                "assembly.character_generation",
                f"missing:{key}",
                detail=f"character_generation.{key} must be a non-empty list in research mode",
                mode="research",
            )
        return list(default)
    return [str(item) for item in value if str(item).strip()]


def _selection_nested_tier_float_map(
    world: WorldState,
    block_name: str,
    nested_key: str,
    default: dict[str, float],
    *,
    lo: float | None = None,
    hi: float | None = None,
) -> dict[str, float]:
    block = _selection_config(world, block_name)
    raw = block.get(nested_key)
    if not isinstance(raw, dict):
        raw = _section_prior_dict(world, "selection_weights", nested_key)
    if current_mode() == "research":
        missing = [str(k) for k in default.keys() if k not in raw]
        if missing:
            audit_fallback_hit(
                "assembly.selection_weights",
                f"missing:{block_name}.{nested_key}",
                detail=f"selection_weights.{block_name}.{nested_key} is missing keys: {', '.join(missing)}",
                mode="research",
            )
    out = {str(k): float(v) for k, v in default.items()}
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                value = float(v)
            except Exception:
                continue
            if lo is not None:
                value = max(float(lo), value)
            if hi is not None:
                value = min(float(hi), value)
            out[str(k)] = float(value)
    return out


def _merge_required_selection_config(
    world: WorldState,
    block_name: str,
    default_cfg: dict[str, Any],
    *,
    optional_keys: set[str] | None = None,
) -> dict[str, Any]:
    raw = _selection_config(world, block_name)
    if current_mode() == "research":
        excluded = set(optional_keys or set())
        missing = [str(key) for key in default_cfg.keys() if key not in excluded and key not in raw]
        if missing:
            audit_fallback_hit(
                "assembly.selection_weights",
                f"missing:{block_name}",
                detail=f"selection_weights.{block_name} is missing keys: {', '.join(missing)}",
                mode="research",
            )
    cfg = dict(default_cfg)
    cfg.update(raw)
    return cfg


def _director_selection_config(world: WorldState) -> dict[str, Any]:
    cfg = _merge_required_selection_config(
        world,
        "director_selection",
        _DEFAULT_DIRECTOR_SELECTION,
        optional_keys={"co_director_probability_by_tier"},
    )
    cfg["co_director_probability_by_tier"] = _section_prior_float_map(
        world,
        "selection_weights",
        "co_director_probability_by_tier",
        _DEFAULT_DIRECTOR_SELECTION["co_director_probability_by_tier"],
        lo=0.0,
        hi=1.0,
    )
    return cfg


def _company_selection_config(world: WorldState) -> dict[str, Any]:
    return _merge_required_selection_config(world, "company_selection", _DEFAULT_COMPANY_SELECTION)


def _cast_selection_config(world: WorldState) -> dict[str, Any]:
    cfg = _merge_required_selection_config(
        world,
        "cast_selection",
        _DEFAULT_CAST_SELECTION,
        optional_keys={"geo_boost_by_tier", "dynamic_cast_base_by_tier"},
    )
    cfg["geo_boost_by_tier"] = _section_prior_float_map(
        world,
        "selection_weights",
        "geo_boost_by_tier",
        _DEFAULT_CAST_SELECTION["geo_boost_by_tier"],
        lo=1.0,
        hi=10.0,
    )
    cfg["dynamic_cast_base_by_tier"] = _section_prior_range_map(
        world,
        "selection_weights",
        "dynamic_cast_base_by_tier",
        _DEFAULT_CAST_SELECTION["dynamic_cast_base_by_tier"],
    )
    return cfg


def _keyword_selection_config(world: WorldState) -> dict[str, Any]:
    cfg = _merge_required_selection_config(
        world,
        "keyword_selection",
        _DEFAULT_KEYWORD_SELECTION,
        optional_keys={"count_by_tier", "year_slate_family_boosts"},
    )
    cfg["count_by_tier"] = _section_prior_range_map(
        world,
        "selection_weights",
        "keyword_count_by_tier",
        _DEFAULT_KEYWORD_SELECTION["count_by_tier"],
    )
    raw_boosts = _section_prior_dict(world, "selection_weights", "keyword_year_slate_family_boosts")
    boosts = dict(_DEFAULT_KEYWORD_SELECTION["year_slate_family_boosts"])
    for family, payload in raw_boosts.items():
        if isinstance(payload, dict):
            boosts[str(family)] = {str(k): float(v) for k, v in payload.items() if isinstance(v, (int, float))}
    cfg["year_slate_family_boosts"] = boosts
    cfg["exact_topic_min_count_by_tier"] = _selection_nested_tier_float_map(
        world,
        "keyword_selection",
        "exact_topic_min_count_by_tier",
        _DEFAULT_EXACT_TOPIC_MIN_COUNT_BY_TIER,
        lo=0.0,
        hi=8.0,
    )
    cfg["primary_plus_related_min_count_by_tier"] = _selection_nested_tier_float_map(
        world,
        "keyword_selection",
        "primary_plus_related_min_count_by_tier",
        _DEFAULT_PRIMARY_PLUS_RELATED_MIN_COUNT_BY_TIER,
        lo=0.0,
        hi=12.0,
    )
    cfg["generic_keyword_cap_by_tier"] = _selection_nested_tier_float_map(
        world,
        "keyword_selection",
        "generic_keyword_cap_by_tier",
        _DEFAULT_GENERIC_KEYWORD_CAP_BY_TIER,
        lo=0.0,
        hi=8.0,
    )
    cfg["off_genre_cap_by_tier"] = _selection_nested_tier_float_map(
        world,
        "keyword_selection",
        "off_genre_cap_by_tier",
        _DEFAULT_OFF_GENRE_CAP_BY_TIER,
        lo=0.0,
        hi=8.0,
    )
    raw_slot_mix = _section_prior_dict(world, "selection_weights", "keyword_selection").get("slot_mix_by_tier")
    slot_mix_cfg: dict[str, dict[str, float]] = {}
    if isinstance(raw_slot_mix, dict):
        for tier in PRODUCTION_TIERS:
            row = raw_slot_mix.get(str(tier))
            if isinstance(row, dict):
                slot_mix_cfg[str(tier)] = {
                    "exact_anchor": float(row.get("exact_anchor", _DEFAULT_KEYWORD_SLOT_MIX_BY_TIER[str(tier)]["exact_anchor"])),
                    "related_support": float(row.get("related_support", _DEFAULT_KEYWORD_SLOT_MIX_BY_TIER[str(tier)]["related_support"])),
                    "story_specific": float(row.get("story_specific", _DEFAULT_KEYWORD_SLOT_MIX_BY_TIER[str(tier)]["story_specific"])),
                    "franchise": float(row.get("franchise", _DEFAULT_KEYWORD_SLOT_MIX_BY_TIER[str(tier)]["franchise"])),
                    "generic": float(row.get("generic", _DEFAULT_KEYWORD_SLOT_MIX_BY_TIER[str(tier)]["generic"])),
                }
    for tier in PRODUCTION_TIERS:
        row = dict(slot_mix_cfg.get(str(tier), _DEFAULT_KEYWORD_SLOT_MIX_BY_TIER[str(tier)]))
        total = float(sum(max(0.0, float(v)) for v in row.values())) or 1.0
        cfg.setdefault("slot_mix_by_tier", {})
        cfg["slot_mix_by_tier"][str(tier)] = {str(k): max(0.0, float(v)) / total for k, v in row.items()}
    raw_related = _section_prior_dict(world, "selection_weights", "keyword_selection").get("related_genres_by_genre")
    related_cfg: dict[str, list[str]] = {}
    if isinstance(raw_related, dict):
        for genre in GENRES:
            values = raw_related.get(str(genre))
            if isinstance(values, list):
                related_cfg[str(genre)] = [str(item) for item in values if str(item) in GENRES and str(item) != str(genre)]
    cfg["related_genres_by_genre"] = {
        str(genre): list(dict.fromkeys(related_cfg.get(str(genre), _DEFAULT_RELATED_GENRES_BY_GENRE.get(str(genre), []))))
        for genre in GENRES
    }
    return cfg


def _geo_boost_for_tier(world: WorldState, tier: str, default: float = 2.0) -> float:
    cfg = _cast_selection_config(world)
    return float(cfg["geo_boost_by_tier"].get(str(tier), float(default)))


def _safe01(value, default: float = 0.5) -> float:
    try:
        val = float(value)
    except Exception:
        val = float(default)
    if val != val:
        val = float(default)
    return float(max(0.0, min(1.0, val)))


def _normalise_dict_weights(weights: dict[str, float], floor: float = 1e-6) -> dict[str, float]:
    if not weights:
        return {}
    arr = {k: max(float(floor), float(v)) for k, v in weights.items()}
    s = sum(arr.values())
    if s <= 0:
        u = 1.0 / max(1, len(arr))
        return {k: u for k in arr}
    return {k: v / s for k, v in arr.items()}


def _cosine_sim(a, b) -> float:
    a_np = np.asarray(a, dtype=np.float32)
    b_np = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a_np) * np.linalg.norm(b_np))
    if denom < 1e-10:
        return 0.0
    return float(max(0.0, min(1.0, float(np.dot(a_np, b_np)) / denom)))


def _concept_csv_target(concept: dict) -> list[float]:
    world = concept.get("_world")
    genre = concept.get("genre", "Drama")
    tier = concept.get("tier", "Mid")
    genre_map = {
        "Action":      [0.3, 0.8, 0.4, 0.9, 0.3, 0.8, 0.7, 0.8],
        "Drama":       [0.6, 0.2, 0.7, 0.2, 0.5, 0.3, 0.3, 0.2],
        "Comedy":      [0.5, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4],
        "Sci-Fi":      [0.4, 0.7, 0.3, 0.6, 0.4, 0.6, 0.7, 0.7],
        "Horror":      [0.3, 0.3, 0.4, 0.7, 0.3, 0.8, 0.4, 0.3],
        "Romance":     [0.7, 0.2, 0.6, 0.2, 0.8, 0.3, 0.3, 0.2],
        "Thriller":    [0.3, 0.4, 0.5, 0.6, 0.7, 0.6, 0.4, 0.4],
        "Fantasy":     [0.5, 0.8, 0.2, 0.5, 0.4, 0.5, 0.8, 0.8],
        "Animation":   [0.6, 0.6, 0.2, 0.5, 0.5, 0.4, 0.7, 0.6],
        "Documentary": [0.4, 0.1, 0.9, 0.1, 0.3, 0.6, 0.1, 0.1],
        "Crime":       [0.4, 0.4, 0.6, 0.5, 0.6, 0.5, 0.4, 0.4],
        "Mystery":     [0.5, 0.3, 0.5, 0.3, 0.6, 0.7, 0.3, 0.3],
        "War":         [0.3, 0.7, 0.5, 0.8, 0.3, 0.7, 0.6, 0.7],
    }
    tier_shift_scalar: dict[str, float] = {"Epic": 0.15, "A": 0.08, "Mid": 0.0, "Indie": -0.10, "Micro": -0.15}
    tier_shift_vector: dict[str, list[float]] = {}
    if isinstance(world, WorldState):
        genre_vector_priors = _section_prior_dict(world, "selection_weights", "concept_style_vector_by_genre")
        raw_vector = genre_vector_priors.get(str(genre))
        if isinstance(raw_vector, (list, tuple)) and len(raw_vector) == 8:
            try:
                genre_map[str(genre)] = [float(v) for v in raw_vector]
            except Exception:
                pass
        elif current_mode() == "research":
            audit_fallback_hit(
                "assembly.selection_weights",
                f"missing:concept_style_vector_by_genre.{genre}",
                detail=f"selection_weights.concept_style_vector_by_genre.{genre} is required in research mode",
                mode="research",
            )
        for k, v in _section_prior_dict(world, "selection_weights", "concept_style_tier_shift_by_tier").items():
            if not k:
                continue
            key = str(k)
            if isinstance(v, (list, tuple)) and len(v) == 8:
                try:
                    tier_shift_vector[key] = [float(x) for x in v]
                    continue
                except Exception:
                    pass
            try:
                tier_shift_scalar[key] = float(v)
            except Exception:
                continue
        if current_mode() == "research" and str(tier) not in tier_shift_scalar and str(tier) not in tier_shift_vector:
            audit_fallback_hit(
                "assembly.selection_weights",
                f"missing:concept_style_tier_shift_by_tier.{tier}",
                detail=f"selection_weights.concept_style_tier_shift_by_tier.{tier} is required in research mode",
                mode="research",
            )
    base = list(genre_map.get(genre, [0.5] * 8))
    vector_shift = tier_shift_vector.get(tier)
    if vector_shift and len(vector_shift) == 8:
        for dim, delta in enumerate(vector_shift):
            base[dim] = max(-1.0, min(1.0, base[dim] + float(delta)))
    else:
        shift = tier_shift_scalar.get(tier, 0.0)
        for dim in (1, 6, 7):
            base[dim] = max(-1.0, min(1.0, base[dim] + shift))
    return base


def _concept_latent_targets(concept: dict) -> tuple[float, float, float]:
    world = concept.get("_world")
    genre = str(concept.get("genre", "Drama"))
    tier = str(concept.get("tier", "Mid"))
    risk_map = {
        "Action": 0.68, "Thriller": 0.62, "Crime": 0.57, "Sci-Fi": 0.63,
        "Fantasy": 0.58, "Horror": 0.74, "Mystery": 0.53, "Comedy": 0.47,
        "Animation": 0.46, "Drama": 0.36, "Romance": 0.32, "Documentary": 0.24,
        "War": 0.59,
    }
    ambition_map = {"Epic": 0.72, "A": 0.64, "Mid": 0.55, "Indie": 0.70, "Micro": 0.66}
    prestige_map = {"Epic": 0.78, "A": 0.68, "Mid": 0.57, "Indie": 0.52, "Micro": 0.42}
    if isinstance(world, WorldState):
        risk_map = {**risk_map, **{str(k): float(v) for k, v in _section_prior_dict(world, "selection_weights", "concept_risk_target_by_genre").items() if k}}
        ambition_map = {**ambition_map, **{str(k): float(v) for k, v in _section_prior_dict(world, "selection_weights", "concept_ambition_target_by_tier").items() if k}}
        prestige_map = {**prestige_map, **{str(k): float(v) for k, v in _section_prior_dict(world, "selection_weights", "concept_prestige_target_by_tier").items() if k}}
        if current_mode() == "research":
            if genre not in risk_map:
                audit_fallback_hit(
                    "assembly.selection_weights",
                    f"missing:concept_risk_target_by_genre.{genre}",
                    detail=f"selection_weights.concept_risk_target_by_genre.{genre} is required in research mode",
                    mode="research",
                )
            if tier not in ambition_map:
                audit_fallback_hit(
                    "assembly.selection_weights",
                    f"missing:concept_ambition_target_by_tier.{tier}",
                    detail=f"selection_weights.concept_ambition_target_by_tier.{tier} is required in research mode",
                    mode="research",
                )
            if tier not in prestige_map:
                audit_fallback_hit(
                    "assembly.selection_weights",
                    f"missing:concept_prestige_target_by_tier.{tier}",
                    detail=f"selection_weights.concept_prestige_target_by_tier.{tier} is required in research mode",
                    mode="research",
                )
    risk_target = risk_map.get(genre, 0.50)
    ambition_target = ambition_map.get(tier, 0.56)
    prestige_target = prestige_map.get(tier, 0.56)
    return float(risk_target), float(ambition_target), float(prestige_target)


def _shortlist_budget(world: WorldState, kind: str, default: int) -> int:
    priors = _workspace_priors(world)
    base = int(default)
    if priors is not None:
        try:
            base = max(8, int(getattr(priors, "shortlist_size", default)))
        except Exception:
            base = int(default)
    if kind == "director":
        return max(12, min(64, max(12, base // 2)))
    if kind == "company":
        return max(10, min(40, max(10, base // 3)))
    if kind == "crew":
        return max(12, min(48, max(12, base // 2)))
    return max(8, int(base))


def _shortlist_indices(weights: np.ndarray, shortlist_size: int, rng, exploration_share: float = 0.25) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    valid = np.flatnonzero(w > 0)
    if valid.size <= shortlist_size:
        return valid
    shortlist_size = max(1, int(shortlist_size))
    exploration_share = float(np.clip(exploration_share, 0.0, 0.60))
    anchor_size = min(valid.size, max(1, int(round(shortlist_size * (1.0 - exploration_share)))))
    valid_weights = w[valid]
    anchor_local = np.argpartition(valid_weights, -anchor_size)[-anchor_size:]
    anchor = valid[anchor_local]
    anchor = anchor[np.argsort(w[anchor])[::-1]]
    explore_size = shortlist_size - anchor.size
    if explore_size <= 0:
        return anchor[:shortlist_size]
    remaining = np.setdiff1d(valid, anchor, assume_unique=False)
    if remaining.size == 0:
        return anchor[:shortlist_size]
    deep_share = _env_float("DATA_SYS_SHORTLIST_DEEP_EXPLORATION_SHARE", 0.35, 0.0, 0.80)
    deep_size = min(remaining.size, int(round(explore_size * deep_share)))
    band_explore_size = max(0, explore_size - deep_size)
    band_size = min(remaining.size, max(band_explore_size * 4, shortlist_size))
    band_local = np.argpartition(w[remaining], -band_size)[-band_size:]
    band = remaining[band_local]
    picks: list[np.ndarray] = []
    if band_explore_size > 0 and band.size > 0:
        band_weights = normalize_weights(w[band])
        picks.append(rng.choice(band, size=min(band_explore_size, band.size), replace=False, p=band_weights))
    if deep_size > 0:
        deep_pool = np.setdiff1d(remaining, band, assume_unique=False)
        if deep_pool.size == 0:
            deep_pool = remaining
        deep_weights = normalize_weights(np.sqrt(np.maximum(w[deep_pool], 0.0)) + 1e-9)
        picks.append(rng.choice(deep_pool, size=min(deep_size, deep_pool.size), replace=False, p=deep_weights))
    explore = np.unique(np.concatenate(picks)) if picks else np.zeros(0, dtype=np.int32)
    merged = np.unique(np.concatenate([anchor, explore]))
    merged = merged[np.argsort(w[merged])[::-1]]
    return merged[:shortlist_size]


def _active_year_subset(df: pd.DataFrame, year: int) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    if "debut_year" in df.columns:
        debut_ok = df["debut_year"].fillna(1900).astype(int) <= int(year)
    else:
        debut_ok = pd.Series(True, index=df.index)
    if "retirement_year" in df.columns:
        retire_ok = df["retirement_year"].fillna(2100).astype(float) >= float(year)
    else:
        retire_ok = pd.Series(True, index=df.index)
    out = df[debut_ok & retire_ok]
    return out if len(out) > 0 else df


def _edge_payload_weight(entry) -> float:
    if isinstance(entry, dict):
        return float(entry.get("weight", 0.0) or 0.0)
    try:
        return float(entry)
    except Exception:
        return 0.0


def _ensure_company_lookup_cache(world: WorldState) -> None:
    if getattr(world, "_company_by_tier_genre", None) is not None:
        return
    world._company_by_tier_genre = {}
    if world.companies is None or len(world.companies) == 0:
        return
    has_tier = "tier" in world.companies.columns
    has_genre = "specialty_genres" in world.companies.columns
    cids = world.companies["company_id"].astype(int).values
    tiers = world.companies["tier"].astype(str).values if has_tier else np.full(len(cids), "")
    genres = world.companies["specialty_genres"].fillna("").astype(str).values if has_genre else np.full(len(cids), "")
    for cid, tier, gspec in zip(cids, tiers, genres):
        pieces = [g.strip().lower() for g in str(gspec).replace(",", ";").split(";") if g.strip()]
        world._company_by_tier_genre.setdefault((str(tier), ""), set()).add(int(cid))
        for g in pieces:
            world._company_by_tier_genre.setdefault((str(tier), g), set()).add(int(cid))
            world._company_by_tier_genre.setdefault(("", g), set()).add(int(cid))


def _ensure_actor_workload_counter(world: WorldState) -> None:
    if getattr(world, "_yearly_workload", None) is not None:
        return
    world._yearly_workload = Counter()
    for pid, years in getattr(world, "person_recent", {}).items():
        for y in years:
            world._yearly_workload[(int(pid), int(y))] += 1


def _bounded_positions(weights: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if limit <= 0 or len(weights) == 0:
        return np.array([], dtype=np.int32)
    if len(weights) <= limit:
        return np.arange(len(weights), dtype=np.int32)
    top_count = max(1, min(limit, int(round(limit * 0.70))))
    if top_count >= len(weights):
        top_idx = np.arange(len(weights), dtype=np.int32)
    else:
        top_idx = np.argpartition(weights, -top_count)[-top_count:].astype(np.int32, copy=False)
    order = np.argsort(weights[top_idx])[::-1]
    top_idx = top_idx[order].astype(np.int32, copy=False)
    extra = max(0, limit - len(top_idx))
    if extra <= 0:
        return top_idx[:limit].astype(np.int32, copy=False)
    mask = np.ones(len(weights), dtype=bool)
    mask[top_idx] = False
    remaining = np.flatnonzero(mask)
    if remaining.size == 0:
        return top_idx[:limit].astype(np.int32, copy=False)
    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
    explore = rng.choice(remaining, size=min(extra, remaining.size), replace=False)
    return np.concatenate([top_idx, np.asarray(explore, dtype=np.int32)])[:limit].astype(np.int32, copy=False)


def _recent_window_count(world: WorldState, pid: int, year: int) -> float:
    _ensure_actor_workload_counter(world)
    workload = getattr(world, "_yearly_workload", None) or {}
    total = 0
    for offset in (-2, -1, 0, 1, 2):
        total += int(workload.get((int(pid), int(year + offset)), 0))
    return float(total)


def _bounded_neighbor_rows(
    world: WorldState,
    edge_type: str,
    pid: int,
    year: int,
    *,
    limit: int,
    seed: int,
) -> list[tuple[int, float, int, int]]:
    if limit <= 0:
        return []
    graph = getattr(world, "graph", None)
    if graph is not None and hasattr(graph, "sample_bounded_neighbors"):
        try:
            rows = graph.sample_bounded_neighbors(edge_type, int(pid), year, limit=int(limit), seed=int(seed))
            if rows:
                return rows
        except Exception:
            pass

    fallback = world._friend_adj_all.get(int(pid), []) if edge_type == "friendship" else world._rival_adj_all.get(int(pid), [])
    if len(fallback) <= limit:
        return list(fallback)
    weights = np.asarray([float(row[1]) for row in fallback], dtype=np.float32)
    chosen = _bounded_positions(weights, int(limit), int(seed))
    return [fallback[int(idx)] for idx in chosen]


@dataclass(slots=True)
class CastYearCache:
    year: int
    candidates: pd.DataFrame
    pids: np.ndarray
    pop_weight: np.ndarray
    peak_start: np.ndarray
    peak_end: np.ndarray
    stage_vals: np.ndarray
    career_stage_mult: np.ndarray
    yearly_max: np.ndarray
    actor_genders: np.ndarray | None
    actor_nationalities: np.ndarray | None
    nat_country: np.ndarray | None
    ga_lower: np.ndarray | None
    st_lower: np.ndarray | None
    market_fit_lower: np.ndarray | None
    actor_tag_sets: list[set[str]] | None
    actor_tag_bitmasks: np.ndarray | None
    li_arr: np.ndarray | None
    valid_li: np.ndarray | None
    cand_agencies: np.ndarray
    cand_communities: np.ndarray
    budget_pref: np.ndarray | None
    genre_match_cache: dict[str, np.ndarray] = field(default_factory=dict)
    style_match_cache: dict[tuple[str, ...], np.ndarray] = field(default_factory=dict)
    market_match_cache: dict[str, np.ndarray] = field(default_factory=dict)
    avoid_genre_cache: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(slots=True)
class CrewYearPool:
    year: int
    role: str
    person_ids: np.ndarray
    weights: np.ndarray
    genre_affinity_lower: np.ndarray | None
    pid_to_local: dict[int, int]
    genre_match_cache: dict[str, np.ndarray] = field(default_factory=dict)
    genre_weight_cache: dict[str, np.ndarray] = field(default_factory=dict)
    genre_band_cache: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(slots=True)
class SelectionYearState:
    year: int
    actor_cache: CastYearCache
    pid_to_local: dict[int, int]
    film_count: np.ndarray
    recent_window: np.ndarray
    yearly_workload: np.ndarray
    unused_flags: np.ndarray
    award_recent: np.ndarray
    actor_views: dict[tuple[Any, ...], "ActorStaticBlock"] = field(default_factory=dict)
    actor_pc_affinity_cache: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    director_pc_affinity_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def get_actor_view(self, concept_key: tuple[Any, ...]) -> "ActorStaticBlock" | None:
        return self.actor_views.get(tuple(concept_key))

    def record_cast_selection(self, actor_ids: Iterable[int], year: int) -> None:
        for pid in actor_ids:
            local_idx = self.pid_to_local.get(int(pid))
            if local_idx is None:
                continue
            self.film_count[local_idx] += 1.0
            self.yearly_workload[local_idx] += 1.0
            self.recent_window[local_idx] += 1.0
            self.unused_flags[local_idx] = False


@dataclass(slots=True)
class CastEnsembleState:
    cast_ids: list[int] = field(default_factory=list)
    cast_set: set[int] = field(default_factory=set)
    agencies: set[int] = field(default_factory=set)
    communities: set[int] = field(default_factory=set)
    genders: set[str] = field(default_factory=set)
    nationalities: set[str] = field(default_factory=set)
    tag_bitmasks: list[np.uint64] = field(default_factory=list)
    latent_anchor_ids: list[int] = field(default_factory=list)
    friend_frontier: dict[int, float] = field(default_factory=dict)
    rival_penalty: dict[int, float] = field(default_factory=dict)
    friend_frontier_vec: np.ndarray | None = None
    rival_penalty_vec: np.ndarray | None = None
    blocked_local_mask: np.ndarray | None = None
    frontier_local_idx: set[int] = field(default_factory=set)
    rival_local_idx: set[int] = field(default_factory=set)
    blocked_local_idx: set[int] = field(default_factory=set)
    forbidden: set[int] = field(default_factory=set)

    def add_actor(self, world: WorldState, static: "ActorStaticBlock", local_idx: int, year: int) -> None:
        pid = int(static.pids[local_idx])
        if pid in self.cast_set:
            return
        self.cast_ids.append(pid)
        self.cast_set.add(pid)
        self.blocked_local_idx.add(int(local_idx))
        agency = int(static.cand_agencies[local_idx]) if len(static.cand_agencies) > local_idx else -1
        if agency >= 0:
            self.agencies.add(agency)
        community = int(static.cand_communities[local_idx]) if len(static.cand_communities) > local_idx else -1
        if community >= 0:
            self.communities.add(community)
        if static.actor_genders is not None:
            self.genders.add(str(static.actor_genders[local_idx]))
        if static.actor_nationalities is not None:
            self.nationalities.add(str(static.actor_nationalities[local_idx]))
        if static.actor_tag_bitmasks is not None:
            self.tag_bitmasks.append(np.uint64(static.actor_tag_bitmasks[local_idx]))
        if static.li_arr is not None:
            li = int(static.li_arr[local_idx])
            if li >= 0:
                self.latent_anchor_ids.append(li)
        if self.blocked_local_mask is not None and 0 <= local_idx < len(self.blocked_local_mask):
            self.blocked_local_mask[local_idx] = True

        friend_limit = max(24, min(96, max(24, len(static.pids) // 192)))
        rival_limit = max(16, min(72, max(16, len(static.pids) // 256)))
        base_seed = (
            int(getattr(world, "seed", 0)) * 1_315_423_911
            + int(year) * 2_654_435_761
            + int(pid) * 97_531
            + len(self.cast_ids) * 17
        ) & 0xFFFFFFFF
        friend_iter = _bounded_neighbor_rows(world, "friendship", pid, year, limit=friend_limit, seed=base_seed)
        rival_iter = _bounded_neighbor_rows(world, "rivalry", pid, year, limit=rival_limit, seed=base_seed ^ 0x9E3779B9)
        for nbr, weight, _vf, _vt in friend_iter:
            nbr_idx = static.pid_to_idx.get(int(nbr))
            if nbr_idx is None or int(nbr) in self.cast_set:
                continue
            boost = float(1.0 + 4.0 * max(0.15, float(weight)))
            self.friend_frontier[int(nbr_idx)] = max(
                float(self.friend_frontier.get(int(nbr_idx), 0.0)),
                boost,
            )
            if self.friend_frontier_vec is not None:
                self.friend_frontier_vec[int(nbr_idx)] = max(
                    float(self.friend_frontier_vec[int(nbr_idx)]),
                    boost,
                )
            self.frontier_local_idx.add(int(nbr_idx))
        for nbr, weight, _vf, _vt in rival_iter:
            nbr_idx = static.pid_to_idx.get(int(nbr))
            if nbr_idx is None or int(nbr) in self.cast_set:
                continue
            penalty = max(0.0, 1.0 - max(0.6, float(weight)))
            current = float(self.rival_penalty.get(int(nbr_idx), 1.0))
            self.rival_penalty[int(nbr_idx)] = min(current, penalty)
            if self.rival_penalty_vec is not None:
                self.rival_penalty_vec[int(nbr_idx)] = min(
                    float(self.rival_penalty_vec[int(nbr_idx)]),
                    penalty,
                )
            if self.blocked_local_mask is not None:
                self.blocked_local_mask[int(nbr_idx)] = True
            self.rival_local_idx.add(int(nbr_idx))
            self.blocked_local_idx.add(int(nbr_idx))
            self.forbidden.add(int(nbr))


def _build_year_mask(df: pd.DataFrame, year: int) -> np.ndarray:
    debut_arr = pd.to_numeric(df.get("debut_year", 1900), errors="coerce").fillna(1900).astype(int).to_numpy()
    retire_arr = pd.to_numeric(df.get("retirement_year", 2100), errors="coerce").fillna(2100).astype(float).to_numpy()
    mask = (debut_arr <= year) & (retire_arr >= float(year))
    if mask.sum() < 10:
        mask = np.ones(len(df), dtype=bool)
    return mask


def _get_cast_year_cache(world: WorldState, year: int) -> CastYearCache:
    cache_map = getattr(world, "_cast_year_cache", None)
    if cache_map is None:
        cache_map = {}
        world._cast_year_cache = cache_map
    cached = cache_map.get(int(year))
    if cached is not None:
        return cached

    if year in getattr(world, "_year_cache", {}):
        mask = world._year_cache[year]
    else:
        mask = _build_year_mask(world.actors, year)
        world._year_cache[year] = mask

    candidates = world.actors.loc[mask].reset_index(drop=True)
    pids = candidates["person_id"].astype(int).to_numpy()
    if "peak_start" in candidates.columns:
        peak_start = pd.to_numeric(candidates["peak_start"], errors="coerce").fillna(1900).astype(int).to_numpy()
    else:
        peak_start = np.full(len(candidates), 1900, dtype=int)
    if "peak_end" in candidates.columns:
        peak_end = pd.to_numeric(candidates["peak_end"], errors="coerce").fillna(1901).astype(int).to_numpy()
    else:
        peak_end = np.full(len(candidates), 1901, dtype=int)
    stage_vals = candidates["career_stage"].fillna("prime").astype(str).str.lower().to_numpy(dtype=str) if "career_stage" in candidates.columns else np.full(len(candidates), "prime", dtype=str)
    stage_map = {
        "legend": _env_float("V16_LEGEND_MULT", 15.0, 4.0, 30.0),
        "prime": _env_float("V16_PRIME_MULT", 5.0, 1.0, 12.0),
        "veteran": _env_float("V16_VETERAN_MULT", 2.5, 0.5, 10.0),
        "rising": _env_float("V16_RISING_MULT", 1.0, 0.2, 4.0),
        "retired": _env_float("V16_RETIRED_MULT", 0.08, 0.01, 1.5),
    }
    career_stage_mult = np.array([stage_map.get(str(s), 1.0) for s in stage_vals], dtype=float)
    yearly_max = candidates["yearly_max"].to_numpy(dtype=float) if "yearly_max" in candidates.columns else np.full(len(candidates), 5.0, dtype=float)
    yearly_max = np.where(np.isnan(yearly_max) | (yearly_max <= 0), 5.0, yearly_max)
    actor_genders = candidates["gender"].fillna("unknown").astype(str).str.lower().to_numpy(dtype=str) if "gender" in candidates.columns else None
    actor_nationalities = candidates["nationality"].fillna("unknown").astype(str).str.lower().to_numpy(dtype=str) if "nationality" in candidates.columns else None
    nat_country = None
    if "nationality" in candidates.columns:
        nat_country = np.array([_NATIONALITY_TO_COUNTRY.get(v, "") for v in candidates["nationality"].fillna("").astype(str).to_numpy(dtype=str)], dtype=str)
    if "_ga_lower" in candidates.columns:
        ga_lower = candidates["_ga_lower"].fillna("").astype(str).to_numpy(dtype=str)
    elif "genre_affinity" in candidates.columns:
        ga_lower = candidates["genre_affinity"].fillna("").astype(str).str.lower().to_numpy(dtype=str)
    else:
        ga_lower = None
    if "_st_lower" in candidates.columns:
        st_lower = candidates["_st_lower"].fillna("").astype(str).to_numpy(dtype=str)
    elif "style_tags" in candidates.columns:
        st_lower = candidates["style_tags"].fillna("").astype(str).str.lower().to_numpy(dtype=str)
    else:
        st_lower = None
    market_fit_lower = candidates["market_fit"].fillna("").astype(str).str.lower().to_numpy(dtype=str) if "market_fit" in candidates.columns else None
    actor_tag_sets = [set(t.strip().lower() for t in raw.replace(",", ";").split(";") if t.strip()) for raw in st_lower] if st_lower is not None else None
    tag_bit_map = _ensure_tag_bit_mapping(world)
    actor_tag_bitmasks = _build_tag_bitmasks(actor_tag_sets, tag_bit_map)
    latent_idx = getattr(world, "_latent_pid_to_idx", None)
    li_arr = np.array([latent_idx.get(int(pid), -1) for pid in pids], dtype=int) if latent_idx is not None else None
    valid_li = li_arr >= 0 if li_arr is not None else None
    cand_agencies = np.array([world.person_agency.get(int(pid), -1) for pid in pids], dtype=np.int32)
    communities = getattr(world, "communities", None) or {}
    cand_communities = np.array([communities.get(int(pid), -1) for pid in pids], dtype=np.int32)

    budget_pref = None
    if li_arr is not None and getattr(world, "_latent_bbp_normed", None) is not None:
        safe = np.clip(li_arr, 0, len(world._latent_bbp_normed) - 1)
        budget_pref = world._latent_bbp_normed[safe].copy()
        budget_pref[~valid_li] = 0.0

    cached = CastYearCache(
        year=int(year),
        candidates=candidates,
        pids=pids,
        pop_weight=candidates["pop_weight"].astype(float).to_numpy().copy(),
        peak_start=peak_start,
        peak_end=peak_end,
        stage_vals=stage_vals,
        career_stage_mult=career_stage_mult,
        yearly_max=yearly_max,
        actor_genders=actor_genders,
        actor_nationalities=actor_nationalities,
        nat_country=nat_country,
        ga_lower=ga_lower,
        st_lower=st_lower,
        market_fit_lower=market_fit_lower,
        actor_tag_sets=actor_tag_sets,
        actor_tag_bitmasks=actor_tag_bitmasks,
        li_arr=li_arr,
        valid_li=valid_li,
        cand_agencies=cand_agencies,
        cand_communities=cand_communities,
        budget_pref=budget_pref,
    )
    cache_map[int(year)] = cached
    return cached


def _award_recent_pid_set(world: WorldState, year: int) -> set[int]:
    award_recent_pids: set[int] = set()
    if getattr(world, "person_award_wins", None):
        for raw_pid, award_info in world.person_award_wins.items():
            pid = int(raw_pid)
            if isinstance(award_info, dict):
                if year - int(award_info.get("year", 0) or 0) <= 3:
                    award_recent_pids.add(pid)
            elif isinstance(award_info, int) and award_info > 0:
                award_recent_pids.add(pid)
    return award_recent_pids


def _get_selection_year_state(world: WorldState, year: int) -> SelectionYearState:
    cache_map = getattr(world, "_selection_year_state_cache", None)
    if cache_map is None:
        cache_map = {}
        world._selection_year_state_cache = cache_map
    cached = cache_map.get(int(year))
    if cached is not None:
        return cached

    actor_cache = _get_cast_year_cache(world, year)
    _ensure_actor_workload_counter(world)
    pids = actor_cache.pids
    recent_window = np.fromiter(
        (_recent_window_count(world, int(pid), int(year)) for pid in pids),
        dtype=float,
        count=len(pids),
    )
    cached = SelectionYearState(
        year=int(year),
        actor_cache=actor_cache,
        pid_to_local={int(pid): idx for idx, pid in enumerate(pids)},
        film_count=np.fromiter((world.person_film_count.get(int(pid), 0) for pid in pids), dtype=float, count=len(pids)),
        recent_window=recent_window,
        yearly_workload=np.fromiter((world._yearly_workload.get((int(pid), int(year)), 0) for pid in pids), dtype=float, count=len(pids)),
        unused_flags=(recent_window <= 0).astype(bool),
        award_recent=(
            np.isin(pids, np.array(sorted(_award_recent_pid_set(world, year)), dtype=int))
            if len(pids) else np.zeros(0, dtype=bool)
        ),
    )
    cache_map[int(year)] = cached
    return cached


def _get_crew_year_pool(world: WorldState, role: str, year: int) -> CrewYearPool | None:
    cache_map = getattr(world, "_crew_year_pool_cache", None)
    if cache_map is None:
        cache_map = {}
        world._crew_year_pool_cache = cache_map
    cache_key = (int(year), str(role))
    cached = cache_map.get(cache_key)
    if cached is not None:
        return cached

    pool = getattr(world, "crew_pools", {}).get(role) if hasattr(world, "crew_pools") else None
    if pool is None or len(pool) == 0:
        fallback_attr = CREW_DEPARTMENTS.get(role, {}).get("pool_fallback", "persons")
        pool = getattr(world, fallback_attr, None)
        if pool is None or len(pool) == 0:
            pool = world.actors
    if pool is None or len(pool) == 0:
        return None

    mask = _build_year_mask(pool, year)
    active = pool.loc[mask].reset_index(drop=True)
    if len(active) == 0:
        return None
    weights = active.get("pop_weight", pd.Series(np.ones(len(active), dtype=float))).astype(float).to_numpy()
    genre_affinity_lower = active["genre_affinity"].fillna("").astype(str).str.lower().to_numpy() if "genre_affinity" in active.columns else None
    cached = CrewYearPool(
        year=int(year),
        role=str(role),
        person_ids=active["person_id"].astype(int).to_numpy(),
        weights=weights,
        genre_affinity_lower=genre_affinity_lower,
        pid_to_local={int(pid): idx for idx, pid in enumerate(active["person_id"].astype(int).to_numpy())},
    )
    cache_map[cache_key] = cached
    return cached


def _crew_genre_match_mask(pool: CrewYearPool, genre: str) -> np.ndarray:
    genre_key = str(genre or "").lower().strip()
    if not genre_key or pool.genre_affinity_lower is None:
        return np.zeros(len(pool.person_ids), dtype=bool)
    cached = pool.genre_match_cache.get(genre_key)
    if cached is not None:
        return cached
    mask = np.char.find(_to_str_ndarray(pool.genre_affinity_lower), genre_key) >= 0
    pool.genre_match_cache[genre_key] = mask
    return mask


def _crew_genre_weights(pool: CrewYearPool, genre: str) -> np.ndarray:
    genre_key = str(genre or "").lower().strip()
    if not genre_key:
        return pool.weights
    cached = pool.genre_weight_cache.get(genre_key)
    if cached is not None:
        return cached
    weights = pool.weights.copy()
    if pool.genre_affinity_lower is not None:
        weights *= (1.0 + 0.60 * _crew_genre_match_mask(pool, genre_key).astype(float))
    pool.genre_weight_cache[genre_key] = weights
    return weights


def _crew_candidate_band(world: WorldState, pool: CrewYearPool, genre: str, target_n: int) -> np.ndarray:
    genre_key = str(genre or "").lower().strip()
    cached = pool.genre_band_cache.get(genre_key)
    if cached is not None:
        return cached
    weights = _crew_genre_weights(pool, genre_key)
    shortlist = _shortlist_budget(world, "crew", max(32, int(target_n) * 8))
    band_size = min(len(pool.person_ids), max(shortlist * 6, 96))
    band = _shortlist_indices(
        weights,
        band_size,
        world.rng,
        exploration_share=_prior_float(world, "crew_exploration_share", 0.35, lo=0.0, hi=0.60),
    )
    if band.size == 0:
        band = np.flatnonzero(weights > 0)
    pool.genre_band_cache[genre_key] = band.astype(np.int32, copy=False)
    return pool.genre_band_cache[genre_key]


def _person_company_multiplier(world: WorldState, pids: np.ndarray, year: int, tier: str, genre: str) -> np.ndarray:
    # Cache P-C scoring once per actor year universe x (tier, genre).
    _ensure_company_lookup_cache(world)
    suitable = (
        world._company_by_tier_genre.get((tier, genre.lower()), set())
        | world._company_by_tier_genre.get(("", genre.lower()), set())
        | world._company_by_tier_genre.get((tier, ""), set())
    )
    if not suitable:
        return np.ones(len(pids), dtype=float)
    selection = _get_selection_year_state(world, year)
    cache_key = (str(tier), str(genre).lower())
    cached = selection.actor_pc_affinity_cache.get(cache_key)
    if cached is None or len(cached) != len(selection.actor_cache.pids):
        cached = world.compute_pc_affinity_batch(selection.actor_cache.pids, suitable)
        selection.actor_pc_affinity_cache[cache_key] = cached
    if np.array_equal(pids, selection.actor_cache.pids):
        return cached
    take = np.array([selection.pid_to_local.get(int(pid), -1) for pid in pids], dtype=int)
    out = np.ones(len(pids), dtype=float)
    valid = take >= 0
    out[valid] = cached[take[valid]]
    return out


def _director_company_multiplier(world: WorldState, dir_pids: np.ndarray, year: int, tier: str, genre: str) -> np.ndarray:
    _ensure_company_lookup_cache(world)
    suitable = (
        world._company_by_tier_genre.get((tier, genre.lower()), set())
        | world._company_by_tier_genre.get(("", genre.lower()), set())
        | world._company_by_tier_genre.get((tier, ""), set())
    )
    if not suitable:
        return np.ones(len(dir_pids), dtype=float)
    selection = _get_selection_year_state(world, year)
    cache_key = (str(tier), str(genre).lower())
    cached = selection.director_pc_affinity_cache.get(cache_key)
    if cached is None or len(cached[0]) != len(dir_pids) or not np.array_equal(cached[0], dir_pids):
        cached = (dir_pids.copy(), world.compute_pc_affinity_batch(dir_pids, suitable))
        selection.director_pc_affinity_cache[cache_key] = cached
    return cached[1]


def _director_edge_arrays(world: WorldState, director_id: int, pids: np.ndarray, year: int) -> tuple[np.ndarray, np.ndarray]:
    cache_map = getattr(world, "_director_year_edge_cache", None)
    if cache_map is None:
        cache_map = {}
        world._director_year_edge_cache = cache_map
    cache_key = (int(year), int(director_id))
    cached = cache_map.get(cache_key)
    if cached is None:
        pref_map: dict[int, float] = {}
        avoid_set: set[int] = set()
        graph = getattr(world, "graph", None)
        pref_edges = graph.get_director_prefs(int(director_id), year) if graph is not None else []
        for aid, weight, _valid_from, _valid_to in pref_edges:
            pref_map[int(aid)] = float(weight)
        avoid_edges = graph.get_director_avoids(int(director_id), year) if graph is not None else []
        for aid, _valid_from, _valid_to in avoid_edges:
            avoid_set.add(int(aid))
        cached = (pref_map, avoid_set)
        cache_map[cache_key] = cached
    else:
        pref_map, avoid_set = cached

    pref = np.ones(len(pids), dtype=float)
    if pref_map:
        arr = np.array([pref_map.get(int(pid), -1.0) for pid in pids], dtype=float)
        mask = arr >= 0.0
        arr = np.clip(arr, 0.0, 1.0)
        pref[mask] = 1.0 + 9.0 * arr[mask]

    avoid_mask = np.isin(pids, np.array(list(avoid_set), dtype=int)) if avoid_set else np.zeros(len(pids), dtype=bool)
    return pref, avoid_mask


@dataclass(slots=True)
class ActorStaticBlock:
    key: tuple[Any, ...]
    candidate_idx: np.ndarray
    pids: np.ndarray
    immutable_scores: np.ndarray
    base_focus_idx: np.ndarray
    yearly_max: np.ndarray
    stage_vals: np.ndarray
    cooldown_floor: np.ndarray
    cooldown_decay: np.ndarray
    li_arr: np.ndarray | None
    actor_genders: np.ndarray | None
    actor_nationalities: np.ndarray | None
    actor_tag_sets: list[set[str]] | None
    actor_tag_bitmasks: np.ndarray | None  # uint64 bitmask per candidate
    pid_to_idx: dict[int, int]
    top_star_mask: np.ndarray
    cand_agencies: np.ndarray
    cand_communities: np.ndarray
    _sim_threshold: float
    policy_rule_mult: np.ndarray | None = None


ActorConceptView = ActorStaticBlock


# ---------------------------------------------------------------------------
# Tag bitmask helpers (vectorized Jaccard over uint64 arrays)
# ---------------------------------------------------------------------------

def _popcount_vec(arr: np.ndarray) -> np.ndarray:
    """Vectorized popcount for a uint64 numpy array (bit-parallel)."""
    x = arr.astype(np.uint64)
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int32)


def _ensure_tag_bit_mapping(world: WorldState) -> dict[str, int]:
    """Build a tag → bit-position mapping (cached on world, O(N) once)."""
    if getattr(world, "_tag_bit_map", None) is not None:
        return world._tag_bit_map
    all_tags: set[str] = set()
    for df in (world.actors, world.persons):
        if df is not None and "style_tags" in df.columns:
            for raw in df["style_tags"].fillna("").astype(str).values:
                for t in raw.replace(",", ";").split(";"):
                    t = t.strip().lower()
                    if t:
                        all_tags.add(t)
            break
    sorted_tags = sorted(all_tags)[:64]  # uint64 supports up to 64 tags
    world._tag_bit_map = {tag: i for i, tag in enumerate(sorted_tags)}
    return world._tag_bit_map


def _build_tag_bitmasks(tag_sets: list[set[str]] | None, bit_map: dict[str, int]) -> np.ndarray | None:
    """Convert a list of tag sets into a uint64 bitmask array."""
    if tag_sets is None:
        return None
    n = len(tag_sets)
    bitmasks = np.zeros(n, dtype=np.uint64)
    for i, tags in enumerate(tag_sets):
        mask = np.uint64(0)
        for t in tags:
            bit = bit_map.get(t)
            if bit is not None:
                mask |= np.uint64(1) << np.uint64(bit)
        bitmasks[i] = mask
    return bitmasks


def _actor_genre_match(cache: CastYearCache, genre_lower: str) -> np.ndarray:
    key = str(genre_lower or "").lower()
    cached = cache.genre_match_cache.get(key)
    if cached is not None:
        return cached
    if not key or cache.ga_lower is None:
        cached = np.zeros(len(cache.pids), dtype=bool)
    else:
        cached = np.char.find(_to_str_ndarray(cache.ga_lower), key) >= 0
    cache.genre_match_cache[key] = cached
    return cached


def _actor_style_match(cache: CastYearCache, hints: Iterable[str]) -> np.ndarray:
    key = tuple(str(h) for h in hints if h)
    cached = cache.style_match_cache.get(key)
    if cached is not None:
        return cached
    if not key or cache.st_lower is None:
        cached = np.zeros(len(cache.pids), dtype=bool)
    else:
        tagged = np.char.add(
            np.char.add(";", np.char.replace(_to_str_ndarray(cache.st_lower), ",", ";")),
            ";",
        )
        cached = np.zeros(len(cache.pids), dtype=bool)
        for hint in key:
            cached |= np.char.find(tagged, f";{hint};") >= 0
    cache.style_match_cache[key] = cached
    return cached


def _actor_market_match(cache: CastYearCache, target_market: str) -> np.ndarray:
    key = str(target_market or "").lower()
    cached = cache.market_match_cache.get(key)
    if cached is not None:
        return cached
    if not key or cache.market_fit_lower is None:
        cached = np.zeros(len(cache.pids), dtype=bool)
    else:
        mf = _to_str_ndarray(cache.market_fit_lower)
        cached = (np.char.find(mf, "global") >= 0) | (np.char.find(mf, key) >= 0)
    cache.market_match_cache[key] = cached
    return cached


def _actor_avoid_genre_match(world: WorldState, cache: CastYearCache, genre: str) -> np.ndarray:
    key = str(genre or "")
    cached = cache.avoid_genre_cache.get(key)
    if cached is not None:
        return cached
    sparse_avoid = getattr(world, "_latent_avoid_genres", None)
    if not sparse_avoid:
        cached = np.zeros(len(cache.pids), dtype=bool)
    else:
        cached = np.fromiter(
            (key in sparse_avoid.get(int(pid), set()) for pid in cache.pids),
            dtype=bool,
            count=len(cache.pids),
        )
    cache.avoid_genre_cache[key] = cached
    return cached


def _build_actor_static_block(world: WorldState, concept: dict) -> ActorStaticBlock:
    year = int(concept["year"])
    genre = str(concept["genre"])
    tier = str(concept["tier"])
    tone = str(concept.get("tone", "neutral"))
    movie_country = str(concept.get("country", ""))
    franchise = concept.get("franchise")
    selection = _get_selection_year_state(world, year)
    cache = selection.actor_cache
    concept_key = (
        int(year),
        genre.lower(),
        str(tier),
        movie_country,
        tone.lower(),
        bool(franchise and franchise.get("movies_generated", 0) > 0),
    )
    cached = selection.get_actor_view(concept_key)
    if cached is not None:
        return cached

    cast_size = _sample_cast_size(world, concept)
    candidate_idx = np.arange(len(cache.pids), dtype=np.int32)
    min_pool = max(cast_size * 4, 50)
    genre_lower = genre.lower()
    genre_match_all = _actor_genre_match(cache, genre_lower) if cache.ga_lower is not None else None

    if candidate_idx.size > min_pool and genre_match_all is not None:
        filtered_idx = candidate_idx[genre_match_all[candidate_idx]]
        if filtered_idx.size >= min_pool:
            candidate_idx = filtered_idx

    if candidate_idx.size > min_pool and cache.budget_pref is not None:
        tier_idx = TIER_TO_LATENT_IDX.get(tier, 2)
        tier_ok = cache.budget_pref[candidate_idx, tier_idx] >= 0.05
        filtered_idx = candidate_idx[tier_ok]
        if filtered_idx.size >= min_pool:
            candidate_idx = filtered_idx

    pids = cache.pids[candidate_idx]
    n = len(candidate_idx)
    base_scores = cache.pop_weight[candidate_idx].copy()

    peak_s = cache.peak_start[candidate_idx]
    peak_e = cache.peak_end[candidate_idx]
    valid = peak_e >= peak_s
    in_peak = valid & (peak_s <= year) & (year <= peak_e)
    career_mult = np.where(in_peak, 3.0, 1.0)

    stage_vals = cache.stage_vals[candidate_idx]
    career_stage_mult = cache.career_stage_mult[candidate_idx]

    geo_boost = _geo_boost_for_tier(world, tier, 2.0)
    if movie_country and cache.nat_country is not None:
        nat_country = cache.nat_country[candidate_idx]
        nationality_mult = np.where(nat_country == movie_country, geo_boost, 1.0)
        actor_nats = cache.actor_nationalities[candidate_idx] if cache.actor_nationalities is not None else None
    else:
        nationality_mult = np.ones(n, dtype=float)
        actor_nats = None

    if genre_match_all is not None:
        genre_match = genre_match_all[candidate_idx]
    else:
        genre_match = np.zeros(n, dtype=bool)
    genre_mult = np.where(genre_match, 5.0, 1.0)

    hints = TONE_STYLE_HINTS.get(tone, [tone])
    policy_enabled = _policy_enabled(world)
    actor_tag_sets = None
    if hints and cache.st_lower is not None:
        style_match = _actor_style_match(cache, hints)[candidate_idx]
        style_boost = _prior_float(world, "cast_style_multiplier", 2.0, lo=1.0, hi=4.0)
        style_mult = np.where(style_match, style_boost, 1.0)
    else:
        style_mult = np.ones(n, dtype=float)
    if policy_enabled and cache.actor_tag_sets is not None:
        actor_tag_sets = [cache.actor_tag_sets[int(i)] for i in candidate_idx] if cache.actor_tag_sets is not None else None

    avoid_genre_mult = np.ones(n, dtype=float)
    sparse_avoid = getattr(world, "_latent_avoid_genres", None)
    if sparse_avoid:
        avoid_genre_mult = np.where(_actor_avoid_genre_match(world, cache, genre)[candidate_idx], 0.15, 1.0)

    market_mult = np.ones(n, dtype=float)
    if cache.market_fit_lower is not None and movie_country:
        target_market = _COUNTRY_TO_MARKET.get(movie_country, "")
        if target_market:
            market_mult = np.where(_actor_market_match(cache, target_market)[candidate_idx], 2.0, 1.0)

    collab_mult = np.ones(n, dtype=float)
    latent_collab = getattr(world, "_latent_collab", None)
    li_arr = cache.li_arr[candidate_idx] if cache.li_arr is not None else None
    valid_li = cache.valid_li[candidate_idx] if cache.valid_li is not None else None

    if li_arr is not None and latent_collab is not None and valid_li.any():
        safe_li = np.clip(li_arr, 0, len(latent_collab) - 1)
        styles = latent_collab[safe_li]
        cs = cast_size
        style_to_mult = {
            "solo": 0.6 if cs >= 6 else (1.5 if cs <= 3 else 1.0),
            "ensemble": 2.0 if cs >= 6 else (0.8 if cs <= 2 else 1.2),
            "chameleon": 1.0,
            "mentorship": 1.3 if cs >= 4 else 1.0,
        }
        for style_name, mult in style_to_mult.items():
            mask = valid_li & (styles == style_name)
            collab_mult[mask] = mult

    tier_ctro = {"Epic": 2.5, "A": 1.8, "Mid": 1.0, "Indie": 0.3, "Micro": 0.1}.get(tier, 1.0)
    controversy_mult = np.ones(n, dtype=float)
    if li_arr is not None and getattr(world, "_latent_controversy", None) is not None and tier_ctro > 0:
        safe = np.clip(li_arr, 0, len(world._latent_controversy) - 1)
        vals = np.where(valid_li, world._latent_controversy[safe], 0.15)
        controversy_mult = np.maximum(0.2, 1.0 - tier_ctro * vals * 0.4)

    tier_vol = {"Epic": 1.5, "A": 1.0, "Mid": 0.5, "Indie": 0.0, "Micro": 0.0}.get(tier, 0.5)
    volatility_mult = np.ones(n, dtype=float)
    if li_arr is not None and getattr(world, "_latent_volatility", None) is not None and tier_vol > 0:
        safe = np.clip(li_arr, 0, len(world._latent_volatility) - 1)
        vals = np.where(valid_li, world._latent_volatility[safe], 0.4)
        volatility_mult = np.maximum(0.4, 1.0 - tier_vol * vals * 0.4)

    csv_mult = np.ones(n, dtype=float)
    csv_target = np.asarray(_concept_csv_target(concept), dtype=np.float32)
    norm = float(np.linalg.norm(csv_target))
    csv_target = csv_target / norm if norm > 1e-10 else csv_target
    if li_arr is not None and getattr(world, "_latent_csv_normed", None) is not None:
        safe = np.clip(li_arr, 0, len(world._latent_csv_normed) - 1)
        sims = np.clip(world._latent_csv_normed[safe] @ csv_target, 0.0, 1.0)
        sims[~valid_li] = 0.0
        csv_mult = 0.7 + 0.8 * sims

    company_aff_mult = _person_company_multiplier(world, cache.pids, year, tier, genre)[candidate_idx]

    actor_genders = cache.actor_genders[candidate_idx] if cache.actor_genders is not None else None
    cand_agencies = cache.cand_agencies[candidate_idx]
    cand_communities = cache.cand_communities[candidate_idx]
    immutable_scores = base_scores.copy()
    immutable_scores *= career_mult
    immutable_scores *= career_stage_mult
    immutable_scores *= nationality_mult
    immutable_scores *= genre_mult
    immutable_scores *= style_mult
    immutable_scores *= avoid_genre_mult
    immutable_scores *= market_mult
    immutable_scores *= collab_mult
    immutable_scores *= controversy_mult
    immutable_scores *= volatility_mult
    immutable_scores *= csv_mult
    immutable_scores *= company_aff_mult
    top_cut = float(np.quantile(immutable_scores, 0.95)) if len(immutable_scores) > 30 else float(np.max(immutable_scores) if len(immutable_scores) else 0.0)
    top_star_mask = immutable_scores >= top_cut
    sorted_base = np.sort(immutable_scores)[::-1]
    sim_threshold = float(sorted_base[min(199, len(sorted_base) - 1)]) * 0.5 if len(sorted_base) else 0.0
    shortlist = _shortlist_budget(world, "cast", 120)
    base_focus_size = min(
        len(candidate_idx),
        max(shortlist * 14, 1600, int(round(len(candidate_idx) * 0.18))),
    )
    base_focus_size_cap = int(_env_float("DATA_SYS_CAST_BASE_FOCUS_SIZE_CAP", float(base_focus_size), float(max(shortlist * 4, 512)), float(base_focus_size)))
    base_focus_size = min(base_focus_size, base_focus_size_cap)
    base_focus_idx = _shortlist_indices(
        immutable_scores,
        base_focus_size,
        world.rng,
        exploration_share=_prior_float(world, "cast_base_focus_exploration", 0.45, lo=0.0, hi=0.60),
    )

    cooldown_floor = np.ones(n, dtype=np.float32)
    cooldown_decay = np.full(n, 0.15, dtype=np.float32)
    if len(stage_vals):
        stage_vals_arr = np.asarray(stage_vals, dtype=object)
        for stage_name, floor in {
            "legend": 0.70,
            "prime": 0.45,
            "veteran": 0.50,
            "rising": 0.25,
            "retired": 0.05,
        }.items():
            stage_mask = stage_vals_arr == stage_name
            if not np.any(stage_mask):
                continue
            cooldown_floor[stage_mask] = np.float32(floor)
            cooldown_decay[stage_mask] = np.float32({
                "legend": 0.07,
                "prime": 0.13,
                "veteran": 0.10,
                "rising": 0.18,
                "retired": 0.30,
            }.get(stage_name, 0.15))

    actor_tag_bitmasks = cache.actor_tag_bitmasks[candidate_idx].copy() if cache.actor_tag_bitmasks is not None else None
    policy_rule_mult = None
    if policy_enabled:
        style_values = None
        if actor_tag_sets is not None:
            style_values = np.asarray([";".join(sorted(tags)) for tags in actor_tag_sets], dtype=object)
        policy_rule_mult = _person_policy_rule_multiplier(
            world,
            pids,
            stage_vals,
            genre,
            style_tags=style_values,
        )

    block = ActorStaticBlock(
        key=concept_key,
        candidate_idx=candidate_idx.copy(),
        pids=pids,
        immutable_scores=immutable_scores,
        base_focus_idx=base_focus_idx.astype(np.int32, copy=False),
        yearly_max=cache.yearly_max[candidate_idx].copy(),
        stage_vals=stage_vals,
        cooldown_floor=cooldown_floor,
        cooldown_decay=cooldown_decay,
        li_arr=li_arr,
        actor_genders=actor_genders,
        actor_nationalities=actor_nats,
        actor_tag_sets=actor_tag_sets,
        actor_tag_bitmasks=actor_tag_bitmasks,
        pid_to_idx={int(pid): i for i, pid in enumerate(pids)},
        top_star_mask=top_star_mask,
        cand_agencies=cand_agencies,
        cand_communities=cand_communities,
        _sim_threshold=sim_threshold,
        policy_rule_mult=policy_rule_mult,
    )
    selection.actor_views[concept_key] = block
    max_views = int(_env_float("DATA_SYS_ACTOR_VIEW_CACHE_MAX", 256.0, 16.0, 4096.0))
    while len(selection.actor_views) > max_views:
        selection.actor_views.pop(next(iter(selection.actor_views)), None)
    return block


def _expand_cast_focus_indices(
    world: WorldState,
    static: ActorStaticBlock,
    scores: np.ndarray,
    cast_id_list: list[int],
    year: int,
    shortlist_size: int,
) -> np.ndarray:
    focus = _shortlist_indices(
        scores,
        max(shortlist_size * 2, 96),
        world.rng,
        exploration_share=_prior_float(world, "cast_focus_exploration", 0.40, lo=0.0, hi=0.60),
    )
    if focus.size == 0 or not cast_id_list:
        return focus
    graph = getattr(world, "graph", None)
    frontier: list[int] = []
    for cid in cast_id_list:
        friend_iter = graph.iter_friend_neighbors(int(cid), year) if graph is not None else world._friend_adj_all.get(int(cid), [])
        rival_iter = graph.iter_rival_neighbors(int(cid), year) if graph is not None else world._rival_adj_all.get(int(cid), [])
        for nbr, _w, _vf, _vt in friend_iter:
            idx = static.pid_to_idx.get(int(nbr))
            if idx is not None and scores[idx] > 0:
                frontier.append(int(idx))
        for nbr, _w, _vf, _vt in rival_iter:
            idx = static.pid_to_idx.get(int(nbr))
            if idx is not None and scores[idx] > 0:
                frontier.append(int(idx))
    if not frontier:
        return focus
    frontier_arr = np.unique(np.asarray(frontier, dtype=np.int32))
    merged = np.unique(np.concatenate([focus.astype(np.int32, copy=False), frontier_arr]))
    cap = max(shortlist_size * 3, 160)
    if merged.size > cap:
        merged = merged[np.argsort(scores[merged])[::-1][:cap]]
    return merged


def _expand_cast_focus_indices_cached(
    world: WorldState,
    scores: np.ndarray,
    shortlist_size: int,
    frontier_local_indices: Iterable[int] | None = None,
) -> np.ndarray:
    focus = _shortlist_indices(
        scores,
        max(shortlist_size * 2, 96),
        world.rng,
        exploration_share=_prior_float(world, "cast_focus_exploration", 0.40, lo=0.0, hi=0.60),
    )
    extras: list[int] = []
    if frontier_local_indices is not None:
        for idx in frontier_local_indices:
            local_idx = int(idx)
            if 0 <= local_idx < len(scores) and scores[local_idx] > 0:
                extras.append(local_idx)
    if not extras:
        return focus
    extra_arr = np.unique(np.asarray(extras, dtype=np.int32))
    if focus.size == 0:
        return extra_arr
    merged = np.unique(np.concatenate([focus.astype(np.int32, copy=False), extra_arr]))
    cap = max(shortlist_size * 3, 160)
    if merged.size > cap:
        merged = merged[np.argsort(scores[merged])[::-1][:cap]]
    return merged


def _cap_focus_indices(scores: np.ndarray, indices: np.ndarray, cap: int) -> np.ndarray:
    if indices.size <= cap:
        return indices
    cap = max(1, int(cap))
    explore_share = _env_float("DATA_SYS_FOCUS_CAP_EXPLORATION_SHARE", 0.30, 0.0, 0.60)
    anchor_size = max(1, min(cap, int(round(cap * (1.0 - explore_share)))))
    ordered = indices[np.argsort(scores[indices])[::-1]]
    anchor = ordered[:anchor_size]
    remaining = ordered[anchor_size:]
    explore_size = cap - anchor.size
    if explore_size <= 0 or remaining.size == 0:
        return anchor[:cap]
    if remaining.size <= explore_size:
        explore = remaining
    else:
        positions = np.linspace(0, remaining.size - 1, num=explore_size, dtype=np.int32)
        explore = remaining[positions]
    merged = np.unique(np.concatenate([anchor, explore]).astype(np.int32, copy=False))
    if merged.size > cap:
        merged = merged[np.argsort(scores[merged])[::-1][:cap]]
    return merged


def _priority_cast_indices(
    static: ActorStaticBlock,
    director_pref_boost: np.ndarray,
    award_mask: np.ndarray,
    franchise_pool_mask: np.ndarray,
    limit: int,
) -> np.ndarray:
    extras: list[np.ndarray] = []
    if np.any(director_pref_boost > 1.0):
        pref = np.flatnonzero(director_pref_boost > 1.0)
        if pref.size > limit:
            pref = pref[np.argsort(static.immutable_scores[pref] * director_pref_boost[pref])[::-1][:limit]]
        extras.append(pref.astype(np.int32, copy=False))
    if np.any(award_mask):
        award_idx = np.flatnonzero(award_mask)
        if award_idx.size > limit:
            award_idx = award_idx[np.argsort(static.immutable_scores[award_idx])[::-1][:limit]]
        extras.append(award_idx.astype(np.int32, copy=False))
    if np.any(franchise_pool_mask):
        pool_idx = np.flatnonzero(franchise_pool_mask)
        if pool_idx.size > limit:
            pool_idx = pool_idx[np.argsort(static.immutable_scores[pool_idx])[::-1][:limit]]
        extras.append(pool_idx.astype(np.int32, copy=False))
    if not extras:
        return np.zeros(0, dtype=np.int32)
    merged = np.unique(np.concatenate(extras))
    if merged.size > limit * 2:
        merged = _cap_focus_indices(static.immutable_scores, merged, limit * 2)
    return merged.astype(np.int32, copy=False)


def _sample_cast_size(world: WorldState, concept: dict) -> int:
    cast_cfg = _cast_selection_config(world)
    tier = str(concept.get("tier", "Mid"))
    genre = str(concept.get("genre", "Drama"))
    franchise = concept.get("franchise")
    lo, hi = cast_cfg["dynamic_cast_base_by_tier"].get(tier, CAST_SIZE_RANGES.get(tier, (3, 8)))
    if genre in _BLOCKBUSTER_GENRES:
        hi += int(cast_cfg["blockbuster_bonus_major"] if tier in ("Epic", "A") else cast_cfg["blockbuster_bonus_other"])
    if franchise is not None:
        hi += int(cast_cfg["franchise_bonus_major"] if tier in ("Epic", "A") else cast_cfg["franchise_bonus_other"])
    cast_max_cap = int(_env_float("V16_CAST_MAX", 120.0, 40.0, 240.0))
    hi = max(lo, min(hi, cast_max_cap))
    if tier == "Epic":
        tail_p = (
            float(cast_cfg["epic_tail_prob_base"])
            + (float(cast_cfg["epic_tail_prob_franchise_bonus"]) if franchise is not None else 0.0)
            + (float(cast_cfg["epic_tail_prob_blockbuster_bonus"]) if genre in _BLOCKBUSTER_GENRES else 0.0)
        )
        tail_p *= _env_float("V16_EPIC_TAIL_SCALE", 1.0, 0.4, 3.0)
        if world.rng.random() < min(float(cast_cfg["epic_tail_prob_cap"]), tail_p):
            return int(
                np.clip(
                    world.rng.lognormal(
                        mean=float(cast_cfg["epic_tail_lognorm_mean"]),
                        sigma=float(cast_cfg["epic_tail_lognorm_sigma"]),
                    ),
                    int(cast_cfg["epic_tail_min"]),
                    cast_max_cap,
                )
            )
    if tier == "A" and (franchise is not None or genre in _BLOCKBUSTER_GENRES):
        a_tail_p = _env_float("V16_A_TAIL_PROB", float(cast_cfg["a_tail_prob"]), 0.01, 0.25)
        if world.rng.random() < a_tail_p:
            hi_a = int(min(cast_max_cap, int(cast_cfg["a_tail_max"])))
            return int(world.rng.randint(int(cast_cfg["a_tail_min"]), max(int(cast_cfg["a_tail_min"]) + 1, hi_a)))
    return int(world.rng.randint(lo, hi + 1))


def _policy_enabled(world: WorldState) -> bool:
    return bool(getattr(world, "enable_llm_world_policy", False) and getattr(world, "world_policy", None))


def _concept_packs_enabled(world: WorldState) -> bool:
    return bool(getattr(world, "enable_llm_concept_packs", False) and getattr(world, "concept_packs", None))


def _year_slates_enabled(world: WorldState) -> bool:
    return bool(getattr(world, "enable_llm_year_slates", False) and getattr(world, "year_slate_plan", None))


def _keyword_motifs_enabled(world: WorldState) -> bool:
    if not getattr(world, "enable_llm_keyword_motifs", False):
        return False
    keywords = getattr(world, "keywords", None)
    return bool(keywords is not None and len(keywords) > 0 and "motif_family" in keywords.columns)


def _franchise_bible(franchise: dict | None) -> dict[str, Any]:
    if not isinstance(franchise, dict):
        return {}
    bible = franchise.get("franchise_bible")
    return dict(bible) if isinstance(bible, dict) else {}


def _year_slate_for_concept(world: WorldState, *, bucket_id: str, market: str, tier: str) -> dict[str, Any]:
    if not _year_slates_enabled(world):
        return {}
    return resolve_year_slate(getattr(world, "year_slate_index", {}), bucket_id=str(bucket_id), market=str(market), tier=str(tier))


def _compatibility_weight(world: WorldState, section: str, key: str, default: float) -> float:
    compatibility = getattr(world, "world_policy", {}).get("compatibility", {}) if _policy_enabled(world) else {}
    values = compatibility.get(section, {}) if isinstance(compatibility, dict) else {}
    try:
        return float(values.get(key, default))
    except Exception:
        return float(default)


def _policy_bucket(world: WorldState, year: int) -> dict[str, Any]:
    if _policy_enabled(world):
        return bucket_for_year(getattr(world, "world_policy", {}), int(year))
    return bucket_for_year({}, int(year))


def _canonical_genre_hint(genre_hint: str | None) -> str:
    text = str(genre_hint or "").strip().lower()
    if not text:
        return ""
    for genre_name in GENRES:
        if str(genre_name).strip().lower() == text:
            return str(genre_name)
    return ""


def _pack_candidates_for_bucket(
    world: WorldState,
    year: int,
    *,
    genre_hint: str | None = None,
    tier_hint: str | None = None,
    country_hint: str | None = None,
) -> list[dict[str, Any]]:
    bucket = _policy_bucket(world, year)
    bucket_id = str(bucket.get("bucket_id"))
    index = getattr(world, "concept_packs_index", {}) or {}
    keys = []
    if genre_hint and tier_hint and country_hint:
        keys.append(f"{genre_hint}|{tier_hint}|{country_hint}")
    if genre_hint and tier_hint:
        keys.append(f"{genre_hint}|{tier_hint}|{_COUNTRY_TO_MARKET.get(country_hint or '', '')}")
        keys.append(f"{genre_hint}|{tier_hint}")
    if genre_hint:
        keys.append(genre_hint)
    keys.append("*")
    seen = set()
    out = []
    ordered_bucket_ids = [bucket_id] + [str(other_id) for other_id in index.keys() if str(other_id) != bucket_id]
    for search_bucket_id in ordered_bucket_ids:
        bucket_index = index.get(search_bucket_id, {})
        for key in keys:
            for pack in bucket_index.get(key, []):
                pack_id = str(pack.get("pack_id"))
                if pack_id and pack_id not in seen:
                    seen.add(pack_id)
                    out.append(dict(pack))
        if out:
            return out
    return out


def _writer_director_probability(world: WorldState, tier: str, default_map: dict[str, float] | None = None) -> float:
    mapping = _section_prior_float_map(
        world,
        "selection_weights",
        "writer_director_probability_by_tier",
        default_map or {"Indie": 0.20, "Micro": 0.25, "Mid": 0.12, "A": 0.08, "Epic": 0.06},
        lo=0.0,
        hi=1.0,
    )
    return float(np.clip(mapping.get(str(tier), 0.10), 0.0, 1.0))


def _genre_tier_distribution(world: WorldState, genre: str) -> np.ndarray | None:
    prior_map = _section_prior_dict(world, "selection_weights", "genre_tier_distribution")
    raw = prior_map.get(str(genre))
    arr: np.ndarray | None = None
    if isinstance(raw, dict):
        try:
            arr = np.asarray([max(1e-6, float(raw.get(tier, 0.0))) for tier in PRODUCTION_TIERS], dtype=float)
        except Exception:
            arr = None
    elif isinstance(raw, (list, tuple)) and len(raw) == len(PRODUCTION_TIERS):
        try:
            arr = np.asarray([max(1e-6, float(v)) for v in raw], dtype=float)
        except Exception:
            arr = None
    if arr is None:
        if current_mode() == "research":
            audit_fallback_hit(
                "assembly.selection_weights",
                f"missing:genre_tier_distribution.{genre}",
                detail=f"selection_weights.genre_tier_distribution.{genre} must be provided in research mode",
                mode="research",
            )
        return None
    total = float(arr.sum()) or 1.0
    return arr / total


def _seasonal_month_weights(world: WorldState, genre: str, season_bias: str | None) -> np.ndarray:
    base_weights = _section_prior_list(world, "selection_weights", "release_month_base_weights", expected_len=12)
    if base_weights is None:
        base_weights = [0.06, 0.06, 0.07, 0.07, 0.10, 0.11, 0.11, 0.10, 0.07, 0.07, 0.09, 0.09]
    month_weights = np.asarray(base_weights, dtype=float)
    season = str(season_bias or "").strip().lower()
    season_bumps = _section_prior_dict(world, "selection_weights", "release_season_month_bumps")
    genre_bumps = _section_prior_dict(world, "selection_weights", "genre_release_month_bumps")

    def _apply_bump_map(raw: Any) -> None:
        nonlocal month_weights
        if isinstance(raw, dict):
            for k, v in raw.items():
                try:
                    idx = int(k) - 1
                    if 0 <= idx < 12:
                        month_weights[idx] += float(v)
                except Exception:
                    continue
        elif isinstance(raw, (list, tuple)) and len(raw) == 12:
            try:
                month_weights += np.asarray([float(v) for v in raw], dtype=float)
            except Exception:
                pass

    if season in season_bumps:
        _apply_bump_map(season_bumps.get(season))
    else:
        if season == "summer":
            month_weights[[4, 5, 6, 7]] += np.array([0.04, 0.05, 0.04, 0.02], dtype=float)
        elif season in {"awards", "festival", "fall"}:
            month_weights[[8, 9, 10, 11]] += np.array([0.04, 0.06, 0.04, 0.03], dtype=float)
        elif season == "winter":
            month_weights[[0, 1, 11]] += np.array([0.02, 0.06, 0.03], dtype=float)
        elif season == "spring":
            month_weights[[2, 3, 4]] += np.array([0.03, 0.04, 0.02], dtype=float)
        elif season == "holiday":
            month_weights[[10, 11]] += np.array([0.05, 0.08], dtype=float)

    if str(genre) in genre_bumps:
        _apply_bump_map(genre_bumps.get(str(genre)))
    else:
        if genre in ("Horror", "Thriller") and season not in {"fall", "festival"}:
            month_weights[8] += 0.06
            month_weights[9] += 0.04
        elif genre == "Romance" and season != "winter":
            month_weights[1] += 0.06
            month_weights[5] += 0.04
        elif genre in ("Action", "Sci-Fi", "Fantasy") and season != "summer":
            month_weights[4] += 0.05
            month_weights[5] += 0.04
        elif genre == "Drama" and season not in {"awards", "festival"}:
            month_weights[8] += 0.04
            month_weights[9] += 0.06
    return month_weights / month_weights.sum()


def _build_pack_backed_concept(
    world: WorldState,
    movie_id: int,
    year: int,
    pack: dict[str, Any],
    franchise: dict[str, Any] | None,
) -> dict[str, Any]:
    genre = str(pack.get("genre", "Drama"))
    tier = str(pack.get("tier", "Mid"))
    country = str(pack.get("country", "USA"))
    if franchise:
        installment = franchise["movies_generated"] + 1
    else:
        installment = None
    franchise_bible = _franchise_bible(franchise)
    market = str(pack.get("market") or _COUNTRY_TO_MARKET.get(country, "Global"))
    year_bucket = str(pack.get("bucket_id") or _policy_bucket(world, year).get("bucket_id"))
    year_slate = _year_slate_for_concept(world, bucket_id=year_bucket, market=market, tier=tier)
    month = int(world.rng.choice(range(1, 13), p=_seasonal_month_weights(world, genre, str(pack.get("release_season_bias", "")))))
    return {
        "movie_id": int(movie_id),
        "genre": genre,
        "tier": tier,
        "year": int(year),
        "country": country,
        "market": market,
        "language": COUNTRY_LANGUAGE.get(country, "English"),
        "tone": str(pack.get("tone_intensity") or _GENRE_TONE.get(genre, "neutral")),
        "month": month,
        "_world": world,
        "franchise": franchise,
        "franchise_bible": franchise_bible,
        "installment": installment,
        "is_writer_director": bool(world.rng.random() < _writer_director_probability(world, tier)),
        "year_bucket": year_bucket,
        "year_slate": year_slate,
        "concept_pack_id": str(pack.get("pack_id", "")),
        "concept_pack": dict(pack),
        "policy_targets": {
            "company_strategy_tag": str(pack.get("company_strategy_tag", "")),
            "cast_chemistry_target": str(pack.get("cast_chemistry_target", "")),
            "keyword_seed_cluster": list(pack.get("keyword_seed_cluster", [])) if isinstance(pack.get("keyword_seed_cluster"), list) else [],
            "title_style": str(pack.get("title_style", "")),
            "tagline_style": str(pack.get("tagline_style", "")),
            "release_season_bias": str(pack.get("release_season_bias", "")),
            "priority_motifs": list(year_slate.get("priority_motifs", [])) if isinstance(year_slate.get("priority_motifs"), list) else [],
            "trending_subgenres": list(year_slate.get("trending_subgenres", [])) if isinstance(year_slate.get("trending_subgenres"), list) else [],
            "motif_drift": list(year_slate.get("motif_drift", [])) if isinstance(year_slate.get("motif_drift"), list) else [],
            "novelty_target": float(year_slate.get("novelty_target", 0.35) or 0.35),
            "sequel_appetite": float(year_slate.get("sequel_appetite", 0.3) or 0.3),
            "franchise_keyword_families": list(franchise_bible.get("keyword_families", [])) if isinstance(franchise_bible.get("keyword_families"), list) else [],
        },
    }


def _log_selection_decision(
    world: WorldState,
    *,
    stage: str,
    concept: dict | None,
    chosen: Any,
    confidence: float,
    candidates: Sequence[Mapping[str, Any]] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    path = getattr(world, "decision_log_path", None)
    if not path:
        return
    row = {
        "stage": stage,
        "movie_id": int(concept.get("movie_id", 0)) if isinstance(concept, dict) else 0,
        "year": int(concept.get("year", 0)) if isinstance(concept, dict) else 0,
        "genre": str(concept.get("genre", "")) if isinstance(concept, dict) else "",
        "tier": str(concept.get("tier", "")) if isinstance(concept, dict) else "",
        "country": str(concept.get("country", "")) if isinstance(concept, dict) else "",
        "concept_pack_id": str(concept.get("concept_pack_id", "")) if isinstance(concept, dict) else "",
        "confidence": round(float(confidence), 4),
        "chosen": chosen,
        "candidates": list(candidates or []),
    }
    if extra:
        row.update(dict(extra))
    append_jsonl(path, row)
    latest_path = getattr(world, "decision_log_latest_path", None)
    if latest_path and str(latest_path) != str(path):
        append_jsonl(latest_path, row)


def _person_policy_rule_multiplier(
    world: WorldState,
    person_ids: np.ndarray,
    career_stages: np.ndarray | None,
    genre: str,
    *,
    style_tags: np.ndarray | None = None,
) -> np.ndarray:
    if not _policy_enabled(world) or person_ids.size == 0:
        return np.ones(len(person_ids), dtype=float)
    rules = getattr(world, "world_policy", {}).get("talent_boost_rules", [])
    if not isinstance(rules, list) or not rules:
        return np.ones(len(person_ids), dtype=float)
    out = np.ones(len(person_ids), dtype=float)
    norm_genre = str(genre or "")
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        rule_genre = str(rule.get("genre", "") or "")
        if rule_genre and rule_genre != norm_genre:
            continue
        boost = max(0.8, min(1.4, float(rule.get("boost", 1.0) or 1.0)))
        stage = str(rule.get("career_stage", "") or "")
        if stage and career_stages is not None:
            out *= np.where(career_stages == stage, boost, 1.0)
        style_tag = str(rule.get("style_tag", "") or "")
        if style_tag and style_tags is not None:
            out *= np.where(np.char.find(_to_str_ndarray(style_tags), style_tag) >= 0, boost, 1.0)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sample_movie_concept(world: WorldState, movie_id: int, forced_year: int = None, title_assignment: dict | None = None) -> dict:
    rng = world.rng
    franchise = world.movie_franchise_map.get(movie_id)
    franchise_bible = _franchise_bible(franchise)
    genre_hint = _canonical_genre_hint((title_assignment or {}).get("genre_hint", ""))
    if not hasattr(world, "_concept_pack_usage_counts"):
        world._concept_pack_usage_counts = defaultdict(int)
    if not hasattr(world, "_concept_country_usage_counts"):
        world._concept_country_usage_counts = defaultdict(int)
    if not hasattr(world, "_concept_bucket_country_usage_counts"):
        world._concept_bucket_country_usage_counts = defaultdict(int)
    if not hasattr(world, "_concept_bucket_genre_usage_counts"):
        world._concept_bucket_genre_usage_counts = defaultdict(int)
    if not hasattr(world, "_concept_bucket_total_counts"):
        world._concept_bucket_total_counts = defaultdict(int)
    target_movie_count = max(1, int(getattr(world, "target_movie_count", len(getattr(world, "title_bank", [])) or 5000)))
    expected_pack_load = max(4.0, float(target_movie_count) / max(1, len(getattr(world, "concept_packs", []) or [])))
    expected_country_load = max(12.0, float(target_movie_count) / max(1, len(COUNTRIES)))

    if forced_year is not None:
        year = int(forced_year)
    elif YEAR_RANGE:
        year = int(rng.randint(YEAR_RANGE[0], YEAR_RANGE[1] + 1))
    else:
        decades = list(DECADE_WEIGHTS.keys())
        dweights = _normalise_dict_weights({str(k): float(v) for k, v in DECADE_WEIGHTS.items()})
        decade = int(rng.choice(decades, p=[dweights[str(k)] for k in decades]))
        year = int(decade + rng.randint(0, 10))

    bucket = _policy_bucket(world, year)
    if franchise:
        genre = franchise["genre"]
        tier = franchise["tier"]
        installment = franchise["movies_generated"] + 1
    else:
        installment = None
        genre = None
        tier = None

    if _concept_packs_enabled(world):
        pack_candidates = _pack_candidates_for_bucket(
            world,
            year,
            genre_hint=genre or genre_hint or None,
            tier_hint=tier,
        )
        if pack_candidates and genre_hint and franchise is None:
            exact_genre_candidates = [
                dict(pack) for pack in pack_candidates
                if str(pack.get("genre", "")).strip().lower() == genre_hint.lower()
            ]
            if exact_genre_candidates:
                pack_candidates = exact_genre_candidates
            else:
                # Preserve the title-bank genre program when concept packs for this
                # bucket collapse to a different genre family.
                pack_candidates = []
        if pack_candidates:
            scores = []
            candidate_concepts = []
            strategy_bonus_weight = _compatibility_weight(world, "company", "strategy_match", 0.24)
            genre_bonus_weight = _compatibility_weight(world, "company", "genre_match", 0.34)
            concept_genre_bias_base = _section_prior_float(world, "selection_weights", "concept_genre_bias_base", 0.8, lo=0.1, hi=3.0)
            concept_genre_bias_scale = _section_prior_float(world, "selection_weights", "concept_genre_bias_scale", 2.2, lo=0.0, hi=6.0)
            concept_country_bias_base = _section_prior_float(world, "selection_weights", "concept_country_bias_base", 0.9, lo=0.1, hi=3.0)
            concept_country_bias_scale = _section_prior_float(world, "selection_weights", "concept_country_bias_scale", 1.9, lo=0.0, hi=6.0)
            concept_market_bias_base = _section_prior_float(world, "selection_weights", "concept_market_bias_base", 0.9, lo=0.1, hi=3.0)
            concept_market_bias_scale = _section_prior_float(world, "selection_weights", "concept_market_bias_scale", 1.6, lo=0.0, hi=6.0)
            concept_exact_genre_hint_boost = _section_prior_float(world, "selection_weights", "concept_exact_genre_hint_boost", 2.10, lo=1.0, hi=6.0)
            concept_genre_hint_miss_penalty = _section_prior_float(world, "selection_weights", "concept_genre_hint_miss_penalty", 0.12, lo=0.01, hi=1.0)
            concept_franchise_genre_match_boost = _section_prior_float(world, "selection_weights", "concept_franchise_genre_match_boost", 1.20, lo=1.0, hi=3.0)
            concept_tier_match_boost = _section_prior_float(world, "selection_weights", "concept_tier_match_boost", 1.15, lo=1.0, hi=3.0)
            concept_franchise_eligible_scale = _section_prior_float(world, "selection_weights", "concept_franchise_eligible_scale", 0.50, lo=0.0, hi=2.0)
            concept_strategy_bonus_scale = _section_prior_float(world, "selection_weights", "concept_strategy_bonus_scale", 0.40, lo=0.0, hi=2.0)
            concept_release_pressure_base = _section_prior_float(world, "selection_weights", "concept_release_pressure_base", 0.92, lo=0.1, hi=2.0)
            concept_release_pressure_scale = _section_prior_float(world, "selection_weights", "concept_release_pressure_scale", 0.55, lo=0.0, hi=2.0)
            concept_novelty_base = _section_prior_float(world, "selection_weights", "concept_novelty_base", 0.95, lo=0.1, hi=2.0)
            concept_novelty_scale = _section_prior_float(world, "selection_weights", "concept_novelty_scale", 0.45, lo=0.0, hi=2.0)
            concept_franchise_strategy_match_boost = _section_prior_float(world, "selection_weights", "concept_franchise_strategy_match_boost", 1.18, lo=1.0, hi=3.0)
            concept_franchise_season_match_boost = _section_prior_float(world, "selection_weights", "concept_franchise_season_match_boost", 1.10, lo=1.0, hi=3.0)
            concept_sequel_pressure_scale = _section_prior_float(world, "selection_weights", "concept_sequel_pressure_scale", 0.35, lo=0.0, hi=2.0)
            concept_pack_usage_capacity = _section_prior_float(world, "selection_weights", "concept_pack_usage_capacity", 4.0, lo=0.5, hi=20.0)
            concept_bucket_country_capacity = _section_prior_float(world, "selection_weights", "concept_bucket_country_capacity", 2.0, lo=0.5, hi=10.0)
            concept_bucket_genre_capacity = _section_prior_float(world, "selection_weights", "concept_bucket_genre_capacity", 1.35, lo=0.25, hi=10.0)
            concept_country_usage_capacity = _section_prior_float(world, "selection_weights", "concept_country_usage_capacity", 1.0, lo=0.25, hi=10.0)
            concept_minor_country_bonus = _section_prior_float(world, "selection_weights", "concept_minor_country_bonus", 1.06, lo=1.0, hi=2.0)
            concept_minor_market_bonus = _section_prior_float(world, "selection_weights", "concept_minor_market_bonus", 1.04, lo=1.0, hi=2.0)
            for pack in pack_candidates:
                pack_genre = str(pack.get("genre", "Drama"))
                pack_tier = str(pack.get("tier", "Mid"))
                pack_country = str(pack.get("country", "USA"))
                pack_market = str(pack.get("market") or _COUNTRY_TO_MARKET.get(pack_country, "Global"))
                pack_bucket_id = str(pack.get("bucket_id") or bucket.get("bucket_id", ""))
                year_slate = _year_slate_for_concept(world, bucket_id=pack_bucket_id, market=pack_market, tier=pack_tier)
                score = 1.0
                score *= concept_genre_bias_base + concept_genre_bias_scale * float(bucket.get("genre_bias", {}).get(pack_genre, 1.0 / max(1, len(GENRES))))
                score *= concept_country_bias_base + concept_country_bias_scale * float(bucket.get("country_bias", {}).get(pack_country, 1.0 / max(1, len(COUNTRIES))))
                score *= concept_market_bias_base + concept_market_bias_scale * float(bucket.get("market_bias", {}).get(pack_market, 0.2))
                if genre_hint and pack_genre.lower() == genre_hint.lower():
                    score *= concept_exact_genre_hint_boost + genre_bonus_weight
                elif genre_hint and franchise is None:
                    score *= concept_genre_hint_miss_penalty
                if genre and pack_genre == genre:
                    score *= concept_franchise_genre_match_boost
                if tier and pack_tier == tier:
                    score *= concept_tier_match_boost
                if pack.get("franchise_eligible"):
                    score *= 1.0 + concept_franchise_eligible_scale * float(bucket.get("franchise_pressure", 0.25))
                strategy_tag = str(pack.get("company_strategy_tag", ""))
                if strategy_tag:
                    score *= 1.0 + concept_strategy_bonus_scale * strategy_bonus_weight
                if year_slate:
                    score *= concept_release_pressure_base + concept_release_pressure_scale * float(year_slate.get("release_pressure", 0.45))
                    score *= concept_novelty_base + concept_novelty_scale * float(year_slate.get("novelty_target", 0.35))
                if franchise_bible:
                    if str(franchise_bible.get("company_strategy_tag", "")) == strategy_tag and strategy_tag:
                        score *= concept_franchise_strategy_match_boost
                    if str(franchise_bible.get("release_season_bias", "")) == str(pack.get("release_season_bias", "")):
                        score *= concept_franchise_season_match_boost
                    score *= 1.0 + concept_sequel_pressure_scale * float(bucket.get("sequel_pressure", 0.25))
                pack_id = str(pack.get("pack_id", ""))
                pack_usage = float(world._concept_pack_usage_counts.get(pack_id, 0))
                country_usage = float(world._concept_country_usage_counts.get(pack_country, 0))
                bucket_country_usage = float(world._concept_bucket_country_usage_counts.get((pack_bucket_id, pack_country), 0))
                bucket_genre_usage = float(world._concept_bucket_genre_usage_counts.get((pack_bucket_id, pack_genre), 0))
                score *= 1.0 / (1.0 + (pack_usage / max(6.0, expected_pack_load * concept_pack_usage_capacity)))
                score *= 1.0 / (1.0 + (bucket_country_usage / max(4.0, expected_pack_load * concept_bucket_country_capacity)))
                score *= 1.0 / (1.0 + (bucket_genre_usage / max(2.0, expected_pack_load * concept_bucket_genre_capacity)))
                country_capacity = expected_country_load * ((3.6 * concept_country_usage_capacity) if pack_country in _MAJOR_HUBS else (2.0 * concept_country_usage_capacity))
                score *= 1.0 / (1.0 + (country_usage / max(10.0, country_capacity)))
                if pack_country not in _MAJOR_HUBS:
                    score *= concept_minor_country_bonus
                if pack_market not in {"North America", "Europe", "Asia"}:
                    score *= concept_minor_market_bonus
                scores.append(max(1e-6, score))
                candidate_concepts.append(_build_pack_backed_concept(world, movie_id, year, pack, franchise=franchise))

            probs = normalize_weights(np.asarray(scores, dtype=float))
            chosen_idx = int(rng.choice(len(candidate_concepts), p=probs))
            concept = candidate_concepts[chosen_idx]
            concept["selection_confidence"] = confidence_from_scores(scores)
            ranked_idx = list(np.argsort(np.asarray(scores))[::-1][:6])
            concept["_rerank_candidates"] = [candidate_concepts[idx] for idx in ranked_idx]
            concept["selection_mode"] = "concept_pack" if franchise is None else "concept_pack_franchise"
            chosen_pack_id = str(concept.get("concept_pack_id", ""))
            chosen_country = str(concept.get("country", ""))
            chosen_bucket_id = str(concept.get("year_bucket", ""))
            if chosen_pack_id:
                world._concept_pack_usage_counts[chosen_pack_id] += 1
            if chosen_country:
                world._concept_country_usage_counts[chosen_country] += 1
            if chosen_bucket_id and chosen_country:
                world._concept_bucket_country_usage_counts[(chosen_bucket_id, chosen_country)] += 1
            if chosen_bucket_id:
                world._concept_bucket_total_counts[chosen_bucket_id] += 1
                world._concept_bucket_genre_usage_counts[(chosen_bucket_id, str(concept.get("genre", "")))] += 1
            _log_selection_decision(
                world,
                stage="sample_movie_concept",
                concept=concept,
                chosen={"pack_id": concept.get("concept_pack_id"), "mode": concept["selection_mode"]},
                confidence=float(concept["selection_confidence"]),
                candidates=[
                    {
                        "pack_id": candidate_concepts[idx].get("concept_pack_id"),
                        "genre": candidate_concepts[idx].get("genre"),
                        "tier": candidate_concepts[idx].get("tier"),
                        "country": candidate_concepts[idx].get("country"),
                        "score": round(float(scores[idx]), 4),
                    }
                    for idx in ranked_idx
                ],
                extra={"year_bucket": concept.get("year_bucket", ""), "genre_hint": genre_hint},
            )
            return concept

    if franchise is None:
        genre_weights = dict(GENRE_WEIGHTS)
        for g, delta in getattr(world, "genre_weight_overrides", {}).items():
            if g in genre_weights:
                genre_weights[g] = max(0.005, float(genre_weights[g]) + float(delta))
        if _policy_enabled(world):
            for genre_name in list(genre_weights.keys()):
                genre_weights[genre_name] = max(
                    0.005,
                    float(genre_weights[genre_name]) * (0.8 + 2.2 * float(bucket.get("genre_bias", {}).get(genre_name, 1.0 / max(1, len(genre_weights))))),
                )
        bucket_id = str(bucket.get("bucket_id", ""))
        for genre_name in list(genre_weights.keys()):
            bucket_usage = float(world._concept_bucket_genre_usage_counts.get((bucket_id, genre_name), 0))
            genre_weights[genre_name] = max(0.005, float(genre_weights[genre_name]) / (1.0 + 0.40 * bucket_usage))
        if genre_hint and genre_hint in genre_weights:
            genre = genre_hint
        else:
            genre_weights = _normalise_dict_weights(genre_weights)
            genre = rng.choice(list(genre_weights.keys()), p=list(genre_weights.values()))
        tier_dist = _genre_tier_distribution(world, genre)
        if tier_dist is None:
            if current_mode() == "research":
                audit_fallback_hit(
                    "assembly.selection_weights",
                    f"missing:genre_tier_distribution.{genre}",
                    detail=f"selection_weights.genre_tier_distribution.{genre} is required in research mode",
                    mode="research",
                )
            tier_dist = _GENRE_TIER_DIST.get(genre)
        if tier_dist is not None:
            tier = rng.choice(PRODUCTION_TIERS, p=tier_dist / tier_dist.sum())
        else:
            base_tier = _normalise_dict_weights({k: float(v) for k, v in TIER_WEIGHTS.items()})
            tier = rng.choice(list(base_tier.keys()), p=list(base_tier.values()))

    country_weights = {k: float(v) for k, v in COUNTRY_WEIGHTS.items()}
    for c, multiplier in getattr(world, "country_weight_overrides", {}).items():
        if c in country_weights:
            country_weights[c] = max(1e-6, country_weights[c] * float(multiplier))
    if _policy_enabled(world):
        for country_name in list(country_weights.keys()):
            country_weights[country_name] = max(
                1e-6,
                float(country_weights[country_name]) * (0.85 + 1.9 * float(bucket.get("country_bias", {}).get(country_name, 1.0 / max(1, len(country_weights))))),
            )
    country_weights = _normalise_dict_weights(country_weights)
    country = rng.choice(list(country_weights.keys()), p=list(country_weights.values()))
    language = COUNTRY_LANGUAGE.get(country, "English")

    if franchise is None and country not in _MAJOR_HUBS and tier in ("Epic", "A"):
        tier = "Mid"

    market = _COUNTRY_TO_MARKET.get(country, "Global")
    year_slate = _year_slate_for_concept(world, bucket_id=str(bucket.get("bucket_id", "")), market=market, tier=str(tier))
    month = int(rng.choice(range(1, 13), p=_seasonal_month_weights(world, str(genre), str(year_slate.get("release_season_bias", "")) if year_slate else None)))
    concept = {
        "movie_id": int(movie_id),
        "genre": genre,
        "tier": tier,
        "year": int(year),
        "country": country,
        "market": market,
        "language": language,
        "tone": _GENRE_TONE.get(genre, "neutral"),
        "month": month,
        "_world": world,
        "franchise": franchise,
        "franchise_bible": franchise_bible,
        "installment": installment,
        "is_writer_director": bool(rng.random() < _writer_director_probability(world, tier)),
        "year_bucket": str(bucket.get("bucket_id", "")),
        "year_slate": year_slate,
        "concept_pack_id": "",
        "concept_pack": None,
        "policy_targets": {
            "priority_motifs": list(year_slate.get("priority_motifs", [])) if isinstance(year_slate.get("priority_motifs"), list) else [],
            "trending_subgenres": list(year_slate.get("trending_subgenres", [])) if isinstance(year_slate.get("trending_subgenres"), list) else [],
            "motif_drift": list(year_slate.get("motif_drift", [])) if isinstance(year_slate.get("motif_drift"), list) else [],
            "release_season_bias": str(year_slate.get("release_season_bias", "")) if year_slate else "",
            "novelty_target": float(year_slate.get("novelty_target", 0.35) or 0.35) if year_slate else 0.35,
            "sequel_appetite": float(year_slate.get("sequel_appetite", 0.3) or 0.3) if year_slate else 0.3,
            "franchise_keyword_families": list(franchise_bible.get("keyword_families", [])) if isinstance(franchise_bible.get("keyword_families"), list) else [],
        },
        "selection_confidence": 0.42,
        "_rerank_candidates": [],
        "selection_mode": "heuristic" if franchise is None else "heuristic_franchise",
    }
    world._concept_country_usage_counts[country] += 1
    world._concept_bucket_country_usage_counts[(str(concept.get("year_bucket", "")), country)] += 1
    world._concept_bucket_total_counts[str(concept.get("year_bucket", ""))] += 1
    world._concept_bucket_genre_usage_counts[(str(concept.get("year_bucket", "")), str(genre))] += 1
    _log_selection_decision(
        world,
        stage="sample_movie_concept",
        concept=concept,
        chosen={"mode": "heuristic", "genre": genre, "tier": tier, "country": country},
        confidence=float(concept["selection_confidence"]),
        candidates=[],
        extra={"year_bucket": concept.get("year_bucket", ""), "genre_hint": genre_hint},
    )
    return concept


def pick_director(world: WorldState, concept: dict) -> int | None:
    if len(world.directors) == 0:
        return None
    genre = str(concept["genre"])
    tier = str(concept.get("tier", "Mid"))
    franchise = concept.get("franchise")
    franchise_bible = concept.get("franchise_bible") or _franchise_bible(franchise)
    year = int(concept.get("year", 2000))
    director_cfg = _director_selection_config(world)

    carryover_director_bias = float(franchise_bible.get("carryover_director_bias", 0.80) or 0.80) if isinstance(franchise_bible, dict) else 0.80
    if franchise and franchise.get("director_id") is not None and world.rng.random() < carryover_director_bias:
        return int(franchise["director_id"])

    dirs = _active_year_subset(world.directors, year)
    if "debut_year" in dirs.columns:
        span = year - dirs["debut_year"].fillna(1970).astype(int)
        soft = dirs[(span >= 0) & (span <= 60)]
        if len(soft) >= 3:
            dirs = soft

    weights = dirs["pop_weight"].astype(float).values.copy()
    if "genre_affinity" in dirs.columns:
        ga = dirs["genre_affinity"].fillna("").astype(str).str.lower()
        weights *= np.where(ga.str.contains(genre.lower(), regex=False).values, float(director_cfg["genre_match_boost"]), 1.0)

    movie_country = str(concept.get("country", ""))
    if movie_country and "nationality" in dirs.columns:
        nat = dirs["nationality"].fillna("").astype(str).values
        dir_country = np.array([_NATIONALITY_TO_COUNTRY.get(n, "") for n in nat])
        geo_boost = _geo_boost_for_tier(world, tier, 2.0) * float(director_cfg["geo_boost_scale"])
        weights *= np.where(dir_country == movie_country, max(float(director_cfg["geo_boost_floor"]), geo_boost), 1.0)

    dir_pids = dirs["person_id"].astype(int).values
    film_counts = np.array([world.director_film_count.get(int(pid), 0) for pid in dir_pids], dtype=float)
    director_recent = getattr(world, "director_recent", {}) or {}
    recent_window = np.array(
        [
            sum(1 for seen_year in director_recent.get(int(pid), []) if abs(int(seen_year) - year) <= 3)
            for pid in dir_pids
        ],
        dtype=float,
    )
    target_director_load = _prior_float(world, "target_director_load", 7.5, lo=2.0, hi=40.0)
    experience_bonus = np.clip(1.0 + 0.05 * np.log1p(np.minimum(film_counts, 8.0)), 1.0, 1.12)
    load_ratio = film_counts / max(target_director_load, 1.0)
    underused_bonus = np.where(
        film_counts <= max(1.0, target_director_load * 0.35),
        1.22,
        np.where(film_counts <= target_director_load, 1.06, 1.0),
    )
    load_decay = np.where(
        load_ratio > 9.0,
        0.08,
        np.where(
            load_ratio > 6.0,
            0.18,
            np.where(
                load_ratio > 3.5,
                0.42,
                np.where(load_ratio > 2.0, 0.72, 1.0),
            ),
        ),
    )
    recent_decay = np.where(recent_window > 0, np.maximum(0.08, 1.0 - 0.30 * recent_window), 1.0)
    usage_mult = _director_usage_multiplier(film_counts)
    weights *= experience_bonus
    weights *= underused_bonus
    weights *= load_decay
    weights *= usage_mult
    weights *= recent_decay

    shortlist = _shortlist_indices(
        weights,
        _shortlist_budget(world, "director", 24),
        world.rng,
        exploration_share=_prior_float(world, "director_exploration_share", 0.30, lo=0.0, hi=0.60),
    )
    if shortlist.size > 0 and shortlist.size < len(dirs):
        dirs = dirs.iloc[shortlist]
        weights = weights[shortlist]
        dir_pids = dirs["person_id"].astype(int).values

    risk_target, ambition_target, prestige_target = _concept_latent_targets(concept)
    alignment = np.ones(len(dirs), dtype=float)
    latent_lookup = getattr(world, "_latent_pid_to_idx", {}) or {}
    latent_indices = np.array([latent_lookup.get(int(pid), -1) for pid in dir_pids], dtype=int)
    valid_latent = latent_indices >= 0
    director_risk_weight = _section_prior_float(world, "selection_weights", "director_risk_weight", 0.40, lo=0.0, hi=1.0)
    director_ambition_weight = _section_prior_float(world, "selection_weights", "director_ambition_weight", 0.35, lo=0.0, hi=1.0)
    director_prestige_weight = _section_prior_float(world, "selection_weights", "director_prestige_weight", 0.25, lo=0.0, hi=1.0)
    director_align_base = _section_prior_float(world, "selection_weights", "director_alignment_base", 0.70, lo=0.1, hi=2.0)
    director_align_scale = _section_prior_float(world, "selection_weights", "director_alignment_scale", 0.80, lo=0.0, hi=3.0)
    director_csv_base = _section_prior_float(world, "selection_weights", "director_csv_base", 0.80, lo=0.1, hi=2.0)
    director_csv_scale = _section_prior_float(world, "selection_weights", "director_csv_scale", 0.50, lo=0.0, hi=3.0)
    if valid_latent.any():
        valid_idx = latent_indices[valid_latent]
        risk_align = 1.0 - np.abs(world._latent_risk[valid_idx] - risk_target)
        ambition_align = 1.0 - np.abs(world._latent_ambition[valid_idx] - ambition_target)
        prestige_source = getattr(world, "_latent_public_reputation", None)
        if prestige_source is not None:
            prestige_align = 1.0 - np.abs(prestige_source[valid_idx] - prestige_target)
        else:
            prestige_align = np.array([
                1.0 - abs(_safe01(get_person_latent(world, int(pid)).get("public_reputation"), 0.5) - prestige_target)
                for pid in dir_pids[valid_latent]
            ], dtype=float)
        alignment[valid_latent] = (
            director_risk_weight * risk_align
            + director_ambition_weight * ambition_align
            + director_prestige_weight * prestige_align
        )
    if (~valid_latent).any():
        for arr_idx, pid in zip(np.flatnonzero(~valid_latent), dir_pids[~valid_latent]):
            lv = get_person_latent(world, int(pid))
            risk_align = 1.0 - abs(_safe01(lv.get("risk_tolerance"), 0.5) - risk_target)
            ambition_align = 1.0 - abs(_safe01(lv.get("artistic_ambition"), 0.5) - ambition_target)
            prestige_align = 1.0 - abs(_safe01(lv.get("public_reputation"), 0.5) - prestige_target)
            alignment[arr_idx] = (
                director_risk_weight * risk_align
                + director_ambition_weight * ambition_align
                + director_prestige_weight * prestige_align
            )
    weights *= np.clip(director_align_base + director_align_scale * alignment, 0.35, 1.75)

    csv_target = np.asarray(_concept_csv_target(concept), dtype=np.float32)
    csv_target_norm = float(np.linalg.norm(csv_target))
    if csv_target_norm > 1e-10:
        csv_target = csv_target / csv_target_norm
    csv_alignment = np.ones(len(dirs), dtype=float)
    if valid_latent.any() and getattr(world, "_latent_csv_normed", None) is not None:
        csv_alignment[valid_latent] = np.clip(
            world._latent_csv_normed[latent_indices[valid_latent]] @ csv_target,
            0.0,
            1.0,
        )
    if (~valid_latent).any():
        for arr_idx, pid in zip(np.flatnonzero(~valid_latent), dir_pids[~valid_latent]):
            csv_alignment[arr_idx] = _cosine_sim(
                get_person_latent(world, int(pid)).get("creative_style_vector", [0.0] * 8),
                csv_target,
            )
    weights *= np.clip(director_csv_base + director_csv_scale * csv_alignment, 0.60, 1.40)

    pc_mult = _director_company_multiplier(world, dir_pids, year, tier, genre)
    if len(pc_mult):
        weights *= np.where(pc_mult > 1.0, pc_mult * float(director_cfg["company_multiplier_rescale"]), 1.0)

    if _policy_enabled(world):
        stage_values = dirs["career_stage"].fillna("prime").astype(str).str.lower().values if "career_stage" in dirs.columns else None
        style_values = dirs["style_tags"].fillna("").astype(str).values if "style_tags" in dirs.columns else None
        weights *= _person_policy_rule_multiplier(
            world,
            dir_pids,
            stage_values,
            genre,
            style_tags=style_values,
        )
        pack = concept.get("concept_pack") or {}
        strategy_tag = str(pack.get("company_strategy_tag") or concept.get("policy_targets", {}).get("company_strategy_tag", ""))
        if strategy_tag == "event_franchise":
            weights *= np.where(
                dirs["pop_weight"].astype(float).values
                >= np.quantile(dirs["pop_weight"].astype(float).values, float(director_cfg["event_franchise_pop_quantile"])),
                float(director_cfg["event_franchise_pop_boost"]),
                1.0,
            )
        elif strategy_tag == "prestige_drama":
            weights *= np.where(
                alignment >= float(director_cfg["prestige_drama_alignment_threshold"]),
                float(director_cfg["prestige_drama_alignment_boost"]),
                1.0,
            )

    probs = normalize_weights(weights)
    if float(np.sum(probs)) <= 0:
        return None
    chosen = int(world.rng.choice(len(dirs), p=probs))
    did = int(dirs.iloc[chosen]["person_id"])
    top_idx = np.argsort(probs)[::-1][:5]
    confidence = confidence_from_scores(probs.tolist())
    _log_selection_decision(
        world,
        stage="pick_director",
        concept=concept,
        chosen={"person_id": did},
        confidence=confidence,
        candidates=[
            {
                "person_id": int(dir_pids[idx]),
                "prob": round(float(probs[idx]), 5),
            }
            for idx in top_idx
        ],
    )
    world.director_film_count[did] += 1
    if hasattr(world, "director_recent"):
        world.director_recent[did].append(int(year))
    return did


def pick_co_director(world: WorldState, concept: dict, primary_dir_id: int) -> int | None:
    tier = str(concept.get("tier", "Mid"))
    genre = str(concept.get("genre", "Drama"))
    director_cfg = _director_selection_config(world)
    prob = float(director_cfg["co_director_probability_by_tier"].get(tier, 0.02))
    if world.rng.random() >= prob or len(world.directors) < 2:
        return None
    dirs = _active_year_subset(world.directors, int(concept.get("year", 2000)))
    weights = dirs["pop_weight"].astype(float).values.copy()
    if "genre_affinity" in dirs.columns:
        ga = dirs["genre_affinity"].fillna("").astype(str).str.lower()
        weights *= np.where(ga.str.contains(genre.lower(), regex=False).values, float(director_cfg["genre_match_boost"]), 1.0)
    weights[dirs["person_id"].astype(int).values == int(primary_dir_id)] = 0.0
    probs = normalize_weights(weights)
    if float(np.sum(probs)) <= 0:
        return None
    chosen = int(world.rng.choice(len(dirs), p=probs))
    did = int(dirs.iloc[chosen]["person_id"])
    world.director_film_count[did] += 1
    return did


def pick_companies(world: WorldState, concept: dict, director_id: int) -> list[dict]:
    comps = world.companies
    tier = str(concept["tier"])
    genre = str(concept["genre"])
    year = int(concept["year"])
    franchise = concept.get("franchise")
    franchise_bible = concept.get("franchise_bible") or _franchise_bible(franchise)
    company_cfg = _company_selection_config(world)

    if franchise and franchise.get("company_ids"):
        rows = []
        for cid in franchise["company_ids"]:
            if int(cid) in set(comps["company_id"].astype(int).tolist()):
                rows.append({"company_id": int(cid), "role": "Production" if not rows else "Co-Production"})
        if rows:
            return rows

    active = comps
    if "founded_year" in active.columns:
        active = active[active["founded_year"].fillna(0).astype(int) <= year]
    if "defunct_year" in active.columns:
        active = active[active["defunct_year"].isna() | (active["defunct_year"].fillna(2100).astype(float) >= float(year))]
    if len(active) < 3:
        active = comps

    n_companies = int(world.rng.choice([1, 1, 2, 2, 3]))
    n_companies = min(n_companies, len(active))
    weights = active["pop_weight"].astype(float).values.copy()
    company_tier_match_boost = _section_prior_float(world, "selection_weights", "company_tier_match_boost", 8.0, lo=1.0, hi=20.0)
    company_tier_mismatch_penalty = _section_prior_float(world, "selection_weights", "company_tier_mismatch_penalty", 0.2, lo=0.01, hi=2.0)
    company_genre_match_boost = _section_prior_float(world, "selection_weights", "company_genre_match_boost", 15.0, lo=1.0, hi=30.0)
    company_genre_mismatch_penalty = _section_prior_float(world, "selection_weights", "company_genre_mismatch_penalty", 0.25, lo=0.01, hi=2.0)

    if "tier" in active.columns:
        tier_match = active["tier"].astype(str).values == tier
        weights *= np.where(tier_match, company_tier_match_boost, company_tier_mismatch_penalty)
    if "specialty_genres" in active.columns:
        sg = active["specialty_genres"].fillna("").astype(str).str.lower()
        match = sg.str.contains(genre.lower(), regex=False).values
        weights *= np.where(match, company_genre_match_boost, company_genre_mismatch_penalty)

    target_company_load = _prior_float(world, "target_company_load", 10.5, lo=2.0, hi=60.0)
    cids_all = active["company_id"].astype(int).values
    company_film_counts = np.array([world.company_film_count.get(int(cid), 0) for cid in cids_all], dtype=float)
    company_recent = getattr(world, "company_recent", {}) or {}
    company_recent_window = np.array(
        [
            sum(1 for seen_year in company_recent.get(int(cid), []) if abs(int(seen_year) - year) <= 4)
            for cid in cids_all
        ],
        dtype=float,
    )
    weights *= _company_usage_multiplier(
        company_film_counts,
        company_recent_window,
        target_load=target_company_load,
    )

    shortlist = _shortlist_indices(
        weights,
        _shortlist_budget(world, "company", 18),
        world.rng,
        exploration_share=_prior_float(world, "company_primary_exploration_share", 0.35, lo=0.0, hi=0.60),
    )
    if shortlist.size > 0 and shortlist.size < len(active):
        active = active.iloc[shortlist]
        weights = weights[shortlist]

    cids = active["company_id"].astype(int).values
    tier_idx = TIER_TO_LATENT_IDX.get(tier, 2)
    risk_target, _, prestige_target = _concept_latent_targets(concept)
    align = np.ones(len(active), dtype=float)
    concept_genre_basis = project_genres_to_company_basis([genre])
    company_latent_lookup = getattr(world, "_pc_c_idx", {}) or {}
    company_latent_indices = np.array([company_latent_lookup.get(int(cid), -1) for cid in cids], dtype=int)
    valid_company_latent = company_latent_indices >= 0
    company_risk_weight = _section_prior_float(world, "selection_weights", "company_risk_weight", 0.28, lo=0.0, hi=1.0)
    company_prestige_weight = _section_prior_float(world, "selection_weights", "company_prestige_weight", 0.22, lo=0.0, hi=1.0)
    company_focus_weight = _section_prior_float(world, "selection_weights", "company_focus_weight", 0.30, lo=0.0, hi=1.0)
    company_genre_fit_weight = _section_prior_float(world, "selection_weights", "company_genre_fit_weight", 0.20, lo=0.0, hi=1.0)
    company_align_base = _section_prior_float(world, "selection_weights", "company_alignment_base", 0.65, lo=0.1, hi=2.0)
    company_align_scale = _section_prior_float(world, "selection_weights", "company_alignment_scale", 0.90, lo=0.0, hi=3.0)
    if valid_company_latent.any() and getattr(world, "_pc_ready", False):
        valid_idx = company_latent_indices[valid_company_latent]
        risk_align = 1.0 - np.abs(world._pc_c_risk[valid_idx] - risk_target)
        prestige_align = 1.0 - np.abs(world._pc_c_prestige[valid_idx] - prestige_target)
        focus = world._pc_c_budget[valid_idx, tier_idx]
        genre_fit = np.clip(world._pc_c_genre[valid_idx] @ concept_genre_basis, 0.0, 1.0)
        align[valid_company_latent] = (
            company_risk_weight * risk_align
            + company_prestige_weight * prestige_align
            + company_focus_weight * focus
            + company_genre_fit_weight * genre_fit
        )
    if (~valid_company_latent).any():
        for arr_idx, cid in zip(np.flatnonzero(~valid_company_latent), cids[~valid_company_latent]):
            lv = world.company_latent.get(int(cid), {}) if hasattr(world, "company_latent") else {}
            risk_align = 1.0 - abs(_safe01(lv.get("risk_appetite"), 0.5) - risk_target)
            prestige_align = 1.0 - abs(_safe01(lv.get("prestige_score"), 0.5) - prestige_target)
            bbp = lv.get("budget_tier_focus") if isinstance(lv, dict) else None
            focus = _safe01(bbp[tier_idx], 0.5) if isinstance(bbp, list) and len(bbp) > tier_idx else 0.5
            gp = canonical_company_genre_vector(lv.get("genre_portfolio") if isinstance(lv, dict) else None)
            genre_fit = float(np.clip(gp @ concept_genre_basis, 0.0, 1.0))
            align[arr_idx] = (
                company_risk_weight * risk_align
                + company_prestige_weight * prestige_align
                + company_focus_weight * focus
                + company_genre_fit_weight * genre_fit
            )
    weights *= np.clip(company_align_base + company_align_scale * align, 0.25, 1.80)
    if _policy_enabled(world):
        strategy_tag = str(
            franchise_bible.get("company_strategy_tag", "")
            or (concept.get("concept_pack") or {}).get("company_strategy_tag")
            or concept.get("policy_targets", {}).get("company_strategy_tag", "")
        )
        if strategy_tag:
            strategy_map = getattr(world, "world_policy", {}).get("company_strategy_assignments", {})
            strategy_match = np.array(
                [str(strategy_map.get(str(cid), "")) == strategy_tag for cid in cids],
                dtype=bool,
            )
            weights *= np.where(strategy_match, float(company_cfg["strategy_match_boost"]), 1.0)
        bucket = _policy_bucket(world, year)
        market_bias = bucket.get("market_bias", {})
        if isinstance(market_bias, dict) and "country" in active.columns:
            company_markets = np.array([_COUNTRY_TO_MARKET.get(str(value), "Global") for value in active["country"].fillna("").astype(str).values])
            weights *= np.array(
                [
                    float(company_cfg["market_bias_base"]) + float(company_cfg["market_bias_scale"]) * float(market_bias.get(market, 0.2))
                    for market in company_markets
                ],
                dtype=float,
            )
    weights = normalize_weights(weights)
    if len(active) == 0 or float(np.sum(weights)) <= 0:
        return []

    primary_shortlist = _shortlist_indices(
        weights,
        min(len(active), _shortlist_budget(world, "company", 18)),
        world.rng,
        exploration_share=_prior_float(world, "company_primary_exploration_share", 0.35, lo=0.0, hi=0.60),
    )
    if primary_shortlist.size == 0:
        return []
    primary_probs = normalize_weights(weights[primary_shortlist])
    chosen_idx = int(world.rng.choice(primary_shortlist, p=primary_probs))
    chosen_indices = [chosen_idx]
    first_cid = int(active.iloc[chosen_idx]["company_id"])

    if n_companies > 1:
        family = getattr(world, "_merge_families", {}) or getattr(world, "company_family", {})
        first_family = family.get(first_cid, set())
        partner_weights = weights.copy()
        for i, cid in enumerate(cids):
            # On-demand C-C scoring: replaces precomputed company_affinity/rivalry index
            cc_aff = world.compute_cc_affinity(first_cid, int(cid))
            if cc_aff > 0:
                partner_weights[i] *= (1.0 + float(company_cfg["partner_affinity_scale"]) * cc_aff)
            cc_riv = world.compute_cc_rivalry(first_cid, int(cid))
            if cc_riv > 0:
                partner_weights[i] *= max(float(company_cfg["partner_rivalry_penalty_floor"]), 1.0 - cc_riv)
            if int(cid) in first_family:
                partner_weights[i] *= float(company_cfg["family_boost"])
        for _ in range(n_companies - 1):
            pw = partner_weights.copy()
            for ci in chosen_indices:
                pw[ci] = 0.0
            if np.count_nonzero(pw > 0) == 0:
                break
            sl = _shortlist_indices(
                pw,
                min(len(active), _shortlist_budget(world, "company", 18)),
                world.rng,
                exploration_share=_prior_float(world, "company_secondary_exploration_share", 0.40, lo=0.0, hi=0.60),
            )
            if sl.size == 0:
                break
            probs = normalize_weights(pw[sl])
            idx = int(world.rng.choice(sl, p=probs))
            if idx not in chosen_indices:
                chosen_indices.append(idx)

    rows = []
    for pos, idx in enumerate(chosen_indices):
        cid = int(active.iloc[idx]["company_id"])
        world.company_film_count[cid] += 1
        if hasattr(world, "company_recent"):
            world.company_recent[cid].append(int(year))
        rows.append({"company_id": cid, "role": "Production" if pos == 0 else "Co-Production"})
    _log_selection_decision(
        world,
        stage="pick_companies",
        concept=concept,
        chosen={"company_ids": [int(row["company_id"]) for row in rows]},
        confidence=confidence_from_scores(weights.tolist()),
        candidates=[
            {"company_id": int(cids[idx]), "prob": round(float(weights[idx]), 5)}
            for idx in np.argsort(weights)[::-1][:5]
        ],
        extra={"director_id": int(director_id) if director_id is not None else None},
    )
    return rows


def _character_name_bank_cache(world: WorldState) -> dict[str, Any]:
    cache = getattr(world, "_character_name_bank_cache", None)
    if cache is not None:
        return cache

    bank = getattr(world, "character_bank", None)
    if not (isinstance(bank, pd.DataFrame) and len(bank) > 0 and "character_name" in bank.columns):
        cb_path = Path(getattr(world, "base_dir", ".")) / "entities" / "character_bank.csv"
        if cb_path.exists():
            try:
                bank = pd.read_csv(cb_path)
                if "character_name" in bank.columns and len(bank) > 0:
                    world.character_bank = bank
            except Exception:
                bank = None

    if not (isinstance(bank, pd.DataFrame) and len(bank) > 0 and "character_name" in bank.columns):
        cache = {
            "size": 0,
            "has_archetype": False,
            "has_gender": False,
            "clean_names": np.array([], dtype=object),
            "raw_names": np.array([], dtype=object),
            "archetypes": np.array([], dtype=object),
            "genders": np.array([], dtype=object),
            "all_indices": np.array([], dtype=np.int32),
            "pool_cache": {},
        }
        world._character_name_bank_cache = cache
        return cache

    raw_names = np.array(
        [
            str(value).strip()
            for value in bank["character_name"].astype(str).values
        ],
        dtype=object,
    )
    clean_names = np.array(
        [sanitize_character_name(value) for value in raw_names],
        dtype=object,
    )
    has_archetype = "archetype" in bank.columns
    has_gender = "gender" in bank.columns
    archetypes = (
        bank["archetype"].fillna("").astype(str).to_numpy(dtype=object)
        if has_archetype
        else np.array([""] * len(raw_names), dtype=object)
    )
    genders = (
        bank["gender"].fillna("").astype(str).str.upper().to_numpy(dtype=object)
        if has_gender
        else np.array([""] * len(raw_names), dtype=object)
    )
    cache = {
        "size": int(len(raw_names)),
        "has_archetype": bool(has_archetype),
        "has_gender": bool(has_gender),
        "clean_names": clean_names,
        "raw_names": raw_names,
        "archetypes": archetypes,
        "genders": genders,
        "all_indices": np.arange(int(len(raw_names)), dtype=np.int32),
        "pool_cache": {},
    }
    world._character_name_bank_cache = cache
    return cache


def _character_name_pool_indices(
    world: WorldState,
    name_cache: dict[str, Any],
    genre: str,
    archetype: str,
    gender: str,
) -> np.ndarray:
    size = int(name_cache["size"])
    if size <= 0:
        return np.array([], dtype=np.int32)
    genre_key = str(genre or "").lower().strip()
    archetype_key = str(archetype or "")
    gender_key = str(gender or "").upper() if str(gender or "").upper() in ("M", "F") else ""
    pool_cache = name_cache.setdefault("pool_cache", {})
    cache_key = (genre_key, archetype_key, gender_key)
    cached = pool_cache.get(cache_key)
    if cached is not None:
        return cached

    def _index_maps() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        archetype_index = name_cache.get("archetype_index")
        gender_index = name_cache.get("gender_index")
        if archetype_index is not None and gender_index is not None:
            return archetype_index, gender_index
        archetype_lists: dict[str, list[int]] = defaultdict(list)
        gender_lists: dict[str, list[int]] = defaultdict(list)
        if bool(name_cache["has_archetype"]):
            for idx, value in enumerate(name_cache["archetypes"]):
                archetype_lists[str(value)].append(int(idx))
        if bool(name_cache["has_gender"]):
            for idx, value in enumerate(name_cache["genders"]):
                gender_lists[str(value)].append(int(idx))
        archetype_index = {
            key: np.asarray(values, dtype=np.int32)
            for key, values in archetype_lists.items()
        }
        gender_index = {
            key: np.asarray(values, dtype=np.int32)
            for key, values in gender_lists.items()
        }
        name_cache["archetype_index"] = archetype_index
        name_cache["gender_index"] = gender_index
        return archetype_index, gender_index

    all_indices = name_cache["all_indices"]
    base_indices = all_indices
    if bool(name_cache["has_archetype"]):
        archetype_index, _gender_index = _index_maps()
        hints = _genre_archetype_candidates(world).get(genre_key, [])
        if hints:
            hint_values: list[str] = []
            seen_hint_values: set[str] = set()
            for value in list(hints) + [archetype_key]:
                value_s = str(value)
                if value_s and value_s not in seen_hint_values:
                    hint_values.append(value_s)
                    seen_hint_values.add(value_s)
            parts = [
                archetype_index[value]
                for value in hint_values
                if value in archetype_index and archetype_index[value].size > 0
            ]
            hinted_indices = np.concatenate(parts).astype(np.int32, copy=False) if parts else np.array([], dtype=np.int32)
            if int(hinted_indices.size) >= 20:
                base_indices = hinted_indices
        else:
            archetype_indices = archetype_index.get(archetype_key)
            if archetype_indices is not None and int(archetype_indices.size) > 0:
                base_indices = archetype_indices
    if bool(name_cache["has_gender"]) and gender_key:
        _archetype_index, gender_index = _index_maps()
        gender_indices = gender_index.get(gender_key)
        if gender_indices is not None and int(gender_indices.size) > 0:
            if base_indices is all_indices:
                candidate_indices = gender_indices
            else:
                candidate_indices = np.intersect1d(base_indices, gender_indices, assume_unique=True).astype(np.int32, copy=False)
            if int(candidate_indices.size) >= 10:
                base_indices = candidate_indices

    pool_indices = base_indices.astype(np.int32, copy=False)
    pool_cache[cache_key] = pool_indices
    return pool_indices


def _pick_character_name(genre: str, archetype: str, slot: int, world: WorldState, gender: str = "M", used_names: set | None = None) -> str:
    if used_names is None:
        used_names = set()
    if not hasattr(world, "_used_char_names_global"):
        world._used_char_names_global = set()
    global_used = world._used_char_names_global
    name_cache = _character_name_bank_cache(world)

    def _select_from_indices(indices: np.ndarray, *, prefer_clean: bool) -> str | None:
        if indices.size == 0:
            return None
        name_source = name_cache["clean_names"] if prefer_clean else name_cache["raw_names"]
        clean_source = name_cache["clean_names"]

        def _try_idx(idx_i: int, *, respect_global: bool) -> str | None:
            name = str(name_source[idx_i] or "")
            if not name or name.lower() == "nan":
                return None
            clean_name = str(clean_source[idx_i] or "")
            candidate = clean_name or name
            if not candidate or candidate in used_names:
                return None
            if respect_global and candidate in global_used:
                return None
            used_names.add(candidate)
            global_used.add(candidate)
            return candidate

        def _try_order(order: np.ndarray, *, respect_global: bool) -> str | None:
            for idx in order:
                picked_name = _try_idx(int(idx), respect_global=respect_global)
                if picked_name:
                    return picked_name
            return None

        probe_budget = name_cache.get("random_probe_budget")
        if probe_budget is None:
            probe_budget = int(_env_float("DATA_SYS_CHARACTER_NAME_RANDOM_PROBES", 96.0, 8.0, 512.0))
            name_cache["random_probe_budget"] = probe_budget
        if indices.size <= probe_budget:
            order = indices.copy()
            world.rng.shuffle(order)
        else:
            probe_pos = world.rng.randint(0, int(indices.size), size=probe_budget)
            order = indices[np.unique(probe_pos)]
            world.rng.shuffle(order)
        picked = _try_order(order, respect_global=True)
        if picked:
            return picked
        picked = _try_order(order, respect_global=False)
        if picked:
            return picked

        # Rare exhaustion path: avoid allocating/shuffling huge pools; scan from
        # a random offset, preserving the same validity rules.
        start = int(world.rng.randint(0, int(indices.size)))
        for respect_global in (True, False):
            for offset in range(int(indices.size)):
                picked = _try_idx(int(indices[(start + offset) % int(indices.size)]), respect_global=respect_global)
                if picked:
                    return picked
        return None

    if int(name_cache["size"]) > 0:
        pool_indices = _character_name_pool_indices(world, name_cache, genre, archetype, gender)
        picked = _select_from_indices(pool_indices, prefer_clean=True)
        if picked:
            return picked
        picked = _select_from_indices(pool_indices, prefer_clean=False)
        if picked:
            return picked

        if pool_indices.size != int(name_cache["size"]):
            all_indices = name_cache["all_indices"]
            picked = _select_from_indices(all_indices, prefer_clean=True)
            if picked:
                return picked
            picked = _select_from_indices(all_indices, prefer_clean=False)
            if picked:
                return picked
    genre_key = genre.lower().strip()
    genre_key = genre_key if genre_key in _GENRE_CHAR_NAMES else _GENRE_KEY_MAP.get(genre_key)
    if current_mode() != "research" and genre_key and genre_key in _GENRE_CHAR_NAMES:
        bank = _GENRE_CHAR_NAMES[genre_key]
        names = bank.get(gender, bank.get("N", [])) or next(iter(bank.values()))
        order = list(range(len(names)))
        world.rng.shuffle(order)
        for idx in order:
            cleaned = sanitize_character_name(names[idx])
            if cleaned not in used_names:
                used_names.add(cleaned)
                return cleaned
    fallback = f"Character_{slot}_{world.rng.randint(1000, 9999)}"
    if current_mode() == "research":
        audit_fallback_hit(
            "assembly.character_names",
            "missing:character_name",
            detail=(
                f"character selection exhausted artifact-backed name options in research mode "
                f"(genre={genre}, archetype={archetype}, bank_rows={len(bank)})"
            ),
            mode="research",
        )
    used_names.add(fallback)
    return fallback


def _pick_cast_fast(world: WorldState, concept: dict, director_id: int, max_retries: int = 3) -> tuple[list[dict], list[tuple[int, int]]]:
    year = int(concept["year"])
    cast_cfg = _cast_selection_config(world)
    cast_size = _sample_cast_size(world, concept)
    static = _build_actor_static_block(world, concept)
    selection = _get_selection_year_state(world, year)
    pids = static.pids
    n_cand = len(pids)
    director_pref_boost, avoid_mask = _director_edge_arrays(world, int(director_id) if director_id is not None else -1, pids, year)
    cast_shortlist_size = _shortlist_budget(world, "cast", 120)
    candidate_local_idx = static.candidate_idx

    if not hasattr(world, "_pid_to_gender"):
        world._pid_to_gender = {}
        if world.persons is not None and "gender" in world.persons.columns:
            _g_pids = world.persons["person_id"].astype(int).values
            _g_vals = world.persons["gender"].fillna("M").astype(str).values
            for _gi in range(len(_g_pids)):
                world._pid_to_gender[int(_g_pids[_gi])] = _g_vals[_gi]

    franchise = concept.get("franchise")
    franchise_bible = concept.get("franchise_bible") or _franchise_bible(franchise)
    franchise_pool = set(int(p) for p in franchise.get("cast_pool", [])) if franchise and franchise.get("movies_generated", 0) > 0 else set()
    franchise_pool_mask = np.isin(pids, np.array(sorted(franchise_pool), dtype=int)) if franchise_pool else np.zeros(n_cand, dtype=bool)
    award_mask = selection.award_recent[candidate_local_idx] if len(candidate_local_idx) else np.zeros(0, dtype=bool)
    policy_rule_mult = static.policy_rule_mult
    chemistry_target = str(
        franchise_bible.get("cast_chemistry_target", "")
        or (concept.get("concept_pack") or {}).get("cast_chemistry_target")
        or concept.get("policy_targets", {}).get("cast_chemistry_target", "")
    )
    priority_focus = _priority_cast_indices(
        static,
        director_pref_boost,
        award_mask,
        franchise_pool_mask,
        max(cast_shortlist_size, 64),
    )
    base_focus = static.base_focus_idx
    default_focus_cap = max(cast_shortlist_size * 16, 2200)
    base_focus_cap = int(_env_float("DATA_SYS_CAST_FOCUS_CAP", float(default_focus_cap), 256.0, float(default_focus_cap)))
    target_actor_load = _prior_float(world, "target_actor_load", 6.0, lo=2.0, hi=20.0)
    top_star_slot_penalty = _env_float("V16_TOP_STAR_SLOT_PENALTY", 0.82, 0.5, 1.0)
    slot_exploration_empty = max(0.48, _prior_float(world, "cast_slot_exploration_empty", 0.35, lo=0.0, hi=0.60))
    slot_exploration_filled = max(0.42, _prior_float(world, "cast_slot_exploration_filled", 0.30, lo=0.0, hi=0.60))
    community_match_mult = _prior_float(world, "cast_community_match_multiplier", 1.5, lo=1.0, hi=3.0)
    unused_boost = float(cast_cfg["unused_actor_boost"])
    award_recent_boost = float(cast_cfg["award_recent_boost"])
    agency_match_boost = float(cast_cfg["agency_match_boost"])
    gender_novelty_boost = float(cast_cfg["gender_novelty_boost"])
    nationality_novelty_boost = float(cast_cfg["nationality_novelty_boost"])
    tag_similarity_penalty = float(cast_cfg["tag_similarity_penalty"])
    franchise_pool_boost = float(cast_cfg["franchise_pool_base_boost"])
    star_vehicle_slot0_boost = float(cast_cfg["star_vehicle_slot0_boost"])
    prestige_pairing_boost = float(cast_cfg["prestige_pairing_boost"])
    volatile_ensemble_boost = float(cast_cfg["volatile_ensemble_boost"])
    balanced_ensemble_boost = float(cast_cfg["balanced_ensemble_boost"])
    award_mask_any = bool(award_mask.any())
    franchise_pool_any = bool(franchise_pool)

    cast: list[dict] = []
    competition_pairs: list[tuple[int, int]] = []
    slot0_snapshot: list[dict[str, Any]] = []
    for _attempt in range(max_retries):
        cast.clear()
        competition_pairs.clear()
        slot0_snapshot = []
        attempt_film_count = selection.film_count.copy()
        attempt_recent_window = selection.recent_window.copy()
        attempt_yearly_workload = selection.yearly_workload.copy()
        attempt_unused_flags = selection.unused_flags.copy()
        ensemble = CastEnsembleState()
        ensemble.friend_frontier_vec = np.zeros(n_cand, dtype=np.float32)
        ensemble.rival_penalty_vec = np.ones(n_cand, dtype=np.float32)
        ensemble.blocked_local_mask = np.zeros(n_cand, dtype=bool)

        for slot in range(cast_size):
            frontier_candidates = ensemble.frontier_local_idx | ensemble.rival_local_idx
            if frontier_candidates:
                frontier_arr = np.fromiter(frontier_candidates, dtype=np.int32)
                candidate_focus = np.unique(np.concatenate([base_focus, priority_focus, frontier_arr]))
            elif priority_focus.size:
                candidate_focus = np.unique(np.concatenate([base_focus, priority_focus]))
            else:
                candidate_focus = base_focus
            candidate_focus = _cap_focus_indices(static.immutable_scores, candidate_focus.astype(np.int32, copy=False), base_focus_cap)
            if candidate_focus.size == 0:
                break

            focus_local = candidate_local_idx[candidate_focus]
            film_count = attempt_film_count[focus_local]
            recent_window = attempt_recent_window[focus_local]
            yearly_workload = attempt_yearly_workload[focus_local]
            unused_flags = attempt_unused_flags[focus_local]

            focus_scores = static.immutable_scores[candidate_focus].copy()
            experience_bonus = np.clip(1.0 + 0.04 * np.log1p(np.minimum(film_count, 10.0)), 1.0, 1.10)
            if slot >= 2:
                experience_bonus = np.minimum(experience_bonus, 1.04)
            usage_mult = _cast_usage_multiplier(
                film_count,
                slot,
                target_load=target_actor_load,
            )
            recent_decay = np.where(recent_window > 0, np.maximum(0.15, 1.0 - 0.22 * recent_window), 1.0)
            focus_scores *= experience_bonus
            focus_scores *= usage_mult
            focus_scores *= recent_decay
            cooldown_mult = np.where(
                recent_window > 0,
                np.maximum(static.cooldown_floor[candidate_focus], 1.0 - static.cooldown_decay[candidate_focus] * recent_window),
                1.0,
            )
            focus_scores *= cooldown_mult
            focus_scores *= np.where(yearly_workload >= static.yearly_max[candidate_focus], 0.1, 1.0)
            focus_scores *= director_pref_boost[candidate_focus]
            focus_scores[avoid_mask[candidate_focus]] = 0.0

            if ensemble.blocked_local_mask is not None and ensemble.blocked_local_idx:
                blocked_focus = ensemble.blocked_local_mask[candidate_focus]
                focus_scores[blocked_focus] = 0.0

            if slot == 0:
                focus_scores[unused_flags] *= max(1.40, unused_boost * 0.90)
            elif slot == 1:
                focus_scores[unused_flags] *= max(1.55, unused_boost)
            else:
                focus_scores[unused_flags] *= max(1.75, unused_boost * 1.10)
            if slot >= 1:
                focus_scores[film_count >= 180] *= 0.40
                focus_scores[film_count >= 240] *= 0.22
            if slot >= 3:
                focus_scores[film_count >= 40] *= 0.62
                focus_scores[film_count >= 90] *= 0.55
            if slot >= 5:
                focus_scores[film_count >= 20] *= 0.58
            if slot >= 4:
                focus_scores[static.top_star_mask[candidate_focus]] *= top_star_slot_penalty
            if award_mask_any:
                focus_scores[award_mask[candidate_focus] & (focus_scores > 0)] *= award_recent_boost
            if franchise_pool_any:
                carryover_cast_bias = float(franchise_bible.get("carryover_cast_bias", 0.66) or 0.66) if isinstance(franchise_bible, dict) else 0.66
                focus_scores[franchise_pool_mask[candidate_focus] & (focus_scores > 0)] *= franchise_pool_boost + carryover_cast_bias
            if _policy_enabled(world):
                if policy_rule_mult is not None:
                    focus_scores *= policy_rule_mult[candidate_focus]
                if chemistry_target == "star_vehicle" and slot == 0:
                    focus_scores[static.top_star_mask[candidate_focus]] *= star_vehicle_slot0_boost
                elif chemistry_target == "prestige_pairing" and slot <= 1:
                    focus_scores[np.isin(static.stage_vals[candidate_focus], np.array(["prime", "veteran", "legend"], dtype=object))] *= prestige_pairing_boost
                elif chemistry_target == "volatile_ensemble" and slot >= 1:
                    focus_scores[np.isin(static.stage_vals[candidate_focus], np.array(["rising", "prime"], dtype=object))] *= volatile_ensemble_boost
                elif chemistry_target == "balanced_ensemble" and slot >= 2:
                    focus_scores[np.isin(static.stage_vals[candidate_focus], np.array(["rising", "veteran"], dtype=object))] *= balanced_ensemble_boost

            if ensemble.cast_ids:
                focus_pids = pids[candidate_focus]
                if ensemble.agencies:
                    agency_match = np.isin(static.cand_agencies[candidate_focus], list(ensemble.agencies))
                    focus_scores[(focus_scores > 0) & agency_match] *= agency_match_boost
                if ensemble.communities:
                    community_match = np.isin(static.cand_communities[candidate_focus], list(ensemble.communities))
                    focus_scores[(focus_scores > 0) & community_match] *= community_match_mult
                if ensemble.genders and static.actor_genders is not None:
                    gender_novel = ~np.isin(static.actor_genders[candidate_focus], list(ensemble.genders))
                    focus_scores[(focus_scores > 0) & gender_novel] *= gender_novelty_boost
                if len(ensemble.nationalities) >= 2 and static.actor_nationalities is not None:
                    nat_novel = ~np.isin(static.actor_nationalities[candidate_focus], list(ensemble.nationalities))
                    focus_scores[(focus_scores > 0) & nat_novel] *= nationality_novelty_boost
                if ensemble.tag_bitmasks and static.actor_tag_bitmasks is not None:
                    cand_bm = static.actor_tag_bitmasks[candidate_focus]
                    forbidden_mask = (
                        ensemble.blocked_local_mask[candidate_focus]
                        if ensemble.blocked_local_mask is not None
                        else np.array([int(pid) in ensemble.forbidden for pid in focus_pids], dtype=bool)
                    )
                    has_tags = cand_bm > np.uint64(0)
                    eligible = (focus_scores > 0) & has_tags & ~forbidden_mask
                    if eligible.any():
                        max_jaccard = np.zeros(len(candidate_focus), dtype=np.float32)
                        for cbm in ensemble.tag_bitmasks:
                            inter = _popcount_vec(cand_bm & cbm)
                            union = _popcount_vec(cand_bm | cbm)
                            jaccard = np.where(union > 0, inter.astype(np.float32) / union.astype(np.float32), 0.0)
                            np.maximum(max_jaccard, jaccard, out=max_jaccard)
                        focus_scores[eligible & (max_jaccard > 0.5)] *= tag_similarity_penalty
                fb_mask = np.zeros(len(candidate_focus), dtype=bool)
                if ensemble.friend_frontier_vec is not None and ensemble.frontier_local_idx:
                    friend_boost = ensemble.friend_frontier_vec[candidate_focus]
                    fb_mask = friend_boost > 0
                    focus_scores[fb_mask] *= friend_boost[fb_mask]
                if ensemble.rival_penalty_vec is not None and ensemble.rival_local_idx:
                    rival_penalty = ensemble.rival_penalty_vec[candidate_focus]
                    focus_scores *= rival_penalty
                need_similarity = np.flatnonzero((focus_scores >= static._sim_threshold) & ~fb_mask)
                if need_similarity.size > 0 and static.li_arr is not None and ensemble.latent_anchor_ids:
                    ns_li = static.li_arr[candidate_focus[need_similarity]]
                    valid_li = ns_li >= 0
                    if valid_li.any():
                        local_targets = need_similarity[valid_li]
                        max_sim = latent_similarity_batch(world, ns_li[valid_li], np.asarray(ensemble.latent_anchor_ids[:3], dtype=int))
                        boost = max_sim > 0.1
                        if boost.any():
                            focus_scores[local_targets[boost]] *= (1.0 + max_sim[boost])

            valid_local = _shortlist_indices(
                focus_scores,
                cast_shortlist_size,
                world.rng,
                exploration_share=slot_exploration_filled if ensemble.cast_ids else slot_exploration_empty,
            )
            if valid_local.size == 0:
                break
            valid = candidate_focus[valid_local]
            exp = 2.5 if slot == 0 else (1.3 if slot == 1 else 0.7)
            probs = normalize_weights(focus_scores[valid_local] ** exp)
            if slot == 0:
                ranked = valid[np.argsort(probs)[::-1][:5]]
                slot0_snapshot = [
                    {
                        "person_id": int(pids[idx]),
                        "prob": round(float(probs[pos]), 5),
                    }
                    for pos, idx in zip(np.argsort(probs)[::-1][:5], ranked, strict=False)
                ]
            chosen_local = int(world.rng.choice(valid_local, p=probs))
            chosen_idx = int(candidate_focus[chosen_local])
            pid = int(pids[chosen_idx])

            if slot <= 1 and valid.size >= 2 and len(competition_pairs) < 4:
                order = valid[np.argsort(focus_scores[valid_local])[::-1]]
                for alt_idx in order:
                    if int(alt_idx) != chosen_idx:
                        competition_pairs.append((min(pid, int(pids[alt_idx])), max(pid, int(pids[alt_idx]))))
                        break

            cast.append({"person_id": pid, "billing_order": slot + 1})
            chosen_focus_local = int(candidate_local_idx[chosen_idx])
            attempt_film_count[chosen_focus_local] += 1.0
            attempt_recent_window[chosen_focus_local] += 1.0
            attempt_yearly_workload[chosen_focus_local] += 1.0
            attempt_unused_flags[chosen_focus_local] = False
            ensemble.add_actor(world, static, chosen_idx, year)

        if len(cast) >= max(1, cast_size - 1):
            break

    committed_ids = [int(row["person_id"]) for row in cast]
    if committed_ids:
        if not hasattr(world, "person_film_count") or world.person_film_count is None:
            world.person_film_count = Counter()
        if not hasattr(world, "person_recent") or world.person_recent is None:
            world.person_recent = {}
        for pid in committed_ids:
            world.person_film_count[pid] = int(world.person_film_count.get(pid, 0)) + 1
            world.person_recent.setdefault(pid, []).append(year)
            if getattr(world, "_yearly_workload", None) is not None:
                world._yearly_workload[(pid, year)] += 1
        selection.record_cast_selection(committed_ids, year)
        _log_selection_decision(
            world,
            stage="pick_cast",
            concept=concept,
            chosen={"cast_ids": committed_ids},
            confidence=0.5 if not slot0_snapshot else confidence_from_scores([entry["prob"] for entry in slot0_snapshot]),
            candidates=slot0_snapshot,
            extra={"director_id": int(director_id) if director_id is not None else None},
        )

    _assign_archetypes(cast, concept, world)
    return cast, competition_pairs


def pick_cast(world: WorldState, concept: dict, director_id: int, max_retries: int = 3) -> tuple[list[dict], list[tuple[int, int]]]:
    return _pick_cast_fast(world, concept, director_id, max_retries=max_retries)

    tier = str(concept["tier"])
    year = int(concept["year"])
    cast_size = _sample_cast_size(world, concept)
    static = _build_actor_static_block(world, concept)
    pids = static.pids
    n_cand = len(pids)
    graph = getattr(world, "graph", None)
    director_pref_boost, avoid_mask = _director_edge_arrays(world, int(director_id) if director_id is not None else -1, pids, year)
    cast_shortlist_size = _shortlist_budget(world, "cast", 120)

    cast: list[dict] = []
    competition_pairs: list[tuple[int, int]] = []
    cast_ids: set[int] = set()
    cast_id_list: list[int] = []
    forbidden: set[int] = set()

    # Pre-build pid→gender dict (cached on world, O(N_persons) once)
    if not hasattr(world, '_pid_to_gender'):
        world._pid_to_gender = {}
        if world.persons is not None and "gender" in world.persons.columns:
            _g_pids = world.persons["person_id"].astype(int).values
            _g_vals = world.persons["gender"].fillna("M").astype(str).values
            for _gi in range(len(_g_pids)):
                world._pid_to_gender[int(_g_pids[_gi])] = _g_vals[_gi]

    # Pre-build award-recent set for O(1) lookup per candidate
    award_recent_pids = set()
    if getattr(world, "person_award_wins", None):
        for _aw_pid, _aw_info in world.person_award_wins.items():
            if isinstance(_aw_info, dict):
                if year - int(_aw_info.get("year", 0) or 0) <= 3:
                    award_recent_pids.add(int(_aw_pid))
            elif isinstance(_aw_info, int) and _aw_info > 0:
                award_recent_pids.add(int(_aw_pid))

    # Pre-build franchise cast pool set
    franchise = concept.get("franchise")
    franchise_pool = set()
    if franchise and franchise.get("movies_generated", 0) > 0:
        franchise_pool = set(int(p) for p in franchise.get("cast_pool", []))
    award_mask = np.isin(pids, np.array(sorted(award_recent_pids), dtype=int)) if award_recent_pids else np.zeros(n_cand, dtype=bool)
    franchise_pool_mask = np.isin(pids, np.array(sorted(franchise_pool), dtype=int)) if franchise_pool else np.zeros(n_cand, dtype=bool)
    for _attempt in range(max_retries):
        cast.clear()
        competition_pairs.clear()
        cast_ids.clear()
        cast_id_list.clear()
        forbidden.clear()
        unused_flags = static.unused_flags.copy()

        # Pre-compute the static portion of scores ONCE.  All 15 multipliers
        # below are identical every slot — only per-slot adjustments differ.
        static_product = static.base_scores.copy()
        static_product *= static.career_mult
        static_product *= static.career_stage_mult
        static_product *= static.genre_mult
        static_product *= static.nationality_mult
        static_product *= static.style_mult
        static_product *= static.avoid_genre_mult
        static_product *= static.market_mult
        static_product *= static.collab_mult
        static_product *= static.controversy_mult
        static_product *= static.volatility_mult
        static_product *= static.csv_mult
        static_product *= static.company_aff_mult
        static_product *= static.cooldown_mult
        static_product *= static.workload_mult
        static_product *= director_pref_boost
        static_product[avoid_mask] = 0.0

        for slot in range(cast_size):
            scores = static_product.copy()

            if cast_ids or forbidden:
                for cid in cast_ids | forbidden:
                    idx = static.pid_to_idx.get(int(cid))
                    if idx is not None:
                        scores[idx] = 0.0

            if slot >= 2:
                scores[unused_flags] *= 1.5
            if slot >= 4:
                scores[static.top_star_mask] *= _env_float("V16_TOP_STAR_SLOT_PENALTY", 0.82, 0.5, 1.0)

            # Awards bump — vectorized via pre-built set
            if award_recent_pids:
                scores[award_mask & (scores > 0)] *= 1.5

            # Franchise cast pool boost — vectorized via pre-built set
            if franchise_pool:
                scores[franchise_pool_mask & (scores > 0)] *= 2.0

            candidate_focus = _expand_cast_focus_indices(
                world,
                static,
                scores,
                cast_id_list,
                year,
                cast_shortlist_size,
            )
            if candidate_focus.size == 0:
                break

            focus_scores = scores[candidate_focus].copy()

            if cast_id_list:
                focus_pids = pids[candidate_focus]

                cast_agency_set = {world.person_agency.get(cid) for cid in cast_id_list}
                cast_agency_set.discard(None)
                if cast_agency_set:
                    agency_match = np.isin(static.cand_agencies[candidate_focus], list(cast_agency_set))
                    focus_scores[(focus_scores > 0) & agency_match] *= 2.0

                cast_com_set = set()
                for cid in cast_id_list:
                    c = static.cand_communities[static.pid_to_idx[cid]] if cid in static.pid_to_idx else -1
                    if c >= 0:
                        cast_com_set.add(c)
                if cast_com_set:
                    com_match = np.isin(static.cand_communities[candidate_focus], list(cast_com_set))
                    focus_scores[(focus_scores > 0) & com_match] *= 1.5

                cast_friend_pids = set()
                for cid in cast_id_list:
                    if graph is not None:
                        for nbr, _w, _vf, _vt in graph.iter_friend_neighbors(int(cid), year):
                            cast_friend_pids.add(int(nbr))
                    else:
                        for nbr, _w, _vf, _vt in world._friend_adj_all.get(int(cid), []):
                            cast_friend_pids.add(int(nbr))

                cast_genders = Counter(
                    static.actor_genders[static.pid_to_idx[cid]]
                    for cid in cast_id_list
                    if static.actor_genders is not None and cid in static.pid_to_idx
                )
                cast_nats = Counter(
                    static.actor_nationalities[static.pid_to_idx[cid]]
                    for cid in cast_id_list
                    if static.actor_nationalities is not None and cid in static.pid_to_idx
                )
                if cast_genders and static.actor_genders is not None:
                    gender_present = set(cast_genders.keys())
                    gender_novel = np.array(
                        [g not in gender_present for g in static.actor_genders[candidate_focus]],
                        dtype=bool,
                    )
                    focus_scores[(focus_scores > 0) & gender_novel] *= 1.5
                if len(cast_nats) >= 2 and static.actor_nationalities is not None:
                    nat_present = set(cast_nats.keys())
                    nat_novel = np.array(
                        [n not in nat_present for n in static.actor_nationalities[candidate_focus]],
                        dtype=bool,
                    )
                    focus_scores[(focus_scores > 0) & nat_novel] *= 1.3

                if static.actor_tag_bitmasks is not None:
                    cast_bitmasks = np.array(
                        [static.actor_tag_bitmasks[static.pid_to_idx[cid]] for cid in cast_id_list if cid in static.pid_to_idx],
                        dtype=np.uint64,
                    )
                    if cast_bitmasks.size > 0:
                        cand_bm = static.actor_tag_bitmasks[candidate_focus]
                        friend_mask = (
                            np.isin(focus_pids, np.array(list(cast_friend_pids), dtype=int))
                            if cast_friend_pids else np.zeros(len(candidate_focus), dtype=bool)
                        )
                        has_tags = cand_bm > np.uint64(0)
                        eligible = (focus_scores > 0) & has_tags & ~friend_mask
                        if eligible.any():
                            max_jaccard = np.zeros(len(candidate_focus), dtype=np.float32)
                            for cbm in cast_bitmasks:
                                inter = _popcount_vec(cand_bm & cbm)
                                union = _popcount_vec(cand_bm | cbm)
                                jaccard = np.where(union > 0, inter.astype(np.float32) / union.astype(np.float32), 0.0)
                                np.maximum(max_jaccard, jaccard, out=max_jaccard)
                            focus_scores[eligible & (max_jaccard > 0.5)] *= 0.5

                friend_boost = np.zeros(len(candidate_focus), dtype=np.float32)
                rival_penalty = np.ones(len(candidate_focus), dtype=np.float32)
                focus_pos = {int(global_idx): pos for pos, global_idx in enumerate(candidate_focus)}
                for cid in cast_id_list:
                    friend_iter = graph.iter_friend_neighbors(int(cid), year) if graph is not None else world._friend_adj_all.get(int(cid), [])
                    rival_iter = graph.iter_rival_neighbors(int(cid), year) if graph is not None else world._rival_adj_all.get(int(cid), [])
                    for nbr, w, _vf, _vt in friend_iter:
                        idx = static.pid_to_idx.get(int(nbr))
                        local_idx = focus_pos.get(int(idx)) if idx is not None else None
                        if local_idx is not None and focus_scores[local_idx] > 0 and friend_boost[local_idx] == 0:
                            friend_boost[local_idx] = 1.0 + 4.0 * max(0.15, w)
                    for nbr, w, _vf, _vt in rival_iter:
                        idx = static.pid_to_idx.get(int(nbr))
                        local_idx = focus_pos.get(int(idx)) if idx is not None else None
                        if local_idx is not None and focus_scores[local_idx] > 0:
                            rival_penalty[local_idx] = min(
                                rival_penalty[local_idx],
                                max(0.0, 1.0 - max(0.6, w)),
                            )
                fb_mask = friend_boost > 0
                focus_scores[fb_mask] *= friend_boost[fb_mask]
                focus_scores *= rival_penalty

                need_similarity = np.flatnonzero((focus_scores >= static._sim_threshold) & ~fb_mask)
                if need_similarity.size > 0 and static.li_arr is not None:
                    latent_idx = getattr(world, "_latent_pid_to_idx", {}) or {}
                    cast_li = np.array([latent_idx.get(cid, -1) for cid in cast_id_list[:3]], dtype=int)
                    cast_li = cast_li[cast_li >= 0]
                    if cast_li.size > 0:
                        ns_li = static.li_arr[candidate_focus[need_similarity]]
                        valid_li = ns_li >= 0
                        if valid_li.any():
                            local_targets = need_similarity[valid_li]
                            max_sim = latent_similarity_batch(world, ns_li[valid_li], cast_li)
                            boost = max_sim > 0.1
                            if boost.any():
                                focus_scores[local_targets[boost]] *= (1.0 + max_sim[boost])

            valid_local = _shortlist_indices(
                focus_scores,
                cast_shortlist_size,
                world.rng,
                exploration_share=0.30 if cast_id_list else 0.35,
            )
            if valid_local.size == 0:
                break
            valid = candidate_focus[valid_local]
            exp = 2.5 if slot == 0 else (1.3 if slot == 1 else 0.7)
            probs = normalize_weights(focus_scores[valid_local] ** exp)
            chosen_local = int(world.rng.choice(valid_local, p=probs))
            chosen_idx = int(candidate_focus[chosen_local])
            pid = int(pids[chosen_idx])

            if slot <= 1 and valid.size >= 2 and len(competition_pairs) < 4:
                order = valid[np.argsort(focus_scores[valid_local])[::-1]]
                for alt_idx in order:
                    if int(alt_idx) != chosen_idx:
                        competition_pairs.append((min(pid, int(pids[alt_idx])), max(pid, int(pids[alt_idx]))))
                        break

            cast.append({"person_id": pid, "billing_order": slot + 1})
            cast_ids.add(pid)
            cast_id_list.append(pid)
            world.person_film_count[pid] += 1
            world.person_recent[pid].append(year)
            if getattr(world, "_yearly_workload", None) is not None:
                world._yearly_workload[(pid, year)] += 1
            idx = static.pid_to_idx.get(pid)
            if idx is not None:
                unused_flags[idx] = False

            # Keep direct rivals from being added later in the same cast
            rival_source = graph.iter_rival_neighbors(int(pid), year) if graph is not None else world._rival_adj_all.get(int(pid), [])
            rival_neighbors = set(nbr for (nbr, _w, _vf, _vt) in rival_source)
            forbidden.update(int(nbr) for nbr in rival_neighbors if int(nbr) not in cast_ids)

        if len(cast) >= max(1, cast_size - 1):
            break

    _assign_archetypes(cast, concept, world)

    return cast, competition_pairs


# ---------------------------------------------------------------------------
# Latent-profile archetype assignment
# ---------------------------------------------------------------------------

# Target latent profiles: [reputation, risk_tolerance, ambition, controversy]
# Each archetype has an "ideal" actor profile — the scoring function measures
# how close a real actor is to that ideal.
_ARCHETYPE_TARGET = {
    "Lead Hero":      np.array([0.80, 0.55, 0.70, 0.12], dtype=np.float32),
    "Lead Villain":   np.array([0.60, 0.75, 0.65, 0.45], dtype=np.float32),
    "Love Interest":  np.array([0.60, 0.30, 0.50, 0.08], dtype=np.float32),
    "Mentor":         np.array([0.75, 0.30, 0.65, 0.08], dtype=np.float32),
    "Sidekick":       np.array([0.40, 0.45, 0.40, 0.15], dtype=np.float32),
    "Comic Relief":   np.array([0.35, 0.50, 0.35, 0.22], dtype=np.float32),
    "Supporting":     np.array([0.40, 0.45, 0.45, 0.15], dtype=np.float32),
    "Authority Figure": np.array([0.62, 0.28, 0.58, 0.12], dtype=np.float32),
    "Henchman":       np.array([0.25, 0.60, 0.30, 0.30], dtype=np.float32),
    "Victim":         np.array([0.26, 0.22, 0.24, 0.10], dtype=np.float32),
    "Mysterious Stranger": np.array([0.52, 0.62, 0.46, 0.18], dtype=np.float32),
    "Extra":          np.array([0.20, 0.40, 0.25, 0.10], dtype=np.float32),
}

# Billing tier → eligible archetypes.  Preserves the slot → importance hierarchy.
_ARCHETYPE_SLOT_TIERS: dict[int, list[str]] = {
    0: ["Lead Hero", "Lead Villain"],
    1: ["Love Interest", "Sidekick", "Mentor", "Lead Villain"],
    2: ["Mentor", "Comic Relief", "Sidekick", "Lead Villain"],
}
_ARCHETYPE_GENERAL = ["Supporting", "Extra", "Henchman"]

# Archetypes that can appear at most ONCE per movie.
_ARCHETYPE_UNIQUE = {"Lead Hero", "Lead Villain", "Mentor"}

# Career stage bonuses: legends gravitate to Mentor, prime actors to leads, etc.
_CAREER_ARCHETYPE_BONUS: dict[str, dict[str, float]] = {
    "legend":  {"Mentor": 0.30, "Lead Hero": 0.12, "Lead Villain": 0.08},
    "prime":   {"Lead Hero": 0.20, "Lead Villain": 0.18, "Love Interest": 0.10},
    "veteran": {"Mentor": 0.22, "Supporting": 0.08, "Lead Villain": 0.10},
    "rising":  {"Sidekick": 0.20, "Supporting": 0.12, "Love Interest": 0.10},
    "retired": {"Mentor": 0.18, "Extra": 0.08, "Supporting": 0.05},
}

# Genre-specific archetype bonuses.
_GENRE_ARCHETYPE_BONUS: dict[str, dict[str, float]] = {
    "Horror":       {"Lead Villain": 0.25, "Henchman": 0.08},
    "Romance":      {"Love Interest": 0.28, "Supporting": 0.05},
    "Comedy":       {"Comic Relief": 0.25, "Sidekick": 0.10},
    "Action":       {"Lead Hero": 0.18, "Henchman": 0.10, "Sidekick": 0.08},
    "Thriller":     {"Lead Villain": 0.18, "Lead Hero": 0.10},
    "Crime":        {"Lead Villain": 0.15, "Henchman": 0.10},
    "Sci-Fi":       {"Lead Hero": 0.12, "Mentor": 0.10},
    "Fantasy":      {"Lead Hero": 0.12, "Mentor": 0.12, "Lead Villain": 0.10},
    "Drama":        {"Lead Hero": 0.10, "Mentor": 0.10, "Supporting": 0.08},
    "War":          {"Lead Hero": 0.15, "Mentor": 0.12, "Henchman": 0.08},
    "Mystery":      {"Lead Hero": 0.10, "Lead Villain": 0.15, "Sidekick": 0.10},
    "Documentary":  {"Supporting": 0.10},
    "Animation":    {"Sidekick": 0.12, "Comic Relief": 0.15, "Lead Villain": 0.10},
}

_COLLAB_ARCHETYPE_BONUS: dict[str, dict[str, float]] = {
    "solo": {"Lead Hero": 0.08, "Lead Villain": 0.08},
    "ensemble": {"Sidekick": 0.06, "Supporting": 0.06},
    "mentorship": {"Mentor": 0.12},
}

_CANONICAL_ARCHETYPE_SET = set(_ARCHETYPE_TARGET.keys())
_ARCHETYPE_ALIAS_MAP: dict[str, str] = {
    "hero": "Lead Hero",
    "lead hero": "Lead Hero",
    "protagonist": "Lead Hero",
    "anti hero": "Lead Hero",
    "antihero": "Lead Hero",
    "chosen one": "Lead Hero",
    "superhero": "Lead Hero",
    "detective": "Lead Hero",
    "investigator": "Lead Hero",
    "sleuth": "Lead Hero",
    "private eye": "Lead Hero",
    "athlete": "Lead Hero",
    "cowboy": "Lead Hero",
    "soldier": "Lead Hero",
    "astronaut": "Lead Hero",
    "knight": "Lead Hero",
    "villain": "Lead Villain",
    "lead villain": "Lead Villain",
    "antagonist": "Lead Villain",
    "supervillain": "Lead Villain",
    "mob boss": "Lead Villain",
    "monster": "Lead Villain",
    "stalker": "Lead Villain",
    "outlaw": "Lead Villain",
    "rival": "Lead Villain",
    "love interest": "Love Interest",
    "lover": "Love Interest",
    "matchmaker": "Love Interest",
    "ex": "Love Interest",
    "spouse": "Love Interest",
    "mentor": "Mentor",
    "guide": "Mentor",
    "master": "Mentor",
    "wizard": "Mentor",
    "oracle": "Mentor",
    "coach": "Mentor",
    "teacher": "Mentor",
    "expert": "Mentor",
    "scientist": "Mentor",
    "sidekick": "Sidekick",
    "student": "Sidekick",
    "child": "Sidekick",
    "pet": "Sidekick",
    "companion": "Sidekick",
    "comic relief": "Comic Relief",
    "straight man": "Comic Relief",
    "bumbling fool": "Comic Relief",
    "mascot": "Comic Relief",
    "supporting": "Supporting",
    "everyman": "Supporting",
    "explorer": "Supporting",
    "creator": "Supporting",
    "confidant": "Supporting",
    "subject": "Supporting",
    "parent": "Supporting",
    "singer": "Supporting",
    "dancer": "Supporting",
    "authority figure": "Authority Figure",
    "bureaucrat": "Authority Figure",
    "commander": "Authority Figure",
    "judge": "Authority Figure",
    "host": "Authority Figure",
    "general": "Authority Figure",
    "king": "Authority Figure",
    "queen": "Authority Figure",
    "sheriff": "Authority Figure",
    "politician": "Authority Figure",
    "henchman": "Henchman",
    "victim": "Victim",
    "witness": "Victim",
    "survivor": "Victim",
    "citizen": "Victim",
    "civilian": "Victim",
    "bystander": "Victim",
    "contestant": "Victim",
    "mysterious stranger": "Mysterious Stranger",
    "observer": "Mysterious Stranger",
    "narrator": "Mysterious Stranger",
    "informant": "Mysterious Stranger",
    "suspect": "Mysterious Stranger",
}
_CAREER_STAGE_ALIAS_MAP: dict[str, str] = {
    "emerging": "rising",
    "established": "prime",
    "midcareer": "prime",
    "mid career": "prime",
    "late career": "veteran",
}
_COLLAB_STYLE_ALIAS_MAP: dict[str, str] = {
    "auteur": "solo",
    "studio": "ensemble",
    "collaborative": "ensemble",
    "team": "ensemble",
    "guided": "mentorship",
    "mentor led": "mentorship",
}


def _normalize_label_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _canonical_archetype_name(value: Any, *, fallback: str = "Supporting") -> str:
    raw = str(value or "").strip()
    if not raw:
        return str(fallback)
    if raw in _CANONICAL_ARCHETYPE_SET:
        return raw
    token = _normalize_label_token(raw)
    mapped = _ARCHETYPE_ALIAS_MAP.get(token)
    if mapped:
        return mapped
    if any(word in token for word in ("hero", "protagon", "chosen one", "superhero")):
        return "Lead Hero"
    if any(word in token for word in ("villain", "antagon", "monster", "stalker", "supervillain")):
        return "Lead Villain"
    if any(word in token for word in ("love", "lover", "romantic", "matchmaker")):
        return "Love Interest"
    if any(word in token for word in ("mentor", "oracle", "master", "guide", "wizard", "teacher")):
        return "Mentor"
    if any(word in token for word in ("sidekick", "student", "child", "companion")):
        return "Sidekick"
    if any(word in token for word in ("comic", "relief", "mascot", "fool")):
        return "Comic Relief"
    if any(word in token for word in ("authority", "bureaucrat", "commander", "judge", "host", "politician")):
        return "Authority Figure"
    if any(word in token for word in ("victim", "witness", "survivor", "civilian", "citizen", "bystander")):
        return "Victim"
    if any(word in token for word in ("mysterious", "stranger", "observer", "narrator", "suspect")):
        return "Mysterious Stranger"
    if "hench" in token:
        return "Henchman"
    return str(fallback)


def _canonical_career_stage_name(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return "prime"
    return _CAREER_STAGE_ALIAS_MAP.get(raw, raw)


def _canonical_collaboration_style_name(value: Any) -> str:
    token = _normalize_label_token(value)
    if not token:
        return "ensemble"
    return _COLLAB_STYLE_ALIAS_MAP.get(token, token)


def _dedupe_strs(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _slot_archetype_candidates(world: WorldState) -> dict[int, list[str]]:
    cached = getattr(world, "_slot_archetype_candidates_cache", None)
    if cached is not None:
        return cached
    default = {int(slot): list(values) for slot, values in _ARCHETYPE_SLOT_TIERS.items()}
    raw = _character_generation_dict(
        world,
        "slot_archetype_candidates",
        {str(slot): list(values) for slot, values in default.items()},
    )
    out: dict[int, list[str]] = {}
    for key, values in raw.items():
        try:
            slot = int(key)
        except Exception:
            continue
        if isinstance(values, list) and values:
            normalized = _dedupe_strs(_canonical_archetype_name(value) for value in values)
            if normalized:
                out[slot] = normalized
    cached = out or default
    world._slot_archetype_candidates_cache = cached
    return cached


def _general_archetypes(world: WorldState) -> list[str]:
    cached = getattr(world, "_general_archetypes_cache", None)
    if cached is not None:
        return cached
    values = _character_generation_list(world, "general_archetypes", list(_ARCHETYPE_GENERAL))
    normalized = _dedupe_strs(_canonical_archetype_name(value) for value in values)
    cached = normalized or list(_ARCHETYPE_GENERAL)
    world._general_archetypes_cache = cached
    return cached


def _unique_archetypes(world: WorldState) -> set[str]:
    cached = getattr(world, "_unique_archetypes_cache", None)
    if cached is not None:
        return cached
    values = _character_generation_list(world, "unique_archetypes", sorted(_ARCHETYPE_UNIQUE))
    normalized = _dedupe_strs(_canonical_archetype_name(value) for value in values)
    cached = set(normalized) or set(_ARCHETYPE_UNIQUE)
    world._unique_archetypes_cache = cached
    return cached


def _genre_archetype_candidates(world: WorldState) -> dict[str, list[str]]:
    cached = getattr(world, "_genre_archetype_candidates_cache", None)
    if cached is not None:
        return cached
    default = {str(key): list(values) for key, values in _GENRE_TO_ARCHETYPES.items()}
    raw = _character_generation_dict(world, "genre_archetype_candidates", default)
    out: dict[str, list[str]] = {}
    for key, values in raw.items():
        if isinstance(values, list) and values:
            out[str(key)] = [str(value) for value in values if str(value).strip()]
    cached = out or default
    world._genre_archetype_candidates_cache = cached
    return cached


def _archetype_target_vectors(world: WorldState) -> dict[str, np.ndarray]:
    cached = getattr(world, "_archetype_target_vectors_cache", None)
    if cached is not None:
        return cached
    default = {str(key): np.asarray(value, dtype=np.float32) for key, value in _ARCHETYPE_TARGET.items()}
    raw = _character_generation_dict(
        world,
        "archetype_target_vectors",
        {str(key): [float(x) for x in value.tolist()] for key, value in default.items()},
    )
    out: dict[str, np.ndarray] = {}
    for key, values in raw.items():
        if not isinstance(values, (list, tuple)) or len(values) != 4:
            continue
        try:
            canonical = _canonical_archetype_name(key)
            out[canonical] = np.asarray([float(v) for v in values], dtype=np.float32)
        except Exception:
            continue
    cached = out or default
    world._archetype_target_vectors_cache = cached
    return cached


def _career_archetype_bias(world: WorldState) -> dict[str, dict[str, float]]:
    cached = getattr(world, "_career_archetype_bias_cache", None)
    if cached is not None:
        return cached
    out: dict[str, dict[str, float]] = {}
    for stage, scores in _character_generation_dict(world, "career_stage_archetype_bias", _CAREER_ARCHETYPE_BONUS).items():
        if not isinstance(scores, dict):
            continue
        canonical_stage = _canonical_career_stage_name(stage)
        bucket = dict(out.get(canonical_stage, {}))
        for archetype, score in scores.items():
            try:
                bucket[_canonical_archetype_name(archetype)] = float(score)
            except Exception:
                continue
        if bucket:
            out[canonical_stage] = bucket
    cached = out or dict(_CAREER_ARCHETYPE_BONUS)
    world._career_archetype_bias_cache = cached
    return cached


def _genre_archetype_bias(world: WorldState) -> dict[str, dict[str, float]]:
    cached = getattr(world, "_genre_archetype_bias_cache", None)
    if cached is not None:
        return cached
    out: dict[str, dict[str, float]] = {}
    for genre, scores in _character_generation_dict(world, "genre_archetype_bias", _GENRE_ARCHETYPE_BONUS).items():
        if not isinstance(scores, dict):
            continue
        bucket = dict(out.get(str(genre), {}))
        for archetype, score in scores.items():
            try:
                bucket[_canonical_archetype_name(archetype)] = float(score)
            except Exception:
                continue
        if bucket:
            out[str(genre)] = bucket
    cached = out or dict(_GENRE_ARCHETYPE_BONUS)
    world._genre_archetype_bias_cache = cached
    return cached


def _collaboration_style_archetype_bias(world: WorldState) -> dict[str, dict[str, float]]:
    cached = getattr(world, "_collaboration_style_archetype_bias_cache", None)
    if cached is not None:
        return cached
    out: dict[str, dict[str, float]] = {}
    for style, scores in _character_generation_dict(world, "collaboration_style_archetype_bias", _COLLAB_ARCHETYPE_BONUS).items():
        if not isinstance(scores, dict):
            continue
        canonical_style = _canonical_collaboration_style_name(style)
        bucket = dict(out.get(canonical_style, {}))
        for archetype, score in scores.items():
            try:
                bucket[_canonical_archetype_name(archetype)] = float(score)
            except Exception:
                continue
        if bucket:
            out[canonical_style] = bucket
    cached = out or dict(_COLLAB_ARCHETYPE_BONUS)
    world._collaboration_style_archetype_bias_cache = cached
    return cached


def _archetype_score(
    lv: dict, archetype: str, career_stage: str, genre: str, jitter: float,
    world: WorldState,
) -> float:
    """Score how well a person's latent profile matches an archetype.

    Returns a value roughly in [0, 1.5] where higher = better match.
    The *jitter* parameter (drawn from rng) adds controlled randomness so
    that the same actor profile doesn't deterministically get the same role.
    """
    canonical_archetype = _canonical_archetype_name(archetype)
    canonical_stage = _canonical_career_stage_name(career_stage)
    targets = _archetype_target_vectors(world)
    target = targets.get(canonical_archetype, _ARCHETYPE_TARGET.get(canonical_archetype, _ARCHETYPE_TARGET["Supporting"]))
    person = np.array([
        _safe01(lv.get("public_reputation"), 0.5),
        _safe01(lv.get("risk_tolerance"), 0.5),
        _safe01(lv.get("artistic_ambition"), 0.5),
        _safe01(lv.get("controversy_score"), 0.15),
    ], dtype=np.float32)

    # Mean absolute error → similarity (1.0 = perfect match)
    similarity = float(1.0 - np.mean(np.abs(person - target)))

    # Career stage bonus
    similarity += _career_archetype_bias(world).get(canonical_stage, {}).get(canonical_archetype, 0.0)

    # Genre bonus
    similarity += _genre_archetype_bias(world).get(genre, {}).get(canonical_archetype, 0.0)

    # Collaboration style alignment
    collab = _canonical_collaboration_style_name(lv.get("collaboration_style", "ensemble"))
    similarity += _collaboration_style_archetype_bias(world).get(collab, {}).get(canonical_archetype, 0.0)

    # Controlled jitter: enough to shuffle close scores, not enough to
    # override strong matches.  Scaled to ~10% of typical score range.
    similarity += jitter * 0.12

    return float(similarity)


def _ensure_career_stage_cache(world: WorldState) -> None:
    """Build pid → career_stage cache (O(N) once, O(1) thereafter)."""
    if getattr(world, "_pid_to_career_stage", None) is not None:
        return
    world._pid_to_career_stage = {}
    for df in (world.actors, world.persons):
        if df is not None and "career_stage" in df.columns and "person_id" in df.columns:
            pids = df["person_id"].astype(int).values
            stages = df["career_stage"].fillna("prime").astype(str).str.lower().values
            for pid, stage in zip(pids, stages):
                world._pid_to_career_stage[int(pid)] = stage
            break


def _assign_archetypes(cast: list[dict], concept: dict, world: WorldState) -> None:
    """Assign archetypes by matching each actor's latent profile to role ideals.

    Billing order is preserved (slot 0 = highest billed). For each slot the
    best-matching *eligible* archetype is chosen, respecting tier constraints
    and uniqueness rules. A small random jitter prevents deterministic mapping.
    """
    genre = str(concept.get("genre", "Drama"))
    n = len(cast)
    if n == 0:
        return

    _ensure_career_stage_cache(world)
    used_unique: set[str] = set()
    used_names: set[str] = set()
    slot_candidates = _slot_archetype_candidates(world)
    general_archetypes = _general_archetypes(world)
    unique_archetypes = _unique_archetypes(world)

    # Pre-draw jitter values for each slot (one rng call, reproducible)
    jitters = world.rng.uniform(-1.0, 1.0, size=n)

    for i, row in enumerate(cast):
        pid = int(row["person_id"])
        lv = get_person_latent(world, pid)
        career_stage = world._pid_to_career_stage.get(pid, "prime")

        # Determine eligible archetypes for this billing position
        if i in slot_candidates:
            eligible = list(slot_candidates[i])
        else:
            eligible = list(general_archetypes)

        # Remove already-used unique archetypes
        eligible = [a for a in eligible if a not in used_unique]

        # Fallback: if all eligible were taken, open up general pool
        if not eligible:
            eligible = [a for a in general_archetypes if a not in used_unique]
        if not eligible:
            eligible = list(general_archetypes)

        # Score each eligible archetype
        best_arch = eligible[0]
        best_score = -999.0
        for arch in eligible:
            score = _archetype_score(lv, arch, career_stage, genre, float(jitters[i]), world)
            if score > best_score:
                best_score = score
                best_arch = arch

        # Mark unique archetypes as used
        if best_arch in unique_archetypes:
            used_unique.add(best_arch)

        # Assign archetype and character name
        gender = world._pid_to_gender.get(pid, "M")
        row["archetype"] = best_arch
        row["character_name"] = _pick_character_name(
            str(concept["genre"]), best_arch, i, world,
            gender=gender, used_names=used_names,
        )


def _generate_research_title(world: WorldState, genre: str, *, title_style: str = "") -> str:
    workspace = getattr(world, "workspace", None)
    base_dir = getattr(workspace, "base_dir", None)
    root = Path(base_dir).resolve() if base_dir else Path(__file__).resolve().parent
    grammar = getattr(world, "_research_title_grammar", None)
    if not isinstance(grammar, dict) or not grammar:
        grammar = _load_title_research_grammar(root, "research")
        world._research_title_grammar = grammar

    best_title = ""
    best_score: tuple[float, float, float, int] | None = None
    normalized_style = str(title_style or "").strip().lower()
    for _ in range(384):
        candidate = sanitize_title(_render_title_bank_title(world.rng, grammar, str(genre), mode="research"))
        if not candidate or candidate in world.used_titles:
            continue
        if contains_placeholder_syntax(candidate) or looks_like_weak_title(candidate):
            continue
        words = len(candidate.split())
        punct_score = 1.0 if (":" in candidate or "-" in candidate) else 0.0
        style_bonus = 0.0
        if normalized_style == "bold":
            style_bonus = 1.0 if words <= 2 else 0.0
        elif normalized_style in {"elegant", "luminous"}:
            style_bonus = 1.0 if 2 <= words <= 4 else 0.0
        elif normalized_style in {"pulp", "urgent", "hardboiled"}:
            style_bonus = punct_score
        token_penalty = float(abs(words - 3))
        score = (style_bonus, punct_score, -token_penalty, -len(candidate))
        if best_score is None or score > best_score:
            best_title = candidate
            best_score = score
            if style_bonus >= 1.0 and punct_score >= 1.0:
                break

    if best_title:
        return best_title
    used_titles = getattr(world, "used_titles", set()) or set()
    fallback_rng = getattr(world, "py_rng", None) or getattr(world, "rng", None)
    if fallback_rng is not None:
        for _ in range(128):
            candidate = sanitize_title(generate_compositional_title(fallback_rng, used_titles))
            if (
                candidate
                and candidate not in used_titles
                and not contains_placeholder_syntax(candidate)
                and not looks_like_weak_title(candidate)
            ):
                return candidate

    genre_token = re.sub(r"[^A-Za-z0-9]+", " ", str(genre or "Feature")).strip() or "Feature"
    base = f"{genre_token} Feature"
    for offset in range(10_000):
        candidate = sanitize_title(f"{base} {len(used_titles) + offset + 1}")
        if candidate and candidate not in used_titles:
            return candidate
    return sanitize_title(f"Untitled Feature {len(used_titles) + 1}")


def _cast_usage_multiplier(film_count: np.ndarray, slot: int, *, target_load: float = 6.0) -> np.ndarray:
    film_count = np.asarray(film_count, dtype=float)
    target_load = max(2.0, float(target_load))
    discovery_boost = np.where(
        film_count <= 0,
        2.05 if slot >= 2 else 1.50,
        np.where(
            film_count <= 2,
            1.60 if slot >= 2 else 1.25,
            np.where(
                film_count <= 5,
                1.28 if slot >= 2 else 1.12,
                1.0,
            ),
        ),
    )
    reuse_decay = np.where(
        film_count > target_load * 50.0,
        0.006,
        np.where(
            film_count > target_load * 38.0,
            0.015,
            np.where(
                film_count > target_load * 30.0,
                0.04,
                np.where(
                    film_count > target_load * 22.0,
                    0.09,
                    np.where(
                        film_count > target_load * 15.0,
                        0.18,
                        np.where(
                            film_count > target_load * 9.0,
                            0.34,
                            np.where(
                                film_count > target_load * 5.0,
                                0.58,
                                np.where(film_count > target_load * 2.5, 0.78, 1.0),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    if slot >= 2:
        reuse_decay = np.where(film_count > target_load * 7.0, reuse_decay * 0.72, reuse_decay)
    if slot >= 4:
        reuse_decay = np.where(film_count > target_load * 3.5, reuse_decay * 0.72, reuse_decay)
    return discovery_boost * reuse_decay


def _director_usage_multiplier(film_counts: np.ndarray) -> np.ndarray:
    film_counts = np.asarray(film_counts, dtype=float)
    discovery_boost = np.where(
        film_counts <= 1,
        1.18,
        np.where(film_counts <= 4, 1.10, 1.0),
    )
    reuse_decay = np.where(
        film_counts > 140,
        0.05,
        np.where(
            film_counts > 100,
            0.12,
            np.where(
                film_counts > 70,
                0.24,
                np.where(
                    film_counts > 45,
                    0.45,
                    np.where(film_counts > 25, 0.72, 1.0),
                ),
            ),
        ),
    )
    return discovery_boost * reuse_decay


def _company_usage_multiplier(
    film_counts: np.ndarray,
    recent_window: np.ndarray,
    *,
    target_load: float,
) -> np.ndarray:
    film_counts = np.asarray(film_counts, dtype=float)
    recent_window = np.asarray(recent_window, dtype=float)
    target_load = max(1.0, float(target_load))

    # Companies should be somewhat more concentrated than talent:
    # give incumbents a moderate momentum bonus, but still stop runaway monopolies.
    cold_start_penalty = np.where(film_counts <= 0, 0.90, 1.0)
    momentum_bonus = 1.0 + 0.12 * np.log1p(np.minimum(film_counts, target_load * 2.0))
    recent_bonus = np.where(recent_window > 0, np.minimum(1.24, 1.0 + 0.05 * recent_window), 1.0)
    saturation_decay = np.where(
        film_counts > target_load * 6.0,
        0.80,
        np.where(
            film_counts > target_load * 4.0,
            0.90,
            1.0,
        ),
    )
    return cold_start_penalty * momentum_bonus * recent_bonus * saturation_decay


_FAST_TAGLINE_MOTIFS: dict[str, list[str]] = {
    "Action": ["loyalty", "vengeance", "the last mission", "the impossible escape"],
    "Adventure": ["the map", "the horizon", "the lost trail", "the old promise"],
    "Animation": ["wonder", "friendship", "a strange little miracle", "the bravest wish"],
    "Comedy": ["one bad plan", "every awkward truth", "a perfect disaster", "the joke that goes too far"],
    "Crime": ["the alibi", "the missing witness", "the perfect score", "one buried secret"],
    "Drama": ["family", "memory", "one choice", "the cost of silence"],
    "Fantasy": ["old magic", "a forbidden crown", "the sleeping kingdom", "a borrowed spell"],
    "Horror": ["the house", "the thing outside", "one midnight warning", "the door that should stay closed"],
    "Mystery": ["the final clue", "every witness", "the locked room", "one impossible question"],
    "Romance": ["a second chance", "the letter never sent", "one summer promise", "the distance between hearts"],
    "Sci-Fi": ["the signal", "the last colony", "a broken future", "the machine that remembers"],
    "Thriller": ["the clock", "the wrong target", "the next call", "one dangerous lie"],
}

_FAST_TAGLINE_PATTERNS: tuple[str, ...] = (
    "In {title}, {motif} changes everything.",
    "Every secret in {title} has a price.",
    "Before {title} ends, {motif} will decide who survives.",
    "{title} begins with {motif} and ends with a choice.",
    "No one leaves {title} unchanged.",
    "Behind {title}, {motif} is waiting.",
    "The road to {title} runs through {motif}.",
    "When {title} arrives, {motif} stops being a story.",
)


def _generate_fast_unique_tagline(genre: str, world, *, title: str | None = None) -> str:
    """Cheap, title-specific tagline for benchmark-first runs.

    The expensive research grammar is useful for archival polish, but benchmark
    step 100 mostly needs nonblank, non-repeated metadata. Including the movie
    title gives high uniqueness without weakening relational/movie modeling.
    """
    clean_title = sanitize_title(title or "")
    if not clean_title or clean_title.lower() == "nan":
        return ""
    motifs = _FAST_TAGLINE_MOTIFS.get(str(genre), _FAST_TAGLINE_MOTIFS.get("Drama", ["one choice"]))
    start = int(world.rng.randint(0, len(_FAST_TAGLINE_PATTERNS))) if len(_FAST_TAGLINE_PATTERNS) else 0
    motif_start = int(world.rng.randint(0, len(motifs))) if motifs else 0
    for attempt in range(max(1, len(_FAST_TAGLINE_PATTERNS) * max(1, len(motifs)))):
        pattern = _FAST_TAGLINE_PATTERNS[(start + attempt) % len(_FAST_TAGLINE_PATTERNS)]
        motif = motifs[(motif_start + attempt) % len(motifs)] if motifs else "one choice"
        candidate = sanitize_tagline(pattern.format(title=clean_title, motif=motif), title=clean_title)
        if (
            candidate
            and not looks_like_weak_tagline(candidate, title=clean_title)
            and _tagline_reuse_score(world, candidate) <= 0.0
        ):
            return candidate
    fallback = sanitize_tagline(f"No one leaves {clean_title} unchanged.", title=clean_title)
    if fallback and not looks_like_weak_tagline(fallback, title=clean_title):
        return fallback
    return ""


def _generate_tagline(genre: str, world, *, title: str | None = None) -> str:
    history = _used_tagline_history(world)
    counts = _used_tagline_counts(world)
    recent_entries = _used_tagline_recent_entries(world)
    if _env_bool("DATA_SYS_FAST_TITLE_TAGLINES", False):
        fast_candidate = _generate_fast_unique_tagline(str(genre), world, title=title)
        if fast_candidate:
            return fast_candidate
    if current_mode() == "research":
        workspace = getattr(world, "workspace", None)
        base_dir = getattr(workspace, "base_dir", None)
        root = Path(base_dir).resolve() if base_dir else Path(__file__).resolve().parent
        grammar = getattr(world, "_research_title_grammar", None)
        if not isinstance(grammar, dict) or not grammar:
            grammar = _load_title_research_grammar(root, "research")
            world._research_title_grammar = grammar
        family_counts = getattr(world, "_used_tagline_template_family_counts", None)
        if not isinstance(family_counts, dict):
            family_counts = {}
            setattr(world, "_used_tagline_template_family_counts", family_counts)
        try:
            candidate, family_sig = _materialize_title_bank_tagline(
                grammar=grammar,
                genre=str(genre),
                rng=world.rng,
                mode="research",
                title=str(title or ""),
                tagline_history=history,
                tagline_counts=counts,
                tagline_template_family_counts=family_counts,
            )
            if candidate:
                if family_sig:
                    family_counts[family_sig] = int(family_counts.get(family_sig, 0)) + 1
                return candidate
        except Exception:
            pass

        # Keep using the authored grammar in research mode, but relax the
        # uniqueness pressure if the stricter materializer is exhausted.
        best_candidate = ""
        best_family_sig = ""
        best_score: tuple[float, float, int] | None = None
        for _ in range(256):
            rendered, family_sig = _render_title_bank_tagline(grammar, str(genre), world.rng, mode="research")
            candidate = sanitize_tagline(rendered, title=title)
            sig, tokens, token_set = _tagline_parts(candidate)
            if not candidate or not sig or looks_like_weak_tagline(candidate, title=title):
                continue
            exact_count = float(counts.get(sig, 0))
            near_penalty = 1.0 if _tagline_is_near_duplicate_cached(sig, tokens, token_set, recent_entries, threshold=0.94) else 0.0
            word_distance = abs(len(candidate.split()) - 6)
            score = (exact_count, near_penalty, int(word_distance))
            if best_score is None or score < best_score:
                best_candidate = candidate
                best_family_sig = str(family_sig or "")
                best_score = score
                if exact_count <= 0.0 and near_penalty <= 0.0:
                    break
        if best_candidate:
            if best_family_sig:
                family_counts[best_family_sig] = int(family_counts.get(best_family_sig, 0)) + 1
            return best_candidate
    templates = _TAGLINE_TEMPLATES.get(genre, _TAGLINE_TEMPLATES.get("Drama", []))
    for _ in range(24):
        candidate = sanitize_tagline(world.py_rng.choice(templates) if templates else "", title=title)
        sig, tokens, token_set = _tagline_parts(candidate)
        if (
            candidate
            and not looks_like_weak_tagline(candidate, title=title)
            and counts.get(sig, 0) == 0
            and not _tagline_is_near_duplicate_cached(sig, tokens, token_set, recent_entries, threshold=0.94)
        ):
            return candidate
    return sanitize_tagline(world.py_rng.choice(templates) if templates else "", title=title)


def _clean_display_text(text: object) -> str:
    return clean_display_text(text)


def _used_tagline_counts(world: WorldState) -> dict[str, int]:
    counts = getattr(world, "_used_tagline_counts", None)
    if not isinstance(counts, dict):
        counts = {}
        setattr(world, "_used_tagline_counts", counts)
    return counts


def _used_tagline_history(world: WorldState) -> list[str]:
    history = getattr(world, "_used_tagline_history", None)
    if not isinstance(history, list):
        history = []
        setattr(world, "_used_tagline_history", history)
    return history


def _used_tagline_recent_entries(world: WorldState) -> list[tuple[str, tuple[str, ...], frozenset[str]]]:
    entries = getattr(world, "_used_tagline_recent_entries", None)
    if not isinstance(entries, list):
        entries = []
        setattr(world, "_used_tagline_recent_entries", entries)
    return entries


def _tagline_parts(clean_tagline: str) -> tuple[str, tuple[str, ...], frozenset[str]]:
    clean = sanitize_tagline(clean_tagline)
    if not clean:
        return "", tuple(), frozenset()
    tokens = tuple(token.lower() for token in re.findall(r"[A-Za-z0-9']+", clean))
    if not tokens:
        return "", tuple(), frozenset()
    return " ".join(tokens), tokens, frozenset(tokens)


def _tagline_is_near_duplicate_cached(
    sig: str,
    tokens: tuple[str, ...],
    token_set: frozenset[str],
    existing: Sequence[tuple[str, tuple[str, ...], frozenset[str]]],
    *,
    threshold: float = 0.94,
) -> bool:
    if not sig or not tokens:
        return False
    prefix = tokens[:4]
    token_len = len(tokens)
    recent = existing[-40:] if len(existing) > 40 else existing
    for prior_sig, prior_tokens, prior_set in recent:
        if not prior_sig:
            continue
        if sig == prior_sig:
            return True
        if prefix and len(prior_tokens) >= len(prefix) and prior_tokens[: len(prefix)] == prefix:
            return True
        if not token_set or not prior_set:
            continue
        overlap = len(token_set & prior_set)
        union = len(token_set | prior_set)
        if union <= 0:
            continue
        jaccard = float(overlap / union)
        if jaccard >= float(threshold):
            return True
        # SequenceMatcher is the expensive part. Only pay for it when token
        # overlap is already very high and the phrasing lengths are comparable.
        if jaccard < 0.75 or abs(len(prior_tokens) - token_len) > 2:
            continue
        ratio = SequenceMatcher(None, sig, prior_sig).ratio()
        if max(jaccard, ratio) >= float(threshold):
            return True
    return False


def _register_tagline_use(world: WorldState, tagline: str) -> None:
    clean = sanitize_tagline(tagline)
    sig, tokens, token_set = _tagline_parts(clean)
    if not sig:
        return
    counts = _used_tagline_counts(world)
    history = _used_tagline_history(world)
    recent_entries = _used_tagline_recent_entries(world)
    counts[sig] = int(counts.get(sig, 0)) + 1
    history.append(clean)
    recent_entries.append((sig, tokens, token_set))
    if len(recent_entries) > 256:
        del recent_entries[:-128]


def _tagline_reuse_score(world: WorldState, tagline: str) -> float:
    clean = sanitize_tagline(tagline)
    sig, tokens, token_set = _tagline_parts(clean)
    if not sig:
        return 0.0
    counts = _used_tagline_counts(world)
    exact_count = int(counts.get(sig, 0))
    recent_entries = _used_tagline_recent_entries(world)
    near_duplicate = _tagline_is_near_duplicate_cached(sig, tokens, token_set, recent_entries, threshold=0.94)
    score = float(exact_count) * 1.5
    if near_duplicate:
        score += 0.75
    return float(score)


def _mark_title_used(world: WorldState, title: str, *, row_idx: int | None = None) -> None:
    marker = getattr(world, "mark_title_used", None)
    if callable(marker):
        try:
            marker(title, row_idx=row_idx)
            return
        except Exception:
            pass
    clean = sanitize_title(title)
    if clean:
        world.used_titles.add(clean)


def _title_bank_candidate_indices(world: WorldState, genre: str) -> np.ndarray:
    getter = getattr(world, "available_title_bank_indices", None)
    if callable(getter):
        try:
            candidates = np.asarray(getter(genre), dtype=np.int32)
            if candidates.size > 0:
                return candidates
            return np.asarray(getter(None), dtype=np.int32)
        except Exception:
            pass
    if world.title_bank is None or len(world.title_bank) == 0:
        return np.zeros(0, dtype=np.int32)
    title_values = (
        world.title_bank["_title_clean"]
        if "_title_clean" in world.title_bank.columns
        else world.title_bank["title"].map(sanitize_title)
    ).fillna("").astype(str)
    if "genre_hint" in world.title_bank.columns:
        mask = (world.title_bank["genre_hint"].astype(str) == genre) & (~title_values.isin(world.used_titles))
        indices = world.title_bank.index[mask].to_numpy(dtype=np.int32, copy=True)
        if indices.size > 0:
            return indices
    fallback = ~title_values.isin(world.used_titles)
    return world.title_bank.index[fallback].to_numpy(dtype=np.int32, copy=True)


def _is_valid_research_title_bank_row(world: WorldState, row: pd.Series) -> bool:
    title = str(row.get("_title_clean", sanitize_title(str(row.get("title", "")))) or "")
    tagline = str(row.get("_tagline_clean", sanitize_tagline(str(row.get("tagline", "")), title=title)) or "")
    if "_research_ok_static" in row and not bool(row.get("_research_ok_static", False)):
        return False
    if "_research_ok_static" not in row:
        if not title or not tagline:
            return False
        if contains_placeholder_syntax(title) or contains_placeholder_syntax(tagline):
            return False
        if looks_like_weak_title(title) or looks_like_weak_tagline(tagline, title=title):
            return False
    if _tagline_reuse_score(world, tagline) > 1.0:
        return False
    return True


def pick_title(world: WorldState, concept: dict) -> tuple[str, str, bool]:
    genre = str(concept["genre"])
    franchise = concept.get("franchise")
    franchise_bible = concept.get("franchise_bible") or _franchise_bible(franchise)
    title_style = str(
        (concept.get("concept_pack") or {}).get("title_style")
        or concept.get("policy_targets", {}).get("title_style", "")
        or franchise_bible.get("title_style", "")
    )
    if current_mode() != "research" and franchise and franchise["movies_generated"] > 0:
        base = franchise["name"]
        inst = concept["installment"]
        subtitle_tokens = list(franchise_bible.get("subtitle_tokens", [])) if isinstance(franchise_bible, dict) else []
        if subtitle_tokens:
            subtitle = subtitle_tokens[(max(2, int(inst)) - 2) % len(subtitle_tokens)]
            title = f"{base}: {subtitle}" if int(inst) > 1 else base
        else:
            title = f"{base} {inst}" if inst <= 3 else f"{base}: Part {inst}"
        title = sanitize_title(title)
        if title not in world.used_titles:
            _mark_title_used(world, title)
            _log_selection_decision(
                world,
                stage="pick_title",
                concept=concept,
                chosen={"title": title},
                confidence=0.95,
                candidates=[],
            )
            tagline = _generate_tagline(genre, world, title=title)
            _register_tagline_use(world, tagline)
            return title, tagline, False

    if world.title_bank is not None and len(world.title_bank) > 0:
        tb = world.title_bank
        candidate_idx = _title_bank_candidate_indices(world, genre)
        if candidate_idx.size > 0:
            title_ok_all = tb["_title_ok_static"].to_numpy(dtype=bool) if "_title_ok_static" in tb.columns else None
            if title_ok_all is not None:
                title_ok = title_ok_all[candidate_idx]
                if bool(title_ok.any()):
                    candidate_idx = candidate_idx[title_ok]

        if candidate_idx.size > 0:
            weights = np.ones(candidate_idx.size, dtype=float)
            titles_all = (
                tb["_title_clean"].to_numpy(dtype=object)
                if "_title_clean" in tb.columns
                else tb["title"].fillna("").astype(str).to_numpy(dtype=object)
            )
            taglines_all = (
                tb["_tagline_clean"].to_numpy(dtype=object)
                if "_tagline_clean" in tb.columns
                else tb["tagline"].fillna("").astype(str).to_numpy(dtype=object)
            )
            titles = titles_all[candidate_idx]

            if current_mode() == "research":
                research_ok_all = tb["_research_ok_static"].to_numpy(dtype=bool) if "_research_ok_static" in tb.columns else None
                if research_ok_all is not None:
                    research_ok = research_ok_all[candidate_idx]
                    if bool(research_ok.any()):
                        candidate_idx = candidate_idx[research_ok]
                        weights = weights[research_ok]
                        titles = titles[research_ok]
                    else:
                        candidate_idx = np.zeros(0, dtype=np.int32)
                        weights = np.zeros(0, dtype=float)
                        titles = np.zeros(0, dtype=object)
            elif "_nonresearch_tagline_weight" in tb.columns:
                weights *= tb["_nonresearch_tagline_weight"].to_numpy(dtype=float)[candidate_idx]

        if candidate_idx.size > 0:
            if "_title_word_count" in tb.columns:
                title_word_count = tb["_title_word_count"].to_numpy(dtype=np.int16)[candidate_idx]
            else:
                title_word_count = np.fromiter((len(str(text).split()) for text in titles), dtype=np.int16, count=len(titles))
            if "_title_has_pulp_punct" in tb.columns:
                title_has_pulp_punct = tb["_title_has_pulp_punct"].to_numpy(dtype=bool)[candidate_idx]
            else:
                title_has_pulp_punct = np.fromiter(((":" in str(text)) or ("-" in str(text)) for text in titles), dtype=bool, count=len(titles))

            if title_style == "bold":
                weights *= np.where(title_word_count <= 2, 1.25, 1.0)
            elif title_style in {"elegant", "luminous"}:
                weights *= np.where((title_word_count >= 2) & (title_word_count <= 4), 1.18, 1.0)
            elif title_style in {"pulp", "urgent", "hardboiled"}:
                weights *= np.where(title_has_pulp_punct, 1.18, 1.0)

            shortlist_size = _shortlist_budget(world, "title", 256)
            if candidate_idx.size > shortlist_size:
                shortlist_local = _shortlist_indices(
                    weights,
                    shortlist_size,
                    world.rng,
                    exploration_share=_prior_float(world, "title_exploration_share", 0.30, lo=0.0, hi=0.60),
                )
                candidate_idx = candidate_idx[shortlist_local]
                weights = weights[shortlist_local]
                titles = titles[shortlist_local]

        if candidate_idx.size > 0:
            taglines = taglines_all[candidate_idx]
            if current_mode() == "research":
                dynamic_ok = np.fromiter(
                    (_tagline_reuse_score(world, str(tagline)) <= 1.0 for tagline in taglines),
                    dtype=bool,
                    count=len(taglines),
                )
                if bool(dynamic_ok.any()):
                    candidate_idx = candidate_idx[dynamic_ok]
                    weights = weights[dynamic_ok]
                    titles = titles[dynamic_ok]
                    taglines = taglines[dynamic_ok]
                else:
                    candidate_idx = np.zeros(0, dtype=np.int32)
                    weights = np.zeros(0, dtype=float)
                    titles = np.zeros(0, dtype=object)
                    taglines = np.zeros(0, dtype=object)
            else:
                tagline_penalties = np.fromiter(
                    (_tagline_reuse_score(world, str(tagline)) for tagline in taglines),
                    dtype=float,
                    count=len(taglines),
                )
                weights *= np.where(tagline_penalties <= 0.0, 1.08, np.where(tagline_penalties <= 1.0, 0.55, 0.18))

        if candidate_idx.size > 0:
            probs = normalize_weights(weights)
            chosen_local = int(world.rng.choice(len(candidate_idx), p=probs))
            row_idx = int(candidate_idx[chosen_local])
            row = tb.iloc[row_idx]
            title = str(titles[chosen_local] or "")
            tagline = str(taglines[chosen_local] or "")
            if current_mode() == "research" and not _is_valid_research_title_bank_row(world, row):
                audit_fallback_hit(
                    "assembly.title_selection",
                    "invalid:title_bank_row",
                    detail=f"title bank row could not satisfy research constraints for genre={genre}",
                    mode="research",
                )
                raise RuntimeError(f"title bank row could not satisfy research constraints for genre={genre}")
            elif current_mode() == "research" and (not tagline or tagline == "nan" or looks_like_weak_tagline(tagline, title=title)):
                tagline = _generate_tagline(genre, world, title=title)
                if not tagline or tagline == "nan" or looks_like_weak_tagline(tagline, title=title):
                    audit_fallback_hit(
                        "assembly.title_selection",
                        "invalid:materialized_tagline_missing",
                        detail=f"title bank row did not provide a valid materialized tagline for genre={genre}",
                        mode="research",
                    )
                    raise RuntimeError(f"title bank row did not provide a valid materialized tagline for genre={genre}")
            elif not bool(row.get("_tagline_ok_static", bool(tagline))) or not tagline or tagline == "nan" or looks_like_weak_tagline(tagline, title=title):
                tagline = _generate_tagline(genre, world, title=title)
            ac_raw = row.get("award_contender", False)
            award_contender = bool(ac_raw) if not (isinstance(ac_raw, float) and ac_raw != ac_raw) else False
            _mark_title_used(world, title, row_idx=row_idx)
            _register_tagline_use(world, tagline)
            if franchise and franchise["movies_generated"] == 0:
                franchise["name"] = title
            _log_selection_decision(
                world,
                stage="pick_title",
                concept=concept,
                chosen={"title": title},
                confidence=confidence_from_scores(weights.tolist()),
                candidates=[
                    {"title": str(titles[idx]), "weight": round(float(weights[idx]), 4)}
                    for idx in np.argsort(weights)[::-1][:5]
                ],
            )
            return title, tagline, award_contender

    if current_mode() == "research":
        title = _generate_research_title(world, genre, title_style=title_style)
        tagline = _generate_tagline(genre, world, title=title)
        if not tagline or tagline == "nan" or looks_like_weak_tagline(tagline, title=title):
            audit_fallback_hit(
                "assembly.title_selection",
                "invalid:generated_tagline_missing",
                detail=f"title grammar synthesized title but could not materialize a strong tagline for genre={genre}",
                mode="research",
                    )
            raise RuntimeError(f"title grammar synthesized title but could not materialize a strong tagline for genre={genre}")
        _mark_title_used(world, title)
        _register_tagline_use(world, tagline)
        if franchise and franchise["movies_generated"] == 0:
            franchise["name"] = title
        _log_selection_decision(
            world,
            stage="pick_title",
            concept=concept,
            chosen={"title": title},
            confidence=0.62,
            candidates=[{"source": "authored_grammar"}],
        )
        return title, tagline, False
    title = sanitize_title(generate_compositional_title(world.py_rng, world.used_titles))
    _mark_title_used(world, title)
    if franchise and franchise["movies_generated"] == 0:
        franchise["name"] = title
    _log_selection_decision(
        world,
        stage="pick_title",
        concept=concept,
        chosen={"title": title},
        confidence=0.3,
        candidates=[],
    )
    tagline = _generate_tagline(genre, world, title=title)
    _register_tagline_use(world, tagline)
    return title, tagline, False


def _ensure_company_genre_cache(world: WorldState) -> None:
    """Build a company_id→set[str] cache of each company's specialty genres."""
    if getattr(world, "_company_genre_cache", None) is not None:
        return
    world._company_genre_cache = {}
    if world.companies is None or len(world.companies) == 0:
        return
    if "specialty_genres" not in world.companies.columns:
        return
    cids = world.companies["company_id"].astype(int).values
    genres = world.companies["specialty_genres"].fillna("").astype(str).values
    for cid, gspec in zip(cids, genres):
        pieces = {g.strip() for g in str(gspec).replace(",", ";").split(";") if g.strip()}
        if pieces:
            world._company_genre_cache[int(cid)] = pieces


# Genre-family clusters: a keyword tagged "Action" should also get a small
# boost from a Thriller/Crime-focused company, since those genres share
# thematic affinity.
_GENRE_CLUSTERS = [
    {"Action", "Thriller", "Crime", "Adventure", "Superhero", "War"},
    {"Drama", "Romance", "Sport"},
    {"Sci-Fi", "Fantasy", "Animation", "Family"},
    {"Horror", "Mystery"},
    {"Comedy"},
    {"Documentary"},
]

# Pre-build a genre→cluster mapping for O(1) lookup.
_GENRE_TO_CLUSTER_IDX: dict[str, int] = {}
for _ci, _cluster in enumerate(_GENRE_CLUSTERS):
    for _g in _cluster:
        _GENRE_TO_CLUSTER_IDX[_g] = _ci

_GENRE_ALIASES = {
    "Adventure": "Action",
    "Superhero": "Action",
    "Family": "Animation",
    "Sport": "Drama",
}


_GENRE_CANONICAL_LOOKUP = {str(genre).strip().lower(): str(genre) for genre in GENRES}


def _canonical_genre_label(value: str | None) -> str:
    text = str(value or "").strip()
    return _GENRE_CANONICAL_LOOKUP.get(text.lower(), text)


def _genre_family(value: str | None) -> set[str]:
    genre = _canonical_genre_label(value)
    cluster_idx = _GENRE_TO_CLUSTER_IDX.get(genre)
    if cluster_idx is None:
        return {genre} if genre else set()
    return {_canonical_genre_label(item) for item in _GENRE_CLUSTERS[cluster_idx]}


def _keyword_related_genres(world: WorldState, genre: str) -> set[str]:
    canonical = _canonical_genre_label(genre)
    cfg = _keyword_selection_config(world)
    related = cfg.get("related_genres_by_genre", {}).get(canonical, [])
    out = {
        _canonical_genre_label(item)
        for item in list(related or [])
        if _canonical_genre_label(item) and _canonical_genre_label(item) != canonical
    }
    return out


def _keyword_slot_targets(total: int, slot_mix: dict[str, float], *, franchise_enabled: bool) -> dict[str, int]:
    mix = {str(key): max(0.0, float(value)) for key, value in dict(slot_mix or {}).items()}
    if not franchise_enabled:
        mix["franchise"] = 0.0
    keys = ("exact_anchor", "related_support", "story_specific", "franchise", "generic")
    weight_total = float(sum(mix.get(key, 0.0) for key in keys)) or 1.0
    raw = {key: (mix.get(key, 0.0) / weight_total) * float(max(0, int(total))) for key in keys}
    counts = {key: int(raw[key]) for key in keys}
    leftovers = sorted(keys, key=lambda key: (raw[key] - counts[key], raw[key]), reverse=True)
    for key in leftovers[: max(0, int(total) - sum(counts.values()))]:
        counts[key] += 1
    return counts


def _selected_keyword_mask_count(chosen_indices: Sequence[int], mask: np.ndarray) -> int:
    if not chosen_indices:
        return 0
    selected = np.asarray(list(chosen_indices), dtype=int)
    return int(mask[selected].sum())


def _append_ranked_keyword_candidates(
    ranked_indices: Sequence[int],
    keyword_ids: np.ndarray,
    chosen_indices: list[int],
    chosen_id_set: set[int],
    mask: np.ndarray,
    limit: int,
    *,
    max_total: int,
) -> int:
    added = 0
    if limit <= 0:
        return added
    for candidate_idx in ranked_indices:
        candidate_idx = int(candidate_idx)
        candidate_id = int(keyword_ids[candidate_idx])
        if candidate_id in chosen_id_set or not bool(mask[candidate_idx]):
            continue
        chosen_indices.append(candidate_idx)
        chosen_id_set.add(candidate_id)
        added += 1
        if added >= int(limit) or len(chosen_indices) >= int(max_total):
            break
    return added


def _replace_keyword_candidate(
    keyword_id: int,
    *,
    keyword_ids: np.ndarray,
    chosen_indices: list[int],
    chosen_id_set: set[int],
    replace_mask: np.ndarray,
    probs: np.ndarray,
) -> bool:
    candidate_positions = np.flatnonzero(keyword_ids == int(keyword_id))
    if candidate_positions.size == 0:
        return False
    replacement_idx = int(candidate_positions[0])
    replacement_kid = int(keyword_ids[replacement_idx])
    if replacement_kid in chosen_id_set:
        return True
    replace_positions = [pos for pos, current_idx in enumerate(chosen_indices) if bool(replace_mask[int(current_idx)])]
    if not replace_positions:
        return False
    drop_pos = min(replace_positions, key=lambda pos: float(probs[int(chosen_indices[pos])]))
    old_idx = int(chosen_indices[drop_pos])
    chosen_id_set.discard(int(keyword_ids[old_idx]))
    chosen_indices[drop_pos] = replacement_idx
    chosen_id_set.add(replacement_kid)
    return True


def _repair_keyword_topic_support(
    *,
    ranked_indices: list[int],
    keyword_ids: np.ndarray,
    chosen_indices: list[int],
    chosen_id_set: set[int],
    probs: np.ndarray,
    exact_genre_mask: np.ndarray,
    related_genre_mask: np.ndarray,
    generic_motif_mask: np.ndarray,
    off_genre_mask: np.ndarray,
    exact_preferred_mask: np.ndarray,
    exact_topic_min: int,
) -> None:
    current_exact = int(_selected_keyword_mask_count(chosen_indices, exact_genre_mask))
    if current_exact >= int(exact_topic_min):
        return

    replacement_masks = (
        off_genre_mask,
        generic_motif_mask,
        related_genre_mask & ~exact_genre_mask,
        ~exact_genre_mask,
    )
    for replacement_idx in ranked_indices:
        replacement_idx = int(replacement_idx)
        replacement_kid = int(keyword_ids[replacement_idx])
        if replacement_kid in chosen_id_set or not bool(exact_preferred_mask[replacement_idx]):
            continue
        replaced = False
        for replace_mask in replacement_masks:
            if _replace_keyword_candidate(
                replacement_kid,
                keyword_ids=keyword_ids,
                chosen_indices=chosen_indices,
                chosen_id_set=chosen_id_set,
                replace_mask=replace_mask,
                probs=probs,
            ):
                replaced = True
                break
        if not replaced:
            continue
        current_exact = int(_selected_keyword_mask_count(chosen_indices, exact_genre_mask))
        if current_exact >= int(exact_topic_min):
            break


def _year_slate_family_boosts(world: WorldState, terms: Sequence[str]) -> dict[str, float]:
    boosts: dict[str, float] = {}
    cfg = _keyword_selection_config(world)
    family_cfg = cfg["year_slate_family_boosts"]
    norm_terms = {str(term).strip().lower().replace(" ", "_") for term in terms if str(term).strip()}
    if {"character_intimacy", "slow_burn", "relationship_heat"} & norm_terms:
        boosts.update({k: max(boosts.get(k, 1.0), float(v)) for k, v in family_cfg.get("relationship", {}).items()})
    if {"ensemble_pressure", "institutional_pressure", "market_balance"} & norm_terms:
        boosts.update({k: max(boosts.get(k, 1.0), float(v)) for k, v in family_cfg.get("ensemble", {}).items()})
    if {"scale_event", "spectacle_push", "franchise_pressure"} & norm_terms:
        boosts.update({k: max(boosts.get(k, 1.0), float(v)) for k, v in family_cfg.get("event", {}).items()})
    if {"global_exchange", "migration_pressure", "cross_border"} & norm_terms:
        boosts.update({k: max(boosts.get(k, 1.0), float(v)) for k, v in family_cfg.get("global", {}).items()})
    if {"analogue_texture", "classical_storytelling"} & norm_terms:
        boosts.update({k: max(boosts.get(k, 1.0), float(v)) for k, v in family_cfg.get("analogue", {}).items()})
    if {"platform_fragmentation", "franchise_acceleration"} & norm_terms:
        boosts.update({k: max(boosts.get(k, 1.0), float(v)) for k, v in family_cfg.get("platform", {}).items()})
    return boosts


def pick_keywords(world: WorldState, concept: dict, n: int = None, company_ids: list | None = None) -> list[int]:
    def _kw_trace(stage: str, **extra: Any) -> None:
        path = getattr(world, "movie_progress_log_path", None)
        if not path:
            return
        payload = {
            "event": "keyword_stage",
            "movie_id": int(concept.get("movie_id", 0) or 0),
            "year": int(concept.get("year", 0) or 0),
            "genre": str(concept.get("genre", "")),
            "tier": str(concept.get("tier", "")),
            "country": str(concept.get("country", "")),
            "stage": str(stage),
            "timestamp": time.time(),
        }
        if extra:
            payload.update(extra)
        append_jsonl(path, payload)
        latest_path = getattr(world, "movie_progress_log_latest_path", None)
        if latest_path and str(latest_path) != str(path):
            append_jsonl(latest_path, payload)

    tier = str(concept.get("tier", "Mid"))
    keyword_cfg = _keyword_selection_config(world)
    franchise = concept.get("franchise")
    franchise_bible = concept.get("franchise_bible") or _franchise_bible(franchise)
    if n is None:
        lo, hi = keyword_cfg["count_by_tier"].get(tier, (4, 7))
        n = int(world.rng.randint(int(lo), int(hi) + 1))
        if franchise is not None:
            n = max(n, int(round(float(keyword_cfg["franchise_min_count"]))))
    kw = world.keywords
    if kw is None or len(kw) == 0:
        return []
    _kw_trace("start", keyword_count=int(len(kw)))

    if not hasattr(world, "_keyword_usage_counts"):
        world._keyword_usage_counts = defaultdict(int)

    keyword_ids = kw["keyword_id"].astype(int).values if "keyword_id" in kw.columns else np.arange(1, len(kw) + 1, dtype=int)
    keyword_text_raw = kw["keyword"].fillna("").astype(str) if "keyword" in kw.columns else pd.Series([""] * len(kw))
    keyword_text = keyword_text_raw.str.lower()
    weights = kw["pop_weight"].astype(float).values.copy() if "pop_weight" in kw.columns else np.ones(len(kw), dtype=float)
    weights = np.clip(weights, 1e-6, None)

    genre = _canonical_genre_label(str(concept.get("genre", "Drama")))
    related_genres = _keyword_related_genres(world, genre)
    if "topic_genre" in kw.columns:
        kw_topic = np.array([_canonical_genre_label(value) for value in kw["topic_genre"].fillna("").astype(str).values], dtype=object)
    else:
        kw_topic = np.array([""] * len(kw), dtype=object)
    if "selection_bucket" in kw.columns:
        bucket_labels = kw["selection_bucket"].fillna("").astype(str).str.strip().str.lower().to_numpy(dtype=object)
    else:
        bucket_labels = np.array([""] * len(kw), dtype=object)
        if current_mode() == "research":
            audit_fallback_hit(
                "assembly.keyword_selection",
                "missing:selection_bucket",
                detail="keyword.csv must include selection_bucket in research mode",
                mode="research",
            )
            raise RuntimeError("keyword.csv missing selection_bucket in research mode")
    exact_genre_mask = kw_topic == genre if genre else np.zeros(len(kw), dtype=bool)
    related_genre_mask = np.array([(bool(topic) and topic in related_genres) for topic in kw_topic], dtype=bool)
    primary_related_mask = exact_genre_mask | related_genre_mask
    off_genre_mask = np.array([(bool(topic) and not bool(primary_related_mask[idx])) for idx, topic in enumerate(kw_topic)], dtype=bool)
    if genre:
        weights *= np.where(exact_genre_mask, float(keyword_cfg["exact_genre_boost"]), 1.0)
        weights *= np.where(~exact_genre_mask & related_genre_mask, float(keyword_cfg["family_genre_boost"]), 1.0)
        weights *= np.where(off_genre_mask, float(keyword_cfg["off_genre_penalty"]), 1.0)

    policy_targets = concept.get("policy_targets", {}) or {}
    seed_terms = list(policy_targets.get("keyword_seed_cluster", []) or [])
    slate_terms = (
        list(policy_targets.get("priority_motifs", []) or [])
        + list(policy_targets.get("trending_subgenres", []) or [])
        + list(policy_targets.get("motif_drift", []) or [])
    )
    franchise_terms = list(policy_targets.get("franchise_keyword_families", []) or []) + list(franchise_bible.get("keyword_families", []) or [])
    raw_lexical_terms = seed_terms + franchise_terms
    lexical_terms: list[str] = []
    seen_terms: set[str] = set()
    for term in raw_lexical_terms:
        norm_term = str(term).strip().lower().replace("_", " ").replace("-", " ")
        norm_term = " ".join(part for part in norm_term.split() if part)
        if not norm_term or len(norm_term) < 3 or len(norm_term) > 48:
            continue
        if norm_term in seen_terms:
            continue
        seen_terms.add(norm_term)
        lexical_terms.append(norm_term)
        if len(lexical_terms) >= 12:
            break
    concept_mask = np.zeros(len(kw), dtype=float)
    if lexical_terms and "keyword" in kw.columns:
        keyword_values = keyword_text.astype(str).tolist()
        for term in lexical_terms:
            matches = np.fromiter((1.0 if term in value else 0.0 for value in keyword_values), dtype=float, count=len(keyword_values))
            concept_mask += matches
        if concept_mask.any():
            weights *= 1.0 + float(keyword_cfg["lexical_match_scale"]) * np.clip(concept_mask, 0.0, float(keyword_cfg["lexical_match_cap"]))
    _kw_trace(
        "lexical_terms",
        lexical_term_count=int(len(lexical_terms)),
        lexical_match_count=int(np.count_nonzero(concept_mask)),
    )

    motif_family = kw["motif_family"].fillna("").astype(str).values if "motif_family" in kw.columns else np.array([""] * len(kw), dtype=object)
    specificity = kw["specificity_tier"].fillna(2).astype(int).values if "specificity_tier" in kw.columns else np.full(len(kw), 2, dtype=int)
    scope_hint = kw["scope_hint"].fillna("").astype(str).values if "scope_hint" in kw.columns else np.array([""] * len(kw), dtype=object)
    franchise_affinity = kw["franchise_affinity"].fillna(0.0).astype(float).values if "franchise_affinity" in kw.columns else np.zeros(len(kw), dtype=float)
    recurrence_strength = kw["recurrence_strength"].fillna(0.0).astype(float).values if "recurrence_strength" in kw.columns else np.zeros(len(kw), dtype=float)
    generic_motif_mask = (bucket_labels == "generic") | ((bucket_labels == "") & (specificity <= 2) & np.isin(motif_family, np.array(["genre", "tone"], dtype=object)))
    specific_story_mask = (bucket_labels == "story_specific") | ((bucket_labels == "") & (specificity >= 3))
    exact_anchor_mask = bucket_labels == "exact_anchor"
    related_support_mask = bucket_labels == "related_support"

    year_slate_family_boosts = _year_slate_family_boosts(world, slate_terms)

    if _keyword_motifs_enabled(world):
        for family_name, boost in year_slate_family_boosts.items():
            weights *= np.where(motif_family == family_name, float(boost), 1.0)
        weights *= np.where(specificity <= 1, float(keyword_cfg["specificity_tier1_penalty"]), 1.0)
        weights *= np.where(generic_motif_mask, float(keyword_cfg["generic_motif_penalty"]), 1.0)
        weights *= np.where(specific_story_mask, float(keyword_cfg["specific_story_boost"]), 1.0)
        if franchise is not None:
            weights *= np.where(
                scope_hint == "franchise",
                float(keyword_cfg["franchise_scope_boost_base"])
                + float(keyword_cfg["franchise_scope_affinity_scale"]) * np.clip(franchise_affinity, 0.0, 1.0),
                1.0,
            )
            weights *= np.where(
                np.isin(motif_family, np.array(["franchise", "sequel_drift"], dtype=object)),
                float(keyword_cfg["franchise_family_boost"]),
                1.0,
            )
            weights *= float(keyword_cfg["franchise_recurrence_base"]) + float(keyword_cfg["franchise_recurrence_scale"]) * np.clip(recurrence_strength, 0.0, 1.0)
        else:
            weights *= np.where(scope_hint == "franchise", float(keyword_cfg["nonfranchise_scope_penalty"]), 1.0)
            weights *= np.where(
                np.isin(motif_family, np.array(["franchise", "sequel_drift"], dtype=object)),
                float(keyword_cfg["nonfranchise_family_penalty"]),
                1.0,
            )
            weights *= np.where(
                franchise_affinity >= float(keyword_cfg["nonfranchise_affinity_threshold"]),
                float(keyword_cfg["nonfranchise_affinity_penalty"]),
                1.0,
            )
        novelty_target = float(policy_targets.get("novelty_target", 0.35) or 0.35)
        weights *= np.where(
            specificity >= 4,
            float(keyword_cfg["high_specificity_novelty_base"]) + float(keyword_cfg["high_specificity_novelty_scale"]) * novelty_target,
            1.0,
        )
        weights *= np.where(
            (scope_hint == "movie") & (specificity >= 3),
            float(keyword_cfg["movie_scope_novelty_base"]) + float(keyword_cfg["movie_scope_novelty_scale"]) * novelty_target,
            1.0,
        )

    usage_counts = np.array([world._keyword_usage_counts.get(int(kid), 0) for kid in keyword_ids], dtype=float)
    weights *= 1.0 / (1.0 + float(keyword_cfg["usage_penalty_scale"]) * usage_counts)

    exact_company_mask = np.zeros(len(kw), dtype=bool)
    related_company_mask = np.zeros(len(kw), dtype=bool)
    if company_ids and "topic_genre" in kw.columns:
        _ensure_company_genre_cache(world)
        company_genres: set[str] = set()
        company_related: set[str] = set()
        for cid in company_ids[:3]:
            cg = world._company_genre_cache.get(int(cid))
            if cg:
                for value in cg:
                    canonical = _canonical_genre_label(value)
                    if canonical:
                        company_genres.add(canonical)
                        company_related.update(_keyword_related_genres(world, canonical))
        if not company_genres:
            company_genres.add(genre)
            company_related.update(related_genres)
        exact_company_mask = np.array([(bool(topic) and topic in company_genres) for topic in kw_topic], dtype=bool)
        related_company_mask = np.array([(bool(topic) and topic in company_related) for topic in kw_topic], dtype=bool)
        weights *= np.where(exact_company_mask, float(keyword_cfg["company_exact_boost"]), 1.0)
        weights *= np.where(~exact_company_mask & related_company_mask, float(keyword_cfg["company_family_boost"]), 1.0)
    _kw_trace("company_bias", company_count=int(len(company_ids or [])))

    core_ids: set[int] = set()
    if franchise is not None:
        core_ids = set(int(kid) for kid in franchise.get("keyword_core_ids", []) if int(kid) > 0)
        if core_ids:
            weights *= np.where(np.isin(keyword_ids, np.array(sorted(core_ids), dtype=int)), float(keyword_cfg["franchise_core_boost"]), 1.0)

    probs = normalize_weights(weights)
    n = min(int(n), len(kw))
    if n <= 0:
        return []

    exact_story_mask = exact_genre_mask & ~generic_motif_mask
    franchise_story_mask = (scope_hint == "franchise") | np.isin(motif_family, np.array(["franchise", "sequel_drift"], dtype=object))
    lexical_story_mask = concept_mask > 0.0
    company_story_mask = exact_company_mask | related_company_mask
    support_story_mask = (primary_related_mask | lexical_story_mask | franchise_story_mask | company_story_mask) & ~generic_motif_mask
    specific_story_support_mask = (specific_story_mask | lexical_story_mask | franchise_story_mask | company_story_mask | exact_story_mask) & ~generic_motif_mask

    exact_topic_min = max(1, int(round(float(keyword_cfg["exact_topic_min_count_by_tier"].get(tier, 1.0) or 1.0))))
    primary_plus_related_min = max(exact_topic_min, int(round(float(keyword_cfg["primary_plus_related_min_count_by_tier"].get(tier, 1.0) or 1.0))))
    generic_limit = max(0, int(round(float(keyword_cfg["generic_keyword_cap_by_tier"].get(tier, 1.0) or 1.0))))
    off_genre_limit = max(0, int(round(float(keyword_cfg["off_genre_cap_by_tier"].get(tier, 1.0) or 1.0))))
    exact_capacity = int(np.count_nonzero(exact_genre_mask & ~generic_motif_mask))
    primary_related_capacity = int(np.count_nonzero(primary_related_mask & ~generic_motif_mask))
    non_off_capacity = int(np.count_nonzero(~off_genre_mask))

    def _fail_research(reason: str, detail: str) -> None:
        if current_mode() == "research":
            audit_fallback_hit("assembly.keyword_selection", reason, detail=detail, mode="research")
            raise RuntimeError(detail)

    if current_mode() == "research":
        if exact_capacity < exact_topic_min:
            _fail_research("invalid:exact_topic_capacity", f"title_id={int(concept.get('movie_id', 0) or 0)} needs {exact_topic_min} exact-topic keywords for {genre}, but bank only has {exact_capacity}")
        if primary_related_capacity < primary_plus_related_min:
            _fail_research("invalid:primary_related_capacity", f"title_id={int(concept.get('movie_id', 0) or 0)} needs {primary_plus_related_min} primary/related keywords for {genre}, but bank only has {primary_related_capacity}")
        if non_off_capacity + off_genre_limit < int(n):
            _fail_research("invalid:slot_plan_infeasible", f"title_id={int(concept.get('movie_id', 0) or 0)} requested {n} keywords but only {non_off_capacity} non-off-genre candidates plus cap {off_genre_limit} are available")

    slot_targets = _keyword_slot_targets(int(n), keyword_cfg.get("slot_mix_by_tier", {}).get(tier, _DEFAULT_KEYWORD_SLOT_MIX_BY_TIER.get(tier, {})), franchise_enabled=franchise is not None)
    slot_targets["generic"] = min(int(slot_targets.get("generic", 0)), int(generic_limit))
    exact_topic_target = max(int(slot_targets.get("exact_anchor", 0)), int(exact_topic_min))
    slot_targets["exact_anchor"] = int(exact_topic_target)
    _kw_trace("slot_plan", requested_count=int(n), exact_topic_min=int(exact_topic_min), exact_topic_target=int(exact_topic_target), primary_plus_related_min=int(primary_plus_related_min), generic_cap=int(generic_limit), off_genre_cap=int(off_genre_limit), related_genres=sorted(related_genres), slot_targets={key: int(value) for key, value in slot_targets.items()})

    ranked_indices = [int(idx) for idx in np.argsort(probs)[::-1]]
    chosen_indices: list[int] = []
    chosen_id_set: set[int] = set()
    exact_slot_mask = exact_anchor_mask & exact_genre_mask & ~generic_motif_mask
    related_slot_mask = related_support_mask & related_genre_mask & ~generic_motif_mask
    story_slot_mask = specific_story_support_mask & ~off_genre_mask
    franchise_slot_mask = franchise_story_mask & ~generic_motif_mask & ~off_genre_mask
    generic_slot_mask = generic_motif_mask & ~off_genre_mask
    exact_backup_mask = exact_story_mask & ~off_genre_mask
    related_backup_mask = primary_related_mask & ~generic_motif_mask & ~off_genre_mask

    _append_ranked_keyword_candidates(ranked_indices, keyword_ids, chosen_indices, chosen_id_set, exact_slot_mask, int(exact_topic_target), max_total=int(n))
    if _selected_keyword_mask_count(chosen_indices, exact_genre_mask) < exact_topic_target:
        _append_ranked_keyword_candidates(ranked_indices, keyword_ids, chosen_indices, chosen_id_set, exact_backup_mask, int(exact_topic_target - _selected_keyword_mask_count(chosen_indices, exact_genre_mask)), max_total=int(n))

    primary_related_target = max(int(primary_plus_related_min), int(slot_targets.get("exact_anchor", 0)) + int(slot_targets.get("related_support", 0)))
    if _selected_keyword_mask_count(chosen_indices, primary_related_mask) < primary_related_target:
        _append_ranked_keyword_candidates(ranked_indices, keyword_ids, chosen_indices, chosen_id_set, related_slot_mask, int(primary_related_target - _selected_keyword_mask_count(chosen_indices, primary_related_mask)), max_total=int(n))
    if _selected_keyword_mask_count(chosen_indices, primary_related_mask) < primary_plus_related_min:
        _append_ranked_keyword_candidates(ranked_indices, keyword_ids, chosen_indices, chosen_id_set, related_backup_mask, int(primary_plus_related_min - _selected_keyword_mask_count(chosen_indices, primary_related_mask)), max_total=int(n))

    if _selected_keyword_mask_count(chosen_indices, specific_story_mask) < int(slot_targets.get("story_specific", 0)):
        _append_ranked_keyword_candidates(ranked_indices, keyword_ids, chosen_indices, chosen_id_set, story_slot_mask, int(slot_targets.get("story_specific", 0) - _selected_keyword_mask_count(chosen_indices, specific_story_mask)), max_total=int(n))
    if franchise is not None and _selected_keyword_mask_count(chosen_indices, franchise_story_mask) < int(slot_targets.get("franchise", 0)):
        _append_ranked_keyword_candidates(ranked_indices, keyword_ids, chosen_indices, chosen_id_set, franchise_slot_mask, int(slot_targets.get("franchise", 0) - _selected_keyword_mask_count(chosen_indices, franchise_story_mask)), max_total=int(n))
    if _selected_keyword_mask_count(chosen_indices, generic_motif_mask) < int(slot_targets.get("generic", 0)):
        _append_ranked_keyword_candidates(ranked_indices, keyword_ids, chosen_indices, chosen_id_set, generic_slot_mask, int(slot_targets.get("generic", 0) - _selected_keyword_mask_count(chosen_indices, generic_motif_mask)), max_total=int(n))

    for mask in (exact_backup_mask, related_backup_mask, story_slot_mask, franchise_slot_mask, support_story_mask & ~off_genre_mask, generic_slot_mask, ~off_genre_mask):
        if len(chosen_indices) >= int(n):
            break
        _append_ranked_keyword_candidates(ranked_indices, keyword_ids, chosen_indices, chosen_id_set, mask, int(n - len(chosen_indices)), max_total=int(n))
    if len(chosen_indices) < int(n):
        remaining_off_budget = max(0, int(off_genre_limit - _selected_keyword_mask_count(chosen_indices, off_genre_mask)))
        if remaining_off_budget > 0:
            _append_ranked_keyword_candidates(ranked_indices, keyword_ids, chosen_indices, chosen_id_set, off_genre_mask & ~generic_motif_mask, min(remaining_off_budget, int(n - len(chosen_indices))), max_total=int(n))

    _repair_keyword_topic_support(
        ranked_indices=ranked_indices,
        keyword_ids=keyword_ids,
        chosen_indices=chosen_indices,
        chosen_id_set=chosen_id_set,
        probs=probs,
        exact_genre_mask=exact_genre_mask,
        related_genre_mask=related_genre_mask,
        generic_motif_mask=generic_motif_mask,
        off_genre_mask=off_genre_mask,
        exact_preferred_mask=(exact_slot_mask | exact_backup_mask) & ~generic_motif_mask,
        exact_topic_min=int(exact_topic_min),
    )

    _kw_trace("post_slot_fill", selected_count=int(len(chosen_indices)), exact_topic_count=int(_selected_keyword_mask_count(chosen_indices, exact_genre_mask)), related_topic_count=int(_selected_keyword_mask_count(chosen_indices, related_genre_mask)), generic_count=int(_selected_keyword_mask_count(chosen_indices, generic_motif_mask)), off_genre_count=int(_selected_keyword_mask_count(chosen_indices, off_genre_mask)))
    if current_mode() == "research":
        if len(chosen_indices) < int(n):
            _fail_research("invalid:selection_underfilled", f"title_id={int(concept.get('movie_id', 0) or 0)} could only fill {len(chosen_indices)} / {n} keyword slots under research constraints")
        if _selected_keyword_mask_count(chosen_indices, exact_genre_mask) < exact_topic_min:
            _fail_research("invalid:exact_topic_min_unsatisfied", f"title_id={int(concept.get('movie_id', 0) or 0)} selected {_selected_keyword_mask_count(chosen_indices, exact_genre_mask)} exact-topic keywords for {genre} (need {exact_topic_min})")
        if _selected_keyword_mask_count(chosen_indices, primary_related_mask) < primary_plus_related_min:
            _fail_research("invalid:primary_related_min_unsatisfied", f"title_id={int(concept.get('movie_id', 0) or 0)} selected {_selected_keyword_mask_count(chosen_indices, primary_related_mask)} primary/related keywords for {genre} (need {primary_plus_related_min})")
        if _selected_keyword_mask_count(chosen_indices, generic_motif_mask) > generic_limit:
            _fail_research("invalid:generic_cap_exceeded", f"title_id={int(concept.get('movie_id', 0) or 0)} selected {_selected_keyword_mask_count(chosen_indices, generic_motif_mask)} generic keywords (cap {generic_limit})")
        if _selected_keyword_mask_count(chosen_indices, off_genre_mask) > off_genre_limit:
            _fail_research("invalid:off_genre_cap_exceeded", f"title_id={int(concept.get('movie_id', 0) or 0)} selected {_selected_keyword_mask_count(chosen_indices, off_genre_mask)} off-genre keywords (cap {off_genre_limit})")

    if franchise is not None:
        existing_core = list(franchise.get("keyword_core_ids", []) or [])
        if existing_core:
            required_overlap = min(len(existing_core), 1 if len(chosen_indices) <= 5 else 2)
            replace_mask = generic_motif_mask | off_genre_mask | ~franchise_story_mask
            for core_id in existing_core:
                current_overlap = int(sum(int(kid) in set(existing_core) for kid in chosen_id_set))
                if current_overlap >= required_overlap:
                    break
                _replace_keyword_candidate(int(core_id), keyword_ids=keyword_ids, chosen_indices=chosen_indices, chosen_id_set=chosen_id_set, replace_mask=replace_mask, probs=probs)
        else:
            preferred_core_pairs = sorted(
                [(int(keyword_ids[int(i)]), float(probs[int(i)])) for i in chosen_indices if bool((franchise_story_mask | exact_genre_mask | related_genre_mask)[int(i)]) and not bool(generic_motif_mask[int(i)])],
                key=lambda item: item[1],
                reverse=True,
            )
            if not preferred_core_pairs:
                preferred_core_pairs = sorted(
                    [(int(keyword_ids[int(i)]), float(probs[int(i)])) for i in chosen_indices],
                    key=lambda item: item[1],
                    reverse=True,
                )
            franchise["keyword_core_ids"] = [kid for kid, _score in preferred_core_pairs[: min(4, len(preferred_core_pairs))]]
    _kw_trace(
        "post_franchise_cleanup",
        franchise=bool(franchise is not None),
        franchise_selected_count=int(_selected_keyword_mask_count(chosen_indices, franchise_story_mask)) if franchise is not None else 0,
    )

    chosen_ids = [int(keyword_ids[int(idx)]) for idx in chosen_indices]

    for kid in chosen_ids:
        world._keyword_usage_counts[int(kid)] += 1

    top_idx = np.argsort(probs)[::-1][:8]
    confidence = confidence_from_scores(probs.tolist())
    selected_exact_count = int(_selected_keyword_mask_count(chosen_indices, exact_genre_mask))
    selected_related_count = int(_selected_keyword_mask_count(chosen_indices, related_genre_mask))
    selected_off_genre_count = int(_selected_keyword_mask_count(chosen_indices, off_genre_mask))
    selected_generic_count = int(_selected_keyword_mask_count(chosen_indices, generic_motif_mask))
    selected_franchise_count = int(
        sum(
            bool(scope_hint[int(idx)] == "franchise")
            or bool(motif_family[int(idx)] in {"franchise", "sequel_drift"})
            or int(keyword_ids[int(idx)]) in core_ids
            for idx in chosen_indices
        )
    )
    specificity_avg = float(np.mean([float(specificity[int(idx)]) for idx in chosen_indices])) if len(chosen_indices) else 0.0
    primary_family_support_count = int(selected_exact_count + selected_related_count)
    keyword_layers = {
        "seed_terms": seed_terms[:6],
        "year_slate_terms": slate_terms[:8],
        "year_slate_family_boosts": {key: round(float(value), 3) for key, value in year_slate_family_boosts.items()},
        "franchise_terms": franchise_terms[:8],
        "company_ids": list(company_ids or [])[:3],
        "related_genres": sorted(related_genres),
    }
    selection_summary = {
        "selected_count": int(len(chosen_ids)),
        "genre_match_count": int(primary_family_support_count),
        "exact_genre_count": int(selected_exact_count),
        "exact_topic_count": int(selected_exact_count),
        "same_family_count": int(selected_related_count),
        "related_count": int(selected_related_count),
        "primary_plus_related_count": int(primary_family_support_count),
        "off_genre_count": int(selected_off_genre_count),
        "generic_count": int(selected_generic_count),
        "franchise_count": int(selected_franchise_count),
        "specificity_avg": round(float(specificity_avg), 3),
        "lexical_match_count": int(sum(float(concept_mask[int(idx)]) > 0.0 for idx in chosen_indices)),
        "generic_cap": int(generic_limit),
        "off_genre_cap": int(off_genre_limit),
        "exact_topic_min": int(exact_topic_min),
        "exact_topic_target": int(exact_topic_target),
        "primary_plus_related_min": int(primary_plus_related_min),
        "slot_targets": {key: int(value) for key, value in slot_targets.items()},
        "franchise_core_missing": bool(franchise is not None and core_ids and not any(int(kid) in core_ids for kid in chosen_ids)),
    }
    concept["_keyword_confidence"] = confidence
    concept["_keyword_candidates"] = [
        {
            "keyword_id": int(keyword_ids[idx]),
            "keyword": str(keyword_text_raw.iloc[idx]),
            "topic_genre": str(kw_topic[idx]),
            "selection_bucket": str(bucket_labels[idx]),
            "motif_family": str(motif_family[idx]),
            "specificity_tier": int(specificity[idx]),
            "scope_hint": str(scope_hint[idx]),
            "genre_match": bool(primary_related_mask[idx]),
            "prob": round(float(probs[idx]), 6),
        }
        for idx in top_idx
    ]
    concept["_keyword_layers"] = keyword_layers
    concept["_keyword_selection_summary"] = selection_summary
    _kw_trace(
        "before_decision_log",
        selected_count=int(len(chosen_ids)),
        confidence=round(float(confidence), 4),
    )
    _log_selection_decision(
        world,
        stage="pick_keywords",
        concept=concept,
        chosen={"keyword_ids": chosen_ids},
        confidence=confidence,
        candidates=concept["_keyword_candidates"][:5],
        extra={**keyword_layers, "selection_summary": selection_summary},
    )
    _kw_trace("after_decision_log", selected_count=int(len(chosen_ids)))
    return chosen_ids


def pick_crew(world: WorldState, concept: dict, director_id: int, cast: list[dict]) -> list[dict]:
    year = int(concept["year"])
    genre = str(concept.get("genre", "Drama"))
    tier = str(concept.get("tier", "Mid-Budget"))
    used = {int(director_id)} if director_id is not None else set()
    for c in cast:
        try:
            used.add(int(c.get("person_id")))
        except Exception:
            pass

    def sample_ids(pool: CrewYearPool | None, n: int, genre_boost_genres: Iterable[str] | None = None, preferred_ids: set[int] | None = None) -> list[int]:
        if pool is None or n <= 0 or len(pool.person_ids) == 0:
            return []
        band_idx = _crew_candidate_band(world, pool, genre, n)
        if band_idx.size == 0:
            return []
        if preferred_ids:
            pref_idx = [pool.pid_to_local.get(int(pid)) for pid in preferred_ids]
            pref_idx = [int(idx) for idx in pref_idx if idx is not None]
            if pref_idx:
                band_idx = np.unique(np.concatenate([band_idx, np.asarray(pref_idx, dtype=np.int32)]))
        pool_ids = pool.person_ids[band_idx]
        weights = _crew_genre_weights(pool, genre)[band_idx].copy()
        if len(weights) == 0:
            return []
        band_local = {int(pid): idx for idx, pid in enumerate(pool_ids)}
        if genre_boost_genres and str(genre).lower() in {str(g).lower() for g in genre_boost_genres}:
            boosted_mask = _crew_genre_match_mask(pool, genre)[band_idx]
            if boosted_mask.any():
                weights[boosted_mask] *= 1.5
        if preferred_ids:
            if len(preferred_ids) <= 24:
                for pid in preferred_ids:
                    local_idx = band_local.get(int(pid))
                    if local_idx is not None:
                        weights[int(local_idx)] *= 8.0
            else:
                pref_mask = np.isin(pool_ids, np.array(sorted(preferred_ids), dtype=int))
                weights[pref_mask] *= 8.0

        if used:
            if len(used) <= 32:
                for pid in used:
                    local_idx = band_local.get(int(pid))
                    if local_idx is not None:
                        weights[int(local_idx)] = 0.0
            else:
                weights[np.isin(pool_ids, np.array(sorted(used), dtype=int))] = 0.0
        eligible = _shortlist_indices(
            weights,
            min(len(pool_ids), _shortlist_budget(world, "crew", max(32, n * 8))),
            world.rng,
            exploration_share=_prior_float(world, "crew_exploration_share", 0.30, lo=0.0, hi=0.60),
        )
        if eligible.size == 0:
            eligible = np.flatnonzero(weights > 0)
        if eligible.size == 0:
            return []

        chosen_count = min(int(n), int(eligible.size))
        probs = normalize_weights(weights[eligible])
        chosen = world.rng.choice(eligible, size=chosen_count, replace=False, p=probs)
        ids: list[int] = []
        for idx in chosen:
            pid = int(pool_ids[int(idx)])
            if pid not in used:
                ids.append(pid)
                used.add(pid)
        return ids

    crew_rows = []
    credit = 1
    for role, cfg in CREW_DEPARTMENTS.items():
        count = int(cfg["count"].get(tier, 0))
        if count <= 0:
            continue
        pool = _get_crew_year_pool(world, role, year)
        if pool is None:
            if current_mode() == "research":
                audit_fallback_hit(
                    "assembly.crew_selection",
                    f"missing:{role}_pool",
                    detail=f"crew pool for role {role} is empty and actor fallback is not allowed in research mode",
                    mode="research",
                )
            if not hasattr(world, "_crew_fallback_warned"):
                world._crew_fallback_warned = set()
            if role not in world._crew_fallback_warned:
                print(f"  [WARN] H1: crew pool empty for role='{role}' -- falling back to actor pool")
                world._crew_fallback_warned.add(role)
            pool = _get_crew_year_pool(world, "actor_fallback", year)
        if role == "writer" and director_id is not None:
            if not hasattr(world, "director_writer_history"):
                world.director_writer_history = {}
            prev_writers = world.director_writer_history.get(int(director_id), set())
        else:
            prev_writers = None
        for pid in sample_ids(pool, count, genre_boost_genres=cfg.get("genre_boost"), preferred_ids=prev_writers):
            crew_rows.append({
                "person_id": int(pid),
                "crew_role": role,
                "credit_order": credit,
                "department": _CREW_DEPT.get(role, "Production"),
            })
            if role == "writer" and director_id is not None:
                if not hasattr(world, "director_writer_history"):
                    world.director_writer_history = {}
                world.director_writer_history.setdefault(int(director_id), set()).add(int(pid))
            credit += 1
    return crew_rows
