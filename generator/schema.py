"""
Mirage -- schema.py
====================
Declarative table schema definitions and generator auto-wiring registry.

Adding a new table:
  1. Add entry to TABLE_DEFS
  2. If secondary: add a SecondaryGenerator to SECONDARY_GENERATORS
  3. Write the generator function (in secondary_tables.py or new module)

Table categories:
  - core:      Built inline by assemble_movies() (tightly coupled to loop state)
  - secondary: Auto-wired per-movie generators (called via registry)
  - global:    Generated once before the movie loop
"""
from dataclasses import dataclass, field
from typing import Callable, Any

# ═══════════════════════════════════════════════════════════════════════
# TABLE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════

TABLE_DEFS: dict[str, dict] = {
    # ─── Core tables (built inline by assembly loop) ──────────────────
    "movie": {
        "columns": [
            "title_id", "title", "year", "country", "language",
            "original_language", "aspect_ratio", "color_format",
            "genre", "production_tier", "budget_usd", "box_office_usd", "runtime_minutes",
            "rating", "num_votes", "certification", "tagline", "plot_summary",
            "franchise_id", "installment_no", "seed", "snapshot_id",
        ],
        "primary_key": "title_id",
        "auto_pk": False,
        "category": "core",
        "description": "Main movie fact table",
    },
    "cast_info": {
        "columns": [
            "title_id", "person_id", "character_name",
            "character_description", "billing_order", "archetype",
            "screen_time_minutes", "salary_usd",
        ],
        "primary_key": None,
        "auto_pk": False,
        "category": "core",
        "foreign_keys": {"title_id": "movie.title_id", "person_id": "person.person_id"},
        "description": "Cast assignments with character info, screen time, and salary",
    },
    "movie_directors": {
        "columns": ["title_id", "director_id"],
        "primary_key": None,
        "auto_pk": False,
        "category": "core",
        "foreign_keys": {"title_id": "movie.title_id", "director_id": "person.person_id"},
        "description": "Director assignments",
    },
    "movie_companies": {
        "columns": ["title_id", "company_id", "role"],
        "primary_key": None,
        "auto_pk": False,
        "category": "core",
        "foreign_keys": {"title_id": "movie.title_id", "company_id": "company.company_id"},
        "description": "Production company assignments",
    },
    "movie_keyword": {
        "columns": ["title_id", "keyword_id"],
        "primary_key": None,
        "auto_pk": False,
        "category": "core",
        "foreign_keys": {"title_id": "movie.title_id", "keyword_id": "keyword.keyword_id"},
        "description": "Keyword tagging",
    },
    "movie_crew": {
        "columns": ["crew_id", "title_id", "person_id", "crew_role", "credit_order", "department"],
        "primary_key": "crew_id",
        "auto_pk": True,
        "category": "core",
        "foreign_keys": {"title_id": "movie.title_id", "person_id": "person.person_id"},
        "description": "Below-the-line crew (writers, cinematographers, editors, composers)",
    },

    # ─── Secondary tables (auto-wired per-movie generators) ───────────
    "release_dates": {
        "columns": ["release_id", "title_id", "country", "release_type", "release_date"],
        "primary_key": "release_id",
        "auto_pk": True,
        "category": "secondary",
        "foreign_keys": {"title_id": "movie.title_id"},
        "description": "Per-title release calendar across countries",
    },
    "box_office_weekly": {
        "columns": ["box_week_id", "title_id", "week_no", "week_start_date",
                     "gross_usd_total", "gross_usd_domestic", "gross_usd_international"],
        "primary_key": "box_week_id",
        "auto_pk": True,
        "category": "secondary",
        "foreign_keys": {"title_id": "movie.title_id"},
        "description": "Weekly box office aggregates",
    },
    "box_office_by_territory": {
        "columns": ["territory_id", "title_id", "territory", "gross_usd",
                    "opening_weekend_usd", "share_pct"],
        "primary_key": "territory_id",
        "auto_pk": True,
        "category": "secondary",
        "foreign_keys": {"title_id": "movie.title_id"},
        "description": "Box office breakdown by geographic territory",
    },
    "box_office_daily": {
        "columns": ["daily_id", "title_id", "day_number", "date",
                    "gross_usd_domestic", "gross_usd_international",
                    "gross_usd_total", "cumulative_usd"],
        "primary_key": "daily_id",
        "auto_pk": True,
        "category": "secondary",
        "foreign_keys": {"title_id": "movie.title_id"},
        "description": "Daily box office for first 30 days after release",
    },
    "reviews": {
        "columns": ["review_id", "title_id", "reviewer_type", "source",
                     "rating_10", "sentiment", "review_date", "review_text"],
        "primary_key": "review_id",
        "auto_pk": True,
        "category": "secondary",
        "foreign_keys": {"title_id": "movie.title_id"},
        "description": "Synthetic critic + audience reviews",
    },
    "awards": {
        "columns": ["award_id", "title_id", "award_year", "ceremony", "category",
                     "outcome", "person_id"],
        "primary_key": "award_id",
        "auto_pk": True,
        "category": "secondary",
        "foreign_keys": {"title_id": "movie.title_id", "person_id": "person.person_id"},
        "description": "Award nominations and wins",
    },
    "locations": {
        "columns": ["location_id", "title_id", "location_order", "city",
                     "country", "location_type"],
        "primary_key": "location_id",
        "auto_pk": True,
        "category": "secondary",
        "foreign_keys": {"title_id": "movie.title_id"},
        "description": "Filming/setting locations",
    },
    "alternate_titles": {
        "columns": ["title_id", "language", "alt_title"],
        "primary_key": None,
        "auto_pk": False,
        "category": "secondary",
        "foreign_keys": {"title_id": "movie.title_id"},
        "description": "Localized/alternative titles",
    },
    "ratings_breakdown": {
        "columns": ["breakdown_id", "title_id", "age_group", "gender",
                     "vote_count", "avg_rating"],
        "primary_key": "breakdown_id",
        "auto_pk": True,
        "category": "secondary",
        "foreign_keys": {"title_id": "movie.title_id"},
        "description": "Rating breakdown by demographic segment",
    },
    "movie_links": {
        "columns": ["title_id", "linked_title_id", "link_type"],
        "primary_key": None,
        "auto_pk": False,
        "category": "secondary",
        "foreign_keys": {
            "title_id": "movie.title_id",
            "linked_title_id": "movie.title_id",
        },
        "description": "Movie-to-movie relationships (sequel, remake, etc.)",
    },

    # ─── Global tables (generated once, not per-movie) ────────────────
    "person_demographics": {
        "columns": ["person_id", "birth_date", "death_date", "birth_city",
                    "birth_country", "height_cm"],
        "primary_key": None,
        "auto_pk": False,
        "category": "global",
        "foreign_keys": {"person_id": "person.person_id"},
        "description": "Demographic data: birth/death dates, birth place, height",
    },
    "tv_series": {
        "columns": ["series_id", "title", "genre", "country", "language",
                    "network", "network_company_id", "creator_person_id",
                    "year_start", "year_end", "status", "total_seasons",
                    "overall_rating", "content_rating"],
        "primary_key": "series_id",
        "auto_pk": False,
        "category": "global",
        "foreign_keys": {
            "creator_person_id": "person.person_id",
            "network_company_id": "company.company_id",
        },
        "description": "TV series master table",
    },
    "seasons": {
        "columns": ["season_id", "series_id", "season_number", "year",
                    "num_episodes", "avg_rating"],
        "primary_key": "season_id",
        "auto_pk": False,
        "category": "global",
        "foreign_keys": {"series_id": "tv_series.series_id"},
        "description": "Season-level data within a TV series",
    },
    "episodes": {
        "columns": ["episode_id", "season_id", "series_id", "episode_number",
                    "title", "runtime_minutes", "rating", "director_person_id",
                    "air_date", "viewership_millions", "writer_person_id"],
        "primary_key": "episode_id",
        "auto_pk": False,
        "category": "global",
        "foreign_keys": {
            "season_id": "seasons.season_id",
            "series_id": "tv_series.series_id",
            "director_person_id": "person.person_id",
            "writer_person_id": "person.person_id",
        },
        "description": "Individual episodes with ratings, viewership, directors and writers",
    },
    "company_links": {
        "columns": ["company_id_1", "company_id_2", "link_type"],
        "primary_key": None,
        "auto_pk": False,
        "category": "global",
        "foreign_keys": {
            "company_id_1": "company.company_id",
            "company_id_2": "company.company_id",
        },
        "description": "Company-to-company relationships (subsidiary, co-production partner)",
    },
    "user_ratings": {
        "columns": ["rating_id", "user_id", "title_id", "rating_10", "rating_date"],
        "primary_key": "rating_id",
        "auto_pk": False,
        "category": "global",
        "foreign_keys": {"title_id": "movie.title_id"},
        "description": "High-volume synthetic user ratings (power-law distribution)",
    },
    "episode_cast": {
        "columns": ["episode_cast_id", "episode_id", "series_id", "person_id",
                    "role_type", "credit_order"],
        "primary_key": "episode_cast_id",
        "auto_pk": False,
        "category": "global",
        "foreign_keys": {
            "episode_id": "episodes.episode_id",
            "series_id": "tv_series.series_id",
            "person_id": "person.person_id",
        },
        "description": "Actor-to-episode assignments (series regulars + guests)",
    },

    # ── V14: Big History Events ────────────────────────────────────────
    "world_events": {
        "columns": [
            "event_id", "year", "event_type", "description",
            "duration_years", "affected_entity_id", "affected_entity_type",
            "parameter_delta_json",
        ],
        "primary_key": "event_id",
        "auto_pk": False,
        "category": "global",
        "foreign_keys": {},
        "description": "LLM-generated structural shocks that changed world-state parameters (V14 Big History Events)",
    },

    # ── V14: A2 Interval / Temporal Tables ──────────────────────────────
    "production_timeline": {
        "columns": ["timeline_id", "movie_id", "phase", "phase_start", "phase_end"],
        "primary_key": "timeline_id",
        "auto_pk": True,
        "category": "global",
        "foreign_keys": {"movie_id": "movie.title_id"},
        "description": "Production phase intervals (announced, pre_production, filming, post_production, released)",
    },
    "streaming_windows": {
        "columns": ["window_id", "movie_id", "platform", "window_start", "window_end", "exclusivity"],
        "primary_key": "window_id",
        "auto_pk": True,
        "category": "global",
        "foreign_keys": {"movie_id": "movie.title_id"},
        "description": "Platform streaming availability windows with temporal validity",
    },
    "person_contracts": {
        "columns": ["contract_id", "person_id", "company_id", "start_date", "end_date",
                    "salary_band", "contract_type"],
        "primary_key": "contract_id",
        "auto_pk": True,
        "category": "global",
        "foreign_keys": {"person_id": "person.person_id", "company_id": "company.company_id"},
        "description": "Exclusive/non-exclusive talent contracts enabling interval overlap queries",
    },

    # ── V14: A3 Cross-Entity Links ───────────────────────────────────
    "movie_sequence": {
        "columns": ["franchise_id", "movie_id", "sequence_no", "predecessor_movie_id"],
        "primary_key": None,
        "auto_pk": False,
        "category": "global",
        "foreign_keys": {"movie_id": "movie.title_id", "predecessor_movie_id": "movie.title_id"},
        "description": "Franchise sequence ordering with predecessor links for multi-hop traversal",
    },
    "person_collaborations": {
        "columns": ["person_a_id", "person_b_id", "collaboration_count",
                    "first_year", "last_year", "shared_genres"],
        "primary_key": None,
        "auto_pk": False,
        "category": "global",
        "foreign_keys": {"person_a_id": "person.person_id", "person_b_id": "person.person_id"},
        "description": "Materialized co-starring counts pre-computed from cast_info self-join",
    },
    "media_links": {
        "columns": ["link_id", "source_id", "source_type", "target_id", "target_type",
                    "link_type", "reason"],
        "primary_key": "link_id",
        "auto_pk": True,
        "category": "global",
        "foreign_keys": {},
        "description": "Sparse cross-media links between movies and TV series (shared_universe, adaptation)",
    },
}


# ═══════════════════════════════════════════════════════════════════════
# SECONDARY GENERATOR REGISTRY
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SecondaryGenerator:
    """Standardized interface for per-movie secondary table generators.

    Attributes:
        table_name: Must match a key in TABLE_DEFS with category='secondary'.
        generate_fn: The generator function to call.
        build_args:  Receives (movie_context dict) -> returns kwargs dict for generate_fn.
        post_hook:   Optional callback(rows, world) for side effects (e.g., award tracking).
    """
    table_name: str
    generate_fn: Callable[..., list[dict]]
    build_args: Callable[[dict], dict]
    post_hook: Callable[[list[dict], Any], None] | None = None


def _award_post_hook(rows: list[dict], world) -> None:
    """Track award wins for causal chain feedback."""
    for aw in rows:
        if aw.get("outcome") == "Won" and aw.get("person_id"):
            pid = int(aw["person_id"])
            world.person_award_wins[pid] = world.person_award_wins.get(pid, 0) + 1


def build_secondary_generators(*, disabled_tables: set[str] | None = None):
    """Build the list of secondary generators.

    Imports are deferred to avoid circular imports at module load time.
    Called once during assemble_movies() initialization.
    """
    from secondary_tables import (
        generate_release_dates, generate_box_office_weekly,
        generate_box_office_by_territory, generate_box_office_daily,
        generate_reviews, generate_awards, generate_locations,
        generate_alternate_titles, generate_ratings_breakdown,
        generate_movie_links,
    )

    disabled = {str(name).strip() for name in (disabled_tables or set()) if str(name).strip()}

    generators = [
        SecondaryGenerator(
            table_name="release_dates",
            generate_fn=generate_release_dates,
            build_args=lambda ctx: {
                "concept": ctx["concept"],
                "title_id": ctx["mid"],
                "rng": ctx["rng"],
            },
        ),
        SecondaryGenerator(
            table_name="box_office_weekly",
            generate_fn=generate_box_office_weekly,
            build_args=lambda ctx: {
                "title_id": ctx["mid"],
                "total_box_office_usd": ctx["fin"]["box_office_usd"],
                "base_release_date": ctx["base_release_date"],
                "rng": ctx["rng"],
                # D29: daily_rows set by generate_movies.py (runs daily first)
                "daily_rows": ctx.get("_daily_rows"),
            },
        ),
        SecondaryGenerator(
            table_name="reviews",
            generate_fn=generate_reviews,
            build_args=lambda ctx: {
                "title_id": ctx["mid"],
                "year": int(ctx["concept"]["year"]),
                "rating": ctx["fin"]["rating"],
                "tier": ctx["concept"]["tier"],
                "rng": ctx["rng"],
                "base_release_date": ctx["base_release_date"],
            },
        ),
        SecondaryGenerator(
            table_name="awards",
            generate_fn=generate_awards,
            build_args=lambda ctx: {
                "title_id": ctx["mid"],
                "year": int(ctx["concept"]["year"]),
                "rating": ctx["fin"]["rating"],
                "tier": ctx["concept"]["tier"],
                "director_id": ctx["director_id"],
                "cast": ctx["cast"],
                "crew_rows": ctx["crew_rows"],
                "rng": ctx["rng"],
                # D23: award_contender flag from title bank boosts nomination probability
                "award_campaign": min(1.0, ctx["fin"].get("award_campaign_strength", 0.0)
                                      + (0.3 if ctx.get("award_contender", False) else 0.0)),
                "world": ctx["world"],
            },
            post_hook=_award_post_hook,
        ),
        SecondaryGenerator(
            table_name="locations",
            generate_fn=generate_locations,
            build_args=lambda ctx: {
                "title_id": ctx["mid"],
                "country": ctx["concept"]["country"],
                "tier": ctx["concept"]["tier"],
                "rng": ctx["rng"],
            },
        ),
        SecondaryGenerator(
            table_name="box_office_by_territory",
            generate_fn=generate_box_office_by_territory,
            build_args=lambda ctx: {
                "title_id": ctx["mid"],
                "total_box_office_usd": ctx["fin"]["box_office_usd"],
                "origin_country": ctx["concept"]["country"],
                "tier": ctx["concept"]["tier"],
                "rng": ctx["rng"],
            },
        ),
        SecondaryGenerator(
            table_name="box_office_daily",
            generate_fn=generate_box_office_daily,
            build_args=lambda ctx: {
                "title_id": ctx["mid"],
                "total_box_office_usd": ctx["fin"]["box_office_usd"],
                "base_release_date": ctx["base_release_date"],
                "rng": ctx["rng"],
            },
        ),
        SecondaryGenerator(
            table_name="alternate_titles",
            generate_fn=generate_alternate_titles,
            build_args=lambda ctx: {
                "title_id": ctx["mid"],
                "title": ctx["title"],
                "language": ctx["concept"]["language"],
                "rng": ctx["rng"],
            },
        ),
        SecondaryGenerator(
            table_name="ratings_breakdown",
            generate_fn=generate_ratings_breakdown,
            build_args=lambda ctx: {
                "title_id": ctx["mid"],
                "rating": ctx["fin"]["rating"],
                "num_votes": ctx["fin"]["num_votes"],
                "rng": ctx["rng"],
            },
        ),
        SecondaryGenerator(
            table_name="movie_links",
            generate_fn=generate_movie_links,
            build_args=lambda ctx: {
                "title_id": ctx["mid"],
                "genre": ctx["concept"]["genre"],
                "year": int(ctx["concept"]["year"]),
                "previous_movies": ctx["previous_movies_for_links"],
                "rng": ctx["rng"],
                "concept": ctx["concept"],
            },
        ),
    ]

    if not disabled:
        return generators
    return [generator for generator in generators if generator.table_name not in disabled]


# ═══════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def get_tables_by_category(category: str) -> list[str]:
    """Return table names for a given category."""
    return [name for name, tdef in TABLE_DEFS.items() if tdef["category"] == category]


def get_auto_pk_tables() -> dict[str, str]:
    """Return {table_name: pk_column} for tables needing auto-generated PKs."""
    return {
        name: tdef["primary_key"]
        for name, tdef in TABLE_DEFS.items()
        if tdef.get("auto_pk") and tdef.get("primary_key")
    }


def validate_table_output(table_name: str, rows: list[dict]) -> list[str]:
    """Validate that output rows match the declared column schema.

    Returns list of warning messages (empty = valid).
    """
    tdef = TABLE_DEFS.get(table_name)
    if not tdef or not rows:
        return []

    expected = set(tdef["columns"])
    # Skip auto-PK column (added after assembly)
    if tdef.get("auto_pk") and tdef.get("primary_key"):
        expected.discard(tdef["primary_key"])

    actual = set(rows[0].keys())
    warnings = []

    missing = expected - actual
    extra = actual - expected
    if missing:
        warnings.append(f"[{table_name}] Missing columns: {sorted(missing)}")
    if extra:
        warnings.append(f"[{table_name}] Unexpected columns: {sorted(extra)}")

    return warnings
"""
Complexity: 5
Description: New file defining 15 table schemas, 8 secondary generators with auto-wiring, and utility functions for validation and PK assignment.
"""
