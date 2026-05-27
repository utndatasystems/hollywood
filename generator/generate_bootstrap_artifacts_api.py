from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Callable

from bootstrap_artifacts import current_mode, load_modeling_priors_artifact, prior_section
from contracts import BUDGET_RANGES, CERTIFICATIONS, CERT_DISTS, COUNTRIES, GENRES, MARKETS, NATIONALITIES, PRODUCTION_TIERS, RELATIONSHIP_TARGETS
from financials import (
    COUNTRY_BUDGET_SCALE,
    GENRE_RATING_OFFSET,
    TIER_LOG_CENTER,
    TIER_MIN_VOTES,
    TIER_RATING_BASE,
    TIER_RATING_STD,
    _DEFAULT_AWARD_CAMPAIGN_WEIGHTS,
    _DEFAULT_MARKET_LATENT_WEIGHTS,
    _DEFAULT_MARKET_REGIME,
    _DEFAULT_PERFORMANCE_MODEL,
    _DEFAULT_QUALITY_LATENT_WEIGHTS,
    _DEFAULT_RUNTIME_MODEL,
    _DEFAULT_VOTE_MODEL,
    _DEFAULT_YEAR_QUALITY,
)
from llm_provider import get_llm_client, safe_json_parse
from model_defaults import model_for_role
from text_polish import clean_display_text, contains_placeholder_syntax, looks_like_weak_tagline
from policy_runtime import (
    character_identity_bank_path,
    company_lexicon_path,
    identity_bank_path,
    keyword_seed_bank_path,
    modeling_priors_path,
    temporal_regime_plan_path,
    title_grammar_bank_path,
)


BASE_DIR = Path(__file__).resolve().parent


def _artifact_specs() -> dict[str, dict[str, Any]]:
    return {
        "identity_bank": {
            "path": identity_bank_path,
            "default_model": model_for_role("artifact_mid"),
            "temperature": 0.35,
            "max_tokens": 18000,
        },
        "character_identity_bank": {
            "path": character_identity_bank_path,
            "default_model": model_for_role("artifact_pro"),
            "temperature": 0.45,
            "max_tokens": 18000,
        },
        "company_lexicon": {
            "path": company_lexicon_path,
            "default_model": model_for_role("artifact_pro"),
            "temperature": 0.40,
            "max_tokens": 16000,
        },
        "keyword_seed_bank": {
            "path": keyword_seed_bank_path,
            "default_model": model_for_role("artifact_mid"),
            "temperature": 0.30,
            "max_tokens": 12000,
        },
        "title_grammar_bank": {
            "path": title_grammar_bank_path,
            "default_model": model_for_role("artifact_pro"),
            "temperature": 0.35,
            "max_tokens": 16000,
        },
        "temporal_regime_plan": {
            "path": temporal_regime_plan_path,
            "default_model": model_for_role("artifact_pro"),
            "temperature": 0.20,
            "max_tokens": 16000,
        },
        "modeling_priors": {
            "path": modeling_priors_path,
            "default_model": model_for_role("artifact_pro"),
            "temperature": 0.15,
            "max_tokens": 16000,
        },
    }


def _context_block(args: argparse.Namespace) -> str:
    return (
        f"Movies: {int(args.n_movies)}\n"
        f"Persons: {int(args.n_persons)}\n"
        f"Companies: {int(args.n_companies)}\n"
        f"Keywords: {int(args.n_keywords)}\n"
        f"Titles: {int(args.n_titles)}\n"
        f"Year range: {int(args.start_year)}-{int(args.end_year)}\n"
        f"Markets: {', '.join(MARKETS)}\n"
        f"Core genres: {', '.join(GENRES)}\n"
        f"Production tiers: {', '.join(PRODUCTION_TIERS)}"
    )


def _prompt_identity_bank(args: argparse.Namespace) -> str:
    nationalities = ", ".join(NATIONALITIES[:48])
    return f"""You are creating a reusable identity bank for a synthetic film-industry database.

This is NOT a row generator. Create a reusable structured bank that a deterministic generator can expand.

Context:
{_context_block(args)}

Requirements:
- Output a JSON object only.
- Provide 18-28 identity families across diverse regions.
- Each family must include:
  - nationality
  - region
  - weight
  - first_m (18-32 entries)
  - first_f (18-32 entries)
  - first_nb (8-18 entries)
  - surnames (24-40 entries)
  - connectors (0-6 entries, like de/van/bin when culturally appropriate)
  - double_surname_probability
  - middle_name_probability
  - middle_initial_probability
  - suffix_probability
  - market_bias (1-3 entries chosen from {MARKETS})
- Also include top-level defaults:
  - role_distribution
  - stage_weights
  - gender_weights
- Names must be realistic and varied, but avoid overusing celebrity names.
- Optimize for deterministic combinatorial expansion without numeric dedupe.

Suggested nationalities to cover include:
{nationalities}

Return JSON with shape:
{{
  "families": [{{ ... }}],
  "defaults": {{
    "role_distribution": {{"actor": 0.55, "director": 0.08}},
    "stage_weights": {{"rising": 0.30, "prime": 0.35, "veteran": 0.20, "legend": 0.10, "retired": 0.05}},
    "gender_weights": {{"M": 0.45, "F": 0.45, "NB": 0.10}}
  }}
}}
"""


_IDENTITY_BANK_GROUPS: tuple[dict[str, Any], ...] = (
    {
        "name": "americas_europe",
        "target_families": 6,
        "temperature": 0.25,
        "max_tokens": 7000,
        "nationalities": (
            "American", "British", "French", "German", "Canadian", "Italian", "Spanish", "Swedish",
            "Irish", "Scottish", "Dutch", "Norwegian", "Danish", "Greek", "Polish", "Portuguese",
            "Belgian", "Czech", "Croatian", "Hungarian", "Romanian", "Swiss", "Serbian", "Welsh",
        ),
    },
    {
        "name": "asia_pacific",
        "target_families": 6,
        "temperature": 0.28,
        "max_tokens": 7000,
        "nationalities": (
            "Indian", "South Korean", "Japanese", "Chinese", "Australian", "Thai", "Filipino", "Vietnamese",
            "Indonesian", "New Zealander", "Pakistani", "Bangladeshi", "Hong Konger", "Malaysian",
            "Mongolian", "Nepali", "Sri Lankan", "Korean",
        ),
    },
    {
        "name": "middle_east_africa",
        "target_families": 5,
        "temperature": 0.28,
        "max_tokens": 6500,
        "nationalities": (
            "Iranian", "Turkish", "Egyptian", "Israeli", "Lebanese", "Ghanaian", "Kenyan", "South African",
            "Moroccan", "Tunisian", "Iraqi", "Jordanian", "Omani", "Saudi", "Saudi Arabian", "Emirati",
            "Palestinian", "Syrian", "Ethiopian", "Tanzanian", "Zimbabwean", "Beninese", "Congolese",
            "Senegalese", "Cape Verdean", "South Sudanese", "Kuwaiti", "Qatari", "Bahraini",
        ),
    },
    {
        "name": "latin_america_eurasia",
        "target_families": 5,
        "temperature": 0.28,
        "max_tokens": 6500,
        "nationalities": (
            "Brazilian", "Mexican", "Argentine", "Argentinian", "Colombian", "Cuban", "Chilean", "Ecuadorian",
            "Peruvian", "Uruguayan", "Venezuelan", "Dominican", "Costa Rican", "Guatemalan", "Honduran",
            "Puerto Rican", "Armenian", "Azerbaijani", "Belarusian", "Bulgarian", "Estonian", "Finnish",
            "Georgian", "Icelandic", "Kazakh", "Kazakhstani", "Luxembourgish", "Macedonian", "Russian",
            "Ukrainian", "Uzbek", "Bolivian",
        ),
    },
)


def _prompt_identity_bank_group(
    args: argparse.Namespace,
    group_name: str,
    nationalities: tuple[str, ...],
    target_families: int,
) -> str:
    allowed = ", ".join(nationalities)
    return f"""Create one grouped subset of a reusable identity bank for a synthetic film-industry database.

This is NOT a row generator. Create reusable identity families that a deterministic generator can expand.

Context:
{_context_block(args)}

Group:
- name: {group_name}
- target family count: {int(target_families)}
- allowed nationalities only: {allowed}

Requirements:
- Output a JSON object only.
- Return exactly one top-level key: `families`.
- `families` must contain exactly {int(target_families)} family objects.
- Use only nationalities from the allowed list above.
- Do not include `defaults` in this grouped response.
- Each family must include:
  - nationality
  - region
  - weight
  - first_m (18-32 entries)
  - first_f (18-32 entries)
  - first_nb (8-18 entries)
  - surnames (24-40 entries)
  - connectors (0-6 entries, only when culturally appropriate)
  - double_surname_probability
  - middle_name_probability
  - middle_initial_probability
  - suffix_probability
  - market_bias (1-3 entries chosen from {MARKETS})
- Avoid celebrity-heavy pools and avoid numeric dedupe assumptions.
- Keep regional diversity inside the allowed set where possible.
"""


def _prompt_identity_bank_defaults(args: argparse.Namespace) -> str:
    return f"""Create the defaults block for a reusable identity bank used by a synthetic film-industry database.

Context:
{_context_block(args)}

Requirements:
- Output a JSON object only.
- Return exactly one top-level key: `defaults`.
- `defaults` must include:
  - role_distribution
  - stage_weights
  - gender_weights
- `role_distribution` should cover the main film-industry roles used by person generation.
- `stage_weights` should cover rising, prime, veteran, legend, retired.
- `gender_weights` should cover M, F, NB.
- Keep the values plausible for a broad global film-industry dataset and normalize them approximately to 1.0.
"""


def _prompt_character_identity_bank(args: argparse.Namespace) -> str:
    return f"""You are creating a reusable character-identity bank for a synthetic film database.

This bank should support both human-style character names and non-literal monikers, codenames,
epithets, mythic aliases, job-title identities, and iconic villain/hero labels.

Context:
{_context_block(args)}

Requirements:
- Output JSON object only.
- Include:
  - archetype_weights
  - title_prefixes_m
  - title_prefixes_f
  - title_prefixes_nb
  - quote_nicknames
  - solo_monikers
  - codename_adjectives
  - codename_nouns
  - mythic_epithets
  - role_epithets
  - alias_templates
  - nonliteral_share_by_archetype
  - human_name_mix_by_archetype
- Use only these canonical alias template placeholders:
  - {{title}}, {{title_prefix}}, {{first}}, {{human_first}}, {{first_name}}, {{given_name}}, {{human_first_name}}
  - {{surname}}, {{human_last}}, {{human_surname}}, {{last_name}}, {{family_name}}, {{human_last_name}}
  - {{full_name}}, {{human_name}}, {{nickname}}, {{quote_nickname}}, {{moniker}}, {{solo_moniker}}
  - {{codename_adj}}, {{codename_adjective}}, {{codename_noun}}, {{mythic_epithet}}, {{role_epithet}}
- Do not invent gendered placeholder variants like {{title_m}}, {{title_f}}, {{title_prefix_m}}, or {{title_prefix_nb}}.
- `nonliteral_share_by_archetype` and `human_name_mix_by_archetype` must cover every archetype that appears in `archetype_weights`.
- Provide enough entries to support large-scale deterministic expansion.
- Ensure a meaningful share of non-human-style names for archetypes like villains, mysterious strangers, mentors, and comic figures.
- Avoid direct reuse of ordinary person naming as the only mode.
"""


def _prompt_company_lexicon(args: argparse.Namespace) -> str:
    countries = ", ".join(COUNTRIES[:64])
    return f"""Create a reusable company lexicon for synthetic film company generation.

Context:
{_context_block(args)}

Requirements:
- Output JSON object only.
- Include:
  - prefixes
  - suffixes
  - abstract_nouns
  - material_words
  - geographic_words
  - motion_words
  - mythic_words
  - templates (using placeholders like {{prefix}}, {{abstract}}, {{suffix}}, {{geo}}, {{motion}}, {{mythic}})
  - tier_styles for tiers {PRODUCTION_TIERS}
  - country_style_bias for representative countries
- Use only these canonical template placeholders:
  - {{prefix}}, {{suffix}}, {{abstract}}, {{material}}, {{geo}}, {{motion}}, {{mythic}}
- Do not invent alias placeholders like {{local_word}} or {{abstract_word}}.
- Names should feel like production/distribution/media companies.
- No numeric dedupe assumptions.
- Bias toward combinations that remain plausible when expanded deterministically.

Representative countries:
{countries}
"""


def _prompt_keyword_seed_bank(args: argparse.Namespace) -> str:
    return f"""Create a reusable keyword seed bank for synthetic movie metadata.

Context:
{_context_block(args)}

Requirements:
- Output JSON object only.
- Include top-level:
  - universal_qualifiers (30-60)
  - universal_contexts (24-50)
  - generic_themes (40-80)
- Include per-genre entries for each of {GENRES} with:
  - genre
  - seeds (18-36)
  - qualifiers (8-18)
  - contexts (8-18)
  - tone_tokens (6-14)
  - exclusion_hints (3-8)
- Every benchmark genre in {GENRES} must appear exactly once in the genre list. Do not omit any and do not substitute aliases.
- Seeds should be specific enough to build benchmark-hard join structures, not just generic emotions.
- Prefer cinematic metadata phrases over dry institutional or task labels.
- Avoid overusing literal bureaucratic or technical process phrases unless they are strongly genre-appropriate.
"""


def _prompt_keyword_seed_bank_globals(args: argparse.Namespace) -> str:
    return f"""Create the global top-level buckets for a reusable keyword seed bank for synthetic movie metadata.

Context:
{_context_block(args)}

Requirements:
- Output JSON object only.
- Return exactly these top-level keys:
  - universal_qualifiers (30-60)
  - universal_contexts (24-50)
  - generic_themes (40-80)
- Do not include any per-genre rows in this response.
- Prefer cinematic metadata phrases that remain reusable across many genres and decades.
- Avoid vague filler such as "interesting", "dramatic", or "good acting" unless embedded in a stronger metadata phrase.
"""


def _prompt_keyword_seed_bank_genre_group(args: argparse.Namespace, genres: Sequence[str]) -> str:
    genre_list = ", ".join(str(genre) for genre in genres)
    return f"""Create one grouped subset of a reusable keyword seed bank for synthetic movie metadata.

Context:
{_context_block(args)}

Requested benchmark genres:
{genre_list}

Requirements:
- Output JSON object only.
- Return exactly one top-level key: `genres`.
- Include exactly one row for each requested genre above. Do not omit any and do not include extra genres.
- Each row must include:
  - genre
  - seeds (18-36)
  - qualifiers (8-18)
  - contexts (8-18)
  - tone_tokens (6-14)
  - exclusion_hints (3-8)
- Do not include universal_qualifiers, universal_contexts, or generic_themes in this grouped response.
- Seeds should be benchmark-useful metadata anchors, not generic emotions or overly abstract themes.
- Prefer terms that support realistic joins, filters, and selectivity patterns in movie metadata.
"""


def _prompt_title_grammar_bank(args: argparse.Namespace) -> str:
    return f"""Create a reusable title grammar bank for synthetic film title generation.

Context:
{_context_block(args)}

Requirements:
- Output JSON object only.
- Include:
  - adjectives
  - nouns
  - abstract_nouns
  - locations
  - celestial_words
  - technology_words
  - mythic_words
  - action_words
  - franchise_affixes
  - genre_templates (per genre in {GENRES})
  - tagline_templates (per genre, with at least 12 templates for every benchmark genre)
  - year_style_phases (4-8 phases with relative-position labels, favored tokens, and title/tagline tendencies)
- Templates should use placeholders like {{adjective}}, {{noun}}, {{location}}, {{franchise_affix}}.
- Titles should feel cinematic and marketable, not like random placeholder word collisions.
- Avoid awkward constructions such as repeated words, nonsensical noun pairs, or question forms built from arbitrary tokens.
- Taglines must be varied and not collapse to 2 repeated phrases.
- Every benchmark genre must have at least 12 distinct tagline templates after exact deduplication.
- Do not reuse the exact same tagline template across multiple genres unless you intentionally place it under a shared default/genre-neutral pool.
- Most tagline templates should read like actual marketing copy, not just bare noun phrases.
- Prefer thematic conflict, stakes, irony, emotional hook, or world intrigue over generic word salad.
- Include enough per-genre tagline variety that rerolling can avoid weak matches.
- Avoid ultra-short noun-phrase taglines that look like alternate titles or sequel subtitles.
- Make at least half of the tagline templates read like a complete marketing thought, not just a branded phrase.
- If a tagline template uses placeholders, keep it grammatical when placeholders are replaced with plain lower-case words.
- Use only canonical curly-brace placeholders. Never use square-bracket placeholders like [Hero] or [Location].
- Avoid tagline templates that depend on awkward token collisions such as "his {{noun}}" with arbitrary location words or transitive clauses that would break with intransitive verbs.
- Prefer zero or one placeholder in tagline templates; use two placeholders only when the sentence still reads like natural marketing copy.
- Mix sentence shapes: complete-sentence marketing copy, irony/hook lines, and a smaller number of one-placeholder punch lines.
- Support both historical and future ranges.
- Keep the bank concise and reusable: short lists, compact templates, no commentary, and no redundant near-duplicate wording.
"""


_TITLE_GRAMMAR_VOCAB_KEYS: tuple[str, ...] = (
    "adjectives",
    "nouns",
    "abstract_nouns",
    "locations",
    "celestial_words",
    "technology_words",
    "mythic_words",
    "action_words",
    "franchise_affixes",
)

_MIN_TITLE_GRAMMAR_VOCAB_ITEMS = 48
_MIN_TITLE_TEMPLATES_PER_GENRE = 18


_TITLE_GRAMMAR_BASE_PLACEHOLDER_ALIASES: dict[str, str] = {
    "adjective": "adjective",
    "adjectives": "adjective",
    "noun": "noun",
    "nouns": "noun",
    "plural_noun": "noun",
    "plural_nouns": "noun",
    "abstract": "abstract",
    "abstract_noun": "abstract",
    "abstract_nouns": "abstract",
    "location": "location",
    "location_word": "location",
    "locations": "location",
    "setting": "location",
    "celestial": "celestial",
    "celestial_word": "celestial",
    "celestial_words": "celestial",
    "technology": "technology",
    "technology_word": "technology",
    "technology_words": "technology",
    "mythic": "mythic",
    "mythic_word": "mythic",
    "mythic_words": "mythic",
    "action": "action",
    "action_word": "action",
    "action_words": "action",
    "franchise_affix": "franchise_affix",
    "franchise_suffix": "franchise_affix",
    "franchise_prefix": "franchise_affix",
    "franchise_affixes": "franchise_affix",
}

_TITLE_GRAMMAR_CONTROLLED_PLACEHOLDERS: set[str] = {
    "adjective",
    "noun",
    "abstract",
    "location",
    "celestial",
    "technology",
    "mythic",
    "action",
    "franchise_affix",
}

_TITLE_GRAMMAR_SEMANTIC_ALIAS_BUCKETS: dict[str, set[str]] = {
    "abstract": {
        "alibi", "betrayal", "chaos", "conflict", "conformity", "concept", "conspiracy", "crime",
        "curse", "danger", "dream", "emotion", "event", "friendship", "history", "idea", "identity",
        "illusion", "impossible", "justice", "law", "legacy", "lies", "love", "madness", "mystery",
        "origin", "past", "power", "problem", "revenge", "romance", "sacrifice", "scandal", "secret",
        "secrets", "sin", "strength", "struggle", "threat", "time", "tragedy", "truth", "unknown",
        "wits", "career", "journey", "vacation", "weekend", "scale",
    },
    "location": {
        "boundary", "city", "destination", "empire", "environment", "kingdom", "landscape",
        "road_trip", "terrain", "underworld", "universe", "world", "civilization",
    },
    "technology": {
        "industry", "invention", "resource", "system", "vehicle", "weapon",
    },
    "mythic": {
        "legend", "magic", "monster", "myth", "spirit",
    },
    "noun": {
        "animal", "artifact", "body_part", "boss", "cartel", "catastrophe", "champion", "character",
        "clues", "creature", "criminals", "detective", "diamonds", "employee", "enemy", "expedition",
        "family", "figure", "genius", "group", "hero", "historical_figure", "icon", "in_laws", "item",
        "job", "killer", "leader", "man", "map", "masterpiece", "mob", "name", "object", "organization",
        "person", "pet", "prize", "profession", "relatives", "relic", "role", "show", "suspect",
        "syndicate", "treasure", "trial", "villains", "voice",
    },
}


def _normalize_title_placeholder_token(token: object) -> str:
    value = str(token or "").strip()
    value = value.replace("-", "_").replace(" ", "_")
    value = re.sub(r"[^A-Za-z0-9_]", "", value)
    value = re.sub(r"_+", "_", value).strip("_").casefold()
    direct = _TITLE_GRAMMAR_BASE_PLACEHOLDER_ALIASES.get(value)
    if direct:
        return direct
    if value in _TITLE_GRAMMAR_CONTROLLED_PLACEHOLDERS:
        return value
    if value in {"fight", "fights", "escape", "escapes", "hunt", "hunts", "run", "runs", "survive", "survives"}:
        return "action"
    for bucket, names in _TITLE_GRAMMAR_SEMANTIC_ALIAS_BUCKETS.items():
        if value in names:
            return bucket
    parts = [part for part in value.split("_") if part]
    for part in parts:
        direct = _TITLE_GRAMMAR_BASE_PLACEHOLDER_ALIASES.get(part)
        if direct:
            return direct
    for bucket, names in _TITLE_GRAMMAR_SEMANTIC_ALIAS_BUCKETS.items():
        if any(part in names for part in parts):
            return bucket
    return value


def _canonicalize_title_template_placeholders(text: object) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    value = re.sub(r"\[([^\[\]]+)\]", lambda match: "{" + _normalize_title_placeholder_token(match.group(1)) + "}", value)
    value = re.sub(r"\{([^{}]+)\}", lambda match: "{" + _normalize_title_placeholder_token(match.group(1)) + "}", value)
    return re.sub(r"\s+", " ", value).strip()


def _title_template_has_square_brackets(text: object) -> bool:
    return bool(re.search(r"\[[^\[\]]+\]", str(text or "")))


def _title_template_placeholders(text: object) -> set[str]:
    value = _canonicalize_title_template_placeholders(text)
    return {
        _normalize_title_placeholder_token(token)
        for token in re.findall(r"\{([^{}]+)\}", value)
        if _normalize_title_placeholder_token(token)
    }


def _title_template_uses_only_controlled_placeholders(text: object) -> bool:
    return _title_template_placeholders(text).issubset(_TITLE_GRAMMAR_CONTROLLED_PLACEHOLDERS)


def _title_smoke_render_values(payload: dict[str, Any]) -> dict[str, list[str]]:
    values = {
        "adjective": [str(item) for item in list(payload.get("adjectives") or ["silent", "reckless", "golden"])[:6] if str(item).strip()],
        "noun": [str(item) for item in list(payload.get("nouns") or ["signal", "shadow", "threshold"])[:6] if str(item).strip()],
        "abstract": [str(item) for item in list(payload.get("abstract_nouns") or ["legacy", "truth", "desire"])[:6] if str(item).strip()],
        "location": [str(item) for item in list(payload.get("locations") or ["harbor", "frontier", "district"])[:6] if str(item).strip()],
        "celestial": [str(item) for item in list(payload.get("celestial_words") or ["eclipse", "aurora", "orbit"])[:6] if str(item).strip()],
        "technology": [str(item) for item in list(payload.get("technology_words") or ["protocol", "network", "circuit"])[:6] if str(item).strip()],
        "mythic": [str(item) for item in list(payload.get("mythic_words") or ["oracle", "phoenix", "titan"])[:6] if str(item).strip()],
        "action": [str(item) for item in list(payload.get("action_words") or ["fight", "survive", "betray"])[:6] if str(item).strip()],
        "franchise_affix": [str(item) for item in list(payload.get("franchise_affixes") or ["legacy", "returns", "origins"])[:6] if str(item).strip()],
    }
    extra_values = payload.get("tagline_placeholder_values")
    if isinstance(extra_values, dict):
        for key, raw in extra_values.items():
            placeholder = _normalize_title_placeholder_token(key)
            bucket = [str(item).strip() for item in list(raw or []) if str(item).strip()]
            if bucket:
                values[placeholder] = bucket[:8]
    return values


def _smoke_render_title_template(template: object, payload: dict[str, Any]) -> str:
    text = _canonicalize_title_template_placeholders(template)
    smoke_values = _title_smoke_render_values(payload)
    rendered = text
    for token in re.findall(r"\{([^{}]+)\}", text):
        placeholder = _normalize_title_placeholder_token(token)
        bucket = smoke_values.get(placeholder) or []
        replacement = str(bucket[0]).strip() if bucket else placeholder.replace("_", " ")
        rendered = rendered.replace("{" + token + "}", replacement)
    return clean_display_text(rendered)


def _validate_smoke_rendered_tagline(template: object, payload: dict[str, Any]) -> None:
    rendered = _smoke_render_title_template(template, payload)
    if contains_placeholder_syntax(rendered):
        raise ValueError(f"title_grammar_bank template did not fully render: {template}")
    if looks_like_weak_tagline(rendered):
        raise ValueError(f"title_grammar_bank template smoke-rendered weak tagline: {template}")


def _genre_chunks(items: Sequence[str], size: int) -> list[tuple[str, ...]]:
    rows = [str(item) for item in items]
    return [tuple(rows[idx: idx + size]) for idx in range(0, len(rows), size)]


def _prompt_title_grammar_vocab(args: argparse.Namespace) -> str:
    keys = "\n".join(f"  - {key}" for key in _TITLE_GRAMMAR_VOCAB_KEYS)
    return f"""Create the reusable vocabulary pools for a synthetic film title grammar bank.

Context:
{_context_block(args)}

Requirements:
- Output one JSON object only.
- Include exactly these top-level keys:
{keys}
- Each key must map to a list of at least {_MIN_TITLE_GRAMMAR_VOCAB_ITEMS} short, cinematic tokens.
- Keep tokens reusable across historical and future ranges.
- Avoid near-duplicate variants, obvious junk, or explanatory text.
"""


def _prompt_title_grammar_genre_templates(args: argparse.Namespace, genres: Sequence[str]) -> str:
    genre_list = ", ".join(str(genre) for genre in genres)
    return f"""Create reusable title templates for these benchmark genres: {genre_list}.

Context:
{_context_block(args)}

Requirements:
- Output one JSON object only.
- Include a top-level key `genre_templates`.
- `genre_templates` must be a JSON object keyed exactly by these genres: {genre_list}.
- Each genre must have at least {_MIN_TITLE_TEMPLATES_PER_GENRE} distinct title templates.
- Most templates should use at least two controlled placeholders so the bank scales to 100k-200k titles without duplicate exhaustion.
- Templates should use placeholders like {{adjective}}, {{noun}}, {{location}}, {{franchise_affix}}.
- Templates must feel cinematic and grammatical, not like placeholder collisions.
"""


def _prompt_title_grammar_taglines(args: argparse.Namespace, genres: Sequence[str]) -> str:
    genre_list = ", ".join(str(genre) for genre in genres)
    return f"""Create reusable tagline templates for these benchmark genres: {genre_list}.

Context:
{_context_block(args)}

Requirements:
- Output one JSON object only.
- Include a top-level key `tagline_templates`.
- `tagline_templates` must be a JSON object keyed exactly by these genres: {genre_list}.
- Every genre should provide 16 to 20 distinct tagline templates so weak or duplicate lines can be discarded safely.
- Most lines should read like real marketing copy rather than noun phrases.
- Mix complete-sentence hook lines, irony/stakes lines, and a smaller number of one-placeholder punch lines.
- Avoid awkward grammar, repeated slogans, or templates that collapse into the same sentence shape.
- Avoid count-up/count-down slogan structures like "One X. Two Y. Zero Z." or "No X. No Y. No problem."
- Use only these canonical placeholders when needed: {{adjective}}, {{noun}}, {{abstract}}, {{location}}, {{celestial}}, {{technology}}, {{mythic}}, {{action}}, {{franchise_affix}}.
- Never invent semantic placeholders like {{betrayal}}, {{hero}}, {{city}}, or {{concept}}.
- Never use square-bracket placeholders like [Hero].
- Do not reuse the exact same tagline template across these genres.
"""


def _prompt_title_grammar_tagline_supplement(
    args: argparse.Namespace,
    genre: str,
    needed: int,
    avoid_templates: Sequence[str],
) -> str:
    avoid_block = "\n".join(f"  - {template}" for template in list(avoid_templates)[:60]) or "  - none"
    return f"""Create additional reusable tagline templates for the benchmark genre `{genre}`.

Context:
{_context_block(args)}

Requirements:
- Output one JSON object only.
- Include a top-level key `tagline_templates`.
- `tagline_templates` must be an object with exactly one key: `{genre}`.
- Return at least {needed} distinct tagline templates for `{genre}`.
- The new templates must not repeat any item from the avoid list below.
- Most lines should read like real marketing copy rather than noun phrases.
- Mix complete-sentence hook lines, irony/stakes lines, and a smaller number of one-placeholder punch lines.
- Avoid awkward grammar, repeated slogans, and generic filler.
- Avoid count-up/count-down slogan structures like "One X. Two Y. Zero Z." or "No X. No Y. No problem."
- Use only these canonical placeholders when needed: {{adjective}}, {{noun}}, {{abstract}}, {{location}}, {{celestial}}, {{technology}}, {{mythic}}, {{action}}, {{franchise_affix}}.
- Never invent semantic placeholders like {{betrayal}}, {{hero}}, {{city}}, or {{concept}}.
- Never use square-bracket placeholders like [Hero].

Avoid these existing templates:
{avoid_block}
"""


def _prompt_title_grammar_phases(args: argparse.Namespace) -> str:
    return f"""Create the year-style phase metadata for a reusable synthetic film title grammar bank.

Context:
{_context_block(args)}

Requirements:
- Output one JSON object only.
- Include a top-level key `year_style_phases`.
- `year_style_phases` must be a list with 4 to 8 phase objects.
- Each phase object must include:
  - label
  - range_hint
  - favored_tokens
  - title_tendencies
  - tagline_tendencies
- Keep the phases relative and reusable across arbitrary year ranges.
"""


def _prompt_title_grammar_placeholder_values(
    args: argparse.Namespace,
    placeholders: Sequence[str],
) -> str:
    placeholder_list = ", ".join(str(item) for item in placeholders)
    return f"""Create reusable value banks for extra title-grammar tagline placeholders.

Context:
{_context_block(args)}

Requirements:
- Output one JSON object only.
- Include a top-level key `tagline_placeholder_values`.
- `tagline_placeholder_values` must be an object keyed exactly by: {placeholder_list}
- For every placeholder, provide at least 8 short replacement values.
- Values must be reusable across many films and should read naturally inside marketing taglines.
- Avoid placeholder names, comments, explanations, or template braces in the values.
"""


def _prompt_temporal_regime_plan(args: argparse.Namespace) -> str:
    span = max(1, int(args.end_year) - int(args.start_year) + 1)
    buckets = max(4, min(10, math.ceil(span / 8)))
    return f"""Create a temporal regime plan for a synthetic film industry over an arbitrary year range.

Context:
{_context_block(args)}

Requirements:
- Output JSON object only.
- Range is exactly {int(args.start_year)} to {int(args.end_year)}.
- Include:
  - start_year
  - end_year
  - year_weights: one entry per year with positive weight, non-uniform across the range
  - phases: 4-{buckets} ordered regime phases with labels, movie_density, economic_heat, prestige_bias, franchise_pressure, experimentation_bias
  - country_ramps: representative countries with ramp_up or ramp_down tendencies over time
  - debut_cadence
  - retirement_cadence
  - sequel_spacing
  - graph_evolution
  - historical_event_cadence
- Do NOT make the distribution uniform.
- Future-only ranges must still have rises, plateaus, drops, clustered eras, and macro-cycle changes.
"""


def _prompt_modeling_priors(args: argparse.Namespace) -> str:
    return f"""Create a modeling priors object for a synthetic film database generation pipeline.

Context:
{_context_block(args)}

Requirements:
- Output JSON object only.
- Include these top-level sections:
  - person_generation
  - company_generation
  - keyword_generation
  - character_generation
  - title_generation
  - company_finance_tiers
  - selection_weights
  - edge_priors
  - scalable_edge_priors
  - financial_priors
  - secondary_table_priors
  - history_event_priors
  - rerank_priors
- Values should be numeric, stable, and explainable.
- Focus on reusable priors, not prose.
- Avoid absolute historical year assumptions; use relative biases where possible.
- Keep safety-friendly bounds (probabilities in [0,1], weights positive).
- The following keys should be present so the runtime can consume them directly:
  - title_generation:
    - genre_base_weights (dict keyed by genre, non-uniform, summing approximately to 1.0)
    - prestige_genres (genre list)
    - franchise_genres (genre list)
    - experimental_genres (genre list)
    - low_cost_genres (genre list)
    - allowed_tagline_placeholders (list of canonical placeholder names)
    - tagline_render_constraints (dict with min_words, max_words, max_placeholder_count, forbid_square_brackets, allow_unresolved_placeholders)
  - keyword_generation:
    - genre_target_weights (dict keyed by benchmark genre)
    - generic_budget_ratio
    - min_specific_story_share
    - selection_bucket_targets (dict with exact_anchor, related_support, story_specific, generic summing approximately to 1.0)
  - selection_weights:
    - cast_base_focus_exploration
    - cast_focus_exploration
    - cast_slot_exploration_empty
    - cast_slot_exploration_filled
    - cast_style_multiplier
    - cast_community_match_multiplier
    - director_exploration_share
    - company_primary_exploration_share
    - company_secondary_exploration_share
    - crew_exploration_share
    - cast_style_multiplier
    - director_risk_weight
    - director_ambition_weight
    - director_prestige_weight
    - director_alignment_base
    - director_alignment_scale
    - director_csv_base
    - director_csv_scale
    - company_tier_match_boost
    - company_tier_mismatch_penalty
    - company_genre_match_boost
    - company_genre_mismatch_penalty
    - company_risk_weight
    - company_prestige_weight
    - company_focus_weight
    - company_genre_fit_weight
    - company_alignment_base
    - company_alignment_scale
    - optional but strongly preferred:
      - concept_genre_bias_base
      - concept_genre_bias_scale
      - concept_country_bias_base
      - concept_country_bias_scale
      - concept_market_bias_base
      - concept_market_bias_scale
      - concept_exact_genre_hint_boost
      - concept_genre_hint_miss_penalty
      - concept_franchise_genre_match_boost
      - concept_tier_match_boost
      - concept_franchise_eligible_scale
      - concept_strategy_bonus_scale
      - concept_release_pressure_base
      - concept_release_pressure_scale
      - concept_novelty_base
      - concept_novelty_scale
      - concept_franchise_strategy_match_boost
      - concept_franchise_season_match_boost
      - concept_sequel_pressure_scale
      - concept_pack_usage_capacity
      - concept_bucket_country_capacity
      - concept_bucket_genre_capacity
      - concept_country_usage_capacity
      - concept_minor_country_bonus
      - concept_minor_market_bonus
      - writer_director_probability_by_tier (dict keyed by production tier)
      - genre_tier_distribution (dict keyed by genre, each value a 5-element list matching production tiers)
      - concept_style_vector_by_genre (dict keyed by genre, each value an 8-element list)
      - concept_style_tier_shift_by_tier (dict keyed by production tier)
      - concept_risk_target_by_genre (dict keyed by genre)
      - concept_ambition_target_by_tier (dict keyed by production tier)
      - concept_prestige_target_by_tier (dict keyed by production tier)
      - release_month_base_weights (12-element list summing approximately to 1.0)
      - release_season_month_bumps (dict keyed by season bias, values are 12-element lists or month->bump maps)
      - genre_release_month_bumps (dict keyed by genre, values are 12-element lists or month->bump maps)
      - keyword_selection.slot_mix_by_tier (dict keyed by production tier, each value has exact_anchor, related_support, story_specific, franchise, generic)
      - keyword_selection.exact_topic_min_count_by_tier (dict keyed by production tier)
      - keyword_selection.primary_plus_related_min_count_by_tier (dict keyed by production tier)
      - keyword_selection.related_genres_by_genre (dict keyed by benchmark genre)
      - director_selection (dict) with keys such as:
        - genre_match_boost
        - geo_boost_scale
        - geo_boost_floor
        - film_count_decay_over_30
        - film_count_decay_over_50
        - film_count_decay_over_80
        - company_multiplier_rescale
        - event_franchise_pop_quantile
        - event_franchise_pop_boost
        - prestige_drama_alignment_threshold
        - prestige_drama_alignment_boost
      - co_director_probability_by_tier (dict keyed by production tier)
      - company_selection (dict) with keys such as:
        - strategy_match_boost
        - market_bias_base
        - market_bias_scale
        - partner_affinity_scale
        - partner_rivalry_penalty_floor
        - family_boost
      - cast_selection (dict) with keys such as:
        - blockbuster_bonus_major
        - blockbuster_bonus_other
        - franchise_bonus_major
        - franchise_bonus_other
        - epic_tail_prob_base
        - epic_tail_prob_franchise_bonus
        - epic_tail_prob_blockbuster_bonus
        - epic_tail_prob_cap
        - epic_tail_lognorm_mean
        - epic_tail_lognorm_sigma
        - epic_tail_min
        - a_tail_prob
        - a_tail_min
        - a_tail_max
        - unused_actor_boost
        - award_recent_boost
        - franchise_pool_base_boost
        - star_vehicle_slot0_boost
        - prestige_pairing_boost
        - volatile_ensemble_boost
        - balanced_ensemble_boost
        - agency_match_boost
        - gender_novelty_boost
        - nationality_novelty_boost
        - tag_similarity_penalty
      - geo_boost_by_tier (dict keyed by production tier)
      - dynamic_cast_base_by_tier (dict keyed by production tier, values are [min, max] or {{min, max}})
      - keyword_selection (dict) with keys such as:
        - franchise_min_count
        - exact_genre_boost
        - family_genre_boost
        - off_genre_penalty
        - lexical_match_scale
        - lexical_match_cap
        - specificity_tier1_penalty
        - generic_motif_penalty
        - specific_story_boost
        - franchise_scope_boost_base
        - franchise_scope_affinity_scale
        - franchise_family_boost
        - franchise_recurrence_base
        - franchise_recurrence_scale
        - nonfranchise_scope_penalty
        - nonfranchise_family_penalty
        - nonfranchise_affinity_penalty
        - nonfranchise_affinity_threshold
        - high_specificity_novelty_base
        - high_specificity_novelty_scale
        - movie_scope_novelty_base
        - movie_scope_novelty_scale
        - usage_penalty_scale
        - company_exact_boost
        - company_family_boost
        - franchise_core_boost
      - keyword_count_by_tier (dict keyed by production tier, values are [min, max] or {{min, max}})
      - keyword_year_slate_family_boosts (dict of family-name -> keyword-family boosts)
  - edge_priors:
    - cross_genre_candidate_k
    - cross_genre_candidate_multiplier
    - cross_genre_threshold_bump
    - person_person_style_weight
    - person_person_genre_weight
    - person_person_risk_weight
    - person_person_stage_weight
    - person_person_noise_weight
    - person_person_policy_weight
    - person_person_logistic_scale
    - person_person_logistic_bias
    - person_person_base_threshold
    - person_person_degree_decay
    - person_person_degree_power
    - person_company_risk_weight
    - person_company_budget_weight
    - person_company_genre_weight
    - person_company_noise_weight
    - person_company_blacklist_threshold
    - person_company_brand_fit_threshold
    - person_person_degree_caps (dict keyed by career stage, each value has mean/std/min/max)
    - person_person_classification (dict) with keys such as:
      - controversy_high_threshold
      - controversy_gap_avoid_threshold
      - controversy_avoid_weight_floor
      - controversy_friendship_weight_base
      - controversy_friendship_score_weight
      - controversy_friendship_jitter
      - mentorship_style_threshold
      - mentorship_stage_gap_threshold
      - mentorship_weight_base
      - mentorship_style_weight
      - mentorship_jitter
      - rivalry_style_max
      - rivalry_genre_min
      - rivalry_stage_gap_max
      - rivalry_weight_base
      - rivalry_genre_weight
      - rivalry_style_distance_weight
      - rivalry_jitter
      - friendship_style_threshold
      - friendship_probability_threshold
      - friendship_style_soft_threshold
      - friendship_weight_base
      - friendship_score_weight
      - friendship_jitter
      - weight_min
      - weight_max
    - person_company_generation (dict) with keys such as:
      - genre_supplement_size
      - controversy_blacklist_person_threshold
      - controversy_blacklist_company_threshold
      - blacklist_weight_base
      - blacklist_weight_controversy_scale
      - event_franchise_micro_budget_penalty_boost
      - market_fit_boost
    - company_company_generation (dict) with keys such as:
      - strategy_match_boost
      - market_match_boost
      - rival_overlap_threshold
      - rival_tier_threshold
      - rival_weight_scale
      - rival_weight_policy_cap
      - coproduction_overlap_threshold
      - coproduction_tier_max
      - coproduction_weight_scale
      - coproduction_policy_cap
    - serendipitous_edges (dict) with:
      - stage_probabilities
      - stage_max_new_edges
      - candidate_multiplier
      - weight_min
      - weight_max
    - triadic_closure (dict) with:
      - stage_probabilities
      - extra_cap
      - weight_min
      - weight_max
    - calibration (dict) with:
      - relationship_targets
      - candidate_sample_k
      - director_actor_supplement
      - upsert_weight_min
      - upsert_weight_max
      - friendship_score_weights
      - rivalry_score_weights
      - preferential_attachment_log_weight
      - best_friend_weight_base
      - best_friend_weight_score_scale
      - best_friend_weight_noise
      - rival_weight_base
      - rival_weight_score_scale
      - rival_weight_noise
      - director_stage_target_base
      - director_stage_target_span
      - director_preferred_score_weights
      - director_preferred_weight_base
      - director_preferred_weight_score_scale
      - director_preferred_weight_noise
      - director_avoid_score_weights
      - director_avoid_weight_base
      - director_avoid_weight_score_scale
      - director_avoid_weight_noise
      - bf_same_community_weight_floor
      - bf_same_community_weight_boost
  - financial_priors:
    - regime_amplitude
    - slate_pressure
    - momentum_decay
    - recent_horizon
    - genre_memory_weight
    - country_budget_scale (dict keyed by country)
    - certification_distribution_by_genre (dict keyed by every benchmark genre; each row uses G/PG/PG-13/R/NR and positive weights)
    - genre_rating_offset (dict keyed by genre)
    - tier_rating_base (dict keyed by tier)
    - tier_rating_std (dict keyed by tier)
    - tier_log_center (dict keyed by tier)
    - tier_min_votes (dict keyed by tier)
    - market_regime (dict) with keys such as:
      - theatrical_cycle_period
      - prestige_cycle_period
      - negative_shock_probability
      - positive_shock_probability
      - negative_shock_base
      - negative_shock_span
      - positive_shock_base
      - positive_shock_span
      - drift_span
      - theatrical_cycle_weight
      - theatrical_shock_weight
      - financing_cycle_weight
      - financing_shock_weight
      - theatrical_min / theatrical_max
      - financing_min / financing_max
      - prestige_bias_base
      - prestige_cycle_weight
      - prestige_shock_weight
      - volatility_base
      - volatility_cycle_weight
      - volatility_shock_weight
      - crowding_base
      - crowding_cycle_weight
      - crowding_noise_weight
      - label_thresholds
    - year_quality (dict) with keys such as:
      - components (list of {{period, amplitude, phase_key}})
      - phase_offsets (list of {{start_frac, end_frac, offset, key}})
      - phase_scale_base
      - phase_scale_span
      - noise_amplitude
      - clip_abs
    - quality_latent_weights (dict)
    - market_latent_weights (dict)
    - performance_model (dict)
    - vote_model (dict)
    - runtime_model (dict)
    - award_campaign_weights (dict)
    - writer_director_rating_bonus
  - secondary_table_priors:
    - demographics (dict) with keys such as:
      - career_stage_age_ranges
      - legacy_death_probability
      - height_by_gender
    - release_dates (dict) with keys such as:
      - genre_month_biases
      - fallback_months
      - major_markets
      - tier_market_count_ranges
      - initial_market_delay_days
      - followup_market_delay_days
      - stream_delay_ranges
    - territory_box_office (dict) with keys such as:
      - profiles
      - dirichlet_concentration
      - territory_counts_by_tier
      - opening_weekend_share
      - first_30_fraction
      - domestic_fraction
    - reviews (dict) with keys such as:
      - volume_by_tier
      - major_sources
      - indie_sources
      - source_mix_by_tier
      - critic_score_std
      - audience_score_std
      - critic_sentiment_noise_std
      - audience_sentiment_noise_std
      - critic_review_delay_days
      - audience_review_delay_days
    - awards (dict) with keys such as:
      - prestige_by_tier
      - history_bonus_scale
      - history_bonus_cap
      - lambda_scale
      - lambda_cap
      - max_nominations
      - ceremonies
      - won_probability
    - tv_generation (dict) with keys such as:
      - title_prefixes
      - title_nouns
      - title_suffixes
      - network_names
      - episode_ranges_by_genre
      - recent_span_share
      - recent_span_min_years
      - ongoing_cutoff_share
      - ongoing_cutoff_min_years
      - genre_top_bias_probability
      - genre_flatten_power
      - network_company_probability
      - season_count_values
      - season_count_probabilities
      - status_probabilities_recent
      - status_probabilities_older
      - overall_rating
      - content_rating_by_genre
      - season_drift_values
      - season_drift_probabilities
      - season_rating_drift_range
      - season_rating_noise_std
      - season_avg_rating_noise_std
      - pilot_rating_bonus
      - finale_rating_bonus
      - episode_title_probability
      - episode_title_the_probability
      - runtime_by_group
      - viewership
    - production_timeline (dict) with keys such as:
      - phase_months_by_tier
      - announcement_fuzz_months
    - streaming_windows (dict) with keys such as:
      - platforms
      - window_count_values
      - window_count_probabilities
      - small_tier_max_windows
      - start_delay_months
      - duration_months
      - open_end_probability
      - exclusive_probability
    - person_contracts (dict) with keys such as:
      - contract_count_values
      - contract_count_probabilities
      - duration_values
      - duration_probabilities
      - gap_values
      - gap_probabilities
      - salary_bands_by_stage
      - contract_types_by_stage
  - history_event_priors:
    - event_specs (list of dicts), where each dict has:
      - event_type
      - description
      - prob
      - default_duration
      - suggested_actions
      - either year_range_fraction ([start_frac, end_frac] in [0,1]) or year_range
    - Prefer relative year windows via year_range_fraction over absolute historical years.
"""


_PROMPTS: dict[str, Callable[[argparse.Namespace], str]] = {
    "identity_bank": _prompt_identity_bank,
    "character_identity_bank": _prompt_character_identity_bank,
    "company_lexicon": _prompt_company_lexicon,
    "keyword_seed_bank": _prompt_keyword_seed_bank,
    "title_grammar_bank": _prompt_title_grammar_bank,
    "temporal_regime_plan": _prompt_temporal_regime_plan,
    "modeling_priors": _prompt_modeling_priors,
}


_MODELING_PRIOR_SECTION_PROMPTS: dict[str, str] = {
    "person_generation": """- person_generation:
  - gender_ratio
  - age_distribution or (age_distribution_mean and age_distribution_std)
  - career_stage_weights
  - base_popularity_mean
  - base_popularity_std""",
    "company_generation": """- company_generation:
  - type_distribution
  - longevity_mean
  - longevity_std
  - company_name_style_mix or country_mix
  - tier_weights (dict keyed by Global, Major, Mid-Budget, Indie, Micro)""",
    "keyword_generation": """- keyword_generation:
  - vocab_size
  - zipf_alpha
  - max_keywords_per_entity
  - genre_target_weights (dict keyed by benchmark genre, non-uniform, approximately summing to 1.0)
  - generic_budget_ratio
  - min_specific_story_share
  - selection_bucket_targets (dict with exact_anchor, related_support, story_specific, generic; approximately summing to 1.0)
  - optional specificity or family weights""",
    "character_generation": """- character_generation:
  - name_diversity_index
  - gender_match_probability
  - age_match_tolerance
  - nonliteral_name_share
  - alias_frequency
  - slot_archetype_candidates
  - general_archetypes
  - unique_archetypes
  - genre_archetype_candidates
  - archetype_target_vectors
  - career_stage_archetype_bias
  - genre_archetype_bias
  - collaboration_style_archetype_bias
  - Use the canonical runtime archetype family rather than inventing free-form roles:
    Lead Hero, Lead Villain, Love Interest, Mentor, Sidekick, Comic Relief, Supporting, Authority Figure, Henchman, Victim, Mysterious Stranger, Extra
  - archetype_target_vectors must be 4-number vectors keyed by those runtime archetypes
  - career stages should use: rising, prime, veteran, legend, retired
  - collaboration styles should use: solo, ensemble, mentorship""",
    "title_generation": """- title_generation:
  - genre_base_weights (dict keyed by genre, non-uniform, approximately summing to 1.0)
  - prestige_genres
  - franchise_genres
  - experimental_genres
  - low_cost_genres
  - allowed_tagline_placeholders (list of canonical placeholder names)
  - tagline_render_constraints (dict with min_words, max_words, max_placeholder_count, forbid_square_brackets, allow_unresolved_placeholders)
  - optional title_length_bias or tagline_tone_weights""",
    "company_finance_tiers": """- company_finance_tiers:
  - either tier objects keyed by production tier with budget ranges
  - or a budget_bands/tier budget policy dict covering at least Epic, A, Mid, Indie, Micro""",
    "selection_weights": """- selection_weights:
  - required top-level numeric keys:
    - cast_focus_exploration
    - cast_base_focus_exploration
    - cast_slot_exploration_empty
    - cast_slot_exploration_filled
    - cast_style_multiplier
    - cast_community_match_multiplier
    - director_exploration_share
    - company_primary_exploration_share
    - company_secondary_exploration_share
    - crew_exploration_share
    - concept_genre_bias_base
    - concept_genre_bias_scale
    - concept_country_bias_base
    - concept_country_bias_scale
    - concept_market_bias_base
    - concept_market_bias_scale
    - concept_exact_genre_hint_boost
    - concept_genre_hint_miss_penalty
    - concept_franchise_genre_match_boost
    - concept_tier_match_boost
    - concept_franchise_eligible_scale
    - concept_strategy_bonus_scale
    - concept_release_pressure_base
    - concept_release_pressure_scale
    - concept_novelty_base
    - concept_novelty_scale
    - concept_franchise_strategy_match_boost
    - concept_franchise_season_match_boost
    - concept_sequel_pressure_scale
    - concept_pack_usage_capacity
    - concept_bucket_country_capacity
    - concept_bucket_genre_capacity
    - concept_country_usage_capacity
    - concept_minor_country_bonus
    - concept_minor_market_bonus
    - director_risk_weight
    - director_ambition_weight
    - director_prestige_weight
    - director_alignment_base
    - director_alignment_scale
    - director_csv_base
    - director_csv_scale
    - company_tier_match_boost
    - company_tier_mismatch_penalty
    - company_genre_match_boost
    - company_genre_mismatch_penalty
    - company_risk_weight
    - company_prestige_weight
    - company_focus_weight
    - company_genre_fit_weight
    - company_alignment_base
    - company_alignment_scale
  - director_selection with exact numeric keys:
    - genre_match_boost
    - geo_boost_scale
    - geo_boost_floor
    - film_count_decay_over_30
    - film_count_decay_over_50
    - film_count_decay_over_80
    - company_multiplier_rescale
    - event_franchise_pop_quantile
    - event_franchise_pop_boost
    - prestige_drama_alignment_threshold
    - prestige_drama_alignment_boost
  - company_selection with exact numeric keys:
    - strategy_match_boost
    - market_bias_base
    - market_bias_scale
    - partner_affinity_scale
    - partner_rivalry_penalty_floor
    - family_boost
  - cast_selection with exact numeric keys:
    - blockbuster_bonus_major
    - blockbuster_bonus_other
    - franchise_bonus_major
    - franchise_bonus_other
    - epic_tail_prob_base
    - epic_tail_prob_franchise_bonus
    - epic_tail_prob_blockbuster_bonus
    - epic_tail_prob_cap
    - epic_tail_lognorm_mean
    - epic_tail_lognorm_sigma
    - epic_tail_min
    - a_tail_prob
    - a_tail_min
    - a_tail_max
    - unused_actor_boost
    - award_recent_boost
    - franchise_pool_base_boost
    - star_vehicle_slot0_boost
    - prestige_pairing_boost
    - volatile_ensemble_boost
    - balanced_ensemble_boost
    - agency_match_boost
    - gender_novelty_boost
    - nationality_novelty_boost
    - tag_similarity_penalty
  - keyword_selection with exact numeric keys:
    - franchise_min_count
    - exact_genre_boost
    - family_genre_boost
    - off_genre_penalty
    - lexical_match_scale
    - lexical_match_cap
    - specificity_tier1_penalty
    - generic_motif_penalty
    - specific_story_boost
    - franchise_scope_boost_base
    - franchise_scope_affinity_scale
    - franchise_family_boost
    - franchise_recurrence_base
    - franchise_recurrence_scale
    - nonfranchise_scope_penalty
    - nonfranchise_family_penalty
    - nonfranchise_affinity_penalty
    - nonfranchise_affinity_threshold
    - high_specificity_novelty_base
    - high_specificity_novelty_scale
    - movie_scope_novelty_base
    - movie_scope_novelty_scale
    - usage_penalty_scale
    - company_exact_boost
    - company_family_boost
    - franchise_core_boost
    - family_genre_max_share
   - genre_tier_distribution (dict keyed by genre, each value is either a tier dict keyed by Epic/A/Mid/Indie/Micro or a length-5 numeric list)
  - writer_director_probability_by_tier (dict keyed by Epic/A/Mid/Indie/Micro)
  - geo_boost_by_tier (dict keyed by Epic/A/Mid/Indie/Micro)
  - dynamic_cast_base_by_tier (dict keyed by Epic/A/Mid/Indie/Micro, each value is [min,max] or {min,max})
  - keyword_count_by_tier (dict keyed by Epic/A/Mid/Indie/Micro, each value is [min,max] or {min,max})
  - concept_style_vector_by_genre (dict keyed by genre, each value a length-8 numeric vector)
  - concept_style_tier_shift_by_tier (dict keyed by production tier, each value either numeric or a length-8 numeric vector)
  - concept_risk_target_by_genre
  - concept_ambition_target_by_tier
  - concept_prestige_target_by_tier
  - release_month_base_weights (length-12 numeric list)
  - release_season_month_bumps (non-empty dict)
    - genre_release_month_bumps (non-empty dict)
    - keyword_year_slate_family_boosts (non-empty dict)
    - keyword_selection.primary_genre_min_count_by_tier (dict keyed by production tier)
    - keyword_selection.exact_topic_min_count_by_tier (dict keyed by production tier)
    - keyword_selection.primary_plus_related_min_count_by_tier (dict keyed by production tier)
    - keyword_selection.generic_keyword_cap_by_tier (dict keyed by production tier)
    - keyword_selection.off_genre_cap_by_tier (dict keyed by production tier)
    - keyword_selection.slot_mix_by_tier (dict keyed by production tier, each value has exact_anchor, related_support, story_specific, franchise, generic)
    - keyword_selection.related_genres_by_genre (dict keyed by benchmark genre)
    - do not substitute generic aliases in cast_selection or keyword_selection; use exactly the field names above""",
    "edge_priors": """- edge_priors:
  - cross_genre_candidate_k
  - cross_genre_candidate_multiplier
  - cross_genre_threshold_bump
  - person_person_style_weight
  - person_person_genre_weight
  - person_person_risk_weight
  - person_person_stage_weight
  - person_person_noise_weight
  - person_person_policy_weight
  - person_person_logistic_scale
  - person_person_logistic_bias
  - person_person_base_threshold
  - person_person_degree_decay
  - person_person_degree_power
  - person_company_risk_weight
  - person_company_budget_weight
  - person_company_genre_weight
  - person_company_noise_weight
  - person_company_blacklist_threshold
  - person_company_brand_fit_threshold
  - person_person_degree_caps (dict keyed by rising/prime/veteran/legend/retired, each value has mean/std/min/max)
  - person_person_classification with the exact runtime keys listed in the full modeling-priors prompt
  - person_company_generation with the exact runtime keys listed in the full modeling-priors prompt
  - company_company_generation with the exact runtime keys listed in the full modeling-priors prompt
  - serendipitous_edges with stage_probabilities, stage_max_new_edges, candidate_multiplier, weight_min, weight_max
  - triadic_closure with stage_probabilities, extra_cap, weight_min, weight_max
  - calibration with exact keys:
    - relationship_targets (dict with best_friend_rate, rival_rate, bf_same_community_rate, director_preferred_actors, director_avoided_actors)
    - candidate_sample_k
    - director_actor_supplement
    - upsert_weight_min
    - upsert_weight_max
    - friendship_score_weights (dict with style, genre, stage)
    - rivalry_score_weights (dict with genre, style_distance, stage)
    - preferential_attachment_log_weight
    - best_friend_weight_base
    - best_friend_weight_score_scale
    - best_friend_weight_noise
    - rival_weight_base
    - rival_weight_score_scale
    - rival_weight_noise
    - director_stage_target_base
    - director_stage_target_span
    - director_preferred_score_weights (dict with style, genre, stage)
    - director_preferred_weight_base
    - director_preferred_weight_score_scale
    - director_preferred_weight_noise
    - director_avoid_score_weights (dict with style_distance, genre_distance, controversy)
    - director_avoid_weight_base
    - director_avoid_weight_score_scale
    - director_avoid_weight_noise
    - bf_same_community_weight_floor
    - bf_same_community_weight_boost
  - do not replace calibration maps with scalar numbers; map-valued keys must remain objects""",
    "scalable_edge_priors": """- scalable_edge_priors:
  - stage_priority (dict keyed by career stage)
  - collaboration_style_codes (dict keyed by collaboration style)
  - valid_year_span
  - person_degree_caps (dict keyed by career stage, each value has mean/std/min/max)
  - profile_overrides with keys such as:
    - friendship_ratio_base
    - friendship_ratio_creative_share_weight
    - friendship_ratio_community_density_weight
    - rivalry_ratio_base
    - rivalry_ratio_competition_density_weight
    - mentorship_ratio_base
    - avoid_ratio_base
    - former_ratio_base
    - collaboration_ratio_base
    - clique_ratio_base
    - closure_scale_base
    - bridge_anchor_ratio_base
    - friendship_threshold_base
    - rivalry_threshold_base
    - mentorship_threshold_base
    - avoid_threshold_base
    - former_threshold_base
    - collaboration_threshold_base
    - clique_threshold_base
    - company_genre_sample_ratio_base
    - company_clique_sample_ratio_base
    - company_random_sample_ratio_base
    - brand_fit_threshold_base
    - blacklist_threshold_base
    - cc_genre_sample_ratio_base
    - cc_clique_sample_ratio_base
    - cc_market_sample_ratio_base
    - cc_random_sample_ratio_base
    - cc_rival_pick_ratio_base
    - cc_copro_pick_ratio_base
    - cc_subsidiary_pick_ratio_base
  - brand_fit_ratio_by_stage
  - employment_ratio_by_stage
  - sampled_union_defaults with ratios/caps for scalable candidate sampling
  - year_validity_offsets (dict with person_edge_end_offset and company_edge_end_offset)""",
    "financial_priors": """- financial_priors:
  - country_budget_scale
  - genre_rating_offset
  - tier_rating_base
  - tier_rating_std
  - tier_log_center
  - tier_min_votes
  - budget_ranges_by_tier
  - certification_distribution_by_genre
    - provide a usable row for every benchmark genre in GENRES
    - each genre row must use certification keys from {G, PG, PG-13, R, NR}
    - each genre row must have positive numeric weights summing approximately to 1.0
  - company_profile_tiers
  - company_profile_coefficients
  - market_regime
  - year_quality
  - quality_latent_weights
  - market_latent_weights
  - performance_model
  - vote_model
  - runtime_model
  - award_campaign_weights
    - include exact numeric keys:
      - company_prestige
      - director_ambition
      - director_reputation
      - cast_reputation
      - company_focus
      - q4_bonus
      - prestige_genre_bonus
      - regime_prestige_bias
      - director_momentum
      - company_momentum
      - graph_synergy
      - quality_signal
      - slate_pressure_penalty
      - controversy_penalty
  - writer_director_rating_bonus
  - company_profile_tiers must be a dict keyed exactly by:
    - Global
    - Major
    - Mid-Budget
    - Indie
    - Micro
  - each company_profile_tiers row must include exact numeric keys:
    - capital
    - margin
    - debt
    - slate
    - buffer
    - growth
    - eff
  - company_profile_coefficients should include at minimum budget_focus_weights (length 5 numeric list) plus numeric coefficient keys for company finance synthesis
  - do not replace company_profile_tiers with studio-label distributions like Conglomerate / Boutique / Independent""",
    "secondary_table_priors": """- secondary_table_priors:
  - demographics (dict) with keys:
    - career_stage_age_ranges
    - legacy_death_probability
    - height_by_gender
    - career_stage_age_ranges should be keyed by rising, prime, veteran, legend, retired
    - height_by_gender should be keyed by F, NB, M
  - release_dates (dict) with keys:
    - genre_month_biases
    - fallback_months
    - major_markets
    - tier_market_count_ranges
    - initial_market_delay_days
    - followup_market_delay_days
    - stream_delay_ranges
  - territory_box_office (dict) with keys:
    - profiles
    - dirichlet_concentration
    - territory_counts_by_tier
    - domestic_fraction
    - opening_weekend_share
    - first_30_fraction
  - reviews (dict) with keys:
    - volume_by_tier
    - freshness_params
    - audience_score_noise
  - awards (dict) with keys:
    - prestige_by_tier
    - history_bonus_scale
    - history_bonus_cap
    - ceremonies
    - won_probability
  - tv_generation (dict) with keys:
    - title_prefixes
    - title_nouns
    - title_suffixes
    - season_count_values
    - season_count_probabilities
    - status_probabilities_recent
    - status_probabilities_older
    - overall_rating
    - content_rating_by_genre
    - season_drift_values
    - season_drift_probabilities
    - episode_ranges_by_genre
    - season_rating_drift_range
    - runtime_by_group
    - viewership
  - production_timeline (dict) with keys:
    - phase_months_by_tier
    - announcement_fuzz_months
    - phase_months_by_tier should be keyed by the production tiers and each tier should include announced, pre_production, filming, post_production
  - streaming_windows (dict) with keys:
    - platforms
    - window_count_values
    - window_count_probabilities
    - start_delay_months
    - duration_months
  - person_contracts (dict) with keys:
    - contract_count_values
    - contract_count_probabilities
    - duration_values
    - duration_probabilities
    - gap_values
    - gap_probabilities
    - salary_bands_by_stage
    - contract_types_by_stage
    - salary_bands_by_stage and contract_types_by_stage should be keyed by rising, prime, veteran, legend, retired""",
    "history_event_priors": """- history_event_priors:
  - event_specs (non-empty list)
  - each event spec should include event_type, description, prob, default_duration, suggested_actions
  - prefer relative time windows via year_range_fraction""",
    "rerank_priors": """- rerank_priors:
  - rerank_budget
  - keyword_rerank_budget
  - critic_weight
  - repair_threshold
  - rerank_temperature""",
}


_MODELING_PRIOR_GROUPS: tuple[dict[str, Any], ...] = (
    {
        "name": "foundation",
        "sections": ("person_generation", "company_generation", "keyword_generation", "character_generation"),
        "temperature": 0.10,
        "max_tokens": 5000,
    },
    {
        "name": "titles",
        "sections": ("title_generation", "company_finance_tiers", "rerank_priors"),
        "temperature": 0.10,
        "max_tokens": 6000,
    },
    {
        "name": "selection",
        "sections": ("selection_weights",),
        "temperature": 0.0,
        "max_tokens": 18000,
    },
    {
        "name": "edges",
        "sections": ("edge_priors", "scalable_edge_priors"),
        "temperature": 0.10,
        "max_tokens": 10000,
    },
    {
        "name": "financials",
        "sections": ("financial_priors",),
        "temperature": 0.10,
        "max_tokens": 7000,
    },
    {
        "name": "secondary",
        "sections": ("secondary_table_priors",),
        "temperature": 0.05,
        "max_tokens": 14000,
    },
    {
        "name": "history",
        "sections": ("history_event_priors",),
        "temperature": 0.10,
        "max_tokens": 4000,
    },
)


def _prompt_modeling_priors_group(args: argparse.Namespace, group_name: str, sections: tuple[str, ...]) -> str:
    section_block = "\n".join(_MODELING_PRIOR_SECTION_PROMPTS[name] for name in sections)
    runtime_schema = _runtime_schema_completion_skeleton(sections)
    runtime_schema_block = ""
    edge_calibration_hint = ""
    if runtime_schema:
        runtime_schema_block = (
            "\nExact runtime schema skeleton to match:\n"
            + json.dumps(runtime_schema, indent=2, ensure_ascii=False)
            + "\nUse these exact nested keys and replace placeholder values with real priors.\n"
        )
    if "edge_priors" in sections:
        edge_calibration_hint = (
            "\nEdge calibration constraints:\n"
            "- relationship_targets.director_preferred_actors and relationship_targets.director_avoided_actors are sparse activation probabilities in [0,1], not literal counts.\n"
            "- Keep director_preferred_actors roughly in the 0.12-0.24 range and director_avoided_actors roughly in the 0.01-0.08 range.\n"
            "- Keep director_stage_target_base near 0.6-1.0 and director_stage_target_span near 0.1-0.5.\n"
            "- Keep bf_same_community_rate as a minimum target in the 0.55-0.75 range, not a forced exact equality.\n"
        )
    return f"""Create one grouped subset of a modeling priors object for a synthetic film database generation pipeline.

Context:
{_context_block(args)}

Rules:
- Output JSON object only.
- Return exactly these top-level sections: {", ".join(sections)}.
- Every requested section must be present and non-empty.
- Use numeric, reusable priors rather than prose.
- Avoid absolute historical assumptions; prefer relative or range-agnostic priors.
- Probabilities must stay in [0,1], weights must be positive, lists must be non-trivial.
- Do not omit sections. Do not wrap them in another envelope.

Required content:
{section_block}
{runtime_schema_block}
{edge_calibration_hint}

Group label:
- {group_name}
"""


_CHARACTER_IDENTITY_SCAFFOLD: dict[str, Any] = {
    "archetype_weights": {
        "Lead Hero": 0.12,
        "Lead Villain": 0.10,
        "Love Interest": 0.06,
        "Sidekick": 0.06,
        "Mentor": 0.07,
        "Supporting": 0.18,
        "Extra": 0.13,
        "Authority Figure": 0.07,
        "Henchman": 0.06,
        "Comic Relief": 0.05,
        "Victim": 0.05,
        "Mysterious Stranger": 0.05,
    },
    "title_prefixes_m": ["Dr.", "Prof.", "Capt.", "Det.", "Agent", "Judge", "Chief", "King", "Lord"],
    "title_prefixes_f": ["Dr.", "Prof.", "Capt.", "Det.", "Agent", "Judge", "Chief", "Queen", "Lady"],
    "title_prefixes_nb": ["Dr.", "Prof.", "Capt.", "Det.", "Agent", "Judge", "Chief"],
    "quote_nicknames": ["The Ghost", "The Fox", "The Blade", "The Raven", "The Oracle", "The Sentinel"],
    "solo_monikers": ["The Collector", "The Architect", "The Warden", "The Harbinger", "The Regent", "The Oracle"],
    "codename_adjectives": ["Silent", "Iron", "Crimson", "Midnight", "Broken", "Golden", "Neon"],
    "codename_nouns": ["Viper", "Falcon", "Cipher", "Ghost", "Torch", "Wraith", "Atlas"],
    "mythic_epithets": ["Stormborn", "Wayfarer", "Ashen", "Radiant", "Veilwalker"],
    "role_epithets": ["Fixer", "Broker", "Handler", "Watcher", "Operator", "Judge"],
    "alias_templates": [
        "{title} {surname}",
        "{first} '{nickname}' {surname}",
        "The {moniker}",
        "{codename_adj} {codename_noun}",
        "{title} {first} {surname}",
    ],
    "nonliteral_share_by_archetype": {
        "Lead Villain": 0.28,
        "Mysterious Stranger": 0.32,
        "Mentor": 0.18,
        "Comic Relief": 0.12,
    },
    "human_name_mix_by_archetype": {
        "Lead Hero": 0.90,
        "Lead Villain": 0.65,
        "Love Interest": 0.96,
        "Supporting": 0.88,
        "Mysterious Stranger": 0.55,
    },
}

_SUPPORTED_CHARACTER_ALIAS_KEYS: set[str] = {
    "title",
    "title_prefix",
    "first",
    "human_first",
    "first_name",
    "given_name",
    "human_first_name",
    "surname",
    "human_last",
    "human_surname",
    "last_name",
    "family_name",
    "human_last_name",
    "full_name",
    "human_name",
    "nickname",
    "quote_nickname",
    "moniker",
    "solo_moniker",
    "codename_adj",
    "codename_adjective",
    "codename_noun",
    "mythic_epithet",
    "role_epithet",
}


_CHARACTER_ALIAS_PLACEHOLDER_ALIASES: dict[str, str] = {
    "title_m": "title",
    "title_f": "title",
    "title_nb": "title",
    "title_male": "title",
    "title_female": "title",
    "title_neutral": "title",
    "title_nonbinary": "title",
    "title_prefix_m": "title_prefix",
    "title_prefix_f": "title_prefix",
    "title_prefix_nb": "title_prefix",
    "title_prefix_male": "title_prefix",
    "title_prefix_female": "title_prefix",
    "title_prefix_neutral": "title_prefix",
    "title_prefix_nonbinary": "title_prefix",
}


def _canonicalize_character_alias_template(template: str) -> str:
    text = str(template or "")
    if not text:
        return ""

    def _replace(match: re.Match[str]) -> str:
        token = str(match.group(1) or "").strip()
        canonical = _CHARACTER_ALIAS_PLACEHOLDER_ALIASES.get(token, token)
        return "{" + canonical + "}"

    return re.sub(r"\{([^{}]+)\}", _replace, text)


_COMPANY_LEXICON_SCAFFOLD: dict[str, Any] = {
    "prefixes": [
        "Apex", "Aurora", "Beacon", "Blue", "Cobalt", "First", "Golden", "Grand",
        "Harbor", "Iron", "Mirage", "Moon", "Neo", "North", "Nova", "Omni",
        "Prime", "Silver", "Solar", "Star", "Summit", "True", "Vector", "Vista",
    ],
    "suffixes": [
        "Pictures", "Studios", "Films", "Entertainment", "Media", "Productions",
        "Cinema", "Group", "Works", "Labs", "Collective", "Ventures",
    ],
    "abstract_nouns": [
        "Vision", "Horizon", "Echo", "Signal", "Legacy", "Canvas", "Story",
        "Pulse", "Summit", "Frontier", "Myth", "Orbit",
    ],
    "material_words": ["Silver", "Golden", "Ivory", "Amber", "Cobalt", "Steel"],
    "geographic_words": ["Harbor", "Coast", "Frontier", "Skyline", "Summit", "Valley"],
    "motion_words": ["Pulse", "Transit", "Orbit", "Drift", "Ascent", "Voyage"],
    "mythic_words": ["Phoenix", "Atlas", "Oracle", "Mirage", "Titan", "Avalon"],
    "templates": [
        "{prefix} {suffix}",
        "{prefix} {abstract} {suffix}",
        "{geo} {prefix} {suffix}",
        "{motion} {abstract} {suffix}",
        "{mythic} {suffix}",
    ],
    "tier_styles": {
        "Global": ["broad", "prestige", "corporate"],
        "Major": ["polished", "commercial", "confident"],
        "Mid-Budget": ["nimble", "genre-forward", "modern"],
        "Indie": ["artful", "personal", "distinctive"],
        "Micro": ["scrappy", "inventive", "local"],
    },
    "country_style_bias": {
        "USA": ["bold", "studio-scale"],
        "UK": ["prestige", "heritage"],
        "India": ["epic", "musicality"],
        "France": ["art-house", "elegant"],
        "Japan": ["precise", "stylized"],
    },
}

_COMPANY_TEMPLATE_ALIAS_MAP: dict[str, str] = {
    "local_word": "geo",
    "local_words": "geo",
    "geographic_word": "geo",
    "geographic_words": "geo",
    "abstract_word": "abstract",
    "abstract_words": "abstract",
    "motion_word": "motion",
    "motion_words": "motion",
    "mythic_word": "mythic",
    "mythic_words": "mythic",
    "material_word": "material",
    "material_words": "material",
}

_SUPPORTED_COMPANY_TEMPLATE_KEYS: set[str] = {"prefix", "suffix", "abstract", "material", "geo", "motion", "mythic"}


def _is_string_list(value: Any, *, minimum: int = 1) -> bool:
    return isinstance(value, list) and len(value) >= minimum and all(str(item).strip() for item in value)


def _validate_identity_bank(payload: dict[str, Any]) -> None:
    families = payload.get("families")
    _validate_identity_bank_family_rows(families, minimum=8)
    _validate_identity_bank_defaults_payload(payload)


def _normalize_identity_defaults(defaults: Any) -> dict[str, Any]:
    if not isinstance(defaults, dict):
        return {}
    out = dict(defaults)
    if "role_distribution" not in out:
        for alias in ("roles", "role_weights", "role_mix"):
            if isinstance(out.get(alias), dict):
                out["role_distribution"] = dict(out.get(alias))
                break
    if "stage_weights" not in out:
        for alias in ("career_stage_weights", "stage_distribution", "stage_mix"):
            if isinstance(out.get(alias), dict):
                out["stage_weights"] = dict(out.get(alias))
                break
    if "gender_weights" not in out:
        for alias in ("gender_ratio", "gender_distribution", "gender_mix"):
            if isinstance(out.get(alias), dict):
                out["gender_weights"] = dict(out.get(alias))
                break
    return out


def _normalize_identity_bank_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    out = dict(payload)
    if "families" not in out:
        for alias in ("identity_families", "family_groups", "family_bank"):
            if isinstance(out.get(alias), list):
                out["families"] = out.get(alias)
                break
    if "defaults" not in out:
        inferred_defaults = _normalize_identity_defaults(out)
        if inferred_defaults.get("role_distribution") or inferred_defaults.get("stage_weights") or inferred_defaults.get("gender_weights"):
            out["defaults"] = inferred_defaults
    elif isinstance(out.get("defaults"), dict):
        out["defaults"] = _normalize_identity_defaults(out.get("defaults"))
    return out


def _strip_llm_json_wrappers(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    while "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>", start)
        if end == -1:
            break
        text = (text[:start] + text[end + len("</think>"):]).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    if text.lower().startswith("json"):
        text = text[4:].strip()
    return text


def _balance_json_suffix(text: str) -> Any:
    cleaned = _strip_llm_json_wrappers(text)
    if not cleaned:
        return None
    start_idx = -1
    for idx, ch in enumerate(cleaned):
        if ch in "[{":
            start_idx = idx
            break
    if start_idx == -1:
        return None
    candidate = cleaned[start_idx:].strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    stack: list[str] = []
    in_string = False
    escape = False
    for ch in candidate:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch == "}":
            if stack and stack[-1] == "{":
                stack.pop()
        elif ch == "]":
            if stack and stack[-1] == "[":
                stack.pop()
    if in_string:
        return None
    closers = "".join("}" if opener == "{" else "]" for opener in reversed(stack))
    if not closers:
        return None
    try:
        return json.loads(candidate + closers)
    except json.JSONDecodeError:
        return None


def _extract_keyed_array_rows(text: str, key: str) -> list[dict[str, Any]]:
    cleaned = _strip_llm_json_wrappers(text)
    if not cleaned:
        return []
    key_pattern = f'"{key}"'
    key_idx = cleaned.find(key_pattern)
    if key_idx == -1:
        return []
    array_idx = cleaned.find("[", key_idx)
    if array_idx == -1:
        return []
    rows: list[dict[str, Any]] = []
    in_string = False
    escape = False
    depth = 0
    obj_start: int | None = None
    for idx in range(array_idx + 1, len(cleaned)):
        ch = cleaned[idx]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                obj_start = idx
            depth += 1
            continue
        if ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and obj_start is not None:
                    snippet = cleaned[obj_start:idx + 1]
                    try:
                        parsed = json.loads(snippet)
                    except json.JSONDecodeError:
                        parsed = None
                    if isinstance(parsed, dict):
                        rows.append(parsed)
                    obj_start = None
            continue
        if ch == "]" and depth == 0:
            break
    return rows


def _extract_keyed_string_list(text: str, key: str) -> list[str]:
    cleaned = _strip_llm_json_wrappers(text)
    if not cleaned:
        return []
    key_pattern = f'"{key}"'
    key_idx = cleaned.find(key_pattern)
    if key_idx == -1:
        return []
    array_idx = cleaned.find("[", key_idx)
    if array_idx == -1:
        return []
    values: list[str] = []
    in_string = False
    escape = False
    depth = 0
    token_chars: list[str] = []
    for idx in range(array_idx + 1, len(cleaned)):
        ch = cleaned[idx]
        if not in_string and depth == 0 and ch == "]":
            break
        if in_string:
            if escape:
                token_chars.append(ch)
                escape = False
                continue
            if ch == "\\":
                escape = True
                token_chars.append(ch)
                continue
            if ch == '"':
                raw_token = "".join(token_chars)
                try:
                    value = json.loads(f'"{raw_token}"')
                except Exception:
                    value = raw_token
                value = str(value or "").strip()
                if value:
                    values.append(value)
                token_chars = []
                in_string = False
                continue
            token_chars.append(ch)
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "[":
            depth += 1
        elif ch == "]" and depth > 0:
            depth -= 1
    return values


def _payload_materiality_score(value: Any) -> int:
    if isinstance(value, dict):
        score = max(1, len(value))
        for child in value.values():
            score += _payload_materiality_score(child)
        return score
    if isinstance(value, list):
        score = len(value)
        for child in value:
            score += _payload_materiality_score(child)
        return score
    if isinstance(value, str):
        return min(len(value), 256)
    return 1 if value is not None else 0


def _coerce_keyword_seed_bank_payload(parsed: Any) -> dict[str, Any] | None:
    if isinstance(parsed, str):
        repaired = _balance_json_suffix(parsed)
        if repaired is not None:
            coerced = _coerce_keyword_seed_bank_payload(repaired)
            if isinstance(coerced, dict):
                return coerced
        payload: dict[str, Any] = {}
        for key in ("universal_qualifiers", "universal_contexts", "generic_themes"):
            values = _extract_keyed_string_list(parsed, key)
            if values:
                payload[key] = values
        genres = _extract_keyed_array_rows(parsed, "genres")
        if genres:
            payload["genres"] = genres
        return payload or None
    if isinstance(parsed, dict):
        payload = dict(parsed)
        if "genres" not in payload and isinstance(payload.get("genre_data"), list):
            payload["genres"] = payload.get("genre_data")
        if "genres" not in payload and isinstance(payload.get("genre_metadata"), list):
            payload["genres"] = payload.get("genre_metadata")
        if "genres" not in payload and isinstance(payload.get("genre_seeds"), list):
            payload["genres"] = payload.get("genre_seeds")
        return payload
    if isinstance(parsed, list):
        genres = [row for row in parsed if isinstance(row, dict) and row.get("genre")]
        if genres:
            return {
                "universal_qualifiers": [],
                "universal_contexts": [],
                "generic_themes": [],
                "genres": genres,
            }
    return None


def _validate_identity_bank_family_rows(rows: Any, minimum: int = 1) -> None:
    if not isinstance(rows, list) or len(rows) < int(minimum):
        got = len(rows) if isinstance(rows, list) else 0
        raise ValueError(f"identity_bank grouped payload requires at least {int(minimum)} family rows; got {int(got)}")
    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"identity_bank family[{idx}] must be an object")
        for key in (
            "nationality",
            "region",
            "weight",
            "first_m",
            "first_f",
            "first_nb",
            "surnames",
            "connectors",
            "double_surname_probability",
            "middle_name_probability",
            "middle_initial_probability",
            "suffix_probability",
            "market_bias",
        ):
            if key not in row:
                raise ValueError(f"identity_bank family missing {key}")


def _validate_identity_bank_defaults_payload(payload: dict[str, Any]) -> None:
    defaults = payload.get("defaults")
    if not isinstance(defaults, dict):
        raise ValueError("identity_bank defaults payload requires defaults")
    for key in ("role_distribution", "stage_weights", "gender_weights"):
        if not isinstance(defaults.get(key), dict) or not defaults.get(key):
            raise ValueError(f"identity_bank defaults missing {key}")


def _coerce_identity_bank_group_payload(parsed: Any) -> dict[str, Any] | None:
    if isinstance(parsed, str):
        repaired = _balance_json_suffix(parsed)
        if repaired is not None:
            coerced = _coerce_identity_bank_group_payload(repaired)
            if isinstance(coerced, dict):
                return coerced
        rows = _extract_keyed_array_rows(parsed, "families")
        if rows:
            return {"families": rows}
        return None
    if isinstance(parsed, dict):
        payload = _normalize_identity_bank_payload(dict(parsed))
        families = payload.get("families")
        if isinstance(families, list):
            return {"families": families}
        if parsed and all(isinstance(value, (str, int, float, list, dict)) for value in parsed.values()):
            required = {"nationality", "region", "first_m", "first_f", "first_nb", "surnames"}
            if required & set(str(key) for key in parsed.keys()):
                return {"families": [dict(parsed)]}
    if isinstance(parsed, list):
        rows = [row for row in parsed if isinstance(row, dict)]
        if rows:
            return {"families": rows}
    return None


def _coerce_identity_bank_defaults_payload(parsed: Any) -> dict[str, Any] | None:
    if isinstance(parsed, dict):
        payload = _normalize_identity_bank_payload(dict(parsed))
        defaults = payload.get("defaults")
        if isinstance(defaults, dict):
            return {"defaults": defaults}
    return None


def _validate_character_identity_bank(payload: dict[str, Any]) -> None:
    required = (
        "archetype_weights",
        "title_prefixes_m",
        "title_prefixes_f",
        "title_prefixes_nb",
        "quote_nicknames",
        "solo_monikers",
        "codename_adjectives",
        "codename_nouns",
        "mythic_epithets",
        "role_epithets",
        "alias_templates",
        "nonliteral_share_by_archetype",
        "human_name_mix_by_archetype",
    )
    for key in required:
        if key not in payload:
            raise ValueError(f"character_identity_bank missing {key}")
    archetypes = {
        str(key).strip()
        for key, value in (payload.get("archetype_weights") or {}).items()
        if str(key).strip() and isinstance(value, (int, float))
    }
    if not archetypes:
        raise ValueError("character_identity_bank missing usable archetype_weights")
    for key in ("nonliteral_share_by_archetype", "human_name_mix_by_archetype"):
        row = payload.get(key)
        if not isinstance(row, dict):
            raise ValueError(f"character_identity_bank missing {key}")
        missing = sorted(archetypes - {str(name).strip() for name in row.keys() if str(name).strip()})
        if missing:
            raise ValueError(f"character_identity_bank {key} missing archetypes: {', '.join(missing[:6])}")
    for template in payload.get("alias_templates", []):
        template = _canonicalize_character_alias_template(str(template or ""))
        tokens = re.findall(r"\{([^{}]+)\}", str(template or ""))
        unsupported = [str(token) for token in tokens if str(token) not in _SUPPORTED_CHARACTER_ALIAS_KEYS]
        if unsupported:
            raise ValueError(
                "character_identity_bank alias_templates use unsupported placeholders: "
                + ", ".join(sorted(set(unsupported)))
            )


def _normalize_character_identity_bank_payload(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload or {})
    if "archetype_weights" not in out:
        # Some models return the weights object directly instead of the full wrapper.
        if out and all(isinstance(key, str) for key in out.keys()) and all(isinstance(value, (int, float)) for value in out.values()):
            out = {"archetype_weights": dict(out)}
    alias_map = {
        "male_title_prefixes": "title_prefixes_m",
        "female_title_prefixes": "title_prefixes_f",
        "neutral_title_prefixes": "title_prefixes_nb",
        "title_prefixes_neutral": "title_prefixes_nb",
        "monikers": "solo_monikers",
        "epithets": "mythic_epithets",
        "job_epithets": "role_epithets",
        "nickname_templates": "alias_templates",
        "human_name_share_by_archetype": "human_name_mix_by_archetype",
        "human_name_share": "human_name_mix_by_archetype",
        "non_literal_share_by_archetype": "nonliteral_share_by_archetype",
    }
    for alias_key, canonical_key in alias_map.items():
        if canonical_key not in out and alias_key in out:
            out[canonical_key] = out.get(alias_key)
    alias_templates = out.get("alias_templates")
    if isinstance(alias_templates, list):
        normalized_templates: list[str] = []
        seen_templates: set[str] = set()
        for raw_template in alias_templates:
            template = _canonicalize_character_alias_template(str(raw_template or "").strip())
            key = template.casefold()
            if not template or key in seen_templates:
                continue
            seen_templates.add(key)
            normalized_templates.append(template)
        out["alias_templates"] = normalized_templates
    if "archetype_weights" not in out:
        for source_key in ("nonliteral_share_by_archetype", "human_name_mix_by_archetype"):
            raw = out.get(source_key)
            if isinstance(raw, dict) and raw:
                weights: dict[str, float] = {}
                for key, value in raw.items():
                    if isinstance(value, dict):
                        try:
                            weights[str(key)] = max(0.001, float(sum(float(v) for v in value.values())))
                        except Exception:
                            continue
                    else:
                        try:
                            weights[str(key)] = max(0.001, float(value))
                        except Exception:
                            continue
                if weights:
                    out["archetype_weights"] = weights
                    break
    for key, default_value in _CHARACTER_IDENTITY_SCAFFOLD.items():
        if key not in out:
            out[key] = default_value
    return out


def _normalize_company_lexicon_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        out: dict[str, Any] = dict(payload)
    elif isinstance(payload, list):
        if payload and all(isinstance(item, str) for item in payload):
            out = {"prefixes": list(payload)}
        else:
            out = {}
            for row in payload:
                if isinstance(row, dict):
                    for key, value in row.items():
                        out.setdefault(str(key), value)
    else:
        out = {}

    alias_map = {
        "company_prefixes": "prefixes",
        "company_suffixes": "suffixes",
        "nouns": "abstract_nouns",
        "abstract_words": "abstract_nouns",
        "materials": "material_words",
        "geo_words": "geographic_words",
        "geographic_terms": "geographic_words",
        "movement_words": "motion_words",
        "motion_terms": "motion_words",
        "mythic_terms": "mythic_words",
        "name_templates": "templates",
        "format_templates": "templates",
        "tier_bias": "tier_styles",
        "country_bias": "country_style_bias",
    }
    for alias_key, canonical_key in alias_map.items():
        if canonical_key not in out and alias_key in out:
            out[canonical_key] = out.get(alias_key)

    def _normalize_string_bucket(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            parts = [part.strip() for part in value.replace("\r", "\n").replace(",", "\n").split("\n")]
            return [part for part in parts if part]
        return []

    def _canonicalize_company_template(template: Any) -> str:
        raw = str(template or "").strip()
        if not raw:
            return ""

        def _replace(match: re.Match[str]) -> str:
            key = str(match.group(1) or "").strip()
            canonical = _COMPANY_TEMPLATE_ALIAS_MAP.get(key, key)
            return "{" + canonical + "}"

        return re.sub(r"\{([^{}]+)\}", _replace, raw)

    for key in (
        "prefixes",
        "suffixes",
        "abstract_nouns",
        "material_words",
        "geographic_words",
        "motion_words",
        "mythic_words",
        "templates",
    ):
        values = _normalize_string_bucket(out.get(key))
        fallback = list(_COMPANY_LEXICON_SCAFFOLD[key])
        out[key] = values if values else fallback
    out["templates"] = [tpl for tpl in (_canonicalize_company_template(item) for item in out.get("templates", [])) if tpl]

    tier_styles = out.get("tier_styles")
    if not isinstance(tier_styles, dict) or not tier_styles:
        out["tier_styles"] = dict(_COMPANY_LEXICON_SCAFFOLD["tier_styles"])
    else:
        normalized_tier_styles: dict[str, list[str]] = {}
        for key, value in tier_styles.items():
            bucket = _normalize_string_bucket(value)
            if bucket:
                normalized_tier_styles[str(key)] = bucket
        out["tier_styles"] = normalized_tier_styles or dict(_COMPANY_LEXICON_SCAFFOLD["tier_styles"])

    country_style_bias = out.get("country_style_bias")
    if not isinstance(country_style_bias, dict) or not country_style_bias:
        out["country_style_bias"] = dict(_COMPANY_LEXICON_SCAFFOLD["country_style_bias"])
    else:
        normalized_country_style_bias: dict[str, list[str]] = {}
        for key, value in country_style_bias.items():
            bucket = _normalize_string_bucket(value)
            if bucket:
                normalized_country_style_bias[str(key)] = bucket
        out["country_style_bias"] = normalized_country_style_bias or dict(_COMPANY_LEXICON_SCAFFOLD["country_style_bias"])

    return out


def _validate_company_lexicon(payload: dict[str, Any]) -> None:
    for key in ("prefixes", "suffixes", "abstract_nouns", "templates"):
        if not _is_string_list(payload.get(key), minimum=5):
            raise ValueError(f"company_lexicon missing robust {key}")
    for template in payload.get("templates", []):
        tokens = re.findall(r"\{([^{}]+)\}", str(template or ""))
        unsupported = [str(token) for token in tokens if str(token) not in _SUPPORTED_COMPANY_TEMPLATE_KEYS]
        if unsupported:
            raise ValueError(
                "company_lexicon template uses unsupported placeholders: "
                + ", ".join(sorted(set(unsupported)))
            )


def _validate_keyword_seed_bank(payload: dict[str, Any]) -> None:
    if not _is_string_list(payload.get("universal_qualifiers"), minimum=8):
        raise ValueError("keyword_seed_bank missing universal_qualifiers")
    genres = payload.get("genres")
    if not isinstance(genres, list):
        raise ValueError("keyword_seed_bank missing genres")
    genre_rows: dict[str, dict[str, Any]] = {}
    for row in genres:
        if not isinstance(row, dict):
            continue
        genre = str(row.get("genre", "") or "").strip()
        if genre:
            genre_rows[genre] = row
    missing = [str(genre) for genre in GENRES if str(genre) not in genre_rows]
    if missing:
        raise ValueError(f"keyword_seed_bank missing benchmark genres: {', '.join(missing)}")
    for genre in GENRES:
        row = genre_rows.get(str(genre), {})
        if not _is_string_list(row.get("seeds"), minimum=8):
            raise ValueError(f"keyword_seed_bank {genre} missing seeds")
        if not _is_string_list(row.get("qualifiers"), minimum=4):
            raise ValueError(f"keyword_seed_bank {genre} missing qualifiers")
        if not _is_string_list(row.get("contexts"), minimum=4):
            raise ValueError(f"keyword_seed_bank {genre} missing contexts")


def _coerce_keyword_seed_bank_globals_payload(parsed: Any) -> dict[str, Any] | None:
    payload = _coerce_keyword_seed_bank_payload(parsed)
    if not isinstance(payload, dict):
        return None
    return {
        "universal_qualifiers": list(payload.get("universal_qualifiers") or []),
        "universal_contexts": list(payload.get("universal_contexts") or []),
        "generic_themes": list(payload.get("generic_themes") or []),
    }


def _validate_keyword_seed_bank_globals_payload(payload: dict[str, Any]) -> None:
    if not _is_string_list(payload.get("universal_qualifiers"), minimum=30):
        raise ValueError("keyword_seed_bank globals missing robust universal_qualifiers")
    if not _is_string_list(payload.get("universal_contexts"), minimum=24):
        raise ValueError("keyword_seed_bank globals missing robust universal_contexts")
    if not _is_string_list(payload.get("generic_themes"), minimum=40):
        raise ValueError("keyword_seed_bank globals missing robust generic_themes")


def _coerce_keyword_seed_bank_genre_group_payload(
    parsed: Any,
    genres: Sequence[str],
) -> dict[str, Any] | None:
    payload = _coerce_keyword_seed_bank_payload(parsed)
    if not isinstance(payload, dict):
        return None
    requested = {str(genre) for genre in genres}
    rows = [
        row
        for row in list(payload.get("genres") or [])
        if isinstance(row, dict) and str(row.get("genre") or "") in requested
    ]
    return {"genres": rows}


def _validate_keyword_seed_bank_genre_group_payload(
    payload: dict[str, Any],
    genres: Sequence[str],
) -> None:
    rows = payload.get("genres")
    if not isinstance(rows, list):
        raise ValueError("keyword_seed_bank group missing genres")
    genre_rows: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        genre = str(row.get("genre", "") or "").strip()
        if genre:
            genre_rows[genre] = row
    missing = [str(genre) for genre in genres if str(genre) not in genre_rows]
    if missing:
        raise ValueError(f"keyword_seed_bank group missing genres: {', '.join(missing)}")
    for genre in genres:
        row = genre_rows[str(genre)]
        if not _is_string_list(row.get("seeds"), minimum=14):
            raise ValueError(f"keyword_seed_bank {genre} group missing robust seeds")
        if not _is_string_list(row.get("qualifiers"), minimum=6):
            raise ValueError(f"keyword_seed_bank {genre} group missing robust qualifiers")
        if not _is_string_list(row.get("contexts"), minimum=6):
            raise ValueError(f"keyword_seed_bank {genre} group missing robust contexts")
        if not _is_string_list(row.get("tone_tokens"), minimum=4):
            raise ValueError(f"keyword_seed_bank {genre} group missing tone_tokens")
        if not _is_string_list(row.get("exclusion_hints"), minimum=2):
            raise ValueError(f"keyword_seed_bank {genre} group missing exclusion_hints")


def _validate_title_grammar_bank(payload: dict[str, Any]) -> None:
    for key in ("adjectives", "nouns", "genre_templates", "tagline_templates", "allowed_tagline_placeholders", "tagline_render_constraints"):
        if key not in payload:
            raise ValueError(f"title_grammar_bank missing {key}")
    allowed_placeholders = payload.get("allowed_tagline_placeholders")
    if not isinstance(allowed_placeholders, list) or not allowed_placeholders:
        raise ValueError("title_grammar_bank missing allowed_tagline_placeholders")
    render_constraints = payload.get("tagline_render_constraints")
    if not isinstance(render_constraints, dict):
        raise ValueError("title_grammar_bank missing tagline_render_constraints")
    for key in ("min_words", "max_words", "max_placeholder_count", "forbid_square_brackets", "allow_unresolved_placeholders"):
        if key not in render_constraints:
            raise ValueError(f"title_grammar_bank tagline_render_constraints missing {key}")
    genre_templates = payload.get("genre_templates")
    if not isinstance(genre_templates, dict) or not genre_templates:
        raise ValueError("title_grammar_bank missing genre_templates")
    for genre in GENRES:
        title_values = genre_templates.get(str(genre))
        if not _is_string_list(title_values, minimum=_MIN_TITLE_TEMPLATES_PER_GENRE):
            count = len(title_values) if isinstance(title_values, list) else 0
            raise ValueError(
                f"title_grammar_bank {genre} must provide at least "
                f"{_MIN_TITLE_TEMPLATES_PER_GENRE} title templates (got {count})"
            )
    tagline_templates = payload.get("tagline_templates")
    if not isinstance(tagline_templates, dict) or not tagline_templates:
        raise ValueError("title_grammar_bank missing tagline_templates")
    duplicate_map: dict[str, set[str]] = {}
    for genre in GENRES:
        values = tagline_templates.get(str(genre))
        if not _is_string_list(values, minimum=12):
            count = len(values) if isinstance(values, list) else 0
            raise ValueError(f"title_grammar_bank {genre} must provide at least 12 tagline templates (got {count})")
        seen_local: set[str] = set()
        for raw in values:
            if _title_template_has_square_brackets(raw):
                raise ValueError(f"title_grammar_bank {genre} contains square-bracket placeholders: {raw}")
            key = re.sub(r"\s+", " ", str(raw or "").strip().lower())
            if not key or key in seen_local:
                continue
            seen_local.add(key)
            duplicate_map.setdefault(key, set()).add(str(genre))
    duplicates = [(template, genres) for template, genres in duplicate_map.items() if len(genres) > 1]
    if duplicates:
        preview = "; ".join(f"{template} -> {', '.join(sorted(genres))}" for template, genres in duplicates[:8])
        raise ValueError(f"title_grammar_bank has cross-genre duplicate tagline templates: {preview}")
    extra_values = payload.get("tagline_placeholder_values")
    if extra_values is None:
        extra_values = {}
    if not isinstance(extra_values, dict):
        raise ValueError("title_grammar_bank tagline_placeholder_values must be an object")
    missing_placeholder_values: list[str] = []
    allowed_placeholder_set = {_normalize_title_placeholder_token(item) for item in allowed_placeholders}
    for field_name in sorted(_title_grammar_placeholder_names(payload)):
        if field_name not in allowed_placeholder_set:
            raise ValueError(f"title_grammar_bank placeholder {field_name} is not allowed by placeholder policy")
        canonical = _TITLE_GRAMMAR_BASE_PLACEHOLDER_ALIASES.get(field_name, field_name)
        if canonical in _TITLE_GRAMMAR_BASE_PLACEHOLDER_ALIASES.values():
            continue
        values = extra_values.get(field_name)
        if not _is_string_list(values, minimum=4):
            missing_placeholder_values.append(field_name)
    if missing_placeholder_values:
        raise ValueError(
            "title_grammar_bank missing tagline_placeholder_values for: "
            + ", ".join(missing_placeholder_values)
        )
    for bucket_name in ("genre_templates", "tagline_templates"):
        section = payload.get(bucket_name)
        if not isinstance(section, dict):
            continue
        for bucket_values in section.values():
            if not isinstance(bucket_values, list):
                continue
            for raw in bucket_values:
                rendered = _smoke_render_title_template(raw, payload)
                if contains_placeholder_syntax(rendered):
                    raise ValueError(f"title_grammar_bank unresolved placeholders after smoke render: {raw}")
                if bucket_name == "tagline_templates":
                    _validate_smoke_rendered_tagline(raw, payload)


def _template_key(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _title_grammar_placeholder_names(payload: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for bucket in ("genre_templates", "tagline_templates"):
        section = payload.get(bucket)
        if not isinstance(section, dict):
            continue
        for values in section.values():
            if not isinstance(values, list):
                continue
            for raw in values:
                names.update(_title_template_placeholders(raw))
    return names


def _coerce_title_grammar_vocab_payload(parsed: Any) -> dict[str, Any] | None:
    if not isinstance(parsed, dict):
        return None
    payload = dict(parsed)
    if "vocab" in payload and isinstance(payload.get("vocab"), dict):
        payload = dict(payload.get("vocab") or {})
    return payload


def _validate_title_grammar_vocab_payload(payload: dict[str, Any]) -> None:
    for key in _TITLE_GRAMMAR_VOCAB_KEYS:
        if not _is_string_list(payload.get(key), minimum=_MIN_TITLE_GRAMMAR_VOCAB_ITEMS):
            count = len(payload.get(key)) if isinstance(payload.get(key), list) else 0
            raise ValueError(
                f"title_grammar_bank vocab {key} must provide at least "
                f"{_MIN_TITLE_GRAMMAR_VOCAB_ITEMS} tokens (got {count})"
            )


def _coerce_title_grammar_genre_templates_payload(parsed: Any, genres: Sequence[str]) -> dict[str, Any] | None:
    if not isinstance(parsed, dict):
        return None
    payload = dict(parsed)
    templates = payload.get("genre_templates")
    if isinstance(templates, dict):
        normalized = {
            str(genre): [
                template
                for item in list(templates.get(str(genre)) or [])
                if (template := _canonicalize_title_template_placeholders(item))
                and _title_template_uses_only_controlled_placeholders(template)
            ]
            for genre in genres
            if templates.get(str(genre)) is not None
        }
        return {"genre_templates": normalized}
    direct = {
        str(genre): [
            template
            for item in list(payload.get(str(genre)) or [])
            if (template := _canonicalize_title_template_placeholders(item))
            and _title_template_uses_only_controlled_placeholders(template)
        ]
        for genre in genres
        if payload.get(str(genre)) is not None
    }
    if direct:
        return {"genre_templates": direct}
    return None


def _validate_title_grammar_genre_templates_payload(payload: dict[str, Any], genres: Sequence[str]) -> None:
    templates = payload.get("genre_templates")
    if not isinstance(templates, dict):
        raise ValueError("title_grammar_bank missing genre_templates")
    for genre in genres:
        values = templates.get(str(genre))
        if not _is_string_list(values, minimum=_MIN_TITLE_TEMPLATES_PER_GENRE):
            count = len(values) if isinstance(values, list) else 0
            raise ValueError(
                f"title_grammar_bank {genre} must provide at least "
                f"{_MIN_TITLE_TEMPLATES_PER_GENRE} title templates (got {count})"
            )
        for raw in values:
            if _title_template_has_square_brackets(raw):
                raise ValueError(f"title_grammar_bank {genre} title template uses square-bracket placeholder: {raw}")
            if not _title_template_uses_only_controlled_placeholders(raw):
                raise ValueError(f"title_grammar_bank {genre} title template uses unsupported placeholder: {raw}")


def _coerce_title_grammar_taglines_payload(parsed: Any, genres: Sequence[str]) -> dict[str, Any] | None:
    if not isinstance(parsed, dict):
        return None
    payload = dict(parsed)
    templates = payload.get("tagline_templates")
    if isinstance(templates, dict):
        normalized = {
            str(genre): [_canonicalize_title_template_placeholders(item) for item in list(templates.get(str(genre)) or [])]
            for genre in genres
            if templates.get(str(genre)) is not None
        }
        normalized = _sanitize_title_grammar_taglines_map(normalized, payload)
        return {"tagline_templates": normalized}
    direct = {
        str(genre): [_canonicalize_title_template_placeholders(item) for item in list(payload.get(str(genre)) or [])]
        for genre in genres
        if payload.get(str(genre)) is not None
    }
    if direct:
        direct = _sanitize_title_grammar_taglines_map(direct, payload)
        return {"tagline_templates": direct}
    return None


def _sanitize_title_grammar_taglines_map(
    templates: dict[str, list[str]],
    payload: dict[str, Any],
) -> dict[str, list[str]]:
    cleaned: dict[str, list[str]] = {}
    seen_global: set[str] = set()
    for genre, raw_values in dict(templates or {}).items():
        local_seen: set[str] = set()
        keep: list[str] = []
        for raw in list(raw_values or []):
            template = _canonicalize_title_template_placeholders(raw)
            if not template or _title_template_has_square_brackets(template):
                continue
            if not _title_template_uses_only_controlled_placeholders(template):
                continue
            try:
                _validate_smoke_rendered_tagline(template, payload)
            except Exception:
                continue
            key = re.sub(r"\s+", " ", str(template).strip().lower())
            if not key or key in local_seen or key in seen_global:
                continue
            local_seen.add(key)
            seen_global.add(key)
            keep.append(template)
        cleaned[str(genre)] = keep
    return cleaned


def _validate_title_grammar_taglines_payload(
    payload: dict[str, Any],
    genres: Sequence[str],
    *,
    minimum_per_genre: int = 12,
) -> None:
    templates = payload.get("tagline_templates")
    if not isinstance(templates, dict):
        raise ValueError("title_grammar_bank missing tagline_templates")
    seen: set[str] = set()
    for genre in genres:
        values = templates.get(str(genre))
        if not _is_string_list(values, minimum=minimum_per_genre):
            count = len(values) if isinstance(values, list) else 0
            raise ValueError(f"title_grammar_bank {genre} must provide at least {minimum_per_genre} tagline templates (got {count})")
        local_seen: set[str] = set()
        for raw in values:
            if _title_template_has_square_brackets(raw):
                raise ValueError(f"title_grammar_bank {genre} tagline template uses square-bracket placeholder: {raw}")
            _validate_smoke_rendered_tagline(raw, payload)
            key = re.sub(r"\s+", " ", str(raw or "").strip().lower())
            if not key:
                continue
            if key in local_seen:
                raise ValueError(f"title_grammar_bank {genre} contains duplicate tagline templates")
            if key in seen:
                raise ValueError(f"title_grammar_bank has cross-genre duplicate tagline template: {key}")
            local_seen.add(key)
            seen.add(key)


def _coerce_title_grammar_placeholder_values_payload(
    parsed: Any,
    placeholders: Sequence[str],
) -> dict[str, Any] | None:
    if not isinstance(parsed, dict):
        return None
    payload = dict(parsed)
    values = payload.get("tagline_placeholder_values")
    if isinstance(values, dict):
        return {
            "tagline_placeholder_values": {
                _normalize_title_placeholder_token(name): raw
                for name, raw in values.items()
            }
        }
    direct = {
        _normalize_title_placeholder_token(name): payload.get(str(name))
        for name in placeholders
        if payload.get(str(name)) is not None
    }
    if direct:
        return {"tagline_placeholder_values": direct}
    return None


def _validate_title_grammar_placeholder_values_payload(
    payload: dict[str, Any],
    placeholders: Sequence[str],
) -> None:
    values = payload.get("tagline_placeholder_values")
    if not isinstance(values, dict):
        raise ValueError("title_grammar_bank missing tagline_placeholder_values")
    for name in placeholders:
        entries = values.get(_normalize_title_placeholder_token(name))
        if not _is_string_list(entries, minimum=8):
            count = len(entries) if isinstance(entries, list) else 0
            raise ValueError(f"title_grammar_bank placeholder {name} must provide at least 8 values (got {count})")


def _dedupe_title_grammar_taglines(payload: dict[str, Any]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    templates = dict(payload.get("tagline_templates") or {})
    kept: dict[str, list[str]] = {}
    owners: dict[str, str] = {}
    duplicates: dict[str, list[str]] = {}
    for genre in GENRES:
        genre_key = str(genre)
        values = templates.get(genre_key)
        clean_values = [str(item).strip() for item in values] if isinstance(values, list) else []
        seen_local: set[str] = set()
        out: list[str] = []
        for raw in clean_values:
            key = _template_key(raw)
            if not key or key in seen_local:
                continue
            seen_local.add(key)
            owner = owners.get(key)
            if owner is None:
                owners[key] = genre_key
                out.append(raw)
            elif owner != genre_key:
                duplicates.setdefault(genre_key, []).append(raw)
        kept[genre_key] = out
    return kept, duplicates


def _coerce_title_grammar_phases_payload(parsed: Any) -> dict[str, Any] | None:
    if isinstance(parsed, dict):
        payload = dict(parsed)
        phases = payload.get("year_style_phases")
        if isinstance(phases, list):
            return {"year_style_phases": phases}
    if isinstance(parsed, list):
        return {"year_style_phases": parsed}
    return None


def _validate_title_grammar_phases_payload(payload: dict[str, Any]) -> None:
    phases = payload.get("year_style_phases")
    if not isinstance(phases, list) or not (4 <= len(phases) <= 8):
        count = len(phases) if isinstance(phases, list) else 0
        raise ValueError(f"title_grammar_bank must provide 4-8 year_style_phases (got {count})")
    required = ("label", "range_hint", "favored_tokens", "title_tendencies", "tagline_tendencies")
    for idx, row in enumerate(phases):
        if not isinstance(row, dict):
            raise ValueError(f"title_grammar_bank year_style_phases[{idx}] is not an object")
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"title_grammar_bank year_style_phases[{idx}] missing {', '.join(missing)}")


def _validate_temporal_regime_plan(payload: dict[str, Any]) -> None:
    if "year_weights" not in payload or not isinstance(payload.get("year_weights"), list):
        raise ValueError("temporal_regime_plan missing year_weights")
    if "phases" not in payload or not isinstance(payload.get("phases"), list):
        raise ValueError("temporal_regime_plan missing phases")


_DEFAULT_PERSON_PERSON_CLASSIFICATION_KEYS = (
    "controversy_high_threshold",
    "controversy_gap_avoid_threshold",
    "controversy_avoid_weight_floor",
    "controversy_friendship_weight_base",
    "controversy_friendship_score_weight",
    "controversy_friendship_jitter",
    "mentorship_style_threshold",
    "mentorship_stage_gap_threshold",
    "mentorship_weight_base",
    "mentorship_style_weight",
    "mentorship_jitter",
    "rivalry_style_max",
    "rivalry_genre_min",
    "rivalry_stage_gap_max",
    "rivalry_weight_base",
    "rivalry_genre_weight",
    "rivalry_style_distance_weight",
    "rivalry_jitter",
    "friendship_style_threshold",
    "friendship_probability_threshold",
    "friendship_style_soft_threshold",
    "friendship_weight_base",
    "friendship_score_weight",
    "friendship_jitter",
    "weight_min",
    "weight_max",
)

_DEFAULT_PERSON_COMPANY_GENERATION_KEYS = (
    "genre_supplement_size",
    "controversy_blacklist_person_threshold",
    "controversy_blacklist_company_threshold",
    "blacklist_weight_base",
    "blacklist_weight_controversy_scale",
    "event_franchise_micro_budget_penalty_boost",
    "market_fit_boost",
)

_DEFAULT_COMPANY_COMPANY_GENERATION_KEYS = (
    "strategy_match_boost",
    "market_match_boost",
    "rival_overlap_threshold",
    "rival_tier_threshold",
    "rival_weight_scale",
    "rival_weight_policy_cap",
    "coproduction_overlap_threshold",
    "coproduction_tier_max",
    "coproduction_weight_scale",
    "coproduction_policy_cap",
)

_DEFAULT_CALIBRATION_KEYS = (
    "relationship_targets",
    "candidate_sample_k",
    "director_actor_supplement",
    "upsert_weight_min",
    "upsert_weight_max",
    "friendship_score_weights",
    "rivalry_score_weights",
    "preferential_attachment_log_weight",
    "best_friend_weight_base",
    "best_friend_weight_score_scale",
    "best_friend_weight_noise",
    "rival_weight_base",
    "rival_weight_score_scale",
    "rival_weight_noise",
    "director_stage_target_base",
    "director_stage_target_span",
    "director_preferred_score_weights",
    "director_preferred_weight_base",
    "director_preferred_weight_score_scale",
    "director_preferred_weight_noise",
    "director_avoid_score_weights",
    "director_avoid_weight_base",
    "director_avoid_weight_score_scale",
    "director_avoid_weight_noise",
    "bf_same_community_weight_floor",
    "bf_same_community_weight_boost",
)


def _require_dict_keys(value: dict[str, Any], label: str, keys: tuple[str, ...]) -> None:
    missing = [str(key) for key in keys if key not in value]
    if missing:
        raise ValueError(f"{label} missing keys: {', '.join(missing)}")


def _require_numeric_keys(value: dict[str, Any], label: str, keys: tuple[str, ...]) -> None:
    missing: list[str] = []
    invalid: list[str] = []
    for key in keys:
        if key not in value:
            missing.append(str(key))
            continue
        try:
            float(value.get(key))
        except Exception:
            invalid.append(str(key))
    if missing:
        raise ValueError(f"{label} missing numeric keys: {', '.join(missing)}")
    if invalid:
        raise ValueError(f"{label} has non-numeric keys: {', '.join(invalid)}")


def _require_certification_distribution_map(value: Any, label: str) -> None:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} must be a non-empty dict")
    missing_genres = [str(genre) for genre in GENRES if str(genre) not in value]
    if missing_genres:
        preview = ", ".join(missing_genres[:10])
        if len(missing_genres) > 10:
            preview += ", ..."
        raise ValueError(f"{label} missing genre rows: {preview}")
    allowed = {str(cert) for cert in CERTIFICATIONS}
    for genre in GENRES:
        row = value.get(str(genre))
        if not isinstance(row, dict) or not row:
            raise ValueError(f"{label}.{genre} must be a non-empty dict")
        positive_total = 0.0
        usable_keys = 0
        for key, raw_value in row.items():
            cert_key = str(key)
            if cert_key not in allowed:
                continue
            try:
                score = float(raw_value)
            except Exception:
                continue
            if score > 0:
                positive_total += score
                usable_keys += 1
        if usable_keys == 0 or positive_total <= 0:
            raise ValueError(f"{label}.{genre} missing usable certification weights")


def _missing_dict_paths(base: Any, update: Any, prefix: str) -> list[str]:
    missing: list[str] = []
    if isinstance(base, dict):
        if not isinstance(update, dict):
            return [prefix]
        for key, value in base.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            if key not in update:
                missing.append(child_prefix)
                continue
            missing.extend(_missing_dict_paths(value, update.get(key), child_prefix))
    return missing


def _require_tier_float_map(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a dict keyed by production tier")
    missing = [tier for tier in PRODUCTION_TIERS if tier not in value]
    if missing:
        raise ValueError(f"{label} missing tiers: {', '.join(missing)}")
    for tier in PRODUCTION_TIERS:
        try:
            float(value.get(tier))
        except Exception:
            raise ValueError(f"{label}.{tier} must be numeric")


def _require_float_map_keys(value: Any, label: str, required_keys: tuple[str, ...] | list[str]) -> None:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} must be a non-empty numeric dict")
    missing = [str(key) for key in required_keys if str(key) not in value]
    if missing:
        preview = ", ".join(missing[:12])
        if len(missing) > 12:
            preview += ", ..."
        raise ValueError(f"{label} missing keys: {preview}")
    invalid: list[str] = []
    for key in required_keys:
        try:
            float(value.get(str(key)))
        except Exception:
            invalid.append(str(key))
    if invalid:
        preview = ", ".join(invalid[:12])
        if len(invalid) > 12:
            preview += ", ..."
        raise ValueError(f"{label} has non-numeric values: {preview}")


def _require_financial_budget_ranges(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a dict keyed by production tier")
    missing = [tier for tier in PRODUCTION_TIERS if tier not in value]
    if missing:
        raise ValueError(f"{label} missing tiers: {', '.join(missing)}")
    for tier in PRODUCTION_TIERS:
        row = value.get(tier)
        try:
            if isinstance(row, dict):
                lo = float(row.get("min_budget"))
                hi = float(row.get("max_budget"))
            elif isinstance(row, (list, tuple)) and len(row) >= 2:
                lo = float(row[0])
                hi = float(row[1])
            else:
                raise ValueError
        except Exception:
            raise ValueError(f"{label}.{tier} must contain numeric min_budget/max_budget")
        if lo <= 0 or hi <= lo:
            raise ValueError(f"{label}.{tier} must have 0 < min_budget < max_budget")
        if hi < 10_000:
            raise ValueError(f"{label}.{tier} appears to be in millions, not absolute USD")


def _require_tier_range_map(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a dict keyed by production tier")
    missing = [tier for tier in PRODUCTION_TIERS if tier not in value]
    if missing:
        raise ValueError(f"{label} missing tiers: {', '.join(missing)}")
    for tier in PRODUCTION_TIERS:
        row = value.get(tier)
        if isinstance(row, dict):
            try:
                lo = int(row.get("min"))
                hi = int(row.get("max"))
            except Exception:
                raise ValueError(f"{label}.{tier} must have integer min/max")
        elif isinstance(row, (list, tuple)) and len(row) >= 2:
            try:
                lo = int(row[0])
                hi = int(row[1])
            except Exception:
                raise ValueError(f"{label}.{tier} must contain integer min/max values")
        else:
            raise ValueError(f"{label}.{tier} must be a [min,max] pair or object with min/max")
        if hi < lo:
            raise ValueError(f"{label}.{tier} has max < min")


def _require_genre_tier_distribution(value: Any, label: str) -> None:
    if not isinstance(value, dict) or len(value) < min(8, len(GENRES)):
        raise ValueError(f"{label} must be a non-trivial dict keyed by genre")
    for genre, row in value.items():
        if isinstance(row, dict):
            missing = [tier for tier in PRODUCTION_TIERS if tier not in row]
            if missing:
                raise ValueError(f"{label}.{genre} missing tiers: {', '.join(missing)}")
            for tier in PRODUCTION_TIERS:
                try:
                    float(row.get(tier))
                except Exception:
                    raise ValueError(f"{label}.{genre}.{tier} must be numeric")
        elif isinstance(row, (list, tuple)) and len(row) == len(PRODUCTION_TIERS):
            try:
                [float(v) for v in row]
            except Exception:
                raise ValueError(f"{label}.{genre} must contain numeric values")
        else:
            raise ValueError(f"{label}.{genre} must be a tier dict or list of length {len(PRODUCTION_TIERS)}")


def _require_vector_map(value: Any, label: str, *, expected_len: int) -> None:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} must be a non-empty dict")
    for key, row in value.items():
        if isinstance(row, (list, tuple)) and len(row) == expected_len:
            try:
                [float(v) for v in row]
            except Exception:
                raise ValueError(f"{label}.{key} must contain numeric values")
            continue
        try:
            float(row)
        except Exception:
            raise ValueError(f"{label}.{key} must be numeric or a length-{expected_len} numeric vector")


_CHARACTER_RUNTIME_ARCHETYPES: tuple[str, ...] = (
    "Lead Hero",
    "Lead Villain",
    "Love Interest",
    "Mentor",
    "Sidekick",
    "Comic Relief",
    "Supporting",
    "Authority Figure",
    "Henchman",
    "Victim",
    "Mysterious Stranger",
    "Extra",
)
_CHARACTER_RUNTIME_ARCHETYPE_SET = set(_CHARACTER_RUNTIME_ARCHETYPES)
_CHARACTER_RUNTIME_ARCHETYPE_ALIASES: dict[str, str] = {
    "hero": "Lead Hero",
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
_CHARACTER_STAGE_ALIASES: dict[str, str] = {
    "emerging": "rising",
    "established": "prime",
    "midcareer": "prime",
    "mid career": "prime",
    "late career": "veteran",
}
_CHARACTER_COLLAB_STYLE_ALIASES: dict[str, str] = {
    "auteur": "solo",
    "studio": "ensemble",
    "collaborative": "ensemble",
    "team": "ensemble",
    "guided": "mentorship",
    "mentor led": "mentorship",
}


def _normalize_runtime_label(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _canonical_character_archetype(value: Any, *, fallback: str = "Supporting") -> str:
    raw = str(value or "").strip()
    if not raw:
        return str(fallback)
    if raw in _CHARACTER_RUNTIME_ARCHETYPE_SET:
        return raw
    token = _normalize_runtime_label(raw)
    mapped = _CHARACTER_RUNTIME_ARCHETYPE_ALIASES.get(token)
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


def _canonical_character_stage(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return "prime"
    return _CHARACTER_STAGE_ALIASES.get(raw, raw)


def _canonical_character_collab_style(value: Any) -> str:
    token = _normalize_runtime_label(value)
    if not token:
        return "ensemble"
    return _CHARACTER_COLLAB_STYLE_ALIASES.get(token, token)


def _normalize_character_generation_priors(section: Any) -> dict[str, Any]:
    if not isinstance(section, dict):
        return {}
    out = dict(section)

    slot_candidates = out.get("slot_archetype_candidates")
    if isinstance(slot_candidates, dict):
        normalized_slots: dict[str, list[str]] = {}
        for slot, values in slot_candidates.items():
            if not isinstance(values, list):
                continue
            deduped: list[str] = []
            seen: set[str] = set()
            for value in values:
                canonical = _canonical_character_archetype(value)
                if canonical and canonical not in seen:
                    deduped.append(canonical)
                    seen.add(canonical)
            if deduped:
                normalized_slots[str(slot)] = deduped
        out["slot_archetype_candidates"] = normalized_slots

    for key in ("general_archetypes", "unique_archetypes"):
        raw_values = out.get(key)
        if isinstance(raw_values, list):
            deduped: list[str] = []
            seen: set[str] = set()
            for value in raw_values:
                canonical = _canonical_character_archetype(value)
                if canonical and canonical not in seen:
                    deduped.append(canonical)
                    seen.add(canonical)
            out[key] = deduped

    target_vectors = out.get("archetype_target_vectors")
    if isinstance(target_vectors, dict):
        normalized_targets: dict[str, list[float]] = {}
        for archetype, values in target_vectors.items():
            if not isinstance(values, (list, tuple)) or len(values) != 4:
                continue
            try:
                normalized_targets[_canonical_character_archetype(archetype)] = [float(v) for v in values]
            except Exception:
                continue
        out["archetype_target_vectors"] = normalized_targets

    for key, key_normalizer in (
        ("career_stage_archetype_bias", _canonical_character_stage),
        ("genre_archetype_bias", str),
        ("collaboration_style_archetype_bias", _canonical_character_collab_style),
    ):
        raw_value = out.get(key)
        if not isinstance(raw_value, dict):
            continue
        normalized_outer: dict[str, dict[str, float]] = {}
        for outer_key, inner in raw_value.items():
            if not isinstance(inner, dict):
                continue
            outer_name = key_normalizer(outer_key)
            bucket = dict(normalized_outer.get(outer_name, {}))
            for archetype, score in inner.items():
                try:
                    bucket[_canonical_character_archetype(archetype)] = float(score)
                except Exception:
                    continue
            if bucket:
                normalized_outer[outer_name] = bucket
        out[key] = normalized_outer

    return out


def _validate_modeling_priors_section(section: str, value: dict[str, Any]) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"modeling_priors missing {section}")
    if not value:
        raise ValueError(f"modeling_priors section {section} is empty")
    if section == "person_generation":
        if not isinstance(value.get("gender_ratio"), dict) or not value.get("gender_ratio"):
            raise ValueError("modeling_priors person_generation missing gender_ratio")
        if not (
            isinstance(value.get("age_distribution"), dict)
            or ("age_distribution_mean" in value and "age_distribution_std" in value)
        ):
            raise ValueError("modeling_priors person_generation missing age distribution")
    elif section == "company_generation":
        if len(value) < 2:
            raise ValueError("modeling_priors company_generation too shallow")
        tier_weights = value.get("tier_weights")
        if not isinstance(tier_weights, dict) or len(tier_weights) < 5:
            raise ValueError("modeling_priors company_generation missing tier_weights")
    elif section == "keyword_generation":
        if len(value) < 2:
            raise ValueError("modeling_priors keyword_generation too shallow")
        genre_target_weights = value.get("genre_target_weights")
        if not isinstance(genre_target_weights, dict) or len(genre_target_weights) < len(GENRES):
            raise ValueError("modeling_priors keyword_generation missing genre_target_weights")
        if "generic_budget_ratio" not in value:
            raise ValueError("modeling_priors keyword_generation missing generic_budget_ratio")
        if "min_specific_story_share" not in value:
            raise ValueError("modeling_priors keyword_generation missing min_specific_story_share")
        selection_bucket_targets = value.get("selection_bucket_targets")
        if not isinstance(selection_bucket_targets, dict):
            raise ValueError("modeling_priors keyword_generation missing selection_bucket_targets")
        _require_numeric_keys(
            selection_bucket_targets,
            "modeling_priors keyword_generation.selection_bucket_targets",
            ("exact_anchor", "related_support", "story_specific", "generic"),
        )
    elif section == "character_generation":
        if len(value) < 6:
            raise ValueError("modeling_priors character_generation too shallow")
        required_keys = (
            "slot_archetype_candidates",
            "general_archetypes",
            "unique_archetypes",
            "genre_archetype_candidates",
            "archetype_target_vectors",
            "career_stage_archetype_bias",
            "genre_archetype_bias",
            "collaboration_style_archetype_bias",
        )
        for required_key in required_keys:
            if required_key not in value:
                raise ValueError(f"modeling_priors character_generation missing {required_key}")
        slot_candidates = value.get("slot_archetype_candidates")
        if not isinstance(slot_candidates, dict) or not slot_candidates:
            raise ValueError("modeling_priors character_generation missing slot_archetype_candidates")
        for slot in ("0", "1", "2"):
            if not isinstance(slot_candidates.get(slot), list) or not slot_candidates.get(slot):
                raise ValueError(f"modeling_priors character_generation slot_archetype_candidates missing slot {slot}")
        if not isinstance(value.get("general_archetypes"), list) or not value.get("general_archetypes"):
            raise ValueError("modeling_priors character_generation missing general_archetypes")
        if not isinstance(value.get("unique_archetypes"), list) or not value.get("unique_archetypes"):
            raise ValueError("modeling_priors character_generation missing unique_archetypes")
        archetype_targets = value.get("archetype_target_vectors")
        if not isinstance(archetype_targets, dict) or not archetype_targets:
            raise ValueError("modeling_priors character_generation missing archetype_target_vectors")
        needed_archetypes = set()
        for rows in slot_candidates.values():
            if isinstance(rows, list):
                needed_archetypes.update(str(item) for item in rows if str(item).strip())
        needed_archetypes.update(str(item) for item in value.get("general_archetypes", []) if str(item).strip())
        needed_archetypes.update(str(item) for item in value.get("unique_archetypes", []) if str(item).strip())
        needed_archetypes.update(("Lead Hero", "Lead Villain", "Mentor", "Supporting"))
        missing_targets = sorted(archetype for archetype in needed_archetypes if archetype not in archetype_targets)
        if missing_targets:
            raise ValueError(
                "modeling_priors character_generation missing archetype targets: "
                + ", ".join(missing_targets[:8])
            )
        _require_vector_map(archetype_targets, "modeling_priors character_generation.archetype_target_vectors", expected_len=4)
    elif section == "title_generation":
        genre_weights = value.get("genre_base_weights")
        if not isinstance(genre_weights, dict) or len(genre_weights) < 8:
            raise ValueError("modeling_priors title_generation missing robust genre_base_weights")
        allowed_placeholders = value.get("allowed_tagline_placeholders")
        if not isinstance(allowed_placeholders, list) or not allowed_placeholders:
            raise ValueError("modeling_priors title_generation missing allowed_tagline_placeholders")
        render_constraints = value.get("tagline_render_constraints")
        if not isinstance(render_constraints, dict):
            raise ValueError("modeling_priors title_generation missing tagline_render_constraints")
        _require_numeric_keys(
            render_constraints,
            "modeling_priors title_generation.tagline_render_constraints",
            ("min_words", "max_words", "max_placeholder_count"),
        )
        for key in ("forbid_square_brackets", "allow_unresolved_placeholders"):
            if key not in render_constraints:
                raise ValueError(f"modeling_priors title_generation.tagline_render_constraints missing {key}")
    elif section == "company_finance_tiers":
        if len(value) < 3:
            raise ValueError("modeling_priors company_finance_tiers too shallow")
    elif section == "selection_weights":
        _require_numeric_keys(
            value,
            "modeling_priors selection_weights",
            (
                "cast_focus_exploration",
                "cast_base_focus_exploration",
                "cast_slot_exploration_empty",
                "cast_slot_exploration_filled",
                "cast_style_multiplier",
                "cast_community_match_multiplier",
                "director_exploration_share",
                "company_primary_exploration_share",
                "company_secondary_exploration_share",
                "crew_exploration_share",
                "concept_genre_bias_base",
                "concept_genre_bias_scale",
                "concept_country_bias_base",
                "concept_country_bias_scale",
                "concept_market_bias_base",
                "concept_market_bias_scale",
                "concept_exact_genre_hint_boost",
                "concept_genre_hint_miss_penalty",
                "concept_franchise_genre_match_boost",
                "concept_tier_match_boost",
                "concept_franchise_eligible_scale",
                "concept_strategy_bonus_scale",
                "concept_release_pressure_base",
                "concept_release_pressure_scale",
                "concept_novelty_base",
                "concept_novelty_scale",
                "concept_franchise_strategy_match_boost",
                "concept_franchise_season_match_boost",
                "concept_sequel_pressure_scale",
                "concept_pack_usage_capacity",
                "concept_bucket_country_capacity",
                "concept_bucket_genre_capacity",
                "concept_country_usage_capacity",
                "concept_minor_country_bonus",
                "concept_minor_market_bonus",
                "director_risk_weight",
                "director_ambition_weight",
                "director_prestige_weight",
                "director_alignment_base",
                "director_alignment_scale",
                "director_csv_base",
                "director_csv_scale",
                "company_tier_match_boost",
                "company_tier_mismatch_penalty",
                "company_genre_match_boost",
                "company_genre_mismatch_penalty",
                "company_risk_weight",
                "company_prestige_weight",
                "company_focus_weight",
                "company_genre_fit_weight",
                "company_alignment_base",
                "company_alignment_scale",
            ),
        )
        for block_name, block_keys in (
            ("director_selection", ("genre_match_boost", "geo_boost_scale", "geo_boost_floor", "film_count_decay_over_30", "film_count_decay_over_50", "film_count_decay_over_80", "company_multiplier_rescale", "event_franchise_pop_quantile", "event_franchise_pop_boost", "prestige_drama_alignment_threshold", "prestige_drama_alignment_boost")),
            ("company_selection", ("strategy_match_boost", "market_bias_base", "market_bias_scale", "partner_affinity_scale", "partner_rivalry_penalty_floor", "family_boost")),
            ("cast_selection", ("blockbuster_bonus_major", "blockbuster_bonus_other", "franchise_bonus_major", "franchise_bonus_other", "epic_tail_prob_base", "epic_tail_prob_franchise_bonus", "epic_tail_prob_blockbuster_bonus", "epic_tail_prob_cap", "epic_tail_lognorm_mean", "epic_tail_lognorm_sigma", "epic_tail_min", "a_tail_prob", "a_tail_min", "a_tail_max", "unused_actor_boost", "award_recent_boost", "franchise_pool_base_boost", "star_vehicle_slot0_boost", "prestige_pairing_boost", "volatile_ensemble_boost", "balanced_ensemble_boost", "agency_match_boost", "gender_novelty_boost", "nationality_novelty_boost", "tag_similarity_penalty")),
            ("keyword_selection", ("franchise_min_count", "exact_genre_boost", "family_genre_boost", "off_genre_penalty", "lexical_match_scale", "lexical_match_cap", "specificity_tier1_penalty", "generic_motif_penalty", "specific_story_boost", "franchise_scope_boost_base", "franchise_scope_affinity_scale", "franchise_family_boost", "franchise_recurrence_base", "franchise_recurrence_scale", "nonfranchise_scope_penalty", "nonfranchise_family_penalty", "nonfranchise_affinity_penalty", "nonfranchise_affinity_threshold", "high_specificity_novelty_base", "high_specificity_novelty_scale", "movie_scope_novelty_base", "movie_scope_novelty_scale", "usage_penalty_scale", "company_exact_boost", "company_family_boost", "franchise_core_boost", "family_genre_max_share")),
        ):
            block = value.get(block_name)
            if not isinstance(block, dict) or not block:
                raise ValueError(f"modeling_priors selection_weights missing nested selector config {block_name}")
            _require_numeric_keys(block, f"modeling_priors selection_weights.{block_name}", block_keys)
        _require_tier_float_map(value.get("co_director_probability_by_tier"), "modeling_priors selection_weights.co_director_probability_by_tier")
        _require_tier_float_map(value.get("writer_director_probability_by_tier"), "modeling_priors selection_weights.writer_director_probability_by_tier")
        _require_tier_float_map(value.get("geo_boost_by_tier"), "modeling_priors selection_weights.geo_boost_by_tier")
        _require_tier_range_map(value.get("dynamic_cast_base_by_tier"), "modeling_priors selection_weights.dynamic_cast_base_by_tier")
        _require_tier_range_map(value.get("keyword_count_by_tier"), "modeling_priors selection_weights.keyword_count_by_tier")
        keyword_selection = value.get("keyword_selection") if isinstance(value.get("keyword_selection"), dict) else {}
        _require_tier_float_map(keyword_selection.get("primary_genre_min_count_by_tier"), "modeling_priors selection_weights.keyword_selection.primary_genre_min_count_by_tier")
        _require_tier_float_map(keyword_selection.get("exact_topic_min_count_by_tier"), "modeling_priors selection_weights.keyword_selection.exact_topic_min_count_by_tier")
        _require_tier_float_map(keyword_selection.get("primary_plus_related_min_count_by_tier"), "modeling_priors selection_weights.keyword_selection.primary_plus_related_min_count_by_tier")
        _require_tier_float_map(keyword_selection.get("generic_keyword_cap_by_tier"), "modeling_priors selection_weights.keyword_selection.generic_keyword_cap_by_tier")
        _require_tier_float_map(keyword_selection.get("off_genre_cap_by_tier"), "modeling_priors selection_weights.keyword_selection.off_genre_cap_by_tier")
        slot_mix = keyword_selection.get("slot_mix_by_tier")
        if not isinstance(slot_mix, dict):
            raise ValueError("modeling_priors selection_weights.keyword_selection missing slot_mix_by_tier")
        for tier in PRODUCTION_TIERS:
            row = slot_mix.get(str(tier))
            if not isinstance(row, dict):
                raise ValueError(f"modeling_priors selection_weights.keyword_selection.slot_mix_by_tier.{tier} must be an object")
            _require_numeric_keys(
                row,
                f"modeling_priors selection_weights.keyword_selection.slot_mix_by_tier.{tier}",
                ("exact_anchor", "related_support", "story_specific", "franchise", "generic"),
            )
        related_genres = keyword_selection.get("related_genres_by_genre")
        if not isinstance(related_genres, dict):
            raise ValueError("modeling_priors selection_weights.keyword_selection missing related_genres_by_genre")
        for genre in GENRES:
            row = related_genres.get(str(genre))
            if not isinstance(row, list) or not row:
                raise ValueError(f"modeling_priors selection_weights.keyword_selection.related_genres_by_genre.{genre} must be a non-empty list")
        _require_genre_tier_distribution(value.get("genre_tier_distribution"), "modeling_priors selection_weights.genre_tier_distribution")
        _require_vector_map(value.get("concept_style_vector_by_genre"), "modeling_priors selection_weights.concept_style_vector_by_genre", expected_len=8)
        _require_vector_map(value.get("concept_style_tier_shift_by_tier"), "modeling_priors selection_weights.concept_style_tier_shift_by_tier", expected_len=8)
        _require_tier_float_map(value.get("concept_ambition_target_by_tier"), "modeling_priors selection_weights.concept_ambition_target_by_tier")
        _require_tier_float_map(value.get("concept_prestige_target_by_tier"), "modeling_priors selection_weights.concept_prestige_target_by_tier")
        if not isinstance(value.get("concept_risk_target_by_genre"), dict) or len(value.get("concept_risk_target_by_genre", {})) < min(8, len(GENRES)):
            raise ValueError("modeling_priors selection_weights missing concept_risk_target_by_genre")
        if not isinstance(value.get("release_month_base_weights"), list) or len(value.get("release_month_base_weights")) != 12:
            raise ValueError("modeling_priors selection_weights missing release_month_base_weights")
        if not isinstance(value.get("keyword_year_slate_family_boosts"), dict) or not value.get("keyword_year_slate_family_boosts"):
            raise ValueError("modeling_priors selection_weights missing keyword_year_slate_family_boosts")
        if not isinstance(value.get("release_season_month_bumps"), dict) or not value.get("release_season_month_bumps"):
            raise ValueError("modeling_priors selection_weights missing release_season_month_bumps")
        if not isinstance(value.get("genre_release_month_bumps"), dict) or not value.get("genre_release_month_bumps"):
            raise ValueError("modeling_priors selection_weights missing genre_release_month_bumps")
    elif section == "edge_priors":
        _require_numeric_keys(
            value,
            "modeling_priors edge_priors",
            (
                "cross_genre_candidate_k",
                "cross_genre_candidate_multiplier",
                "cross_genre_threshold_bump",
                "person_person_style_weight",
                "person_person_genre_weight",
                "person_person_risk_weight",
                "person_person_stage_weight",
                "person_person_noise_weight",
                "person_person_policy_weight",
                "person_person_logistic_scale",
                "person_person_logistic_bias",
                "person_person_base_threshold",
                "person_person_degree_decay",
                "person_person_degree_power",
                "person_company_risk_weight",
                "person_company_budget_weight",
                "person_company_genre_weight",
                "person_company_noise_weight",
                "person_company_blacklist_threshold",
                "person_company_brand_fit_threshold",
            ),
        )
        _require_dict_keys(
            value.get("person_person_classification", {}) if isinstance(value.get("person_person_classification"), dict) else {},
            "modeling_priors edge_priors.person_person_classification",
            tuple(_DEFAULT_PERSON_PERSON_CLASSIFICATION_KEYS),
        )
        _require_dict_keys(
            value.get("person_company_generation", {}) if isinstance(value.get("person_company_generation"), dict) else {},
            "modeling_priors edge_priors.person_company_generation",
            tuple(_DEFAULT_PERSON_COMPANY_GENERATION_KEYS),
        )
        _require_dict_keys(
            value.get("company_company_generation", {}) if isinstance(value.get("company_company_generation"), dict) else {},
            "modeling_priors edge_priors.company_company_generation",
            tuple(_DEFAULT_COMPANY_COMPANY_GENERATION_KEYS),
        )
        _require_dict_keys(
            value.get("serendipitous_edges", {}) if isinstance(value.get("serendipitous_edges"), dict) else {},
            "modeling_priors edge_priors.serendipitous_edges",
            ("stage_probabilities", "stage_max_new_edges", "candidate_multiplier", "weight_min", "weight_max"),
        )
        serendipitous = value.get("serendipitous_edges", {}) if isinstance(value.get("serendipitous_edges"), dict) else {}
        stage_probabilities = serendipitous.get("stage_probabilities")
        stage_caps = serendipitous.get("stage_max_new_edges")
        if not isinstance(stage_probabilities, dict):
            raise ValueError("modeling_priors edge_priors.serendipitous_edges.stage_probabilities must be a dict")
        if not isinstance(stage_caps, dict):
            raise ValueError("modeling_priors edge_priors.serendipitous_edges.stage_max_new_edges must be a dict")
        for stage in ("rising", "prime", "veteran", "legend", "retired"):
            if stage not in stage_probabilities:
                raise ValueError(
                    f"modeling_priors edge_priors.serendipitous_edges.stage_probabilities missing stage {stage}"
                )
            if stage not in stage_caps:
                raise ValueError(
                    f"modeling_priors edge_priors.serendipitous_edges.stage_max_new_edges missing stage {stage}"
                )
            try:
                float(stage_probabilities.get(stage))
            except Exception as exc:
                raise ValueError(
                    f"modeling_priors edge_priors.serendipitous_edges.stage_probabilities.{stage} must be numeric"
                ) from exc
            try:
                int(round(float(stage_caps.get(stage))))
            except Exception as exc:
                raise ValueError(
                    f"modeling_priors edge_priors.serendipitous_edges.stage_max_new_edges.{stage} must be numeric"
                ) from exc
        _require_dict_keys(
            value.get("triadic_closure", {}) if isinstance(value.get("triadic_closure"), dict) else {},
            "modeling_priors edge_priors.triadic_closure",
            ("stage_probabilities", "extra_cap", "weight_min", "weight_max"),
        )
        _require_dict_keys(
            value.get("calibration", {}) if isinstance(value.get("calibration"), dict) else {},
            "modeling_priors edge_priors.calibration",
            tuple(_DEFAULT_CALIBRATION_KEYS),
        )
        calibration = value.get("calibration", {}) if isinstance(value.get("calibration"), dict) else {}
        _require_numeric_keys(
            calibration,
            "modeling_priors edge_priors.calibration",
            (
                "candidate_sample_k",
                "director_actor_supplement",
                "upsert_weight_min",
                "upsert_weight_max",
                "preferential_attachment_log_weight",
                "best_friend_weight_base",
                "best_friend_weight_score_scale",
                "best_friend_weight_noise",
                "rival_weight_base",
                "rival_weight_score_scale",
                "rival_weight_noise",
                "director_stage_target_base",
                "director_stage_target_span",
                "director_preferred_weight_base",
                "director_preferred_weight_score_scale",
                "director_preferred_weight_noise",
                "director_avoid_weight_base",
                "director_avoid_weight_score_scale",
                "director_avoid_weight_noise",
                "bf_same_community_weight_floor",
                "bf_same_community_weight_boost",
            ),
        )
        relationship_targets = calibration.get("relationship_targets")
        if not isinstance(relationship_targets, dict):
            raise ValueError("modeling_priors edge_priors.calibration.relationship_targets must be a dict")
        _require_numeric_keys(
            relationship_targets,
            "modeling_priors edge_priors.calibration.relationship_targets",
            tuple(str(key) for key in RELATIONSHIP_TARGETS.keys()),
        )
        for block_name, block_keys in (
            ("friendship_score_weights", ("style", "genre", "stage")),
            ("rivalry_score_weights", ("genre", "style_distance", "stage")),
            ("director_preferred_score_weights", ("style", "genre", "stage")),
            ("director_avoid_score_weights", ("style_distance", "genre_distance", "controversy")),
        ):
            block = calibration.get(block_name)
            if not isinstance(block, dict):
                raise ValueError(f"modeling_priors edge_priors.calibration.{block_name} must be a dict")
            _require_numeric_keys(
                block,
                f"modeling_priors edge_priors.calibration.{block_name}",
                block_keys,
            )
        degree_caps = value.get("person_person_degree_caps")
        if not isinstance(degree_caps, dict) or not degree_caps:
            raise ValueError("modeling_priors edge_priors missing person_person_degree_caps")
        for stage in ("rising", "prime", "veteran", "legend", "retired"):
            row = degree_caps.get(stage)
            if not isinstance(row, dict):
                raise ValueError(f"modeling_priors edge_priors.person_person_degree_caps.{stage} must be an object")
            _require_numeric_keys(row, f"modeling_priors edge_priors.person_person_degree_caps.{stage}", ("mean", "std", "min", "max"))
    elif section == "scalable_edge_priors":
        required_keys = (
            "stage_priority",
            "collaboration_style_codes",
            "profile_overrides",
            "person_degree_caps",
            "brand_fit_ratio_by_stage",
            "employment_ratio_by_stage",
            "sampled_union_defaults",
            "valid_year_span",
            "year_validity_offsets",
        )
        if any(not isinstance(value.get(key), dict) or not value.get(key) for key in required_keys):
            raise ValueError("modeling_priors scalable_edge_priors missing scalable graph configs")
    elif section == "financial_priors":
        required_keys = (
            "country_budget_scale",
            "genre_rating_offset",
            "tier_rating_base",
            "tier_rating_std",
            "tier_log_center",
            "tier_min_votes",
            "market_regime",
            "year_quality",
            "budget_ranges_by_tier",
            "certification_distribution_by_genre",
            "company_profile_tiers",
            "company_profile_coefficients",
            "quality_latent_weights",
            "market_latent_weights",
            "performance_model",
            "vote_model",
            "runtime_model",
            "award_campaign_weights",
        )
        if any(not isinstance(value.get(key), dict) or not value.get(key) for key in required_keys):
            raise ValueError("modeling_priors financial_priors missing core finance configs")
        _require_float_map_keys(
            value.get("country_budget_scale"),
            "modeling_priors financial_priors.country_budget_scale",
            [str(key) for key in COUNTRY_BUDGET_SCALE.keys()],
        )
        _require_float_map_keys(
            value.get("genre_rating_offset"),
            "modeling_priors financial_priors.genre_rating_offset",
            [str(key) for key in GENRE_RATING_OFFSET.keys()],
        )
        _require_float_map_keys(
            value.get("tier_rating_base"),
            "modeling_priors financial_priors.tier_rating_base",
            [str(key) for key in TIER_RATING_BASE.keys()],
        )
        _require_float_map_keys(
            value.get("tier_rating_std"),
            "modeling_priors financial_priors.tier_rating_std",
            [str(key) for key in TIER_RATING_STD.keys()],
        )
        _require_float_map_keys(
            value.get("tier_log_center"),
            "modeling_priors financial_priors.tier_log_center",
            [str(key) for key in TIER_LOG_CENTER.keys()],
        )
        _require_float_map_keys(
            value.get("tier_min_votes"),
            "modeling_priors financial_priors.tier_min_votes",
            [str(key) for key in TIER_MIN_VOTES.keys()],
        )
        _require_float_map_keys(
            value.get("quality_latent_weights"),
            "modeling_priors financial_priors.quality_latent_weights",
            [str(key) for key in _DEFAULT_QUALITY_LATENT_WEIGHTS.keys()],
        )
        _require_float_map_keys(
            value.get("market_latent_weights"),
            "modeling_priors financial_priors.market_latent_weights",
            [str(key) for key in _DEFAULT_MARKET_LATENT_WEIGHTS.keys()],
        )
        _require_float_map_keys(
            value.get("performance_model"),
            "modeling_priors financial_priors.performance_model",
            [str(key) for key in _DEFAULT_PERFORMANCE_MODEL.keys()],
        )
        _require_float_map_keys(
            value.get("vote_model"),
            "modeling_priors financial_priors.vote_model",
            [str(key) for key in _DEFAULT_VOTE_MODEL.keys()],
        )
        _require_financial_budget_ranges(
            value.get("budget_ranges_by_tier"),
            "modeling_priors financial_priors.budget_ranges_by_tier",
        )
        _require_certification_distribution_map(
            value.get("certification_distribution_by_genre"),
            "modeling_priors financial_priors.certification_distribution_by_genre",
        )
        company_profile_tiers = value.get("company_profile_tiers")
        if not isinstance(company_profile_tiers, dict):
            raise ValueError("modeling_priors financial_priors.company_profile_tiers must be a dict")
        for tier_name in ("Global", "Major", "Mid-Budget", "Indie", "Micro"):
            row = company_profile_tiers.get(tier_name)
            if not isinstance(row, dict):
                raise ValueError(f"modeling_priors financial_priors.company_profile_tiers missing tier {tier_name}")
            _require_numeric_keys(
                row,
                f"modeling_priors financial_priors.company_profile_tiers.{tier_name}",
                ("capital", "margin", "debt", "slate", "buffer", "growth", "eff"),
            )
        coeffs = value.get("company_profile_coefficients")
        if not isinstance(coeffs, dict) or not coeffs:
            raise ValueError("modeling_priors financial_priors.company_profile_coefficients must be a non-empty dict")
        budget_focus_weights = coeffs.get("budget_focus_weights")
        if not isinstance(budget_focus_weights, list) or len(budget_focus_weights) != 5:
            raise ValueError("modeling_priors financial_priors.company_profile_coefficients missing budget_focus_weights")
        award_weights = value.get("award_campaign_weights")
        required_award_weight_keys = (
            "company_prestige",
            "director_ambition",
            "director_reputation",
            "cast_reputation",
            "company_focus",
            "q4_bonus",
            "prestige_genre_bonus",
            "regime_prestige_bias",
            "director_momentum",
            "company_momentum",
            "graph_synergy",
            "quality_signal",
            "slate_pressure_penalty",
            "controversy_penalty",
        )
        if not isinstance(award_weights, dict) or any(key not in award_weights for key in required_award_weight_keys):
            raise ValueError("modeling_priors financial_priors.award_campaign_weights missing calibrated award-campaign coefficients")
    elif section == "secondary_table_priors":
        from secondary_tables import (
            _DEFAULT_AWARD_PRIORS,
            _DEFAULT_DEMOGRAPHICS_PRIORS,
            _DEFAULT_PERSON_CONTRACT_PRIORS,
            _DEFAULT_PRODUCTION_TIMELINE_PRIORS,
            _DEFAULT_RELEASE_DATE_PRIORS,
            _DEFAULT_REVIEW_PRIORS,
            _DEFAULT_STREAMING_WINDOW_PRIORS,
            _DEFAULT_TERRITORY_BOX_OFFICE_PRIORS,
            _DEFAULT_TV_GENERATION_PRIORS,
        )

        required_blocks = {
            "demographics": _DEFAULT_DEMOGRAPHICS_PRIORS,
            "release_dates": _DEFAULT_RELEASE_DATE_PRIORS,
            "territory_box_office": _DEFAULT_TERRITORY_BOX_OFFICE_PRIORS,
            "reviews": _DEFAULT_REVIEW_PRIORS,
            "awards": _DEFAULT_AWARD_PRIORS,
            "tv_generation": _DEFAULT_TV_GENERATION_PRIORS,
            "production_timeline": _DEFAULT_PRODUCTION_TIMELINE_PRIORS,
            "streaming_windows": _DEFAULT_STREAMING_WINDOW_PRIORS,
            "person_contracts": _DEFAULT_PERSON_CONTRACT_PRIORS,
        }
        for block_name, defaults in required_blocks.items():
            block = value.get(block_name)
            if not isinstance(block, dict) or not block:
                raise ValueError(f"modeling_priors secondary_table_priors missing {block_name}")
            missing_keys = _missing_dict_paths(defaults, block, f"secondary_table_priors.{block_name}")
            if missing_keys:
                missing = ", ".join(missing_keys[:10])
                if len(missing_keys) > 10:
                    missing += ", ..."
                raise ValueError(
                    f"modeling_priors secondary_table_priors.{block_name} missing keys: {missing}"
                )
    elif section == "history_event_priors":
        specs = value.get("event_specs")
        if not isinstance(specs, list) or len(specs) < 2:
            raise ValueError("modeling_priors history_event_priors missing event_specs")
        for idx, row in enumerate(specs):
            if not isinstance(row, dict):
                raise ValueError(f"modeling_priors history_event_priors.event_specs[{idx}] must be an object")
            if not str(row.get("event_type", "") or "").strip():
                raise ValueError(f"modeling_priors history_event_priors.event_specs[{idx}] missing event_type")
            if not str(row.get("description", "") or "").strip():
                raise ValueError(f"modeling_priors history_event_priors.event_specs[{idx}] missing description")
            raw_range = row.get("year_range")
            raw_fraction = row.get("year_range_fraction")
            if isinstance(raw_range, (list, tuple)) and len(raw_range) == 2:
                pass
            elif isinstance(raw_fraction, (list, tuple)) and len(raw_fraction) == 2:
                pass
            else:
                try:
                    float(raw_fraction)
                except Exception as exc:
                    raise ValueError(
                        f"modeling_priors history_event_priors.event_specs[{idx}] missing usable year range"
                    ) from exc
    elif section == "rerank_priors":
        if len(value) < 2:
            raise ValueError("modeling_priors rerank_priors too shallow")


def _validate_modeling_priors_sections(payload: dict[str, Any], sections: tuple[str, ...] | None = None) -> None:
    target_sections = sections or _MODELING_PRIOR_REQUIRED_SECTIONS
    for key in target_sections:
        _validate_modeling_priors_section(key, payload.get(key, {}))


def _validate_modeling_priors(payload: dict[str, Any]) -> None:
    _validate_modeling_priors_sections(payload)


_VALIDATORS: dict[str, Callable[[dict[str, Any]], None]] = {
    "identity_bank": _validate_identity_bank,
    "character_identity_bank": _validate_character_identity_bank,
    "company_lexicon": _validate_company_lexicon,
    "keyword_seed_bank": _validate_keyword_seed_bank,
    "title_grammar_bank": _validate_title_grammar_bank,
    "temporal_regime_plan": _validate_temporal_regime_plan,
    "modeling_priors": _validate_modeling_priors,
}


_MODELING_PRIOR_REQUIRED_SECTIONS: tuple[str, ...] = (
    "person_generation",
    "company_generation",
    "keyword_generation",
    "character_generation",
    "title_generation",
    "company_finance_tiers",
    "selection_weights",
    "edge_priors",
    "scalable_edge_priors",
    "financial_priors",
    "secondary_table_priors",
    "history_event_priors",
    "rerank_priors",
)


_MODELING_PRIOR_SECTION_HINTS: dict[str, set[str]] = {
    "person_generation": {
        "gender_ratio", "age_distribution_mean", "age_distribution_std",
        "base_career_length_mean", "base_career_length_std", "role_mix",
        "nationality_mix", "career_stage_mix", "debut_curve", "retirement_curve",
    },
    "company_generation": {
        "tier_mix", "country_mix", "specialty_mix", "risk_profile",
        "company_name_style_mix", "catalog_density", "tier_weights",
    },
    "keyword_generation": {
        "scope_mix", "specificity_mix", "keyword_family_weights",
        "movie_keyword_count_mean", "cross_genre_keyword_share",
        "genre_target_weights", "generic_budget_ratio", "min_specific_story_share",
        "selection_bucket_targets",
    },
    "character_generation": {
        "archetype_mix", "nonliteral_name_share", "alias_frequency",
        "team_name_share", "character_role_mix", "slot_archetype_candidates",
        "general_archetypes", "unique_archetypes", "genre_archetype_candidates",
        "archetype_target_vectors", "career_stage_archetype_bias", "genre_archetype_bias",
        "collaboration_style_archetype_bias",
    },
    "title_generation": {
        "genre_base_weights", "prestige_genres", "franchise_genres",
        "experimental_genres", "low_cost_genres", "title_length_bias",
        "tagline_tone_weights", "allowed_tagline_placeholders", "tagline_render_constraints",
    },
    "company_finance_tiers": {
        "tiers", "budget_bands", "revenue_profiles", "cash_reserve_policy",
    },
    "selection_weights": {
        "cast_focus_exploration", "cast_base_focus_exploration",
        "cast_slot_exploration_empty", "cast_slot_exploration_filled",
        "cast_style_multiplier", "cast_community_match_multiplier",
        "director_exploration_share", "company_primary_exploration_share",
        "company_secondary_exploration_share", "crew_exploration_share",
        "keyword_selection", "director_selection", "cast_selection",
        "company_selection", "geo_boost_by_tier", "dynamic_cast_base_by_tier",
        "keyword_count_by_tier", "keyword_year_slate_family_boosts",
    },
    "edge_priors": {
        "cross_genre_candidate_k", "cross_genre_candidate_multiplier",
        "person_person_degree_caps", "person_person_classification",
        "person_company_generation", "company_company_generation",
        "serendipitous_edges", "triadic_closure", "calibration",
    },
    "scalable_edge_priors": {
        "stage_priority", "collaboration_style_codes", "profile_overrides",
        "brand_fit_ratio_by_stage", "employment_ratio_by_stage",
        "sampled_union_defaults", "year_validity_offsets", "person_degree_caps",
        "valid_year_span",
    },
    "financial_priors": {
        "country_budget_scale", "genre_rating_offset", "tier_rating_base",
        "tier_rating_std", "tier_log_center", "tier_min_votes",
        "budget_ranges_by_tier", "certification_distribution_by_genre",
        "company_profile_tiers", "company_profile_coefficients",
        "market_regime", "year_quality", "quality_latent_weights",
        "market_latent_weights", "performance_model", "vote_model",
        "runtime_model", "award_campaign_weights",
    },
    "secondary_table_priors": {
        "demographics", "release_dates", "territory_box_office", "reviews",
        "awards", "tv_generation", "production_timeline",
        "streaming_windows", "person_contracts",
    },
    "history_event_priors": {
        "event_specs", "event_families", "macro_event_mix",
    },
    "rerank_priors": {
        "rerank_budget", "keyword_rerank_budget", "critic_weight",
        "repair_threshold", "rerank_temperature",
    },
}


def _guess_modeling_priors_section(payload: dict[str, Any], sections: tuple[str, ...] | None = None) -> str:
    keys = {str(key) for key in payload.keys()}
    candidate_sections = sections or _MODELING_PRIOR_REQUIRED_SECTIONS
    best_section = candidate_sections[0]
    best_score = -1
    for section in candidate_sections:
        hints = _MODELING_PRIOR_SECTION_HINTS.get(section, set())
        score = len(keys & hints)
        if section.endswith("_priors"):
            score += sum(1 for key in keys if key.startswith(section.replace("_priors", "").split("_")[0]))
        if section == "person_generation":
            score += sum(1 for key in keys if "career" in key or "gender" in key or "age" in key)
        elif section == "company_generation":
            score += sum(1 for key in keys if key.startswith("company_"))
        elif section == "financial_priors":
            score += sum(1 for key in keys if "budget" in key or "rating" in key or "votes" in key)
        elif section == "secondary_table_priors":
            score += sum(1 for key in keys if key in {"reviews", "awards", "tv_generation"} or "release" in key)
        elif section == "history_event_priors":
            score += sum(1 for key in keys if "event" in key)
        if score > best_score:
            best_score = score
            best_section = section
    return best_section


def _normalize_secondary_table_priors(block: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(block, dict):
        return {}
    out = dict(block)

    demographics = out.get("demographics")
    if isinstance(demographics, dict):
        demo = dict(demographics)
        height = demo.get("height_by_gender")
        if isinstance(height, dict):
            height_map = dict(height)
            normalized_height: dict[str, Any] = {}
            alias_groups = {
                "F": ("F", "female", "women", "woman"),
                "NB": ("NB", "non_binary", "non-binary", "nb", "neutral"),
                "M": ("M", "male", "men", "man"),
            }
            for target_key, aliases in alias_groups.items():
                for alias in aliases:
                    if alias in height_map:
                        raw_value = height_map[alias]
                        if isinstance(raw_value, dict):
                            normalized_height[target_key] = raw_value
                        elif isinstance(raw_value, (list, tuple)) and len(raw_value) >= 2:
                            try:
                                lo = float(raw_value[0])
                                hi = float(raw_value[1])
                            except Exception:
                                normalized_height[target_key] = raw_value
                            else:
                                if hi < lo:
                                    lo, hi = hi, lo
                                spread = max(1.0, hi - lo)
                                normalized_height[target_key] = {
                                    "mean": round((lo + hi) / 2.0, 3),
                                    "std": round(max(spread / 6.0, 1.0), 3),
                                    "min": round(lo, 3),
                                    "max": round(hi, 3),
                                }
                        else:
                            normalized_height[target_key] = raw_value
                        break
            for key, value in height_map.items():
                if key not in {"F", "female", "women", "woman", "NB", "non_binary", "non-binary", "nb", "neutral", "M", "male", "men", "man"}:
                    normalized_height.setdefault(str(key), value)
            demo["height_by_gender"] = normalized_height
        age_ranges = demo.get("career_stage_age_ranges")
        if isinstance(age_ranges, dict):
            age_map = dict(age_ranges)
            normalized_ranges: dict[str, Any] = {}
            stage_aliases = {
                "rising": ("rising", "emerging", "debut"),
                "prime": ("prime", "established"),
                "veteran": ("veteran",),
                "legend": ("legend", "veteran"),
                "retired": ("retired", "legend", "veteran"),
            }
            for target_key, aliases in stage_aliases.items():
                for alias in aliases:
                    if alias in age_map:
                        normalized_ranges[target_key] = age_map[alias]
                        break
            for key, value in age_map.items():
                if key not in {"rising", "emerging", "debut", "prime", "established", "veteran", "legend", "retired"}:
                    normalized_ranges.setdefault(str(key), value)
            demo["career_stage_age_ranges"] = normalized_ranges
        out["demographics"] = demo

    contracts = out.get("person_contracts")
    if isinstance(contracts, dict):
        contract_block = dict(contracts)
        for field_name in ("salary_bands_by_stage", "contract_types_by_stage"):
            field = contract_block.get(field_name)
            if not isinstance(field, dict):
                continue
            field_map = dict(field)
            normalized_field: dict[str, Any] = {}
            stage_aliases = {
                "rising": ("rising", "emerging", "debut"),
                "prime": ("prime", "established"),
                "veteran": ("veteran",),
                "legend": ("legend", "veteran"),
                "retired": ("retired", "legend", "veteran"),
            }
            for target_key, aliases in stage_aliases.items():
                for alias in aliases:
                    if alias in field_map:
                        normalized_field[target_key] = field_map[alias]
                        break
            for key, value in field_map.items():
                if key not in {"rising", "emerging", "debut", "prime", "established", "veteran", "legend", "retired"}:
                    normalized_field.setdefault(str(key), value)
            contract_block[field_name] = normalized_field
        out["person_contracts"] = contract_block

    awards = out.get("awards")
    if isinstance(awards, dict):
        award_block = dict(awards)
        prestige = award_block.get("prestige_by_tier")
        if isinstance(prestige, dict):
            prestige_ranges = {
                "Epic": (0.16, 0.32, 0.24),
                "A": (0.12, 0.26, 0.18),
                "A-List": (0.12, 0.26, 0.18),
                "Mid": (0.04, 0.14, 0.08),
                "Mid-Budget": (0.04, 0.14, 0.08),
                "Indie": (0.05, 0.18, 0.11),
                "Micro": (0.0, 0.06, 0.02),
                "Micro-Budget": (0.0, 0.06, 0.02),
            }
            normalized_prestige: dict[str, float] = {}
            for tier_name, (lo_value, hi_value, default_value) in prestige_ranges.items():
                raw_value = prestige.get(tier_name, default_value)
                try:
                    normalized_prestige[tier_name] = float(np.clip(float(raw_value), float(lo_value), float(hi_value)))
                except Exception:
                    normalized_prestige[tier_name] = float(default_value)
            award_block["prestige_by_tier"] = normalized_prestige
        scalar_ranges = {
            "history_bonus_scale": (0.0, 0.06, 0.04),
            "history_bonus_cap": (0.02, 0.14, 0.10),
            "lambda_scale": (0.20, 0.90, 0.55),
            "lambda_cap": (0.8, 2.2, 1.6),
        }
        for key, (lo_value, hi_value, default_value) in scalar_ranges.items():
            raw_value = award_block.get(key, default_value)
            try:
                award_block[key] = float(np.clip(float(raw_value), float(lo_value), float(hi_value)))
            except Exception:
                award_block[key] = float(default_value)
        raw_max_nominations = award_block.get("max_nominations", 3)
        try:
            award_block["max_nominations"] = int(np.clip(int(round(float(raw_max_nominations))), 1, 4))
        except Exception:
            award_block["max_nominations"] = 3
        entry_probability = award_block.get("entry_probability")
        if isinstance(entry_probability, dict):
            entry_block = dict(entry_probability)
            entry_ranges = {
                "base": (0.0, 0.03, 0.0),
                "scale": (0.08, 0.26, 0.18),
                "min": (0.0, 0.05, 0.0),
                "max": (0.08, 0.36, 0.28),
            }
            for key, (lo_value, hi_value, default_value) in entry_ranges.items():
                raw_value = entry_block.get(key, default_value)
                try:
                    entry_block[key] = float(np.clip(float(raw_value), float(lo_value), float(hi_value)))
                except Exception:
                    entry_block[key] = float(default_value)
            entry_block["min"] = min(entry_block["min"], entry_block["max"])
            entry_block["base"] = min(entry_block["base"], entry_block["max"])
            award_block["entry_probability"] = entry_block
        won_probability = award_block.get("won_probability")
        if isinstance(won_probability, dict):
            won_block = dict(won_probability)
            won_ranges = {
                "base": (0.0, 0.04, 0.0),
                "scale": (0.02, 0.10, 0.06),
                "min": (0.0, 0.04, 0.0),
                "max": (0.06, 0.18, 0.16),
            }
            for key, (lo_value, hi_value, default_value) in won_ranges.items():
                raw_value = won_block.get(key, default_value)
                try:
                    won_block[key] = float(np.clip(float(raw_value), float(lo_value), float(hi_value)))
                except Exception:
                    won_block[key] = float(default_value)
            won_block["min"] = min(won_block["min"], won_block["max"])
            won_block["base"] = min(won_block["base"], won_block["max"])
            award_block["won_probability"] = won_block
        out["awards"] = award_block

    return out


def _normalize_edge_priors(block: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(block, dict):
        return {}
    out = dict(block)

    serendipitous = out.get("serendipitous_edges")
    if isinstance(serendipitous, dict):
        serendipitous_block = dict(serendipitous)
        stage_caps = serendipitous_block.get("stage_max_new_edges")
        if not isinstance(stage_caps, dict):
            try:
                scalar_cap = int(round(float(stage_caps)))
            except Exception:
                scalar_cap = None
            if scalar_cap is not None:
                scalar_cap = max(0, min(20, scalar_cap))
                baseline_caps = {
                    "legend": 3,
                    "veteran": 2,
                    "prime": 1,
                    "rising": 1,
                    "retired": 1,
                }
                serendipitous_block["stage_max_new_edges"] = {
                    stage: min(default_cap, scalar_cap) if scalar_cap > 0 else 0
                    for stage, default_cap in baseline_caps.items()
                }
        out["serendipitous_edges"] = serendipitous_block

    triadic = out.get("triadic_closure")
    if isinstance(triadic, dict):
        triadic_block = dict(triadic)
        extra_cap = triadic_block.get("extra_cap")
        if isinstance(extra_cap, dict):
            numeric_values: list[int] = []
            for value in extra_cap.values():
                try:
                    numeric_values.append(int(round(float(value))))
                except Exception:
                    continue
            if numeric_values:
                # Runtime consumes a single integer cap, so collapse stage-local maps
                # into one permissive scalar instead of failing later in step 80.
                triadic_block["extra_cap"] = max(0, min(100, max(numeric_values)))
        out["triadic_closure"] = triadic_block

    calibration = out.get("calibration")
    if isinstance(calibration, dict):
        calibration_block = dict(calibration)
        relationship_defaults = {str(key): float(value) for key, value in RELATIONSHIP_TARGETS.items()}
        raw_targets = calibration_block.get("relationship_targets")

        def _coerce_float(raw_value: Any, default: float, *, lo: float, hi: float) -> float:
            try:
                value = float(raw_value)
            except Exception:
                value = float(default)
            return min(float(hi), max(float(lo), value))

        def _coerce_sparse_target(
            raw_value: Any,
            default: float,
            *,
            legacy_scale: float,
            cap: float,
        ) -> float:
            try:
                value = float(raw_value)
            except Exception:
                value = float(default)
            if value <= 0:
                return 0.0
            if value > 1.0:
                # Legacy prompts sometimes emitted literal per-director counts.
                # Convert those back into sparse activation probabilities so
                # step 80 stays in the validated regime instead of exploding.
                value *= float(legacy_scale)
            return min(float(cap), max(0.0, value))

        normalized_targets = dict(relationship_defaults)
        if isinstance(raw_targets, dict):
            normalized_targets["best_friend_rate"] = _coerce_float(
                raw_targets.get("best_friend_rate"),
                relationship_defaults["best_friend_rate"],
                lo=0.0,
                hi=0.35,
            )
            normalized_targets["rival_rate"] = _coerce_float(
                raw_targets.get("rival_rate"),
                relationship_defaults["rival_rate"],
                lo=0.0,
                hi=0.20,
            )
            normalized_targets["bf_same_community_rate"] = _coerce_float(
                raw_targets.get("bf_same_community_rate"),
                relationship_defaults["bf_same_community_rate"],
                lo=0.45,
                hi=0.95,
            )
            normalized_targets["director_preferred_actors"] = _coerce_sparse_target(
                raw_targets.get("director_preferred_actors"),
                relationship_defaults["director_preferred_actors"],
                legacy_scale=0.06,
                cap=0.35,
            )
            normalized_targets["director_avoided_actors"] = _coerce_sparse_target(
                raw_targets.get("director_avoided_actors"),
                relationship_defaults["director_avoided_actors"],
                legacy_scale=0.03,
                cap=0.12,
            )
        calibration_block["relationship_targets"] = normalized_targets
        calibration_block["director_stage_target_base"] = _coerce_float(
            calibration_block.get("director_stage_target_base"),
            0.80,
            lo=0.50,
            hi=1.25,
        )
        calibration_block["director_stage_target_span"] = _coerce_float(
            calibration_block.get("director_stage_target_span"),
            0.40,
            lo=0.0,
            hi=0.75,
        )
        out["calibration"] = calibration_block

    return out


_KEYWORD_SELECTION_EXACT_FLOORS = {"Epic": 3.0, "A": 3.0, "Mid": 2.0, "Indie": 2.0, "Micro": 1.0}
_KEYWORD_SELECTION_PRIMARY_RELATED_FLOORS = {"Epic": 5.0, "A": 5.0, "Mid": 4.0, "Indie": 3.0, "Micro": 2.0}
_KEYWORD_SELECTION_GENERIC_CAPS = {"Epic": 1.0, "A": 1.0, "Mid": 1.0, "Indie": 1.0, "Micro": 1.0}
_KEYWORD_SELECTION_OFF_GENRE_CAPS = {"Epic": 1.0, "A": 1.0, "Mid": 1.0, "Indie": 1.0, "Micro": 1.0}
_KEYWORD_SELECTION_SLOT_EXACT_FLOORS = {"Epic": 0.48, "A": 0.48, "Mid": 0.42, "Indie": 0.40, "Micro": 0.36}
_KEYWORD_SELECTION_SLOT_GENERIC_CAPS = {"Epic": 0.06, "A": 0.06, "Mid": 0.08, "Indie": 0.08, "Micro": 0.10}
_KEYWORD_SELECTION_SLOT_RELATED_CAPS = {"Epic": 0.14, "A": 0.14, "Mid": 0.16, "Indie": 0.18, "Micro": 0.20}
_KEYWORD_SELECTION_SLOT_FRANCHISE_CAPS = {"Epic": 0.10, "A": 0.10, "Mid": 0.10, "Indie": 0.05, "Micro": 0.00}


def _normalize_selection_weights_priors(block: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(block, dict):
        return {}
    out = dict(block)
    keyword_selection = out.get("keyword_selection")
    if not isinstance(keyword_selection, dict):
        return out

    keyword_block = dict(keyword_selection)

    def _coerce_tier_map(raw: Any, defaults: dict[str, float], *, minimum: bool) -> dict[str, float]:
        out_map: dict[str, float] = {}
        raw_map = dict(raw) if isinstance(raw, dict) else {}
        for tier_name, default_value in defaults.items():
            raw_value = raw_map.get(tier_name, default_value)
            try:
                value = float(raw_value)
            except Exception:
                value = float(default_value)
            out_map[tier_name] = max(float(default_value), value) if minimum else min(float(default_value), value)
        return out_map

    keyword_block["exact_topic_min_count_by_tier"] = _coerce_tier_map(
        keyword_block.get("exact_topic_min_count_by_tier"),
        _KEYWORD_SELECTION_EXACT_FLOORS,
        minimum=True,
    )
    keyword_block["primary_plus_related_min_count_by_tier"] = _coerce_tier_map(
        keyword_block.get("primary_plus_related_min_count_by_tier"),
        _KEYWORD_SELECTION_PRIMARY_RELATED_FLOORS,
        minimum=True,
    )
    keyword_block["generic_keyword_cap_by_tier"] = _coerce_tier_map(
        keyword_block.get("generic_keyword_cap_by_tier"),
        _KEYWORD_SELECTION_GENERIC_CAPS,
        minimum=False,
    )
    keyword_block["off_genre_cap_by_tier"] = _coerce_tier_map(
        keyword_block.get("off_genre_cap_by_tier"),
        _KEYWORD_SELECTION_OFF_GENRE_CAPS,
        minimum=False,
    )

    raw_slot_mix = keyword_block.get("slot_mix_by_tier")
    if isinstance(raw_slot_mix, dict):
        normalized_slot_mix: dict[str, dict[str, float]] = {}
        for tier_name in PRODUCTION_TIERS:
            row = dict(raw_slot_mix.get(tier_name) or {}) if isinstance(raw_slot_mix.get(tier_name), dict) else {}
            exact_anchor = max(_KEYWORD_SELECTION_SLOT_EXACT_FLOORS[tier_name], float(row.get("exact_anchor", 0.0) or 0.0))
            related_support = min(_KEYWORD_SELECTION_SLOT_RELATED_CAPS[tier_name], max(0.0, float(row.get("related_support", 0.0) or 0.0)))
            generic = min(_KEYWORD_SELECTION_SLOT_GENERIC_CAPS[tier_name], max(0.0, float(row.get("generic", 0.0) or 0.0)))
            franchise = min(_KEYWORD_SELECTION_SLOT_FRANCHISE_CAPS[tier_name], max(0.0, float(row.get("franchise", 0.0) or 0.0)))
            remainder = max(0.0, 1.0 - exact_anchor - related_support - generic - franchise)
            story_specific = remainder
            normalized_slot_mix[tier_name] = {
                "exact_anchor": exact_anchor,
                "related_support": related_support,
                "story_specific": story_specific,
                "franchise": franchise,
                "generic": generic,
            }
        keyword_block["slot_mix_by_tier"] = normalized_slot_mix

    out["keyword_selection"] = keyword_block
    return out


def _normalize_financial_priors(block: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(block, dict):
        return {}
    out = dict(block)

    def _clone_jsonish(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): _clone_jsonish(child) for key, child in value.items()}
        if isinstance(value, (list, tuple)):
            return [_clone_jsonish(child) for child in value]
        return value

    def _merge_nested_defaults(defaults: dict[str, Any], raw: Any) -> dict[str, Any]:
        merged = _clone_jsonish(defaults)
        if not isinstance(raw, dict):
            return merged
        for key, value in raw.items():
            str_key = str(key)
            if isinstance(value, dict) and isinstance(merged.get(str_key), dict):
                merged[str_key] = _merge_nested_defaults(merged[str_key], value)
            elif isinstance(value, (list, tuple)):
                merged[str_key] = [_clone_jsonish(child) for child in value]
            else:
                merged[str_key] = value
        return merged

    def _merge_float_map(
        defaults: dict[str, float] | dict[str, int],
        raw: Any,
        *,
        key_fn: Callable[[Any], str | None] | None = None,
        keep_unknown: bool = True,
    ) -> dict[str, float]:
        merged = {str(key): float(value) for key, value in defaults.items()}
        if not isinstance(raw, dict):
            return merged
        for raw_key, raw_value in raw.items():
            key = key_fn(raw_key) if key_fn is not None else str(raw_key or "").strip()
            if not key:
                continue
            if not keep_unknown and str(key) not in merged:
                continue
            try:
                merged[str(key)] = float(raw_value)
            except Exception:
                continue
        return merged

    def _canon_genre_key(raw_key: Any) -> str | None:
        key = str(raw_key or "").strip()
        if not key:
            return None
        if key == "default":
            return "default"
        normalized = re.sub(r"[^a-z0-9]+", "", key.casefold())
        alias_map = {
            re.sub(r"[^a-z0-9]+", "", str(genre).casefold()): str(genre)
            for genre in GENRES
        }
        alias_map.update(
            {
                "sciencefiction": "Sci-Fi",
                "scifi": "Sci-Fi",
                "sci fi": "Sci-Fi",
                "martialarts": "Martial Arts",
                "realitytv": "Reality-TV",
                "filmnoir": "Film-Noir",
                "superherofilm": "Superhero",
            }
        )
        return alias_map.get(normalized)

    def _canon_cert_key(raw_key: Any) -> str | None:
        key = str(raw_key or "").strip()
        if not key:
            return None
        normalized = re.sub(r"[^a-z0-9]+", "", key.casefold())
        alias_map = {
            "g": "G",
            "pg": "PG",
            "pg13": "PG-13",
            "r": "R",
            "nr": "NR",
            "notrated": "NR",
            "unrated": "NR",
            "nongrated": "NR",
        }
        return alias_map.get(normalized)

    out["country_budget_scale"] = _merge_float_map(
        {str(key): float(value) for key, value in COUNTRY_BUDGET_SCALE.items()},
        out.get("country_budget_scale"),
        keep_unknown=True,
    )
    out["genre_rating_offset"] = _merge_float_map(
        {str(key): float(value) for key, value in GENRE_RATING_OFFSET.items()},
        out.get("genre_rating_offset"),
        key_fn=_canon_genre_key,
        keep_unknown=False,
    )
    out["tier_rating_base"] = _merge_float_map(
        {str(key): float(value) for key, value in TIER_RATING_BASE.items()},
        out.get("tier_rating_base"),
        keep_unknown=False,
    )
    out["tier_rating_std"] = _merge_float_map(
        {str(key): float(value) for key, value in TIER_RATING_STD.items()},
        out.get("tier_rating_std"),
        keep_unknown=False,
    )
    out["tier_log_center"] = _merge_float_map(
        {str(key): float(value) for key, value in TIER_LOG_CENTER.items()},
        out.get("tier_log_center"),
        keep_unknown=False,
    )
    out["tier_min_votes"] = _merge_float_map(
        {str(key): float(value) for key, value in TIER_MIN_VOTES.items()},
        out.get("tier_min_votes"),
        keep_unknown=False,
    )

    cert_dist = out.get("certification_distribution_by_genre")
    normalized_cert_dist: dict[str, dict[str, float]] = {
        str(genre): {str(cert): float(score) for cert, score in CERT_DISTS[str(genre)].items()}
        for genre in GENRES
    }
    if isinstance(cert_dist, dict):
        for raw_genre, raw_row in cert_dist.items():
            genre_key = _canon_genre_key(raw_genre)
            if genre_key is None or not isinstance(raw_row, dict):
                continue
            row_out: dict[str, float] = {}
            for raw_cert, raw_value in raw_row.items():
                cert_key = _canon_cert_key(raw_cert)
                if cert_key is None:
                    continue
                try:
                    score = float(raw_value)
                except Exception:
                    continue
                if score > 0:
                    row_out[cert_key] = score
            if row_out:
                existing = dict(normalized_cert_dist.get(genre_key, {}))
                for cert_key, score in row_out.items():
                    existing[cert_key] = float(existing.get(cert_key, 0.0)) + float(score)
                normalized_cert_dist[genre_key] = existing
    out["certification_distribution_by_genre"] = normalized_cert_dist

    budget_ranges = out.get("budget_ranges_by_tier")
    normalized_ranges: dict[str, dict[str, float]] = {
        str(tier): {"min_budget": float(row[0]), "max_budget": float(row[1])}
        for tier, row in BUDGET_RANGES.items()
    }
    if isinstance(budget_ranges, dict):
        for raw_tier, raw_row in budget_ranges.items():
            tier_key = str(raw_tier or "").strip()
            if tier_key not in PRODUCTION_TIERS:
                continue
            row_out: dict[str, float] = {}
            if isinstance(raw_row, dict):
                alias_map = {
                    "min": "min_budget",
                    "minimum": "min_budget",
                    "minbudget": "min_budget",
                    "max": "max_budget",
                    "maximum": "max_budget",
                    "maxbudget": "max_budget",
                }
                for raw_key, raw_value in raw_row.items():
                    key_norm = re.sub(r"[^a-z0-9]+", "", str(raw_key or "").casefold())
                    key = alias_map.get(key_norm, str(raw_key or "").strip())
                    try:
                        row_out[key] = float(raw_value)
                    except Exception:
                        continue
            elif isinstance(raw_row, (list, tuple)) and len(raw_row) >= 2:
                try:
                    row_out["min_budget"] = float(raw_row[0])
                    row_out["max_budget"] = float(raw_row[1])
                except Exception:
                    row_out = {}
            if (
                "min_budget" in row_out
                and "max_budget" in row_out
                and row_out["min_budget"] > 0
                and row_out["max_budget"] > row_out["min_budget"]
            ):
                if row_out["max_budget"] < 10_000:
                    row_out["min_budget"] *= 1_000_000.0
                    row_out["max_budget"] *= 1_000_000.0
                normalized_ranges[tier_key] = row_out
    out["budget_ranges_by_tier"] = normalized_ranges

    regime_block = _merge_nested_defaults(_DEFAULT_MARKET_REGIME, out.get("market_regime"))
    for key, floor_value in (
        ("prestige_bias_base", 0.56),
        ("prestige_cycle_weight", 0.24),
        ("prestige_bias_min", 0.08),
    ):
        raw_value = regime_block.get(key, floor_value)
        try:
            regime_block[key] = max(float(floor_value), float(raw_value))
        except Exception:
            regime_block[key] = float(floor_value)
    label_thresholds = regime_block.get("label_thresholds")
    if isinstance(label_thresholds, dict):
        thresholds = dict(label_thresholds)
        raw_value = thresholds.get("prestige_boom", 0.62)
        try:
            thresholds["prestige_boom"] = max(0.62, float(raw_value))
        except Exception:
            thresholds["prestige_boom"] = 0.62
        regime_block["label_thresholds"] = thresholds
    out["market_regime"] = regime_block
    out["year_quality"] = _merge_nested_defaults(_DEFAULT_YEAR_QUALITY, out.get("year_quality"))
    out["quality_latent_weights"] = _merge_float_map(
        {str(key): float(value) for key, value in _DEFAULT_QUALITY_LATENT_WEIGHTS.items()},
        out.get("quality_latent_weights"),
        keep_unknown=True,
    )
    out["market_latent_weights"] = _merge_float_map(
        {str(key): float(value) for key, value in _DEFAULT_MARKET_LATENT_WEIGHTS.items()},
        out.get("market_latent_weights"),
        keep_unknown=True,
    )
    out["performance_model"] = _merge_float_map(
        {str(key): float(value) for key, value in _DEFAULT_PERFORMANCE_MODEL.items()},
        out.get("performance_model"),
        keep_unknown=True,
    )
    out["vote_model"] = _merge_float_map(
        {str(key): float(value) for key, value in _DEFAULT_VOTE_MODEL.items()},
        out.get("vote_model"),
        keep_unknown=True,
    )
    out["runtime_model"] = _merge_nested_defaults(_DEFAULT_RUNTIME_MODEL, out.get("runtime_model"))

    award_weights = out.get("award_campaign_weights")
    weight_floors = {
        "company_prestige": 0.28,
        "director_ambition": 0.24,
        "director_reputation": 0.16,
        "cast_reputation": 0.09,
        "company_focus": 0.07,
        "q4_bonus": 0.22,
        "prestige_genre_bonus": 0.20,
        "regime_prestige_bias": 0.14,
        "director_momentum": 0.05,
        "company_momentum": 0.03,
        "graph_synergy": 0.09,
        "quality_signal": 0.16,
    }
    penalty_bounds = {
        "slate_pressure_penalty": (0.08, 0.16),
        "controversy_penalty": (0.06, 0.14),
    }
    raw_award = award_weights if isinstance(award_weights, dict) else {}
    normalized_weights = _merge_float_map(
        {str(key): float(value) for key, value in _DEFAULT_AWARD_CAMPAIGN_WEIGHTS.items()},
        raw_award,
        keep_unknown=False,
    )
    for key, floor_value in weight_floors.items():
        try:
            normalized_weights[key] = max(float(floor_value), float(normalized_weights.get(key, floor_value)))
        except Exception:
            normalized_weights[key] = float(floor_value)
    for key, (lo, hi) in penalty_bounds.items():
        try:
            normalized_weights[key] = min(float(hi), max(float(lo), float(normalized_weights.get(key, lo))))
        except Exception:
            normalized_weights[key] = float(lo)
    out["award_campaign_weights"] = normalized_weights

    return out


def _normalize_history_event_priors(block: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(block, dict):
        return {}
    out = dict(block)
    raw_specs = out.get("event_specs")
    if isinstance(raw_specs, list):
        normalized_specs: list[dict[str, Any]] = []
        for row in raw_specs:
            if not isinstance(row, dict):
                continue
            spec = dict(row)
            raw_range = spec.get("year_range")
            if not (isinstance(raw_range, (list, tuple)) and len(raw_range) == 2):
                raw_fraction = spec.get("year_range_fraction")
                if isinstance(raw_fraction, (list, tuple)) and len(raw_fraction) == 2:
                    spec["year_range"] = [float(raw_fraction[0]), float(raw_fraction[1])]
                else:
                    try:
                        frac = float(raw_fraction)
                    except Exception:
                        frac = None
                    if frac is not None:
                        frac = max(0.0, min(1.0, frac))
                        spec["year_range"] = [0.0, frac]
            normalized_specs.append(spec)
        out["event_specs"] = normalized_specs
    return out


def _normalize_modeling_priors_payload(payload: dict[str, Any], sections: tuple[str, ...] | None = None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    target_sections = sections or _MODELING_PRIOR_REQUIRED_SECTIONS
    working = dict(payload)
    for wrapper_key in ("modeling_priors", "priors", "sections", "payload", "data"):
        nested = working.get(wrapper_key)
        if isinstance(nested, dict):
            working = dict(nested)
            break

    normalized: dict[str, Any] = {}
    stray: dict[str, Any] = {}
    for key, value in working.items():
        if key in target_sections and isinstance(value, dict):
            normalized[str(key)] = dict(value)
        elif key != "meta":
            stray[str(key)] = value

    if stray:
        guessed = _guess_modeling_priors_section(stray, target_sections)
        bucket = dict(normalized.get(guessed, {}))
        bucket.update(stray)
        normalized[guessed] = bucket

    for section in target_sections:
        value = normalized.get(section)
        normalized[section] = dict(value) if isinstance(value, dict) else {}

    if isinstance(normalized.get("secondary_table_priors"), dict):
        normalized["secondary_table_priors"] = _normalize_secondary_table_priors(normalized["secondary_table_priors"])
    if isinstance(normalized.get("edge_priors"), dict):
        normalized["edge_priors"] = _normalize_edge_priors(normalized["edge_priors"])
    if isinstance(normalized.get("selection_weights"), dict):
        normalized["selection_weights"] = _normalize_selection_weights_priors(normalized["selection_weights"])
    if isinstance(normalized.get("financial_priors"), dict):
        normalized["financial_priors"] = _normalize_financial_priors(normalized["financial_priors"])
    if isinstance(normalized.get("character_generation"), dict):
        normalized["character_generation"] = _normalize_character_generation_priors(normalized["character_generation"])
    if isinstance(normalized.get("history_event_priors"), dict):
        normalized["history_event_priors"] = _normalize_history_event_priors(normalized["history_event_priors"])

    if isinstance(payload.get("meta"), dict):
        normalized["meta"] = dict(payload["meta"])
    return normalized


def _coerce_artifact_payload(artifact_name: str, parsed: Any) -> dict[str, Any] | None:
    if artifact_name == "keyword_seed_bank":
        payload = _coerce_keyword_seed_bank_payload(parsed)
        if isinstance(payload, dict):
            return payload
    if isinstance(parsed, dict):
        if artifact_name == "identity_bank":
            return _normalize_identity_bank_payload(dict(parsed))
        nested = parsed.get(str(artifact_name))
        if isinstance(nested, dict):
            if artifact_name == "identity_bank":
                return _normalize_identity_bank_payload(nested)
            if artifact_name == "character_identity_bank":
                return _normalize_character_identity_bank_payload(nested)
            if artifact_name == "company_lexicon":
                return _normalize_company_lexicon_payload(nested)
            return nested
        if artifact_name == "character_identity_bank":
            return _normalize_character_identity_bank_payload(dict(parsed))
        if artifact_name == "company_lexicon":
            return _normalize_company_lexicon_payload(dict(parsed))
        if artifact_name == "temporal_regime_plan":
            payload = dict(parsed)
            year_weights = payload.get("year_weights")
            if isinstance(year_weights, dict):
                rows: list[dict[str, Any]] = []
                for key, value in year_weights.items():
                    try:
                        rows.append({"year": int(key), "weight": float(value)})
                    except Exception:
                        continue
                payload["year_weights"] = rows
            return payload
        if artifact_name == "modeling_priors":
            return _normalize_modeling_priors_payload(dict(parsed))
        return parsed
    if isinstance(parsed, list):
        if artifact_name == "identity_bank":
            rows = [row for row in parsed if isinstance(row, dict)]
            if rows:
                return {"families": rows}
        if len(parsed) == 1 and isinstance(parsed[0], dict):
            if artifact_name == "company_lexicon":
                return _normalize_company_lexicon_payload(parsed[0])
            if artifact_name == "modeling_priors":
                return _normalize_modeling_priors_payload(parsed[0])
            return parsed[0]
        if artifact_name == "company_lexicon":
            return _normalize_company_lexicon_payload(parsed)
    return None


def _coerce_modeling_priors_group_payload(parsed: Any, sections: tuple[str, ...]) -> dict[str, Any] | None:
    if isinstance(parsed, (dict, list)):
        payload = _coerce_artifact_payload("modeling_priors", parsed)
        if isinstance(payload, dict):
            return _normalize_modeling_priors_payload(payload, sections)
    return None


def _deep_merge(base: Any, overlay: Any) -> Any:
    if isinstance(base, dict) and isinstance(overlay, dict):
        merged = {str(k): v for k, v in base.items()}
        for key, value in overlay.items():
            key = str(key)
            if key in merged:
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged
    if isinstance(overlay, list) and overlay:
        return overlay
    if overlay not in (None, {}, []):
        return overlay
    return base


def _modeling_priors_richness_score(payload: dict[str, Any], sections: tuple[str, ...] | None = None) -> int:
    target_sections = sections or _MODELING_PRIOR_REQUIRED_SECTIONS
    score = 0
    for section in target_sections:
        value = payload.get(section)
        if not isinstance(value, dict) or not value:
            continue
        score += 10
        score += len(value)
        for nested_value in value.values():
            if isinstance(nested_value, dict):
                score += len(nested_value)
            elif isinstance(nested_value, list):
                score += min(len(nested_value), 8)
    return score


def _prune_nested_modeling_sections(payload: dict[str, Any], sections: tuple[str, ...] | None = None) -> dict[str, Any]:
    target_sections = sections or _MODELING_PRIOR_REQUIRED_SECTIONS
    cleaned: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "meta":
            cleaned[key] = value
            continue
        if isinstance(value, dict):
            section_clean = {}
            for inner_key, inner_value in value.items():
                if inner_key in target_sections:
                    continue
                section_clean[inner_key] = inner_value
            cleaned[key] = section_clean
        else:
            cleaned[key] = value
    return cleaned


def _artifact_debug_dir(base_dir: Path) -> Path:
    path = base_dir / "_dev" / "bootstrap_artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_failed_artifact_debug(base_dir: Path, artifact_name: str, response_text: str, parsed: Any = None) -> None:
    debug_dir = _artifact_debug_dir(base_dir)
    (debug_dir / f"{artifact_name}_failed_raw.txt").write_text(str(response_text or ""), encoding="utf-8")
    if parsed is not None:
        try:
            (debug_dir / f"{artifact_name}_failed_parsed.json").write_text(
                json.dumps(parsed, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass


def _repair_prompt(artifact_name: str, response_text: str) -> str:
    return f"""Convert the following model output into one strict JSON object for the artifact `{artifact_name}`.

Rules:
- Return JSON object only.
- No markdown fences.
- Preserve as much valid content as possible.
- If the content is a wrapped list, unwrap or merge it into one object.
- Do not add explanations.

Original output:
{response_text}
"""


def _validation_repair_prompt(
    artifact_name: str,
    original_prompt: str,
    validation_error: Exception,
    parsed_payload: dict[str, Any],
) -> str:
    modeling_hint = ""
    if artifact_name.startswith("modeling_priors"):
        if artifact_name == "modeling_priors":
            target_sections = _MODELING_PRIOR_REQUIRED_SECTIONS
        else:
            group_name = artifact_name.split("__", 1)[-1]
            target_sections = next(
                (tuple(group["sections"]) for group in _MODELING_PRIOR_GROUPS if str(group["name"]) == group_name),
                _MODELING_PRIOR_REQUIRED_SECTIONS,
            )
        modeling_hint = (
            "\nRequired top-level sections:\n"
            + "\n".join(f"- {name}" for name in target_sections)
            + "\nIf the current JSON looks like only one section, preserve it under the best-matching required section"
              " and still return every required top-level section as a populated JSON object.\n"
            "Do not return empty objects for the requested sections.\n"
        )
    return f"""Repair the JSON artifact `{artifact_name}` so it satisfies the original schema requirements.

Rules:
- Return one JSON object only.
- No markdown fences.
- Keep as much valid content as possible.
- Preserve all strong existing entries.
- Add missing required structure instead of deleting content.
{modeling_hint}

Original generation brief:
{original_prompt}

Validation error:
{validation_error}

Current parsed JSON:
{json.dumps(parsed_payload, indent=2, ensure_ascii=False)}
"""


def _validation_missing_field_map(validation_error: Exception) -> dict[str, list[str]]:
    text = str(validation_error or "")
    out: dict[str, list[str]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.search(r"modeling_priors ([A-Za-z0-9_\.]+) missing(?: numeric)? keys: (.+)", line)
        if not match:
            continue
        path = str(match.group(1)).strip()
        keys = [part.strip() for part in str(match.group(2)).split(",") if part.strip()]
        if not keys:
            continue
        bucket = out.setdefault(path, [])
        for key in keys:
            if key not in bucket:
                bucket.append(key)
    return out


def _validation_completion_skeleton(sections: tuple[str, ...], missing: dict[str, list[str]]) -> dict[str, Any]:
    root: dict[str, Any] = {str(section): {} for section in sections}
    for dotted_path, keys in missing.items():
        parts = str(dotted_path).split(".")
        if not parts or parts[0] not in root:
            continue
        node: dict[str, Any] = root
        for part in parts:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        for key in keys:
            node.setdefault(str(key), 0.0)
    return root


def _shape_only_skeleton(node: Any) -> Any:
    if isinstance(node, dict):
        return {str(key): _shape_only_skeleton(value) for key, value in node.items()}
    if isinstance(node, (list, tuple)):
        if not node:
            return []
        sample = node[0]
        if isinstance(sample, dict):
            return [_shape_only_skeleton(sample)]
        if isinstance(sample, (int, float)):
            return [0.0, 1.0] if len(node) >= 2 else [0.0]
        return ["value"]
    if isinstance(node, bool):
        return False
    if isinstance(node, int):
        return 0
    if isinstance(node, float):
        return 0.0
    return "value"


def _runtime_schema_completion_skeleton(sections: tuple[str, ...]) -> dict[str, Any]:
    root: dict[str, Any] = {}
    if "person_generation" in sections:
        root["person_generation"] = {
            "gender_ratio": {
                "male": 0.52,
                "female": 0.45,
                "non_binary": 0.03,
            },
            "age_distribution_mean": 42.5,
            "age_distribution_std": 14.2,
            "career_stage_weights": {
                "rising": 0.25,
                "prime": 0.40,
                "veteran": 0.25,
                "legend": 0.08,
                "retired": 0.02,
            },
            "base_popularity_mean": 45.0,
            "base_popularity_std": 18.5,
        }
    if "company_generation" in sections:
        root["company_generation"] = {
            "type_distribution": {
                "production": 0.60,
                "distribution": 0.20,
                "special_effects": 0.10,
                "financing": 0.10,
            },
            "longevity_mean": 25.0,
            "longevity_std": 15.0,
            "country_mix": {
                "US": 0.40,
                "UK": 0.10,
                "FR": 0.10,
                "JP": 0.10,
                "IN": 0.10,
                "Other": 0.20,
            },
            "tier_weights": {
                "Global": 0.05,
                "Major": 0.15,
                "Mid-Budget": 0.30,
                "Indie": 0.35,
                "Micro": 0.15,
            },
        }
    if "title_generation" in sections:
        root["title_generation"] = {
            "genre_base_weights": {str(genre): 0.0 for genre in GENRES},
            "prestige_genres": [str(GENRES[0])],
            "franchise_genres": [str(GENRES[1])],
            "experimental_genres": [str(GENRES[2])],
            "low_cost_genres": [str(GENRES[3])],
            "allowed_tagline_placeholders": ["adjective", "noun", "abstract", "location", "celestial", "technology", "mythic", "action", "franchise_affix"],
            "tagline_render_constraints": {
                "min_words": 4,
                "max_words": 10,
                "max_placeholder_count": 1,
                "forbid_square_brackets": True,
                "allow_unresolved_placeholders": False,
            },
        }
    if "financial_priors" in sections:
        root["financial_priors"] = {
            "country_budget_scale": {
                str(key): float(value)
                for key, value in COUNTRY_BUDGET_SCALE.items()
            },
            "genre_rating_offset": {
                str(key): float(value)
                for key, value in GENRE_RATING_OFFSET.items()
            },
            "tier_rating_base": {
                str(key): float(value)
                for key, value in TIER_RATING_BASE.items()
            },
            "tier_rating_std": {
                str(key): float(value)
                for key, value in TIER_RATING_STD.items()
            },
            "tier_log_center": {
                str(key): float(value)
                for key, value in TIER_LOG_CENTER.items()
            },
            "tier_min_votes": {
                str(key): float(value)
                for key, value in TIER_MIN_VOTES.items()
            },
            "market_regime": {
                str(key): value
                for key, value in _DEFAULT_MARKET_REGIME.items()
            },
            "year_quality": {
                str(key): value
                for key, value in _DEFAULT_YEAR_QUALITY.items()
            },
            "budget_ranges_by_tier": {
                str(tier): {"min_budget": float(row[0]), "max_budget": float(row[1])}
                for tier, row in BUDGET_RANGES.items()
            },
            "certification_distribution_by_genre": {
                str(genre): {str(cert): float(score) for cert, score in CERT_DISTS[str(genre)].items()}
                for genre in GENRES
            },
            "company_profile_tiers": {
                "Global": {"capital": 1000.0, "margin": 0.15, "debt": 0.40, "slate": 20.0, "buffer": 0.20, "growth": 0.05, "eff": 0.80},
                "Major": {"capital": 500.0, "margin": 0.12, "debt": 0.50, "slate": 12.0, "buffer": 0.15, "growth": 0.04, "eff": 0.85},
                "Mid-Budget": {"capital": 100.0, "margin": 0.10, "debt": 0.30, "slate": 5.0, "buffer": 0.10, "growth": 0.08, "eff": 0.90},
                "Indie": {"capital": 20.0, "margin": 0.08, "debt": 0.20, "slate": 3.0, "buffer": 0.05, "growth": 0.10, "eff": 0.95},
                "Micro": {"capital": 2.0, "margin": 0.05, "debt": 0.10, "slate": 1.0, "buffer": 0.02, "growth": 0.15, "eff": 1.00},
            },
            "company_profile_coefficients": {
                "budget_focus_weights": [0.40, 0.30, 0.15, 0.10, 0.05],
                "risk_tolerance": 0.25,
                "marketing_multiplier": 1.50,
                "franchise_propensity": 0.35,
                "co_production_likelihood": 0.40,
            },
            "quality_latent_weights": {
                str(key): float(value)
                for key, value in _DEFAULT_QUALITY_LATENT_WEIGHTS.items()
            },
            "market_latent_weights": {
                str(key): float(value)
                for key, value in _DEFAULT_MARKET_LATENT_WEIGHTS.items()
            },
            "performance_model": {
                str(key): float(value)
                for key, value in _DEFAULT_PERFORMANCE_MODEL.items()
            },
            "vote_model": {
                str(key): float(value)
                for key, value in _DEFAULT_VOTE_MODEL.items()
            },
            "runtime_model": {
                str(key): value
                for key, value in _DEFAULT_RUNTIME_MODEL.items()
            },
            "award_campaign_weights": {
                str(key): float(value)
                for key, value in _DEFAULT_AWARD_CAMPAIGN_WEIGHTS.items()
            },
        }
    if "keyword_generation" in sections:
        root["keyword_generation"] = {
            "genre_target_weights": {str(genre): 0.0 for genre in GENRES},
            "generic_budget_ratio": 0.0,
            "min_specific_story_share": 0.0,
            "selection_bucket_targets": {
                "exact_anchor": 0.0,
                "related_support": 0.0,
                "story_specific": 0.0,
                "generic": 0.0,
            },
        }
    if "character_generation" in sections:
        root["character_generation"] = {
            "slot_archetype_candidates": {
                "0": ["Lead Hero", "Lead Villain"],
                "1": ["Love Interest", "Sidekick", "Mentor", "Lead Villain"],
                "2": ["Mentor", "Comic Relief", "Henchman"],
            },
            "general_archetypes": ["Supporting", "Authority Figure", "Victim", "Extra"],
            "unique_archetypes": ["Lead Hero", "Lead Villain", "Mentor", "Mysterious Stranger"],
            "genre_archetype_candidates": {str(genre): [] for genre in GENRES},
            "archetype_target_vectors": {name: [0.0, 0.0, 0.0, 0.0] for name in _CHARACTER_RUNTIME_ARCHETYPES},
            "career_stage_archetype_bias": {stage: {} for stage in ("rising", "prime", "veteran", "legend", "retired")},
            "genre_archetype_bias": {str(genre): {} for genre in GENRES},
            "collaboration_style_archetype_bias": {style: {} for style in ("solo", "ensemble", "mentorship")},
        }
    if "secondary_table_priors" in sections:
        from secondary_tables import (
            _DEFAULT_AWARD_PRIORS,
            _DEFAULT_DEMOGRAPHICS_PRIORS,
            _DEFAULT_PERSON_CONTRACT_PRIORS,
            _DEFAULT_PRODUCTION_TIMELINE_PRIORS,
            _DEFAULT_RELEASE_DATE_PRIORS,
            _DEFAULT_REVIEW_PRIORS,
            _DEFAULT_STREAMING_WINDOW_PRIORS,
            _DEFAULT_TERRITORY_BOX_OFFICE_PRIORS,
            _DEFAULT_TV_GENERATION_PRIORS,
        )

        root["secondary_table_priors"] = {
            "demographics": _shape_only_skeleton(_DEFAULT_DEMOGRAPHICS_PRIORS),
            "release_dates": _shape_only_skeleton(_DEFAULT_RELEASE_DATE_PRIORS),
            "territory_box_office": _shape_only_skeleton(_DEFAULT_TERRITORY_BOX_OFFICE_PRIORS),
            "reviews": _shape_only_skeleton(_DEFAULT_REVIEW_PRIORS),
            "awards": _shape_only_skeleton(_DEFAULT_AWARD_PRIORS),
            "tv_generation": _shape_only_skeleton(_DEFAULT_TV_GENERATION_PRIORS),
            "production_timeline": _shape_only_skeleton(_DEFAULT_PRODUCTION_TIMELINE_PRIORS),
            "streaming_windows": _shape_only_skeleton(_DEFAULT_STREAMING_WINDOW_PRIORS),
            "person_contracts": _shape_only_skeleton(_DEFAULT_PERSON_CONTRACT_PRIORS),
        }
    if "selection_weights" in sections:
        from assembly import (
            _DEFAULT_CAST_SELECTION,
            _DEFAULT_COMPANY_SELECTION,
            _DEFAULT_DIRECTOR_SELECTION,
            _DEFAULT_KEYWORD_SELECTION,
        )

        selection_skeleton = {
            "cast_focus_exploration": 0.40,
            "cast_base_focus_exploration": 0.45,
            "cast_slot_exploration_empty": 0.35,
            "cast_slot_exploration_filled": 0.30,
            "cast_style_multiplier": 2.0,
            "cast_community_match_multiplier": 1.5,
            "director_exploration_share": 0.30,
            "company_primary_exploration_share": 0.35,
            "company_secondary_exploration_share": 0.40,
            "crew_exploration_share": 0.35,
            "concept_genre_bias_base": 0.80,
            "concept_genre_bias_scale": 2.20,
            "concept_country_bias_base": 0.90,
            "concept_country_bias_scale": 1.90,
            "concept_market_bias_base": 0.90,
            "concept_market_bias_scale": 1.60,
            "concept_exact_genre_hint_boost": 2.10,
            "concept_genre_hint_miss_penalty": 0.12,
            "concept_franchise_genre_match_boost": 1.20,
            "concept_tier_match_boost": 1.15,
            "concept_franchise_eligible_scale": 0.50,
            "concept_strategy_bonus_scale": 0.40,
            "concept_release_pressure_base": 0.92,
            "concept_release_pressure_scale": 0.55,
            "concept_novelty_base": 0.95,
            "concept_novelty_scale": 0.45,
            "concept_franchise_strategy_match_boost": 1.18,
            "concept_franchise_season_match_boost": 1.10,
            "concept_sequel_pressure_scale": 0.35,
            "concept_pack_usage_capacity": 4.0,
            "concept_bucket_country_capacity": 2.0,
            "concept_bucket_genre_capacity": 1.35,
            "concept_country_usage_capacity": 1.0,
            "concept_minor_country_bonus": 1.06,
            "concept_minor_market_bonus": 1.04,
            "director_risk_weight": 0.40,
            "director_ambition_weight": 0.35,
            "director_prestige_weight": 0.25,
            "director_alignment_base": 0.70,
            "director_alignment_scale": 0.80,
            "director_csv_base": 0.80,
            "director_csv_scale": 0.50,
            "company_tier_match_boost": 8.0,
            "company_tier_mismatch_penalty": 0.20,
            "company_genre_match_boost": 15.0,
            "company_genre_mismatch_penalty": 0.25,
            "company_risk_weight": 0.28,
            "company_prestige_weight": 0.22,
            "company_focus_weight": 0.30,
            "company_genre_fit_weight": 0.20,
            "company_alignment_base": 0.65,
            "company_alignment_scale": 0.90,
            "director_selection": _shape_only_skeleton(
                {k: v for k, v in _DEFAULT_DIRECTOR_SELECTION.items() if k != "co_director_probability_by_tier"}
            ),
            "co_director_probability_by_tier": {str(tier): 0.0 for tier in PRODUCTION_TIERS},
            "company_selection": _shape_only_skeleton(_DEFAULT_COMPANY_SELECTION),
            "cast_selection": _shape_only_skeleton(
                {k: v for k, v in _DEFAULT_CAST_SELECTION.items() if k not in {"geo_boost_by_tier", "dynamic_cast_base_by_tier"}}
            ),
            "geo_boost_by_tier": {str(tier): 0.0 for tier in PRODUCTION_TIERS},
            "dynamic_cast_base_by_tier": {str(tier): {"min": 0, "max": 1} for tier in PRODUCTION_TIERS},
            "keyword_selection": _shape_only_skeleton(
                {k: v for k, v in _DEFAULT_KEYWORD_SELECTION.items() if k not in {"count_by_tier", "year_slate_family_boosts"}}
            ) | {
                "primary_genre_min_count_by_tier": {str(tier): 0.0 for tier in PRODUCTION_TIERS},
                "exact_topic_min_count_by_tier": {str(tier): 0.0 for tier in PRODUCTION_TIERS},
                "primary_plus_related_min_count_by_tier": {str(tier): 0.0 for tier in PRODUCTION_TIERS},
                "generic_keyword_cap_by_tier": {str(tier): 0.0 for tier in PRODUCTION_TIERS},
                "off_genre_cap_by_tier": {str(tier): 0.0 for tier in PRODUCTION_TIERS},
                "slot_mix_by_tier": {
                    str(tier): {
                        "exact_anchor": 0.0,
                        "related_support": 0.0,
                        "story_specific": 0.0,
                        "franchise": 0.0,
                        "generic": 0.0,
                    }
                    for tier in PRODUCTION_TIERS
                },
                "related_genres_by_genre": {str(genre): [str(genre)] for genre in GENRES},
            },
            "keyword_count_by_tier": {str(tier): {"min": 0, "max": 1} for tier in PRODUCTION_TIERS},
            "keyword_year_slate_family_boosts": _shape_only_skeleton(_DEFAULT_KEYWORD_SELECTION.get("year_slate_family_boosts", {})),
            "writer_director_probability_by_tier": {str(tier): 0.0 for tier in PRODUCTION_TIERS},
            "concept_ambition_target_by_tier": {str(tier): 0.0 for tier in PRODUCTION_TIERS},
            "concept_prestige_target_by_tier": {str(tier): 0.0 for tier in PRODUCTION_TIERS},
            "concept_style_tier_shift_by_tier": {str(tier): [0.0] * 8 for tier in PRODUCTION_TIERS},
            "concept_risk_target_by_genre": {str(genre): 0.0 for genre in GENRES},
            "concept_style_vector_by_genre": {str(genre): [0.0] * 8 for genre in GENRES},
            "genre_tier_distribution": {str(genre): [0.0] * len(PRODUCTION_TIERS) for genre in GENRES},
            "release_month_base_weights": [0.0] * 12,
            "release_season_month_bumps": {"summer": [0.0] * 12},
            "genre_release_month_bumps": {str(GENRES[0]): [0.0] * 12},
        }
        root["selection_weights"] = selection_skeleton
    if "scalable_edge_priors" in sections:
        root["scalable_edge_priors"] = {
            "stage_priority": {
                "legend": 0,
                "veteran": 1,
                "prime": 2,
                "retired": 3,
                "rising": 4,
            },
            "collaboration_style_codes": {
                "ensemble": 0,
                "selective": 1,
                "auteur": 2,
            },
            "profile_overrides": {},
            "person_degree_caps": {
                "rising": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
                "prime": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
                "veteran": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
                "legend": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
                "retired": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
            },
            "brand_fit_ratio_by_stage": {str(stage): 0.0 for stage in ("legend", "veteran", "prime", "rising", "retired")},
            "employment_ratio_by_stage": {str(stage): 0.0 for stage in ("legend", "veteran", "prime", "rising", "retired")},
            "sampled_union_defaults": {"person_sample_ratio": 0.0, "company_sample_ratio": 0.0, "max_candidates_per_person": 0.0},
            "valid_year_span": {"start_year": 0.0, "end_year": 0.0},
            "year_validity_offsets": {
                "person_edge_end_offset": 0.0,
                "company_edge_end_offset": 0.0,
            },
        }
    return root


def _modeling_priors_completion_prompt(
    args: argparse.Namespace,
    group_name: str,
    sections: tuple[str, ...],
    validation_error: Exception,
    current_payload: dict[str, Any],
) -> str:
    section_block = "\n".join(
        f"{section}:\n{_MODELING_PRIOR_SECTION_PROMPTS.get(section, '- provide a populated object')}"
        for section in sections
    )
    missing_map = _validation_missing_field_map(validation_error)
    missing_block = ""
    if missing_map:
        bullet_lines = [f"- {path}: {', '.join(keys)}" for path, keys in missing_map.items()]
        skeleton = _validation_completion_skeleton(sections, missing_map)
        missing_block = (
            "\nExact missing fields to fill:\n"
            + "\n".join(bullet_lines)
            + "\n\nReturn them under the correct nested paths. Use this skeleton as a guide and replace the placeholder numeric values with real values:\n"
            + json.dumps(skeleton, indent=2, ensure_ascii=False)
            + "\n"
        )
    runtime_schema = _runtime_schema_completion_skeleton(sections)
    runtime_schema_block = ""
    if runtime_schema:
        runtime_schema_block = (
            "\nExact runtime schema skeleton to match:\n"
            + json.dumps(runtime_schema, indent=2, ensure_ascii=False)
            + "\nUse these exact nested keys. Replace placeholder values with real priors and keep semantically valid current values where possible.\n"
        )
    return f"""Complete the missing or invalid parts of the `modeling_priors` group `{group_name}`.

Rules:
- Return one JSON object only.
- Return only the requested top-level sections.
- Preserve valid existing values from the current payload unless a field is clearly malformed.
- Fill every missing required field referenced by the validation error.
- If the validation error names a nested block, rewrite that entire nested block using the exact required runtime keys rather than inventing near-synonyms.
- Do not return placeholders, comments, markdown fences, or empty objects.
- Use the requested movie counts and year range as context.

Context:
{_context_block(args)}

Validation error:
{validation_error}
{missing_block}{runtime_schema_block}

Requested sections and required content:
{section_block}

Current payload to complete:
{json.dumps(current_payload, indent=2, ensure_ascii=False)}
"""


def _save_artifact(path: Path, artifact_name: str, payload: dict[str, Any], model: str) -> None:
    meta = dict(payload.get("meta") or {})
    meta.update(
        {
            "schema_version": 1,
            "artifact": artifact_name,
            "generator_mode": "llm",
            "model": model,
        }
    )
    payload = dict(payload)
    payload["meta"] = meta
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _generate_json_artifact(
    *,
    client: Any,
    artifact_name: str,
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    thinking_budget: int | None,
    base_dir: Path,
    coerce: Callable[[Any], dict[str, Any] | None],
    validate: Callable[[dict[str, Any]], None],
    timeout_sec: float = 120.0,
    max_attempts: int = 4,
) -> dict[str, Any]:
    response = client.generate(
        prompt,
        model=model,
        json_mode=True,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_sec=timeout_sec,
        max_attempts=max_attempts,
        thinking_budget=thinking_budget,
    )
    parsed_raw = safe_json_parse(response.text)
    parsed = coerce(parsed_raw)
    if not isinstance(parsed, dict):
        raw_salvage = coerce(response.text)
        if isinstance(raw_salvage, dict):
            parsed = raw_salvage
    if not isinstance(parsed, dict):
        repair = client.generate(
            _repair_prompt(artifact_name, response.text),
            model=model,
            json_mode=True,
            temperature=0.0,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
            max_attempts=2,
            thinking_budget=thinking_budget,
        )
        parsed_raw = safe_json_parse(repair.text)
        parsed = coerce(parsed_raw)
        if not isinstance(parsed, dict):
            raw_salvage = coerce(repair.text)
            if isinstance(raw_salvage, dict):
                parsed = raw_salvage
    if not isinstance(parsed, dict):
        _write_failed_artifact_debug(base_dir, artifact_name, response.text, parsed_raw)
        raise ValueError(f"{artifact_name} response was not a JSON object")
    try:
        validate(parsed)
    except Exception as exc:
        raw_salvage = coerce(response.text)
        if isinstance(raw_salvage, dict):
            try:
                validate(raw_salvage)
                return raw_salvage
            except Exception as raw_exc:
                if _payload_materiality_score(raw_salvage) > _payload_materiality_score(parsed):
                    parsed = raw_salvage
                    exc = raw_exc
        repair = client.generate(
            _validation_repair_prompt(artifact_name, prompt, exc, parsed),
            model=model,
            json_mode=True,
            temperature=0.0,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
            max_attempts=2,
            thinking_budget=thinking_budget,
        )
        repaired_raw = safe_json_parse(repair.text)
        repaired = coerce(repaired_raw)
        if not isinstance(repaired, dict):
            _write_failed_artifact_debug(base_dir, artifact_name, response.text, repaired_raw)
            raise
        try:
            validate(repaired)
        except Exception:
            _write_failed_artifact_debug(base_dir, artifact_name, response.text, repaired)
            raise
        parsed = repaired
    return parsed


def _generate_identity_bank_payload(
    *,
    client: Any,
    args: argparse.Namespace,
    model: str,
    base_dir: Path,
    thinking_budget: int | None,
) -> dict[str, Any]:
    assembled: dict[str, Any] = {"families": [], "defaults": {}}
    seen_nationalities: set[str] = set()

    for group in _IDENTITY_BANK_GROUPS:
        group_name = str(group["name"])
        nationalities = tuple(str(item) for item in group["nationalities"])
        target_families = int(group["target_families"])
        payload = _generate_json_artifact(
            client=client,
            artifact_name=f"identity_bank__{group_name}",
            prompt=_prompt_identity_bank_group(args, group_name, nationalities, target_families),
            model=model,
            temperature=float(group["temperature"]),
            max_tokens=int(group["max_tokens"]),
            thinking_budget=thinking_budget,
            base_dir=base_dir,
            coerce=_coerce_identity_bank_group_payload,
            validate=lambda row, _minimum=target_families: _validate_identity_bank_family_rows(row.get("families"), minimum=_minimum),
            timeout_sec=150.0,
            max_attempts=4,
        )
        for raw in payload.get("families", []):
            if not isinstance(raw, dict):
                continue
            nationality = str(raw.get("nationality", "") or "").strip()
            nationality_key = nationality.casefold()
            if not nationality or nationality_key in seen_nationalities:
                continue
            seen_nationalities.add(nationality_key)
            assembled["families"].append(raw)

    defaults_payload = _generate_json_artifact(
        client=client,
        artifact_name="identity_bank__defaults",
        prompt=_prompt_identity_bank_defaults(args),
        model=model,
        temperature=0.10,
        max_tokens=2500,
        thinking_budget=thinking_budget,
        base_dir=base_dir,
        coerce=_coerce_identity_bank_defaults_payload,
        validate=_validate_identity_bank_defaults_payload,
        timeout_sec=120.0,
        max_attempts=4,
    )
    assembled["defaults"] = defaults_payload.get("defaults", {})
    assembled = _normalize_identity_bank_payload(assembled)
    _validate_identity_bank(assembled)
    if len(assembled.get("families", [])) < 18:
        raise ValueError("identity_bank artifact is too shallow after grouped generation")
    return assembled


def _generate_keyword_seed_bank_payload(
    *,
    client: Any,
    args: argparse.Namespace,
    model: str,
    base_dir: Path,
    thinking_budget: int | None,
) -> dict[str, Any]:
    debug_dir = _artifact_debug_dir(base_dir)
    assembled: dict[str, Any] = {
        "universal_qualifiers": [],
        "universal_contexts": [],
        "generic_themes": [],
        "genres": [],
    }

    globals_payload = _generate_json_artifact(
        client=client,
        artifact_name="keyword_seed_bank__globals",
        prompt=_prompt_keyword_seed_bank_globals(args),
        model=model,
        temperature=0.22,
        max_tokens=7000,
        thinking_budget=thinking_budget,
        base_dir=base_dir,
        coerce=_coerce_keyword_seed_bank_globals_payload,
        validate=_validate_keyword_seed_bank_globals_payload,
        timeout_sec=120.0,
        max_attempts=4,
    )
    assembled.update(globals_payload)
    (debug_dir / "keyword_seed_bank__globals_accepted.json").write_text(
        json.dumps(globals_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    genre_rows: dict[str, dict[str, Any]] = {}
    for idx, chunk in enumerate(_genre_chunks(GENRES, 4), start=1):
        payload = _generate_json_artifact(
            client=client,
            artifact_name=f"keyword_seed_bank__genres_{idx}",
            prompt=_prompt_keyword_seed_bank_genre_group(args, chunk),
            model=model,
            temperature=0.25,
            max_tokens=7000,
            thinking_budget=thinking_budget,
            base_dir=base_dir,
            coerce=lambda raw, _chunk=chunk: _coerce_keyword_seed_bank_genre_group_payload(raw, _chunk),
            validate=lambda payload, _chunk=chunk: _validate_keyword_seed_bank_genre_group_payload(payload, _chunk),
            timeout_sec=120.0,
            max_attempts=4,
        )
        accepted_rows: list[dict[str, Any]] = []
        for row in list(payload.get("genres") or []):
            if not isinstance(row, dict):
                continue
            genre = str(row.get("genre", "") or "").strip()
            if not genre:
                continue
            genre_rows[genre] = row
            accepted_rows.append(row)
        (debug_dir / f"keyword_seed_bank__genres_{idx}_accepted.json").write_text(
            json.dumps({"genres": accepted_rows}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    assembled["genres"] = [genre_rows[str(genre)] for genre in GENRES if str(genre) in genre_rows]
    _validate_keyword_seed_bank(assembled)
    return assembled


def _generate_modeling_priors_payload(
    *,
    client: Any,
    args: argparse.Namespace,
    model: str,
    base_dir: Path,
    thinking_budget: int | None,
) -> dict[str, Any]:
    assembled: dict[str, Any] = {}
    debug_dir = _artifact_debug_dir(base_dir)

    def _complete_group_payload(
        *,
        group_name: str,
        sections: tuple[str, ...],
        payload: dict[str, Any],
        validation_error: Exception,
        max_tokens: int,
    ) -> dict[str, Any]:
        working = _prune_nested_modeling_sections(_normalize_modeling_priors_payload(payload, sections))
        last_error: Exception = validation_error
        for attempt in range(4):
            completion = _generate_json_artifact(
                client=client,
                artifact_name=f"modeling_priors__{group_name}__completion_{attempt + 1}",
                prompt=_modeling_priors_completion_prompt(args, group_name, sections, last_error, working),
                model=model,
                temperature=0.0,
                max_tokens=max_tokens,
                thinking_budget=thinking_budget,
                base_dir=base_dir,
                coerce=lambda raw, _sections=sections: _coerce_modeling_priors_group_payload(raw, _sections),
                validate=lambda _payload: None,
            )
            working = _prune_nested_modeling_sections(
                _normalize_modeling_priors_payload(_deep_merge(working, completion), sections)
            )
            try:
                _validate_modeling_priors_sections(working, sections)
                return working
            except Exception as exc:
                last_error = exc
        raise last_error

    for group in _MODELING_PRIOR_GROUPS:
        sections = tuple(group["sections"])
        prompt = _prompt_modeling_priors_group(args, str(group["name"]), sections)
        artifact_debug_name = f"modeling_priors__{group['name']}"

        def _coerce_group(raw: Any, _sections: tuple[str, ...] = sections) -> dict[str, Any] | None:
            return _coerce_modeling_priors_group_payload(raw, _sections)

        def _validate_group(payload: dict[str, Any], _sections: tuple[str, ...] = sections) -> None:
            _validate_modeling_priors_sections(payload, _sections)

        payload = _generate_json_artifact(
            client=client,
            artifact_name=artifact_debug_name,
            prompt=prompt,
            model=model,
            temperature=float(group["temperature"]),
            max_tokens=int(group["max_tokens"]),
            thinking_budget=thinking_budget,
            base_dir=base_dir,
            coerce=_coerce_group,
            validate=lambda _payload: None,
        )
        payload = _prune_nested_modeling_sections(payload, sections)
        try:
            _validate_group(payload)
        except Exception as exc:
            payload = _complete_group_payload(
                group_name=str(group["name"]),
                sections=sections,
                payload=payload,
                validation_error=exc,
                max_tokens=max(8000, int(group["max_tokens"])),
            )
        for section in sections:
            assembled[section] = _deep_merge(assembled.get(section, {}), payload.get(section, {}))
        (debug_dir / f"{artifact_debug_name}_accepted.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    assembled = _prune_nested_modeling_sections(_normalize_modeling_priors_payload(assembled))
    _validate_modeling_priors(assembled)
    if _modeling_priors_richness_score(assembled) < 120:
        raise ValueError("modeling_priors artifact is too shallow after grouped generation")
    return assembled


def _generate_title_grammar_bank_payload(
    *,
    client: Any,
    args: argparse.Namespace,
    model: str,
    base_dir: Path,
    thinking_budget: int | None,
) -> dict[str, Any]:
    debug_dir = _artifact_debug_dir(base_dir)
    assembled: dict[str, Any] = {}

    vocab = _generate_json_artifact(
        client=client,
        artifact_name="title_grammar_bank__vocab",
        prompt=_prompt_title_grammar_vocab(args),
        model=model,
        temperature=0.25,
        max_tokens=5000,
        thinking_budget=thinking_budget,
        base_dir=base_dir,
        coerce=_coerce_title_grammar_vocab_payload,
        validate=_validate_title_grammar_vocab_payload,
        timeout_sec=120.0,
        max_attempts=4,
    )
    assembled.update(vocab)
    (debug_dir / "title_grammar_bank__vocab_accepted.json").write_text(
        json.dumps(vocab, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    genre_templates_assembled: dict[str, list[str]] = {}
    for idx, chunk in enumerate(_genre_chunks(GENRES, 7), start=1):
        payload = _generate_json_artifact(
            client=client,
            artifact_name=f"title_grammar_bank__genre_templates_{idx}",
            prompt=_prompt_title_grammar_genre_templates(args, chunk),
            model=model,
            temperature=0.25,
            max_tokens=6000,
            thinking_budget=thinking_budget,
            base_dir=base_dir,
            coerce=lambda raw, _chunk=chunk: _coerce_title_grammar_genre_templates_payload(raw, _chunk),
            validate=lambda payload, _chunk=chunk: _validate_title_grammar_genre_templates_payload(payload, _chunk),
            timeout_sec=120.0,
            max_attempts=4,
        )
        genre_templates_assembled.update(dict(payload.get("genre_templates") or {}))
        (debug_dir / f"title_grammar_bank__genre_templates_{idx}_accepted.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    assembled["genre_templates"] = genre_templates_assembled

    tagline_templates_assembled: dict[str, list[str]] = {}
    for idx, chunk in enumerate(_genre_chunks(GENRES, 6), start=1):
        payload = _generate_json_artifact(
            client=client,
            artifact_name=f"title_grammar_bank__tagline_templates_{idx}",
            prompt=_prompt_title_grammar_taglines(args, chunk),
            model=model,
            temperature=0.20,
            max_tokens=7000,
            thinking_budget=thinking_budget,
            base_dir=base_dir,
            coerce=lambda raw, _chunk=chunk: _coerce_title_grammar_taglines_payload(raw, _chunk),
            validate=lambda payload, _chunk=chunk: _validate_title_grammar_taglines_payload(payload, _chunk, minimum_per_genre=10),
            timeout_sec=120.0,
            max_attempts=4,
        )
        tagline_templates_assembled.update(dict(payload.get("tagline_templates") or {}))
        (debug_dir / f"title_grammar_bank__tagline_templates_{idx}_accepted.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    assembled["tagline_templates"] = tagline_templates_assembled

    repaired_taglines, duplicate_map = _dedupe_title_grammar_taglines(assembled)
    assembled["tagline_templates"] = repaired_taglines
    if duplicate_map:
        (debug_dir / "title_grammar_bank__tagline_duplicates_detected.json").write_text(
            json.dumps(duplicate_map, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    for genre in GENRES:
        genre_key = str(genre)
        current = list(assembled.get("tagline_templates", {}).get(genre_key) or [])
        attempts = 0
        while len(current) < 12 and attempts < 4:
            attempts += 1
            avoid_templates: list[str] = []
            seen: set[str] = set()
            for other_genre, values in dict(assembled.get("tagline_templates") or {}).items():
                for raw in list(values or []):
                    key = _template_key(raw)
                    if key and key not in seen:
                        seen.add(key)
                        avoid_templates.append(str(raw).strip())
            payload = _generate_json_artifact(
                client=client,
                artifact_name=f"title_grammar_bank__tagline_supplement_{genre_key}_{attempts}",
                prompt=_prompt_title_grammar_tagline_supplement(args, genre_key, max(12 - len(current), 4), avoid_templates),
                model=model,
                temperature=0.15,
                max_tokens=5000,
                thinking_budget=thinking_budget,
                base_dir=base_dir,
                coerce=lambda raw, _genre=genre_key: _coerce_title_grammar_taglines_payload(raw, (_genre,)),
                validate=lambda payload, _genre=genre_key, _needed=max(12 - len(current), 4): _validate_title_grammar_taglines_payload(
                    payload,
                    (_genre,),
                    minimum_per_genre=_needed,
                ),
                timeout_sec=120.0,
                max_attempts=4,
            )
            additions = list((payload.get("tagline_templates") or {}).get(genre_key) or [])
            merged_payload = {"tagline_templates": {genre_key: current + additions}}
            local_validated = _coerce_title_grammar_taglines_payload(merged_payload, (genre_key,)) or {"tagline_templates": {genre_key: []}}
            current = list((local_validated.get("tagline_templates") or {}).get(genre_key) or [])
            # Filter against globally owned templates again after supplementation.
            assembled["tagline_templates"][genre_key] = current
            repaired_taglines, duplicate_map = _dedupe_title_grammar_taglines(assembled)
            assembled["tagline_templates"] = repaired_taglines
            current = list(assembled.get("tagline_templates", {}).get(genre_key) or [])
            (debug_dir / f"title_grammar_bank__tagline_supplement_{genre_key}_{attempts}_accepted.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    extra_placeholders = sorted(
        field_name
        for field_name in _title_grammar_placeholder_names(assembled)
        if _TITLE_GRAMMAR_BASE_PLACEHOLDER_ALIASES.get(field_name, field_name)
        not in _TITLE_GRAMMAR_BASE_PLACEHOLDER_ALIASES.values()
    )
    if extra_placeholders:
        placeholder_values = _generate_json_artifact(
            client=client,
            artifact_name="title_grammar_bank__tagline_placeholder_values",
            prompt=_prompt_title_grammar_placeholder_values(args, extra_placeholders),
            model=model,
            temperature=0.15,
            max_tokens=6000,
            thinking_budget=thinking_budget,
            base_dir=base_dir,
            coerce=lambda raw, _fields=tuple(extra_placeholders): _coerce_title_grammar_placeholder_values_payload(raw, _fields),
            validate=lambda payload, _fields=tuple(extra_placeholders): _validate_title_grammar_placeholder_values_payload(payload, _fields),
            timeout_sec=120.0,
            max_attempts=4,
        )
        assembled["tagline_placeholder_values"] = dict(placeholder_values.get("tagline_placeholder_values") or {})
        (debug_dir / "title_grammar_bank__tagline_placeholder_values_accepted.json").write_text(
            json.dumps(placeholder_values, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    else:
        assembled["tagline_placeholder_values"] = {}

    phases = _generate_json_artifact(
        client=client,
        artifact_name="title_grammar_bank__phases",
        prompt=_prompt_title_grammar_phases(args),
        model=model,
        temperature=0.15,
        max_tokens=4000,
        thinking_budget=thinking_budget,
        base_dir=base_dir,
        coerce=_coerce_title_grammar_phases_payload,
        validate=_validate_title_grammar_phases_payload,
        timeout_sec=120.0,
        max_attempts=4,
    )
    assembled["year_style_phases"] = list(phases.get("year_style_phases") or [])
    (debug_dir / "title_grammar_bank__phases_accepted.json").write_text(
        json.dumps(phases, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    title_priors = prior_section(load_modeling_priors_artifact(base_dir, mode=str(args.mode)) or {}, "title_generation")
    allowed_placeholders = [
        _normalize_title_placeholder_token(item)
        for item in list(title_priors.get("allowed_tagline_placeholders") or [])
        if _normalize_title_placeholder_token(item)
    ]
    if not allowed_placeholders:
        allowed_placeholders = sorted(_title_grammar_placeholder_names(assembled))
    constraints = title_priors.get("tagline_render_constraints")
    if not isinstance(constraints, dict):
        constraints = {}
    assembled["allowed_tagline_placeholders"] = list(dict.fromkeys(allowed_placeholders))
    assembled["tagline_render_constraints"] = {
        "min_words": int(constraints.get("min_words", 4) or 4),
        "max_words": int(constraints.get("max_words", 10) or 10),
        "max_placeholder_count": int(constraints.get("max_placeholder_count", 1) or 1),
        "forbid_square_brackets": bool(constraints.get("forbid_square_brackets", True)),
        "allow_unresolved_placeholders": bool(constraints.get("allow_unresolved_placeholders", False)),
    }

    _validate_title_grammar_bank(assembled)
    return assembled


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Mirage bootstrap artifacts.")
    parser.add_argument("--artifact", required=True, choices=sorted(_artifact_specs().keys()))
    parser.add_argument("--base-dir", default=str(BASE_DIR))
    parser.add_argument("--mode", default=current_mode())
    parser.add_argument("--model", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n-movies", type=int, default=10000)
    parser.add_argument("--n-persons", type=int, default=32000)
    parser.add_argument("--n-companies", type=int, default=1000)
    parser.add_argument("--n-keywords", type=int, default=1500)
    parser.add_argument("--n-titles", type=int, default=10000)
    parser.add_argument("--start-year", type=int, default=1950)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--thinking-budget", type=int, default=None)
    args = parser.parse_args()

    mode = current_mode(args.mode)
    if mode != "research":
        print(f"Skipping bootstrap artifact generation in {mode} mode")
        return

    spec = _artifact_specs()[args.artifact]
    path = spec["path"](Path(args.base_dir).resolve())
    if path.exists() and not args.force:
        print(f"{args.artifact} already exists at {path}")
        return

    prompt = _PROMPTS[args.artifact](args)
    client = get_llm_client()
    model = str(args.model or spec["default_model"])
    thinking_budget = int(args.thinking_budget) if args.thinking_budget is not None else None
    base_dir = Path(args.base_dir).resolve()
    if args.artifact == "modeling_priors":
        parsed = _generate_modeling_priors_payload(
            client=client,
            args=args,
            model=model,
            base_dir=base_dir,
            thinking_budget=thinking_budget,
        )
    elif args.artifact == "title_grammar_bank":
        parsed = _generate_title_grammar_bank_payload(
            client=client,
            args=args,
            model=model,
            base_dir=base_dir,
            thinking_budget=thinking_budget,
        )
    elif args.artifact == "identity_bank":
        parsed = _generate_identity_bank_payload(
            client=client,
            args=args,
            model=model,
            base_dir=base_dir,
            thinking_budget=thinking_budget,
        )
    elif args.artifact == "keyword_seed_bank":
        parsed = _generate_keyword_seed_bank_payload(
            client=client,
            args=args,
            model=model,
            base_dir=base_dir,
            thinking_budget=thinking_budget,
        )
    else:
        prompt = _PROMPTS[args.artifact](args)
        parsed = _generate_json_artifact(
            client=client,
            artifact_name=args.artifact,
            prompt=prompt,
            model=model,
            temperature=float(spec["temperature"]),
            max_tokens=int(spec["max_tokens"]),
            thinking_budget=thinking_budget,
            base_dir=base_dir,
            coerce=lambda raw: _coerce_artifact_payload(args.artifact, raw),
            validate=_VALIDATORS[args.artifact],
        )
    _save_artifact(path, args.artifact, parsed, model)
    print(f"Saved {args.artifact} -> {path}")


if __name__ == "__main__":
    main()
