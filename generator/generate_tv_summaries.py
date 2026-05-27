"""
Mirage -- Generate TV series and episode summaries via the configured LLM provider.
===============================================================
D4 fix: Fills `plot_summary` in tv_series.csv and `description` in
episodes.csv using a two-tier LLM approach.

Tier 1 -- configured Gemini model (1 call per series):
    Generates a 2-3 paragraph series overview given all metadata
    (title, genre, cast, season count, status).

Tier 2 -- configured Gemini model (1 call per season):
    Generates all episode descriptions in one batch, using:
    - The series overview from Tier 1
    - All episode metadata for the season (titles, ratings, air dates)
    - Cast list for the season
    - The finale line from the previous season (continuity)

Run AFTER generate_movies.py has produced tv_series.csv, seasons.csv,
episodes.csv, episode_cast.csv, and person.csv.

Usage:
    python generate_tv_summaries.py                  # full run
    python generate_tv_summaries.py --dry-run        # no API calls
    python generate_tv_summaries.py --batch-verify 1 # 1 series only
    python generate_tv_summaries.py --series-limit N # first N series
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from feather_sink import df_to_arrow, read_table
from llm_provider import get_llm_client
from model_defaults import model_for_role

BASE_DIR = Path(__file__).parent

# ── Model config ────────────────────────────────────────────────────────
_DEFAULT_MODEL = model_for_role("plot_summaries")
TIER1_MODEL = _DEFAULT_MODEL
TIER2_MODEL = _DEFAULT_MODEL

MAX_RETRIES = 7
_RETRY_DELAYS = [4, 8, 15, 30, 45, 60, 90]
_API_TIMEOUT = 70  # seconds -- hard limit per call
SLEEP_BETWEEN_CALLS = 1.0
SLEEP_ON_ERROR = 5.0
PIPELINE_MODE = str(os.environ.get("DATA_SYS_PIPELINE_MODE", "research") or "research").strip().lower()
DEFAULT_EPISODE_BATCH_SIZE = 4

# Cost rates (per 1M tokens) -- conservative estimates
COST_RATE = {
    "gemini-3.1-flash-lite": {"in": 0.02, "out": 0.08},
    "gemini-2.0-flash":        {"in": 0.10, "out": 0.40},
    "gemini-2.0-flash-lite":   {"in": 0.02, "out": 0.08},
}


def _resolve_tv_save_every_series(total_series: int) -> int:
    raw = str(os.environ.get("DATA_SYS_TV_SAVE_EVERY_SERIES", "") or "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except Exception:
            pass
    return 1


def _resolve_episode_batch_size() -> int:
    raw = str(os.environ.get("DATA_SYS_TV_EPISODE_BATCH_SIZE", "") or "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except Exception:
            pass
    return DEFAULT_EPISODE_BATCH_SIZE


# ═══════════════════════════════════════════════════════════════════════
# TOKEN / COST TRACKER
# ═══════════════════════════════════════════════════════════════════════

class TokenTracker:
    def __init__(self):
        self.by_model: dict[str, dict] = {}

    def record(self, model: str, response) -> tuple[int, int, float]:
        inp = getattr(response, "input_tokens", 0) or 0
        out = getattr(response, "output_tokens", 0) or 0
        cost = getattr(response, "cost_usd", 0.0) or 0.0
        if model not in self.by_model:
            self.by_model[model] = {"calls": 0, "inp": 0, "out": 0, "cost": 0.0, "errors": 0}
        m = self.by_model[model]
        m["calls"] += 1
        m["inp"] += inp
        m["out"] += out
        m["cost"] += cost
        return inp, out, cost

    def record_error(self, model: str):
        if model not in self.by_model:
            self.by_model[model] = {"calls": 0, "inp": 0, "out": 0, "cost": 0.0, "errors": 0}
        self.by_model[model]["errors"] += 1

    def summary(self) -> str:
        lines = ["\n" + "="*60, "  TOKEN USAGE SUMMARY", "="*60]
        total_cost = 0.0
        for model, m in sorted(self.by_model.items()):
            lines.append(f"  {model}")
            lines.append(f"    Calls:  {m['calls']} ({m['errors']} errors)")
            lines.append(f"    Tokens: {m['inp']:,} in + {m['out']:,} out")
            lines.append(f"    Cost:   ${m['cost']:.4f}")
            total_cost += m["cost"]
        lines += ["-"*60, f"  TOTAL COST: ${total_cost:.4f}", "="*60]
        return "\n".join(lines)


def _normalise_text(text: object) -> str:
    return " ".join(str(text or "").split()).strip()


def _word_count(text: object) -> int:
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'/-]*", str(text or "")))


def _is_valid_series_summary(text: object) -> bool:
    value = _normalise_text(text)
    if not value:
        return False
    low = value.lower()
    if "synthetic" in low:
        return False
    return _word_count(value) >= 45


def _is_valid_episode_description(text: object) -> bool:
    value = _normalise_text(text)
    if not value:
        return False
    low = value.lower()
    if "synthetic" in low:
        return False
    return _word_count(value) >= 14


def _is_generic_episode_title(text: object) -> bool:
    value = _normalise_text(text)
    return bool(re.match(r"^Episode\s+\d+$", value, flags=re.IGNORECASE))


def _is_valid_episode_title(text: object) -> bool:
    value = _normalise_text(text)
    if not value or _is_generic_episode_title(value):
        return False
    if re.search(r"[\[\]{}]", value):
        return False
    words = _word_count(value)
    return 2 <= words <= 8


def _is_transient_llm_error(exc: Exception) -> bool:
    message = str(exc or "").lower()
    transient_markers = (
        "503",
        "unavailable",
        "timeout",
        "timed out",
        "exceeded 70s",
        "deadline",
        "temporarily",
        "rate limit",
        "resource exhausted",
    )
    return any(marker in message for marker in transient_markers)


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_data(base_dir: Path) -> dict:
    """Load all TV-related tables into DataFrames, preferring Arrow."""
    def _load(name, table_name: str | None = None):
        df = read_table(str(base_dir / name), table_name)
        if df is None or df.empty:
            print(f"  WARNING: {name}.arrow/.csv not found or empty")
            return pd.DataFrame()
        return df

    series_df = _load("tv_series", "tv_series")
    seasons_df = _load("seasons", "seasons")
    episodes_df = _load("episodes", "episodes")
    ep_cast_df = _load("episode_cast", "episode_cast")
    persons_df = _load("person")
    if persons_df.empty:
        persons_df = _load("persons_enriched")

    # Build person name lookup
    name_map: dict[int, str] = {}
    if not persons_df.empty and "person_id" in persons_df.columns:
        for _, row in persons_df.iterrows():
            name_map[int(row["person_id"])] = str(row.get("name", f"Person {row['person_id']}"))

    # Build cast lists per series (from episode_cast)
    series_cast: dict[int, list[str]] = defaultdict(list)
    if not ep_cast_df.empty:
        for _, row in ep_cast_df.iterrows():
            sid = int(row.get("series_id", 0))
            pid = int(row.get("person_id", 0))
            name = name_map.get(pid, f"Actor {pid}")
            if name not in series_cast[sid]:
                series_cast[sid].append(name)

    # Build episode lists per season
    season_episodes: dict[int, list[dict]] = defaultdict(list)
    if not episodes_df.empty:
        for _, row in episodes_df.iterrows():
            sn_id = int(row.get("season_id", 0))
            season_episodes[sn_id].append({
                "episode_id": int(row["episode_id"]),
                "episode_number": int(row.get("episode_number", 0)),
                "title": str(row.get("title", "")),
                "runtime_minutes": int(row.get("runtime_minutes", 45)),
                "rating": float(row.get("rating", 7.0)),
                "air_date": str(row.get("air_date", "")),
                # skip if already has description
                "description": str(row.get("description", "")) if "description" in episodes_df.columns else "",
            })
        # Sort by episode number within each season
        for sn_id in season_episodes:
            season_episodes[sn_id].sort(key=lambda e: e["episode_number"])

    # Build seasons per series
    series_seasons: dict[int, list[dict]] = defaultdict(list)
    if not seasons_df.empty:
        for _, row in seasons_df.iterrows():
            sid = int(row.get("series_id", 0))
            series_seasons[sid].append({
                "season_id": int(row["season_id"]),
                "season_number": int(row.get("season_number", 1)),
                "year": int(row.get("year", 2020)),
                "episode_count": int(row.get("episode_count", 8)),
                "avg_rating": float(row.get("avg_rating", 7.0)),
            })
        for sid in series_seasons:
            series_seasons[sid].sort(key=lambda s: s["season_number"])

    return {
        "series_df": series_df,
        "seasons_df": seasons_df,
        "episodes_df": episodes_df,
        "series_cast": series_cast,
        "series_seasons": series_seasons,
        "season_episodes": season_episodes,
    }


def _save_tv_series_outputs(base_dir: Path, series_df: pd.DataFrame, all_series_summaries: dict[int, str]) -> None:
    series_df["plot_summary"] = series_df["series_id"].map(all_series_summaries).fillna("")
    df_to_arrow(series_df, str(base_dir / "tv_series.arrow"), table_name="tv_series")
    series_df.to_csv(base_dir / "tv_series.csv", index=False)


def _save_episode_outputs(
    base_dir: Path,
    episodes_df: pd.DataFrame,
    all_ep_descriptions: dict[int, str],
    all_ep_titles: dict[int, str] | None = None,
) -> None:
    if all_ep_titles:
        existing_titles = episodes_df["title"] if "title" in episodes_df.columns else pd.Series([""] * len(episodes_df))
        episodes_df["title"] = episodes_df["episode_id"].map(all_ep_titles).fillna(existing_titles)
    episodes_df["description"] = episodes_df["episode_id"].map(all_ep_descriptions).fillna("")
    df_to_arrow(episodes_df, str(base_dir / "episodes.arrow"), table_name="episodes")
    episodes_df.to_csv(base_dir / "episodes.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════
# PROMPT BUILDERS
# ═══════════════════════════════════════════════════════════════════════

def build_series_prompt(row: dict, cast_names: list[str], seasons: list[dict]) -> str:
    title = row.get("title", "Unknown")
    genre = row.get("genre", "Drama")
    country = row.get("country", "USA")
    language = row.get("language", "English")
    network = row.get("network", "")
    year_start = row.get("year_start", 2015)
    year_end = row.get("year_end", "")
    status = row.get("status", "Ended")
    n_seasons = row.get("total_seasons", len(seasons))
    overall_rating = row.get("overall_rating", 7.0)

    year_range = f"{year_start}-{year_end}" if year_end else f"{year_start}-present"
    cast_str = ", ".join(cast_names[:8]) if cast_names else "ensemble cast"
    season_summary = ", ".join(
        f"S{s['season_number']} ({s['episode_count']} eps, {s['year']}, {s['avg_rating']:.1f}★)"
        for s in seasons
    )

    return f"""Write a compelling 2-3 paragraph series overview for this TV show, as it would appear on a streaming platform's "About" page. This is SYNTHETIC FICTIONAL data.

SERIES METADATA:
  Title:    {title}
  Genre:    {genre}
  Country:  {country} | Language: {language}
  Network:  {network}
  Years:    {year_range} | Status: {status}
  Seasons:  {n_seasons} ({season_summary})
  Rating:   {overall_rating:.1f}/10
  Cast:     {cast_str}

Requirements:
- Paragraph 1 (3-4 sentences): Premise and setting. Who are the main characters? What world do they inhabit?
- Paragraph 2 (2-3 sentences): Themes and tone. What makes this show distinctive?
- Paragraph 3 (1-2 sentences): Any series arc or something that draws viewers back season after season.
- Match the genre tone: {genre} -- be authentic to that register.
- Use the cast names naturally (don't just list them).
- The series should feel like a real show with a coherent identity.
- Do NOT mention it is synthetic or fictional.
- 120-200 words total.

Output ONLY the series overview text, no JSON, no headings."""


def build_series_retry_prompt(row: dict, cast_names: list[str], seasons: list[dict]) -> str:
    title = row.get("title", "Unknown")
    genre = row.get("genre", "Drama")
    country = row.get("country", "USA")
    language = row.get("language", "English")
    network = row.get("network", "")
    year_start = row.get("year_start", 2015)
    year_end = row.get("year_end", "")
    status = row.get("status", "Ended")
    n_seasons = row.get("total_seasons", len(seasons))
    overall_rating = row.get("overall_rating", 7.0)

    year_range = f"{year_start}-{year_end}" if year_end else f"{year_start}-present"
    cast_str = ", ".join(cast_names[:8]) if cast_names else "ensemble cast"
    season_summary = ", ".join(
        f"S{s['season_number']} ({s['episode_count']} eps, {s['year']}, {s['avg_rating']:.1f}★)"
        for s in seasons
    )

    return f"""Write exactly one polished streaming-platform series overview paragraph for this TV show.

SERIES METADATA:
  Title:    {title}
  Genre:    {genre}
  Country:  {country} | Language: {language}
  Network:  {network}
  Years:    {year_range} | Status: {status}
  Seasons:  {n_seasons} ({season_summary})
  Rating:   {overall_rating:.1f}/10
  Cast:     {cast_str}

Requirements:
- Write 90-150 words in one paragraph.
- Include the premise, tone, and the season-to-season hook.
- Mention at least two cast names naturally in the text.
- Match the genre tone: {genre}.
- Do NOT mention that the data is synthetic or fictional.
- Do NOT use bullets, headings, or quotes.

Output ONLY the paragraph."""


def build_season_prompt(
    series_title: str,
    series_summary: str,
    genre: str,
    season: dict,
    episodes: list[dict],
    cast_names: list[str],
    continuity_note: str,
) -> str:
    sn = season["season_number"]
    year = season["year"]
    avg_rating = season["avg_rating"]
    cast_str = ", ".join(cast_names[:6]) if cast_names else "ensemble cast"

    prev_context = f'\nContinuity note: "{continuity_note}"' if continuity_note else ""

    ep_lines = "\n".join(
        f'  [{ep["episode_id"]}] Ep{ep["episode_number"]}: "{ep["title"]}" | '
        f'{ep["runtime_minutes"]}min | {ep["air_date"]} | ⭐{ep["rating"]:.1f}'
        for ep in episodes
    )

    return f"""Write one-paragraph episode descriptions for every episode in this TV season. This is SYNTHETIC FICTIONAL data.

SERIES: "{series_title}" ({genre})
SERIES OVERVIEW:
{series_summary}
{prev_context}

SEASON {sn} ({year}, avg rating {avg_rating:.1f}★, cast: {cast_str}):
{ep_lines}

Requirements per episode description:
- 2-3 sentences, 35-60 words
- Must advance the season arc logically (earlier episodes set up later ones)
- Higher-rated episodes should feel like standout/pivotal moments
- The LAST episode of the season should end on a note that could be resolved or a cliffhanger
- Genre tone: {genre}
- Use character names from the cast list naturally
- Do NOT repeat the episode title verbatim as the first words
- Do NOT mention ratings or runtime
- Each description must be distinct -- no template repetition

Output ONLY a JSON object mapping episode_id (integer) to description (string), plus a special key "_finale_line" containing the final 1-2 sentences of the last episode described in this batch for continuity.

Example format:
{{"123": "Description...", "124": "Description...", "_finale_line": "The finale's closing lines..."}}

No markdown fences, no explanation."""


def build_single_episode_retry_prompt(
    series_title: str,
    series_summary: str,
    genre: str,
    season: dict,
    episode: dict,
    cast_names: list[str],
    previous_episode_desc: str,
    next_episode_title: str,
) -> str:
    sn = season["season_number"]
    year = season["year"]
    cast_str = ", ".join(cast_names[:6]) if cast_names else "ensemble cast"
    prev_context = previous_episode_desc.strip() or "No prior episode summary available."
    next_context = next_episode_title.strip() or "Unknown"
    return f"""Write one polished episode description for a synthetic TV episode.

SERIES: "{series_title}" ({genre})
SERIES OVERVIEW:
{series_summary}

SEASON:
- Season {sn}
- Year: {year}
- Cast: {cast_str}

EPISODE METADATA:
- Episode id: {episode["episode_id"]}
- Episode number: {episode["episode_number"]}
- Title: {episode["title"]}
- Runtime: {episode["runtime_minutes"]} minutes
- Air date: {episode["air_date"]}
- Rating: {episode["rating"]:.1f}/10

CONTINUITY:
- Previous episode summary: {prev_context}
- Next episode title: {next_context}

Requirements:
- Write 2-3 sentences, 35-70 words total.
- It must sound like a real streaming-platform episode synopsis.
- Advance the season arc logically from the previous episode.
- Match the genre tone: {genre}.
- Use at least one cast name naturally when appropriate.
- Do not mention that the data is synthetic or fictional.
- Do not mention ratings or runtime.
- Do not start by repeating the episode title verbatim.

Output ONLY the episode description text."""


def build_episode_title_refresh_prompt(
    *,
    series_title: str,
    genre: str,
    episodes: list[dict],
) -> str:
    ep_lines = "\n".join(
        f'  [{ep["episode_id"]}] Ep{ep["episode_number"]}: current="{ep.get("title", "")}" | desc="{_normalise_text(ep.get("description", ""))}"'
        for ep in episodes
    )
    return f"""Write improved episode titles for a synthetic TV series. Replace generic placeholders like "Episode 4" with concise, specific titles that fit the description.

SERIES: "{series_title}" ({genre})

EPISODES:
{ep_lines}

Requirements:
- Write 2-6 word titles.
- Keep them specific to the episode description and tone.
- Do not use "Episode N" or any numbered placeholder format.
- Avoid spoilers and avoid repeating the series title verbatim.
- Use title case.

Output ONLY a JSON object mapping episode_id (integer) to title (string).
No markdown fences, no explanation."""


def _recover_missing_episode_descriptions(
    *,
    llm,
    t2_model: str,
    tracker: TokenTracker,
    all_series: list[dict],
    series_seasons: dict[int, list[dict]],
    season_episodes: dict[int, list[dict]],
    series_cast: dict[int, list[str]],
    all_series_summaries: dict[int, str],
    all_ep_descriptions: dict[int, str],
    dry_run: bool,
) -> int:
    recovered = 0
    for series_row in all_series:
        sid = int(series_row["series_id"])
        title = str(series_row.get("title", f"Series {sid}"))
        genre = str(series_row.get("genre", "Drama"))
        cast_names = series_cast.get(sid, [])
        series_summary = all_series_summaries.get(sid, f"A {genre} series titled '{title}'.")
        seasons = series_seasons.get(sid, [])
        for season in seasons:
            sn_id = int(season["season_id"])
            episodes = season_episodes.get(sn_id, [])
            missing = [ep for ep in episodes if ep["episode_id"] not in all_ep_descriptions]
            if not missing:
                continue
            print(f"    Recovery pass for '{title}' S{season['season_number']} ({len(missing)} missing episode descriptions)")
            for idx, episode in enumerate(missing, start=1):
                ep_num = int(episode["episode_number"])
                prev_desc = ""
                if ep_num > 1:
                    prev_candidates = [ep for ep in episodes if int(ep["episode_number"]) == ep_num - 1]
                    if prev_candidates:
                        prev_desc = all_ep_descriptions.get(int(prev_candidates[0]["episode_id"]), "")
                next_title = ""
                next_candidates = [ep for ep in episodes if int(ep["episode_number"]) == ep_num + 1]
                if next_candidates:
                    next_title = str(next_candidates[0].get("title", ""))

                for attempt in range(1, 4):
                    print(f"      Recover episode {episode['episode_id']} attempt {attempt}/3...", end=" ", flush=True)
                    prompt = build_single_episode_retry_prompt(
                        series_title=title,
                        series_summary=series_summary,
                        genre=genre,
                        season=season,
                        episode=episode,
                        cast_names=cast_names,
                        previous_episode_desc=prev_desc,
                        next_episode_title=next_title,
                    )
                    text = call_api(
                        llm,
                        t2_model,
                        prompt,
                        tracker,
                        dry_run=dry_run,
                        label=f"{title}/S{season['season_number']}/E{episode['episode_number']}/repair",
                    )
                    clean = _normalise_text(text)
                    if text and _is_valid_episode_description(clean):
                        all_ep_descriptions[int(episode["episode_id"])] = clean
                        recovered += 1
                        print("recovered")
                        break
                    print("invalid")
                    if not dry_run and attempt < 3:
                        time.sleep(SLEEP_BETWEEN_CALLS)
    return recovered


def _recover_missing_series_summaries(
    *,
    llm,
    t1_model: str,
    tracker: TokenTracker,
    all_series: list[dict],
    series_cast: dict[int, list[str]],
    series_seasons: dict[int, list[dict]],
    all_series_summaries: dict[int, str],
    dry_run: bool,
) -> int:
    recovered = 0
    for series_row in all_series:
        sid = int(series_row["series_id"])
        current = _normalise_text(all_series_summaries.get(sid, ""))
        if _is_valid_series_summary(current):
            continue

        title = str(series_row.get("title", f"Series {sid}"))
        cast_names = series_cast.get(sid, [])
        seasons = series_seasons.get(sid, [])
        print(f"    Recovery pass for series '{title}' ({len(seasons)} seasons)")

        prompts = [
            build_series_retry_prompt(series_row, cast_names, seasons),
            build_series_prompt(series_row, cast_names, seasons),
        ]
        recovered_here = False
        for attempt in range(1, 4):
            prompt = prompts[0] if attempt < 3 else prompts[1]
            print(f"      Recover series {sid} attempt {attempt}/3...", end=" ", flush=True)
            text = call_api(
                llm,
                t1_model,
                prompt,
                tracker,
                dry_run=dry_run,
                label=f"{title}/series_repair",
            )
            clean = _normalise_text(text)
            if text and _is_valid_series_summary(clean):
                all_series_summaries[sid] = clean
                recovered += 1
                recovered_here = True
                print("recovered")
                break
            print("invalid")
            if not dry_run and attempt < 3:
                time.sleep(SLEEP_BETWEEN_CALLS)
        if not recovered_here and sid not in all_series_summaries:
            all_series_summaries[sid] = current
    return recovered


def _refresh_generic_episode_titles(
    *,
    llm,
    t2_model: str,
    tracker: TokenTracker,
    all_series: list[dict],
    series_seasons: dict[int, list[dict]],
    season_episodes: dict[int, list[dict]],
    all_ep_descriptions: dict[int, str],
    all_ep_titles: dict[int, str],
    dry_run: bool,
    batch_size: int = 8,
) -> int:
    refreshed = 0
    for series_row in all_series:
        sid = int(series_row["series_id"])
        title = str(series_row.get("title", f"Series {sid}"))
        genre = str(series_row.get("genre", "Drama"))
        generic_eps: list[dict] = []
        for season in series_seasons.get(sid, []):
            sn_id = int(season["season_id"])
            for episode in season_episodes.get(sn_id, []):
                eid = int(episode["episode_id"])
                current_title = str(all_ep_titles.get(eid, episode.get("title", "")) or "")
                description = str(all_ep_descriptions.get(eid, "") or "")
                if _is_generic_episode_title(current_title) and _is_valid_episode_description(description):
                    generic_eps.append(
                        {
                            "episode_id": eid,
                            "episode_number": int(episode["episode_number"]),
                            "title": current_title,
                            "description": description,
                        }
                    )
        if not generic_eps:
            continue
        print(f"    Refreshing {len(generic_eps)} generic episode titles for '{title}'")
        for start in range(0, len(generic_eps), batch_size):
            batch = generic_eps[start:start + batch_size]
            text = call_api(
                llm,
                t2_model,
                build_episode_title_refresh_prompt(series_title=title, genre=genre, episodes=batch),
                tracker,
                dry_run=dry_run,
                label=f"{title}/episode_titles/{start // batch_size + 1}",
            )
            if not text:
                continue
            try:
                parsed = parse_json_response(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue
            for ep_id_str, new_title in parsed.items():
                try:
                    eid = int(ep_id_str)
                except (TypeError, ValueError):
                    continue
                clean_title = _normalise_text(new_title)
                if _is_valid_episode_title(clean_title):
                    old_title = str(all_ep_titles.get(eid, "") or "")
                    if _is_generic_episode_title(old_title):
                        all_ep_titles[eid] = clean_title
                        refreshed += 1
            if not dry_run:
                time.sleep(SLEEP_BETWEEN_CALLS)
    return refreshed


def _chunk_sequence(items: list[dict], chunk_size: int) -> list[list[dict]]:
    if chunk_size <= 0:
        return [items]
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def _continuity_tail(text: str) -> str:
    value = _normalise_text(text)
    if not value:
        return ""
    sentences = [s.strip() for s in value.split(".") if s.strip()]
    if not sentences:
        return value
    if len(sentences) >= 2:
        return ". ".join(sentences[-2:]) + "."
    return sentences[-1] + "."


# ═══════════════════════════════════════════════════════════════════════
# JSON PARSING HELPERS
# ═══════════════════════════════════════════════════════════════════════

def parse_json_response(text: str) -> dict | list:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return json.loads(text)


def call_api(llm, model: str, prompt: str, tracker: TokenTracker,
             dry_run: bool = False, label: str = "") -> str | None:
    """Single API call with transient-error backoff on the requested model."""
    if dry_run:
        print(f"  [DRY-RUN] Would call {model} for: {label}")
        print(f"  [DRY-RUN] Prompt ({len(prompt)} chars): {prompt[:200]}...")
        return None

    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.time()
            response = llm.generate(
                prompt,
                model=model,
                temperature=0.75,
                max_tokens=8192,
                timeout_sec=_API_TIMEOUT,
                max_attempts=2,
            )
            elapsed = time.time() - t0
            inp, out, cost = tracker.record(model, response)
            print(f"    {model}: {inp:,}in + {out:,}out = ${cost:.4f} ({elapsed:.1f}s)")
            return response.text
        except Exception as e:
            tracker.record_error(model)
            print(f"    API ERROR [{label}] attempt {attempt + 1}/{MAX_RETRIES} on {model}: {e}")
            if attempt < MAX_RETRIES - 1:
                delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
                print(f"    Retrying in {delay}s...")
                time.sleep(delay)
    return None


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate TV series and episode summaries")
    parser.add_argument("--base-dir", default=str(BASE_DIR), help="Dataset directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompts, make zero API calls")
    parser.add_argument("--series-limit", type=int, default=None,
                        help="Process only first N series (default: all)")
    parser.add_argument("--batch-verify", type=int, default=None,
                        help="Process exactly N series then exit (for inspection)")
    parser.add_argument("--tier1-model", default=TIER1_MODEL,
                        help=f"Model for series summaries (default: {TIER1_MODEL})")
    parser.add_argument("--tier2-model", default=TIER2_MODEL,
                        help=f"Model for episode batches (default: {TIER2_MODEL})")
    parser.add_argument("--episode-batch-size", type=int, default=_resolve_episode_batch_size(),
                        help=f"Max episodes per Tier 2 prompt (default: {_resolve_episode_batch_size()})")
    args = parser.parse_args()

    t1_model = args.tier1_model
    t2_model = args.tier2_model

    # Load .env
    base_dir = Path(args.base_dir).resolve()
    load_dotenv(BASE_DIR.parent / ".env")
    llm = get_llm_client() if not args.dry_run else None

    tracker = TokenTracker()

    # Load data
    print("Loading TV data...")
    data = load_data(base_dir)
    series_df = data["series_df"]
    episodes_df = data["episodes_df"]
    series_cast = data["series_cast"]
    series_seasons = data["series_seasons"]
    season_episodes = data["season_episodes"]

    if series_df.empty:
        print("tv_series.arrow / tv_series.csv not found or empty -- run generate_movies.py first")
        sys.exit(1)

    # Add columns if missing
    if "plot_summary" not in series_df.columns:
        series_df["plot_summary"] = ""
    if "description" not in episodes_df.columns:
        episodes_df["description"] = ""

    # Build existing summary maps for resume support
    existing_series_summaries: dict[int, str] = {}
    for _, row in series_df.iterrows():
        ps = row.get("plot_summary", "")
        if pd.notna(ps) and _is_valid_series_summary(ps):
            existing_series_summaries[int(row["series_id"])] = _normalise_text(ps)

    existing_ep_descriptions: dict[int, str] = {}
    if not episodes_df.empty and "description" in episodes_df.columns:
        for _, row in episodes_df.iterrows():
            d = row.get("description", "")
            if pd.notna(d) and _is_valid_episode_description(d):
                existing_ep_descriptions[int(row["episode_id"])] = _normalise_text(d)
    all_ep_titles: dict[int, str] = {}
    if not episodes_df.empty and "title" in episodes_df.columns:
        for _, row in episodes_df.iterrows():
            all_ep_titles[int(row["episode_id"])] = _normalise_text(row.get("title", ""))

    # Determine which series to process
    all_series = series_df.to_dict("records")
    limit = args.batch_verify or args.series_limit
    if limit:
        all_series = all_series[:limit]

    n_series = len(all_series)
    save_every_series = _resolve_tv_save_every_series(n_series)
    episode_batch_size = max(1, int(args.episode_batch_size))
    n_seasons_total = sum(len(series_seasons.get(int(s["series_id"]), [])) for s in all_series)
    print(f"  Series to process: {n_series} | Seasons: {n_seasons_total}")
    print(f"  Already have series summary: {len(existing_series_summaries)}")
    print(f"  Already have episode descriptions: {len(existing_ep_descriptions)}")
    print(f"  Checkpoint save cadence: every {save_every_series} series")
    if args.dry_run:
        print("  [DRY-RUN MODE -- no API calls will be made]")

    # ── TIER 1: Series summaries ─────────────────────────────────────────
    print(f"\n{'='*60}\nTIER 1: Series summaries ({t1_model})\n{'='*60}")

    all_series_summaries: dict[int, str] = dict(existing_series_summaries)

    for i, series_row in enumerate(all_series):
        sid = int(series_row["series_id"])
        title = series_row.get("title", f"Series {sid}")

        if sid in all_series_summaries:
            print(f"  [{i+1}/{n_series}] '{title}' -- already has summary, skipping")
            continue

        print(f"  [{i+1}/{n_series}] '{title}'...")

        cast_names = series_cast.get(sid, [])
        seasons = series_seasons.get(sid, [])
        text = ""
        for attempt in range(1, 4):
            prompt = (
                build_series_prompt(series_row, cast_names, seasons)
                if attempt == 1
                else build_series_retry_prompt(series_row, cast_names, seasons)
            )
            text = call_api(
                llm,
                t1_model,
                prompt,
                tracker,
                dry_run=args.dry_run,
                label=f"series/{title}",
            )
            if text and _is_valid_series_summary(text):
                all_series_summaries[sid] = _normalise_text(text)
                break
            if args.dry_run:
                break
            if attempt < 3:
                print(f"    Invalid series summary for series {sid}, retrying ({attempt + 1}/3)...")
                time.sleep(SLEEP_BETWEEN_CALLS)
        else:
            if not args.dry_run:
                print(f"    FAILED or invalid summary for series {sid}, using empty string")
                all_series_summaries[sid] = ""

        if not args.dry_run:
            time.sleep(SLEEP_BETWEEN_CALLS)

    # Write Tier 1 results back to series_df
        if not args.dry_run and (((i + 1) % save_every_series == 0) or ((i + 1) == n_series)):
            _save_tv_series_outputs(base_dir, series_df, all_series_summaries)
            print(f"\n  Saved series summaries -> {base_dir / 'tv_series.csv'}")

    # ── TIER 2: Episode descriptions ──────────────────────────────────────
    print(f"\n{'='*60}\nTIER 2: Episode descriptions ({t2_model})\n{'='*60}")

    all_ep_descriptions: dict[int, str] = dict(existing_ep_descriptions)

    for i, series_row in enumerate(all_series):
        sid = int(series_row["series_id"])
        title = series_row.get("title", f"Series {sid}")
        genre = str(series_row.get("genre", "Drama"))
        cast_names = series_cast.get(sid, [])
        series_summary = all_series_summaries.get(sid, f"A {genre} series titled '{title}'.")
        seasons = series_seasons.get(sid, [])

        print(f"\n  Series [{i+1}/{n_series}]: '{title}' ({len(seasons)} seasons)")

        previous_finale_line = ""

        for sn_idx, season in enumerate(seasons):
            sn_id = season["season_id"]
            sn_num = season["season_number"]
            episodes = season_episodes.get(sn_id, [])

            if not episodes:
                continue

            # Check if all episodes in this season already have descriptions
            ep_ids = [ep["episode_id"] for ep in episodes]
            already_done = sum(1 for eid in ep_ids if eid in all_ep_descriptions)
            if already_done == len(ep_ids):
                print(f"    S{sn_num}: all {len(episodes)} eps already described, skipping")
                # Still need to set continuity from last episode
                last_ep_id = ep_ids[-1]
                last_desc = all_ep_descriptions.get(last_ep_id, "")
                if last_desc:
                    previous_finale_line = _continuity_tail(last_desc)
                continue

            missing_eps = [ep for ep in episodes if ep["episode_id"] not in all_ep_descriptions]
            episode_batches = _chunk_sequence(missing_eps, episode_batch_size)

            for batch_index, batch in enumerate(episode_batches, start=1):
                batch_ep_ids = [ep["episode_id"] for ep in batch]
                batch_note = previous_finale_line
                for semantic_attempt in range(3):
                    if semantic_attempt == 0:
                        print(
                            f"    S{sn_num} batch {batch_index}/{len(episode_batches)} ({len(batch)} eps)...",
                            end=" ",
                            flush=True,
                        )
                    else:
                        print(
                            f"    S{sn_num} batch {batch_index}/{len(episode_batches)} retry {semantic_attempt + 1}/3...",
                            end=" ",
                            flush=True,
                        )
                    prompt = build_season_prompt(
                        series_title=title,
                        series_summary=series_summary,
                        genre=genre,
                        season=season,
                        episodes=batch,
                        cast_names=cast_names,
                        continuity_note=batch_note,
                    )

                    text = call_api(
                        llm,
                        t2_model,
                        prompt,
                        tracker,
                        dry_run=args.dry_run,
                        label=f"{title}/S{sn_num}/batch{batch_index}",
                    )

                    if text:
                        try:
                            parsed = parse_json_response(text)
                            finale_line = _normalise_text(parsed.pop("_finale_line", ""))

                            for ep_id_str, desc in parsed.items():
                                try:
                                    eid = int(ep_id_str)
                                except (ValueError, TypeError):
                                    continue
                                clean_desc = _normalise_text(desc)
                                if _is_valid_episode_description(clean_desc):
                                    all_ep_descriptions[eid] = clean_desc

                            got = sum(1 for eid in batch_ep_ids if eid in all_ep_descriptions)
                            print(f"got {got}/{len(batch)}")

                            if got < len(batch):
                                values = [_normalise_text(v) for v in parsed.values() if _is_valid_episode_description(v)]
                                if len(values) == len(batch):
                                    for ep, desc in zip(batch, values):
                                        eid = ep["episode_id"]
                                        if eid not in all_ep_descriptions:
                                            all_ep_descriptions[eid] = desc
                                    got = sum(1 for eid in batch_ep_ids if eid in all_ep_descriptions)
                                    print(
                                        f"    Positional fallback applied for S{sn_num} batch {batch_index} ({got}/{len(batch)})"
                                    )

                            if got == len(batch):
                                if finale_line:
                                    previous_finale_line = finale_line
                                else:
                                    previous_finale_line = _continuity_tail(
                                        all_ep_descriptions.get(batch_ep_ids[-1], "")
                                    )
                                break
                            if semantic_attempt < 2:
                                print(f"    Incomplete batch coverage for S{sn_num} batch {batch_index}, retrying...")

                        except json.JSONDecodeError as e:
                            print(f"    JSON PARSE ERROR for S{sn_num} batch {batch_index}: {e}")
                            raw_dir = BASE_DIR / "_dev"
                            raw_dir.mkdir(exist_ok=True)
                            raw_path = raw_dir / f"tv_s{sid}_sn{sn_num}_batch{batch_index}_raw.txt"
                            raw_path.write_text(text, encoding="utf-8")
                            print(f"    Raw saved -> {raw_path}")
                    else:
                        print("FAILED")

                    if not args.dry_run:
                        time.sleep(SLEEP_BETWEEN_CALLS)

            if ep_ids:
                last_ep_id = ep_ids[-1]
                last_desc = all_ep_descriptions.get(last_ep_id, "")
                if last_desc:
                    previous_finale_line = _continuity_tail(last_desc)

        # Incremental save after each series
        if not args.dry_run and (((i + 1) % save_every_series == 0) or ((i + 1) == n_series)):
            _save_episode_outputs(base_dir, episodes_df, all_ep_descriptions, all_ep_titles)
            print(f"  Saved episode summaries -> {base_dir / 'episodes.csv'}")

    # ── FINAL SAVE ────────────────────────────────────────────────────────
    if not args.dry_run:
        missing_series_ids = [
            int(row["series_id"])
            for _, row in series_df.iterrows()
            if not _is_valid_series_summary(all_series_summaries.get(int(row["series_id"]), ""))
        ]
        if missing_series_ids:
            print(f"\n  Missing series summaries before final validation: {len(missing_series_ids)}")
            recovered_series = _recover_missing_series_summaries(
                llm=llm,
                t1_model=t1_model,
                tracker=tracker,
                all_series=all_series,
                series_cast=series_cast,
                series_seasons=series_seasons,
                all_series_summaries=all_series_summaries,
                dry_run=args.dry_run,
            )
            print(f"  Recovery filled {recovered_series} series summaries")

        missing_ep_ids = [
            int(row["episode_id"])
            for _, row in episodes_df.iterrows()
            if int(row["episode_id"]) not in all_ep_descriptions
        ]
        if missing_ep_ids:
            print(f"\n  Missing episode descriptions before final validation: {len(missing_ep_ids)}")
            recovered = _recover_missing_episode_descriptions(
                llm=llm,
                t2_model=t2_model,
                tracker=tracker,
                all_series=all_series,
                series_seasons=series_seasons,
                season_episodes=season_episodes,
                series_cast=series_cast,
                all_series_summaries=all_series_summaries,
                all_ep_descriptions=all_ep_descriptions,
                dry_run=args.dry_run,
            )
            print(f"  Recovery filled {recovered} episode descriptions")

        refreshed_titles = _refresh_generic_episode_titles(
            llm=llm,
            t2_model=t2_model,
            tracker=tracker,
            all_series=all_series,
            series_seasons=series_seasons,
            season_episodes=season_episodes,
            all_ep_descriptions=all_ep_descriptions,
            all_ep_titles=all_ep_titles,
            dry_run=args.dry_run,
        )
        if refreshed_titles:
            print(f"  Refreshed {refreshed_titles} generic episode titles")

        print(f"\n{'='*60}")
        _save_tv_series_outputs(base_dir, series_df, all_series_summaries)
        _save_episode_outputs(base_dir, episodes_df, all_ep_descriptions, all_ep_titles)

        filled_series = int(series_df["plot_summary"].fillna("").map(_is_valid_series_summary).sum())
        filled_eps = int(episodes_df["description"].fillna("").map(_is_valid_episode_description).sum())
        print(f"  Series summaries: {filled_series}/{len(series_df)}")
        print(f"  Episode descriptions: {filled_eps}/{len(episodes_df)}")
        if PIPELINE_MODE == "research":
            missing_series = int(len(series_df) - filled_series)
            missing_eps = int(len(episodes_df) - filled_eps)
            if missing_series or missing_eps:
                raise RuntimeError(
                    f"Research-mode TV summary generation incomplete: missing {missing_series} series summaries and {missing_eps} episode descriptions"
                )

    if args.batch_verify:
        print(f"\n[batch-verify] Processed {args.batch_verify} series. Inspect output before full run.")

    print(tracker.summary())


if __name__ == "__main__":
    main()
