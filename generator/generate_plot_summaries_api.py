"""
Generate Plot Summaries via API / unified LLM provider.
======================================================
Fills the `plot_summary` column in movie.arrow / movie.csv using genre, cast,
director, and keywords as context.

Run AFTER generate_movies.py (needs assembled movie data).

Usage:
    python generate_plot_summaries_api.py          # interactive
    python generate_plot_summaries_api.py --auto    # skip prompts
    python generate_plot_summaries_api.py --model gemini-3.1-flash-lite
"""
import argparse
import os, json, sys, time, csv, re
from pathlib import Path
import pandas as pd

from dotenv import load_dotenv
from feather_sink import df_to_arrow, read_table
from llm_provider import get_llm_client
from model_defaults import model_for_role

MODEL = model_for_role("plot_summaries")

BASE_DIR = Path(__file__).parent
BATCH_SIZE = 12
MAX_RETRIES = 5
_RETRY_DELAYS = [4, 4, 4, 15, 20]
_API_TIMEOUT = 70  # seconds -- hard limit per call
PIPELINE_MODE = str(os.environ.get("DATA_SYS_PIPELINE_MODE", "research") or "research").strip().lower()
TARGETED_REWRITE_ATTEMPTS = 4


def _resolve_plot_save_every_batches(total_movies: int) -> int:
    raw = str(os.environ.get("DATA_SYS_PLOT_SAVE_EVERY_BATCHES", "") or "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except Exception:
            pass
    if total_movies >= 100000:
        return 25
    if total_movies >= 5000:
        return 1
    if total_movies >= 20000:
        return 5
    return 1


def _resolve_plot_batch_size(total_movies: int) -> int:
    raw = str(os.environ.get("DATA_SYS_PLOT_BATCH_SIZE", "") or "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except Exception:
            pass
    if total_movies >= 20000:
        return 16
    if total_movies >= 5000:
        return 8
    if total_movies >= 500:
        return 10
    return BATCH_SIZE


# ═══════════════════════════════════════════════════════════════════════
# TOKEN TRACKING
# ═══════════════════════════════════════════════════════════════════════

class TokenTracker:
    """Track cumulative token usage and cost across all API calls."""
    def __init__(self, model_name: str = MODEL):
        self.model_name = model_name
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.calls = 0
        self.errors = 0

    def record(self, response):
        """Extract and record token usage from LLMResponse.

        Called immediately after API response arrives, so even if
        JSON parsing fails, we still have accurate token counts.
        """
        inp = getattr(response, "input_tokens", 0) or 0
        out = getattr(response, "output_tokens", 0) or 0
        cost = getattr(response, "cost_usd", 0.0) or 0.0

        self.total_input_tokens += inp
        self.total_output_tokens += out
        self.total_cost += cost
        self.calls += 1

        return inp, out, cost

    def record_error(self):
        self.errors += 1

    def summary(self) -> str:
        return (
            f"\n{'='*60}\n"
            f"  TOKEN USAGE SUMMARY\n"
            f"{'='*60}\n"
            f"  Model:         {self.model_name}\n"
            f"  API calls:     {self.calls} ({self.errors} errors)\n"
            f"  Input tokens:  {self.total_input_tokens:,}\n"
            f"  Output tokens: {self.total_output_tokens:,}\n"
            f"  Total tokens:  {self.total_input_tokens + self.total_output_tokens:,}\n"
            f"  Total cost:    ${self.total_cost:.4f}\n"
            f"{'='*60}"
        )


# ═══════════════════════════════════════════════════════════════════════
# PROMPTS
# ═══════════════════════════════════════════════════════════════════════

def _levenshtein_distance(s1: str, s2: str) -> int:
    """Simple Levenshtein distance for near-duplicate detection."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def _extract_key_phrases(summary: str) -> list:
    """Extract 2-3 key phrases from a summary for the anti-duplication blacklist."""
    phrases = []
    sentences = summary.split('.')
    if sentences:
        # First sentence opening (up to 8 words)
        first = sentences[0].strip()
        words = first.split()[:8]
        if len(words) >= 3:
            phrases.append(' '.join(words))
    # Any distinctive multi-word fragment from later in the text
    if len(sentences) > 1:
        second = sentences[1].strip()
        words2 = second.split()[:6]
        if len(words2) >= 3:
            phrases.append(' '.join(words2))
    return phrases


def _normalise_summary_text(text: object) -> str:
    return " ".join(str(text or "").split()).strip()


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'/-]*", str(text or "")))


def _split_sentences(text: str) -> list[str]:
    value = _normalise_summary_text(text)
    if not value:
        return []
    parts = re.split(r"(?<=[.!?])\s+", value)
    return [part.strip() for part in parts if part.strip()]


_GENERIC_STORY_FRAGMENTS_BY_GENRE = {
    "Action": (
        "mercenary",
        "ancient artifact",
        "hostage situation",
        "high-speed chase",
        "double-cross",
        "future city",
        "warehouse",
        "mission is compromised",
        "survive the crossfire",
        "team of mercenaries",
    ),
    "Adventure": (
        "ancient artifact",
        "hidden temple",
        "dangerous mission",
        "mountain escape",
        "team of explorers",
        "uncharted wilderness",
    ),
    "Family": (
        "hidden world",
        "anthropomorphic animal",
        "value of family and friendship",
        "neighborhood mystery",
        "summer reunion",
        "simple family reunion",
    ),
    "Fantasy": (
        "otherworldly origins",
        "chosen one",
        "hidden world",
        "magical journey",
        "artifact of power",
    ),
    "Thriller": (
        "shadow network",
        "rival syndicate",
        "global blackout",
        "deadly mission",
        "race against time",
    ),
}

_GENERIC_SUMMARY_START_PATTERNS = (
    r"^(a|an)\s+reclusive\s+[a-z-]+",
    r"^(a|an)\s+dedicated\s+[a-z-]+",
    r"^(a|an)\s+struggling\s+[a-z-]+",
    r"^(a|an)\s+dysfunctional\s+family\b",
    r"^(a|an)\s+group of\s+[a-z-]+",
    r"^(a|an)\s+team of\s+[a-z-]+",
    r"^a high-stakes\s+[a-z-]+\s+following\b",
    r"^a visually striking\s+[a-z-]+\s+film\b",
    r"^an investigative look at\b",
    r"^an intrepid\s+[a-z-]+\s+leads\b",
    r"^in a near-future\b",
    r"^in a vibrant animated world\b",
)


def _looks_like_placeholder_summary(summary: str, genre: str | None = None) -> bool:
    text = _normalise_summary_text(summary)
    low = text.lower()
    if not text:
        return True
    sentence_count = len(_split_sentences(text))
    if sentence_count < 2 or sentence_count > 4:
        return True
    if _word_count(text) < 28 or _word_count(text) > 110:
        return True
    if "synthetic" in low:
        return True
    if low.startswith(("a ", "an ")) and " film from " in low and re.search(r"\(\d{4}\)", low):
        return True
    if re.match(r"^(a|an)\s+[a-z-]+\s+.+?\s+film from\s+.+?\(\d{4}\)", low):
        return True
    if "rated " in low and _word_count(text) < 28:
        return True
    if any(re.search(pattern, low) for pattern in _GENERIC_SUMMARY_START_PATTERNS):
        return True
    genre_key = str(genre or "").strip()
    if genre_key:
        markers = _GENERIC_STORY_FRAGMENTS_BY_GENRE.get(genre_key, ())
        if markers:
            hits = sum(1 for marker in markers if marker in low)
            if hits >= 3:
                return True
    if re.search(r"\b(?:learn|discovers?) the value of family and friendship\b", low):
        return True
    if re.search(r"\ba disgraced [a-z-]+ is hired to\b", low):
        return True
    if re.search(r"\bteam of mercenar(?:y|ies)\b", low):
        return True
    return False


def _is_acceptable_summary(summary: str, *, genre: str | None = None) -> bool:
    text = _normalise_summary_text(summary)
    return bool(text) and not _looks_like_placeholder_summary(text, genre=genre)


_PLOT_CUE_DROP_EXACT = {
    "action words",
    "abstract nouns",
    "mythic words",
    "technology words",
    "celestial words",
}

_PLOT_CUE_DROP_FRAGMENTS = (
    "dynamic-lighting",
    "layered-composition",
    "fluid-dynamics",
    "cel-shaded-world",
    "hand-drawn-aesthetic",
    "stop-motion-creature",
    "abstract-visuals",
    "voxel-environment",
    "particle-effects",
    "stylized-character-design",
    "morphing-geometry",
)

_PLOT_CUE_GENRE_GATES = {
    "geological survey": {"Adventure", "Documentary", "Disaster", "Experimental", "History", "Short", "Western"},
    "interview transcript": {"Biography", "Documentary", "History", "Mystery"},
}


def _prepare_story_cues(raw_keywords: object, genre: str | None = None) -> str:
    parts = re.split(r"\s*,\s*", str(raw_keywords or ""))
    cleaned: list[str] = []
    seen: set[str] = set()
    for part in parts:
        value = " ".join(str(part or "").split()).strip(" ,;")
        if not value:
            continue
        low = value.lower()
        if low in _PLOT_CUE_DROP_EXACT:
            continue
        if any(fragment in low for fragment in _PLOT_CUE_DROP_FRAGMENTS):
            continue
        natural = value.replace("-", " ")
        natural = re.sub(r"\s+", " ", natural).strip()
        low_natural = natural.lower()
        allowed_genres = _PLOT_CUE_GENRE_GATES.get(low_natural)
        if allowed_genres and genre and str(genre) not in allowed_genres:
            continue
        if low_natural in seen:
            continue
        seen.add(low_natural)
        cleaned.append(natural)
    return ", ".join(cleaned[:6])


def _extract_plain_text_response(text: object) -> str:
    value = str(text or "").strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    return _normalise_summary_text(value)


def build_summary_prompt(movies_batch, blacklist=None):
    """Build prompt for generating plot summaries for a batch of movies."""
    movie_lines = []
    for m in movies_batch:
        mid = m["title_id"]
        title = m["title"]
        genre = m["genre"]
        year = m["year"]
        director = m.get("director", "Unknown")
        actors = m.get("actors", "")
        keywords = _prepare_story_cues(m.get("keywords", ""), genre=genre)
        tier = m.get("production_tier", "Mid")
        tagline = str(m.get("tagline", "") or "")
        country = str(m.get("country", "") or "")
        language = str(m.get("language", "") or "")

        movie_lines.append(
            f'[{mid}] "{title}" ({year}) | Genre: {genre} | Tier: {tier} '
            f'| Country: {country} | Language: {language} | Tagline: {tagline} '
            f'| Director: {director} | Cast: {actors[:150]} '
            f'| Keywords: {keywords}'
        )

    movies_block = "\n".join(movie_lines)

    blacklist_block = ""
    if blacklist:
        # Take last 40 phrases max to keep prompt manageable
        recent = blacklist[-40:]
        blacklist_block = f"""\n6. AVOID these phrasings used in prior batches (do NOT repeat them):\n   {'; '.join(recent)}\n7. Each summary MUST start with a DIFFERENT opening word/phrase than any other summary"""

    return f"""Write brief plot summaries for these {len(movies_batch)} movies. Each summary should be 2-3 sentences that describe the basic plot, tone, and themes -- as if it's the "Overview" on a movie database page.

=== MOVIES ===
{movies_block}

Rules:
1. Each summary is exactly 2-3 sentences, 40-85 words
2. Match the genre and tone (horror = dark, comedy = light, etc.)
3. Use the cast/story cues/director to create coherent plot details
4. Tell a concrete mini-story with an inciting incident, pressure, or turning point
5. Each summary must be UNIQUE and specific to the movie{blacklist_block}
6. Do NOT start with formulas like "A sweeping X film from Y"
7. Do NOT mention production tier, country, rating, or that it is "a [genre] film"
8. Ground the summary in at least two of the provided story cues or title-specific clues
9. Do NOT introduce fantasy, magic, supernatural, or mythical elements unless the genre or keywords clearly support them
10. Do NOT introduce advanced technology, space travel, AI, or futuristic devices unless the genre or keywords clearly support them
11. Translate the cues into natural prose; do NOT copy awkward hyphenated cue phrases verbatim if a normal phrasing exists
12. Avoid stock skeletons like "a disgraced operative is hired", "a team of mercenaries", "ancient artifact + betrayal + warehouse", or "they learn the value of family and friendship"
13. Make each synopsis feel specific to the title, setting, tagline, and cues rather than a reusable genre boilerplate
14. Do NOT begin with weak generic openings like "A high-stakes thriller following...", "An investigative look at...", or "A dysfunctional family attempts..."
15. Use the tagline as an emotional clue, but do not quote or paraphrase it directly

Output a JSON array of objects: {{"title_id": <id>, "plot_summary": "<summary>"}}
No markdown fences, no explanation."""


def build_single_summary_rewrite_prompt(movie: dict, weak_summary: str = "") -> str:
    title = str(movie.get("title", "") or "")
    genre = str(movie.get("genre", "") or "")
    year = str(movie.get("year", "") or "")
    country = str(movie.get("country", "") or "")
    language = str(movie.get("language", "") or "")
    tier = str(movie.get("production_tier", "") or "")
    tagline = str(movie.get("tagline", "") or "")
    director = str(movie.get("director", "") or "Unknown")
    cast = str(movie.get("actors", "") or "")[:180]
    keywords = _prepare_story_cues(movie.get("keywords", ""), genre=genre)
    weak_block = f"\nWEAK SUMMARY TO REPLACE:\n{weak_summary}\n" if weak_summary else ""
    return f"""Write one polished movie plot summary for a synthetic film database entry.

MOVIE:
- Title: {title}
- Year: {year}
- Genre: {genre}
- Country: {country}
- Language: {language}
- Production tier: {tier}
- Tagline: {tagline}
- Director: {director}
- Cast: {cast}
- Story cues: {keywords}
{weak_block}
Requirements:
- Write exactly 2-3 sentences, 40-85 words total.
- Start with a concrete incident, discovery, betrayal, dilemma, or collision specific to this movie.
- Use at least two story-specific clues from the title, tagline, cast context, or story cues.
- Match the genre cleanly: no supernatural elements unless supported; no futuristic technology unless supported.
- Do not use generic openings like "A high-stakes thriller following...", "An investigative look at...", or "A dysfunctional family attempts...".
- Do not mention that the data is synthetic.
- Do not mention ratings, runtime, or production tier.
- Do not restate the title verbatim as the opening words.

Output ONLY the summary text."""


def parse_json_response(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return json.loads(text)


def _save_movie_outputs(base_dir: Path, movies_df: pd.DataFrame, all_summaries: dict[int, str]) -> None:
    movies_df["plot_summary"] = movies_df["title_id"].map(all_summaries).fillna("")
    df_to_arrow(movies_df, str(base_dir / "movie.arrow"), table_name="movie")
    movies_df.to_csv(base_dir / "movie.csv", index=False)


def _save_flat_and_analysis_outputs(base_dir: Path, flat_df: pd.DataFrame | None, movies_df: pd.DataFrame, all_summaries: dict[int, str]) -> None:
    if flat_df is None or flat_df.empty:
        return
    flat_df = flat_df.copy()
    flat_df["description"] = flat_df["title_id"].map(all_summaries).fillna("")
    df_to_arrow(flat_df, str(base_dir / "movies_flat.arrow"))
    flat_df.to_csv(base_dir / "movies_flat.csv", index=False)

    analysis = flat_df.copy()
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
        if col in movies_df.columns:
            analysis[col] = movies_df[col].values
    df_to_arrow(analysis, str(base_dir / "movies_analysis.arrow"))
    analysis.to_csv(base_dir / "movies_analysis.csv", index=False)


def _generate_batch_summaries(
    *,
    llm,
    model: str,
    batch: list[dict],
    batch_num_label: str,
    tracker: TokenTracker,
    phrase_blacklist: list[str] | None,
) -> dict[int, str]:
    batch_ids = {int(m["title_id"]) for m in batch}
    batch_genre_by_id = {int(m["title_id"]): str(m.get("genre", "") or "") for m in batch}
    prompt = build_summary_prompt(batch, blacklist=phrase_blacklist if phrase_blacklist else None)
    best_valid: dict[int, str] = {}

    for retry in range(MAX_RETRIES):
        t0 = time.time()
        try:
            response = llm.generate(
                prompt,
                model=model,
                temperature=0.6,
                max_tokens=16384,
                timeout_sec=_API_TIMEOUT,
                max_attempts=1,
            )
            elapsed = time.time() - t0
        except Exception as e:
            print(f"  API ERROR (retry {retry+1}/{MAX_RETRIES}): {e}")
            tracker.record_error()
            if retry < MAX_RETRIES - 1:
                delay = _RETRY_DELAYS[retry]
                print(f"  Retrying in {delay}s...")
                time.sleep(delay)
            continue

        inp, out, cost = tracker.record(response)
        print(f"  Tokens: {inp:,} in + {out:,} out = ${cost:.4f} | {elapsed:.1f}s")

        try:
            summaries = parse_json_response(response.text)
            valid_this_response: dict[int, str] = {}
            for s in summaries:
                mid = s.get("title_id")
                ps = _normalise_summary_text(s.get("plot_summary", ""))
                try:
                    mid = int(mid)
                except Exception:
                    mid = None
                if mid and _is_acceptable_summary(ps, genre=batch_genre_by_id.get(mid, "")):
                    valid_this_response[mid] = ps

            for mid, ps in valid_this_response.items():
                best_valid.setdefault(mid, ps)

            missing = batch_ids - set(best_valid.keys())

            if len(missing) > 0 and len(summaries) == len(batch):
                positional_mapped = 0
                for s, m in zip(summaries, batch):
                    ps = _normalise_summary_text(s.get("plot_summary", ""))
                    mid = int(m["title_id"])
                    if _is_acceptable_summary(ps, genre=str(m.get("genre", "") or "")) and mid not in best_valid:
                        best_valid[mid] = ps
                        positional_mapped += 1
                if positional_mapped > 0:
                    print(f"  Positional fallback: mapped {positional_mapped} summaries")
                    missing = batch_ids - set(best_valid.keys())

            if len(missing) > 0 and retry < MAX_RETRIES - 1:
                print(f"  Missing {len(missing)} summaries, retrying...")
                continue

            print(f"  Got {len(batch_ids - missing)}/{len(batch_ids)} valid summaries")
            break

        except Exception as e:
            print(f"  JSON PARSE ERROR (retry {retry+1}): {e}")
            raw_path = BASE_DIR / f"_dev/plots_batch{batch_num_label}_raw.txt"
            os.makedirs(BASE_DIR / "_dev", exist_ok=True)
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            print(f"  Raw response saved -> {raw_path}")
            if retry < MAX_RETRIES - 1:
                delay = _RETRY_DELAYS[retry]
                time.sleep(delay)

    missing = batch_ids - set(best_valid.keys())
    if missing and len(batch) > 1:
        missing_batch = [m for m in batch if int(m["title_id"]) in missing]
        if len(missing_batch) == len(batch):
            print(f"  Splitting failed batch {batch_num_label} into smaller salvage batches...")
        else:
            print(f"  Salvaging {len(missing_batch)} unresolved summaries from batch {batch_num_label}...")
        split_at = max(1, len(missing_batch) // 2)
        left = missing_batch[:split_at]
        right = missing_batch[split_at:]
        if left:
            best_valid.update(
                _generate_batch_summaries(
                    llm=llm,
                    model=model,
                    batch=left,
                    batch_num_label=f"{batch_num_label}a",
                    tracker=tracker,
                    phrase_blacklist=phrase_blacklist,
                )
            )
        if right:
            best_valid.update(
                _generate_batch_summaries(
                    llm=llm,
                    model=model,
                    batch=right,
                    batch_num_label=f"{batch_num_label}b",
                    tracker=tracker,
                    phrase_blacklist=phrase_blacklist,
                )
            )

    return best_valid


def _recover_plot_summaries(
    *,
    llm,
    model: str,
    tracker: TokenTracker,
    movies_data: list[dict],
    all_summaries: dict[int, str],
    base_dir: Path,
    movies_df: pd.DataFrame,
    flat_df: pd.DataFrame | None,
) -> int:
    unresolved = [
        movie for movie in movies_data
        if not _is_acceptable_summary(all_summaries.get(int(movie["title_id"]), ""), genre=str(movie.get("genre", "") or ""))
    ]
    if not unresolved:
        return 0
    recovered = 0
    print(f"\n  Targeted recovery for {len(unresolved)} weak/missing plot summaries...")
    for idx, movie in enumerate(unresolved, start=1):
        mid = int(movie["title_id"])
        title = str(movie.get("title", "") or f"title_id={mid}")
        weak_summary = all_summaries.get(mid, "")
        print(f"    [{idx}/{len(unresolved)}] {title}...", end=" ", flush=True)
        for attempt in range(TARGETED_REWRITE_ATTEMPTS):
            t0 = time.time()
            try:
                response = llm.generate(
                    build_single_summary_rewrite_prompt(movie, weak_summary=weak_summary),
                    model=model,
                    temperature=0.55,
                    max_tokens=1200,
                    timeout_sec=_API_TIMEOUT,
                    max_attempts=1,
                )
                elapsed = time.time() - t0
                inp, out, cost = tracker.record(response)
                print(f"tokens {inp:,}+{out:,} (${cost:.4f}, {elapsed:.1f}s)", end=" ", flush=True)
            except Exception as exc:
                tracker.record_error()
                if attempt < TARGETED_REWRITE_ATTEMPTS - 1:
                    delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
                    print(f"api error, retrying in {delay}s...", end=" ", flush=True)
                    time.sleep(delay)
                    continue
                print(f"failed: {exc}")
                break
            candidate = _extract_plain_text_response(getattr(response, "text", ""))
            if _is_acceptable_summary(candidate, genre=str(movie.get("genre", "") or "")):
                all_summaries[mid] = candidate
                recovered += 1
                print("ok")
                break
            if attempt < TARGETED_REWRITE_ATTEMPTS - 1:
                weak_summary = candidate or weak_summary
                print("weak, retrying...", end=" ", flush=True)
                time.sleep(2)
            else:
                print("still weak")
        if recovered > 0 and recovered % 5 == 0:
            _save_movie_outputs(base_dir, movies_df, all_summaries)
            _save_flat_and_analysis_outputs(base_dir, flat_df, movies_df, all_summaries)
            print(f"    Recovery checkpoint saved after {recovered} repaired summaries")
        time.sleep(1)
    return recovered


def main():
    parser = argparse.ArgumentParser(description="Generate plot summaries via the configured LLM provider.")
    parser.add_argument("--auto", action="store_true", help="Skip interactive prompts")
    parser.add_argument("--base-dir", default=str(BASE_DIR), help="Dataset directory")
    parser.add_argument("--model", default=MODEL,
                        help=f"LLM model to use (default: {MODEL})")
    args = parser.parse_args()

    # Override module-level MODEL with CLI arg
    _model = args.model

    # Load API key from .env or environment
    load_dotenv(BASE_DIR.parent / ".env")
    llm = get_llm_client()
    tracker = TokenTracker()

    # Load movie + flat data
    base_dir = Path(args.base_dir).resolve()
    movie_path = base_dir / "movie"
    flat_path = base_dir / "movies_flat"

    movies_df = read_table(str(movie_path), "movie")
    if movies_df.empty:
        print("movie.arrow / movie.csv not found -- run generate_movies.py first")
        return
    flat_df = read_table(str(flat_path))
    if flat_df.empty:
        flat_df = None

    # Build movie context for prompts
    movies_data = []
    flat_title_ids = set(flat_df["title_id"].values) if flat_df is not None else set()
    for _, row in movies_df.iterrows():
        mid = row["title_id"]
        m = {
            "title_id": mid,
            "title": row["title"],
            "genre": row["genre"],
            "year": row["year"],
            "production_tier": row.get("production_tier", "Mid"),
            "country": row.get("country", ""),
            "language": row.get("language", ""),
            "tagline": row.get("tagline", ""),
        }

        # Get director + actors + keywords from flat table
        if flat_df is not None and mid in flat_title_ids:
            flat_row = flat_df[flat_df["title_id"] == mid].iloc[0]
            m["director"] = str(flat_row.get("director", "Unknown"))
            m["actors"] = str(flat_row.get("actors", ""))
            m["keywords"] = str(flat_row.get("keywords", ""))
        else:
            m["director"] = "Unknown"
            m["actors"] = ""
            m["keywords"] = ""

        movies_data.append(m)

    # Check which movies already have summaries
    existing_summaries = {}
    invalid_existing = 0
    if "plot_summary" in movies_df.columns:
        for _, row in movies_df.iterrows():
            ps = row.get("plot_summary")
            if pd.notna(ps) and str(ps).strip():
                clean = _normalise_summary_text(ps)
                if _is_acceptable_summary(clean, genre=str(row.get("genre", "") or "")):
                    existing_summaries[row["title_id"]] = clean
                else:
                    invalid_existing += 1

    movies_needing = [m for m in movies_data if m["title_id"] not in existing_summaries]
    print(f"Total movies: {len(movies_data)}")
    print(f"Already have summary: {len(existing_summaries)}")
    print(f"Generic/placeholder summaries to replace: {invalid_existing}")
    print(f"Needing summary: {len(movies_needing)}")

    if not movies_needing:
        print("All movies already have summaries!")
        return

    batch_size = _resolve_plot_batch_size(len(movies_data))
    n_batches = (len(movies_needing) + batch_size - 1) // batch_size
    save_every_batches = _resolve_plot_save_every_batches(len(movies_data))
    est_cost = n_batches * (5000 / 1e6 * 0.25 + 5000 / 1e6 * 1.50)
    print(f"Batches: {n_batches}, Est. cost: ${est_cost:.3f}")
    print(f"Plot batch size: {batch_size}")
    print(f"Checkpoint save cadence: every {save_every_batches} batch(es)")

    if not args.auto:
        proceed = input("Proceed? [y/N]: ").strip().lower()
        if proceed != "y":
            print("Aborted.")
            return

    all_summaries = dict(existing_summaries)
    # v11 P1: Anti-duplication -- rolling phrase blacklist
    phrase_blacklist = []
    recent_summaries = [text.lower().strip() for text in existing_summaries.values() if text.strip()][-30:]
    for text in existing_summaries.values():
        phrase_blacklist.extend(_extract_key_phrases(text))
    dup_warnings = 0

    for batch_start in range(0, len(movies_needing), batch_size):
        batch = movies_needing[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        print(f"\n  Batch {batch_num}/{n_batches} ({len(batch)} movies)...")
        batch_results = _generate_batch_summaries(
            llm=llm,
            model=_model,
            batch=batch,
            batch_num_label=f"{batch_num}",
            tracker=tracker,
            phrase_blacklist=phrase_blacklist,
        )
        all_summaries.update(batch_results)

        should_checkpoint = (batch_num % save_every_batches == 0) or (batch_num == n_batches)
        if should_checkpoint:
            _save_movie_outputs(base_dir, movies_df, all_summaries)
            print(f"  Checkpoint saved after batch {batch_num}/{n_batches}")

        # v11 P1: Update anti-duplication blacklist with this batch's phrases
        for mid in [m["title_id"] for m in batch]:
            ps = all_summaries.get(mid, "")
            if ps:
                phrase_blacklist.extend(_extract_key_phrases(ps))
                # Levenshtein near-duplicate check against recent summaries
                ps_lower = ps.lower().strip()
                for prev in recent_summaries:
                    dist = _levenshtein_distance(ps_lower[:80], prev[:80])
                    if dist < 15:
                        dup_warnings += 1
                        if dup_warnings <= 5:  # don't spam
                            print(f"  [warn] Near-duplicate detected (Levenshtein={dist}): {ps[:60]}...")
                recent_summaries.append(ps_lower)
                if len(recent_summaries) > 30:
                    recent_summaries.pop(0)
        # Keep blacklist bounded
        if len(phrase_blacklist) > 100:
            phrase_blacklist = phrase_blacklist[-60:]

        # Keep a small pacing gap to avoid bursty provider traffic, but do not
        # burn minutes of wall-clock time across thousands of batches.
        time.sleep(0.15)

    recovered = _recover_plot_summaries(
        llm=llm,
        model=_model,
        tracker=tracker,
        movies_data=movies_data,
        all_summaries=all_summaries,
        base_dir=base_dir,
        movies_df=movies_df,
        flat_df=flat_df,
    )
    if recovered:
        print(f"  Targeted recovery repaired {recovered} summaries")

    # Final save: movie.csv
    _save_movie_outputs(base_dir, movies_df, all_summaries)

    # Keep downstream deliverables in sync with the improved movie table.
    _save_flat_and_analysis_outputs(base_dir, flat_df, movies_df, all_summaries)

    genre_lookup = {int(row["title_id"]): str(row.get("genre", "") or "") for _, row in movies_df.iterrows()}
    valid_filled = sum(1 for mid, v in all_summaries.items() if _is_acceptable_summary(v, genre=genre_lookup.get(int(mid), "")))
    print(f"\n  Filled: {valid_filled}/{len(movies_df)} plot summaries")
    print(f"  Saved -> {base_dir / 'movie.arrow'}")
    if PIPELINE_MODE == "research" and valid_filled < len(movies_df):
        raise RuntimeError(
            f"Research-mode plot summary generation incomplete: only {valid_filled}/{len(movies_df)} valid summaries"
        )

    # Token summary
    print(tracker.summary())


if __name__ == "__main__":
    main()
