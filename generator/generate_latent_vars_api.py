# -*- coding: utf-8 -*-
"""
Mirage -- Generate latent variables via the configured LLM provider.
===================================================
Step 1 of the Hybrid Graph Architecture:
Asks the LLM to read each person's/company's bio and assign numeric
latent variables that capture nuance a procedural rule can't.

These latent vectors are then consumed by generate_edges_hybrid.py
(Step 2) to build the graph procedurally.

Usage:
    python generate_latent_vars_api.py          # interactive
    python generate_latent_vars_api.py --auto    # skip prompts
    python generate_latent_vars_api.py --model gemini-3.1-flash-lite
"""
import argparse
import os, json, sys, time
import shutil
from pathlib import Path

from llm_provider import get_llm_client
from model_defaults import model_for_role

MODEL = model_for_role("latent_vars")

import random as _rnd
from contracts import GENRES  # V17: used in avoid_genres prompt

API_TIMEOUT = 70  # seconds; kills hung API calls


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return max(minimum, int(default))
    try:
        return max(minimum, int(float(str(raw).strip())))
    except Exception:
        return max(minimum, int(default))


def _jitter_to_3dp(latent_vars: list[dict]) -> list[dict]:
    """Post-process latent vars to ensure 3 decimal precision.

    LLMs tend to output round numbers (0.3, 0.25) despite explicit
    prompt instructions. This adds a tiny seeded jitter (+/-0.005) to
    any float with fewer than 3 significant decimals, then rounds to 3dp.
    """
    for entry in latent_vars:
        seed_str = entry.get("name", str(entry.get("person_id", 0)))
        rng = _rnd.Random(seed_str)
        for k, v in entry.items():
            if not isinstance(v, float):
                if isinstance(v, list):
                    entry[k] = [
                        round(x + rng.uniform(-0.005, 0.005), 3)
                        if isinstance(x, float) else x
                        for x in v
                    ]
                continue
            # Check if already 3dp
            s = f"{v:.10f}".rstrip('0')
            dec_part = s.split('.')[-1] if '.' in s else ''
            if len(dec_part) < 3:
                jitter = rng.uniform(-0.005, 0.005)
                entry[k] = round(v + jitter, 3)
    return latent_vars

BASE_DIR = Path(__file__).parent
ENTITY_DIR = BASE_DIR / "entities"
PERSON_SHARD_DIR = ENTITY_DIR / "_person_latent_shards"
COMPANY_SHARD_DIR = ENTITY_DIR / "_company_latent_shards"

BATCH_SIZE = _env_int("DATA_SYS_LATENT_BATCH_SIZE", 50)
MAX_RETRIES = _env_int("DATA_SYS_LATENT_MAX_RETRIES", 5)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _clamp01(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))), 3)


def _deterministic_person_latent(person: dict) -> dict:
    pid = int(person.get("person_id", 0) or 0)
    rng = _rnd.Random(f"person-latent:{pid}:{person.get('name', '')}")
    roles = person.get("roles", [])
    if isinstance(roles, str):
        roles = [part.strip() for part in roles.replace(";", ",").split(",") if part.strip()]
    stage = str(person.get("career_stage", "prime")).lower()
    genres = person.get("genre_affinity", [])
    if isinstance(genres, str):
        genres = [part.strip() for part in genres.replace(";", ",").split(",") if part.strip()]
    if not genres:
        genres = ["Drama", "Comedy"]
    style_vector = [round(rng.uniform(-0.75, 0.75), 3) for _ in range(8)]
    rep_base = {"rising": 0.25, "prime": 0.55, "veteran": 0.62, "legend": 0.86, "retired": 0.50}.get(stage, 0.45)
    ambition_base = 0.62 if any(role in roles for role in ("director", "writer", "cinematographer")) else 0.45
    risk_base = 0.58 if any(role in roles for role in ("director", "writer", "producer")) else 0.42
    budget = [rng.uniform(0.15, 0.85) for _ in range(5)]
    if stage == "rising":
        budget[0] += 0.25
        budget[1] += 0.20
    elif stage == "legend":
        budget[3] += 0.25
        budget[4] += 0.20
    avoid_pool = [g for g in GENRES if g not in genres] or list(GENRES)
    rng.shuffle(avoid_pool)
    return {
        "person_id": pid,
        "creative_style_vector": style_vector,
        "risk_tolerance": _clamp01(risk_base + rng.uniform(-0.18, 0.18)),
        "collaboration_style": rng.choice(["solo", "ensemble", "chameleon", "mentorship"]),
        "controversy_score": _clamp01(0.18 + rng.uniform(-0.10, 0.25)),
        "public_reputation": _clamp01(rep_base + rng.uniform(-0.12, 0.12)),
        "budget_band_pref": [_clamp01(v) for v in budget],
        "artistic_ambition": _clamp01(ambition_base + rng.uniform(-0.20, 0.20)),
        "volatility": _clamp01(0.38 + rng.uniform(-0.18, 0.22)),
        "avoid_genres": avoid_pool[:2],
    }


def _deterministic_company_latent(company: dict) -> dict:
    cid = int(company.get("company_id", 0) or 0)
    rng = _rnd.Random(f"company-latent:{cid}:{company.get('name', '')}")
    tier = str(company.get("tier", "Mid-Budget"))
    genre_focus = [rng.uniform(0.05, 0.45) for _ in range(12)]
    total = sum(genre_focus) or 1.0
    genre_focus = [round(v / total, 3) for v in genre_focus]
    tier_idx = {"Micro": 0, "Indie": 1, "Mid-Budget": 2, "A-List": 3, "Epic": 4}.get(tier, 2)
    budget = [0.08 + rng.uniform(0.0, 0.12) for _ in range(5)]
    budget[tier_idx] += 0.55
    total_budget = sum(budget) or 1.0
    return {
        "company_id": cid,
        "risk_appetite": _clamp01(0.55 - 0.07 * tier_idx + rng.uniform(-0.15, 0.15)),
        "prestige_score": _clamp01(0.18 + 0.15 * tier_idx + rng.uniform(-0.10, 0.14)),
        "genre_portfolio": genre_focus,
        "budget_tier_focus": [round(v / total_budget, 3) for v in budget],
        "market_trend_sensitivity": _clamp01(0.45 + rng.uniform(-0.20, 0.20)),
        "controversy_tolerance": _clamp01(0.35 + rng.uniform(-0.18, 0.22)),
    }


def _write_deterministic_latents(all_persons: list[dict]) -> None:
    companies_path = ENTITY_DIR / "companies.json"
    if not companies_path.exists():
        raise FileNotFoundError(f"companies.json not found at {companies_path}")
    with open(companies_path, encoding="utf-8") as f:
        companies = json.load(f)
    cids_changed = False
    for i, c in enumerate(companies):
        if "company_id" not in c:
            c["company_id"] = i + 1
            cids_changed = True
    if cids_changed:
        with open(companies_path, "w", encoding="utf-8") as f:
            json.dump(companies, f, indent=2, ensure_ascii=False)
    person_latents = [_deterministic_person_latent(p) for p in all_persons]
    company_latents = [_deterministic_company_latent(c) for c in companies]
    with open(ENTITY_DIR / "persons_latent.json", "w", encoding="utf-8") as f:
        json.dump(person_latents, f, indent=2, ensure_ascii=False)
    with open(ENTITY_DIR / "companies_latent.json", "w", encoding="utf-8") as f:
        json.dump(company_latents, f, indent=2, ensure_ascii=False)
    if PERSON_SHARD_DIR.exists():
        shutil.rmtree(PERSON_SHARD_DIR, ignore_errors=True)
    if COMPANY_SHARD_DIR.exists():
        shutil.rmtree(COMPANY_SHARD_DIR, ignore_errors=True)
    print(f"Deterministic latent mode wrote {len(person_latents)} person latents and {len(company_latents)} company latents.")


def _retry_delay_seconds(retry_index: int, *, is_503: bool) -> int:
    profile = str(os.getenv("DATA_SYS_OVERLOAD_PROFILE", "default") or "default").strip().lower()
    if profile == "smoke":
        schedule_503 = [4, 7, 10, 15, 20]
        schedule_other = [2, 3, 5, 8, 12]
    else:
        schedule_503 = [8, 15, 25, 35, 45]
        schedule_other = [3, 5, 8, 12, 16]
    if is_503:
        schedule = schedule_503
    else:
        schedule = schedule_other
    idx = min(max(0, int(retry_index)), len(schedule) - 1)
    return int(schedule[idx])

# -----------------------------------------------------------------------
# TOKEN TRACKING
# -----------------------------------------------------------------------

class TokenTracker:
    """Track cumulative token usage and cost across all API calls."""
    def __init__(self, model_name: str = MODEL):
        self.model_name = model_name
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.calls = 0
        self.errors = 0

    def record_llm_response(self, resp):
        """Record token usage from an LLMResponse object."""
        self.total_input_tokens += resp.input_tokens
        self.total_output_tokens += resp.output_tokens
        self.total_cost += resp.cost_usd
        self.calls += 1
        return resp.input_tokens, resp.output_tokens, resp.cost_usd

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


# -----------------------------------------------------------------------
# PROMPTS
# -----------------------------------------------------------------------

def build_person_latent_prompt(batch_persons):
    """Build prompt to assign latent variables to a batch of persons."""
    person_lines = []
    for p in batch_persons:
        pid = p.get("person_id", "?")
        name = p["name"]
        bio = str(p.get("bio", ""))[:200]
        styles = p.get("style_tags", [])
        if isinstance(styles, str):
            styles = [s.strip() for s in styles.split(",")]
        genres = p.get("genre_affinity", [])
        if isinstance(genres, str):
            genres = [g.strip() for g in genres.split(",")]
        stage = p.get("career_stage", "prime")
        roles = p.get("roles", ["actor"])
        if isinstance(roles, str):
            roles = [roles]

        person_lines.append(
            f'[{pid}] {name} | roles: {",".join(roles)} | stage: {stage} '
            f'| styles: {",".join(styles[:4])} | genres: {",".join(genres)} '
            f'| bio: {bio}'
        )

    persons_block = "\n".join(person_lines)

    return f"""You are assigning numeric latent variables to movie industry persons based on their profiles.
Read each person's bio, styles, genres, and career stage, then assign scores that capture the nuance of their profile.

IMPORTANT: All float values MUST use 3 decimal places (e.g., 0.347, not 0.3). Each person should
have meaningfully different values -- avoid rounding to 0.1 increments.

=== PERSONS ({len(batch_persons)}) ===
{persons_block}

For EACH person, output a JSON object with their person_id and these latent variables:

1. "creative_style_vector": array of 8 floats in [-1, 1] with 3 decimal places. Dimensions:
   [0] methodical (-1) vs spontaneous (+1)
   [1] minimalist (-1) vs maximalist (+1)
   [2] dark/edgy (-1) vs light/uplifting (+1)
   [3] realistic (-1) vs stylized (+1)
   [4] introverted (-1) vs extroverted (+1)
   [5] traditional (-1) vs experimental (+1)
   [6] character-driven (-1) vs plot-driven (+1)
   [7] intimate (-1) vs spectacle (+1)

2. "risk_tolerance": float [0, 1] with 3 decimal places. How willing to take unconventional roles.
   Bio mentions "controversial", "avant-garde", "experimental" -> high.
   "family-friendly", "mainstream", "blockbuster" -> low.

3. "collaboration_style": MUST be exactly one of: "solo", "ensemble", "chameleon", "mentorship".
   No other values allowed. Solo = prefers tight, small crews. Ensemble = thrives in big groups.
   Chameleon = adapts to any format. Mentorship = tends to pair with newcomers / proteges.

4. "controversy_score": float [0, 1] with 3 decimal places. How controversial the person is.
   Mentions of scandals, provocative work, political activism -> high.
   Clean reputation, family-friendly -> low.

5. "public_reputation": float [0, 1] with 3 decimal places. Star power / name recognition.
   Legends and mega-stars -> 0.8-1.0. Rising newcomers -> 0.1-0.3.

6. "budget_band_pref": array of 5 floats [0, 1] with 3 decimal places, one per budget tier [Micro, Indie, Mid, A, Epic].
   How comfortable this person is in each tier. Sum doesn't need to be 1.

7. "artistic_ambition": float [0, 1] with 3 decimal places. High = art-house / experimental / festival focus.
   Low = commercial / mainstream / crowd-pleasing focus.

8. "volatility": float [0, 1] with 3 decimal places. High = swings between hit and flop. Low = consistent performer.
   Use bio cues like "unpredictable", "polarizing", "erratic" -> high.

9. "avoid_genres": array of 1-3 genre strings from this list: {GENRES}.
   Genres this person would NOT want to appear in. Infer from bio + style:
   Family-friendly actors avoid Horror/Thriller; cerebral actors avoid Action/Superhero;
   comedy specialists may avoid Documentary. Pick 1-3 genres that conflict with their profile.

Output ONLY a JSON array of objects. Each object has "person_id" and the 9 variables above.
No markdown fences, no explanation, no extra text."""


def build_company_latent_prompt(batch_companies):
    """Build prompt to assign latent variables to a batch of companies."""
    company_lines = []
    for c in batch_companies:
        cid = c.get("company_id", "?")
        name = c["name"]
        desc = str(c.get("description", ""))[:150]
        genres = c.get("specialty_genres", [])
        if isinstance(genres, str):
            genres = [g.strip() for g in genres.split(";")]
        tier = c.get("tier", "Mid-Budget")

        company_lines.append(
            f'[{cid}] {name} | tier: {tier} | genres: {",".join(genres)} | {desc}'
        )

    companies_block = "\n".join(company_lines)

    return f"""You are assigning numeric latent variables to movie production companies based on their profiles.

=== COMPANIES ({len(batch_companies)}) ===
{companies_block}

For EACH company, output a JSON object with their company_id and these latent variables:

1. "risk_appetite": float [0, 1]. How willing to take risky, unconventional projects.
   Indie studios doing art-house films -> high. Big tentpole studios -> low.

2. "prestige_score": float [0, 1]. Studio prestige / brand recognition.
   Global studios with Oscar history -> 0.8-1.0. New micro studios -> 0.1-0.3.

3. "genre_portfolio": array of 12 floats [0, 1], one per genre
   [Action, Drama, Comedy, Sci-Fi, Horror, Romance, Thriller, Fantasy, Mystery, Documentary, Crime, Animation].
   How much of their slate is in each genre. Sum should be roughly 1.

4. "budget_tier_focus": array of 5 floats [0, 1], one per tier [Micro, Indie, Mid, A, Epic].
   How much they operate in each tier. Sum should be roughly 1.

5. "market_trend_sensitivity": float [0, 1]. High = chases trends / shifts genres over decades.
   Low = sticks to brand identity and signature slate.

6. "controversy_tolerance": float [0, 1]. How much controversy they tolerate.
   Studios known for provocative content -> high. Family studios -> low.

Output ONLY a JSON array of objects. Each has "company_id" and the 6 variables above.
No markdown fences, no explanation."""


def parse_json_response(text):
    """Extract JSON array from response, handling markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return json.loads(text)


def _load_json_shards(shard_dir: Path) -> list[dict]:
    rows: list[dict] = []
    if not shard_dir.exists():
        return rows
    for path in sorted(shard_dir.glob("batch_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _coerce_entity_id(row: dict, id_key: str) -> int | None:
    raw = row.get(id_key)
    if raw is None or isinstance(raw, bool):
        return None
    try:
        return int(raw)
    except Exception:
        return None


def _dedupe_latents(rows: list[dict], id_key: str, label: str) -> list[dict]:
    cleaned: list[dict] = []
    seen: set[int] = set()
    dropped_missing = 0
    dropped_duplicates = 0
    for row in rows:
        if not isinstance(row, dict):
            dropped_missing += 1
            continue
        entity_id = _coerce_entity_id(row, id_key)
        if entity_id is None:
            dropped_missing += 1
            continue
        if entity_id in seen:
            dropped_duplicates += 1
            continue
        fixed = dict(row)
        fixed[id_key] = entity_id
        cleaned.append(fixed)
        seen.add(entity_id)
    if dropped_missing or dropped_duplicates:
        print(
            f"  Cleaned {label}: kept {len(cleaned)}, "
            f"dropped {dropped_missing} missing-id rows and {dropped_duplicates} duplicate rows"
        )
    return cleaned


def _validate_batch_latents(
    raw_rows: list[dict],
    batch_ids: set[int],
    id_key: str,
) -> tuple[list[dict], list[int], int, int, int]:
    if not isinstance(raw_rows, list):
        raise ValueError(f"Expected JSON array, got {type(raw_rows).__name__}")

    cleaned: list[dict] = []
    seen: set[int] = set()
    duplicate_rows = 0
    extra_rows = 0
    missing_id_rows = 0
    expected = {int(entity_id) for entity_id in batch_ids}

    for row in raw_rows:
        if not isinstance(row, dict):
            missing_id_rows += 1
            continue
        entity_id = _coerce_entity_id(row, id_key)
        if entity_id is None:
            missing_id_rows += 1
            continue
        if entity_id not in expected:
            extra_rows += 1
            continue
        if entity_id in seen:
            duplicate_rows += 1
            continue
        fixed = dict(row)
        fixed[id_key] = entity_id
        cleaned.append(fixed)
        seen.add(entity_id)

    missing_ids = sorted(expected - seen)
    return cleaned, missing_ids, duplicate_rows, extra_rows, missing_id_rows


def _write_jsonl_shard(shard_dir: Path, batch_num: int, rows: list[dict]) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)
    path = shard_dir / f"batch_{batch_num:06d}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _next_jsonl_shard_number(shard_dir: Path) -> int:
    if not shard_dir.exists():
        return 0
    highest = 0
    for path in shard_dir.glob("batch_*.jsonl"):
        try:
            highest = max(highest, int(path.stem.rsplit("_", 1)[1]))
        except Exception:
            continue
    return highest


def main():
    parser = argparse.ArgumentParser(description="Generate latent variables via the configured LLM provider.")
    parser.add_argument("--auto",            action="store_true", help="Skip interactive prompts")
    parser.add_argument("--companies-only",  action="store_true", help="Only process companies (skip persons)")
    parser.add_argument("--persons-only",    action="store_true", help="Only process persons (skip companies)")
    parser.add_argument("--model",           default=None,
                        help=f"LLM model override (default: provider default)")
    parser.add_argument("--batch-size",      type=int, default=BATCH_SIZE,
                        help=f"Persons/companies per batch (default: {BATCH_SIZE})")
    parser.add_argument("--mode", choices=("research", "debug"), default=os.getenv("DATA_SYS_PIPELINE_MODE", "research"))
    args = parser.parse_args()
    _model = args.model  # None = use provider default
    _batch_size = args.batch_size

    os.makedirs(ENTITY_DIR, exist_ok=True)

    # --- Load persons -------------------------------------------------
    with open(ENTITY_DIR / "persons.json", encoding="utf-8") as f:
        all_persons = json.load(f)

    # Assign latent vars to ALL persons (core + extras)
    # The incremental logic below skips already-processed persons automatically.
    all_process = all_persons

    # F4: Stable IDs -- persist person_id so it survives re-runs
    ids_changed = False
    for i, p in enumerate(all_process):
        if "person_id" not in p:
            p["person_id"] = i + 1
            ids_changed = True
    if ids_changed:
        with open(ENTITY_DIR / "persons.json", "w", encoding="utf-8") as f:
            json.dump(all_persons, f, indent=2, ensure_ascii=False)
        print("  Persisted person_id values to persons.json")

    if _env_bool("DATA_SYS_DETERMINISTIC_LATENTS", False):
        _write_deterministic_latents(all_persons)
        return

    llm = get_llm_client()
    tracker = TokenTracker(getattr(llm, '_default_model', MODEL))

    core_count = sum(1 for p in all_persons if not p.get("is_extra", False))
    print(f"Loaded {len(all_persons)} total persons")

    # Load existing persons latent vars (always needed to find what's missing)
    latent_path = ENTITY_DIR / "persons_latent.json"
    existing_ids = set()
    existing_latent = []
    if latent_path.exists():
        existing_latent = json.loads(latent_path.read_text(encoding="utf-8"))
        print(f"  Found {len(existing_latent)} existing person latent vars")
    if args.auto and PERSON_SHARD_DIR.exists():
        shard_latent = _load_json_shards(PERSON_SHARD_DIR)
        if shard_latent:
            print(f"  Recovered {len(shard_latent)} person latent vars from shards")
            existing_latent.extend(shard_latent)
    existing_latent = _dedupe_latents(existing_latent, "person_id", "person latent resume rows")
    existing_ids = {lv["person_id"] for lv in existing_latent}

    if args.companies_only:
        print("  --companies-only: skipping persons")
        persons_needing = []
    else:
        persons_needing = [p for p in all_process if p["person_id"] not in existing_ids]
        print(f"Persons needing latent vars: {len(persons_needing)}")

    # Approximate cost estimate for Gemini Flash-style pricing.
    n_batches = (len(persons_needing) + BATCH_SIZE - 1) // BATCH_SIZE
    est_cost = n_batches * (5000 / 1e6 * 0.25 + 3000 / 1e6 * 1.50)
    print(f"\nBatches: {n_batches}, Est. cost: ${est_cost:.3f}")

    if len(persons_needing) == 0:
        print("All persons already have latent vars. Skipping.")
    else:
        if not args.auto:
            proceed = input("Proceed with person latent vars? [y/N]: ").strip().lower()
            if proceed != "y":
                print("Skipped persons.")
                persons_needing = []

        if persons_needing:
            all_person_latent = list(existing_latent)
            person_batch_offset = _next_jsonl_shard_number(PERSON_SHARD_DIR)

            for batch_start in range(0, len(persons_needing), _batch_size):
                batch = persons_needing[batch_start:batch_start + _batch_size]
                batch_num = person_batch_offset + batch_start // _batch_size + 1
                print(f"\n  Batch {batch_num}/{n_batches} ({len(batch)} persons)...")

                prompt = build_person_latent_prompt(batch)
                batch_latents = None

                for retry in range(MAX_RETRIES):
                    t0 = time.time()
                    try:
                        response = llm.generate(
                            prompt,
                            model=_model,
                            json_mode=True,
                            temperature=0.3,
                            max_tokens=16384,
                            timeout_sec=API_TIMEOUT,
                            max_attempts=1,
                        )
                        elapsed = time.time() - t0
                    except TimeoutError as e:
                        print(f"  TIMEOUT (retry {retry+1}/{MAX_RETRIES}): {e}")
                        tracker.record_error()
                        delay = _retry_delay_seconds(retry, is_503=False)
                        print(f"  Waiting {delay}s before retry...")
                        time.sleep(delay)
                        continue
                    except Exception as e:
                        is_503 = "503" in str(e) or "unavailable" in str(e).lower()
                        print(f"  {'503 ' if is_503 else ''}API ERROR (retry {retry+1}/{MAX_RETRIES}): {e}")
                        tracker.record_error()
                        delay = _retry_delay_seconds(retry, is_503=is_503)
                        print(f"  Waiting {delay}s before retry...")
                        time.sleep(delay)
                        continue

                    inp, out, cost = tracker.record_llm_response(response)
                    print(f"  Tokens: {inp:,} in + {out:,} out = ${cost:.4f} | {elapsed:.1f}s")

                    try:
                        latent_vars = parse_json_response(response.text)

                        # Validate: check we got enough
                        batch_ids = {p["person_id"] for p in batch}
                        cleaned, missing, duplicate_rows, extra_rows, missing_id_rows = _validate_batch_latents(
                            latent_vars,
                            batch_ids,
                            "person_id",
                        )
                        if duplicate_rows or extra_rows or missing_id_rows:
                            print(
                                f"  Cleaned batch rows: dropped {duplicate_rows} duplicates, "
                                f"{extra_rows} extras, {missing_id_rows} missing-id rows"
                            )

                        if missing:
                            if retry < MAX_RETRIES - 1:
                                print(f"  Missing {len(missing)} persons, retrying...")
                                continue
                            raise RuntimeError(
                                f"Batch {batch_num} returned incomplete person latents: "
                                f"missing {len(missing)} ids, sample={missing[:10]}"
                            )

                        batch_latents = _jitter_to_3dp(cleaned)
                        all_person_latent.extend(batch_latents)
                        print(f"  Got {len(batch_latents)} latent vars (3dp jittered)")
                        break

                    except (json.JSONDecodeError, Exception) as e:
                        print(f"  JSON PARSE ERROR (retry {retry+1}/{MAX_RETRIES}): {e}")
                        raw_path = ENTITY_DIR / f"latent_batch{batch_num}_raw.txt"
                        with open(raw_path, "w", encoding="utf-8") as f:
                            f.write(response.text)
                        print(f"  Raw response saved -> {raw_path}")
                        if retry < MAX_RETRIES - 1:
                            time.sleep(3)
                            continue

                if batch_latents is None:
                    raise RuntimeError(f"Failed to generate a complete person latent batch {batch_num}")
                _write_jsonl_shard(PERSON_SHARD_DIR, batch_num, batch_latents)

                time.sleep(0.5)

            with open(latent_path, "w", encoding="utf-8") as f:
                json.dump(all_person_latent, f, indent=2, ensure_ascii=False)
            if PERSON_SHARD_DIR.exists():
                shutil.rmtree(PERSON_SHARD_DIR, ignore_errors=True)
            print(f"\nSaved {len(all_person_latent)} person latent vars -> {latent_path}")

    # --- Load companies -----------------------------------------------
    with open(ENTITY_DIR / "companies.json", encoding="utf-8") as f:
        companies = json.load(f)
    # F4: Stable company IDs
    cids_changed = False
    for i, c in enumerate(companies):
        if "company_id" not in c:
            c["company_id"] = i + 1
            cids_changed = True
    if cids_changed:
        with open(ENTITY_DIR / "companies.json", "w", encoding="utf-8") as f:
            json.dump(companies, f, indent=2, ensure_ascii=False)
        print("  Persisted company_id values to companies.json")

    print(f"\nLoaded {len(companies)} companies")

    company_latent_path = ENTITY_DIR / "companies_latent.json"
    existing_company_latent = []
    existing_cids = set()
    if company_latent_path.exists():
        existing_company_latent = json.loads(company_latent_path.read_text(encoding="utf-8"))
        print(f"  Found {len(existing_company_latent)} existing company latent vars")
    if args.auto and COMPANY_SHARD_DIR.exists():
        shard_company_latent = _load_json_shards(COMPANY_SHARD_DIR)
        if shard_company_latent:
            print(f"  Recovered {len(shard_company_latent)} company latent vars from shards")
            existing_company_latent.extend(shard_company_latent)
    existing_company_latent = _dedupe_latents(
        existing_company_latent,
        "company_id",
        "company latent resume rows",
    )
    existing_cids = {cl["company_id"] for cl in existing_company_latent}

    companies_needing = [c for c in companies if c["company_id"] not in existing_cids]
    n_cbatches = (len(companies_needing) + BATCH_SIZE - 1) // BATCH_SIZE

    if len(companies_needing) == 0:
        print("All companies already have latent vars.")
    else:
        print(f"Companies needing latent vars: {len(companies_needing)} ({n_cbatches} batches)")

        if not args.auto:
            proceed = input("Proceed with company latent vars? [y/N]: ").strip().lower()
            if proceed != "y":
                print("Skipped companies.")
                companies_needing = []
        if args.persons_only:
            print("  --persons-only: skipping companies")
            companies_needing = []

        if companies_needing:
            all_company_latent = list(existing_company_latent)
            company_batch_offset = _next_jsonl_shard_number(COMPANY_SHARD_DIR)

            for batch_start in range(0, len(companies_needing), _batch_size):
                batch = companies_needing[batch_start:batch_start + _batch_size]
                batch_num = company_batch_offset + batch_start // _batch_size + 1
                print(f"\n  Company batch {batch_num}/{n_cbatches} ({len(batch)} companies)...")

                prompt = build_company_latent_prompt(batch)
                batch_latents = None

                for retry in range(MAX_RETRIES):
                    t0 = time.time()
                    try:
                        response = llm.generate(
                            prompt,
                            model=_model,
                            json_mode=True,
                            temperature=0.3,
                            max_tokens=16384,
                            timeout_sec=API_TIMEOUT,
                            max_attempts=1,
                        )
                        elapsed = time.time() - t0
                    except TimeoutError as e:
                        print(f"  TIMEOUT (retry {retry+1}/{MAX_RETRIES}): {e}")
                        tracker.record_error()
                        delay = _retry_delay_seconds(retry, is_503=False)
                        print(f"  Waiting {delay}s before retry...")
                        time.sleep(delay)
                        continue
                    except Exception as e:
                        is_503 = "503" in str(e) or "unavailable" in str(e).lower()
                        print(f"  {'503 ' if is_503 else ''}API ERROR (retry {retry+1}/{MAX_RETRIES}): {e}")
                        tracker.record_error()
                        delay = _retry_delay_seconds(retry, is_503=is_503)
                        print(f"  Waiting {delay}s before retry...")
                        time.sleep(delay)
                        continue

                    inp, out, cost = tracker.record_llm_response(response)
                    print(f"  Tokens: {inp:,} in + {out:,} out = ${cost:.4f} | {elapsed:.1f}s")

                    try:
                        latent_vars = parse_json_response(response.text)
                        batch_ids = {c["company_id"] for c in batch}
                        cleaned, missing, duplicate_rows, extra_rows, missing_id_rows = _validate_batch_latents(
                            latent_vars,
                            batch_ids,
                            "company_id",
                        )
                        if duplicate_rows or extra_rows or missing_id_rows:
                            print(
                                f"  Cleaned company batch rows: dropped {duplicate_rows} duplicates, "
                                f"{extra_rows} extras, {missing_id_rows} missing-id rows"
                            )
                        if missing:
                            if retry < MAX_RETRIES - 1:
                                print(f"  Missing {len(missing)} companies, retrying...")
                                continue
                            raise RuntimeError(
                                f"Company batch {batch_num} returned incomplete latents: "
                                f"missing {len(missing)} ids, sample={missing[:10]}"
                            )
                        batch_latents = cleaned
                        all_company_latent.extend(batch_latents)
                        print(f"  Got {len(batch_latents)} latent vars")
                        break
                    except (json.JSONDecodeError, Exception) as e:
                        print(f"  JSON PARSE ERROR (retry {retry+1}/{MAX_RETRIES}): {e}")
                        if retry < MAX_RETRIES - 1:
                            time.sleep(3)

                if batch_latents is None:
                    raise RuntimeError(f"Failed to generate a complete company latent batch {batch_num}")
                _write_jsonl_shard(COMPANY_SHARD_DIR, batch_num, batch_latents)

                time.sleep(0.5)

            with open(company_latent_path, "w", encoding="utf-8") as f:
                json.dump(all_company_latent, f, indent=2, ensure_ascii=False)
            if COMPANY_SHARD_DIR.exists():
                shutil.rmtree(COMPANY_SHARD_DIR, ignore_errors=True)
            print(f"\nSaved {len(all_company_latent)} company latent vars -> {company_latent_path}")

    # Final token summary
    print(tracker.summary())


if __name__ == "__main__":
    main()
