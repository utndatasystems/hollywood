"""
Mirage -- JSON-to-CSV converter + mock data generator for dry runs.
Convert LLM JSON outputs to CSV entities for the assembly engine.
"""
import json
import pandas as pd
import numpy as np
import random
import os, sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(errors="backslashreplace")

sys.path.insert(0, os.path.dirname(__file__))
from contracts import (
    GENRES, GENRE_WEIGHTS, COUNTRIES, COMPANY_COUNTRY_WEIGHTS, NATIONALITIES,
    STYLE_TAGS, DIRECTOR_STYLES, CAREER_STAGES, MARKETS,
    COMPANY_TIERS, ARCHETYPES, ROLE_TYPES, validate_batch, validate_person,
    validate_company, normalize_name, find_near_duplicates,
    normalize_style_tag, normalize_company_styles,
    load_json_batch, save_json_batch,
)


# ═════════════════════════════════════════════════════════════════════
# NORMALIZATION HELPERS (keep IDs stable; do NOT drop records)
# ═════════════════════════════════════════════════════════════════════

_GENRE_ALIASES = {
    "musical": "Drama", "historical-drama": "Drama",
    "historical drama": "Drama", "social-commentary": "Drama",
    "social commentary": "Drama",
}


def _normalize_list_field(value):
    """Accept list or string and return a list[str]."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        tmp = value.replace("|", ";").replace(",", ";")
        return [x.strip() for x in tmp.split(";") if x.strip()]
    return [str(value).strip()]


def normalize_genres(genres):
    """Normalize genre tokens into the controlled GENRES vocabulary."""
    out = []
    for g in _normalize_list_field(genres):
        key = g.strip().lower()
        key_h = key.replace("_", "-")
        key_s = key_h.replace("-", " ")
        canon = (
            _GENRE_ALIASES.get(key) or
            _GENRE_ALIASES.get(key_h) or
            _GENRE_ALIASES.get(key_s)
        )
        if canon:
            out.append(canon)
            continue
        for vg in GENRES:
            if key == vg.lower():
                out.append(vg)
                break
    # De-dup preserving order
    seen = set()
    return [g for g in out if g not in seen and not seen.add(g)]


def normalize_person_record(p: dict) -> dict:
    """In-place normalization for persons."""
    # ── Roles: split compound strings like 'actor/director' -> ['actor', 'director']
    raw_roles = _normalize_list_field(p.get("roles", ["actor"]))
    role_set = set(ROLE_TYPES)
    roles = []
    for r in raw_roles:
        r_lo = r.strip().lower()
        if r_lo in role_set:
            roles.append(r_lo)
        elif "/" in r_lo:
            # Split compound: 'actor/director' -> ['actor', 'director']
            for part in r_lo.split("/"):
                part = part.strip()
                if part in role_set:
                    roles.append(part)
        # else: unknown role token -- silently drop
    if not roles:
        roles = ["actor"]
    # Deduplicate preserving order
    seen = set()
    p["roles"] = [x for x in roles if x not in seen and not seen.add(x)]

    p["genre_affinity"] = normalize_genres(p.get("genre_affinity", []))

    # ── Nationality: map unknown sub-regional variants to nearest known nationality
    _NAT_KEYWORDS = [
        # keyword -> known nationality (order matters -- longer/more specific first)
        ("eritrean", "Ethiopian"), ("oromo", "Ethiopian"), ("amhara", "Ethiopian"),
        ("slovenian", "Czech"), ("slovak", "Czech"), ("maltese", "Italian"),
        ("algerian", "Moroccan"), ("tunisian", "Moroccan"), ("ivorian", "Ghanaian"),
        ("malian", "Ghanaian"), ("cameroonian", "Nigerian"), ("rwandan", "Kenyan"),
        ("ugandan", "Kenyan"), ("tanzanian swahili", "Tanzanian"),
        ("south african zulu", "South African"), ("south african (zulu)", "South African"),
        ("haitian", "Dominican"), ("puerto rican", "Puerto Rican"),
        ("chicano", "American"), ("cajun", "American"), ("african-american", "American"),
        ("native american", "American"), ("hawaiian", "American"),
        ("quebecois", "Canadian"), ("british-columbian", "Canadian"), ("acadian", "Canadian"),
        ("scottish gaelic", "Scottish"), ("northern irish", "Irish"),
        ("bavarian", "German"), ("venetian", "Italian"), ("sicilian", "Italian"),
        ("neapolitan", "Italian"), ("andalusian", "Spanish"), ("catalan", "Spanish"),
        ("basque", "Spanish"), ("galician", "Spanish"), ("provencal", "French"),
        ("breton", "French"), ("faroese", "Danish"), ("saami", "Norwegian"),
        ("aboriginal australian", "Australian"), ("maori", "New Zealander"),
        ("cantonese", "Chinese"), ("hokkien", "Chinese"), ("sichuan", "Chinese"),
        ("mandarin northern chinese", "Chinese"), ("northern chinese", "Chinese"),
        ("taiwanese", "Chinese"), ("uyghur", "Chinese"), ("tibetan", "Chinese"),
        ("malayali", "Indian"), ("tamil indian", "Indian"), ("punjabi", "Indian"),
        ("assamese", "Indian"), ("bengali indian", "Indian"), ("marathi", "Indian"),
        ("odia", "Indian"), ("kannada", "Indian"), ("telugu", "Indian"),
        ("sundan", "Indonesian"), ("balinese", "Indonesian"), ("javanese", "Indonesian"),
        ("cebuano", "Filipino"), ("tagalog", "Filipino"),
        ("malay", "Malaysian"), ("malaysian chinese", "Chinese"),
        ("sinhalese", "Sri Lankan"), ("sri lankan tamil", "Sri Lankan"),
        ("khmer", "Vietnamese"), ("lao", "Vietnamese"), ("burmese", "Vietnamese"),
        ("okinawan", "Japanese"), ("ainu", "Japanese"),
        ("afghan", "Iranian"), ("pashtun", "Iranian"), ("kurdish", "Turkish"),
        ("levantine", "Lebanese"), ("gulf arab", "Saudi Arabian"),
        ("egyptian arab", "Egyptian"), ("moroccan amazigh", "Moroccan"),
        ("yoruba", "Nigerian"), ("igbo", "Nigerian"), ("hausa", "Nigerian"),
        ("kenyan kikuyu", "Kenyan"), ("kenyan luo", "Kenyan"), ("kenyan maasai", "Kenyan"),
        ("ghanaian akan", "Ghanaian"), ("ghanaian ewe", "Ghanaian"),
        ("senegalese wolof", "Senegalese"), ("lusophone african", "South African"),
        ("afrikaner", "South African"), ("fijian", "Australian"),
        ("samoan", "New Zealander"), ("papua", "Australian"),
        ("singaporean", "Malaysian"), ("tajik", "Kazakhstani"), ("kyrgyz", "Kazakhstani"),
        ("austrian", "German"), ("albanian", "Greek"), ("macedonian", "Bulgarian"),
        ("armenian", "Armenian"), ("azerbaijani", "Azerbaijani"),
        ("georgian", "Georgian"), ("belarusian", "Ukrainian"),
        ("uzbek", "Uzbek"), ("nepalese", "Nepali"),
    ]
    nat = p.get("nationality", "")
    nat_known = set(NATIONALITIES)
    if nat not in nat_known:
        nat_lo = nat.strip().lower()
        mapped = None
        if nat_lo in ("m", "f", "nb", "nz", ""):
            mapped = "American"  # gender/code leaked into nationality field
        else:
            for kw, target in _NAT_KEYWORDS:
                if kw in nat_lo:
                    mapped = target
                    break
        if mapped is None:
            mapped = "American"  # safe fallback
        p["nationality"] = mapped

    # ── Style tags: allow both actor and director styles (union vocab)
    raw_tags = _normalize_list_field(p.get("style_tags", []))
    vocab = list(STYLE_TAGS) + list(DIRECTOR_STYLES)
    norm_tags = []
    for t in raw_tags:
        nt = normalize_style_tag(t, vocab=vocab)
        if nt and nt not in norm_tags:
            norm_tags.append(nt)
    p["style_tags"] = norm_tags
    p["market_fit"] = [m.strip() for m in _normalize_list_field(p.get("market_fit", [])) if m.strip()]
    if "career_stage" in p and isinstance(p["career_stage"], str):
        cs = p["career_stage"].strip().lower()
        _STAGE_TYPOS = {"veterant": "veteran", "legand": "legend", "priime": "prime"}
        p["career_stage"] = _STAGE_TYPOS.get(cs, cs)
    elif not p.get("career_stage"):
        p["career_stage"] = "prime"  # safe default for malformed records
    return p



def normalize_company_record(c: dict) -> dict:
    """In-place normalization for companies."""
    c["specialty_genres"] = normalize_genres(c.get("specialty_genres", []))
    normalize_company_styles(c)
    if "tier" in c and isinstance(c["tier"], str):
        c["tier"] = c["tier"].strip()
    # ── Country normalization ────────────────────────────────────────
    country = c.get("country", "")
    country_known = set(COUNTRIES)
    if country and country not in country_known:
        # Strip city/region suffixes: 'India (Mumbai)' -> 'India'
        if "(" in country:
            country = country.split("(")[0].strip()
        # Map remaining unknown countries to nearest known one
        _COUNTRY_MAP = {
            # original entries
            "Rwanda": "Congo", "Uganda": "Congo", "Mali": "Senegal",
            "Cameroon": "Nigeria", "Ivory Coast": "Ghana",
            "Kyrgyzstan": "Kazakhstan", "Palestine": "Jordan",
            # new countries from company topup
            "Algeria": "Morocco", "Angola": "Congo",
            "Burkina Faso": "Senegal", "Cote d'Ivoire": "Ghana",
            "Cambodia": "Thailand", "Laos": "Thailand",
            "Myanmar": "Thailand", "Fiji": "Australia",
            "Papua New Guinea": "Australia", "Samoa": "Australia",
            "Solomon Islands": "Australia", "Vanuatu": "Australia",
            "Tajikistan": "Kazakhstan", "Turkmenistan": "Kazakhstan",
            "Timor-Leste": "Australia", "Tonga": "Australia",
            "Unknown": "USA",
        }
        country = _COUNTRY_MAP.get(country, country)
        if country not in country_known:
            country = "USA"  # safe fallback (was 'United States' — not in vocab)
        c["country"] = country
    return c



def json_to_csv(base_dir: str):
    """Convert all JSON entity files to CSVs expected by generate_movies.py."""
    edir = os.path.join(base_dir, "entities")

    # ─── Persons ──────────────────────────────────────────────────────
    persons_path = os.path.join(edir, "persons.json")
    if os.path.exists(persons_path):
        persons = load_json_batch(persons_path)
        # v11: Stable IDs -- persist person_id so it survives re-runs
        ids_changed = False
        for i, p in enumerate(persons):
            if "person_id" not in p:
                p["person_id"] = i + 1
                ids_changed = True
        if ids_changed:
            save_json_batch(persons, persons_path)
            print("  Persisted person_id values to persons.json")
        # Normalize in-place (do NOT drop records; IDs must remain stable)
        persons = [normalize_person_record(p) for p in persons]
        result = validate_batch(persons, validate_person, "persons")
        if result["invalid"]:
            raise ValueError(
                f"Refusing to write person.csv: {result['invalid']} invalid person records."
                " Fix JSON or extend normalization."
            )
        clean = result["clean"]

        # Convert list fields to strings for CSV
        rows = []
        for p in clean:
            row = {
                "person_id": int(p.get("person_id", 0) or 0),
                "name": p["name"],
                "nationality": p["nationality"],
                "gender": p["gender"],
                "bio": p["bio"],
                "style_tags": ";".join(p.get("style_tags", [])),
                "genre_affinity": ";".join(p.get("genre_affinity", [])),
                "career_stage": p.get("career_stage", "prime"),
                "roles": ";".join(p.get("roles", ["actor"])),
                "market_fit": ";".join(p.get("market_fit", [])),
            }
            for timeline_field in ("debut_year", "peak_start", "peak_end", "retirement_year", "yearly_max"):
                if timeline_field in p:
                    row[timeline_field] = p.get(timeline_field)
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(edir, "person.csv"), index=False)
        print(f"Saved person.csv ({len(df)} persons)")

        # person_roles.csv
        roles = []
        for i, p in enumerate(clean):
            pid = int(p.get("person_id", i + 1))
            for r in p.get("roles", ["actor"]):
                roles.append({"person_id": pid, "role_type": r})
        pd.DataFrame(roles).to_csv(os.path.join(edir, "person_roles.csv"), index=False)

        # Near-duplicate checks are quadratic; keep them for small smoke runs only.
        names = [p["name"] for p in clean]
        near_duplicate_limit = int(os.getenv("DATA_SYS_NEAR_DUPLICATE_LIMIT", "50000"))
        if len(names) <= near_duplicate_limit:
            dupes = find_near_duplicates(names)
            if dupes:
                print(f"  WARNING: {len(dupes)} near-duplicate names found:")
                for a, b, d in dupes[:5]:
                    print(f"    '{a}' ~ '{b}' (distance={d})")
        else:
            print(
                f"  Skipping near-duplicate person-name check for {len(names)} names "
                f"(limit {near_duplicate_limit})"
            )

    # ─── Companies ────────────────────────────────────────────────────
    companies_path = os.path.join(edir, "companies.json")
    if os.path.exists(companies_path):
        companies = load_json_batch(companies_path)
        # v11: Stable IDs -- persist company_id so it survives re-runs
        ids_changed = False
        for i, c in enumerate(companies):
            if "company_id" not in c:
                c["company_id"] = i + 1
                ids_changed = True
        if ids_changed:
            save_json_batch(companies, companies_path)
            print("  Persisted company_id values to companies.json")
        companies = [normalize_company_record(c) for c in companies]
        result = validate_batch(companies, validate_company, "companies")
        if result["invalid"]:
            raise ValueError(
                f"Refusing to write company.csv: {result['invalid']} invalid company records."
                " Fix JSON or extend normalization."
            )
        clean = result["clean"]

        rows = []
        for c in clean:
            row = {
                "company_id": int(c.get("company_id", 0) or 0),
                "name": c["name"],
                "country": c["country"],
                "description": c["description"],
                "specialty_genres": ";".join(c.get("specialty_genres", [])),
                "tier": c.get("tier", "Mid-Budget"),
                "founded_year": c.get("founded_year"),
                "defunct_year": c.get("defunct_year"),
                "preferred_actor_styles": ";".join(c.get("preferred_actor_styles", [])),
                "preferred_director_styles": ";".join(c.get("preferred_director_styles", [])),
                "avoid_actor_styles": ";".join(c.get("avoid_actor_styles", [])),
                "avoid_director_styles": ";".join(c.get("avoid_director_styles", [])),
                "pop_weight": float(c.get("pop_weight", 0.5)),  # preserve API-generated weights
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(edir, "company.csv"), index=False)
        print(f"Saved company.csv ({len(df)} companies)")

    # ─── Keywords ─────────────────────────────────────────────────────
    kw_path = os.path.join(edir, "keywords.json")
    if os.path.exists(kw_path):
        keywords = load_json_batch(kw_path)
        # v11: Stable IDs -- persist keyword_id so it survives re-runs
        ids_changed = False
        for i, k in enumerate(keywords):
            if "keyword_id" not in k:
                k["keyword_id"] = i + 1
                ids_changed = True
        if ids_changed:
            save_json_batch(keywords, kw_path)
            print("  Persisted keyword_id values to keywords.json")
        rows = []
        for k in keywords:
            rows.append({
                "keyword_id": int(k.get("keyword_id", 0) or 0),
                "keyword": k.get("keyword", k.get("name", "?")),
                "topic_genre": k.get("topic_genre", ""),
                "pop_weight": float(k.get("pop_weight", 0.5)),  # preserve Zipf weights
                "selection_bucket": k.get("selection_bucket", "story_specific"),
                "motif_family": k.get("motif_family", ""),
                "specificity_tier": int(k.get("specificity_tier", 2) or 2),
                "scope_hint": k.get("scope_hint", ""),
                "franchise_affinity": float(k.get("franchise_affinity", 0.0) or 0.0),
                "cooccurrence_cluster": k.get("cooccurrence_cluster", ""),
                "recurrence_strength": float(k.get("recurrence_strength", 0.0) or 0.0),
            })
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(edir, "keyword.csv"), index=False)
        print(f"Saved keyword.csv ({len(df)} keywords)")

    # ─── Title bank ───────────────────────────────────────────────────
    # V12: file is movie_titlebank.json; fallback to titles.json for compat
    # FIX: do NOT overwrite title_bank.csv if it already has MORE titles than
    # movie_titlebank.json provides. generate_titles_llm.py writes title_bank.csv
    # directly with the full LLM-generated bank; entities_to_csv must not clobber it.
    title_path = os.path.join(edir, "movie_titlebank.json")
    if not os.path.exists(title_path):
        title_path = os.path.join(edir, "titles.json")
    if os.path.exists(title_path):
        titles = load_json_batch(title_path)
        rows = []
        for t in titles:
            row = {
                "title": t.get("title", ""),
                "tagline": t.get("tagline", ""),
                "genre_hint": t.get("genre", t.get("genre_hint", "Drama")),
            }
            # Preserve extra V12 fields if present
            if "year" in t:
                row["year"] = t["year"]
            if "movie_id" in t:
                row["movie_id"] = t["movie_id"]
            if "award_contender" in t:
                row["award_contender"] = t["award_contender"]
            if "sub_genre" in t:
                row["sub_genre"] = t["sub_genre"]
            rows.append(row)
        df = pd.DataFrame(rows)
        df = df.drop_duplicates(subset='title')
        out_path = os.path.join(edir, "title_bank.csv")
        existing_count = 0
        if os.path.exists(out_path):
            try:
                existing_count = sum(1 for _ in open(out_path, encoding="utf-8")) - 1
            except Exception:
                existing_count = 0
        if existing_count > len(df):
            print(f"title_bank.csv already has {existing_count} titles > {len(df)} from JSON — keeping existing.")
        else:
            df.to_csv(out_path, index=False)
            print(f"Saved title_bank.csv ({len(df)} titles)")

    # ─── Character bank ───────────────────────────────────────────────
    char_path = os.path.join(edir, "characters.json")
    if os.path.exists(char_path):
        chars = load_json_batch(char_path)
        rows = [{"character_name": c["character_name"],
                 "archetype": c.get("archetype", "Supporting")} for c in chars]
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(edir, "character_bank.csv"), index=False)
        print(f"Saved character_bank.csv ({len(df)} characters)")


# ═══════════════════════════════════════════════════════════════════════
# MOCK DATA GENERATOR (for dry runs without LLM)
# ═══════════════════════════════════════════════════════════════════════

def generate_mock_entities(base_dir: str, n_persons: int = 50, n_companies: int = 30,
                           n_keywords: int = 40, n_titles: int = 200, n_chars: int = 100):
    """Generate minimal mock entities for dry-run testing."""
    rng = random.Random(42)
    np_rng = np.random.RandomState(42)
    edir = os.path.join(base_dir, "entities")
    os.makedirs(edir, exist_ok=True)

    first_names_m = ["James", "Marcus", "Viktor", "Hiroshi", "Omar", "Jin", "Nikolai",
                     "Sebastian", "Diego", "Kwame", "Raj", "Leo", "Felix", "Anton"]
    first_names_f = ["Elena", "Priya", "Yuki", "Celine", "Maya", "Sofia", "Isabella",
                     "Aria", "Luna", "Zara", "Mei", "Ingrid", "Fatima", "Amara"]
    last_names = ["Storm", "Vance", "Volkov", "Chen", "Kim", "Moreau", "Patel",
                  "Torres", "Okafor", "Sato", "Fischer", "Berg", "Cruz", "Kovacs",
                  "Abbas", "Petrov", "Lund", "Silva", "Diaz", "Hart", "Blake",
                  "Frost", "Drake", "Steele", "Fox", "Night", "Stone", "Reed"]

    # Persons
    persons = []
    used_names = set()
    for i in range(n_persons):
        gender = rng.choice(["M", "M", "M", "F", "F", "F", "F", "NB"])
        first = rng.choice(first_names_m if gender == "M" else first_names_f)
        last = rng.choice(last_names)
        name = f"{first} {last}"
        while name in used_names:
            last = rng.choice(last_names)
            name = f"{first} {last}"
        used_names.add(name)

        stage = rng.choice(CAREER_STAGES)
        roles = ["actor"]
        if i < n_persons * 0.2:  # 20% are also directors
            roles.append("director")

        persons.append({
            "name": name,
            "nationality": rng.choice(NATIONALITIES),
            "gender": gender,
            "bio": f"A {stage} {rng.choice(STYLE_TAGS)} performer known for {rng.choice(GENRES).lower()} roles.",
            "style_tags": rng.sample(STYLE_TAGS, rng.randint(2, 4)),
            "genre_affinity": rng.sample(GENRES, rng.randint(1, 3)),
            "career_stage": stage,
            "roles": roles,
            "market_fit": rng.sample(MARKETS, rng.randint(1, 2)),
        })
    save_json_batch(persons, os.path.join(edir, "persons.json"))

    # Companies
    company_prefixes = ["Apex", "Nova", "Zenith", "Eclipse", "Prism", "Titan",
                        "Nebula", "Solaris", "Vortex", "Atlas", "Phoenix", "Onyx",
                        "Stellar", "Crimson", "Azure", "Polar", "Emerald"]
    company_suffixes = ["Studios", "Pictures", "Films", "Entertainment", "Media",
                        "Productions", "Cinema", "Cinematic", "Motion Pictures"]
    companies = []
    used_cnames = set()
    for i in range(n_companies):
        name = f"{rng.choice(company_prefixes)} {rng.choice(company_suffixes)}"
        while name in used_cnames:
            name = f"{rng.choice(company_prefixes)} {rng.choice(company_suffixes)}"
        used_cnames.add(name)
        companies.append({
            "name": name,
            "country": rng.choices(
                list(COMPANY_COUNTRY_WEIGHTS.keys()),
                weights=list(COMPANY_COUNTRY_WEIGHTS.values()),
                k=1,
            )[0],
            "description": f"A {rng.choice(COMPANY_TIERS).lower()} studio specializing in {rng.choice(GENRES).lower()}.",
            "specialty_genres": rng.sample(GENRES, rng.randint(1, 3)),
            "tier": rng.choice(COMPANY_TIERS),
            "preferred_actor_styles": rng.sample(STYLE_TAGS, rng.randint(1, 3)),
            "preferred_director_styles": rng.sample(DIRECTOR_STYLES, rng.randint(1, 3)),
        })
    save_json_batch(companies, os.path.join(edir, "companies.json"))

    # Keywords
    kw_topics = ["revenge", "betrayal", "survival", "love", "deception", "justice",
                 "heist", "conspiracy", "redemption", "identity", "war", "escape",
                 "sacrifice", "corruption", "discovery", "transformation", "obsession",
                 "haunting", "chase", "rivalry", "prophecy", "invasion", "paradox"]
    keywords = []
    for i in range(n_keywords):
        kw = rng.choice(kw_topics) if i < len(kw_topics) else f"topic_{i}"
        keywords.append({
            "keyword": kw,
            "topic_genre": rng.choice(GENRES),
        })
    save_json_batch(keywords, os.path.join(edir, "keywords.json"))

    # Titles
    adjectives = ["Dark", "Silent", "Broken", "Eternal", "Final", "Lost", "Hidden"]
    nouns = ["Shadow", "Storm", "Fire", "Crown", "Edge", "Ghost", "Mirror", "Code"]
    titles = []
    used_titles = set()
    for i in range(n_titles):
        t = f"The {rng.choice(adjectives)} {rng.choice(nouns)}"
        while t in used_titles:
            t = f"{rng.choice(adjectives)} {rng.choice(nouns)} {rng.randint(1,999)}"
        used_titles.add(t)
        titles.append({
            "title": t,
            "tagline": f"A story of {rng.choice(kw_topics)}.",
            "genre_hint": rng.choice(GENRES),
        })
    save_json_batch(titles, os.path.join(edir, "titles.json"))

    # Characters
    char_firsts = ["Jack", "Rose", "Blade", "Nova", "Rex", "Ivy", "Max", "Stone",
                   "Raven", "Ash", "Echo", "Storm", "Volt", "Sage", "Cruz"]
    chars = []
    for i in range(n_chars):
        chars.append({
            "character_name": f"{rng.choice(char_firsts)} {rng.choice(last_names)}",
            "archetype": rng.choice(ARCHETYPES),
        })
    save_json_batch(chars, os.path.join(edir, "characters.json"))

    # Mock edges
    gdir = os.path.join(base_dir, "graph")
    os.makedirs(gdir, exist_ok=True)
    edges = []
    for i in range(0, min(30, n_persons), 2):
        edges.append({
            "src": persons[i]["name"], "dst": persons[(i+1) % n_persons]["name"],
            "edge_type": "friendship", "weight": round(rng.uniform(0.4, 0.9), 2),
            "reason": "Shared training background"
        })
    for i in range(0, min(10, n_persons), 3):
        edges.append({
            "src": persons[i]["name"], "dst": persons[(i+2) % n_persons]["name"],
            "edge_type": "rivalry", "weight": round(rng.uniform(0.5, 1.0), 2),
            "reason": "Award competition"
        })
    # Director prefs
    for i in range(n_persons):
        if "director" in persons[i]["roles"]:
            for j in rng.sample(range(n_persons), min(5, n_persons)):
                if j != i:
                    edges.append({
                        "src": persons[i]["name"], "dst": persons[j]["name"],
                        "edge_type": "mentorship", "sign": "+",
                        "weight": round(rng.uniform(0.5, 0.9), 2),
                        "reason": "Compatible styles"
                    })
    save_json_batch(edges, os.path.join(gdir, "mock_edges.json"))

    print(f"\nMock data generated: {n_persons} persons, {n_companies} companies, "
          f"{n_keywords} keywords, {n_titles} titles, {n_chars} characters, {len(edges)} edges")


if __name__ == "__main__":
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    cmd = sys.argv[2] if len(sys.argv) > 2 else "convert"

    if cmd == "mock":
        # Safety: do not overwrite real LLM entities by accident
        if os.path.exists(os.path.join(base, "entities", "persons.json")) or os.path.exists(os.path.join(base, "entities", "companies.json")):
            raise SystemExit("Refusing to overwrite existing entities/*.json in mock mode. Use cmd=mock_force if you really want to regenerate mock JSON.")
        generate_mock_entities(base)
        json_to_csv(base)
    elif cmd == "mock_force":
        generate_mock_entities(base)
        json_to_csv(base)
    elif cmd == "convert":
        json_to_csv(base)
    else:
        print(f"Usage: python entities_to_csv.py [base_dir] [mock|mock_force|convert]")
