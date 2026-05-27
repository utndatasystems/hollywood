from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from bootstrap_artifacts import audit_critic_report
from contracts import GENRES
from text_polish import (
    contains_placeholder_syntax,
    looks_like_title_phrase,
    looks_like_weak_tagline,
    sanitize_alternate_title,
    sanitize_tagline,
    tagline_signature,
)


DEFAULT_CRITIC_SAMPLE_SIZE = 12
DEFAULT_CRITIC_MAX_ACTIONS = 20
DEFAULT_CRITIC_MAX_REPAIRS_PER_TITLE = 2

_GENERIC_PLOT_OPENING_PATTERNS = (
    r"^a high-stakes\s+[a-z-]+\s+following\b",
    r"^an investigative look at\b",
    r"^a visually striking\s+[a-z-]+\s+film\b",
    r"^(a|an)\s+dysfunctional\s+family\b",
    r"^(a|an)\s+group of\s+[a-z-]+",
    r"^(a|an)\s+team of\s+[a-z-]+",
    r"^in a near-future\b",
    r"^in a vibrant animated world\b",
)

_CANON_GENRE_MAP = {str(genre).strip().lower(): str(genre) for genre in GENRES}
_CANON_GENRE_MAP.update({
    "film noir": "Film-Noir",
    "film-noir": "Film-Noir",
    "sci fi": "Sci-Fi",
    "sci-fi": "Sci-Fi",
    "scifi": "Sci-Fi",
    "martial arts": "Martial Arts",
    "super hero": "Superhero",
})


def _canonical_genre_label(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return _CANON_GENRE_MAP.get(text.lower(), text)


def _split_plot_sentences(text: str) -> list[str]:
    value = " ".join(str(text or "").split()).strip()
    if not value:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", value) if part.strip()]


def _looks_generic_plot_summary(text: str) -> bool:
    value = " ".join(str(text or "").split()).strip()
    low = value.lower()
    if not value:
        return True
    sentences = _split_plot_sentences(value)
    words = len(value.split())
    if len(sentences) < 2 or len(sentences) > 4:
        return True
    if words < 28 or words > 110:
        return True
    if any(re.search(pattern, low) for pattern in _GENERIC_PLOT_OPENING_PATTERNS):
        return True
    if low.startswith(("a ", "an ")) and " film from " in low:
        return True
    if "synthetic" in low:
        return True
    return False


def _duplicate_tagline_clusters(movies: pd.DataFrame | None) -> dict[str, Any]:
    if movies is None or len(movies) == 0 or 'title_id' not in movies.columns or 'tagline' not in movies.columns:
        return {"title_ids": set(), "counts": {}, "clusters": []}
    work = movies[['title_id', 'tagline']].copy()
    work['tagline_signature'] = work['tagline'].map(tagline_signature)
    work = work[work['tagline_signature'].astype(str) != ""]
    if len(work) == 0:
        return {"title_ids": set(), "counts": {}, "clusters": []}
    grouped = (
        work.groupby('tagline_signature')['title_id']
        .apply(lambda rows: [int(v) for v in rows.tolist()])
        .to_dict()
    )
    clusters = [
        {"signature": str(signature), "title_ids": ids, "count": len(ids)}
        for signature, ids in grouped.items()
        if len(ids) > 1
    ]
    clusters.sort(key=lambda row: (-int(row["count"]), min(row["title_ids"])))
    title_ids = {int(title_id) for row in clusters for title_id in row["title_ids"]}
    counts = {int(title_id): int(row["count"]) for row in clusters for title_id in row["title_ids"]}
    return {"title_ids": title_ids, "counts": counts, "clusters": clusters}


def _persist_critic_logs(log_dir: str | None, prompt: str, raw_text: str, report: dict[str, Any]) -> None:
    if not log_dir:
        return
    try:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, 'post_generation_critic_prompt.txt'), 'w', encoding='utf-8') as f:
            f.write(str(prompt or ''))
        with open(os.path.join(log_dir, 'post_generation_critic_raw.json'), 'w', encoding='utf-8') as f:
            f.write(str(raw_text or ''))
        with open(os.path.join(log_dir, 'post_generation_critic_report.json'), 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _persist_and_audit_critic(log_dir: str | None, prompt: str, raw_text: str, report: dict[str, Any]) -> None:
    _persist_critic_logs(log_dir, prompt, raw_text, report)
    try:
        audit_critic_report(report)
    except Exception:
        pass


def _safe_json_loads(raw_text: str) -> Any:
    text = str(raw_text or '').strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    if '```' in text:
        parts = text.split('```')
        for part in parts:
            candidate = part.strip()
            if candidate.lower().startswith('json'):
                candidate = candidate[4:].strip()
            if not candidate:
                continue
            try:
                return json.loads(candidate)
            except Exception:
                continue
    start = text.find('{')
    end = text.rfind('}')
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return None
    return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _stable_seed(*parts: Any) -> int:
    raw = '|'.join(str(part) for part in parts).encode('utf-8')
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


def _keyword_column(keyword_df: pd.DataFrame) -> str | None:
    for col in ('keyword', 'name', 'text'):
        if col in keyword_df.columns:
            return col
    if len(keyword_df.columns) >= 2:
        return str(keyword_df.columns[1])
    return None


def _keyword_lookup(world) -> dict[str, int]:
    keyword_df = getattr(world, 'keywords', None)
    if keyword_df is None or len(keyword_df) == 0 or 'keyword_id' not in keyword_df.columns:
        return {}
    col = _keyword_column(keyword_df)
    if not col:
        return {}
    lookup: dict[str, int] = {}
    for row in keyword_df[['keyword_id', col]].itertuples(index=False):
        keyword_id = int(getattr(row, 'keyword_id'))
        text = str(getattr(row, col) or '').strip().lower()
        if text:
            lookup[text] = keyword_id
    return lookup


def _name_lookup(df: pd.DataFrame | None, id_col: str, name_col: str = 'name') -> dict[int, str]:
    if df is None or len(df) == 0 or id_col not in df.columns or name_col not in df.columns:
        return {}
    lookup = {}
    for row in df[[id_col, name_col]].itertuples(index=False):
        try:
            lookup[int(getattr(row, id_col))] = str(getattr(row, name_col) or '')
        except Exception:
            continue
    return lookup


def _precritic_structural_gate(result: dict[str, pd.DataFrame], world) -> dict[str, Any]:
    movies = result.get('movie')
    if movies is None or len(movies) == 0 or 'title_id' not in movies.columns:
        return {"placeholder_title_ids": [], "zero_exact_topic_support_ids": []}

    placeholder_title_ids: list[int] = []
    if 'tagline' in movies.columns:
        placeholder_title_ids = [
            int(row.title_id)
            for row in movies[['title_id', 'tagline']].itertuples(index=False)
            if contains_placeholder_syntax(str(getattr(row, 'tagline') or ''))
        ]

    zero_exact_topic_support_ids: list[int] = []
    movie_keyword = result.get('movie_keyword')
    keyword_df = getattr(world, 'keywords', None)
    if (
        movie_keyword is not None
        and len(movie_keyword) > 0
        and keyword_df is not None
        and len(keyword_df) > 0
        and 'keyword_id' in movie_keyword.columns
        and 'keyword_id' in keyword_df.columns
        and 'topic_genre' in keyword_df.columns
        and 'genre' in movies.columns
    ):
        topic_lookup = {
            int(row.keyword_id): str(getattr(row, 'topic_genre') or '').strip()
            for row in keyword_df[['keyword_id', 'topic_genre']].itertuples(index=False)
        }
        support_counts: Counter[int] = Counter()
        movie_genre_lookup = {
            int(row.title_id): str(getattr(row, 'genre') or '')
            for row in movies[['title_id', 'genre']].itertuples(index=False)
        }
        for row in movie_keyword[['title_id', 'keyword_id']].itertuples(index=False):
            title_id = int(getattr(row, 'title_id'))
            keyword_id = int(getattr(row, 'keyword_id'))
            topic = _canonical_genre_label(topic_lookup.get(keyword_id, ''))
            if not topic:
                continue
            movie_genre = _canonical_genre_label(movie_genre_lookup.get(title_id, ''))
            if topic == movie_genre:
                support_counts[title_id] += 1
        zero_exact_topic_support_ids = [
            int(row.title_id)
            for row in movies[['title_id', 'genre']].itertuples(index=False)
            if int(support_counts.get(int(row.title_id), 0)) <= 0
        ]

    return {
        "placeholder_title_ids": placeholder_title_ids,
        "zero_exact_topic_support_ids": zero_exact_topic_support_ids,
    }


def _summarize_dataset(result: dict[str, pd.DataFrame]) -> dict[str, Any]:
    movies = result.get('movie')
    cast_info = result.get('cast_info')
    movie_companies = result.get('movie_companies')
    awards = result.get('awards')
    if movies is None or len(movies) == 0:
        return {'movie_count': 0}

    movie_count = int(len(movies))
    cast_counts = cast_info.groupby('title_id').size().to_dict() if cast_info is not None and len(cast_info) > 0 and 'title_id' in cast_info.columns else {}
    company_counts = movie_companies.groupby('title_id').size().to_dict() if movie_companies is not None and len(movie_companies) > 0 and 'title_id' in movie_companies.columns else {}
    award_counts = awards.groupby('title_id').size().to_dict() if awards is not None and len(awards) > 0 and 'title_id' in awards.columns else {}

    budgets = movies['budget_usd'].astype(float) if 'budget_usd' in movies.columns else pd.Series(dtype=float)
    grosses = movies['box_office_usd'].astype(float) if 'box_office_usd' in movies.columns else pd.Series(dtype=float)
    ratings = movies['rating'].astype(float) if 'rating' in movies.columns else pd.Series(dtype=float)

    return {
        'movie_count': movie_count,
        'year_range': [int(movies['year'].min()), int(movies['year'].max())] if 'year' in movies.columns and movie_count else [],
        'genre_counts': movies['genre'].value_counts().head(8).to_dict() if 'genre' in movies.columns else {},
        'country_counts': movies['country'].value_counts().head(8).to_dict() if 'country' in movies.columns else {},
        'avg_cast_size': round(float(np.mean(list(cast_counts.values()))), 3) if cast_counts else 0.0,
        'multi_company_rate': round(float(np.mean([1.0 if cnt > 1 else 0.0 for cnt in company_counts.values()])), 3) if company_counts else 0.0,
        'award_rate': round(float(np.mean([1.0 if award_counts.get(int(mid), 0) > 0 else 0.0 for mid in movies['title_id'].tolist()])), 3) if award_counts else 0.0,
        'budget_boxoffice_corr': round(float(budgets.corr(grosses)), 3) if len(budgets) > 1 and len(grosses) > 1 else 0.0,
        'rating_mean': round(float(ratings.mean()), 3) if len(ratings) else 0.0,
    }


def select_movies_for_critic(result: dict[str, pd.DataFrame], world, sample_size: int = DEFAULT_CRITIC_SAMPLE_SIZE) -> list[int]:
    movies = result.get('movie')
    if movies is None or len(movies) == 0:
        return []

    cast_info = result.get('cast_info')
    movie_companies = result.get('movie_companies')
    movie_keyword = result.get('movie_keyword')
    awards = result.get('awards')

    cast_counts = cast_info.groupby('title_id').size().to_dict() if cast_info is not None and len(cast_info) > 0 and 'title_id' in cast_info.columns else {}
    company_counts = movie_companies.groupby('title_id').size().to_dict() if movie_companies is not None and len(movie_companies) > 0 and 'title_id' in movie_companies.columns else {}
    keyword_counts = movie_keyword.groupby('title_id').size().to_dict() if movie_keyword is not None and len(movie_keyword) > 0 and 'title_id' in movie_keyword.columns else {}
    award_counts = awards.groupby('title_id').size().to_dict() if awards is not None and len(awards) > 0 and 'title_id' in awards.columns else {}
    duplicate_clusters = _duplicate_tagline_clusters(movies)
    duplicate_title_ids = set(duplicate_clusters["title_ids"])
    duplicate_counts = dict(duplicate_clusters["counts"])

    work = movies.copy()
    work['cast_count'] = work['title_id'].map(cast_counts).fillna(0).astype(int)
    work['company_count'] = work['title_id'].map(company_counts).fillna(0).astype(int)
    work['keyword_count'] = work['title_id'].map(keyword_counts).fillna(0).astype(int)
    work['award_count'] = work['title_id'].map(award_counts).fillna(0).astype(int)
    work['budget_num'] = work['budget_usd'].map(lambda v: _safe_float(v, 0.0)) if 'budget_usd' in work.columns else 0.0
    work['gross_num'] = work['box_office_usd'].map(lambda v: _safe_float(v, 0.0)) if 'box_office_usd' in work.columns else 0.0
    work['rating_num'] = work['rating'].map(lambda v: _safe_float(v, 0.0)) if 'rating' in work.columns else 0.0
    work['performance_ratio'] = work['gross_num'] / work['budget_num'].clip(lower=1.0)
    work['summary_words'] = work['plot_summary'].fillna('').astype(str).str.split().str.len() if 'plot_summary' in work.columns else 0
    work['tagline_words'] = work['tagline'].fillna('').astype(str).str.split().str.len() if 'tagline' in work.columns else 0
    work['tagline_duplicate_count'] = work['title_id'].map(duplicate_counts).fillna(1).astype(int)

    perf_median = work.groupby('genre')['performance_ratio'].median().to_dict() if 'genre' in work.columns else {}
    rating_median = work.groupby('genre')['rating_num'].median().to_dict() if 'genre' in work.columns else {}

    def _row_score(row) -> float:
        genre = str(getattr(row, 'genre', '') or '')
        title = str(getattr(row, 'title', '') or '')
        tagline = str(getattr(row, 'tagline', '') or '')
        plot_summary = str(getattr(row, 'plot_summary', '') or '')
        summary_words = int(getattr(row, 'summary_words', 0) or 0)
        tagline_words = int(getattr(row, 'tagline_words', 0) or 0)
        keyword_count = int(getattr(row, 'keyword_count', 0) or 0)
        company_count = int(getattr(row, 'company_count', 0) or 0)
        cast_count = int(getattr(row, 'cast_count', 0) or 0)
        award_count = int(getattr(row, 'award_count', 0) or 0)
        budget_num = float(getattr(row, 'budget_num', 0.0) or 0.0)
        rating_num = float(getattr(row, 'rating_num', 0.0) or 0.0)
        perf_ratio = float(getattr(row, 'performance_ratio', 0.0) or 0.0)
        genre_perf = float(perf_median.get(genre, perf_ratio) or perf_ratio)
        genre_rating = float(rating_median.get(genre, rating_num) or rating_num)

        score = 0.0
        if summary_words < 28:
            score += 1.6
        if _looks_generic_plot_summary(plot_summary):
            score += 1.2
        if tagline_words == 0:
            score += 0.8
        elif looks_like_weak_tagline(tagline, title=title):
            score += 1.4
        elif looks_like_title_phrase(tagline):
            score += 0.8
        if int(getattr(row, 'tagline_duplicate_count', 1) or 1) > 1:
            score += min(2.0, 0.45 * int(getattr(row, 'tagline_duplicate_count', 1) or 1))
        if keyword_count < 3:
            score += 0.8
        if cast_count < 4 or cast_count > 12:
            score += 0.7
        if budget_num >= 45_000_000 and company_count <= 1:
            score += 1.0
        if abs(perf_ratio - genre_perf) >= 2.2:
            score += 0.9
        if abs(rating_num - genre_rating) >= 1.5:
            score += 0.6
        if rating_num >= 7.7 and award_count == 0:
            score += 0.5
        if title.lower().startswith('untitled-'):
            score += 1.5
        return float(score)

    work['critic_risk_score'] = [_row_score(row) for row in work.itertuples(index=False)]
    ranked = work.sort_values(['critic_risk_score', 'year', 'title_id'], ascending=[False, True, True])

    max_per_genre = max(2, int(np.ceil(sample_size / 4)))
    selected: list[int] = []
    seen_genres: Counter[str] = Counter()
    for cluster in duplicate_clusters["clusters"][: max(1, min(4, sample_size // 3))]:
        for title_id in cluster["title_ids"][: min(3, len(cluster["title_ids"]))]:
            if int(title_id) in selected:
                continue
            selected.append(int(title_id))
            genre_row = work.loc[work['title_id'] == int(title_id), 'genre']
            genre = str(genre_row.iloc[0]) if len(genre_row) > 0 else ''
            seen_genres[genre] += 1
            if len(selected) >= sample_size:
                return selected[:sample_size]
    for row in ranked.itertuples(index=False):
        title_id = int(getattr(row, 'title_id'))
        genre = str(getattr(row, 'genre', '') or '')
        if title_id in selected:
            continue
        if seen_genres[genre] >= max_per_genre and len(selected) < max(0, sample_size - 2):
            continue
        selected.append(title_id)
        seen_genres[genre] += 1
        if len(selected) >= sample_size:
            break

    if len(selected) < sample_size:
        for title_id in ranked['title_id'].tolist():
            tid = int(title_id)
            if tid in selected:
                continue
            selected.append(tid)
            if len(selected) >= sample_size:
                break
    return selected


def _build_movie_dossiers(result: dict[str, pd.DataFrame], world, title_ids: list[int]) -> list[dict[str, Any]]:
    movies = result.get('movie')
    if movies is None or len(movies) == 0 or not title_ids:
        return []

    cast_info = result.get('cast_info')
    movie_companies = result.get('movie_companies')
    movie_keyword = result.get('movie_keyword')
    awards = result.get('awards')
    alternate_titles = result.get('alternate_titles')

    people = _name_lookup(getattr(world, 'persons', None), 'person_id')
    companies = _name_lookup(getattr(world, 'companies', None), 'company_id')
    keywords_lookup = {}
    keyword_df = getattr(world, 'keywords', None)
    keyword_col = _keyword_column(keyword_df) if keyword_df is not None else None
    if keyword_df is not None and keyword_col and 'keyword_id' in keyword_df.columns:
        keywords_lookup = {int(row.keyword_id): str(getattr(row, keyword_col) or '') for row in keyword_df[['keyword_id', keyword_col]].itertuples(index=False)}

    cast_by_title = {}
    if cast_info is not None and len(cast_info) > 0:
        work = cast_info.sort_values(['title_id', 'billing_order']) if 'billing_order' in cast_info.columns else cast_info
        for title_id, chunk in work.groupby('title_id'):
            cast_by_title[int(title_id)] = [people.get(int(pid), str(pid)) for pid in chunk['person_id'].head(5).tolist()]

    company_by_title = {}
    if movie_companies is not None and len(movie_companies) > 0:
        for title_id, chunk in movie_companies.groupby('title_id'):
            labels = []
            for row in chunk.itertuples(index=False):
                cid = int(getattr(row, 'company_id'))
                role = str(getattr(row, 'role', '') or '')
                label = companies.get(cid, str(cid))
                if role:
                    label = f'{label} ({role})'
                labels.append(label)
            company_by_title[int(title_id)] = labels[:4]

    keyword_by_title = {}
    if movie_keyword is not None and len(movie_keyword) > 0:
        for title_id, chunk in movie_keyword.groupby('title_id'):
            keyword_by_title[int(title_id)] = [keywords_lookup.get(int(kid), str(kid)) for kid in chunk['keyword_id'].head(8).tolist()]

    award_by_title = {}
    if awards is not None and len(awards) > 0:
        for title_id, chunk in awards.groupby('title_id'):
            award_by_title[int(title_id)] = {
                'count': int(len(chunk)),
                'wins': int((chunk['outcome'].astype(str).str.lower() == 'win').sum()) if 'outcome' in chunk.columns else 0,
            }

    alt_by_title = {}
    if alternate_titles is not None and len(alternate_titles) > 0 and {'title_id', 'language', 'alt_title'}.issubset(alternate_titles.columns):
        for title_id, chunk in alternate_titles.groupby('title_id'):
            alt_by_title[int(title_id)] = [
                {'language': str(row.language), 'alt_title': str(row.alt_title)}
                for row in chunk.head(3).itertuples(index=False)
            ]

    dossiers: list[dict[str, Any]] = []
    movie_by_id = movies.set_index('title_id')
    for title_id in title_ids:
        if int(title_id) not in movie_by_id.index:
            continue
        row = movie_by_id.loc[int(title_id)]
        dossiers.append({
            'title_id': int(title_id),
            'title': str(row.get('title', '') or ''),
            'year': int(row.get('year', 0) or 0),
            'genre': str(row.get('genre', '') or ''),
            'country': str(row.get('country', '') or ''),
            'production_tier': str(row.get('production_tier', '') or ''),
            'budget_usd': int(_safe_float(row.get('budget_usd'), 0.0)),
            'box_office_usd': int(_safe_float(row.get('box_office_usd'), 0.0)),
            'rating': round(_safe_float(row.get('rating'), 0.0), 2),
            'tagline': str(row.get('tagline', '') or ''),
            'plot_summary': str(row.get('plot_summary', '') or ''),
            'cast': cast_by_title.get(int(title_id), []),
            'companies': company_by_title.get(int(title_id), []),
            'keywords': keyword_by_title.get(int(title_id), []),
            'awards': award_by_title.get(int(title_id), {'count': 0, 'wins': 0}),
            'alternate_titles': alt_by_title.get(int(title_id), []),
        })
    return dossiers


def build_post_generation_critic_prompt(result: dict[str, pd.DataFrame], world, title_ids: list[int]) -> str:
    summary = _summarize_dataset(result)
    dossiers = _build_movie_dossiers(result, world, title_ids)
    duplicate_clusters = _duplicate_tagline_clusters(result.get('movie'))
    prompt = {
        'task': 'post_generation_critic',
        'goal': 'Review a bounded sample of generated films, flag coherence issues, and propose only safe deterministic repairs.',
        'dataset_summary': summary,
        'sampled_movies': dossiers,
        'duplicate_tagline_clusters': duplicate_clusters.get('clusters', [])[:6],
        'allowed_actions': {
            'rewrite_plot_summary': {
                'required_fields': ['title_id', 'plot_summary'],
                'rule': 'Write exactly 2-3 sentences and roughly 40-85 words, keep factual fields unchanged, and make the synopsis more specific, concrete, and less template-like. Avoid generic openings like "A high-stakes thriller following..." or "An investigative look at...".',
            },
            'rewrite_tagline': {
                'required_fields': ['title_id', 'tagline'],
                'rule': 'Write a 4-10 word market-facing tagline aligned with genre and tone; avoid generic inspirational slogans, avoid restating the title, and break duplicate-tagline clusters when present. When rewriting because of a duplicate cluster, set reason=duplicate_tagline_cluster.',
            },
            'append_keyword': {
                'required_fields': ['title_id', 'keyword'],
                'rule': 'Only use concise keyword text that plausibly exists in a film keyword taxonomy.',
            },
            'remove_keyword': {
                'required_fields': ['title_id', 'keyword'],
                'rule': 'Remove only if the keyword is clearly irrelevant or misleading for the movie.',
            },
            'add_alternate_title': {
                'required_fields': ['title_id', 'language', 'alt_title'],
                'rule': 'Add only if the film is likely to have a market-specific alternate title.',
            },
        },
        'constraints': [
            'Return JSON only.',
            'Do not invent new IDs or modify title_id/year/country/genre/budget/box_office/rating.',
            'Prefer no-op over weak or speculative repairs.',
            'At most 2 repairs per title and at most 20 actions overall.',
        ],
        'response_schema': {
            'critic_summary': 'string',
            'flagged_titles': [
                {'title_id': 'int', 'issues': ['string']}
            ],
            'actions': [
                {'action': 'rewrite_plot_summary|rewrite_tagline|append_keyword|remove_keyword|add_alternate_title', 'params': 'object', 'reason': 'string optional'}
            ],
        },
    }
    return json.dumps(prompt, ensure_ascii=False, indent=2)


def apply_post_generation_repairs(
    result: dict[str, pd.DataFrame],
    world,
    payload: dict[str, Any],
    *,
    max_actions: int = DEFAULT_CRITIC_MAX_ACTIONS,
    max_repairs_per_title: int = DEFAULT_CRITIC_MAX_REPAIRS_PER_TITLE,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    movies = result.get('movie')
    if movies is None or len(movies) == 0:
        return result, []

    movie_keyword = result.get('movie_keyword')
    alt_titles = result.get('alternate_titles')
    if movie_keyword is None:
        movie_keyword = pd.DataFrame(columns=['title_id', 'keyword_id'])
        result['movie_keyword'] = movie_keyword
    if alt_titles is None:
        alt_titles = pd.DataFrame(columns=['title_id', 'language', 'alt_title'])
        result['alternate_titles'] = alt_titles

    keyword_lookup = _keyword_lookup(world)
    titles_by_id = {
        int(row.title_id): str(row.title or '')
        for row in movies[['title_id', 'title']].itertuples(index=False)
    }
    movie_title_ids = set(int(v) for v in movies['title_id'].tolist())
    existing_keywords = set()
    if len(movie_keyword) > 0 and {'title_id', 'keyword_id'}.issubset(movie_keyword.columns):
        existing_keywords = {(int(row.title_id), int(row.keyword_id)) for row in movie_keyword[['title_id', 'keyword_id']].itertuples(index=False)}
    existing_alt_titles = set()
    if len(alt_titles) > 0 and {'title_id', 'language', 'alt_title'}.issubset(alt_titles.columns):
        existing_alt_titles = {
            (int(row.title_id), str(row.language).strip().lower(), str(row.alt_title).strip().lower())
            for row in alt_titles[['title_id', 'language', 'alt_title']].itertuples(index=False)
        }
    current_taglines_by_title = {
        int(row.title_id): str(row.tagline or '').strip()
        for row in movies[['title_id', 'tagline']].itertuples(index=False)
    }
    tagline_signature_counts: Counter[str] = Counter(
        tagline_signature(tagline)
        for tagline in current_taglines_by_title.values()
        if str(tagline).strip()
    )
    duplicate_tagline_title_ids = set(_duplicate_tagline_clusters(movies).get("title_ids", []))

    repairs: list[dict[str, Any]] = []
    repair_budget: Counter[int] = Counter()
    pending_keyword_rows: list[dict[str, Any]] = []
    pending_keyword_removals: set[tuple[int, int]] = set()
    pending_alt_rows: list[dict[str, Any]] = []

    actions = payload.get('actions') or []
    if not isinstance(actions, list):
        return result, repairs

    for action in actions[:max_actions]:
        if not isinstance(action, dict):
            continue
        op = str(action.get('action', '') or '').strip()
        params = action.get('params') if isinstance(action.get('params'), dict) else {}
        title_id = int(_safe_float(params.get('title_id'), 0))
        repair_reason = str(action.get('reason', '') or params.get('reason', '') or '').strip()
        if title_id not in movie_title_ids:
            repairs.append({'title_id': title_id, 'repair_type': op, 'repair_reason': repair_reason, 'status': 'skipped', 'detail': 'unknown title_id'})
            continue
        if repair_budget[title_id] >= max_repairs_per_title:
            repairs.append({'title_id': title_id, 'repair_type': op, 'repair_reason': repair_reason, 'status': 'skipped', 'detail': 'repair budget exhausted'})
            continue

        detail = ''
        applied = False
        if op == 'rewrite_plot_summary':
            plot_summary = str(params.get('plot_summary', '') or '').strip()
            if not _looks_generic_plot_summary(plot_summary):
                movies.loc[movies['title_id'] == title_id, 'plot_summary'] = plot_summary
                detail = plot_summary
                applied = True
            else:
                detail = 'plot summary weak, too short, or still generic'
        elif op == 'rewrite_tagline':
            if not repair_reason and title_id in duplicate_tagline_title_ids:
                repair_reason = 'duplicate_tagline_cluster'
            tagline = sanitize_tagline(str(params.get('tagline', '') or '').strip(), title=titles_by_id.get(title_id, ''))
            signature = tagline_signature(tagline) if tagline else ""
            old_tagline = current_taglines_by_title.get(title_id, '')
            old_signature = tagline_signature(old_tagline) if old_tagline else ""
            competing_count = int(tagline_signature_counts.get(signature, 0)) - (1 if signature and signature == old_signature and old_tagline else 0)
            if 4 <= len(tagline.split()) <= 12 and not looks_like_weak_tagline(tagline, title=titles_by_id.get(title_id, '')) and competing_count <= 0:
                if old_signature:
                    tagline_signature_counts[old_signature] = max(0, int(tagline_signature_counts.get(old_signature, 0)) - 1)
                movies.loc[movies['title_id'] == title_id, 'tagline'] = tagline
                current_taglines_by_title[title_id] = tagline
                if signature:
                    tagline_signature_counts[signature] += 1
                detail = tagline
                applied = True
            else:
                detail = 'tagline weak, malformed, or still duplicated'
        elif op == 'append_keyword':
            keyword = str(params.get('keyword', '') or '').strip().lower()
            keyword_id = keyword_lookup.get(keyword)
            if keyword_id is None:
                detail = f'keyword not found: {keyword}'
            elif (title_id, keyword_id) in existing_keywords:
                detail = f'keyword already present: {keyword}'
            else:
                pending_keyword_removals.discard((title_id, keyword_id))
                pending_keyword_rows.append({'title_id': title_id, 'keyword_id': keyword_id})
                existing_keywords.add((title_id, keyword_id))
                detail = keyword
                applied = True
        elif op == 'remove_keyword':
            keyword = str(params.get('keyword', '') or '').strip().lower()
            keyword_id = keyword_lookup.get(keyword)
            if keyword_id is None:
                detail = f'keyword not found: {keyword}'
            elif (title_id, keyword_id) not in existing_keywords:
                detail = f'keyword absent: {keyword}'
            else:
                existing_keywords.discard((title_id, keyword_id))
                pending_keyword_removals.add((title_id, keyword_id))
                detail = keyword
                applied = True
        elif op == 'add_alternate_title':
            language = str(params.get('language', '') or '').strip()
            alt_title = sanitize_alternate_title(str(params.get('alt_title', '') or '').strip())
            key = (title_id, language.lower(), alt_title.lower())
            if not language or len(alt_title.split()) < 1:
                detail = 'alternate title missing fields'
            elif key in existing_alt_titles:
                detail = f'alternate title already present: {language}/{alt_title}'
            else:
                pending_alt_rows.append({'title_id': title_id, 'language': language, 'alt_title': alt_title})
                existing_alt_titles.add(key)
                detail = f'{language}:{alt_title}'
                applied = True
        else:
            detail = 'unsupported repair action'

        status = 'applied' if applied else 'skipped'
        repairs.append({'title_id': title_id, 'repair_type': op, 'repair_reason': repair_reason, 'status': status, 'detail': detail})
        if applied:
            repair_budget[title_id] += 1

    if pending_keyword_rows or pending_keyword_removals:
        work_keywords = movie_keyword.copy()
        if pending_keyword_removals and len(work_keywords) > 0:
            remove_df = pd.DataFrame(sorted(pending_keyword_removals), columns=['title_id', 'keyword_id'])
            work_keywords = work_keywords.merge(remove_df.assign(__drop__=1), on=['title_id', 'keyword_id'], how='left')
            work_keywords = work_keywords[work_keywords['__drop__'].isna()].drop(columns='__drop__')
        if pending_keyword_rows:
            work_keywords = pd.concat([work_keywords, pd.DataFrame(pending_keyword_rows)], ignore_index=True)
        result['movie_keyword'] = work_keywords.reset_index(drop=True)
    if pending_alt_rows:
        result['alternate_titles'] = pd.concat([alt_titles, pd.DataFrame(pending_alt_rows)], ignore_index=True)
    result['movie'] = movies
    return result, repairs


def run_post_generation_critic(
    result: dict[str, pd.DataFrame],
    world,
    *,
    enabled: bool = True,
    llm_manager=None,
    model: str | None = None,
    log_dir: str | None = None,
    sample_size: int | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    if not enabled:
        report = {'status': 'disabled', 'applied': 0, 'sampled_titles': []}
        audit_critic_report(report)
        return result, report

    workspace = getattr(world, 'workspace', None)
    priors = getattr(getattr(workspace, 'config', SimpleNamespace()), 'priors', SimpleNamespace())
    sample_size = int(sample_size or getattr(priors, 'critic_sample_size', DEFAULT_CRITIC_SAMPLE_SIZE))
    max_actions = int(getattr(priors, 'critic_max_actions', DEFAULT_CRITIC_MAX_ACTIONS))
    max_repairs_per_title = int(getattr(priors, 'critic_max_repairs_per_title', DEFAULT_CRITIC_MAX_REPAIRS_PER_TITLE))
    movies = result.get('movie')
    movie_count = int(len(movies)) if movies is not None else 0
    coverage_floor = max(sample_size, int(np.ceil(float(movie_count) * 0.12))) if movie_count > 0 else sample_size
    sample_size = min(movie_count, max(sample_size, coverage_floor)) if movie_count > 0 else sample_size
    max_actions = min(40, max(max_actions, sample_size * 2))

    structural_gate = _precritic_structural_gate(result, world)
    placeholder_title_ids = list(structural_gate.get('placeholder_title_ids') or [])
    zero_exact_topic_support_ids = list(structural_gate.get('zero_exact_topic_support_ids') or [])
    if placeholder_title_ids or zero_exact_topic_support_ids:
        reason_parts: list[str] = []
        if placeholder_title_ids:
            reason_parts.append(f"placeholder taglines in title_ids={placeholder_title_ids[:8]}")
        if zero_exact_topic_support_ids:
            reason_parts.append(f"zero exact-topic keyword support in title_ids={zero_exact_topic_support_ids[:8]}")
        reason_text = "; ".join(reason_parts) or "pre-critic structural gate failed"
        report = {
            'status': 'structural_failure',
            'reason': reason_text,
            'applied': 0,
            'sampled_titles': [],
            'placeholder_title_ids': placeholder_title_ids,
            'zero_exact_topic_support_ids': zero_exact_topic_support_ids,
            'non_fatal': True,
        }
        _persist_and_audit_critic(log_dir, '', '', report)
        return result, report

    sampled_titles = select_movies_for_critic(result, world, sample_size=sample_size)
    if not sampled_titles:
        report = {'status': 'empty', 'applied': 0, 'sampled_titles': []}
        audit_critic_report(report)
        return result, report

    prompt = build_post_generation_critic_prompt(result, world, sampled_titles)
    raw_text = ''
    parsed = None
    cache_hit = False
    _persist_critic_logs(
        log_dir,
        prompt,
        raw_text,
        {
            'status': 'started',
            'applied': 0,
            'sampled_titles': sampled_titles,
        },
    )

    manager = llm_manager
    if manager is None and workspace is not None:
        try:
            from llm_provider import LLMManager
            manager = LLMManager(workspace)
        except Exception as exc:
            report = {
                'status': 'skipped',
                'reason': f'critic manager unavailable: {exc}',
                'applied': 0,
                'sampled_titles': sampled_titles,
            }
            _persist_and_audit_critic(log_dir, prompt, raw_text, report)
            return result, report
    if manager is None:
        report = {'status': 'skipped', 'reason': 'no llm manager', 'applied': 0, 'sampled_titles': sampled_titles}
        _persist_and_audit_critic(log_dir, prompt, raw_text, report)
        return result, report

    try:
        response, _inp, _out, _cost, cache_hit = manager.generate(
            role='critic',
            contents=prompt,
            schema_name='post_generation_critic',
            seed=_stable_seed(getattr(world, 'seed', 0), 'post_generation_critic', ','.join(map(str, sampled_titles))),
            model=model,
        )
        raw_text = str(getattr(response, 'text', '') or '')
        parsed = _safe_json_loads(raw_text)
    except Exception as exc:
        report = {
            'status': 'error',
            'reason': str(exc),
            'applied': 0,
            'sampled_titles': sampled_titles,
            'cache_hit': False,
        }
        _persist_and_audit_critic(log_dir, prompt, raw_text, report)
        return result, report

    if not isinstance(parsed, dict):
        report = {
            'status': 'invalid',
            'reason': 'critic output was not a JSON object',
            'applied': 0,
            'sampled_titles': sampled_titles,
            'cache_hit': cache_hit,
        }
        _persist_and_audit_critic(log_dir, prompt, raw_text, report)
        return result, report

    result, repairs = apply_post_generation_repairs(
        result,
        world,
        parsed,
        max_actions=max_actions,
        max_repairs_per_title=max_repairs_per_title,
    )
    repairs_df = pd.DataFrame(repairs)
    result['critic_repairs'] = repairs_df
    repair_types = Counter(str(row.get('repair_type', '') or '').strip() for row in repairs if isinstance(row, dict) and str(row.get('repair_type', '') or '').strip())
    repair_reasons = Counter(str(row.get('repair_reason', '') or '').strip() for row in repairs if isinstance(row, dict) and str(row.get('repair_reason', '') or '').strip())
    llm_rewrite_actions = int(sum(1 for row in repairs if isinstance(row, dict) and str(row.get('status', '') or '') == 'applied' and str(row.get('repair_type', '') or '') in {'rewrite_plot_summary', 'rewrite_tagline'}))
    deterministic_sanitation_actions = int(sum(1 for row in repairs if isinstance(row, dict) and str(row.get('status', '') or '') == 'applied' and str(row.get('repair_type', '') or '') not in {'rewrite_plot_summary', 'rewrite_tagline'}))
    duplicate_tagline_rewrite_actions = int(sum(
        1
        for row in repairs
        if isinstance(row, dict)
        and str(row.get('status', '') or '') == 'applied'
        and str(row.get('repair_type', '') or '') == 'rewrite_tagline'
        and str(row.get('repair_reason', '') or '') == 'duplicate_tagline_cluster'
    ))
    report = {
        'status': 'ok',
        'sampled_titles': sampled_titles,
        'flagged_titles': parsed.get('flagged_titles', []),
        'critic_summary': parsed.get('critic_summary', ''),
        'applied': int((repairs_df['status'] == 'applied').sum()) if len(repairs_df) > 0 and 'status' in repairs_df.columns else 0,
        'skipped': int((repairs_df['status'] == 'skipped').sum()) if len(repairs_df) > 0 and 'status' in repairs_df.columns else 0,
        'actions_proposed': len(parsed.get('actions', []) or []),
        'cache_hit': bool(cache_hit),
        'repairs': repairs,
        'repair_types': dict(repair_types),
        'repair_reasons': dict(repair_reasons),
        'llm_rewrite_actions': llm_rewrite_actions,
        'deterministic_sanitation_actions': deterministic_sanitation_actions,
        'duplicate_tagline_rewrite_actions': duplicate_tagline_rewrite_actions,
    }

    _persist_and_audit_critic(log_dir, prompt, raw_text, report)

    return result, report
