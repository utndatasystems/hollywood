from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Iterable


_ARTICLES = {
    "a",
    "an",
    "the",
    "le",
    "la",
    "les",
    "der",
    "die",
    "das",
    "el",
    "los",
    "las",
    "il",
    "lo",
    "gli",
    "i",
    "o",
    "os",
    "un",
    "una",
    "um",
}
_ARTICLE_ALLOW_PREV = {
    "and",
    "at",
    "before",
    "behind",
    "beneath",
    "beyond",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "over",
    "through",
    "to",
    "under",
    "versus",
    "vs",
    "with",
    "within",
}
_SHORT_TITLE_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "the",
    "to",
    "vs",
    "with",
}
_BAD_TAGLINE_PATTERNS = (
    r"[.!?]\s+[a-z]",
    r"\bwill\s+die\s+them\s+all\b",
    r"\b(?:his|her|their)\s+(?:ocean|river|forest|desert|city|brother|sister)\b",
    r"\bthe\s+last\s+end\b",
)


def _token_key(token: object) -> str:
    return re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", str(token or "")).lower()


def clean_display_text(text: object) -> str:
    value = " ".join(str(text or "").split()).strip()
    if not value:
        return ""
    value = re.sub(r"\s+([:;,.!?])", r"\1", value)
    value = re.sub(r"([:;,.!?])([A-Za-z])", r"\1 \2", value)
    tokens = value.split()
    cleaned: list[str] = []
    for token in tokens:
        bare = _token_key(token)
        prev = _token_key(cleaned[-1]) if cleaned else ""
        if bare and bare == prev:
            continue
        cleaned.append(token)
    value = " ".join(cleaned).strip(" -:")
    value = value.replace(" :", ":").replace(" ,", ",")
    value = re.sub(r"\bA ([AEIOUaeiou])", r"An \1", value)
    value = re.sub(r"\b([aA])n ([^AEIOUaeiou\W])", lambda m: f"{m.group(1)} {m.group(2)}", value)
    return value.strip()


def strip_leading_article(text: object) -> str:
    value = clean_display_text(text)
    tokens = value.split()
    if len(tokens) >= 2 and _token_key(tokens[0]) in _ARTICLES:
        return " ".join(tokens[1:]).strip()
    return value


def _collapse_article_artifacts(value: str) -> str:
    tokens = value.split()
    if not tokens:
        return ""

    cleaned: list[str] = []
    seen_leading_article = _token_key(tokens[0]) in _ARTICLES

    for idx, token in enumerate(tokens):
        key = _token_key(token)
        next_token = tokens[idx + 1] if idx + 1 < len(tokens) else ""
        prev_key = _token_key(cleaned[-1]) if cleaned else ""

        if key in _ARTICLES and prev_key in _ARTICLES:
            cleaned[-1] = token
            continue

        if (
            key in _ARTICLES
            and seen_leading_article
            and cleaned
            and prev_key not in _ARTICLE_ALLOW_PREV
            and next_token
            and str(next_token or "")[:1].isupper()
        ):
            continue

        cleaned.append(token)

    return " ".join(cleaned).strip()


def sanitize_title(text: object) -> str:
    value = clean_display_text(text)
    value = _collapse_article_artifacts(value)
    return clean_display_text(value)


def sanitize_tagline(text: object, *, title: object | None = None) -> str:
    value = clean_display_text(text)
    value = _collapse_article_artifacts(value)
    if title and clean_display_text(title).casefold() == value.casefold():
        return ""
    return clean_display_text(value)


def sanitize_character_name(text: object) -> str:
    value = clean_display_text(text)
    value = _collapse_article_artifacts(value)
    return clean_display_text(value)


def sanitize_alternate_title(text: object) -> str:
    value = clean_display_text(text)
    value = _collapse_article_artifacts(value)
    return clean_display_text(value)


def looks_like_title_phrase(text: object) -> bool:
    value = clean_display_text(text)
    if not value or re.search(r"[.!?;:]", value):
        return False
    tokens = re.findall(r"[A-Za-z0-9']+", value)
    if not (1 <= len(tokens) <= 5):
        return False
    major = [tok for tok in tokens if tok.lower() not in _SHORT_TITLE_STOPWORDS]
    if not major:
        return False
    return all(tok[:1].isupper() or tok.isupper() for tok in major)


def contains_placeholder_syntax(text: object) -> bool:
    raw = str(text or "")
    if not raw:
        return False
    return bool(re.search(r"\[[^\[\]]+\]|\{[^{}]+\}", raw))


def looks_like_weak_tagline(text: object, *, title: object | None = None) -> bool:
    if contains_placeholder_syntax(text):
        return True
    value = sanitize_tagline(text, title=title)
    if not value:
        return True
    low = value.lower()
    for pattern in _BAD_TAGLINE_PATTERNS:
        if re.search(pattern, value) or re.search(pattern, low):
            return True
    tokens = re.findall(r"[A-Za-z0-9']+", value)
    if len(tokens) < 3 and not re.search(r"[.!?]", value):
        return True
    if looks_like_title_phrase(value):
        return True
    return False


def looks_like_weak_title(text: object) -> bool:
    if contains_placeholder_syntax(text):
        return True
    value = sanitize_title(text)
    if not value:
        return True
    low = value.lower()
    if re.search(r"\babstract nouns\b|\baction words\b|\bmythic words\b|\btechnology words\b|\bcelestial words\b", low):
        return True
    if re.search(r"\b([a-z][a-z' -]+)\s+(?:and|or|vs\.?|versus)\s+\1\b", low):
        return True
    if re.search(r"\b([a-z][a-z' -]+)\s+on\s+the\s+\1\b", low):
        return True
    return False


def normalized_word_tokens(text: object) -> list[str]:
    value = sanitize_tagline(text)
    if not value:
        return []
    return [token.lower() for token in re.findall(r"[A-Za-z0-9']+", value)]


def tagline_signature(text: object) -> str:
    return " ".join(normalized_word_tokens(text))


def tagline_similarity(a: object, b: object) -> float:
    sig_a = tagline_signature(a)
    sig_b = tagline_signature(b)
    if not sig_a or not sig_b:
        return 0.0
    if sig_a == sig_b:
        return 1.0
    tokens_a = sig_a.split()
    tokens_b = sig_b.split()
    if tokens_a[:4] and tokens_a[:4] == tokens_b[:4]:
        return 0.96
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    overlap = len(set_a & set_b)
    union = len(set_a | set_b)
    jaccard = float(overlap / union) if union else 0.0
    ratio = SequenceMatcher(None, sig_a, sig_b).ratio()
    return float(max(jaccard, ratio))


def tagline_is_near_duplicate(candidate: object, existing: Iterable[object], *, threshold: float = 0.9) -> bool:
    for prior in existing:
        if tagline_similarity(candidate, prior) >= float(threshold):
            return True
    return False


def unique_preserving_order(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = clean_display_text(item).casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(clean_display_text(item))
    return out
