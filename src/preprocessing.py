"""Cleaning, spam filtering, de-duplication and language filtering.

Order of stages (and therefore of the funnel counts):
    1. text cleaning (HTML entities / markup / whitespace)
    2. empty rows removed
    3. likely spam removed (conservative heuristics, reasons recorded)
    4. duplicates removed (first occurrence kept)
    5. optional language filter

Definition of the target: the number of *valid unique comments* remaining
after all five stages. Demo rows are never counted.

De-duplication method: rows are duplicates if they share a ``comment_id`` or the
same *dedupe key*: lower-cased text with ASCII punctuation removed, runs of a
repeated character collapsed to two, and whitespace collapsed. Emojis are kept
in the key so "love it \U0001F60D" and "love it \U0001F62D" are different.
Meaningful negations ("not good") and short comments ("love it") are never
removed by cleaning.
"""
from __future__ import annotations

import html
import logging
import re
import unicodedata
from dataclasses import dataclass

import pandas as pd

from config import DEFAULT_TARGET
from src.schemas import DEMO_METHOD, PreprocessReport, ensure_schema

logger = logging.getLogger(__name__)

_TAG_RE = re.compile(r"</?[a-zA-Z][^<>]*>")
_BR_RE = re.compile(r"<br\s*/?>", re.I)
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ASCII_PUNCT_RE = re.compile(r"[!-/:-@\[-`{-~]")
_REPEAT_RE = re.compile(r"(.)\1{2,}")
_WS_RE = re.compile(r"\s+")

_PROMO_RE = re.compile(
    r"\b(sub4sub|subscribe to my|check out my (?:channel|page|profile|video)|free followers|"
    r"earn \$?\d+|make money online|click (?:the )?link in (?:my )?(?:bio|description)|"
    r"whatsapp\s*\+?\d{6,})", re.I)
_CHAR_RUN_RE = re.compile(r"([A-Za-z0-9])\1{14,}")
_WORD_REPEAT_RE = re.compile(r"\b(\w+)(?:\s+\1\b){5,}", re.I)

_EN_STOPWORDS = frozenset(
    "the a an and or but is are was were be been it this that these those i you he she we they "
    "my your our their me to of in on for with at by from as not no so very too just have has "
    "had do does did can could would should will what which who how where when why if than then "
    "there here its im ive dont cant wont".split())
_EN_HINTS = frozenset(
    "love loved lovely like nice good great best bad worst beautiful pretty gorgeous stunning "
    "cute amazing awesome perfect ordered order bought buy price size fit quality ring necklace "
    "bracelet earrings dress shirt color delivery shipping thanks thank please wish need want "
    "look looks looking wear wore wearing gift jewelry jewellery cheap expensive".split())


def is_emoji(ch: str) -> bool:
    o = ord(ch)
    return (0x1F000 <= o <= 0x1FAFF) or (0x2600 <= o <= 0x27BF) or (0x2B00 <= o <= 0x2BFF) \
        or o in (0x2764, 0x2B50, 0x203C, 0x2049)


def clean_text(value: object) -> str:
    """Normalize whitespace and entities and strip markup. Keeps emojis/negations."""
    if value is None or (isinstance(value, float) and value != value):
        return ""
    text = html.unescape(str(value))
    text = _BR_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    text = _CTRL_RE.sub("", text)
    text = unicodedata.normalize("NFC", text)
    return _WS_RE.sub(" ", text).strip()


def has_content(text: str) -> bool:
    return any(c.isalnum() or is_emoji(c) for c in text)


def spam_reason(text: str) -> str | None:
    """Return a short reason if the text is likely spam, else None (conservative)."""
    if _PROMO_RE.search(text):
        return "promo_phrase"
    urls = _URL_RE.findall(text)
    if len(urls) >= 2:
        return "multiple_urls"
    if len(urls) == 1 and len(_URL_RE.sub(" ", text).split()) <= 3:
        return "url_only"
    if _CHAR_RUN_RE.search(text):
        return "repeated_characters"
    if _WORD_REPEAT_RE.search(text):
        return "repeated_word"
    return None


def dedupe_key(text: str) -> str:
    t = _URL_RE.sub(" ", text.casefold())
    t = _ASCII_PUNCT_RE.sub("", t)
    t = _REPEAT_RE.sub(lambda m: m.group(1) * 2, t)
    return _WS_RE.sub(" ", t).strip() or text.casefold()


def detect_language(text: str) -> str:
    """Dependency-free, *approximate* language guess.

    Returns ``en``, ``latin-other`` (Latin script but few English cues, e.g.
    Roman Urdu / Spanish), a script label (``arabic-script``, ``cyrillic``,
    ``devanagari``, ``cjk``, ``other-script``) or ``unknown`` for text too short
    to judge. Not a substitute for a trained language identifier.
    """
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 3:
        return "unknown"
    scripts = {"latin": 0, "arabic-script": 0, "cyrillic": 0, "devanagari": 0, "cjk": 0, "other-script": 0}
    for c in letters:
        o = ord(c)
        if o < 0x250:
            scripts["latin"] += 1
        elif 0x600 <= o <= 0x6FF or 0x750 <= o <= 0x77F or 0xFB50 <= o <= 0xFEFF:
            scripts["arabic-script"] += 1
        elif 0x400 <= o <= 0x4FF:
            scripts["cyrillic"] += 1
        elif 0x900 <= o <= 0x97F:
            scripts["devanagari"] += 1
        elif 0x3040 <= o <= 0x9FFF or 0xAC00 <= o <= 0xD7AF:
            scripts["cjk"] += 1
        else:
            scripts["other-script"] += 1
    top, count = max(scripts.items(), key=lambda kv: kv[1])
    if top != "latin":
        return top if count / len(letters) > 0.6 else "unknown"
    tokens = re.findall(r"[a-z']+", text.lower())
    if len(tokens) < 3:
        return "unknown"
    cues = sum(1 for t in tokens if t in _EN_STOPWORDS or t in _EN_HINTS)
    return "en" if cues / len(tokens) >= 0.25 else "latin-other"


@dataclass
class PreprocessResult:
    clean: pd.DataFrame
    removed: pd.DataFrame  # includes a ``removal_reason`` column
    report: PreprocessReport


def preprocess(df: pd.DataFrame, target: int = DEFAULT_TARGET,
               language_filter: str = "any") -> PreprocessResult:
    """Run the cleaning pipeline. Never mutates ``df``.

    ``language_filter``: ``"any"`` keeps everything; ``"en"`` drops rows whose
    detected language is a definite non-English one (``unknown`` is kept).
    """
    if language_filter not in ("any", "en"):
        raise ValueError("language_filter must be 'any' or 'en'")

    work = ensure_schema(df)
    report = PreprocessReport(fetched=len(work), target=target, language_filter=language_filter)
    work["text"] = work["text"].map(clean_text)
    work["language"] = work["text"].map(detect_language)
    reason = pd.Series("", index=work.index, dtype=object)

    # 2. empty
    empty = ~work["text"].map(has_content)
    reason[empty] = "empty"
    report.empty_removed = int(empty.sum())

    # 3. spam
    spam_reasons = work["text"].map(spam_reason)
    spam = (reason == "") & spam_reasons.notna()
    reason[spam] = "spam:" + spam_reasons[spam].astype(str)
    work.loc[spam, "is_spam"] = True
    report.spam_removed = int(spam.sum())

    # 4. duplicates (comment_id first, then normalized text)
    live = reason == ""
    id_dup = live & work.duplicated("comment_id", keep="first")
    reason[id_dup] = "duplicate:id"
    live = reason == ""
    keys = work["text"].map(dedupe_key)
    key_dup = live & keys.where(live).duplicated(keep="first")
    reason[key_dup] = "duplicate:text"
    dup = id_dup | key_dup
    work.loc[dup, "is_duplicate"] = True
    report.duplicates_removed = int(dup.sum())

    # 5. language
    if language_filter == "en":
        live = reason == ""
        lang_drop = live & ~work["language"].isin(["en", "unknown"])
        reason[lang_drop] = "language:" + work["language"][lang_drop]
        report.language_removed = int(lang_drop.sum())

    keep = reason == ""
    clean = work[keep].reset_index(drop=True)
    removed = work[~keep].assign(removal_reason=reason[~keep]).reset_index(drop=True)
    report.valid = len(clean)
    report.demo_valid = int((clean["collection_method"] == DEMO_METHOD).sum())
    if report.demo_valid:
        report.notes.append("Synthetic demo rows are present and are not counted toward the target.")
    if language_filter == "en":
        report.notes.append("Language detection is approximate; Roman Urdu or mixed-language "
                            "comments may be misclassified.")
    logger.info("preprocess: fetched=%d empty=%d spam=%d dup=%d lang=%d valid=%d",
                report.fetched, report.empty_removed, report.spam_removed,
                report.duplicates_removed, report.language_removed, report.valid)
    return PreprocessResult(clean=clean, removed=removed, report=report)
