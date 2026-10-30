"""Secret scrubbing, safe file names and export allow-lists."""
from __future__ import annotations

import os
import re

import pandas as pd

from config import SECRET_NAMES, get_secret

# Columns that may appear in downloadable files. Anything else (for example an
# author name column from an uploaded CSV) is dropped from exports by default.
RAW_EXPORT_COLUMNS = ["comment_id", "text", "created_at", "source", "brand", "product_name",
                      "product_url", "post_url", "like_count", "reply_count", "rating",
                      "collection_method", "video_id", "video_title"]
ANALYZED_EXTRA_COLUMNS = ["language", "sentiment", "sentiment_score", "aspects"]

_KEY_LIKE_RE = re.compile(r"AIza[0-9A-Za-z_\-]{30,}")
_SECRET_ENV_NAMES = list(SECRET_NAMES.values())


def _secret_values() -> list[str]:
    """Configured secret values from the environment *or* Streamlit secrets (never logged)."""
    vals = []
    for name in _SECRET_ENV_NAMES:
        v = get_secret(name)
        if v and len(v) >= 8:
            vals.append(v)
    return vals


def scrub_secrets(text: str, secrets: list[str] | None = None) -> str:
    """Replace any configured secret value (and Google-API-key-shaped strings) with [redacted]."""
    out = str(text)
    for v in (_secret_values() if secrets is None else secrets):
        out = out.replace(v, "[redacted]")
    return _KEY_LIKE_RE.sub("[redacted]", out)


def _is_text_column(series: pd.Series) -> bool:
    dt = series.dtype
    return pd.api.types.is_object_dtype(dt) or pd.api.types.is_string_dtype(dt)


def scrub_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Redact secrets from every text cell (comment text is untrusted and could contain one)."""
    secrets = _secret_values()
    out = df.copy()
    for col in out.columns:
        if _is_text_column(out[col]):
            out[col] = out[col].map(lambda v: scrub_secrets(v, secrets) if isinstance(v, str) else v)
    return out


def contains_secret(payload: str | bytes) -> bool:
    text = payload.decode("utf-8", "replace") if isinstance(payload, (bytes, bytearray)) else str(payload)
    return scrub_secrets(text) != text


def safe_object_name(*parts: str, extension: str = "csv") -> str:
    """Build a storage-safe object / file name: lowercase [a-z0-9._-] only, no path separators."""
    cleaned = []
    for p in parts:
        s = re.sub(r"[^A-Za-z0-9._-]+", "-", str(p).strip()).strip("-.").lower()
        if s:
            cleaned.append(s[:60])
    stem = "_".join(cleaned) or "export"
    return f"{stem}.{extension.lstrip('.')}"


def spreadsheet_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Neutralize CSV/Excel formula injection: prefix cells starting with = + - @ with an apostrophe.

    Comment text is untrusted; a comment such as ``=HYPERLINK(...)`` must not run when the
    export is opened in Excel. Numeric columns are untouched.
    """
    out = df.copy()
    for col in out.columns:
        if _is_text_column(out[col]):
            out[col] = out[col].map(
                lambda v: "'" + v if isinstance(v, str) and v[:1] in ("=", "+", "@") or
                (isinstance(v, str) and v[:1] == "-" and not _looks_numeric(v)) else v)
    return out


def unguard_formula(value: object) -> object:
    """Reverse ``spreadsheet_safe`` on re-import: drop the apostrophe it added.

    Only a leading apostrophe directly followed by ``= + - @`` is removed, so ordinary
    text that happens to start with an apostrophe is left alone.
    """
    if isinstance(value, str) and value[:1] == "'" and value[1:2] in ("=", "+", "-", "@") and value[1:2]:
        return value[1:]
    return value


def _looks_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False
