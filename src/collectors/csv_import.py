"""CSV import with auto-suggested column mapping, and reuse of exported CSVs."""
from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from src.schemas import MalformedCSVError, MissingColumnError, ensure_schema
from src.security import unguard_formula

MAX_BYTES = 50 * 1024 * 1024

# normalized field -> header names we recognise (compared case-insensitively)
COLUMN_HINTS: dict[str, list[str]] = {
    "text": ["text", "comment", "comment_text", "commenttext", "review", "review_text",
             "body", "content", "message", "feedback", "textdisplay"],
    "created_at": ["created_at", "date", "published_at", "publishedat", "timestamp", "time",
                   "created", "datetime"],
    "source": ["source", "platform", "channel"],
    "product_name": ["product_name", "product", "item", "item_name"],
    "product_url": ["product_url"],
    "post_url": ["post_url", "video_url", "url", "link"],
    "like_count": ["like_count", "likecount", "likes", "upvotes"],
    "reply_count": ["reply_count", "replycount", "replies", "totalreplycount"],
    "rating": ["rating", "stars", "star_rating"],
    "comment_id": ["comment_id", "commentid", "id"],
}

EXAMPLE_CSV = (
    "text,date,source,product,likes,rating\n"
    '"Love this ring, the design is so elegant!",2026-01-05,instagram_export,Luna Ring,12,5\n'
    '"The necklace tarnished after two weeks, not happy.",2026-01-07,website_reviews,Aria Necklace,0,2\n'
)


def _read_bytes(source: str | Path | bytes | BinaryIO) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    elif isinstance(source, (str, Path)):
        try:
            data = Path(source).read_bytes()
        except OSError as exc:
            raise MalformedCSVError(f"Could not read file: {exc.strerror or exc}") from exc
    else:
        data = source.read()
        if isinstance(data, str):
            data = data.encode("utf-8")
    if len(data) > MAX_BYTES:
        raise MalformedCSVError(
            f"File is larger than {MAX_BYTES // (1024 * 1024)} MB. Split it and import in parts.")
    if not data.strip():
        raise MalformedCSVError("The file is empty.")
    return data


def _detect_delimiter(text: str) -> str:
    """Pick the delimiter that appears most often in the header line (default comma).

    Safer than csv.Sniffer, which mis-detects single-column files.
    """
    header = text.lstrip("\ufeff").splitlines()[0] if text.strip() else ""
    counts = {d: header.count(d) for d in (",", ";", "\t", "|")}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def read_csv_raw(source: str | Path | bytes | BinaryIO) -> pd.DataFrame:
    """Parse a CSV as text (all values kept as strings). Raises MalformedCSVError."""
    data = _read_bytes(source)
    text = None
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):  # UTF-16 only when a BOM says so
        encodings.insert(0, "utf-16")
    for encoding in encodings:
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:  # pragma: no cover - latin-1 never fails
        raise MalformedCSVError("Could not decode the file. Save it as UTF-8 CSV.")
    try:
        df = pd.read_csv(io.StringIO(text), sep=_detect_delimiter(text), engine="python",
                         dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError as exc:
        raise MalformedCSVError("The file has no columns.") from exc
    except (pd.errors.ParserError, csv.Error) as exc:
        raise MalformedCSVError(f"The file is not valid CSV: {exc}") from exc
    if df.shape[1] == 0:
        raise MalformedCSVError("The file has no columns.")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def suggest_mapping(columns: list[str]) -> dict[str, str | None]:
    """Guess which uploaded column feeds each normalized field."""
    lowered = {c.strip().lower(): c for c in columns}
    mapping: dict[str, str | None] = {}
    used: set[str] = set()
    for field_name, hints in COLUMN_HINTS.items():
        found = None
        for hint in hints:
            col = lowered.get(hint)
            if col is not None and col not in used:
                found = col
                break
        mapping[field_name] = found
        if found:
            used.add(found)
    return mapping


def import_csv(source: str | Path | bytes | BinaryIO | pd.DataFrame,
               mapping: dict[str, str | None] | None = None, brand: str = "",
               default_source: str = "csv_upload") -> pd.DataFrame:
    """Import a CSV (or DataFrame) into the normalized schema.

    ``mapping`` maps normalized field -> uploaded column name. If omitted it is
    auto-suggested. The ``text`` field is mandatory.
    """
    raw = source if isinstance(source, pd.DataFrame) else read_csv_raw(source)
    if raw.empty:
        raise MalformedCSVError("The file has headers but no rows.")
    mapping = dict(mapping) if mapping else suggest_mapping(list(raw.columns))

    text_col = mapping.get("text")
    if not text_col:
        raise MissingColumnError(
            "No comment-text column is mapped. Choose the column containing the comment "
            f"text. Available columns: {', '.join(map(str, raw.columns))}")
    for field_name, col in mapping.items():
        if col and col not in raw.columns:
            raise MissingColumnError(
                f"Mapped column '{col}' (for {field_name}) is not in the file. "
                f"Available columns: {', '.join(map(str, raw.columns))}")

    out = pd.DataFrame(index=raw.index)
    for field_name, col in mapping.items():
        if col:
            out[field_name] = raw[col]
    if brand:
        out["brand"] = brand
    return ensure_schema(out, default_source=default_source, collection_method="csv_import")


def load_exported_csv(source: str | Path | bytes | BinaryIO) -> tuple[pd.DataFrame, dict]:
    """Load a CSV previously exported by this app so analysis can be re-run.

    Returns (normalized DataFrame, metadata). Raises MissingColumnError if the
    file has no ``text`` column.
    """
    raw = read_csv_raw(source)
    if "text" not in raw.columns:
        raise MissingColumnError(
            "This does not look like an export from this app: no 'text' column. "
            "Use the CSV import form with column mapping instead.")
    for col in raw.columns:  # undo the apostrophe added to formula-like cells on export
        raw[col] = raw[col].map(unguard_formula)
    df = ensure_schema(raw, default_source="csv_upload", collection_method="csv_reload")
    meta = {
        "rows": len(df),
        "columns": list(raw.columns),
        "has_sentiment": bool((df["sentiment"] != "").any()),
        "sources": sorted(df["source"].unique().tolist()),
        "date_min": None if df["created_at"].isna().all() else str(df["created_at"].min().date()),
        "date_max": None if df["created_at"].isna().all() else str(df["created_at"].max().date()),
        "contains_demo": bool((df["collection_method"] == "demo_synthetic").any()),
    }
    return df, meta
