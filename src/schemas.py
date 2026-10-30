"""Normalized internal schema, shared exceptions and schema enforcement."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field

import pandas as pd

NORMALIZED_COLUMNS: list[str] = [
    "comment_id",
    "text",
    "created_at",
    "source",
    "brand",
    "product_name",
    "product_url",
    "post_url",
    "like_count",
    "reply_count",
    "rating",
    "language",
    "sentiment",
    "sentiment_score",
    "aspects",
    "is_duplicate",
    "is_spam",
    "collection_method",
]

# Collection-method value used for synthetic demo rows. Demo rows are never
# counted towards the real-data target.
DEMO_METHOD = "demo_synthetic"

TEXT_COLUMNS = ["comment_id", "text", "source", "brand", "product_name", "product_url",
                "post_url", "language", "sentiment", "aspects", "collection_method"]


class PipelineError(Exception):
    """Base class for expected, user-actionable errors."""


class MissingColumnError(PipelineError):
    """A required column is absent or not mapped."""


class MalformedCSVError(PipelineError):
    """The uploaded file cannot be parsed as CSV."""


class ConnectorError(PipelineError):
    """A data connector failed (auth, quota, permission, network)."""


class ConnectorNotConfigured(ConnectorError):
    """A connector has no credentials / approval configured."""


def make_comment_id(source: str, text: str, index: int) -> str:
    digest = hashlib.sha1(f"{source}|{text}|{index}".encode("utf-8", "replace")).hexdigest()
    return f"c-{digest[:14]}"


def ensure_schema(df: pd.DataFrame, default_source: str = "unknown",
                  collection_method: str = "import") -> pd.DataFrame:
    """Return a copy with every normalized column present and typed.

    Unknown extra columns are preserved after the normalized ones. Missing values
    are handled safely (empty strings, NaT, NaN, zero counts).
    """
    out = df.copy().reset_index(drop=True)

    for col in TEXT_COLUMNS:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].where(out[col].notna(), "").astype(str)

    out["source"] = out["source"].where(out["source"].str.strip() != "", default_source)
    out["collection_method"] = out["collection_method"].where(
        out["collection_method"].str.strip() != "", collection_method)

    if "created_at" not in out.columns:
        out["created_at"] = pd.NaT
    out["created_at"] = pd.to_datetime(out["created_at"], errors="coerce", utc=True)

    for col in ("like_count", "reply_count"):
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).clip(lower=0).astype(int)

    if "rating" not in out.columns:
        out["rating"] = float("nan")
    out["rating"] = pd.to_numeric(out["rating"], errors="coerce")

    if "sentiment_score" not in out.columns:
        out["sentiment_score"] = float("nan")
    out["sentiment_score"] = pd.to_numeric(out["sentiment_score"], errors="coerce")

    for col in ("is_duplicate", "is_spam"):
        if col not in out.columns:
            out[col] = False
        out[col] = out[col].map(lambda v: str(v).strip().lower() in ("true", "1", "yes")
                                if not isinstance(v, bool) else v).astype(bool)

    missing_id = out["comment_id"].str.strip() == ""
    if missing_id.any():
        out.loc[missing_id, "comment_id"] = [
            make_comment_id(out.at[i, "source"], out.at[i, "text"], i)
            for i in out.index[missing_id]
        ]

    extras = [c for c in out.columns if c not in NORMALIZED_COLUMNS]
    return out[NORMALIZED_COLUMNS + extras]


@dataclass
class PreprocessReport:
    """Counts describing how many rows survived each cleaning stage.

    The target is measured on *valid unique comments after preprocessing*:
    non-empty, non-spam, de-duplicated and passing the chosen language filter.
    Synthetic demo rows are never counted towards the target.
    """

    fetched: int = 0
    empty_removed: int = 0
    spam_removed: int = 0
    duplicates_removed: int = 0
    language_removed: int = 0
    valid: int = 0
    demo_valid: int = 0
    target: int = 2000
    language_filter: str = "any"
    notes: list[str] = field(default_factory=list)

    @property
    def real_valid(self) -> int:
        return max(self.valid - self.demo_valid, 0)

    @property
    def target_met(self) -> bool:
        return self.real_valid >= self.target

    @property
    def shortfall(self) -> int:
        return max(self.target - self.real_valid, 0)

    def funnel(self) -> list[tuple[str, int]]:
        after_empty = self.fetched - self.empty_removed
        after_spam = after_empty - self.spam_removed
        after_dedup = after_spam - self.duplicates_removed
        return [
            ("Fetched / imported", self.fetched),
            ("After empty-row removal", after_empty),
            ("After spam removal", after_spam),
            ("After de-duplication", after_dedup),
            ("Valid after language filter", self.valid),
        ]

    def status_message(self) -> str:
        if self.valid == 0:
            return "No valid comments remain after preprocessing."
        if self.demo_valid and self.real_valid == 0:
            return (f"{self.valid:,} synthetic DEMO comments - not real data and not counted "
                    f"toward the {self.target:,} target.")
        pct = 100 * self.real_valid / self.target if self.target else 0
        if self.target_met:
            return (f"Target reached: {self.real_valid:,} valid unique comments "
                    f"(target {self.target:,}).")
        return (f"Below target: {self.real_valid:,} of {self.target:,} valid unique comments "
                f"({pct:.0f}%). Add sources or import more data, or analyze the smaller "
                f"sample with the sample-size warning.")

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(real_valid=self.real_valid, target_met=self.target_met,
                 shortfall=self.shortfall)
        return d
