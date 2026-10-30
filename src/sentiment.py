"""Explainable sentiment: VADER plus a small fashion/jewelry lexicon.

VADER is free, CPU-only and handles negation ("not good"), intensifiers and
common emojis. The domain lexicon adds words VADER scores as neutral
("tarnished", "flimsy", "overpriced"). Known limits: sarcasm, slang, mixed
opinions and non-English / Roman Urdu text are frequently misread.
"""
from __future__ import annotations

import re
from math import sqrt

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import SENTIMENT_MODEL_VERSION

DOMAIN_LEXICON: dict[str, float] = {
    # negative
    "tarnish": -2.5, "tarnished": -2.5, "tarnishes": -2.5, "tarnishing": -2.5,
    "rust": -2.0, "rusty": -2.5, "rusted": -2.5, "fake": -2.5, "flimsy": -2.2,
    "overpriced": -2.5, "pricey": -1.2, "faded": -1.8, "fades": -1.8, "fading": -1.8,
    "itchy": -2.0, "allergic": -2.0, "rash": -2.0, "scratchy": -2.0, "pilling": -2.0,
    "peeling": -2.0, "peeled": -2.0, "discolored": -2.2, "discoloured": -2.2,
    "shrunk": -2.0, "shrinks": -2.0, "delayed": -1.5, "crushed": -1.5, "defective": -2.5,
    "unraveled": -2.0, "thin": -0.8, "dull": -1.2, "snapped": -2.0, "broken": -2.2,
    # positive
    "stunning": 2.8, "elegant": 2.3, "dainty": 1.5, "chic": 2.0, "obsessed": 2.5,
    "sturdy": 1.8, "durable": 2.0, "gorgeous": 3.0, "flawless": 2.8, "slay": 2.0,
    "exquisite": 2.8, "shiny": 1.5, "premium": 1.5, "classy": 2.0,
}

# Multi-word phrases scored as a unit. VADER only applies its own idiom table next to a
# lexicon word, so phrases like "never again" are handled here instead.
PHRASE_SCORES: dict[str, float] = {
    "waste of money": -2.5, "not worth": -2.0, "rip off": -2.5, "ripoff": -2.5,
    "never again": -2.0, "would not recommend": -2.2, "do not buy": -2.5, "don't buy": -2.5,
    "never buy": -2.2, "will never buy": -2.2, "fell apart": -2.5, "turned green": -2.5,
    "came apart": -2.2, "runs small": -1.2, "too small": -1.2, "too big": -1.0,
    "nobody replied": -2.0, "no response": -2.0, "never replied": -2.2, "did not reply": -1.8,
    "didn't reply": -1.8, "took forever": -2.0, "took over": -1.0, "took too long": -2.0,
    "very late": -1.5, "still waiting": -1.8, "never arrived": -2.5, "never received": -2.5,
    "highly recommend": 2.5, "worth every penny": 2.8, "will buy again": 2.2,
}
_PHRASE_RES = [(re.compile(r"(?<!\w)" + re.escape(p) + r"(?!\w)"), v) for p, v in PHRASE_SCORES.items()]


class SentimentAnalyzer:
    """Positive / neutral / negative labels from VADER's compound score.

    Thresholds follow VADER's recommended +/-0.05 neutral band.
    """

    version = SENTIMENT_MODEL_VERSION

    def __init__(self, positive_threshold: float = 0.05, negative_threshold: float = -0.05):
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self._vader = SentimentIntensityAnalyzer()
        self._vader.lexicon.update(DOMAIN_LEXICON)

    def score(self, text: str) -> float:
        if not isinstance(text, str) or not text.strip():
            return 0.0
        compound = float(self._vader.polarity_scores(text)["compound"])
        lowered = text.lower().replace("\u2019", "'")
        raw = sum(v for rx, v in _PHRASE_RES if rx.search(lowered))
        if raw:
            phrase = raw / sqrt(raw * raw + 15)
            if compound == 0 or compound * phrase < 0:
                compound = max(-1.0, min(1.0, compound + phrase))
            else:  # same direction: keep the stronger signal rather than double counting
                compound = max(compound, phrase, key=abs)
        return round(compound, 4)

    def label(self, score: float) -> str:
        if score >= self.positive_threshold:
            return "positive"
        if score <= self.negative_threshold:
            return "negative"
        return "neutral"

    def analyze_text(self, text: str) -> tuple[str, float]:
        score = self.score(text)
        return self.label(score), score

    def analyze_frame(self, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        """Return a copy with ``sentiment`` and ``sentiment_score`` filled in."""
        out = df.copy()
        scores = out[text_col].map(self.score)
        out["sentiment_score"] = scores.round(4)
        out["sentiment"] = scores.map(self.label)
        return out


def sentiment_distribution(df: pd.DataFrame) -> dict[str, float]:
    """Share of positive / neutral / negative rows (0-1). Empty input -> zeros."""
    if df.empty or "sentiment" not in df.columns:
        return {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
    counts = df["sentiment"].value_counts(normalize=True)
    return {k: float(counts.get(k, 0.0)) for k in ("positive", "neutral", "negative")}


def net_sentiment_snapshot(df: pd.DataFrame, target: int = 2000) -> dict:
    """Transparent 'feedback snapshot' (not an absolute verdict on the brand).

    Net sentiment = (positive - negative) / total, range -1..+1. A 95% normal-
    approximation margin of error is reported so small samples are visible.
    """
    n = len(df)
    if n == 0:
        return {"n": 0, "net_sentiment": None, "margin_of_error": None,
                "formula": "(positive - negative) / total comments", "sample_size_warning": True}
    d = sentiment_distribution(df)
    net = d["positive"] - d["negative"]
    var = (d["positive"] + d["negative"]) - net ** 2
    moe = 1.96 * (max(var, 0.0) / n) ** 0.5
    return {"n": n, "net_sentiment": round(net, 4), "margin_of_error": round(moe, 4),
            "formula": "(positive - negative) / total comments; +/- 95% margin of error",
            "sample_size_warning": n < target}
