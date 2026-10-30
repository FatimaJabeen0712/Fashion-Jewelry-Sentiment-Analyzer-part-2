"""Product mention extraction and ranking.

Rankings are about *discussion* and *response*, never sales. Output labels are
"Most discussed" and "Highest positive response"; verified sales data would be
required before any sales claim is made.
"""
from __future__ import annotations

import re
from math import sqrt

import pandas as pd

from config import MIN_PRODUCT_MENTIONS
from src.aspect_analysis import split_clauses
from src.sentiment import SentimentAnalyzer

LABEL_MOST_DISCUSSED = "Most discussed"
LABEL_HIGHEST_POSITIVE = "Highest positive response"
DISCLAIMER = ("Mentions and sentiment are proxies for customer interest, not sales figures. "
              "Import verified sales data before making any sales claim.")

PRODUCT_TERMS: dict[str, dict[str, list[str]]] = {
    "jewelry": {
        "ring": [r"rings?"],
        "necklace": [r"necklaces?", r"chains?", r"lockets?"],
        "pendant": [r"pendants?"],
        "choker": [r"chokers?"],
        "bracelet": [r"bracelets?", r"cuffs?"],
        "bangle": [r"bangles?"],
        "earrings": [r"earrings?", r"studs?", r"hoops?", r"jhumk\w*"],
        "anklet": [r"anklets?", r"payal"],
        "nose pin": [r"nose ?pins?", r"nose rings?"],
        "tikka": [r"tikka", r"maang ?tikka"],
        "jewelry set": [r"(?:jewel(?:le)?ry|bridal|necklace|earring) sets?"],
    },
    "clothing": {
        "dress": [r"dress(?:es)?", r"gowns?"],
        "shirt": [r"shirts?", r"t-?shirts?", r"tees?", r"blouses?"],
        "kurta": [r"kurtas?", r"kurtis?", r"kameez", r"shalwar"],
        "abaya": [r"abayas?"],
        "hijab": [r"hijabs?"],
        "scarf": [r"scarf", r"scarves", r"dupattas?", r"shawls?"],
        "hoodie": [r"hoodies?", r"sweatshirts?"],
        "sweater": [r"sweaters?", r"cardigans?"],
        "jacket": [r"jackets?", r"coats?", r"blazers?"],
        "jeans": [r"jeans", r"denim"],
        "trousers": [r"trousers?", r"pants", r"palazzos?"],
        "skirt": [r"skirts?"],
        "suit set": [r"(?:2|3|two|three)[ -]?piece"],
    },
}

_COMPILED = {niche: {canon: re.compile(r"\b(?:" + "|".join(p) + r")\b", re.I)
                     for canon, p in terms.items()} for niche, terms in PRODUCT_TERMS.items()}


def _niches(niche: str) -> list[str]:
    n = niche.lower()
    return ["jewelry", "clothing"] if n == "both" else [n]


def _term_map(niche: str) -> dict[str, re.Pattern]:
    merged: dict[str, re.Pattern] = {}
    for n in _niches(niche):
        merged.update(_COMPILED[n])
    return merged


def extract_product_mentions(df: pd.DataFrame, niche: str = "both",
                             analyzer: SentimentAnalyzer | None = None,
                             text_col: str = "text") -> pd.DataFrame:
    """One row per (comment, product type) with clause-level sentiment.

    Scanning is text-based; if a ``product_name`` value is present its product
    type is also counted for that comment.
    """
    analyzer = analyzer or SentimentAnalyzer()
    terms = _term_map(niche)
    cols = ["comment_id", "product", "sentiment", "sentiment_score", "like_count", "source"]
    records = []
    for _, row in df.iterrows():
        text = row[text_col] if isinstance(row[text_col], str) else ""
        found: dict[str, list[float]] = {}
        for clause in split_clauses(text):
            hits = [p for p, rx in terms.items() if rx.search(clause)]
            if hits:
                score = analyzer.score(clause)
                for p in hits:
                    found.setdefault(p, []).append(score)
        pname = str(row.get("product_name", "") or "")
        if pname:
            for p, rx in terms.items():
                if rx.search(pname) and p not in found:
                    found[p] = [float(row.get("sentiment_score", 0.0) or 0.0)]
        for p, scores in found.items():
            mean = sum(scores) / len(scores)
            records.append({"comment_id": row["comment_id"], "product": p,
                            "sentiment": analyzer.label(mean), "sentiment_score": round(mean, 4),
                            "like_count": int(row.get("like_count", 0) or 0),
                            "source": row.get("source", "")})
    return pd.DataFrame(records, columns=cols)


def wilson_lower_bound(positive: int, n: int, z: float = 1.96) -> float:
    """Lower bound of the Wilson interval for the positive share (small-sample safe)."""
    if n == 0:
        return 0.0
    p = positive / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return (centre - margin) / denom


def rank_products(mentions: pd.DataFrame, min_mentions: int = MIN_PRODUCT_MENTIONS) -> pd.DataFrame:
    """Rank products by discussion volume and by positive response.

    ``positive_response_rank`` uses the Wilson lower bound of the positive share and
    is only given to products with at least ``min_mentions`` mentions.
    """
    cols = ["product", "mentions", "share_of_mentions", "positive", "neutral", "negative",
            "pos_pct", "neg_pct", "avg_score", "total_likes", "most_discussed_rank",
            "positive_response_score", "positive_response_rank", "labels", "note"]
    if mentions.empty:
        return pd.DataFrame(columns=cols)
    g = mentions.groupby("product")
    r = pd.DataFrame({
        "mentions": g.size(),
        "positive": g["sentiment"].apply(lambda s: int((s == "positive").sum())),
        "neutral": g["sentiment"].apply(lambda s: int((s == "neutral").sum())),
        "negative": g["sentiment"].apply(lambda s: int((s == "negative").sum())),
        "avg_score": g["sentiment_score"].mean().round(3),
        "total_likes": g["like_count"].sum(),
    }).reset_index()
    r["share_of_mentions"] = (r["mentions"] / r["mentions"].sum()).round(3)
    r["pos_pct"] = (100 * r["positive"] / r["mentions"]).round(1)
    r["neg_pct"] = (100 * r["negative"] / r["mentions"]).round(1)
    r = r.sort_values(["mentions", "product"], ascending=[False, True]).reset_index(drop=True)
    r["most_discussed_rank"] = range(1, len(r) + 1)
    r["positive_response_score"] = [round(wilson_lower_bound(p, n), 3)
                                    for p, n in zip(r["positive"], r["mentions"])]
    eligible = r["mentions"] >= min_mentions
    ranks = r.loc[eligible, "positive_response_score"].rank(ascending=False, method="min")
    r["positive_response_rank"] = pd.Series(ranks, dtype="Int64").reindex(r.index)

    labels = [""] * len(r)
    if len(r):
        labels[0] = LABEL_MOST_DISCUSSED
    if eligible.any():
        best = r.loc[eligible, "positive_response_score"].idxmax()
        labels[best] = (labels[best] + "; " if labels[best] else "") + LABEL_HIGHEST_POSITIVE
    r["labels"] = labels
    r["note"] = DISCLAIMER
    return r[cols]
