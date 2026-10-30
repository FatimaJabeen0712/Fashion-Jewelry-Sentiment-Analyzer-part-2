"""Rule-based aspect extraction, aspect sentiment, complaints and feature requests.

Method (explainable baseline): a comment is split into clauses; for every aspect
whose keywords appear in a clause, that clause is scored with the sentiment
analyzer. "Beautiful ring but the clasp broke" therefore yields design/style
positive and durability negative rather than one blended score.
"""
from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd

from src.sentiment import SentimentAnalyzer

ASPECT_KEYWORDS: dict[str, list[str]] = {
    "price": [r"price[sd]?", r"pricing", r"priced", r"expensive", r"overpriced", r"pricey",
              r"cost\w*", r"afford\w*", r"cheap\w*", r"discount\w*", r"how much"],
    "value_for_money": [r"worth", r"value", r"money", r"bang for", r"rip ?off", r"waste of"],
    "design_style": [r"design\w*", r"styles?", r"stylish", r"the look", r"aesthetic",
                     r"pretty", r"beautiful", r"elegant", r"gorgeous", r"stunning", r"trendy",
                     r"chic", r"dainty", r"minimal\w*", r"collection", r"unique", r"lovely",
                     r"cute"],
    "material": [r"material\w*", r"gold", r"silver", r"plated", r"plating", r"stainless",
                 r"steel", r"brass", r"copper", r"pearls?", r"gemstones?", r"stones?",
                 r"diamonds?", r"zircon\w*", r"fabric\w*", r"cotton", r"linen", r"silk",
                 r"polyester", r"wool", r"leather", r"denim", r"lawn", r"chiffon", r"velvet",
                 r"viscose", r"metal"],
    "comfort": [r"comfort\w*", r"itch\w*", r"soft", r"scratchy", r"lightweight", r"heavy",
                r"breathable", r"skin", r"allerg\w*", r"hypoallergenic", r"irritat\w*", r"rash"],
    "fit_sizing": [r"sizes?", r"sized", r"sizing", r"fits?", r"fitted", r"fitting", r"tight",
                   r"loose", r"too big", r"too small", r"adjustable", r"length", r"shrunk",
                   r"shrink\w*", r"runs (?:small|large|big)"],
    "quality": [r"quality", r"well[- ]made", r"poorly made", r"cheaply made", r"flimsy",
                r"sturdy", r"finish\w*", r"stitch\w*", r"seams?", r"craftsmanship", r"defect\w*",
                r"damaged", r"thin"],
    "durability": [r"last\w*", r"durab\w*", r"tarnish\w*", r"rust\w*", r"fad(?:e|es|ed|ing)",
                   r"peel\w*", r"scratch\w*", r"broke", r"broken", r"snapped", r"clasps?",
                   r"discolou?r\w*", r"turned green", r"fell apart", r"came apart", r"pilling",
                   r"after (?:a|one|two|three|\d+) (?:day|days|week|weeks|month|months|wash|washes)"],
    "color": [r"colou?rs?", r"colou?red", r"shades?", r"hues?", r"bright", r"dull"],
    "packaging": [r"packag\w*", r"box(?:es)?", r"wrapp?(?:ed|ing)?", r"gift ?box", r"pouch",
                  r"unboxing", r"ribbon"],
    "delivery": [r"deliver\w*", r"shipping", r"shipped", r"ship", r"arriv\w*", r"courier",
                 r"tracking", r"late", r"delay\w*", r"dispatch\w*", r"parcel"],
    "customer_service": [r"customer service", r"support", r"respon(?:se|d|ded|ds)", r"repl(?:y|ied|ies)",
                         r"refund\w*", r"return(?:s|ed)?", r"exchange\w*", r"replacement",
                         r"seller", r"staff", r"rude", r"helpful"],
}

_ASPECT_RE = {a: re.compile(r"\b(?:" + "|".join(p) + r")\b", re.I) for a, p in ASPECT_KEYWORDS.items()}
_CLAUSE_SPLIT_RE = re.compile(
    r"[.!?;\n]+|\bbut\b|\bhowever\b|\balthough\b|\bthough\b|,\s*(?:and|but)\b", re.I)

ASPECT_LABELS = {a: a.replace("_", " ").title() for a in ASPECT_KEYWORDS}


def split_clauses(text: str) -> list[str]:
    """Split a comment into clauses so each opinion is scored separately."""
    parts = [p.strip(" ,-–—") for p in _CLAUSE_SPLIT_RE.split(text)]
    return [p for p in parts if p]


def aspects_in_text(text: str) -> list[str]:
    return [a for a, rx in _ASPECT_RE.items() if rx.search(text)]


def extract_aspects(df: pd.DataFrame, analyzer: SentimentAnalyzer | None = None,
                    text_col: str = "text") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add an ``aspects`` column and build a long-form aspect table.

    Returns ``(df_with_aspects, aspect_rows)``. ``aspect_rows`` has one row per
    (comment, aspect) with the supporting clause as ``evidence``.
    """
    analyzer = analyzer or SentimentAnalyzer()
    out = df.copy()
    records: list[dict] = []
    aspect_lists: list[str] = []
    meta_cols = [c for c in ("source", "product_name", "created_at", "like_count") if c in out.columns]

    for idx, row in out.iterrows():
        text = row[text_col] if isinstance(row[text_col], str) else ""
        scores: dict[str, list[tuple[float, str]]] = defaultdict(list)
        for clause in split_clauses(text):
            hits = aspects_in_text(clause)
            if not hits:
                continue
            score = analyzer.score(clause)
            for a in hits:
                scores[a].append((score, clause))
        aspect_lists.append("|".join(sorted(scores)))
        for aspect, items in scores.items():
            mean = sum(s for s, _ in items) / len(items)
            evidence = min(items, key=lambda it: it[0])[1] if mean < 0 else max(items, key=lambda it: it[0])[1]
            rec = {"comment_id": row["comment_id"], "aspect": aspect,
                   "aspect_sentiment": analyzer.label(mean), "aspect_score": round(mean, 4),
                   "evidence": evidence[:280]}
            rec.update({c: row[c] for c in meta_cols})
            records.append(rec)

    out["aspects"] = aspect_lists
    columns = ["comment_id", "aspect", "aspect_sentiment", "aspect_score", "evidence"] + meta_cols
    return out, pd.DataFrame(records, columns=columns)


def summarize_aspects(aspect_rows: pd.DataFrame) -> pd.DataFrame:
    """Per-aspect mention counts and sentiment split, most mentioned first."""
    cols = ["aspect", "mentions", "positive", "neutral", "negative", "pos_pct", "neg_pct",
            "net_sentiment", "avg_score"]
    if aspect_rows.empty:
        return pd.DataFrame(columns=cols)
    g = aspect_rows.groupby("aspect")
    summary = pd.DataFrame({
        "mentions": g.size(),
        "positive": g["aspect_sentiment"].apply(lambda s: int((s == "positive").sum())),
        "neutral": g["aspect_sentiment"].apply(lambda s: int((s == "neutral").sum())),
        "negative": g["aspect_sentiment"].apply(lambda s: int((s == "negative").sum())),
        "avg_score": g["aspect_score"].mean().round(3),
    }).reset_index()
    summary["pos_pct"] = (100 * summary["positive"] / summary["mentions"]).round(1)
    summary["neg_pct"] = (100 * summary["negative"] / summary["mentions"]).round(1)
    summary["net_sentiment"] = ((summary["positive"] - summary["negative"]) / summary["mentions"]).round(3)
    return summary.sort_values("mentions", ascending=False)[cols].reset_index(drop=True)


def _top_by_sentiment(aspect_rows: pd.DataFrame, sentiment: str, count_col: str,
                      n_examples: int) -> pd.DataFrame:
    cols = ["aspect", count_col, f"share_of_{sentiment}", "examples", "example_comment_ids"]
    sel = aspect_rows[aspect_rows["aspect_sentiment"] == sentiment] if not aspect_rows.empty else aspect_rows
    if sel.empty:
        return pd.DataFrame(columns=cols)
    total = len(sel)
    rows = []
    for aspect, grp in sel.groupby("aspect"):
        # most extreme examples first: most negative for complaints, most positive for likes
        grp = grp.sort_values("aspect_score", ascending=(sentiment == "negative")).drop_duplicates("evidence")
        top = grp.head(n_examples)
        n = int((sel["aspect"] == aspect).sum())
        rows.append({"aspect": aspect, count_col: n, f"share_of_{sentiment}": round(n / total, 3),
                     "examples": list(top["evidence"]), "example_comment_ids": list(top["comment_id"])})
    return pd.DataFrame(rows, columns=cols).sort_values(count_col, ascending=False).reset_index(drop=True)


def top_complaints(aspect_rows: pd.DataFrame, n_examples: int = 3) -> pd.DataFrame:
    """Rank aspects by negative mentions with representative (most negative) examples."""
    out = _top_by_sentiment(aspect_rows, "negative", "negative_mentions", n_examples)
    return out.rename(columns={"share_of_negative": "share_of_negative"})


def top_likes(aspect_rows: pd.DataFrame, n_examples: int = 3) -> pd.DataFrame:
    """Rank aspects by positive mentions with representative (most positive) examples."""
    return _top_by_sentiment(aspect_rows, "positive", "positive_mentions", n_examples)


# ----------------------------------------------------------------------------
# Feature requests / unmet needs
# ----------------------------------------------------------------------------
_REQUEST_PATTERNS = [re.compile(p, re.I) for p in (
    r"\bwish (?:it|they|you|this|there|i could)\b[^.!?\n]*",
    r"\bplease (?:make|add|bring|restock|launch|create|do|offer|have|sell|ship|include|consider)\b[^.!?\n]*",
    r"\bwould be (?:better|nicer|great|good|amazing|perfect|awesome) if\b[^.!?\n]*",
    r"\bshould (?:have|come|make|add|offer|include)\b[^.!?\n]*",
    r"\b(?:can|could) (?:you|u) (?:please )?(?:make|add|bring|offer|do|restock|launch|create|ship|include|sell)\b[^.!?\n]*",
    r"\bneeds? (?:more|to have|an?)\b[^.!?\n]*",
    r"\bhope (?:you|they) (?:will |can )?(?:make|add|bring|launch|restock|offer)\b[^.!?\n]*",
    r"\bwould love (?:to see|a|an|some|more)\b[^.!?\n]*",
    r"\bdo you (?:have|make|offer|sell)\b[^.!?\n]*",
)]

REQUEST_THEMES: dict[str, str] = {
    "waterproof_tarnish_free": r"waterproof|water ?proof|tarnish|sweat|anti-?tarnish|hypoallergenic|nickel|sensitive skin|allerg",
    "size_range": r"\bsiz(?:e|es|ing)\b|plus|petite|adjustable|length|\bxl\b|\bxs\b",
    "color_options": r"colou?rs?|shades?|silver|gold|rose gold|black|white|pastel",
    "personalization": r"personali[sz]\w*|custom\w*|engrav\w*|initials?|monogram|name",
    "sets_bundles": r"\bsets?\b|bundle\w*|matching|combo|pairs?",
    "restock": r"restock|back in stock|sold out|out of stock|available",
    "price_offers": r"discount|cheaper|affordable|\bsale\b|offers?|lower price",
    "new_categories": r"\bmen\b|\bmens\b|kids?|children|baby|bridal|wedding|menswear",
    "gift_packaging": r"gift|packag\w*|\bbox\b",
    "delivery_options": r"ship\w*|deliver\w*|international|cash on delivery|\bcod\b|express",
}
_THEME_RE = {k: re.compile(v, re.I) for k, v in REQUEST_THEMES.items()}


def find_feature_requests(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Detect requests / unmet needs ("wish it had...", "please make...") and tag a theme."""
    cols = ["comment_id", "request_text", "theme", "source", "product_name"]
    rows = []
    for _, row in df.iterrows():
        text = row[text_col] if isinstance(row[text_col], str) else ""
        for rx in _REQUEST_PATTERNS:
            m = rx.search(text)
            if m:
                span = m.group(0).strip()
                theme = next((k for k, trx in _THEME_RE.items() if trx.search(span)), None) \
                    or next((k for k, trx in _THEME_RE.items() if trx.search(text)), "other")
                rows.append({"comment_id": row["comment_id"], "request_text": span[:280],
                             "theme": theme, "source": row.get("source", ""),
                             "product_name": row.get("product_name", "")})
                break
    return pd.DataFrame(rows, columns=cols)


def group_similar_requests(requests: pd.DataFrame, threshold: float = 0.6,
                           max_rows: int = 3000) -> pd.DataFrame:
    """Group near-identical request phrasings with TF-IDF cosine similarity (greedy).

    Adds ``request_group`` (int) and ``group_size``. Capped at ``max_rows`` to keep
    the similarity matrix small on modest machines.
    """
    out = requests.copy()
    if out.empty:
        out["request_group"] = pd.Series(dtype=int)
        out["group_size"] = pd.Series(dtype=int)
        return out
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    out = out.head(max_rows).reset_index(drop=True)
    tfidf = TfidfVectorizer(ngram_range=(1, 2), lowercase=True).fit_transform(out["request_text"])
    sim = cosine_similarity(tfidf)
    group = [-1] * len(out)
    next_id = 0
    for i in range(len(out)):
        if group[i] != -1:
            continue
        group[i] = next_id
        for j in range(i + 1, len(out)):
            if group[j] == -1 and sim[i, j] >= threshold:
                group[j] = next_id
        next_id += 1
    out["request_group"] = group
    out["group_size"] = out.groupby("request_group")["request_group"].transform("size")
    return out


def summarize_requests(requests: pd.DataFrame) -> pd.DataFrame:
    """Theme counts with example request texts and their comment ids."""
    cols = ["theme", "requests", "examples", "example_comment_ids"]
    if requests.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for theme, grp in requests.groupby("theme"):
        uniq = grp.drop_duplicates("request_text")
        rows.append({"theme": theme, "requests": len(grp),
                     "examples": list(uniq["request_text"].head(3)),
                     "example_comment_ids": list(uniq["comment_id"].head(3))})
    return pd.DataFrame(rows, columns=cols).sort_values("requests", ascending=False).reset_index(drop=True)
