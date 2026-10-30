"""Sentiment evaluation against a *manually labeled* sample.

No scores are shipped with the project: run this on your own labeled subset
(recommended: at least 300 comments, ~100 per class where possible, labeled by
two people if you can) and report the output together with its limitations.
"""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src.collectors.csv_import import read_csv_raw
from src.schemas import MissingColumnError, PipelineError
from src.sentiment import SentimentAnalyzer

LABELS = ["positive", "neutral", "negative"]
TEMPLATE_CSV = ("text,label\n"
                "Love this ring so much,positive\n"
                "The necklace tarnished after a week,negative\n"
                "Where can I buy this?,neutral\n")
_ALIASES = {"pos": "positive", "neg": "negative", "neu": "neutral", "+": "positive", "-": "negative",
            "0": "neutral", "1": "positive", "-1": "negative"}


def normalize_label(value: object) -> str | None:
    v = str(value).strip().lower()
    v = _ALIASES.get(v, v)
    return v if v in LABELS else None


def evaluate_sentiment(labeled: pd.DataFrame | bytes, analyzer: SentimentAnalyzer | None = None) -> dict:
    """Compare model labels with human labels. Input needs ``text`` and ``label`` columns."""
    df = read_csv_raw(labeled) if isinstance(labeled, (bytes, bytearray)) else labeled.copy()
    missing = [c for c in ("text", "label") if c not in df.columns]
    if missing:
        raise MissingColumnError(f"Labeled file needs columns 'text' and 'label' (missing: {', '.join(missing)}).")
    df["true"] = df["label"].map(normalize_label)
    bad = int(df["true"].isna().sum())
    df = df[df["true"].notna() & (df["text"].astype(str).str.strip() != "")]
    if df.empty:
        raise PipelineError("No usable rows: labels must be positive / neutral / negative.")
    analyzer = analyzer or SentimentAnalyzer()
    df["pred"] = df["text"].astype(str).map(lambda t: analyzer.analyze_text(t)[0])

    p, r, f1, support = precision_recall_fscore_support(
        df["true"], df["pred"], labels=LABELS, zero_division=0)
    per_class = pd.DataFrame({"label": LABELS, "precision": p.round(3), "recall": r.round(3),
                              "f1": f1.round(3), "support": support})
    cm = pd.DataFrame(confusion_matrix(df["true"], df["pred"], labels=LABELS),
                      index=[f"true_{l}" for l in LABELS], columns=[f"pred_{l}" for l in LABELS])
    n = len(df)
    warnings = []
    if n < 300:
        warnings.append(f"Only {n} labeled rows - scores will be unstable (aim for 300+).")
    thin = [l for l, s in zip(LABELS, support) if s < 30]
    if thin:
        warnings.append(f"Class imbalance / few examples for: {', '.join(thin)}. "
                        "Prefer macro-F1 over accuracy and interpret per-class rows.")
    if bad:
        warnings.append(f"{bad} row(s) skipped because the label was not positive/neutral/negative.")
    warnings.append("Model: VADER + domain lexicon; sarcasm, slang, mixed opinions and "
                    "non-English text are known weak spots.")
    return {"n": n, "accuracy": round(float(accuracy_score(df["true"], df["pred"])), 3),
            "macro_precision": round(float(p.mean()), 3), "macro_recall": round(float(r.mean()), 3),
            "macro_f1": round(float(f1.mean()), 3), "per_class": per_class,
            "confusion_matrix": cm, "warnings": warnings, "predictions": df[["text", "true", "pred"]]}
