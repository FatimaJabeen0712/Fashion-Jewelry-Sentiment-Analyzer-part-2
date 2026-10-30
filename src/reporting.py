"""CSV / JSON export builders. Exports never contain secrets or commenter identities.

All CSVs are UTF-8 with a BOM (``utf-8-sig``) so Excel opens emojis and non-Latin
text correctly; the csv module quotes commas, quotes and newlines.
"""
from __future__ import annotations

import io
import json

import pandas as pd

from config import APP_NAME, APP_VERSION
from src.pipeline import AnalysisResult
from src.schemas import DEMO_METHOD
from src.security import (ANALYZED_EXTRA_COLUMNS, RAW_EXPORT_COLUMNS, scrub_frame, scrub_secrets,
                          spreadsheet_safe)

LIST_JOIN = " | "


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize to UTF-8-BOM CSV with secret redaction and formula-injection protection."""
    buf = io.StringIO()
    spreadsheet_safe(scrub_frame(df)).to_csv(buf, index=False, lineterminator="\r\n")
    return buf.getvalue().encode("utf-8-sig")


def _allowed(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[[c for c in columns if c in df.columns]].copy()


def _stringify_lists(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].map(lambda v: isinstance(v, list)).any():
            out[col] = out[col].map(lambda v: LIST_JOIN.join(map(str, v)) if isinstance(v, list) else v)
    return out


def _meta(res: AnalysisResult) -> dict:
    return {"brand": res.brand, "niche": res.niche, "analyzed_at_utc": res.analyzed_at,
            "app": f"{APP_NAME} {APP_VERSION}", "model_version": res.model_version}


def raw_export(res: AnalysisResult) -> pd.DataFrame:
    """Permitted raw fields as imported/collected (before cleaning)."""
    return _allowed(res.raw, RAW_EXPORT_COLUMNS)


def cleaned_export(res: AnalysisResult) -> pd.DataFrame:
    return _allowed(res.clean, RAW_EXPORT_COLUMNS + ["language"])


def analyzed_export(res: AnalysisResult, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Analyzed comments with sentiment and aspect labels (optionally a filtered subset)."""
    base = res.analyzed if df is None else df
    return _allowed(base, RAW_EXPORT_COLUMNS + ANALYZED_EXTRA_COLUMNS)


def summary_export(res: AnalysisResult) -> pd.DataFrame:
    """Key/value summary: provenance, preprocessing counts, sentiment snapshot."""
    r, snap = res.report, res.snapshot
    rng = res.date_range
    rows = [
        ("collection_analysis_timestamp_utc", res.analyzed_at),
        ("brand", res.brand), ("niche", res.niche),
        ("sources", "; ".join(f"{k}={v}" for k, v in res.sources.items())),
        ("date_range", f"{rng[0]} to {rng[1]}" if rng else ""),
        ("fetched_or_imported", r.fetched), ("empty_removed", r.empty_removed),
        ("spam_removed", r.spam_removed), ("duplicates_removed", r.duplicates_removed),
        ("language_removed", r.language_removed), ("language_filter", r.language_filter),
        ("valid_unique_comments", r.valid), ("synthetic_demo_comments", r.demo_valid),
        ("real_valid_unique_comments", r.real_valid), ("target", r.target),
        ("target_met", r.target_met), ("sample_size_warning", res.small_sample),
        ("net_sentiment", snap.get("net_sentiment")),
        ("net_sentiment_margin_of_error_95", snap.get("margin_of_error")),
        ("net_sentiment_formula", snap.get("formula")),
        ("model_version", res.model_version),
        ("contains_synthetic_demo_data", r.demo_valid > 0),
    ]
    for label in ("positive", "neutral", "negative"):
        rows.append((f"{label}_comments", int((res.analyzed["sentiment"] == label).sum())))
    return pd.DataFrame(rows, columns=["metric", "value"])


def aspect_export(res: AnalysisResult) -> pd.DataFrame:
    return res.aspect_summary.assign(**{k: v for k, v in _meta(res).items()
                                         if k in ("brand", "analyzed_at_utc", "model_version")})


def recommendations_export(res: AnalysisResult) -> pd.DataFrame:
    rec = _stringify_lists(res.recommendations).assign(kind="improvement")
    opp = _stringify_lists(res.opportunities)
    if not opp.empty:
        opp = opp.rename(columns={"concept": "recommendation", "validation_test": "action"}) \
                 .assign(kind="future_opportunity_hypothesis")
    out = pd.concat([rec, opp], ignore_index=True) if not opp.empty else rec
    return out.assign(brand=res.brand, analyzed_at_utc=res.analyzed_at, model_version=res.model_version)


def products_export(res: AnalysisResult) -> pd.DataFrame:
    return res.product_ranking


def json_report(res: AnalysisResult) -> bytes:
    payload = {
        "meta": _meta(res), "preprocessing": res.report.to_dict(), "snapshot": res.snapshot,
        "aspects": res.aspect_summary.to_dict("records"),
        "products": res.product_ranking.to_dict("records"),
        "recommendations": _stringify_lists(res.recommendations).to_dict("records"),
        "opportunities": _stringify_lists(res.opportunities).to_dict("records"),
        "disclaimer": ("Mentions and sentiment are proxies for interest, not sales. Opportunities "
                       "are hypotheses, not guaranteed demand."),
    }
    return scrub_secrets(json.dumps(payload, ensure_ascii=False, indent=2, default=str)).encode("utf-8")


def all_exports(res: AnalysisResult) -> dict[str, bytes]:
    """Every downloadable file, keyed by file name. Demo data is flagged in the name."""
    tag = "DEMO-SYNTHETIC_" if (res.report.demo_valid and not res.report.real_valid) else ""
    files = {
        f"{tag}raw_comments.csv": raw_export(res),
        f"{tag}cleaned_comments.csv": cleaned_export(res),
        f"{tag}analyzed_comments.csv": analyzed_export(res),
        f"{tag}analysis_summary.csv": summary_export(res),
        f"{tag}aspect_breakdown.csv": aspect_export(res),
        f"{tag}product_ranking.csv": products_export(res),
        f"{tag}recommendations.csv": recommendations_export(res),
    }
    out = {name: to_csv_bytes(df) for name, df in files.items()}
    out[f"{tag}report.json"] = json_report(res)
    return out


# ---------------------------------------------------------------------------
# Chart / comparison tables used by the dashboard
# ---------------------------------------------------------------------------
SENTIMENTS = ["positive", "neutral", "negative"]


def sentiment_by_source(df: pd.DataFrame) -> pd.DataFrame:
    """Long table (source, sentiment, comments, pct) with pct = share within each source."""
    cols = ["source", "sentiment", "comments", "pct"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    counts = df.groupby(["source", "sentiment"]).size().unstack(fill_value=0).reindex(
        columns=SENTIMENTS, fill_value=0)
    pct = counts.div(counts.sum(axis=1), axis=0) * 100
    out = counts.stack().rename("comments").reset_index().merge(
        pct.stack().rename("pct").reset_index(), on=["source", "sentiment"])
    out["pct"] = out["pct"].round(1)
    return out[cols]


def sentiment_over_time(df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """Comments per period and sentiment (wide). Undated comments are excluded.

    ``freq`` is ``"W"`` (weekly) or ``"M"`` (monthly).
    """
    cols = ["period"] + SENTIMENTS
    if freq not in ("W", "M"):
        raise ValueError("freq must be 'W' or 'M'")
    dated = df.dropna(subset=["created_at"])
    if dated.empty:
        return pd.DataFrame(columns=cols)
    ts = dated["created_at"].dt.tz_convert(None) if dated["created_at"].dt.tz is not None \
        else dated["created_at"]
    period = ts.dt.to_period(freq).dt.start_time
    wide = pd.crosstab(period, dated["sentiment"]).reindex(columns=SENTIMENTS, fill_value=0)
    wide.index.name = "period"
    return wide.reset_index()[cols]


def competitor_comparison(results: dict[str, AnalysisResult]) -> pd.DataFrame:
    """Side-by-side snapshot per brand with dataset size, sources and date range visible.

    No significance test is run; margins of error are shown so small samples are obvious.
    """
    rows = []
    for name, res in results.items():
        snap = res.snapshot
        dist = res.analyzed["sentiment"].value_counts(normalize=True)
        rng = res.date_range
        top_complaint = res.complaints.iloc[0]["aspect"] if len(res.complaints) else ""
        top_aspect = res.aspect_summary.iloc[0]["aspect"] if len(res.aspect_summary) else ""
        net = snap.get("net_sentiment")
        rows.append({
            "Brand": name, "Valid comments": res.n,
            "Sources": ", ".join(sorted(res.sources)) or "-",
            "Date range": f"{rng[0]} to {rng[1]}" if rng else "no dates",
            "Positive %": round(100 * dist.get("positive", 0.0), 1),
            "Neutral %": round(100 * dist.get("neutral", 0.0), 1),
            "Negative %": round(100 * dist.get("negative", 0.0), 1),
            "Net sentiment": None if net is None else round(net, 3),
            "Margin of error (95%)": snap.get("margin_of_error"),
            "Top complaint aspect": top_complaint, "Most discussed aspect": top_aspect,
            "Below target": res.small_sample})
    return pd.DataFrame(rows)
