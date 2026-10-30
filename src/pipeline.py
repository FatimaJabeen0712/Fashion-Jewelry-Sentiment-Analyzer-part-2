"""Orchestrates preprocessing -> sentiment -> aspects -> products -> recommendations.

Kept separate from the Streamlit UI so it can be tested and reused from scripts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from config import DEFAULT_TARGET, SENTIMENT_MODEL_VERSION
from src.aspect_analysis import (extract_aspects, find_feature_requests, group_similar_requests,
                                 summarize_aspects, summarize_requests, top_complaints)
from src.preprocessing import PreprocessResult, preprocess
from src.product_insights import extract_product_mentions, rank_products
from src.recommendations import build_opportunities, build_recommendations
from src.schemas import PreprocessReport
from src.sentiment import SentimentAnalyzer, net_sentiment_snapshot


@dataclass
class AnalysisResult:
    brand: str
    niche: str
    analyzed_at: str
    report: PreprocessReport
    raw: pd.DataFrame
    clean: pd.DataFrame
    analyzed: pd.DataFrame
    removed: pd.DataFrame
    aspect_rows: pd.DataFrame
    aspect_summary: pd.DataFrame
    complaints: pd.DataFrame
    requests: pd.DataFrame
    request_summary: pd.DataFrame
    product_mentions: pd.DataFrame
    product_ranking: pd.DataFrame
    recommendations: pd.DataFrame
    opportunities: pd.DataFrame
    snapshot: dict
    sources: dict[str, int] = field(default_factory=dict)
    model_version: str = SENTIMENT_MODEL_VERSION

    @property
    def n(self) -> int:
        return len(self.analyzed)

    @property
    def contains_demo(self) -> bool:
        return self.report.demo_valid > 0

    @property
    def target(self) -> int:
        return self.report.target

    @property
    def small_sample(self) -> bool:
        return self.report.real_valid < self.report.target

    @property
    def date_range(self) -> tuple[str, str] | None:
        s = self.analyzed["created_at"].dropna()
        return None if s.empty else (str(s.min().date()), str(s.max().date()))


def run_analysis(df: pd.DataFrame, niche: str = "Both", brand: str = "",
                 target: int = DEFAULT_TARGET, language_filter: str = "any") -> AnalysisResult:
    """Run the full pipeline on a normalized DataFrame (never mutates ``df``)."""
    pre: PreprocessResult = preprocess(df, target=target, language_filter=language_filter)
    analyzer = SentimentAnalyzer()
    scored = analyzer.analyze_frame(pre.clean)
    if brand:
        scored["brand"] = scored["brand"].where(scored["brand"].str.strip() != "", brand)
    analyzed, aspect_rows = extract_aspects(scored, analyzer)
    mentions = extract_product_mentions(analyzed, niche, analyzer)
    requests = group_similar_requests(find_feature_requests(analyzed))
    size = len(analyzed)
    return AnalysisResult(
        brand=brand, niche=niche, analyzed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        report=pre.report, raw=df, clean=pre.clean, analyzed=analyzed, removed=pre.removed,
        aspect_rows=aspect_rows, aspect_summary=summarize_aspects(aspect_rows),
        complaints=top_complaints(aspect_rows), requests=requests,
        request_summary=summarize_requests(requests), product_mentions=mentions,
        product_ranking=rank_products(mentions),
        recommendations=build_recommendations(aspect_rows, niche, size, target),
        opportunities=build_opportunities(requests, niche, size, target),
        snapshot=net_sentiment_snapshot(analyzed, target),
        sources=analyzed["source"].value_counts().to_dict(), model_version=analyzer.version)
