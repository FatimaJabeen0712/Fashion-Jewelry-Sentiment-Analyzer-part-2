import pandas as pd

from src.aspect_analysis import extract_aspects, find_feature_requests
from src.collectors.demo import generate_demo_dataset
from src.preprocessing import preprocess
from src.product_insights import (DISCLAIMER, extract_product_mentions, rank_products,
                                  wilson_lower_bound)
from src.recommendations import HYPOTHESIS_LABEL, build_opportunities, build_recommendations
from src.schemas import ensure_schema
from src.sentiment import SentimentAnalyzer

A = SentimentAnalyzer()


def analyzed(texts_or_df):
    df = texts_or_df if isinstance(texts_or_df, pd.DataFrame) else ensure_schema(
        pd.DataFrame({"text": texts_or_df}), default_source="t")
    return A.analyze_frame(df)


def test_recommendations_are_traceable_to_comments():
    texts = [f"The necklace tarnished after {i} weeks" for i in range(1, 6)] + \
            [f"Clasp number {i} broke quickly" for i in range(4)] + ["Love it"]
    df, rows = extract_aspects(analyzed(texts), A)
    recs = build_recommendations(rows, niche="jewelry", dataset_size=len(df))
    assert not recs.empty
    ids = set(df["comment_id"])
    for _, r in recs.iterrows():
        assert r["evidence_count"] == len(r["evidence_comment_ids"]) > 0
        assert set(r["evidence_comment_ids"]) <= ids
        assert r["example_comments"] and r["limitations"]
        assert r["priority"] in ("high", "medium", "low") and r["confidence"] in ("high", "medium", "low")
    assert {"Improve tarnish and finish resistance", "Strengthen clasps and closures"} <= set(recs["recommendation"])


def test_small_dataset_limitation_is_stated():
    texts = [f"The necklace tarnished badly on day {i}" for i in range(4)]
    _, rows = extract_aspects(analyzed(texts), A)
    recs = build_recommendations(rows, "jewelry", dataset_size=4, target=2000)
    assert "2,000" in recs.iloc[0]["limitations"]


def test_min_evidence_and_empty_inputs():
    _, rows = extract_aspects(analyzed(["The necklace tarnished"]), A)
    assert build_recommendations(rows, "jewelry", min_evidence=3).empty
    assert build_recommendations(pd.DataFrame()).empty
    assert build_opportunities(pd.DataFrame()).empty


def test_niche_filters_rules():
    _, rows = extract_aspects(analyzed([f"The gold plating tarnished again {i}" for i in range(5)]), A)
    assert build_recommendations(rows, "clothing").empty or \
        "tarnish" not in " ".join(build_recommendations(rows, "clothing")["recommendation"]).lower()


def test_opportunities_are_labeled_hypotheses_with_evidence():
    texts = [f"I wish it came in more sizes {i}" for i in range(4)]
    df = analyzed(texts)
    opp = build_opportunities(find_feature_requests(df), "jewelry", dataset_size=4)
    assert not opp.empty
    row = opp.iloc[0]
    assert row["label"] == HYPOTHESIS_LABEL and "not guaranteed" in row["label"]
    assert row["validation_test"] and row["risk_assumption"]
    assert set(row["evidence_comment_ids"]) <= set(df["comment_id"])


def test_product_ranking_never_claims_best_selling():
    res = preprocess(generate_demo_dataset(n=800))
    df = analyzed(res.clean)
    ranking = rank_products(extract_product_mentions(df, "both", A))
    assert not ranking.empty
    blob = " ".join(map(str, ranking.columns)) + " " + " ".join(ranking["labels"]) + " " + " ".join(ranking["note"])
    assert "best" not in blob.lower() and "sell" not in blob.lower().replace("sales", "")
    assert set(ranking["labels"]) - {""} <= {"Most discussed", "Highest positive response",
                                           "Most discussed; Highest positive response"}
    assert ranking.iloc[0]["most_discussed_rank"] == 1 and "not sales figures" in DISCLAIMER


def test_positive_response_requires_minimum_mentions():
    df = analyzed(["Love this ring"] * 2 + ["Hate this necklace, it tarnished"] * 8)
    df["comment_id"] = [f"id{i}" for i in range(len(df))]
    ranking = rank_products(extract_product_mentions(df, "jewelry", A), min_mentions=5)
    ring = ranking[ranking["product"] == "ring"].iloc[0]
    assert pd.isna(ring["positive_response_rank"])
    assert wilson_lower_bound(2, 2) < wilson_lower_bound(80, 100)
    assert wilson_lower_bound(0, 0) == 0.0
