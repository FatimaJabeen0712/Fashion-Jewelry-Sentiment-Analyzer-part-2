import pandas as pd

from src.aspect_analysis import (extract_aspects, find_feature_requests, group_similar_requests,
                                 split_clauses, summarize_aspects, summarize_requests,
                                 top_complaints)
from src.schemas import ensure_schema


def frame(texts):
    return ensure_schema(pd.DataFrame({"text": texts}), default_source="t")


def aspects_for(text):
    df, rows = extract_aspects(frame([text]))
    return df.at[0, "aspects"].split("|") if df.at[0, "aspects"] else [], rows


def test_jewelry_terms():
    asp, _ = aspects_for("The gold plating tarnished and the clasp broke")
    assert "material" in asp and "durability" in asp


def test_clothing_terms():
    asp, _ = aspects_for("The cotton fabric is soft but the size runs small")
    assert {"material", "comfort", "fit_sizing"} <= set(asp)


def test_mixed_opinion_is_scored_per_clause():
    _, rows = aspects_for("Beautiful design but the clasp broke after a week")
    by = rows.set_index("aspect")["aspect_sentiment"].to_dict()
    assert by["design_style"] == "positive"
    assert by["durability"] == "negative"
    assert split_clauses("Great price. Bad quality") == ["Great price", "Bad quality"]


def test_no_aspect_means_empty_rows():
    df, rows = extract_aspects(frame(["hello there"]))
    assert df.at[0, "aspects"] == "" and rows.empty


def test_summary_and_complaints_have_evidence():
    texts = ["The necklace tarnished after two weeks"] * 1 + [
        "Clasp broke on day one", "Gold plating tarnished fast", "Great design, love it"]
    df, rows = extract_aspects(frame(texts))
    summary = summarize_aspects(rows)
    assert summary.loc[summary["aspect"] == "durability", "negative"].iloc[0] >= 3
    comp = top_complaints(rows)
    assert comp.iloc[0]["aspect"] == "durability"
    assert comp.iloc[0]["examples"] and set(comp.iloc[0]["example_comment_ids"]) <= set(df["comment_id"])
    assert summarize_aspects(pd.DataFrame()).empty and top_complaints(pd.DataFrame()).empty


def test_feature_requests_detected_and_grouped():
    texts = ["I wish it came in rose gold", "I wish it came in rose gold and silver",
             "Please make the ring in more sizes", "Would be better if it was waterproof",
             "Can you please restock the necklace?", "Love it"]
    df = frame(texts)
    req = find_feature_requests(df)
    assert len(req) == 5 and "Love it" not in " ".join(req["request_text"])
    assert set(req["comment_id"]) <= set(df["comment_id"])
    themes = dict(zip(req["comment_id"], req["theme"]))
    assert themes[df.at[3, "comment_id"]] == "waterproof_tarnish_free"
    grouped = group_similar_requests(req)
    assert grouped["request_group"].nunique() < len(grouped)  # near-duplicates merged
    assert not summarize_requests(grouped).empty
    assert group_similar_requests(pd.DataFrame(columns=req.columns)).empty
