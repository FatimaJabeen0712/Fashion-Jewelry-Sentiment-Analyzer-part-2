import pandas as pd
import pytest

from src.sentiment import SentimentAnalyzer, net_sentiment_snapshot, sentiment_distribution

A = SentimentAnalyzer()


@pytest.mark.parametrize("text,expected", [
    ("Love this ring, so elegant", "positive"),
    ("love it", "positive"),
    ("Absolutely stunning necklace \U0001F60D", "positive"),
    ("not good", "negative"),
    ("not worth the price", "negative"),
    ("The necklace tarnished after two weeks", "negative"),
    ("Waste of money, never again", "negative"),
    ("The clasp broke on the first day", "negative"),
    ("Where can I buy this ring?", "neutral"),
    ("", "neutral"),
])
def test_representative_labels(text, expected):
    assert A.analyze_text(text)[0] == expected


def test_negation_flips_polarity():
    assert A.score("good") > 0 > A.score("not good")


def test_analyze_frame_adds_columns_without_mutating():
    df = pd.DataFrame({"text": ["love it", "not good", "where to buy?"]})
    out = A.analyze_frame(df)
    assert list(out["sentiment"]) == ["positive", "negative", "neutral"]
    assert "sentiment" not in df.columns


def test_distribution_and_snapshot():
    df = pd.DataFrame({"sentiment": ["positive"] * 6 + ["negative"] * 2 + ["neutral"] * 2})
    d = sentiment_distribution(df)
    assert d["positive"] == pytest.approx(0.6)
    snap = net_sentiment_snapshot(df, target=2000)
    assert snap["net_sentiment"] == pytest.approx(0.4)
    assert snap["sample_size_warning"] is True and snap["margin_of_error"] > 0
    assert net_sentiment_snapshot(pd.DataFrame(), 2000)["net_sentiment"] is None
    assert sentiment_distribution(pd.DataFrame()) == {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
