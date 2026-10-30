import pandas as pd

from src.collectors.demo import generate_demo_dataset
from src.preprocessing import clean_text, detect_language, preprocess, spam_reason
from src.schemas import ensure_schema


def make(texts, **extra):
    df = pd.DataFrame({"text": texts})
    for k, v in extra.items():
        df[k] = v
    return ensure_schema(df, default_source="test", collection_method="csv_import")


def test_clean_text_strips_markup_and_entities():
    assert clean_text("Love it &amp; <b>will</b> buy<br>again &#128525;") == "Love it & will buy again \U0001F60D"
    assert clean_text(None) == ""
    assert clean_text("  a \n\t b  ") == "a b"


def test_clean_text_keeps_less_than_heart_emoticon():
    assert "<3" in clean_text("love it <3")


def test_empty_duplicate_and_spam_counts():
    df = make(["Love this ring", "love this ring!!", "", "   ", "<br>",
               "Subscribe to my channel for free followers", "Great quality", "Not good at all"])
    res = preprocess(df, target=2000)
    r = res.report
    assert r.fetched == 8
    assert r.empty_removed == 3
    assert r.spam_removed == 1
    assert r.duplicates_removed == 1
    assert r.valid == 3 == len(res.clean)
    assert len(res.removed) == 5
    assert set(res.removed["removal_reason"]) >= {"empty", "duplicate:text", "spam:promo_phrase"}


def test_funnel_is_consistent():
    res = preprocess(generate_demo_dataset(n=300))
    stages = [n for _, n in res.report.funnel()]
    assert stages == sorted(stages, reverse=True)
    assert stages[-1] == res.report.valid


def test_short_and_negated_comments_are_kept():
    res = preprocess(make(["love it", "not good", "ok", "\U0001F60D\U0001F60D"]))
    assert res.report.valid == 4
    assert "not good" in set(res.clean["text"])


def test_duplicates_by_comment_id():
    df = make(["first text", "different text"])
    df["comment_id"] = ["same", "same"]
    res = preprocess(df)
    assert res.report.duplicates_removed == 1 and res.report.valid == 1


def test_spam_heuristics_are_conservative():
    assert spam_reason("love it") is None
    assert spam_reason("Check this out www.shop.example/abc nice") is None
    assert spam_reason("http://a.example http://b.example") == "multiple_urls"
    assert spam_reason("aaaaaaaaaaaaaaaaaaaa") == "repeated_characters"
    assert spam_reason("buy buy buy buy buy buy buy") == "repeated_word"


def test_language_filter_keeps_unknown_and_english():
    df = make(["lovely necklace beautiful design", "love it", "ہمیں یہ بہت پسند آیا",
               "bohat acha hai yeh ring"])
    res = preprocess(df, language_filter="en")
    assert set(res.clean["text"]) == {"lovely necklace beautiful design", "love it"}
    assert res.report.language_removed == 2
    assert "approximate" in " ".join(res.report.notes)


def test_language_detection_basics():
    assert detect_language("this ring is so beautiful and I love it") == "en"
    assert detect_language("😍😍") == "unknown"
    assert detect_language("ہمیں یہ بہت پسند آیا") == "arabic-script"


def test_target_met_uses_real_valid_only_and_demo_not_counted():
    demo = generate_demo_dataset(n=300)
    res = preprocess(demo, target=100)
    assert res.report.demo_valid == res.report.valid > 100
    assert res.report.real_valid == 0 and not res.report.target_met
    assert "DEMO" in res.report.status_message()


def test_target_met_for_real_rows():
    df = make([f"unique comment number {i} about the ring" for i in range(50)])
    assert preprocess(df, target=50).report.target_met
    small = preprocess(df, target=2000).report
    assert not small.target_met and small.shortfall == 1950
    assert "Below target" in small.status_message()


def test_input_not_mutated():
    df = make(["Love it", "love it"])
    before = df.copy()
    preprocess(df)
    pd.testing.assert_frame_equal(df, before)
