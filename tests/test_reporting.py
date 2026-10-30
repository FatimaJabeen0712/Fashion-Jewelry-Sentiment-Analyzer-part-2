import io
import json

import pandas as pd
import pytest

from src.collectors.csv_import import load_exported_csv
from src.collectors.demo import generate_demo_dataset
from src.pipeline import run_analysis
from src.reporting import (all_exports, analyzed_export, competitor_comparison, sentiment_by_source,
                           sentiment_over_time, summary_export, to_csv_bytes)
from src.schemas import ConnectorError, ConnectorNotConfigured, ensure_schema
from src.security import (contains_secret, safe_object_name, scrub_secrets, spreadsheet_safe,
                          unguard_formula)
from src import storage

SECRET = "AIzaSyFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE12"


@pytest.fixture(scope="module")
def demo_result():
    return run_analysis(generate_demo_dataset(n=700), niche="Both", brand="Demo Brand", target=2000)


def real_frame(n=60, extra=None):
    df = pd.DataFrame({"text": [f"The necklace number {i} tarnished badly, clasp broke" if i % 2
                                else f"Love ring {i}, so elegant" for i in range(n)]})
    if extra:
        for k, v in extra.items():
            df[k] = v
    return ensure_schema(df, default_source="csv_upload", collection_method="csv_import")


def test_exports_exist_and_are_utf8_sig(demo_result):
    files = all_exports(demo_result)
    assert any(name.endswith("report.json") for name in files)
    csvs = {k: v for k, v in files.items() if k.endswith(".csv")}
    assert len(csvs) == 7
    for name, data in csvs.items():
        assert data.startswith(b"\xef\xbb\xbf"), name
        pd.read_csv(io.BytesIO(data), encoding="utf-8-sig")  # parseable
    assert all(k.startswith("DEMO-SYNTHETIC_") for k in files)  # demo is visibly labeled


def test_csv_roundtrip_with_hard_characters():
    df = ensure_schema(pd.DataFrame({"text": ['Love it, "really" \U0001F60D', "two\nlines", "café سلام"]}))
    back, meta = load_exported_csv(to_csv_bytes(df))
    assert back["text"].tolist() == df["text"].tolist() and meta["rows"] == 3


def test_analyzed_export_roundtrips_and_can_be_reanalyzed(demo_result):
    data = to_csv_bytes(analyzed_export(demo_result))
    back, meta = load_exported_csv(data)
    assert meta["has_sentiment"] and meta["contains_demo"]
    again = run_analysis(back, target=2000)
    assert again.n == demo_result.n


def test_exports_exclude_non_allowlisted_columns_and_secrets(monkeypatch):
    monkeypatch.setenv("YOUTUBE_API_KEY", "super-secret-key-value-123")
    df = real_frame(extra={"author_name": "Jane Private", "profile_url": "http://x/jane"})
    df.loc[0, "text"] = f"my key is super-secret-key-value-123 and {SECRET}"
    res = run_analysis(df, target=10)
    for name, data in all_exports(res).items():
        text = data.decode("utf-8-sig")
        assert "Jane Private" not in text and "profile_url" not in text and "author_name" not in text, name
        assert "super-secret-key-value-123" not in text, name   # every CSV and the JSON
        assert SECRET not in text, name


def test_scrub_and_detect_secrets(monkeypatch):
    monkeypatch.setenv("YOUTUBE_API_KEY", "super-secret-key-value-123")
    assert scrub_secrets("x super-secret-key-value-123 y") == "x [redacted] y"
    assert scrub_secrets(f"k={SECRET}") == "k=[redacted]"
    assert contains_secret(f"k={SECRET}") and not contains_secret("harmless")


def test_formula_injection_is_neutralized():
    df = pd.DataFrame({"text": ["=HYPERLINK(\"http://evil\")", "+1+1", "@SUM(A1)", "-5", "-cmd|x", "fine"]})
    out = spreadsheet_safe(df)["text"].tolist()
    assert out[0].startswith("'=") and out[1].startswith("'+") and out[2].startswith("'@")
    assert out[3] == "-5" and out[4].startswith("'-") and out[5] == "fine"


def test_safe_object_name():
    assert safe_object_name("../../Evil Name!", "a/b") == "evil-name_a-b.csv"
    assert "/" not in safe_object_name("a/b\\c") and safe_object_name("", "") == "export.csv"


def test_summary_states_demo_and_target(demo_result):
    s = summary_export(demo_result).set_index("metric")["value"]
    assert s["contains_synthetic_demo_data"] is True or str(s["contains_synthetic_demo_data"]) == "True"
    assert int(s["real_valid_unique_comments"]) == 0 and int(s["target"]) == 2000
    assert str(s["target_met"]) == "False" and str(s["sample_size_warning"]) == "True"
    assert "model_version" in s.index and "collection_analysis_timestamp_utc" in s.index


def test_real_data_summary_and_recommendations_traceable():
    res = run_analysis(real_frame(80), niche="Jewelry", brand="Luna", target=2000)
    s = summary_export(res).set_index("metric")["value"]
    assert int(s["real_valid_unique_comments"]) == res.report.valid > 0
    assert not any(k.startswith("DEMO") for k in all_exports(res))
    assert not res.recommendations.empty
    ids = set(res.analyzed["comment_id"])
    assert all(set(r) <= ids for r in res.recommendations["evidence_comment_ids"])
    rec_csv = pd.read_csv(io.BytesIO(all_exports(res)["recommendations.csv"]), encoding="utf-8-sig")
    assert {"recommendation", "evidence_count", "priority", "confidence", "example_comments"} <= set(rec_csv.columns)


def test_json_report_valid(demo_result):
    data = json.loads(all_exports(demo_result)["DEMO-SYNTHETIC_report.json"])
    assert data["preprocessing"]["demo_valid"] > 0 and "not sales" in data["disclaimer"]


# ---- optional Cloud Storage (fake client, no network) ---------------------------
class FakeBlob:
    def __init__(self, store, name): self.store, self.name = store, name
    def upload_from_string(self, data, content_type=None): self.store[self.name] = (data, content_type)
    def download_as_bytes(self): return self.store[self.name][0]


class FakeBucket:
    def __init__(self, store): self.store = store
    def blob(self, name): return FakeBlob(self.store, name)


class FakeClient:
    def __init__(self): self.store = {}
    def bucket(self, name): return FakeBucket(self.store)
    def list_blobs(self, bucket, prefix="", max_results=50):
        return [type("B", (), {"name": n})() for n in self.store if n.startswith(prefix)][:max_results]


def test_gcs_requires_configuration():
    assert not storage.is_configured()
    with pytest.raises(ConnectorNotConfigured):
        storage.save_exports({"a.csv": b"x"}, "label")


def test_gcs_save_list_load_with_fake_client(monkeypatch):
    monkeypatch.setenv("GCS_BUCKET_NAME", "my-private-bucket")
    c = FakeClient()
    names = storage.save_exports({"cleaned_comments.csv": b"a,b", "report.json": b"{}"}, "My Brand!", client=c)
    assert all(n.startswith("exports/") and ".." not in n for n in names)
    assert c.store[names[1]][1] == "application/json"
    assert storage.list_exports(client=c) == names
    assert storage.load_export(names[0], client=c) == b"a,b"
    with pytest.raises(ConnectorError):
        storage.load_export("../secrets.txt", client=c)
    with pytest.raises(ConnectorError):
        storage.load_export("other/file.csv", client=c)


def test_formula_guard_roundtrips_through_export_and_reload():
    df = ensure_schema(pd.DataFrame({"text": ["=1+1 is my size", "+1 great", "-5 stars", "@brand love it"]}))
    data = to_csv_bytes(df)
    assert b"'=1+1" in data and b"'+1" in data and b"'@brand" in data
    back, _ = load_exported_csv(data)
    assert back["text"].tolist() == df["text"].tolist()
    assert unguard_formula("'normal quote") == "'normal quote" and unguard_formula("'") == "'"


def test_secret_from_streamlit_style_source_is_scrubbed(monkeypatch):
    monkeypatch.setattr("src.security.get_secret", lambda n, d=None: "toml-only-secret-9999" if n == "YOUTUBE_API_KEY" else None)
    df = ensure_schema(pd.DataFrame({"text": ["leak toml-only-secret-9999 here"]}))
    assert b"toml-only-secret-9999" not in to_csv_bytes(df)


def test_sentiment_by_source_and_over_time(demo_result):
    bs = sentiment_by_source(demo_result.analyzed)
    assert set(bs["sentiment"]) == {"positive", "neutral", "negative"}
    assert bs.groupby("source")["pct"].sum().round(0).eq(100).all()
    for freq in ("W", "M"):
        ot = sentiment_over_time(demo_result.analyzed, freq)
        assert list(ot.columns) == ["period", "positive", "neutral", "negative"]
        assert int(ot[["positive", "neutral", "negative"]].sum().sum()) == demo_result.n
    assert sentiment_over_time(demo_result.analyzed.assign(created_at=pd.NaT), "W").empty
    assert sentiment_by_source(demo_result.analyzed.iloc[0:0]).empty
    with pytest.raises(ValueError):
        sentiment_over_time(demo_result.analyzed, "D")


def test_competitor_comparison_shows_sizes_and_warns_small(demo_result):
    small = run_analysis(real_frame(30), niche="Jewelry", brand="Rival", target=2000)
    table = competitor_comparison({"Us": demo_result, "Rival": small})
    assert list(table["Brand"]) == ["Us", "Rival"]
    assert table.loc[1, "Valid comments"] == small.n and bool(table.loc[1, "Below target"])
    assert {"Sources", "Date range", "Margin of error (95%)"} <= set(table.columns)
