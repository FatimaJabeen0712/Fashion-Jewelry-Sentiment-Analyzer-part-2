import io

import pandas as pd
import pytest

from src.collectors.base import MockCollector
from src.collectors.csv_import import (EXAMPLE_CSV, import_csv, load_exported_csv,
                                       read_csv_raw, suggest_mapping)
from src.collectors.demo import DEMO_LABEL, generate_demo_dataset, is_demo
from src.schemas import (DEMO_METHOD, NORMALIZED_COLUMNS, MalformedCSVError, MissingColumnError,
                         ensure_schema)


def test_example_csv_imports_with_auto_mapping():
    df = import_csv(EXAMPLE_CSV.encode(), brand="Luna")
    assert list(df.columns[: len(NORMALIZED_COLUMNS)]) == NORMALIZED_COLUMNS
    assert len(df) == 2 and df["brand"].eq("Luna").all()
    assert df["like_count"].tolist() == [12, 0] and df["rating"].tolist() == [5.0, 2.0]
    assert df["created_at"].notna().all() and df["collection_method"].eq("csv_import").all()


def test_suggest_mapping():
    m = suggest_mapping(["Comment", "Date", "Likes", "Product", "Stars"])
    assert m["text"] == "Comment" and m["like_count"] == "Likes" and m["rating"] == "Stars"
    assert suggest_mapping(["foo"])["text"] is None


def test_missing_text_column_raises_actionable_error():
    with pytest.raises(MissingColumnError, match="Available columns"):
        import_csv(b"a,b\n1,2\n")
    with pytest.raises(MissingColumnError):
        import_csv(b"a,b\n1,2\n", mapping={"text": "nope"})


def test_malformed_files():
    with pytest.raises(MalformedCSVError):
        import_csv(b"")
    with pytest.raises(MalformedCSVError):
        import_csv(b"text,date\n")
    with pytest.raises(MalformedCSVError):
        read_csv_raw("/nonexistent/path.csv")


def test_encoding_roundtrip_with_emojis_commas_quotes_newlines():
    original = pd.DataFrame({"text": ['Love it, "really" \U0001F60D', "line1\nline2", "café سلام"]})
    ensured = ensure_schema(original, default_source="t")
    buf = io.StringIO()
    ensured.to_csv(buf, index=False)
    data = ("﻿" + buf.getvalue()).encode("utf-8")  # utf-8-sig like the app's downloads
    back, meta = load_exported_csv(data)
    assert back["text"].tolist() == original["text"].tolist()
    assert meta["rows"] == 3


def test_cp1252_fallback():
    df = import_csv("text\ncafé is nice\n".encode("cp1252"))
    assert df.at[0, "text"] == "café is nice"


def test_semicolon_delimiter_is_sniffed():
    df = import_csv(b"comment;likes\nnice ring;3\nbad clasp;0\n")
    assert len(df) == 2 and df["like_count"].tolist() == [3, 0]


def test_load_exported_requires_text():
    with pytest.raises(MissingColumnError):
        load_exported_csv(b"a,b\n1,2\n")


def test_demo_is_labeled_synthetic_and_deterministic():
    d1, d2 = generate_demo_dataset(n=50), generate_demo_dataset(n=50)
    pd.testing.assert_frame_equal(d1, d2)
    assert is_demo(d1) and d1["collection_method"].eq(DEMO_METHOD).all()
    assert d1["source"].eq("demo_synthetic").all() and d1["comment_id"].str.startswith("demo-").all()
    assert "SYNTHETIC" in DEMO_LABEL and not is_demo(import_csv(EXAMPLE_CSV.encode()))
    _, meta = load_exported_csv(d1.to_csv(index=False).encode())
    assert meta["contains_demo"] is True


def test_mock_collector_makes_no_network_calls():
    c = MockCollector([{"text": "hi there", "like_count": "2"}])
    assert c.status().available
    res = c.collect()
    assert res.fetched == 1 and res.per_source == {"mock": 1}
