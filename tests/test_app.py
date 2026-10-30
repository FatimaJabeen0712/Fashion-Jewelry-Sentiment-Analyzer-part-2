"""Headless smoke tests of the Streamlit UI (no network, no API key)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def fresh():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    return at


def click(at, label):
    [b for b in at.button if b.label == label][0].click().run()


def test_app_starts_with_empty_states_and_no_exceptions():
    at = fresh()
    assert not at.exception
    assert len(at.tabs) == 7
    assert [t.label for t in at.tabs] == ["Collect Data", "Dashboard", "Product & Aspect Insights",
                                         "Recommendations", "Data Explorer", "CSV Library / Reuse",
                                         "Setup & Methodology"]
    texts = " ".join(i.value for i in at.info)
    assert "No analysis yet" in texts


def test_youtube_without_key_shows_setup_and_disables_collect():
    at = fresh()
    at.radio[0].set_value("YouTube (official API)").run()
    assert not at.exception
    collect = [b for b in at.button if b.label == "Collect comments"][0]
    assert collect.disabled
    assert any("No YouTube API key" in w.value for w in at.warning)


def test_invalid_youtube_links_are_reported():
    at = fresh()
    at.radio[0].set_value("YouTube (official API)").run()
    [t for t in at.text_area if t.label.startswith("YouTube video links")][0].set_value("https://www.instagram.com/p/abc/").run()
    assert not at.exception
    assert any("not a YouTube video link" in e.value for e in at.error)


def test_demo_flow_loads_analyzes_and_labels_demo():
    at = fresh()
    at.radio[0].set_value("Demo dataset (synthetic)").run()
    click(at, "Load synthetic demo data")
    assert not at.exception
    click(at, "Preprocess & analyze")
    assert not at.exception
    assert at.session_state["bundle"] is not None
    b = at.session_state["bundle"]
    assert b.contains_demo and not b.report.target_met
    md = " ".join(m.value for m in at.markdown)
    assert "SYNTHETIC DEMO DATA" in md
    assert any("Net sentiment" in m.label for m in at.metric)
    assert any("Valid comments analyzed" in m.label for m in at.metric)


def test_below_target_requires_acknowledgement_for_real_data():
    import pandas as pd

    from src.schemas import ensure_schema

    at = fresh()
    df = ensure_schema(pd.DataFrame({"text": [f"Lovely ring design number {i}" for i in range(30)]}),
                       collection_method="csv_import")
    at.session_state["raw"] = df
    at.session_state["sources"] = [{"platform": "csv", "identifier": "t.csv", "count": 30}]
    at.run()
    analyze = [b for b in at.button if b.label == "Preprocess & analyze"][0]
    assert analyze.disabled  # needs the sample-size acknowledgement
    at.checkbox[-1].check().run()
    click(at, "Preprocess & analyze")
    assert not at.exception and at.session_state["bundle"].small_sample
    assert any("Sample-size warning" in w.value for w in at.warning)


def test_untrusted_comment_text_is_never_rendered_as_html():
    """Recommendation examples are comment text: they must go through st.text, not markdown/HTML."""
    import pandas as pd

    from src.schemas import ensure_schema

    evil = "tarnished clasp broke badly [click here](http://phish/login)"
    df = ensure_schema(pd.DataFrame({"text": [f"{evil} {i}" for i in range(12)]}), collection_method="csv_import")
    at = fresh()
    at.session_state["raw"] = df
    at.session_state["sources"] = [{"platform": "csv", "identifier": "t.csv", "count": 12}]
    at.run()
    at.checkbox[-1].check().run()
    click(at, "Preprocess & analyze")
    assert not at.exception and at.session_state["bundle"] is not None
    plain = " ".join(t.value for t in at.text)
    md = " ".join(m.value for m in at.markdown)
    assert "phish/login" in plain   # shown, as inert plain text
    assert "phish/login" not in md  # never passed to markdown (no live link / image beacon)


# ---- automatic agent panel ------------------------------------------------------------
def agent_app(monkeypatch, tmp_path, **fake_kw):
    import config
    import src.agent as agent_mod
    from tests.test_agent import FakeYT

    monkeypatch.setenv("YOUTUBE_API_KEY", "fake-key-for-status-only")
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(agent_mod, "YouTubeCollector", lambda: FakeYT(**fake_kw))
    return fresh()


def widget(items, label_start):
    return [w for w in items if w.label.startswith(label_start)][0]


def test_agent_panel_disabled_without_key_or_input():
    at = fresh()
    run = [b for b in at.button if b.label.startswith("Run agent")][0]
    assert run.disabled
    assert any("No YouTube API key" in w.value for w in at.warning)


def test_agent_one_click_collects_saves_csv_and_analyzes(monkeypatch, tmp_path):
    at = agent_app(monkeypatch, tmp_path, per_video=60)
    run = [b for b in at.button if b.label.startswith("Run agent")][0]
    assert run.disabled                                    # key present but nothing to search for yet
    widget(at.text_input, "Brand name").set_value("Luna Jewelry").run()
    widget(at.number_input, "Minimum valid comments").set_value(100).run()
    widget(at.text_area, "Extra YouTube links").set_value("aaaaaaaaaaa bbbbbbbbbbb").run()
    click(at, "Run agent: collect \u2192 export CSV \u2192 analyze")
    assert not at.exception
    res = at.session_state["agent_result"]
    assert res.ok and res.raw_csv_path.parent.parent == tmp_path and res.raw_csv_path.exists()
    assert at.session_state["bundle"] is not None and at.session_state["bundle"].n == 120
    assert len(at.session_state["raw"]) == 120             # the reloaded CSV is the session dataset
    assert any("Videos used" in m.label for m in at.metric)
    assert any("Analysis is ready" in s.value for s in at.success)


def test_agent_below_target_loads_csv_but_waits_for_acknowledgement(monkeypatch, tmp_path):
    at = agent_app(monkeypatch, tmp_path, per_video=30)
    widget(at.text_input, "Brand name").set_value("Luna Jewelry").run()
    widget(at.text_area, "Extra YouTube links").set_value("aaaaaaaaaaa").run()
    click(at, "Run agent: collect \u2192 export CSV \u2192 analyze")
    assert not at.exception
    res = at.session_state["agent_result"]
    assert res.status == "below_target" and at.session_state["bundle"] is None
    assert at.session_state["raw"] is not None and len(at.session_state["raw"]) == 30
    assert any("below the target" in w.value for w in at.warning)
    analyze = [b for b in at.button if b.label == "Preprocess & analyze"][0]
    assert analyze.disabled                                # still needs the sample-size acknowledgement


def test_ad_optimizer_page_renders_with_simulation_warning():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "pages" / "1_Ad_Budget_Optimizer.py"), default_timeout=120).run()
    assert not at.exception
    assert any("SIMULATED" in w.value for w in at.warning)
