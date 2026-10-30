"""The agent end to end with a fake YouTube collector (no network)."""
import json

import pandas as pd
import pytest

import src.agent as agent_mod
from run_agent import main as cli_main
from src.agent import AgentConfig, brand_matches, run_agent
from src.collectors.base import CollectionResult, ConnectorStatus
from src.collectors.youtube import QuotaExceeded
from src.schemas import ConnectorNotConfigured, PipelineError, ensure_schema

V1, V2, V3, V4, V5 = ("aaaaaaaaaaa", "bbbbbbbbbbb", "ccccccccccc", "ddddddddddd", "eeeeeeeeeee")


class FakeYT:
    """Stands in for YouTubeCollector: same methods, canned data, records what was asked."""

    def __init__(self, comments=None, hits=None, search_error=None, quota_videos=(), per_video=30):
        self.comments, self.hits, self.search_error = comments or {}, hits or [], search_error
        self.quota_videos, self.per_video = set(quota_videos), per_video
        self.requests_made, self.search_calls = 0, 0
        self.queries, self.collected = [], []

    def status(self):
        return ConnectorStatus("youtube", True, "fake")

    def search_videos(self, query, max_results=50, max_pages=2, **kw):
        self.search_calls += 1
        self.queries.append(query)
        if self.search_error:
            raise self.search_error
        return self.hits

    def collect(self, video_inputs=None, brand="", max_comments_per_video=1000, fetch_titles=True, **kw):
        vid = video_inputs[0]
        self.collected.append(vid)
        self.requests_made += 1
        if vid in self.quota_videos:
            return CollectionResult(df=ensure_schema(pd.DataFrame()), per_source={vid: 0},
                                    errors=[f"{vid}: YouTube API daily quota is exhausted."])
        n = self.comments.get(vid, self.per_video)
        texts = [f"The necklace number {i} from {vid} tarnished, but I love the design" if i % 2
                 else f"Love this ring {i} from {vid}, so elegant" for i in range(n)]
        df = ensure_schema(pd.DataFrame({"text": texts, "comment_id": [f"yt-{vid}-{i}" for i in range(n)],
                                         "source": "youtube", "collection_method": "youtube_api_v3",
                                         "video_id": vid, "brand": brand}))
        return CollectionResult(df=df, per_source={vid: n})


def hit(vid, title, desc=""):
    return {"video_id": vid, "title": title, "description": desc, "published_at": "2026-01-01T00:00:00Z"}


def cfg(tmp_path, **kw):
    base = dict(brand="Luna Jewelry", niche="Jewelry", target=50, out_dir=tmp_path, discover=False)
    base.update(kw)
    return AgentConfig(**base)


def test_full_flow_saves_csv_reloads_it_and_analyzes(tmp_path):
    yt = FakeYT()
    steps = []
    res = run_agent(cfg(tmp_path, video_inputs=[V1, V2]), yt, progress=lambda s, d: steps.append(s))
    assert res.ok and res.status == "complete" and res.analysis.n == 60
    raw = res.raw_csv_path
    assert raw.parent == res.out_dir and raw.read_bytes().startswith(b"\xef\xbb\xbf")
    names = {p.name for p in res.files}
    assert {"raw_comments.csv", "analysis_summary.csv", "recommendations.csv", "report.json", "agent_log.json"} <= names
    # the analysis ran on what was read back from the saved file
    saved = pd.read_csv(raw, encoding="utf-8-sig")
    assert set(res.analysis.analyzed["comment_id"]) == set(saved["comment_id"]) and len(saved) == 60
    assert [s for s in steps if s in ("Saved CSV", "Reloaded CSV", "Analysis complete")] == \
        ["Saved CSV", "Reloaded CSV", "Analysis complete"]
    assert {"video_id", "video_title"} <= set(saved.columns)
    assert [v.comments_fetched for v in res.videos] == [30, 30]


def test_stops_collecting_once_target_of_valid_comments_is_reached(tmp_path):
    yt = FakeYT()
    res = run_agent(cfg(tmp_path, video_inputs=[V1, V2, V3, V4, V5], target=50), yt)
    assert yt.collected == [V1, V2] and res.ok          # 60 >= 50 after two videos; the rest not requested
    assert any("Target reached" in s for s in res.steps)


def test_discovery_uses_default_query_and_brand_filter(tmp_path):
    hits = [hit(V1, "Luna Jewelry unboxing"), hit(V2, "Random haul", "best rings"),
            hit(V3, "my luna ring review"), hit(V4, "Other brand", "mentions Luna Jewelry in description")]
    yt = FakeYT(hits=hits)
    res = run_agent(cfg(tmp_path, discover=True, target=80), yt)
    assert yt.queries == ["Luna Jewelry review"] and yt.search_calls == 1
    assert yt.collected == [V1, V3, V4] and [v["video_id"] for v in res.skipped_videos] == [V2]
    saved = pd.read_csv(res.raw_csv_path, encoding="utf-8-sig")
    assert saved.loc[saved["video_id"] == V1, "video_title"].iloc[0] == "Luna Jewelry unboxing"
    log = json.loads((res.out_dir / "agent_log.json").read_text(encoding="utf-8"))
    assert log["skipped_as_not_matching_brand"][0]["video_id"] == V2 and log["search_calls"] == 1


def test_brand_filter_can_be_turned_off_and_custom_query_used(tmp_path):
    yt = FakeYT(hits=[hit(V1, "Random haul"), hit(V2, "Other")])
    run_agent(cfg(tmp_path, discover=True, require_brand_match=False, search_query="gold rings pakistan", target=50), yt)
    assert yt.queries == ["gold rings pakistan"] and yt.collected == [V1, V2]


def test_explicit_links_always_collected_and_not_duplicated_by_search(tmp_path):
    yt = FakeYT(hits=[hit(V1, "Luna Jewelry again"), hit(V2, "Luna Jewelry two")])
    run_agent(cfg(tmp_path, discover=True, video_inputs=[f"https://youtu.be/{V1}"], target=50), yt)
    assert yt.collected == [V1, V2]


def test_below_target_saves_csv_but_does_not_analyze_unless_allowed(tmp_path):
    res = run_agent(cfg(tmp_path, video_inputs=[V1], target=2000), FakeYT())
    assert res.status == "below_target" and res.analysis is None and not res.ok
    assert res.raw_csv_path.exists() and not (res.out_dir / "analysis_summary.csv").exists()
    assert res.reloaded is not None and "below the target" in res.message and res.valid_comments == 30
    allowed = run_agent(cfg(tmp_path, video_inputs=[V1], target=2000, allow_small=True), FakeYT())
    assert allowed.ok and allowed.analysis.small_sample and "sample-size warning" in allowed.message


def test_quota_exhaustion_stops_but_keeps_what_was_collected(tmp_path):
    yt = FakeYT(quota_videos=[V2])
    res = run_agent(cfg(tmp_path, video_inputs=[V1, V2, V3], target=500), yt)
    assert yt.collected == [V1, V2] and res.raw_csv_path.exists()
    assert any("quota" in e.lower() for e in res.errors) and res.valid_comments == 30
    assert any("quota" in s.lower() for s in res.steps)


def test_search_failure_does_not_stop_collection_of_provided_videos(tmp_path):
    yt = FakeYT(search_error=QuotaExceeded("search quota used up"))
    res = run_agent(cfg(tmp_path, discover=True, video_inputs=[V1, V2], target=50), yt)
    assert res.ok and any("Video search failed" in e for e in res.errors)


def test_no_comments_gives_no_data_and_writes_nothing(tmp_path):
    res = run_agent(cfg(tmp_path, video_inputs=[V1]), FakeYT(comments={V1: 0}))
    assert res.status == "no_data" and res.raw_csv_path is None and list(tmp_path.iterdir()) == []


def test_all_spam_gives_no_valid_comments(tmp_path):
    class Spammy(FakeYT):
        def collect(self, video_inputs=None, **kw):
            r = super().collect(video_inputs=video_inputs, **kw)
            r.df["text"] = "Subscribe to my channel for free followers"
            return r
    res = run_agent(cfg(tmp_path, video_inputs=[V1]), Spammy())
    assert res.status == "no_valid_comments" and res.analysis is None and res.raw_csv_path.exists()


def test_missing_key_and_missing_input_raise_clear_errors(tmp_path):
    with pytest.raises(ConnectorNotConfigured, match="YOUTUBE_API_KEY"):
        run_agent(cfg(tmp_path, video_inputs=[V1]))                 # real collector, no key (conftest clears it)
    with pytest.raises(PipelineError, match="brand name or search phrase"):
        run_agent(AgentConfig(out_dir=tmp_path), FakeYT())
    with pytest.raises(PipelineError):
        run_agent(cfg(tmp_path, video_inputs=["https://www.instagram.com/p/x/"], discover=False), FakeYT())
    with pytest.raises(PipelineError):
        run_agent(cfg(tmp_path, target=0), FakeYT())


def test_outputs_never_contain_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("YOUTUBE_API_KEY", "agent-secret-key-12345")
    res = run_agent(cfg(tmp_path, video_inputs=[V1], brand="Luna agent-secret-key-12345"), FakeYT())
    for p in res.files:
        assert "agent-secret-key-12345" not in p.read_text(encoding="utf-8-sig"), p.name


def test_brand_matching_rules():
    assert brand_matches("Hiven Jewelry", "Unboxing HIVEN rings!") and brand_matches("Hiven Jewelry", "hiven jewelry")
    assert not brand_matches("Hiven Jewelry", "Best jewelry haul 2026")      # generic word alone is not enough
    assert brand_matches("Luna & Co", "luna co sale") and not brand_matches("Luna & Co", "lunar eclipse")
    assert brand_matches("", "anything") and not brand_matches("Jewelry Shop", "cool shop")


# ---- command line ---------------------------------------------------------------
def test_cli_exit_codes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(agent_mod, "YouTubeCollector", lambda: FakeYT())
    ok = cli_main(["--brand", "Luna Jewelry", "--videos", V1, V2, "--no-discover", "--target", "50", "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert ok == 0 and "COMPLETE" in out and "Files saved in" in out
    small = cli_main(["--brand", "Luna Jewelry", "--videos", V1, "--no-discover", "--target", "2000", "--out", str(tmp_path)])
    assert small == 2 and "BELOW_TARGET" in capsys.readouterr().out
    monkeypatch.undo()
    assert cli_main(["--brand", "x", "--videos", V1, "--no-discover", "--out", str(tmp_path)]) == 1  # no API key
    assert "ERROR" in capsys.readouterr().err
