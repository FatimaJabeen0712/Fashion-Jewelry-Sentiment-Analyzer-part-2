import pytest
import requests

from src.collectors.meta import MetaCollector
from src.collectors.tiktok_research import TikTokResearchCollector
from src.collectors.youtube import (COMMENT_THREADS_URL, SEARCH_URL, InvalidYouTubeSource, QuotaExceeded,
                                    YouTubeCollector, parse_video_id, parse_video_inputs)
from src.schemas import ConnectorError, ConnectorNotConfigured

KEY = "FAKE-KEY-DO-NOT-LEAK-123"
VID = "dQw4w9WgXcQ"
VID2 = "abcdefghijk"


def thread(cid, text, likes=0, ts="2026-03-01T10:00:00Z", replies=0):
    return {"id": cid, "snippet": {"totalReplyCount": replies, "topLevelComment": {
        "id": cid, "snippet": {"textOriginal": text, "textDisplay": text, "publishedAt": ts,
                               "likeCount": likes, "authorDisplayName": "SHOULD-NOT-BE-STORED"}}}}


class Resp:
    def __init__(self, status=200, body=None):
        self.status_code, self._body = status, body or {}

    def json(self):
        return self._body


def err(status, reason):
    return Resp(status, {"error": {"code": status, "message": "boom", "errors": [{"reason": reason}]}})


class FakeSession:
    """Scripted responses per call; records every request. No network."""

    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params or {}), "headers": dict(headers or {}),
                           "timeout": timeout})
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def collector(responses, **kw):
    return YouTubeCollector(api_key=KEY, session=FakeSession(responses), sleep=lambda s: None, **kw)


# ---- URL parsing -------------------------------------------------------------
@pytest.mark.parametrize("value", [
    VID, f"https://www.youtube.com/watch?v={VID}", f"https://youtube.com/watch?v={VID}&t=30s",
    f"https://youtu.be/{VID}?si=abc", f"https://www.youtube.com/shorts/{VID}",
    f"https://m.youtube.com/watch?v={VID}", f"www.youtube.com/embed/{VID}",
    f"https://www.youtube.com/live/{VID}", f"  https://youtu.be/{VID}  "])
def test_parse_valid(value):
    assert parse_video_id(value) == VID


@pytest.mark.parametrize("value", [
    "", "   ", "https://www.instagram.com/p/abc123/", "https://www.tiktok.com/@brand/video/1",
    "https://brand-website.com/products/ring", "https://www.youtube.com/watch",
    "https://www.youtube.com/watch?v=short", "not a url at all", "https://youtu.be/"])
def test_parse_rejects(value):
    with pytest.raises(InvalidYouTubeSource):
        parse_video_id(value)


def test_channel_links_are_not_silently_expanded():
    for url in ("https://www.youtube.com/@brand", "https://www.youtube.com/channel/UC123",
                "https://www.youtube.com/playlist?list=PL1"):
        with pytest.raises(InvalidYouTubeSource, match="not expanded"):
            parse_video_id(url)


def test_parse_inputs_dedupes_and_reports_errors():
    ids, errors = parse_video_inputs(f"https://youtu.be/{VID}, {VID}\nhttps://tiktok.com/x {VID2}")
    assert ids == [VID, VID2] and len(errors) == 1


# ---- configuration -----------------------------------------------------------
def test_missing_key_gives_setup_instructions_and_status():
    c = YouTubeCollector(session=FakeSession([]))
    st = c.status()
    assert not st.available and "console.cloud.google.com" in st.message
    with pytest.raises(ConnectorNotConfigured, match="YOUTUBE_API_KEY"):
        c.collect(video_inputs=[VID])


def test_status_never_exposes_key():
    st = collector([]).status()
    assert st.available and KEY not in st.message


# ---- pagination --------------------------------------------------------------
def test_pagination_follows_next_page_token_and_maps_fields():
    pages = [Resp(200, {"items": [thread("a", "Love it", 3), thread("b", "Bad clasp", 0, replies=2)],
                        "nextPageToken": "P2"}),
             Resp(200, {"items": [thread("c", "Nice ring")]})]
    c = collector(pages)
    res = c.collect(video_inputs=[VID], fetch_titles=False)
    s = c._session
    assert [x["params"].get("pageToken") for x in s.calls] == [None, "P2"]
    assert all(x["url"] == COMMENT_THREADS_URL and x["params"]["videoId"] == VID for x in s.calls)
    assert s.calls[0]["params"]["maxResults"] == 100 and s.calls[0]["params"]["textFormat"] == "plainText"
    df = res.df
    assert len(df) == 3 and res.per_source == {VID: 3} and not res.errors
    assert df["comment_id"].tolist() == ["yt-a", "yt-b", "yt-c"]
    assert df.loc[0, "like_count"] == 3 and df.loc[1, "reply_count"] == 2
    assert df["source"].eq("youtube").all() and df["collection_method"].eq("youtube_api_v3").all()
    assert df["post_url"].iloc[0].endswith(VID) and df["created_at"].notna().all()
    assert "SHOULD-NOT-BE-STORED" not in df.to_csv()  # no commenter data kept


def test_limit_stops_early_and_shrinks_page_size():
    c = collector([Resp(200, {"items": [thread(f"x{i}", f"c{i}") for i in range(5)],
                              "nextPageToken": "MORE"})])
    res = c.collect(video_inputs=[VID], max_comments_per_video=5, fetch_titles=False)
    assert len(res.df) == 5 and len(c._session.calls) == 1
    assert c._session.calls[0]["params"]["maxResults"] == 5


def test_max_pages_limit():
    pages = [Resp(200, {"items": [thread(f"p{i}", "t")], "nextPageToken": "n"}) for i in range(5)]
    c = collector(pages)
    res = c.collect(video_inputs=[VID], max_pages_per_video=2, fetch_titles=False)
    assert len(c._session.calls) == 2 and len(res.df) == 2


def test_duplicate_comment_ids_across_pages_and_videos_are_dropped():
    c = collector([Resp(200, {"items": [thread("a", "one"), thread("a", "one")]}),
                   Resp(200, {"items": [thread("a", "one"), thread("z", "two")]})])
    res = c.collect(video_inputs=[VID, VID2], fetch_titles=False)
    assert res.df["comment_id"].tolist() == ["yt-a", "yt-z"]
    assert res.per_source == {VID: 1, VID2: 1}


def test_multiple_videos_report_per_video_counts():
    c = collector([Resp(200, {"items": [thread("a", "x1"), thread("b", "x2")]}),
                   Resp(200, {"items": [thread("c", "y1")]})])
    res = c.collect(video_inputs=f"{VID} {VID2}".split(), fetch_titles=False)
    assert res.per_source == {VID: 2, VID2: 1} and res.fetched == 3


def test_title_fetched_when_enabled_and_failure_is_non_fatal():
    ok = collector([Resp(200, {"items": [{"snippet": {"title": "My Video"}}]}),
                    Resp(200, {"items": [thread("a", "hi there")]})])
    assert ok.collect(video_inputs=[VID]).df.loc[0, "video_title"] == "My Video"
    bad = collector([err(403, "forbidden"), Resp(200, {"items": [thread("a", "hi there")]})])
    res = bad.collect(video_inputs=[VID])
    assert len(res.df) == 1 and res.df.loc[0, "video_title"] == ""


def test_date_filter_stops_on_time_order():
    items = [thread("new", "n", ts="2026-03-10T00:00:00Z"), thread("old", "o", ts="2026-01-01T00:00:00Z")]
    from datetime import datetime
    c = collector([Resp(200, {"items": items, "nextPageToken": "MORE"})])
    res = c.collect(video_inputs=[VID], published_after=datetime(2026, 2, 1), fetch_titles=False)
    assert res.df["comment_id"].tolist() == ["yt-new"] and len(c._session.calls) == 1


# ---- errors ------------------------------------------------------------------
def test_comments_disabled_and_not_found_are_per_video_errors():
    c = collector([err(403, "commentsDisabled"), err(404, "videoNotFound"),
                   Resp(200, {"items": [thread("a", "fine")]})])
    res = c.collect(video_inputs=[VID, VID2, "zzzzzzzzzzz"], fetch_titles=False)
    assert len(res.df) == 1 and len(res.errors) == 2
    assert "disabled" in res.errors[0] and "not found" in res.errors[1].lower()
    assert res.per_source[VID] == 0


def test_quota_exhaustion_keeps_partial_data_and_lists_skipped_videos():
    c = collector([Resp(200, {"items": [thread("a", "kept")]}), err(403, "quotaExceeded")])
    res = c.collect(video_inputs=[VID, VID2, "zzzzzzzzzzz"], fetch_titles=False)
    assert len(res.df) == 1
    assert "quota" in res.errors[0].lower() and "zzzzzzzzzzz" in res.errors[0]
    assert len(c._session.calls) == 2  # stopped, no further videos attempted
    assert isinstance(YouTubeCollector._translate(403, "quotaExceeded"), QuotaExceeded)


def test_invalid_key_and_forbidden_messages():
    for status, reason, text in [(400, "keyInvalid", "key was rejected"),
                                 (403, "accessNotConfigured", "enabled")]:
        res = collector([err(status, reason)]).collect(video_inputs=[VID], fetch_titles=False)
        assert text in res.errors[0]


def test_transient_errors_retry_with_backoff_then_succeed():
    sleeps = []
    s = FakeSession([err(503, ""), requests.exceptions.Timeout(), Resp(200, {"items": [thread("a", "ok")]})])
    c = YouTubeCollector(api_key=KEY, session=s, sleep=sleeps.append, backoff=1.0)
    res = c.collect(video_inputs=[VID], fetch_titles=False)
    assert len(res.df) == 1 and sleeps == [1.0, 2.0]


def test_gives_up_after_retries_with_safe_message():
    s = FakeSession([err(500, "")] * 4)
    c = YouTubeCollector(api_key=KEY, session=s, sleep=lambda x: None, max_retries=3)
    res = c.collect(video_inputs=[VID], fetch_titles=False)
    assert res.df.empty and "not responding" in res.errors[0]


def test_no_valid_input_raises_actionable_error():
    with pytest.raises(InvalidYouTubeSource):
        collector([]).collect(video_inputs=["https://www.instagram.com/p/x/"])
    with pytest.raises(InvalidYouTubeSource):
        collector([]).collect(video_inputs=[])


# ---- secrets -----------------------------------------------------------------
def test_key_sent_in_header_never_in_url_params_or_errors():
    c = collector([Resp(200, {"items": [thread("a", "ok")]}), err(403, "commentsDisabled")])
    res = c.collect(video_inputs=[VID, VID2], fetch_titles=False)
    for call in c._session.calls:
        assert "key" not in call["params"] and KEY not in str(call["params"]) and KEY not in call["url"]
        assert call["headers"]["X-Goog-Api-Key"] == KEY and call["timeout"] == 15.0
    assert KEY not in " ".join(res.errors) and KEY not in res.df.to_csv()


# ---- restricted connectors ---------------------------------------------------
@pytest.mark.parametrize("cls", [MetaCollector, TikTokResearchCollector])
def test_restricted_connectors_are_honestly_unavailable(cls):
    c = cls()
    assert not c.status().available
    with pytest.raises(ConnectorNotConfigured):
        c.collect()
    assert issubclass(ConnectorNotConfigured, ConnectorError)


# ---- video search (search.list) ---------------------------------------------------
def hit(vid, title="Ring review", desc="d"):
    return {"id": {"kind": "youtube#video", "videoId": vid}, "snippet": {"title": title, "description": desc,
            "publishedAt": "2026-02-01T00:00:00Z", "channelTitle": "SHOULD-NOT-BE-STORED"}}


def test_search_paginates_filters_non_videos_and_dedupes():
    pages = [Resp(200, {"items": [hit("aaaaaaaaaaa"), {"id": {"kind": "youtube#channel", "channelId": "UC1"}},
                                  hit("aaaaaaaaaaa"), hit("bbbbbbbbbbb")], "nextPageToken": "N2"}),
             Resp(200, {"items": [hit("ccccccccccc")]})]
    c = collector(pages)
    out = c.search_videos("Luna Jewelry review", max_results=10, max_pages=3)
    assert [v["video_id"] for v in out] == ["aaaaaaaaaaa", "bbbbbbbbbbb", "ccccccccccc"]
    assert set(out[0]) == {"video_id", "title", "description", "published_at"}  # no uploader info kept
    calls = c._session.calls
    assert len(calls) == 2 and c.search_calls == 2
    assert all(x["url"] == SEARCH_URL and x["params"]["type"] == "video" for x in calls)
    assert calls[0]["params"]["q"] == "Luna Jewelry review" and calls[1]["params"]["pageToken"] == "N2"
    assert all("key" not in x["params"] and KEY not in str(x["params"]) and KEY not in x["url"] for x in calls)
    assert all(x["headers"]["X-Goog-Api-Key"] == KEY for x in calls)


def test_search_respects_max_results_and_max_pages():
    c = collector([Resp(200, {"items": [hit(f"{i:011d}") for i in range(5)], "nextPageToken": "MORE"})])
    assert len(c.search_videos("x", max_results=3, max_pages=5)) == 3 and c.search_calls == 1
    pages = [Resp(200, {"items": [hit(f"p{i}aaaaaaaaa")], "nextPageToken": "n"}) for i in range(5)]
    c2 = collector(pages)
    assert len(c2.search_videos("x", max_results=50, max_pages=2)) == 2 and c2.search_calls == 2


def test_search_optional_params_and_validation():
    from datetime import datetime
    c = collector([Resp(200, {"items": []})])
    c.search_videos("x", published_after=datetime(2026, 1, 2), relevance_language="en", region_code="PK", order="date")
    p = c._session.calls[0]["params"]
    assert p["publishedAfter"] == "2026-01-02T00:00:00Z" and p["relevanceLanguage"] == "en" and p["regionCode"] == "PK"
    with pytest.raises(InvalidYouTubeSource):
        collector([]).search_videos("   ")
    with pytest.raises(ValueError):
        collector([]).search_videos("x", order="bogus")
    with pytest.raises(ConnectorNotConfigured):
        YouTubeCollector(session=FakeSession([])).search_videos("x")


def test_search_quota_error_raises_quota_exceeded():
    with pytest.raises(QuotaExceeded):
        collector([err(403, "quotaExceeded")]).search_videos("x")
