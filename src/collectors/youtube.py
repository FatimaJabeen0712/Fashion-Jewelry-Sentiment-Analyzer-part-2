"""YouTube Data API v3 connector (official ``commentThreads.list`` endpoint).

Collects *top-level* comments of public videos you name. Not collected: commenter
names, channel ids or profile links. Replies are not fetched in this version.

Quota: each ``commentThreads.list`` call costs 1 unit (a default project quota is
10,000 units/day), and returns up to 100 threads, so ~1,000 comments cost ~10
units. A brand URL or channel is NOT silently expanded: only video links/IDs are
accepted. The API key is sent in the ``X-Goog-Api-Key`` header, never in the URL,
and is never logged or included in error messages.
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from typing import Callable
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests

from config import SECRET_NAMES, get_secret
from src.collectors.base import BaseCollector, CollectionResult, ConnectorStatus
from src.schemas import ConnectorError, ConnectorNotConfigured, ensure_schema

logger = logging.getLogger(__name__)

API_BASE = "https://www.googleapis.com/youtube/v3"
COMMENT_THREADS_URL = f"{API_BASE}/commentThreads"
VIDEOS_URL = f"{API_BASE}/videos"
SEARCH_URL = f"{API_BASE}/search"
SEARCH_ORDERS = ("relevance", "date", "viewCount", "rating")
SEARCH_PAGE_SIZE = 50  # documented maximum for search maxResults
PAGE_SIZE = 100  # documented maximum for maxResults
COLLECTION_METHOD = "youtube_api_v3"

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YT_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com",
             "youtube-nocookie.com", "www.youtube-nocookie.com"}

SETUP_INSTRUCTIONS = (
    "YouTube collection needs a free YouTube Data API v3 key:\n"
    "1. Open https://console.cloud.google.com/ and create or select a project.\n"
    "2. Enable 'YouTube Data API v3' (https://console.cloud.google.com/apis/library/youtube.googleapis.com).\n"
    "3. APIs & Services > Credentials > Create credentials > API key.\n"
    "4. Restrict the key: under API restrictions choose 'YouTube Data API v3' only.\n"
    "5. Locally, put YOUTUBE_API_KEY=... in a .env file (never commit it). On Streamlit "
    "Community Cloud, add it under App settings > Secrets.\n"
    "CSV upload and demo mode keep working without a key."
)


class InvalidYouTubeSource(ConnectorError):
    """The supplied text is not a supported YouTube video link or ID."""


class QuotaExceeded(ConnectorError):
    """Daily API quota is exhausted."""


def parse_video_id(value: str) -> str:
    """Extract an 11-character video ID from a video URL or bare ID.

    Supports watch?v=, youtu.be/, /shorts/, /embed/, /live/, /v/ links. Channel,
    playlist, other-platform and generic website URLs are rejected rather than
    guessed at.
    """
    raw = (value or "").strip()
    if not raw:
        raise InvalidYouTubeSource("Enter a YouTube video link or video ID.")
    if _VIDEO_ID_RE.match(raw):
        return raw
    candidate = raw if "://" in raw else "https://" + raw
    try:
        parsed = urlparse(candidate)
    except ValueError as exc:
        raise InvalidYouTubeSource(f"'{raw[:60]}' is not a valid link.") from exc
    host = (parsed.hostname or "").lower()
    if host == "youtu.be":
        vid = parsed.path.strip("/").split("/")[0]
    elif host in _YT_HOSTS:
        parts = [p for p in parsed.path.split("/") if p]
        if parsed.path.rstrip("/") == "/watch":
            vid = (parse_qs(parsed.query).get("v") or [""])[0]
        elif len(parts) >= 2 and parts[0] in ("shorts", "embed", "live", "v"):
            vid = parts[1]
        elif parts and (parts[0] in ("channel", "c", "user", "playlist") or parts[0].startswith("@")):
            raise InvalidYouTubeSource(
                "Channel and playlist links are not expanded automatically. Open the videos you "
                "want analyzed and paste each video link (or ID) instead.")
        else:
            vid = ""
    else:
        raise InvalidYouTubeSource(
            f"'{raw[:60]}' is not a YouTube video link. Only YouTube video URLs/IDs are accepted "
            "here; use CSV upload for other platforms or websites.")
    if not _VIDEO_ID_RE.match(vid):
        raise InvalidYouTubeSource(f"Could not find a valid video ID in '{raw[:60]}'.")
    return vid


def parse_video_inputs(text: str) -> tuple[list[str], list[str]]:
    """Split pasted text (newline/comma/space separated) into (unique ids, error messages)."""
    ids: list[str] = []
    errors: list[str] = []
    for token in re.split(r"[\s,]+", text or ""):
        if not token:
            continue
        try:
            vid = parse_video_id(token)
        except InvalidYouTubeSource as exc:
            errors.append(str(exc))
            continue
        if vid not in ids:
            ids.append(vid)
    return ids, errors


def _error_reason(response) -> tuple[str, str]:
    """(reason, message) from an API error body without echoing the body itself."""
    try:
        err = response.json().get("error", {})
        reason = (err.get("errors") or [{}])[0].get("reason", "") or err.get("status", "")
        return str(reason), str(err.get("message", ""))[:200]
    except Exception:  # noqa: BLE001
        return "", ""


class YouTubeCollector(BaseCollector):
    name = "youtube"

    def __init__(self, api_key: str | None = None, session: requests.Session | None = None,
                 timeout: float = 15.0, max_retries: int = 3, backoff: float = 1.0,
                 sleep: Callable[[float], None] = time.sleep):
        self._api_key = api_key
        self._session = session or requests.Session()
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self._sleep = sleep
        self.requests_made = 0  # API calls made (each costs 1 unit)
        self.search_calls = 0   # search.list calls: they have their own, small daily limit

    # -- configuration ---------------------------------------------------
    def _key(self) -> str | None:
        return self._api_key or get_secret(SECRET_NAMES["youtube"])

    def status(self) -> ConnectorStatus:
        if self._key():
            return ConnectorStatus(self.name, True, "API key configured (value hidden).")
        return ConnectorStatus(self.name, False, "No YOUTUBE_API_KEY found.\n" + SETUP_INSTRUCTIONS)

    # -- HTTP --------------------------------------------------------------
    def _get(self, url: str, params: dict, key: str) -> dict:
        headers = {"X-Goog-Api-Key": key, "Accept": "application/json"}
        last = "network error"
        for attempt in range(self.max_retries + 1):
            if attempt:
                self._sleep(self.backoff * (2 ** (attempt - 1)))
            try:
                self.requests_made += 1
                resp = self._session.get(url, params=params, headers=headers, timeout=self.timeout)
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                last = "the request timed out or the network is unreachable"
                continue
            code = resp.status_code
            if code == 200:
                return resp.json()
            reason, _ = _error_reason(resp)
            if code in (500, 502, 503, 504) or (code == 429) or reason == "rateLimitExceeded":
                last = f"temporary API error (HTTP {code})"
                continue
            raise self._translate(code, reason)
        raise ConnectorError(f"YouTube API is not responding after {self.max_retries + 1} "
                             f"attempts ({last}). Try again later.")

    @staticmethod
    def _translate(code: int, reason: str) -> ConnectorError:
        if reason in ("quotaExceeded", "dailyLimitExceeded"):
            return QuotaExceeded("YouTube API daily quota is exhausted. Quota resets daily "
                                 "(Pacific time); request more quota in Google Cloud Console or "
                                 "try again tomorrow. Data collected so far is kept.")
        if reason == "commentsDisabled":
            return ConnectorError("Comments are disabled for this video.")
        if reason == "videoNotFound" or code == 404:
            return ConnectorError("Video not found - it may be deleted, private or the ID is wrong.")
        if reason in ("keyInvalid", "badRequest") and code == 400:
            return ConnectorError("The API key was rejected. Check YOUTUBE_API_KEY.")
        if reason in ("accessNotConfigured", "ipRefererBlocked", "forbidden") or code == 403:
            return ConnectorError("Access denied (HTTP 403). Make sure YouTube Data API v3 is "
                                  "enabled for the project and the key's restrictions allow it; "
                                  "the video may also restrict comments.")
        if code in (400, 401):
            return ConnectorError(f"The API rejected the request (HTTP {code}). Check the key and the video link.")
        return ConnectorError(f"Unexpected YouTube API response (HTTP {code}).")

    # -- collection ----------------------------------------------------------
    def _video_title(self, video_id: str, key: str) -> str:
        try:
            data = self._get(VIDEOS_URL, {"part": "snippet", "id": video_id}, key)
            items = data.get("items") or []
            return str(items[0]["snippet"].get("title", ""))[:200] if items else ""
        except (ConnectorError, KeyError, ValueError):
            return ""  # title is optional; never fail the collection over it

    def search_videos(self, query: str, max_results: int = 50, max_pages: int = 2,
                      order: str = "relevance", published_after: datetime | None = None,
                      relevance_language: str | None = None,
                      region_code: str | None = None) -> list[dict]:
        """Find public videos with the official ``search.list`` endpoint (``type=video``).

        Returns ``[{video_id, title, description, published_at}]`` (channel and playlist
        results are ignored; no uploader or commenter details are kept). Search has its own,
        much smaller daily call limit than other endpoints (the docs state 100 calls/day by
        default), so pages are capped by ``max_pages`` and ``search_calls`` is tracked.
        """
        key = self._key()
        if not key:
            raise ConnectorNotConfigured(SETUP_INSTRUCTIONS)
        q = (query or "").strip()
        if not q:
            raise InvalidYouTubeSource("Enter a search phrase (for example the brand name).")
        if order not in SEARCH_ORDERS:
            raise ValueError(f"order must be one of {SEARCH_ORDERS}")
        found: list[dict] = []
        seen: set[str] = set()
        token = None
        for _ in range(max(1, max_pages)):
            params = {"part": "snippet", "q": q, "type": "video", "order": order,
                      "maxResults": max(1, min(SEARCH_PAGE_SIZE, max_results - len(found)))}
            if token:
                params["pageToken"] = token
            if published_after:
                params["publishedAfter"] = _aware(published_after).strftime("%Y-%m-%dT%H:%M:%SZ")
            if relevance_language:
                params["relevanceLanguage"] = relevance_language
            if region_code:
                params["regionCode"] = region_code
            self.search_calls += 1
            data = self._get(SEARCH_URL, params, key)
            for item in data.get("items", []):
                vid = (item.get("id") or {}).get("videoId")
                if not vid or vid in seen or not _VIDEO_ID_RE.match(vid):
                    continue
                seen.add(vid)
                snip = item.get("snippet") or {}
                found.append({"video_id": vid, "title": str(snip.get("title", ""))[:200],
                              "description": str(snip.get("description", ""))[:500],
                              "published_at": str(snip.get("publishedAt", ""))})
                if len(found) >= max_results:
                    break
            token = data.get("nextPageToken")
            if len(found) >= max_results or not token:
                break
        logger.info("youtube search: results=%d search_calls=%d", len(found), self.search_calls)
        return found

    def collect(self, video_inputs: list[str] | str | None = None, brand: str = "",
                max_comments_per_video: int = 1000, max_pages_per_video: int = 50,
                order: str = "time", published_after: datetime | None = None,
                published_before: datetime | None = None, fetch_titles: bool = True,
                progress: Callable[[str, int], None] | None = None, **_: object) -> CollectionResult:
        key = self._key()
        if not key:
            raise ConnectorNotConfigured(SETUP_INSTRUCTIONS)
        if order not in ("time", "relevance"):
            raise ValueError("order must be 'time' or 'relevance'")
        if isinstance(video_inputs, str):
            video_inputs = [video_inputs]
        ids: list[str] = []
        errors: list[str] = []
        for item in video_inputs or []:
            new_ids, errs = parse_video_inputs(item)
            ids += [i for i in new_ids if i not in ids]
            errors += errs
        if not ids:
            raise InvalidYouTubeSource(errors[0] if errors else "Enter at least one YouTube video link.")

        records: list[dict] = []
        seen: set[str] = set()
        per_source: dict[str, int] = {}
        for n, vid in enumerate(ids):
            try:
                got = self._collect_video(vid, key, brand, max_comments_per_video,
                                          max_pages_per_video, order, published_after,
                                          published_before, fetch_titles, seen, records)
                per_source[vid] = got
                if progress:
                    progress(vid, got)
            except QuotaExceeded as exc:
                per_source.setdefault(vid, 0)
                skipped = ids[n + 1:]
                errors.append(f"{vid}: {exc}" + (f" Not collected: {', '.join(skipped)}." if skipped else ""))
                break
            except ConnectorError as exc:
                per_source[vid] = 0
                errors.append(f"{vid}: {exc}")
        df = ensure_schema(pd.DataFrame(records), default_source="youtube",
                           collection_method=COLLECTION_METHOD)
        logger.info("youtube collect: videos=%d comments=%d api_calls=%d errors=%d",
                    len(ids), len(df), self.requests_made, len(errors))
        return CollectionResult(df=df, per_source=per_source, errors=errors)

    def _collect_video(self, vid, key, brand, limit, max_pages, order, after, before,
                       fetch_titles, seen, records) -> int:
        title = self._video_title(vid, key) if fetch_titles else ""
        page_token = None
        got = 0
        for _ in range(max_pages):
            params = {"part": "snippet", "videoId": vid, "maxResults": min(PAGE_SIZE, limit - got),
                      "order": order, "textFormat": "plainText"}
            if page_token:
                params["pageToken"] = page_token
            data = self._get(COMMENT_THREADS_URL, params, key)
            stop = False
            for thread in data.get("items", []):
                top = (thread.get("snippet") or {}).get("topLevelComment") or {}
                snip = top.get("snippet") or {}
                cid = top.get("id") or thread.get("id")
                text = snip.get("textOriginal") or snip.get("textDisplay") or ""
                if not cid or cid in seen:
                    continue
                published = pd.to_datetime(snip.get("publishedAt"), errors="coerce", utc=True)
                if pd.notna(published):
                    if before and published > _aware(before):
                        continue
                    if after and published < _aware(after):
                        if order == "time":  # newest first: everything further is older
                            stop = True
                            break
                        continue
                seen.add(cid)
                records.append({
                    "comment_id": f"yt-{cid}", "text": text,
                    "created_at": published, "source": "youtube", "brand": brand,
                    "post_url": f"https://www.youtube.com/watch?v={vid}",
                    "like_count": snip.get("likeCount", 0),
                    "reply_count": (thread.get("snippet") or {}).get("totalReplyCount", 0),
                    "collection_method": COLLECTION_METHOD,
                    "video_id": vid, "video_title": title})
                got += 1
                if got >= limit:
                    stop = True
                    break
            page_token = data.get("nextPageToken")
            if stop or not page_token:
                break
        return got


def _aware(dt: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(dt)
    return ts.tz_localize(timezone.utc) if ts.tzinfo is None else ts.tz_convert(timezone.utc)
