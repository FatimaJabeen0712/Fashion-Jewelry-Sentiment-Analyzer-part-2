"""End-to-end agent: find videos -> collect comments -> save CSV -> reload that CSV -> analyze.

One call does the whole job. Rules it follows (all from the project brief):

* Official YouTube Data API only. Videos you list are always collected; extra videos are
  *discovered* with ``search.list`` and kept only if the title/description mentions the brand
  (switch off with ``require_brand_match=False``) so unrelated comments don't pollute results.
* Collection continues video by video until the number of *valid unique comments after
  cleaning* reaches the target, the video budget is used up, or quota runs out. Nothing is
  padded or fabricated.
* The raw comments are written to a CSV, and the analysis is run on the **reloaded CSV**
  (so the file on disk is exactly what was analyzed).
* If the result is below the target the agent saves the CSV but only analyzes when
  ``allow_small=True`` was chosen up front; the result then carries the sample-size warning.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from config import DEFAULT_TARGET, RUNS_DIR
from src.collectors.csv_import import load_exported_csv
from src.collectors.youtube import SEARCH_PAGE_SIZE, YouTubeCollector, parse_video_inputs
from src.pipeline import AnalysisResult, run_analysis
from src.preprocessing import preprocess
from src.reporting import all_exports, to_csv_bytes
from src.schemas import ConnectorError, ConnectorNotConfigured, PipelineError
from src.security import RAW_EXPORT_COLUMNS, safe_object_name, scrub_secrets

logger = logging.getLogger(__name__)

GENERIC_WORDS = frozenset(
    "jewelry jewellery jewels jewelers jewellers clothing clothes fashion brand official store shop "
    "boutique wear apparel the and by co company review reviews collection".split())

ProgressFn = Callable[[str, str], None]


@dataclass
class AgentConfig:
    brand: str = ""
    niche: str = "Both"
    video_inputs: list[str] = field(default_factory=list)  # links / IDs that are always collected
    search_query: str = ""            # default: "<brand> review"
    discover: bool = True             # find extra videos with YouTube search
    require_brand_match: bool = True  # keep discovered videos only if they mention the brand
    max_videos: int = 10
    max_comments_per_video: int = 1000
    search_pages: int = 2             # search has its own small daily limit: keep this low
    target: int = DEFAULT_TARGET
    language_filter: str = "any"
    allow_small: bool = False         # analyze even if below target (result carries a warning)
    out_dir: Path | None = None       # default: config.RUNS_DIR


@dataclass
class VideoRun:
    video_id: str
    title: str
    origin: str          # "provided" or "search"
    comments_fetched: int
    error: str = ""


@dataclass
class AgentResult:
    status: str          # complete | below_target | no_data | no_valid_comments
    message: str
    videos: list[VideoRun] = field(default_factory=list)
    skipped_videos: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    out_dir: Path | None = None
    raw_csv_path: Path | None = None
    files: list[Path] = field(default_factory=list)
    reloaded: pd.DataFrame | None = None       # the data read back from the saved CSV
    analysis: AnalysisResult | None = None
    valid_comments: int = 0
    target: int = DEFAULT_TARGET
    api_calls: int = 0
    search_calls: int = 0

    @property
    def ok(self) -> bool:
        return self.status == "complete"


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).casefold()).strip()


def brand_matches(brand: str, text: str) -> bool:
    """True if ``text`` mentions the brand: the full name, or every distinctive word of it.

    Generic words ("jewelry", "shop", ...) and 1-2 letter words are ignored when checking
    individual words, so "Hiven Jewelry" matches a title containing "Hiven".
    """
    b = _norm(brand)
    if not b:
        return True
    t = f" {_norm(text)} "
    if f" {b} " in t:
        return True
    core = [w for w in b.split() if w not in GENERIC_WORDS and len(w) >= 3]
    return bool(core) and all(f" {w} " in t for w in core)


def _slug(text: str) -> str:
    return safe_object_name(text or "run", extension="")[:-1] or "run"


def _write_log(run_dir: Path, cfg: AgentConfig, result: AgentResult, started: str) -> Path:
    payload = {
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": result.status, "message": result.message,
        "config": {"brand": cfg.brand, "niche": cfg.niche, "search_query": cfg.search_query,
                   "discover": cfg.discover, "require_brand_match": cfg.require_brand_match,
                   "max_videos": cfg.max_videos, "max_comments_per_video": cfg.max_comments_per_video,
                   "target": cfg.target, "language_filter": cfg.language_filter,
                   "allow_small": cfg.allow_small},
        "videos": [v.__dict__ for v in result.videos],
        "skipped_as_not_matching_brand": [{"video_id": v["video_id"], "title": v["title"]}
                                          for v in result.skipped_videos],
        "errors": result.errors, "valid_comments": result.valid_comments,
        "api_calls": result.api_calls, "search_calls": result.search_calls, "steps": result.steps,
        "note": "No API keys or commenter identities are stored in this file.",
    }
    path = run_dir / "agent_log.json"
    path.write_text(scrub_secrets(json.dumps(payload, ensure_ascii=False, indent=2)), encoding="utf-8")
    return path


def run_agent(cfg: AgentConfig, collector=None, progress: ProgressFn | None = None) -> AgentResult:
    """Run the whole pipeline. Raises ConnectorNotConfigured if no YouTube key is available,
    PipelineError for unusable inputs; everything else is reported in the result."""
    if cfg.max_videos < 1 or cfg.max_comments_per_video < 1 or cfg.target < 1:
        raise PipelineError("max_videos, max_comments_per_video and target must be at least 1.")
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    steps: list[str] = []

    def log(step: str, detail: str = "") -> None:
        steps.append(f"{step}: {detail}" if detail else step)
        if progress:
            progress(step, detail)

    collector = collector or YouTubeCollector()
    status = collector.status()
    if not status.available:
        raise ConnectorNotConfigured(status.message)

    brand = cfg.brand.strip()
    provided, errors = parse_video_inputs("\n".join(cfg.video_inputs))
    query = cfg.search_query.strip() or (f"{brand} review" if brand else "")
    if not provided and not (cfg.discover and query):
        raise PipelineError("Enter a brand name or search phrase, or at least one valid YouTube video link.")

    candidates: list[dict] = [{"video_id": v, "title": "", "origin": "provided"} for v in provided]
    skipped: list[dict] = []
    if cfg.discover and query:
        log("Searching YouTube", query)
        try:
            wanted = min(SEARCH_PAGE_SIZE * max(cfg.search_pages, 1), max(cfg.max_videos * 3, 10))
            hits = collector.search_videos(query, max_results=wanted, max_pages=cfg.search_pages)
        except ConnectorError as exc:  # quota, key, network: keep going with provided links
            hits = []
            errors.append(f"Video search failed: {exc}")
        known = set(provided)
        kept = 0
        for h in hits:
            if h["video_id"] in known:
                continue
            if cfg.require_brand_match and brand and not brand_matches(brand, f"{h['title']} {h['description']}"):
                skipped.append(h)
                continue
            known.add(h["video_id"])
            candidates.append({"video_id": h["video_id"], "title": h["title"], "origin": "search"})
            kept += 1
        log("Videos found", f"{kept} relevant" + (f", {len(skipped)} skipped (did not mention '{brand}')" if skipped else ""))
    candidates = candidates[: max(cfg.max_videos, len(provided))]

    result = AgentResult(status="no_data", message="", skipped_videos=skipped, errors=errors,
                         steps=steps, target=cfg.target)
    if not candidates:
        result.message = ("No videos to collect from. Search found nothing that mentions the brand; "
                          "paste video links or turn off the brand-match filter.")
        _finish_counts(result, collector)
        return result

    frames: list[pd.DataFrame] = []
    videos: list[VideoRun] = []
    for i, cand in enumerate(candidates, 1):
        log(f"Collecting comments {i}/{len(candidates)}", f"{cand['video_id']} {cand['title'][:60]}".strip())
        try:
            res = collector.collect(video_inputs=[cand["video_id"]], brand=brand,
                                    max_comments_per_video=cfg.max_comments_per_video, fetch_titles=False)
        except ConnectorError as exc:
            videos.append(VideoRun(cand["video_id"], cand["title"], cand["origin"], 0, str(exc)))
            errors.append(f"{cand['video_id']}: {exc}")
            continue
        n = len(res.df)
        if n:
            df = res.df.copy()
            df["video_title"] = cand["title"]
            frames.append(df)
        videos.append(VideoRun(cand["video_id"], cand["title"], cand["origin"], n, "; ".join(res.errors)))
        errors.extend(res.errors)
        if any("quota" in e.lower() for e in res.errors):
            log("Stopped early", "YouTube API quota exhausted")
            break
        fetched = sum(len(f) for f in frames)
        if fetched >= cfg.target:  # only then can the target possibly be met
            valid = preprocess(pd.concat(frames, ignore_index=True), cfg.target, cfg.language_filter).report.valid
            log("Progress", f"{valid:,} valid of {cfg.target:,} target")
            if valid >= cfg.target:
                log("Target reached", "no more videos needed")
                break
    result.videos = videos
    result.errors = errors
    if not frames:
        result.status, result.message = "no_data", "No comments could be collected. See the errors below."
        _finish_counts(result, collector)
        return result

    raw = pd.concat(frames, ignore_index=True)
    run_dir = Path(cfg.out_dir or RUNS_DIR) / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{_slug(brand)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / "raw_comments.csv"
    raw_path.write_bytes(to_csv_bytes(raw[[c for c in RAW_EXPORT_COLUMNS if c in raw.columns]]))
    log("Saved CSV", f"{len(raw):,} comments -> {raw_path}")

    reloaded, meta = load_exported_csv(raw_path)   # the analysis reads the file that was just saved
    log("Reloaded CSV", f"{meta['rows']:,} rows")
    analysis = run_analysis(reloaded, cfg.niche, brand, cfg.target, cfg.language_filter)
    result.out_dir, result.raw_csv_path, result.reloaded = run_dir, raw_path, reloaded
    result.valid_comments = analysis.report.real_valid
    result.files = [raw_path]

    if analysis.n == 0:
        result.status = "no_valid_comments"
        result.message = "Every collected comment was removed as empty, spam, duplicate or by the language filter."
    elif analysis.small_sample and not cfg.allow_small:
        result.status = "below_target"
        result.message = (f"Collected {analysis.report.real_valid:,} valid unique comments, below the target of "
                          f"{cfg.target:,}. The CSV is saved but not analyzed: add more video links, raise the "
                          "video budget, or re-run allowing a smaller sample.")
    else:
        result.analysis = analysis
        result.status = "complete"
        for name, data in all_exports(analysis).items():
            path = run_dir / name
            path.write_bytes(data)
            result.files.append(path)
        log("Analysis complete", f"{analysis.n:,} valid comments")
        result.message = (f"Done: {analysis.n:,} valid unique comments analyzed." if not analysis.small_sample else
                          f"Done with a sample-size warning: {analysis.n:,} valid comments vs a target of "
                          f"{cfg.target:,}. Treat findings as directional.")
    _finish_counts(result, collector)
    result.files.append(_write_log(run_dir, cfg, result, started))
    return result


def _finish_counts(result: AgentResult, collector) -> None:
    result.api_calls = int(getattr(collector, "requests_made", 0))
    result.search_calls = int(getattr(collector, "search_calls", 0))
