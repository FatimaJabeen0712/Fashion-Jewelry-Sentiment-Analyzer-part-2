"""Command-line agent: find YouTube videos for a brand, collect comments, save a CSV, analyze it.

Example (PowerShell):
    python run_agent.py --brand "Hiven Jewelry" --niche Jewelry --target 2000
    python run_agent.py --brand "Hiven Jewelry" --videos https://youtu.be/XXXXXXXXXXX --no-discover

Needs YOUTUBE_API_KEY (in .env or the environment). Output goes to data/runs/<time>_<brand>/.
Exit code: 0 complete, 2 below target / nothing analyzable, 1 error.
"""
from __future__ import annotations

import argparse
import sys

from config import DEFAULT_TARGET, NICHES
from src.agent import AgentConfig, run_agent
from src.schemas import PipelineError


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--brand", default="", help="brand name (also used to find and filter videos)")
    ap.add_argument("--niche", default="Both", choices=NICHES)
    ap.add_argument("--videos", nargs="*", default=[], help="YouTube video links or IDs to always include")
    ap.add_argument("--query", default="", help='search phrase (default: "<brand> review")')
    ap.add_argument("--no-discover", action="store_true", help="only use the videos given with --videos")
    ap.add_argument("--no-brand-filter", action="store_true",
                    help="keep discovered videos even if they do not mention the brand")
    ap.add_argument("--max-videos", type=int, default=10)
    ap.add_argument("--per-video", type=int, default=1000, help="max comments per video")
    ap.add_argument("--target", type=int, default=DEFAULT_TARGET, help="valid unique comments wanted")
    ap.add_argument("--language", choices=["any", "en"], default="any")
    ap.add_argument("--allow-small", action="store_true",
                    help="analyze even if below the target (the result carries a sample-size warning)")
    ap.add_argument("--out", default=None, help="output folder (default data/runs)")
    a = ap.parse_args(argv)

    cfg = AgentConfig(brand=a.brand, niche=a.niche, video_inputs=a.videos, search_query=a.query,
                      discover=not a.no_discover, require_brand_match=not a.no_brand_filter,
                      max_videos=a.max_videos, max_comments_per_video=a.per_video, target=a.target,
                      language_filter=a.language, allow_small=a.allow_small, out_dir=a.out)
    try:
        res = run_agent(cfg, progress=lambda step, detail: print(f"[agent] {step}" + (f": {detail}" if detail else "")))
    except PipelineError as exc:  # includes ConnectorNotConfigured (missing key) and bad input
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    print(f"\n{res.status.upper()}: {res.message}")
    print(f"Valid unique comments: {res.valid_comments:,} (target {res.target:,}) | "
          f"API calls: {res.api_calls} (search calls: {res.search_calls})")
    for e in res.errors:
        print(f"  ! {e}")
    if res.out_dir:
        print(f"Files saved in: {res.out_dir}")
    return 0 if res.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
