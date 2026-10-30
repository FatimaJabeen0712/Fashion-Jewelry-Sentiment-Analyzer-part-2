"""Fashion & Jewelry Customer Voice Intelligence - Streamlit app.

Run locally:  streamlit run app.py
Every control is functional or explicitly labeled unavailable / not configured.
"""
from __future__ import annotations

import math
import re
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (APP_NAME, APP_VERSION, DEFAULT_TARGET, LANGUAGE_FILTERS, NICHES, RUNS_DIR,
                    SENTIMENT_MODEL_VERSION)
from src.aspect_analysis import top_likes
from src.collectors.csv_import import (EXAMPLE_CSV, import_csv, load_exported_csv, read_csv_raw,
                                       suggest_mapping)
from src.collectors.demo import DEMO_LABEL, generate_demo_dataset, is_demo
from src.collectors.meta import MetaCollector
from src.collectors.tiktok_research import TikTokResearchCollector
from src.collectors.youtube import SETUP_INSTRUCTIONS, YouTubeCollector, parse_video_inputs
from src.evaluation import TEMPLATE_CSV, evaluate_sentiment
from src.preprocessing import preprocess
from src.product_insights import DISCLAIMER
from src.recommendations import HYPOTHESIS_LABEL
from src import storage
from src.agent import AgentConfig, run_agent
from src.pipeline import run_analysis
from src.reporting import (all_exports, competitor_comparison, sentiment_by_source,
                           sentiment_over_time, to_csv_bytes)
from src.schemas import ConnectorError, PipelineError

st.set_page_config(page_title=APP_NAME, page_icon="\U0001F48E", layout="wide")

SENT_COLORS = {"positive": "#2E7D6B", "neutral": "#B8B0A8", "negative": "#C4553B"}
ROSE, PLUM = "#B76E79", "#3B2A35"
PLOT_LAYOUT = dict(template="plotly_white", margin=dict(l=10, r=10, t=40, b=10),
                   font=dict(color=PLUM))

st.markdown(f"""
<style>
.block-container {{padding-top: 1.4rem;}}
.fjvi-hero {{background: linear-gradient(120deg, #F7E7CE 0%, #F3D5D8 60%, #E9C8CF 100%);
  border-radius: 14px; padding: 1.1rem 1.4rem; margin-bottom: 1rem; color: {PLUM};}}
.fjvi-hero h1 {{font-size: 1.7rem; margin: 0 0 .2rem 0; color: {PLUM};}}
.fjvi-hero p {{margin: 0; opacity: .85;}}
div[data-testid="stMetric"] {{background: #FFF8F3; border: 1px solid #EBD9D0; border-radius: 12px;
  padding: .6rem .8rem;}}
@media (prefers-color-scheme: dark) {{ div[data-testid="stMetric"] {{background: #2A2024; border-color:#4A3A40;}} }}
.fjvi-demo {{background:#C4553B; color:white; padding:.5rem .9rem; border-radius:8px; font-weight:600;}}
</style>
<div class="fjvi-hero"><h1>\U0001F48E {APP_NAME}</h1>
<p>Turn customer comments into evidence-backed product feedback. A feedback snapshot, not a verdict
- and mentions are interest signals, not sales.</p></div>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
def init_state() -> None:
    defaults = {"raw": None, "sources": [], "bundle": None, "competitor": None,
                "collect_log": None, "data_version": 0, "notice": None, "agent_result": None}
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


init_state()
S = st.session_state


def bump() -> None:
    S.data_version += 1
    S.bundle = None


def add_raw(new_df: pd.DataFrame, sources: list[dict], replace: bool = False) -> None:
    """Add rows to the working dataset. Demo rows are never mixed with real rows."""
    cur = S.raw
    demo_new = is_demo(new_df)
    if demo_new:
        replace = True  # a demo load always replaces everything
    elif cur is not None and is_demo(cur):
        cur = None  # real data replaces demo data
        S.sources = []
        S.notice = "Synthetic demo rows were removed because real data was added."
    S.raw = new_df.copy() if (replace or cur is None) else pd.concat([cur, new_df], ignore_index=True)
    S.sources = sources if replace else S.sources + sources
    bump()


def clear_data() -> None:
    S.raw, S.sources, S.collect_log, S.agent_result = None, [], None, None
    bump()


@st.cache_data(show_spinner=False, ttl=3600, max_entries=8)  # bounded retention of comment data in memory
def cached_preprocess(raw: pd.DataFrame, target: int, language_filter: str):
    """Preprocessing is re-evaluated on every rerun (all tabs render), so cache it."""
    return preprocess(raw, target=target, language_filter=language_filter)


def get_files(b) -> dict[str, bytes]:
    """Export files for the current analysis, built once per analysis."""
    if S.get("_files_for") != id(b):
        S["_files"], S["_files_for"] = all_exports(b), id(b)
    return S["_files"]


def mime_for(name: str) -> str:
    return "application/json" if name.endswith(".json") else "text/csv"


def valid_url(u: str) -> bool:
    return bool(re.match(r"^https?://[^\s/]+\.[^\s/]+", u.strip()))


# ----------------------------------------------------------------------------
# Sidebar: project inputs
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Project setup")
    niche = st.selectbox("Niche", NICHES, index=2)
    brand = st.text_input("Brand name", placeholder="e.g. Hiven Jewelry")
    website = st.text_input("Brand website (optional)", placeholder="https://example.com")
    if website and not valid_url(website):
        st.warning("That does not look like a valid http(s) URL.")
    st.caption("Website and product URLs are stored as labels only. This app never scrapes "
               "websites - a URL is not permission to collect from it.")
    product_urls = st.text_area("Product URL(s) (optional, labels only)", height=68)
    bad_urls = [u for u in product_urls.split() if u and not valid_url(u)]
    if bad_urls:
        st.warning(f"Ignoring {len(bad_urls)} invalid URL(s).")
    target = st.number_input("Minimum valid comments - TARGET", min_value=100, max_value=100000,
                             value=DEFAULT_TARGET, step=100,
                             help="A goal measured after cleaning, de-duplication and filters. "
                                  "It is not guaranteed and never met with fabricated data.")
    language_filter = st.selectbox("Language filter", list(LANGUAGE_FILTERS),
                                   format_func=LANGUAGE_FILTERS.get)
    st.caption("Analysis is tuned for English. Multilingual accuracy (incl. Roman Urdu) is not "
               "validated; language detection is approximate.")

    with st.expander("Competitor (optional)"):
        comp_name = st.text_input("Competitor name")
        comp_file = st.file_uploader("Competitor CSV", type=["csv"], key="comp_csv")
        if st.button("Load competitor data", disabled=not (comp_name and comp_file)):
            try:
                comp_raw = import_csv(comp_file.getvalue(), brand=comp_name, default_source="competitor_csv")
                comp_res = run_analysis(comp_raw, niche, comp_name, int(target), language_filter)
                if comp_res.n == 0:
                    st.error("No valid competitor comments remain after preprocessing.")
                else:
                    S.competitor = (comp_name, comp_res)
                    st.success(f"Loaded {comp_res.n:,} valid competitor comments.")
            except PipelineError as exc:
                st.error(str(exc))
        if S.competitor:
            st.caption(f"Loaded: {S.competitor[0]}")
    with st.expander("Responsible use"):
        st.markdown(
            "- Collect only data you are permitted to use; respect each platform's terms.\n"
            "- Only necessary fields are kept; commenter names/profiles are not stored.\n"
            "- Honor retention limits and deletion requests; don't re-share exports publicly.\n"
            "- Sentiment is a proxy for interest, **not sales**.")
    st.caption(f"v{APP_VERSION}")

tabs = st.tabs(["Collect Data", "Dashboard", "Product & Aspect Insights", "Recommendations",
                "Data Explorer", "CSV Library / Reuse", "Setup & Methodology"])


def get_bundle():
    return S.bundle


def banners(b) -> None:
    if b.contains_demo:
        st.markdown(f'<div class="fjvi-demo">{DEMO_LABEL}</div>', unsafe_allow_html=True)
    if b.small_sample and not b.contains_demo:
        st.warning(f"Sample-size warning: {b.report.real_valid:,} valid comments analyzed vs a target "
                   f"of {b.target:,}. Treat findings as directional.")
    elif b.contains_demo:
        st.caption("Demo rows are never counted toward the target.")


def fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def empty_state() -> None:
    st.info("No analysis yet. Go to **Collect Data**, add comments (YouTube, CSV or demo), "
            "then click **Preprocess & analyze**.")


# ----------------------------------------------------------------------------
# Tab 1: Collect Data
# ----------------------------------------------------------------------------
with tabs[0]:
    if S.notice:
        st.info(S.notice)
        S.notice = None

    # ---- automatic agent ---------------------------------------------------------------
    st.subheader("Automatic agent")
    st.caption("One click: find YouTube videos about the brand, collect their comments, save a CSV, then "
               "analyze that CSV. Official YouTube API only (Instagram, Facebook and TikTok cannot be "
               "collected automatically - use CSV upload for those).")
    agent_status = YouTubeCollector().status()
    if not agent_status.available:
        st.warning("No YouTube API key configured - the agent needs one. CSV and demo modes below still work.")
        with st.expander("How to get a YouTube Data API v3 key"):
            st.code(SETUP_INSTRUCTIONS, language="text")
    ag1, ag2 = st.columns(2)
    agent_query = ag1.text_input("Search phrase (optional)", key="agent_query",
                                 placeholder="default: '<brand name> review'")
    agent_links = ag2.text_area("Extra YouTube links (optional, one per line)", height=68, key="agent_links")
    ag3, ag4, ag5 = st.columns(3)
    agent_max_videos = ag3.number_input("Max videos to use", 1, 50, 10, key="agent_max_videos")
    agent_per_video = ag4.number_input("Max comments per video", 100, 20000, 1000, step=100, key="agent_per_video")
    agent_match = ag5.checkbox("Only videos that mention the brand", value=True, key="agent_match",
                               help="Search results are filtered by title/description so unrelated videos "
                                    "do not pollute the analysis.")
    agent_small = st.checkbox("If fewer than the target are found, analyze the smaller sample anyway "
                              "(results carry a sample-size warning)", key="agent_small")
    can_run = agent_status.available and bool(brand.strip() or agent_query.strip() or agent_links.strip())
    if agent_status.available and not can_run:
        st.info("Enter a brand name in the sidebar (or a search phrase / video links) to start.")
    if st.button("Run agent: collect \u2192 export CSV \u2192 analyze", type="primary", disabled=not can_run,
                 key="agent_run"):
        cfg = AgentConfig(brand=brand, niche=niche, video_inputs=agent_links.split(), search_query=agent_query,
                          require_brand_match=agent_match, max_videos=int(agent_max_videos),
                          max_comments_per_video=int(agent_per_video), target=int(target),
                          language_filter=language_filter, allow_small=agent_small, out_dir=RUNS_DIR)
        agent_res = None
        with st.status("Agent running...", expanded=True) as run_status:
            try:
                agent_res = run_agent(cfg, progress=lambda step, detail: run_status.write(f"**{step}** {detail}"))
                run_status.update(label=agent_res.message, state="complete" if agent_res.ok else "error")
            except PipelineError as exc:
                run_status.update(label="The agent could not run", state="error")
                st.error(str(exc))
        if agent_res is not None:
            S.agent_result = agent_res
            if agent_res.reloaded is not None:  # the data read back from the saved CSV
                add_raw(agent_res.reloaded, [{"platform": "youtube", "identifier": v.video_id,
                                              "count": v.comments_fetched}
                                             for v in agent_res.videos if v.comments_fetched], replace=True)
                if agent_res.analysis is not None:
                    S.bundle = agent_res.analysis
    ar = S.agent_result
    if ar is not None:
        (st.success if ar.ok else st.warning)(ar.message)
        k = st.columns(4)
        k[0].metric("Videos used", sum(1 for v in ar.videos if v.comments_fetched))
        k[1].metric("Valid unique comments", f"{ar.valid_comments:,}")
        k[2].metric("Target (goal)", f"{ar.target:,}")
        k[3].metric("API calls (search)", f"{ar.api_calls} ({ar.search_calls})")
        if ar.videos:
            st.dataframe(pd.DataFrame([{"video_id": v.video_id, "title": v.title, "found via": v.origin,
                                        "comments": v.comments_fetched, "note": v.error} for v in ar.videos]),
                         hide_index=True)
        if ar.skipped_videos:
            with st.expander(f"{len(ar.skipped_videos)} search result(s) skipped - did not mention the brand"):
                st.dataframe(pd.DataFrame(ar.skipped_videos)[["video_id", "title"]], hide_index=True)
        for e in ar.errors:
            st.error(e)
        if ar.raw_csv_path is not None and ar.raw_csv_path.exists():
            st.caption(f"CSV saved on the machine running the app: {ar.raw_csv_path} (temporary on Streamlit "
                       "Community Cloud - use the download button).")
            st.download_button("Download the collected comments (CSV)", ar.raw_csv_path.read_bytes(),
                               "raw_comments.csv", "text/csv", key="agent_dl_raw")
        if ar.ok:
            st.success("Analysis is ready - open the **Dashboard** tab.")
        elif ar.status == "below_target":
            st.info("The collected comments are loaded below. Add more videos and run the agent again, or "
                    "tick the sample-size box in step 3 to analyze them as a small sample.")
    st.divider()
    st.markdown("#### Manual mode (advanced)")

    st.subheader("1 - Connectors")
    yt, meta_c, tt = YouTubeCollector(), MetaCollector(), TikTokResearchCollector()
    rows = []
    for label, c in (("YouTube Data API v3", yt), ("Facebook / Instagram (Meta)", meta_c),
                     ("TikTok Research API", tt)):
        stt = c.status()
        rows.append({"Connector": label, "Status": "Configured" if stt.available else "Not available / not configured",
                     "Details": stt.message.split("\n")[0]})
    rows += [{"Connector": "CSV upload", "Status": "Always available", "Details": "Your own exports or reviews"},
             {"Connector": "Demo dataset", "Status": "Always available", "Details": "Synthetic, clearly labeled"}]
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    st.subheader("2 - Add comments")
    mode = st.radio("Data source", ["YouTube (official API)", "CSV upload", "Demo dataset (synthetic)"],
                    horizontal=True)

    if mode.startswith("YouTube"):
        if not yt.status().available:
            st.warning("No YouTube API key configured - collection is disabled, but you can still "
                       "validate links and use CSV / demo mode.")
            with st.expander("How to get a YouTube Data API v3 key", expanded=True):
                st.code(SETUP_INSTRUCTIONS, language="text")
        links = st.text_area("YouTube video links or IDs (one per line)", height=110,
                             placeholder="https://www.youtube.com/watch?v=...")
        ids, errs = parse_video_inputs(links)
        for e in errs:
            st.error(e)
        if ids:
            st.success(f"{len(ids)} valid video(s): {', '.join(ids)}")
        c1, c2, c3 = st.columns(3)
        max_per = c1.number_input("Max comments per video", 100, 20000, 1000, step=100)
        order = c2.selectbox("Order", ["time", "relevance"])
        use_dates = c3.checkbox("Filter by date range")
        after = before = None
        if use_dates:
            d1, d2 = st.columns(2)
            after = datetime.combine(d1.date_input("From", value=datetime(2025, 1, 1)), datetime.min.time())
            before = datetime.combine(d2.date_input("To", value=datetime.now()), datetime.max.time())
        calls = len(ids) * (math.ceil(max_per / 100) + 1)
        st.caption(f"Estimated cost: up to ~{calls} API calls (~{calls} quota units; the default daily "
                   "quota is 10,000). A video's comment count may be lower than the limit.")
        if st.button("Collect comments", type="primary", disabled=not (ids and yt.status().available)):
            with st.spinner("Collecting from the YouTube Data API..."):
                try:
                    res = YouTubeCollector().collect(
                        video_inputs=ids, brand=brand, max_comments_per_video=int(max_per), order=order,
                        published_after=after, published_before=before)
                    S.collect_log = {"per_source": res.per_source, "errors": res.errors}
                    if len(res.df):
                        add_raw(res.df, [{"platform": "youtube", "identifier": v, "count": n}
                                         for v, n in res.per_source.items()])
                        st.success(f"Collected {len(res.df):,} comments from {len(ids)} video(s).")
                    else:
                        st.warning("No comments were collected - see the messages below.")
                except ConnectorError as exc:
                    st.error(str(exc))
        if S.collect_log:
            st.write("**Per-video results**")
            st.dataframe(pd.DataFrame({"video_id": list(S.collect_log["per_source"]),
                                       "comments_fetched": list(S.collect_log["per_source"].values())}),
                         hide_index=True)
            for e in S.collect_log["errors"]:
                st.error(e)

    elif mode.startswith("CSV"):
        st.caption("Upload a CSV of comments or reviews you are permitted to use. Need an example?")
        st.download_button("Download example CSV", EXAMPLE_CSV.encode("utf-8-sig"),
                           "example_comments.csv", "text/csv")
        up = st.file_uploader("CSV file", type=["csv"], key="main_csv")
        if up is not None:
            try:
                raw_up = read_csv_raw(up.getvalue())
                st.write(f"Preview ({len(raw_up):,} rows, {len(raw_up.columns)} columns)")
                st.dataframe(raw_up.head(8), hide_index=True)
                sug = suggest_mapping(list(raw_up.columns))
                st.write("**Column mapping**")
                opts = ["(none)"] + list(raw_up.columns)
                labels = {"text": "Comment text *", "created_at": "Date", "source": "Source / platform",
                          "product_name": "Product", "like_count": "Likes", "reply_count": "Replies",
                          "rating": "Rating", "post_url": "Post / video URL", "comment_id": "Comment ID"}
                mapping = {}
                cols = st.columns(3)
                for i, (field_name, label) in enumerate(labels.items()):
                    default = opts.index(sug[field_name]) if sug.get(field_name) in opts else 0
                    choice = cols[i % 3].selectbox(label, opts, index=default, key=f"map_{field_name}")
                    mapping[field_name] = None if choice == "(none)" else choice
                src_label = st.text_input("Source label (used if no source column)", "csv_upload")
                if mapping["text"] is None:
                    st.error("Choose the column that contains the comment text.")
                if st.button("Import CSV", type="primary", disabled=mapping["text"] is None):
                    try:
                        df_in = import_csv(raw_up, mapping, brand=brand, default_source=src_label or "csv_upload")
                        add_raw(df_in, [{"platform": "csv", "identifier": up.name, "count": len(df_in)}])
                        st.success(f"Imported {len(df_in):,} rows.")
                    except PipelineError as exc:
                        st.error(str(exc))
            except PipelineError as exc:
                st.error(str(exc))
        else:
            st.info("Choose a CSV file to begin.")

    else:
        st.markdown(f'<div class="fjvi-demo">{DEMO_LABEL}</div>', unsafe_allow_html=True)
        st.caption("Generated from templates for exploring the app. It replaces any loaded data, is never "
                   "mixed with real data, and is never counted toward the target.")
        n_demo = st.slider("Rows to generate", 200, 3000, 800, step=100)
        if st.button("Load synthetic demo data", type="primary"):
            demo_df = generate_demo_dataset(niche, n_demo, brand=brand or "Demo Brand")
            add_raw(demo_df, [{"platform": "demo_synthetic", "identifier": "synthetic",
                               "count": len(demo_df)}], replace=True)
            st.success(f"Loaded {n_demo:,} synthetic rows (labeled DEMO).")

    # ---- dataset status + preprocessing --------------------------------------------
    st.divider()
    st.subheader("3 - Check, clean and analyze")
    if S.raw is None or S.raw.empty:
        st.info("No data loaded yet.")
    else:
        raw = S.raw
        if is_demo(raw):
            st.markdown(f'<div class="fjvi-demo">{DEMO_LABEL}</div>', unsafe_allow_html=True)
        st.write(f"**{len(raw):,} rows loaded** from {len(S.sources)} source(s)")
        st.dataframe(pd.DataFrame(S.sources), hide_index=True)
        pre = cached_preprocess(raw, int(target), language_filter)
        r = pre.report
        funnel = pd.DataFrame(r.funnel(), columns=["Stage", "Rows"])
        c1, c2 = st.columns([1, 1])
        c1.dataframe(funnel, hide_index=True)
        c2.metric("Valid unique comments", f"{r.real_valid:,}" if not r.demo_valid else f"{r.valid:,} (demo)")
        c2.metric("Target (goal, not guaranteed)", f"{r.target:,}")
        c2.progress(min(r.real_valid / r.target, 1.0) if r.target else 0.0)
        (st.success if r.target_met else st.warning)(r.status_message())
        st.caption(f"Removed: {r.empty_removed:,} empty, {r.spam_removed:,} spam, "
                   f"{r.duplicates_removed:,} duplicates, {r.language_removed:,} by language filter. "
                   "Definition: valid = non-empty, non-spam, de-duplicated (unique comment text) and "
                   "passing the language filter.")
        with st.expander("Rows removed (with reason)"):
            st.dataframe(pre.removed[["text", "removal_reason", "source"]].head(500), hide_index=True)
        ack = True
        if r.valid and not r.target_met and not r.demo_valid:
            ack = st.checkbox("I understand the dataset is below the target and results come from a "
                              "small sample (a warning will be shown on every result).")
        b1, b2 = st.columns([1, 1])
        if b1.button("Preprocess & analyze", type="primary", disabled=(r.valid == 0 or not ack)):
            with st.spinner("Analyzing..."):
                S.bundle = run_analysis(raw, niche, brand, int(target), language_filter)
            st.success("Analysis complete - open the **Dashboard** tab.")
        if r.valid == 0:
            st.error("No valid comments remain after preprocessing. Add more data or relax the filters.")
        if b2.button("Clear dataset"):
            clear_data()
            st.rerun()


# ----------------------------------------------------------------------------
# Tab 2: Dashboard
# ----------------------------------------------------------------------------
with tabs[1]:
    b = get_bundle()
    if b is None:
        empty_state()
    else:
        banners(b)
        df, rep, snap = b.analyzed, b.report, b.snapshot
        dist = df["sentiment"].value_counts(normalize=True)
        top_prod = b.product_ranking.iloc[0]["product"] if len(b.product_ranking) else "n/a"
        top_asp = b.aspect_summary.iloc[0]["aspect"].replace("_", " ") if len(b.aspect_summary) else "n/a"
        top_comp = b.complaints.iloc[0]["aspect"].replace("_", " ") if len(b.complaints) else "n/a"
        m = st.columns(5)
        m[0].metric("Valid comments analyzed", f"{len(df):,}")
        m[1].metric("Sources", df["source"].nunique())
        m[2].metric("Positive", fmt_pct(dist.get("positive", 0)))
        m[3].metric("Neutral", fmt_pct(dist.get("neutral", 0)))
        m[4].metric("Negative", fmt_pct(dist.get("negative", 0)))
        m2 = st.columns(4)
        m2[0].metric("Net sentiment", f"{snap['net_sentiment']:+.2f}")
        m2[0].caption(f"95% margin of error: +/-{snap['margin_of_error']:.2f}")
        m2[1].metric("Most discussed product", top_prod)
        m2[2].metric("Most discussed aspect", top_asp)
        m2[3].metric("Top complaint", top_comp)
        dates = df["created_at"].dropna()
        st.caption(f"Feedback snapshot - {len(df):,} comments - sources: {', '.join(sorted(df['source'].unique()))} - "
                   f"time range: {dates.min().date() if len(dates) else 'n/a'} to "
                   f"{dates.max().date() if len(dates) else 'n/a'} - analysis {SENTIMENT_MODEL_VERSION}")
        with st.expander("How is the snapshot calculated? (not a brand verdict)"):
            st.markdown(f"Net sentiment = (positive - negative) / total comments, range -1 to +1, shown with a "
                        f"95% margin of error. {snap['formula']}. It reflects only the comments collected "
                        "from the listed sources and dates; sarcasm, slang and mixed opinions can be misread.")

        a, c = st.columns(2)
        fig = go.Figure(go.Funnel(y=[s for s, _ in rep.funnel()], x=[n for _, n in rep.funnel()],
                                  marker=dict(color=ROSE)))
        fig.update_layout(title="Data quality funnel", **PLOT_LAYOUT)
        a.plotly_chart(fig)
        counts = df["sentiment"].value_counts().reindex(["positive", "neutral", "negative"], fill_value=0)
        fig = px.pie(values=counts.values, names=counts.index, hole=0.55, color=counts.index,
                     color_discrete_map=SENT_COLORS, title="Sentiment distribution")
        fig.update_layout(**PLOT_LAYOUT)
        c.plotly_chart(fig)

        a, c = st.columns(2)
        bs = sentiment_by_source(df)
        fig = px.bar(bs, x="source", y="pct", color="sentiment", color_discrete_map=SENT_COLORS,
                     title="Sentiment by source (%)", barmode="stack")
        fig.update_layout(**PLOT_LAYOUT)
        a.plotly_chart(fig)
        freq = c.radio("Time grouping", ["W", "M"], horizontal=True, format_func=lambda x: "Weekly" if x == "W" else "Monthly")
        ot = sentiment_over_time(df, freq)
        if ot.empty:
            c.info("No comment dates available for a time chart.")
        else:
            long = ot.melt(id_vars="period", value_vars=["positive", "neutral", "negative"],
                           var_name="sentiment", value_name="comments")
            fig = px.bar(long, x="period", y="comments", color="sentiment", color_discrete_map=SENT_COLORS,
                         title="Sentiment over time")
            fig.update_layout(**PLOT_LAYOUT)
            c.plotly_chart(fig)
            undated = int(df["created_at"].isna().sum())
            if undated:
                c.caption(f"{undated:,} comments without a date are excluded from this chart.")

        a, c = st.columns(2)
        if len(b.product_ranking):
            top = b.product_ranking.head(10).sort_values("mentions")
            fig = px.bar(top, x="mentions", y="product", orientation="h", color="pos_pct",
                         color_continuous_scale=["#C4553B", "#EBD9D0", "#2E7D6B"], range_color=(0, 100),
                         title="Most discussed products (colour = % positive)")
            fig.update_layout(**PLOT_LAYOUT)
            a.plotly_chart(fig)
            a.caption(DISCLAIMER)
        else:
            a.info("No product names or types were found in the comments.")
        if len(b.aspect_summary):
            long = b.aspect_summary.melt(id_vars="aspect", value_vars=["positive", "neutral", "negative"],
                                         var_name="sentiment", value_name="count")
            fig = px.bar(long, x="count", y="aspect", color="sentiment", orientation="h",
                         color_discrete_map=SENT_COLORS, title="Aspect sentiment (mentions)")
            fig.update_layout(yaxis=dict(categoryorder="total ascending"), **PLOT_LAYOUT)
            c.plotly_chart(fig)
        else:
            c.info("No aspects detected.")

        st.subheader("Recurring complaints")
        if b.complaints.empty:
            st.info("No negative aspect mentions found.")
        else:
            fig = px.bar(b.complaints.sort_values("negative_mentions"), x="negative_mentions", y="aspect",
                         orientation="h", title="Negative mentions by aspect", color_discrete_sequence=[SENT_COLORS["negative"]])
            fig.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig)
            ev = b.complaints.assign(example=b.complaints["examples"].map(lambda x: x[0] if x else ""),
                                     comment_id=b.complaints["example_comment_ids"].map(lambda x: x[0] if x else ""))
            st.dataframe(ev[["aspect", "negative_mentions", "example", "comment_id"]], hide_index=True)

        if S.competitor:
            st.subheader("Competitor comparison")
            st.dataframe(competitor_comparison({b.brand or "This brand": b, S.competitor[0]: S.competitor[1]}),
                         hide_index=True)
            st.caption("Dataset sizes, sources and date ranges are shown. No statistical-significance test is run.")


# ----------------------------------------------------------------------------
# Tab 3: Product & Aspect Insights
# ----------------------------------------------------------------------------
with tabs[2]:
    b = get_bundle()
    if b is None:
        empty_state()
    else:
        banners(b)
        st.subheader("What customers like")
        likes = top_likes(b.aspect_rows)
        if likes.empty:
            st.info("No positive aspect mentions found.")
        else:
            lk = likes.assign(example=likes["examples"].map(lambda x: x[0] if x else ""))
            st.dataframe(lk[["aspect", "positive_mentions", "example"]], hide_index=True)
        st.subheader("Aspect explorer")
        if b.aspect_rows.empty:
            st.info("No aspects detected.")
        else:
            c1, c2, c3 = st.columns(3)
            aspect = c1.selectbox("Aspect", sorted(b.aspect_rows["aspect"].unique()))
            sent = c2.selectbox("Sentiment", ["all", "positive", "neutral", "negative"])
            query = c3.text_input("Contains text")
            ev = b.aspect_rows[b.aspect_rows["aspect"] == aspect]
            if sent != "all":
                ev = ev[ev["aspect_sentiment"] == sent]
            if query:
                ev = ev[ev["evidence"].str.contains(query, case=False, regex=False, na=False)]
            st.caption(f"{len(ev):,} supporting comment(s). Use the comment_id in **Data Explorer** to see the full row.")
            st.dataframe(ev[["comment_id", "aspect_sentiment", "aspect_score", "evidence", "source"]].head(300),
                         hide_index=True)

        st.subheader("Products - most discussed / highest positive response")
        st.caption(DISCLAIMER)
        if b.product_ranking.empty:
            st.info("No product names or types found in this dataset.")
        else:
            show = b.product_ranking[["product", "mentions", "share_of_mentions", "pos_pct", "neg_pct", "avg_score",
                              "total_likes", "most_discussed_rank", "positive_response_rank", "labels"]]
            st.dataframe(show, hide_index=True)
            st.caption("Highest positive response uses the lower bound of a Wilson interval and only ranks "
                       "products with enough mentions, so tiny samples cannot win.")
            prod = st.selectbox("Inspect a product", list(b.product_ranking["product"]))
            ids = b.product_mentions.loc[b.product_mentions["product"] == prod, "comment_id"]
            st.dataframe(b.analyzed[b.analyzed["comment_id"].isin(ids)][
                ["comment_id", "text", "sentiment", "like_count", "source"]].head(200), hide_index=True)

        st.subheader("Feature requests and unmet needs")
        if b.request_summary.empty:
            st.info("No request-style comments (\"I wish...\", \"please make...\") detected.")
        else:
            rs = b.request_summary.assign(example=b.request_summary["examples"].map(lambda x: x[0] if x else ""))
            st.dataframe(rs[["theme", "requests", "example"]], hide_index=True)
            with st.expander("All detected requests"):
                st.dataframe(b.requests[["comment_id", "theme", "request_text", "group_size"]].head(500),
                             hide_index=True)


# ----------------------------------------------------------------------------
# Tab 4: Recommendations
# ----------------------------------------------------------------------------
with tabs[3]:
    b = get_bundle()
    if b is None:
        empty_state()
    else:
        banners(b)
        st.subheader("Improve existing products")
        st.caption("Each recommendation is traceable to specific comments. Rules are keyword-based - "
                   "read the examples before acting.")
        if b.recommendations.empty:
            st.info("Not enough negative evidence yet (a recommendation needs at least 3 supporting comments).")
        else:
            for _, r in b.recommendations.iterrows():
                with st.expander(f"[{r['priority'].upper()}] {r['recommendation']} - {r['evidence_count']} comments, "
                                 f"{r['confidence']} confidence"):
                    st.write(r["action"])
                    st.write(f"Aspect: `{r['aspects']}` - {100 * r['share_of_negative']:.0f}% of negative mentions")
                    st.write("**Example comments**")
                    for txt, cid in zip(r["example_comments"], r["example_comment_ids"]):
                        st.text(f"\u201c{txt}\u201d  (comment_id: {cid})")  # untrusted: plain text only
                    st.caption(r["limitations"])
        st.subheader("Future product ideas (hypotheses)")
        st.warning(HYPOTHESIS_LABEL)
        if b.opportunities.empty:
            st.info("Not enough request-style comments to suggest concepts (needs at least 3 per theme).")
        else:
            for _, o in b.opportunities.iterrows():
                with st.expander(f"{o['concept']} - {o['evidence_count']} requests"):
                    st.markdown(f"**Customer need:** {o['customer_need']}  \n**Desired features:** {o['desired_features']}  \n"
                                f"**Differentiator:** {o['differentiator']}  \n**Risk / assumption:** {o['risk_assumption']}  \n"
                                f"**Low-cost validation test:** {o['validation_test']}")
                    for txt, cid in zip(o["example_comments"], o["example_comment_ids"]):
                        st.text(f"\u201c{txt}\u201d  (comment_id: {cid})")  # untrusted: plain text only
                    st.caption(o["limitations"])
        files = get_files(b)
        rec_name = [k for k in files if k.endswith("recommendations.csv")][0]
        st.download_button("Download recommendations CSV", files[rec_name], rec_name, "text/csv")


# ----------------------------------------------------------------------------
# Tab 5: Data Explorer
# ----------------------------------------------------------------------------
with tabs[4]:
    b = get_bundle()
    if b is None:
        empty_state()
    else:
        banners(b)
        df = b.analyzed
        c1, c2, c3, c4 = st.columns(4)
        srcs = c1.multiselect("Source", sorted(df["source"].unique()))
        sents = c2.multiselect("Sentiment", ["positive", "neutral", "negative"])
        prods = c3.multiselect("Product type", sorted(b.product_mentions["product"].unique()) if len(b.product_mentions) else [])
        asps = c4.multiselect("Aspect", sorted(b.aspect_rows["aspect"].unique()) if len(b.aspect_rows) else [])
        q, dr = st.columns([2, 2])
        text_q = q.text_input("Search comment text")
        dated = df["created_at"].dropna()
        date_rng = dr.date_input("Date range", value=(dated.min().date(), dated.max().date())) if len(dated) else None
        view = df
        if srcs:
            view = view[view["source"].isin(srcs)]
        if sents:
            view = view[view["sentiment"].isin(sents)]
        if prods:
            ids = b.product_mentions.loc[b.product_mentions["product"].isin(prods), "comment_id"]
            view = view[view["comment_id"].isin(ids)]
        if asps:
            view = view[view["aspects"].map(lambda s: bool(set(s.split("|")) & set(asps)))]
        if text_q:
            view = view[view["text"].str.contains(text_q, case=False, regex=False, na=False)]
        if date_rng is not None and isinstance(date_rng, tuple) and len(date_rng) == 2:
            d = view["created_at"].dt.date
            view = view[d.isna() | ((d >= date_rng[0]) & (d <= date_rng[1]))]
        st.write(f"**{len(view):,} of {len(df):,} comments match.** The table below is exactly what the export contains.")
        show_cols = ["comment_id", "text", "created_at", "source", "sentiment", "sentiment_score", "aspects",
                     "like_count", "language"]
        st.dataframe(view[show_cols], hide_index=True)
        st.download_button("Download these rows (CSV)", to_csv_bytes(view[show_cols]),
                           "filtered_comments.csv", "text/csv", disabled=view.empty)


# ----------------------------------------------------------------------------
# Tab 6: CSV Library / Reuse
# ----------------------------------------------------------------------------
with tabs[5]:
    b = get_bundle()
    st.subheader("Download exports")
    st.caption("UTF-8 with BOM (opens correctly in Excel). Commenter identity columns and anything that looks "
               "like a credential are stripped; cells beginning with = + @ get a leading apostrophe.")
    if b is None:
        st.info("Run an analysis to generate downloadable files.")
        files = {}
    else:
        banners(b)
        files = get_files(b)
        cols = st.columns(2)
        for i, (name, data) in enumerate(files.items()):
            cols[i % 2].download_button(name, data, name, mime_for(name), key=f"dl_{name}")
        st.caption("Downloads go to your browser's download folder. This app does not write files to the "
                   "server; on Streamlit Community Cloud the server disk is temporary anyway.")

    st.subheader("Reuse a previous export")
    up = st.file_uploader("Upload a CSV exported from this app", type=["csv"], key="reuse_csv")
    if up is not None:
        try:
            reloaded, meta = load_exported_csv(up.getvalue())
            st.write(f"**{meta['rows']:,} rows** - sources: {', '.join(meta['sources'])} - "
                     f"dates: {meta['date_min']} to {meta['date_max']} - sentiment present: {meta['has_sentiment']}")
            if meta["contains_demo"]:
                st.markdown(f'<div class="fjvi-demo">{DEMO_LABEL}</div>', unsafe_allow_html=True)
            if st.button("Load into session"):
                add_raw(reloaded, [{"platform": "reloaded_csv", "identifier": up.name, "count": len(reloaded)}], replace=True)
                st.success("Loaded. Go to **Collect Data > Preprocess & analyze** to re-run the analysis "
                           "(the same target and sample-size checks apply). Nothing is re-collected.")
        except PipelineError as exc:
            st.error(str(exc))

    st.subheader("Optional: Google Cloud Storage")
    gcs_ready = storage.is_configured()
    st.caption("Configured." if gcs_ready else "Not configured. " + storage.SETUP)
    st.caption("Separate from YouTube access. A bucket may incur charges and is not automatically free. "
               "Keep it private. Everyone who can open this deployment shares the same export folder, so "
               "keep the app private.")
    if gcs_ready:
        if files and st.button("Save all exports to Cloud Storage"):
            try:
                saved = storage.save_exports(files, brand or "export")
                st.success(f"Saved {len(saved)} files under {saved[0].rsplit('/', 1)[0]}/")
            except ConnectorError as exc:
                st.error(str(exc))
        try:
            objs = [o for o in storage.list_exports() if o.endswith("analyzed_comments.csv")]
            if objs:
                pick = st.selectbox("Saved analyzed exports", objs)
                if st.button("Load selected export"):
                    reloaded, meta = load_exported_csv(storage.load_export(pick))
                    add_raw(reloaded, [{"platform": "gcs", "identifier": pick, "count": len(reloaded)}], replace=True)
                    st.success(f"Loaded {meta['rows']:,} rows. Go to Collect Data to analyze.")
            else:
                st.info("No analyzed exports saved yet.")
        except (ConnectorError, PipelineError) as exc:
            st.error(str(exc))


# ----------------------------------------------------------------------------
# Tab 7: Setup & Methodology
# ----------------------------------------------------------------------------
with tabs[6]:
    st.subheader("Setup")
    st.markdown("**No credentials needed** for CSV upload and demo mode. For YouTube comments:")
    st.code(SETUP_INSTRUCTIONS, language="text")
    st.markdown(
        "**What a brand URL can and cannot do:** a URL is never treated as permission to collect. "
        "Google Cloud's YouTube Data API v3 covers eligible public YouTube video comments only - it does not "
        "extract Facebook, Instagram or TikTok comments. Meta needs official access to accounts you own or manage; "
        "TikTok's Research API is approval-restricted. For those platforms, use an export you are allowed to "
        "download and upload it as CSV.")
    st.subheader("Methodology")
    st.markdown(f"""
- **Cleaning:** HTML/entities stripped, whitespace normalized; negations, emojis and short comments ("love it") are kept.
- **Valid unique comment:** non-empty, not flagged as spam, de-duplicated (same comment id, or same text after
  lower-casing, removing ASCII punctuation and collapsing repeated letters), and passing the language filter.
  The target is counted on this definition; demo rows never count.
- **Spam:** conservative rules (promo phrases, multiple/only URLs, long character or word repeats); removed rows are listed with reasons.
- **Sentiment:** VADER plus a fashion/jewelry lexicon (`{SENTIMENT_MODEL_VERSION}`); positive >= 0.05, negative <= -0.05.
  It can misread sarcasm, slang, mixed opinions and non-English text.
- **Aspects:** keyword rules per clause (price, design/style, material, comfort, fit/sizing, quality, durability, color,
  packaging, delivery, customer service, value for money); every insight links to comment ids.
- **Products:** product names/types found in text; ranked as *most discussed* and *highest positive response*
  (Wilson lower bound, minimum mentions). Not sales.
- **Recommendations:** rule table mapping negative evidence to actions, with evidence counts and limits. Opportunity
  ideas are hypotheses, not demand forecasts.
- **Snapshot:** net sentiment with a 95% margin of error - never an absolute verdict on a brand.
""")
    st.subheader("Evaluate sentiment on your own labeled sample")
    st.caption("Label 300+ comments yourself (positive / neutral / negative) and upload them. No scores are "
               "pre-filled - results come only from your file.")
    st.download_button("Download labeled-file template", TEMPLATE_CSV.encode("utf-8-sig"),
                       "labeled_template.csv", "text/csv")
    lab = st.file_uploader("Labeled CSV (columns: text, label)", type=["csv"], key="eval_csv")
    if lab is not None:
        try:
            ev = evaluate_sentiment(lab.getvalue())
            k = st.columns(4)
            k[0].metric("Labeled rows", ev["n"])
            k[1].metric("Accuracy", f"{ev['accuracy']:.3f}")
            k[2].metric("Macro precision / recall", f"{ev['macro_precision']:.2f} / {ev['macro_recall']:.2f}")
            k[3].metric("Macro-F1", f"{ev['macro_f1']:.3f}")
            st.dataframe(ev["per_class"], hide_index=True)
            st.write("Confusion matrix (rows = human label, columns = model)")
            st.dataframe(ev["confusion_matrix"])
            for w in ev["warnings"]:
                st.warning(w)
        except PipelineError as exc:
            st.error(str(exc))
