"""Step 2 - Digital Advertising Budget Optimizer (simulated data)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DATA_DIR  # noqa: E402
from src.ads.metrics import summarize  # noqa: E402
from src.ads.optimizer import DISCLAIMER, recommend  # noqa: E402
from src.ads.simulate import simulate  # noqa: E402

st.set_page_config(page_title="Ad Budget Optimizer", layout="wide")
st.title("Digital Advertising Budget Optimizer")
st.warning("SIMULATED DATA. The numbers below come from a synthetic data generator, not from a real ad account.")


@st.cache_data(show_spinner="Simulating campaign data...", max_entries=3)
def load(rows: int, seed: int) -> pd.DataFrame:
    return simulate(rows, seed)


with st.sidebar:
    rows = st.select_slider("Simulated records", [10_000, 50_000, 100_000, 250_000, 500_000], value=100_000)
    seed = st.number_input("Random seed", 0, 10_000, 42)
    budget = st.number_input("Budget to allocate (Rs)", 100_000, 100_000_000, 1_000_000, step=100_000)
    extra = st.radio("This budget is...", ["Extra money on top of current spend", "The whole monthly budget"]) \
        .startswith("Extra")
    lo, hi = st.slider("Allowed share per platform (%)", 0, 100, (5, 45))

df = load(rows, int(seed))
tabs = st.tabs(["Performance", "Budget recommendation", "Audience", "Pandas vs MapReduce vs Spark"])

with tabs[0]:
    plat = summarize(df, "platform")
    tot = summarize(df.assign(all="All platforms"), "all").iloc[0]
    c = st.columns(5)
    c[0].metric("Spend", f"Rs {tot.spend:,.0f}")
    c[1].metric("Revenue", f"Rs {tot.revenue:,.0f}")
    c[2].metric("ROAS", f"{tot.roas:.2f}")
    c[3].metric("CTR", f"{tot.ctr:.2%}")
    c[4].metric("CPA", f"Rs {tot.cpa:,.0f}")
    st.dataframe(plat.round(4), use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(plat, x="platform", y="roas", title="ROAS by platform (revenue / spend)"),
                    use_container_width=True)
    st.dataframe(summarize(df, ["platform", "campaign"]).sort_values("roas", ascending=False).round(4),
                 use_container_width=True, hide_index=True)

with tabs[1]:
    try:
        rec = recommend(df, float(budget), incremental=extra, min_share=lo / 100, max_share=hi / 100)
    except ValueError as exc:
        st.error(str(exc))
    else:
        st.subheader("Recommended split")
        st.write("  |  ".join(f"**{p}: {s:.0f}%**" for p, s in rec.mix.items()))
        st.plotly_chart(px.pie(rec.table, names="platform", values="budget", hole=0.45), use_container_width=True)
        st.dataframe(rec.table.round(3), use_container_width=True, hide_index=True)
        st.metric("Expected extra revenue from this budget", f"Rs {rec.expected_revenue:,.0f}",
                  f"{rec.uplift_pct:+.1f}% vs spreading it like current spend")
        st.caption("b = response exponent: closer to 1 means spend keeps paying off; lower means saturating. "
                   "Range is the 10th-90th percentile over 100 bootstrap resamples.")
        st.info(DISCLAIMER)

with tabs[2]:
    for col in ("age", "gender", "city", "device"):
        s = summarize(df, col).sort_values("roas", ascending=False)
        st.plotly_chart(px.bar(s, x=col, y="roas", title=f"ROAS by {col}"), use_container_width=True)

with tabs[3]:
    path = DATA_DIR / "ads" / "benchmark_results.json"
    if not path.exists():
        st.info("No benchmark yet. Run `python run_ads.py benchmark` in a terminal, then reload this page.")
    else:
        res = json.loads(path.read_text(encoding="utf-8"))
        b = pd.DataFrame(res["rows"])
        st.plotly_chart(px.line(b, x="rows", y="total_s", color="engine", markers=True, log_x=True, log_y=True,
                                title="Total time by data size (lower is better, log scales)"),
                        use_container_width=True)
        st.dataframe(b, use_container_width=True, hide_index=True)
        for k, v in res.get("skipped", {}).items():
            st.warning(f"{k} was skipped: {v}")
        st.caption("Single machine, 2 cores. Spark time excludes JVM start-up. Every engine returned the same "
                   "totals (matches_pandas).")
