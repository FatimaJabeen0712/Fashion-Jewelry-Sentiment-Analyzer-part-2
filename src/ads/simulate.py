"""Synthetic advertising data (clearly simulated - the assignment asks for simulation).

Each row is one *ad set on one day for one audience segment*. Ground truth built into the
simulation (so the optimizer can be checked against it):

* each platform has its own CPM, click-through rate, conversion rate and order value;
* **diminishing returns** - click-through falls as an ad set's spend rises
  (``ctr = ctr0 * (spend / reference) ** -gamma``), so revenue grows like ``spend ** (1 - gamma)``;
* audience segments (age, gender, city, device), campaigns and seasonality shift performance.

Total yearly spend is held at about Rs 60 million whatever the row count, so 10k and 10M rows
describe the same business at different levels of detail.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

YEAR_SPEND = 60_000_000.0
START = np.datetime64("2025-01-01")
DAYS = 365

# share of spend, CPM (Rs per 1000 impressions), base CTR, base conv rate (per click),
# average order value (Rs), saturation gamma
PLATFORMS = {
    "Facebook":  dict(share=0.28, cpm=250, ctr=0.012, cvr=0.0080, aov=6500, gamma=0.20),
    "Instagram": dict(share=0.22, cpm=320, ctr=0.010, cvr=0.0150, aov=7500, gamma=0.12),
    "Google":    dict(share=0.25, cpm=400, ctr=0.035, cvr=0.0070, aov=5500, gamma=0.30),
    "TikTok":    dict(share=0.13, cpm=180, ctr=0.008, cvr=0.0060, aov=5000, gamma=0.05),
    "YouTube":   dict(share=0.12, cpm=220, ctr=0.006, cvr=0.0080, aov=5500, gamma=0.18),
}
CAMPAIGNS = {  # name: (spend weight, ctr multiplier, cvr multiplier)
    "Eid Collection": (0.30, 1.10, 1.10),
    "Bridal Jewelry": (0.25, 0.95, 1.30),
    "Winter Sale": (0.25, 1.05, 0.90),
    "Retargeting": (0.20, 1.20, 1.60),
}
AGES = {"18-24": (0.22, 1.10, 0.85), "25-34": (0.36, 1.00, 1.15), "35-44": (0.24, 0.95, 1.10),
        "45-54": (0.12, 0.90, 0.90), "55+": (0.06, 0.85, 0.70)}
GENDERS = {"Female": (0.72, 1.05, 1.10), "Male": (0.28, 0.90, 0.75)}
CITIES = {"Karachi": (0.25, 1.0, 1.00), "Lahore": (0.24, 1.0, 1.10), "Islamabad": (0.12, 0.95, 1.20),
          "Rawalpindi": (0.09, 1.0, 0.95), "Faisalabad": (0.10, 1.05, 0.85), "Multan": (0.08, 1.05, 0.80),
          "Peshawar": (0.07, 1.00, 0.80), "Quetta": (0.05, 1.0, 0.70)}
DEVICES = {"Mobile": (0.78, 1.05, 0.95), "Desktop": (0.17, 0.85, 1.25), "Tablet": (0.05, 0.90, 1.00)}
MONTH_CVR = np.array([0.95, 0.95, 1.00, 1.15, 1.10, 0.90, 0.85, 0.85, 0.95, 1.05, 1.15, 1.25])  # Eid / wedding / winter
SPEND_SIGMA = 0.7
CHUNK = 1_000_000
COLUMNS = ["date", "campaign", "platform", "impressions", "clicks", "spend", "conversions", "revenue",
           "age", "gender", "city", "device"]


def _pick(rng, table: dict, n: int):
    names = list(table)
    p = np.array([table[k][0] for k in names], dtype=float)
    idx = rng.choice(len(names), size=n, p=p / p.sum())
    ctr_m = np.array([table[k][1] for k in names])[idx]
    cvr_m = np.array([table[k][2] for k in names])[idx]
    return pd.Categorical.from_codes(idx, names), ctr_m, cvr_m


def _chunk(n: int, rng: np.random.Generator, total_rows: int) -> pd.DataFrame:
    plats = list(PLATFORMS)
    p = np.array([PLATFORMS[k]["share"] for k in plats])
    pidx = rng.choice(len(plats), size=n, p=p)
    day = rng.integers(0, DAYS, size=n)
    month = ((START + day.astype("timedelta64[D]")).astype("datetime64[M]").astype(int) % 12)

    camp, c_ctr, c_cvr = _pick(rng, CAMPAIGNS, n)
    age, a_ctr, a_cvr = _pick(rng, AGES, n)
    gen, g_ctr, g_cvr = _pick(rng, GENDERS, n)
    city, y_ctr, y_cvr = _pick(rng, CITIES, n)
    dev, d_ctr, d_cvr = _pick(rng, DEVICES, n)

    cols = {k: np.array([PLATFORMS[q][k] for q in plats])[pidx] for k in ("cpm", "ctr", "cvr", "aov", "gamma")}
    # mean spend per row of each platform keeps the platform's yearly total fixed
    rows_p = total_rows * p
    mean_spend = (YEAR_SPEND * p / rows_p)[pidx]
    spend = mean_spend * rng.lognormal(-SPEND_SIGMA ** 2 / 2, SPEND_SIGMA, n)
    imp = np.round(spend / cols["cpm"] * 1000 * rng.lognormal(-0.02, 0.2, n))
    ctr = cols["ctr"] * c_ctr * a_ctr * g_ctr * y_ctr * d_ctr * (spend / mean_spend) ** (-cols["gamma"])
    clicks = rng.poisson(imp * np.clip(ctr, 0, 0.5))
    cvr = np.clip(cols["cvr"] * c_cvr * a_cvr * g_cvr * y_cvr * d_cvr * MONTH_CVR[month], 0, 0.9)
    conv = rng.binomial(clicks, cvr)
    revenue = conv * cols["aov"] * rng.lognormal(-0.045, 0.3, n)

    return pd.DataFrame({
        "date": (START + day.astype("timedelta64[D]")).astype("datetime64[s]"),
        "campaign": camp, "platform": pd.Categorical.from_codes(pidx, plats),
        "impressions": imp.astype("int64"), "clicks": clicks.astype("int64"),
        "spend": np.round(spend, 2), "conversions": conv.astype("int64"), "revenue": np.round(revenue, 2),
        "age": age, "gender": gen, "city": city, "device": dev,
    })[COLUMNS]


def simulate(n_rows: int, seed: int = 42) -> pd.DataFrame:
    """Generate ``n_rows`` records in memory (use ``write_dataset`` for 1M+)."""
    if n_rows < 1:
        raise ValueError("n_rows must be at least 1")
    return _chunk(n_rows, np.random.default_rng(seed), n_rows)


def write_dataset(n_rows: int, out_dir: Path, seed: int = 42, force: bool = False) -> Path:
    """Write ``n_rows`` records as Parquet partitions of up to 1M rows. Reuses an existing dataset."""
    out_dir = Path(out_dir)
    marker = out_dir / "_SUCCESS"
    if marker.exists() and not force:
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("part-*.parquet"):
        old.unlink()
    rng = np.random.default_rng(seed)
    done, part = 0, 0
    while done < n_rows:
        k = min(CHUNK, n_rows - done)
        _chunk(k, rng, n_rows).to_parquet(out_dir / f"part-{part:03d}.parquet", index=False)
        done += k
        part += 1
    marker.write_text(f"{n_rows} rows, seed {seed}\n")
    return out_dir
