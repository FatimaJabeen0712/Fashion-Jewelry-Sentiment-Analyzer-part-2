"""Advertising KPIs. All divisions are safe: a zero denominator gives NaN, never an error."""
from __future__ import annotations

import numpy as np
import pandas as pd

SUM_COLS = ["impressions", "clicks", "spend", "conversions", "revenue"]
KPI_COLS = ["ctr", "cpc", "cpa", "conversion_rate", "roas", "profit"]


def _div(a, b):
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / b, np.nan)


def add_kpis(totals: pd.DataFrame) -> pd.DataFrame:
    """Add CTR, CPC, CPA, conversion rate, ROAS and profit to a frame of summed counts."""
    out = totals.copy()
    out["ctr"] = _div(out["clicks"], out["impressions"])
    out["cpc"] = _div(out["spend"], out["clicks"])
    out["cpa"] = _div(out["spend"], out["conversions"])
    out["conversion_rate"] = _div(out["conversions"], out["clicks"])
    out["roas"] = _div(out["revenue"], out["spend"])
    out["profit"] = out["revenue"] - out["spend"]
    return out


def summarize(df: pd.DataFrame, by: str | list[str]) -> pd.DataFrame:
    """Sum counts by one or more columns, then compute the KPIs from the sums
    (ratio of sums, never an average of row ratios)."""
    g = df.groupby(by, observed=True)[SUM_COLS].sum().reset_index()
    return add_kpis(g)
