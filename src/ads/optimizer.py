"""Marketing Budget Recommendation Engine.

For each platform, estimate a diminishing-returns curve from the historical data and then split the
next budget so that the *marginal* return is as equal as possible across platforms.

Model (per platform):  revenue(B) = R0 * (B / B0) ** b      with 0 < b <= 1
  * B0, R0 = current monthly spend and revenue (history / months covered)
  * b      = response exponent, fitted from how ROAS changes with ad-set spend:
             ROAS ~ spend ** (b - 1), using ten equal-count spend bins (ratio of sums per bin)
Allocation: greedy in small steps - each step goes to the platform with the highest marginal revenue
that is under its cap; optimal for concave curves. Guardrails: each platform keeps a minimum share
and a maximum share of the budget being allocated.

Everything here is based on the data you give it. With simulated data the result illustrates the method;
it is not a forecast for a real business.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import SUM_COLS, add_kpis

MIN_SHARE = 0.05
MAX_SHARE = 0.45
STEPS = 2000
FIT_SAMPLE = 200_000
BOOTSTRAPS = 100
DISCLAIMER = ("Based on historical (simulated) performance with a diminishing-returns assumption. "
              "Past results do not guarantee future results; validate with a small test before moving real budget.")


@dataclass
class PlatformCurve:
    platform: str
    b: float
    b_low: float
    b_high: float
    monthly_spend: float
    monthly_revenue: float

    def revenue(self, budget: float) -> float:
        return self.monthly_revenue * (max(budget, 0.0) / self.monthly_spend) ** self.b

    def marginal(self, budget: float) -> float:
        budget = max(budget, 1e-9)
        return self.monthly_revenue * self.b * budget ** (self.b - 1) / self.monthly_spend ** self.b


@dataclass
class Recommendation:
    budget: float
    table: pd.DataFrame
    expected_revenue: float
    baseline_revenue: float
    uplift_pct: float
    curves: list[PlatformCurve] = field(default_factory=list)
    mix: dict = field(default_factory=dict)
    note: str = DISCLAIMER


def _fit_b(spend: np.ndarray, revenue: np.ndarray, bins: int = 10) -> float | None:
    """Slope of log(ROAS) on log(mean spend) over equal-count spend bins, plus one."""
    if len(spend) < bins * 20:
        return None
    order = np.argsort(spend, kind="stable")
    xs, ys = [], []
    for idx in np.array_split(order, bins):
        s, r = spend[idx].sum(), revenue[idx].sum()
        if s > 0 and r > 0:
            xs.append(np.log(spend[idx].mean()))
            ys.append(np.log(r / s))
    if len(xs) < 5:
        return None
    slope = np.polyfit(xs, ys, 1)[0]
    return float(np.clip(1.0 + slope, 0.2, 1.0))


def fit_curves(df: pd.DataFrame, months: float | None = None, seed: int = 7) -> list[PlatformCurve]:
    """Fit one curve per platform. ``df`` needs platform, spend, revenue, date (for the month count)."""
    if months is None:
        d = pd.to_datetime(df["date"])
        months = max((d.max() - d.min()).days / 30.44, 1.0)
    rng = np.random.default_rng(seed)
    curves = []
    for plat, g in df.groupby("platform", observed=True):
        spend_tot, rev_tot = float(g["spend"].sum()), float(g["revenue"].sum())
        if spend_tot <= 0 or rev_tot <= 0:
            continue
        sample = g if len(g) <= FIT_SAMPLE else g.sample(FIT_SAMPLE, random_state=seed)
        s, r = sample["spend"].to_numpy(float), sample["revenue"].to_numpy(float)
        b = _fit_b(s, r)
        if b is None:
            b, lo, hi = 0.8, 0.8, 0.8   # too little data: neutral assumption, flagged by equal bounds
        else:
            boots = []
            small = min(len(s), 50_000)
            for _ in range(BOOTSTRAPS):
                ix = rng.integers(0, len(s), small)
                bb = _fit_b(s[ix], r[ix])
                if bb is not None:
                    boots.append(bb)
            lo, hi = (np.percentile(boots, [10, 90]) if boots else (b, b))
        curves.append(PlatformCurve(str(plat), b, float(lo), float(hi), spend_tot / months, rev_tot / months))
    if not curves:
        raise ValueError("No platform has positive spend and revenue to fit.")
    return curves


def allocate(curves: list[PlatformCurve], budget: float, incremental: bool = True,
             min_share: float = MIN_SHARE, max_share: float = MAX_SHARE) -> dict[str, float]:
    """Greedy equal-marginal-return allocation of ``budget``.

    ``incremental=True``: the budget is *extra* money on top of each platform's current monthly spend.
    ``incremental=False``: the budget is the *whole* monthly budget (current spend is ignored).
    """
    n = len(curves)
    min_share = min(min_share, 1.0 / n)
    max_share = max(max_share, 1.0 / n)
    alloc = {c.platform: budget * min_share for c in curves}
    left = budget - sum(alloc.values())
    step = budget / STEPS
    base = {c.platform: (c.monthly_spend if incremental else 0.0) for c in curves}
    cap = budget * max_share
    while left > 1e-9:
        live = [c for c in curves if alloc[c.platform] < cap - 1e-9]
        if not live:
            break
        best = max(live, key=lambda c: c.marginal(base[c.platform] + alloc[c.platform]))
        add = min(step, left, cap - alloc[best.platform])
        alloc[best.platform] += add
        left -= add
    return alloc


def recommend(df: pd.DataFrame, budget: float = 1_000_000, incremental: bool = True,
              min_share: float = MIN_SHARE, max_share: float = MAX_SHARE) -> Recommendation:
    if budget <= 0:
        raise ValueError("budget must be positive")
    curves = fit_curves(df)
    alloc = allocate(curves, budget, incremental, min_share, max_share)
    by = {c.platform: c for c in curves}
    # comparison: spread the same budget in proportion to current spend
    tot_spend = sum(c.monthly_spend for c in curves)
    even = {c.platform: budget * c.monthly_spend / tot_spend for c in curves}

    def total_rev(a: dict) -> float:
        return sum(by[p].revenue((by[p].monthly_spend if incremental else 0.0) + x) for p, x in a.items())

    def gained(a: dict) -> float:   # revenue attributable to the new budget
        base = sum(by[p].monthly_revenue for p in a) if incremental else 0.0
        return total_rev(a) - base

    rows = []
    for p, x in alloc.items():
        c = by[p]
        rows.append({"platform": p, "share_pct": round(100 * x / budget, 1), "budget": round(x, 0),
                     "current_roas": c.monthly_revenue / c.monthly_spend,
                     "marginal_roas_after": c.marginal((c.monthly_spend if incremental else 0.0) + x),
                     "response_exponent_b": round(c.b, 3),
                     "b_range_p10_p90": f"{c.b_low:.2f}-{c.b_high:.2f}",
                     "current_mix_share_pct": round(100 * even[p] / budget, 1)})
    table = pd.DataFrame(rows).sort_values("share_pct", ascending=False).reset_index(drop=True)
    new_rev, even_rev = gained(alloc), gained(even)
    uplift = 100 * (new_rev - even_rev) / even_rev if even_rev > 0 else float("nan")
    return Recommendation(budget=budget, table=table, expected_revenue=new_rev, baseline_revenue=even_rev,
                          uplift_pct=uplift, curves=curves, mix={p: round(100 * x / budget, 1) for p, x in alloc.items()})
