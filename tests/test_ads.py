import numpy as np
import pandas as pd
import pytest

from src.ads import engines
from src.ads.benchmark import results_agree
from src.ads.metrics import add_kpis, summarize
from src.ads.optimizer import allocate, fit_curves, recommend
from src.ads.simulate import PLATFORMS, YEAR_SPEND, simulate, write_dataset


@pytest.fixture(scope="module")
def df():
    return simulate(60_000, seed=1)


def test_kpis_use_ratio_of_sums_and_zero_denominators_are_nan():
    t = pd.DataFrame({"impressions": [1000, 0], "clicks": [50, 0], "spend": [500.0, 10.0],
                      "conversions": [5, 0], "revenue": [2500.0, 0.0]})
    k = add_kpis(t)
    assert k.loc[0, "ctr"] == pytest.approx(0.05)
    assert k.loc[0, "cpc"] == pytest.approx(10)
    assert k.loc[0, "cpa"] == pytest.approx(100)
    assert k.loc[0, "conversion_rate"] == pytest.approx(0.1)
    assert k.loc[0, "roas"] == pytest.approx(5)
    assert np.isnan(k.loc[1, "ctr"]) and np.isnan(k.loc[1, "cpa"]) and k.loc[1, "roas"] == 0


def test_simulation_is_reproducible_complete_and_sane(df):
    again = simulate(60_000, seed=1)
    pd.testing.assert_frame_equal(df, again)
    assert list(df.columns) == ["date", "campaign", "platform", "impressions", "clicks", "spend",
                                "conversions", "revenue", "age", "gender", "city", "device"]
    assert set(df.platform.astype(str)) == set(PLATFORMS)
    assert (df.clicks <= df.impressions).all() and (df.conversions <= df.clicks).all()
    assert df.spend.sum() == pytest.approx(YEAR_SPEND, rel=0.05)
    assert df.isna().sum().sum() == 0


def test_pandas_and_mapreduce_agree(df, tmp_path):
    d = write_dataset(60_000, tmp_path / "d", seed=1)
    a, _ = engines.pandas_engine(d)
    b, _ = engines.mapreduce_engine(d, workers=1)
    assert results_agree(a, b)
    assert a.impressions.sum() == df.impressions.sum()
    assert a.revenue.sum() == pytest.approx(df.revenue.sum())


def test_multi_partition_mapreduce(tmp_path, monkeypatch):
    monkeypatch.setattr("src.ads.simulate.CHUNK", 7_000)
    d = write_dataset(20_000, tmp_path / "d", seed=3)
    assert len(list(d.glob("part-*.parquet"))) == 3
    a, _ = engines.pandas_engine(d)
    b, _ = engines.mapreduce_engine(d, workers=2)
    assert results_agree(a, b)


@pytest.mark.skipif(not engines.spark_available(), reason="pyspark/Java not installed")
def test_spark_agrees(tmp_path):
    d = write_dataset(20_000, tmp_path / "d", seed=3)
    a, _ = engines.pandas_engine(d)
    b, _ = engines.spark_engine(d)
    assert results_agree(a, b)


def test_fitted_exponents_are_close_to_the_simulation_truth():
    big = simulate(150_000, seed=5)
    curves = {c.platform: c for c in fit_curves(big)}
    for name, p in PLATFORMS.items():
        assert abs(curves[name].b - (1 - p["gamma"])) < 0.08, name


def test_allocation_sums_to_budget_and_respects_limits(df):
    curves = fit_curves(df)
    alloc = allocate(curves, 1_000_000, min_share=0.05, max_share=0.40)
    assert sum(alloc.values()) == pytest.approx(1_000_000)
    assert all(0.05 * 1e6 - 1 <= v <= 0.40 * 1e6 + 1 for v in alloc.values())


def test_recommendation_beats_or_matches_the_current_mix(df):
    rec = recommend(df, 1_000_000)
    assert sum(rec.mix.values()) == pytest.approx(100, abs=0.5)
    assert rec.expected_revenue >= rec.baseline_revenue
    assert rec.uplift_pct >= 0
    top = rec.table.iloc[0]
    assert top.platform in ("Instagram", "Google")      # highest ROAS platforms in the simulation


def test_bad_budget_and_tiny_data():
    with pytest.raises(ValueError):
        recommend(simulate(1000), 0)
    c = fit_curves(simulate(300, seed=2))                # too little data: neutral exponent, no crash
    assert all(0.2 <= x.b <= 1.0 for x in c)


def test_summarize_totals_match(df):
    s = summarize(df, ["platform", "campaign"])
    assert s.spend.sum() == pytest.approx(df.spend.sum())
