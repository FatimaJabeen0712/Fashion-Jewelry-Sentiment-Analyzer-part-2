# Step 2 - Digital Advertising Budget Optimizer

Question: **where should the next Rs 1 million of advertising spend go?** Platforms: Facebook,
Instagram, Google, TikTok, YouTube. **All data here is simulated** (the assignment asks for simulation);
nothing comes from a real ad account.

## What was built
| Part | File |
|---|---|
| Simulator (12 columns: date, campaign, platform, impressions, clicks, spend, conversions, revenue, age, gender, city, device) | `src/ads/simulate.py` |
| KPIs: CTR, CPC, CPA, conversion rate, ROAS, profit (ratio of sums) | `src/ads/metrics.py` |
| Same job on Pandas, MapReduce (map -> shuffle -> reduce) and Spark | `src/ads/engines.py`, `src/ads/benchmark.py` |
| Budget Recommendation Engine | `src/ads/optimizer.py` |
| Dashboard page (4 tabs) | `pages/1_Ad_Budget_Optimizer.py` |
| Command line | `run_ads.py` |

## How the simulation works
Each row is one ad set on one day for one audience segment. Platforms differ in CPM, click-through,
conversion rate and order value; campaigns, age, gender, city, device and season (Eid, wedding season,
winter) shift performance. Click-through **falls as an ad set's spend rises** (diminishing returns), with
a different strength per platform. Total yearly spend is fixed at about Rs 60 million for any row count,
so 10k and 10M rows are the same business at different detail.

## How the recommendation works
1. For each platform fit `revenue(B) = R0 * (B / B0) ** b` (B0, R0 = current monthly spend and revenue).
   `b` comes from how ROAS changes across ten equal-count spend bins; 100 bootstrap resamples give a
   10th-90th percentile range.
2. Spend the budget in 2,000 small steps, each to the platform with the highest marginal revenue that is
   under its cap (optimal for concave curves). Guardrails: each platform gets at least 5% and at most 45%
   of the new budget (adjustable).
3. Compare with spreading the same money in proportion to current spend.

Check against the known truth: with 100k+ rows the fitted `b` is within about 0.08 of the value built
into the simulation for every platform (covered by a test).

Example (100,000 rows, Rs 1,000,000 extra): Instagram 45%, Google 40%, Facebook 5%, TikTok 5%,
YouTube 5% - about +20% more revenue from the new money than spreading it like current spend. The 5% /
45% limits are binding here, so read the result as "Instagram and Google first, the rest at the minimum".

## Pandas vs MapReduce vs Spark (2 CPU cores, one machine, median of 2-3 runs)
All three returned identical totals at every size.

| Rows | Pandas | MapReduce | Spark |
|---:|---:|---:|---:|
| 10,000 | 0.02 s | 0.02 s | 0.33 s |
| 100,000 | 0.03 s | 0.23 s | 0.36 s |
| 1,000,000 | 0.09 s | 2.5 s | 0.45 s |
| 10,000,000 | 2.4 s | 14.2 s | 1.4 s |

Reading it honestly: Pandas wins while the data is small because it has no start-up or coordination
cost. Spark overtakes it at about 10M rows even on 2 cores, and its real advantage (scaling across
machines, handling data larger than memory) is not tested here because everything fits in RAM.
The MapReduce version is plain Python and shows the programming model, not tuned performance. Spark's
time excludes JVM start-up.

## Limits
* Simulated data: it shows the method, not real marketing performance. Run it on real exports before
  moving real money, and test any shift with a small budget first.
* The model assumes smooth diminishing returns per platform and ignores interactions between platforms
  (for example YouTube views lifting Google searches).
* Timings depend on the machine and are for comparison only.

## Run it
```
python run_ads.py recommend --rows 100000 --budget 1000000
python run_ads.py benchmark --sizes 10000 100000 1000000 10000000
streamlit run app.py        # then open "Ad Budget Optimizer" in the sidebar
```
Spark needs `pip install pyspark` and Java 17+; if either is missing the Spark row is skipped.
