"""Step 2 command line: simulate ad data, benchmark Pandas vs MapReduce vs Spark, recommend a budget split.

    python run_ads.py recommend --rows 100000 --budget 1000000
    python run_ads.py benchmark --sizes 10000 100000 1000000 10000000
    python run_ads.py simulate --rows 1000000

Data is written under data/ads/ (git-ignored). The Spark engine is skipped if pyspark or Java is missing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import DATA_DIR

ADS_DIR = DATA_DIR / "ads"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate", help="write a simulated dataset")
    s.add_argument("--rows", type=int, default=100_000)
    b = sub.add_parser("benchmark", help="Pandas vs MapReduce vs Spark")
    b.add_argument("--sizes", type=int, nargs="*", default=[10_000, 100_000, 1_000_000, 10_000_000])
    b.add_argument("--repeats", type=int, default=3)
    b.add_argument("--engines", nargs="*", default=["pandas", "mapreduce", "spark"])
    r = sub.add_parser("recommend", help="budget recommendation from simulated history")
    r.add_argument("--rows", type=int, default=100_000)
    r.add_argument("--budget", type=float, default=1_000_000)
    r.add_argument("--whole-budget", action="store_true", help="budget is the whole month, not extra money")
    a = ap.parse_args(argv)

    from src.ads import benchmark as bm
    from src.ads.optimizer import recommend
    from src.ads.simulate import simulate, write_dataset

    if a.cmd == "simulate":
        out = write_dataset(a.rows, ADS_DIR / f"ads_{a.rows}")
        print(f"Wrote {a.rows:,} simulated rows to {out}")
    elif a.cmd == "benchmark":
        res = bm.run_benchmark(ADS_DIR, a.sizes, a.engines, a.repeats)
        bm.save(res, ADS_DIR / "benchmark_results.json")
        print(f"Saved {ADS_DIR / 'benchmark_results.json'}")
        for k, v in res["skipped"].items():
            print(f"Skipped {k}: {v}")
    else:
        rec = recommend(simulate(a.rows), a.budget, incremental=not a.whole_budget)
        print(rec.table.round(2).to_string(index=False))
        print(f"\nExpected revenue from the new budget: Rs {rec.expected_revenue:,.0f} "
              f"({rec.uplift_pct:+.1f}% vs spreading it like current spend)\n{rec.note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
