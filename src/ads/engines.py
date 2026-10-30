"""The same job on three engines: total impressions, clicks, spend, conversions and revenue per
(platform, campaign), then KPIs.

* ``pandas_engine``     - single-machine DataFrame ``groupby``.
* ``mapreduce_engine``  - a real map -> shuffle -> reduce: mappers run in parallel worker processes,
                          each reads one partition and emits per-key partial sums (with a combiner);
                          the shuffle groups partials by key; reducers add them up.
* ``spark_engine``      - Apache Spark (local mode) DataFrame ``groupBy``; needs ``pyspark`` + Java.

Each returns ``(totals_frame, timings)`` where timings separate ``load_s`` / ``compute_s`` where the
engine allows it. All three must give the same numbers - ``benchmark.py`` checks that.
"""
from __future__ import annotations

import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from .metrics import SUM_COLS, add_kpis

KEYS = ["platform", "campaign"]
NEEDED = KEYS + SUM_COLS


def _parts(data_dir: Path) -> list[Path]:
    parts = sorted(Path(data_dir).glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No part-*.parquet files in {data_dir}")
    return parts


def _finish(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(KEYS).reset_index(drop=True)
    return add_kpis(frame)


# ---------------------------------------------------------------- pandas
def pandas_engine(data_dir: Path):
    t0 = time.perf_counter()
    df = pd.concat([pd.read_parquet(p, columns=NEEDED) for p in _parts(data_dir)], ignore_index=True)
    t1 = time.perf_counter()
    out = df.groupby(KEYS, observed=True)[SUM_COLS].sum().reset_index()
    out[KEYS] = out[KEYS].astype(str)
    t2 = time.perf_counter()
    return _finish(out), {"load_s": t1 - t0, "compute_s": t2 - t1, "total_s": t2 - t0}


# ------------------------------------------------------------- MapReduce
def _map_partition(path: str) -> dict:
    """MAP + COMBINE: read one partition, emit {(platform, campaign): [imp, clicks, spend, conv, rev]}."""
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=NEEDED).to_pydict()
    acc: dict = defaultdict(lambda: [0, 0, 0.0, 0, 0.0])
    for p, c, i, k, s, v, r in zip(t["platform"], t["campaign"], t["impressions"], t["clicks"],
                                   t["spend"], t["conversions"], t["revenue"]):
        a = acc[(p, c)]
        a[0] += i
        a[1] += k
        a[2] += s
        a[3] += v
        a[4] += r
    return dict(acc)


def _reduce_key(item):
    """REDUCE: add the partial sums for one key."""
    key, partials = item
    return key, [sum(col) for col in zip(*partials)]


def mapreduce_engine(data_dir: Path, workers: int = 2):
    t0 = time.perf_counter()
    paths = [str(p) for p in _parts(data_dir)]
    if workers > 1 and len(paths) > 1:
        with Pool(min(workers, len(paths))) as pool:
            mapped = pool.map(_map_partition, paths)
    else:
        mapped = [_map_partition(p) for p in paths]
    t1 = time.perf_counter()
    shuffled: dict = defaultdict(list)                  # SHUFFLE: group partials by key
    for part in mapped:
        for key, vals in part.items():
            shuffled[key].append(vals)
    reduced = [_reduce_key(item) for item in shuffled.items()]
    t2 = time.perf_counter()
    rows = [{"platform": k[0], "campaign": k[1], **dict(zip(SUM_COLS, v))} for k, v in reduced]
    return _finish(pd.DataFrame(rows)), {"load_s": t1 - t0, "compute_s": t2 - t1, "total_s": t2 - t0,
                                         "note": "load_s = parallel map phase (includes reading)"}


# ----------------------------------------------------------------- Spark
def spark_available() -> bool:
    try:
        import pyspark  # noqa: F401
    except ImportError:
        return False
    import shutil
    return shutil.which("java") is not None


def spark_session(cores: int = 2):
    from pyspark.sql import SparkSession
    return (SparkSession.builder.master(f"local[{cores}]").appName("ads-budget-optimizer")
            .config("spark.ui.enabled", "false").config("spark.sql.shuffle.partitions", "8")
            .config("spark.driver.memory", "2g").getOrCreate())


def spark_engine(data_dir: Path, session=None):
    """Spark job time excludes JVM start-up, which is returned separately as ``startup_s``."""
    from pyspark.sql import functions as F
    t0 = time.perf_counter()
    spark = session or spark_session()
    startup = time.perf_counter() - t0
    t1 = time.perf_counter()
    sdf = spark.read.option("pathGlobFilter", "part-*.parquet").parquet(str(Path(data_dir))).select(*NEEDED)
    agg = sdf.groupBy(*KEYS).agg(*[F.sum(c).alias(c) for c in SUM_COLS])
    out = agg.toPandas()                                 # action: triggers read + compute
    t2 = time.perf_counter()
    return _finish(out), {"load_s": 0.0, "compute_s": t2 - t1, "total_s": t2 - t1, "startup_s": startup,
                          "note": "read and compute are lazy and fused in Spark; startup_s is JVM start"}


ENGINES = {"pandas": pandas_engine, "mapreduce": mapreduce_engine, "spark": spark_engine}
