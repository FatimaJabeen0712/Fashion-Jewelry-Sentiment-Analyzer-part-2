"""Pandas vs MapReduce vs Spark over growing data sizes (10k -> 100k -> 1M -> 10M rows)."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

from .engines import ENGINES, spark_available, spark_session
from .metrics import SUM_COLS
from .simulate import write_dataset

DEFAULT_SIZES = (10_000, 100_000, 1_000_000, 10_000_000)


def results_agree(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    a = a.sort_values(["platform", "campaign"]).reset_index(drop=True)
    b = b.sort_values(["platform", "campaign"]).reset_index(drop=True)
    if not a[["platform", "campaign"]].equals(b[["platform", "campaign"]]):
        return False
    return bool(np.allclose(a[SUM_COLS].to_numpy(float), b[SUM_COLS].to_numpy(float), rtol=1e-9, atol=1e-6))


def run_benchmark(data_root: Path, sizes=DEFAULT_SIZES, engines=("pandas", "mapreduce", "spark"),
                  repeats: int = 3, seed: int = 42, progress=print) -> dict:
    """Returns {"rows": [...one per size x engine...], "skipped": {...}}. Reports the *median* of repeats."""
    data_root = Path(data_root)
    skipped, rows = {}, []
    use = [e for e in engines if e != "spark" or spark_available()]
    if "spark" in engines and "spark" not in use:
        skipped["spark"] = "pyspark or Java not found"
    session = spark_session() if "spark" in use else None
    if session is not None:   # warm-up so the first size is not charged for JIT / class loading
        write_dataset(1000, data_root / "warmup", seed)
        ENGINES["spark"](data_root / "warmup", session)
    try:
        for n in sizes:
            d = write_dataset(n, data_root / f"ads_{n}", seed)
            reference = None
            for name in use:
                reps = repeats if n <= 1_000_000 else max(1, min(repeats, 2))
                runs = []
                for _ in range(reps):
                    out, tm = ENGINES[name](d, session) if name == "spark" else ENGINES[name](d)
                    runs.append(tm)
                if reference is None:
                    reference = out
                med = {k: statistics.median(r[k] for r in runs) for k in ("load_s", "compute_s", "total_s")}
                rows.append({"rows": n, "engine": name, **{k: round(v, 3) for k, v in med.items()},
                             "repeats": reps, "matches_pandas": results_agree(reference, out)})
                progress(f"{n:>10,} rows  {name:<9} {med['total_s']:.2f}s")
    finally:
        if session is not None:
            session.stop()
    return {"rows": rows, "skipped": skipped}


def save(result: dict, path: Path) -> None:
    Path(path).write_text(json.dumps(result, indent=2), encoding="utf-8")
