"""Measure memory before/after optimisation on ONE raw daily file (evidence for report Section 3).

A: naive          pd.read_csv(all 8 columns, default dtypes, whole file)
B: cols + dtypes  usecols + narrow dtypes, whole file
C: C + chunked    chunked reading + bincount aggregation (the pipeline used in data_loading.py)
Each variant runs in a fresh subprocess so that peak RSS is not polluted by the others.
The full-dataset figures are linear extrapolations (x N_DAYS) and are labelled as such.
"""
import argparse
import multiprocessing as mp
import time

import numpy as np
import pandas as pd

from src import config, data_loading, utils

ALL_COLS = ["sq", "ts", "country", "smsin", "smsout", "callin", "callout", "internet"]


def _variant(name, path, q):
    t0 = time.perf_counter()
    base = utils.peak_rss_mb()
    if name == "A_naive":
        df = pd.read_csv(path, sep="\t", header=None, names=ALL_COLS)
        df_mb = df.memory_usage(deep=True).sum() / 2 ** 20
        rows = len(df)
    elif name == "B_cols_dtypes":
        df = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 7], names=["sq", "ts", "internet"],
                         dtype={"sq": "uint16", "ts": "int64", "internet": "float32"})
        df_mb = df.memory_usage(deep=True).sum() / 2 ** 20
        rows = len(df)
    else:
        arr, _ = data_loading.aggregate_day(path, day=0)
        df_mb = arr.nbytes / 2 ** 20          # what is finally kept in memory for the day
        rows = -1
    q.put(dict(variant=name, in_memory_mb=df_mb, peak_rss_mb=utils.peak_rss_mb(), base_rss_mb=base,
               seconds=time.perf_counter() - t0, rows=rows))


def main(path):
    ctx = mp.get_context("spawn")
    res = []
    for name in ["A_naive", "B_cols_dtypes", "C_chunked_aggregated"]:
        q = ctx.Queue()
        p = ctx.Process(target=_variant, args=(name, path, q))
        p.start()
        res.append(q.get())
        p.join()
    df = pd.DataFrame(res)
    df["peak_rss_reduction_vs_A_%"] = 100 * (1 - df["peak_rss_mb"] / df.loc[0, "peak_rss_mb"])
    df["in_memory_x_days_(extrapolated_MB)"] = df["in_memory_mb"] * config.N_DAYS
    out = config.OUT_DIR / "memory"
    out.mkdir(parents=True, exist_ok=True)
    df.round(1).to_csv(out / "memory_benchmark.csv", index=False)
    print(df.round(1).to_string(index=False))
    full = config.N_SQUARES * config.T
    print(f"\nFinal dataset: {full * 4 / 2 ** 20:.0f} MB as float32 vs {full * 8 / 2 ** 20:.0f} MB as float64 "
          f"({config.N_SQUARES}x{config.T}); stored as memmap so RAM use is per-slice.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(data_loading.raw_path(0)), help="one raw daily .txt file")
    main(ap.parse_args().file)
