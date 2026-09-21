"""Memory-efficient ingestion of the raw daily files into ONE (N_SQUARES, T) float32 array.

Raw format (tab separated, no header): square_id, time_ms, country_code, smsin, smsout, callin,
callout, internet. There is one row per (square, interval, country code) and only where activity
was recorded, so rows must be SUMMED over country codes.

Memory strategy (see memory_benchmark.py for the measured effect):
  1. usecols=[0,1,7]      -> never materialise the 5 unused columns
  2. explicit narrow dtypes (uint16 / int64 / float32) instead of pandas defaults
  3. chunked reading      -> peak RAM is bounded by the chunk size, not the file size
  4. np.bincount aggregation into a dense (N_SQUARES x 144) buffer per day (no groupby, no index)
  5. float32 storage in a disk-backed .npy memmap -> later stages slice single areas on demand
"""
import argparse
import time

import numpy as np
import pandas as pd

from src import config, utils


def raw_path(day: int):
    date = config.START_DATE + pd.Timedelta(days=day)
    return config.RAW_DIR / f"sms-call-internet-mi-{date:%Y-%m-%d}.txt"


def aggregate_day(path, day: int, chunksize: int = 2_000_000):
    """Return (traffic[N,144] float32, observed[N,144] bool) for one raw daily file."""
    n = config.N_SQUARES * config.SLOTS_PER_DAY
    acc = np.zeros(n, dtype=np.float64)
    seen = np.zeros(n, dtype=bool)
    offset = day * config.SLOTS_PER_DAY
    reader = pd.read_csv(
        path, sep="\t", header=None, usecols=[0, 1, 7], names=["sq", "ts", "internet"],
        dtype={"sq": "uint16", "ts": "int64", "internet": "float32"}, chunksize=chunksize)
    for chunk in reader:
        sq = chunk["sq"].to_numpy().astype(np.int64) - 1
        slot = (chunk["ts"].to_numpy() - config.START_MS) // config.INTERVAL_MS - offset
        val = np.nan_to_num(chunk["internet"].to_numpy(), nan=0.0)   # NaN = no internet record
        ok = (sq >= 0) & (sq < config.N_SQUARES) & (slot >= 0) & (slot < config.SLOTS_PER_DAY)
        flat = sq[ok] * config.SLOTS_PER_DAY + slot[ok]
        acc += np.bincount(flat, weights=val[ok], minlength=n)
        seen[flat] = True
    shape = (config.N_SQUARES, config.SLOTS_PER_DAY)
    return acc.reshape(shape).astype(np.float32), seen.reshape(shape)


def build_dataset(force: bool = False) -> None:
    if config.TRAFFIC_NPY.exists() and not force:
        print(f"[skip] {config.TRAFFIC_NPY} already exists (use --force to rebuild)")
        return
    missing = [str(raw_path(d).name) for d in range(config.N_DAYS) if not raw_path(d).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} raw files missing in {config.RAW_DIR}, e.g. {missing[:3]}")
    config.PROC_DIR.mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap(config.TRAFFIC_NPY, mode="w+", dtype=np.float32,
                                   shape=(config.N_SQUARES, config.T))
    unobserved = 0
    t_start = time.perf_counter()
    for day in range(config.N_DAYS):
        t0 = time.perf_counter()
        arr, seen = aggregate_day(raw_path(day), day)
        mm[:, day * config.SLOTS_PER_DAY:(day + 1) * config.SLOTS_PER_DAY] = arr
        unobserved += int((~seen).sum())
        print(f"day {day + 1:2d}/{config.N_DAYS}  {raw_path(day).name}  {time.perf_counter() - t0:5.1f}s  "
              f"peak RSS {utils.peak_rss_mb():7.0f} MB")
    mm.flush()
    del mm

    traffic = np.load(config.TRAFFIC_NPY, mmap_mode="r")
    totals = np.zeros(config.N_SQUARES, dtype=np.float64)
    for s in range(0, config.N_SQUARES, 1000):       # stream over the memmap, never load it all
        totals[s:s + 1000] = traffic[s:s + 1000].sum(axis=1, dtype=np.float64)
    np.save(config.TOTALS_NPY, totals)
    utils.save_json({
        "shape": [config.N_SQUARES, config.T], "dtype": "float32",
        "file_size_mb": round(config.TRAFFIC_NPY.stat().st_size / 2 ** 20, 1),
        "cells_without_any_record": unobserved,
        "pct_cells_without_any_record": round(100 * unobserved / (config.N_SQUARES * config.T), 3),
        "build_seconds": round(time.perf_counter() - t_start, 1),
        "peak_rss_mb": round(utils.peak_rss_mb(), 1),
    }, config.META_JSON)
    print("done ->", config.TRAFFIC_NPY)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--force", action="store_true")
    build_dataset(ap.parse_args().force)
