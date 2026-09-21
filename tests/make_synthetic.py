"""Generate a SMALL synthetic dataset in the exact raw format (for smoke-testing the pipeline only).
    TRAFFIC_RAW_DIR=/tmp/raw TRAFFIC_N_SQUARES=5000 python tests/make_synthetic.py
Only ~42 squares carry traffic; the rest are absent (=0), as in sparse areas of the real data."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, data_loading

rng = np.random.default_rng(0)
squares = np.array(list(range(1, 41)) + [4159, 4556])
base = rng.uniform(50, 500, len(squares)); base[3] = 3000; base[10] = 2000; base[20] = 1500
config.RAW_DIR.mkdir(parents=True, exist_ok=True)
for day in range(config.N_DAYS):
    date = config.START_DATE + pd.Timedelta(days=day)
    slots = np.arange(config.SLOTS_PER_DAY)
    weekend = date.dayofweek >= 5
    prof = 1 + 0.6 * np.sin(2 * np.pi * (slots / 144 - 0.35)) - (0.25 if weekend else 0)
    S, SL = np.meshgrid(np.arange(len(squares)), slots, indexing="ij")
    val = base[S] * prof[SL] * rng.lognormal(0, 0.15, S.shape)
    ts = config.START_MS + (day * 144 + SL) * config.INTERVAL_MS
    frames = []
    for cc, frac in [(0, 0.7), (39, 0.3)]:
        v = val * frac
        v = np.where(rng.random(v.shape) < 0.03, np.nan, v)
        frames.append(pd.DataFrame({"a": squares[S].ravel(), "b": ts.ravel(), "c": cc, "d": np.nan, "e": np.nan,
                                    "f": np.nan, "g": np.nan, "h": v.ravel()}))
    pd.concat(frames).to_csv(data_loading.raw_path(day), sep="\t", header=False, index=False)
print("synthetic files written to", config.RAW_DIR)
