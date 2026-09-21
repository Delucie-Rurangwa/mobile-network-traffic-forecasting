"""Input representation: scaling, calendar features and sliding windows.

Per time step i the model sees 5 features:
  [ z_i , sin/cos(time-of-day of i+1) , sin/cos(day-of-week of i+1) ]
where z_i = (log1p(x_i) - mu) / sigma, with mu, sigma estimated on the TRAINING period only.
Calendar features describe the step being predicted (t+1), which is known in advance, so no leakage.
A sample for target index j is the window data[j-L : j] and the label z_j.
"""
import numpy as np
import pandas as pd

from src import config


class LogStdScaler:
    def fit(self, x):
        z = np.log1p(x.astype(np.float64))
        self.mean, self.std = float(z.mean()), float(z.std() + 1e-8)
        return self

    def transform(self, x):
        return ((np.log1p(x.astype(np.float64)) - self.mean) / self.std).astype(np.float32)

    def inverse(self, z):
        return np.clip(np.expm1(np.asarray(z, dtype=np.float64) * self.std + self.mean), 0, None)


def next_step_time_features():
    nxt = config.TIME_INDEX + pd.Timedelta(minutes=10)
    tod = (nxt.hour * 60 + nxt.minute) / 1440.0
    dow = nxt.dayofweek / 7.0
    return np.stack([np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod),
                     np.sin(2 * np.pi * dow), np.cos(2 * np.pi * dow)], axis=1).astype(np.float32)


def prepare_splits(series, scaler, L):
    """Return {'train'|'val'|'test': (X[n,L,5], y[n], target_idx[n])} in normalised space."""
    data = np.concatenate([scaler.transform(series)[:, None], next_step_time_features()], axis=1)
    ranges = {"train": (L, config.VAL_START), "val": (config.VAL_START, config.TEST_START),
              "test": (config.TEST_START, config.TEST_END)}
    out = {}
    for name, (a, b) in ranges.items():
        idx = np.arange(a, b)
        X = data[idx[:, None] + np.arange(-L, 0)[None, :]]
        out[name] = (X.astype(np.float32), data[idx, 0].astype(np.float32), idx)
    return out
