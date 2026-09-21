"""Metrics (computed on the ORIGINAL traffic scale) and inference timing."""
import time

import numpy as np
import torch

from src.train import _sync, predict


def metrics(y, p, eps=1e-8):
    e = p - y
    nz = y > eps                                    # MAPE undefined where the truth is 0
    return dict(MAE=float(np.mean(np.abs(e))), RMSE=float(np.sqrt(np.mean(e ** 2))),
                MAPE=float(np.mean(np.abs(e[nz]) / y[nz]) * 100),
                sMAPE=float(np.mean(2 * np.abs(e) / (np.abs(y) + np.abs(p) + eps)) * 100))


def time_inference(model, X, device, repeats=5, n_single=200):
    """Batched: predict the whole test week (median of `repeats`). Single: batch-size-1 latency."""
    predict(model, X[:64], device)                                # warm-up
    times = []
    for _ in range(repeats):
        _sync(device); t0 = time.perf_counter()
        predict(model, X, device)
        _sync(device); times.append(time.perf_counter() - t0)
    n = min(n_single, len(X))
    model.eval(); _sync(device); t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(n):
            model(torch.from_numpy(X[i:i + 1]).to(device))
    _sync(device)
    return dict(infer_batch_total_s=float(np.median(times)), infer_batch_ms_per_sample=float(np.median(times) / len(X) * 1e3),
                infer_single_ms=float((time.perf_counter() - t0) / n * 1e3))
