"""Staged (coordinate-wise) grid search on the VALIDATION week (2013-12-09..15) of the top-1 area.

Each stage grid-searches a small group of hyperparameters while everything else is fixed at the best
values of previous stages: 1) input window  2) model capacity  3) optimisation/regularisation.
Every experiment is appended to outputs/tuning/experiment_log.csv (fill the `notes` column with your
reasoning for the next step - the assignment requires documented, justified iterations).
Best parameters -> outputs/tuning/best_<model>.json.  Test data is never touched here.

    python -m src.tune --models lstm tcn transformer
    python -m src.tune --quick        # tiny smoke run
"""
import argparse
import itertools
import json
from datetime import datetime

import numpy as np
import pandas as pd

from src import config, utils
from src.datasets import LogStdScaler, prepare_splits
from src.evaluate import metrics
from src.train import fit, get_device, predict

STAGES = {
    "lstm": [("1_window", {"window": [36, 72, 144, 288]}),
             ("2_capacity", {"hidden": [32, 64, 128], "layers": [1, 2]}),
             ("3_optim", {"lr": [3e-3, 1e-3, 3e-4], "dropout": [0.0, 0.1, 0.3]})],
    "tcn": [("1_window", {"window": [36, 72, 144, 288]}),
            ("2_capacity", {"channels": [16, 32, 64], "levels": [4, 5, 6]}),
            ("3_optim", {"lr": [3e-3, 1e-3, 3e-4], "dropout": [0.0, 0.1, 0.3]})],
    "transformer": [("1_window", {"window": [36, 72, 144]}),
                    ("2_capacity", {"d_model": [32, 64], "layers": [1, 2, 3]}),
                    ("3_optim", {"lr": [1e-3, 3e-4], "dropout": [0.0, 0.1, 0.3]})],
}
LOG = config.OUT_DIR / "tuning" / "experiment_log.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=config.MODELS)
    ap.add_argument("--area", type=int, default=None, help="default: highest-traffic area")
    ap.add_argument("--metric", default="RMSE", choices=["MAE", "RMSE"])
    ap.add_argument("--max-epochs", type=int, default=config.MAX_EPOCHS)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    area = a.area or utils.top_areas(1)[0]
    series = utils.load_area(area)
    scaler = LogStdScaler().fit(series[:config.VAL_START])
    device = get_device()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(LOG) if LOG.exists() else pd.DataFrame()
    cache = {}
    if len(done):
        cache = {(r.model, r.area, r.params): r for r in done.itertuples()}

    for model in a.models:
        current = dict(config.DEFAULT_PARAMS[model])
        for stage, grid in STAGES[model]:
            grid = {k: v[:2] if a.quick else v for k, v in grid.items()}
            results = []
            for combo in itertools.product(*grid.values()):
                p = {**current, **dict(zip(grid.keys(), combo))}
                key = (model, area, json.dumps(p, sort_keys=True))
                if key in cache:
                    r = cache[key]; vm = dict(MAE=r.val_MAE, RMSE=r.val_RMSE)
                else:
                    splits = prepare_splits(series, scaler, p["window"])
                    net, info = fit(model, p, splits, device, max_epochs=3 if a.quick else a.max_epochs)
                    X, _, idx = splits["val"]
                    vm = metrics(series[idx], scaler.inverse(predict(net, X, device)))
                    row = dict(time=datetime.now().isoformat(timespec="seconds"), model=model, area=area,
                               stage=stage, params=json.dumps(p, sort_keys=True), val_MAE=vm["MAE"],
                               val_RMSE=vm["RMSE"], val_MAPE=vm["MAPE"], best_epoch=info["best_epoch"],
                               epochs_run=info["epochs_run"], train_seconds=round(info["train_seconds"], 2),
                               n_params=info["n_params"], notes="")
                    pd.DataFrame([row]).to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
                    print(f"[{model}:{stage}] {dict(zip(grid.keys(), combo))} -> val MAE {vm['MAE']:.2f} "
                          f"RMSE {vm['RMSE']:.2f} ({info['best_epoch']}/{info['epochs_run']} ep)")
                results.append((vm[a.metric], p))
            current = min(results, key=lambda t: t[0])[1]
            print(f"  => best after {stage}: { {k: current[k] for k in grid} }")
        utils.save_json(current, config.OUT_DIR / "tuning" / f"best_{model}.json")
        print(f"[{model}] final params: {current}")


if __name__ == "__main__":
    main()
