"""Re-run the top-K logged configs per model with several seeds; pick by MEAN validation RMSE.

    python -m src.confirm --top-k 3 --seeds 3
Writes outputs/tuning/confirmation_log.csv and overwrites best_<model>.json with the confirmed winner
(the greedy single-seed result is kept in best_<model>_greedy.json).
"""
import argparse
import json
import shutil

import pandas as pd

from src import config, utils
from src.datasets import LogStdScaler, prepare_splits
from src.evaluate import metrics
from src.train import fit, get_device, predict

TUNE = config.OUT_DIR / "tuning"
LOG = TUNE / "experiment_log.csv"
OUT = TUNE / "confirmation_log.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=config.MODELS)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-epochs", type=int, default=config.MAX_EPOCHS)
    a = ap.parse_args()

    log = pd.read_csv(LOG)
    device = get_device()
    rows = []
    for model in a.models:
        sub = log[log.model == model]
        area = int(sub["area"].iloc[0])                       # the area used during tuning
        series = utils.load_area(area)
        scaler = LogStdScaler().fit(series[:config.VAL_START])
        cands = sub.drop_duplicates("params").nsmallest(a.top_k, "val_RMSE")
        model_rows = []
        for r in cands.itertuples():
            p = json.loads(r.params)
            splits = prepare_splits(series, scaler, p["window"])
            X, _, idx = splits["val"]
            ms = []
            for s in range(a.seeds):
                net, _ = fit(model, p, splits, device, seed=config.SEED + s, max_epochs=a.max_epochs)
                ms.append(metrics(series[idx], scaler.inverse(predict(net, X, device))))
            df = pd.DataFrame(ms)
            row = dict(model=model, params=r.params, single_seed_val_RMSE=r.val_RMSE,
                       val_MAE_mean=df["MAE"].mean(), val_RMSE_mean=df["RMSE"].mean(),
                       val_RMSE_std=df["RMSE"].std(ddof=0))
            model_rows.append(row)
            print(f"[{model}] {r.params}\n   1-seed RMSE {r.val_RMSE:.2f} -> {a.seeds}-seed "
                  f"{row['val_RMSE_mean']:.2f} ± {row['val_RMSE_std']:.2f}", flush=True)
        rows += model_rows
        best = min(model_rows, key=lambda d: d["val_RMSE_mean"])
        f = TUNE / f"best_{model}.json"
        backup = TUNE / f"best_{model}_greedy.json"
        if f.exists() and not backup.exists():
            shutil.copy(f, backup)
        f.write_text(json.dumps(json.loads(best["params"]), indent=2))
        print(f"==> {model} confirmed params: {best['params']}\n", flush=True)
    pd.DataFrame(rows).round(3).to_csv(OUT, index=False)


if __name__ == "__main__":
    main()