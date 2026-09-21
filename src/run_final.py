"""Final experiments (report Sections Methodology / Results): train the three models on each of the
top-3 areas, forecast the test week 2013-12-16..22 one step ahead using the TRUE history, and write
tables, the 9 overlay plots, timing statistics and a failure analysis.

    python -m src.run_final --seeds 3
Hyperparameters are read from outputs/tuning/best_<model>.json (created by tune.py), else defaults.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config, utils
from src.datasets import LogStdScaler, prepare_splits
from src.evaluate import metrics, time_inference
from src.train import fit, get_device, predict

OUT = config.OUT_DIR / "results"
NAMES = {"lstm": "LSTM", "tcn": "TCN", "transformer": "Transformer",
         "persistence": "Persistence (x[t])", "seasonal_naive_1d": "Seasonal naive (1 day)",
         "seasonal_naive_1w": "Seasonal naive (1 week)"}


def load_params(model):
    f = config.OUT_DIR / "tuning" / f"best_{model}.json"
    if f.exists():
        return json.load(open(f))
    print(f"[warn] {f.name} not found -> using default parameters for {model}")
    return dict(config.DEFAULT_PARAMS[model])


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def plot_overlay(t, y, p, title, path):
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.plot(t, y, color="black", lw=1.0, label="actual")
    ax.plot(t, p, color="tab:red", lw=1.0, alpha=0.85, label="predicted")
    ax.set(title=title, xlabel="time", ylabel="Internet activity"); ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def failure_analysis(area, t, y, preds, best, tag_dir):
    err = {m: np.abs(preds[m] - y) for m in preds}
    df = pd.DataFrame({"t": t, **{m: err[m] for m in err}}); df["day"] = df["t"].dt.date; df["hour"] = df["t"].dt.hour
    df.groupby("day")[list(err)].mean().round(2).to_csv(tag_dir / f"area{area}_MAE_by_day.csv")
    df.groupby("hour")[list(err)].mean().round(2).to_csv(tag_dir / f"area{area}_MAE_by_hour.csv")
    worst = df.groupby("day")[best].mean().idxmax()
    top = df.nlargest(10, best)[["t", best]].assign(actual=y[df.nlargest(10, best).index.values],
                                                    predicted=preds[best][df.nlargest(10, best).index.values])
    top.to_csv(tag_dir / f"area{area}_top10_errors_{best}.csv", index=False, float_format="%.2f")
    mask = (df["day"] == worst).values
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t[mask], y[mask], "k", lw=1.4, label="actual")
    for m in preds:
        ax.plot(t[mask], preds[m][mask], lw=1, label=NAMES[m])
    ax.set(title=f"Area {area}: worst day for {NAMES[best]} ({worst})", xlabel="time", ylabel="activity")
    ax.legend(ncol=3, fontsize=8); fig.tight_layout()
    fig.savefig(tag_dir / f"area{area}_worst_day.png", dpi=150); plt.close(fig)
    return str(worst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--areas", nargs="+", type=int, default=None)
    ap.add_argument("--models", nargs="+", default=config.MODELS)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-epochs", type=int, default=config.MAX_EPOCHS)
    a = ap.parse_args()

    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    (OUT / "failure").mkdir(parents=True, exist_ok=True)
    areas = a.areas or utils.top_areas(3)
    device = get_device()
    hw = utils.hardware_info(); hw["torch_device_used"] = str(device)
    utils.save_json(hw, OUT / "hardware.json")
    print("areas:", areas, "| device:", device)

    timing_rows, all_tables = [], {}
    t_idx = np.arange(config.TEST_START, config.TEST_END)
    times = config.TIME_INDEX[t_idx]
    for area in areas:
        series = utils.load_area(area)
        scaler = LogStdScaler().fit(series[:config.VAL_START])       # train-only statistics
        y = series[t_idx].astype(np.float64)
        preds, rows, hist_store = {}, [], {}
        for model in a.models:
            p = load_params(model)
            splits = prepare_splits(series, scaler, p["window"])
            X_test = splits["test"][0]
            seed_metrics = []
            for seed in range(a.seeds):
                net, info = fit(model, p, splits, device, seed=config.SEED + seed, max_epochs=a.max_epochs)
                pred = scaler.inverse(predict(net, X_test, device))
                seed_metrics.append(metrics(y, pred))
                tm = time_inference(net, X_test, device)
                timing_rows.append(dict(area=area, model=model, seed=seed, train_seconds=info["train_seconds"],
                                        epochs_run=info["epochs_run"], best_epoch=info["best_epoch"],
                                        n_params=info["n_params"], **tm))
                if seed == 0:
                    preds[model] = pred
                    hist_store[model] = info["history"]
                print(f"area {area} {model} seed {seed}: {seed_metrics[-1]}")
            df = pd.DataFrame(seed_metrics)
            rows.append(dict(model=NAMES[model], **{f"{k}_mean": df[k].mean() for k in df},
                             **{f"{k}_std": df[k].std(ddof=0) for k in df}))
            plot_overlay(times, y, preds[model], f"Area {area} - {NAMES[model]}: actual vs predicted, 16-22 Dec 2013",
                         OUT / "plots" / f"area{area}_{model}.png")
        base = {"persistence": series[t_idx - 1], "seasonal_naive_1d": series[t_idx - config.SLOTS_PER_DAY],
                "seasonal_naive_1w": series[t_idx - 7 * config.SLOTS_PER_DAY]}
        for name, pr in base.items():
            rows.append(dict(model=NAMES[name], **{f"{k}_mean": v for k, v in metrics(y, pr.astype(np.float64)).items()},
                             **{f"{k}_std": 0.0 for k in ["MAE", "RMSE", "MAPE", "sMAPE"]}))
        tab = pd.DataFrame(rows)
        tab.round(3).to_csv(OUT / f"metrics_area{area}.csv", index=False)
        pretty = pd.DataFrame({"Model": tab["model"], **{k: [f"{m:.2f} ± {s:.2f}" for m, s in zip(tab[f"{k}_mean"], tab[f"{k}_std"])]
                                                        for k in ["MAE", "RMSE", "MAPE", "sMAPE"]}})
        pretty = pretty.rename(columns={"MAPE": "MAPE (%)", "sMAPE": "sMAPE (%)"})
        (OUT / f"metrics_area{area}.md").write_text(md_table(pretty))
        all_tables[area] = tab
        # 3-panel summary grid + failure analysis on the best model (lowest mean RMSE among the 3 nets)
        nets = tab[tab["model"].isin([NAMES[m] for m in a.models])]
        best = a.models[int(np.argmin(nets["RMSE_mean"].values))]
        preds_plus = {**preds, "persistence": base["persistence"].astype(np.float64)}
        pd.DataFrame({"time": times, "actual": y, **preds_plus}).to_csv(
            OUT / f"predictions_area{area}.csv", index=False, float_format="%.3f")
        worst_day = failure_analysis(area, times, y, preds_plus, best, OUT / "failure")
        pd.DataFrame(hist_store).to_json(OUT / f"training_history_area{area}.json")
        print(f"area {area}: best={best}, worst day={worst_day}")

    tr = pd.DataFrame(timing_rows)
    tr.round(4).to_csv(OUT / "timing_raw.csv", index=False)
    agg = tr.groupby("model").agg(train_s=("train_seconds", "mean"), train_s_std=("train_seconds", "std"),
                                  epochs=("epochs_run", "mean"), params=("n_params", "first"),
                                  infer_week_s=("infer_batch_total_s", "mean"),
                                  infer_ms_per_sample_batched=("infer_batch_ms_per_sample", "mean"),
                                  infer_ms_single=("infer_single_ms", "mean")).round(3)
    agg["train_s_per_epoch"] = (agg["train_s"] / agg["epochs"]).round(3)
    agg.to_csv(OUT / "timing_summary.csv")
    print(agg.to_string())
    print("\nTiming = mean over all areas and seeds; train time = epoch loop incl. validation; "
          "inference = full 1008-step test week, batched (median of 5) and batch-size-1 latency.")


if __name__ == "__main__":
    main()
