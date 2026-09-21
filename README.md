# Mobile Network Traffic Forecasting - Milan (LSTM vs TCN vs Transformer)

One-step-ahead forecasting of Internet activity in Milan telecom grid squares (10-min resolution,
2013-11-01 .. 2014-01-01, 10,000 squares). Test week: **2013-12-16 .. 2013-12-22**.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Download the 62 daily files `sms-call-internet-mi-YYYY-MM-DD.txt` from the Harvard Dataverse
(doi:10.7910/DVN/EGZHFV) and put them in `data/raw/` (or set `TRAFFIC_RAW_DIR`). Do not rename them.

## Pipeline (run in order, from the repo root)
| Step | Command | Produces |
|---|---|---|
| 1. Build compact dataset | `python -m src.data_loading` | `data/processed/internet_10000x8928_float32.npy` (~340 MB) |
| 2. Memory evidence | `python -m src.memory_benchmark` | `outputs/memory/memory_benchmark.csv` |
| 3. EDA | `python -m src.eda` | `outputs/eda/*.png, *.json, *.csv` |
| 4. Tune (validation week only) | `python -m src.tune` | `outputs/tuning/experiment_log.csv`, `best_<model>.json` |
| 5. Final runs + tables + plots | `python -m src.run_final --seeds 3` | `outputs/results/` |

Optional environment variables: `TRAFFIC_RAW_DIR`, `TRAFFIC_PROC_DIR`, `TRAFFIC_OUT_DIR`, `TRAFFIC_N_SQUARES`.

## Design summary
* **Split (temporal, no shuffling across periods):** train 1 Nov-8 Dec, validation 9-15 Dec (early stopping and
  tuning), test 16-22 Dec. Scaler statistics use the training period only.
* **Input representation:** window of L past steps x 5 features (log1p+z-scored traffic, sin/cos time-of-day and
  day-of-week of the step being predicted). Target: next-step traffic.
* **Models:** LSTM (recurrent), TCN (dilated causal convolutions), Transformer encoder (self-attention).
  Reference baselines: persistence, seasonal-naive (1 day / 1 week).
* **Tuning:** staged grid search (window -> capacity -> optimisation) on the top-1 area's validation week; every run is
  logged with a `notes` column for your written justification.
* **Metrics:** MAE, RMSE, MAPE, sMAPE on the original scale; mean ± std over seeds.
* **Timing:** train time = epoch loop wall-clock (incl. validation); inference = full test week batched, plus
  batch-size-1 latency. Hardware in `outputs/results/hardware.json`.

## Smoke test without the real data
```bash
export TRAFFIC_RAW_DIR=/tmp/raw TRAFFIC_PROC_DIR=/tmp/proc TRAFFIC_OUT_DIR=/tmp/out TRAFFIC_N_SQUARES=5000
python tests/make_synthetic.py && python -m src.data_loading && python -m src.eda
python -m src.tune --quick && python -m src.run_final --seeds 1 --max-epochs 3
```

## Data reference
G. Barlacchi et al., "A multi-source dataset of urban life in the city of Milan and the Province of Trentino,"
Sci. Data 2, 150055 (2015). https://doi.org/10.1038/sdata.2015.55
