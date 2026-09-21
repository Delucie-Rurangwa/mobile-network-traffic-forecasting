"""Central configuration: paths, dataset geometry, split definition, default hyperparameters."""
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = Path(os.environ.get("TRAFFIC_RAW_DIR", ROOT / "data" / "raw"))
PROC_DIR = Path(os.environ.get("TRAFFIC_PROC_DIR", ROOT / "data" / "processed"))
OUT_DIR = Path(os.environ.get("TRAFFIC_OUT_DIR", ROOT / "outputs"))

# ---- dataset geometry (Milan telecom dataset, Barlacchi et al. 2015) -------------------------
N_SQUARES = int(os.environ.get("TRAFFIC_N_SQUARES", 10_000))   # 100 x 100 grid
N_DAYS = 62                                                     # 2013-11-01 .. 2014-01-01
SLOTS_PER_DAY = 144                                             # 10-minute intervals
T = N_DAYS * SLOTS_PER_DAY
START_DATE = pd.Timestamp("2013-11-01")
START_MS = 1_383_260_400_000          # 2013-11-01 00:00 CET (UTC+1) in epoch milliseconds
INTERVAL_MS = 600_000
TIME_INDEX = pd.date_range(START_DATE, periods=T, freq="10min")  # local (CET) wall-clock time

TRAFFIC_NPY = PROC_DIR / f"internet_{N_SQUARES}x{T}_float32.npy"
TOTALS_NPY = PROC_DIR / f"totals_{N_SQUARES}.npy"
META_JSON = PROC_DIR / "build_meta.json"


def day_index(date_str: str) -> int:
    return (pd.Timestamp(date_str) - START_DATE).days


# ---- temporal split (all in target-index space) ------------------------------------------------
# train: 2013-11-01 .. 2013-12-08 | validation: 2013-12-09 .. 2013-12-15 | test: 2013-12-16 .. 2013-12-22
VAL_START = day_index("2013-12-09") * SLOTS_PER_DAY
TEST_START = day_index("2013-12-16") * SLOTS_PER_DAY
TEST_END = day_index("2013-12-23") * SLOTS_PER_DAY      # exclusive

SEED = 42

# ---- default hyperparameters (starting point of the staged tuning in tune.py) -------------------
DEFAULT_PARAMS = {
    "lstm": dict(window=144, hidden=64, layers=1, dropout=0.1, lr=1e-3, batch_size=64),
    "tcn": dict(window=144, channels=32, levels=5, kernel_size=3, dropout=0.1, lr=1e-3, batch_size=64),
    "transformer": dict(window=72, d_model=64, n_heads=4, layers=2, dim_ff=128, dropout=0.1, lr=1e-3,
                        batch_size=64),
}
MAX_EPOCHS = 60
PATIENCE = 8
MODELS = ["lstm", "tcn", "transformer"]
