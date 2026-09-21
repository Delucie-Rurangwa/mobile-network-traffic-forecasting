"""Training loop with early stopping + prediction helper."""
import copy
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src import config, utils
from src.models import build_model, count_params


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def predict(model, X, device, batch_size=512):
    model.eval()
    out = [model(torch.from_numpy(X[i:i + batch_size]).to(device)).cpu().numpy()
           for i in range(0, len(X), batch_size)]
    return np.concatenate(out)


def fit(name, params, splits, device, seed=config.SEED, max_epochs=config.MAX_EPOCHS,
        patience=config.PATIENCE):
    """Adam + MSE on normalised targets, ReduceLROnPlateau, early stopping on validation MSE.
    Reported train time = wall-clock of the epoch loop (includes per-epoch validation)."""
    utils.set_seed(seed)
    model = build_model(name, params).to(device)
    Xtr, ytr, _ = splits["train"]
    Xva, yva, _ = splits["val"]
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                        batch_size=params["batch_size"], shuffle=True, generator=g)
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    lossf = torch.nn.MSELoss()
    best, best_state, best_epoch, bad = float("inf"), None, 0, 0
    hist = {"train_loss": [], "val_loss": []}
    _sync(device); t0 = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train(); tot = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(xb)
        vloss = float(np.mean((predict(model, Xva, device) - yva) ** 2))
        hist["train_loss"].append(tot / len(Xtr)); hist["val_loss"].append(vloss)
        sched.step(vloss)
        if vloss < best - 1e-6:
            best, best_state, best_epoch, bad = vloss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= patience:
                break
    _sync(device)
    seconds = time.perf_counter() - t0
    model.load_state_dict(best_state)
    info = dict(train_seconds=seconds, epochs_run=epoch, best_epoch=best_epoch, best_val_mse=best,
                n_params=count_params(model), history=hist)
    return model, info
