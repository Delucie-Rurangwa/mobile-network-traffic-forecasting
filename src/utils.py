"""Small helpers: seeding, hardware description, memory measurement, data access."""
import json
import platform
import random

import numpy as np

from src import config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def peak_rss_mb() -> float:
    """Peak resident memory of the current process in MB (cross-platform)."""
    import psutil
    info = psutil.Process().memory_info()
    if hasattr(info, "peak_wset"):                      # Windows
        return info.peak_wset / 2 ** 20
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2 ** 20 if platform.system() == "Darwin" else r / 1024


def hardware_info() -> dict:
    import psutil
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 2 ** 30, 1),
        "python": platform.python_version(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def save_json(obj, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def load_traffic(mmap: bool = True) -> np.ndarray:
    """(N_SQUARES, T) float32 matrix. Memory-mapped by default, so nothing is loaded until sliced."""
    return np.load(config.TRAFFIC_NPY, mmap_mode="r" if mmap else None)


def load_area(square_id: int) -> np.ndarray:
    """One area's full series as an in-memory float32 vector (square ids are 1-based)."""
    return np.array(load_traffic()[square_id - 1], dtype=np.float32)


def load_totals() -> np.ndarray:
    return np.load(config.TOTALS_NPY)


def top_areas(k: int = 3) -> list:
    totals = load_totals()
    return [int(i) + 1 for i in np.argsort(totals)[::-1][:k]]
