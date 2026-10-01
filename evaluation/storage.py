"""Result storage (Phase 6): reproducible images + metric data per experiment.

Folder layout (root defaults to ./data):
    data/original/<image>.png
    data/noisy/<experiment_id>.png
    data/results/wavelet/<experiment_id>.png
    data/results/shearlet/<experiment_id>.png
    data/results/tables/<experiment_id>.csv     one comparison table each
    data/results/results.csv                    master file, one row per
                                                (experiment, method)

Experiment id:  <image>_<noise>_<level>_seed<seed>   e.g. camera_gaussian_20_seed42
All files are written to a temp file first and then moved into place, so a
crash can never leave a half-written (corrupted) result file behind.
"""
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import config

MASTER_COLUMNS = ["experiment_id", "image", "noise_type", "noise_level", "seed",
                  "method", "mse", "psnr", "ssim", "processing_time", "params"]


# --- naming -------------------------------------------------------------
def _safe(name) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(name))


def experiment_id(image: str, noise_type: str, level, seed: int) -> str:
    lvl = f"{level:g}".replace(".", "p")
    return f"{_safe(image)}_{noise_type}_{lvl}_seed{seed}"


def effective_params(sigma=None, overrides=None) -> dict:
    """Parameters each method actually ran with: config defaults + overrides."""
    overrides = overrides or {}
    wavelet = {"wavelet": config.WAVELET, "level": config.WAVELET_LEVEL,
               "mode": config.WAVELET_THRESHOLD_MODE,
               "method": config.WAVELET_THRESHOLD_METHOD,
               "threshold_scale": config.WAVELET_THRESHOLD_SCALE, "sigma": sigma}
    shearlet = {"scales": config.SHEARLET_SCALES,
                "factor": config.SHEARLET_THRESHOLD_FACTOR,
                "mode": config.SHEARLET_THRESHOLD_MODE, "sigma": sigma}
    wavelet.update({k: v for k, v in overrides.get("wavelet", {}).items() if v is not None})
    shearlet.update({k: v for k, v in overrides.get("shearlet", {}).items() if v is not None})
    return {"wavelet": wavelet, "shearlet": shearlet}


# --- safe writing -------------------------------------------------------
def _atomic(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        writer(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def save_png(arr: np.ndarray, path) -> Path:
    """Save a 0-255 float image as an 8-bit PNG (clipped and rounded)."""
    path = Path(path)
    out = np.clip(np.rint(np.asarray(arr, dtype=np.float64)), 0, 255).astype(np.uint8)
    _atomic(path, lambda tmp: Image.fromarray(out, mode="L").save(tmp, format="PNG"))
    return path


def save_csv(df: pd.DataFrame, path) -> Path:
    path = Path(path)
    _atomic(path, lambda tmp: df.to_csv(tmp, index=False))
    return path


# --- main entry points --------------------------------------------------
def save_experiment(result, image: str, noise_type: str, level, seed: int,
                    sigma=None, params=None, root="data") -> dict:
    """Store everything about one finished comparison.

    result : evaluation.comparison.ComparisonResult
    Returns the paths written. Running the same experiment again replaces
    its rows in the master CSV instead of duplicating them.
    """
    root = Path(root)
    exp = experiment_id(image, noise_type, level, seed)
    used = effective_params(sigma, params)

    paths = {
        "original": save_png(result.images["clean"], root / "original" / f"{_safe(image)}.png"),
        "noisy": save_png(result.images["noisy"], root / "noisy" / f"{exp}.png"),
    }
    for method in used:
        if method in result.images:
            paths[method] = save_png(result.images[method],
                                     root / "results" / method / f"{exp}.png")

    table = result.table.copy()
    paths["table"] = save_csv(table, root / "results" / "tables" / f"{exp}.csv")

    rows = table.assign(
        experiment_id=exp, image=image, noise_type=noise_type,
        noise_level=level, seed=seed,
        params=[json.dumps(used.get(m, {}), sort_keys=True) for m in table["method"]],
    )[MASTER_COLUMNS]

    master_path = root / "results" / "results.csv"
    if master_path.exists():
        old = pd.read_csv(master_path)
        old = old[old["experiment_id"] != exp]               # replace, don't duplicate
        rows = pd.concat([old, rows], ignore_index=True)
    paths["master"] = save_csv(rows, master_path)
    paths["experiment_id"] = exp
    return paths


def load_results(root="data") -> pd.DataFrame:
    """Read the master results CSV (empty frame if nothing saved yet)."""
    path = Path(root) / "results" / "results.csv"
    if not path.exists():
        return pd.DataFrame(columns=MASTER_COLUMNS)
    return pd.read_csv(path)