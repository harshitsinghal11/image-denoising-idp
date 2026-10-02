"""Logic behind the Streamlit app (Phase 9), kept free of any Streamlit code
so it can be tested on its own. The UI file only collects inputs, calls these
functions and displays what they return.
"""
import copy
import io
import json
import zipfile

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

import config
from algorithms.wavelet_denoising import estimate_sigma
from evaluation.comparison import METHODS, compare, difference_table
from evaluation.metrics import psnr
from evaluation.storage import effective_params, experiment_id
from experiments.experiment_runner import BENCHMARK_COLUMNS
from preprocessing.image_loader import prepare
from preprocessing.noise import add_noise

MIN_SIDE = 64                     # smallest uploaded image we accept (pixels)
SIZES = (128, 256, 512)           # working sizes offered in the UI
SIGMA_MODES = ("config", "estimate")

# label, level description, min, max, default, step
NOISE_UI = {
    "gaussian":    ("Gaussian", "σ, noise std on a 0–255 scale", 1.0, 60.0, 20.0, 1.0),
    "salt_pepper": ("Salt & pepper", "fraction of pixels corrupted", 0.01, 0.30, 0.05, 0.01),
    "speckle":     ("Speckle", "std of the multiplicative noise", 0.02, 0.60, 0.20, 0.01),
    "mixed":       ("Mixed (Gaussian + impulses)", "Gaussian σ (impulses fixed at "
                    f"{config.MIXED_SP_AMOUNT:.0%} of pixels)", 1.0, 60.0, 20.0, 1.0),
}


# --- images -------------------------------------------------------------
def load_uploaded(data: bytes, size: int):
    """Decode uploaded image bytes into the project's standard image.

    Returns (image, (width, height) of the original). Colour is converted to
    grayscale, the image is center-cropped to a square and resized to `size`.
    """
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            original = im.size
            arr = np.asarray(im.convert("RGB") if im.mode not in ("L", "F") else im)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("Could not read this file as an image. "
                         "Use PNG, JPG, BMP or TIFF.") from exc
    if min(original) < MIN_SIDE:
        raise ValueError(f"Image is too small ({original[0]}×{original[1]}). "
                         f"The shorter side must be at least {MIN_SIDE} px.")
    return prepare(arr, size), original


def to_uint8(arr) -> np.ndarray:
    return np.clip(np.rint(np.asarray(arr, dtype=np.float64)), 0, 255).astype(np.uint8)


def png_bytes(arr) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(to_uint8(arr), mode="L").save(buf, format="PNG")
    return buf.getvalue()


def display_image(arr, scale: int = 2) -> Image.Image:
    """8-bit PIL image enlarged by whole pixels (nearest) so noise stays crisp."""
    im = Image.fromarray(to_uint8(arr), mode="L")
    return im.resize((im.width * scale, im.height * scale), Image.NEAREST)


def crop_region(arr, center_x: float, center_y: float, window: int) -> np.ndarray:
    """Square crop around a point given as fractions (0-1) of width/height."""
    h, w = arr.shape
    window = max(1, min(window, h, w))
    x0 = int(round(center_x * w - window / 2))
    y0 = int(round(center_y * h - window / 2))
    x0 = min(max(x0, 0), w - window)
    y0 = min(max(y0, 0), h - window)
    return arr[y0:y0 + window, x0:x0 + window]


# --- running an experiment ---------------------------------------------
def default_params() -> dict:
    return {
        "wavelet": {"wavelet": config.WAVELET, "level": config.WAVELET_LEVEL,
                    "mode": config.WAVELET_THRESHOLD_MODE,
                    "method": config.WAVELET_THRESHOLD_METHOD,
                    "threshold_scale": config.WAVELET_THRESHOLD_SCALE},
        "shearlet": {"scales": config.SHEARLET_SCALES,
                     "factor": config.SHEARLET_THRESHOLD_FACTOR,
                     "mode": config.SHEARLET_THRESHOLD_MODE},
    }


def noise_sigma(noise_type: str, level: float, noisy, sigma_mode: str = "config"):
    """Noise level handed to both denoisers: (sigma, 'true' | 'estimated')."""
    if sigma_mode not in SIGMA_MODES:
        raise ValueError(f"sigma_mode must be one of {SIGMA_MODES}")
    if sigma_mode == "config" and config.SIGMA_SOURCE[noise_type] == "true":
        return float(level), "true"
    return estimate_sigma(noisy), "estimated"


def tune_knob(method: str, clean, noisy, sigma, base: dict):
    """Try the method's tuning grid (same size for both methods) and keep the
    value with the highest PSNR against the clean image (oracle tuning).

    Returns (knob name, best value, number of candidates, at_grid_edge).
    """
    (knob, values), = config.TUNING_GRIDS[method].items()
    best = None
    for v in values:
        img, _ = METHODS[method](noisy, sigma, **{**base, knob: v})
        score = psnr(clean, img)
        if best is None or score > best[0]:
            best = (score, v)
    v = best[1]
    return knob, v, len(values), v in (min(values), max(values))


def run_experiment(clean, noise_type: str, level: float, seed: int,
                   sigma_mode: str = "config", params: dict = None,
                   tune: bool = False) -> dict:
    """Add noise, run both methods on the SAME noisy image and score them."""
    params = copy.deepcopy(params) if params else default_params()
    noisy = add_noise(clean, noise_type, level, seed=seed)
    sigma, source = noise_sigma(noise_type, level, noisy, sigma_mode)
    tuned = {}
    if tune:
        for method in METHODS:
            knob, value, n, edge = tune_knob(method, clean, noisy, sigma, params[method])
            params[method][knob] = value
            tuned[method] = {"knob": knob, "value": value, "candidates": n, "at_edge": edge}
    result = compare(clean, noisy, sigma, params=params)
    return {"result": result, "sigma": sigma, "sigma_source": source, "params": params,
            "tuned": tuned, "tune": tune, "noise_type": noise_type, "level": level,
            "seed": seed, "sigma_mode": sigma_mode}


# --- exports ------------------------------------------------------------
def results_dataframe(run: dict, image_name: str) -> pd.DataFrame:
    """One row per method, in the same columns as the benchmark CSV."""
    table = run["result"].table
    used = effective_params(run["sigma"], run["params"])
    exp = experiment_id(image_name, run["noise_type"], run["level"], run["seed"])
    rows = []
    for _, r in table.iterrows():
        m = r["method"]
        info = run["tuned"].get(m)
        rows.append({
            "experiment_id": exp, "tuning": "tuned" if run["tune"] else "custom",
            "image": image_name, "noise_type": run["noise_type"],
            "noise_level": run["level"], "seed": run["seed"], "method": m,
            "mse": r["mse"], "psnr": r["psnr"], "ssim": r["ssim"],
            "processing_time": r["processing_time"], "sigma_used": run["sigma"],
            "sigma_source": run["sigma_source"],
            "n_candidates": info["candidates"] if info else (0 if m == "noisy" else 1),
            "tuned_value": info["value"] if info else np.nan,
            "params": json.dumps(used[m], sort_keys=True) if m in used else "{}",
        })
    return pd.DataFrame(rows, columns=BENCHMARK_COLUMNS)


def build_zip(run: dict, image_name: str) -> bytes:
    """All four images, the results CSV and the parameters, in one ZIP."""
    images = run["result"].images
    df = results_dataframe(run, image_name)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for key, name in (("clean", "original"), ("noisy", "noisy"),
                          ("wavelet", "wavelet_denoised"), ("shearlet", "shearlet_denoised")):
            z.writestr(f"{name}.png", png_bytes(images[key]))
        z.writestr("results.csv", df.to_csv(index=False))
        z.writestr("parameters.json", json.dumps(
            {"image": image_name, "noise_type": run["noise_type"],
             "noise_level": run["level"], "seed": run["seed"],
             "sigma_used": run["sigma"], "sigma_source": run["sigma_source"],
             "auto_tuned": run["tune"], "methods": run["params"]}, indent=2))
    return buf.getvalue()


def rounded_table(df: pd.DataFrame) -> pd.DataFrame:
    """Display copy of a result table with sensible rounding."""
    out = df.copy()
    for col, digits in (("mse", 2), ("psnr", 2), ("ssim", 4), ("processing_time", 4),
                        ("difference", 4), ("relative_change_pct", 1)):
        if col in out.columns:
            out[col] = out[col].astype(float).round(digits)
    return out


def takeaways(run: dict) -> list:
    """Plain-language facts about this single run (no 'winner' claims)."""
    t = run["result"].table.set_index("method")
    out = []
    for m in ("wavelet", "shearlet"):
        gain = t.loc[m, "psnr"] - t.loc["noisy", "psnr"]
        out.append(f"{m.capitalize()} changed PSNR by {gain:+.2f} dB and SSIM by "
                   f"{t.loc[m, 'ssim'] - t.loc['noisy', 'ssim']:+.3f} compared with the noisy image.")
    d = run["result"].differences.set_index("metric")
    ratio = t.loc["shearlet", "processing_time"] / t.loc["wavelet", "processing_time"]
    out.append(f"Shearlet minus wavelet on this run: {d.loc['psnr', 'difference']:+.2f} dB PSNR, "
               f"{d.loc['ssim', 'difference']:+.3f} SSIM; shearlet took {ratio:.0f}× as long.")
    return out