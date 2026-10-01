"""Image-quality metrics (Phase 5).

All metrics compare a processed image against the CLEAN reference, both
2-D arrays on a 0-255 scale. The processed image is clipped to [0, 255]
first, so every method is judged on the image you would actually save.

Result contract returned by evaluate() for every method:
    method, mse, psnr, ssim, processing_time
"""
import math

import numpy as np
from skimage.metrics import structural_similarity

import config

METRIC_COLUMNS = ("method", "mse", "psnr", "ssim", "processing_time")


def _prepare(clean, image):
    clean = np.asarray(clean, dtype=np.float64)
    image = np.asarray(image, dtype=np.float64)
    if clean.ndim != 2 or image.ndim != 2:
        raise ValueError("Metrics expect 2-D grayscale images")
    if clean.shape != image.shape:
        raise ValueError(f"Shape mismatch: {clean.shape} vs {image.shape}")
    return clean, np.clip(image, 0.0, config.INTENSITY_RANGE)


def mse(clean, image) -> float:
    """Mean squared error. Lower is better; 0 means identical."""
    clean, image = _prepare(clean, image)
    return float(np.mean((clean - image) ** 2))


def psnr(clean, image) -> float:
    """Peak signal-to-noise ratio in dB. Higher is better; inf if identical."""
    err = mse(clean, image)
    if err == 0.0:
        return math.inf
    return float(10.0 * math.log10(config.INTENSITY_RANGE ** 2 / err))


def ssim(clean, image) -> float:
    """Structural similarity, about 0 to 1. Higher is better; 1 is identical."""
    clean, image = _prepare(clean, image)
    return float(structural_similarity(
        clean, image, data_range=config.INTENSITY_RANGE))


def evaluate(method: str, clean, image, processing_time: float = float("nan")) -> dict:
    """Return the standard result record for one method."""
    return {
        "method": method,
        "mse": mse(clean, image),
        "psnr": psnr(clean, image),
        "ssim": ssim(clean, image),
        "processing_time": float(processing_time),
    }