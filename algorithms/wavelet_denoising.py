"""Wavelet denoising engine (Phase 3).

Pipeline:  noisy -> 2-D DWT -> threshold DETAIL coefficients -> inverse DWT

Public contract (the Shearlet module will match it):
    denoise(noisy, sigma=None, **params) -> np.ndarray   same shape as input
    denoise_timed(...)                   -> (np.ndarray, seconds)
"""
import time

import numpy as np
import pywt

import config

THRESHOLD_MODES = ("soft", "hard")
THRESHOLD_METHODS = ("bayes", "universal")


# --- noise estimation ---------------------------------------------------
def estimate_sigma(noisy: np.ndarray, wavelet: str = None) -> float:
    """Robust noise estimate (Donoho's MAD) from the finest diagonal band."""
    wavelet = wavelet or config.WAVELET
    _, (_, _, cd) = pywt.dwt2(noisy, wavelet, mode="symmetric")
    return float(np.median(np.abs(cd)) / 0.6745)


# --- thresholds ---------------------------------------------------------
def _universal_threshold(sigma: float, n_pixels: int) -> float:
    """VisuShrink: sigma * sqrt(2 ln N). Same value for every subband."""
    return sigma * np.sqrt(2.0 * np.log(n_pixels))


def _bayes_threshold(band: np.ndarray, sigma: float) -> float:
    """BayesShrink: sigma^2 / sigma_signal, estimated per subband."""
    var_signal = max(float(np.mean(band ** 2)) - sigma ** 2, 0.0)
    if var_signal == 0.0:                      # band is pure noise
        return float(np.max(np.abs(band)))     # -> zero the whole band
    return sigma ** 2 / np.sqrt(var_signal)


# --- main ---------------------------------------------------------------
def denoise(noisy: np.ndarray, sigma: float = None, wavelet: str = None,
            level: int = None, mode: str = None, method: str = None,
            threshold_scale: float = None) -> np.ndarray:
    """Wavelet-threshold denoising of a 2-D image on a 0-255 scale.

    sigma            true noise std; if None it is estimated (MAD)
    wavelet, level   DWT family and decomposition depth (capped to the max
                     depth the image size allows)
    mode             "soft" (default) or "hard" thresholding
    method           "bayes" or "universal" threshold rule
    threshold_scale  multiplies every threshold (the tuning knob)
    """
    wavelet = wavelet or config.WAVELET
    level = level or config.WAVELET_LEVEL
    mode = mode or config.WAVELET_THRESHOLD_MODE
    method = method or config.WAVELET_THRESHOLD_METHOD
    scale = config.WAVELET_THRESHOLD_SCALE if threshold_scale is None else threshold_scale

    noisy = np.asarray(noisy, dtype=np.float64)
    if noisy.ndim != 2:
        raise ValueError("Wavelet denoising expects a 2-D grayscale image")
    if mode not in THRESHOLD_MODES:
        raise ValueError(f"mode must be one of {THRESHOLD_MODES}")
    if method not in THRESHOLD_METHODS:
        raise ValueError(f"method must be one of {THRESHOLD_METHODS}")
    if level < 1 or scale < 0 or (sigma is not None and sigma < 0):
        raise ValueError("level must be >= 1; sigma and threshold_scale >= 0")
    try:
        w = pywt.Wavelet(wavelet)
    except ValueError as exc:
        raise ValueError(f"Unknown wavelet '{wavelet}'") from exc

    level = min(level, pywt.dwt_max_level(min(noisy.shape), w.dec_len))
    if level < 1:
        raise ValueError("Image too small for this wavelet")
    if sigma is None:
        sigma = estimate_sigma(noisy, wavelet)

    coeffs = pywt.wavedec2(noisy, w, level=level, mode="symmetric")
    approx, details = coeffs[0], coeffs[1:]          # approximation untouched

    out = [approx]
    for (ch, cv, cd) in details:                     # detail bands only
        bands = []
        for band in (ch, cv, cd):
            if method == "universal":
                t = _universal_threshold(sigma, noisy.size)
            else:
                t = _bayes_threshold(band, sigma)
            bands.append(pywt.threshold(band, scale * t, mode=mode))
        out.append(tuple(bands))

    rec = pywt.waverec2(out, w, mode="symmetric")
    rec = rec[: noisy.shape[0], : noisy.shape[1]]    # undo odd-size padding
    return np.clip(rec, 0.0, config.INTENSITY_RANGE)


def denoise_timed(noisy, sigma=None, **params):
    """Same as denoise() but also returns elapsed seconds."""
    t0 = time.perf_counter()
    result = denoise(noisy, sigma, **params)
    return result, time.perf_counter() - t0