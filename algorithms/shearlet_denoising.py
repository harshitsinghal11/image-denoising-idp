"""Shearlet denoising engine (Phase 4).

Pipeline:  noisy -> shearlet transform -> threshold -> inverse transform

Same public contract as the Wavelet module:
    denoise(noisy, sigma=None, **params) -> np.ndarray   same shape as input
    denoise_timed(...)                   -> (np.ndarray, seconds)

pyShearLab restrictions handled here (found by testing, see docs):
  * the shearlet system is built for a SQUARE image of one fixed size;
    non-square inputs are reflect-padded to a square and cropped back
  * 3 scales need an image of at least ~128 px per side; smaller sizes make
    the library crash (it even calls sys.exit), so they are rejected with a
    readable ValueError instead
  * building the system is slow (~1-2 s at 256 px), so it is built once per
    (size, scales) and cached; setup time is reported separately from the
    transform time
"""
import time

import numpy as np
import pyshearlab

import config
from algorithms.wavelet_denoising import estimate_sigma   # same noise estimator for both methods

THRESHOLD_MODES = ("hard", "soft")

_SYSTEM_CACHE = {}   # (size, scales) -> (system, build_seconds)


# --- system handling ----------------------------------------------------
def get_system(size: int, scales: int):
    """Build (or fetch from cache) the shearlet system for size x size images."""
    key = (size, scales)
    if key not in _SYSTEM_CACHE:
        t0 = time.perf_counter()
        try:
            system = pyshearlab.SLgetShearletSystem2D(0, size, size, scales)
            build = time.perf_counter() - t0
            # self-test: the library only fails at transform time for some sizes
            probe = np.zeros((size, size))
            pyshearlab.SLshearrec2D(pyshearlab.SLsheardec2D(probe, system), system)
        except (Exception, SystemExit) as exc:        # SystemExit: pyShearLab calls exit()
            raise ValueError(
                f"Shearlet system not available for {size}x{size} images with "
                f"{scales} scales. Use a larger image (>= 128 px for 3 scales) "
                f"or fewer scales.") from exc
        _SYSTEM_CACHE[key] = (system, build)
    return _SYSTEM_CACHE[key][0]


def setup_seconds(size: int, scales: int) -> float:
    """Time it took to build the system (0-cost if already cached earlier)."""
    get_system(size, scales)
    return _SYSTEM_CACHE[(size, scales)][1]


def _lowpass_index(system) -> int:
    """Index of the low-pass (scaling) shearlet: the row [0, 0, 0]."""
    idxs = np.asarray(system["shearletIdxs"])
    return int(np.where(idxs[:, 0] == 0)[0][0])


def _pad_to_square(img: np.ndarray):
    h, w = img.shape
    s = max(h, w)
    if h == w:
        return img, (h, w)
    padded = np.pad(img, ((0, s - h), (0, s - w)), mode="symmetric")
    return padded, (h, w)


# --- main ---------------------------------------------------------------
def denoise(noisy: np.ndarray, sigma: float = None, scales: int = None,
            factor: float = None, mode: str = None) -> np.ndarray:
    """Shearlet-threshold denoising of a 2-D image on a 0-255 scale.

    sigma   true noise std; if None it is estimated (same MAD estimator
            as the Wavelet method)
    scales  number of shearlet scales (config.SHEARLET_SCALES)
    factor  cutoff multiplier: a coefficient is cut when
            |c| < factor * sigma * RMS_j   (RMS_j = norm of shearlet j)
    mode    "hard" (default) or "soft"
    The low-pass shearlet is never thresholded (it holds the coarse image).
    """
    scales = scales or config.SHEARLET_SCALES
    factor = config.SHEARLET_THRESHOLD_FACTOR if factor is None else factor
    mode = mode or config.SHEARLET_THRESHOLD_MODE

    noisy = np.asarray(noisy, dtype=np.float64)
    if noisy.ndim != 2:
        raise ValueError("Shearlet denoising expects a 2-D grayscale image")
    if mode not in THRESHOLD_MODES:
        raise ValueError(f"mode must be one of {THRESHOLD_MODES}")
    if scales < 1 or factor < 0 or (sigma is not None and sigma < 0):
        raise ValueError("scales must be >= 1; sigma and factor >= 0")
    if sigma is None:
        sigma = estimate_sigma(noisy)

    padded, (h, w) = _pad_to_square(noisy)
    system = get_system(padded.shape[0], scales)          # cached

    coeffs = np.real(pyshearlab.SLsheardec2D(padded, system))
    weights = np.asarray(system["RMS"])[None, None, :]
    cutoff = factor * sigma * weights                     # per-shearlet cutoff

    keep_low = coeffs[:, :, _lowpass_index(system)].copy()
    if mode == "hard":
        coeffs = np.where(np.abs(coeffs) < cutoff, 0.0, coeffs)
    else:
        coeffs = np.sign(coeffs) * np.maximum(np.abs(coeffs) - cutoff, 0.0)
    coeffs[:, :, _lowpass_index(system)] = keep_low       # restore low-pass

    rec = np.real(pyshearlab.SLshearrec2D(coeffs, system))[:h, :w]
    return np.clip(rec, 0.0, config.INTENSITY_RANGE)


def denoise_timed(noisy, sigma=None, **params):
    """Like denoise() but returns (image, seconds). The one-off system setup
    is done before the clock starts, so only transform + threshold + inverse
    are timed (use setup_seconds() for the setup cost)."""
    padded_size = max(np.asarray(noisy).shape)
    get_system(padded_size, params.get("scales") or config.SHEARLET_SCALES)
    t0 = time.perf_counter()
    result = denoise(noisy, sigma, **params)
    return result, time.perf_counter() - t0