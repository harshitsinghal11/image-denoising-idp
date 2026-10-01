"""Image loading and preprocessing (Phase 2).

Contract: every image leaves this module as a 2-D float64 array,
square (IMAGE_SIZE x IMAGE_SIZE), with values in [0, 255].
"""
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import data as skdata

import config

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _to_gray(arr: np.ndarray) -> np.ndarray:
    """Return a 2-D float64 array on a 0-255 scale."""
    arr = np.asarray(arr)
    if arr.ndim == 3:
        if arr.shape[2] == 4:            # drop alpha
            arr = arr[..., :3]
        if arr.shape[2] != 3:
            raise ValueError(f"Unsupported channel count: {arr.shape[2]}")
        # ITU-R BT.601 luma, same weights as PIL's convert("L")
        arr = arr.astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    elif arr.ndim != 2:
        raise ValueError(f"Expected 2-D or 3-D image, got {arr.ndim}-D")
    return arr.astype(np.float64)


def _to_square(arr: np.ndarray, size: int) -> np.ndarray:
    """Center-crop to a square, then resize to size x size."""
    h, w = arr.shape
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    arr = arr[top:top + s, left:left + s]
    if s != size:
        img = Image.fromarray(arr.astype(np.float32), mode="F")
        arr = np.asarray(img.resize((size, size), Image.LANCZOS), dtype=np.float64)
    return np.clip(arr, 0.0, config.INTENSITY_RANGE)


def prepare(arr: np.ndarray, size: int = None) -> np.ndarray:
    """Turn any array (gray/RGB/RGBA) into the project's standard image."""
    size = size or config.IMAGE_SIZE
    if size < 16:
        raise ValueError("Image size must be at least 16 pixels")
    arr = np.asarray(arr)
    if arr.dtype != np.uint8 and arr.max() <= 1.0:   # float images in [0, 1]
        arr = arr * 255.0
    return _to_square(_to_gray(arr), size)


def load_image(path, size: int = None) -> np.ndarray:
    """Load an image file and return the standard float64 square image."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No such image file: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported format '{path.suffix}'. "
            f"Use one of {sorted(SUPPORTED_EXTENSIONS)}")
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB") if im.mode not in ("L", "F") else im)
    return prepare(arr, size)


def load_sample(name: str, size: int = None) -> np.ndarray:
    """Load a built-in scikit-image test image (e.g. 'camera', 'moon')."""
    if not hasattr(skdata, name):
        raise ValueError(f"Unknown sample image '{name}'")
    return prepare(getattr(skdata, name)(), size)


def save_image(arr: np.ndarray, path) -> None:
    """Save a 0-255 float image as an 8-bit file (clipped, rounded)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    Image.fromarray(out, mode="L").save(path)