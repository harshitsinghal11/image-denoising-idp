"""Wavelet vs Shearlet comparison (Phase 5).

compare(clean, noisy, sigma) is the one function that runs both methods on
the SAME noisy image and returns a complete, structured comparison.

It reports numbers and differences only. It deliberately does NOT declare
a winner: that needs the full experiment set (Phases 7-8).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from algorithms import shearlet_denoising, wavelet_denoising
from evaluation.metrics import METRIC_COLUMNS, evaluate

METHODS = {
    "wavelet": wavelet_denoising.denoise_timed,
    "shearlet": shearlet_denoising.denoise_timed,
}
_METRICS = ("mse", "psnr", "ssim", "processing_time")


@dataclass
class ComparisonResult:
    table: pd.DataFrame          # one row per method, plus the noisy baseline
    differences: pd.DataFrame    # method_b minus method_a, per metric
    images: dict                 # {"clean", "noisy", "<method>": image}


def difference_table(table: pd.DataFrame, a: str = "wavelet", b: str = "shearlet") -> pd.DataFrame:
    """Compare two methods metric by metric.

    difference          = b - a   (raw; for mse and time LOWER is better,
                                    for psnr and ssim HIGHER is better)
    relative_change_pct = 100 * (b - a) / |a|, left empty for PSNR because
                          it is already a log scale, so use the dB difference.
    """
    rows = table.set_index("method")
    out = []
    for metric in _METRICS:
        va, vb = rows.loc[a, metric], rows.loc[b, metric]
        diff = vb - va
        rel = np.nan
        if metric != "psnr" and np.isfinite(va) and va != 0:
            rel = 100.0 * diff / abs(va)
        out.append({"metric": metric, a: va, b: vb,
                    "difference": diff, "relative_change_pct": rel})
    return pd.DataFrame(out)


def compare(clean, noisy, sigma=None, methods=("wavelet", "shearlet"),
            params=None) -> ComparisonResult:
    """Run each method on the same noisy image and evaluate against `clean`.

    params: optional {"wavelet": {...}, "shearlet": {...}} keyword arguments
            forwarded to each denoiser (e.g. {"shearlet": {"factor": 2.5}}).
    """
    params = params or {}
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise ValueError(f"Unknown method(s) {unknown}. Available: {list(METHODS)}")

    images = {"clean": np.asarray(clean), "noisy": np.asarray(noisy)}
    records = [evaluate("noisy", clean, noisy)]          # baseline: no processing
    for name in methods:
        result, seconds = METHODS[name](noisy, sigma, **params.get(name, {}))
        images[name] = result
        records.append(evaluate(name, clean, result, seconds))

    table = pd.DataFrame(records, columns=list(METRIC_COLUMNS))
    if "wavelet" in methods and "shearlet" in methods:
        diffs = difference_table(table)
    else:
        diffs = pd.DataFrame()
    return ComparisonResult(table=table, differences=diffs, images=images)