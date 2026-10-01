"""Plots for the comparison (Phase 6).

Every function builds and returns a matplotlib Figure (OO API, no pyplot, so
there is no global state and the same functions work in scripts and in
Streamlit via st.pyplot(fig)). Pass save_path to also write a PNG.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

COLORS = {"noisy": "#8c8c8c", "wavelet": "#1f77b4", "shearlet": "#ff7f0e"}
METRIC_INFO = {          # label, "higher"/"lower" is better
    "psnr": ("PSNR (dB)", "higher"),
    "ssim": ("SSIM", "higher"),
    "mse": ("MSE", "lower"),
    "processing_time": ("Processing time (s)", "lower"),
}
_PANELS = (("clean", "Original"), ("noisy", "Noisy"),
           ("wavelet", "Wavelet"), ("shearlet", "Shearlet"))


def _finish(fig: Figure, save_path):
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def _layout(fig: Figure, title):
    """Tidy layout; reserve headroom so the figure title never overlaps panels."""
    if title:
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
    else:
        fig.tight_layout()


def plot_image_comparison(images: dict, table: pd.DataFrame = None,
                          title: str = None, save_path=None) -> Figure:
    """Original | Noisy | Wavelet | Shearlet, with PSNR/SSIM under each image."""
    panels = [(k, label) for k, label in _PANELS if k in images]
    fig = Figure(figsize=(3.4 * len(panels), 3.9))
    rows = table.set_index("method") if table is not None else None
    for i, (key, label) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, len(panels), i)
        ax.imshow(np.clip(images[key], 0, 255), cmap="gray", vmin=0, vmax=255)
        ax.axis("off")
        caption = label
        if rows is not None and key in rows.index:
            caption += f"\n{rows.loc[key, 'psnr']:.2f} dB | SSIM {rows.loc[key, 'ssim']:.3f}"
        ax.set_title(caption, fontsize=10)
    _layout(fig, title)
    return _finish(fig, save_path)


def _bar(ax, table: pd.DataFrame, metric: str):
    label, better = METRIC_INFO[metric]
    data = table if metric != "processing_time" else table[table["method"] != "noisy"]
    data = data.dropna(subset=[metric])
    methods = list(data["method"])
    values = list(data[metric])
    bars = ax.bar(methods, values, color=[COLORS.get(m, "#555555") for m in methods])
    for b, v in zip(bars, values):
        ax.annotate(f"{v:.4g}", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=9)
    if metric == "processing_time" and len(values) > 1 and min(values) > 0 \
            and max(values) / min(values) > 20:
        ax.set_yscale("log")
        label += ", log scale"
    ax.set_ylabel(label)
    ax.set_title(f"{label.split(',')[0]}  ({better} is better)", fontsize=10)
    ax.margins(y=0.15)


def plot_metric_bar(table: pd.DataFrame, metric: str, title: str = None,
                    save_path=None) -> Figure:
    """One bar chart for one metric (psnr, ssim, mse or processing_time)."""
    if metric not in METRIC_INFO:
        raise ValueError(f"metric must be one of {list(METRIC_INFO)}")
    fig = Figure(figsize=(4.6, 3.8))
    _bar(fig.add_subplot(111), table, metric)
    _layout(fig, title)
    return _finish(fig, save_path)


def plot_all_metrics(table: pd.DataFrame, title: str = None, save_path=None) -> Figure:
    """PSNR, SSIM, MSE and runtime charts in one 2x2 figure."""
    fig = Figure(figsize=(9.5, 7.2))
    for i, metric in enumerate(("psnr", "ssim", "mse", "processing_time"), start=1):
        _bar(fig.add_subplot(2, 2, i), table, metric)
    _layout(fig, title)
    return _finish(fig, save_path)


def plot_metric_vs_noise(results: pd.DataFrame, metric: str, noise_type: str,
                         image: str = None, save_path=None) -> Figure:
    """Metric against noise level, one line per method (from results.csv).

    With several images (image=None) the lines show the mean over images.
    The unprocessed noisy baseline is drawn dashed for scale.
    """
    if metric not in METRIC_INFO:
        raise ValueError(f"metric must be one of {list(METRIC_INFO)}")
    df = results[results["noise_type"] == noise_type]
    if image is not None:
        df = df[df["image"] == image]
    if df.empty:
        raise ValueError("No saved results match this noise type / image")
    label, better = METRIC_INFO[metric]
    fig = Figure(figsize=(5.4, 4.0))
    ax = fig.add_subplot(111)
    for method in ("noisy", "wavelet", "shearlet"):
        d = df[df["method"] == method].dropna(subset=[metric])
        if d.empty:
            continue
        g = d.groupby("noise_level")[metric].mean().sort_index()
        ax.plot(g.index, g.values, marker="o", color=COLORS[method], label=method,
                linestyle="--" if method == "noisy" else "-")
    ax.set_xlabel(f"{noise_type} noise level")
    ax.set_ylabel(label)
    n_img = df["image"].nunique()
    ax.set_title(f"{label} vs noise level ({image or f'mean of {n_img} images'})", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return _finish(fig, save_path)