"""Research analysis (Phase 8): turn benchmark_results.csv into evidence.

    python -m analysis.analyze_results
    python -m analysis.analyze_results --csv data/results/benchmark_results.csv --root data

Reads the benchmark CSV and writes, under <out> (default data/results/analysis):
    tables/*.csv      summary, paired differences, per-image, failures, runtime
    figures/*.png     metric-vs-level, difference (forest) plots, per-image,
                      runtime trade-off, visual gallery of notable cases
    analysis_report.md  the research questions answered with numbers

Method of comparison: every experiment (image x noise x level x seed) gives
one PAIRED observation, shearlet minus wavelet, for each metric. Per condition
we report the mean difference with a 95% t confidence interval and how many
runs favoured each method. "Clear difference" means the interval excludes 0.
The report states numbers and their limits; it never uses the word "best".
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from PIL import Image
from scipy import stats

import config
from evaluation.storage import _safe
from experiments.experiment_runner import grid_edge_report, read_benchmark_csv
from visualization.plots import COLORS

KEYS = ["tuning", "noise_type", "noise_level"]
METHODS3 = ("noisy", "wavelet", "shearlet")
NOISE_ORDER = ["gaussian", "salt_pepper", "speckle", "mixed"]
REQUIRED = {"experiment_id", "tuning", "image", "noise_type", "noise_level",
            "seed", "method", "mse", "psnr", "ssim", "processing_time"}


# --- loading and reshaping ----------------------------------------------
def load_benchmark(path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m experiments.experiment_runner --preset full")
    df = read_benchmark_csv(path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Benchmark CSV is missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Benchmark CSV has no rows")
    return df


def to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (experiment, tuning) with every method side by side."""
    meta = ["experiment_id", "tuning", "image", "noise_type", "noise_level", "seed"]
    wide = None
    for m in METHODS3:
        part = df[df["method"] == m][meta + ["mse", "psnr", "ssim", "processing_time"]]
        part = part.rename(columns={c: f"{c}_{m}" for c in
                                    ("mse", "psnr", "ssim", "processing_time")})
        wide = part if wide is None else wide.merge(part, on=meta, how="inner")
    wide = wide.drop(columns="processing_time_noisy")
    for m in ("wavelet", "shearlet"):
        wide[f"psnr_gain_{m}"] = wide[f"psnr_{m}"] - wide["psnr_noisy"]
        wide[f"ssim_gain_{m}"] = wide[f"ssim_{m}"] - wide["ssim_noisy"]
    wide["psnr_diff"] = wide["psnr_shearlet"] - wide["psnr_wavelet"]
    wide["ssim_diff"] = wide["ssim_shearlet"] - wide["ssim_wavelet"]
    wide["time_ratio"] = wide["processing_time_shearlet"] / wide["processing_time_wavelet"]
    return wide.reset_index(drop=True)


# --- statistics ---------------------------------------------------------
def mean_ci(x, conf: float = 0.95):
    """(mean, ci_low, ci_high, n) using a t interval; CI is NaN if n < 2."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    m = float(x.mean())
    if n < 2:
        return m, np.nan, np.nan, n
    se = float(x.std(ddof=1) / np.sqrt(n))
    if se == 0.0:
        return m, m, m, n
    lo, hi = stats.t.interval(conf, n - 1, loc=m, scale=se)
    return m, float(lo), float(hi), n


def direction(lo: float, hi: float) -> int:
    """+1 if the whole interval is above 0, -1 if below, else 0."""
    if np.isnan(lo) or np.isnan(hi):
        return 0
    return 1 if lo > 0 else (-1 if hi < 0 else 0)


def verdict(psnr_dir: int, ssim_dir: int, n: int) -> str:
    if n < 3:
        return "insufficient data (n<3)"
    if psnr_dir > 0 and ssim_dir > 0:
        return "shearlet higher on PSNR and SSIM"
    if psnr_dir < 0 and ssim_dir < 0:
        return "wavelet higher on PSNR and SSIM"
    if psnr_dir * ssim_dir < 0:
        return "metrics disagree"
    if psnr_dir > 0:
        return "shearlet higher on PSNR only"
    if psnr_dir < 0:
        return "wavelet higher on PSNR only"
    if ssim_dir > 0:
        return "shearlet higher on SSIM only"
    if ssim_dir < 0:
        return "wavelet higher on SSIM only"
    return "no clear difference"


# --- tables -------------------------------------------------------------
def summary_table(w: pd.DataFrame) -> pd.DataFrame:
    """Mean/std of each metric per condition and method, plus gain over noisy."""
    rows = []
    for keys, g in w.groupby(KEYS, sort=False):
        for m in METHODS3:
            r = dict(zip(KEYS, keys), method=m, n=len(g))
            for metric in ("psnr", "ssim", "mse"):
                col = g[f"{metric}_{m}"]
                r[f"{metric}_mean"] = col.mean()
                r[f"{metric}_std"] = col.std(ddof=1) if len(col) > 1 else np.nan
            if m == "noisy":
                r.update(time_median=np.nan, psnr_gain=np.nan, ssim_gain=np.nan)
            else:
                r.update(time_median=g[f"processing_time_{m}"].median(),
                         psnr_gain=g[f"psnr_gain_{m}"].mean(),
                         ssim_gain=g[f"ssim_gain_{m}"].mean())
            rows.append(r)
    return pd.DataFrame(rows)


def paired_table(w: pd.DataFrame, by=KEYS) -> pd.DataFrame:
    """Shearlet-minus-wavelet differences with CIs, win counts and a verdict."""
    rows = []
    for keys, g in w.groupby(list(by), sort=False):
        r = dict(zip(by, keys), n=len(g))
        dirs = {}
        for metric in ("psnr", "ssim"):
            m, lo, hi, _ = mean_ci(g[f"{metric}_diff"])
            r.update({f"{metric}_diff_mean": m, f"{metric}_diff_lo": lo,
                      f"{metric}_diff_hi": hi,
                      f"{metric}_shearlet_higher_runs": int((g[f"{metric}_diff"] > 0).sum())})
            dirs[metric] = direction(lo, hi)
        r["metrics_disagree_runs"] = int(
            (np.sign(g["psnr_diff"]) * np.sign(g["ssim_diff"]) < 0).sum())
        r["time_ratio_median"] = g["time_ratio"].median()
        r["verdict"] = verdict(dirs["psnr"], dirs["ssim"], len(g))
        rows.append(r)
    return pd.DataFrame(rows)


def failure_cases(w: pd.DataFrame) -> pd.DataFrame:
    """Runs where a method scored below the noisy input, or the metrics disagree."""
    rows = []
    base = ["tuning", "experiment_id", "image", "noise_type", "noise_level", "seed"]
    for _, r in w.iterrows():
        info = {k: r[k] for k in base}
        for m in ("wavelet", "shearlet"):
            for metric in ("psnr", "ssim"):
                gain = r[f"{metric}_gain_{m}"]
                if gain <= 0:
                    rows.append(dict(info, kind=f"{m} below noisy input on {metric.upper()}",
                                     value=gain))
        if np.sign(r["psnr_diff"]) * np.sign(r["ssim_diff"]) < 0:
            rows.append(dict(info, kind="PSNR and SSIM favour different methods",
                             value=r["psnr_diff"]))
    return pd.DataFrame(rows, columns=base + ["kind", "value"])


def runtime_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-method runtime in milliseconds (shearlet setup time excluded)."""
    t = df[df["method"].isin(["wavelet", "shearlet"])].copy()
    t["ms"] = t["processing_time"] * 1000.0
    out = (t.groupby("method")["ms"]
             .agg(runs="size", median_ms="median", mean_ms="mean", std_ms="std",
                  min_ms="min", max_ms="max")
             .reset_index())
    med = out.set_index("method")["median_ms"]
    out["median_vs_wavelet"] = out["method"].map(lambda m: med[m] / med["wavelet"])
    return out


# --- figures ------------------------------------------------------------
def _types(frame):
    present = set(frame["noise_type"])
    return [t for t in NOISE_ORDER if t in present]


def fig_metric_vs_level(summary: pd.DataFrame, metric: str, tuning: str) -> Figure:
    s = summary[summary["tuning"] == tuning]
    types = _types(s)
    fig = Figure(figsize=(5.2 * min(len(types), 2), 3.8 * ((len(types) + 1) // 2)))
    for i, nt in enumerate(types, start=1):
        ax = fig.add_subplot((len(types) + 1) // 2, min(len(types), 2), i)
        for m in METHODS3:
            d = s[(s["noise_type"] == nt) & (s["method"] == m)].sort_values("noise_level")
            ax.plot(d["noise_level"], d[f"{metric}_mean"], marker="o", color=COLORS[m],
                    label=m, linestyle="--" if m == "noisy" else "-")
        ax.set_title(f"{nt}", fontsize=10)
        ax.set_xlabel("noise level")
        ax.set_ylabel(f"mean {metric.upper()}")
        ax.grid(alpha=0.3)
        if i == 1:
            ax.legend(fontsize=8)
    fig.suptitle(f"{metric.upper()} vs noise level ({tuning} parameters; mean over images and seeds)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def fig_difference_forest(paired: pd.DataFrame, tuning: str) -> Figure:
    p = paired[paired["tuning"] == tuning].copy()
    p["order"] = p["noise_type"].map({t: i for i, t in enumerate(NOISE_ORDER)})
    p = p.sort_values(["order", "noise_level"], ascending=[False, False])
    labels = [f"{r.noise_type} {r.noise_level:g}" for r in p.itertuples()]
    fig = Figure(figsize=(10.5, 0.42 * len(p) + 2.2))
    for i, (metric, unit) in enumerate((("psnr", "dB"), ("ssim", "")), start=1):
        ax = fig.add_subplot(1, 2, i)
        y = np.arange(len(p))
        mean, lo, hi = (p[f"{metric}_diff_{k}"].to_numpy() for k in ("mean", "lo", "hi"))
        colors = ["#ff7f0e" if l > 0 else "#1f77b4" if h < 0 else "#8c8c8c"
                  for l, h in zip(lo, hi)]
        for yi, m_, l_, h_, c in zip(y, mean, lo, hi, colors):
            if not np.isnan(l_):
                ax.plot([l_, h_], [yi, yi], color=c, lw=2)
            ax.plot(m_, yi, "o", color=c)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels if i == 1 else [], fontsize=8)
        ax.set_xlabel(f"shearlet minus wavelet {metric.upper()} {('(' + unit + ')') if unit else ''}")
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle(f"Paired difference with 95% CI ({tuning}). Orange: shearlet higher, "
                 f"blue: wavelet higher, gray: no clear difference", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def fig_by_image(by_image: pd.DataFrame, tuning: str) -> Figure:
    b = by_image[by_image["tuning"] == tuning]
    types, images = _types(b), sorted(b["image"].unique())
    fig = Figure(figsize=(9, 4.2))
    ax = fig.add_subplot(111)
    width = 0.8 / max(len(images), 1)
    for j, img in enumerate(images):
        vals = [b[(b["noise_type"] == t) & (b["image"] == img)]["psnr_diff_mean"].mean()
                for t in types]
        ax.bar(np.arange(len(types)) + j * width, vals, width, label=img)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(np.arange(len(types)) + width * (len(images) - 1) / 2)
    ax.set_xticklabels(types)
    ax.set_ylabel("shearlet minus wavelet PSNR (dB)")
    ax.set_title(f"Difference by image type ({tuning}; mean over levels and seeds)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def fig_tradeoff(summary: pd.DataFrame, tuning: str) -> Figure:
    s = summary[(summary["tuning"] == tuning) & (summary["method"] != "noisy")]
    g = s.groupby(["method", "noise_type"]).agg(
        time=("time_median", "median"), gain=("psnr_gain", "mean")).reset_index()
    fig = Figure(figsize=(6, 4.2))
    ax = fig.add_subplot(111)
    for r in g.itertuples():
        ax.scatter(r.time, r.gain, color=COLORS[r.method], s=60)
        ax.annotate(r.noise_type, (r.time, r.gain), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("median processing time (s, log scale)")
    ax.set_ylabel("mean PSNR gain over noisy input (dB)")
    ax.set_title(f"Quality gain vs runtime ({tuning}). Blue: wavelet, orange: shearlet", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# --- visual gallery -----------------------------------------------------
def select_cases(w: pd.DataFrame) -> list:
    """Pick notable runs to inspect by eye: (label, row)."""
    cases, used = [], set()

    def add(label, idx):
        row = w.loc[idx]
        if row["experiment_id"] not in used:
            used.add(row["experiment_id"])
            cases.append((label, row))

    add("Largest shearlet PSNR advantage", w["psnr_diff"].idxmax())
    add("Largest wavelet PSNR advantage (or smallest shearlet advantage)", w["psnr_diff"].idxmin())
    dis = w[np.sign(w["psnr_diff"]) * np.sign(w["ssim_diff"]) < 0]
    if not dis.empty:
        add("PSNR and SSIM favour different methods", dis["ssim_diff"].abs().idxmax())
    worst = w[["psnr_gain_wavelet", "psnr_gain_shearlet"]].min(axis=1)
    add("Lowest PSNR gain over the noisy input (either method)", worst.idxmin())
    return cases


def _case_images(row, root: Path):
    exp, tuning = row["experiment_id"], row["tuning"]
    paths = {"Original": root / "original" / f"{_safe(row['image'])}.png",
             "Noisy": root / "noisy" / f"{exp}.png",
             "Wavelet": root / "results" / "wavelet" / f"{exp}_{tuning}.png",
             "Shearlet": root / "results" / "shearlet" / f"{exp}_{tuning}.png"}
    if not all(p.is_file() for p in paths.values()):
        return None
    return {k: np.asarray(Image.open(p)) for k, p in paths.items()}


def make_gallery(w: pd.DataFrame, root, tuning: str):
    """Figure with Original | Noisy | Wavelet | Shearlet for notable cases.

    Returns (figure or None, list of case descriptions). None if the saved
    images are missing (the benchmark was run with --no-images).
    """
    root = Path(root)
    w = w[w["tuning"] == tuning].reset_index(drop=True)
    if w.empty:
        return None, []
    shown = [(label, row, imgs) for label, row in select_cases(w)
             if (imgs := _case_images(row, root)) is not None]
    if not shown:
        return None, []
    fig = Figure(figsize=(13, 3.7 * len(shown)))
    for r, (label, row, imgs) in enumerate(shown):
        for c, name in enumerate(("Original", "Noisy", "Wavelet", "Shearlet")):
            ax = fig.add_subplot(len(shown), 4, r * 4 + c + 1)
            ax.imshow(imgs[name], cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
            if name in ("Noisy", "Wavelet", "Shearlet"):
                key = name.lower()
                caption = f"{name}\n{row[f'psnr_{key}']:.2f} dB | SSIM {row[f'ssim_{key}']:.3f}"
            else:
                caption = f"{label}\n{row['experiment_id']}"
            ax.set_title(caption, fontsize=8)
    fig.tight_layout()
    cases = [f"{label}: {row['experiment_id']} ({tuning}); PSNR gain over noisy input: "
             f"wavelet {row['psnr_gain_wavelet']:+.2f} dB, shearlet {row['psnr_gain_shearlet']:+.2f} dB; "
             f"shearlet minus wavelet: {row['psnr_diff']:+.2f} dB PSNR, {row['ssim_diff']:+.3f} SSIM"
             for label, row, _ in shown]
    return fig, cases


# --- report -------------------------------------------------------------
def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append("–" if isinstance(v, (float, np.floating)) and np.isnan(v)
                         else f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _ci(m, lo, hi, fmt="{:+.2f}"):
    if np.isnan(lo):
        return fmt.format(m) + " [n/a]"
    return f"{fmt.format(m)} [{fmt.format(lo)}, {fmt.format(hi)}]"


def _paired_view(p: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "noise": p["noise_type"], "level": p["noise_level"].map(lambda v: f"{v:g}"),
        "n": p["n"],
        "PSNR diff dB [95% CI]": [_ci(a, b, c) for a, b, c in
                                  zip(p.psnr_diff_mean, p.psnr_diff_lo, p.psnr_diff_hi)],
        "SSIM diff [95% CI]": [_ci(a, b, c, "{:+.3f}") for a, b, c in
                               zip(p.ssim_diff_mean, p.ssim_diff_lo, p.ssim_diff_hi)],
        "PSNR runs shearlet higher": [f"{a}/{n}" for a, n in
                                      zip(p.psnr_shearlet_higher_runs, p.n)],
        "verdict": p["verdict"]})


def build_report(w, summary, paired, paired_type, by_image, failures, runtime,
                 df, gallery_cases, csv_path) -> str:
    tunings = [t for t in ("tuned", "default") if t in set(w["tuning"])] or sorted(set(w["tuning"]))
    main = tunings[0]
    L = []
    add = L.append

    n_exp = w["experiment_id"].nunique()
    seeds, images = sorted(w["seed"].unique()), sorted(w["image"].unique())
    levels = {t: sorted(w[w.noise_type == t]["noise_level"].unique()) for t in _types(w)}
    add("# Analysis report: Wavelet vs Shearlet denoising\n")
    add(f"_Generated from `{csv_path}`. Numbers only; read the limitations at the end._\n")
    add("## 1. Data scope\n")
    add(f"- Experiments: {n_exp} (images: {', '.join(images)}; seeds: {', '.join(map(str, seeds))})")
    add("- Noise types and levels: " + "; ".join(
        f"{t}: {', '.join(f'{v:g}' for v in vs)}" for t, vs in levels.items()))
    add(f"- Parameter conditions: {', '.join(tunings)} "
        "(tuned = best of an equal-size grid per method, chosen by PSNR against the clean image)")
    if len(seeds) < 3:
        add(f"- **WARNING: only {len(seeds)} seed(s).** Results rest on very few noise draws; "
            "re-run with at least 5 seeds before drawing conclusions.")
    if len(images) < 3:
        add(f"- **WARNING: only {len(images)} image(s).** Image-type conclusions are not possible.")
    add("\n**How to read the tables.** Differences are shearlet minus wavelet, so a positive PSNR or "
        "SSIM difference means shearlet scored higher. `[95% CI]` is a t interval over paired runs; "
        "if it excludes 0 the difference is called clear. n is the number of paired runs "
        "(images × seeds). Runs from the same image are not independent, so intervals are "
        "somewhat optimistic.\n")

    add("## 2. Which method reduces Gaussian noise more effectively?\n")
    if "gaussian" in levels:
        for t in tunings:
            p = paired[(paired.tuning == t) & (paired.noise_type == "gaussian")]
            add(f"**{t} parameters**\n")
            add(_md_table(_paired_view(p)) + "\n")
    else:
        add("Gaussian noise was not part of this benchmark.\n")

    add("## 3. How do results change as noise intensity increases?\n")
    s = summary[summary.tuning == main]
    for nt in _types(s):
        d = s[s.noise_type == nt]
        lv = sorted(d.noise_level.unique())
        lo, hi = lv[0], lv[-1]
        if lo == hi:
            continue
        g = lambda m, lvl, col: d[(d.method == m) & (d.noise_level == lvl)][col].iloc[0]
        pp = paired[(paired.tuning == main) & (paired.noise_type == nt)].set_index("noise_level")
        add(f"- **{nt}** ({main}): from level {lo:g} to {hi:g}, mean PSNR gain over the noisy input "
            f"goes {g('wavelet', lo, 'psnr_gain'):.2f} → {g('wavelet', hi, 'psnr_gain'):.2f} dB (wavelet) "
            f"and {g('shearlet', lo, 'psnr_gain'):.2f} → {g('shearlet', hi, 'psnr_gain'):.2f} dB (shearlet); "
            f"the shearlet−wavelet PSNR gap goes {pp.loc[lo, 'psnr_diff_mean']:+.2f} → "
            f"{pp.loc[hi, 'psnr_diff_mean']:+.2f} dB.")
    add("")

    add("## 4. How well are image structures preserved? (SSIM)\n")
    add("SSIM is a proxy for structural similarity, not a direct measure of edge preservation. "
        f"Pooled over levels ({main} parameters):\n")
    pt = paired_type[paired_type.tuning == main]
    add(_md_table(pd.DataFrame({
        "noise": pt.noise_type, "n": pt.n,
        "SSIM diff [95% CI]": [_ci(a, b, c, "{:+.3f}") for a, b, c in
                               zip(pt.ssim_diff_mean, pt.ssim_diff_lo, pt.ssim_diff_hi)],
        "runs where PSNR and SSIM disagree": pt.metrics_disagree_runs,
        "verdict": pt.verdict})) + "\n")

    add("## 5. What is the runtime trade-off?\n")
    add(_md_table(runtime) + "\n")
    faster = float((w["time_ratio"] > 1).mean()) * 100
    add(f"Wavelet was faster in {faster:.0f}% of paired runs. Timings come from one machine; "
        "shearlet's one-off system setup (about 1–2 s per image size) is excluded from "
        "`processing_time` and would add to a first run.\n")

    add("## 6. Does performance change for different image types?\n")
    bi = by_image[by_image.tuning == main]
    add(f"Mean over levels and seeds ({main} parameters):\n")
    add(_md_table(pd.DataFrame({
        "noise": bi.noise_type, "image": bi.image, "n": bi.n,
        "PSNR diff dB [95% CI]": [_ci(a, b, c) for a, b, c in
                                  zip(bi.psnr_diff_mean, bi.psnr_diff_lo, bi.psnr_diff_hi)],
        "SSIM diff [95% CI]": [_ci(a, b, c, "{:+.3f}") for a, b, c in
                               zip(bi.ssim_diff_mean, bi.ssim_diff_lo, bi.ssim_diff_hi)],
        "verdict": bi.verdict})) + "\n")
    if len(images) < 3:
        add("_Too few images to say anything general about image types._\n")

    add("## 7. Are there cases where one method is preferable for a specific criterion?\n")
    pm = paired[paired.tuning == main]
    for v, grp in pm.groupby("verdict", sort=False):
        conds = ", ".join(f"{r.noise_type} {r.noise_level:g}" for r in grp.itertuples())
        add(f"- **{v}** ({main}): {conds}")
    add(f"- **runtime**: wavelet faster in {faster:.0f}% of paired runs.\n")

    add("## 8. Failure cases and artifacts\n")
    if failures.empty:
        add("No run scored below its noisy input, and PSNR and SSIM never favoured different methods.\n")
    else:
        cnt = (failures.groupby(["tuning", "noise_type", "kind"]).size()
                       .reset_index(name="runs"))
        add(_md_table(cnt) + "\n")
        add("Full list: `tables/failure_cases.csv`.\n")
    if gallery_cases:
        add("Notable cases to inspect by eye (`figures/gallery.png`):\n")
        for c in gallery_cases:
            add(f"- {c}")
        add("\nArtifacts (ringing, streaks, blur, blockiness) cannot be read from the metrics. "
            "Look at the gallery and describe what you see in your own words.\n")
    else:
        add("_No gallery: the saved images were not found (benchmark run with `--no-images`?)._\n")

    add("## 9. Limitations\n")
    lim = [
        f"Only {len(images)} test image(s) ({', '.join(images)}) and {len(seeds)} seed(s): "
        "results describe these images, not images in general.",
        "'Tuned' uses the clean image to choose each method's setting (oracle tuning). It shows "
        "each method's best case within an equal-size grid, not what a user without the clean "
        "image could obtain.",
        "For salt-and-pepper and speckle noise the noise level given to both methods is a blind "
        "estimate (MAD), because those noise levels are not standard deviations. "
        "Both transforms assume roughly Gaussian-like noise.",
        "Runtime was measured on one machine with Python/NumPy and is excluded from "
        "setup cost; relative timings may differ elsewhere.",
        "PSNR and SSIM are global scores. They do not capture local artifacts or perceptual "
        "quality, and they can disagree.",
        "Only one wavelet family (db4, level 3) and one shearlet configuration (3 scales) "
        "were tested.",
    ]
    edge = grid_edge_report(df)
    if not edge.empty:
        total = int(edge["at_edge"].sum())
        lim.append(f"In {total} tuned runs the best setting was at the edge of the tuning grid "
                   "(see runner warning), so tuned scores for those cases may understate what "
                   "the method could reach.")
    for item in lim:
        add(f"- {item}")
    add("")
    return "\n".join(L)


# --- driver -------------------------------------------------------------
def run_analysis(csv_path="data/results/benchmark_results.csv", root="data",
                 out_dir=None, progress=print) -> dict:
    csv_path, root = Path(csv_path), Path(root)
    out = Path(out_dir) if out_dir else root / "results" / "analysis"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    df = load_benchmark(csv_path)
    w = to_wide(df)
    if w.empty:
        raise ValueError("No complete (noisy + wavelet + shearlet) experiments found")
    summary = summary_table(w)
    paired = paired_table(w)
    paired_type = paired_table(w, by=["tuning", "noise_type"])
    by_image = paired_table(w, by=["tuning", "noise_type", "image"])
    failures = failure_cases(w)
    runtime = runtime_table(df)

    tables = {"summary_by_condition": summary, "paired_differences": paired,
              "paired_by_noise_type": paired_type, "paired_by_image": by_image,
              "failure_cases": failures, "runtime": runtime}
    for name, t in tables.items():
        t.to_csv(out / "tables" / f"{name}.csv", index=False)

    paths = {"tables": out / "tables"}
    for tuning in sorted(set(w["tuning"])):
        for metric in ("psnr", "ssim"):
            fig_metric_vs_level(summary, metric, tuning).savefig(
                out / "figures" / f"{metric}_vs_level_{tuning}.png", dpi=120, bbox_inches="tight")
        fig_difference_forest(paired, tuning).savefig(
            out / "figures" / f"difference_{tuning}.png", dpi=120, bbox_inches="tight")
        fig_by_image(by_image, tuning).savefig(
            out / "figures" / f"by_image_{tuning}.png", dpi=120, bbox_inches="tight")
        fig_tradeoff(summary, tuning).savefig(
            out / "figures" / f"tradeoff_{tuning}.png", dpi=120, bbox_inches="tight")

    main_tuning = "tuned" if "tuned" in set(w["tuning"]) else sorted(set(w["tuning"]))[0]
    gallery, cases = make_gallery(w, root, main_tuning)
    if gallery is not None:
        gallery.savefig(out / "figures" / "gallery.png", dpi=110, bbox_inches="tight")

    report = build_report(w, summary, paired, paired_type, by_image, failures,
                          runtime, df, cases, csv_path)
    report_path = out / "analysis_report.md"
    report_path.write_text(report, encoding="utf-8")
    paths["report"] = report_path
    paths["figures"] = out / "figures"
    progress(f"Report:  {report_path}\nTables:  {out / 'tables'}\nFigures: {out / 'figures'}")
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", default="data/results/benchmark_results.csv")
    ap.add_argument("--root", default="data", help="folder holding the saved images")
    ap.add_argument("--out", help="output folder (default: <root>/results/analysis)")
    a = ap.parse_args(argv)
    run_analysis(a.csv, a.root, a.out)


if __name__ == "__main__":
    main()