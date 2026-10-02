"""Experiment runner (Phase 7): one command -> complete benchmark CSV.

    python -m experiments.experiment_runner --preset initial
    python -m experiments.experiment_runner --preset full

For every image x noise type x noise level x seed it
  1. makes the noisy image (reproducible: fixed seed)
  2. runs Wavelet and Shearlet in two conditions:
       default : the config defaults
       tuned   : the best of an equal-size grid of threshold settings per
                 method (config.TUNING_GRIDS), chosen by PSNR against the
                 clean image ("oracle" tuning: a best-case for each method,
                 not something a user could do without the clean image)
  3. scores everything and appends rows to one CSV.

If the run is interrupted (or the CSV was locked), start it again with
--resume: experiments already in the CSV are skipped.

Outputs under <root> (default ./data):
    results/benchmark_results.csv        one row per (experiment, tuning, method)
    noisy/<id>.png, results/<method>/<id>_<tuning>.png   (unless --no-images)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config
from algorithms.wavelet_denoising import estimate_sigma
from evaluation.comparison import METHODS
from evaluation.metrics import evaluate
from evaluation.storage import _safe, effective_params, experiment_id, save_csv, save_png
from preprocessing.image_loader import load_image, load_sample
from preprocessing.noise import NOISE_TYPES, add_noise

BENCHMARK_COLUMNS = [
    "experiment_id", "tuning", "image", "noise_type", "noise_level", "seed",
    "method", "mse", "psnr", "ssim", "processing_time", "sigma_used",
    "sigma_source", "n_candidates", "tuned_value", "params"]
TUNINGS = ("default", "tuned")
_TEXT_COLUMNS = ("experiment_id", "tuning", "image", "noise_type", "method",
                 "sigma_source", "params")


def read_benchmark_csv(path) -> pd.DataFrame:
    """Read a benchmark CSV without letting pandas guess types wrongly.

    Plain read_csv turns the text 'true' (sigma_source) into a boolean and
    can lose the last digit of a float; this keeps text as text and floats exact.
    """
    return pd.read_csv(path, dtype={c: str for c in _TEXT_COLUMNS},
                       float_precision="round_trip")


# --- helpers ------------------------------------------------------------
def _load(image):
    """Image identifier: a built-in sample name or a path to an image file."""
    p = Path(str(image))
    if p.is_file():
        return p.stem, load_image(p)
    return str(image), load_sample(str(image))


def _image_name(image) -> str:
    p = Path(str(image))
    return p.stem if p.is_file() else str(image)


def _save_progress(rows, out_csv, progress, warned, final=False):
    """Write the CSV. A locked file (e.g. open in Excel) must not kill a long run."""
    df = pd.DataFrame(rows, columns=BENCHMARK_COLUMNS)
    try:
        save_csv(df, out_csv)
    except PermissionError:
        if not final:
            if not warned:
                progress("  note: cannot update the CSV right now (open in Excel or "
                         "another program?). Continuing; it is saved again at the end.")
                warned.append(True)
            return
        alt = out_csv.with_name(f"{out_csv.stem}_{time.strftime('%Y%m%d_%H%M%S')}{out_csv.suffix}")
        save_csv(df, alt)
        progress(f"\nWARNING: {out_csv.name} is locked. Close it in Excel / the CSV "
                 f"preview. Results were saved to {alt} instead.")


def _sigma_for(noise_type, level, noisy):
    source = config.SIGMA_SOURCE[noise_type]
    if source == "true":
        return float(level), "true"
    return estimate_sigma(noisy), "estimated"      # same estimator for both methods


def _grid(method):
    (knob, values), = config.TUNING_GRIDS[method].items()   # exactly one knob
    return knob, list(values)


def _median_time(method, noisy, sigma, kwargs, first_secs, repeats):
    times = [first_secs]
    for _ in range(max(repeats, 1) - 1):
        times.append(METHODS[method](noisy, sigma, **kwargs)[1])
    return float(np.median(times))


def _run_method(method, tuning, clean, noisy, sigma, repeats):
    """Return (image, seconds, knob_kwargs, n_candidates, tuned_value)."""
    knob, values = _grid(method)
    if tuning == "default":
        img, secs = METHODS[method](noisy, sigma)
        return img, _median_time(method, noisy, sigma, {}, secs, repeats), {}, 1, np.nan
    best = None
    for v in values:                                         # equal budget per method
        img, secs = METHODS[method](noisy, sigma, **{knob: v})
        score = evaluate(method, clean, img)["psnr"]
        if best is None or score > best[0]:
            best = (score, v, img, secs)
    _, v, img, secs = best
    kwargs = {knob: v}
    return img, _median_time(method, noisy, sigma, kwargs, secs, repeats), kwargs, len(values), float(v)


def _rows(exp, tuning, name, noise_type, level, seed, clean, noisy, sigma, source,
          outputs):
    """Table rows for one (experiment, tuning): baseline + each method."""
    common = dict(experiment_id=exp, tuning=tuning, image=name, noise_type=noise_type,
                  noise_level=level, seed=seed, sigma_used=sigma, sigma_source=source)
    rows = [dict(common, **evaluate("noisy", clean, noisy), n_candidates=0,
                 tuned_value=np.nan, params="{}")]
    for method, (img, secs, kwargs, n_cand, tuned_value) in outputs.items():
        used = effective_params(sigma, {method: kwargs})[method]
        rows.append(dict(common, **evaluate(method, clean, img, secs),
                         n_candidates=n_cand, tuned_value=tuned_value,
                         params=json.dumps(used, sort_keys=True)))
    return rows


# --- main ---------------------------------------------------------------
def run_matrix(images, noise_types, levels=None, seeds=None, tunings=TUNINGS,
               save_images=True, root="data", timing_repeats=None,
               out_csv=None, progress=print, resume=False) -> pd.DataFrame:
    """Run the whole benchmark and return (and save) the results table."""
    levels = levels or config.NOISE_LEVELS
    seeds = list(seeds) if seeds else [config.SEED]
    repeats = config.TIMING_REPEATS if timing_repeats is None else timing_repeats
    root = Path(root)
    out_csv = Path(out_csv) if out_csv else root / "results" / "benchmark_results.csv"
    for nt in noise_types:
        if nt not in NOISE_TYPES:
            raise ValueError(f"Unknown noise type '{nt}'. Choose from {NOISE_TYPES}")
    for t in tunings:
        if t not in TUNINGS:
            raise ValueError(f"Unknown tuning '{t}'. Choose from {TUNINGS}")

    jobs = [(img, nt, lv, sd) for img in images for nt in noise_types
            for lv in levels[nt] for sd in seeds]
    all_rows, t_start, warned = [], time.perf_counter(), []
    if resume and out_csv.exists():
        old = read_benchmark_csv(out_csv)
        if list(old.columns) != BENCHMARK_COLUMNS:
            raise ValueError(f"Cannot resume: {out_csv} has different columns")
        per_exp = old.groupby("experiment_id").size()
        done = set(per_exp[per_exp == len(tunings) * (1 + len(METHODS))].index)
        all_rows = old[old["experiment_id"].isin(done)].to_dict("records")
        before = len(jobs)
        jobs = [j for j in jobs
                if experiment_id(_image_name(j[0]), j[1], j[2], j[3]) not in done]
        progress(f"Resuming: {before - len(jobs)} of {before} experiments already done.")
    for i, (image, nt, lv, sd) in enumerate(jobs, start=1):
        t0 = time.perf_counter()
        name, clean = _load(image)
        noisy = add_noise(clean, nt, lv, seed=sd)
        sigma, source = _sigma_for(nt, lv, noisy)
        exp = experiment_id(name, nt, lv, sd)
        if save_images:
            save_png(clean, root / "original" / f"{_safe(name)}.png")
            save_png(noisy, root / "noisy" / f"{exp}.png")
        for tuning in tunings:
            outputs = {m: _run_method(m, tuning, clean, noisy, sigma, repeats)
                       for m in METHODS}
            all_rows += _rows(exp, tuning, name, nt, lv, sd, clean, noisy,
                              sigma, source, outputs)
            if save_images:
                for m, (img, *_rest) in outputs.items():
                    save_png(img, root / "results" / m / f"{exp}_{tuning}.png")
        _save_progress(all_rows, out_csv, progress, warned)                   # crash-safe
        progress(f"[{i}/{len(jobs)}] {exp}  ({time.perf_counter() - t0:.1f} s)")
    _save_progress(all_rows, out_csv, progress, warned, final=True)
    df = pd.DataFrame(all_rows, columns=BENCHMARK_COLUMNS)
    progress(f"\n{len(df)} rows -> {out_csv}  (total {time.perf_counter() - t_start:.0f} s)")
    return df


def grid_edge_report(df: pd.DataFrame) -> pd.DataFrame:
    """Tuned runs whose best value is the smallest or largest candidate.

    If this is not empty, the true optimum may lie outside the grid, so the
    'tuned' score for that group is a lower bound on what the method could do.
    """
    t = df[(df["tuning"] == "tuned") & (df["method"] != "noisy")].copy()
    if t.empty:
        return pd.DataFrame(columns=["noise_type", "method", "runs", "at_edge"])
    def at_edge(row):
        _, values = _grid(row["method"])
        return row["tuned_value"] in (min(values), max(values))
    t["at_edge"] = t.apply(at_edge, axis=1)
    out = (t.groupby(["noise_type", "method"], sort=False)["at_edge"]
             .agg(runs="size", at_edge="sum").reset_index())
    return out[out["at_edge"] > 0]


def overview(df: pd.DataFrame) -> pd.DataFrame:
    """Mean metrics per tuning/noise type/method. A summary, not a conclusion."""
    return (df.groupby(["tuning", "noise_type", "method"], sort=False)
              [["psnr", "ssim", "mse", "processing_time"]].mean())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preset", choices=("initial", "full"), default="initial",
                    help="initial: first image, Gaussian only; full: all images, all noise types")
    ap.add_argument("--images", nargs="+", help="sample names or file paths (overrides preset)")
    ap.add_argument("--noise", nargs="+", choices=NOISE_TYPES, help="noise types (overrides preset)")
    ap.add_argument("--seeds", nargs="+", type=int, help="random seeds (default: config.SEED)")
    ap.add_argument("--no-tuning", action="store_true", help="only the default-parameter runs")
    ap.add_argument("--no-images", action="store_true", help="do not save image files")
    ap.add_argument("--root", default="data", help="output folder (default: data)")
    ap.add_argument("--repeats", type=int, help="timing repeats (default: config.TIMING_REPEATS)")
    ap.add_argument("--resume", action="store_true",
                    help="skip experiments already in the CSV (continue an interrupted run)")
    a = ap.parse_args(argv)

    if a.preset == "initial":
        images, noise = config.TEST_IMAGES[:1], ["gaussian"]
    else:
        images, noise = config.TEST_IMAGES, list(NOISE_TYPES)
    df = run_matrix(a.images or images, a.noise or noise, seeds=a.seeds,
                    tunings=("default",) if a.no_tuning else TUNINGS,
                    save_images=not a.no_images, root=a.root, timing_repeats=a.repeats,
                    resume=a.resume)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
    print("\nOVERVIEW (means; not conclusions)")
    print(overview(df).to_string())
    edge = grid_edge_report(df)
    if not edge.empty:
        print("\nWARNING: best tuned value was at the edge of the grid for:")
        print(edge.to_string(index=False))
        print("The true optimum may be outside the grid; treat those tuned scores"
              " as lower bounds.")


if __name__ == "__main__":
    main()