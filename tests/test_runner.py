"""Phase 7 checks. Run from the project root:  python -m tests.test_runner"""
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import config
from experiments.experiment_runner import (BENCHMARK_COLUMNS, grid_edge_report,
                                           main, read_benchmark_csv, run_matrix)

QUIET = dict(progress=lambda *_: None, timing_repeats=1)


def _run(root, **kw):
    args = dict(images=["camera"], noise_types=["gaussian"], levels={"gaussian": [20]},
                root=root, **QUIET)
    args.update(kw)
    return run_matrix(**args)


def test_structure_and_files():
    with tempfile.TemporaryDirectory() as d:
        df = _run(d)
        assert list(df.columns) == BENCHMARK_COLUMNS
        assert len(df) == 6                                  # 2 tunings x (noisy+2 methods)
        assert set(df["tuning"]) == {"default", "tuned"}
        assert set(df["method"]) == {"noisy", "wavelet", "shearlet"}
        csv = pd.read_csv(Path(d) / "results" / "benchmark_results.csv")
        assert len(csv) == 6
        exp = "camera_gaussian_20_seed42"
        for rel in (f"noisy/{exp}.png", "original/camera.png",
                    f"results/wavelet/{exp}_default.png", f"results/wavelet/{exp}_tuned.png",
                    f"results/shearlet/{exp}_default.png", f"results/shearlet/{exp}_tuned.png"):
            assert (Path(d) / rel).is_file(), rel
        assert not list(Path(d).rglob("*.tmp"))


def test_values_are_valid_and_params_stored():
    with tempfile.TemporaryDirectory() as d:
        df = _run(d)
        num = df[["mse", "psnr", "ssim", "sigma_used"]]
        assert np.isfinite(num.to_numpy()).all()
        assert (df.loc[df["method"] != "noisy", "processing_time"] > 0).all()
        assert df.loc[df["method"] == "noisy", "processing_time"].isna().all()
        assert (df["sigma_used"] == 20).all() and (df["sigma_source"] == "true").all()
        row = df[(df.method == "shearlet") & (df.tuning == "tuned")].iloc[0]
        p = json.loads(row["params"])
        assert p["factor"] == row["tuned_value"] and p["scales"] == config.SHEARLET_SCALES
        assert row["n_candidates"] == len(config.TUNING_GRIDS["shearlet"]["factor"])


def test_equal_tuning_budget_and_default_in_grid():
    n = {m: len(next(iter(g.values()))) for m, g in config.TUNING_GRIDS.items()}
    assert len(set(n.values())) == 1                         # same budget per method
    assert config.WAVELET_THRESHOLD_SCALE in config.TUNING_GRIDS["wavelet"]["threshold_scale"]
    assert config.SHEARLET_THRESHOLD_FACTOR in config.TUNING_GRIDS["shearlet"]["factor"]


def test_tuned_is_never_worse_than_default():
    with tempfile.TemporaryDirectory() as d:
        df = _run(d, noise_types=["speckle"], levels={"speckle": [0.2]})
        for m in ("wavelet", "shearlet"):
            psnr = df[df.method == m].set_index("tuning")["psnr"]
            assert psnr["tuned"] >= psnr["default"] - 1e-12


def test_reproducible_apart_from_timing():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        x, y = _run(a, save_images=False), _run(b, save_images=False)
        cols = [c for c in BENCHMARK_COLUMNS if c != "processing_time"]
        pd.testing.assert_frame_equal(x[cols], y[cols])


def test_seeds_and_sigma_sources():
    with tempfile.TemporaryDirectory() as d:
        df = _run(d, noise_types=["salt_pepper"], levels={"salt_pepper": [0.05]},
                  seeds=[1, 2], tunings=("default",), save_images=False)
        assert df["experiment_id"].nunique() == 2
        assert (df["sigma_source"] == "estimated").all()
        # both methods got the same estimated sigma within an experiment
        assert df.groupby("experiment_id")["sigma_used"].nunique().eq(1).all()
        a, b = (df[df.seed == s].query("method == 'wavelet'")["psnr"].iloc[0] for s in (1, 2))
        assert a != b


def test_option_switches_and_errors():
    with tempfile.TemporaryDirectory() as d:
        df = _run(d, tunings=("default",), save_images=False)
        assert set(df["tuning"]) == {"default"} and len(df) == 3
        assert not (Path(d) / "noisy").exists()
        for bad in (lambda: _run(d, noise_types=["nope"]),
                    lambda: _run(d, tunings=("best",)),
                    lambda: _run(d, images=["not_an_image"])):
            try:
                bad()
            except ValueError:
                continue
            raise AssertionError("expected ValueError")


def test_grid_edge_report_and_cli():
    df = pd.DataFrame({"tuning": ["tuned"] * 2, "method": ["wavelet", "shearlet"],
                       "noise_type": ["gaussian"] * 2,
                       "tuned_value": [max(config.TUNING_GRIDS["wavelet"]["threshold_scale"]), 3.0]})
    rep = grid_edge_report(df)
    assert list(rep["method"]) == ["wavelet"]                # only the edge hit
    with tempfile.TemporaryDirectory() as d:
        main(["--images", "camera", "--noise", "gaussian", "--no-tuning",
              "--no-images", "--repeats", "1", "--root", d])
        assert (Path(d) / "results" / "benchmark_results.csv").is_file()


def test_replace_retries_then_succeeds():
    import evaluation.storage as storage
    real_replace, real_sleep, calls = os.replace, storage.time.sleep, {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError("locked")        # what Windows raises while a file is in use
        return real_replace(src, dst)

    with tempfile.TemporaryDirectory() as d:
        os.replace, storage.time.sleep = flaky, lambda *_: None
        try:
            storage.save_csv(pd.DataFrame({"a": [1]}), Path(d) / "x.csv")
        finally:
            os.replace, storage.time.sleep = real_replace, real_sleep
        assert (Path(d) / "x.csv").is_file() and calls["n"] == 3
        assert not list(Path(d).glob("*.tmp"))


def test_locked_csv_does_not_crash_the_run():
    import evaluation.storage as storage
    real_replace, real_sleep = os.replace, storage.time.sleep

    def locked(src, dst):
        if str(dst).endswith("benchmark_results.csv"):
            raise PermissionError("locked")        # CSV open in Excel the whole time
        return real_replace(src, dst)

    with tempfile.TemporaryDirectory() as d:
        msgs = []
        os.replace, storage.time.sleep = locked, lambda *_: None
        try:
            df = _run(d, levels={"gaussian": [10, 20]}, save_images=False, progress=msgs.append)
        finally:
            os.replace, storage.time.sleep = real_replace, real_sleep
        assert len(df) == 12                                   # nothing lost
        saved = list((Path(d) / "results").glob("benchmark_results_*.csv"))
        assert len(saved) == 1 and len(pd.read_csv(saved[0])) == 12
        assert any("locked" in m for m in msgs)


def test_resume_skips_finished_experiments():
    lv = {"gaussian": [10, 20]}
    with tempfile.TemporaryDirectory() as d:
        full = _run(d, levels=lv, save_images=False)
        csv = Path(d) / "results" / "benchmark_results.csv"
        part = read_benchmark_csv(csv)
        first = part["experiment_id"].iloc[0]
        part[part["experiment_id"] == first].to_csv(csv, index=False)   # pretend: interrupted
        msgs = []
        again = _run(d, levels=lv, save_images=False, resume=True, progress=msgs.append)
        assert len(again) == 12 and again["experiment_id"].nunique() == 2
        assert sum(m.startswith("[") for m in msgs) == 1               # only one experiment re-run
        assert any("Resuming: 1 of 2" in m for m in msgs)
        cols = [c for c in BENCHMARK_COLUMNS if c != "processing_time"]
        pd.testing.assert_frame_equal(full[cols], again[cols], check_dtype=False)
        msgs.clear()
        _run(d, levels=lv, save_images=False, resume=True, progress=msgs.append)
        assert not any(m.startswith("[") for m in msgs)                # complete file: nothing to do


if __name__ == "__main__":
    for fn in [v for k, v in dict(globals()).items() if k.startswith("test_")]:
        fn(); print("PASS", fn.__name__)