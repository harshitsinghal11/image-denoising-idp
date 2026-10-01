"""Phase 6 checks. Run from the project root:  python -m tests.test_visualization"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import config
from evaluation.comparison import compare
from evaluation.storage import (MASTER_COLUMNS, experiment_id, load_results,
                                save_experiment, save_png)
from preprocessing.image_loader import load_sample
from preprocessing.noise import add_noise
from visualization.plots import (plot_all_metrics, plot_image_comparison,
                                 plot_metric_bar, plot_metric_vs_noise)


def _result(sigma=20, name="camera"):
    clean = load_sample(name)
    noisy = add_noise(clean, "gaussian", sigma, seed=config.SEED)
    return compare(clean, noisy, sigma)


def test_experiment_id():
    assert experiment_id("camera", "gaussian", 20, 42) == "camera_gaussian_20_seed42"
    assert experiment_id("camera", "speckle", 0.2, 42) == "camera_speckle_0p2_seed42"
    assert experiment_id("my photo.jpg", "gaussian", 10, 1) == "my_photo_jpg_gaussian_10_seed1"
    assert experiment_id("a", "gaussian", 10, 1) != experiment_id("a", "gaussian", 20, 1)


def test_save_png_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        arr = np.random.default_rng(0).uniform(-20, 300, (32, 32))
        p = save_png(arr, Path(d) / "x" / "a.png")
        back = np.asarray(Image.open(p))
        assert back.shape == (32, 32) and back.dtype == np.uint8
        assert back.min() >= 0 and back.max() <= 255


def test_save_experiment_files_and_master_csv():
    res = _result()
    with tempfile.TemporaryDirectory() as d:
        paths = save_experiment(res, "camera", "gaussian", 20, 42, sigma=20, root=d)
        exp = "camera_gaussian_20_seed42"
        root = Path(d)
        for rel in (f"original/camera.png", f"noisy/{exp}.png",
                    f"results/wavelet/{exp}.png", f"results/shearlet/{exp}.png",
                    f"results/tables/{exp}.csv", "results/results.csv"):
            assert (root / rel).is_file(), rel
        master = load_results(d)
        assert list(master.columns) == MASTER_COLUMNS
        assert list(master["method"]) == ["noisy", "wavelet", "shearlet"]
        p = json.loads(master.loc[master["method"] == "shearlet", "params"].iloc[0])
        assert p["scales"] == config.SHEARLET_SCALES and p["sigma"] == 20
        assert not list(root.rglob("*.tmp"))                    # no leftovers


def test_resave_replaces_and_other_experiments_kept():
    with tempfile.TemporaryDirectory() as d:
        r20, r10 = _result(20), _result(10)
        save_experiment(r20, "camera", "gaussian", 20, 42, sigma=20, root=d)
        save_experiment(r10, "camera", "gaussian", 10, 42, sigma=10, root=d)
        save_experiment(r20, "camera", "gaussian", 20, 42, sigma=20, root=d)   # again
        m = load_results(d)
        assert len(m) == 6                                       # 2 experiments x 3 rows
        assert m.groupby("experiment_id").size().eq(3).all()


def test_param_overrides_are_recorded():
    res = _result()
    with tempfile.TemporaryDirectory() as d:
        save_experiment(res, "camera", "gaussian", 20, 42, sigma=20,
                        params={"shearlet": {"factor": 2.5}}, root=d)
        m = load_results(d)
        p = json.loads(m.loc[m["method"] == "shearlet", "params"].iloc[0])
        assert p["factor"] == 2.5


def test_plots_return_and_save_figures():
    res = _result()
    with tempfile.TemporaryDirectory() as d:
        figs = [
            plot_image_comparison(res.images, res.table, "t", Path(d) / "a.png"),
            plot_all_metrics(res.table, "t", Path(d) / "b.png"),
            plot_metric_bar(res.table, "psnr", save_path=Path(d) / "c.png"),
            plot_metric_bar(res.table, "processing_time", save_path=Path(d) / "d.png"),
        ]
        assert all(f is not None for f in figs)
        for n in "abcd":
            assert (Path(d) / f"{n}.png").stat().st_size > 1000


def test_metric_vs_noise_and_errors():
    with tempfile.TemporaryDirectory() as d:
        for s in (10, 20, 30):
            save_experiment(_result(s), "camera", "gaussian", s, 42, sigma=s, root=d)
        m = load_results(d)
        fig = plot_metric_vs_noise(m, "psnr", "gaussian", save_path=Path(d) / "n.png")
        assert fig is not None and (Path(d) / "n.png").is_file()
        for bad in (lambda: plot_metric_bar(_result().table, "nope"),
                    lambda: plot_metric_vs_noise(m, "psnr", "speckle")):
            try:
                bad()
            except ValueError:
                continue
            raise AssertionError("expected ValueError")


if __name__ == "__main__":
    for fn in [v for k, v in dict(globals()).items() if k.startswith("test_")]:
        fn(); print("PASS", fn.__name__)