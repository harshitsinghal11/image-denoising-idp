"""Phase 9 checks. Run from the project root:  python -m tests.test_app"""
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import config
from app import core
from experiments.experiment_runner import BENCHMARK_COLUMNS
from preprocessing.image_loader import load_sample


def _png(arr_rgb) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr_rgb).save(buf, format="PNG")
    return buf.getvalue()


def test_load_uploaded_colour_and_nonsquare():
    rgb = np.random.default_rng(0).integers(0, 256, (150, 220, 3), dtype=np.uint8)
    img, original = core.load_uploaded(_png(rgb), 128)
    assert original == (220, 150)
    assert img.shape == (128, 128) and img.dtype == np.float64
    assert 0 <= img.min() and img.max() <= 255


def test_load_uploaded_errors():
    for bad in (lambda: core.load_uploaded(b"not an image", 128),
                lambda: core.load_uploaded(_png(np.zeros((40, 40, 3), np.uint8)), 128)):
        try:
            bad()
        except ValueError as exc:
            assert str(exc)
            continue
        raise AssertionError("expected ValueError")


def test_image_helpers():
    a = np.arange(64 * 64, dtype=float).reshape(64, 64) % 256
    assert core.display_image(a, 2).size == (128, 128)
    back = np.asarray(Image.open(io.BytesIO(core.png_bytes(a))))
    assert back.shape == (64, 64) and back.dtype == np.uint8
    for cx, cy in ((0, 0), (1, 1), (0.5, 0.5)):                # edges stay inside
        assert core.crop_region(a, cx, cy, 32).shape == (32, 32)
    assert core.crop_region(a, 0.5, 0.5, 500).shape == (64, 64)


def test_noise_sigma_sources():
    clean = load_sample("camera", 128)
    from preprocessing.noise import add_noise
    g = add_noise(clean, "gaussian", 20, seed=1)
    assert core.noise_sigma("gaussian", 20, g, "config") == (20.0, "true")
    s, src = core.noise_sigma("gaussian", 20, g, "estimate")
    assert src == "estimated" and 14 < s < 26
    sp = add_noise(clean, "salt_pepper", 0.05, seed=1)
    assert core.noise_sigma("salt_pepper", 0.05, sp, "config")[1] == "estimated"
    try:
        core.noise_sigma("gaussian", 20, g, "bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_run_experiment_and_exports():
    clean = load_sample("camera", 128)
    run = core.run_experiment(clean, "gaussian", 20, 42)
    res = run["result"]
    assert list(res.table["method"]) == ["noisy", "wavelet", "shearlet"]
    assert set(res.images) == {"clean", "noisy", "wavelet", "shearlet"}
    assert run["sigma"] == 20 and run["sigma_source"] == "true" and not run["tuned"]
    # same seed -> identical noisy image
    run2 = core.run_experiment(clean, "gaussian", 20, 42)
    assert np.array_equal(run["result"].images["noisy"], run2["result"].images["noisy"])

    df = core.results_dataframe(run, "camera")
    assert list(df.columns) == BENCHMARK_COLUMNS and len(df) == 3
    assert (df["experiment_id"] == "camera_gaussian_20_seed42").all()
    p = json.loads(df.loc[df.method == "shearlet", "params"].iloc[0])
    assert p["factor"] == config.SHEARLET_THRESHOLD_FACTOR

    z = zipfile.ZipFile(io.BytesIO(core.build_zip(run, "camera")))
    assert set(z.namelist()) == {"original.png", "noisy.png", "wavelet_denoised.png",
                                 "shearlet_denoised.png", "results.csv", "parameters.json"}
    assert Image.open(io.BytesIO(z.read("wavelet_denoised.png"))).size == (128, 128)
    assert len(pd.read_csv(io.BytesIO(z.read("results.csv")))) == 3
    assert len(core.takeaways(run)) == 3
    assert "psnr" in core.rounded_table(res.table).columns


def test_custom_params_reach_the_denoisers():
    clean = load_sample("camera", 128)
    p = core.default_params()
    p["shearlet"]["factor"] = 6.0
    p["wavelet"]["wavelet"] = "haar"
    a = core.run_experiment(clean, "gaussian", 20, 42)
    b = core.run_experiment(clean, "gaussian", 20, 42, params=p)
    assert a["result"].table.loc[2, "psnr"] != b["result"].table.loc[2, "psnr"]
    assert a["result"].table.loc[1, "psnr"] != b["result"].table.loc[1, "psnr"]
    assert core.default_params()["shearlet"]["factor"] == config.SHEARLET_THRESHOLD_FACTOR  # untouched


def test_autotune_picks_from_grid_and_is_not_worse():
    clean = load_sample("camera", 128)
    base = core.run_experiment(clean, "speckle", 0.2, 42)
    tuned = core.run_experiment(clean, "speckle", 0.2, 42, tune=True)
    for m, knob in (("wavelet", "threshold_scale"), ("shearlet", "factor")):
        info = tuned["tuned"][m]
        assert info["knob"] == knob and info["candidates"] == 8
        assert info["value"] in config.TUNING_GRIDS[m][knob]
        row = 1 if m == "wavelet" else 2
        assert tuned["result"].table.loc[row, "psnr"] >= base["result"].table.loc[row, "psnr"] - 1e-9
    df = core.results_dataframe(tuned, "camera")
    assert (df["tuning"] == "tuned").all() and df.loc[df.method == "wavelet", "tuned_value"].notna().all()


def test_every_noise_type_runs():
    clean = load_sample("moon", 128)
    for nt, (_, _, _, _, default, _) in core.NOISE_UI.items():
        run = core.run_experiment(clean, nt, default, 1)
        assert np.isfinite(run["result"].table[["mse", "psnr", "ssim"]].to_numpy()).all(), nt


def test_streamlit_app_end_to_end():
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        print("SKIP streamlit AppTest unavailable in this Streamlit version")
        return
    script = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")
    at = AppTest.from_file(script, default_timeout=180).run()
    assert not at.exception, at.exception
    assert any("Run comparison" in b.label for b in at.button)
    assert not at.dataframe                                    # nothing shown before a run
    at.selectbox(key="size").select(128)
    at.selectbox(key="noise_type").select("salt_pepper")
    at.button(key="run_button").click().run()
    assert not at.exception, at.exception
    assert "run" in at.session_state
    run = at.session_state["run"]
    assert run["noise_type"] == "salt_pepper" and run["sigma_source"] == "estimated"
    assert len(at.dataframe) >= 2                              # metrics + differences
    assert len(at.tabs) == 5
    # tuned run through the UI
    at.checkbox(key="tune").check()
    at.button(key="run_button").click().run()
    assert not at.exception, at.exception
    assert at.session_state["run"]["tune"] and at.session_state["run"]["tuned"]


if __name__ == "__main__":
    for fn in [v for k, v in dict(globals()).items() if k.startswith("test_")]:
        fn(); print("PASS", fn.__name__)