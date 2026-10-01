"""Phase 5 checks. Run from the project root:  python -m tests.test_evaluation"""
import math

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

import config
from evaluation.comparison import compare, difference_table
from evaluation.metrics import METRIC_COLUMNS, evaluate, mse, psnr, ssim
from preprocessing.image_loader import load_sample
from preprocessing.noise import add_noise


def _case(sigma=20):
    clean = load_sample("camera")
    return clean, add_noise(clean, "gaussian", sigma, seed=config.SEED)


def test_identical_images():
    clean, _ = _case()
    assert mse(clean, clean) == 0.0
    assert psnr(clean, clean) == math.inf
    assert abs(ssim(clean, clean) - 1.0) < 1e-12


def test_metrics_match_reference_implementations():
    clean, noisy = _case()
    n = np.clip(noisy, 0, 255)
    assert abs(psnr(clean, noisy) - sk_psnr(clean, n, data_range=255)) < 1e-9
    assert abs(ssim(clean, noisy) - sk_ssim(clean, n, data_range=255)) < 1e-12
    assert abs(mse(clean, noisy) - np.mean((clean - n) ** 2)) < 1e-9


def test_known_value():
    a, b = np.zeros((16, 16)), np.full((16, 16), 10.0)      # mse = 100
    assert mse(a, b) == 100.0
    assert abs(psnr(a, b) - 10 * math.log10(255 ** 2 / 100)) < 1e-12


def test_metrics_clip_the_processed_image_only():
    clean = np.full((16, 16), 255.0)
    over = np.full((16, 16), 300.0)                          # clipped to 255
    assert mse(clean, over) == 0.0


def test_bad_inputs():
    z = np.zeros((16, 16))
    for bad in (lambda: mse(z, np.zeros((8, 8))), lambda: mse(np.zeros((4, 4, 3)), z)):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_metric_contract():
    clean, noisy = _case()
    rec = evaluate("x", clean, noisy, 0.5)
    assert tuple(rec) == METRIC_COLUMNS


def test_compare_structure_and_same_input():
    clean, noisy = _case()
    before = noisy.copy()
    res = compare(clean, noisy, 20)
    assert list(res.table["method"]) == ["noisy", "wavelet", "shearlet"]
    assert tuple(res.table.columns) == METRIC_COLUMNS
    assert np.array_equal(noisy, before)                     # input untouched
    assert np.array_equal(res.images["noisy"], noisy)        # both saw this one
    assert np.isnan(res.table.loc[0, "processing_time"])     # baseline: no time
    assert (res.table.loc[1:, "processing_time"] > 0).all()
    assert "winner" not in " ".join(res.differences.columns).lower()


def test_differences_are_correct():
    clean, noisy = _case()
    res = compare(clean, noisy, 20)
    t = res.table.set_index("method")
    d = res.differences.set_index("metric")
    for m in ("mse", "psnr", "ssim", "processing_time"):
        assert abs(d.loc[m, "difference"] - (t.loc["shearlet", m] - t.loc["wavelet", m])) < 1e-12
    expect = 100 * (t.loc["shearlet", "mse"] - t.loc["wavelet", "mse"]) / t.loc["wavelet", "mse"]
    assert abs(d.loc["mse", "relative_change_pct"] - expect) < 1e-9
    assert np.isnan(d.loc["psnr", "relative_change_pct"])


def test_params_are_forwarded_and_single_method():
    clean, noisy = _case()
    a = compare(clean, noisy, 20, methods=("shearlet",), params={"shearlet": {"factor": 2.0}})
    b = compare(clean, noisy, 20, methods=("shearlet",), params={"shearlet": {"factor": 4.0}})
    assert a.table.loc[1, "psnr"] != b.table.loc[1, "psnr"]
    assert a.differences.empty


def test_unknown_method():
    clean, noisy = _case()
    try:
        compare(clean, noisy, 20, methods=("nope",))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


if __name__ == "__main__":
    for fn in [v for k, v in dict(globals()).items() if k.startswith("test_")]:
        fn(); print("PASS", fn.__name__)