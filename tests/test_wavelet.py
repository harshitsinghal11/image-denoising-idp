"""Phase 3 checks. Run from the project root:  python -m tests.test_wavelet"""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import config
from algorithms.wavelet_denoising import denoise, denoise_timed, estimate_sigma
from preprocessing.image_loader import load_sample
from preprocessing.noise import add_noise


def _case(name="camera", sigma=20):
    clean = load_sample(name)
    return clean, add_noise(clean, "gaussian", sigma, seed=config.SEED)


def test_output_contract():
    clean, noisy = _case()
    before = noisy.copy()
    out = denoise(noisy, 20)
    assert out.shape == noisy.shape and out.dtype == np.float64
    assert np.isfinite(out).all() and out.min() >= 0 and out.max() <= 255
    assert np.array_equal(noisy, before)                   # input untouched


def test_improves_psnr_and_ssim():
    for name in config.TEST_IMAGES:
        for sigma in config.GAUSSIAN_SIGMAS:
            clean, noisy = _case(name, sigma)
            out = denoise(noisy, sigma)
            n = np.clip(noisy, 0, 255)
            assert psnr(clean, out, data_range=255) > psnr(clean, n, data_range=255), (name, sigma)
            assert ssim(clean, out, data_range=255) > ssim(clean, n, data_range=255), (name, sigma)


def test_all_modes_and_methods_run():
    clean, noisy = _case()
    for mode in ("soft", "hard"):
        for method in ("bayes", "universal"):
            out = denoise(noisy, 20, mode=mode, method=method)
            assert psnr(clean, out, data_range=255) > psnr(clean, np.clip(noisy, 0, 255), data_range=255)


def test_zero_threshold_is_perfect_reconstruction():
    clean, _ = _case()
    assert np.abs(denoise(clean, 20, threshold_scale=0) - clean).max() < 1e-8


def test_sigma_estimate():
    _, noisy = _case(sigma=20)
    assert abs(estimate_sigma(noisy) - 20) < 3
    assert denoise(noisy).shape == noisy.shape            # sigma=None path


def test_non_square_and_odd_sizes():
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 255, (97, 130))
    assert denoise(img, 10).shape == (97, 130)


def test_level_is_capped_and_timed():
    _, noisy = _case()
    assert denoise(noisy, 20, level=50).shape == noisy.shape
    out, secs = denoise_timed(noisy, 20)
    assert secs > 0 and out.shape == noisy.shape


def test_bad_inputs():
    _, noisy = _case()
    for bad in (lambda: denoise(noisy, 20, mode="x"),
                lambda: denoise(noisy, 20, method="x"),
                lambda: denoise(noisy, 20, wavelet="nope"),
                lambda: denoise(noisy, -1),
                lambda: denoise(np.zeros((8, 8, 3)), 20)):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


if __name__ == "__main__":
    for fn in [v for k, v in dict(globals()).items() if k.startswith("test_")]:
        fn(); print("PASS", fn.__name__)