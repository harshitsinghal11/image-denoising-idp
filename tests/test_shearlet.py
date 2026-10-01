"""Phase 4 checks. Run from the project root:  python -m tests.test_shearlet"""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import config
from algorithms.shearlet_denoising import denoise, denoise_timed, get_system, setup_seconds
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
    assert np.array_equal(noisy, before)                    # input untouched


def test_improves_psnr_and_ssim():
    for name in config.TEST_IMAGES:
        for sigma in config.GAUSSIAN_SIGMAS:
            clean, noisy = _case(name, sigma)
            out = denoise(noisy, sigma)
            n = np.clip(noisy, 0, 255)
            assert psnr(clean, out, data_range=255) > psnr(clean, n, data_range=255), (name, sigma)
            assert ssim(clean, out, data_range=255) > ssim(clean, n, data_range=255), (name, sigma)


def test_zero_factor_is_perfect_reconstruction():
    clean, _ = _case()
    assert np.abs(denoise(clean, 20, factor=0) - clean).max() < 1e-6


def test_hard_and_soft_run():
    clean, noisy = _case()
    for mode in ("hard", "soft"):
        out = denoise(noisy, 20, mode=mode)
        assert psnr(clean, out, data_range=255) > psnr(clean, np.clip(noisy, 0, 255), data_range=255)


def test_sigma_none_uses_estimate():
    _, noisy = _case(sigma=20)
    assert denoise(noisy).shape == noisy.shape


def test_non_square_is_padded_and_cropped():
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 255, (200, 256))
    assert denoise(img, 10).shape == (200, 256)


def test_system_is_cached_and_timed():
    get_system(256, 3)
    assert setup_seconds(256, 3) > 0
    assert get_system(256, 3) is get_system(256, 3)         # same object -> cached
    _, noisy = _case()
    out, secs = denoise_timed(noisy, 20)
    assert secs > 0 and out.shape == noisy.shape


def test_too_small_image_gives_readable_error():
    for bad in (lambda: denoise(np.zeros((64, 64)), 10),
                lambda: denoise(np.zeros((32, 32)), 10)):
        try:
            bad()
        except ValueError as exc:
            assert "not available" in str(exc)
            continue
        raise AssertionError("expected ValueError")


def test_bad_inputs():
    _, noisy = _case()
    for bad in (lambda: denoise(noisy, 20, mode="x"),
                lambda: denoise(noisy, -1),
                lambda: denoise(noisy, 20, factor=-1),
                lambda: denoise(np.zeros((8, 8, 3)), 20)):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


if __name__ == "__main__":
    for fn in [v for k, v in dict(globals()).items() if k.startswith("test_")]:
        fn(); print("PASS", fn.__name__)