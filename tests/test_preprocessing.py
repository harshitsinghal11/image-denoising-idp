import numpy as np
import config

from preprocessing.image_loader import load_sample, prepare, load_image
from preprocessing.noise import NOISE_TYPES, add_noise

LEVELS = {"gaussian": 20, "salt_pepper": 0.05, "speckle": 0.2, "mixed": 20}


def test_loader_contract():
    for name in config.TEST_IMAGES:
        img = load_sample(name)
        assert img.shape == (config.IMAGE_SIZE, config.IMAGE_SIZE), name
        assert img.dtype == np.float64
        assert 0.0 <= img.min() and img.max() <= 255.0


def test_rgb_and_nonsquare_inputs():
    rgb = np.random.default_rng(0).integers(0, 256, (120, 200, 3), dtype=np.uint8)
    assert prepare(rgb).shape == (config.IMAGE_SIZE, config.IMAGE_SIZE)


def test_all_noise_reproducible_and_clean_untouched():
    img = load_sample("camera")
    before = img.copy()
    for nt in NOISE_TYPES:
        a = add_noise(img, nt, LEVELS[nt], seed=1)
        b = add_noise(img, nt, LEVELS[nt], seed=1)
        c = add_noise(img, nt, LEVELS[nt], seed=2)
        assert a.shape == img.shape and a.dtype == np.float64, nt
        assert np.array_equal(a, b), nt                  # same seed -> same
        assert not np.array_equal(a, c), nt              # new seed -> new
        assert np.array_equal(img, before), nt           # clean preserved


def test_gaussian_level():
    img = load_sample("camera")
    noisy = add_noise(img, "gaussian", 20, seed=config.SEED)
    assert abs((noisy - img).std() - 20) < 0.5


def test_salt_pepper_properties():
    img = load_sample("camera")
    noisy = add_noise(img, "salt_pepper", 0.05, seed=config.SEED)
    changed = noisy != img
    # a corrupted pixel can coincide with its original value (0 or 255), so
    # the changed fraction is a little below the amount
    assert 0.04 < changed.mean() <= 0.055
    assert set(np.unique(noisy[changed])) <= {0.0, 255.0}
    salt = (noisy[changed] == 255).mean()
    assert 0.45 < salt < 0.55                            # roughly half/half


def test_speckle_is_multiplicative():
    img = load_sample("camera")
    noisy = add_noise(img, "speckle", 0.2, seed=config.SEED)
    bright = img > 30
    rel = (noisy[bright] - img[bright]) / img[bright]
    assert abs(rel.std() - 0.2) < 0.01                   # relative std ~ level
    dark, light = img < 60, img > 180
    assert (noisy - img)[light].std() > 2 * (noisy - img)[dark].std()


def test_mixed_has_both_components():
    img = load_sample("camera")
    noisy = add_noise(img, "mixed", 20, seed=config.SEED)
    impulses = (noisy == 0) | (noisy == 255)
    assert 0.015 < impulses.mean() < 0.04                # ~ MIXED_SP_AMOUNT
    assert abs((noisy - img)[~impulses].std() - 20) < 1.5   # gaussian part


def test_bad_inputs():
    z = np.zeros((16, 16))
    for bad in (lambda: load_image("nope.png"),
                lambda: add_noise(z, "unknown", 1),
                lambda: add_noise(z, "salt_pepper", 1.5),
                lambda: add_noise(z, "gaussian", -1)):
        try:
            bad()
        except (FileNotFoundError, ValueError):
            continue
        raise AssertionError("expected an error")


if __name__ == "__main__":
    for fn in [v for k, v in dict(globals()).items() if k.startswith("test_")]:
        fn(); print("PASS", fn.__name__)