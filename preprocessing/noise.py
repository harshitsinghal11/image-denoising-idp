import numpy as np
import config

NOISE_TYPES = ("gaussian", "salt_pepper", "speckle", "mixed")
_ALIASES = {"salt_and_pepper": "salt_pepper", "s&p": "salt_pepper"}


def _rng(seed):
    return np.random.default_rng(config.SEED if seed is None else seed)


def _finish(noisy, clip):
    return np.clip(noisy, 0, config.INTENSITY_RANGE) if clip else noisy


def add_gaussian_noise(img, sigma, seed=None, clip=False):
    """Additive white Gaussian noise, std `sigma` (0-255 units).

    clip=False keeps the noise exactly Gaussian, so the sigma the
    denoisers are given is the true one.
    """
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    return _finish(img + _rng(seed).normal(0.0, sigma, img.shape), clip)


def _impulses(img, amount, rng):
    """Replace a random `amount` fraction of pixels with 0 or 255."""
    if not 0.0 <= amount <= 1.0:
        raise ValueError("amount must be between 0 and 1")
    out = np.array(img, dtype=np.float64, copy=True)
    corrupt = rng.random(img.shape) < amount          # which pixels
    salt = rng.random(img.shape) < 0.5                # salt or pepper
    out[corrupt & salt] = config.INTENSITY_RANGE
    out[corrupt & ~salt] = 0.0
    return out


def add_salt_pepper_noise(img, amount, seed=None):
    """Impulse noise: `amount` fraction of pixels set to 0 or 255."""
    return _impulses(img, amount, _rng(seed))


def add_speckle_noise(img, std, seed=None, clip=False):
    """Multiplicative noise: img + img * N(0, std^2)."""
    if std < 0:
        raise ValueError("std must be non-negative")
    n = _rng(seed).normal(0.0, std, img.shape)
    return _finish(img + img * n, clip)


def add_mixed_noise(img, sigma, seed=None, sp_amount=None):
    """Gaussian(sigma) followed by salt-and-pepper impulses."""
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    sp_amount = config.MIXED_SP_AMOUNT if sp_amount is None else sp_amount
    rng = _rng(seed)
    noisy = img + rng.normal(0.0, sigma, img.shape)
    return _impulses(noisy, sp_amount, rng)


def add_noise(img, noise_type, level, seed=None, **kwargs):
    """Single entry point used by the experiment runner and the app."""
    name = _ALIASES.get(noise_type, noise_type)
    if name == "gaussian":
        return add_gaussian_noise(img, level, seed, **kwargs)
    if name == "salt_pepper":
        return add_salt_pepper_noise(img, level, seed, **kwargs)
    if name == "speckle":
        return add_speckle_noise(img, level, seed, **kwargs)
    if name == "mixed":
        return add_mixed_noise(img, level, seed, **kwargs)
    raise ValueError(
        f"Unknown noise type '{noise_type}'. Choose from {NOISE_TYPES}")