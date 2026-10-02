# config.py — single source of truth for the experiment

SEED = 42

# Images
IMAGE_SIZE = 256              # square; Shearlet needs this (see Phase 4)
INTENSITY_RANGE = 255.0       # work in float 0–255 everywhere
TEST_IMAGES = ["camera", "moon", "coins"]   # skimage.data names; extend later

# Noise Levels
GAUSSIAN_SIGMAS = [10, 20, 30]
SALT_PEPPER_AMOUNTS = [0.02, 0.05, 0.10]   # fraction of pixels corrupted
SPECKLE_STDS = [0.1, 0.2, 0.3]             # std of the multiplicative factor

NOISE_LEVELS = {
    "gaussian": GAUSSIAN_SIGMAS,           # sigma on 0-255 scale
    "salt_pepper": SALT_PEPPER_AMOUNTS,
    "speckle": SPECKLE_STDS,
    "mixed": GAUSSIAN_SIGMAS,              # gaussian sigma; impulses fixed
}


# Wavelet
WAVELET = "db4"
WAVELET_LEVEL = 3
WAVELET_THRESHOLD_MODE = "soft"

# Shearlet
SHEARLET_SCALES = 3
SHEARLET_THRESHOLD_FACTOR = 3.0   # coefficient cutoff = factor * sigma * RMS


# --- Phase 2 additions: all noise types -------------------------------
MIXED_SP_AMOUNT = 0.02                     # impulse fraction inside "mixed"

WAVELET_THRESHOLD_METHOD = "bayes"   # "bayes" (per-subband) or "universal"
WAVELET_THRESHOLD_SCALE = 1.0        # multiplier on the threshold (tuning knob)

SHEARLET_THRESHOLD_MODE = "hard"     # "hard" or "soft" (hard is the pyShearLab standard)


TUNING_GRIDS = {
    "wavelet":  {"threshold_scale": [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]},
    "shearlet": {"factor":          [2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]},
}
TIMING_REPEATS = 3          # processing_time = median of this many runs
# Where the noise level (sigma) given to the denoisers comes from:
#   "true"      = the sigma used to generate the noise
#   "estimated" = robust MAD estimate from the noisy image (blind setting)
SIGMA_SOURCE = {
    "gaussian": "true",
    "mixed": "true",            # level IS the Gaussian sigma; impulses are unmodelled
    "salt_pepper": "estimated", # level is a pixel fraction, not a sigma
    "speckle": "estimated",     # level is a relative std, not a sigma
}
 