"""Phase 0 exit check: real transform/reconstruction for BOTH libraries.
Run from project root:  python phase0_shearlet_check.py
"""
import time
import numpy as np
import pywt
import pyshearlab
from skimage import data
from skimage.metrics import peak_signal_noise_ratio as psnr

SIGMA = 25.0                       # noise std on a 0-255 scale
rng = np.random.default_rng(42)    # fixed seed -> reproducible
X = data.camera()[::2, ::2].astype(float)          # 256x256, square
Xn = X + SIGMA * rng.standard_normal(X.shape)

# ---- 1. Perfect-reconstruction checks (no thresholding) -------------
t = time.perf_counter()
S = pyshearlab.SLgetShearletSystem2D(0, X.shape[0], X.shape[1], 3)
print(f"shearlet system: {time.perf_counter()-t:.2f}s, "
      f"{S['nShearlets']} shearlets")
C = pyshearlab.SLsheardec2D(X, S)
R = pyshearlab.SLshearrec2D(C, S)
print("shearlet coeffs:", C.shape, "| max recon error:", np.abs(R - X).max())

cw = pywt.wavedec2(X, "db4", level=3)
Rw = pywt.waverec2(cw, "db4")[: X.shape[0], : X.shape[1]]
print("wavelet  max recon error:", np.abs(Rw - X).max())

# ---- 2. Minimal denoising sanity check -------------------------------
# Shearlet: hard threshold, scale-aware via S['RMS'] (from pyShearLab example)
Cn = np.real(pyshearlab.SLsheardec2D(Xn, S))
w = np.array(S["RMS"])[None, None, :]
Cn[np.abs(Cn) < 3 * SIGMA * w] = 0
Xs = pyshearlab.SLshearrec2D(Cn, S)

# Wavelet: soft threshold, universal threshold, sigma from finest diagonal band
cw = pywt.wavedec2(Xn, "db4", level=3)
sig = np.median(np.abs(cw[-1][2])) / 0.6745
thr = sig * np.sqrt(2 * np.log(Xn.size))
cw = [cw[0]] + [tuple(pywt.threshold(d, thr, "soft") for d in lvl) for lvl in cw[1:]]
Xw = pywt.waverec2(cw, "db4")[: X.shape[0], : X.shape[1]]

for name, img in [("noisy", Xn), ("wavelet", Xw), ("shearlet", Xs)]:
    print(f"{name:9s} PSNR = {psnr(X, np.clip(img, 0, 255), data_range=255):.2f} dB")