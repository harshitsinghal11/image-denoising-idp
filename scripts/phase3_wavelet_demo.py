"""Phase 3 visual + numeric check. Run from the project root:
    python -m scripts.phase3_wavelet_demo
Prints a table and saves data/results/wavelet/phase3_demo_<image>.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import config
from algorithms.wavelet_denoising import denoise_timed
from preprocessing.image_loader import load_sample
from preprocessing.noise import add_noise

OUT = Path("data/results/wavelet")
OUT.mkdir(parents=True, exist_ok=True)
VARIANTS = [("soft", "bayes"), ("soft", "universal"), ("hard", "bayes")]


def m(clean, img):
    img = np.clip(img, 0, 255)
    return psnr(clean, img, data_range=255), ssim(clean, img, data_range=255)


print(f"{'image':8s}{'sigma':>6s}  {'variant':18s}{'PSNR':>7s}{'SSIM':>7s}{'ms':>7s}")
for name in config.TEST_IMAGES:
    clean = load_sample(name)
    fig, axes = plt.subplots(len(config.GAUSSIAN_SIGMAS), 2 + len(VARIANTS),
                             figsize=(14, 3.4 * len(config.GAUSSIAN_SIGMAS)))
    for r, sigma in enumerate(config.GAUSSIAN_SIGMAS):
        noisy = add_noise(clean, "gaussian", sigma, seed=config.SEED)
        p, s = m(clean, noisy)
        print(f"{name:8s}{sigma:6d}  {'noisy':18s}{p:7.2f}{s:7.3f}")
        panels = [("clean", clean), (f"noisy s={sigma}\n{p:.2f} dB", noisy)]
        for mode, method in VARIANTS:
            out, secs = denoise_timed(noisy, sigma, mode=mode, method=method)
            p, s = m(clean, out)
            print(f"{name:8s}{sigma:6d}  {mode + '/' + method:18s}{p:7.2f}{s:7.3f}{secs*1000:7.1f}")
            panels.append((f"{mode}/{method}\n{p:.2f} dB", out))
        for c, (title, img) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(np.clip(img, 0, 255), cmap="gray", vmin=0, vmax=255)
            ax.set_title(title, fontsize=9); ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / f"phase3_demo_{name}.png", dpi=110)
    plt.close(fig)
print("saved figures to", OUT)