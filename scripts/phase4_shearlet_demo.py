"""Phase 4 check: Shearlet on the SAME noisy images the Wavelet demo used.
    python -m scripts.phase4_shearlet_demo
Prints a table and saves data/results/shearlet/phase4_demo_<image>.png
This is a sanity check with default settings, NOT the final comparison.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import config
from algorithms import shearlet_denoising as sh
from algorithms import wavelet_denoising as wv
from preprocessing.image_loader import load_sample
from preprocessing.noise import add_noise

OUT = Path("data/results/shearlet")
OUT.mkdir(parents=True, exist_ok=True)


def m(clean, img):
    img = np.clip(img, 0, 255)
    return psnr(clean, img, data_range=255), ssim(clean, img, data_range=255)


print(f"shearlet system setup ({config.IMAGE_SIZE}px, {config.SHEARLET_SCALES} scales): "
      f"{sh.setup_seconds(config.IMAGE_SIZE, config.SHEARLET_SCALES):.2f} s (one-off)\n")
print(f"{'image':8s}{'sigma':>6s}  {'method':10s}{'PSNR':>7s}{'SSIM':>7s}{'ms':>8s}")
for name in config.TEST_IMAGES:
    clean = load_sample(name)
    fig, axes = plt.subplots(len(config.GAUSSIAN_SIGMAS), 4,
                             figsize=(12, 3.4 * len(config.GAUSSIAN_SIGMAS)))
    for r, sigma in enumerate(config.GAUSSIAN_SIGMAS):
        noisy = add_noise(clean, "gaussian", sigma, seed=config.SEED)
        panels = [("clean", clean)]
        p, s = m(clean, noisy)
        print(f"{name:8s}{sigma:6d}  {'noisy':10s}{p:7.2f}{s:7.3f}")
        panels.append((f"noisy s={sigma}\n{p:.2f} dB", noisy))
        for label, mod in (("wavelet", wv), ("shearlet", sh)):
            out, secs = mod.denoise_timed(noisy, sigma)
            p, s = m(clean, out)
            print(f"{name:8s}{sigma:6d}  {label:10s}{p:7.2f}{s:7.3f}{secs*1000:8.1f}")
            panels.append((f"{label}\n{p:.2f} dB / SSIM {s:.3f}", out))
        for c, (title, img) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(np.clip(img, 0, 255), cmap="gray", vmin=0, vmax=255)
            ax.set_title(title, fontsize=9); ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / f"phase4_demo_{name}.png", dpi=110)
    plt.close(fig)
print("\nsaved figures to", OUT)