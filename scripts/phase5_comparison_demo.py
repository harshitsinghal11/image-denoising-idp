"""Phase 5 demo: one call gives the full comparison.
    python -m scripts.phase5_comparison_demo
"""
import pandas as pd

import config
from evaluation.comparison import compare
from preprocessing.image_loader import load_sample
from preprocessing.noise import add_noise

pd.set_option("display.width", 120)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

clean = load_sample("camera")
sigma = 20
noisy = add_noise(clean, "gaussian", sigma, seed=config.SEED)

result = compare(clean, noisy, sigma)
print(f"camera, gaussian sigma={sigma}\n")
print("RESULT TABLE (processing_time in seconds)")
print(result.table.to_string(index=False))
print("\nDIFFERENCE TABLE (shearlet minus wavelet)")
print(result.differences.to_string(index=False))