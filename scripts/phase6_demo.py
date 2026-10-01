"""Phase 6 demo: run experiments, store everything, draw the figures.
    python -m scripts.phase6_demo
Writes images/CSVs under data/ and figures under data/results/figures/.
"""
from pathlib import Path

import config
from evaluation.comparison import compare
from evaluation.storage import load_results, save_experiment
from preprocessing.image_loader import load_sample
from preprocessing.noise import add_noise
from visualization.plots import (plot_all_metrics, plot_image_comparison,
                                 plot_metric_vs_noise)

ROOT = Path("data")
FIG = ROOT / "results" / "figures"
NOISE = "gaussian"

for name in config.TEST_IMAGES:
    clean = load_sample(name)
    for sigma in config.GAUSSIAN_SIGMAS:
        noisy = add_noise(clean, NOISE, sigma, seed=config.SEED)
        result = compare(clean, noisy, sigma)
        paths = save_experiment(result, name, NOISE, sigma, config.SEED,
                                sigma=sigma, root=ROOT)
        exp = paths["experiment_id"]
        plot_image_comparison(result.images, result.table, exp, FIG / f"{exp}_images.png")
        plot_all_metrics(result.table, exp, FIG / f"{exp}_metrics.png")
        print("saved", exp)

master = load_results(ROOT)
for metric in ("psnr", "ssim", "mse", "processing_time"):
    plot_metric_vs_noise(master, metric, NOISE, save_path=FIG / f"vs_noise_{metric}.png")
print(f"\n{len(master)} rows in {ROOT / 'results' / 'results.csv'}")
print("figures in", FIG)