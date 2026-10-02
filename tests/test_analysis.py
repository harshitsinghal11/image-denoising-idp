"""Phase 8 checks. Run from the project root:  python -m tests.test_analysis"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.analyze_results import (build_report, direction, failure_cases,
                                      load_benchmark, make_gallery, mean_ci,
                                      paired_table, run_analysis, runtime_table,
                                      select_cases, summary_table, to_wide, verdict)
from experiments.experiment_runner import run_matrix


def _synthetic(shear_psnr_extra=1.0, shear_ssim_extra=0.05, seeds=(1, 2, 3, 4)):
    """Small hand-made benchmark with known differences."""
    rows = []
    rng = np.random.default_rng(0)
    for tuning in ("default", "tuned"):
        for image in ("a", "b"):
            for seed in seeds:
                for level in (10, 20):
                    exp = f"{image}_gaussian_{level}_seed{seed}"
                    jitter = rng.normal(0, 0.2)
                    base = 30 - level / 2 + jitter
                    for method, extra_p, extra_s, t in (
                            ("noisy", -6.0, -0.4, np.nan), ("wavelet", 0.0, 0.0, 0.003),
                            ("shearlet", shear_psnr_extra, shear_ssim_extra, 0.2)):
                        rows.append(dict(
                            experiment_id=exp, tuning=tuning, image=image,
                            noise_type="gaussian", noise_level=level, seed=seed,
                            method=method, mse=100.0, psnr=base + extra_p,
                            ssim=0.7 + extra_s, processing_time=t, sigma_used=level,
                            sigma_source="true", n_candidates=1, tuned_value=np.nan,
                            params="{}"))
    return pd.DataFrame(rows)


def test_mean_ci_known_values():
    m, lo, hi, n = mean_ci([1, 2, 3])
    assert (m, n) == (2.0, 3)
    assert abs((hi - lo) / 2 - 2.4843) < 1e-3                # t(0.975, 2) / sqrt(3)
    assert np.isnan(mean_ci([5.0])[1])                       # n=1 -> no CI
    assert mean_ci([2, 2, 2])[1:3] == (2.0, 2.0)             # zero variance


def test_direction_and_verdict():
    assert direction(0.1, 0.5) == 1 and direction(-0.5, -0.1) == -1
    assert direction(-0.1, 0.5) == 0 and direction(np.nan, np.nan) == 0
    assert verdict(1, 1, 5) == "shearlet higher on PSNR and SSIM"
    assert verdict(-1, -1, 5) == "wavelet higher on PSNR and SSIM"
    assert verdict(1, -1, 5) == "metrics disagree"
    assert verdict(1, 0, 5) == "shearlet higher on PSNR only"
    assert verdict(0, -1, 5) == "wavelet higher on SSIM only"
    assert verdict(0, 0, 5) == "no clear difference"
    assert verdict(1, 1, 2) == "insufficient data (n<3)"


def test_wide_and_paired_known_differences():
    w = to_wide(_synthetic())
    assert len(w) == 2 * 2 * 4 * 2                           # tuning x image x seed x level
    assert np.allclose(w["psnr_diff"], 1.0) and np.allclose(w["ssim_diff"], 0.05)
    assert np.allclose(w["psnr_gain_wavelet"], 6.0)          # wavelet - noisy
    assert np.allclose(w["time_ratio"], 0.2 / 0.003)
    p = paired_table(w)
    assert len(p) == 4                                       # 2 tunings x 2 levels
    assert (p["n"] == 8).all()
    assert np.allclose(p["psnr_diff_mean"], 1.0)
    assert (p["psnr_shearlet_higher_runs"] == 8).all()
    assert (p["verdict"] == "shearlet higher on PSNR and SSIM").all()


def test_verdict_flips_when_wavelet_is_higher_and_on_disagreement():
    p = paired_table(to_wide(_synthetic(-1.0, -0.05)))
    assert (p["verdict"] == "wavelet higher on PSNR and SSIM").all()
    p = paired_table(to_wide(_synthetic(1.0, -0.05)))
    assert (p["verdict"] == "metrics disagree").all()
    assert (p["metrics_disagree_runs"] == 8).all()


def test_summary_failures_runtime():
    df = _synthetic()
    w = to_wide(df)
    s = summary_table(w)
    assert set(s["method"]) == {"noisy", "wavelet", "shearlet"}
    assert s.loc[s.method == "noisy", "psnr_gain"].isna().all()
    assert failure_cases(w).empty                            # shearlet == wavelet + 1, both > noisy
    bad = df.copy()
    idx = bad[(bad.method == "wavelet") & (bad.experiment_id == "a_gaussian_10_seed1")
              & (bad.tuning == "tuned")].index[0]
    bad.loc[idx, "psnr"] = 0.0                               # worse than noisy
    f = failure_cases(to_wide(bad))
    assert (f["kind"] == "wavelet below noisy input on PSNR").sum() == 1
    r = runtime_table(df).set_index("method")
    assert abs(r.loc["shearlet", "median_ms"] - 200.0) < 1e-9
    assert abs(r.loc["shearlet", "median_vs_wavelet"] - 200 / 3) < 1e-6


def test_load_benchmark_errors():
    with tempfile.TemporaryDirectory() as d:
        for bad in (lambda: load_benchmark(Path(d) / "nope.csv"),):
            try:
                bad()
            except FileNotFoundError:
                pass
            else:
                raise AssertionError("expected FileNotFoundError")
        pd.DataFrame({"x": [1]}).to_csv(Path(d) / "bad.csv", index=False)
        try:
            load_benchmark(Path(d) / "bad.csv")
        except ValueError:
            return
        raise AssertionError("expected ValueError")


def test_report_warns_and_gallery_skips_missing_images():
    df = _synthetic(seeds=(1,))                              # a single seed
    w = to_wide(df)
    rep = build_report(w, summary_table(w), paired_table(w),
                       paired_table(w, by=["tuning", "noise_type"]),
                       paired_table(w, by=["tuning", "noise_type", "image"]),
                       failure_cases(w), runtime_table(df), df, [], "x.csv")
    assert "only 1 seed" in rep and "Limitations" in rep
    assert "best" not in rep.lower().replace("best of", "").replace("best case", "") \
        .replace("best setting", "")                         # no "best method" claims
    fig, cases = make_gallery(w, "/nonexistent", "tuned")
    assert fig is None and cases == []
    # synthetic differences are identical in every run, so all "notable" picks
    # collapse onto one experiment; real data gives several distinct cases
    assert len(select_cases(w[w.tuning == "tuned"].reset_index(drop=True))) >= 1


def test_end_to_end_with_real_benchmark():
    with tempfile.TemporaryDirectory() as d:
        run_matrix(["camera"], ["gaussian", "speckle"],
                   levels={"gaussian": [10, 20], "speckle": [0.1, 0.2]},
                   seeds=[1, 2], root=d, timing_repeats=1, progress=lambda *_: None)
        paths = run_analysis(Path(d) / "results" / "benchmark_results.csv", d,
                             progress=lambda *_: None)
        out = Path(d) / "results" / "analysis"
        for rel in ("analysis_report.md", "tables/paired_differences.csv",
                    "tables/summary_by_condition.csv", "tables/failure_cases.csv",
                    "tables/runtime.csv", "figures/difference_tuned.png",
                    "figures/psnr_vs_level_tuned.png", "figures/ssim_vs_level_default.png",
                    "figures/tradeoff_tuned.png", "figures/by_image_tuned.png",
                    "figures/gallery.png"):
            assert (out / rel).is_file() and (out / rel).stat().st_size > 0, rel
        text = (out / "analysis_report.md").read_text(encoding="utf-8")
        assert "## 2. Which method reduces Gaussian noise" in text
        assert "only 1 image" in text                        # honest scope warning


if __name__ == "__main__":
    for fn in [v for k, v in dict(globals()).items() if k.startswith("test_")]:
        fn(); print("PASS", fn.__name__)