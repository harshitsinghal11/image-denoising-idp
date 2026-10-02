import sys
from pathlib import Path

# make the project root importable when Streamlit runs this file directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

import config
from app import core
from evaluation.storage import experiment_id
from preprocessing.image_loader import load_sample
from visualization.plots import plot_all_metrics

st.set_page_config(page_title="Image Denoising", layout="wide")

FAMILIES = ["haar", "db4", "sym4"]


def _index(options, value, fallback=0):
    return list(options).index(value) if value in options else fallback


# =========================== sidebar: inputs =============================
with st.sidebar:
    st.header("1 · Image")
    size = st.selectbox(
        "Working size (pixels)", core.SIZES,
        index=_index(core.SIZES, config.IMAGE_SIZE, 1), key="size",
        help="Images are cropped to a square and resized to this. "
             "Shearlet gets slower with size (512 px: ~10 s one-off setup, ~1 s per run).")
    source = st.radio("Source", ["Sample image", "Upload my own"], key="source")

    clean, image_name, upload_info = None, None, None
    if source == "Sample image":
        image_name = st.selectbox("Sample", list(config.TEST_IMAGES), key="sample")
        clean = load_sample(image_name, size)
    else:
        upload = st.file_uploader("Image file", key="upload",
                                  type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
        if upload is not None:
            try:
                clean, original = core.load_uploaded(upload.getvalue(), size)
                image_name = Path(upload.name).stem
                upload_info = (f"Uploaded {original[0]}×{original[1]} px → grayscale, "
                               f"centre-cropped to a square, resized to {size}×{size}.")
            except ValueError as exc:
                st.error(str(exc))
    if clean is not None:
        st.image(core.display_image(clean, 1), caption=f"{image_name} ({size}×{size}, grayscale)")
        if upload_info:
            st.caption(upload_info)

    st.header("2 · Noise")
    noise_type = st.selectbox("Noise type", list(core.NOISE_UI), key="noise_type",
                              format_func=lambda k: core.NOISE_UI[k][0])
    _, desc, lo, hi, default, step = core.NOISE_UI[noise_type]
    level = st.slider(f"Level: {desc}", lo, hi, default, step, key=f"level_{noise_type}")
    st.caption("Levels used in the benchmark: "
               + ", ".join(f"{v:g}" for v in config.NOISE_LEVELS[noise_type]))
    seed = st.number_input("Random seed", min_value=0, max_value=2**31 - 1,
                           value=int(config.SEED), step=1, key="seed",
                           help="Same seed + same settings = exactly the same noisy image.")

    st.header("3 · Denoising")
    tune = st.checkbox(
        "Auto-tune the threshold of each method", key="tune",
        help="Tries 8 threshold settings per method and keeps the one with the highest PSNR "
             "against the clean image (as the 'tuned' benchmark does). This is a best case; "
             "it needs the clean image. The threshold sliders below are ignored when on.")
    with st.expander("Wavelet", expanded=True):
        w_family = st.selectbox("Wavelet family", FAMILIES, key="w_family",
                                index=_index(FAMILIES, config.WAVELET, 2))
        w_level = st.slider("Decomposition level", 1, 5, int(config.WAVELET_LEVEL), key="w_level",
                            help="Capped automatically if the image is too small for it.")
        w_mode = st.radio("Thresholding", ["soft", "hard"], horizontal=True, key="w_mode",
                          index=_index(["soft", "hard"], config.WAVELET_THRESHOLD_MODE))
        w_method = st.radio("Threshold rule", ["bayes", "universal"], horizontal=True, key="w_method",
                            index=_index(["bayes", "universal"], config.WAVELET_THRESHOLD_METHOD))
        w_scale = st.slider("Threshold scale", 0.25, 6.0, float(config.WAVELET_THRESHOLD_SCALE),
                            0.05, key="w_scale", disabled=tune,
                            help="Multiplies the threshold. Larger = more smoothing.")
    with st.expander("Shearlet", expanded=True):
        s_scales = config.SHEARLET_SCALES
        st.caption(f"{s_scales} scales (fixed; needs an image of at least 128 px)")
        s_mode = st.radio("Thresholding", ["hard", "soft"], horizontal=True, key="s_mode",
                          index=_index(["hard", "soft"], config.SHEARLET_THRESHOLD_MODE))
        s_factor = st.slider("Threshold factor", 1.0, 10.0, float(config.SHEARLET_THRESHOLD_FACTOR),
                             0.1, key="s_factor", disabled=tune,
                             help="Coefficients below factor × σ × weight are removed.")
    sigma_mode = "config"      # true sigma where one exists, an estimate otherwise (as in the benchmark)
    run_clicked = st.button("Run comparison", type="primary", key="run_button",
                            disabled=clean is None)

# ============================ run on click ===============================
if run_clicked and clean is not None:
    params = {
        "wavelet": {"wavelet": w_family, "level": int(w_level), "mode": w_mode,
                    "method": w_method, "threshold_scale": float(w_scale)},
        "shearlet": {"scales": int(s_scales), "factor": float(s_factor), "mode": s_mode},
    }
    try:
        with st.spinner("Running both methods. The first shearlet run at a new size builds "
                        "its transform, which can take a few seconds."):
            run = core.run_experiment(clean, noise_type, float(level), int(seed),
                                      sigma_mode, params, tune)
        st.session_state["run"] = run
        st.session_state["meta"] = {"name": image_name, "size": size}
    except ValueError as exc:                      # readable message, no traceback
        st.error(f"Could not run the comparison: {exc}")

# ============================== main area ================================
st.title("Image Denoising using Wavlet & Shearlet")
st.caption("Add controlled noise to an image, clean it with a Wavelet and a Shearlet method, "
           "and compare both against the original.")

run = st.session_state.get("run")
if run is None:
    st.info("Pick an image and noise in the sidebar, then press **Run comparison**.")
    st.stop()

meta = st.session_state["meta"]
result, table = run["result"], run["result"].table
exp = experiment_id(meta["name"], run["noise_type"], run["level"], run["seed"])
st.subheader(f"{meta['name']} · {core.NOISE_UI[run['noise_type']][0]} {run['level']:g} · seed {run['seed']}")
st.caption(f"Working size {meta['size']}×{meta['size']} · noise level given to both methods: "
           f"{run['sigma']:.2f} ({run['sigma_source']}) · "
           f"{'auto-tuned thresholds' if run['tune'] else 'thresholds as set in the sidebar'}")
if run["tuned"]:
    st.caption("Auto-tuned: " + "; ".join(
        f"{m} {v['knob']} = {v['value']:g} (best of {v['candidates']})"
        for m, v in run["tuned"].items()))
    for m, v in run["tuned"].items():
        if v["at_edge"]:
            st.warning(f"The best {m} setting was at the edge of the search range, so a "
                       "better value may exist outside it.")

tab_img, tab_metrics, tab_charts, tab_inspect, tab_download = st.tabs(
    ["Images", "Metrics", "Charts", "Inspect a region", "Download"])

with tab_img:
    cols = st.columns(4)
    t = table.set_index("method")
    for col, (key, label, row) in zip(cols, (("clean", "Original", None), ("noisy", "Noisy", "noisy"),
                                             ("wavelet", "Wavelet", "wavelet"),
                                             ("shearlet", "Shearlet", "shearlet"))):
        caption = label if row is None else (
            f"{label}: {t.loc[row, 'psnr']:.2f} dB · SSIM {t.loc[row, 'ssim']:.3f}")
        col.image(core.display_image(result.images[key]), caption=caption)
    for line in core.takeaways(run):
        st.markdown(f"- {line}")
    for m in ("wavelet", "shearlet"):
        if t.loc[m, "psnr"] <= t.loc["noisy", "psnr"]:
            st.warning(f"{m.capitalize()} scored no higher than the noisy image on PSNR in this run.")
    st.caption("One run is an observation, not evidence that one method is better in general. "
               "The benchmark analysis averages over many images, noise levels and seeds.")

with tab_metrics:
    st.markdown("**Scores against the original (clean) image**")
    st.dataframe(core.rounded_table(table), hide_index=True)
    st.markdown("**Shearlet minus wavelet**")
    st.dataframe(core.rounded_table(result.differences), hide_index=True)
    st.caption("For PSNR and SSIM higher is better; for MSE and time lower is better. "
               "`noisy` is the image before any denoising, shown for reference. "
               "Relative change is left empty for PSNR because PSNR is already a log scale. "
               "Processing time excludes shearlet's one-off setup.")
    with st.expander("How to read these numbers"):
        st.markdown(
            "- **PSNR (dB)**: closeness to the original. Higher is better. "
            "About 20 is poor, about 30 decent, 40+ excellent.\n"
            "- **SSIM** (0 to 1): similarity of structure such as edges and texture. 1 means identical.\n"
            "- **MSE**: average squared pixel error. Lower is better.\n"
            "- **Time**: seconds for one denoising run.")

with tab_charts:
    st.pyplot(plot_all_metrics(table, title=exp))

with tab_inspect:
    st.caption("Zoom into the same region of all four images to look for artifacts "
               "such as streaks, blur or leftover noise.")
    c1, c2, c3 = st.columns(3)
    cx = c1.slider("Horizontal position (%)", 0, 100, 50, key="zoom_x") / 100
    cy = c2.slider("Vertical position (%)", 0, 100, 50, key="zoom_y") / 100
    win = c3.slider("Window (pixels)", 16, 128, 64, 8, key="zoom_w")
    zcols = st.columns(4)
    for col, (key, label) in zip(zcols, (("clean", "Original"), ("noisy", "Noisy"),
                                         ("wavelet", "Wavelet"), ("shearlet", "Shearlet"))):
        patch = core.crop_region(result.images[key], cx, cy, win)
        col.image(core.display_image(patch, max(1, 256 // patch.shape[0])), caption=label)

with tab_download:
    df = core.results_dataframe(run, meta["name"])
    st.download_button("Download results (CSV)", df.to_csv(index=False), f"{exp}_results.csv",
                       "text/csv", key="dl_csv")
    st.download_button("Download everything (ZIP: images + CSV + parameters)",
                       core.build_zip(run, meta["name"]), f"{exp}.zip",
                       "application/zip", key="dl_zip")
    d1, d2, d3, d4 = st.columns(4)
    for col, (key, label) in zip((d1, d2, d3, d4), (("clean", "original"), ("noisy", "noisy"),
                                                     ("wavelet", "wavelet"), ("shearlet", "shearlet"))):
        col.download_button(f"{label.capitalize()} (PNG)", core.png_bytes(result.images[key]),
                            f"{exp}_{label}.png", "image/png", key=f"dl_{label}")
    st.caption("Saved images are 8-bit PNGs (clipped and rounded). "
               "All scores are computed before saving, from the full-precision images.")