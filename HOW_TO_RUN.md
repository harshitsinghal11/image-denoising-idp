# How to run: Image Denoising Lab (Wavelet vs Shearlet)

All commands are for **PowerShell**. `data\` only holds generated output, so it is safe to delete and
regenerate at any time. The Streamlit app does not need it.

## 0. Open the project (do this first, every time)

```powershell
cd C:\Harshit\mruLABS\image-denoising
.venv\Scripts\Activate.ps1
```

You should see `(.venv)` at the start of the prompt.
If PowerShell refuses to run the script: `Set-ExecutionPolicy -Scope Process Bypass`, then try again.

## 1. The main demo: the Streamlit app

```powershell
streamlit run app\streamlit_app.py
```

Opens `http://localhost:8501`. Stop it with Ctrl+C.
Suggested demo: pick `camera`, Gaussian noise, level 20, press **Run comparison**; then try salt & pepper,
tick **Auto-tune**, and open the **Inspect a region** tab. Upload your own photo under "Upload my own".

## 2. Quick terminal demo (about 1-2 minutes)

One image, Gaussian noise at 3 levels, 3 random seeds, both methods, then the written analysis.

```powershell
python -m experiments.experiment_runner --preset initial --seeds 42 43 44 --root data_demo
python -m analysis.analyze_results --csv data_demo\results\benchmark_results.csv --root data_demo
explorer data_demo\results\analysis
```

Open `analysis_report.md` (the report) and the `figures` folder (the charts and `gallery.png`).

## 3. Full benchmark and report (about 15 minutes)

3 images x 4 noise types x 3 levels x 5 seeds = 180 experiments, each with default and tuned settings.

```powershell
python -m experiments.experiment_runner --preset full --seeds 42 43 44 45 46
python -m analysis.analyze_results
explorer data\results\analysis
```

- Do **not** open `data\results\benchmark_results.csv` in Excel while this runs (Windows locks the file).
- If it gets interrupted, continue where it stopped by adding `--resume`:

```powershell
python -m experiments.experiment_runner --preset full --seeds 42 43 44 45 46 --resume
```

Results appear in:

| What | Where |
|---|---|
| Written report (answers the research questions) | `data\results\analysis\analysis_report.md` |
| Charts, difference plots, visual gallery | `data\results\analysis\figures\` |
| Tables as CSV | `data\results\analysis\tables\` |
| Every raw result (one row per run and method) | `data\results\benchmark_results.csv` |
| Denoised images | `data\results\wavelet\`, `data\results\shearlet\` |

## 4. Run on your own images

```powershell
python -m experiments.experiment_runner --preset full --images "C:\path\to\photo1.jpg" "C:\path\to\photo2.png" --seeds 42 43 44 --root data_mine
python -m analysis.analyze_results --csv data_mine\results\benchmark_results.csv --root data_mine
```

Photos are converted to grayscale, centre-cropped to a square and resized to 256 px.
The short side must be at least 128 px. Two files with the same name overwrite each other.

## 5. Check that the code works (72 tests)

```powershell
foreach ($t in "test_preprocessing","test_wavelet","test_shearlet","test_evaluation","test_visualization","test_runner","test_analysis","test_app") { python -m tests.$t }
```

Expected `PASS` counts: preprocessing 8, wavelet 8, shearlet 9, evaluation 10, visualization 9,
runner 11, analysis 8, app 9 (72 in total). Ignore the harmless Streamlit `ScriptRunContext` warning and the
shearlet "filters were automatically set" warning.

## 6. If something goes wrong

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'config'` | You are not in the project folder. Run the `cd` line from section 0 |
| `PermissionError ... benchmark_results.csv` | The CSV is open in Excel or a VS Code preview. Close it, then rerun with `--resume` |
| `AttributeError: ... SLgetShearletSystem2D` or `inhomogeneous shape` | pyShearLab is missing or unpatched: `pip install git+https://github.com/stefanloock/pyshearlab.git` then `python scripts\patch_shearlab.py` |
| `FileNotFoundError ... benchmark_results.csv` when running the analysis | `data\` was deleted. Run the benchmark (section 2 or 3) first |
| Streamlit shows an old version of the page | Stop it (Ctrl+C) and start it again |

## 7. Clean up (delete all generated files)

```powershell
Remove-Item -Recurse -Force data, data_demo, data_mine -ErrorAction SilentlyContinue
Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
```

## 8. Keep a copy of the final results (small zip, run before cleaning up)

```powershell
Compress-Archive -Path data\results\benchmark_results.csv, data\results\analysis -DestinationPath final_results.zip -Force
```

## 9. One-paragraph summary for an examiner

The project adds controlled noise (Gaussian, salt-and-pepper, speckle, mixed) to test images, removes it
with a wavelet-threshold method and a shearlet-threshold method, and scores both against the clean original
using PSNR, SSIM, MSE and runtime. A benchmark runs every combination with fixed random seeds for
reproducibility, and an analysis script reports paired differences with confidence intervals. A Streamlit app
lets anyone upload an image, choose the noise and settings, and compare the two methods interactively.
