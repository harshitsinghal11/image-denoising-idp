# P071 Image Denoising Prototype — Project Phases

> **Project focus:** Image denoising using **Wavelet Transform** and **Shearlet Transform**, followed by a controlled experimental comparison.
>
> **Prototype goal:** Build a working Python-based research prototype that can load an image, introduce controlled noise, denoise it using both transforms, calculate objective metrics, visualize the outputs, and compare performance.
>
> **Important scope decision:** Edge detection and PDE-based processing are **out of the current implementation scope** unless the project supervisor adds them back later.

---

## Phase 0 — Technical Setup ✅

### Environment
- [x] Create project directory
- [x] Create Python virtual environment
- [x] Activate `.venv`
- [x] Verify Python environment

### Core Libraries
- [x] NumPy
- [x] SciPy
- [x] Matplotlib
- [x] Pandas
- [x] Pillow
- [x] OpenCV
- [x] scikit-image
- [x] Streamlit
- [x] PyWavelets

### Shearlet Dependency
- [x] Install `pyShearLab`
- [x] Verify `import pyshearlab`
- [x] Run a real Shearlet transform/reconstruction test
- [x] Confirm compatibility with our image-processing pipeline

**Exit condition:** Both Wavelet and Shearlet libraries can perform a basic transform/reconstruction successfully.

---

# Phase 1 — Finalize Scope & Technical Design

### Define the exact research problem
- [x] Finalize project title
- [x] Finalize research question
- [x] Define what "effective denoising" means for this project
- [x] Define comparison metrics

### Fix the experimental pipeline

```text
Input Image
    ↓
Image Preprocessing
    ↓
Controlled Noise Generation
    ↓
┌───────────────────┬───────────────────┐
│                   │                   │
▼                   ▼
Wavelet          Shearlet
Denoising        Denoising
│                   │
└───────────────────┬───────────────────┘
                    ↓
              Metric Evaluation
                    ↓
              Visual Comparison
                    ↓
              Final Analysis
```

### Decide initial parameters
- [x] Initial image format(s)
- [x] Image size policy
- [x] Grayscale vs RGB handling
- [x] Wavelet family
- [x] Wavelet decomposition level
- [x] Thresholding method
- [x] Shearlet configuration
- [x] Noise types
- [x] Noise levels

**Exit condition:** We can describe exactly what happens to an image from input to final comparison.

---

# Phase 2 — Image Preprocessing

## `preprocessing/image_loader.py`
- [ ] Load image from file
- [ ] Validate supported formats
- [ ] Convert image to required numeric representation
- [ ] Convert RGB to grayscale where required
- [ ] Normalize pixel values
- [ ] Validate dimensions
- [ ] Add reusable image utility functions

## `preprocessing/noise.py`
Implement controlled noise generation:

- [ ] Gaussian noise
- [ ] Salt-and-pepper noise
- [ ] Speckle noise
- [ ] Mixed noise

### Noise controls
- [ ] Noise intensity / variance
- [ ] Reproducible random seed
- [ ] Preserve original clean image
- [ ] Save generated noisy images when needed

**Exit condition:** Given one clean image, the system can reliably generate the same noisy image for repeated experiments.

---

# Phase 3 — Wavelet Denoising Engine

## `algorithms/wavelet_denoising.py`

### Transform
- [ ] Implement 2D Discrete Wavelet Transform
- [ ] Inspect approximation/detail coefficients
- [ ] Support configurable wavelet family
- [ ] Support configurable decomposition level

### Denoising
- [ ] Choose noise-estimation strategy
- [ ] Implement coefficient thresholding
- [ ] Implement soft thresholding
- [ ] Keep hard thresholding available for experimentation
- [ ] Apply thresholding to appropriate detail coefficients

### Reconstruction
- [ ] Implement inverse DWT
- [ ] Restore image dimensions
- [ ] Clip/normalize output correctly
- [ ] Return denoised image

### Initial validation
- [ ] Compare clean image vs noisy image
- [ ] Compare noisy image vs Wavelet result
- [ ] Verify no unexpected artifacts
- [ ] Record runtime

**Exit condition:** Wavelet denoising produces a valid, viewable image and improves at least one evaluation metric under the initial test setup.

---

# Phase 4 — Shearlet Denoising Engine

## `algorithms/shearlet_denoising.py`

### Transform validation
- [ ] Create/load Shearlet system
- [ ] Perform forward Shearlet transform
- [ ] Inspect coefficient structure
- [ ] Perform inverse Shearlet transform
- [ ] Confirm reconstruction dimensions

### Denoising
- [ ] Choose coefficient thresholding strategy
- [ ] Estimate/define threshold
- [ ] Suppress noisy coefficients
- [ ] Reconstruct image

### Compatibility handling
- [ ] Confirm image-size requirements
- [ ] Handle padding/cropping if needed
- [ ] Document any square-image restrictions
- [ ] Document all Shearlet parameters used

### Initial validation
- [ ] Run on the same noisy image used for Wavelet
- [ ] Verify visual output
- [ ] Record runtime
- [ ] Check for reconstruction artifacts

**Exit condition:** Shearlet denoising successfully processes the same controlled test case used by the Wavelet pipeline.

---

# Phase 5 — Unified Evaluation Engine

## `evaluation/metrics.py`

Implement:

- [ ] MSE
- [ ] PSNR
- [ ] SSIM
- [ ] Processing time
- [ ] Optional image-quality statistics

### Metric contract
For every method, return a consistent result:

```text
method
mse
psnr
ssim
processing_time
```

## `evaluation/comparison.py`

- [ ] Compare Wavelet vs Shearlet
- [ ] Produce a structured result table
- [ ] Calculate differences
- [ ] Calculate relative changes where meaningful
- [ ] Avoid declaring a winner before experiments are complete

**Exit condition:** One function can take the clean image + denoised outputs and return a complete comparison.

---

# Phase 6 — Visualization & Result Storage

## `visualization/plots.py`

Create:

- [ ] Original vs noisy vs Wavelet vs Shearlet view
- [ ] PSNR comparison chart
- [ ] SSIM comparison chart
- [ ] MSE comparison chart
- [ ] Runtime comparison chart
- [ ] Optional metric-vs-noise-level plots

## Result storage
- [ ] Save denoised images
- [ ] Save comparison tables
- [ ] Save experiment results as CSV
- [ ] Use consistent experiment naming

Suggested output structure:

```text
data/
├── original/
├── noisy/
└── results/
    ├── wavelet/
    └── shearlet/
```

**Exit condition:** Every completed experiment leaves behind reproducible images and metric data.

---

# Phase 7 — Experiment Runner

## `experiments/experiment_runner.py`

Automate combinations of:

```text
Images
×
Noise Types
×
Noise Levels
×
Wavelet Parameters
×
Shearlet Parameters
```

### Initial experiment matrix
- [ ] Start with one image
- [ ] Start with Gaussian noise
- [ ] Test 2–3 noise levels
- [ ] Run both methods
- [ ] Validate results manually

### Expanded experiment matrix
- [ ] Multiple images
- [ ] Gaussian noise
- [ ] Salt-and-pepper noise
- [ ] Speckle noise
- [ ] Mixed noise
- [ ] Multiple noise levels

### Experiment reproducibility
- [ ] Fixed random seeds
- [ ] Store parameters with each run
- [ ] Store method name
- [ ] Store metrics
- [ ] Store runtime
- [ ] Store image identifiers

**Exit condition:** A single command can execute the benchmark and generate a complete results CSV.

---

# Phase 8 — Research Analysis

- [ ] Calculate average metrics
- [ ] Compare method behavior across noise levels
- [ ] Compare method behavior across noise types
- [ ] Analyze runtime trade-offs
- [ ] Analyze visual quality
- [ ] Identify cases where Wavelet performs well
- [ ] Identify cases where Shearlet performs well
- [ ] Identify failure cases and artifacts
- [ ] Record limitations

### Research questions to answer
- [ ] Which method reduces Gaussian noise more effectively?
- [ ] How do results change as noise intensity increases?
- [ ] How well are important image structures preserved?
- [ ] What is the runtime trade-off?
- [ ] Does performance change for different image types?
- [ ] Are there cases where one method is preferable for a specific criterion?

**Exit condition:** We have evidence-based observations rather than assumptions.

---

# Phase 9 — Streamlit Prototype

## `app/streamlit_app.py`

### Input
- [ ] Image upload
- [ ] Image preview
- [ ] Optional sample images

### Noise controls
- [ ] Noise type selector
- [ ] Noise strength slider
- [ ] Random seed control (optional)

### Denoising controls
- [ ] Wavelet selection
- [ ] Wavelet decomposition level
- [ ] Wavelet threshold settings
- [ ] Shearlet settings
- [ ] Run button

### Results
- [ ] Original image
- [ ] Noisy image
- [ ] Wavelet denoised image
- [ ] Shearlet denoised image
- [ ] Metric table
- [ ] Processing time

### Comparison
- [ ] Side-by-side comparison
- [ ] Charts
- [ ] Download processed images
- [ ] Download result CSV

**Exit condition:** A user can complete the entire workflow without touching Python code.

---

# Phase 10 — Prototype Hardening & Testing

## `tests/`

### Unit tests
- [ ] Test image loading
- [ ] Test grayscale conversion
- [ ] Test each noise generator
- [ ] Test Wavelet output shape
- [ ] Test Shearlet output shape
- [ ] Test metric calculations

### Edge cases
- [ ] Very small image
- [ ] Large image
- [ ] Non-square image
- [ ] Unsupported image
- [ ] Extreme noise level
- [ ] Constant/low-contrast image

### Reliability
- [ ] Handle invalid parameters
- [ ] Handle failed transform
- [ ] Provide readable error messages
- [ ] Prevent corrupted output files

**Exit condition:** Prototype handles expected inputs without crashing.

---

# Phase 11 — Final Presentation / Demonstration Preparation

### Prototype demo flow

```text
1. Upload clean image
        ↓
2. Select noise type
        ↓
3. Select noise level
        ↓
4. Generate noisy image
        ↓
5. Run Wavelet
        ↓
6. Run Shearlet
        ↓
7. Show side-by-side outputs
        ↓
8. Show PSNR / SSIM / MSE / Time
        ↓
9. Show comparison graphs
        ↓
10. Explain observations
```

### Prepare
- [ ] Final architecture diagram
- [ ] Final workflow diagram
- [ ] Algorithm explanation
- [ ] Parameter explanation
- [ ] Experimental methodology
- [ ] Results tables
- [ ] Graphs
- [ ] Limitations
- [ ] Future work
- [ ] Final demo script

**Exit condition:** A complete reproducible demo can be given from start to finish.

---

# Phase 12 — Documentation & Final Deliverables

- [ ] `README.md`
- [ ] `requirements.txt`
- [ ] Algorithm documentation
- [ ] Experiment methodology
- [ ] Dataset / image-source documentation
- [ ] Parameter documentation
- [ ] Final results
- [ ] Final comparison
- [ ] Limitations
- [ ] Future work
- [ ] Final project report
- [ ] Final presentation

---

# Final Prototype Definition

The project is considered complete only when all of the following work together:

```text
                ┌──────────────────────┐
                │      Streamlit       │
                │    Prototype UI      │
                └──────────┬───────────┘
                           │
                           ▼
                 ┌──────────────────┐
                 │ Image Preprocess  │
                 └────────┬─────────┘
                          │
                          ▼
                  ┌───────────────┐
                  │ Noise Engine  │
                  └───────┬───────┘
                          │
                 ┌────────┴────────┐
                 ▼                 ▼
          ┌────────────┐    ┌────────────┐
          │  Wavelet   │    │  Shearlet  │
          │  Denoising │    │  Denoising │
          └─────┬──────┘    └──────┬─────┘
                │                  │
                └────────┬─────────┘
                         ▼
                ┌─────────────────┐
                │ Metrics Engine  │
                │ MSE / PSNR/SSIM │
                │ Runtime         │
                └────────┬────────┘
                         ▼
                ┌─────────────────┐
                │   Comparison    │
                │   + Charts      │
                └─────────────────┘
```

## Definition of Done

- [ ] Both algorithms work on the same noisy input
- [ ] Both algorithms reconstruct valid images
- [ ] Metrics are calculated consistently
- [ ] Runtime is measured consistently
- [ ] Multiple noise conditions can be tested
- [ ] Experiments are reproducible
- [ ] Results are stored
- [ ] Prototype provides visual comparison
- [ ] Streamlit demo works end-to-end
- [ ] Final observations are supported by experimental data
- [ ] No unsupported claim that one method is universally better

---

## Current Status

### ✅ Completed
- Phase 0 core environment
- Core libraries verified
- `PyWavelets 1.8.0` verified
- `scikit-image 0.26.0` verified
- `pyShearLab 0.0.1` installed
- `pyshearlab` import verified
- Project directory structure created

### 🔵 Current Phase
**Phase 1 — Finalize Scope & Technical Design**

### ▶️ Next Immediate Tasks
1. Run a real pyShearLab transform/reconstruction test.
2. Create `preprocessing/image_loader.py`.
3. Create `preprocessing/noise.py`.
4. Build the first Wavelet denoising prototype.
5. Build the first Shearlet denoising prototype.
