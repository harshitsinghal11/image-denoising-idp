# P071 — An Efficient Comparative Approach for Image Denoising Using Wavelet and Shearlet Transforms

## 1. Project Title

**An Efficient Comparative Approach for Image Denoising Using Wavelet and Shearlet Transforms**

### Short Name

**Wavelet vs Shearlet Image Denoising**

---

## 2. Project Description

Digital images are often affected by noise during image acquisition, transmission, storage, or processing. Noise reduces visual quality and can hide important image structures and details.

The original problem statement proposes an efficient hybrid approach for image denoising and edge detection using grayscale conversion, wavelet-based denoising, and second-order PDE-based edge detection.

For the current project implementation, the scope has been refined based on the project guidance:

> **Edge detection and PDE-based processing are removed from the prototype. The project will focus entirely on image denoising using Wavelet Transform and Shearlet Transform, followed by a systematic comparison of their performance.**

The prototype will take an image, introduce controlled noise, process the noisy image using both Wavelet and Shearlet based denoising methods, and compare the resulting images using objective image-quality metrics and visual analysis.

---

# 3. Problem We Are Solving

When an image contains noise, simply applying conventional smoothing techniques can remove noise but may also blur important image structures.

The fundamental problem is therefore:

> **How can we reduce image noise while preserving important image details and structures?**

We are investigating two transform-based approaches:

1. **Wavelet Transform**
2. **Shearlet Transform**

The project does not assume beforehand that one method is universally better.

Instead, we will experimentally evaluate both methods under controlled conditions.

---

# 4. Why Wavelet and Shearlet?

## Wavelet Transform

Wavelets provide a multi-scale representation of an image.

An image is decomposed into:

* Approximation information
* Horizontal details
* Vertical details
* Diagonal details

Noise can then be reduced by modifying or thresholding the detail coefficients before reconstructing the image.

Conceptually:

```text
Noisy Image
     ↓
2D Wavelet Transform
     ↓
Wavelet Coefficients
     ↓
Coefficient Thresholding
     ↓
Inverse Wavelet Transform
     ↓
Denoised Image
```

---

## Shearlet Transform

Shearlets provide a multi-scale and directional representation.

They are particularly useful for representing directional and anisotropic structures such as:

* Edges
* Curves
* Lines
* Oriented structures

The basic denoising concept is:

```text
Noisy Image
     ↓
Shearlet Transform
     ↓
Directional Shearlet Coefficients
     ↓
Coefficient Thresholding
     ↓
Inverse Shearlet Transform
     ↓
Denoised Image
```

The project therefore allows us to investigate how the two different representations behave when removing noise while retaining image structures.

---

# 5. Main Objective

The primary objective is to build a functional Python prototype that:

* Accepts an input image.
* Converts/preprocesses the image appropriately.
* Introduces controlled noise for experimentation.
* Applies Wavelet-based denoising.
* Applies Shearlet-based denoising.
* Reconstructs the processed images.
* Calculates image-quality metrics.
* Visually compares the results.
* Records processing time.
* Performs experiments across different noise conditions.
* Presents the comparison through an interactive interface.

---

# 6. Research Question

The central research question is:

> **How do Wavelet and Shearlet transform-based approaches compare for image denoising under different noise conditions, considering both image quality and computational cost?**

Supporting questions include:

* How effectively does each method reduce noise?
* How well does each method preserve image details?
* How does performance change as noise intensity increases?
* Does the type of noise affect the relative performance?
* What is the computational cost of each approach?
* What artifacts or limitations appear in each method?

---

# 7. Overall System

The complete prototype will follow this pipeline:

```text
                     INPUT IMAGE
                          │
                          ▼
                 IMAGE PREPROCESSING
                          │
                          ▼
                   NOISE GENERATOR
                          │
                          ▼
                    NOISY IMAGE
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
       WAVELET METHOD          SHEARLET METHOD
              │                       │
              ▼                       ▼
       DENOISED IMAGE           DENOISED IMAGE
              │                       │
              └───────────┬───────────┘
                          ▼
                   QUALITY METRICS
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
             MSE         PSNR        SSIM
              │           │           │
              └───────────┼───────────┘
                          ▼
                    COMPARISON
                          │
                          ▼
               VISUALIZATION & REPORT
                          │
                          ▼
                  STREAMLIT PROTOTYPE
```

---

# 8. What the User Will Be Able to Do

The final prototype will provide an interactive interface.

A typical user workflow will be:

```text
1. Upload an image
        ↓
2. View original image
        ↓
3. Select noise type
        ↓
4. Select noise intensity
        ↓
5. Generate noisy image
        ↓
6. Run Wavelet Denoising
        ↓
7. Run Shearlet Denoising
        ↓
8. View both results
        ↓
9. View PSNR / SSIM / MSE
        ↓
10. Compare processing time
        ↓
11. View comparison graphs
```

The interface is intended primarily for demonstration and experimentation.

The actual research logic remains in separate Python modules.

---

# 9. Noise Models

The prototype will support controlled noise generation.

Initial target noise models:

### Gaussian Noise

Used to simulate random intensity variations.

### Salt-and-Pepper Noise

Introduces random black and white pixels.

### Speckle Noise

Represents multiplicative noise commonly associated with certain imaging processes.

### Mixed Noise

Combines multiple noise characteristics to create a more challenging denoising condition.

The initial implementation will begin with a small controlled experiment and gradually expand.

---

# 10. Image Quality Metrics

The prototype will compare the denoised images using quantitative metrics.

## MSE — Mean Squared Error

Measures the average squared difference between the reference image and the processed image.

Lower MSE generally indicates smaller pixel-level error.

---

## PSNR — Peak Signal-to-Noise Ratio

Measures reconstruction quality using a logarithmic scale.

Higher PSNR generally indicates better similarity to the reference image.

---

## SSIM — Structural Similarity Index

Measures similarity in structural characteristics between images.

It is useful because denoising quality is not only about individual pixel differences; preservation of image structure is also important.

---

## Processing Time

The prototype will also record the time required by each method.

This allows us to investigate the trade-off between:

```text
Image Quality
       vs
Computational Cost
```

---

# 11. Experimental Methodology

The comparison must be controlled.

For a given experiment:

```text
Same Original Image
        │
        ▼
Same Noise Type
        │
        ▼
Same Noise Level
        │
        ├───────────────┐
        ▼               ▼
     Wavelet         Shearlet
        │               │
        ▼               ▼
   Result A          Result B
        │               │
        └───────┬───────┘
                ▼
          Same Metrics
```

This prevents the comparison from being affected by different input conditions.

---

# 12. Experimental Dataset

The prototype will initially use a small collection of standard/test images.

The experiment framework will eventually support:

```text
Multiple Images
       ×
Multiple Noise Types
       ×
Multiple Noise Levels
       ×
Wavelet Method
       ×
Shearlet Method
```

The exact final image dataset and number of images will be finalized during the experimentation phase.

---

# 13. Expected Output

The prototype should produce four primary visual outputs:

### 1. Original Image

The clean reference image.

### 2. Noisy Image

The controlled corrupted version of the image.

### 3. Wavelet Denoised Image

The result obtained using the Wavelet-based method.

### 4. Shearlet Denoised Image

The result obtained using the Shearlet-based method.

Example presentation:

```text
┌─────────────┬─────────────┐
│   Original  │    Noisy    │
├─────────────┼─────────────┤
│   Wavelet   │  Shearlet   │
└─────────────┴─────────────┘
```

Alongside the images:

```text
             Wavelet       Shearlet

MSE             ...            ...
PSNR            ...            ...
SSIM            ...            ...
Time            ...            ...
```

---

# 14. Technology Stack

## Programming Language

**Python**

Python is the primary implementation language for the complete research prototype.

## Numerical Computing

**NumPy**

Used for numerical arrays and mathematical operations.

## Scientific Computing

**SciPy**

Used for scientific and numerical processing where required.

## Wavelet Processing

**PyWavelets**

Used for Wavelet decomposition and reconstruction.

## Shearlet Processing

**pyShearLab**

Used for the Shearlet transform and reconstruction pipeline.

The package is being tested for compatibility with the project's current Python/NumPy environment.

## Image Processing

**OpenCV**

Used for image processing utilities.

**Pillow**

Used for image loading and handling.

**scikit-image**

Used for image-quality metrics and supporting image-processing functionality.

## Data Analysis

**Pandas**

Used to organize experiment results and generate comparison tables.

## Visualization

**Matplotlib**

Used to generate research graphs and visual comparisons.

## Prototype Interface

**Streamlit**

Used to build the interactive demonstration interface.

---

# 15. Software Architecture

The project will separate the research algorithms from the user interface.

```text
image-denoising/
│
├── app/
│   └── streamlit_app.py
│
├── algorithms/
│   ├── wavelet_denoising.py
│   └── shearlet_denoising.py
│
├── preprocessing/
│   ├── image_loader.py
│   └── noise.py
│
├── evaluation/
│   ├── metrics.py
│   └── comparison.py
│
├── visualization/
│   └── plots.py
│
├── experiments/
│   └── experiment_runner.py
│
├── data/
│   ├── original/
│   ├── noisy/
│   └── results/
│
├── tests/
│
├── requirements.txt
├── README.md
└── phases.md
```

---

# 16. Role of Each Module

### `preprocessing/`

Responsible for preparing images and generating controlled noise.

### `algorithms/`

Contains the actual Wavelet and Shearlet denoising implementations.

### `evaluation/`

Calculates MSE, PSNR, SSIM, runtime, and comparison results.

### `visualization/`

Creates graphs and visual comparisons.

### `experiments/`

Automates repeated experiments across images, noise types, and parameters.

### `app/`

Provides the Streamlit user interface.

### `tests/`

Checks that individual components work correctly.

### `data/`

Stores input images, noisy images, processed outputs, and experiment results.

---

# 17. What We Are NOT Building

To keep the project scope controlled, the current prototype does not include:

* Edge detection
* PDE-based edge detection
* Object detection
* Image classification
* Deep-learning-based denoising
* Real-time camera processing
* Mobile application
* Cloud deployment

The current focus is specifically:

> **Wavelet-based image denoising vs Shearlet-based image denoising.**

Additional features can be considered as future work if required.

---

# 18. Research Contribution / Project Value

The prototype is not simply an image filter.

The main value of the project is the **controlled comparison framework**.

Instead of only showing:

```text
"Noisy Image → Denoised Image"
```

the project investigates:

```text
Same Image
Same Noise
Same Noise Level
       │
       ├───────────────┐
       ▼               ▼
    Wavelet         Shearlet
       │               │
       ▼               ▼
    Metrics         Metrics
       │               │
       └───────┬───────┘
               ▼
         Evidence-Based
           Comparison
```

This allows the project to study the relationship between:

* Noise reduction
* Structural preservation
* Image quality
* Directional representation
* Computational cost

---

# 19. Expected Final Demonstration

During the final demonstration, we should be able to show:

```text
                 USER UPLOADS IMAGE
                         ↓
                  SELECTS NOISE
                         ↓
                  GENERATES NOISE
                         ↓
             ┌───────────┴───────────┐
             ▼                       ▼
          WAVELET                 SHEARLET
          DENOISING               DENOISING
             │                       │
             └───────────┬───────────┘
                         ▼
                   VISUAL RESULTS
                         +
                   METRIC RESULTS
                         +
                  PROCESSING TIME
                         ↓
                    COMPARISON
```

The final prototype should therefore be both:

1. **A functional image-denoising application**
2. **A small experimental research platform**

---

# 20. Final Project Goal

The final goal is to develop a Python-based interactive prototype that demonstrates and experimentally compares Wavelet and Shearlet transform-based image denoising.

The system will take a clean image, introduce controlled noise, apply both denoising approaches under identical conditions, reconstruct the images, evaluate them using MSE, PSNR, SSIM and processing time, and present the results through visual comparisons and an interactive Streamlit interface.

The project will use the experimental results to understand the strengths, limitations, quality characteristics, and computational trade-offs of the two transform-based approaches.

---

# 21. One-Line Explanation

> **We are building a Python-based prototype that adds controlled noise to images, removes that noise using Wavelet and Shearlet transforms, and experimentally compares both methods based on image quality, structural preservation, and processing time.**

---

# 22. Simple Explanation for Viva / Anyone

If someone asks:

**"What exactly are you building?"**

You can explain:

> "Our project is an image-denoising prototype. We take an image and introduce controlled noise into it. Then we use two mathematical transform-based techniques, Wavelet and Shearlet transforms, to remove the noise. We reconstruct the images and compare the two approaches using metrics such as PSNR, SSIM, MSE and processing time. Finally, we provide an interactive Streamlit interface where the complete process can be demonstrated visually."

If they ask:

**"What is the research part?"**

> "The research part is the controlled comparison. We don't assume that one technique is better. We test both methods on the same images, with the same noise conditions, and analyze their image quality and computational performance."

If they ask:

**"Why two methods?"**

> "Wavelets provide a multi-scale representation, while Shearlets provide a multi-scale and directional representation. Comparing them allows us to investigate how these different representations affect noise removal and preservation of image structures."

---

# 23. Current Development Status

### Completed

* Python environment
* Virtual environment
* Core scientific libraries
* Image-processing libraries
* PyWavelets installation
* pyShearLab installation
* Basic pyShearLab import verification
* Project folder structure
* Project phase roadmap

### Currently Working On

**Phase 0 — Shearlet compatibility and transform/reconstruction validation**

Before implementing the actual denoising algorithm, we must verify:

```text
Image
 ↓
Shearlet Transform
 ↓
Coefficients
 ↓
Inverse Shearlet Transform
 ↓
Reconstructed Image
```

### After Phase 0

The development sequence will be:

```text
Phase 1
Technical Design
      ↓
Phase 2
Image Preprocessing
      ↓
Phase 3
Wavelet Denoising
      ↓
Phase 4
Shearlet Denoising
      ↓
Phase 5
Metrics & Comparison
      ↓
Phase 6
Visualization
      ↓
Phase 7
Experiments
      ↓
Phase 8
Research Analysis
      ↓
Phase 9
Streamlit Prototype
      ↓
Phase 10
Testing
      ↓
Phase 11
Final Demonstration
      ↓
Phase 12
Documentation & Submission
```
