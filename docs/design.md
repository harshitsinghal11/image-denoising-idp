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

* Effective denoising: higher PSNR and SSIM than the noisy input, with no visible artifacts.
* Comparison rule: a method wins a condition only if it wins on PSNR and SSIM across multiple images.
* Fairness rule: both methods get the same threshold-tuning budget.
* Timing rule: shearlet system setup is timed separately from transform time.
* Fixed pipeline: the order from phases.md (image, preprocessing, noise, both methods, metrics, comparison).