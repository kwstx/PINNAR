# Manuscript Outline

**Working Title:** Physics-Informed Neural Networks for Robust and Calibrated Hyperspectral Mineral Mapping of Planetary Surfaces

**Target Journals:** *Icarus*, *Remote Sensing of Environment (RSE)*

## Abstract
- Briefly state the challenge of planetary hyperspectral unmixing (data sparsity, non-linear mixing, instrument variability).
- Introduce PINNAR: A hybrid spatial-spectral deep learning model constrained by Hapke radiative transfer physics.
- Highlight the three novel contributions:
  1. Physics-constrained abundance retrieval bridging analytical models and data-driven methods.
  2. Cross-mission domain adaptation ensuring robust performance across diverse sensor responses.
  3. Calibrated epistemic and aleatoric uncertainty maps crucial for autonomous In-Situ Resource Utilization (ISRU) site selection.
- Summarize key results (improved spectral fidelity, physically valid single-scattering albedos, robust performance on held-out PDS datasets).

## 1. Introduction
- The need for large-scale, automated mineral mapping for lunar and Martian exploration.
- Limitations of current methods (purely analytical models are slow; purely data-driven models lack physical generalizability and fail on out-of-distribution spectra).
- Enter Physics-Informed Deep Learning: combining the speed of neural networks with the rigorous constraints of physics.
- **Specific Objectives of this Paper**: Detail the PINNAR architecture, validate against PDS subsets, and demonstrate uncertainty quantification for mission planning.

## 2. Methodology
- **2.1 Spatial-Spectral Architecture**: 
  - Fourier feature mapping for high-frequency spectral absorption resolution.
  - Spatial U-Net branch for enforcing geological continuity (Markov Random Field concepts).
- **2.2 Physics-Informed Loss Construction**:
  - Differentiable implementation of the two-stream Hapke approximation.
  - Linear unmixing in Single-Scattering Albedo (SSA) space.
  - Formulation of the composite loss (Data MSE/SAM + Physics Residual).
- **2.3 Uncertainty Quantification**:
  - Implementation of Deep Ensembles.
  - Derivation of predictive mean, epistemic variance, and aleatoric variance.
- **2.4 Cross-Mission Domain Adaptation**:
  - Sensor-specific calibration heads regularized by the shared physical latent space.

## 3. Data & Experiments
- **3.1 Dataset Construction**:
  - Preprocessing of CRISM/M3 datasets (atmospheric correction, destriping).
  - Generation of labeled subsets via analog matching (Apollo soils, Perseverance PIXL).
- **3.2 Training Regimen**:
  - Two-stage training: unsupervised pre-training on physics loss, supervised fine-tuning.
- **3.3 Experimental Setup**:
  - Ablation studies (removing physics loss, removing spatial branch).
  - Hyperparameter sweeps and Global Sensitivity Analysis (Sobol indices on SSA and g-parameter).

## 4. Results
- **4.1 Quantitative Performance**:
  - RMSE, SAM, and Abundance MAE comparisons against baseline unmixing methods.
- **4.2 Global Sensitivity Analysis**:
  - Presentation of Sobol indices indicating the strong governing effect of SSA and phase functions on the network's predictive stability.
- **4.3 Uncertainty Calibration**:
  - Reliability diagrams and Expected Calibration Error (ECE).
  - Visualization of high-uncertainty regions (e.g., shadowed craters).
- **4.4 Cross-Mission Harmonization**:
  - Demonstrating consistency of mineral maps when applying the model across different orbital tracks and instrument configurations.

## 5. Discussion
- **5.1 Interpretability of Physics Constraints**: How the Hapke residual acts as a regularizer in data-sparse regimes.
- **5.2 Implications for ISRU**: Utilizing calibrated uncertainty maps for risk-averse robotic landing site selection.
- **5.3 Limitations**: Sensitivity to atmospheric residuals and assumptions of linear SSA unmixing.

## 6. Conclusion
- Summary of PINNAR's capabilities.
- Open-source release of the framework.
- Future work (incorporating thermal emissions, full non-linear mixing models).
