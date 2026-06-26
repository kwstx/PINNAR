# Model Card: PINNAR (Physics-Informed Neural Network for Planetary Resource Mapping)

## Model Details
- **Architecture**: Hybrid Spatial-Spectral model utilizing Fourier feature mapping for high-frequency spectral absorption features and a U-Net spatial branch for enforcing geological continuity.
- **Physics Integration**: Incorporates a differentiable two-stream Hapke radiative transfer model. The network directly predicts endmember abundances and Single-Scattering Albedos (SSA), ensuring predictions are constrained to physically plausible regions.
- **Output Types**:
  - Pixel-wise predictive mean abundance maps for specific mineral endmembers (e.g., Olivine, Pyroxene, Water Ice).
  - Epistemic uncertainty maps.
  - Total predictive variance (Aleatoric + Epistemic).

## Intended Use
- **Primary Use Case**: Automated, physics-constrained unmixing of hyperspectral datasets (e.g., CRISM, M3) for precise planetary mineralogical mapping.
- **Downstream Applications**: In-Situ Resource Utilization (ISRU) site selection, mission planning, and global geological surveys on the Moon and Mars.
- **Out-of-Scope Uses**: High-precision quantitative trace element analysis (where dedicated laboratory spectroscopy is required).

## Training Data
- **Pre-training**: Large corpora of unlabeled orbital hyperspectral imagery (e.g., CRISM uncalibrated radiances) using a purely physics-constrained loss.
- **Fine-tuning**: High-quality, sparsely labeled pixels mapped to known ground-truth mineralogy derived from rover analogs (e.g., Perseverance PIXL/SHERLOC data) or Apollo returned samples.

## Performance Metrics
- **Quantitative Metrics**: Mean Absolute Error (MAE) per endmember, Spectral Angle Mapper (SAM) for spectral fidelity.
- **Reliability Metrics**: Expected Calibration Error (ECE) on the uncertainty predictions.
- **Domain Adaptation**: Evaluated via cross-mission consistency residuals (e.g., transferring models trained on M3 to analyze synthetic lunar hyperspectral data).

## Limitations and Uncertainties
- Performance is heavily dependent on the atmospheric correction applied during preprocessing, particularly for Martian datasets (CRISM).
- The assumption of linear unmixing in SSA space simplifies complex non-linear macroscopic mixing scenarios.
- High uncertainty regions are expected in deeply shadowed craters, which the epistemic uncertainty map reliably captures.

## Ethical Considerations
Open-sourcing this model accelerates global scientific collaboration for space exploration. Ensure datasets used for fine-tuning acknowledge the original planetary missions and their principal investigators.
