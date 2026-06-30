# PINNAR: Physics-Informed Neural Network for Planetary Resource Mapping

## Overview
PINNAR is an advanced machine learning pipeline designed to identify and map valuable resources, such as water ice and specific minerals, on the Moon and Mars. By processing orbital hyperspectral imagery (e.g., CRISM, M3 datasets), PINNAR enables automated, physics-constrained unmixing for precise planetary mineralogical mapping.

## Core Characteristics
- **Hybrid Architecture**: Combines a Spatial-Spectral model utilizing Fourier feature mapping for high-frequency spectral absorption features with a U-Net spatial branch to enforce geological continuity.
- **Physics Integration**: Incorporates a differentiable two-stream Hapke radiative transfer model. It directly predicts endmember abundances and Single-Scattering Albedos (SSA), ensuring physically plausible outputs.
- **Uncertainty Quantification**: Generates predictive mean abundance maps alongside epistemic uncertainty and total predictive variance maps, providing reliability metrics for predictions.
- **Two-Stage Training**: Utilizes pre-training on large, unlabeled orbital hyperspectral corpora with physics-constrained loss, followed by fine-tuning on high-quality, ground-truth-mapped pixels.

## Project Structure
- `src/`: Contains the core Python machine learning codebase, including the hybrid model architecture, deep ensembles, loss functions (physics and composite), and the training/validation pipeline.
- `client/`: A front-end web application for data visualization and interaction.
- `server/`: A backend Node.js server that provides the API and supports the client interface.
- `notebooks/`: Jupyter notebooks containing examples and data exploration workflows.
- `run_pipeline.py`: The primary CLI entry point for executing model training (pre-training/fine-tuning) and sensitivity analysis (Sobol/hyperparameter sweeps).
- `verify_pipeline.py`: Utility script for verifying the integrity and configuration of the data pipeline.

## Usage

### Training the Model
Use the `run_pipeline.py` script to initiate the training process. The training stage (`pretrain`, `finetune`, or `both`) can be specified via command-line arguments.

```bash
python run_pipeline.py train --stage both --config config.yaml
```

### Running Sensitivity Analysis
The pipeline supports extensive sensitivity analysis, including Sobol analysis and hyperparameter sweeps.

```bash
python run_pipeline.py sensitivity --type all
```

## Intended Applications
PINNAR is engineered to support In-Situ Resource Utilization (ISRU) site selection, planetary mission planning, and global geological surveys. It is optimized for automated mapping rather than high-precision quantitative trace element analysis (which necessitates laboratory spectroscopy).

## Known Limitations
- Model performance is highly dependent on the accuracy of the atmospheric correction applied during dataset preprocessing (especially for Martian CRISM data).
- The use of linear unmixing in SSA space simplifies complex, non-linear macroscopic mixing scenarios.
- Predictions in deeply shadowed regions (e.g., steep craters) will natively exhibit high uncertainty, which is explicitly quantified in the output epistemic uncertainty maps.
