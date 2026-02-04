# TESS Light Curve Anomaly Detection - Weighted VAE

This repository contains a weighted Variational Autoencoder (VAE) implementation for creating latent vectors and detecting anomalies in TESS light curves.

## Overview

This project uses a weighted VAE to analyze TESS (Transiting Exoplanet Survey Satellite) light curve data, enabling:
- **Latent Vector Creation**: Generate compact representations of light curve data
- **Anomaly Detection**: Identify unusual or anomalous light curves in the TESS dataset
- **Uncertainty Quantification**: Quantify reconstruction uncertainty and model confidence

## Repository Structure

```
.
├── README.md                                          # This file
├── .gitignore                                         # Git ignore rules
├── src/
│   └── util.py                                        # Utility functions and helpers
├── docs/
│   └── paper/
│       ├── CCIR_William_NHSJS_online_rev1.docx       # Online paper revision
│       ├── CCIR_William_NHSJS_standard_rev1.docx     # Standard paper revision
│       └── William_NHSJS_Reviewer_Response_Letter.docx # Reviewer responses
└── Jupyter Notebooks/
    ├── 5k-cap-v20251213_basic_norm_uncert_recon_3m.ipynb      # 3-minute analysis
    ├── 5k-cap-v20251213_basic_norm_uncert_recon_5m.ipynb      # 5-minute analysis
    ├── 5k-cap-v20251213_basic_norm_uncert_recon_10m.ipynb     # 10-minute analysis
    ├── 5k-cap-v20251213_basic_norm_uncert_recon_15m.ipynb     # 15-minute analysis
    ├── 5k-cap-v20251213_basic_norm_uncert_recon_25m.ipynb     # 25-minute analysis
    └── 5k-cap-v20251213_basic_norm_uncert_recon_25m_early_stopping.ipynb # 25m with early stopping
```

## Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab
- Common dependencies: numpy, pandas, matplotlib, scikit-learn, PyTorch 

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd TESS-light-curve-anomaly-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter:
```bash
jupyter notebook
```

## Notebooks

Each notebook implements VAE-based analysis with:
- **Normalization**: Preprocessing and normalization of TESS light curves
- **Uncertainty Quantification**: Estimation of reconstruction uncertainty
- **Reconstruction**: VAE-based light curve reconstruction

Available time resolutions:
- **3m, 5m, 10m, 15m, 25m**: Different temporal resolutions for light curve analysis
- **Early Stopping**: Version with early stopping to prevent overfitting

## Utilities

The `src/util.py` file contains helper functions for:
- Data loading and preprocessing
- VAE model components
- Anomaly detection and scoring

## Usage

Open any notebook and run the cells in sequence. Notebooks are self-contained and include:
1. Data loading from TESS
2. VAE model training
3. Latent space visualization
4. Anomaly detection results

## Publication

See `docs/paper/` for manuscript drafts and reviewer response materials.

