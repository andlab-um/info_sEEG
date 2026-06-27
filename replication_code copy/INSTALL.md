# Installation

## Tested Or Assumed Systems

The lightweight demo was prepared for macOS or Linux with Python 3.9 or newer. It uses only the Python standard library.

## Lightweight Demo Environment

From the repository root:

```bash
python3 --version
python3 replication_code/scripts/run_demo.py --config replication_code/config/demo_config.yaml
```

Typical installation time is under one minute because no packages are installed.

## Full Analysis Environment

The original notebooks indicate the following scientific stack for manuscript-scale reproduction:

- Python 3.9 or newer.
- NumPy, SciPy, pandas.
- matplotlib and seaborn.
- PyMC, ArviZ, JAX, NumPyro for hierarchical Bayesian modeling.
- MNE-Python and mne-connectivity for sEEG time-frequency and PSI analyses.
- scikit-learn and statsmodels for PCA, decoding, RSA, regression, and mixed-effects summaries.
- MATLAB with EEGLAB, Brainstorm, SPM12, and CAT12 for raw clinical SEEG preprocessing and electrode localization.

A typical Python setup may look like:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy pandas matplotlib seaborn scikit-learn statsmodels pymc arviz jax numpyro mne mne-connectivity
```

The full PCA-family reviewer scripts require only this subset:

```bash
python -m pip install numpy scipy pandas matplotlib scikit-learn statsmodels
```

The full reviewer-facing manuscript scripts additionally require:

```bash
python -m pip install pymc arviz pytensor jax numpyro mne
```

`mne` is required for EEGLAB `.set` epoch loading in the TFR, PLV, and PSI scripts. `pymc`, `arviz`, and the optional JAX/NumPyro stack are required for behavioral model refitting.

The exact versions used in the manuscript should be pinned before archival release. The manuscript methods mention PyMC v5.25.1 and MATLAB R2023b/R2025b.

## Optional Acceleration

JAX/NumPyro acceleration is optional for the demo but useful for full hierarchical Bayesian model fitting. GPU acceleration is optional and should not change model definitions.

## Expected Installation Time

- Lightweight demo: under one minute.
- Full Python environment: about 10 to 30 minutes on a normal desktop, depending on network and compiled dependencies.
- MATLAB toolboxes and raw preprocessing environment: depends on local licenses and existing installation.
