# LSTM-Based State Estimation with Moving Horizon Estimation for Penicillin Fermentation

A hybrid data-driven and optimization-based framework for estimating key states in a penicillin fermentation process under sparse and delayed measurements.

## Overview

This project develops an **LSTM-based state estimator coupled with Moving Horizon Estimation (MHE)** for the estimation of:

- pH
- Penicillin concentration
- Dissolved oxygen (DO₂)

The workflow combines time-series machine learning, dimensionality reduction, missing-data imputation, delayed-measurement handling, and constrained optimization through GEKKO.

The notebook also evaluates a **baseline one-step-ahead LSTM** against the LSTM + fixed-lag MHE framework.

## Problem Statement

Fermentation processes contain important states that may be difficult to measure continuously. Penicillin concentration, in particular, may be available only intermittently and with measurement delay.

This project simulates that situation by introducing:

- sparse penicillin measurements
- delayed penicillin measurements
- continuous pH measurements
- continuous DO₂ measurements

The estimator then combines available measurements, LSTM predictions, imputed values, and temporal consistency within a moving estimation horizon.

## Methodology

```text
Raw fermentation data
        │
        ▼
Data cleaning and outlier removal
        │
        ▼
Robust scaling
        │
        ▼
PCA dimensionality reduction
        │
        ▼
LSTM one-step-ahead state prediction
        │
        ├──────────────► Baseline LSTM evaluation
        │
        ▼
Sparse / delayed measurement simulation
        │
        ▼
Penicillin missing-value imputation
        │
        ▼
Fixed-lag Moving Horizon Estimation
        │
        ▼
Estimated pH, Penicillin and DO₂
        │
        ▼
Comparison with baseline LSTM
```

## Dataset

The notebook uses the **IndPensim** penicillin fermentation dataset.

The original dataset contains 4,630 observations and 22 variables. The variables include process inputs, historical DO₂ measurements, process measurements, and the target state variables.

Key variables include:

- Time
- Historical DO₂ measurements
- Aeration rate
- Sugar feed rate
- Base flow rate
- Heating/cooling water flow rate
- Vessel volume and weight
- CO₂ in off-gas
- PAA flow
- Oil flow
- Oxygen Uptake Rate (OUR)
- Carbon Evolution Rate (CER)
- pH
- Penicillin concentration
- Dissolved oxygen concentration

The notebook removes rows containing zero values and applies statistical outlier removal before model development.

## Data Preparation

The preprocessing workflow includes:

1. Removal of rows containing zero values.
2. Statistical outlier removal using a median ± 3 standard deviation criterion.
3. Time-series train/test splitting without shuffling.
4. Robust scaling using `RobustScaler`.
5. Principal Component Analysis (PCA) retaining 95% explained variance.
6. Preparation of current-state and next-time-step targets.

The final state vector is:

```text
[pH, Penicillin concentration, DO₂]
```

## LSTM State Prediction

The notebook uses Keras/TensorFlow to build a one-step-ahead LSTM model.

The hyperparameter search explores:

- Number of LSTM layers
- LSTM units
- LSTM dropout
- Number of dense layers
- Dense-layer units
- Dense-layer dropout
- Optimizer
- Learning rate

The hyperparameter search uses Keras Tuner random search with validation loss as the objective.

Training uses:

- Mean squared error loss
- Early stopping
- Learning-rate reduction
- Model checkpointing

The notebook uses a one-step sequence length for the LSTM input.

## Sparse and Delayed Measurements

To test estimator robustness under realistic measurement constraints, the test data are converted into a relaxed measurement set.

The implemented scenario is:

| State | Sampling interval | Delay |
|---|---:|---:|
| pH | Every sample | 0 samples |
| Penicillin | Every 60 samples | 20 samples |
| DO₂ | Every sample | 0 samples |

This creates a challenging estimation problem in which penicillin concentration is both **sparsely measured and delayed**.

## Penicillin Imputation

A Gradient Boosting Regressor is used within the estimation workflow to reconstruct missing penicillin measurements from the available process information.

Observed penicillin values are retained, while missing values are estimated before being supplied to the MHE workflow.

## Moving Horizon Estimation

The MHE stage uses **GEKKO** to estimate the process states over a sliding horizon.

The implemented configuration includes:

- MHE window size: 30 samples
- Fixed-lag smoothing: 25 samples
- Measurement consistency terms
- LSTM prediction terms
- Initial-state alignment
- Temporal smoothness regularization

The test set contains 723 samples, producing the corresponding sliding-window MHE iterations.

The MHE objective combines measurement information with model predictions and temporal consistency to obtain state estimates that are more robust to sparse and delayed observations.

## Evaluation

The notebook evaluates both the baseline LSTM and the LSTM + MHE estimator using:

- R²
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

The metrics are calculated directly from the prediction arrays in the notebook.

### Baseline LSTM

| State | R² | RMSE | MAE |
|---|---:|---:|---:|
| pH | 0.7810 | 0.0276 | 0.0135 |
| Penicillin (g/L) | 0.6507 | 3.7648 | 2.9696 |
| DO₂ (mg/L) | 0.9084 | 0.5561 | 0.2551 |

### LSTM + Fixed-Lag MHE

The MHE metrics are computed in the notebook using the variables:

```python
r2_ph_mhe
r2_do2_mhe
r2_penicillin_mhe

rmse_ph_mhe
rmse_do2_mhe
rmse_penicillin_mhe

mae_ph_mhe
mae_do2_mhe
mae_penicillin_mhe
```

This keeps the reported MHE results tied directly to the executed notebook rather than manually entered values.

## Repository Structure

```text
.
├── LSTM_MHE_Estimator.ipynb
├── IndPensim_data.csv          # not included by default
├── requirements.txt
├── README.md
└── figures/                    # optional
```

## Installation

Create a Python environment and install the required packages:

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter notebook
```

Then open:

```text
LSTM_MHE_Estimator.ipynb
```

## Data Path

The original notebook contains a local Windows path:

```python
file_path = "D:\College\Projects\LSTM-MHE\IndPensim_data.csv"
```

For a portable repository, change this to a relative path such as:

```python
file_path = "data/IndPensim_data.csv"
```

and place the dataset in the corresponding `data/` directory if redistribution is permitted.

## Reproducibility

For reproducible execution:

1. Create a clean Python environment.
2. Install `requirements.txt`.
3. Place the permitted dataset at the expected relative path.
4. Run the notebook from top to bottom.
5. Use the metric variables generated by the notebook for the final comparison.

The notebook currently includes trained-model checkpoint paths and Keras Tuner output directories generated during execution. These generated artifacts do not need to be committed unless you specifically want to distribute trained models.

## Key Technologies

- Python
- TensorFlow / Keras
- Keras Tuner
- Scikit-learn
- GEKKO
- NumPy
- Pandas
- Matplotlib
- LSTM neural networks
- PCA
- Robust scaling
- Gradient Boosting regression
- Moving Horizon Estimation
- Fixed-lag smoothing

## Limitations

- The current LSTM uses a one-step sequence length.
- The measurement scenario is simulated from the available test data rather than obtained from a live process.
- MHE performance depends on the selected horizon, lag, weights, and measurement configuration.
- The notebook contains local file paths that should be replaced with repository-relative paths.
- Public redistribution of the underlying dataset should be verified separately from publication of the code.

## Future Work

Potential extensions include:

- Longer LSTM temporal windows
- Adaptive measurement-delay handling
- Online MHE deployment
- Automated hyperparameter optimization of MHE weights
- Uncertainty-aware state estimation
- Comparison with alternative recurrent architectures
- Integration with real fermentation measurements
- Computational benchmarking for online deployment

## Citation

If this repository is used in academic work, please cite the associated research work and acknowledge the original dataset source as required by its terms of use.

## Author

**Vishnu Roshan**

Chemical Engineering

---

**Project focus:** Data-driven state estimation and optimization-based estimation for penicillin fermentation.
