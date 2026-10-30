# AI-based Predictive Maintenance for Prediction of Simulated Turbofan Engine

**NASA C-MAPSS FD001 — LSTM-based Remaining Useful Life (RUL) prediction, with Ridge Regression, SimpleRNN, PCA, and a controlled five-model comparison**

## Overview
This project predicts the Remaining Useful Life (RUL) of simulated turbofan engines from sensor time-series data. It includes LSTM-based prediction, Ridge Regression and SimpleRNN experiments, PCA experiments, and a controlled five-model comparison of Ridge, SimpleRNN, and LSTM with and without PCA.

## Dataset
- NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset
- FD001: 100 training engines and 100 test engines
- One operating condition and one simulated fault mode

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) and place `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt` in `data/raw/`. Dataset files are not included in this repository.

## Method
- Uses 14 sensor features and 30-cycle input windows.
- Splits training engines into training and validation groups.
- The controlled comparison fits its preprocessing objects using training-engine data only. Notebook 01 is exploratory and has a documented limitation: it fits its scaler before the engine split, so its Ridge metrics are not part of the controlled comparison.
- Caps training RUL targets at 125 cycles.
- Evaluates using RMSE, MAE, mean signed error, and the NASA scoring function.

## Test Results

These are historical baseline records, not the controlled comparison below.
The README values for the original LSTM correspond to the saved seed-42 model;
`reports/results.csv` records metrics from the mean predictions of three seeds.
Those are different prediction protocols and should not be read as identical
results.

| Model | Target | RMSE | MAE | NASA score |
|---|---|---:|---:|---:|
| Ridge | Raw RUL | 21.61 | 17.33 | 1087.2 |
| LSTM | Raw RUL | 14.71 | 10.96 | 357.6 |
| Ridge | Clipped RUL | 20.41 | 16.32 | 1011.7 |
| LSTM | Clipped RUL | 13.42 | 9.89 | 322.3 |

These are historical baseline results; the controlled comparison below is the
like-for-like evaluation. See `reports/results.md` and `reports/results.csv`
for the original recorded metrics.
The separate exploration notebook fits its scaler before its engine split, so
its Ridge metrics are exploratory and are not used in the controlled comparison.

## PCA Experiment (21 Sensors)

A separate experiment uses all 21 sensor measurements, standardizes them using
training-engine rows only, and applies PCA retaining at least 95% of the
training-engine variance. PCA is dimensionality reduction: each component is a
linear combination of sensor measurements, not a selected original sensor.
The experiment uses 30-cycle windows and the same 80/20 engine split protocol
as the sequence baseline. The official test evaluation uses one final window
per engine and uncapped official RUL labels.

The completed run retained 10 principal components (95.1594% cumulative
explained variance). These results belong to the separate PCA experiment and
should not be treated as directly interchangeable with the existing 14-sensor
baseline:

| Model | RMSE | MAE | NASA score |
|---|---:|---:|---:|
| Ridge Regression | 21.7092 | 17.4440 | 1079.8589 |
| Simple RNN | 20.1165 | 15.0525 | 1141.1451 |
| LSTM | 15.9969 | 11.6757 | 670.7957 |

The verified experiment details and metrics are in
`reports/pca_model_comparison.md` and `reports/pca_model_comparison.csv`.
The explained-variance plot is
`reports/figures/pca_cumulative_explained_variance.png`.

## Controlled Baseline vs PCA Comparison

The controlled comparison retrains the original 14-sensor Ridge and LSTM
baselines alongside PCA Ridge, PCA SimpleRNN, and PCA LSTM. All models use
the same 80/20 engine split (seed 42), 30-cycle windows, capped training
targets, and the same 100 uncapped official test RUL labels. Each scaler/PCA
transform is fit only on training-engine sensor rows. Neural models use seeds
42, 43, and 44; their values below are mean ± sample standard deviation.
Ridge is a single deterministic fit.

| Model | Features | PCA components | RMSE | MAE | NASA score |
|---|---|---:|---:|---:|---:|
| Original 14-sensor Ridge (retrained) | 14 selected sensors | — | 21.6072 | 17.3268 | 1087.1635 |
| PCA Ridge | All 21 sensors transformed by PCA | 10 | 21.7092 | 17.4440 | 1079.8589 |
| Original 14-sensor LSTM (3 seeds) | 14 selected sensors | — | 14.8876 ± 0.1792 | 11.1554 ± 0.1757 | 380.3457 ± 13.2517 |
| PCA SimpleRNN (3 seeds) | All 21 sensors transformed by PCA | 10 | 19.1310 ± 1.1944 | 14.0696 ± 1.0689 | 988.6472 ± 227.4474 |
| PCA LSTM (3 seeds) | All 21 sensors transformed by PCA | 10 | 15.4767 ± 0.6264 | 11.7092 ± 0.2079 | 494.5570 ± 159.9626 |

These are the like-for-like comparison results. Per-seed metrics,
full-precision test predictions, the data/pipeline audit, and metric definitions
are recorded in `reports/fd001_fair_comparison.csv`,
`reports/fd001_fair_comparison_predictions.csv`,
`reports/fd001_data_pipeline_audit.md`, and
`reports/fd001_fair_comparison.md`.

## Figures
- `reports/figures/true_vs_pred_test_raw.png`
- `reports/figures/lstm_loss_curves.png`
- `reports/figures/residual_hist_lstm_test_raw.png`

## Setup (Windows PowerShell)

Activate the existing environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies if needed:

```powershell
python -m pip install -r requirements.txt
```

## Run the Notebook

From the project root:

```powershell
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebooks/02_baseline_lstm.ipynb
```

Notebook execution can take several minutes because it trains the LSTM.

## Run the PCA Model Comparison

From the project root in Windows PowerShell, run the separate PCA experiment:

```powershell
.\.venv\Scripts\python.exe scripts\run_pca_experiments.py
```

By default, the script refuses to overwrite PCA artifacts or reports. To
rerun and replace only those PCA experiment outputs:

```powershell
.\.venv\Scripts\python.exe scripts\run_pca_experiments.py --overwrite
```

To run the audited five-model comparison without overwriting existing outputs,
use a new run tag:

```powershell
.\.venv\Scripts\python.exe scripts\audit_compare_fd001.py --run-tag rerun_01
```

The audit can also be run without training:

```powershell
.\.venv\Scripts\python.exe scripts\audit_compare_fd001.py --audit-only --run-tag audit_01
```

## Run Inference

`src/predict.py` loads `models/lstm_rul_fd001.keras` and the matching
`models/scaler_lstm_fd001.pkl`. The scaler uses the final notebook's
training-only MinMax preprocessing and expects raw sensor values ordered as
2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, and 21.

```python
from src.predict import predict_rul

rul_cycles = predict_rul(window)
```

`window` must have shape `(30, 14)`. Run `python -m pytest` to test inference
input validation and model execution.

## Project Structure
- `data/raw/` — downloaded dataset files
- `notebooks/01_data_exploration.ipynb` — existing exploration notebook
- `notebooks/02_baseline_lstm.ipynb` — baseline and LSTM experiment
- `reports/` — metrics, logs, and figures
- `models/` — saved model and scaler artifacts

## Limitations
This is an educational experiment using simulated FD001 data, not live aircraft telemetry. The controlled comparison uses one validation split and three neural-model seeds. This project is not a certified aircraft safety system.

## Future Work
Evaluate more C-MAPSS subsets, test multiple random seeds, and explore
alternative sequence models.