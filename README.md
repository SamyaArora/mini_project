# AI-based Predictive Maintenance for Prediction of Simulated Turbofan Engine

**NASA C-MAPSS FD001 — LSTM-based Remaining Useful Life (RUL) prediction**

## Overview
This project predicts the Remaining Useful Life (RUL) of simulated turbofan engines from sensor time-series data. It compares an LSTM neural network with a Ridge Regression baseline.

## Dataset
- NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset
- FD001: 100 training engines and 100 test engines
- One operating condition and one simulated fault mode

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) and place `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt` in `data/raw/`. Dataset files are not included in this repository.

## Method
- Uses 14 sensor features and 30-cycle input windows.
- Splits training engines into training and validation groups.
- Fits the scaler on training-split engines only.
- Caps training RUL targets at 125 cycles.
- Evaluates using RMSE, MAE, mean signed error, and the NASA scoring function.

## Test Results

| Model | Target | RMSE | MAE | NASA score |
|---|---|---:|---:|---:|
| Ridge | Raw RUL | 21.61 | 17.33 | 1087.2 |
| LSTM | Raw RUL | 14.71 | 10.96 | 357.6 |
| Ridge | Clipped RUL | 20.41 | 16.32 | 1011.7 |
| LSTM | Clipped RUL | 13.42 | 9.89 | 322.3 |

These are the recorded results from the current experiment. The LSTM has lower test RMSE and MAE than the Ridge baseline. See `reports/results.md` and `reports/results.csv` for the reported metrics.

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
This is an educational experiment using simulated FD001 data, not live aircraft telemetry. Results use one validation split and one recorded LSTM seed. This project is not a certified aircraft safety system.

## Future Work
Evaluate more C-MAPSS subsets, test multiple random seeds, and explore
alternative sequence models.