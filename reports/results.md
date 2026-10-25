# AI-based Predictive Maintenance for Prediction of Simulated Turbofan Engine

NASA C-MAPSS FD001, LSTM-based RUL prediction

Mean signed error = mean(pred − true): **positive** = late (over-predicting RUL); **negative** = early.

## Metrics (from run_log.txt)

| Model | Split | Target | RMSE | MAE | Mean signed error | NASA |
| --- | --- | --- | --- | --- | --- | --- |
| Ridge | validation | clipped | 19.97 | 15.84 | 0.40 | 23761.0 |
| Ridge | test | raw | 21.61 | 17.33 | 1.97 | 1087.2 |
| Ridge | test | clipped | 20.41 | 16.32 | 3.04 | 1011.7 |
| LSTM | validation | clipped | 11.51 | 8.21 | -0.28 | 7284.5 |
| LSTM | test | raw | 14.75 | 11.22 | 2.09 | 363.2 |
| LSTM | test | clipped | 13.58 | 10.15 | 3.16 | 332.1 |

## Ridge comparison to notebook 01

Notebook `01_data_exploration.ipynb` fit MinMaxScaler on all 100 training engines before the engine split; this notebook fits the scaler on training-split engines only (protocol). Validation/test metrics may differ.

Reference (01, all-engine scaler): test RMSE 21.61 / NASA 1087.2 vs **raw** RUL; 20.41 / 1011.7 vs **clipped**; validation RMSE 19.97 vs clipped.

## Figures

- `figures/true_vs_pred_test_raw.png`
- `figures/lstm_loss_curves.png`
- `figures/residual_hist_lstm_test_raw.png`
