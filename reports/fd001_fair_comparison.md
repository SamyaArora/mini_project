# Controlled FD001 Model Comparison

All five models were newly fit with this script. They use the same raw training labels capped at 125, 80/20 engine split (seed 42), 30-cycle windows, 100 official test engines, uncapped official test RUL labels, and metric definitions. No official test labels were used for fitting, early stopping, or model selection.

The 14-sensor models use the baseline's 14 named sensors and a training-engine-only MinMaxScaler. PCA models use all 21 sensors, a training-engine-only StandardScaler and PCA(n_components=0.95). The fitted PCA retained 10 components. PCA is dimensionality reduction (linear combinations of sensors), not feature selection.

Neural models use seeds 42, 43, 44, Adam (0.001), batch size 128, at most 60 epochs, and validation-loss EarlyStopping (patience 10, restore best weights).

Displayed values are mean ± sample standard deviation over the listed neural-model seeds. Ridge is one deterministic fit; its standard deviation is not applicable.

| Model | Representation | PCA components | Runs | RMSE | MAE | NASA score |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Original 14-sensor Ridge (retrained) | 14 selected sensors + MinMaxScaler | — | 1 | 21.6072 (single run) | 17.3268 (single run) | 1087.1635 (single run) |
| PCA Ridge | 21 sensors -> StandardScaler -> PCA | 10 | 1 | 21.7092 (single run) | 17.4440 (single run) | 1079.8589 (single run) |
| Original 14-sensor LSTM | 14 selected sensors + MinMaxScaler | — | 3 | 14.8876 ± 0.1792 | 11.1554 ± 0.1757 | 380.3457 ± 13.2517 |
| PCA SimpleRNN | 21 sensors -> StandardScaler -> PCA | 10 | 3 | 19.1310 ± 1.1944 | 14.0696 ± 1.0689 | 988.6472 ± 227.4474 |
| PCA LSTM | 21 sensors -> StandardScaler -> PCA | 10 | 3 | 15.4767 ± 0.6264 | 11.7092 ± 0.2079 | 494.5570 ± 159.9626 |

NASA asymmetric score: with `d = prediction - actual`, each engine contributes `exp(-d/13)-1` for `d < 0` (early prediction) and `exp(d/10)-1` for `d >= 0` (late prediction); the reported score is the sum over 100 engines.

The original saved 14-sensor LSTM/Ridge history is not substituted for this controlled comparison. The table contains retrained models evaluated under one common protocol. Per-seed metrics are in the CSV; official test predictions are in `fd001_fair_comparison_predictions.csv`.
