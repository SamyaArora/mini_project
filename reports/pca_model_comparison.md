# FD001 PCA Model Comparison

PCA is dimensionality reduction: each component is a linear combination of the 21 sensor measurements, not a selected original sensor.

The scaler and PCA were fit only on training-engine sensor rows. Validation and official test rows were transformed with those fitted objects.

- PCA components retained: 10 (cumulative explained variance: 0.951594)
- Input sensor measurements: 21
- Window length: 30 cycles
- Training target: labeled RUL capped at 125 cycles
- Evaluation target: official, uncapped FD001 test RUL
- Official test evaluation: one final window for each of 100 engines
- Ridge: alpha=1.0, using the final timestep of each window
- SimpleRNN: seed=42; LSTM: seed=43
- Recurrent models: Adam (learning rate 0.001), batch size 128, up to 50 epochs, EarlyStopping on validation loss (patience 7, restore best weights)

NASA score formula, with d = prediction - actual:

- d < 0: exp(-d / 13) - 1
- d >= 0: exp(d / 10) - 1
- The reported score is the sum of per-engine penalties; lower is better.

| Model | Feature representation | PCA components | Window | RMSE | MAE | NASA score |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Ridge | PCA-transformed 21 sensors (95% variance) | 10 | 30 | 21.7092 | 17.4440 | 1079.8589 |
| SimpleRNN | PCA-transformed 21 sensors (95% variance) | 10 | 30 | 20.1165 | 15.0525 | 1141.1451 |
| LSTM | PCA-transformed 21 sensors (95% variance) | 10 | 30 | 15.9969 | 11.6757 | 670.7957 |

These results are from the PCA experiment and are not directly interchangeable with the existing 14-sensor baseline.

Explained-variance plot: `figures/pca_cumulative_explained_variance.png`
