# FD001 Raw Data and Pipeline Audit

**Audit result: 28/28 checks passed; 0 failed.**

This audit compared the local raw NASA C-MAPSS FD001 files against the derived labeled files and inspected the saved PCA/original-LSTM artifacts. Raw data files were read only.

## Data and preprocessing checks

- **PASS** Raw training dimensions: shape=(20631, 26); expected (20631, 26)
- **PASS** Raw training engines: 100 unique engine IDs; expected 100
- **PASS** Raw test dimensions: shape=(13096, 26), engines=100; expected (13096, 26), 100 engines
- **PASS** Official test RUL labels: 100 labels; expected 100
- **PASS** Processed CSV schema and dimensions: shape=(20631, 27); columns match raw 26 plus final RUL
- **PASS** Raw-to-CSV row-by-row values: all 26 original columns match exactly in the original row order
- **PASS** Engine/cycle ordering: engine IDs and cycles match row-for-row
- **PASS** Independently recalculated training RUL: min(max cycle - current cycle, 125) matches every processed label
- **PASS** Processed RUL type and range: dtype=int64, range=0..125
- **PASS** Processed TXT/CSV consistency: all 27 columns match exactly
- **PASS** Processed schema JSON: 27 names and dimensions agree with labeled CSV
- **PASS** Missing and non-finite values: none found in raw train/test, processed CSV, or official labels
- **PASS** Duplicate data rows/cycles: full duplicate rows=0; duplicate engine-cycle keys=0
- **PASS** Engine IDs and cycle validity: integer IDs 1..100 and positive integer cycles
- **PASS** Sensor and setting values: all sensor measurements finite/nonnegative; operating settings finite (negative settings_1/setting_2 values are present in raw data)
- **PASS** Raw training row order: cycles increase within each engine
- **PASS** Official RUL alignment: test engine IDs are ordered 1..100 for the official RUL vector
- **PASS** Official test window availability: test sequence length range=31..303; all have at least 30 cycles
- **PASS** Saved PCA scaler: 21 sensor columns and 16561 training rows; parameters match reproducible training-only fit=True
- **PASS** Saved PCA transformer: input_features=21, components=10, fit_rows=16561; parameters match reproducible training-only fit=True
- **PASS** Saved PCA Ridge inputs/predictions: input compatibility=True, predictions=100, finite=True
- **PASS** Saved PCA Ridge reported metrics: recomputed from saved-model predictions and official raw RUL: exact match
- **PASS** Saved PCA SimpleRNN inputs/predictions: input compatibility=True, predictions=100, finite=True
- **PASS** Saved PCA SimpleRNN reported metrics: recomputed from saved-model predictions and official raw RUL: matches within numerical inference tolerance
- **PASS** Saved PCA LSTM inputs/predictions: input compatibility=True, predictions=100, finite=True
- **PASS** Saved PCA LSTM reported metrics: recomputed from saved-model predictions and official raw RUL: matches within numerical inference tolerance
- **PASS** Saved original 14-sensor LSTM/scaler: model input=(30, 14), scaler fit rows=16561, training-only scaler match=True, predictions=100
- **PASS** Historical baseline result interpretation: saved seed-42 model metrics differ from results.csv's recorded three-seed mean prediction (same metrics=False); saved model RMSE/MAE/NASA=14.7069/10.9609/357.6348

## Code-level preprocessing and leakage review

The original LSTM notebook uses 14 selected sensors, 30-cycle windows, GroupShuffleSplit with seed 42, training-only MinMax scaling, and capped endpoint RUL labels. The saved baseline scaler's 14 feature names, training-row count, and extrema match a fresh fit on the same training-engine rows.

The separate `01_data_exploration.ipynb` fits its MinMaxScaler on all training engines before its later engine split. That exposes validation-engine sensor ranges to preprocessing; its Ridge scores are exploratory and were excluded from the controlled comparison. The baseline notebook and its artifacts were left intact.

The PCA script reads the existing labeled CSV, supplies only `sensor_1` through `sensor_21` to StandardScaler/PCA, uses `PCA(n_components=0.95)`, fits both transforms on training-engine rows, and transforms validation/test data without refitting. The saved scaler and PCA parameters match a reproducible fit on the 16,561 training-engine rows.

Training windows are built within each engine in cycle order; each label comes from that window's last cycle. The test pipeline uses exactly the final 30 observed rows per engine and aligns sorted IDs 1..100 with `RUL_FD001.txt`. Official RUL labels enter only metric calculation, never training, scaling, PCA, stopping, or model selection.

The raw dataset contains negative values in operating settings 1 and 2; these are expected operating-condition values, not invalid negative sensor readings. The 21 sensor columns are finite and nonnegative.

Training labels were independently computed as `min(max_cycle_for_engine - cycle, 125)`. Official test labels were read directly from `RUL_FD001.txt` and aligned to sorted engine IDs 1..100; they were not derived using the training-label formula.

## Recomputed saved-artifact metrics

PCA model predictions were recreated from saved scaler/PCA/model artifacts and compared with the current `pca_model_comparison.csv` using a 1e-6 relative tolerance. Small floating-point differences arise from Keras inference batching. Predictions were not saved by that original PCA run, so this is artifact-based reinference, not comparison against stored prediction arrays.

| Saved model | RMSE | MAE | NASA score |
| --- | ---: | ---: | ---: |
| PCA Ridge | 21.709157 | 17.444019 | 1079.858901 |
| PCA SimpleRNN | 20.116518 | 15.052484 | 1141.145388 |
| PCA LSTM | 15.996947 | 11.675745 | 670.795667 |
| Original saved 14-sensor LSTM | 14.706885 | 10.960888 | 357.634825 |

NASA score definition: let `d = predicted_RUL - actual_RUL`; penalty is `exp(-d/13)-1` for `d < 0` and `exp(d/10)-1` otherwise. The total is the sum over official test engines.

## Findings and corrections

No raw-to-labeled row, sensor, ordering, or RUL mismatch was found; no derived data correction was needed. Existing baseline notebooks, models, scalers, inference code, and raw files were preserved.

A confirmed scaler leakage in the separate exploration notebook (all 100 training engines scaled before splitting) is documented rather than edited; the new comparison fits each scaler only on the training-engine partition. The original LSTM notebook's training-only fit was verified against its saved scaler.

The historical baseline `results.csv` contains the mean prediction of three LSTM seeds, while `models/lstm_rul_fd001.keras` is the seed-42 model. The saved model reproduces the README's rounded single-model raw-test metrics, not the ensemble row. Those records must not be treated as identical runs. The new comparison retrains all five models under one explicit protocol and records predictions.

The first generated fair-comparison prediction table was rewritten from saved models using 17-significant-digit float serialization; all 11 recorded metric rows then reproduced from that table to 1e-10 absolute tolerance.

Validation: the full `.venv` pytest suite passed (6 tests), including three new window-label and NASA-score tests. All nine new neural artifacts also accepted their expected input shapes and returned 100 finite predictions.

See `fd001_fair_comparison.md` and `fd001_fair_comparison.csv` for the controlled rerun.
