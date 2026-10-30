"""Compare Ridge, SimpleRNN, and LSTM using PCA-transformed FD001 sensors."""

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers


ROOT = Path(__file__).resolve().parents[1]
LABELED_PATH = ROOT / "data" / "processed" / "train_FD001_labeled.csv"
TEST_PATH = ROOT / "data" / "raw" / "test_FD001.txt"
TEST_RUL_PATH = ROOT / "data" / "raw" / "RUL_FD001.txt"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

WINDOW = 30
RUL_CAP = 125
PCA_VARIANCE = 0.95
SPLIT_SEED = 42
RNN_SEED = 42
LSTM_SEED = 43
MAX_EPOCHS = 50
BATCH_SIZE = 128

SENSOR_COLUMNS = [f"sensor_{number}" for number in range(1, 22)]
RAW_COLUMNS = [
    "unit_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    *SENSOR_COLUMNS,
]

SCALER_PATH = MODELS_DIR / "scaler_pca_fd001.pkl"
PCA_PATH = MODELS_DIR / "pca_fd001.pkl"
RIDGE_PATH = MODELS_DIR / "ridge_pca_fd001.pkl"
RNN_PATH = MODELS_DIR / "simple_rnn_pca_fd001.keras"
LSTM_PATH = MODELS_DIR / "lstm_pca_fd001.keras"
VARIANCE_PLOT_PATH = FIGURES_DIR / "pca_cumulative_explained_variance.png"
RESULTS_CSV_PATH = REPORTS_DIR / "pca_model_comparison.csv"
RESULTS_MD_PATH = REPORTS_DIR / "pca_model_comparison.md"


def check_output_paths_are_new(overwrite: bool) -> None:
    output_paths = [
        SCALER_PATH,
        PCA_PATH,
        RIDGE_PATH,
        RNN_PATH,
        LSTM_PATH,
        VARIANCE_PLOT_PATH,
        RESULTS_CSV_PATH,
        RESULTS_MD_PATH,
    ]
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing PCA experiment outputs. "
            "Pass --overwrite to replace only these PCA experiment files:\n"
            + formatted
        )


def check_finite(frame: pd.DataFrame, description: str) -> None:
    if frame.isna().any().any():
        raise ValueError(f"{description} contains missing values")
    if not np.isfinite(frame.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{description} contains non-finite values")


def make_training_windows(
    frame: pd.DataFrame,
    transformed_sensors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create one labeled window for each available 30-cycle training endpoint."""
    windows = []
    targets = []
    window_units = []
    target_values = frame["RUL"].to_numpy(dtype=np.float32)

    for unit_id, row_indices in frame.groupby("unit_id", sort=True).indices.items():
        unit_sensors = transformed_sensors[row_indices]
        unit_targets = target_values[row_indices]
        if len(row_indices) < WINDOW:
            continue
        for end in range(WINDOW, len(row_indices) + 1):
            windows.append(unit_sensors[end - WINDOW : end])
            targets.append(unit_targets[end - 1])
            window_units.append(unit_id)

    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(window_units),
    )


def make_final_test_windows(
    frame: pd.DataFrame,
    transformed_sensors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Make exactly one final chronological 30-cycle window per test engine."""
    windows = []
    unit_ids = []

    for unit_id, row_indices in frame.groupby("unit_id", sort=True).indices.items():
        unit_sensors = transformed_sensors[row_indices]
        if len(unit_sensors) < WINDOW:
            padding = np.repeat(
                unit_sensors[:1],
                WINDOW - len(unit_sensors),
                axis=0,
            )
            unit_sensors = np.concatenate((padding, unit_sensors), axis=0)
        windows.append(unit_sensors[-WINDOW:])
        unit_ids.append(unit_id)

    return np.asarray(windows, dtype=np.float32), np.asarray(unit_ids)


def nasa_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Return the NASA asymmetric score for d = prediction - actual."""
    difference = np.asarray(predicted, dtype=np.float64) - np.asarray(
        actual, dtype=np.float64
    )
    penalties = np.where(
        difference < 0,
        np.expm1(-difference / 13.0),
        np.expm1(difference / 10.0),
    )
    return float(np.sum(penalties))


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.float64).ravel()
    predicted = np.asarray(predicted, dtype=np.float64).ravel()
    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "nasa_score": nasa_score(actual, predicted),
    }


def build_recurrent_model(
    model_type: str,
    input_shape: tuple[int, int],
) -> keras.Model:
    """Build a compact SimpleRNN or LSTM regressor for a transformed window."""
    inputs = keras.Input(shape=input_shape, name="pca_window")
    if model_type == "SimpleRNN":
        sequence = layers.SimpleRNN(32, return_sequences=True)(inputs)
        sequence = layers.Dropout(0.2)(sequence)
        sequence = layers.SimpleRNN(16)(sequence)
    elif model_type == "LSTM":
        sequence = layers.LSTM(32, return_sequences=True)(inputs)
        sequence = layers.Dropout(0.2)(sequence)
        sequence = layers.LSTM(16)(sequence)
    else:
        raise ValueError(f"Unknown recurrent model: {model_type}")

    hidden = layers.Dense(16, activation="relu")(sequence)
    outputs = layers.Dense(1, name="rul")(hidden)
    model = keras.Model(inputs=inputs, outputs=outputs, name=model_type.lower())
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
    )
    return model


def train_recurrent_model(
    model_type: str,
    seed: int,
    input_shape: tuple[int, int],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
) -> keras.Model:
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = build_recurrent_model(model_type, input_shape)
    if model.input_shape[1:] != input_shape:
        raise ValueError(
            f"{model_type} input shape {model.input_shape[1:]} "
            f"does not match {input_shape}"
        )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
    )
    print(
        f"Training {model_type} (seed={seed}, max_epochs={MAX_EPOCHS}, "
        f"batch_size={BATCH_SIZE})..."
    )
    model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=2,
    )
    return model


def save_variance_plot(pca: PCA) -> None:
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    component_numbers = np.arange(1, len(cumulative_variance) + 1)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(component_numbers, cumulative_variance, marker="o")
    axis.axhline(PCA_VARIANCE, color="red", linestyle="--", label="95% target")
    axis.axvline(pca.n_components_, color="gray", linestyle=":")
    axis.set(
        xlabel="Number of principal components",
        ylabel="Cumulative explained variance",
        title="FD001 PCA cumulative explained variance (training engines only)",
        ylim=(0, 1.02),
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(VARIANCE_PLOT_PATH, dpi=150)
    plt.close(figure)


def write_results(
    results: list[dict[str, object]],
    n_components: int,
    overwrite: bool,
) -> None:
    results_frame = pd.DataFrame(results)
    if overwrite:
        results_frame.to_csv(RESULTS_CSV_PATH, index=False)
    else:
        with RESULTS_CSV_PATH.open("x", encoding="utf-8", newline="") as csv_file:
            results_frame.to_csv(csv_file, index=False)

    markdown_lines = [
        "# FD001 PCA Model Comparison",
        "",
        "PCA is dimensionality reduction: each component is a linear combination "
        "of the 21 sensor measurements, not a selected original sensor.",
        "",
        "The scaler and PCA were fit only on training-engine sensor rows. "
        "Validation and official test rows were transformed with those fitted objects.",
        "",
        f"- PCA components retained: {n_components} "
        f"(cumulative explained variance: "
        f"{results[0]['cumulative_explained_variance']:.6f})",
        f"- Input sensor measurements: {len(SENSOR_COLUMNS)}",
        f"- Window length: {WINDOW} cycles",
        "- Training target: labeled RUL capped at 125 cycles",
        "- Evaluation target: official, uncapped FD001 test RUL",
        "- Official test evaluation: one final window for each of 100 engines",
        "- Ridge: alpha=1.0, using the final timestep of each window",
        f"- SimpleRNN: seed={RNN_SEED}; LSTM: seed={LSTM_SEED}",
        "- Recurrent models: Adam (learning rate 0.001), batch size 128, "
        "up to 50 epochs, EarlyStopping on validation loss (patience 7, "
        "restore best weights)",
        "",
        "NASA score formula, with d = prediction - actual:",
        "",
        "- d < 0: exp(-d / 13) - 1",
        "- d >= 0: exp(d / 10) - 1",
        "- The reported score is the sum of per-engine penalties; lower is better.",
        "",
        "| Model | Feature representation | PCA components | Window | RMSE | MAE | NASA score |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        markdown_lines.append(
            f"| {row['model']} | {row['feature_representation']} | "
            f"{row['pca_components']} | {row['window_length']} | "
            f"{row['rmse']:.4f} | {row['mae']:.4f} | "
            f"{row['nasa_score']:.4f} |"
        )
    markdown_lines.extend(
        [
            "",
            "These results are from the PCA experiment and are not directly "
            "interchangeable with the existing 14-sensor baseline.",
            "",
            f"Explained-variance plot: `figures/{VARIANCE_PLOT_PATH.name}`",
        ]
    )
    mode = "w" if overwrite else "x"
    with RESULTS_MD_PATH.open(mode, encoding="utf-8", newline="") as report_file:
        report_file.write("\n".join(markdown_lines) + "\n")


def main(overwrite: bool = False) -> None:
    check_output_paths_are_new(overwrite)
    for input_path in (LABELED_PATH, TEST_PATH, TEST_RUL_PATH):
        if not input_path.is_file():
            raise FileNotFoundError(f"Required input file not found: {input_path}")

    train_frame = pd.read_csv(LABELED_PATH)
    if train_frame.shape != (20631, 27):
        raise ValueError(
            f"Expected labeled training data shape (20631, 27), "
            f"got {train_frame.shape}"
        )
    expected_train_columns = [
        "unit_id",
        "cycle",
        "setting_1",
        "setting_2",
        "setting_3",
        *SENSOR_COLUMNS,
        "RUL",
    ]
    if list(train_frame.columns) != expected_train_columns:
        raise ValueError("Labeled training CSV columns do not match the expected schema")
    if train_frame["unit_id"].nunique() != 100:
        raise ValueError("Expected 100 training engines")

    train_frame = train_frame.sort_values(
        ["unit_id", "cycle"], kind="stable"
    ).reset_index(drop=True)
    check_finite(
        train_frame[SENSOR_COLUMNS + ["RUL"]],
        "Labeled training sensor and RUL values",
    )
    if not train_frame["RUL"].eq(train_frame["RUL"].round()).all():
        raise ValueError("Training RUL labels must be integers")
    if not train_frame["RUL"].between(0, RUL_CAP).all():
        raise ValueError(f"Training RUL labels must be between 0 and {RUL_CAP}")

    test_frame = pd.read_csv(
        TEST_PATH,
        sep=r"\s+",
        header=None,
        names=RAW_COLUMNS,
    ).sort_values(["unit_id", "cycle"], kind="stable").reset_index(drop=True)
    official_rul = pd.read_csv(
        TEST_RUL_PATH,
        sep=r"\s+",
        header=None,
        names=["RUL"],
    )["RUL"].to_numpy(dtype=np.float64)
    check_finite(test_frame[SENSOR_COLUMNS], "Official test sensor values")
    if len(official_rul) != 100 or not np.isfinite(official_rul).all():
        raise ValueError("Expected 100 finite official FD001 test RUL labels")

    test_engine_ids = np.sort(test_frame["unit_id"].unique())
    if not np.array_equal(test_engine_ids, np.arange(1, 101)):
        raise ValueError("Official FD001 test engines must have IDs 1 through 100")
    if len(test_engine_ids) != len(official_rul):
        raise ValueError("Official test engine count does not match RUL label count")

    engine_ids = np.sort(train_frame["unit_id"].unique())
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=SPLIT_SEED,
    )
    train_engine_positions, validation_engine_positions = next(
        splitter.split(engine_ids, groups=engine_ids)
    )
    train_engine_ids = engine_ids[train_engine_positions]
    validation_engine_ids = engine_ids[validation_engine_positions]
    if set(train_engine_ids).intersection(validation_engine_ids):
        raise AssertionError("Training and validation engine IDs overlap")

    training_rows = train_frame["unit_id"].isin(train_engine_ids)
    scaler = StandardScaler()
    scaler.fit(train_frame.loc[training_rows, SENSOR_COLUMNS])

    scaled_training_sensor_rows = scaler.transform(
        train_frame.loc[training_rows, SENSOR_COLUMNS]
    )
    pca = PCA(n_components=PCA_VARIANCE, svd_solver="full")
    pca.fit(scaled_training_sensor_rows)

    if pca.n_features_in_ != len(SENSOR_COLUMNS):
        raise AssertionError("PCA was not fitted on all 21 sensor features")
    if pca.n_samples_ != int(training_rows.sum()):
        raise AssertionError("PCA was not fitted exclusively on training rows")
    cumulative_explained_variance = float(pca.explained_variance_ratio_.sum())
    if cumulative_explained_variance < PCA_VARIANCE:
        raise AssertionError("PCA did not retain the required 95% variance")
    if scaler.n_samples_seen_ != int(training_rows.sum()):
        raise AssertionError("Scaler was not fitted exclusively on training rows")

    train_pca = pca.transform(
        scaler.transform(train_frame[SENSOR_COLUMNS])
    ).astype(np.float32)
    test_pca = pca.transform(
        scaler.transform(test_frame[SENSOR_COLUMNS])
    ).astype(np.float32)
    if train_pca.shape != (len(train_frame), pca.n_components_):
        raise AssertionError("Unexpected PCA-transformed training data shape")
    if test_pca.shape != (len(test_frame), pca.n_components_):
        raise AssertionError("Unexpected PCA-transformed test data shape")

    all_windows, all_targets, window_units = make_training_windows(
        train_frame,
        train_pca,
    )
    if all_windows.shape[1:] != (WINDOW, pca.n_components_):
        raise AssertionError("Unexpected PCA training-window tensor shape")
    if len(all_windows) == 0:
        raise ValueError("No complete training windows were created")
    training_windows = np.isin(window_units, train_engine_ids)
    validation_windows = np.isin(window_units, validation_engine_ids)
    X_train = all_windows[training_windows]
    y_train = all_targets[training_windows]
    X_validation = all_windows[validation_windows]
    y_validation = all_targets[validation_windows]
    if np.isin(window_units[training_windows], validation_engine_ids).any():
        raise AssertionError("A validation engine entered the training windows")
    if np.isin(window_units[validation_windows], train_engine_ids).any():
        raise AssertionError("A training engine entered the validation windows")

    X_test, ordered_test_ids = make_final_test_windows(test_frame, test_pca)
    if not np.array_equal(ordered_test_ids, test_engine_ids):
        raise AssertionError("Test windows are not aligned with official engine order")
    if X_test.shape != (100, WINDOW, pca.n_components_):
        raise AssertionError(f"Unexpected final test window shape: {X_test.shape}")

    input_shape = (WINDOW, pca.n_components_)
    if X_train[:, -1, :].shape[1:] != (pca.n_components_,):
        raise AssertionError("Ridge input does not match the PCA component count")
    for model_name in ("SimpleRNN", "LSTM"):
        shape_check_model = build_recurrent_model(model_name, input_shape)
        if shape_check_model.input_shape[1:] != input_shape:
            raise AssertionError(
                f"{model_name} does not accept the common input shape"
            )
        del shape_check_model
    tf.keras.backend.clear_session()

    print("Pre-training and shape checks passed:")
    print(f"  Train/validation engines: {len(train_engine_ids)}/{len(validation_engine_ids)}")
    print(f"  Engine overlap: {set(train_engine_ids).intersection(validation_engine_ids)}")
    print(f"  Scaler/PCA fitting rows: {int(training_rows.sum())} training-engine rows")
    print(f"  PCA input/output rows: {len(train_frame)} x 21 -> {train_pca.shape}")
    print(f"  Retained components: {pca.n_components_}")
    print(f"  Cumulative explained variance: {cumulative_explained_variance:.6f}")
    print(
        f"  Window tensors: train={X_train.shape}, "
        f"validation={X_validation.shape}, test={X_test.shape}"
    )
    print(f"  Shared recurrent input shape: {input_shape}")

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train[:, -1, :], y_train)

    rnn = train_recurrent_model(
        "SimpleRNN",
        RNN_SEED,
        input_shape,
        X_train,
        y_train,
        X_validation,
        y_validation,
    )
    lstm = train_recurrent_model(
        "LSTM",
        LSTM_SEED,
        input_shape,
        X_train,
        y_train,
        X_validation,
        y_validation,
    )
    for model_name, model in (("SimpleRNN", rnn), ("LSTM", lstm)):
        if model.input_shape[1:] != input_shape:
            raise AssertionError(f"{model_name} does not accept the common input shape")

    predictions = {
        "Ridge": ridge.predict(X_test[:, -1, :]).reshape(-1),
        "SimpleRNN": rnn.predict(X_test, verbose=0).reshape(-1),
        "LSTM": lstm.predict(X_test, verbose=0).reshape(-1),
    }
    results = []
    for model_name, prediction in predictions.items():
        if prediction.shape != (100,):
            raise AssertionError(
                f"{model_name} must predict exactly 100 test engines; "
                f"got {prediction.shape}"
            )
        if not np.isfinite(prediction).all():
            raise ValueError(f"{model_name} produced non-finite test predictions")
        results.append(
            {
                "model": model_name,
                "feature_representation": "PCA-transformed 21 sensors (95% variance)",
                "pca_components": int(pca.n_components_),
                "cumulative_explained_variance": cumulative_explained_variance,
                "window_length": WINDOW,
                **calculate_metrics(official_rul, prediction),
            }
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(pca, PCA_PATH)
    joblib.dump(ridge, RIDGE_PATH)
    rnn.save(RNN_PATH)
    lstm.save(LSTM_PATH)
    save_variance_plot(pca)
    write_results(results, pca.n_components_, overwrite)

    print(f"Saved scaler: {SCALER_PATH}")
    print(f"Saved PCA: {PCA_PATH}")
    print(f"Saved Ridge: {RIDGE_PATH}")
    print(f"Saved SimpleRNN: {RNN_PATH}")
    print(f"Saved LSTM: {LSTM_PATH}")
    print(f"Saved explained-variance plot: {VARIANCE_PLOT_PATH}")
    print(f"Saved results: {RESULTS_CSV_PATH} and {RESULTS_MD_PATH}")
    print("\nOfficial test metrics (one prediction per engine):")
    for result in results:
        print(
            f"  {result['model']}: RMSE={result['rmse']:.4f}, "
            f"MAE={result['mae']:.4f}, NASA={result['nasa_score']:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace only this experiment's PCA artifacts and reports",
    )
    arguments = parser.parse_args()
    main(overwrite=arguments.overwrite)
