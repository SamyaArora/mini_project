"""Generate train, validation, and official-test RUL comparisons from saved models."""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

if __package__:
    from . import audit_compare_fd001 as pipeline
else:
    import audit_compare_fd001 as pipeline


PREDICTION_PATHS = {
    split: pipeline.REPORTS_DIR / f"fd001_rul_predictions_{split}.csv"
    for split in ("train", "validation", "test")
}
METRICS_PATH = pipeline.REPORTS_DIR / "fd001_rul_split_metrics.csv"
FIGURES_DIR = pipeline.REPORTS_DIR / "figures"
FIGURE_PATHS = {
    split: FIGURES_DIR / f"fd001_rul_actual_vs_predicted_{split}.png"
    for split in ("train", "validation", "test")
}

PCA_MODEL_SEEDS = pipeline.MODEL_SEEDS
MODEL_NAMES = (
    "Original 14-sensor Ridge (retrained)",
    "PCA Ridge",
    "Original 14-sensor LSTM",
    "PCA SimpleRNN",
    "PCA LSTM",
)


def _target_cycles(frame: pd.DataFrame) -> np.ndarray:
    cycles = []
    for _, engine in frame.groupby("unit_id", sort=True):
        engine_cycles = engine["cycle"].to_numpy()
        cycles.extend(
            engine_cycles[pipeline.WINDOW - 1 :]
        )
    return np.asarray(cycles)


def _prediction_rows(
    split: str,
    model: str,
    run_seed: str,
    feature_representation: str,
    engine_ids: np.ndarray,
    target_cycles: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> pd.DataFrame:
    engine_ids = np.asarray(engine_ids).reshape(-1)
    target_cycles = np.asarray(target_cycles).reshape(-1)
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    arrays = (engine_ids, target_cycles, actual, predicted)
    if any(len(values) != len(engine_ids) for values in arrays):
        raise ValueError(f"{split} {model} prediction arrays have mismatched lengths")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError(f"{split} {model} predictions contain non-finite values")
    return pd.DataFrame(
        {
            "split": split,
            "model": model,
            "feature_representation": feature_representation,
            "run_seed": run_seed,
            "unit_id": engine_ids,
            "target_cycle": target_cycles,
            "actual_RUL": actual,
            "predicted_RUL": predicted,
            "error": predicted - actual,
        }
    )


def _write_scatter_plot(
    split: str,
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(16, 10), squeeze=False)
    axes_flat = axes.ravel()
    colors = ("tab:blue", "tab:orange", "tab:green")
    lower = min(0.0, float(predictions["actual_RUL"].min()))
    upper = float(
        max(
            predictions["actual_RUL"].max(),
            predictions["predicted_RUL"].max(),
        )
    )
    padding = max(3.0, (upper - lower) * 0.04)
    lower -= padding
    upper += padding

    for axis, model_name in zip(axes_flat, MODEL_NAMES):
        model_rows = predictions[predictions["model"] == model_name]
        for color, (seed, run_rows) in zip(
            colors,
            model_rows.groupby("run_seed", sort=True),
        ):
            axis.scatter(
                run_rows["actual_RUL"],
                run_rows["predicted_RUL"],
                s=10 if split == "train" else 18,
                alpha=0.25 if split == "train" else 0.55,
                color=color,
                label=f"seed {seed}" if seed != "deterministic" else "deterministic",
                edgecolors="none",
            )
        axis.plot([lower, upper], [lower, upper], "k--", linewidth=1)
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.set_title(model_name)
        axis.set_xlabel("Actual RUL at target cycle")
        axis.set_ylabel("Predicted RUL")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)

    axes_flat[-1].set_visible(False)
    split_label = "Training (in-sample; not evidence of generalization)" if split == "train" else split.title()
    figure.suptitle(f"FD001 Actual vs Predicted RUL — {split_label}", fontsize=15)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _assert_output_paths_are_new() -> None:
    paths = [*PREDICTION_PATHS.values(), METRICS_PATH, *FIGURE_PATHS.values()]
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing RUL comparison outputs:\n"
            + "\n".join(f"  {path}" for path in existing)
        )


def run() -> None:
    _assert_output_paths_are_new()
    raw_train, raw_test, official_rul, labeled = pipeline._read_inputs()
    data_checks, _ = pipeline._audit_data(
        raw_train,
        raw_test,
        official_rul,
        labeled,
    )
    failed_checks = [check for check in data_checks if "**FAIL**" in check]
    if failed_checks:
        raise ValueError(
            "Input audit failed; no comparisons were written:\n"
            + "\n".join(failed_checks)
        )
    if labeled["RUL"].isna().any():
        raise ValueError("Training labels contain missing RUL values")

    train_frame = labeled.sort_values(
        ["unit_id", "cycle"], kind="stable"
    ).reset_index(drop=True)
    test_frame = raw_test.sort_values(
        ["unit_id", "cycle"], kind="stable"
    ).reset_index(drop=True)
    engine_ids = np.sort(train_frame["unit_id"].unique())
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=pipeline.SPLIT_SEED,
    )
    train_positions, validation_positions = next(
        splitter.split(engine_ids, groups=engine_ids)
    )
    train_engine_ids = engine_ids[train_positions]
    validation_engine_ids = engine_ids[validation_positions]
    if set(train_engine_ids).intersection(validation_engine_ids):
        raise AssertionError("Training and validation engine IDs overlap")

    training_rows = train_frame["unit_id"].isin(train_engine_ids)
    pca_sensor_columns = [
        column
        for column in pipeline.SENSOR_COLUMNS
        if train_frame.loc[training_rows, column].nunique(dropna=False) > 1
    ]
    if len(pca_sensor_columns) != 15:
        raise AssertionError(
            f"Expected 15 non-constant training sensors; found {len(pca_sensor_columns)}"
        )

    pca_scaler = StandardScaler().fit(
        train_frame.loc[training_rows, pca_sensor_columns]
    )
    pca = PCA(n_components=10, svd_solver="full").fit(
        pca_scaler.transform(train_frame.loc[training_rows, pca_sensor_columns])
    )
    if pca.n_samples_ != int(training_rows.sum()):
        raise AssertionError("PCA was not fit exclusively on training-engine rows")

    all_train_15 = pca.transform(
        pca_scaler.transform(train_frame[pca_sensor_columns])
    )
    all_test_15 = pca.transform(
        pca_scaler.transform(test_frame[pca_sensor_columns])
    )
    saved_scaler_21 = joblib.load(
        pipeline.MODELS_DIR / "scaler_21sensor_fair_fd001.pkl"
    )
    saved_pca_21 = joblib.load(pipeline.MODELS_DIR / "pca_fair_fd001.pkl")
    if saved_pca_21.n_components_ != 10 or saved_scaler_21.n_features_in_ != 21:
        raise AssertionError("Saved PCA models do not match the expected 21-to-10 input")
    all_train_21 = saved_pca_21.transform(
        saved_scaler_21.transform(train_frame[pipeline.SENSOR_COLUMNS])
    )
    all_test_21 = saved_pca_21.transform(
        saved_scaler_21.transform(test_frame[pipeline.SENSOR_COLUMNS])
    )
    if not np.allclose(all_train_15, all_train_21, rtol=1e-8, atol=1e-8):
        raise AssertionError(
            "The 15-sensor PCA transform is not compatible with the saved PCA models"
        )
    if not np.allclose(all_test_15, all_test_21, rtol=1e-8, atol=1e-8):
        raise AssertionError(
            "The 15-sensor PCA test transform differs from the saved model input"
        )

    baseline_scaler = joblib.load(
        pipeline.MODELS_DIR / "scaler_14sensor_fair_fd001.pkl"
    )
    baseline_train = baseline_scaler.transform(
        train_frame[pipeline.BASELINE_SENSORS]
    ).astype(np.float32)
    baseline_test = baseline_scaler.transform(
        test_frame[pipeline.BASELINE_SENSORS]
    ).astype(np.float32)
    pca_train = all_train_15.astype(np.float32)
    pca_test = all_test_15.astype(np.float32)

    baseline_windows, labels, window_engine_ids = pipeline.make_training_windows(
        train_frame,
        baseline_train,
    )
    pca_windows, pca_labels, pca_window_engine_ids = pipeline.make_training_windows(
        train_frame,
        pca_train,
    )
    if not np.array_equal(labels, pca_labels) or not np.array_equal(
        window_engine_ids,
        pca_window_engine_ids,
    ):
        raise AssertionError("The original and PCA windows have different RUL labels")
    window_cycles = _target_cycles(train_frame)
    if len(window_cycles) != len(labels):
        raise AssertionError("Window target-cycle labels are misaligned")
    if not np.isfinite(labels).all():
        raise ValueError("Window RUL labels contain missing or non-finite values")
    train_mask = np.isin(window_engine_ids, train_engine_ids)
    validation_mask = np.isin(window_engine_ids, validation_engine_ids)
    if not train_mask.any() or not validation_mask.any():
        raise AssertionError("Training or validation split has no windows")
    if set(window_engine_ids[train_mask]) != set(train_engine_ids):
        raise AssertionError("Some training engines have no 30-cycle windows")
    if set(window_engine_ids[validation_mask]) != set(validation_engine_ids):
        raise AssertionError("Some validation engines have no 30-cycle windows")
    window_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=pipeline.SPLIT_SEED,
    )
    train_window_indices, validation_window_indices = next(
        window_splitter.split(
            baseline_windows,
            labels,
            groups=window_engine_ids,
        )
    )
    if (
        set(window_engine_ids[train_window_indices]) != set(train_engine_ids)
        or set(window_engine_ids[validation_window_indices])
        != set(validation_engine_ids)
    ):
        raise AssertionError(
            "The explicit engine split differs from the existing window-wise group split"
        )

    test_engine_ids = np.sort(test_frame["unit_id"].unique())
    if not np.array_equal(test_engine_ids, np.arange(1, 101)):
        raise AssertionError("Official test engine IDs do not match RUL file order 1..100")
    if len(official_rul) != len(test_engine_ids):
        raise AssertionError("Official RUL label count does not match test engines")
    test_windows_14, aligned_test_ids = pipeline.make_final_test_windows(
        test_frame,
        baseline_test,
    )
    test_windows_pca, pca_test_ids = pipeline.make_final_test_windows(
        test_frame,
        pca_test,
    )
    if not np.array_equal(aligned_test_ids, test_engine_ids) or not np.array_equal(
        pca_test_ids,
        test_engine_ids,
    ):
        raise AssertionError("Test windows do not align one-to-one with official RUL")
    test_cycles = (
        test_frame.groupby("unit_id", sort=True)["cycle"].max().to_numpy()
    )

    model_specs = [
        (
            "Original 14-sensor Ridge (retrained)",
            "14 selected sensors + MinMaxScaler",
            "deterministic",
            joblib.load(pipeline.MODELS_DIR / "ridge_14sensor_fair_fd001.pkl"),
            baseline_windows,
            test_windows_14,
            False,
        ),
        (
            "PCA Ridge",
            "15 non-constant sensors -> StandardScaler -> PCA(10)",
            "deterministic",
            joblib.load(pipeline.MODELS_DIR / "ridge_pca_fair_fd001.pkl"),
            pca_windows,
            test_windows_pca,
            False,
        ),
    ]
    for model_name, model_prefix, feature_rows, test_rows in (
        (
            "Original 14-sensor LSTM",
            "lstm_14sensor_fair_fd001",
            baseline_windows,
            test_windows_14,
        ),
        (
            "PCA SimpleRNN",
            "simple_rnn_pca_fair_fd001",
            pca_windows,
            test_windows_pca,
        ),
        (
            "PCA LSTM",
            "lstm_pca_fair_fd001",
            pca_windows,
            test_windows_pca,
        ),
    ):
        representation = (
            "14 selected sensors + MinMaxScaler"
            if model_name == "Original 14-sensor LSTM"
            else "15 non-constant sensors -> StandardScaler -> PCA(10)"
        )
        for seed in PCA_MODEL_SEEDS:
            model_specs.append(
                (
                    model_name,
                    representation,
                    str(seed),
                    tf.keras.models.load_model(
                        pipeline.MODELS_DIR / f"{model_prefix}_seed{seed}.keras",
                        compile=False,
                    ),
                    feature_rows,
                    test_rows,
                    True,
                )
            )

    actual_by_split = {
        "train": labels[train_mask],
        "validation": labels[validation_mask],
        "test": official_rul,
    }
    engine_ids_by_split = {
        "train": window_engine_ids[train_mask],
        "validation": window_engine_ids[validation_mask],
        "test": aligned_test_ids,
    }
    cycles_by_split = {
        "train": window_cycles[train_mask],
        "validation": window_cycles[validation_mask],
        "test": test_cycles,
    }
    masks = {
        "train": train_mask,
        "validation": validation_mask,
        "test": None,
    }
    prediction_frames = {split: [] for split in PREDICTION_PATHS}
    metric_rows = []
    for (
        model_name,
        representation,
        run_seed,
        model,
        training_windows,
        test_windows,
        is_neural,
    ) in model_specs:
        for split in ("train", "validation", "test"):
            if split == "test":
                X = test_windows
                predicted = (
                    model.predict(X, verbose=0).reshape(-1)
                    if is_neural
                    else model.predict(X[:, -1, :])
                )
            else:
                mask = masks[split]
                X = training_windows[mask]
                predicted = (
                    model.predict(X, verbose=0).reshape(-1)
                    if is_neural
                    else model.predict(X[:, -1, :])
                )
            actual = actual_by_split[split]
            if len(predicted) != len(actual):
                raise AssertionError(
                    f"{split} {model_name} expected {len(actual)} predictions; "
                    f"received {len(predicted)}"
                )
            prediction_frames[split].append(
                _prediction_rows(
                    split,
                    model_name,
                    run_seed,
                    representation,
                    engine_ids_by_split[split],
                    cycles_by_split[split],
                    actual,
                    predicted,
                )
            )
            metric_rows.append(
                {
                    "split": split,
                    "model": model_name,
                    "feature_representation": representation,
                    "run_seed": run_seed,
                    "n": len(actual),
                    **pipeline.calculate_metrics(actual, predicted),
                }
            )

    combined_predictions = {
        split: pd.concat(frames, ignore_index=True)
        for split, frames in prediction_frames.items()
    }
    for split, frame in combined_predictions.items():
        if split == "test":
            expected_pairs = len(test_engine_ids) * len(model_specs)
            if len(frame) != expected_pairs:
                raise AssertionError(
                    f"Official test table has {len(frame)} rows; expected {expected_pairs}"
                )
            for _, model_rows in frame.groupby(["model", "run_seed"]):
                if not np.array_equal(model_rows["unit_id"].to_numpy(), test_engine_ids):
                    raise AssertionError(
                        "A model's official-test rows do not contain each engine once in order"
                    )

    pipeline.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metric_rows)
    for split, frame in combined_predictions.items():
        frame.to_csv(PREDICTION_PATHS[split], index=False, mode="x")
        _write_scatter_plot(split, frame, FIGURE_PATHS[split])
    metrics.to_csv(METRICS_PATH, index=False, mode="x")

    print(
        f"Training/validation engines: {len(train_engine_ids)}/"
        f"{len(validation_engine_ids)}; overlap: "
        f"{len(set(train_engine_ids) & set(validation_engine_ids))}"
    )
    print(f"PCA features: {len(pca_sensor_columns)} -> {pca.n_components_} components")
    print(f"Maximum 15-vs-saved-21 PCA transform difference: "
          f"{max(np.max(np.abs(all_train_15-all_train_21)), np.max(np.abs(all_test_15-all_test_21))):.3e}")
    print(f"Window counts: train={int(train_mask.sum())}, "
          f"validation={int(validation_mask.sum())}, test={len(test_engine_ids)}")
    print("Metrics (training metrics are in-sample, not evidence of generalization):")
    print(metrics.to_string(index=False))
    for split in PREDICTION_PATHS:
        print(f"Predictions: {PREDICTION_PATHS[split]}")
        print(f"Plot: {FIGURE_PATHS[split]}")
    print(f"Metrics: {METRICS_PATH}")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
