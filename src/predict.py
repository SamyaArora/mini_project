from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "lstm_rul_fd001.keras"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler_lstm_fd001.pkl"

WINDOW_SHAPE = (30, 14)
SENSOR_COLUMNS = (
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21",
)

model = keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

if getattr(scaler, "n_features_in_", None) != len(SENSOR_COLUMNS):
    raise ValueError(
        f"scaler must be fitted on {len(SENSOR_COLUMNS)} sensor features"
    )

scaler_columns = getattr(scaler, "feature_names_in_", None)
if scaler_columns is not None and tuple(scaler_columns) != SENSOR_COLUMNS:
    raise ValueError("scaler sensor feature order does not match the LSTM input")


def predict_rul(window: np.ndarray) -> float:
    """Predict remaining useful life for one 30-cycle raw sensor window."""
    try:
        values = np.asarray(window, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("window must contain numeric sensor values") from exc

    if values.shape != WINDOW_SHAPE:
        raise ValueError(f"window must have shape {WINDOW_SHAPE}; got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("window must contain only finite sensor values")

    scaled = scaler.transform(
        pd.DataFrame(values, columns=SENSOR_COLUMNS)
    ).reshape((1, *WINDOW_SHAPE))
    prediction = np.asarray(model.predict(scaled, verbose=0))
    if prediction.size != 1:
        raise RuntimeError(
            f"model must return one RUL prediction; got shape {prediction.shape}"
        )

    rul = float(prediction.reshape(-1)[0])
    if not np.isfinite(rul):
        raise RuntimeError("model returned a non-finite RUL prediction")
    return rul
