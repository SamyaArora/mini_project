import numpy as np
import pytest

from src.predict import predict_rul


def test_predict_rul_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"window must have shape \(30, 14\)"):
        predict_rul(np.zeros((29, 14), dtype=np.float32))


def test_predict_rul_rejects_nan() -> None:
    window = np.zeros((30, 14), dtype=np.float32)
    window[0, 0] = np.nan

    with pytest.raises(ValueError, match="only finite sensor values"):
        predict_rul(window)


def test_predict_rul_returns_one_finite_number() -> None:
    prediction = predict_rul(np.ones((30, 14), dtype=np.float32))

    assert np.isscalar(prediction)
    assert np.isfinite(prediction)
