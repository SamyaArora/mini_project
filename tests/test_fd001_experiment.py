import numpy as np
import pandas as pd

from scripts.audit_compare_fd001 import (
    WINDOW,
    make_final_test_windows,
    make_training_windows,
    nasa_score,
)


def test_nasa_score_uses_asymmetric_penalties() -> None:
    actual = np.array([10.0, 10.0])
    predicted = np.array([9.0, 11.0])

    expected = np.expm1(1.0 / 13.0) + np.expm1(1.0 / 10.0)

    assert np.isclose(nasa_score(actual, predicted), expected)


def test_training_windows_keep_engine_and_final_cycle_label_aligned() -> None:
    rows = []
    values = []
    for unit_id in (1, 2):
        for cycle in range(1, WINDOW + 2):
            rows.append(
                {
                    "unit_id": unit_id,
                    "cycle": cycle,
                    "RUL": WINDOW + 1 - cycle,
                }
            )
            values.append([unit_id * 100 + cycle])
    frame = pd.DataFrame(rows)

    windows, labels, engine_ids = make_training_windows(
        frame,
        np.asarray(values, dtype=np.float32),
    )

    assert windows.shape == (4, WINDOW, 1)
    assert labels.tolist() == [1.0, 0.0, 1.0, 0.0]
    assert engine_ids.tolist() == [1, 1, 2, 2]
    assert np.all(windows[:2, :, 0] // 100 == 1)
    assert np.all(windows[2:, :, 0] // 100 == 2)
    assert windows[0, -1, 0] == 130
    assert windows[1, -1, 0] == 131


def test_final_test_window_is_one_ordered_chronological_window_per_engine() -> None:
    rows = []
    values = []
    for unit_id in (2, 1):
        for cycle in range(1, WINDOW + 2):
            rows.append({"unit_id": unit_id, "cycle": cycle})
            values.append([unit_id * 100 + cycle])
    frame = pd.DataFrame(rows)

    windows, engine_ids = make_final_test_windows(
        frame,
        np.asarray(values, dtype=np.float32),
    )

    assert windows.shape == (2, WINDOW, 1)
    assert engine_ids.tolist() == [1, 2]
    assert windows[0, 0, 0] == 102
    assert windows[0, -1, 0] == 131
    assert windows[1, 0, 0] == 202
    assert windows[1, -1, 0] == 231
