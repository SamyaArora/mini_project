import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "processed"
LABELED_PATH = OUTPUT_DIR / "train_FD001_labeled.txt"
SCHEMA_PATH = OUTPUT_DIR / "train_FD001_labeled_schema.json"

EXPECTED_COLUMNS = [
    "unit_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    *[f"sensor_{number}" for number in range(1, 22)],
    "RUL",
]


def report(label: str, actual: object, expected: object, passed: bool) -> bool:
    result = "PASS" if passed else "FAIL"
    print(f"{label}: {actual} (expected {expected}) - {result}")
    return passed


def main() -> None:
    with SCHEMA_PATH.open("r", encoding="utf-8") as schema_file:
        schema = json.load(schema_file)

    schema_columns = schema["columns"]
    schema_names = [column["name"] for column in schema_columns]
    labeled = pd.read_csv(
        LABELED_PATH,
        sep=r"\s+",
        header=None,
        names=schema_names,
    )

    rul = labeled["RUL"]
    maximum_cycles = labeled.groupby("unit_id")["cycle"].transform("max")
    maximum_cycle_rows = labeled["cycle"].eq(maximum_cycles)
    every_engine_max_cycle_has_zero_rul = (
        labeled.loc[maximum_cycle_rows, "RUL"].eq(0).all()
    )
    rul_values_valid = (
        rul.notna().all()
        and rul.eq(rul.round()).all()
        and rul.between(0, 125).all()
    )
    nan_count = int(labeled.isna().sum().sum())
    checks = [
        report("Rows", len(labeled), 20631, len(labeled) == 20631),
        report("Columns", labeled.shape[1], 27, labeled.shape[1] == 27),
        report(
            "RUL min/max",
            f"{rul.min()}/{rul.max()}",
            "0/125",
            rul.min() == 0 and rul.max() == 125,
        ),
        report(
            "RUL values are integers in [0, 125]",
            bool(rul_values_valid),
            True,
            bool(rul_values_valid),
        ),
        report(
            "Unique engines",
            labeled["unit_id"].nunique(),
            100,
            labeled["unit_id"].nunique() == 100,
        ),
        report("NaN count", nan_count, 0, nan_count == 0),
        report(
            "Every engine's maximum-cycle row has RUL == 0",
            bool(every_engine_max_cycle_has_zero_rul),
            True,
            bool(every_engine_max_cycle_has_zero_rul),
        ),
        report(
            "Schema column entries",
            len(schema_columns),
            27,
            len(schema_columns) == 27,
        ),
        report(
            "Schema column names match data columns",
            schema_names == EXPECTED_COLUMNS and list(labeled.columns) == schema_names,
            True,
            schema_names == EXPECTED_COLUMNS and list(labeled.columns) == schema_names,
        ),
        report(
            "Schema dimensions match data",
            (
                f"n_rows={schema['n_rows']} vs rows={len(labeled)}, "
                f"n_columns={schema['n_columns']} vs columns={labeled.shape[1]}"
            ),
            "both schema dimensions equal data dimensions",
            schema["n_rows"] == len(labeled)
            and schema["n_columns"] == labeled.shape[1],
        ),
    ]

    if not all(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
