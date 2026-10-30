import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "raw" / "train_FD001.txt"
OUTPUT_DIR = ROOT / "data" / "processed"
LABELED_PATH = OUTPUT_DIR / "train_FD001_labeled.txt"
SCHEMA_PATH = OUTPUT_DIR / "train_FD001_labeled_schema.json"

RAW_COLUMNS = [
    "unit_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    *[f"sensor_{number}" for number in range(1, 22)],
]

COLUMN_DESCRIPTIONS = [
    ("unit_id", "engine unit identifier"),
    ("cycle", "operating cycle"),
    ("setting_1", "altitude"),
    ("setting_2", "Mach number"),
    ("setting_3", "throttle resolver angle"),
    ("sensor_1", "T2 total temperature at fan inlet"),
    ("sensor_2", "T24 total temperature at LPC outlet"),
    ("sensor_3", "T30 total temperature at HPC outlet"),
    ("sensor_4", "T50 total temperature at LPT outlet"),
    ("sensor_5", "P2 pressure at fan inlet"),
    ("sensor_6", "P15 total pressure in bypass duct"),
    ("sensor_7", "P30 total pressure at HPC outlet"),
    ("sensor_8", "Nf physical fan speed"),
    ("sensor_9", "Nc physical core speed"),
    ("sensor_10", "EPR engine pressure ratio (P50/P2)"),
    ("sensor_11", "Ps30 static pressure at HPC outlet"),
    ("sensor_12", "phi ratio of fuel flow to Ps30"),
    ("sensor_13", "NRf corrected fan speed"),
    ("sensor_14", "NRc corrected core speed"),
    ("sensor_15", "BPR bypass ratio"),
    ("sensor_16", "farB burner fuel-air ratio"),
    ("sensor_17", "htBleed bleed enthalpy"),
    ("sensor_18", "Nf_dmd demanded fan speed"),
    ("sensor_19", "PCNfR_dmd demanded corrected fan speed"),
    ("sensor_20", "W31 HPT coolant bleed"),
    ("sensor_21", "W32 LPT coolant bleed"),
    ("RUL", "remaining useful life in cycles, capped at 125"),
]


def main() -> None:
    raw = pd.read_csv(
        INPUT_PATH,
        sep=r"\s+",
        header=None,
        names=RAW_COLUMNS,
    )

    if raw.shape != (20631, 26):
        raise ValueError(f"Expected input shape (20631, 26), got {raw.shape}")

    rul_values = raw.groupby("unit_id")["cycle"].transform("max") - raw["cycle"]
    capped_rul = rul_values.clip(upper=125)

    if not capped_rul.eq(capped_rul.round()).all():
        raise ValueError("Computed RUL contains non-integer values")

    labeled = raw.copy()
    labeled["RUL"] = capped_rul.astype("int64")

    if labeled.shape != (20631, 27):
        raise ValueError(f"Expected labeled shape (20631, 27), got {labeled.shape}")
    if labeled["unit_id"].nunique() != 100:
        raise ValueError(
            f"Expected 100 unique engines, got {labeled['unit_id'].nunique()}"
        )
    if labeled.isna().any().any():
        raise ValueError("Labeled data contains NaN values")
    if not labeled["RUL"].between(0, 125).all():
        raise ValueError("RUL values must be between 0 and 125")
    if not labeled.groupby("unit_id")["RUL"].min().eq(0).all():
        raise ValueError("Every engine must have a minimum RUL of 0")

    schema = {
        "source": "NASA C-MAPSS FD001 training data (data/raw/train_FD001.txt, unmodified)",
        "output_file": "data/processed/train_FD001_labeled.txt",
        "format": "space-delimited, no header",
        "n_rows": int(len(labeled)),
        "n_columns": int(labeled.shape[1]),
        "rul_definition": "maximum cycle for the engine minus the row cycle",
        "rul_cap": 125,
        "rul_cap_note": "The cap is a modelling choice (Heimes 2008), not part of the original NASA data.",
        "columns": [
            {"index": i, "name": n, "description": d}
            for i, (n, d) in enumerate(COLUMN_DESCRIPTIONS, start=1)
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if LABELED_PATH.exists():
        raise FileExistsError(f"Output already exists: {LABELED_PATH}")
    if SCHEMA_PATH.exists():
        raise FileExistsError(f"Output already exists: {SCHEMA_PATH}")

    created_paths = []
    try:
        with LABELED_PATH.open("x", encoding="utf-8", newline="") as data_file:
            created_paths.append(LABELED_PATH)
            with SCHEMA_PATH.open("x", encoding="utf-8", newline="") as schema_file:
                created_paths.append(SCHEMA_PATH)
                labeled.to_csv(
                    data_file,
                    sep=" ",
                    header=False,
                    index=False,
                    lineterminator="\n",
                )
                json.dump(schema, schema_file, indent=2)
                schema_file.write("\n")
    except Exception:
        for path in reversed(created_paths):
            if path.exists():
                path.unlink()
        raise


if __name__ == "__main__":
    main()
