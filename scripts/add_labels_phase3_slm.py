"""
Add 2-class and 4-class labels to phase3 SLM parquet files using the same
algorithm as add_2class_labels_golden.py and add_4class_labels_golden.py.
"""
import sys
from pathlib import Path

import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from labels import recalculate_4class_label
from add_2class_labels_golden import row_to_2class, ATTR_WINNER_COLS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PARQUET_PATHS = [
    DATA_DIR / "phase3_slm_labeled.parquet",
    DATA_DIR / "phase3_slm_labeledkimi.parquet",
]


def main():
    for parquet_path in PARQUET_PATHS:
        if not parquet_path.exists():
            print(f"File not found: {parquet_path}", file=sys.stderr)
            continue
        df = pd.read_parquet(parquet_path)
        missing = [c for c in ATTR_WINNER_COLS if c not in df.columns]
        if missing:
            print(f"{parquet_path.name}: missing columns {missing}", file=sys.stderr)
            continue
        df["2class_testlabels"] = df.apply(row_to_2class, axis=1)
        df["4class_testlabels"] = df.apply(recalculate_4class_label, axis=1)
        df.to_parquet(parquet_path, index=False)
        print(f"\n{parquet_path.name}:")
        print("  2class_testlabels:", df["2class_testlabels"].value_counts().to_dict())
        print("  4class_testlabels:", df["4class_testlabels"].value_counts().to_dict())


if __name__ == "__main__":
    main()
