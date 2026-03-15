"""
Calculate 4-class labels (alt, base, both, none) for golden_dataset_200.parquet
from the attr_*_winner columns, using the shared logic in labels.py.
"""
import sys
from pathlib import Path

import pandas as pd

# Ensure scripts/ is on path when run as python scripts/add_4class_labels_golden.py
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from labels import recalculate_4class_label

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PARQUET_PATH = DATA_DIR / "golden_dataset_200.parquet"


def main():
    if not PARQUET_PATH.exists():
        print(f"File not found: {PARQUET_PATH}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(PARQUET_PATH)

    required = ["attr_name_winner", "attr_phone_winner", "attr_web_winner", "attr_address_winner", "attr_category_winner"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    df["4class_testlabels"] = df.apply(recalculate_4class_label, axis=1)
    df.to_parquet(PARQUET_PATH, index=False)

    counts = df["4class_testlabels"].value_counts()
    print("4class_testlabels added. Distribution:")
    for label in ("none", "alt", "base", "both"):
        n = counts.get(label, 0)
        print(f"  {label}: {n}")


if __name__ == "__main__":
    main()
