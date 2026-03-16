"""
Create a 2class_testlabels column in golden_dataset_200.parquet.
Label each row as 'base' or 'alt' from attr_*_winner columns;
uses scripts.labels.row_to_2class (same rule as binary model training).
"""
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PARQUET_PATH = DATA_DIR / "golden_dataset_200.parquet"

try:
    from scripts.labels import row_to_2class
except ImportError:
    sys.path.insert(0, str(DATA_DIR.parent))
    from scripts.labels import row_to_2class


ATTR_WINNER_COLS = [
    "attr_name_winner",
    "attr_phone_winner",
    "attr_web_winner",
    "attr_address_winner",
    "attr_category_winner",
]


def main():
    if not PARQUET_PATH.exists():
        print(f"File not found: {PARQUET_PATH}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(PARQUET_PATH)

    missing = [c for c in ATTR_WINNER_COLS if c not in df.columns]
    if missing:
        print(f"Missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    df["2class_testlabels"] = df.apply(row_to_2class, axis=1)
    df.to_parquet(PARQUET_PATH, index=False)

    counts = df["2class_testlabels"].value_counts()
    print("2class_testlabels added. Distribution:")
    for label, n in counts.items():
        print(f"  {label}: {n}")


if __name__ == "__main__":
    main()
