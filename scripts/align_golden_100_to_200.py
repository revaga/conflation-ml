"""
Align data/golden_dataset_100.parquet to match columns of data/golden_dataset_200.parquet.
Takes values from golden_dataset_200 for the first 100 rows (same row order by id) and
adds any new/changed columns to golden_dataset_100. Writes data/golden_dataset_100_aligned.parquet.
Run from repo root: python scripts/align_golden_100_to_200.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
G100_PATH = DATA_DIR / "golden_dataset_100.parquet"
G200_PATH = DATA_DIR / "golden_dataset_200.parquet"
OUTPUT_PATH = DATA_DIR / "golden_dataset_100_aligned.parquet"


def main() -> int:
    if not G100_PATH.exists():
        print(f"Error: {G100_PATH} not found", file=sys.stderr)
        return 1
    if not G200_PATH.exists():
        print(f"Error: {G200_PATH} not found", file=sys.stderr)
        return 1

    g100 = pd.read_parquet(G100_PATH)
    g200 = pd.read_parquet(G200_PATH)

    # First 100 rows of 200 have same ids as 100 in same order (verified)
    g200_first100 = g200.iloc[:100].copy()
    cols_in_200_not_100 = set(g200.columns) - set(g100.columns)
    if not cols_in_200_not_100:
        print("golden_dataset_100 already has all columns from golden_dataset_200.")
        # Still write aligned so output path exists with same data
        g100.to_parquet(OUTPUT_PATH, index=False)
        print(f"Wrote {OUTPUT_PATH} ({len(g100)} rows)")
        return 0

    # Add new columns from 200 (first 100 rows, same order)
    for col in sorted(cols_in_200_not_100):
        g100[col] = g200_first100[col].values

    # Optional: drop columns that are in 100 but not in 200 so schema exactly matches
    cols_in_100_not_200 = set(g100.columns) - set(g200.columns)
    if cols_in_100_not_200:
        # Keep 100's extra columns (e.g. golden_label) so we don't lose data; verification uses 200 schema
        pass

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    g100.to_parquet(OUTPUT_PATH, index=False)
    print(f"Added columns from golden_dataset_200 (first 100 rows): {sorted(cols_in_200_not_100)}")
    print(f"Wrote {OUTPUT_PATH} ({len(g100)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
