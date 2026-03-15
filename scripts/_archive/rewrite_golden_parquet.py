"""
Re-save golden_dataset_200.parquet so PyArrow can read it.
Reads with fastparquet (avoids PyArrow 'Repetition level histogram size mismatch'),
writes with PyArrow to produce a clean file.
Run from project root: python scripts/rewrite_golden_parquet.py
"""
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = PROJECT_ROOT / "data" / "golden_dataset_200.parquet"


def main():
    print(f"Reading {GOLDEN_PATH} with fastparquet ...")
    df = pd.read_parquet(GOLDEN_PATH, engine="fastparquet")
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"Writing back with PyArrow ...")
    df.to_parquet(GOLDEN_PATH, index=False, engine="pyarrow")
    print("Done. Golden file can now be read with PyArrow.")


if __name__ == "__main__":
    main()
