"""
Show all columns in golden_dataset_200.parquet with unique values and null counts.
"""
import sys
from pathlib import Path

import pandas as pd

# Path relative to repo root
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PARQUET_PATH = DATA_DIR / "golden_dataset_200.parquet"

# Max unique values to print per column (avoid huge dumps)
MAX_UNIQUE_PRINT = 50


def main():
    # Avoid UnicodeEncodeError on Windows when printing values (e.g. Č in brand names)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not PARQUET_PATH.exists():
        print(f"File not found: {PARQUET_PATH}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(PARQUET_PATH)
    n_rows = len(df)

    print("=" * 70)
    print(f"golden_dataset_200.parquet — columns, unique values, nulls")
    print("=" * 70)
    print(f"Rows: {n_rows:,}\n")

    for col in df.columns:
        s = df[col]
        null_count = s.isna().sum()
        n_unique = s.nunique(dropna=False)
        uniques = s.dropna().unique()

        print("-" * 70)
        print(f"Column: {col}")
        print(f"  dtype: {s.dtype}")
        print(f"  null/missing: {null_count:,} ({100 * null_count / n_rows:.1f}%)")
        print(f"  unique values (excluding nulls): {n_unique if not s.isna().all() else 0}")

        def sort_key(x):
            try:
                return (str(x)[:50], repr(x)[:80])
            except Exception:
                return ("", "")

        if len(uniques) == 0:
            print(f"  values: (all null)")
        elif len(uniques) <= MAX_UNIQUE_PRINT:
            # Show each unique with a sample count
            for u in sorted(uniques, key=sort_key):
                count = (s == u).sum()
                disp = repr(u)
                if len(disp) > 60:
                    disp = disp[:57] + "..."
                print(f"    {count:>5} x  {disp}")
        else:
            # Too many uniques: show first N and summary
            sample = sorted(uniques, key=sort_key)[:MAX_UNIQUE_PRINT]
            for u in sample:
                count = (s == u).sum()
                disp = repr(u)
                if len(disp) > 60:
                    disp = disp[:57] + "..."
                print(f"    {count:>5} x  {disp}")
            print(f"    ... and {len(uniques) - MAX_UNIQUE_PRINT} more unique values")

        print()

    print("=" * 70)


if __name__ == "__main__":
    main()
