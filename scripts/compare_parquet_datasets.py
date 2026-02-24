"""
Compare two parquet datasets: print sample rows and report columns/rows
that appear in one dataset but not the other.
"""
import os
import sys
import pandas as pd

# Paths relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FILE1 = os.path.join(DATA_DIR, "phase1_processed.parquet")
FILE2 = os.path.join(DATA_DIR, "project_a_samples.parquet")

N_SAMPLE_ROWS = 5


def load_or_none(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None


def show_sample(df: pd.DataFrame, name: str, n: int = N_SAMPLE_ROWS) -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.max_colwidth", 40)
    print(f"\n--- {name} ---")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nColumns:", list(df.columns))
    print(f"\nFirst {n} rows:")
    print(df.head(n).to_string())


def compare_columns(cols1: list, cols2: list, name1: str, name2: str) -> None:
    set1, set2 = set(cols1), set(cols2)
    only_in_1 = set1 - set2
    only_in_2 = set2 - set1
    common = set1 & set2
    print(f"\n--- Column comparison ---")
    print(f"Common columns: {len(common)}")
    if only_in_1:
        print(f"Only in {name1}: {sorted(only_in_1)}")
    else:
        print(f"Only in {name1}: (none)")
    if only_in_2:
        print(f"Only in {name2}: {sorted(only_in_2)}")
    else:
        print(f"Only in {name2}: (none)")


def compare_rows(df1: pd.DataFrame, df2: pd.DataFrame, name1: str, name2: str) -> None:
    """Compare row counts and try to find rows in one set but not the other by index."""
    print(f"\n--- Row comparison ---")
    print(f"{name1}: {len(df1)} rows")
    print(f"{name2}: {len(df2)} rows")
    # If both have an index we can compare, or a common ID column
    id_candidates = [c for c in df1.columns if "id" in c.lower() or "key" in c.lower()]
    id_col = None
    for c in id_candidates:
        if c in df2.columns:
            id_col = c
            break
    if id_col:
        ids1 = set(df1[id_col].dropna().astype(str))
        ids2 = set(df2[id_col].dropna().astype(str))
        only_in_1 = ids1 - ids2
        only_in_2 = ids2 - ids1
        print(f"Using key column: '{id_col}'")
        print(f"IDs only in {name1}: {len(only_in_1)}")
        if only_in_1 and len(only_in_1) <= 20:
            print(f"  {sorted(only_in_1)}")
        elif only_in_1:
            print(f"  (first 20) {sorted(only_in_1)[:20]}")
        print(f"IDs only in {name2}: {len(only_in_2)}")
        if only_in_2 and len(only_in_2) <= 20:
            print(f"  {sorted(only_in_2)}")
        elif only_in_2:
            print(f"  (first 20) {sorted(only_in_2)[:20]}")
    else:
        print("No common ID-like column found; row comparison by index only.")
        print(f"Row index range {name1}: 0 .. {len(df1) - 1}")
        print(f"Row index range {name2}: 0 .. {len(df2) - 1}")


def main() -> None:
    print("Comparing parquet datasets")
    print("File 1:", FILE1)
    print("File 2:", FILE2)

    df1 = load_or_none(FILE1)
    df2 = load_or_none(FILE2)

    if df1 is None and df2 is None:
        sys.exit(1)
    if df1 is None:
        print("\nOnly second file available. Showing sample of second file only.")
        show_sample(df2, "project_a_samples.parquet")
        return
    if df2 is None:
        print("\nOnly first file available. Showing sample of first file only.")
        show_sample(df1, "phase1_processed.parquet")
        return

    show_sample(df1, "phase1_processed.parquet")
    show_sample(df2, "project_a_samples.parquet")
    compare_columns(
        df1.columns.tolist(),
        df2.columns.tolist(),
        "phase1_processed.parquet",
        "project_a_samples.parquet",
    )
    compare_rows(df1, df2, "phase1_processed.parquet", "project_a_samples.parquet")


if __name__ == "__main__":
    main()
