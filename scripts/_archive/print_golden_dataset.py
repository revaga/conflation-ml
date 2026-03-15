"""Print contents of the golden dataset parquet."""
import sys
from pathlib import Path

import pandas as pd
from parquet_io import read_parquet_safe

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = PROJECT_ROOT / "data" / "golden_dataset_100.parquet"


def main():
    if not GOLDEN_PATH.exists():
        print(f"File not found: {GOLDEN_PATH}", file=sys.stderr)
        sys.exit(1)
    df = read_parquet_safe(str(GOLDEN_PATH))
    print(f"Rows: {len(df)}, Columns: {list(df.columns)}\n")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", 50)
    print(df.to_string())


if __name__ == "__main__":
    main()
