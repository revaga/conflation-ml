import os
import sys
import logging
import pandas as pd
from scripts.parquet_io import read_parquet_safe

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Paths relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FILE1 = os.path.join(DATA_DIR, "phase1_processed.parquet")
FILE2 = os.path.join(DATA_DIR, "project_a_samples.parquet")

N_SAMPLE_ROWS = 5

def load_or_none(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        logger.error(f"File not found: {path}")
        return None
    try:
        return read_parquet_safe(path)
    except Exception as e:
        logger.error(f"Error reading {path}: {e}")
        return None

def show_sample(df: pd.DataFrame, name: str, n: int = N_SAMPLE_ROWS) -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.max_colwidth", 40)
    logger.info(f"\n--- {name} ---")
    logger.info(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"\nFirst {n} rows:")
    logger.info("\n" + df.head(n).to_string())

def compare_columns(cols1: list, cols2: list, name1: str, name2: str) -> None:
    set1, set2 = set(cols1), set(cols2)
    only_in_1 = set1 - set2
    only_in_2 = set2 - set1
    common = set1 & set2
    logger.info(f"\n--- Column comparison ---")
    logger.info(f"Common columns: {len(common)}")
    logger.info(f"Only in {name1}: {sorted(only_in_1) if only_in_1 else '(none)'}")
    logger.info(f"Only in {name2}: {sorted(only_in_2) if only_in_2 else '(none)'}")

def compare_rows(df1: pd.DataFrame, df2: pd.DataFrame, name1: str, name2: str) -> None:
    logger.info(f"\n--- Row comparison ---")
    logger.info(f"{name1}: {len(df1)} rows")
    logger.info(f"{name2}: {len(df2)} rows")
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
        logger.info(f"Using key column: '{id_col}'")
        logger.info(f"IDs only in {name1}: {len(only_in_1)}")
        if only_in_1:
            logger.info(f"  (first 20) {sorted(only_in_1)[:20]}")
        logger.info(f"IDs only in {name2}: {len(only_in_2)}")
        if only_in_2:
            logger.info(f"  (first 20) {sorted(only_in_2)[:20]}")

def main() -> None:
    logger.info("Comparing parquet datasets")
    logger.info(f"File 1: {FILE1}")
    logger.info(f"File 2: {FILE2}")

    df1 = load_or_none(FILE1)
    df2 = load_or_none(FILE2)

    if df1 is None and df2 is None:
        sys.exit(1)
    if df1 is None:
        show_sample(df2, "File 2")
        return
    if df2 is None:
        show_sample(df1, "File 1")
        return

    show_sample(df1, "File 1")
    show_sample(df2, "File 2")
    compare_columns(df1.columns.tolist(), df2.columns.tolist(), "File 1", "File 2")
    compare_rows(df1, df2, "File 1", "File 2")

if __name__ == "__main__":
    main()
