"""
Apply 3-class labels (match/both/base) to golden_dataset_200.parquet
derived from the attr_*_winner columns.
"""
import pandas as pd
import numpy as np
from parquet_io import read_parquet_safe

GOLDEN_PATH = "data/golden_dataset_200.parquet"
ATTR_ATTRS = ("name", "phone", "web", "address", "category")
TIE_BREAK_TO_BOTH = True

def _normalize_attr_winner(val):
    if pd.isna(val) or val is None:
        return "none"
    v = str(val).strip().lower()
    if v in ("base", "alt", "both", "none"):
        return v
    return "none"

def derive_4class(row):
    counts = {"none": 0, "both": 0, "base": 0, "alt": 0}
    for attr in ATTR_ATTRS:
        col = f"attr_{attr}_winner"
        w = _normalize_attr_winner(row.get(col))
        counts[w] += 1
    
    if counts["none"] >= 3:
        return "none"
    if counts["both"] >= 3:
        return "both"
    if counts["base"] > counts["alt"]:
        return "base"
    if counts["alt"] > counts["base"]:
        return "alt"
    return "both" if TIE_BREAK_TO_BOTH else "alt"

def fourclass_to_threeclass(label):
    if label == "alt":
        return "alt"  # Map 'alt' to 'alt' to match original data in golden_dataset_100
    if label in ("base", "none"):
        return "base"
    return "both"

def main():
    print(f"Reading {GOLDEN_PATH}...")
    df = read_parquet_safe(GOLDEN_PATH)
    
    # Check if we have the winner columns
    missing = [f"attr_{a}_winner" for a in ATTR_ATTRS if f"attr_{a}_winner" not in df.columns]
    if missing:
        print(f"Error: Missing columns {missing}")
        return

    print("Deriving classes for missing labels...")
    # Derive only where missing in either target column (if they exist)
    # We want to fill 3class_testlabels and xgboost_testlabels
    
    if "3class_testlabels" not in df.columns:
        df["3class_testlabels"] = np.nan
    if "xgboost_testlabels" not in df.columns:
        df["xgboost_testlabels"] = np.nan
        
    mask_3class = df["3class_testlabels"].isna()
    mask_xgboost = df["xgboost_testlabels"].isna()
    
    any_missing = mask_3class | mask_xgboost
    if any_missing.any():
        print(f"Found {any_missing.sum()} rows with missing labels. Applying derivation...")
        derived_3class = df[any_missing].apply(derive_4class, axis=1).map(fourclass_to_threeclass)
        df.loc[mask_3class, "3class_testlabels"] = derived_3class
        df.loc[mask_xgboost, "xgboost_testlabels"] = derived_3class

    print("\nValue counts for 3class_testlabels:")
    print(df["3class_testlabels"].value_counts())
    
    df.to_parquet(GOLDEN_PATH, index=False)
    print(f"\nSaved {GOLDEN_PATH}")

if __name__ == "__main__":
    main()
