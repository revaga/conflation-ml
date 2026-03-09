import pandas as pd
import numpy as np
from parquet_io import read_parquet_safe

GOLDEN_100 = "data/golden_dataset_100.parquet"
GOLDEN_200 = "data/golden_dataset_200.parquet"

def _normalize_attr_winner(val):
    if pd.isna(val) or val is None:
        return "none"
    v = str(val).strip().lower()
    if v in ("base", "alt", "both", "none"):
        return v
    return "none"

def derive_4_to_3(row):
    ATTR_ATTRS = ("name", "phone", "web", "address", "category")
    counts = {"none": 0, "both": 0, "base": 0, "alt": 0}
    for attr in ATTR_ATTRS:
        col = f"attr_{attr}_winner"
        w = _normalize_attr_winner(row.get(col))
        counts[w] += 1
    
    label = "both" # Default
    if counts["none"] >= 3:
        label = "none"
    elif counts["both"] >= 3:
        label = "both"
    elif counts["base"] > counts["alt"]:
        label = "base"
    elif counts["alt"] > counts["base"]:
        label = "alt"
    
    # Map to 3-class (using 'alt' for consistency)
    if label in ("alt", "match"): return "alt"
    if label in ("base", "none"): return "base"
    return "both"

def main():
    df100 = read_parquet_safe(GOLDEN_100)
    df200 = read_parquet_safe(GOLDEN_200)
    
    print(f"Original 100 has column '3class_testlabels': {'3class_testlabels' in df100.columns}")
    
    # Restore first 100 from df100
    id_to_label = dict(zip(df100["id"], df100["3class_testlabels"]))
    
    df200["3class_testlabels"] = df200["id"].map(id_to_label)
    df200["xgboost_testlabels"] = df200["3class_testlabels"] # Mirror it
    
    # Count missing
    missing_count = df200["3class_testlabels"].isna().sum()
    print(f"Rows 101-200 (or unknown): {missing_count}")
    
    if missing_count > 0:
        mask = df200["3class_testlabels"].isna()
        print("Applying majority-vote derivation to missing labels...")
        derived = df200[mask].apply(derive_4_to_3, axis=1)
        df200.loc[mask, "3class_testlabels"] = derived
        df200.loc[mask, "xgboost_testlabels"] = derived
        
    print("\nFinal Value Counts for 3class_testlabels:")
    print(df200["3class_testlabels"].value_counts())
    
    df200.to_parquet(GOLDEN_200, index=False)
    print(f"\nSuccessfully updated {GOLDEN_200}")

if __name__ == "__main__":
    main()
