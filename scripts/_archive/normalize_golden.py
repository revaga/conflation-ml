import pandas as pd
from scripts.labels import fourclass_to_threeclass, recalculate_4class_label
from scripts.parquet_io import read_parquet_safe
from pathlib import Path

DATA_DIR = Path("data")
FILES = ["golden_dataset_100.parquet", "golden_dataset_200.parquet"]

def normalize_golden_parquet(filename):
    path = DATA_DIR / filename
    if not path.exists():
        print(f"Skipping {filename} (not found)")
        return

    print(f"Normalizing {filename}...")
    df = read_parquet_safe(str(path))
    
    # 1. Ensure we have 3class_testlabels
    # If missing, try to derive from attr_*_winner if available
    if "3class_testlabels" not in df.columns:
        if all(f"attr_{a}_winner" in df.columns for a in ["name", "phone", "web", "address", "category"]):
            print(f"  Deriving 3class_testlabels for {filename}")
            df["3class_testlabels"] = df.apply(lambda r: fourclass_to_threeclass(recalculate_4class_label(r)), axis=1)
        elif "xgboost_testlabels" in df.columns:
            df["3class_testlabels"] = df["xgboost_testlabels"]
        else:
            print(f"  WARNING: No way to derive labels for {filename}")
            return

    # 2. Map labels to canonical match/both/base
    # some might have 'alt' instead of 'match'
    label_map = {
        "alt": "match",
        "match": "match",
        "both": "both",
        "base": "base",
        "none": "base"
    }
    df["3class_testlabels"] = df["3class_testlabels"].str.lower().map(lambda x: label_map.get(x, x))
    
    # 3. Remove legacy columns
    if "xgboost_testlabels" in df.columns:
        df = df.drop(columns=["xgboost_testlabels"])
    
    # 4. Save back clean
    df.to_parquet(path, index=False)
    print(f"  Successfully normalized {filename}. Value counts:")
    print(df["3class_testlabels"].value_counts())

if __name__ == "__main__":
    for f in FILES:
        normalize_golden_parquet(f)
