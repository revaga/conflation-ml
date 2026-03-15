import pandas as pd
import numpy as np
from parquet_io import read_parquet_safe

GOLDEN_PATH = "data/golden_dataset_200.parquet"
df = read_parquet_safe(GOLDEN_PATH)
print(f"Total rows: {len(df)}")
if "xgboost_testlabels" in df.columns:
    print(f"Nulls in xgboost_testlabels: {df['xgboost_testlabels'].isna().sum()}")
    print("\nValue counts for xgboost_testlabels:")
    print(df["xgboost_testlabels"].value_counts(dropna=False))
else:
    print("xgboost_testlabels column not found.")
