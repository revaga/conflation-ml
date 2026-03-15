import pandas as pd
import os
from parquet_io import read_parquet_safe

files = [
    r"c:\Users\revaa\neha-reva-places-attribute-conflation\data\golden_dataset_100.parquet",
    r"c:\Users\revaa\neha-reva-places-attribute-conflation\data\golden_dataset_200.parquet"
]

for file_path in files:
    if os.path.exists(file_path):
        print(f"Processing: {os.path.basename(file_path)}")
        df = read_parquet_safe(file_path)
        
        changed = False
        if 'xgboost_testlabels' in df.columns:
            if '3class_testlabels' not in df.columns:
                print(f"  Renaming 'xgboost_testlabels' to '3class_testlabels'")
                df.rename(columns={'xgboost_testlabels': '3class_testlabels'}, inplace=True)
                changed = True
            else:
                print(f"  Removing 'xgboost_testlabels' (identical to '3class_testlabels')")
                df.drop(columns=['xgboost_testlabels'], inplace=True)
                changed = True
        
        if changed:
            df.to_parquet(file_path, index=False)
            print(f"  Saved changes to {os.path.basename(file_path)}")
        else:
            print(f"  No changes needed for {os.path.basename(file_path)}")
    else:
        print(f"File not found: {file_path}")
