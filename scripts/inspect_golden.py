import pandas as pd
import os
from parquet_io import read_parquet_safe
import sys

# Set encoding to utf-8 for safety
sys.stdout = open(r'c:\Users\revaa\neha-reva-places-attribute-conflation\data\inspection_output.txt', 'w', encoding='utf-8')

files = [
    r"c:\Users\revaa\neha-reva-places-attribute-conflation\data\golden_dataset_100.parquet",
    r"c:\Users\revaa\neha-reva-places-attribute-conflation\data\golden_dataset_200.parquet"
]

output_file = r'c:\Users\revaa\neha-reva-places-attribute-conflation\data\inspection_output.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    sys.stdout = f
    for file_path in files:
        if os.path.exists(file_path):
            print(f"\n--- Investigating: {os.path.basename(file_path)} ---")
            df = read_parquet_safe(file_path)
            
            # Check if xgboost_testlabels and 3class_testlabels are the same
            if 'xgboost_testlabels' in df.columns and '3class_testlabels' in df.columns:
                are_identical = (df['xgboost_testlabels'] == df['3class_testlabels']).all()
                print(f"Are 'xgboost_testlabels' and '3class_testlabels' identical? {are_identical}")
                if not are_identical:
                    mismatches = (df['xgboost_testlabels'] != df['3class_testlabels']).sum()
                    print(f"Number of mismatches: {mismatches} out of {len(df)} rows")
            
            print("-" * 60)
        else:
            print(f"File not found: {file_path}")
    sys.stdout = sys.__stdout__
