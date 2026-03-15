import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root and scripts folder to sys.path
root = Path(__file__).parent.parent
sys.path.append(str(root))
sys.path.append(str(root / "scripts"))

from scripts.features import engineer_features as new_engineer_features
from scripts.parquet_io import read_parquet_safe

# Import the old function from multiclass script
# We'll use a trick to import it without running the whole script
import importlib.util
spec = importlib.util.spec_from_file_location("old_multiclass", str(root / "scripts" / "xgboost_multiclass.py"))
old_multiclass = importlib.util.module_from_spec(spec)
# Patch print and some variables to avoid execution side effects if any
old_multiclass.print = lambda *args, **kwargs: None
spec.loader.exec_module(old_multiclass)

def test_multiclass_parity():
    print("Testing Feature Parity for Multiclass...")
    data_path = root / "data" / "phase1_processed.parquet"
    df = read_parquet_safe(str(data_path)).head(20)
    
    # Run old
    df_old = old_multiclass.engineer_features(df.copy())
    
    # Run new
    # Multiclass validates both URLs and Phones
    df_new = new_engineer_features(df.copy(), validate_urls=True, validate_phones=True)
    
    # Check common columns
    common_cols = [c for c in df_old.columns if c.startswith("feat_")]
    
    mismatches = []
    for col in common_cols:
        if col not in df_new.columns:
            mismatches.append(f"Missing column: {col}")
            continue
            
        # Compare values with 1e-6 tolerance
        try:
            pd.testing.assert_series_equal(df_old[col], df_new[col], rtol=1e-6, atol=1e-6, check_names=False)
        except AssertionError as e:
            mismatches.append(f"Mismatch in {col}: {str(e)[:200]}...")
            
    if mismatches:
        print(f"FAIL: Found {len(mismatches)} mismatches:")
        for m in mismatches[:10]:
            print(f"  - {m}")
        if len(mismatches) > 10:
            print("  ...")
    else:
        print("SUCCESS: All common features match within 1e-6 tolerance.")

if __name__ == "__main__":
    test_multiclass_parity()
