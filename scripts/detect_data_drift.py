"""
Data Drift Detection
=====================================================================================
Checks for distribution drift between the Phase 3 training dataset and the 
live incoming project samples to ensure the models aren't operating out-of-distribution.
"""

import os
import logging
import argparse
import pandas as pd
import numpy as np
from scripts.parquet_io import read_parquet_safe
from scripts.features import engineer_features
from scripts.xgboost_multiclass import FEATURE_COLS

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

TRAIN_PATH = "data/phase3_slm_labeled.parquet"
LIVE_PATH = "data/phase1_processed.parquet"

def check_drift(df_train: pd.DataFrame, df_live: pd.DataFrame, threshold: float = 0.2):
    """
    Compares the means of key features. If the drift in mean is > threshold standard 
    deviations of the training set, raises a warning.
    """
    logger.info(f"Comparing {len(df_train)} training rows vs {len(df_live)} live rows...")
    
    # Needs features
    df_train = engineer_features(df_train)
    df_live = engineer_features(df_live)
    
    drift_detected = False
    
    for col in FEATURE_COLS:
        if col not in df_train.columns or col not in df_live.columns:
            continue
            
        train_mean = df_train[col].mean()
        train_std = df_train[col].std()
        
        live_mean = df_live[col].mean()
        
        if train_std == 0 or pd.isna(train_std):
            continue
            
        drift_z_score = abs(live_mean - train_mean) / train_std
        
        if drift_z_score > threshold:
            drift_detected = True
            logger.warning(f"DRIFT DETECTED in '{col}': Train Mean={train_mean:.3f}, Live Mean={live_mean:.3f} (z-score: {drift_z_score:.2f})")
            
    if not drift_detected:
        logger.info("No significant data drift detected across features.")
    
    return drift_detected

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.25, help="Z-score threshold for drift alert")
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("Data Drift Detection")
    logger.info("=" * 70)
    
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(LIVE_PATH):
        logger.error("Missing datasets for drift comparison.")
        return
        
    df_train = read_parquet_safe(TRAIN_PATH)
    df_live = read_parquet_safe(LIVE_PATH)
    
    check_drift(df_train, df_live, args.threshold)

if __name__ == "__main__":
    main()
