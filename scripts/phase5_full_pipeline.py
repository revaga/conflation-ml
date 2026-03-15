"""
Phase 5 — Full Conflict Resolution & Abstention Pipeline
=========================================================
Unified pipeline that:
1. Reads raw data/project_a_samples.parquet
2. Engineers features (centralized)
3. Applies XGBoost-style 2-stage classification
4. Implements Highest Confidence baseline
5. Evaluates against Golden Dataset

Refactored to use centralized logic from scripts/features.py and scripts/labels.py.
"""

import os
import warnings
import pandas as pd
import numpy as np
import logging
import argparse
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report

# Centralized imports
from scripts.features import engineer_features
from scripts.labels import recalculate_4class_label, fourclass_to_threeclass
from scripts.parquet_io import read_parquet_safe
from scripts.validator_cache import clear_validator_cache
from scripts.schema import validate_phase1_processed

# We reuse the Stage 2 detection from xgboostbinary
from scripts.xgboostbinary import _identify_both

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
RAW_INPUT_PATH = "data/project_a_samples.parquet"
PROCESSED_PATH = "data/phase1_processed.parquet"
GOLDEN_PATH = "data/golden_dataset_200.parquet"
OUTPUT_PATH = "data/phase5_full_results.parquet"

def apply_highest_confidence_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Strategy: Pick match if confidence > base_confidence, else keep base."""
    df["baseline_selection"] = np.where(df["confidence"] > df["base_confidence"], "alt", "base")
    return df

def main():
    parser = argparse.ArgumentParser(description="Phase 5 -- Full Pipeline & Comparison")
    parser.add_argument("--clear-cache", action="store_true", help="Clear the validator cache before running")
    parser.add_argument("--validate", action="store_true", help="Run website and phone validation (slow)")
    args = parser.parse_args()

    if args.clear_cache:
        clear_validator_cache()

    logger.info("=" * 70)
    logger.info("PHASE 5 -- Full Pipeline & Comparison")
    logger.info("=" * 70)

    # 1. Load Data
    # We use processing path if available, else raw
    if os.path.exists(PROCESSED_PATH):
        logger.info(f"Loading processed data from {PROCESSED_PATH} ...")
        df = read_parquet_safe(PROCESSED_PATH)
    else:
        logger.info(f"Loading raw data from {RAW_INPUT_PATH} ...")
        df = read_parquet_safe(RAW_INPUT_PATH)
        
    # Enforce Schema on Phase 1 processed similarities / confidences
    validate_phase1_processed(df)
    
    df = engineer_features(df, validate_urls=args.validate, validate_phones=args.validate)
    
    # 2. Add Baseline
    logger.info("Applying Highest Confidence baseline ...")
    df = apply_highest_confidence_baseline(df)
    
    # 3. Apply Unified logic (Stage 2 Both-ness) to help baseline identify 'both'
    # High similarity + Confidence delta < 0.05 -> 'both'
    is_both = _identify_both(df)
    df.loc[is_both, "baseline_selection"] = "both"
    
    # 4. Apply Single 3-Class ML Model
    import xgboost as xgb
    from scripts.xgboost_multiclass import FEATURE_COLS

    MODEL_PATH = "data/models/refiner_3class.json"
    
    X_all = df[FEATURE_COLS].fillna(0).values

    if os.path.exists(MODEL_PATH):
        logger.info(f"Running Inference with Single 3-Class Model: {MODEL_PATH} ...")
        try:
            booster = xgb.Booster()
            booster.load_model(MODEL_PATH)
            
            dmatrix_all = xgb.DMatrix(X_all)
            proba = booster.predict(dmatrix_all)
            
            # Map index to native 3-class output (0:alt, 1:base, 2:both)
            IDX_MAP = {0: "alt", 1: "base", 2: "both"}
            preds_idx = np.argmax(proba, axis=1)
            preds_mapped = [IDX_MAP[i] for i in preds_idx]
            
            df["model_3class_prediction"] = preds_mapped
            
            logger.info("ML Inference Complete.")
        except Exception as e:
            logger.warning(f"Failed to run inference: {e}")
            import traceback
            traceback.print_exc()
    else:
        logger.warning(f"Missing model! Need {MODEL_PATH}")

    # 5. Evaluation against Golden
    golden = read_parquet_safe(GOLDEN_PATH) if os.path.exists(GOLDEN_PATH) else pd.DataFrame()
    if not golden.empty:
        logger.info(f"Reviewing Golden Dataset ({len(golden)} rows) ...")
        df["is_golden"] = df["id"].isin(golden["id"])
        test_df = df[df["is_golden"]].merge(golden[["id", "3class_testlabels"]], on="id", suffixes=('', '_truth'))
        
        truth = test_df["3class_testlabels"].str.lower().map({"alt": "alt", "match": "alt", "both": "both", "base": "base", "none": "base"})
        
        # Strategies to compare
        strategies = {
            "Baseline (Conf + Heuristic Both)": test_df["baseline_selection"],
        }
        if "model_3class_prediction" in test_df.columns:
            strategies["XGBoost 3-Class"] = test_df["model_3class_prediction"]

        for name, pred in strategies.items():
            mask = truth.notna() & pred.notna()
            cur_truth, cur_pred = truth[mask], pred[mask]
            logger.info(f"--- {name} Performance ---")
            logger.info(f"Accuracy: {accuracy_score(cur_truth, cur_pred):.4%}")
            if "XGBoost" in name:
                logger.info("\n" + classification_report(cur_truth, cur_pred))

    # 6. Save Results
    if "model_3class_prediction" in df.columns:
        df["final_prediction"] = df["model_3class_prediction"]
    else:
        df["final_prediction"] = df["baseline_selection"]
    
    keep_cols = [c for c in df.columns if not c.startswith("_")]
    df[keep_cols].to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"Pipeline Complete. Results saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
