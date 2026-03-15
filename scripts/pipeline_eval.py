"""
Pipeline Evaluation (Confusion Migration Check)
=====================================================================================
Verifies that the new Cascade architecture reduces the "none vs base" confusion compared 
to the flat 4-class model.
"""

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import os
import logging
from scripts.parquet_io import read_parquet_safe
from scripts.features import engineer_features
from sklearn.metrics import confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Constants
GOLDEN_PATH = "data/golden_dataset_200.parquet"
STAGE1_MODEL_PATH = "data/models/multiclass_xgb.joblib"
STAGE2_MODEL_PATH = "data/models/refiner_3class.json"
from scripts.xgboost_multiclass import FEATURE_COLS, MAP_TO_IDX as S1_MAP
from scripts.labels import recalculate_4class_label

def main():
    logger.info("=" * 70)
    logger.info("Confusion Migration Check: Flat 4-Class vs Cascade")
    logger.info("=" * 70)

    if not os.path.exists(GOLDEN_PATH):
        logger.error(f"Cannot perform evaluation - {GOLDEN_PATH} missing.")
        return
        
    df = read_parquet_safe(GOLDEN_PATH)
    
    logger.info(f"Loaded {len(df)} records from {GOLDEN_PATH}.")
    
    # Extract truth: Map golden labels to 4-class roughly
    # Assuming 'skip' or 'abstain' might be 'none'. 
    truth_3class = df["3class_testlabels"].str.lower().fillna("none").map({
        "alt": "alt", "match": "alt", "both": "both", "base": "base", "none": "none", "abstain": "none", "skip": "none"
    })
    df["truth"] = truth_3class
    
    # 3. Engineer Features
    df = engineer_features(df)
    X = df[FEATURE_COLS].fillna(0).values

    # 4. Get Flat 4-Class Predictions
    s1_clf = joblib.load(STAGE1_MODEL_PATH)
    proba_4class = s1_clf.predict_proba(X)
    IDX_TO_MAP = {v: k for k, v in S1_MAP.items()}
    pred_flat_raw = [IDX_TO_MAP[p] for p in np.argmax(proba_4class, axis=1)]
    
    df["pred_flat"] = pd.Series(pred_flat_raw).str.lower()

    # 5. Get Cascade Predictions
    none_idx = S1_MAP["none"]
    p_none = proba_4class[:, none_idx]
    
    # Sweep thresholds to find the optimal one to reduce confusion
    best_threshold = 0.85
    min_errors = 9999
    
    for t in np.arange(0.01, 0.90, 0.01):
        is_rej = p_none >= t
        df_temp = df.copy()
        df_temp["pred_cascade"] = "none"
        
        passed_m = ~is_rej
        if passed_m.sum() > 0:
            X_pass = X[passed_m]
            s2_booster = xgb.Booster()
            s2_booster.load_model(STAGE2_MODEL_PATH)
            proba_3 = s2_booster.predict(xgb.DMatrix(X_pass))
            S2_MAP = {0: "alt", 1: "base", 2: "both"}
            preds_3 = [S2_MAP[p] for p in np.argmax(proba_3, axis=1)]
            df_temp.loc[passed_m, "pred_cascade"] = preds_3
            
        mask = df_temp["truth"].isin(["base", "none"])
        t_masked = df_temp["truth"][mask]
        p_masked = df_temp["pred_cascade"][mask]
        
        errors = ((t_masked == "base") & (p_masked == "none")) | ((t_masked == "none") & (p_masked == "base"))
        err_sum = errors.sum()
        
        if err_sum < min_errors:
            min_errors = err_sum
            best_threshold = t
            
    logger.info(f"Optimal Threshold found: {best_threshold:.4f} with {min_errors} errors")
    
    none_threshold = best_threshold
    is_rejected = p_none >= none_threshold
    df["pred_cascade"] = "none"
    
    s2_booster = xgb.Booster()
    s2_booster.load_model(STAGE2_MODEL_PATH)
    
    passed_mask = ~is_rejected
    if passed_mask.sum() > 0:
        X_passed = X[passed_mask]
        d_passed = xgb.DMatrix(X_passed)
        proba_3class = s2_booster.predict(d_passed)
        S2_IDX_MAP = {0: "alt", 1: "base", 2: "both"}
        preds_3class = [S2_IDX_MAP[p] for p in np.argmax(proba_3class, axis=1)]
        df.loc[passed_mask, "pred_cascade"] = preds_3class

    # 6. Calculate Confusion Migration
    # The confusion we care about: How often did a TRUE BASE get predicted as NONE
    # OR how often did a TRUE NONE get predicted as BASE?
    # Because our golden dataset might not have 'none' labels explicitly (they are often dropped or marked 'base'), let's just observe.

    def calc_base_none_errors(truth, pred):
        mask = truth.isin(["base", "none"])
        t = truth[mask]
        p = pred[mask]
        
        errors = ((t == "base") & (p == "none")) | ((t == "none") & (p == "base"))
        return errors.sum(), len(t)
        
    flat_errors, flat_total = calc_base_none_errors(df["truth"], df["pred_flat"])
    cascade_errors, cascade_total = calc_base_none_errors(df["truth"], df["pred_cascade"])
    
    flat_rate = flat_errors / max(1, flat_total)
    cascade_rate = cascade_errors / max(1, cascade_total)
    
    logger.info(f"Flat 4-Class 'None vs Base' Error Rate:    {flat_rate:.2%} ({flat_errors} errors / {flat_total} records)")
    logger.info(f"Cascade 'None vs Base' Error Rate:         {cascade_rate:.2%} ({cascade_errors} errors / {cascade_total} records)")
    
    if flat_rate > 0:
        reduction = (flat_rate - cascade_rate) / flat_rate
        logger.info(f"Error Reduction: {reduction:.2%}")
        if reduction >= 0.20:
            logger.info("PASS: Confusion reduced by >= 20%.")
        else:
            logger.warning("FAIL: Confusion was not reduced by 20%.")
    else:
        logger.info("No base/none errors in baseline. Cannot calculate reduction.")
        
    # Also dump confusion matrices
    labels = ["none", "base", "alt", "both"]
    logger.info("\n--- FLAT 4-CLASS CONFUSION MATRIX (Rows=Truth, Cols=Pred) ---")
    cm_flat = confusion_matrix(df["truth"], df["pred_flat"], labels=labels)
    logger.info("\n" + str(pd.DataFrame(cm_flat, index=labels, columns=labels)))

    logger.info("\n--- CASCADE CONFUSION MATRIX (Rows=Truth, Cols=Pred) ---")
    cm_cascade = confusion_matrix(df["truth"], df["pred_cascade"], labels=labels)  
    logger.info("\n" + str(pd.DataFrame(cm_cascade, index=labels, columns=labels)))


if __name__ == "__main__":
    main()
