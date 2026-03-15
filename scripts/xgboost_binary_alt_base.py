"""
XGBoost Binary Classifier (Refiner): alt (1) vs base (0)
=====================================================================================
Specialized script to distinguish 'alt' (Class 1) from 'base' (Class 0).
Prioritizes identifying subtle differences that make an 'alt' record superior.

Usage:
    python scripts/xgboost_binary_alt_base.py --input data/phase3_slm_labeled.parquet
"""

import os
import argparse
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    log_loss, 
    roc_auc_score
)

# Centralized imports
try:
    from scripts.features import engineer_features
    from scripts.labels import recalculate_4class_label
    from scripts.parquet_io import read_parquet_safe
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from scripts.features import engineer_features
    from scripts.labels import recalculate_4class_label
    from scripts.parquet_io import read_parquet_safe

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
# Authoritative feature list based on engineer_features() in features.py
FEATURE_COLS = [
    "feat_existence_conf_delta",
    "feat_match_exists_score",
    "feat_base_exists_score",
    "feat_addr_richness_delta",
    "feat_match_has_phone",
    "feat_base_has_phone",
    "feat_match_has_web",
    "feat_base_has_web",
    "feat_phone_presence_delta",
    "feat_web_presence_delta",
    "feat_match_has_brand",
    "feat_base_has_brand",
    "feat_brand_delta",
    "feat_match_has_social",
    "feat_base_has_social",
    "feat_social_delta",
    "feat_match_web_valid",
    "feat_base_web_valid",
    "feat_web_valid_delta",
    "feat_match_phone_valid",
    "feat_base_phone_valid",
    "feat_phone_valid_delta",
    "feat_name_similarity",
    "feat_addr_similarity",
    "feat_phone_similarity",
    "feat_phone_exact_match",
    "feat_web_similarity",
    "feat_category_similarity",
    "feat_category_exact_match",
    "feat_is_msft_match",
    "feat_is_meta_match",
    "feat_src_count_delta",
    "feat_match_recency_days",
    "feat_base_recency_days",
    "feat_recency_delta",
    "feat_match_completeness",
    "feat_base_completeness",
    "feat_completeness_delta",
    "feat_valid_count_delta",
    "feat_name_addr_sim_product",
    "feat_avg_similarity",
    "feat_adds_new_info",
    "feat_addr_dissimilarity",
    "feat_conf_x_addr_richness",
    "feat_phone_exclusive",
    "feat_web_exclusive",
]

RANDOM_STATE = 42

def check_schema(df: pd.DataFrame):
    """Ensure all required FEATURE_COLS exist in the dataframe."""
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns in engineered dataframe: {missing}")
    logger.info("Schema check PASSED: All FEATURE_COLS are present.")

def main():
    parser = argparse.ArgumentParser(description="XGBoost Binary Classifier (alt vs base)")
    parser.add_argument("--input", type=str, default="data/phase3_slm_labeled.parquet", help="Path to labeled training data")
    parser.add_argument("--golden", type=str, default="data/golden_dataset_200.parquet", help="Path to golden evaluation set")
    parser.add_argument("--output", type=str, default="data/xgboost_binary_results.parquet", help="Path to save predictions")
    parser.add_argument("--model_path", type=str, default="data/models/binary_alt_base.json", help="Path to save the trained model")
    args = parser.parse_args()

    # 1. Load and Label Data
    logger.info(f"Loading data from {args.input} ...")
    df = read_parquet_safe(args.input)
    
    logger.info("Calculating labels using labels.py logic ...")
    df['temp_label'] = df.apply(recalculate_4class_label, axis=1)
    
    # Filtering: only 'alt' or 'base'. Ignore 'both' and 'none'.
    filtered_df = df[df['temp_label'].isin(['alt', 'base'])].copy()
    logger.info(f"Filtered to 'alt' and 'base' labels. Count: {len(filtered_df)} (Dropped {len(df) - len(filtered_df)} rows)")
    
    if filtered_df.empty:
        logger.error("No data remains after filtering for 'alt' and 'base'.")
        return

    # Label Mapping: 0 for base, 1 for alt
    label_map = {"base": 0, "alt": 1}
    filtered_df['target'] = filtered_df['temp_label'].map(label_map)
    
    # 2. Feature Engineering
    logger.info("Engineering features ...")
    processed_df = engineer_features(filtered_df)
    
    # Schema safety check
    check_schema(processed_df)
    
    # 3. Training Preparation
    X = processed_df[FEATURE_COLS].fillna(0)
    y = processed_df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Calculate scale_pos_weight (base count / alt count)
    n_base = (y_train == 0).sum()
    n_alt = (y_train == 1).sum()
    pos_weight = n_base / n_alt if n_alt > 0 else 1.0
    logger.info(f"Train set balance: base={n_base}, alt={n_alt}. scale_pos_weight={pos_weight:.4f}")
    
    # 4. Model Training
    logger.info("Training XGBClassifier ...")
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=pos_weight,
        random_state=RANDOM_STATE,
        eval_metric='logloss',
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1
    )
    
    model.fit(X_train, y_train)
    
    # 5. Internal Evaluation
    logger.info("--- Internal Evaluation (80/20 Test Split) ---")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['base', 'alt']))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    loss = log_loss(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    print(f"\nLog Loss: {loss:.4f}")
    print(f"AUC-ROC:  {auc:.4f}")
    
    # 6. Save Model
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    model.save_model(args.model_path)
    logger.info(f"Model saved to {args.model_path}")
    
    # 7. Golden Set Validation
    if os.path.exists(args.golden):
        logger.info(f"\nRunning Golden Set Validation ({args.golden}) ...")
        golden_df = read_parquet_safe(args.golden)
        
        # Consistent label handling: Map 'match' to 'alt' if present
        if '3class_testlabels' in golden_df.columns:
            # Drop rows without 'alt' or 'base'
            # Normalize labels: match/alt -> alt, base -> base
            golden_df['temp_label'] = golden_df['3class_testlabels'].str.lower().replace({'match': 'alt'})
            golden_filtered = golden_df[golden_df['temp_label'].isin(['alt', 'base'])].copy()
            
            if not golden_filtered.empty:
                logger.info(f"Golden Set filtered to {len(golden_filtered)} binary rows.")
                golden_processed = engineer_features(golden_filtered)
                X_gold = golden_processed[FEATURE_COLS].fillna(0)
                y_gold = golden_filtered['temp_label'].map(label_map)
                
                y_gold_pred = model.predict(X_gold)
                y_gold_proba = model.predict_proba(X_gold)[:, 1]
                
                print("\nGolden Set Classification Report:")
                print(classification_report(y_gold, y_gold_pred, target_names=['base', 'alt']))
                print(f"Log Loss: {log_loss(y_gold, y_gold_proba):.4f}")
                print(f"AUC-ROC:  {roc_auc_score(y_gold, y_gold_proba):.4f}")
            else:
                logger.warning("Golden set contains no 'alt' or 'base' rows.")
        else:
            logger.warning(f"Golden dataset missing '3class_testlabels' column.")
            
    # 8. Save Results (Full input dataset predictions)
    logger.info(f"Predicting all rows in {args.input} ...")
    full_processed = engineer_features(df)
    X_full = full_processed[FEATURE_COLS].fillna(0)
    df['xgb_binary_proba'] = model.predict_proba(X_full)[:, 1]
    df['xgb_binary_pred'] = [ ('alt' if p >= 0.5 else 'base') for p in df['xgb_binary_proba']]
    
    # Keep only important columns for the output parquet
    keep_cols = [c for c in df.columns if not c.startswith('_')]
    df[keep_cols].to_parquet(args.output, index=False)
    logger.info(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
