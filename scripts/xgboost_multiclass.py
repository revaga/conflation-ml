"""
Multiclass XGBoost (none / alt / base / both) with hyperparameter tuning.
=====================================================================================
Refactored to use centralized logic from scripts/features.py and scripts/labels.py.
"""

import os
import warnings
import argparse
import joblib
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Centralized imports
from scripts.features import engineer_features
from scripts.labels import recalculate_4class_label, ATTR_ATTRS
from scripts.parquet_io import read_parquet_safe
from scripts.schema import validate_phase3_output

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
INPUT_PATH = "data/phase3_slm_labeled.parquet"
GOLDEN_PATH = "data/golden_dataset_200.parquet"
NEGATIVES_PATH = "data/negative_samples.parquet"
OUTPUT_PATH = "data/xgboost_multiclass_results.parquet"
MODEL_PATH = "data/models/multiclass_xgb.joblib"
RANDOM_STATE = 42
LABEL_COL = "label_4class"
CLASS_ORDER = ("none", "alt", "base", "both")
MAP_TO_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}
IDX_TO_MAP = {i: c for i, c in enumerate(CLASS_ORDER)}

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
    "feat_name_addr_sim_product",
    "feat_avg_similarity",
]

def main():
    parser = argparse.ArgumentParser(description="Multiclass XGBoost Classifier")
    parser.add_argument("--predict-only", action="store_true", help="Load existing model and predict")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Multiclass XGBoost (none/alt/base/both)")
    logger.info("=" * 70)

    # 1. Load Data
    df = read_parquet_safe(INPUT_PATH)
    logger.info(f"Loaded {len(df)} rows from {INPUT_PATH}")
    validate_phase3_output(df)
    
    golden = read_parquet_safe(GOLDEN_PATH) if os.path.exists(GOLDEN_PATH) else pd.DataFrame()
    
    # 2. Join Golden Labels (if any) and flag them
    df["is_golden"] = False
    if not golden.empty:
        golden_subset = golden[["id", "3class_testlabels"]].drop_duplicates("id")
        df = df.merge(golden_subset, on="id", how="left")
        df["is_golden"] = df["3class_testlabels"].notna()

    # 3. Feature Engineering
    logger.info("Engineering features ...")
    df = engineer_features(df)
    X_all = df[FEATURE_COLS].fillna(0).values

    # 4. Model Loading or Training
    clf = None
    if args.predict_only:
        if os.path.exists(MODEL_PATH):
            logger.info(f"Loading model from {MODEL_PATH} ...")
            clf = joblib.load(MODEL_PATH)
        else:
            logger.warning(f"Model not found at {MODEL_PATH}. Switching to training mode.")

    if clf is None:
        # Apply Labels to main dataset
        logger.info("Applying labels (recalculating 4-class from winners) ...")
        df[LABEL_COL] = df.apply(recalculate_4class_label, axis=1)

        # 3b. Load and Incorporate Negative Samples
        if os.path.exists(NEGATIVES_PATH):
            logger.info(f"Loading synthetic negative samples from {NEGATIVES_PATH} ...")
            neg_df = pd.read_parquet(NEGATIVES_PATH)
            neg_df["is_golden"] = False
            neg_df[LABEL_COL] = "none"
            # Ensure we have engineered features for negatives too
            if "feat_avg_similarity" not in neg_df.columns:
                neg_df = engineer_features(neg_df)
            
            # Combine Main (non-golden) + Negatives
            # train_pool = pd.concat([df[~df["is_golden"]], neg_df], ignore_index=True)
            # logger.info(f"Added {len(neg_df)} negative samples. Total training pool: {len(train_pool)}")
            train_pool = df[~df["is_golden"]].copy()
            logger.info(f"Using only SLM labeled data. Pool: {len(train_pool)}")
        else:
            train_pool = df[~df["is_golden"]].copy()

        train_df = train_pool.dropna(subset=[LABEL_COL]).copy()
        X_train = train_df[FEATURE_COLS].fillna(0).values
        y_train = train_df[LABEL_COL].map(MAP_TO_IDX).values

        logger.info(f"Setting up GridSearchCV for Multiclass XGBoost on {len(X_train)} rows...")
        base_clf = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=4,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        
        param_grid = {
            'max_depth': [3, 4],
            'learning_rate': [0.1],
            'n_estimators': [100]
        }
        
        # We use StratifiedKFold to ensure class balance across folds
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        
        grid_search = GridSearchCV(
            estimator=base_clf,
            param_grid=param_grid,
            scoring='f1_macro',
            cv=cv,
            verbose=1,
            n_jobs=-1
        )
        
        # MAP_TO_IDX: none=0, alt=1, base=2, both=3
        weight_dict = {0: 1.0, 1: 3.0, 2: 3.0, 3: 1.5}
        custom_weights = np.array([weight_dict[y] for y in y_train])

        logger.info("Starting hyperparameter sweep ...")
        grid_search.fit(X_train, y_train, sample_weight=custom_weights)
        
        logger.info(f"Best parameters found: {grid_search.best_params_}")
        clf = grid_search.best_estimator_
        
        # Save
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(clf, MODEL_PATH)
        logger.info(f"Model saved to {MODEL_PATH}")

    # 5. Holdout Evaluation (if golden exists)
    if not golden.empty:
        test_df = df[df["is_golden"]].copy()
        X_holdout = test_df[FEATURE_COLS].fillna(0).values
        y_holdout_raw = test_df["3class_testlabels"].str.lower().map({"alt": "alt", "both": "both", "base": "base"})
        y_holdout = y_holdout_raw.map(MAP_TO_IDX).fillna(0).astype(int).values
        
        y_pred = clf.predict(X_holdout)
        acc = accuracy_score(y_holdout, y_pred)
        f1 = f1_score(y_holdout, y_pred, average='macro')
        
        logger.info("Performance on Golden (3-class holdout):")
        logger.info(f"Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
        
        target_names = [CLASS_ORDER[i] for i in sorted(np.unique(np.concatenate([y_holdout, y_pred])))]
        logger.info("Classification Report:")
        logger.info("\n" + classification_report(y_holdout, y_pred, target_names=target_names))

    # 6. Global Prediction and Save
    logger.info("Predicting all rows ...")
    proba_all = clf.predict_proba(X_all)
    df["xgb_4class_pred"] = [IDX_TO_MAP[p] for p in np.argmax(proba_all, axis=1)]
    
    for i, c in enumerate(CLASS_ORDER):
        df[f"xgb_4class_proba_{c}"] = proba_all[:, i]
    
    # Save results
    keep_cols = [c for c in df.columns if not c.startswith("_")]
    df[keep_cols].to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"Results saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
