"""
Multiclass XGBoost (none / alt / base / both) with hyperparameter tuning.
=====================================================================================
4-class model (none/alt/base/both). Labels from recalculate_4class_label (attr_*_winner).
GridSearchCV with f1_macro; golden rows excluded from training pool. Uses centralized
logic from scripts/features.py and scripts/labels.py.
"""

import os
import sys
import warnings
import argparse
import joblib
import logging
from pathlib import Path

# Ensure this repo is first on the path so "scripts" is from conflation-ml
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

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
SYNTHETIC_KIMI_PATH = "data/synthetic_4class_kimi.parquet"
RANDOM_STATE = 42
LABEL_COL = "label_4class"
CLASS_ORDER = ("none", "alt", "base", "both")
MAP_TO_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}
IDX_TO_MAP = {i: c for i, c in enumerate(CLASS_ORDER)}

# Aligned with engineer_features() and xgboost_binary_alt_base.py
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


def check_schema(df: pd.DataFrame) -> None:
    """Ensure all required FEATURE_COLS exist in the dataframe."""
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns in engineered dataframe: {missing}")
    logger.info("Schema check PASSED: All FEATURE_COLS are present.")


def main():
    parser = argparse.ArgumentParser(description="Multiclass XGBoost Classifier")
    parser.add_argument("--predict-only", action="store_true", help="Load existing model and predict")
    parser.add_argument(
        "--train-on-synthetic-kimi",
        action="store_true",
        help="Train classifier using synthetic 4-class Kimi dataset instead of SLM pool",
    )
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
    check_schema(df)
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
        # Apply Labels to main dataset (used for non-synthetic training and for predictions)
        logger.info("Applying labels (recalculating 4-class from winners) on main dataset ...")
        df[LABEL_COL] = df.apply(recalculate_4class_label, axis=1)

        # --- Select training data ---
        if args.train_on_synthetic_kimi and os.path.exists(SYNTHETIC_KIMI_PATH):
            logger.info(f"Training on synthetic 4-class Kimi dataset: {SYNTHETIC_KIMI_PATH}")
            train_df = read_parquet_safe(SYNTHETIC_KIMI_PATH)
            # Ensure labels exist
            if LABEL_COL not in train_df.columns:
                logger.info("LABEL_COL not found in synthetic set; recalculating 4-class labels ...")
                train_df[LABEL_COL] = train_df.apply(recalculate_4class_label, axis=1)
            # Ensure features exist
            if "feat_avg_similarity" not in train_df.columns:
                logger.info("Engineering features for synthetic Kimi training data ...")
                train_df = engineer_features(train_df)
            check_schema(train_df)
        else:
            # 3b. Load and Incorporate Negative Samples (if any), using non-golden SLM pool
            if os.path.exists(NEGATIVES_PATH):
                logger.info(f"Loading synthetic negative samples from {NEGATIVES_PATH} ...")
                neg_df = pd.read_parquet(NEGATIVES_PATH)
                neg_df["is_golden"] = False
                neg_df[LABEL_COL] = "none"
                # Ensure we have engineered features for negatives too
                if "feat_avg_similarity" not in neg_df.columns:
                    neg_df = engineer_features(neg_df)
                
                train_pool = df[~df["is_golden"]].copy()
                logger.info(f"Using only SLM labeled data. Pool: {len(train_pool)}")
            else:
                train_pool = df[~df["is_golden"]].copy()

            train_df = train_pool.dropna(subset=[LABEL_COL]).copy()
            check_schema(train_df)
        X_train = train_df[FEATURE_COLS].fillna(0).values
        y_train = train_df[LABEL_COL].map(MAP_TO_IDX).values

        logger.info(f"Setting up GridSearchCV for Multiclass XGBoost on {len(X_train)} rows...")
        base_clf = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=4,
            random_state=RANDOM_STATE,
            eval_metric="mlogloss",
        )
        
        param_grid = {
            "max_depth": [3, 4, 5],
            "learning_rate": [0.05, 0.1, 0.2],
            "n_estimators": [100, 150],
            "subsample": [0.8, 1.0],
            "reg_alpha": [0.05, 0.1],
            "reg_lambda": [0.5, 1.0],
        }

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        grid_search = GridSearchCV(
            estimator=base_clf,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=cv,
            verbose=1,
            n_jobs=-1,
        )

        # Data-driven class weights (balanced, capped) so minority classes get higher weight
        classes = np.unique(y_train)
        balanced = compute_class_weight(
            "balanced", classes=classes, y=y_train
        )
        weight_dict = dict(zip(classes, balanced))
        max_weight = 10.0
        for k in weight_dict:
            weight_dict[k] = min(float(weight_dict[k]), max_weight)
        custom_weights = np.array([weight_dict[y] for y in y_train])
        train_counts = np.bincount(y_train, minlength=len(CLASS_ORDER))
        logger.info(
            f"Train class distribution: {dict(zip(CLASS_ORDER, train_counts))}; "
            f"weights (capped at {max_weight}): {dict((CLASS_ORDER[i], round(w, 2)) for i, w in weight_dict.items())}"
        )

        logger.info("Starting hyperparameter sweep ...")
        grid_search.fit(X_train, y_train, sample_weight=custom_weights)

        logger.info(f"Best parameters found: {grid_search.best_params_}")
        logger.info(f"Best CV F1-macro score: {grid_search.best_score_:.4f}")
        clf = grid_search.best_estimator_
        
        # Save
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(clf, MODEL_PATH)
        logger.info(f"Model saved to {MODEL_PATH}")

    # 5. Holdout Evaluation (if golden exists)
    if not golden.empty:
        test_df = df[df["is_golden"]].copy()
        X_holdout = test_df[FEATURE_COLS].fillna(0).values
        # Golden has 3-class labels only (alt/both/base); "none" is not in golden holdout
        y_holdout_raw = test_df["3class_testlabels"].str.lower().map({"alt": "alt", "both": "both", "base": "base"})
        y_holdout = y_holdout_raw.map(MAP_TO_IDX).fillna(0).astype(int).values

        logger.info("Golden holdout uses 3-class labels (alt/both/base only; 'none' not in golden).")
        y_pred = clf.predict(X_holdout)
        acc = accuracy_score(y_holdout, y_pred)
        f1 = f1_score(y_holdout, y_pred, average="macro")

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
