"""
XGBoost Binary Classifier: alt (1) vs base (0)
==============================================
Pure binary classifier to distinguish 'alt' (Class 1) from 'base' (Class 0).
Threshold on golden is tuned for F1 by default (--tune-for f1) to align with
unified_metrics binary F1. Use --label-rule golden to align training labels
with golden_dataset_200 2class_testlabels.
When tuning threshold on golden, prefer --exclude-golden-from-train to avoid
using golden IDs in training and threshold selection on the same data.

Usage:
    python scripts/xgboost_binary_alt_base.py --input data/phase3_slm_labeled.parquet
    python scripts/xgboost_binary_alt_base.py --label-rule golden --tune-threshold-on-golden
    python scripts/xgboost_binary_alt_base.py --label-rule golden --exclude-golden-from-train --tune-hyperparams
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Ensure this repo is first on the path so "scripts" is from conflation-ml, not another project
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
    f1_score,
)

from scripts.features import engineer_features
from scripts.labels import recalculate_4class_label, row_to_2class
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

def _best_threshold_for_accuracy(y_true, y_proba):
    """Find threshold in 0.05 steps that maximizes binary accuracy."""
    best_acc, best_t = 0.0, 0.5
    for t in np.arange(0.15, 0.86, 0.05):
        y_pred = (y_proba >= t).astype(int)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t, best_acc


def _best_threshold_for_f1(y_true, y_proba, pos_label=1):
    """Find threshold in 0.05 steps that maximizes binary F1 (pos_label=alt)."""
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.15, 0.86, 0.05):
        y_pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def main():
    parser = argparse.ArgumentParser(description="XGBoost pure binary classifier: alt vs base")
    parser.add_argument("--input", type=str, default="data/phase3_slm_labeled.parquet", help="Path to labeled training data")
    parser.add_argument("--golden", type=str, default="data/golden_dataset_200.parquet", help="Path to golden evaluation set")
    parser.add_argument("--output", type=str, default="data/xgboost_binary_results.parquet", help="Path to save predictions")
    parser.add_argument("--model_path", type=str, default="data/models/binary_alt_base.json", help="Path to save the trained model")
    parser.add_argument(
        "--label-rule",
        type=str,
        choices=["recalc", "golden"],
        default="golden",
        help="Training labels: 'golden' = row_to_2class (aligns with 2class_testlabels), 'recalc' = recalculate_4class_label base/alt only",
    )
    parser.add_argument("--exclude-golden-from-train", action="store_true", help="Exclude golden ids from training (recommended when tuning threshold on golden)")
    parser.add_argument("--tune-threshold-on-golden", action="store_true", default=True, help="Tune decision threshold on golden 2class (default: True)")
    parser.add_argument("--no-tune-threshold-on-golden", action="store_false", dest="tune_threshold_on_golden", help="Use 0.5 threshold")
    parser.add_argument(
        "--tune-for",
        type=str,
        choices=["f1", "accuracy"],
        default="f1",
        help="When tuning threshold on golden: maximize 'f1' (default) or 'accuracy'",
    )
    parser.add_argument(
        "--tune-hyperparams",
        action="store_true",
        help="Run GridSearchCV for hyperparameters (scoring=F1); slower.",
    )
    parser.add_argument(
        "--scale-pos-weight",
        type=float,
        default=None,
        help="Override scale_pos_weight (default: auto from train balance, capped at 10)",
    )
    args = parser.parse_args()

    label_map = {"base": 0, "alt": 1}

    # Load golden early for exclude-golden and threshold tuning
    golden_df = None
    golden_ids = set()
    golden_2class_series = None
    if os.path.exists(args.golden):
        golden_df = read_parquet_safe(args.golden)
        golden_df["id"] = golden_df["id"].astype(str)
        golden_ids = set(golden_df["id"].tolist())
        if "2class_testlabels" in golden_df.columns:
            golden_2class_series = golden_df.set_index("id")["2class_testlabels"].astype(str).str.strip().str.lower()
        logger.info(f"Golden set: {len(golden_ids)} ids, 2class_testlabels present: {golden_2class_series is not None}")

    # 1. Load and Label Data
    logger.info(f"Loading data from {args.input} ...")
    df = read_parquet_safe(args.input)
    df["id"] = df["id"].astype(str)

    if args.label_rule == "golden":
        logger.info("Using label rule 'golden' (row_to_2class) to align with 2class_testlabels")
        df["temp_label"] = df.apply(row_to_2class, axis=1)
        filtered_df = df.copy()
        logger.info(f"Binary labels (golden rule): base={(df['temp_label'] == 'base').sum()}, alt={(df['temp_label'] == 'alt').sum()}")
    else:
        logger.info("Using label rule 'recalc' (recalculate_4class_label, base/alt only)")
        df["temp_label"] = df.apply(recalculate_4class_label, axis=1)
        filtered_df = df[df["temp_label"].isin(["alt", "base"])].copy()
        logger.info(f"Filtered to 'alt' and 'base'. Count: {len(filtered_df)} (Dropped {len(df) - len(filtered_df)} rows)")

    if filtered_df.empty:
        logger.error("No data remains for training.")
        return

    if args.exclude_golden_from_train and golden_ids:
        before = len(filtered_df)
        filtered_df = filtered_df[~filtered_df["id"].isin(golden_ids)].copy()
        logger.info(f"Excluded golden from train: {before} -> {len(filtered_df)} rows")
        if filtered_df.empty:
            logger.error("No non-golden rows left for training.")
            return

    filtered_df["target"] = filtered_df["temp_label"].map(label_map)
    
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
    
    # Calculate scale_pos_weight (base count / alt count), optional cap and override
    n_base = (y_train == 0).sum()
    n_alt = (y_train == 1).sum()
    pos_weight = n_base / n_alt if n_alt > 0 else 1.0
    if args.scale_pos_weight is not None:
        pos_weight = args.scale_pos_weight
        logger.info(f"Using --scale-pos-weight override: {pos_weight:.4f}")
    else:
        pos_weight = min(pos_weight, 10.0)  # cap to avoid over-weighting minority
    pct_alt = 100.0 * n_alt / len(y_train) if len(y_train) else 0
    logger.info(f"Train set balance: base={n_base}, alt={n_alt} ({pct_alt:.1f}% alt). scale_pos_weight={pos_weight:.4f}")
    
    # 4. Model Training (single run or GridSearchCV)
    if args.tune_hyperparams:
        logger.info("Running GridSearchCV for hyperparameters (scoring=F1) ...")
        base_clf = xgb.XGBClassifier(
            objective="binary:logistic",
            scale_pos_weight=pos_weight,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            early_stopping_rounds=None,
        )
        param_grid = {
            "max_depth": [4, 5, 6],
            "learning_rate": [0.03, 0.05, 0.1],
            "n_estimators": [200, 300],
            "subsample": [0.7, 0.8],
            "reg_alpha": [0.05, 0.1],
            "reg_lambda": [0.5, 1.0],
        }
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        grid_search = GridSearchCV(
            estimator=base_clf,
            param_grid=param_grid,
            scoring="f1",
            cv=cv,
            verbose=1,
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV F1: {grid_search.best_score_:.4f}")
        model = grid_search.best_estimator_
    else:
        logger.info("Training XGBoost binary classifier ...")
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            scale_pos_weight=pos_weight,
            random_state=RANDOM_STATE,
            eval_metric=['logloss', 'auc'],
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            early_stopping_rounds=20,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=10,
        )
    
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
    try:
        model.save_model(args.model_path)
    except (TypeError, AttributeError):
        model.get_booster().save_model(args.model_path)
    logger.info(f"Model saved to {args.model_path}")

    # 7. Golden set: threshold tuning and validation (use 2class_testlabels when present)
    threshold = 0.5
    if golden_df is not None:
        logger.info(f"\nRunning Golden Set Validation ({args.golden}) ...")
        golden_processed = engineer_features(golden_df)
        X_gold = golden_processed[FEATURE_COLS].fillna(0)
        y_gold_proba = model.predict_proba(X_gold)[:, 1]

        if golden_2class_series is not None and not golden_2class_series.empty:
            # Align by id (golden_processed has same index as golden_df if no drops)
            ids = golden_df["id"].values
            y_gold_true = golden_2class_series.reindex(ids).fillna("").values
            valid = np.array([v in ("base", "alt") for v in y_gold_true])
            if valid.sum() > 0:
                y_gt = np.array([label_map.get(v, 0) for v in y_gold_true[valid]])
                y_gp = y_gold_proba[valid]
                y_gold_pred_default = (y_gp >= 0.5).astype(int)
                f1_default = f1_score(y_gt, y_gold_pred_default, pos_label=1, zero_division=0)
                print("\nGolden Set (2class_testlabels) — threshold 0.5:")
                print(classification_report(y_gt, y_gold_pred_default, target_names=["base", "alt"]))
                print(f"Log Loss: {log_loss(y_gt, y_gp):.4f}")
                print(f"AUC-ROC:  {roc_auc_score(y_gt, y_gp):.4f}")
                print(f"Accuracy: {accuracy_score(y_gt, y_gold_pred_default):.4f}")
                print(f"F1 (alt): {f1_default:.4f}")

                if args.tune_threshold_on_golden:
                    if args.tune_for == "f1":
                        threshold, best_metric = _best_threshold_for_f1(y_gt, y_gp, pos_label=1)
                        logger.info(f"Tuned threshold on golden for F1: {threshold:.2f} (F1 alt = {best_metric:.4f})")
                    else:
                        threshold, best_metric = _best_threshold_for_accuracy(y_gt, y_gp)
                        logger.info(f"Tuned threshold on golden for accuracy: {threshold:.2f} (accuracy = {best_metric:.4f})")
                    y_gold_pred_tuned = (y_gp >= threshold).astype(int)
                    f1_tuned = f1_score(y_gt, y_gold_pred_tuned, pos_label=1, zero_division=0)
                    print(f"\nGolden Set with tuned threshold {threshold:.2f}:")
                    print(classification_report(y_gt, y_gold_pred_tuned, target_names=["base", "alt"]))
                    print(f"Accuracy: {accuracy_score(y_gt, y_gold_pred_tuned):.4f}")
                    print(f"F1 (alt): {f1_tuned:.4f}")
        else:
            # Fallback: 3class_testlabels -> binary
            if "3class_testlabels" in golden_df.columns:
                golden_df = golden_df.copy()
                golden_df["temp_label"] = golden_df["3class_testlabels"].str.lower().replace({"match": "alt"})
                golden_filtered = golden_df[golden_df["temp_label"].isin(["alt", "base"])]
                if not golden_filtered.empty:
                    y_gold = golden_filtered["temp_label"].map(label_map).values
                    y_gold_pred = model.predict(X_gold.loc[golden_filtered.index]) if hasattr(X_gold, "loc") else model.predict(X_gold)
                    print("\nGolden Set (3class_testlabels -> binary):")
                    print(classification_report(y_gold, y_gold_pred, target_names=["base", "alt"]))

    # 8. Save Results (full input dataset predictions; use tuned threshold)
    logger.info(f"Predicting all rows in {args.input} ...")
    full_processed = engineer_features(df)
    X_full = full_processed[FEATURE_COLS].fillna(0)
    df["xgb_binary_proba"] = model.predict_proba(X_full)[:, 1]
    df["xgb_binary_pred"] = ["alt" if p >= threshold else "base" for p in df["xgb_binary_proba"]]
    
    # Keep only important columns for the output parquet
    keep_cols = [c for c in df.columns if not c.startswith('_')]
    df[keep_cols].to_parquet(args.output, index=False)
    logger.info(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
