"""
Train one XGBoost model per attribute (name, phone, web, address, category)
and write per-attribute predictions to a parquet file.

Usage (from repo root):
    python -m xgboostmodels.train_per_attribute_xgb \
        --input data/phase3_slm_labeled.parquet \
        --output data/xgboost_per_attr_results.parquet \
        --models-dir data/models
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, GridSearchCV

_REPO_ROOT = Path(__file__).resolve().parent.parent

import sys

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.features import engineer_features  # type: ignore
from scripts.labels import LABEL_4CLASS, ATTR_ATTRS  # type: ignore
from scripts.parquet_io import read_parquet_safe  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42

# Reuse the same feature set as the multiclass XGBoost model
FEATURE_COLS: List[str] = [
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

CLASS_ORDER = tuple(LABEL_4CLASS)  # ("none", "alt", "base", "both")
MAP_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CLASS_ORDER)}
IDX_TO_MAP: Dict[int, str] = {i: c for i, c in enumerate(CLASS_ORDER)}


def _check_schema(df: pd.DataFrame) -> None:
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns in engineered dataframe: {missing}")
    logger.info("Schema check PASSED: all FEATURE_COLS are present.")


def _normalize_attr_label(val) -> str:
    """Normalize attribute-level winner into LABEL_4CLASS space."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "none"
    v = str(val).strip().lower()
    if v in CLASS_ORDER:
        return v
    if v in ("a", "match", "m"):
        return "alt"
    if v == "b":
        return "base"
    if v == "t":
        return "both"
    return "none"


def train_one_attribute(
    df_features: pd.DataFrame,
    attr: str,
    models_dir: Path,
    tune_hyperparams: bool = True,
) -> xgb.XGBClassifier:
    label_col = f"attr_{attr}_winner"
    if label_col not in df_features.columns:
        raise KeyError(f"{label_col} not found in input data.")

    labels = df_features[label_col].apply(_normalize_attr_label)
    # Require a valid 4-class label
    mask = labels.isin(CLASS_ORDER)
    train_df = df_features[mask].copy()
    y = labels[mask].map(MAP_TO_IDX).values
    X = train_df[FEATURE_COLS].fillna(0).values

    if len(np.unique(y)) < 2:
        raise ValueError(f"Attribute '{attr}' has <2 classes after filtering; cannot train.")

    logger.info(
        "Training attribute '%s' on %d rows (class distribution: %s)",
        attr,
        len(train_df),
        {CLASS_ORDER[i]: int((y == i).sum()) for i in np.unique(y)},
    )

    base_clf = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(CLASS_ORDER),
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
    )

    if tune_hyperparams:
        param_grid = {
            "max_depth": [3, 4],
            "learning_rate": [0.05, 0.1],
            "n_estimators": [120, 180],
            "subsample": [0.8, 1.0],
            "reg_alpha": [0.05, 0.1],
            "reg_lambda": [0.5, 1.0],
        }
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        grid = GridSearchCV(
            estimator=base_clf,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=cv,
            verbose=1,
            n_jobs=-1,
        )
        grid.fit(X, y)
        clf: xgb.XGBClassifier = grid.best_estimator_
        logger.info(
            "Best params for '%s': %s (F1-macro=%.4f)",
            attr,
            grid.best_params_,
            grid.best_score_,
        )
    else:
        clf = base_clf
        clf.set_params(
            n_estimators=160,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            reg_alpha=0.1,
            reg_lambda=0.8,
        )
        clf.fit(X, y)

    # Simple in-sample report (for quick sanity check)
    y_pred = clf.predict(X)
    report = classification_report(
        y, y_pred, target_names=list(CLASS_ORDER), zero_division=0
    )
    logger.info("In-sample report for attribute '%s':\n%s", attr, report)

    # Save model
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"xgb_attr_{attr}.json"
    try:
        clf.save_model(str(model_path))
    except (TypeError, AttributeError):
        clf.get_booster().save_model(str(model_path))
    logger.info("Saved model for '%s' to %s", attr, model_path)

    return clf


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one XGBoost model per attribute and write per-attribute predictions."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/phase3_slm_labeled.parquet",
        help="Input SLM-labeled dataset with attr_*_winner columns.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/xgboost_per_attr_results.parquet",
        help="Output parquet with per-attribute predictions.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="data/models",
        help="Directory to store per-attribute XGBoost models.",
    )
    parser.add_argument(
        "--no-tune-hyperparams",
        action="store_true",
        help="Disable GridSearchCV hyperparameter tuning (faster).",
    )
    args = parser.parse_args()

    models_dir = (_REPO_ROOT / args.models_dir).resolve()

    logger.info("Loading data from %s", args.input)
    df = read_parquet_safe(args.input)
    df["id"] = df["id"].astype(str)

    logger.info("Engineering features ...")
    df_feat = engineer_features(df)
    _check_schema(df_feat)

    attr_models: Dict[str, xgb.XGBClassifier] = {}
    for attr in ATTR_ATTRS:
        logger.info("=" * 60)
        logger.info("Attribute: %s", attr)
        logger.info("=" * 60)
        try:
            clf = train_one_attribute(
                df_feat,
                attr=attr,
                models_dir=models_dir,
                tune_hyperparams=not args.no_tune_hyperparams,
            )
            attr_models[attr] = clf
        except Exception as e:
            logger.error("Failed to train attribute '%s': %s", attr, e)

    if not attr_models:
        logger.error("No attribute models were successfully trained; aborting.")
        return

    # Predict for all rows for each attribute and attach to original df
    X_all = df_feat[FEATURE_COLS].fillna(0).values
    for attr, clf in attr_models.items():
        logger.info("Predicting attribute '%s' for all rows ...", attr)
        proba = clf.predict_proba(X_all)
        pred_idx = np.argmax(proba, axis=1)
        pred_labels = [IDX_TO_MAP[i] for i in pred_idx]

        df[f"xgb_attr_{attr}_pred"] = pred_labels
        for i, c in enumerate(CLASS_ORDER):
            df[f"xgb_attr_{attr}_proba_{c}"] = proba[:, i]

    # Keep non-temporary columns only
    keep_cols = [c for c in df.columns if not c.startswith("_")]
    output_path = _REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[keep_cols].to_parquet(output_path, index=False)
    logger.info("Per-attribute XGBoost results written to %s", output_path)


if __name__ == "__main__":
    main()

