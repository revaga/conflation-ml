"""
Stage 2 (Refiner) 3-Class XGBoost Model
=====================================================================================
Trains a 3-class classifier ('base', 'alt', 'both') on records with 4-class label in {alt, base, both} only.
Rows with label 'none' are excluded from training (no none considerations).
"""

import os
import argparse
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Centralized imports
from scripts.features import engineer_features
from scripts.labels import recalculate_4class_label
from scripts.parquet_io import read_parquet_safe
from scripts.schema import validate_phase3_output

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
INPUT_PATH = "data/phase3_slm_labeled.parquet"
GOLDEN_PATH = "data/golden_dataset_200.parquet"
OUTPUT_MODEL_PATH = "data/models/refiner_3class.json"
RANDOM_STATE = 42

LABEL_COL = "label_4class"
CLASS_ORDER = ("alt", "base", "both")
MAP_TO_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}
IDX_TO_MAP = {i: c for i, c in enumerate(CLASS_ORDER)}

from scripts.xgboost_multiclass import FEATURE_COLS

def main():
    parser = argparse.ArgumentParser(description="Train Stage 2 3-class Refiner Model")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Stage 2 Refiner Training (alt/base/both)")
    logger.info("=" * 70)

    # 1. Load Data
    df = read_parquet_safe(INPUT_PATH)
    logger.info(f"Loaded {len(df)} rows from {INPUT_PATH}")
    validate_phase3_output(df)
    
    # 2. Apply Truth Labels
    df[LABEL_COL] = df.apply(recalculate_4class_label, axis=1)

    # 3. Filter for only 3-class training data (exclude none; no Hard Nones injection)
    train_df = df[df[LABEL_COL].isin(["alt", "base", "both"])].copy()
    logger.info(f"Refiner training shape (alt/base/both only, none excluded): {len(train_df)}")

    # 5. Feature Engineering
    logger.info("Engineering features ...")
    train_df = engineer_features(train_df)
    
    X_train = train_df[FEATURE_COLS].fillna(0).values
    y_train = train_df[LABEL_COL].map(MAP_TO_IDX).values

    # 6. Train the Model using GridSearchCV
    logger.info(f"Setting up GridSearchCV for 3-Class Refiner on {len(X_train)} rows...")
    base_clf = xgb.XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric='mlogloss'
    )
    
    param_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'n_estimators': [100, 150]
    }
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    
    grid_search = GridSearchCV(
        estimator=base_clf,
        param_grid=param_grid,
        scoring='f1_macro',
        cv=cv,
        verbose=1,
        n_jobs=-1
    )
    
    # Custom sample weights: alt and base are often under-recalled compared to 'both'
    weight_dict = {MAP_TO_IDX['alt']: 1.5, MAP_TO_IDX['base']: 1.5, MAP_TO_IDX['both']: 1.0}
    custom_weights = np.array([weight_dict[y] for y in y_train])

    grid_search.fit(X_train, y_train, sample_weight=custom_weights)
    
    logger.info(f"Best parameters found: {grid_search.best_params_}")
    logger.info(f"Best CV F1-macro score: {grid_search.best_score_:.4f}")
    
    clf = grid_search.best_estimator_

    # 7. Save native JSON format
    os.makedirs(os.path.dirname(OUTPUT_MODEL_PATH), exist_ok=True)
    # The sklearn wrapper can sometimes throw a TypeError internally when saving to JSON directly on newer versions.
    # Extract the native booster and save that instead.
    booster = clf.get_booster()
    booster.save_model(OUTPUT_MODEL_PATH)
    logger.info(f"Model successfully saved to {OUTPUT_MODEL_PATH}")

if __name__ == "__main__":
    main()
