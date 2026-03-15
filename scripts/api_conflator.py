"""
Real-Time API Class
=====================================================================================
Wraps the inference components into a stateful `PlacesConflator` class that 
processes single JSON dictionaries without writing to Parquet.
"""

import os
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from scripts.features import engineer_features
from scripts.xgboost_multiclass import FEATURE_COLS
from scripts.normalization import process_addresses, standardize_phone, normalize_website

logger = logging.getLogger(__name__)

class PlacesConflator:
    def __init__(self, model_path: str = "data/models/refiner_3class.json"):
        """Initialize the conflator and load the pre-trained 3-Class XGBoost model."""
        self.model_path = model_path
        self.booster = None
        
        if not os.path.exists(self.model_path):
            logger.warning(f"Model not found at {self.model_path}. Inference will fail.")
        else:
            self.booster = xgb.Booster()
            self.booster.load_model(self.model_path)
            logger.info("PlacesConflator initialized with 3-Class XGBoost model.")

    def predict(self, record: dict) -> dict:
        """
        Process a single JSON dictionary representing an incoming conflation pair.
        Returns a dict containing the prediction (`alt`, `base`, `both`) and probabilities.
        """
        if self.booster is None:
            raise RuntimeError("Model is not loaded. Cannot run predict().")

        # 1. Convert to 1-row DataFrame
        df = pd.DataFrame([record])
        
        # 1a. Run Phase 1-style Normalization
        from rapidfuzz import fuzz

        df = process_addresses(df)

        # Safely extract single values allowing for potential missing keys
        df["phones"] = df.get("phones", "")
        df["base_phones"] = df.get("base_phones", "")
        df["websites"] = df.get("websites", "")
        df["base_websites"] = df.get("base_websites", "")

        df["norm_conflated_phone"] = df["phones"].apply(standardize_phone)
        df["norm_base_phone"] = df["base_phones"].apply(standardize_phone)

        df["norm_conflated_website"] = df["websites"].apply(normalize_website)
        df["norm_base_website"] = df["base_websites"].apply(normalize_website)

        def compute_sim(row, c1, c2, scorer=fuzz.ratio):
            v1 = row.get(c1, "")
            v2 = row.get(c2, "")
            if not v1 or not v2:
                return 0
            return scorer(v1, v2)

        df["addr_similarity_ratio"] = df.apply(
            lambda x: compute_sim(x, "norm_conflated_addr", "norm_base_addr", fuzz.ratio), axis=1
        )
        df["addr_token_sort"] = df.apply(
            lambda x: compute_sim(x, "norm_conflated_addr", "norm_base_addr", fuzz.token_sort_ratio), axis=1
        )
        df["phone_similarity"] = df.apply(
            lambda x: compute_sim(x, "norm_conflated_phone", "norm_base_phone", fuzz.ratio), axis=1
        )
        df["website_similarity"] = df.apply(
            lambda x: compute_sim(x, "norm_conflated_website", "norm_base_website", fuzz.ratio), axis=1
        )
        
        # 2. Engineer Phase 2/3 Features
        df = engineer_features(df, validate_urls=False, validate_phones=False)
        
        # 3. Extract Feature Vector
        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0
                
        X = df[FEATURE_COLS].fillna(0.0).values
        dmatrix = xgb.DMatrix(X)
        
        # 4. Inference
        proba = self.booster.predict(dmatrix)[0]
        
        # Map back to classes (0:alt, 1:base, 2:both)
        IDX_MAP = {0: "alt", 1: "base", 2: "both"}
        pred_idx = int(np.argmax(proba))
        prediction = IDX_MAP[pred_idx]
        
        return {
            "prediction": prediction,
            "probabilities": {
                "alt": float(proba[0]),
                "base": float(proba[1]),
                "both": float(proba[2])
            }
        }
