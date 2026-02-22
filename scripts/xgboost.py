"""
XGBoost-style Gradient Boosted Classifier for Places Attribute Conflation
=====================================================================================
Uses data/golden_dataset_100.parquet as the truth/golden set for training labels.
Uses the broader data/project_a_samples.parquet (via phase1_processed) for input,
excluding the first 100 rows from the broader set so the golden set is not duplicated.
Leverages pre-computed similarity scores, engineers advanced features, and trains a
gradient-boosted decision-tree classifier.

Run from project root:
    python scripts/xgboost.py
"""

import json
import re
import warnings
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from website_validator import verify_website
from phonenumber_validator import validate_phone_number

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Processed version of project_a_samples (phase1 adds norm_*, addr_similarity, etc.)
INPUT_PATH = "data/phase1_processed.parquet"
# Truth/golden labels: first 100 rows of project_a_samples, human-labeled (golden_label: base/alt/both/none)
GOLDEN_PATH = "data/golden_dataset_100.parquet"
# First N rows of project_a_samples are the golden set; broader input is the rest
GOLDEN_NROWS = 100
OUTPUT_PATH = "data/xgboost_results.parquet"
LABEL_COL = "final_label"
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1.  JSON Helpers
# ---------------------------------------------------------------------------

def safe_json(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return {}
    if isinstance(x, (dict, list)):
        return x
    try:
        return json.loads(str(x))
    except (json.JSONDecodeError, TypeError):
        return {}

def extract_primary_name(val):
    obj = safe_json(val)
    if isinstance(obj, dict):
        return obj.get("primary", "")
    return ""

def extract_primary_category(val):
    obj = safe_json(val)
    if isinstance(obj, dict):
        return obj.get("primary", "")
    return ""

def extract_sources_info(val):
    arr = safe_json(val)
    if not isinstance(arr, list):
        return 0, None, set()
    count = len(arr)
    datasets = set()
    latest_dt = None
    for item in arr:
        if not isinstance(item, dict): continue
        ds = item.get("dataset", "")
        if ds: datasets.add(ds.lower())
        ut = item.get("update_time")
        if ut:
            try:
                dt = datetime.fromisoformat(ut.replace("Z", "+00:00"))
                if latest_dt is None or dt > latest_dt: latest_dt = dt
            except: pass
    return count, latest_dt, datasets

# ---------------------------------------------------------------------------
# 2.  Feature Engineering
# ---------------------------------------------------------------------------

def _check_website(url) -> bool:
    """Return True if the URL is non-empty and resolves (status < 400)."""
    if not url or (isinstance(url, float) and np.isnan(url)):
        return False
    url = str(url).strip()
    if not url:
        return False
    # Ensure URL has a scheme
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    is_valid, _ = verify_website(url)
    return is_valid

def _check_phone_number(phone_number: str) -> bool:
    """Return True if the phone number is non-empty and valid."""
    if not phone_number or (isinstance(phone_number, float) and np.isnan(phone_number)):
        return False
    phone_number = str(phone_number).strip()
    if not phone_number:
        return False
    valid, _ = validate_phone_number(phone_number)
    return valid

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("  Extracting names and categories from JSON ...")
    df["_name"] = df["names"].apply(extract_primary_name)
    df["_base_name"] = df["base_names"].apply(extract_primary_name)
    df["_category"] = df["categories"].apply(extract_primary_category)
    df["_base_category"] = df["base_categories"].apply(extract_primary_category)

    print("  Extracting source metadata ...")
    src_info = df["sources"].apply(extract_sources_info)
    df["_src_count"] = src_info.apply(lambda x: x[0])
    df["_src_latest"] = src_info.apply(lambda x: x[1])
    df["_src_datasets"] = src_info.apply(lambda x: x[2])

    base_src_info = df["base_sources"].apply(extract_sources_info)
    df["_base_src_count"] = base_src_info.apply(lambda x: x[0])
    df["_base_src_latest"] = base_src_info.apply(lambda x: x[1])

    print("  Calculating Existence and Completeness features ...")

    # A. Trust (Existence Confidence)
    df["feat_existence_conf_delta"] = df["confidence"] - df["base_confidence"]
    df["feat_match_exists_score"] = df["confidence"]
    df["feat_base_exists_score"] = df["base_confidence"]
    
    # B. Completeness (Attribute Richness)
    df["feat_match_addr_len"] = df["norm_conflated_addr"].fillna("").str.len()
    df["feat_base_addr_len"] = df["norm_base_addr"].fillna("").str.len()
    df["feat_addr_richness_delta"] = df["feat_match_addr_len"] - df["feat_base_addr_len"]
    
    # Check for presence of Phone/Website (binary richness)
    df["feat_match_has_phone"] = (df["norm_conflated_phone"].fillna("") != "").astype(int)
    df["feat_base_has_phone"] = (df["norm_base_phone"].fillna("") != "").astype(int)
    df["feat_match_has_web"] = (df["norm_conflated_website"].fillna("") != "").astype(int)
    df["feat_base_has_web"] = (df["norm_base_website"].fillna("") != "").astype(int)

    # Website Validation (is the website actually reachable?)
    print("  Validating website URLs (this may take a while) ...")
    df["feat_match_web_valid"] = df["norm_conflated_website"].apply(_check_website).astype(int)
    df["feat_base_web_valid"] = df["norm_base_website"].apply(_check_website).astype(int)


    # Phone Validation (is the phone number actually valid?)
    print("  Validating phone numbers (this may take a while) ...")
    df["feat_match_phone_valid"] = df["norm_conflated_phone"].apply(_check_phone_number).astype(int)
    df["feat_base_phone_valid"] = df["norm_base_phone"].apply(_check_phone_number).astype(int)


    # C. Similarity
    df["feat_name_similarity"] = df.apply(
        lambda r: fuzz.token_sort_ratio(r["_name"], r["_base_name"]) / 100.0
        if r["_name"] and r["_base_name"] else 0.0, axis=1
    )
    df["feat_addr_similarity"] = df["addr_similarity_ratio"] / 100.0

    # D. Source Signal
    df["feat_is_msft_match"] = df["_src_datasets"].apply(lambda s: int("msft" in s) if isinstance(s, set) else 0)
    df["feat_is_meta_match"] = df["_src_datasets"].apply(lambda s: int("meta" in s) if isinstance(s, set) else 0)

    return df

# ---------------------------------------------------------------------------
# 3.  Hybrid Labeling (Golden + Heuristic)
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "feat_existence_conf_delta",
    "feat_match_exists_score",
    "feat_addr_richness_delta",
    "feat_match_has_phone",
    "feat_base_has_phone",
    "feat_match_has_web",
    "feat_base_has_web",
    "feat_match_web_valid",
    "feat_base_web_valid",
    "feat_name_similarity",
    "feat_addr_similarity",
    "feat_is_msft_match",
    "feat_is_meta_match"
]

def apply_labels(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Start with Heuristics (Fallback)
    # Higher focus on completeness now
    cond_pos = (
        (df["feat_name_similarity"] > 0.85) # Very likely same place
        & (
            (df["feat_addr_richness_delta"] > 5) | # Match is more descriptive
            (df["feat_match_has_phone"] > df["feat_base_has_phone"]) |
            (df["feat_match_has_web"] > df["feat_base_has_web"])
        )
    )
    
    # Base is clearly better (e.g. contains unit info while match doesn't)
    cond_neg = (
        (df["feat_addr_richness_delta"] < -10) | # Base is much longer (likely has suite)
        (df["feat_name_similarity"] < 0.40) # Different places entirely
    )

    df[LABEL_COL] = np.nan
    df.loc[cond_pos, LABEL_COL] = 1.0
    df.loc[cond_neg & ~cond_pos, LABEL_COL] = 0.0

    # 2. Overwrite with Golden Labels (The Ground Truth) from golden_dataset_100.parquet
    if os.path.exists(GOLDEN_PATH):
        print(f"  Integrating manual labels from {GOLDEN_PATH} ...")
        golden = pd.read_parquet(GOLDEN_PATH)
        # Map golden_label (base/alt/both/none) -> binary: base=0 (keep base), alt/both=1 (use match), none=drop
        def golden_to_binary(val):
            if pd.isna(val) or val is None or str(val).strip().lower() in ("none", ""):
                return np.nan
            v = str(val).strip().lower()
            if v == "base":
                return 0.0
            if v in ("alt", "match", "both"):
                return 1.0
            return np.nan
        golden[LABEL_COL] = golden["golden_label"].apply(golden_to_binary)
        # Keep only rows with a valid binary label for the map
        golden_labeled = golden.dropna(subset=[LABEL_COL])
        label_map = dict(zip(golden_labeled["id"], golden_labeled[LABEL_COL]))
        # Apply labels (will overwrite heuristics) for any row whose id is in the golden set
        df["is_golden"] = df["id"].isin(label_map.keys())
        for idx, row in df[df["is_golden"]].iterrows():
            df.at[idx, LABEL_COL] = label_map[row["id"]]
        n_gold = df["is_golden"].sum()
        print(f"  Applied {n_gold} human-verified labels from golden_dataset_100.parquet.")

    n_pos = (df[LABEL_COL] == 1).sum()
    n_neg = (df[LABEL_COL] == 0).sum()
    print(f"  Total Training Pool — Positive: {n_pos}, Negative: {n_neg}")
    return df

# ---------------------------------------------------------------------------
# 4.  Training & Prediction (Pure Numpy)
# ---------------------------------------------------------------------------

class DecisionStump:
    def __init__(self):
        self.feature_idx = 0
        self.threshold = 0.0
        self.left_value = 0.0
        self.right_value = 0.0

    def fit(self, X, residuals):
        best_loss = np.inf
        for feat_idx in range(X.shape[1]):
            col = X[:, feat_idx]
            thresholds = np.unique(np.percentile(col, np.arange(5, 100, 5)))
            for thr in thresholds:
                l_m = col <= thr; r_m = ~l_m
                if l_m.sum() < 2 or r_m.sum() < 2: continue
                l_v = residuals[l_m].mean(); r_v = residuals[r_m].mean()
                loss = np.mean((residuals - np.where(l_m, l_v, r_v))**2)
                if loss < best_loss:
                    best_loss = loss; self.feature_idx = feat_idx; self.threshold = thr
                    self.left_value = l_v; self.right_value = r_v

    def predict(self, X):
        return np.where(X[:, self.feature_idx] <= self.threshold, self.left_value, self.right_value)

class GradientBoostedClassifier:
    def __init__(self, n_estimators=150, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.trees = []; self.init_pred = 0.0

    def fit(self, X, y):
        p = y.mean()
        self.init_pred = np.log(p / (1 - p)) if 0 < p < 1 else 0.0
        raw_preds = np.full(len(y), self.init_pred)
        for i in range(self.n_estimators):
            probs = 1.0 / (1.0 + np.exp(-np.clip(raw_preds, -500, 500)))
            residuals = y - probs
            stump = DecisionStump()
            stump.fit(X, residuals)
            self.trees.append(stump)
            raw_preds += self.learning_rate * stump.predict(X)

    def predict_proba(self, X):
        raw_preds = np.full(X.shape[0], self.init_pred)
        for tree in self.trees:
            raw_preds += self.learning_rate * tree.predict(X)
        return 1.0 / (1.0 + np.exp(-np.clip(raw_preds, -500, 500)))

def main():
    print("="*60 + "\nPHASE 5 — XGBoost-style Classifier (v3: Existence vs Completeness)\n" + "="*60)
    # INPUT_PATH is phase1_processed (built from project_a_samples). Exclude first GOLDEN_NROWS
    # (those are the golden set); use only the broader input for training and prediction.
    full_df = pd.read_parquet(INPUT_PATH)
    df = full_df.iloc[GOLDEN_NROWS:].reset_index(drop=True)
    print(f"  Loaded {len(df)} rows (broader input only; first {GOLDEN_NROWS} rows excluded).")
    df = engineer_features(df)
    df = apply_labels(df)

    labeled = df.dropna(subset=[LABEL_COL]).copy()
    X = labeled[FEATURE_COLS].fillna(0).values
    y = labeled[LABEL_COL].astype(int).values

    n_labeled = len(labeled)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    print(f"\n  Labeled pool: {n_labeled} rows (positive: {n_pos}, negative: {n_neg})")

    # Simple 80/20 train/validation
    indices = np.arange(len(y))
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(indices)
    split = int(0.8 * len(y))
    train_idx, val_idx = indices[:split], indices[split:]
    n_train, n_val = len(train_idx), len(val_idx)
    print(f"  Train size: {n_train}  |  Validation size: {n_val}")

    model = GradientBoostedClassifier()
    print("\n  Training model on combined labels ...")
    model.fit(X[train_idx], y[train_idx])

    # Accuracy on training set
    y_pred_train = (model.predict_proba(X[train_idx]) >= 0.5).astype(int)
    acc_train = (y_pred_train == y[train_idx]).sum() / len(y_pred_train)
    print(f"\n  Accuracy on Training Set: {acc_train:.4%} (n={n_train})")

    # Accuracy on validation set
    y_pred_val = (model.predict_proba(X[val_idx]) >= 0.5).astype(int)
    acc_val = (y_pred_val == y[val_idx]).sum() / len(y_pred_val)
    print(f"  Accuracy on Validation Set: {acc_val:.4%} (n={n_val})")

    # Feature usage distribution (how often each feature was used in splits)
    feature_usage = np.zeros(len(FEATURE_COLS), dtype=np.int64)
    for tree in model.trees:
        feature_usage[tree.feature_idx] += 1
    feature_prob = feature_usage / len(model.trees)
    print("")

    print("\n  Feature usage distribution (fraction of trees using each feature):")
    print("  " + "-" * 60)
    for name, prob, count in sorted(
        zip(FEATURE_COLS, feature_prob, feature_usage),
        key=lambda x: -x[1],
    ):
        print(f"    {name}: {prob:.4f}  ({count} trees)")
    print("  " + "-" * 60)
    print(f"  Sum of fractions: {feature_prob.sum():.4f} (one feature per tree)")

    # Predict all
    print(f"\n  Predicting conflation for all {len(df)} records ...")
    X_all = df[FEATURE_COLS].fillna(0).values
    df["xgb_prediction"] = (model.predict_proba(X_all) >= 0.5).astype(int)
    
    print(f"  Decision: Use Match = {df['xgb_prediction'].sum()} | Keep Base = {len(df) - df['xgb_prediction'].sum()}")
    
    temp_cols = [c for c in df.columns if c.startswith("_")]
    df.drop(columns=temp_cols).to_parquet(OUTPUT_PATH, index=False)
    print(f"\n  Done! Results saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
