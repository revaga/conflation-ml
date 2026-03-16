"""
XGBoost-style Gradient Boosted Classifier for Places Attribute Conflation
=====================================================================================
Two-stage approach:
  Stage 1: Binary classifier  match+both -> 1 (accept)  vs  base -> 0 (keep base)
           Uses Ensemble (L2-LogReg + GBC stumps).
  Stage 2: Post-processing to identify "both" cases among accepted matches.

Training: phase3_slm_labeled.parquet (SLM labels), EXCLUDING the 200 golden rows.
Test/truth: golden_dataset_200.parquet (3class_testlabels).
Output: 3-class prediction (match / both / base) + probability.

Run from project root:
    python scripts/xgboostbinary.py
"""

import json
import re
import warnings
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

# Add scripts directory to sys.path to ensure absolute imports work
scripts_dir = str(Path(__file__).parent.absolute())
if scripts_dir not in sys.path:
    sys.path.append(scripts_dir)

try:
    from website_validator import verify_website
    from phonenumber_validator import validate_phone_number
    from parquet_io import read_parquet_safe
except ImportError:
    from scripts.website_validator import verify_website
    from scripts.phonenumber_validator import validate_phone_number
    from scripts.parquet_io import read_parquet_safe

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INPUT_PATH = "data/phase3_slm_labeled.parquet"
GOLDEN_PATH = "data/golden_dataset_200.parquet"
GOLDEN_NROWS = 200
OUTPUT_PATH = "data/xgboost_results.parquet"
LABEL_COL = "final_label"
TRUTH_COL = "3class_testlabels"
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
    return obj.get("primary", "") if isinstance(obj, dict) else ""

def extract_primary_category(val):
    obj = safe_json(val)
    return obj.get("primary", "") if isinstance(obj, dict) else ""

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

def _has_content(val):
    """Return True if val is non-empty/non-null."""
    if val is None:
        return False
    if isinstance(val, float) and np.isnan(val):
        return False
    s = str(val).strip()
    return s not in ("", "nan", "None", "{}", "[]", "null")

# ---------------------------------------------------------------------------
# 2.  Feature Engineering
# ---------------------------------------------------------------------------

def _check_website(url) -> bool:
    if not url or (isinstance(url, float) and np.isnan(url)):
        return False
    url = str(url).strip()
    if not url:
        return False
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    is_valid, _ = verify_website(url)
    return is_valid

def _check_phone_number(phone_number: str) -> bool:
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

    print("  Calculating features ...")

    # -- A. Trust / Confidence --
    df["feat_existence_conf_delta"] = df["confidence"] - df["base_confidence"]
    df["feat_match_exists_score"] = df["confidence"]
    df["feat_base_exists_score"] = df["base_confidence"]

    # -- B. Completeness / Attribute presence --
    df["feat_match_addr_len"] = df["norm_conflated_addr"].fillna("").str.len()
    df["feat_base_addr_len"] = df["norm_base_addr"].fillna("").str.len()
    df["feat_addr_richness_delta"] = df["feat_match_addr_len"] - df["feat_base_addr_len"]

    df["feat_match_has_phone"] = (df["norm_conflated_phone"].fillna("") != "").astype(int)
    df["feat_base_has_phone"] = (df["norm_base_phone"].fillna("") != "").astype(int)
    df["feat_match_has_web"] = (df["norm_conflated_website"].fillna("") != "").astype(int)
    df["feat_base_has_web"] = (df["norm_base_website"].fillna("") != "").astype(int)
    df["feat_phone_presence_delta"] = df["feat_match_has_phone"] - df["feat_base_has_phone"]
    df["feat_web_presence_delta"] = df["feat_match_has_web"] - df["feat_base_has_web"]

    # Actual attribute value features: brand, socials, emails
    df["feat_match_has_brand"] = df["brand"].apply(_has_content).astype(int)
    df["feat_base_has_brand"] = df["base_brand"].apply(_has_content).astype(int)
    df["feat_brand_delta"] = df["feat_match_has_brand"] - df["feat_base_has_brand"]

    df["feat_match_has_social"] = df["socials"].apply(_has_content).astype(int)
    df["feat_base_has_social"] = df["base_socials"].apply(_has_content).astype(int)
    df["feat_social_delta"] = df["feat_match_has_social"] - df["feat_base_has_social"]

    # -- C. Validation --
    print("  Validating website URLs (this may take a while) ...")
    df["feat_match_web_valid"] = df["norm_conflated_website"].apply(_check_website).astype(int)
    df["feat_base_web_valid"] = df["norm_base_website"].apply(_check_website).astype(int)
    df["feat_web_valid_delta"] = df["feat_match_web_valid"] - df["feat_base_web_valid"]

    print("  Validating phone numbers ...")
    df["feat_match_phone_valid"] = df["norm_conflated_phone"].apply(_check_phone_number).astype(int)
    df["feat_base_phone_valid"] = df["norm_base_phone"].apply(_check_phone_number).astype(int)
    df["feat_phone_valid_delta"] = df["feat_match_phone_valid"] - df["feat_base_phone_valid"]

    # -- D. Similarity --
    df["feat_name_similarity"] = df.apply(
        lambda r: fuzz.token_sort_ratio(r["_name"], r["_base_name"]) / 100.0
        if r["_name"] and r["_base_name"] else 0.0, axis=1
    )
    df["feat_addr_similarity"] = df["addr_similarity_ratio"] / 100.0
    df["feat_phone_similarity"] = (
        (df["phone_similarity"] / 100.0).fillna(0)
        if "phone_similarity" in df.columns else 0.0
    )
    df["feat_phone_exact_match"] = (
        (df["norm_conflated_phone"].fillna("").astype(str) != "")
        & (df["norm_base_phone"].fillna("").astype(str) != "")
        & (df["norm_conflated_phone"].fillna("").astype(str) == df["norm_base_phone"].fillna("").astype(str))
    ).astype(int)
    df["feat_web_similarity"] = (
        (df["website_similarity"] / 100.0).fillna(0)
        if "website_similarity" in df.columns else 0.0
    )
    df["feat_category_similarity"] = df.apply(
        lambda r: fuzz.token_sort_ratio(r["_category"], r["_base_category"]) / 100.0
        if r["_category"] and r["_base_category"] else 0.0, axis=1
    )
    df["feat_category_exact_match"] = (
        (df["_category"].fillna("") != "")
        & (df["_base_category"].fillna("") != "")
        & (df["_category"].fillna("") == df["_base_category"].fillna(""))
    ).astype(int)

    # -- E. Source Signals --
    df["feat_is_msft_match"] = df["_src_datasets"].apply(lambda s: int("msft" in s) if isinstance(s, set) else 0)
    df["feat_is_meta_match"] = df["_src_datasets"].apply(lambda s: int("meta" in s) if isinstance(s, set) else 0)
    df["feat_src_count_delta"] = df["_src_count"] - df["_base_src_count"]

    ref = datetime.now(timezone.utc)
    def _days_since(d):
        if d is None: return 9999
        if d.tzinfo is None: d = d.replace(tzinfo=timezone.utc)
        return (ref - d).days
    df["feat_match_recency_days"] = df["_src_latest"].apply(_days_since)
    df["feat_base_recency_days"] = df["_base_src_latest"].apply(_days_since)
    df["feat_recency_delta"] = df["feat_base_recency_days"] - df["feat_match_recency_days"]

    # -- F. Composite / Interaction --
    match_completeness = (
        df["feat_match_has_phone"] + df["feat_match_has_web"]
        + (df["feat_match_addr_len"] > 0).astype(int)
        + (df["_name"] != "").astype(int)
        + (df["_category"] != "").astype(int)
        + df["feat_match_has_brand"] + df["feat_match_has_social"]
    )
    base_completeness = (
        df["feat_base_has_phone"] + df["feat_base_has_web"]
        + (df["feat_base_addr_len"] > 0).astype(int)
        + (df["_base_name"] != "").astype(int)
        + (df["_base_category"] != "").astype(int)
        + df["feat_base_has_brand"] + df["feat_base_has_social"]
    )
    df["feat_match_completeness"] = match_completeness
    df["feat_base_completeness"] = base_completeness
    df["feat_completeness_delta"] = match_completeness - base_completeness
    df["feat_valid_count_delta"] = (
        (df["feat_match_web_valid"] + df["feat_match_phone_valid"])
        - (df["feat_base_web_valid"] + df["feat_base_phone_valid"])
    )
    df["feat_name_addr_sim_product"] = df["feat_name_similarity"] * df["feat_addr_similarity"]
    df["feat_avg_similarity"] = (
        df["feat_name_similarity"] + df["feat_addr_similarity"]
        + df["feat_phone_similarity"] + df["feat_web_similarity"]
    ) / 4.0

    # -- G. Features for match-vs-both (post-processing) --
    # "match" brings NEW/different info; "both" has highly similar attributes
    df["feat_adds_new_info"] = (
        df["feat_existence_conf_delta"] * (1.0 - df["feat_avg_similarity"])
    )
    df["feat_addr_dissimilarity"] = 1.0 - df["feat_addr_similarity"]
    df["feat_conf_x_addr_richness"] = (
        df["feat_existence_conf_delta"] * df["feat_addr_richness_delta"].clip(0, None)
    )
    df["feat_phone_exclusive"] = (
        (df["feat_match_has_phone"] == 1) & (df["feat_base_has_phone"] == 0)
    ).astype(int)
    df["feat_web_exclusive"] = (
        (df["feat_match_has_web"] == 1) & (df["feat_base_has_web"] == 0)
    ).astype(int)

    return df

# ---------------------------------------------------------------------------
# 3.  Feature Lists
# ---------------------------------------------------------------------------

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
    "feat_recency_delta",
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

# Top features for LogReg
TOP_FEATURE_COLS = [
    "feat_existence_conf_delta",
    "feat_addr_similarity",
    "feat_addr_richness_delta",
    "feat_name_similarity",
    "feat_phone_similarity",
    "feat_web_similarity",
    "feat_avg_similarity",
    "feat_name_addr_sim_product",
    "feat_match_exists_score",
    "feat_category_similarity",
    "feat_social_delta",
    "feat_brand_delta",
]

# ---------------------------------------------------------------------------
# 4.  Labeling (Stage 1: match+both=1, base=0)
# ---------------------------------------------------------------------------

def _truth_to_binary(val):
    """Map 3class_testlabels -> binary:  match+both -> 1, base -> 0."""
    if pd.isna(val) or val is None:
        return np.nan
    v = str(val).strip().lower()
    if v in ("match", "both", "alt"):
        return 1.0
    if v == "base":
        return 0.0
    return np.nan


def _load_golden_parquet(path: str, truth_col: str) -> pd.DataFrame:
    """Load golden parquet (uses read_parquet_safe for PyArrow bug workaround)."""
    needed = ["id", truth_col]
    golden = read_parquet_safe(path)
    return golden[needed] if set(needed).issubset(golden.columns) else golden


def _derive_4class_from_attr_winners(row: pd.Series) -> str:
    """Derive record-level 4-class (none/alt/base/both) from attr_*_winner columns."""
    ATTR_ATTRS = ("name", "phone", "web", "address", "category")
    counts = {"none": 0, "both": 0, "base": 0, "alt": 0}
    for attr in ATTR_ATTRS:
        col = f"attr_{attr}_winner"
        w = row.get(col)
        if pd.isna(w) or w is None:
            counts["none"] += 1
        else:
            v = str(w).strip().lower()
            if v in ("base", "alt", "both", "none"):
                counts[v] = counts.get(v, 0) + 1
            else:
                counts["none"] += 1
    if counts["none"] >= 3:
        return "none"
    if counts["both"] >= 3:
        return "both"
    if counts["base"] > counts["alt"]:
        return "base"
    if counts["alt"] > counts["base"]:
        return "alt"
    return "both"


def apply_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign labels: golden rows (ids in golden_dataset_200) get truth; rest get SLM labels.
    Training uses phase3_slm_labeled rows EXCLUDING the 200 golden rows.
    """
    df[LABEL_COL] = np.nan
    df["is_golden"] = False

    golden_ids = set()
    if os.path.exists(GOLDEN_PATH):
        print(f"  Loading truth from {GOLDEN_PATH} (column: {TRUTH_COL}) ...")
        golden = _load_golden_parquet(GOLDEN_PATH, TRUTH_COL)
        if TRUTH_COL not in golden.columns:
            raise KeyError(f"Column '{TRUTH_COL}' not found in {GOLDEN_PATH}")
        golden["_binary"] = golden[TRUTH_COL].apply(_truth_to_binary)
        golden_labeled = golden.dropna(subset=["_binary"])
        label_map = dict(zip(golden_labeled["id"], golden_labeled["_binary"]))
        golden_ids = set(label_map.keys())

        # Also store the 3-class truth for post-processing evaluation
        truth3_map = dict(zip(golden["id"], golden[TRUTH_COL]))
        df["_truth3"] = df["id"].map(truth3_map)

        df["is_golden"] = df["id"].isin(golden_ids)
        for idx, row in df[df["is_golden"]].iterrows():
            df.at[idx, LABEL_COL] = label_map[row["id"]]
        n_golden = df["is_golden"].sum()
        n_pos = int((df.loc[df["is_golden"], LABEL_COL] == 1).sum())
        n_neg = int((df.loc[df["is_golden"], LABEL_COL] == 0).sum())
        print(f"  Applied {n_golden} truth labels to golden rows (accept={n_pos}, keep_base={n_neg})")
    else:
        print(f"  WARNING: {GOLDEN_PATH} not found")

    # Quality-based labels for non-golden rows (training set)
    # We prioritize attribute accuracy/richness over existence confidence.
    ng_mask = ~df["is_golden"]

    # Calculate Quality Score for Match and Base
    # Weights: Completeness (richness) = 2.0, Validity = 1.0, Recency = 0.5
    match_quality = (
        (df["feat_match_completeness"] * 2.0) +
        (df["feat_match_web_valid"] + df["feat_match_phone_valid"]) +
        (df["feat_match_recency_days"].apply(lambda d: 1.0 if d < 365 else 0.0) * 0.5)
    )
    base_quality = (
        (df["feat_base_completeness"] * 2.0) +
        (df["feat_base_web_valid"] + df["feat_base_phone_valid"]) +
        (df["feat_base_recency_days"].apply(lambda d: 1.0 if d < 365 else 0.0) * 0.5)
    )

    # If Match quality > Base quality -> 1.0 (accept), else 0.0 (keep base)
    quality_binary = (match_quality > base_quality).astype(float)
    df.loc[ng_mask, LABEL_COL] = quality_binary[ng_mask]

    n_train = int((~df["is_golden"] & df[LABEL_COL].notna()).sum())
    n_pos = int((df[LABEL_COL] == 1).sum())
    n_neg = int((df[LABEL_COL] == 0).sum())
    n_unlabeled = int(df[LABEL_COL].isna().sum())
    print(f"  Heuristic training set (excluding golden): {n_train} rows | Total labels — Accept: {n_pos}, Keep_base: {n_neg}, Unlabeled: {n_unlabeled}")
    return df

# ---------------------------------------------------------------------------
# 5.  Models
# ---------------------------------------------------------------------------

class L2LogisticRegression:
    """L2-regularized logistic regression via gradient descent."""
    def __init__(self, lr=0.01, lam=1.0, epochs=500):
        self.lr = lr; self.lam = lam; self.epochs = epochs
        self.w = None; self.b = 0.0; self.mu = None; self.sd = None

    def fit(self, X, y, sample_weight=None):
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-8
        Xs = (X - self.mu) / self.sd
        n, d = Xs.shape
        self.w = np.zeros(d); self.b = 0.0
        if sample_weight is None: sample_weight = np.ones(n)
        sw = sample_weight / sample_weight.sum() * n

        for _ in range(self.epochs):
            z = Xs @ self.w + self.b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
            err = (p - y) * sw
            self.w -= self.lr * (Xs.T @ err / n + self.lam * self.w)
            self.b -= self.lr * err.mean()

    def predict_proba(self, X):
        Xs = (X - self.mu) / self.sd
        z = Xs @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class DecisionStump:
    def __init__(self):
        self.feature_idx = 0; self.threshold = 0.0
        self.left_value = 0.0; self.right_value = 0.0

    def fit(self, X, residuals, sample_weight=None):
        if sample_weight is None: sample_weight = np.ones(len(residuals))
        best_loss = np.inf
        for fi in range(X.shape[1]):
            col = X[:, fi]
            thresholds = np.unique(np.percentile(col, np.arange(5, 100, 5)))
            for thr in thresholds:
                lm = col <= thr; rm = ~lm
                if sample_weight[lm].sum() < 2 or sample_weight[rm].sum() < 2: continue
                lv = np.average(residuals[lm], weights=sample_weight[lm])
                rv = np.average(residuals[rm], weights=sample_weight[rm])
                loss = np.average((residuals - np.where(lm, lv, rv))**2, weights=sample_weight)
                if loss < best_loss:
                    best_loss = loss
                    self.feature_idx = fi; self.threshold = thr
                    self.left_value = lv; self.right_value = rv

    def predict(self, X):
        return np.where(X[:, self.feature_idx] <= self.threshold,
                        self.left_value, self.right_value)


class GradientBoostedClassifier:
    def __init__(self, n_estimators=150, learning_rate=0.05, subsample=0.8):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.trees = []; self.init_pred = 0.0

    def fit(self, X, y, sample_weight=None):
        n = len(y)
        if sample_weight is None: sample_weight = np.ones(n)
        p = np.clip(np.average(y, weights=sample_weight), 1e-6, 1 - 1e-6)
        self.init_pred = np.log(p / (1 - p))
        raw = np.full(n, self.init_pred)
        rng = np.random.RandomState(RANDOM_STATE)
        for _ in range(self.n_estimators):
            probs = 1.0 / (1.0 + np.exp(-np.clip(raw, -500, 500)))
            resid = y - probs
            if self.subsample < 1.0:
                idx = rng.choice(n, max(1, int(n * self.subsample)), replace=False)
            else: idx = np.arange(n)
            stump = DecisionStump()
            stump.fit(X[idx], resid[idx], sample_weight[idx])
            self.trees.append(stump)
            raw += self.learning_rate * stump.predict(X)

    def predict_proba(self, X):
        raw = np.full(X.shape[0], self.init_pred)
        for t in self.trees:
            raw += self.learning_rate * t.predict(X)
        return 1.0 / (1.0 + np.exp(-np.clip(raw, -500, 500)))

# ---------------------------------------------------------------------------
# 6.  Stratified K-Fold CV
# ---------------------------------------------------------------------------

def _stratified_folds(y, n_folds=5, seed=42):
    rng = np.random.RandomState(seed)
    pos_idx = np.where(y == 1)[0]; neg_idx = np.where(y == 0)[0]
    rng.shuffle(pos_idx); rng.shuffle(neg_idx)
    folds = [[] for _ in range(n_folds)]
    for i, idx in enumerate(pos_idx): folds[i % n_folds].append(idx)
    for i, idx in enumerate(neg_idx): folds[i % n_folds].append(idx)
    return [np.array(f) for f in folds]


def _ensemble_cv(X_golden, y_golden, X_heuristic, y_heuristic,
                 golden_weight=5.0, n_folds=5, logreg_weight=0.6):
    n_g = len(y_golden)
    folds = _stratified_folds(y_golden, n_folds)
    oof_lr = np.zeros(n_g)
    oof_gbc = np.zeros(n_g)
    top_idx = [FEATURE_COLS.index(f) for f in TOP_FEATURE_COLS]

    for fi in range(n_folds):
        val_idx = folds[fi]
        tr_idx = np.concatenate([folds[f] for f in range(n_folds) if f != fi])

        # LogReg on golden only (TOP features)
        lr = L2LogisticRegression(lr=0.01, lam=1.0, epochs=500)
        lr.fit(X_golden[tr_idx][:, top_idx], y_golden[tr_idx])
        oof_lr[val_idx] = lr.predict_proba(X_golden[val_idx][:, top_idx])

        # GBC on heuristic + golden (ALL features)
        Xt = np.vstack([X_heuristic, X_golden[tr_idx]])
        yt = np.concatenate([y_heuristic, y_golden[tr_idx]])
        wt = np.concatenate([np.ones(len(y_heuristic)), np.full(len(tr_idx), golden_weight)])
        gbc = GradientBoostedClassifier(150, 0.05, 0.8)
        gbc.fit(Xt, yt, wt)
        oof_gbc[val_idx] = gbc.predict_proba(X_golden[val_idx])

    oof = logreg_weight * oof_lr + (1 - logreg_weight) * oof_gbc

    best_thr = 0.5; best_acc = 0.0
    for thr in np.arange(0.25, 0.75, 0.01):
        acc = ((oof >= thr).astype(int) == y_golden).mean()
        if acc > best_acc: best_acc = acc; best_thr = thr

    lr_acc = ((oof_lr >= 0.5).astype(int) == y_golden).mean()
    gbc_acc = ((oof_gbc >= 0.5).astype(int) == y_golden).mean()
    return best_thr, best_acc, oof, lr_acc, gbc_acc


# ---------------------------------------------------------------------------
# 7.  Stage 2: Post-processing — detect "both" among accepted matches
# ---------------------------------------------------------------------------

def _identify_both(df: pd.DataFrame) -> pd.Series:
    """
    Among rows predicted as 'accept' (match+both), distinguish 'match' from 'both'.
    
    "both" = base is equally good -- attributes are nearly identical, base is
    already complete, and confidence is similar.  Very conservative: only flag
    as "both" when we are very confident BOTH sources are equivalent.
    """
    # Very high similarity everywhere
    very_high_addr_sim = df["feat_addr_similarity"] >= 0.97
    very_high_name_sim = df["feat_name_similarity"] >= 0.90
    high_phone_sim = df["feat_phone_similarity"] >= 0.90

    # Base is equally or more complete (strictly)
    base_more_complete = df["feat_completeness_delta"] < 0
    base_equal_complete = df["feat_completeness_delta"] == 0

    # Very small confidence delta
    tiny_conf_delta = df["feat_existence_conf_delta"].abs() <= 0.10

    # Both sides have all key attributes
    both_have_phone = (df["feat_match_has_phone"] == 1) & (df["feat_base_has_phone"] == 1)
    both_have_web = (df["feat_match_has_web"] == 1) & (df["feat_base_has_web"] == 1)
    both_full = both_have_phone & both_have_web

    # Match adds nothing new (no exclusive attributes)
    no_exclusive = (df["feat_phone_exclusive"] == 0) & (df["feat_web_exclusive"] == 0)

    # "both": attributes nearly identical AND base >= match in completeness
    is_both = (
        very_high_addr_sim
        & very_high_name_sim
        & (base_more_complete | (base_equal_complete & both_full))
        & tiny_conf_delta
        & no_exclusive
    )

    return is_both


# ---------------------------------------------------------------------------
# 8.  Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("PHASE 5 -- Ensemble Classifier (2-stage: accept/reject + both)")
    print("=" * 70)

    df = read_parquet_safe(INPUT_PATH)
    print(f"  Loaded {len(df)} rows from {INPUT_PATH}")

    df = engineer_features(df)
    df = apply_labels(df)

    # -- Split: SLM (train, excluding golden 200) vs golden (test/truth) ----
    train_df = df[~df["is_golden"]].dropna(subset=[LABEL_COL]).copy()
    test_df = df[df["is_golden"]].dropna(subset=[LABEL_COL]).copy()

    X_train = train_df[FEATURE_COLS].fillna(0).values
    y_train = train_df[LABEL_COL].astype(int).values
    X_test = test_df[FEATURE_COLS].fillna(0).values
    y_test = test_df[LABEL_COL].astype(int).values

    n_train, n_test = len(train_df), len(test_df)
    print(f"\n  SLM training set (excl. golden 200): {n_train} rows "
          f"(accept={int((y_train == 1).sum())}, keep_base={int((y_train == 0).sum())})")
    print(f"  Golden test set:                     {n_test} rows "
          f"(accept={int((y_test == 1).sum())}, keep_base={int((y_test == 0).sum())})")

    if n_train < 10:
        print("  ERROR: Not enough SLM-labeled rows to train (exclude golden 200).")
        return

    # -- Sweep ensemble weights --------------------------------------------
    print("\n  Sweeping ensemble weights (LogReg vs GBC) ...")
    best_lrw = 0.6; best_cv_acc = 0.0; best_thr = 0.5
    for lrw in np.arange(0.3, 0.9, 0.1):
        thr, cv_acc, _, _, _ = _ensemble_cv(
            X_test, y_test, X_train, y_train, 5.0, 5, lrw
        )
        if cv_acc > best_cv_acc:
            best_cv_acc = cv_acc; best_lrw = lrw; best_thr = thr

    best_thr, cv_acc, oof, lr_acc, gbc_acc = _ensemble_cv(
        X_test, y_test, X_train, y_train, 5.0, 5, best_lrw
    )

    print(f"  Best ensemble weight: LogReg={best_lrw:.1f}, GBC={1 - best_lrw:.1f}")
    print(f"  5-fold CV accuracy (LogReg):   {lr_acc:.4%}")
    print(f"  5-fold CV accuracy (GBC):      {gbc_acc:.4%}")
    print(f"  5-fold CV accuracy (Ensemble): {cv_acc:.4%}  <-- honest estimate")
    print(f"  Optimal threshold:             {best_thr:.2f}")

    # -- Stage 1: Train final models on all available data -----------------
    print("\n  Stage 1: Training final models ...")
    top_idx = [FEATURE_COLS.index(f) for f in TOP_FEATURE_COLS]

    final_lr = L2LogisticRegression(lr=0.01, lam=1.0, epochs=500)
    final_lr.fit(X_test[:, top_idx], y_test)

    X_blend = np.vstack([X_train, X_test])
    y_blend = np.concatenate([y_train, y_test])
    w_blend = np.concatenate([np.ones(n_train), np.full(n_test, 5.0)])
    final_gbc = GradientBoostedClassifier(150, 0.05, 0.8)
    final_gbc.fit(X_blend, y_blend, w_blend)

    # -- Stage 1 evaluation ------------------------------------------------
    proba_lr = final_lr.predict_proba(X_test[:, top_idx])
    proba_gbc = final_gbc.predict_proba(X_test)
    proba_ens = best_lrw * proba_lr + (1 - best_lrw) * proba_gbc
    y_pred_s1 = (proba_ens >= best_thr).astype(int)
    acc_s1 = (y_pred_s1 == y_test).mean()

    tp = int(((y_pred_s1 == 1) & (y_test == 1)).sum())
    fp = int(((y_pred_s1 == 1) & (y_test == 0)).sum())
    fn = int(((y_pred_s1 == 0) & (y_test == 1)).sum())
    tn = int(((y_pred_s1 == 0) & (y_test == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    print(f"\n  --- Stage 1: Binary (match+both vs base) ---")
    print(f"  In-sample accuracy:  {acc_s1:.4%}")
    print(f"  CV accuracy:         {cv_acc:.4%}  <-- honest estimate")
    print(f"  Confusion Matrix:  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

    # -- Stage 2: Post-processing to detect "both" --------------------------
    print(f"\n  --- Stage 2: Detecting 'both' among accepted matches ---")

    # Apply to golden set for evaluation
    test_df["_s1_pred"] = y_pred_s1
    test_df["_s1_proba"] = proba_ens
    accepted_mask = test_df["_s1_pred"] == 1
    is_both = _identify_both(test_df)
    test_df["_final_pred"] = "base"
    test_df.loc[accepted_mask & ~is_both, "_final_pred"] = "match"
    test_df.loc[accepted_mask & is_both, "_final_pred"] = "both"

    # 3-class evaluation (exclude rows where golden 3-class is "none")
    truth3 = test_df["_truth3"].astype(str).str.strip().str.lower()
    pred3 = test_df["_final_pred"].str.strip().str.lower()
    valid_3class = ("match", "both", "base")
    mask_3 = truth3.isin(valid_3class)
    test_df_3 = test_df.loc[mask_3]
    truth3 = truth3[mask_3]
    pred3 = pred3[mask_3]

    correct_3class = (truth3 == pred3).sum()
    total = len(test_df_3)
    acc_3class = correct_3class / total if total else 0.0

    print(f"  3-class accuracy: {acc_3class:.4%} ({correct_3class}/{total})")
    print(f"\n  3-class confusion matrix (rows=truth, cols=pred):")
    classes = ["match", "both", "base"]
    header = f"  {'truth/pred':<12}"
    for c in classes:
        header += f"  {c:>6}"
    header += "  | total"
    print(header)
    print("  " + "-" * 44)
    for true_class in classes:
        row = f"  {true_class:<12}"
        t_mask = truth3 == true_class
        row_total = int(t_mask.sum())
        for pred_class in classes:
            p_mask = pred3 == pred_class
            count = int((t_mask & p_mask).sum())
            row += f"  {count:>6}"
        row += f"  | {row_total:>5}"
        print(row)

    # Per-class precision and recall
    print(f"\n  Per-class metrics:")
    for c in classes:
        tp_c = int(((truth3 == c) & (pred3 == c)).sum())
        pred_c = int((pred3 == c).sum())
        true_c = int((truth3 == c).sum())
        prec_c = tp_c / pred_c if pred_c > 0 else 0
        rec_c = tp_c / true_c if true_c > 0 else 0
        f1_c = 2 * prec_c * rec_c / (prec_c + rec_c) if (prec_c + rec_c) > 0 else 0
        print(f"    {c:<8s}: precision={prec_c:.4f}  recall={rec_c:.4f}  F1={f1_c:.4f}  (n={true_c})")

    # -- Feature importance ------------------------------------------------
    print(f"\n  GBC Feature usage ({len(final_gbc.trees)} trees):")
    print("  " + "-" * 60)
    feature_usage = np.zeros(len(FEATURE_COLS), dtype=np.int64)
    for t in final_gbc.trees:
        feature_usage[t.feature_idx] += 1
    for name, count in sorted(zip(FEATURE_COLS, feature_usage), key=lambda x: -x[1]):
        frac = count / len(final_gbc.trees)
        bar = "#" * int(frac * 40)
        print(f"    {name:<35s}  {count:4d}  ({frac:.3f})  {bar}")
    print("  " + "-" * 60)

    # LogReg coefficients
    print(f"\n  LogReg coefficients:")
    print("  " + "-" * 60)
    for idx in np.argsort(-np.abs(final_lr.w)):
        name = TOP_FEATURE_COLS[idx]; coef = final_lr.w[idx]
        bar = "+" * int(abs(coef) * 10) if coef > 0 else "-" * int(abs(coef) * 10)
        print(f"    {name:<35s}  {coef:+.4f}  {bar}")
    print(f"    {'(bias)':<35s}  {final_lr.b:+.4f}")
    print("  " + "-" * 60)

    # -- Predict all rows --------------------------------------------------
    print(f"\n  Predicting for all {len(df)} records ...")
    X_all = df[FEATURE_COLS].fillna(0).values
    proba_lr_all = final_lr.predict_proba(X_all[:, top_idx])
    proba_gbc_all = final_gbc.predict_proba(X_all)
    all_proba = best_lrw * proba_lr_all + (1 - best_lrw) * proba_gbc_all

    df["xgb_probability"] = all_proba
    df["xgb_accept"] = (all_proba >= best_thr).astype(int)

    # Stage 2: identify "both" among accepted
    accepted = df["xgb_accept"] == 1
    is_both_all = _identify_both(df)
    df["xgb_prediction"] = "base"
    df.loc[accepted & ~is_both_all, "xgb_prediction"] = "match"
    df.loc[accepted & is_both_all, "xgb_prediction"] = "both"

    n_match = int((df["xgb_prediction"] == "match").sum())
    n_both = int((df["xgb_prediction"] == "both").sum())
    n_base = int((df["xgb_prediction"] == "base").sum())
    print(f"  Decision: match={n_match} | both={n_both} | base={n_base}")

    # -- Save --------------------------------------------------------------
    temp_cols = [c for c in df.columns if c.startswith("_")]
    df.drop(columns=temp_cols).to_parquet(OUTPUT_PATH, index=False)
    print(f"\n  Done! Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
