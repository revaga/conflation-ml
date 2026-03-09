"""
Multiclass XGBoost (none / alt / base / both) with hyperparameter tuning.
=====================================================================================
- Golden labels: recalculated from attr_*_winner (majority none/both/base/alt; tie -> both).
- Non-golden labels: heuristic 4-class (validation and completeness emphasized; less conf_delta).
- Model: XGBoost 4-class. Tuning: GridSearchCV, RandomizedSearchCV, Optuna; pick best by macro F1.
- Output: data/xgboost_multiclass_results.parquet with xgb_4class_pred and probabilities.

Run from project root:
    python scripts/xgboost_multiclass.py
"""

import json
import os
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from website_validator import verify_website
from phonenumber_validator import validate_phone_number
from parquet_io import read_parquet_safe

# Avoid importing scripts/xgboost.py instead of the installed xgboost package
_script_dir = os.path.dirname(os.path.abspath(__file__))
_path_save = sys.path.copy()
sys.path = [p for p in sys.path if os.path.abspath(p) != os.path.abspath(_script_dir)]
import xgboost as xgb
sys.path = _path_save

# Optional sklearn (may fail with DLL load on some Windows envs)
try:
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
    from sklearn.metrics import f1_score, classification_report, confusion_matrix
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INPUT_PATH = "data/phase3_slm_labeled.parquet"
GOLDEN_PATH = "data/golden_dataset_200.parquet"
OUTPUT_PATH = "data/xgboost_multiclass_results.parquet"
RANDOM_STATE = 42
LABEL_COL_4CLASS = "label_4class"
ATTR_ATTRS = ("name", "phone", "web", "address", "category")
# Class order for encoding: none=0, alt=1, base=2, both=3
CLASS_ORDER = ("none", "alt", "base", "both")
TIE_BREAK_TO_BOTH = True  # When base_count == alt_count, assign "both"

# ---------------------------------------------------------------------------
# 1. JSON Helpers (from xgboost.py)
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
        if not isinstance(item, dict):
            continue
        ds = item.get("dataset", "")
        if ds:
            datasets.add(ds.lower())
        ut = item.get("update_time")
        if ut:
            try:
                dt = datetime.fromisoformat(ut.replace("Z", "+00:00"))
                if latest_dt is None or dt > latest_dt:
                    latest_dt = dt
            except Exception:
                pass
    return count, latest_dt, datasets

def _has_content(val):
    if val is None:
        return False
    if isinstance(val, float) and np.isnan(val):
        return False
    s = str(val).strip()
    return s not in ("", "nan", "None", "{}", "[]", "null")

# ---------------------------------------------------------------------------
# 2. Feature Engineering (from xgboost.py)
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
    df["feat_existence_conf_delta"] = df["confidence"] - df["base_confidence"]
    df["feat_match_exists_score"] = df["confidence"]
    df["feat_base_exists_score"] = df["base_confidence"]

    df["feat_match_addr_len"] = df["norm_conflated_addr"].fillna("").str.len()
    df["feat_base_addr_len"] = df["norm_base_addr"].fillna("").str.len()
    df["feat_addr_richness_delta"] = df["feat_match_addr_len"] - df["feat_base_addr_len"]

    df["feat_match_has_phone"] = (df["norm_conflated_phone"].fillna("") != "").astype(int)
    df["feat_base_has_phone"] = (df["norm_base_phone"].fillna("") != "").astype(int)
    df["feat_match_has_web"] = (df["norm_conflated_website"].fillna("") != "").astype(int)
    df["feat_base_has_web"] = (df["norm_base_website"].fillna("") != "").astype(int)
    df["feat_phone_presence_delta"] = df["feat_match_has_phone"] - df["feat_base_has_phone"]
    df["feat_web_presence_delta"] = df["feat_match_has_web"] - df["feat_base_has_web"]

    df["feat_match_has_brand"] = df["brand"].apply(_has_content).astype(int)
    df["feat_base_has_brand"] = df["base_brand"].apply(_has_content).astype(int)
    df["feat_brand_delta"] = df["feat_match_has_brand"] - df["feat_base_has_brand"]

    df["feat_match_has_social"] = df["socials"].apply(_has_content).astype(int)
    df["feat_base_has_social"] = df["base_socials"].apply(_has_content).astype(int)
    df["feat_social_delta"] = df["feat_match_has_social"] - df["feat_base_has_social"]

    print("  Validating website URLs ...")
    df["feat_match_web_valid"] = df["norm_conflated_website"].apply(_check_website).astype(int)
    df["feat_base_web_valid"] = df["norm_base_website"].apply(_check_website).astype(int)
    df["feat_web_valid_delta"] = df["feat_match_web_valid"] - df["feat_base_web_valid"]

    print("  Validating phone numbers ...")
    df["feat_match_phone_valid"] = df["norm_conflated_phone"].apply(_check_phone_number).astype(int)
    df["feat_base_phone_valid"] = df["norm_base_phone"].apply(_check_phone_number).astype(int)
    df["feat_phone_valid_delta"] = df["feat_match_phone_valid"] - df["feat_base_phone_valid"]

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

    df["feat_is_msft_match"] = df["_src_datasets"].apply(lambda s: int("msft" in s) if isinstance(s, set) else 0)
    df["feat_is_meta_match"] = df["_src_datasets"].apply(lambda s: int("meta" in s) if isinstance(s, set) else 0)
    df["feat_src_count_delta"] = df["_src_count"] - df["_base_src_count"]

    ref = datetime.now(timezone.utc)
    def _days_since(d):
        if d is None:
            return 9999
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return (ref - d).days
    df["feat_match_recency_days"] = df["_src_latest"].apply(_days_since)
    df["feat_base_recency_days"] = df["_base_src_latest"].apply(_days_since)
    df["feat_recency_delta"] = df["feat_base_recency_days"] - df["feat_match_recency_days"]

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

# ---------------------------------------------------------------------------
# 3. Recalculate 4-class label from attr_*_winner
# ---------------------------------------------------------------------------

def _normalize_attr_winner(val):
    """Return one of 'base', 'alt', 'both', 'none'. Treat missing/invalid as 'none'."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "none"
    v = str(val).strip().lower()
    if v in ("base", "alt", "both", "none"):
        return v
    return "none"


def recalculate_4class_label(row: pd.Series) -> str:
    """
    Compute record-level 4-class label from 5 attr_*_winner columns.
    Majority none -> none; else majority both -> both; else base vs alt by count; tie -> both.
    Missing attr treated as 'none'.
    """
    counts = {"none": 0, "both": 0, "base": 0, "alt": 0}
    for attr in ATTR_ATTRS:
        col = f"attr_{attr}_winner"
        w = _normalize_attr_winner(row.get(col))
        counts[w] = counts.get(w, 0) + 1

    n_none = counts["none"]
    n_both = counts["both"]
    n_base = counts["base"]
    n_alt = counts["alt"]

    if n_none >= 3:
        return "none"
    if n_both >= 3:
        return "both"
    if n_base > n_alt:
        return "base"
    if n_alt > n_base:
        return "alt"
    # tie base == alt
    return "both" if TIE_BREAK_TO_BOTH else "alt"


def apply_golden_4class_labels(df: pd.DataFrame, golden: pd.DataFrame) -> pd.DataFrame:
    """Set is_golden and label_4class for golden rows from recalculated attr_*_winner."""
    df["is_golden"] = df["id"].isin(golden["id"].tolist())
    df[LABEL_COL_4CLASS] = None  # object dtype for string labels
    if not golden.empty:
        golden_ids = golden["id"].tolist()
        for idx, row in golden.iterrows():
            rec_label = recalculate_4class_label(row)
            id_val = row["id"]
            mask = df["id"] == id_val
            df.loc[mask, LABEL_COL_4CLASS] = rec_label
    return df


# ---------------------------------------------------------------------------
# 4. Heuristic 4-class for non-golden rows
# ---------------------------------------------------------------------------

def apply_heuristic_4class(df: pd.DataFrame, t_name: float = 0.80, t_addr: float = 0.78) -> pd.DataFrame:
    """
    Assign none/alt/base/both for non-golden rows. Validation and completeness emphasized;
    conf_delta de-emphasized. Order: none (not same_place) -> base -> both -> alt; remainder -> both.
    """
    ng = ~df["is_golden"]
    same_place = (
        (df["feat_name_similarity"] > t_name)
        | ((df["feat_name_similarity"] > t_name - 0.12) & (df["feat_addr_similarity"] > t_addr))
    )

    # none: not same place
    df.loc[ng & ~same_place, LABEL_COL_4CLASS] = "none"

    # base: same_place and (similarity low, or base wins validation, or valid_count_delta < 0 / completeness)
    base_similarity = (df["feat_name_similarity"] < t_name - 0.40) | (df["feat_avg_similarity"] < 0.45)
    base_valid_phone = (df["feat_base_phone_valid"] == 1) & (df["feat_match_phone_valid"] == 0)
    base_valid_web = (df["feat_base_web_valid"] == 1) & (df["feat_match_web_valid"] == 0)
    base_validity = base_valid_phone | base_valid_web
    base_overall = (df["feat_valid_count_delta"] < 0) | (
        (df["feat_valid_count_delta"] <= 0) & (df["feat_completeness_delta"] <= 0)
    )
    cond_base = ng & same_place & (base_similarity | base_validity | base_overall)
    df.loc[cond_base, LABEL_COL_4CLASS] = "base"

    # both: same_place, not base, high similarity, completeness_delta<=0, validation parity, no exclusive
    high_sim = (df["feat_addr_similarity"] >= 0.97) & (df["feat_name_similarity"] >= 0.90)
    validation_parity = (df["feat_phone_valid_delta"] == 0) & (df["feat_web_valid_delta"] == 0)
    no_exclusive = (df["feat_phone_exclusive"] == 0) & (df["feat_web_exclusive"] == 0)
    cond_both = (
        ng & same_place & ~cond_base
        & high_sim
        & (df["feat_completeness_delta"] <= 0)
        & validation_parity
        & no_exclusive
    )
    df.loc[cond_both, LABEL_COL_4CLASS] = "both"

    # alt: same_place, match adds value (validation first, then completeness), not both
    match_adds_value = (
        (df["feat_phone_valid_delta"] > 0)
        | (df["feat_web_valid_delta"] > 0)
        | (df["feat_completeness_delta"] > 0)
        | (df["feat_addr_richness_delta"] > 3)
        | (df["feat_phone_presence_delta"] > 0)
        | (df["feat_web_presence_delta"] > 0)
        | (df["feat_existence_conf_delta"] > 0)
    )
    cond_alt = ng & same_place & ~cond_base & ~cond_both & match_adds_value
    df.loc[cond_alt, LABEL_COL_4CLASS] = "alt"

    # remainder: same_place but no strong signal -> both
    remainder = ng & same_place & df[LABEL_COL_4CLASS].isna()
    df.loc[remainder, LABEL_COL_4CLASS] = "both"

    return df


# ---------------------------------------------------------------------------
# 5. Label encoding
# ---------------------------------------------------------------------------

def encode_labels(series: pd.Series) -> np.ndarray:
    """Map none/alt/base/both -> 0,1,2,3."""
    return series.map(lambda x: CLASS_ORDER.index(x) if x in CLASS_ORDER else np.nan).values


def decode_labels(arr) -> np.ndarray:
    """Map 0,1,2,3 -> none/alt/base/both."""
    return np.array([CLASS_ORDER[int(i)] if 0 <= int(i) < len(CLASS_ORDER) else "none" for i in arr])


# ---------------------------------------------------------------------------
# 6. Manual CV and metrics (when sklearn unavailable)
# ---------------------------------------------------------------------------

def _manual_stratified_kfold(y, n_splits=5, random_state=42):
    """Return list of (train_idx, test_idx) for stratified K-fold."""
    rng = np.random.RandomState(random_state)
    y = np.asarray(y)
    classes, y_inv = np.unique(y, return_inverse=True)
    folds = [[] for _ in range(n_splits)]
    for c in range(len(classes)):
        idx = np.where(y_inv == c)[0]
        rng.shuffle(idx)
        for i, j in enumerate(idx):
            folds[i % n_splits].append(j)
    return [
        (np.array([j for f in folds if f != fold for j in f]), np.array(fold))
        for fold in folds
    ]


def _manual_f1_macro(y_true, y_pred, n_classes=4):
    """Macro F1 for multi-class."""
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()
    f1s = []
    for c in range(n_classes):
        tp = ((y_t == c) & (y_p == c)).sum()
        fp = ((y_t != c) & (y_p == c)).sum()
        fn = ((y_t == c) & (y_p != c)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return np.mean(f1s)


def _manual_classification_report(y_true, y_pred):
    """Per-class precision/recall/F1 and accuracy."""
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()
    lines = [f"{'':>10} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}"]
    for i, c in enumerate(CLASS_ORDER):
        tp = ((y_t == i) & (y_p == i)).sum()
        fp = ((y_t != i) & (y_p == i)).sum()
        fn = ((y_t == i) & (y_p != i)).sum()
        support = int((y_t == i).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        lines.append(f"{c:>10} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {support:>10}")
    acc = (y_t == y_p).mean()
    lines.append(f"{'accuracy':>10} {'':>10} {'':>10} {acc:>10.4f} {len(y_t):>10}")
    return "\n".join(lines)


def _manual_confusion_matrix(y_true, y_pred, n_classes=4):
    """Confusion matrix (rows=truth, cols=pred)."""
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()
    m = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(len(y_t)):
        m[int(y_t[i]), int(y_p[i])] += 1
    return m


# ---------------------------------------------------------------------------
# 7. Native XGBoost training (no sklearn dependency)
# ---------------------------------------------------------------------------

def _xgb_train_params(**kwargs):
    """Build param dict for xgb.train (multi:softprob)."""
    return {
        "max_depth": kwargs.get("max_depth", 5),
        "eta": kwargs.get("learning_rate", 0.1),
        "objective": "multi:softprob",
        "num_class": 4,
        "eval_metric": "mlogloss",
        "subsample": kwargs.get("subsample", 0.8),
        "colsample_bytree": kwargs.get("colsample_bytree", 0.8),
        "min_child_weight": kwargs.get("min_child_weight", 1),
        "reg_alpha": kwargs.get("reg_alpha", 0.1),
        "reg_lambda": kwargs.get("reg_lambda", 0.1),
        "seed": RANDOM_STATE,
    }


def _train_booster(X, y, **kwargs):
    """Train xgb.Booster with given tuning params. Returns booster."""
    params = _xgb_train_params(**kwargs)
    dtrain = xgb.DMatrix(X, label=y, feature_names=FEATURE_COLS)
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=kwargs.get("n_estimators", 200),
        verbose_eval=False,
    )
    return booster


def _predict_booster(booster, X):
    """Predict class indices (0..3) from booster. X: 2d array."""
    d = xgb.DMatrix(X, feature_names=FEATURE_COLS)
    proba = booster.predict(d)
    if proba.ndim == 1:
        proba = proba.reshape(-1, 4)
    return np.argmax(proba, axis=1)


def _cv_score_native(X, y, n_splits=5, **params):
    """Mean macro F1 over stratified K-fold using native xgb."""
    folds = _manual_stratified_kfold(y, n_splits=n_splits, random_state=RANDOM_STATE)
    scores = []
    for train_idx, test_idx in folds:
        booster = _train_booster(X[train_idx], y[train_idx], **params)
        pred = _predict_booster(booster, X[test_idx])
        scores.append(_manual_f1_macro(y[test_idx], pred))
    return np.mean(scores)


def run_grid_search(X, y):
    # Grid: 2^4 = 16 combinations for reasonable runtime (5-fold CV per combo)
    param_grid = {
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "n_estimators": [100, 200],
        "subsample": [0.8, 1.0],
        # fixed: colsample_bytree=0.8, min_child_weight=1, reg_alpha=0.1, reg_lambda=0.1
    }
    defaults = {"colsample_bytree": 0.8, "min_child_weight": 1, "reg_alpha": 0.1, "reg_lambda": 0.1}
    keys = list(param_grid.keys())
    best_score, best_params = -1.0, None
    from itertools import product
    n_combos = 1
    for k in keys:
        n_combos *= len(param_grid[k])
    print(f"    GridSearchCV: {n_combos} combinations x 5-fold CV ...")
    for vals in product(*(param_grid[k] for k in keys)):
        params = {**defaults, **dict(zip(keys, vals))}
        score = _cv_score_native(X, y, **params)
        if score > best_score:
            best_score, best_params = score, params
    return best_score, best_params, "GridSearchCV"


def run_random_search(X, y, n_iter=25):
    param_dist = {
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": [0.03, 0.05, 0.1, 0.15],
        "n_estimators": [80, 120, 160, 200],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 2, 3],
        "reg_alpha": [0.05, 0.1, 0.5, 1.0],
        "reg_lambda": [0.05, 0.1, 0.5, 1.0],
    }
    rng = np.random.RandomState(RANDOM_STATE)
    keys = list(param_dist.keys())
    best_score, best_params = -1.0, None
    for _ in range(n_iter):
        params = {k: rng.choice(param_dist[k]) for k in keys}
        score = _cv_score_native(X, y, **params)
        if score > best_score:
            best_score, best_params = score, params
    return best_score, best_params, "RandomizedSearchCV"


def run_optuna(X, y, n_trials=25):
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        return None, None, "Optuna (skip)"

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15),
            "n_estimators": trial.suggest_int("n_estimators", 80, 200),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 3),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.05, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.05, 1.0),
        }
        return _cv_score_native(X, y, **params)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE, n_startup_trials=10))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_value, study.best_params, "Optuna"


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Multiclass XGBoost (none / alt / base / both) with hyperparameter tuning")
    print("=" * 70)

    df = read_parquet_safe(INPUT_PATH)
    print(f"  Loaded {len(df)} rows from {INPUT_PATH}")

    df = engineer_features(df)

    # Load golden and apply 4-class labels to golden rows
    golden = read_parquet_safe(GOLDEN_PATH) if os.path.exists(GOLDEN_PATH) else pd.DataFrame()
    if golden.empty:
        print(f"  WARNING: {GOLDEN_PATH} not found or empty")
    else:
        # Ensure golden has attr_*_winner columns
        required = [f"attr_{a}_winner" for a in ATTR_ATTRS]
        missing = [c for c in required if c not in golden.columns]
        if missing:
            print(f"  WARNING: Golden missing columns: {missing}; 4-class labels may be all none")

    df = apply_golden_4class_labels(df, golden)

    # Thresholds from golden (alt+both vs base) if we have variety
    t_name, t_addr = 0.80, 0.78
    golden_labeled = df.loc[df["is_golden"]].dropna(subset=[LABEL_COL_4CLASS])
    if len(golden_labeled) > 10:
        pos = golden_labeled[golden_labeled[LABEL_COL_4CLASS].isin(("alt", "both"))]
        neg = golden_labeled[golden_labeled[LABEL_COL_4CLASS] == "base"]
        if len(pos) > 0 and len(neg) > 0:
            t_name = float((pos["feat_name_similarity"].median() + neg["feat_name_similarity"].median()) / 2)
            t_addr = float((pos["feat_addr_similarity"].median() + neg["feat_addr_similarity"].median()) / 2)
            print(f"  Thresholds from golden: t_name={t_name:.3f}, t_addr={t_addr:.3f}")

    # USE SLM LABELS recalculated from winners
    if any(f"attr_{a}_winner" in df.columns for a in ATTR_ATTRS):
        print("  Applying labels from SLM (recalculating 4-class from winners) ...")
        ng_mask = ~df["is_golden"]
        df.loc[ng_mask, LABEL_COL_4CLASS] = df[ng_mask].apply(recalculate_4class_label, axis=1)

    # Disable heuristics
    """
    df = apply_heuristic_4class(df, t_name=t_name, t_addr=t_addr)
    """

    train_df = df[~df["is_golden"]].dropna(subset=[LABEL_COL_4CLASS]).copy()
    test_df = df[df["is_golden"]].dropna(subset=[LABEL_COL_4CLASS]).copy()

    if len(train_df) < 20:
        print("  ERROR: Not enough heuristic-labeled training rows.")
        return
    if len(test_df) < 1:
        print("  WARNING: No golden test rows with 4-class label.")

    X_train = train_df[FEATURE_COLS].fillna(0).values
    y_train = encode_labels(train_df[LABEL_COL_4CLASS])
    X_test = test_df[FEATURE_COLS].fillna(0).values
    y_test = encode_labels(test_df[LABEL_COL_4CLASS])

    print(f"\n  Train: {len(train_df)} rows. Test (golden): {len(test_df)} rows.")
    print(f"  Train class counts: {dict(pd.Series(decode_labels(y_train)).value_counts())}")

    # Tuning: run all three, pick best by macro F1
    print("\n  Hyperparameter tuning (macro F1)...")
    results = []
    score, params, name = run_grid_search(X_train, y_train)
    results.append((score, params, name))
    print(f"    {name}: best CV macro F1 = {score:.4f}")

    score, params, name = run_random_search(X_train, y_train)
    results.append((score, params, name))
    print(f"    {name}: best CV macro F1 = {score:.4f}")

    score, params, name = run_optuna(X_train, y_train)
    if score is not None:
        results.append((score, params, name))
        print(f"    {name}: best CV macro F1 = {score:.4f}")

    best_score, best_params, best_name = max(results, key=lambda x: x[0])
    print(f"\n  Best: {best_name} with macro F1 = {best_score:.4f}")
    print(f"  Best params: {best_params}")

    # Train final model on full training set (native API works with or without sklearn)
    print("\n  Training final model ...")
    final_booster = _train_booster(X_train, y_train, **best_params)

    # Evaluate on golden holdout
    y_pred = _predict_booster(final_booster, X_test)
    acc = (y_pred == y_test).mean()
    macro_f1 = _manual_f1_macro(y_test, y_pred)
    print(f"\n  --- Holdout (golden) ---")
    print(f"  Accuracy: {acc:.4%}")
    print(f"  Macro F1: {macro_f1:.4f}")
    print("\n  Classification report:")
    print(_manual_classification_report(y_test, y_pred))
    print("  Confusion matrix (rows=truth, cols=pred):")
    print(_manual_confusion_matrix(y_test, y_pred))

    # Predict on full dataframe
    print(f"\n  Predicting for all {len(df)} rows ...")
    X_all = df[FEATURE_COLS].fillna(0).values
    d_all = xgb.DMatrix(X_all, feature_names=FEATURE_COLS)
    proba_all = final_booster.predict(d_all)
    if proba_all.ndim == 1:
        proba_all = proba_all.reshape(-1, 4)
    pred_all = np.argmax(proba_all, axis=1)

    df["xgb_4class_pred"] = decode_labels(pred_all)
    for i, c in enumerate(CLASS_ORDER):
        df[f"xgb_4class_proba_{c}"] = proba_all[:, i]

    # Drop temp columns before save
    drop_cols = [c for c in df.columns if c.startswith("_")]
    out = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    out.to_parquet(OUTPUT_PATH, index=False)
    print(f"  Done. Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
