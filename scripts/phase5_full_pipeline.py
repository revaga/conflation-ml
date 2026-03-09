"""
Phase 5 — Full Conflict Resolution & Abstention Pipeline
=========================================================
Unified pipeline that reads raw data/project_a_samples.parquet, applies
enhanced normalization, engineers features, implements three selection
strategies (Highest Confidence, Rule-Based + Abstention, XGBoost +
Abstention), generates a comparison matrix, and classifies conflict
root causes.

Run from project root:
    python scripts/phase5_full_pipeline.py
"""

import json
import os
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from parquet_io import read_parquet_safe
import xgboost as xgb

# NOTE: scikit-learn removed due to DLL load issues on this Windows environment.
# Implementing metrics and train/test split manually.

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Manual sklearn replacements
# ---------------------------------------------------------------------------

def manual_train_test_split(X, y, test_size=0.2, random_state=42, stratify=None):
    """Stratified train/test split without sklearn."""
    rng = np.random.RandomState(random_state)
    n = len(X)

    if stratify is not None:
        train_idx, test_idx = [], []
        for cls in np.unique(stratify):
            cls_idx = np.where(stratify.values == cls)[0]
            rng.shuffle(cls_idx)
            n_test = max(1, int(len(cls_idx) * test_size))
            test_idx.extend(cls_idx[:n_test])
            train_idx.extend(cls_idx[n_test:])
    else:
        indices = np.arange(n)
        rng.shuffle(indices)
        split = int(n * (1 - test_size))
        train_idx = indices[:split]
        test_idx = indices[split:]

    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def manual_accuracy_score(y_true, y_pred):
    """Simple accuracy."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    return (y_t == y_p).sum() / len(y_t) if len(y_t) > 0 else 0.0


def manual_confusion_matrix(y_true, y_pred):
    """2-class confusion matrix [[TN, FP], [FN, TP]]."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    tp = int(((y_t == 1) & (y_p == 1)).sum())
    tn = int(((y_t == 0) & (y_p == 0)).sum())
    fp = int(((y_t == 0) & (y_p == 1)).sum())
    fn = int(((y_t == 1) & (y_p == 0)).sum())
    return np.array([[tn, fp], [fn, tp]])


def manual_classification_report(y_true, y_pred, target_names=None):
    """Minimal classification report string."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    classes = sorted(set(y_t) | set(y_p))
    if target_names is None:
        target_names = [str(c) for c in classes]

    lines = [f"{'':>20} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}"]
    lines.append("")

    for i, cls in enumerate(classes):
        tp = int(((y_t == cls) & (y_p == cls)).sum())
        fp = int(((y_t != cls) & (y_p == cls)).sum())
        fn = int(((y_t == cls) & (y_p != cls)).sum())
        support = int((y_t == cls).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        name = target_names[i] if i < len(target_names) else str(cls)
        lines.append(f"{name:>20} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10}")

    total = len(y_t)
    acc = manual_accuracy_score(y_t, y_p)
    lines.append("")
    lines.append(f"{'accuracy':>20} {'':>10} {'':>10} {acc:>10.4f} {total:>10}")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "project_a_samples.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "phase5_full_results.parquet"
REPORTS_DIR = PROJECT_ROOT / "reports"
LABEL_COL = "_heuristic_label"
RANDOM_STATE = 42

# ===========================================================================
#  SECTION 1 — Enhanced Normalization
# ===========================================================================

# --- Null Canonicalization ---
NULL_TOKENS = {"null", "none", "nan", "n/a", "na", ""}


def canonicalize_null(val):
    """Map null-like values to None consistently."""
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    s = str(val).strip()
    # Handle JSON-encoded nulls like '[""]', '[null]', '["NULL"]'
    if s in ('[""]', "[null]", '["NULL"]', '["None"]', "[]"):
        return None
    if s.lower() in NULL_TOKENS:
        return None
    return val


# --- JSON Helpers ---
def safe_json(x):
    """Parse a JSON string, returning {} or [] on failure."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return {}
    if isinstance(x, (dict, list)):
        return x
    try:
        return json.loads(str(x))
    except (json.JSONDecodeError, TypeError):
        return {}


def extract_primary_name(val):
    """Extract the 'primary' name from a names JSON object."""
    obj = safe_json(val)
    if isinstance(obj, dict):
        return obj.get("primary", "")
    return ""


def extract_freeform_address(val):
    """Extract the freeform address string from an addresses JSON array."""
    obj = safe_json(val)
    if isinstance(obj, list) and len(obj) > 0:
        first = obj[0]
        if isinstance(first, dict):
            return first.get("freeform", "")
    if isinstance(obj, dict):
        return obj.get("freeform", "")
    return ""


def extract_first_item(val):
    """Get the first string element from a JSON array (phones, websites)."""
    obj = safe_json(val)
    if isinstance(obj, list) and len(obj) > 0:
        return str(obj[0])
    if isinstance(obj, str) and obj:
        return obj
    return ""


def extract_primary_category(val):
    """Extract 'primary' category from a categories JSON object."""
    obj = safe_json(val)
    if isinstance(obj, dict):
        return obj.get("primary", "")
    return ""


# --- Phone Normalization ---
def normalize_phone(phone):
    """Strip all non-digit characters from phone number."""
    if not phone:
        return ""
    return re.sub(r"\D", "", str(phone))


def phone_format_valid(phone_digits: str) -> int:
    """Binary: does phone meet 7–15 digit length requirement?"""
    if not phone_digits:
        return 0
    length = len(phone_digits)
    return 1 if 7 <= length <= 15 else 0


# --- Address Normalization ---
ABBR_MAP = {
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bdr\b": "drive",
    r"\brd\b": "road",
    r"\bblvd\b": "boulevard",
    r"\bln\b": "lane",
    r"\bct\b": "court",
    r"\bpl\b": "place",
    r"\bsq\b": "square",
    r"\bpkwy\b": "parkway",
    r"\bcir\b": "circle",
    r"\bhwy\b": "highway",
}

SUITE_PATTERN = re.compile(
    r"\b(ste|suite|apt|unit|#)\s*[\w-]*", re.IGNORECASE
)


def normalize_address(addr: str) -> str:
    """Lowercase, expand abbreviations, strip suite/unit for semantic core."""
    if not addr or pd.isna(addr):
        return ""
    normalized = str(addr).lower().strip()
    for pattern, replacement in ABBR_MAP.items():
        normalized = re.sub(pattern, replacement, normalized)
    # Strip suite/unit variations for semantic comparison
    normalized = SUITE_PATTERN.sub("", normalized).strip()
    # Collapse whitespace
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


# --- Website / Email Domain ---
def normalize_domain(url: str) -> str:
    """Strip protocol & www to get a bare domain for comparison."""
    if not url:
        return ""
    url = url.lower().strip()
    url = re.sub(r"^https?://", "", url)
    url = re.sub(r"^www\.", "", url)
    return url.strip("/").split("/")[0]


def extract_email_domain(email_val) -> str:
    """Extract domain from an email address."""
    if not email_val or (isinstance(email_val, float) and np.isnan(email_val)):
        return ""
    s = str(email_val).strip()
    # Handle JSON arrays of emails
    obj = safe_json(s)
    if isinstance(obj, list) and len(obj) > 0:
        s = str(obj[0])
    if "@" in s:
        return s.split("@")[-1].lower().strip()
    return ""


# --- Source Metadata ---
def extract_sources_info(val):
    """Return (source_count, latest_update_datetime, set_of_dataset_names)."""
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
            except (ValueError, TypeError):
                pass
    return count, latest_dt, datasets


# --- Jaccard Similarity ---
def jaccard_similarity(s1: str, s2: str) -> float:
    """Compute Jaccard similarity between word token sets."""
    if not s1 or not s2:
        return 0.0
    tokens_a = set(s1.lower().split())
    tokens_b = set(s2.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


# ===========================================================================
#  SECTION 2 — Feature Engineering
# ===========================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create all numeric features from raw columns. Returns the DataFrame."""

    print("  Canonicalizing nulls ...")
    null_cols = ["phones", "base_phones", "websites", "base_websites",
                 "emails", "base_emails", "names", "base_names",
                 "addresses", "base_addresses"]
    for col in null_cols:
        if col in df.columns:
            df[col] = df[col].apply(canonicalize_null)

    print("  Flattening JSON columns ...")
    # --- Flatten ---
    df["_name"] = df["names"].apply(extract_primary_name)
    df["_base_name"] = df["base_names"].apply(extract_primary_name)

    df["_addr_raw"] = df["addresses"].apply(extract_freeform_address)
    df["_base_addr_raw"] = df["base_addresses"].apply(extract_freeform_address)

    df["_addr"] = df["_addr_raw"].apply(normalize_address)
    df["_base_addr"] = df["_base_addr_raw"].apply(normalize_address)

    df["_phone"] = df["phones"].apply(extract_first_item).apply(normalize_phone)
    df["_base_phone"] = df["base_phones"].apply(extract_first_item).apply(normalize_phone)

    df["_website"] = df["websites"].apply(extract_first_item)
    df["_base_website"] = df["base_websites"].apply(extract_first_item)

    df["_category"] = df["categories"].apply(extract_primary_category)
    df["_base_category"] = df["base_categories"].apply(extract_primary_category)

    df["_email_domain"] = df["emails"].apply(extract_email_domain)
    df["_website_domain"] = df["_website"].apply(normalize_domain)

    # Sources metadata
    src_info = df["sources"].apply(extract_sources_info)
    df["_src_count"] = src_info.apply(lambda x: x[0])
    df["_src_latest"] = src_info.apply(lambda x: x[1])
    df["_src_datasets"] = src_info.apply(lambda x: x[2])

    base_src_info = df["base_sources"].apply(extract_sources_info)
    df["_base_src_count"] = base_src_info.apply(lambda x: x[0])
    df["_base_src_latest"] = base_src_info.apply(lambda x: x[1])

    print("  Computing features ...")

    # A. Trust & Confidence
    df["feat_confidence_delta"] = df["confidence"] - df["base_confidence"]
    df["feat_source_count"] = df["_src_count"]
    df["feat_base_source_count"] = df["_base_src_count"]

    # B. Freshness — recency gap in days (positive = match is newer)
    def recency_gap(row):
        match_dt = row["_src_latest"]
        base_dt = row["_base_src_latest"]
        if match_dt is None or base_dt is None:
            return 0.0
        try:
            if match_dt.tzinfo is None:
                match_dt = match_dt.replace(tzinfo=timezone.utc)
            if base_dt.tzinfo is None:
                base_dt = base_dt.replace(tzinfo=timezone.utc)
            return (match_dt - base_dt).total_seconds() / 86400.0
        except Exception:
            return 0.0

    df["feat_recency_gap_days"] = df.apply(recency_gap, axis=1)

    # C. Similarity (NLP) — Token Sort Ratio
    df["feat_name_similarity"] = df.apply(
        lambda r: fuzz.token_sort_ratio(r["_name"], r["_base_name"]) / 100.0
        if r["_name"] and r["_base_name"] else 0.0,
        axis=1,
    )
    df["feat_addr_similarity"] = df.apply(
        lambda r: fuzz.token_sort_ratio(r["_addr"], r["_base_addr"]) / 100.0
        if r["_addr"] and r["_base_addr"] else 0.0,
        axis=1,
    )

    # D. NEW — Jaccard Similarity
    df["feat_name_jaccard"] = df.apply(
        lambda r: jaccard_similarity(r["_name"], r["_base_name"]), axis=1
    )
    df["feat_addr_jaccard"] = df.apply(
        lambda r: jaccard_similarity(r["_addr"], r["_base_addr"]), axis=1
    )

    # E. Phone
    df["feat_phone_match"] = (
        (df["_phone"] != "") & (df["_phone"] == df["_base_phone"])
    ).astype(int)

    # NEW — Phone format score
    df["feat_phone_format_score"] = df["_phone"].apply(phone_format_valid)

    # F. Website domain match
    df["feat_website_domain_match"] = df.apply(
        lambda r: int(
            normalize_domain(r["_website"]) == normalize_domain(r["_base_website"])
            and normalize_domain(r["_website"]) != ""
        ),
        axis=1,
    )

    # G. NEW — Internal Consistency (website domain == email domain)
    df["feat_internal_consistency"] = df.apply(
        lambda r: int(
            r["_website_domain"] != ""
            and r["_email_domain"] != ""
            and r["_website_domain"] == r["_email_domain"]
        ),
        axis=1,
    )

    # H. Completeness
    df["feat_match_addr_completeness"] = df["_addr"].str.len()
    df["feat_base_addr_completeness"] = df["_base_addr"].str.len()
    df["feat_addr_completeness_delta"] = (
        df["feat_match_addr_completeness"] - df["feat_base_addr_completeness"]
    )

    # I. Source-specific flags
    df["feat_is_meta_source"] = df["_src_datasets"].apply(
        lambda s: int("meta" in s) if isinstance(s, set) else 0
    )
    df["feat_is_msft_source"] = df["_src_datasets"].apply(
        lambda s: int("msft" in s) if isinstance(s, set) else 0
    )

    # J. Category match
    df["feat_category_match"] = (
        (df["_category"] != "") & (df["_category"] == df["_base_category"])
    ).astype(int)

    return df


# ===========================================================================
#  SECTION 3 — Tier 1: Abstention Logic
# ===========================================================================

def apply_abstention(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag high-risk rows for human review.

    Rule: confidence < 0.6 AND base_confidence < 0.6
          AND at least one key attribute disagrees (similarity < 0.80).
    """
    low_confidence = (df["confidence"] < 0.6) & (df["base_confidence"] < 0.6)

    # Check if any key attribute disagrees
    name_disagree = df["feat_name_similarity"] < 0.80
    addr_disagree = df["feat_addr_similarity"] < 0.80
    phone_disagree = df["feat_phone_match"] == 0

    any_disagree = name_disagree | addr_disagree | phone_disagree

    df["abstention_flag"] = np.where(
        low_confidence & any_disagree,
        "ABSTAIN",
        "RESOLVABLE",
    )

    n_abstain = (df["abstention_flag"] == "ABSTAIN").sum()
    n_resolve = (df["abstention_flag"] == "RESOLVABLE").sum()
    pct = n_abstain / len(df) * 100

    print(f"\n  Abstention: {n_abstain} rows flagged ({pct:.1f}%)")
    print(f"  Resolvable: {n_resolve} rows")

    return df


# ===========================================================================
#  SECTION 4 — Feature Columns & Heuristic Labeling
# ===========================================================================

FEATURE_COLS = [
    "feat_confidence_delta",
    "feat_source_count",
    "feat_base_source_count",
    "feat_recency_gap_days",
    "feat_name_similarity",
    "feat_addr_similarity",
    "feat_name_jaccard",
    "feat_addr_jaccard",
    "feat_phone_match",
    "feat_phone_format_score",
    "feat_website_domain_match",
    "feat_internal_consistency",
    "feat_match_addr_completeness",
    "feat_base_addr_completeness",
    "feat_addr_completeness_delta",
    "feat_is_meta_source",
    "feat_is_msft_source",
    "feat_category_match",
]


def generate_heuristic_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign heuristic training labels.
        1 = matched candidate is better than base
        0 = base is better / keep base
       NaN = ambiguous (excluded from training)
    """
    conditions_positive = (
        (df["feat_confidence_delta"] > 0.1)
        & (df["feat_name_similarity"] > 0.60)
        & (df["feat_addr_similarity"] > 0.50)
    )

    conditions_negative = (
        (df["feat_confidence_delta"] < -0.05)
        | (df["feat_name_similarity"] < 0.40)
        | (df["feat_addr_similarity"] < 0.30)
    )

    df[LABEL_COL] = np.nan
    df.loc[conditions_positive, LABEL_COL] = 1.0
    df.loc[conditions_negative & ~conditions_positive, LABEL_COL] = 0.0

    n_pos = (df[LABEL_COL] == 1).sum()
    n_neg = (df[LABEL_COL] == 0).sum()
    n_amb = df[LABEL_COL].isna().sum()
    print(f"\n  Heuristic labels — Positive: {n_pos}, Negative: {n_neg}, Ambiguous (dropped): {n_amb}")

    return df


# ===========================================================================
#  SECTION 5 — Three Selection Strategies
# ===========================================================================

def strategy_highest_confidence(df: pd.DataFrame) -> pd.Series:
    """Baseline: pick whichever source has higher confidence."""
    return (df["confidence"] >= df["base_confidence"]).astype(int)


def strategy_rule_based(df: pd.DataFrame) -> pd.Series:
    """
    Rule-based selection on resolvable rows.
    Pick match (1) if: confidence_delta > 0 AND name_sim > 0.6 AND addr_sim > 0.5.
    Abstained rows get prediction = -1 (human review).
    """
    rule_pred = np.where(
        (df["feat_confidence_delta"] > 0)
        & (df["feat_name_similarity"] > 0.60)
        & (df["feat_addr_similarity"] > 0.50),
        1, 0
    )
    # Mark abstained rows as -1
    rule_pred = np.where(df["abstention_flag"] == "ABSTAIN", -1, rule_pred)
    return pd.Series(rule_pred, index=df.index)


def strategy_xgboost(df: pd.DataFrame) -> tuple:
    """
    XGBoost + Abstention. Train on labeled, resolvable rows.
    Returns (predictions_series, model, test_accuracy).
    Abstained rows get prediction = -1.
    Uses native xgb API (not sklearn wrapper) to avoid DLL issues.
    """
    # Train only on resolvable, labeled rows
    trainable = df[
        (df["abstention_flag"] == "RESOLVABLE") & df[LABEL_COL].notna()
    ].copy()

    if len(trainable) < 20:
        print("  ERROR: Not enough labeled resolvable rows to train XGBoost.")
        return pd.Series(-1, index=df.index), None, 0.0

    X = trainable[FEATURE_COLS].fillna(0)
    y = trainable[LABEL_COL].astype(int)

    X_train, X_test, y_train, y_test = manual_train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y,
    )

    print(f"\n  Training set: {len(X_train)} rows  |  Test set: {len(X_test)} rows")
    print(f"  Train class dist: {dict(y_train.value_counts())}")
    print(f"  Test  class dist: {dict(y_test.value_counts())}")

    # Native XGBoost API (avoids sklearn DLL dependency)
    dtrain = xgb.DMatrix(X_train.values, label=y_train.values, feature_names=FEATURE_COLS)
    dtest = xgb.DMatrix(X_test.values, label=y_test.values, feature_names=FEATURE_COLS)

    params = {
        "max_depth": 5,
        "eta": 0.1,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "seed": RANDOM_STATE,
    }

    model = xgb.train(params, dtrain, num_boost_round=200)

    # Evaluation on test set
    y_prob_test = model.predict(dtest)
    y_pred_test = (y_prob_test >= 0.5).astype(int)
    test_acc = manual_accuracy_score(y_test, y_pred_test)

    print(f"\n  XGBoost Test Accuracy: {test_acc:.4f}")
    print("\n  Classification Report:")
    print(manual_classification_report(y_test, y_pred_test,
          target_names=["Keep Base (0)", "Use Match (1)"]))

    cm = manual_confusion_matrix(y_test, y_pred_test)
    print("  Confusion Matrix:")
    print(f"                    Pred: Keep Base   Pred: Use Match")
    if cm.shape == (2, 2):
        print(f"    Actual Keep Base    {cm[0][0]:<16}  {cm[0][1]}")
        print(f"    Actual Use Match    {cm[1][0]:<16}  {cm[1][1]}")
    else:
        print(f"    {cm}")

    # Feature importance
    importance_raw = model.get_score(importance_type="gain")
    total_gain = sum(importance_raw.values()) if importance_raw else 1
    importance = {k: v / total_gain for k, v in importance_raw.items()}
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print("\n  Feature Importance (Gain):")
    print("  " + "-" * 55)
    for feat, score in sorted_imp:
        bar = "█" * int(score * 40)
        print(f"    {feat:<35} {score:.4f}  {bar}")

    # Predict on ALL rows
    X_all = df[FEATURE_COLS].fillna(0)
    dall = xgb.DMatrix(X_all.values, feature_names=FEATURE_COLS)
    probs = model.predict(dall)
    preds = (probs >= 0.5).astype(int)

    # Override abstained rows to -1
    preds = np.where(df["abstention_flag"] == "ABSTAIN", -1, preds)

    df["xgb_probability"] = probs

    return pd.Series(preds, index=df.index), model, test_acc


# ===========================================================================
#  SECTION 6 — Comparison Matrix
# ===========================================================================

def compute_comparison_matrix(df: pd.DataFrame, test_acc_xgb: float):
    """
    Compare all three strategies against heuristic labels.
    Only evaluates on labeled rows (where we have ground truth proxy).
    """
    labeled = df[df[LABEL_COL].notna()].copy()
    y_true = labeled[LABEL_COL].astype(int)

    results = {}

    # 1. Highest Confidence
    y_base = labeled["baseline_pred"]
    acc_base = manual_accuracy_score(y_true, y_base)
    err_base = 1 - acc_base
    results["Highest Confidence"] = {
        "Accuracy": f"{acc_base:.2%}",
        "Error Rate": f"{err_base:.2%}",
        "Automation %": "100.0%",
        "Abstention %": "N/A",
    }

    # 2. Rule-Based + Abstention
    rule_labeled = labeled[labeled["rule_pred"] != -1]
    if len(rule_labeled) > 0:
        y_rule = rule_labeled["rule_pred"].astype(int)
        y_true_rule = rule_labeled[LABEL_COL].astype(int)
        acc_rule = manual_accuracy_score(y_true_rule, y_rule)
        err_rule = 1 - acc_rule
    else:
        acc_rule, err_rule = 0.0, 1.0

    n_auto_rule = (labeled["rule_pred"] != -1).sum()
    auto_pct_rule = n_auto_rule / len(labeled) * 100
    abstain_pct_rule = 100 - auto_pct_rule
    results["Rule-Based + Abstention"] = {
        "Accuracy": f"{acc_rule:.2%}",
        "Error Rate": f"{err_rule:.2%}",
        "Automation %": f"{auto_pct_rule:.1f}%",
        "Abstention %": f"{abstain_pct_rule:.1f}%",
    }

    # 3. XGBoost + Abstention
    xgb_labeled = labeled[labeled["xgb_pred"] != -1]
    if len(xgb_labeled) > 0:
        y_xgb = xgb_labeled["xgb_pred"].astype(int)
        y_true_xgb = xgb_labeled[LABEL_COL].astype(int)
        acc_xgb = manual_accuracy_score(y_true_xgb, y_xgb)
        err_xgb = 1 - acc_xgb
    else:
        acc_xgb, err_xgb = 0.0, 1.0

    n_auto_xgb = (labeled["xgb_pred"] != -1).sum()
    auto_pct_xgb = n_auto_xgb / len(labeled) * 100
    abstain_pct_xgb = 100 - auto_pct_xgb
    results["XGBoost + Abstention"] = {
        "Accuracy": f"{acc_xgb:.2%}",
        "Error Rate": f"{err_xgb:.2%}",
        "Automation %": f"{auto_pct_xgb:.1f}%",
        "Abstention %": f"{abstain_pct_xgb:.1f}%",
    }

    # Format table
    metrics = ["Accuracy", "Error Rate", "Automation %", "Abstention %"]
    strategies = ["Highest Confidence", "Rule-Based + Abstention", "XGBoost + Abstention"]

    header = f"{'Metric':<18} | {'Highest Confidence':<22} | {'Rule-Based + Abstention':<26} | {'XGBoost + Abstention':<22}"
    sep = "-" * len(header)

    lines = [sep, header, sep]
    for metric in metrics:
        vals = [results[s][metric] for s in strategies]
        line = f"{metric:<18} | {vals[0]:<22} | {vals[1]:<26} | {vals[2]:<22}"
        lines.append(line)
    lines.append(sep)

    table_str = "\n".join(lines)
    return table_str, results


# ===========================================================================
#  SECTION 7 — Conflict Root Cause Classifier
# ===========================================================================

def classify_root_cause(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label the root cause of each conflict.

    Labels:
      - Formatting Variation: values differ raw but normalized similarity > 95%
      - Outdated Info: recency gap > 90 days AND values differ
      - Semantic Difference: normalized similarity < 60%
      - High-Risk Ambiguity: row flagged for abstention
      - Agreement: no meaningful conflict (similarity >= 95%)
    """
    causes = []

    for _, row in df.iterrows():
        # Check if there's any conflict at all
        name_sim = row.get("feat_name_similarity", 1.0)
        addr_sim = row.get("feat_addr_similarity", 1.0)
        phone_match = row.get("feat_phone_match", 1)

        # No conflict — high agreement
        if name_sim >= 0.95 and addr_sim >= 0.95 and phone_match == 1:
            causes.append("Agreement")
            continue

        # High-Risk Ambiguity (abstention flag)
        if row.get("abstention_flag") == "ABSTAIN":
            causes.append("High-Risk Ambiguity")
            continue

        # Formatting Variation — raw values differ but normalized very similar
        if name_sim >= 0.95 or addr_sim >= 0.95:
            causes.append("Formatting Variation")
            continue

        # Outdated Info — large recency gap with differences
        recency = abs(row.get("feat_recency_gap_days", 0))
        if recency > 90 and (name_sim < 0.90 or addr_sim < 0.90):
            causes.append("Outdated Info")
            continue

        # Semantic Difference — fundamentally different values
        if name_sim < 0.60 or addr_sim < 0.60:
            causes.append("Semantic Difference")
            continue

        # Default: moderate difference
        causes.append("Formatting Variation")

    df["conflict_root_cause"] = causes
    return df


def generate_root_cause_report(df: pd.DataFrame) -> str:
    """Generate a root cause distribution summary."""
    dist = df["conflict_root_cause"].value_counts()
    total = len(df)

    lines = [
        "=" * 50,
        "CONFLICT ROOT CAUSE DISTRIBUTION",
        "=" * 50,
        "",
    ]

    for label, count in dist.items():
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        lines.append(f"  {label:<25} {count:>5}  ({pct:>5.1f}%)  {bar}")

    lines.append("")
    lines.append(f"  {'TOTAL':<25} {total:>5}")
    lines.append("=" * 50)

    return "\n".join(lines)


# ===========================================================================
#  SECTION 8 — Main Pipeline
# ===========================================================================

def main():
    print("=" * 65)
    print("  PHASE 5 — Full Conflict Resolution & Abstention Pipeline")
    print("=" * 65)

    # --- Load ---
    print(f"\nLoading {INPUT_PATH} ...")
    df = read_parquet_safe(str(INPUT_PATH))
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns.")

    # --- Feature Engineering ---
    print("\n--- Section 1 & 2: Enhanced Normalization + Feature Engineering ---")
    df = engineer_features(df)

    # --- Abstention ---
    print("\n--- Section 3: Tier 1 — Abstention Logic ---")
    df = apply_abstention(df)

    # --- Heuristic Labeling ---
    print("\n--- Section 4: Heuristic Labeling ---")
    df = generate_heuristic_labels(df)

    # --- Three Strategies ---
    print("\n--- Section 5: Selection Strategies ---")

    print("\n  [1/3] Baseline: Highest Confidence Wins")
    df["baseline_pred"] = strategy_highest_confidence(df)
    n_match_base = (df["baseline_pred"] == 1).sum()
    print(f"        Use Match: {n_match_base}, Keep Base: {len(df) - n_match_base}")

    print("\n  [2/3] Rule-Based + Abstention")
    df["rule_pred"] = strategy_rule_based(df)
    n_abstain_rule = (df["rule_pred"] == -1).sum()
    n_match_rule = (df["rule_pred"] == 1).sum()
    n_keep_rule = (df["rule_pred"] == 0).sum()
    print(f"        Use Match: {n_match_rule}, Keep Base: {n_keep_rule}, Human Review: {n_abstain_rule}")

    print("\n  [3/3] XGBoost + Abstention")
    xgb_preds, model, test_acc_xgb = strategy_xgboost(df)
    df["xgb_pred"] = xgb_preds
    n_abstain_xgb = (df["xgb_pred"] == -1).sum()
    n_match_xgb = (df["xgb_pred"] == 1).sum()
    n_keep_xgb = (df["xgb_pred"] == 0).sum()
    print(f"\n        Use Match: {n_match_xgb}, Keep Base: {n_keep_xgb}, Human Review: {n_abstain_xgb}")

    # --- Comparison Matrix ---
    print("\n\n" + "=" * 65)
    print("  COMPARISON MATRIX")
    print("=" * 65 + "\n")
    table_str, results = compute_comparison_matrix(df, test_acc_xgb)
    print(table_str)

    # --- Root Cause Classifier ---
    print("\n\n--- Section 7: Conflict Root Cause Classifier ---")
    df = classify_root_cause(df)
    root_cause_report = generate_root_cause_report(df)
    print("\n" + root_cause_report)

    # --- Save Outputs ---
    print("\n--- Saving Outputs ---")

    # Create reports directory
    REPORTS_DIR.mkdir(exist_ok=True)

    # 1. Full results parquet (drop temp cols)
    temp_cols = [c for c in df.columns if c.startswith("_")]
    df_save = df.drop(columns=temp_cols)
    df_save.to_parquet(OUTPUT_PATH, index=False)
    print(f"  Saved full results to {OUTPUT_PATH}")

    # 2. Comparison matrix
    matrix_path = REPORTS_DIR / "comparison_matrix.txt"
    with open(matrix_path, "w", encoding="utf-8") as f:
        f.write("CONFLICT RESOLUTION — STRATEGY COMPARISON\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Dataset: {len(df)} rows\n\n")
        f.write(table_str + "\n")
    print(f"  Saved comparison matrix to {matrix_path}")

    # 3. Root cause report
    rc_path = REPORTS_DIR / "root_cause_report.txt"
    with open(rc_path, "w", encoding="utf-8") as f:
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Dataset: {len(df)} rows\n\n")
        f.write(root_cause_report + "\n")
    print(f"  Saved root cause report to {rc_path}")

    # 4. Abstention candidates CSV
    abstain_df = df[df["abstention_flag"] == "ABSTAIN"].copy()
    if len(abstain_df) > 0:
        export_cols = ["id", "base_id", "confidence", "base_confidence",
                       "_name", "_base_name", "_addr", "_base_addr",
                       "_phone", "_base_phone",
                       "feat_name_similarity", "feat_addr_similarity",
                       "conflict_root_cause"]
        # Only use columns that exist (temp cols already dropped in df_save but not in df)
        available = [c for c in export_cols if c in df.columns]
        abstain_path = REPORTS_DIR / "abstention_candidates.csv"
        abstain_df[available].to_csv(abstain_path, index=False, encoding="utf-8")
        print(f"  Saved {len(abstain_df)} abstention candidates to {abstain_path}")
    else:
        print("  No abstention candidates to save.")

    print("\n" + "=" * 65)
    print("  Pipeline complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()
