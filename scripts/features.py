import pandas as pd
import numpy as np
import json
import re
import logging
from datetime import datetime, timezone
from rapidfuzz import fuzz
from pathlib import Path

# Import local utility scripts
try:
    from scripts.website_validator import verify_website
    from scripts.phonenumber_validator import validate_phone_number
    from scripts.validator_cache import cached_validate
    from scripts.normalization import standardize_phone, normalize_website
except ImportError:
    # Fallback for direct script execution if PYTHONPATH is not set
    import sys
    sys.path.append(str(Path(__file__).parent))
    from website_validator import verify_website
    from phonenumber_validator import validate_phone_number
    from validator_cache import cached_validate
    from normalization import standardize_phone, normalize_website

logger = logging.getLogger(__name__)

# --- 1. JSON Helpers & Constants ---
NULL_TOKENS = {"null", "none", "nan", "n/a", "na", ""}

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
    obj = safe_json(val)
    return obj.get("primary", "") if isinstance(obj, dict) else ""

def extract_primary_category(val):
    obj = safe_json(val)
    return obj.get("primary", "") if isinstance(obj, dict) else ""

def extract_freeform_address(val):
    obj = safe_json(val)
    if isinstance(obj, list) and len(obj) > 0:
        first = obj[0]
        if isinstance(first, dict):
            return first.get("freeform", "")
    if isinstance(obj, dict):
        return obj.get("freeform", "")
    return ""

def extract_first_item(val):
    obj = safe_json(val)
    if isinstance(obj, list) and len(obj) > 0:
        return str(obj[0])
    if isinstance(obj, str) and obj:
        return obj
    return ""

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

# --- 2. Normalization Helpers ---
def normalize_domain(url: str) -> str:
    if not url:
        return ""
    # Delegate core normalization to shared website normalizer
    return normalize_website(url)

def _has_content(val):
    """Check if a field has non-null content (strings, lists, dicts)."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return False
    if isinstance(val, (list, dict, str)):
        return len(val) > 0
    return bool(val)

# --- 3. External Validator Wrappers ---
def _check_website(url):
    """Wrapper for verify_website. Prepends https:// if missing. Uses diskcache."""
    if not url or pd.isna(url): return 0
    url = str(url).strip()
    if not url: return 0
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    valid, _ = cached_validate('website', url, verify_website)
    return 1 if valid else 0

def _check_phone_number(phone):
    """Wrapper for validate_phone_number. Uses diskcache."""
    if not phone or pd.isna(phone): return 0
    valid, _ = cached_validate('phone', phone, validate_phone_number)
    return 1 if valid else 0

def jaccard_similarity(s1: str, s2: str) -> float:
    if not s1 or not s2: return 0.0
    tokens_a = set(s1.lower().split())
    tokens_b = set(s2.lower().split())
    if not tokens_a or not tokens_b: return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

REQUIRED_PHASE1_INPUT_COLS = [
    "confidence",
    "base_confidence",
    "norm_conflated_addr",
    "norm_base_addr",
    "addr_similarity_ratio",
    "norm_conflated_phone",
    "norm_base_phone",
    "phone_similarity",
    "norm_conflated_website",
    "norm_base_website",
    "website_similarity",
    "names",
    "base_names",
    "categories",
    "base_categories",
    "sources",
    "base_sources",
]


def assert_required_feature_inputs(df: pd.DataFrame) -> None:
    """Fail fast if required Phase 1/metadata columns are missing."""
    missing = [c for c in REQUIRED_PHASE1_INPUT_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"engineer_features missing required input columns: {missing}")


def standardize_feature_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize a few key inputs so that training and inference see the same
    representation for missing values and basic types.
    """
    df = df.copy()

    # Phones / websites: normalize None/NaN to empty string then canonical form
    for col in ["norm_conflated_phone", "norm_base_phone"]:
        if col in df.columns:
            df[col] = df[col].apply(standardize_phone)
    for col in ["norm_conflated_website", "norm_base_website"]:
        if col in df.columns:
            df[col] = df[col].apply(normalize_website)

    # Similarity scores: ensure numeric with NaNs for missing
    for col in ["addr_similarity_ratio", "phone_similarity", "website_similarity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# --- 4. Core Feature Engineering ---
def engineer_features(df: pd.DataFrame, validate_urls=False, validate_phones=False) -> pd.DataFrame:
    """
    Unified feature engineering for all models.
    Produces a superset of columns required by binary, multiclass, and pipeline scripts.
    """
    df = df.copy()
    assert_required_feature_inputs(df)
    df = standardize_feature_inputs(df)
    
    # Extraction
    df["_name"] = df["names"].apply(extract_primary_name)
    df["_base_name"] = df["base_names"].apply(extract_primary_name)
    df["_category"] = df["categories"].apply(extract_primary_category)
    df["_base_category"] = df["base_categories"].apply(extract_primary_category)

    src_info = df["sources"].apply(extract_sources_info)
    df["_src_count"] = src_info.apply(lambda x: x[0])
    df["_src_latest"] = src_info.apply(lambda x: x[1])
    df["_src_datasets"] = src_info.apply(lambda x: x[2])
    base_src_info = df["base_sources"].apply(extract_sources_info)
    df["_base_src_count"] = base_src_info.apply(lambda x: x[0])
    df["_base_src_latest"] = base_src_info.apply(lambda x: x[1])

    # A. Confidence / Existence
    df["feat_existence_conf_delta"] = df["confidence"] - df["base_confidence"]
    df["feat_match_exists_score"] = df["confidence"]
    df["feat_base_exists_score"] = df["base_confidence"]
    df["feat_confidence_delta"] = df["feat_existence_conf_delta"] # Synonym used in some scripts

    # B. Completeness
    df["feat_match_addr_len"] = df["norm_conflated_addr"].fillna("").str.len()
    df["feat_base_addr_len"] = df["norm_base_addr"].fillna("").str.len()
    df["feat_addr_richness_delta"] = df["feat_match_addr_len"] - df["feat_base_addr_len"]

    df["feat_match_has_phone"] = (df["norm_conflated_phone"].fillna("") != "").astype(int)
    df["feat_base_has_phone"] = (df["norm_base_phone"].fillna("") != "").astype(int)
    df["feat_match_has_web"] = (df["norm_conflated_website"].fillna("") != "").astype(int)
    df["feat_base_has_web"] = (df["norm_base_website"].fillna("") != "").astype(int)
    df["feat_phone_presence_delta"] = df["feat_match_has_phone"] - df["feat_base_has_phone"]
    df["feat_web_presence_delta"] = df["feat_match_has_web"] - df["feat_base_has_web"]

    # Brand & Socials
    for prefix in ["", "base_"]:
        brand_col = "brand" if prefix == "" else "base_brand"
        social_col = "socials" if prefix == "" else "base_socials"
        out_prefix = "match" if prefix == "" else "base"
        if brand_col in df.columns:
            df[f"feat_{out_prefix}_has_brand"] = df[brand_col].apply(_has_content).astype(int)
        if social_col in df.columns:
            df[f"feat_{out_prefix}_has_social"] = df[social_col].apply(_has_content).astype(int)
    
    if "feat_match_has_brand" in df.columns and "feat_base_has_brand" in df.columns:
        df["feat_brand_delta"] = df["feat_match_has_brand"] - df["feat_base_has_brand"]
    if "feat_match_has_social" in df.columns and "feat_base_has_social" in df.columns:
        df["feat_social_delta"] = df["feat_match_has_social"] - df["feat_base_has_social"]

    # C. Validation (Slow)
    if validate_urls:
        df["feat_match_web_valid"] = df["norm_conflated_website"].apply(_check_website).astype(int)
        df["feat_base_web_valid"] = df["norm_base_website"].apply(_check_website).astype(int)
        df["feat_web_valid_delta"] = df["feat_match_web_valid"] - df["feat_base_web_valid"]
    else:
        # Default to 0 or neutral if not validating
        df["feat_match_web_valid"] = 0
        df["feat_base_web_valid"] = 0
        df["feat_web_valid_delta"] = 0

    if validate_phones:
        df["feat_match_phone_valid"] = df["norm_conflated_phone"].apply(_check_phone_number).astype(int)
        df["feat_base_phone_valid"] = df["norm_base_phone"].apply(_check_phone_number).astype(int)
        df["feat_phone_valid_delta"] = df["feat_match_phone_valid"] - df["feat_base_phone_valid"]
    else:
        df["feat_match_phone_valid"] = 0
        df["feat_base_phone_valid"] = 0
        df["feat_phone_valid_delta"] = 0

    # D. Similarity
    df["feat_name_similarity"] = df.apply(
        lambda r: fuzz.token_sort_ratio(r["_name"], r["_base_name"]) / 100.0
        if r["_name"] and r["_base_name"] else 0.0, axis=1
    )
    df["feat_addr_similarity"] = df["addr_similarity_ratio"] / 100.0
    df["feat_phone_similarity"] = (df["phone_similarity"] / 100.0).fillna(0)
    df["feat_web_similarity"] = (df["website_similarity"] / 100.0).fillna(0)
    
    df["feat_phone_exact_match"] = (
        (df["norm_conflated_phone"].fillna("").astype(str) != "")
        & (df["norm_base_phone"].fillna("").astype(str) != "")
        & (df["norm_conflated_phone"].fillna("").astype(str) == df["norm_base_phone"].fillna("").astype(str))
    ).astype(int)
    
    df["feat_category_similarity"] = df.apply(
        lambda r: fuzz.token_sort_ratio(r["_category"], r["_base_category"]) / 100.0
        if r["_category"] and r["_base_category"] else 0.0, axis=1
    )
    df["feat_category_exact_match"] = (
        (df["_category"].fillna("") != "")
        & (df["_base_category"].fillna("") != "")
        & (df["_category"].fillna("") == df["_base_category"].fillna(""))
    ).astype(int)

    # E. Source Signals
    df["feat_is_msft_match"] = df["_src_datasets"].apply(lambda s: int("msft" in s) if isinstance(s, set) else 0)
    df["feat_is_meta_match"] = df["_src_datasets"].apply(lambda s: int("meta" in s) if isinstance(s, set) else 0)
    df["feat_src_count_delta"] = df["_src_count"] - df["_base_src_count"]

    # F. Recency
    ref = datetime.now(timezone.utc)
    def _days_since(d):
        if d is None: return 9999
        if d.tzinfo is None: d = d.replace(tzinfo=timezone.utc)
        return (ref - d).days
    
    df["feat_match_recency_days"] = df["_src_latest"].apply(_days_since)
    df["feat_base_recency_days"] = df["_base_src_latest"].apply(_days_since)
    df["feat_recency_delta"] = df["feat_base_recency_days"] - df["feat_match_recency_days"]

    # F. Composite
    # We use a safer approach for completeness to avoid KeyErrors if brand/social are missing
    for p in ["match", "base"]:
        brand_val = df[f"feat_{p}_has_brand"] if f"feat_{p}_has_brand" in df.columns else 0
        social_val = df[f"feat_{p}_has_social"] if f"feat_{p}_has_social" in df.columns else 0
        name_val = df["_name" if p == "match" else "_base_name"] != ""
        cat_val = df["_category" if p == "match" else "_base_category"] != ""
        addr_len = df[f"feat_{p}_addr_len"]
        
        df[f"feat_{p}_completeness"] = (
            df[f"feat_{p}_has_phone"] + df[f"feat_{p}_has_web"]
            + (addr_len > 0).astype(int) + name_val.astype(int) + cat_val.astype(int)
            + brand_val + social_val
        )
    
    df["feat_completeness_delta"] = df["feat_match_completeness"] - df["feat_base_completeness"]
    df["feat_valid_count_delta"] = (
        (df["feat_match_web_valid"] + df["feat_match_phone_valid"])
        - (df["feat_base_web_valid"] + df["feat_base_phone_valid"])
    )
    df["feat_name_addr_sim_product"] = df["feat_name_similarity"] * df["feat_addr_similarity"]
    df["feat_avg_similarity"] = (
        df["feat_name_similarity"] + df["feat_addr_similarity"]
        + df["feat_phone_similarity"] + df["feat_web_similarity"]
    ) / 4.0

    # G. Special Post-processing features
    df["feat_adds_new_info"] = df["feat_existence_conf_delta"] * (1.0 - df["feat_avg_similarity"])
    df["feat_addr_dissimilarity"] = 1.0 - df["feat_addr_similarity"]
    df["feat_conf_x_addr_richness"] = df["feat_existence_conf_delta"] * df["feat_addr_richness_delta"].clip(0, None)
    df["feat_phone_exclusive"] = ((df["feat_match_has_phone"] == 1) & (df["feat_base_has_phone"] == 0)).astype(int)
    df["feat_web_exclusive"] = ((df["feat_match_has_web"] == 1) & (df["feat_base_has_web"] == 0)).astype(int)
    
    return df
