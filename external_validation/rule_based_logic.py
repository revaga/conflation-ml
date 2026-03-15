"""
Rule-based: choose between base and alternate (conflated) with no external data.
When prefer_alt=False: pick base when base and alt disagree (original behavior).
When prefer_alt=True: use alt values as "real" so we prefer alt when they disagree → more alt.
Output: data/rule_based.parquet (same schema as other pipelines). Run verify_truth.py on it to compare to golden.
Run from repo root: python external_validation/rule_based_logic.py [--limit N] [--prefer-alt]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.parquet_io import read_parquet_safe
from scripts.normalization import process_addresses, standardize_phone, normalize_website, normalize_address_json
from scripts.features import extract_primary_name, extract_primary_category

from external_validation.compare import compare_row, truth_columns

# Normalized truth columns (compare normalized with normalized in verification)
TRUTH_NORM_COLUMNS = (
    "truth_phone_value_norm",
    "truth_web_value_norm",
    "truth_address_value_norm",
    "truth_category_value_norm",
)


def _norm_str(val: Any) -> str:
    if val is None or (isinstance(val, float) and str(val) == "nan"):
        return ""
    return (str(val) or "").strip().lower()


def _normalize_truth_value(attr: str, raw: Any) -> str:
    """Normalize a single truth_*_value for storage and comparison."""
    if raw is None or (isinstance(raw, float) and str(raw) == "nan"):
        raw = ""
    if attr == "phone":
        return standardize_phone(raw) or ""
    if attr == "web":
        if isinstance(raw, list):
            raw = (raw[0] if raw else "") or ""
        elif isinstance(raw, str) and raw.strip().startswith("["):
            try:
                arr = json.loads(raw)
                raw = (arr[0] if isinstance(arr, list) and arr else "") or ""
            except Exception:
                pass
        return normalize_website(raw) or ""
    if attr == "address":
        return normalize_address_json(raw) or ""
    if attr == "category":
        return _norm_str(raw)
    return ""

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = _REPO_ROOT / "data"
GOLDEN_PATH = DATA_DIR / "golden_dataset_200.parquet"
OUTPUT_PATH = DATA_DIR / "rule_based.parquet"

# Empty "real" dict: no external data, prefer base when disagree
EMPTY_REAL = {"phone": "", "web": "", "address": "", "category": ""}


def _str_val(v) -> str:
    if v is None or (isinstance(v, float) and str(v) == "nan"):
        return ""
    return str(v).strip() if v else ""


def real_from_alt(row: pd.Series) -> dict:
    """Build a 'real' dict from the row's alt (conflated) values. Passing this to compare_row makes it prefer alt when base and alt disagree."""
    return {
        "phone": _str_val(row.get("norm_conflated_phone")),
        "web": _str_val(row.get("norm_conflated_website")),
        "address": _str_val(row.get("norm_conflated_addr")),
        "category": _str_val(row.get("_category")),
    }


def ensure_phase1_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add norm_* and _name, _base_name, _category, _base_category if missing."""
    df = df.copy()
    if "norm_conflated_addr" not in df.columns:
        df = process_addresses(df)
    if "norm_conflated_phone" not in df.columns:
        df["norm_conflated_phone"] = df["phones"].apply(standardize_phone)
        df["norm_base_phone"] = df["base_phones"].apply(standardize_phone)
    if "norm_conflated_website" not in df.columns:
        df["norm_conflated_website"] = df["websites"].apply(normalize_website)
        df["norm_base_website"] = df["base_websites"].apply(normalize_website)
    if "_name" not in df.columns:
        df["_name"] = df["names"].apply(extract_primary_name)
        df["_base_name"] = df["base_names"].apply(extract_primary_name)
        df["_category"] = df["categories"].apply(extract_primary_category)
        df["_base_category"] = df["base_categories"].apply(extract_primary_category)
    return df


def compute_rule_based_truth(row: pd.Series, prefer_alt: bool = True) -> dict:
    """
    Compute truth_* winner and value for a row using only base vs alt (no external data).
    If prefer_alt=False: when base and alt disagree, choose base (empty real).
    If prefer_alt=True: use alt values as "real" so when they disagree we choose alt.
    """
    real = real_from_alt(row) if prefer_alt else EMPTY_REAL
    return compare_row(row, real)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rule-based truth (base vs alt only) for golden set")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N rows")
    parser.add_argument("--prefer-alt", action="store_true", help="Prefer alt when base and alt disagree (more alt)")
    parser.add_argument("--input", type=str, default=None, help="Input parquet path")
    parser.add_argument("--output", type=str, default=None, help="Output parquet path (default: data/rule_based.parquet)")
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else GOLDEN_PATH
    output_path = Path(args.output) if args.output else OUTPUT_PATH

    if not input_path.exists():
        logger.error("Input not found: %s", input_path)
        sys.exit(1)

    logger.info("Loading %s", input_path)
    df = read_parquet_safe(str(input_path))
    df = ensure_phase1_columns(df)

    if args.limit:
        df = df.head(args.limit)
        logger.info("Limited to %d rows", len(df))

    for col in truth_columns():
        if col not in df.columns:
            df[col] = ""
    for col in TRUTH_NORM_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    if "truth_source" not in df.columns:
        df["truth_source"] = ""

    if args.prefer_alt:
        logger.info("Using prefer_alt: will choose alt when base and alt disagree.")
    for idx, row in df.iterrows():
        cmp = compute_rule_based_truth(row, prefer_alt=args.prefer_alt)
        for k, v in cmp.items():
            df.at[idx, k] = v
        df.at[idx, "truth_source"] = "rule_based"
        # Normalized truth columns (for compare-normalized-with-normalized in verification)
        df.at[idx, "truth_phone_value_norm"] = _normalize_truth_value("phone", cmp.get("truth_phone_value"))
        df.at[idx, "truth_web_value_norm"] = _normalize_truth_value("web", cmp.get("truth_web_value"))
        df.at[idx, "truth_address_value_norm"] = _normalize_truth_value("address", cmp.get("truth_address_value"))
        df.at[idx, "truth_category_value_norm"] = _normalize_truth_value("category", cmp.get("truth_category_value"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Wrote %s (%d rows)", output_path, len(df))


if __name__ == "__main__":
    main()
