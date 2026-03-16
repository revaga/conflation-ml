"""
Pipeline A: Fetch real-world place data from Google Maps/Places API and compare with base/alternate.
Reads data/golden_dataset_200.parquet, adds truth_* columns, writes data/ground_truth_google_golden.parquet.
Run from repo root: python external_validation/fetch_truth_google.py [--limit N] [--dry-run]
"""
from __future__ import annotations

import argparse
from typing import Any
import logging
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import json

from scripts.parquet_io import read_parquet_safe
from scripts.normalization import process_addresses, standardize_phone, normalize_website, normalize_address_json
from scripts.features import extract_primary_name, extract_primary_category

from external_validation.google_places_client import fetch_real_data, _load_api_key
from external_validation.compare import compare_row, truth_columns

# Normalized truth columns (compare normalized with normalized in verification)
TRUTH_NORM_COLUMNS = (
    "truth_phone_value_norm",
    "truth_web_value_norm",
    "truth_address_value_norm",
    "truth_category_value_norm",
)


def _norm_str(val) -> str:
    if val is None or (isinstance(val, float) and str(val) == "nan"):
        return ""
    return (str(val) or "").strip().lower()


def _normalize_truth_value(attr: str, raw: Any) -> str:
    """Normalize a single truth_*_value for storage and comparison (phone, web, address, category)."""
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
OUTPUT_PATH = DATA_DIR / "ground_truth_google_golden.parquet"


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Google Places truth for golden set")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N rows")
    parser.add_argument("--dry-run", action="store_true", help="Print queries only, no API calls")
    parser.add_argument("--input", type=str, default=None, help="Input parquet path (default: data/golden_dataset_200.parquet)")
    parser.add_argument("--output", type=str, default=None, help="Output parquet path (default: data/ground_truth_google_golden.parquet)")
    parser.add_argument("--no-fallback", action="store_true", help="Use no_data when real-world value missing (no base/alt fallback)")
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else GOLDEN_PATH
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = DATA_DIR / ("ground_truth_google_no_fallback.parquet" if args.no_fallback else "ground_truth_google_golden.parquet")

    if not input_path.exists():
        logger.error("Input not found: %s", input_path)
        sys.exit(1)

    logger.info("Loading %s", input_path)
    df = read_parquet_safe(str(input_path))
    df = ensure_phase1_columns(df)

    if args.limit:
        df = df.head(args.limit)
        logger.info("Limited to %d rows", len(df))

    api_key = _load_api_key()
    if not api_key and not args.dry_run:
        logger.warning("GOOGLE_PLACES_API_KEY not set; all real data will be empty.")

    # Add truth columns and normalized truth columns
    for col in truth_columns():
        if col not in df.columns:
            df[col] = ""
    for col in TRUTH_NORM_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    if "truth_source" not in df.columns:
        df["truth_source"] = ""

    for idx, row in df.iterrows():
        name = row.get("_name") or row.get("_base_name") or ""
        addr = row.get("norm_base_addr") or row.get("norm_conflated_addr") or ""
        if args.dry_run:
            logger.info("Query: %s | %s", name, addr)
            continue
        real = fetch_real_data(name, addr, api_key)
        cmp = compare_row(row, real, allow_fallback=not args.no_fallback)
        for k, v in cmp.items():
            df.at[idx, k] = v
        df.at[idx, "truth_source"] = "google_places"
        # Populate normalized truth columns (for compare-normalized-with-normalized in verification)
        df.at[idx, "truth_phone_value_norm"] = _normalize_truth_value("phone", cmp.get("truth_phone_value"))
        df.at[idx, "truth_web_value_norm"] = _normalize_truth_value("web", cmp.get("truth_web_value"))
        df.at[idx, "truth_address_value_norm"] = _normalize_truth_value("address", cmp.get("truth_address_value"))
        df.at[idx, "truth_category_value_norm"] = _normalize_truth_value("category", cmp.get("truth_category_value"))

    if args.dry_run:
        logger.info("Dry run done. Exiting without writing.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Wrote %s (%d rows)", output_path, len(df))


if __name__ == "__main__":
    main()
