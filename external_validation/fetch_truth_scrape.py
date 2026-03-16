"""
Pipeline B: Fetch real-world place data by scraping business websites and non-Google search.
Reads data/golden_dataset_200.parquet, adds truth_* columns, writes data/ground_truth_scrape_golden.parquet.
Run from repo root: python external_validation/fetch_truth_scrape.py [--limit N]
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

from external_validation.scrape_place import scrape_place
from external_validation.non_google_search import search_place as search_place_ddg
from external_validation.compare import compare_row, truth_columns

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
OUTPUT_PATH = DATA_DIR / "ground_truth_scrape_golden.parquet"


def _normalize_website_for_url(val) -> str:
    """Get a usable URL for scraping (add scheme if missing)."""
    if val is None or (isinstance(val, float) and str(val) == "nan"):
        return ""
    s = str(val).strip()
    if not s:
        return ""
    if not s.startswith(("http://", "https://")):
        s = "https://" + s
    return s


def ensure_phase1_columns(df: pd.DataFrame) -> pd.DataFrame:
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


def merge_real(scraped: dict, search_result: dict) -> dict:
    """Merge scraped and search result; scraped wins when present."""
    out = {"phone": "", "web": "", "address": "", "category": ""}
    for k in out:
        out[k] = (scraped.get(k) or "").strip() or (search_result.get(k) or "").strip()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch scrape+search truth for golden set")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N rows")
    parser.add_argument("--input", type=str, default=None, help="Input parquet path")
    parser.add_argument("--output", type=str, default=None, help="Output parquet path")
    parser.add_argument("--no-fallback", action="store_true", help="Use no_data when real-world value missing (no base/alt fallback)")
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else GOLDEN_PATH
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = DATA_DIR / ("ground_truth_scrape_no_fallback.parquet" if args.no_fallback else "ground_truth_scrape_golden.parquet")

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

    for idx, row in df.iterrows():
        name = row.get("_name") or row.get("_base_name") or ""
        addr = row.get("norm_base_addr") or row.get("norm_conflated_addr") or ""
        query = f"{name} {addr}".strip()

        # (a) Scrape base or conflated website if present
        scraped = {"phone": "", "web": "", "address": "", "category": ""}
        for url_col, name_col in [("base_websites", "_base_name"), ("websites", "_name")]:
            url = row.get(url_col)
            url = _normalize_website_for_url(url)
            if url:
                s = scrape_place(url, row.get(name_col), use_cache=True)
                for k in scraped:
                    if s.get(k):
                        scraped[k] = s[k]
                if any(scraped.values()):
                    break

        # (b) Non-Google search
        search_result = search_place_ddg(query)

        # (c) Merge
        real = merge_real(scraped, search_result)

        # (d) Compare and set truth columns
        cmp = compare_row(row, real, allow_fallback=not args.no_fallback)
        for k, v in cmp.items():
            df.at[idx, k] = v
        df.at[idx, "truth_source"] = "scrape_and_search"
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
