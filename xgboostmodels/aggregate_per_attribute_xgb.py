"""
Aggregate per-attribute XGBoost predictions into record-level
binary / 3-class / 4-class labels via the existing label rules.

This script expects the output of `train_per_attribute_xgb.py`, which contains
`xgb_attr_{attr}_pred` columns for each attribute in `ATTR_ATTRS`. It maps
those into a parallel set of `attr_{attr}_winner_xgb` columns, then uses the
label utilities from `scripts.labels` to compute:

- 4-class: none / alt / base / both
- 3-class: alt / both / base
- binary:  alt (1) / base (0)

Usage (from repo root):
    python -m xgboostmodels.aggregate_per_attribute_xgb \
        --input data/xgboost_per_attr_results.parquet \
        --output data/xgboost_per_attr_ensemble_results.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd

import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.labels import (  # type: ignore
    ATTR_ATTRS,
    recalculate_4class_label,
    recalculate_3class_label,
    row_to_2class,
)
from scripts.parquet_io import read_parquet_safe  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _normalize_attr_pred(val: str) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "none"
    v = str(val).strip().lower()
    if v in ("base", "alt", "both", "none"):
        return v
    if v in ("a", "match", "m"):
        return "alt"
    if v == "b":
        return "base"
    if v == "t":
        return "both"
    return "none"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate per-attribute XGBoost predictions into overall labels."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/xgboost_per_attr_results.parquet",
        help="Per-attribute XGBoost results parquet.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/xgboost_per_attr_ensemble_results.parquet",
        help="Output parquet with overall binary / 3-class / 4-class predictions.",
    )
    args = parser.parse_args()

    input_path = _REPO_ROOT / args.input
    logger.info("Loading per-attribute results from %s", input_path)
    df = read_parquet_safe(str(input_path))
    df["id"] = df["id"].astype(str)

    # Map per-attribute predictions into a suffix space the label helpers can use.
    suffix = "_xgb"
    attr_winner_cols: List[str] = []
    for attr in ATTR_ATTRS:
        src_col = f"xgb_attr_{attr}_pred"
        dst_col = f"attr_{attr}_winner{suffix}"
        if src_col not in df.columns:
            logger.warning("Column %s not found; skipping attribute '%s'.", src_col, attr)
            continue
        df[dst_col] = df[src_col].apply(_normalize_attr_pred)
        attr_winner_cols.append(dst_col)

    if not attr_winner_cols:
        logger.error("No per-attribute prediction columns found; nothing to aggregate.")
        return

    logger.info("Computing 4-class, 3-class, and binary labels from XGBoost attr winners.")
    df["xgb_4class_from_attr"] = df.apply(
        lambda row: recalculate_4class_label(row, suffix=suffix), axis=1
    )
    df["xgb_3class_from_attr"] = df.apply(
        lambda row: recalculate_3class_label(row, suffix=suffix), axis=1
    )
    df["xgb_binary_from_attr"] = df.apply(
        lambda row: row_to_2class(row, suffix=suffix), axis=1
    )

    output_path = _REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    keep_cols = [c for c in df.columns if not c.startswith("_")]
    df[keep_cols].to_parquet(output_path, index=False)
    logger.info("Aggregated XGBoost ensemble results written to %s", output_path)


if __name__ == "__main__":
    main()

