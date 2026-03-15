import pandas as pd
import logging
from typing import Iterable, Sequence, Dict, List

from scripts.labels import LABEL_3CLASS, LABEL_4CLASS, ATTR_ATTRS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column sets for different stages of the pipeline
# ---------------------------------------------------------------------------

# Phase 1 *processed* data – produced by phase1_data_prep.py and consumed by
# feature engineering and downstream models.
PHASE1_COLUMNS: List[str] = [
    "id",
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
]

# Golden 3-class labels used for evaluation.
GOLDEN_COLUMNS: List[str] = [
    "id",
    "3class_testlabels",
]

# Attribute-level winners coming from the SLM / golden labelling.
ATTR_WINNER_COLUMNS: List[str] = [
    f"attr_{attr}_winner" for attr in ATTR_ATTRS
]

# Phase 3 (SLM-labeled) output is expected to contain Phase 1 processed
# similarity columns plus attribute winners.
PHASE3_COLUMNS: List[str] = PHASE1_COLUMNS + ATTR_WINNER_COLUMNS


def validate_schema(df: pd.DataFrame, required_cols: Sequence[str], stage_name: str) -> bool:
    """
    Verify that the dataframe contains all required columns for the specified stage.

    This is a structural check only; value-domain checks are handled separately.
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        error_msg = f"Schema Validation FAILED for {stage_name}. Missing columns: {missing}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    logger.info(f"Schema Validation PASSED for {stage_name}.")
    return True


def _validate_allowed_values(
    df: pd.DataFrame,
    columns: Iterable[str],
    allowed_values: Iterable[str],
    stage_name: str,
) -> bool:
    """
    Ensure that each of the given columns (if present) only contains values from
    the allowed set (case-insensitive, with basic string normalization).
    """
    allowed_norm = {str(v).strip().lower() for v in allowed_values}
    invalid_by_col: Dict[str, List[str]] = {}

    for col in columns:
        if col not in df.columns:
            continue
        series = df[col].dropna().map(lambda x: str(x).strip().lower())
        invalid = sorted(set(series) - allowed_norm)
        if invalid:
            invalid_by_col[col] = invalid

    if invalid_by_col:
        parts = [f"{c}: {vals}" for c, vals in invalid_by_col.items()]
        error_msg = (
            f"Value Validation FAILED for {stage_name}. "
            f"Invalid label values detected -> {', '.join(parts)}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(f"Value Validation PASSED for {stage_name}.")
    return True


# ---------------------------------------------------------------------------
# Stage-specific helpers
# ---------------------------------------------------------------------------

def validate_phase1_processed(df: pd.DataFrame) -> bool:
    """Validate the Phase 1 processed dataframe structure."""
    return validate_schema(df, PHASE1_COLUMNS, "Phase 1 Processed Data")


def validate_phase3_output(df: pd.DataFrame) -> bool:
    """
    Validate the Phase 3 / SLM output:
    - Structural: Phase 1 processed columns + attr_*_winner columns.
    - Value-domain: attr_*_winner values must be one of LABEL_4CLASS.
    """
    validate_schema(df, PHASE3_COLUMNS, "Phase 3 / SLM Output")
    _validate_allowed_values(df, ATTR_WINNER_COLUMNS, LABEL_4CLASS, "Phase 3 / SLM Output")
    return True


def validate_golden_3class(df: pd.DataFrame) -> bool:
    """
    Validate the golden 3-class dataset structure and label domain.

    We allow the canonical ('match', 'both', 'base') plus a small set of
    synonyms that other code already normalizes: 'alt' and 'none'.
    """
    validate_schema(df, GOLDEN_COLUMNS, "Golden 3-class Dataset")
    allowed = {"match", "both", "base", "alt", "none"}
    _validate_allowed_values(df, ["3class_testlabels"], allowed, "Golden 3-class Dataset")
    return True
