import pytest
import pandas as pd
from scripts.schema import (
    validate_schema,
    validate_phase1_processed,
    validate_phase3_output,
    validate_golden_3class,
    PHASE1_COLUMNS,
    GOLDEN_COLUMNS,
    ATTR_WINNER_COLUMNS,
    PHASE3_COLUMNS,
)


def test_valid_schema_generic():
    # Generic schema validation over PHASE1_COLUMNS
    df = pd.DataFrame(columns=PHASE1_COLUMNS)
    assert validate_schema(df, PHASE1_COLUMNS, "test") is True


def test_invalid_schema_generic():
    # Generic schema validation should fail when a column is missing
    df = pd.DataFrame(columns=PHASE1_COLUMNS[:-1])
    with pytest.raises(ValueError):
        validate_schema(df, PHASE1_COLUMNS, "test")


def test_validate_phase1_processed():
    df = pd.DataFrame(columns=PHASE1_COLUMNS)
    assert validate_phase1_processed(df) is True


def test_validate_phase3_output_value_domain():
    # Happy path: all attr winners are in the allowed 4-class vocabulary
    # Plus structural validation columns from PHASE1_COLUMNS
    cols = list(set(PHASE3_COLUMNS))
    data = {c: ["alt"] if "winner" in c else [0.0] for c in cols}
    data["id"] = [1]
    df = pd.DataFrame(data)
    assert validate_phase3_output(df) is True

    # Invalid value should raise
    bad_df = df.copy()
    bad_df[ATTR_WINNER_COLUMNS[0]] = ["weird_label"]
    with pytest.raises(ValueError):
        validate_phase3_output(bad_df)


def test_validate_golden_3class_allows_synonyms():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "3class_testlabels": ["match", "both", "base", "alt", "none"],
        }
    )
    assert validate_golden_3class(df) is True
