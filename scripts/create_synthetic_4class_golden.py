"""
Create synthetic_4class_golden.parquet from golden_dataset_200.parquet:
- Keep all 200 original rows (with synthetic modifications).
- 70 random rows: set to 4-class "none" (set all attr_*_winner to "none").
- Half of "both" rows: drop/corrupt alt attributes so row becomes "base".
- Other half of "both" rows: drop/corrupt base attributes so row becomes "alt".
- Append the original unmodified "both" rows so they appear in the dataset as well.
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from labels import recalculate_4class_label

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GOLDEN_PATH = DATA_DIR / "golden_dataset_200.parquet"
OUT_PATH = DATA_DIR / "synthetic_4class_golden.parquet"

ATTR_WINNER_COLS = [
    "attr_name_winner",
    "attr_phone_winner",
    "attr_web_winner",
    "attr_address_winner",
    "attr_category_winner",
]

# Attribute -> (conflated/alt columns, base columns) for drop/corrupt
ATTR_DATA_COLS = {
    "name": (["names"], ["base_names"]),
    "phone": (["phones", "norm_conflated_phone"], ["base_phones", "norm_base_phone"]),
    "web": (["websites", "norm_conflated_website"], ["base_websites", "norm_base_website"]),
    "address": (["addresses", "norm_conflated_addr"], ["base_addresses", "norm_base_addr"]),
    "category": (["categories"], ["base_categories"]),
}


def ensure_4class(df: pd.DataFrame) -> pd.DataFrame:
    if "4class_testlabels" not in df.columns:
        df["4class_testlabels"] = df.apply(recalculate_4class_label, axis=1)
    return df


def set_row_to_none(df: pd.DataFrame, idx: pd.Index) -> None:
    """Set 50 random rows to 4-class 'none' by setting all attr_*_winner to 'none'."""
    for col in ATTR_WINNER_COLS:
        if col in df.columns:
            df.loc[idx, col] = "none"


def corrupt_alt_to_favor_base(df: pd.DataFrame, idx: pd.Index, attrs_to_set: list) -> None:
    """Drop/corrupt alt (conflated) attributes so base wins: set conflated = base."""
    for attr in attrs_to_set:
        conflated_cols, base_cols = ATTR_DATA_COLS[attr]
        for c_col, b_col in zip(conflated_cols, base_cols):
            if c_col in df.columns and b_col in df.columns:
                df.loc[idx, c_col] = df.loc[idx, b_col].values
        col = f"attr_{attr}_winner"
        if col in df.columns:
            df.loc[idx, col] = "base"


def corrupt_base_to_favor_alt(df: pd.DataFrame, idx: pd.Index, attrs_to_set: list) -> None:
    """Drop/corrupt base attributes so alt wins: set base = conflated."""
    for attr in attrs_to_set:
        conflated_cols, base_cols = ATTR_DATA_COLS[attr]
        for c_col, b_col in zip(conflated_cols, base_cols):
            if c_col in df.columns and b_col in df.columns:
                df.loc[idx, b_col] = df.loc[idx, c_col].values
        col = f"attr_{attr}_winner"
        if col in df.columns:
            df.loc[idx, col] = "alt"


def main():
    if not GOLDEN_PATH.exists():
        print(f"File not found: {GOLDEN_PATH}", file=sys.stderr)
        sys.exit(1)

    rng = np.random.default_rng(42)
    df = pd.read_parquet(GOLDEN_PATH)

    missing = [c for c in ATTR_WINNER_COLS if c not in df.columns]
    if missing:
        print(f"Missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    df = ensure_4class(df)

    # Save original "both" rows to append later (unchanged)
    both_mask = df["4class_testlabels"] == "both"
    both_idx = df.index[both_mask].tolist()
    original_both = df.loc[both_idx].copy()

    # 1) Both rows: split into two halves and convert to base / alt
    rng.shuffle(both_idx)
    half = len(both_idx) // 2
    both_to_base_idx = both_idx[:half]
    both_to_alt_idx = both_idx[half:]

    attrs = list(ATTR_DATA_COLS.keys())
    for i in both_to_base_idx:
        pick = rng.choice(attrs, size=3, replace=False).tolist()
        corrupt_alt_to_favor_base(df, [i], pick)
    for i in both_to_alt_idx:
        pick = rng.choice(attrs, size=3, replace=False).tolist()
        corrupt_base_to_favor_alt(df, [i], pick)

    # 2) 70 random rows -> none (after both->base/alt so we get exactly 70 none)
    n_none = 70
    all_idx = df.index.tolist()
    none_idx = rng.choice(all_idx, size=min(n_none, len(all_idx)), replace=False)
    set_row_to_none(df, none_idx)

    # Recompute 4class after all edits
    df["4class_testlabels"] = df.apply(recalculate_4class_label, axis=1)

    # Append original "both" rows back (still labeled "both")
    df = pd.concat([df, original_both], ignore_index=True)

    df.to_parquet(OUT_PATH, index=False)

    counts = df["4class_testlabels"].value_counts()
    print(f"Wrote {OUT_PATH} ({len(df)} rows)")
    print("4class_testlabels distribution:")
    for label in ("none", "alt", "base", "both"):
        print(f"  {label}: {counts.get(label, 0)}")


if __name__ == "__main__":
    main()
