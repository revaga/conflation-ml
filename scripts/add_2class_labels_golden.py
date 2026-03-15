"""
Create a 2class_testlabels column in golden_dataset_200.parquet.
Label each row as 'base' or 'alt' from attr_*_winner columns;
break ties using address and name (prefer side that wins on both; else address).
"""
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PARQUET_PATH = DATA_DIR / "golden_dataset_200.parquet"

ATTR_WINNER_COLS = [
    "attr_name_winner",
    "attr_phone_winner",
    "attr_web_winner",
    "attr_address_winner",
    "attr_category_winner",
]


def _normalize(w):
    if w is None or (isinstance(w, float) and pd.isna(w)):
        return None
    v = str(w).strip().lower()
    if v in ("base", "alt", "both", "none"):
        return v
    return None


def row_to_2class(row: pd.Series) -> str:
    """
    Decide 'base' or 'alt' from attr_*_winner columns.
    Break ties by preferring the side that wins on address and name; if they disagree, use address.
    """
    n_base = 0
    n_alt = 0
    for col in ATTR_WINNER_COLS:
        w = _normalize(row.get(col))
        if w == "base":
            n_base += 1
        elif w == "alt":
            n_alt += 1
        # 'both' and 'none' (or invalid) don't count toward either

    if n_base > n_alt:
        return "base"
    if n_alt > n_base:
        return "alt"

    # Tie: break by address and name (prefer side where address and name agree)
    addr = _normalize(row.get("attr_address_winner"))
    name = _normalize(row.get("attr_name_winner"))

    if addr == "base" and name == "base":
        return "base"
    if addr == "alt" and name == "alt":
        return "alt"
    # Disagree or both/none: use phone as tie-breaker
    phone = _normalize(row.get("attr_phone_winner"))
    if phone == "base":
        return "base"
    if phone == "alt":
        return "alt"
    if name == "base":
        return "base"
    if name == "alt":
        return "alt"
    if addr == "base":
        return "base"
    if addr == "alt":
        return "alt"
    # All both/none: default to base
    return "base"


def main():
    if not PARQUET_PATH.exists():
        print(f"File not found: {PARQUET_PATH}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(PARQUET_PATH)

    missing = [c for c in ATTR_WINNER_COLS if c not in df.columns]
    if missing:
        print(f"Missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    df["2class_testlabels"] = df.apply(row_to_2class, axis=1)
    df.to_parquet(PARQUET_PATH, index=False)

    counts = df["2class_testlabels"].value_counts()
    print("2class_testlabels added. Distribution:")
    for label, n in counts.items():
        print(f"  {label}: {n}")


if __name__ == "__main__":
    main()
