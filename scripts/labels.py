import pandas as pd
import numpy as np

# Single source of truth for labels and attribute conflation logic
LABEL_3CLASS = ("alt", "both", "base")
LABEL_4CLASS = ("none", "alt", "base", "both")
ATTR_ATTRS = ("name", "phone", "web", "address", "category")
# Attributes used to decide record-level "none": only name, address, website
ATTRS_FOR_NONE = ("name", "address", "web")

# Vocabulary mapping: 'alt' in the golden data corresponds to 'alt' in model outputs/evals
FOUR_TO_THREE = {
    "alt": "alt",
    "none": "base",
    "base": "base",
    "both": "both"
}

EPSILON = 1  # Threshold for winner delta; if less than EPSILON, consistency wins

def _normalize_attr_winner(val):
    """Normalize attribute-level winners to one of ('base', 'alt', 'both', 'none')."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "none"
    v = str(val).strip().lower()
    if v in ("base", "alt", "both", "none"):
        return v
    # Handle synonyms or short names if necessary
    if v in ("a", "match", "m"): return "alt"
    if v in ("b"): return "base"
    if v in ("t"): return "both"
    return "none"

def recalculate_4class_label(row: pd.Series, suffix: str = "") -> str:
    """
    Compute record-level 4-class label from 5 attr_*_winner columns.
    Uses an epsilon-based consistency scoring approach:
    - If all three of name, address, website are none -> none
    - If n_both > abs(n_alt - n_base) -> both (agreement wins over slight winner)
    - Else majority base vs alt
    """
    counts = {"none": 0, "both": 0, "base": 0, "alt": 0}
    for attr in ATTR_ATTRS:
        col = f"attr_{attr}_winner{suffix}"
        w = _normalize_attr_winner(row.get(col))
        counts[w] += 1

    # "none" only based on name, address, website (all 3 must be none)
    n_none_name_addr_web = sum(
        1 for attr in ATTRS_FOR_NONE
        if _normalize_attr_winner(row.get(f"attr_{attr}_winner{suffix}")) == "none"
    )
    n_none = n_none_name_addr_web
    n_both = counts["both"]
    n_base = counts["base"]
    n_alt = counts["alt"]

    if n_none >= 3:  # all 3 of name, address, web are none
        return "none"
    
    delta = n_alt - n_base
    
    # If more attributes agree ('both') than distinguish (delta), it's 'both'
    # Or if it's an exact tie in winners
    if n_both > abs(delta) or (delta == 0 and n_both > 0):
        return "both"
    
    if delta > 0:
        return "alt"
    if delta < 0:
        return "base"
        
    # Full tie (0 base, 0 alt, but maybe some none/both)
    if n_both > 0:
        return "both"
    return "none"

def fourclass_to_threeclass(label: str) -> str:
    """Map 4-class labels (from data) to canonical 3-class (alt/both/base)."""
    return FOUR_TO_THREE.get(label, "base")

def recalculate_3class_label(row: pd.Series, suffix: str = "") -> str:
    """Shortcut to get 3-class label directly from attribute winners."""
    l4 = recalculate_4class_label(row, suffix)
    return fourclass_to_threeclass(l4)
