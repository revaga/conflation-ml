"""
Shared comparison: real-world data vs base vs alternate (conflated).
Produces per-attribute winner (base | alt | real | both) and canonical value.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Tuple

# Use repo root for scripts
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rapidfuzz import fuzz

try:
    from scripts.normalization import standardize_phone, normalize_website, normalize_address_json_full
except ImportError:
    from normalization import standardize_phone, normalize_website, normalize_address_json_full  # type: ignore

# Attributes we compare (align with labels.ATTR_ATTRS: name, phone, web, address, category)
# We do not assign "truth" for name here; plan focuses on phone, web, category, address.
TRUTH_ATTRS = ("phone", "web", "category", "address")

# Minimum ratio to consider "match" for fuzzy fields (address, category)
FUZZY_MATCH_THRESHOLD = 85


def _norm_phone(val: Any) -> str:
    if val is None or (isinstance(val, float) and str(val) == "nan"):
        return ""
    return standardize_phone(val) or ""


def _norm_web(val: Any) -> str:
    if val is None or (isinstance(val, float) and str(val) == "nan"):
        return ""
    return normalize_website(val) or ""


def _first_url(val: Any) -> str:
    """Extract a single URL string from cell value (list, JSON array string, or plain string). So normalized comparison uses one domain."""
    if val is None or (isinstance(val, float) and str(val) == "nan"):
        return ""
    if isinstance(val, list):
        return (val[0] if val else "") if isinstance(val[0], str) else str(val[0]) if val else ""
    s = str(val).strip()
    if not s:
        return ""
    if s.startswith("["):
        try:
            arr = json.loads(s)
            if isinstance(arr, list) and arr:
                return str(arr[0]).strip()
        except Exception:
            pass
    return s


def _norm_str(val: Any) -> str:
    if val is None or (isinstance(val, float) and str(val) == "nan"):
        return ""
    return (str(val) or "").strip().lower()


def _fuzzy_match(a: str, b: str, threshold: int = FUZZY_MATCH_THRESHOLD) -> bool:
    if not a or not b:
        return a == b
    return fuzz.ratio(a, b) >= threshold or fuzz.token_sort_ratio(a, b) >= threshold


def _compare_phone(real: str, base: str, alt: str) -> Tuple[str, str]:
    r, b, a = _norm_phone(real), _norm_phone(base), _norm_phone(alt)
    if not r:
        if b and a:
            return ("both" if b == a else "base", b if b == a else (b or a))
        return ("base" if b else "alt", b or a)
    if r == b and r == a:
        return ("both", base or alt if isinstance(base, str) else str(base) if base else str(alt))
    if r == b:
        return ("base", base if isinstance(base, str) else str(base))
    if r == a:
        return ("alt", alt if isinstance(alt, str) else str(alt))
    return ("real", real if isinstance(real, str) else str(real))


def _compare_web(real: str, base: str, alt: str) -> Tuple[str, str]:
    """Compare using normalized URLs. Exact match wins; then fuzzy match so same/similar URLs (e.g. redwing vs redwingshoes same path) are base/alt/both, not real."""
    r, b, a = _norm_web(real), _norm_web(base), _norm_web(alt)
    if not r:
        if b and a:
            return ("both" if b == a else "base", b if b == a else (b or a))
        return ("base" if b else "alt", b or a)
    if r == b and r == a:
        return ("both", base or alt if isinstance(base, str) else str(base) if base else str(alt))
    if r == b:
        return ("base", base if isinstance(base, str) else str(base))
    if r == a:
        return ("alt", alt if isinstance(alt, str) else str(alt))
    # Fuzzy match: same or very similar URL (e.g. stores.redwingshoes.com vs stores.redwing.com same path) -> not "real"
    if _fuzzy_match(r, b) and _fuzzy_match(r, a):
        return ("both", base or alt if isinstance(base, str) else str(base) if base else str(alt))
    if _fuzzy_match(r, b):
        return ("base", base if isinstance(base, str) else str(base))
    if _fuzzy_match(r, a):
        return ("alt", alt if isinstance(alt, str) else str(alt))
    return ("real", real if isinstance(real, str) else str(real))


def _compare_category(real: str, base: str, alt: str) -> Tuple[str, str]:
    r, b, a = _norm_str(real), _norm_str(base), _norm_str(alt)
    if not r:
        if b and a:
            return ("both" if b == a else "base", base if b == a else (base or alt))
        return ("base" if b else "alt", base or alt)
    if _fuzzy_match(r, b) and _fuzzy_match(r, a):
        return ("both", base or alt)
    if _fuzzy_match(r, b):
        return ("base", base if isinstance(base, str) else str(base))
    if _fuzzy_match(r, a):
        return ("alt", alt if isinstance(alt, str) else str(alt))
    return ("real", real if isinstance(real, str) else str(real))


def _compare_address(real: str, base: str, alt: str) -> Tuple[str, str]:
    """Compare using full normalized addresses (real=Google, base/alt=record). All three are normalized for comparison."""
    r = normalize_address_json_full(real) or ""
    b = normalize_address_json_full(base) or ""
    a = normalize_address_json_full(alt) or ""
    if not r:
        if b and a:
            return ("both" if b == a else "base", base if b == a else (base or alt))
        return ("base" if b else "alt", base or alt)
    if _fuzzy_match(r, b) and _fuzzy_match(r, a):
        return ("both", base or alt)
    if _fuzzy_match(r, b):
        return ("base", base if isinstance(base, str) else str(base))
    if _fuzzy_match(r, a):
        return ("alt", alt if isinstance(alt, str) else str(alt))
    return ("real", real if isinstance(real, str) else str(real))


def compare_row(
    row: Any,
    real: Dict[str, str],
    allow_fallback: bool = True,
) -> Dict[str, str]:
    """
    Compare real-world dict to base and alternate (conflated) from a dataframe row.
    Row must have: norm_base_phone, norm_conflated_phone, norm_base_website, norm_conflated_website,
    norm_base_addr, norm_conflated_addr; and for category we use base_categories/categories
    primary if available, else norm-style. We expect _base_category and _category if from
    feature engineering; otherwise we compare raw.

    When allow_fallback is False, attributes with no real-world value get winner "no_data"
    and value "" instead of falling back to base/alt/both.
    """
    # Base/alt values from row (conflated = alt)
    def _base_phone():
        return _norm_phone(row.get("norm_base_phone"))

    def _alt_phone():
        return _norm_phone(row.get("norm_conflated_phone"))

    def _base_web():
        raw = _first_url(row.get("base_websites")) or str(row.get("norm_base_website") or "").strip()
        return _norm_web(raw)

    def _alt_web():
        raw = _first_url(row.get("websites")) or str(row.get("norm_conflated_website") or "").strip()
        return _norm_web(raw)

    def _base_addr():
        v = row.get("norm_base_addr") or row.get("base_addresses")
        return _norm_str(v) if v is not None else ""

    def _alt_addr():
        v = row.get("norm_conflated_addr") or row.get("addresses")
        return _norm_str(v) if v is not None else ""

    def _base_addr_full():
        """Full normalized address from base (for address winner comparison)."""
        return normalize_address_json_full(row.get("norm_base_addr") or row.get("base_addresses") or "") or ""

    def _alt_addr_full():
        """Full normalized address from alt (for address winner comparison)."""
        return normalize_address_json_full(row.get("norm_conflated_addr") or row.get("addresses") or "") or ""

    def _base_cat():
        if "_base_category" in row and row.get("_base_category") not in (None, ""):
            return _norm_str(row["_base_category"])
        # Fallback: extract from base_categories JSON
        v = row.get("base_categories")
        if isinstance(v, str) and v:
            try:
                import json
                o = json.loads(v)
                if isinstance(o, dict) and o.get("primary"):
                    return _norm_str(o["primary"])
            except Exception:
                pass
        return _norm_str(v)

    def _alt_cat():
        if "_category" in row and row.get("_category") not in (None, ""):
            return _norm_str(row["_category"])
        v = row.get("categories")
        if isinstance(v, str) and v:
            try:
                import json
                o = json.loads(v)
                if isinstance(o, dict) and o.get("primary"):
                    return _norm_str(o["primary"])
            except Exception:
                pass
        return _norm_str(v)

    # For value output we want the display value (original form), not normalized
    def _base_phone_val():
        v = row.get("norm_base_phone")
        return v if isinstance(v, str) and v else (str(v) if v else "")

    def _alt_phone_val():
        v = row.get("norm_conflated_phone")
        return v if isinstance(v, str) and v else (str(v) if v else "")

    def _base_web_val():
        val = _first_url(row.get("base_websites")) or str(row.get("norm_base_website") or "").strip()
        return val or ""

    def _alt_web_val():
        val = _first_url(row.get("websites")) or str(row.get("norm_conflated_website") or "").strip()
        return val or ""

    def _base_addr_val():
        v = row.get("norm_base_addr")
        return v if isinstance(v, str) and v else (str(v) if v else "")

    def _alt_addr_val():
        v = row.get("norm_conflated_addr")
        return v if isinstance(v, str) and v else (str(v) if v else "")

    def _base_cat_val():
        if "_base_category" in row:
            v = row["_base_category"]
            return v if isinstance(v, str) and v else (str(v) if v else "")
        return str(row.get("base_categories") or "")

    def _alt_cat_val():
        if "_category" in row:
            v = row["_category"]
            return v if isinstance(v, str) and v else (str(v) if v else "")
        return str(row.get("categories") or "")

    out = {}
    def _str(v):
        if v is None or (isinstance(v, float) and str(v) == "nan"):
            return ""
        return str(v).strip()

    # Phone
    w, v = _compare_phone(
        real.get("phone", ""),
        _base_phone_val() if _base_phone() else "",
        _alt_phone_val() if _alt_phone() else "",
    )
    out["truth_phone_winner"] = _str(w)
    out["truth_phone_value"] = _str(v) or _str(real.get("phone", ""))

    # Web
    w, v = _compare_web(
        real.get("web", ""),
        _base_web_val() if _base_web() else "",
        _alt_web_val() if _alt_web() else "",
    )
    out["truth_web_winner"] = _str(w)
    out["truth_web_value"] = _str(v) or _str(real.get("web", ""))

    # Category (compare using normalized; value from base/alt/real)
    w, v = _compare_category(
        real.get("category", ""),
        _base_cat_val() if _base_cat() else "",
        _alt_cat_val() if _alt_cat() else "",
    )
    out["truth_category_winner"] = _str(w)
    out["truth_category_value"] = _str(v) or _str(real.get("category", ""))

    # Address: use full normalized base/alt and normalize Google address for comparison
    w, v = _compare_address(
        real.get("address", ""),
        _base_addr_full() if _base_addr() else "",
        _alt_addr_full() if _alt_addr() else "",
    )
    out["truth_address_winner"] = _str(w)
    out["truth_address_value"] = _str(v) or _str(real.get("address", ""))

    if not allow_fallback:
        attrs = [
            ("phone", "truth_phone_winner", "truth_phone_value"),
            ("web", "truth_web_winner", "truth_web_value"),
            ("category", "truth_category_winner", "truth_category_value"),
            ("address", "truth_address_winner", "truth_address_value"),
        ]
        for key, winner_col, value_col in attrs:
            raw = real.get(key, "")
            if raw is None or (isinstance(raw, str) and not raw.strip()):
                out[winner_col] = "no_data"
                out[value_col] = ""

    return out


def truth_columns() -> Tuple[str, ...]:
    """Column names added by compare_row (for schema / dataframe)."""
    return (
        "truth_phone_winner",
        "truth_phone_value",
        "truth_web_winner",
        "truth_web_value",
        "truth_category_winner",
        "truth_category_value",
        "truth_address_winner",
        "truth_address_value",
    )
