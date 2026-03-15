"""
Verification: compare internet-derived truth_*_winner to golden attr_*_winner and 3class_testlabels.
Normalizes both sides before comparing and reports similarity scores.
Run after fetch_truth_google or fetch_truth_scrape.
Usage: python external_validation/verify_truth.py [--input data/ground_truth_google_golden.parquet]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from scripts.labels import fourclass_to_threeclass
    from scripts.normalization import standardize_phone, normalize_website, normalize_address_json
except ImportError:
    from labels import fourclass_to_threeclass  # type: ignore
    from normalization import standardize_phone, normalize_website, normalize_address_json  # type: ignore

# Map: (truth_winner_col, truth_value_col, attr_winner_col) and how to get base/alt value cols
ATTR_MAP = [
    ("truth_phone_winner", "truth_phone_value", "attr_phone_winner", "norm_base_phone", "norm_conflated_phone"),
    ("truth_web_winner", "truth_web_value", "attr_web_winner", "norm_base_website", "norm_conflated_website"),
    ("truth_address_winner", "truth_address_value", "attr_address_winner", "norm_base_addr", "norm_conflated_addr"),
    ("truth_category_winner", "truth_category_value", "attr_category_winner", "_base_category", "_category"),
]


def _norm_label(v) -> str:
    """Normalize winner label for comparison."""
    if v is None or (isinstance(v, float) and str(v) == "nan"):
        return ""
    return str(v).strip().lower()


def _norm_phone(val) -> str:
    if val is None or (isinstance(val, float) and str(val) == "nan"):
        return ""
    return standardize_phone(val) or ""


def _norm_web(val) -> str:
    if val is None or (isinstance(val, float) and str(val) == "nan"):
        return ""
    return normalize_website(val) or ""


def _norm_str(val) -> str:
    if val is None or (isinstance(val, float) and str(val) == "nan"):
        return ""
    return (str(val) or "").strip().lower()


def _golden_value(row: pd.Series, winner_col: str, base_col: str, alt_col: str) -> str:
    """Pick base or alt value based on attr_*_winner; normalize later per attribute."""
    w = _norm_label(row.get(winner_col))
    if w == "base" or w == "none":
        return row.get(base_col) if base_col in row.index else ""
    if w == "alt":
        return row.get(alt_col) if alt_col in row.index else ""
    # both: prefer base
    return row.get(base_col) if base_col in row.index else row.get(alt_col) if alt_col in row.index else ""


def _primary_category_from_row(row: pd.Series, col: str) -> str:
    """Get primary category string from row (column may be _base_category/_category or JSON)."""
    v = row.get(col)
    if v is None or (isinstance(v, float) and str(v) == "nan"):
        return ""
    if isinstance(v, str) and v.strip():
        if v.strip().startswith("{"):
            try:
                o = json.loads(v)
                if isinstance(o, dict) and o.get("primary"):
                    return str(o["primary"]).strip().lower()
            except Exception:
                pass
        return str(v).strip().lower()
    return ""


def _norm_address(val) -> str:
    """Normalize address for comparison (expand abbreviations, lower)."""
    if val is None or (isinstance(val, float) and str(val) == "nan"):
        return ""
    return normalize_address_json(val) or ""

def _normalize_value(attr_key: str, raw: str) -> str:
    """Normalize a single value for comparison (phone, web, address, category)."""
    if attr_key == "phone":
        return _norm_phone(raw)
    if attr_key == "web":
        return _norm_web(raw)
    if attr_key == "address":
        return _norm_address(raw)
    # category: lower + strip
    return _norm_str(raw)


def _similarity(attr_key: str, norm_truth: str, norm_golden: str) -> float:
    """Similarity score 0-100. Phone/web: exact match; address/category/name: fuzzy."""
    if not norm_truth and not norm_golden:
        return 100.0
    if not norm_truth or not norm_golden:
        return 0.0
    if attr_key in ("phone", "web"):
        return 100.0 if norm_truth == norm_golden else 0.0
    if attr_key == "address":
        return float(fuzz.ratio(norm_truth, norm_golden))
    if attr_key in ("category", "name"):
        return float(fuzz.token_sort_ratio(norm_truth, norm_golden))
    return 0.0


def _primary_name_from_val(v) -> str:
    """Extract primary name from raw value (string or JSON)."""
    if v is None or (isinstance(v, float) and str(v) == "nan"):
        return ""
    if isinstance(v, str) and v.strip():
        if v.strip().startswith("{"):
            try:
                o = json.loads(v)
                return str(o.get("primary", "")).strip().lower() if isinstance(o, dict) else ""
            except Exception:
                pass
        return str(v).strip().lower()
    return ""


def _primary_name_from_row(row: pd.Series, col: str) -> str:
    """Get primary name from row (column may be _base_name/_name or names JSON)."""
    return _primary_name_from_val(row.get(col))


def _derive_truth_name_winner(row: pd.Series, truth_winner_cols: list) -> str:
    """Derive implied name winner from majority of other truth_*_winner (base vs alt)."""
    base_count = sum(1 for c in truth_winner_cols if _norm_label(row.get(c)) == "base")
    alt_count = sum(1 for c in truth_winner_cols if _norm_label(row.get(c)) == "alt")
    both_count = sum(1 for c in truth_winner_cols if _norm_label(row.get(c)) == "both")
    if alt_count > base_count:
        return "alt"
    if base_count > alt_count:
        return "base"
    if both_count > 0:
        return "both"
    return "base"


# Attribute key for normalization/similarity
ATTR_KEYS = ["phone", "web", "address", "category"]

# Normalized value columns written by fetch_truth_google / rule_based_logic / fetch_truth_scrape; use when present for compare-normalized-with-normalized
TRUTH_VALUE_NORM_COLUMNS = {
    "phone": "truth_phone_value_norm",
    "web": "truth_web_value_norm",
    "address": "truth_address_value_norm",
    "category": "truth_category_value_norm",
}


def _norm_truth_value(row: pd.Series, attr_key: str, truth_value_col: str) -> str:
    """Truth value for comparison: use truth_*_value_norm if present, else normalize truth_*_value."""
    norm_col = TRUTH_VALUE_NORM_COLUMNS.get(attr_key)
    if norm_col and norm_col in row.index:
        v = row.get(norm_col)
        if v is not None and str(v).strip():
            return str(v).strip()
    return _normalize_value(attr_key, row.get(truth_value_col))


def verify(input_path: Path) -> None:
    df = pd.read_parquet(input_path)
    needed = []
    for t in ATTR_MAP:
        needed.extend([t[0], t[1], t[2]])
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(f"Missing columns: {missing}. Run a fetch_truth_* script first.", file=sys.stderr)
        return

    total = len(df)
    print(f"Verification: {input_path.name} ({total} rows)\n")

    # Ensure we have base/alt value columns for category (fallback from categories JSON)
    if "_base_category" not in df.columns and "base_categories" in df.columns:
        def _prim_cat(val):
            v = val
            if isinstance(v, str) and v.strip().startswith("{"):
                try:
                    o = json.loads(v)
                    return str(o.get("primary", "")).strip().lower() if isinstance(o, dict) else ""
                except Exception:
                    pass
            return _norm_str(v)
        df["_base_category"] = df["base_categories"].apply(_prim_cat)
    if "_category" not in df.columns and "categories" in df.columns:
        def _prim_cat_alt(val):
            v = val
            if isinstance(v, str) and v.strip().startswith("{"):
                try:
                    o = json.loads(v)
                    return str(o.get("primary", "")).strip().lower() if isinstance(o, dict) else ""
                except Exception:
                    pass
            return _norm_str(v)
        df["_category"] = df["categories"].apply(_prim_cat_alt)

    # Ensure primary name columns for name similarity
    if "_base_name" not in df.columns and "base_names" in df.columns:
        df["_base_name"] = df["base_names"].apply(_primary_name_from_val)
    if "_name" not in df.columns and "names" in df.columns:
        df["_name"] = df["names"].apply(_primary_name_from_val)

    # Per-attribute: normalized value comparison and similarity scores (compare normalized with normalized)
    print("--- Normalized value comparison (truth_*_value_norm vs golden normalized) ---")
    sim_cols = []
    for attr_key, (truth_winner_col, truth_value_col, golden_winner_col, base_col, alt_col) in zip(ATTR_KEYS, ATTR_MAP):
        sim_col = f"_sim_{attr_key}"
        sim_cols.append(sim_col)
        df[sim_col] = 0.0
        for idx, row in df.iterrows():
            norm_truth = _norm_truth_value(row, attr_key, truth_value_col)
            golden_raw = _golden_value(row, golden_winner_col, base_col, alt_col)
            if attr_key == "category":
                if not golden_raw and base_col in row.index:
                    golden_raw = _primary_category_from_row(row, base_col)
                if not golden_raw and alt_col in row.index:
                    golden_raw = _primary_category_from_row(row, alt_col)
            norm_golden = _normalize_value(attr_key, golden_raw)
            df.at[idx, sim_col] = _similarity(attr_key, norm_truth, norm_golden)
        mean_sim = df[sim_col].mean()
        print(f"  {attr_key:10} mean similarity (0-100): {mean_sim:.2f}")

    # Name similarity: derive truth name from majority of truth_*_winner, compare to golden attr_name_winner
    truth_winner_cols = [t[0] for t in ATTR_MAP]
    if "_base_name" in df.columns and "_name" in df.columns and "attr_name_winner" in df.columns:
        df["_sim_name"] = 0.0
        for idx, row in df.iterrows():
            truth_winner = _derive_truth_name_winner(row, truth_winner_cols)
            truth_name = _norm_str(row.get("_base_name") if truth_winner == "base" else row.get("_name"))
            golden_winner = _norm_label(row.get("attr_name_winner"))
            golden_name = _norm_str(_golden_value(row, "attr_name_winner", "_base_name", "_name"))
            if not golden_name and golden_winner == "base":
                golden_name = _primary_name_from_row(row, "_base_name")
            if not golden_name and golden_winner == "alt":
                golden_name = _primary_name_from_row(row, "_name")
            df.at[idx, "_sim_name"] = _similarity("name", truth_name, golden_name)
        sim_cols.append("_sim_name")
        print(f"  {'name':10} mean similarity (0-100): {df['_sim_name'].mean():.2f}")

    # Golden label value matches Pipeline A value (per attribute and overall)
    print("\n--- Golden label value matches Pipeline A value ---")
    print("  (Does the value golden selects for each attribute equal Pipeline A's truth_*_value?)")
    match_cols = []
    for attr_key, (_, truth_value_col, golden_winner_col, base_col, alt_col) in zip(ATTR_KEYS, ATTR_MAP):
        sim_col = f"_sim_{attr_key}"
        match_col = f"_golden_match_{attr_key}"
        # Match = normalized equality (exact); for fuzzy fields we use similarity == 100
        df[match_col] = df[sim_col] >= 100.0
        match_cols.append(match_col)
        n_match = df[match_col].sum()
        pct = 100.0 * n_match / total
        print(f"  {attr_key:10} golden value == Pipeline A value: {n_match:3} / {total} ({pct:5.1f}%)")
    if "_sim_name" in df.columns:
        df["_golden_match_name"] = df["_sim_name"] >= 100.0
        match_cols.append("_golden_match_name")
        n_match = df["_golden_match_name"].sum()
        pct = 100.0 * n_match / total
        print(f"  {'name':10} golden value == Pipeline A value: {n_match:3} / {total} ({pct:5.1f}%)")
    # Mean per-attribute match rate (over the 4 or 5 attributes)
    mean_match_rate = df[match_cols].mean(axis=1).mean() * 100.0
    print(f"  Mean per-attribute match rate: {mean_match_rate:.1f}%")
    # Overall: rows where all attributes match
    df["_golden_match_all"] = df[match_cols].all(axis=1)
    n_all = df["_golden_match_all"].sum()
    pct_all = 100.0 * n_all / total
    print(f"  All attributes match (golden == Pipeline A): {n_all:3} / {total} ({pct_all:5.1f}%)")

    # Pipeline A value vs base / alt / neither (normalized value comparison)
    print("\n--- Pipeline A value: same as base, same as alt, same as both, or neither ---")
    for attr_key, (_, truth_value_col, _, base_col, alt_col) in zip(ATTR_KEYS, ATTR_MAP):
        n_base = n_alt = n_both = n_neither = 0
        for idx, row in df.iterrows():
            norm_truth = _norm_truth_value(row, attr_key, truth_value_col)
            base_raw = row.get(base_col) if base_col in row.index else ""
            alt_raw = row.get(alt_col) if alt_col in row.index else ""
            if attr_key == "category":
                if not base_raw or (isinstance(base_raw, str) and base_raw.strip().startswith("{")):
                    base_raw = _primary_category_from_row(row, base_col) or base_raw
                if not alt_raw or (isinstance(alt_raw, str) and alt_raw.strip().startswith("{")):
                    alt_raw = _primary_category_from_row(row, alt_col) or alt_raw
            norm_base = _normalize_value(attr_key, base_raw)
            norm_alt = _normalize_value(attr_key, alt_raw)
            match_base = norm_truth == norm_base
            match_alt = norm_truth == norm_alt
            if match_base and match_alt:
                n_both += 1
            elif match_base:
                n_base += 1
            elif match_alt:
                n_alt += 1
            else:
                n_neither += 1
        pct = lambda n: 100.0 * n / total
        print(f"  {attr_key:10} base_only={n_base:3} ({pct(n_base):5.1f}%)  alt_only={n_alt:3} ({pct(n_alt):5.1f}%)  both={n_both:3} ({pct(n_both):5.1f}%)  neither={n_neither:3} ({pct(n_neither):5.1f}%)")

    # Binary labels (base=0, alt=1): rule-based vs golden per attribute and record-level
    print("\n--- Binary labels (base=0, alt=1): rule-based vs golden ---")
    def winner_to_binary(w: str) -> int:
        """Map winner to binary: alt=1, base/both/none/real=0."""
        w = _norm_label(w)
        return 1 if w == "alt" else 0
    # Per-attribute binary agreement
    for truth_winner_col, _, golden_winner_col, _, _ in ATTR_MAP:
        agree = 0
        for _, row in df.iterrows():
            rb = winner_to_binary(row.get(truth_winner_col))
            gb = winner_to_binary(row.get(golden_winner_col))
            if rb == gb:
                agree += 1
        pct = 100.0 * agree / total
        print(f"  {truth_winner_col.replace('truth_', '').replace('_winner', ''):10} binary agree: {agree:3} / {total} ({pct:5.1f}%)")
    # Record-level binary: derive rule-based 2class from truth_*_winner (majority base vs alt)
    truth_winner_cols = [t[0] for t in ATTR_MAP]
    def row_to_2class_rule(row):
        n_base = sum(1 for c in truth_winner_cols if _norm_label(row.get(c)) == "base")
        n_alt = sum(1 for c in truth_winner_cols if _norm_label(row.get(c)) == "alt")
        if n_alt > n_base:
            return "alt"
        return "base"
    df["_rule_2class"] = df.apply(row_to_2class_rule, axis=1)
    golden_2col = "2class_testlabels"
    if golden_2col in df.columns:
        g2 = df[golden_2col].fillna("").astype(str).str.strip().str.lower()
        agree_2 = (df["_rule_2class"] == g2).sum()
        pct_2 = 100.0 * agree_2 / total
        print(f"  Record-level (rule-based 2class vs {golden_2col}): agree {agree_2:3} / {total} ({pct_2:5.1f}%)")
        # Confusion: golden base/alt vs rule base/alt
        tn = ((g2 == "base") & (df["_rule_2class"] == "base")).sum()
        fp = ((g2 == "base") & (df["_rule_2class"] == "alt")).sum()
        fn = ((g2 == "alt") & (df["_rule_2class"] == "base")).sum()
        tp = ((g2 == "alt") & (df["_rule_2class"] == "alt")).sum()
        print(f"  Confusion (golden \\ rule):  base\\base={tn}, base\\alt={fp}, alt\\base={fn}, alt\\alt={tp}")
    else:
        print(f"  (No {golden_2col} in data; skipping record-level binary.)")

    # Per-attribute winner agreement (truth_*_winner vs attr_*_winner)
    print("\n--- Per-attribute winner agreement (truth_*_winner vs attr_*_winner) ---")
    for truth_winner_col, _, golden_winner_col, _, _ in ATTR_MAP:
        agree = 0
        disagree = 0
        suggested = 0
        for _, row in df.iterrows():
            t = _norm_label(row.get(truth_winner_col))
            g = _norm_label(row.get(golden_winner_col))
            if t == "real":
                suggested += 1
            elif t == g:
                agree += 1
            else:
                disagree += 1
        print(f"  {truth_winner_col}: agree={agree}, disagree={disagree}, suggested_update(real)={suggested}")

    # Record-level: derive 3-class from truth_* (base/alt/both only; treat real as alt)
    truth_winner_cols = [t[0] for t in ATTR_MAP]
    if all(c in df.columns for c in truth_winner_cols):
        def truth_to_4class(row):
            counts = {"base": 0, "alt": 0, "both": 0, "none": 0}
            for truth_col in truth_winner_cols:
                v = _norm_label(row.get(truth_col))
                if v == "real":
                    v = "alt"
                if v in counts:
                    counts[v] += 1
            if counts["alt"] > counts["base"]:
                return "alt"
            if counts["base"] > counts["alt"]:
                return "base"
            if counts["both"] > 0:
                return "both"
            return "none"

        df["_truth_4class"] = df.apply(truth_to_4class, axis=1)
        df["_truth_3class"] = df["_truth_4class"].apply(
            lambda x: fourclass_to_threeclass(x) if x else "base"
        )
        golden_3col = "3class_testlabels"
        if golden_3col in df.columns:
            agree_r = (df["_truth_3class"] == df[golden_3col].astype(str).str.strip().str.lower()).sum()
            print(f"\n--- Record-level (derived 3-class vs {golden_3col}) ---")
            print(f"  Agree: {agree_r} / {total}")

    # Overall similarity summary (includes name when attr_name_winner present)
    if sim_cols:
        df["_sim_overall"] = df[sim_cols].mean(axis=1)
        attrs_label = "phone, web, address, category" + (", name" if "_sim_name" in sim_cols else "")
        print(f"\n--- Overall value similarity (mean over {attrs_label}) ---")
        print(f"  Mean: {df['_sim_overall'].mean():.2f}  Median: {df['_sim_overall'].median():.2f}")

    # Sample disagreements (winner)
    print("\n--- Sample rows where truth_winner != golden winner (first 5) ---")
    shown = 0
    for idx, row in df.iterrows():
        if shown >= 5:
            break
        parts = []
        for truth_winner_col, _, golden_winner_col, _, _ in ATTR_MAP:
            t, g = _norm_label(row.get(truth_winner_col)), _norm_label(row.get(golden_winner_col))
            if t != g:
                parts.append(f"{truth_winner_col}={t} vs {golden_winner_col}={g}")
        if parts:
            print(f"  id={row.get('id', idx)}: {'; '.join(parts)}")
            shown += 1
    if shown == 0:
        print("  (none)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify truth_*_winner vs golden labels")
    parser.add_argument("--input", type=str, default=None, help="Parquet with truth_* and attr_* columns")
    args = parser.parse_args()
    input_path = Path(args.input) if args.input else _REPO_ROOT / "data" / "ground_truth_google_golden.parquet"
    if not input_path.exists():
        input_path = _REPO_ROOT / "data" / "ground_truth_scrape_golden.parquet"
    if not input_path.exists():
        print("No ground truth parquet found. Run fetch_truth_google.py or fetch_truth_scrape.py first.", file=sys.stderr)
        sys.exit(1)
    verify(input_path)


if __name__ == "__main__":
    main()
