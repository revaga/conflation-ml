"""
Phase 2 similarity: score base vs alt (conflated) with legacy metrics and
unified fuzzy metrics (RapidFuzz token_sort_ratio, partial_ratio, token_set_ratio,
Jaro-Winkler). Writes phase2_scored.parquet. Optional --accuracy reports base
accuracy vs golden_label at threshold 0.671.

Run from project root: python scripts/phase2_similarity.py [--accuracy]
"""
import argparse
import re
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz
from parquet_io import read_parquet_safe
from rapidfuzz import distance

# --- Config ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PHASE1_PATH = DATA_DIR / "phase1_processed.parquet"
PHASE2_SCORED_PATH = DATA_DIR / "phase2_scored.parquet"
PHASE3_PATH = DATA_DIR / "phase3_slm_labeled.parquet"
FUZZY_FIELDS = ("name", "phone", "web", "address", "category")
BASE_ACCURACY_THRESHOLD = 0.671


# ---------- Shared field extraction (aligned with SLM / golden_dataset_maker) ----------
def _str(val, empty="(empty)"):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return empty
    return str(val).strip() or empty


def _name_display(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "(empty)"
    if isinstance(val, dict):
        return (val.get("primary") or val.get("raw") or str(val)).strip() or "(empty)"
    return str(val).strip() or "(empty)"


def _first_str(val):
    if isinstance(val, list) and len(val) > 0:
        return str(val[0]).strip() or ""
    return _str(val, "")


def _category_display(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "(empty)"
    if isinstance(val, dict):
        s = (val.get("primary") or val.get("raw") or "").strip()
        return s or "(empty)"
    return str(val).strip() or "(empty)"


def get_field_strings(row):
    """Return dict of (base_str, alt_str) for each attribute."""
    base_name = _name_display(row.get("base_names"))
    alt_name = _name_display(row.get("names"))
    base_phone = _str(_first_str(row.get("norm_base_phone") or row.get("base_phones")), "(empty)")
    alt_phone = _str(_first_str(row.get("norm_conflated_phone") or row.get("phones")), "(empty)")
    base_web = _str(_first_str(row.get("norm_base_website") or row.get("base_websites")), "(empty)")
    alt_web = _str(_first_str(row.get("norm_conflated_website") or row.get("websites")), "(empty)")
    base_addr = _str(row.get("norm_base_addr") or row.get("base_addresses"), "(empty)")
    alt_addr = _str(row.get("norm_conflated_addr") or row.get("addresses"), "(empty)")
    base_cat = _category_display(row.get("base_categories"))
    alt_cat = _category_display(row.get("categories"))
    return {
        "name": (base_name, alt_name),
        "phone": (base_phone, alt_phone),
        "web": (base_web, alt_web),
        "address": (base_addr, alt_addr),
        "category": (base_cat, alt_cat),
    }


def score_pair_fuzzy(s1: str, s2: str):
    """
    Compare two strings with four metrics (all normalized to 0–1):
    token_sort_ratio, partial_ratio, token_set_ratio, Jaro-Winkler.
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    token_sort = fuzz.token_sort_ratio(s1, s2) / 100.0
    partial = fuzz.partial_ratio(s1, s2) / 100.0
    token_set = fuzz.token_set_ratio(s1, s2) / 100.0
    jw = distance.JaroWinkler.similarity(s1, s2)
    return (token_sort + partial + token_set + jw) / 4.0


def compute_row_fuzzy_scores(row):
    """Compute per-field fuzzy scores and total_similarity for one row."""
    field_strs = get_field_strings(row)
    out = {}
    total = 0.0
    for field in FUZZY_FIELDS:
        base_s, alt_s = field_strs[field]
        s = score_pair_fuzzy(base_s, alt_s)
        out[f"fuzzy_score_{field}"] = s
        total += s
    out["total_similarity"] = total / len(FUZZY_FIELDS)
    out["total_similarity_sum"] = total
    return out


def add_fuzzy_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add fuzzy_score_* and total_similarity columns to df. Returns new DataFrame."""
    rows = [compute_row_fuzzy_scores(row) for _, row in df.iterrows()]
    fuzzy_df = pd.DataFrame(rows)
    return pd.concat([df.reset_index(drop=True), fuzzy_df], axis=1)


# ---------- Legacy scoring (name/address components, match_score) ----------
def calculate_jaro_winkler(s1, s2):
    if pd.isna(s1) or pd.isna(s2):
        return 0.0
    return distance.JaroWinkler.similarity(str(s1), str(s2))


def parse_address_custom(addr):
    if not isinstance(addr, str):
        return {"num": "", "street": "", "unit": ""}
    match = re.search(r"^(\d+)\s+(.*)", addr)
    if match:
        num, rest = match.groups()
    else:
        num, rest = "", addr
    unit_match = re.search(r"\b(ste|suite|apt|unit|#)\s*([\w-]+)", rest)
    unit = ""
    if unit_match:
        unit = f"{unit_match.group(1)} {unit_match.group(2)}"
        rest = re.sub(r"\b(ste|suite|apt|unit|#)\s*[\w-]+", "", rest).strip()
    return {"num": num, "street": rest, "unit": unit}


def score_address_components(row):
    addr1 = parse_address_custom(row["norm_conflated_addr"])
    addr2 = parse_address_custom(row["norm_base_addr"])
    num_match = 1.0 if addr1["num"] == addr2["num"] and addr1["num"] else 0.0
    if not addr1["num"] and not addr2["num"]:
        num_match = 0.5
    street_sim = fuzz.token_sort_ratio(addr1["street"], addr2["street"]) / 100.0
    if not addr1["unit"] and not addr2["unit"]:
        unit_match = 1.0
    elif addr1["unit"] == addr2["unit"]:
        unit_match = 1.0
    else:
        unit_match = 0.0
    return (num_match * 0.4) + (street_sim * 0.4) + (unit_match * 0.2)


def run_legacy_scoring(df: pd.DataFrame) -> pd.DataFrame:
    """Add legacy score_* and match_score, pred_label. Expects phase1 columns."""
    # Name
    df = df.copy()
    df["name_tfidf_proxy"] = df.apply(
        lambda x: fuzz.token_set_ratio(str(x["names"]), str(x["base_names"])) / 100.0, axis=1
    )
    df["name_jaro"] = df.apply(lambda x: calculate_jaro_winkler(x["names"], x["base_names"]), axis=1)
    df["name_fuzz"] = df.apply(
        lambda x: fuzz.ratio(str(x["names"]), str(x["base_names"])) / 100.0, axis=1
    )
    df["score_name"] = (
        df["name_tfidf_proxy"] * 0.3 + df["name_jaro"] * 0.3 + df["name_fuzz"] * 0.4
    )
    # Address
    df["score_address"] = df.apply(score_address_components, axis=1)
    # Phone / website (from phase1)
    df["score_phone"] = df["phone_similarity"] / 100.0
    df["score_website"] = df["website_similarity"] / 100.0
    # Match score and pred
    df["match_score"] = (
        df["score_name"] * 0.3
        + df["score_address"] * 0.4
        + df["score_phone"] * 0.2
        + df["score_website"] * 0.1
    )
    df["pred_label"] = (df["match_score"] >= 0.7).astype(int)
    return df


# ---------- Base accuracy (fuzzy threshold vs golden_label) ----------
def report_base_accuracy(
    scored_path: Path,
    golden_path: Path | None = None,
    threshold: float = BASE_ACCURACY_THRESHOLD,
) -> float:
    """
    Load scored parquet (must have total_similarity), merge golden_label from
    golden_path (default phase3_slm_labeled), report accuracy. Returns accuracy in [0, 1].
    """
    golden_path = golden_path or PHASE3_PATH
    df = read_parquet_safe(str(scored_path))
    if "total_similarity" not in df.columns:
        raise ValueError(f"No 'total_similarity' in {scored_path}. Run scoring first.")
    if not golden_path.exists():
        print(f"Golden path not found: {golden_path}. Skipping base accuracy.")
        return float("nan")
    golden = read_parquet_safe(str(golden_path))
    if "golden_label" not in golden.columns:
        print("No 'golden_label' in golden file. Skipping base accuracy.")
        return float("nan")
    merge_cols = ["id"] + (["golden_label"] if "golden_label" not in df.columns else [])
    golden_sub = golden[["id", "golden_label"]].drop_duplicates()
    merged = df.merge(golden_sub, on="id", how="left")
    eval_df = merged.dropna(subset=["golden_label"]).copy()
    eval_df = eval_df[eval_df["golden_label"].isin(("base", "alt"))].copy()
    if len(eval_df) == 0:
        print("No rows with golden_label in (base, alt). Skipping base accuracy.")
        return float("nan")
    eval_df["truth"] = (eval_df["golden_label"] == "alt").astype(int)
    eval_df["pred"] = (eval_df["total_similarity"] >= threshold).astype(int)
    correct = (eval_df["pred"] == eval_df["truth"]).sum()
    n = len(eval_df)
    accuracy = correct / n
    print(f"Threshold: {threshold}")
    print(f"Rows with golden_label in (base, alt): {n}")
    print(f"Correct: {correct}")
    print(f"Base accuracy: {accuracy:.2%} (~{accuracy * 100:.0f}%)")
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Phase 2 similarity scoring and optional base accuracy")
    parser.add_argument(
        "--accuracy",
        action="store_true",
        help="Report base accuracy vs golden_label (phase3) at threshold 0.671",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PHASE1_PATH,
        help="Input parquet (default: data/phase1_processed.parquet)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PHASE2_SCORED_PATH,
        help="Output parquet (default: data/phase2_scored.parquet)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input not found: {args.input}")
        return 1

    print("Loading data...")
    df = read_parquet_safe(str(args.input))

    # Legacy scoring (phase2 original)
    print("Calculating legacy name/address/phone/website scores...")
    df = run_legacy_scoring(df)

    # Fuzzy multi-metric scores (token_sort, partial, token_set, Jaro-Winkler per field)
    print("Calculating fuzzy match scores (token_sort, partial, token_set, Jaro-Winkler)...")
    df = add_fuzzy_scores(df)

    print("Saving results...")
    df.to_parquet(args.output, index=False)
    print(f"Saved {args.output} ({len(df)} rows)")

    # Summary
    score_cols = [c for c in df.columns if c.startswith("fuzzy_score_") or c == "total_similarity"]
    print("\nFuzzy score columns:", score_cols)
    if "total_similarity" in df.columns:
        print(
            "total_similarity: min={:.4f} max={:.4f} mean={:.4f}".format(
                df["total_similarity"].min(),
                df["total_similarity"].max(),
                df["total_similarity"].mean(),
            )
        )
    print("\n--- Scoring Summary ---")
    print(df[["match_score", "pred_label"]].describe())
    print(f"\nPredicted Matches: {df['pred_label'].sum()} / {len(df)}")

    if args.accuracy:
        print("\n--- Base Accuracy (fuzzy threshold) ---")
        report_base_accuracy(args.output, threshold=BASE_ACCURACY_THRESHOLD)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
