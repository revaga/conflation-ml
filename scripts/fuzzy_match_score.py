"""
Fuzzy match score: compute similarity between base and alt for each attribute
using the same logic as phase2_similarity (RapidFuzz + Jaro-Winkler). Writes
data/fuzzy_scored.parquet. Use this when you want fuzzy scores on phase3 (or
another) input without running the full phase2 legacy scoring.

Run from project root: python scripts/fuzzy_match_score.py
"""
import sys
from pathlib import Path

# Allow importing phase2_similarity when run as script from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from phase2_similarity import add_fuzzy_scores, PHASE1_PATH, PHASE3_PATH

# --- Config ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "fuzzy_scored.parquet"


def main():
    # Prefer phase3 so output has golden_label for base_accuracy
    path = PHASE3_PATH if PHASE3_PATH.exists() else PHASE1_PATH
    if not path.exists():
        print(f"Neither {PHASE3_PATH} nor {PHASE1_PATH} found.")
        return 1

    print(f"Loading {path}...")
    df = pd.read_parquet(path)
    print("Computing fuzzy match scores...")
    df = add_fuzzy_scores(df)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {OUTPUT_PATH} ({len(df)} rows)")

    if "total_similarity" in df.columns:
        print(
            "total_similarity: min={:.4f} max={:.4f} mean={:.4f}".format(
                df["total_similarity"].min(),
                df["total_similarity"].max(),
                df["total_similarity"].mean(),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
