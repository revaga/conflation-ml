"""Report winner distribution (no_data, base, alt, both, real) for no-fallback pipeline outputs."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
WINNER_COLS = [
    "truth_phone_winner",
    "truth_web_winner",
    "truth_address_winner",
    "truth_category_winner",
]


def report(path: Path, label: str) -> None:
    df = pd.read_parquet(path)
    total = len(df)
    print(f"\n{'='*60}")
    print(f"{label} (n={total})")
    print("=" * 60)
    for col in WINNER_COLS:
        attr = col.replace("truth_", "").replace("_winner", "")
        vc = df[col].fillna("").astype(str).str.strip().str.lower()
        counts = vc.value_counts()
        print(f"\n  {attr:10}")
        for winner in ["no_data", "base", "alt", "both", "real"]:
            n = int(counts.get(winner, 0))
            pct = 100.0 * n / total
            print(f"    {winner:10} {n:4} ({pct:5.1f}%)")
    # Rows with at least one no_data
    any_no = df[WINNER_COLS].apply(lambda r: (r.fillna("").astype(str).str.strip().str.lower() == "no_data").any(), axis=1)
    all_no = df[WINNER_COLS].apply(lambda r: (r.fillna("").astype(str).str.strip().str.lower() == "no_data").all(), axis=1)
    print(f"\n  Rows with any no_data:  {any_no.sum()} ({100.0 * any_no.sum() / total:.1f}%)")
    print(f"  Rows with all no_data:  {all_no.sum()} ({100.0 * all_no.sum() / total:.1f}%)")


def main() -> None:
    data_dir = _REPO_ROOT / "data"
    paths = [
        (data_dir / "ground_truth_google_no_fallback.parquet", "Pipeline A (Google) - no fallback"),
        (data_dir / "ground_truth_scrape_no_fallback.parquet", "Pipeline B (Scrape) - no fallback"),
    ]
    for path, label in paths:
        if path.exists():
            report(path, label)
        else:
            print(f"\nSkip (not found): {path}", file=sys.stderr)
    print()


if __name__ == "__main__":
    main()
