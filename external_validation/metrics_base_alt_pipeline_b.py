"""
Metrics: compare base, alt, and Pipeline B (scrape) to golden labels.
Reports per-attribute and record-level agreement, and Pipeline B winner distribution.
Usage: python external_validation/metrics_base_alt_pipeline_b.py [--input data/ground_truth_scrape_golden.parquet]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from external_validation.verify_truth import (
    ATTR_MAP,
    _norm_label,
)


def _row_to_2class(row: pd.Series, winner_cols: list, get_winner) -> str:
    """Derive record-level 2-class (base/alt) from per-attribute winners."""
    n_base = sum(1 for c in winner_cols if _norm_label(get_winner(row, c)) == "base")
    n_alt = sum(1 for c in winner_cols if _norm_label(get_winner(row, c)) == "alt")
    return "alt" if n_alt > n_base else "base"


def run_metrics(input_path: Path) -> None:
    df = pd.read_parquet(input_path)
    truth_winner_cols = [t[0] for t in ATTR_MAP]
    golden_winner_cols = [t[2] for t in ATTR_MAP]
    attr_names = [t[0].replace("truth_", "").replace("_winner", "") for t in ATTR_MAP]
    missing = [c for c in truth_winner_cols + golden_winner_cols if c not in df.columns]
    if missing:
        print(f"Missing columns: {missing}. Need Pipeline B output and golden attr_*_winner.", file=sys.stderr)
        return

    total = len(df)
    print(f"Metrics: base vs alt vs Pipeline B (input: {input_path.name}, n={total})\n")

    # --- 1. Pipeline B winner distribution (per attribute) ---
    print("--- Pipeline B: winner distribution (base / alt / both / real) ---")
    for truth_col, attr_name in zip(truth_winner_cols, attr_names):
        counts = {"base": 0, "alt": 0, "both": 0, "real": 0}
        for _, row in df.iterrows():
            w = _norm_label(row.get(truth_col))
            counts[w] = counts.get(w, 0) + 1
        pct = lambda n: 100.0 * n / total
        print(f"  {attr_name:10} base={counts['base']:3} ({pct(counts['base']):5.1f}%)  "
              f"alt={counts['alt']:3} ({pct(counts['alt']):5.1f}%)  "
              f"both={counts['both']:3} ({pct(counts['both']):5.1f}%)  "
              f"real={counts.get('real', 0):3} ({pct(counts.get('real', 0)):5.1f}%)")

    # --- 2. Per-attribute agreement with golden: base, alt, Pipeline B ---
    print("\n--- Per-attribute agreement with golden (attr_*_winner) ---")
    print("  Strategy: fraction of rows where strategy winner == golden winner (base/alt/both count as match when same).\n")
    for (truth_col, _, golden_col, _, _), attr_name in zip(ATTR_MAP, attr_names):
        agree_base = agree_alt = agree_pipeline_b = 0
        for _, row in df.iterrows():
            g = _norm_label(row.get(golden_col))
            t = _norm_label(row.get(truth_col))
            # Always-base: agree if golden is base or both (we treat both as base for "base" strategy? No - base strategy means we pick base; agree iff golden==base)
            if g == "base":
                agree_base += 1
            if g == "alt":
                agree_alt += 1
            # Pipeline B agrees if truth_winner == golden_winner (treat real as distinct, so no match unless golden also "real" which we don't have; real vs golden is disagree)
            if t == g:
                agree_pipeline_b += 1
        pct = lambda n: 100.0 * n / total
        print(f"  {attr_name:10}  base_agree={agree_base:3} ({pct(agree_base):5.1f}%)  "
              f"alt_agree={agree_alt:3} ({pct(agree_alt):5.1f}%)  "
              f"Pipeline_B_agree={agree_pipeline_b:3} ({pct(agree_pipeline_b):5.1f}%)")

    # --- 3. Record-level 2-class: base vs alt vs Pipeline B vs golden ---
    golden_2col = "2class_testlabels"
    if golden_2col not in df.columns:
        print(f"\n  (No {golden_2col} in data; skipping record-level 2-class comparison.)")
    else:
        g2 = df[golden_2col].fillna("").astype(str).str.strip().str.lower()
        # Always base
        base_2class = pd.Series(["base"] * total, index=df.index)
        # Always alt
        alt_2class = pd.Series(["alt"] * total, index=df.index)
        # Pipeline B 2-class from truth_*_winner
        pipeline_b_2class = df.apply(
            lambda row: _row_to_2class(row, truth_winner_cols, lambda r, c: r.get(c)),
            axis=1
        )
        agree_base_r = (base_2class == g2).sum()
        agree_alt_r = (alt_2class == g2).sum()
        agree_pb_r = (pipeline_b_2class == g2).sum()
        print("\n--- Record-level 2-class (vs 2class_testlabels) ---")
        print(f"  Always base:  agree {agree_base_r:3} / {total} ({100.0 * agree_base_r / total:.1f}%)")
        print(f"  Always alt:   agree {agree_alt_r:3} / {total} ({100.0 * agree_alt_r / total:.1f}%)")
        print(f"  Pipeline B:   agree {agree_pb_r:3} / {total} ({100.0 * agree_pb_r / total:.1f}%)")
        # Confusion for Pipeline B vs golden
        tn = ((g2 == "base") & (pipeline_b_2class == "base")).sum()
        fp = ((g2 == "base") & (pipeline_b_2class == "alt")).sum()
        fn = ((g2 == "alt") & (pipeline_b_2class == "base")).sum()
        tp = ((g2 == "alt") & (pipeline_b_2class == "alt")).sum()
        print(f"  Pipeline B vs golden confusion: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # --- 4. Summary table: strategy comparison ---
    print("\n--- Summary: strategy vs golden (per-attribute mean agreement) ---")
    base_agrees = []
    alt_agrees = []
    pb_agrees = []
    for (truth_col, _, golden_col, _, _) in ATTR_MAP:
        g_norm = df[golden_col].apply(_norm_label)
        t_norm = df[truth_col].apply(_norm_label)
        base_agrees.append((g_norm == "base").sum() / total * 100)
        alt_agrees.append((g_norm == "alt").sum() / total * 100)
        pb_agrees.append((t_norm == g_norm).sum() / total * 100)
    print(f"  Always base (mean over attributes): {sum(base_agrees)/len(base_agrees):.1f}%")
    print(f"  Always alt  (mean over attributes):  {sum(alt_agrees)/len(alt_agrees):.1f}%")
    print(f"  Pipeline B  (mean over attributes): {sum(pb_agrees)/len(pb_agrees):.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Base vs alt vs Pipeline B metrics vs golden")
    parser.add_argument("--input", type=str, default=None, help="Pipeline B parquet (e.g. ground_truth_scrape_golden.parquet)")
    args = parser.parse_args()
    input_path = Path(args.input) if args.input else _REPO_ROOT / "data" / "ground_truth_scrape_golden.parquet"
    if not input_path.exists():
        print(f"File not found: {input_path}. Run fetch_truth_scrape.py first.", file=sys.stderr)
        sys.exit(1)
    run_metrics(input_path)


if __name__ == "__main__":
    main()
