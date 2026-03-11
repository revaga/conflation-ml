"""
Phase 4 Evaluation: Golden truth vs SLM predictions.
Uses data/golden_dataset_200.parquet as truth and data/phase3_slm_labeled.parquet as SLM.
Reports binary, 4-class, and 3-class metrics aligned with xgboost_multiclass and xgboostbinary.
No sklearn dependency; metrics implemented manually with pandas/numpy.
"""

import numpy as np
import pandas as pd
import os

from parquet_io import read_parquet_safe

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GOLDEN_PATH = "data/golden_dataset_200.parquet"
SLM_PATH = "data/phase3_slm_labeledkimi.parquet"
ATTR_ATTRS = ("name", "phone", "web", "address", "category")
# 4-class order: none=0, alt=1, base=2, both=3 (align with xgboost_multiclass)
CLASS_ORDER_4 = ("none", "alt", "base", "both")
CLASS_ORDER_3 = ("match", "both", "base")
TIE_BREAK_TO_BOTH = True


def _normalize_attr_winner(val):
    """Return one of 'base', 'alt', 'both', 'none'. Treat missing/invalid as 'none'."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "none"
    v = str(val).strip().lower()
    if v in ("base", "alt", "both", "none"):
        return v
    return "none"


def recalculate_4class_label(row: pd.Series, suffix: str = "") -> str:
    """
    Compute record-level 4-class label from 5 attr_*_winner columns.
    suffix: e.g. '_truth' or '_slm' to pick which set of columns to use.
    """
    counts = {"none": 0, "both": 0, "base": 0, "alt": 0}
    for attr in ATTR_ATTRS:
        col = f"attr_{attr}_winner{suffix}"
        w = _normalize_attr_winner(row.get(col))
        counts[w] = counts.get(w, 0) + 1

    n_none = counts["none"]
    n_both = counts["both"]
    n_base = counts["base"]
    n_alt = counts["alt"]

    if n_none >= 3:
        return "none"
    if n_both >= 3:
        return "both"
    if n_base > n_alt:
        return "base"
    if n_alt > n_base:
        return "alt"
    return "both" if TIE_BREAK_TO_BOTH else "alt"


def fourclass_to_binary(label: str) -> int:
    """alt or both -> 1; base or none -> 0."""
    return 1 if label in ("alt", "both") else 0


def fourclass_to_threeclass(label: str) -> str:
    """none->base, base->base, alt->match, both->both."""
    if label == "alt":
        return "match"
    if label in ("base", "none"):
        return "base"
    return "both"


def normalize_threeclass(val) -> str:
    """Normalize 3-class label to match/both/base."""
    if pd.isna(val) or val is None:
        return "base"
    v = str(val).strip().lower()
    if v in ("match", "alt", "m", "a"):
        return "match"
    if v == "both":
        return "both"
    return "base"


# ---------------------------------------------------------------------------
# Manual metrics (no sklearn)
# ---------------------------------------------------------------------------

def _manual_f1_macro(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 4) -> float:
    """Macro F1 for multi-class (numeric labels 0..n_classes-1)."""
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()
    f1s = []
    for c in range(n_classes):
        tp = ((y_t == c) & (y_p == c)).sum()
        fp = ((y_t != c) & (y_p == c)).sum()
        fn = ((y_t == c) & (y_p != c)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def _classification_report_4class(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Per-class precision/recall/F1 and accuracy for 4-class."""
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()
    lines = [f"{'':>10} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}"]
    for i, c in enumerate(CLASS_ORDER_4):
        tp = ((y_t == i) & (y_p == i)).sum()
        fp = ((y_t != i) & (y_p == i)).sum()
        fn = ((y_t == i) & (y_p != i)).sum()
        support = int((y_t == i).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        lines.append(f"{c:>10} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {support:>10}")
    acc = (y_t == y_p).mean()
    lines.append(f"{'accuracy':>10} {'':>10} {'':>10} {acc:>10.4f} {len(y_t):>10}")
    return "\n".join(lines)


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Confusion matrix (rows=truth, cols=pred)."""
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()
    m = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(len(y_t)):
        ti, pi = int(y_t[i]), int(y_p[i])
        if 0 <= ti < n_classes and 0 <= pi < n_classes:
            m[ti, pi] += 1
    return m


def _label_to_index_4class(labels: pd.Series) -> np.ndarray:
    """Map 4-class string labels to 0..3; unknown -> -1 (drop later)."""
    return labels.map(lambda x: CLASS_ORDER_4.index(x) if x in CLASS_ORDER_4 else -1).values


def _label_to_index_3class(labels: pd.Series) -> np.ndarray:
    """Map 3-class string labels to 0..2; unknown -> -1."""
    return labels.map(lambda x: CLASS_ORDER_3.index(x) if x in CLASS_ORDER_3 else -1).values


def main():
    print("Loading data...")
    try:
        golden_df = read_parquet_safe(GOLDEN_PATH)
        slm_df = read_parquet_safe(SLM_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Expected: {GOLDEN_PATH} (truth) and {SLM_PATH} (SLM predictions).")
        return

    # Merge on id; keep both attr_*_winner sets with suffixes
    required = [f"attr_{a}_winner" for a in ATTR_ATTRS]
    for name, df in [("Golden", golden_df), ("SLM", slm_df)]:
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"Error: {name} missing columns: {missing}")
            return

    merged = pd.merge(
        golden_df[["id"] + required].copy(),
        slm_df[["id"] + required].copy(),
        on="id",
        how="inner",
        suffixes=("_truth", "_slm"),
    )
    print(f"Loaded {len(merged)} records (intersection of golden and SLM by id).")

    if len(merged) == 0:
        print("No overlapping ids; nothing to evaluate.")
        return

    # Derive 4-class labels
    merged["truth_4class"] = merged.apply(lambda r: recalculate_4class_label(r, "_truth"), axis=1)
    merged["slm_4class"] = merged.apply(lambda r: recalculate_4class_label(r, "_slm"), axis=1)

    # Binary: alt+both -> 1, base+none -> 0
    merged["truth_binary"] = merged["truth_4class"].map(fourclass_to_binary)
    merged["slm_binary"] = merged["slm_4class"].map(fourclass_to_binary)

    # 3-class: from xgboost_testlabels if present, else from 4-class
    if "xgboost_testlabels" in golden_df.columns:
        truth_3_map = golden_df.set_index("id")["xgboost_testlabels"].map(normalize_threeclass)
        merged["truth_3class"] = merged["id"].map(truth_3_map)
    else:
        merged["truth_3class"] = merged["truth_4class"].map(fourclass_to_threeclass)
    merged["slm_3class"] = merged["slm_4class"].map(fourclass_to_threeclass)

    # Drop any row with missing labels (should not happen after derivation)
    merged = merged.dropna(subset=["truth_4class", "slm_4class", "truth_binary", "slm_binary"])
    n = len(merged)

    # ---------- Binary metrics ----------
    y_true_bin = merged["truth_binary"].astype(int).values
    y_pred_bin = merged["slm_binary"].astype(int).values
    tp = int(((y_true_bin == 1) & (y_pred_bin == 1)).sum())
    tn = int(((y_true_bin == 0) & (y_pred_bin == 0)).sum())
    fp = int(((y_true_bin == 0) & (y_pred_bin == 1)).sum())
    fn = int(((y_true_bin == 1) & (y_pred_bin == 0)).sum())
    acc_bin = (y_true_bin == y_pred_bin).mean()
    prec_bin = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_bin = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_bin = 2 * prec_bin * rec_bin / (prec_bin + rec_bin) if (prec_bin + rec_bin) > 0 else 0.0

    print("\n" + "=" * 60)
    print("--- Binary (accept vs keep_base) ---")
    print("=" * 60)
    print(f"Accuracy:  {acc_bin:.4%}")
    print(f"Precision: {prec_bin:.4f}  Recall: {rec_bin:.4f}  F1: {f1_bin:.4f}")
    print("Confusion matrix:  TP={}  FP={}  FN={}  TN={}".format(tp, fp, fn, tn))
    print(f"                 Predicted 0   Predicted 1")
    print(f"Truth 0 (base/none)  {tn:<12}   {fp:<12}")
    print(f"Truth 1 (alt/both)   {fn:<12}   {tp:<12}")

    # ---------- 4-class metrics ----------
    y_true_4 = _label_to_index_4class(merged["truth_4class"])
    y_pred_4 = _label_to_index_4class(merged["slm_4class"])
    valid_4 = (y_true_4 >= 0) & (y_pred_4 >= 0)
    y_true_4 = y_true_4[valid_4]
    y_pred_4 = y_pred_4[valid_4]
    acc_4 = (y_true_4 == y_pred_4).mean()
    macro_f1_4 = _manual_f1_macro(y_true_4, y_pred_4, n_classes=4)
    cm_4 = _confusion_matrix(y_true_4, y_pred_4, 4)

    print("\n" + "=" * 60)
    print("--- 4-class (none / alt / base / both) ---")
    print("=" * 60)
    print(f"Accuracy:  {acc_4:.4%}")
    print(f"Macro F1: {macro_f1_4:.4f}")
    print("\nClassification report:")
    print(_classification_report_4class(y_true_4, y_pred_4))
    print("\nConfusion matrix (rows=truth, cols=pred):")
    header = "  " + " ".join(f"{c:>6}" for c in CLASS_ORDER_4)
    print(header)
    for i, c in enumerate(CLASS_ORDER_4):
        print(f"  {c:>6} " + " ".join(f"{cm_4[i, j]:>6}" for j in range(4)))

    # ---------- 3-class metrics ----------
    y_true_3 = _label_to_index_3class(merged["truth_3class"])
    y_pred_3 = _label_to_index_3class(merged["slm_3class"])
    valid_3 = (y_true_3 >= 0) & (y_pred_3 >= 0)
    y_true_3 = y_true_3[valid_3]
    y_pred_3 = y_pred_3[valid_3]
    acc_3 = (y_true_3 == y_pred_3).mean()
    cm_3 = _confusion_matrix(y_true_3, y_pred_3, 3)

    print("\n" + "=" * 60)
    print("--- 3-class (match / both / base) ---")
    print("=" * 60)
    print(f"Accuracy: {acc_3:.4%}")
    print("\nConfusion matrix (rows=truth, cols=pred):")
    header = "  " + " ".join(f"{c:>6}" for c in CLASS_ORDER_3)
    print(header)
    for i, c in enumerate(CLASS_ORDER_3):
        print(f"  {c:>6} " + " ".join(f"{cm_3[i, j]:>6}" for j in range(3)))
    print("\nPer-class metrics:")
    for c in CLASS_ORDER_3:
        idx = CLASS_ORDER_3.index(c)
        tp_c = int(((y_true_3 == idx) & (y_pred_3 == idx)).sum())
        pred_c = int((y_pred_3 == idx).sum())
        true_c = int((y_true_3 == idx).sum())
        prec_c = tp_c / pred_c if pred_c > 0 else 0.0
        rec_c = tp_c / true_c if true_c > 0 else 0.0
        f1_c = 2 * prec_c * rec_c / (prec_c + rec_c) if (prec_c + rec_c) > 0 else 0.0
        print(f"  {c:<8}: precision={prec_c:.4f}  recall={rec_c:.4f}  F1={f1_c:.4f}  (n={true_c})")

    # ---------- Disagreements (4-class truth != SLM 4-class) ----------
    disagreements = merged[merged["truth_4class"] != merged["slm_4class"]].copy()
    print(f"\nDisagreements (truth_4class != slm_4class): {len(disagreements)}")

    if len(disagreements) > 0:
        output_disagreement = "incorrect_predictions.csv"
        desired_cols = [
            "id", "truth_4class", "slm_4class", "truth_3class", "slm_3class",
            "truth_binary", "slm_binary",
        ]
        # Add any extra columns from merged that might be useful (from golden or slm side)
        extra = [c for c in merged.columns if c.startswith("attr_") and "winner" in c]
        desired_cols = desired_cols + [c for c in extra if c in merged.columns]
        available = [c for c in desired_cols if c in merged.columns]
        disagreements[available].to_csv(output_disagreement, index=False)
        print(f"Saved {len(disagreements)} disagreements to {output_disagreement}")

    # ---------- Summary file ----------
    summary_path = "evaluation_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Phase 4 Evaluation: Golden (truth) vs SLM predictions\n")
        f.write(f"Truth: {GOLDEN_PATH}  |  SLM: {SLM_PATH}\n")
        f.write(f"Sample size (merged on id): {n}\n\n")
        f.write("--- Binary ---\n")
        f.write(f"Accuracy: {acc_bin:.4%}\n")
        f.write(f"Precision: {prec_bin:.4f}  Recall: {rec_bin:.4f}  F1: {f1_bin:.4f}\n")
        f.write(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}\n\n")
        f.write("--- 4-class ---\n")
        f.write(f"Accuracy: {acc_4:.4%}\n")
        f.write(f"Macro F1: {macro_f1_4:.4f}\n\n")
        f.write("--- 3-class ---\n")
        f.write(f"Accuracy: {acc_3:.4%}\n")
        f.write(f"Disagreements (4-class): {len(disagreements)}\n")
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
