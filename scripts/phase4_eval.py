import numpy as np
import pandas as pd
import os
import logging
from scripts.parquet_io import read_parquet_safe
from scripts.labels import (
    recalculate_4class_label,
    fourclass_to_threeclass,
    ATTR_ATTRS,
    LABEL_4CLASS,
    LABEL_3CLASS,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GOLDEN_PATH = "data/golden_dataset_200.parquet"
SLM_PATH = "data/phase3_slm_labeled.parquet"  # Fixed path
# 4-class order: none=0, alt=1, base=2, both=3 (must align with scripts.labels)
CLASS_ORDER_4 = LABEL_4CLASS
CLASS_ORDER_3 = LABEL_3CLASS

def fourclass_to_binary(label: str) -> int:
    return 1 if label in ("alt", "both") else 0

def normalize_threeclass(val) -> str:
    if pd.isna(val) or val is None:
        return "base"
    v = str(val).strip().lower()
    if v in ("match", "alt", "m", "a"):
        return "alt"
    if v == "both":
        return "both"
    return "base"

def _manual_f1_macro(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 4) -> float:
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
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()
    m = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(len(y_t)):
        ti, pi = int(y_t[i]), int(y_p[i])
        if 0 <= ti < n_classes and 0 <= pi < n_classes:
            m[ti, pi] += 1
    return m

def _label_to_index_4class(labels: pd.Series) -> np.ndarray:
    return labels.map(lambda x: CLASS_ORDER_4.index(x) if x in CLASS_ORDER_4 else -1).values

def _label_to_index_3class(labels: pd.Series) -> np.ndarray:
    return labels.map(lambda x: CLASS_ORDER_3.index(x) if x in CLASS_ORDER_3 else -1).values

def main():
    logger.info("Loading data...")
    try:
        golden_df = read_parquet_safe(GOLDEN_PATH)
        slm_df = read_parquet_safe(SLM_PATH)
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.error(f"Expected: {GOLDEN_PATH} (truth) and {SLM_PATH} (SLM predictions).")
        return

    required = [f"attr_{a}_winner" for a in ATTR_ATTRS]
    for name, df in [("Golden", golden_df), ("SLM", slm_df)]:
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.error(f"Error: {name} missing columns: {missing}")
            return

    merged = pd.merge(
        golden_df[["id"] + required].copy(),
        slm_df[["id"] + required].copy(),
        on="id",
        how="inner",
        suffixes=("_truth", "_slm"),
    )
    logger.info(f"Loaded {len(merged)} records (intersection of golden and SLM by id).")

    if len(merged) == 0:
        logger.warning("No overlapping ids; nothing to evaluate.")
        return

    merged["truth_4class"] = merged.apply(lambda r: recalculate_4class_label(r, "_truth"), axis=1)
    merged["slm_4class"] = merged.apply(lambda r: recalculate_4class_label(r, "_slm"), axis=1)
    merged["truth_binary"] = merged["truth_4class"].map(fourclass_to_binary)
    merged["slm_binary"] = merged["slm_4class"].map(fourclass_to_binary)

    if "xgboost_testlabels" in golden_df.columns:
        truth_3_map = golden_df.set_index("id")["xgboost_testlabels"].map(normalize_threeclass)
        merged["truth_3class"] = merged["id"].map(truth_3_map)
    else:
        merged["truth_3class"] = merged["truth_4class"].map(fourclass_to_threeclass)
    merged["slm_3class"] = merged["slm_4class"].map(fourclass_to_threeclass)

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

    logger.info("\n" + "=" * 60)
    logger.info("--- Binary (accept vs keep_base) ---")
    logger.info("=" * 60)
    logger.info(f"Accuracy:  {acc_bin:.4%}")
    logger.info(f"Precision: {prec_bin:.4f}  Recall: {rec_bin:.4f}  F1: {f1_bin:.4f}")
    logger.info(f"Confusion matrix:  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    # ---------- 4-class metrics ----------
    y_true_4 = _label_to_index_4class(merged["truth_4class"])
    y_pred_4 = _label_to_index_4class(merged["slm_4class"])
    valid_4 = (y_true_4 >= 0) & (y_pred_4 >= 0)
    y_true_4 = y_true_4[valid_4]
    y_pred_4 = y_pred_4[valid_4]
    acc_4 = (y_true_4 == y_pred_4).mean()
    macro_f1_4 = _manual_f1_macro(y_true_4, y_pred_4, n_classes=4)
    cm_4 = _confusion_matrix(y_true_4, y_pred_4, 4)

    logger.info("\n" + "=" * 60)
    logger.info("--- 4-class (none / alt / base / both) ---")
    logger.info("=" * 60)
    logger.info(f"Accuracy:  {acc_4:.4%}")
    logger.info(f"Macro F1: {macro_f1_4:.4f}")
    logger.info("\nClassification report:\n" + _classification_report_4class(y_true_4, y_pred_4))

    # ---------- 3-class metrics ----------
    y_true_3 = _label_to_index_3class(merged["truth_3class"])
    y_pred_3 = _label_to_index_3class(merged["slm_3class"])
    valid_3 = (y_true_3 >= 0) & (y_pred_3 >= 0)
    y_true_3 = y_true_3[valid_3]
    y_pred_3 = y_pred_3[valid_3]
    acc_3 = (y_true_3 == y_pred_3).mean()
    cm_3 = _confusion_matrix(y_true_3, y_pred_3, 3)

    logger.info("\n" + "=" * 60)
    logger.info("--- 3-class (alt / both / base) ---")
    logger.info("=" * 60)
    logger.info(f"Accuracy: {acc_3:.4%}")

    # Disagreements
    disagreements = merged[merged["truth_4class"] != merged["slm_4class"]].copy()
    logger.info(f"\nDisagreements (4-class): {len(disagreements)}")

    if len(disagreements) > 0:
        output_disagreement = "incorrect_predictions.csv"
        disagreements.to_csv(output_disagreement, index=False)
        logger.info(f"Saved disagreements to {output_disagreement}")

    # Summary file
    summary_path = "evaluation_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Phase 4 Evaluation summary\nTruth: {GOLDEN_PATH} | SLM: {SLM_PATH}\nSample: {n}\n")
        f.write(f"Binary Acc: {acc_bin:.4%}\n4-Class Acc: {acc_4:.4%}\n3-Class Acc: {acc_3:.4%}\n")
    logger.info(f"Saved summary to {summary_path}")

if __name__ == "__main__":
    main()
