
import pandas as pd
import numpy as np
from scripts.labels import recalculate_3class_label, FOUR_TO_THREE

def map_3class(val):
    if pd.isna(val) or val is None: return "base"
    v = str(val).lower().strip()
    if v in ("match", "alt"): return "alt"
    if v in ("none", "keep_base"): return "base"
    return FOUR_TO_THREE.get(v, "base")

GOLDEN_PATH = "data/golden_dataset_200.parquet"
SLM_PATH = "data/phase3_slm_labeled.parquet"

golden = pd.read_parquet(GOLDEN_PATH)
slm = pd.read_parquet(SLM_PATH)

# Compare Truth Columns
print("--- Truth Column Comparison ---")
if "3class_testlabels" in golden.columns:
    print(f"3class_testlabels distribution:\n{golden['3class_testlabels'].value_counts()}")
else:
    print("3class_testlabels NOT found")

# Recalculate Truth from attr_*_winner in golden
golden_truth_recalc = golden.apply(lambda r: recalculate_3class_label(r), axis=1)
print(f"\nRecalculated Golden Truth distribution:\n{golden_truth_recalc.value_counts()}")

# Compare Prediction Logic
merged = golden.merge(slm, on="id", suffixes=("_truth", "_slm"))
pred_recalc = merged.apply(lambda r: recalculate_3class_label(r, "_slm"), axis=1)

print(f"\nPrediction distribution (Recalculated SLM):\n{pred_recalc.value_counts()}")

# Check phase4_eval logic
# It recalculates truth from _truth suffix and SLM from _slm suffix
truth_4 = merged.apply(lambda r: recalculate_3class_label(r, "_truth"), axis=1)
pred_4 = merged.apply(lambda r: recalculate_3class_label(r, "_slm"), axis=1)
acc_4 = (truth_4 == pred_4).mean()
print(f"\nPhase4-style Accuracy (Recalc vs Recalc): {acc_4:.4%}")

# Check slm_deep_eval logic
# It uses 3class_testlabels as truth and Recalc from SLM as prediction
truth_deep = merged["3class_testlabels"].apply(lambda x: "alt" if str(x).lower() == "match" else str(x).lower())
pred_deep = merged.apply(lambda r: recalculate_3class_label(r, "_slm"), axis=1)
acc_deep = (truth_deep == pred_deep).mean()
print(f"SLM-Deep-style Accuracy (3class_testlabels vs Recalc): {acc_deep:.4%}")
