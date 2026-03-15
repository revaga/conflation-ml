
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score
from scripts.labels import recalculate_3class_label

GOLDEN_PATH = "data/golden_dataset_200.parquet"
SLM_PATH = "data/phase3_slm_labeled.parquet"

golden = pd.read_parquet(GOLDEN_PATH)
slm = pd.read_parquet(SLM_PATH)

merged = golden.merge(slm, on="id", suffixes=("_truth", "_slm"))

truth_3 = merged.apply(lambda r: recalculate_3class_label(r, "_truth"), axis=1)
pred_3 = merged.apply(lambda r: recalculate_3class_label(r, "_slm"), axis=1)

report = classification_report(truth_3, pred_3, labels=["alt", "both", "base"], output_dict=True, zero_division=0)
macro_f1 = f1_score(truth_3, pred_3, average='macro', zero_division=0)

print(f"Method: phase4_eval Recalculated")
print(f"Accuracy: {(truth_3==pred_3).mean():.4f}")
print(f"F1-Macro: {macro_f1:.4f}")
print(f"F1-alt: {report['alt']['f1-score']:.4f}")
print(f"F1-both: {report['both']['f1-score']:.4f}")
print(f"F1-base: {report['base']['f1-score']:.4f}")
print(f"N: {len(merged)}")
