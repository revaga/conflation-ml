
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

def map_3class(val):
    if pd.isna(val): return "base"
    v = str(val).lower().strip()
    mapping = {
        "alt": "alt", "match": "alt", 
        "both": "both", 
        "base": "base", "none": "base", "keep_base": "base"
    }
    return mapping.get(v, "base")

def map_binary(val):
    v = map_3class(val)
    if v in ["alt", "both"]:
        return "accept"
    return "reject"

def evaluate(slm_path, golden_path, name):
    print(f"\n{'='*20} {name} {'='*20}")
    slm = pd.read_parquet(slm_path)
    golden = pd.read_parquet(golden_path)
    merged = golden[["id", "3class_testlabels"]].merge(slm, on="id", how="inner")
    
    if "golden_label" not in merged.columns:
        print(f"Column 'golden_label' missing. Columns: {merged.columns.tolist()}")
        return

    truth_3 = merged["3class_testlabels"].apply(map_3class)
    pred_3 = merged["golden_label"].apply(map_3class)
    
    acc_3 = accuracy_score(truth_3, pred_3)
    
    truth_bin = merged["3class_testlabels"].apply(map_binary)
    pred_bin = merged["golden_label"].apply(map_binary)
    acc_bin = accuracy_score(truth_bin, pred_bin)
    
    print(f"3-Class Accuracy: {acc_3:.4%}")
    print(f"Binary Accuracy:  {acc_bin:.4%} (Accept vs Reject)")
    
    print("\n3-Class Distribution (Pred):")
    print(merged["golden_label"].value_counts())

if __name__ == "__main__":
    evaluate("data/phase3_slm_labeled.parquet", "data/golden_dataset_200.parquet", "Base SLM (Gemma3 4B)")
    evaluate("data/phase3_slm_labeledkimi.parquet", "data/golden_dataset_200.parquet", "Kimi SLM")
