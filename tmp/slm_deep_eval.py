
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from scripts.labels import recalculate_3class_label, FOUR_TO_THREE

def map_3class(val):
    if pd.isna(val) or val is None: return "base"
    v = str(val).lower().strip()
    if v in ("match", "alt"): return "alt"
    if v in ("none", "keep_base"): return "base"
    return FOUR_TO_THREE.get(v, "base")

def get_metrics(truth, pred, name):
    acc = accuracy_score(truth, pred)
    f1_macro = f1_score(truth, pred, average='macro', zero_division=0)
    
    classes = ["alt", "both", "base"]
    p, r, f1, s = precision_recall_fscore_support(truth, pred, labels=classes, zero_division=0)
    
    return {
        "Method": name,
        "Accuracy": acc,
        "F1-Macro": f1_macro,
        "F1-alt": f1[0],
        "F1-both": f1[1],
        "F1-base": f1[2],
        "Support": len(truth)
    }

def main():
    GOLDEN_PATH = "data/golden_dataset_200.parquet"
    SLM_BASE_PATH = "data/phase3_slm_labeled.parquet"
    SLM_KIMI_PATH = "data/phase3_slm_labeledkimi.parquet"

    # 1. Load Truth
    golden = pd.read_parquet(GOLDEN_PATH)[["id", "3class_testlabels"]]
    golden["truth"] = golden["3class_testlabels"].apply(map_3class)
    
    results = []

    # 2. Recalculated SLM (GPT-4o mini)
    slm_base = pd.read_parquet(SLM_BASE_PATH)
    slm_base["recalc_pred"] = slm_base.apply(lambda r: recalculate_3class_label(r), axis=1)
    
    merged_recalc = golden.merge(slm_base[["id", "recalc_pred"]], on="id")
    results.append(get_metrics(merged_recalc["truth"], merged_recalc["recalc_pred"], "SLM Recalculated (GPT-4o mini)"))

    # 3. Base SLM (GPT-4o mini) - Raw golden_label
    # golden_label in slm_base tends to be 'base', 'alt', 'abstain'
    slm_base["raw_pred"] = slm_base["golden_label"].apply(map_3class)
    merged_base = golden.merge(slm_base[["id", "raw_pred"]], on="id")
    results.append(get_metrics(merged_base["truth"], merged_base["raw_pred"], "SLM Raw (GPT-4o mini)"))

    # 4. Kimi SLM - Raw golden_label
    slm_kimi = pd.read_parquet(SLM_KIMI_PATH)
    slm_kimi["raw_pred"] = slm_kimi["golden_label"].apply(map_3class)
    merged_kimi = golden.merge(slm_kimi[["id", "raw_pred"]], on="id")
    results.append(get_metrics(merged_kimi["truth"], merged_kimi["raw_pred"], "SLM Raw (Kimi)"))

    # Create summary table
    res_df = pd.DataFrame(results)
    
    # Print nice table
    print("\n" + "="*80)
    print(f"{'SLM Output Evaluation Comparison':^80}")
    print("="*80)
    
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(res_df.to_string(index=False))
    print("="*80)
    
    # Save to a new report
    res_df.to_csv("reports/slm_comparison_metrics.csv", index=False)
    print(f"\nSaved metrics to reports/slm_comparison_metrics.csv")

if __name__ == "__main__":
    main()
