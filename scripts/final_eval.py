import pandas as pd
import numpy as np
import os
import logging
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from scripts.labels import FOUR_TO_THREE, recalculate_3class_label

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

GOLDEN_PATH = "data/golden_dataset_200.parquet"

def map_4to3(label):
    if label is None: return "base"
    return FOUR_TO_THREE.get(str(label).lower(), "base")

def main():
    logger.info("="*70)
    logger.info("Final Performance Evaluation (against Golden 200)")
    logger.info("="*70)

    # 1. Load Ground Truth
    if not os.path.exists(GOLDEN_PATH):
        logger.error(f"Golden dataset not found at {GOLDEN_PATH}")
        return
    
    gt_df = pd.read_parquet(GOLDEN_PATH)[["id", "3class_testlabels"]].copy()
    gt_df["id"] = gt_df["id"].astype(str)
    gt_df["truth"] = gt_df["3class_testlabels"].str.lower()
    
    results = []

    # Helper for scoring
    def score_df(df, pred_col, name, mapping=None):
        df = df.copy()
        df["id"] = df["id"].astype(str)
        merged = gt_df.merge(df[["id", pred_col]], on="id", how="inner")
        
        if mapping:
            merged["pred"] = merged[pred_col].apply(mapping)
        else:
            merged["pred"] = merged[pred_col].str.lower()
        
        # Filter for rows that actually have a prediction
        mask = merged["truth"].notna() & merged["pred"].notna()
        t = merged.loc[mask, "truth"]
        p = merged.loc[mask, "pred"]
        
        if len(t) == 0:
            return
            
        acc = accuracy_score(t, p)
        f1 = f1_score(t, p, average='macro')
        
        # Per-class F1
        classes = ["alt", "both", "base"]
        p_class, r_class, f1_class, _ = precision_recall_fscore_support(t, p, labels=classes, zero_division=0)
        
        res = {
            "Method": name,
            "Accuracy": acc,
            "F1-Macro": f1,
            "F1-alt": f1_class[0],
            "F1-both": f1_class[1],
            "F1-base": f1_class[2],
            "N": len(t)
        }
        results.append(res)
        logger.info(f"Evaluated {name} on {len(t)} rows.")

    # 2. Baseline & Pipeline earlier predictions
    if os.path.exists("data/phase5_full_results.parquet"):
        p5_df = pd.read_parquet("data/phase5_full_results.parquet")
        if "baseline_selection" in p5_df.columns:
            score_df(p5_df, "baseline_selection", "Highest Conf Baseline")
        if "model_3class_prediction" in p5_df.columns:
            score_df(p5_df, "model_3class_prediction", "Earlier 3-Class Filter")

    # 3. XGBoost 2-Stage (Binary + Refiner)
    if os.path.exists("data/xgboost_results.parquet"):
        xgb_df = pd.read_parquet("data/xgboost_results.parquet")
        if "xgb_prediction" in xgb_df.columns:
            score_df(xgb_df, "xgb_prediction", "XGBoost 2-Stage (Binary + Refiner)")

    # 4. XGBoost 4-Class Multiclass
    if os.path.exists("data/xgboost_multiclass_results.parquet"):
        multi_df = pd.read_parquet("data/xgboost_multiclass_results.parquet")
        if "xgb_4class_pred" in multi_df.columns:
            score_df(multi_df, "xgb_4class_pred", "XGBoost 4-Class Multiclass", mapping=map_4to3)

    # 5. Native SLM Output
    if os.path.exists("data/phase3_slm_labeled.parquet"):
        slm_df = pd.read_parquet("data/phase3_slm_labeled.parquet")
        if "golden_label" in slm_df.columns:
            score_df(slm_df, "golden_label", "Native SLM Output", mapping=map_4to3)

    # Display Summary Table
    if results:
        res_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
        pd.set_option('display.float_format', '{:.4f}'.format)
        logger.info("\n" + res_df.to_string(index=False))
        
        # Save summary
        res_df.to_csv("reports/final_metrics_summary.csv", index=False)
        logger.info("\nPerformance summary saved to reports/final_metrics_summary.csv")
    else:
        logger.warning("No results to evaluate.")

if __name__ == "__main__":
    main()
