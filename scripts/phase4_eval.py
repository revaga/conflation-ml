import pandas as pd
import os

# NOTE: scikit-learn dependency removed to avoid DLL load issues.
# Implementing metrics manually using pandas.

def main():
    print("Loading data...")
    try:
        slm_df = pd.read_csv('SLM_predictions.csv')
        base_df = pd.read_parquet('data/phase2_scored.parquet')
    except FileNotFoundError:
        print("Error: SLM_predictions.csv or data/phase2_scored.parquet not found.")
        print("Please run phase2_similarity.py and phase3_slm.py first.")
        return

    # Merge on ID
    # SLM predictions might be a subset
    # Join on 'id' -> base_df has 'id' too.
    merged = pd.merge(slm_df, base_df, on='id', suffixes=('_slm', '_base'))
    
    print(f"Loaded {len(merged)} records for evaluation.")
    
    # metrics calculation
    y_true = merged['SLM_same_place'].astype(int) # 0 or 1
    y_pred = merged['pred_label_base'].astype(int)     # 0 or 1
    
    # Accuracy
    correct = (y_true == y_pred).sum()
    total = len(merged)
    acc = correct / total if total > 0 else 0
    
    # Confusion Matrix
    # TP: True=1, Pred=1
    # TN: True=0, Pred=0
    # FP: True=0, Pred=1
    # FN: True=1, Pred=0
    
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    
    print("\n--- Evaluation: Baseline Model vs SLM Verification ---")
    print(f"Accuracy (Agreement): {acc:.2%}")
    print("\nConfusion Matrix:")
    print(f"                 Predicted NO   Predicted YES")
    print(f"SLM says NO      {tn:<12}   {fp:<12}")
    print(f"SLM says YES     {fn:<12}   {tp:<12}")
    
    # Extract Disagreements
    disagreements = merged[y_true != y_pred].copy()
    
    print(f"\nDisagreements (Potential Baseline Errors): {len(disagreements)}")
    
    if len(disagreements) > 0:
        output_disagreement = 'incorrect_predictions.csv'
        # Select relevant columns if available
        # base_df has 'names', 'norm_conflated_addr', etc.
        # These are in merged too.
        desired_cols = ['id', 'names', 'norm_conflated_addr', 'norm_base_addr', 'match_score', 'pred_label', 'SLM_same_place', 'SLM_reason']
        
        # Check availability
        available_cols = [c for c in desired_cols if c in merged.columns]
        
        disagreements[available_cols].to_csv(output_disagreement, index=False)
        print(f"Saved {len(disagreements)} disagreements to {output_disagreement}")
        
    # Save Summary
    summary_path = 'evaluation_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"Validation against SLM (Sample Size: {len(merged)})\n")
        f.write(f"Accuracy: {acc:.2%}\n")
        f.write(f"Confusion Matrix:\n")
        f.write(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}\n")
    print(f"Saved summary to {summary_path}")

if __name__ == "__main__":
    main()
