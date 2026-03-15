import pandas as pd
import numpy as np
import logging
import argparse
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve
from scripts.parquet_io import read_parquet_safe
from scripts.features import engineer_features

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def calibrate_both_thresholds():
    logger.info("Calibrating 'Both' Classification using Logistic Regression...")
    if not os.path.exists("data/golden_dataset_200.parquet"):
        logger.error("Golden dataset not found.")
        return

    df = read_parquet_safe("data/golden_dataset_200.parquet")
    df = engineer_features(df)
    
    # Ground Truth mapping
    df["truth"] = df["3class_testlabels"].str.lower().map({"alt": "alt", "match": "alt", "both": "both", "base": "base"})
    
    # We focus on the 'both' vs 'alt' distinction among accepted matches
    # Mask for records that are either 'alt' or 'both' in ground truth
    mask = df["truth"].isin(["alt", "both"])
    eval_df = df[mask].copy()
    
    if len(eval_df) < 10:
        logger.error("Not enough golden alt/both cases to calibrate.")
        return

    # Binary target for PR curve: 1 if 'both', 0 if 'alt'
    y_true = (eval_df["truth"] == "both").astype(int)
    
    eval_df["abs_conf_delta"] = eval_df["feat_existence_conf_delta"].abs()
    
    features = [
        "feat_name_similarity",
        "feat_addr_similarity",
        "feat_phone_similarity",
        "feat_web_similarity",
        "abs_conf_delta"
    ]
    
    X = eval_df[features].fillna(0).values
    
    # Train Logistic Regression
    clf = LogisticRegression(random_state=42, class_weight='balanced')
    clf.fit(X, y_true)
    
    # Save the model
    os.makedirs("data/models", exist_ok=True)
    model_path = "data/models/both_logistic.joblib"
    joblib.dump(clf, model_path)
    logger.info(f"Saved Logistic Regression model to {model_path}")
    
    for fname, coef in zip(features, clf.coef_[0]):
        logger.info(f"Feature: {fname:25} Weight: {coef:.4f}")
    
    eval_df["both_probability"] = clf.predict_proba(X)[:, 1]
    
    # Plot PR Curve
    precision, recall, thresholds = precision_recall_curve(y_true, eval_df["both_probability"])
    
    os.makedirs("data/experiments", exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label='PR Curve (Both Class / Logistic Prob)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for "Both" Classification (Logistic)')
    plt.grid(True)
    plt.savefig("data/experiments/both_logistic_pr_curve.png")
    logger.info("Saved PR curve to data/experiments/both_logistic_pr_curve.png")

    # Find threshold that maximizes F1
    f1_scores = []
    # precision_recall_curve returns thresholds of length len(precision)-1
    for t in thresholds:
        y_pred = (eval_df["both_probability"] >= t).astype(int)
        f1_scores.append(f1_score(y_true, y_pred))
    
    if not f1_scores:
        logger.warning("No thresholds generated.")
        return
        
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    logger.info(f"Best Prob Threshold for 'Both' scoring: {best_threshold:.4f}")
    logger.info(f"Max F1 Score (Both vs alt): {best_f1:.4f}")
    
    # Distribution of probabilities
    logger.info("Prob Distribution:")
    logger.info("\n" + str(eval_df.groupby("truth")["both_probability"].describe()))

if __name__ == "__main__":
    calibrate_both_thresholds()
