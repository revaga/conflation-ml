import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from scripts.parquet_io import read_parquet_safe
from scripts.features import engineer_features

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def analyze_both_distribution():
    logger.info("Analyzing Golden Dataset 'both' cases...")
    df = read_parquet_safe("data/golden_dataset_200.parquet")
    
    # We need features to analyze them
    df = engineer_features(df)
    
    # Filter for 'both' ground truth
    both_df = df[df["3class_testlabels"] == "both"]
    alt_df = df[df["3class_testlabels"] == "alt"]
    base_df = df[df["3class_testlabels"] == "base"]
    
    logger.info(f"Total rows: {len(df)}")
    logger.info(f"  both:  {len(both_df)}")
    logger.info(f"  alt:   {len(alt_df)}")
    logger.info(f"  base:  {len(base_df)}")
    
    features = [
        "feat_name_similarity",
        "feat_addr_similarity",
        "feat_phone_similarity",
        "feat_web_similarity",
        "feat_existence_conf_delta"
    ]
    
    stats = []
    for f in features:
        stats.append({
            "feature": f,
            "both_mean": both_df[f].mean(),
            "both_median": both_df[f].median(),
            "both_std": both_df[f].std(),
            "alt_mean": alt_df[f].mean(),
            "base_mean": base_df[f].mean()
        })
    
    stats_df = pd.DataFrame(stats)
    logger.info("Feature Distributions across classes:")
    logger.info("\n" + stats_df.to_string())
    
    # Simple Logistic Scoring heuristic based on means
    # Weight features by how well they separate 'both' from 'alt'
    logger.info("Proposed Logistic Thresholds based on Golden Distribution:")
    for f in features:
        q10 = both_df[f].quantile(0.1)
        logger.info(f"  {f} 10th percentile for 'both': {q10:.4f}")

if __name__ == "__main__":
    analyze_both_distribution()
