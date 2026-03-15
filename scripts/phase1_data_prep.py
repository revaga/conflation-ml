import pandas as pd
import rapidfuzz
import logging
import numpy as np
from scripts.parquet_io import read_parquet_safe
from rapidfuzz import fuzz
from scripts.normalization import process_addresses, standardize_phone, normalize_website

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Loading data...")
    try:
        df = read_parquet_safe('data/project_a_samples.parquet')
    except FileNotFoundError:
        logger.error("Data file not found. Please ensure 'data/project_a_samples.parquet' exists.")
        return

    logger.info(f"Loaded {len(df)} rows.")

    logger.info("Normalizing data...")
    df = process_addresses(df)
    df["norm_conflated_phone"] = df["phones"].apply(standardize_phone)
    df["norm_base_phone"] = df["base_phones"].apply(standardize_phone)
    df["norm_conflated_website"] = df["websites"].apply(normalize_website)
    df["norm_base_website"] = df["base_websites"].apply(normalize_website)

    logger.info("Computing fuzzy matching scores...")
    
    def compute_similarity(row, col1, col2, scorer=fuzz.ratio):
        val1 = row[col1]
        val2 = row[col2]
        if not val1 or not val2:
            return 0
        return scorer(val1, val2)

    df['addr_similarity_ratio'] = df.apply(lambda x: compute_similarity(x, 'norm_conflated_addr', 'norm_base_addr', fuzz.ratio), axis=1)
    df['addr_token_sort'] = df.apply(lambda x: compute_similarity(x, 'norm_conflated_addr', 'norm_base_addr', fuzz.token_sort_ratio), axis=1)
    df['phone_similarity'] = df.apply(lambda x: compute_similarity(x, 'norm_conflated_phone', 'norm_base_phone', fuzz.ratio), axis=1)
    df['website_similarity'] = df.apply(lambda x: compute_similarity(x, 'norm_conflated_website', 'norm_base_website', fuzz.ratio), axis=1)

    output_path = 'data/phase1_processed.parquet'
    df.to_parquet(output_path)
    logger.info(f"Processed data saved to {output_path}")
    
    logger.info("Sample Results (first 5 rows):")
    cols = ['id', 'norm_conflated_addr', 'norm_base_addr', 'addr_similarity_ratio', 'phone_similarity', 'website_similarity']
    logger.info("\n" + df[cols].head().to_string())

if __name__ == "__main__":
    main()
