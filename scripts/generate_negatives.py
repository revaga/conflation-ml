import pandas as pd
import numpy as np
import json
import logging
import random
from tqdm import tqdm
from scripts.features import extract_primary_name, safe_json
from scripts.normalization import process_addresses, standardize_phone, normalize_website
from scripts.labels import LABEL_4CLASS

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def extract_locality_postcode(val):
    obj = safe_json(val)
    if isinstance(obj, list) and len(obj) > 0:
        obj = obj[0]
    if isinstance(obj, dict):
        return obj.get("locality", "").lower().strip(), obj.get("postcode", "").lower().strip()
    return "", ""

def main():
    logger.info("Generating negative samples for 'none' class...")
    
    # Load base dataset
    input_file = 'data/project_a_samples.parquet'
    try:
        df = pd.read_parquet(input_file)
    except FileNotFoundError:
        logger.error(f"Input file {input_file} not found.")
        return

    logger.info(f"Loaded {len(df)} source records.")

    # We treat each row as a source of one 'alt' record and one 'base' record.
    # Actually, the base record is defined by base_* columns.
    
    # Separate base attributes and alt attributes
    alt_cols = ['id', 'sources', 'names', 'categories', 'confidence', 'websites', 'socials', 'emails', 'phones', 'brand', 'addresses']
    base_cols = ['base_id', 'base_sources', 'base_names', 'base_categories', 'base_confidence', 'base_websites', 'base_socials', 'base_emails', 'base_phones', 'base_brand', 'base_addresses']
    
    # Filter out rows that are missing critical base/alt info if any
    df_clean = df.dropna(subset=['names', 'base_names', 'addresses', 'base_addresses'])
    
    alt_df = df_clean[alt_cols].copy()
    base_df = df_clean[base_cols].copy()
    
    # Pre-extract names and locations for easier matching
    logger.info("Pre-extracting names and locations...")
    alt_df['_name'] = alt_df['names'].apply(extract_primary_name)
    base_df['_base_name'] = base_df['base_names'].apply(extract_primary_name)
    
    alt_df['_loc'] = alt_df['addresses'].apply(extract_locality_postcode)
    base_df['_base_loc'] = base_df['base_addresses'].apply(extract_locality_postcode)

    negatives = []

    # 1. Random Negatives (5,000)
    logger.info("Generating 5,000 Random Negatives...")
    count = 0
    attempts = 0
    max_attempts = 50000
    
    while count < 5000 and attempts < max_attempts:
        attempts += 1
        alt_idx = random.randint(0, len(alt_df) - 1)
        base_idx = random.randint(0, len(base_df) - 1)
        
        # Avoid same row (unlikely but good to check) or same base_id/id if match
        if alt_df.iloc[alt_idx]['id'] == base_df.iloc[base_idx]['base_id']:
            continue
            
        alt_loc = alt_df.iloc[alt_idx]['_loc']
        base_loc = base_df.iloc[base_idx]['_base_loc']
        
        # Check if city or postcode is different
        if alt_loc[0] != base_loc[0] or alt_loc[1] != base_loc[1]:
            alt_row = alt_df.iloc[alt_idx].to_dict()
            base_row = base_df.iloc[base_idx].to_dict()
            combined = {**alt_row, **base_row}
            negatives.append(combined)
            count += 1
            if count % 1000 == 0:
                logger.info(f"Generated {count} random negatives...")

    # 2. Hard Negatives (1,000) - Same name, different location
    logger.info("Generating 1,000 Hard Negatives...")
    
    # Group by name
    name_to_alt_indices = {}
    for i, name in enumerate(alt_df['_name']):
        if name and len(name) > 3: # Ignore very short names
            if name not in name_to_alt_indices:
                name_to_alt_indices[name] = []
            name_to_alt_indices[name].append(i)
            
    name_to_base_indices = {}
    for i, name in enumerate(base_df['_base_name']):
        if name and len(name) > 3:
            if name not in name_to_base_indices:
                name_to_base_indices[name] = []
            name_to_base_indices[name].append(i)
            
    # Common names
    common_names = set(name_to_alt_indices.keys()) & set(name_to_base_indices.keys())
    logger.info(f"Found {len(common_names)} common names between alt and base.")
    
    hard_count = 0
    shuffled_names = list(common_names)
    random.shuffle(shuffled_names)
    
    for name in shuffled_names:
        if hard_count >= 1000:
            break
            
        alt_indices = name_to_alt_indices[name]
        base_indices = name_to_base_indices[name]
        
        # Try to find a pair with different locations
        found_for_name = False
        for ai in alt_indices:
            if found_for_name: break
            for bi in base_indices:
                alt_loc = alt_df.iloc[ai]['_loc']
                base_loc = base_df.iloc[bi]['_base_loc']
                
                # If they have different locations, it's a hard negative
                if alt_loc != base_loc:
                    # Double check they are not already the same record
                    if alt_df.iloc[ai]['id'] != base_df.iloc[bi]['base_id']:
                        alt_row = alt_df.iloc[ai].to_dict()
                        base_row = base_df.iloc[bi].to_dict()
                        combined = {**alt_row, **base_row}
                        negatives.append(combined)
                        hard_count += 1
                        found_for_name = True
                        break
        
        if hard_count % 200 == 0 and hard_count > 0:
            logger.info(f"Generated {hard_count} hard negatives...")

    if not negatives:
        logger.warning("No negatives generated.")
        return

    neg_df = pd.DataFrame(negatives)
    
    # Remove internal columns
    internal_cols = ['_name', '_base_name', '_loc', '_base_loc']
    neg_df = neg_df.drop(columns=[col for col in internal_cols if col in neg_df.columns])
    
    # Assign 'none' label
    neg_df['golden_label'] = 'none'
    
    # Add other required winners if needed (can be 'none')
    for attr in ["name", "phone", "web", "address", "category"]:
        neg_df[f"attr_{attr}_winner"] = 'none'
        
    # Re-normalize/Process just in case
    logger.info("Normalizing address/phone/website for negative samples...")
    neg_df = process_addresses(neg_df)
    
    # Fixed types for Parquet
    # Ensure id/base_id are strings
    neg_df['id'] = neg_df['id'].astype(str)
    neg_df['base_id'] = neg_df['base_id'].astype(str)
    
    # Ensure confidence is float
    neg_df['confidence'] = neg_df['confidence'].astype(float)
    neg_df['base_confidence'] = neg_df['base_confidence'].astype(float)

    # Ensure phone/web normalization
    neg_df["norm_conflated_phone"] = neg_df["phones"].apply(standardize_phone)
    neg_df["norm_base_phone"] = neg_df["base_phones"].apply(standardize_phone)
    neg_df["norm_conflated_website"] = neg_df["websites"].apply(normalize_website)
    neg_df["norm_base_website"] = neg_df["base_websites"].apply(normalize_website)
    
    # Dummy scores (should be low since they are negatives)
    from rapidfuzz import fuzz
    def compute_similarity(row, col1, col2):
        val1 = row[col1]
        val2 = row[col2]
        if not val1 or not val2 or pd.isna(val1) or pd.isna(val2):
            return 0.0
        return float(fuzz.ratio(str(val1), str(val2)))

    logger.info("Computing fuzzy scores for negatives...")
    neg_df['addr_similarity_ratio'] = neg_df.apply(lambda x: compute_similarity(x, 'norm_conflated_addr', 'norm_base_addr'), axis=1)
    neg_df['addr_token_sort'] = neg_df.apply(lambda x: compute_similarity(x, 'norm_conflated_addr', 'norm_base_addr'), axis=1)
    neg_df['phone_similarity'] = neg_df.apply(lambda x: compute_similarity(x, 'norm_conflated_phone', 'norm_base_phone'), axis=1)
    neg_df['website_similarity'] = neg_df.apply(lambda x: compute_similarity(x, 'norm_conflated_website', 'norm_base_website'), axis=1)

    # Cast similarity columns to numeric explicitly
    for col in ['addr_similarity_ratio', 'addr_token_sort', 'phone_similarity', 'website_similarity']:
        neg_df[col] = pd.to_numeric(neg_df[col], errors='coerce').fillna(0.0)

    output_path = 'data/negative_samples.parquet'
    # Final check on object columns - convert any mixed types to string
    for col in neg_df.columns:
        if neg_df[col].dtype == 'object':
            # If it's not a list or dict, convert to string
            pass

    neg_df.to_parquet(output_path)
    logger.info(f"Generated {len(neg_df)} negative samples saved to {output_path}")

if __name__ == "__main__":
    main()
