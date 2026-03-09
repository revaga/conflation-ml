import pandas as pd
import rapidfuzz
from parquet_io import read_parquet_safe
from rapidfuzz import fuzz
import json
import re
import numpy as np

def expand_abbreviations(address):
    if not address or pd.isna(address):
        return ""
    
    # Basic mapping - can be expanded
    abbr_map = {
        r"\bst\b": "street",
        r"\bave\b": "avenue",
        r"\bdr\b": "drive",
        r"\brd\b": "road",
        r"\bblvd\b": "boulevard",
        r"\bln\b": "lane",
        r"\bct\b": "court",
        r"\bpl\b": "place",
        r"\bsq\b": "square",
        r"\bpkwy\b": "parkway",
        r"\bcir\b": "circle",
        r"\bhwy\b": "highway",
    }
    
    normalized = address.lower()
    for pattern, replacement in abbr_map.items():
        normalized = re.sub(pattern, replacement, normalized)
    return normalized

def standardize_phone(phone):
    if not phone or pd.isna(phone):
        return ""
    # Remove all non-digit characters
    return re.sub(r"\D", "", str(phone))

def normalize_website(url):
    if not url or pd.isna(url):
        return ""
    
    url = url.lower().strip()
    url = re.sub(r"^https?://", "", url)
    url = re.sub(r"^www\.", "", url)
    return url.strip("/")

def safe_json_parse(x):
    if pd.isna(x):
        return {}
    try:
        return json.loads(x)
    except:
        return {}

def process_addresses(df):
    # Parse JSON if columns are strings
    def get_address_text(val):
        if isinstance(val, str):
            try:
                data = json.loads(val)
                # Try to get freeform or construct from components
                if isinstance(data, list) and len(data) > 0:
                     data = data[0] # Take first address
                
                if isinstance(data, dict):
                     return data.get('freeform', '')
                return str(val)
            except:
                return str(val)
        return ""

    df['norm_conflated_addr'] = df['addresses'].apply(lambda x: expand_abbreviations(get_address_text(x)))
    df['norm_base_addr'] = df['base_addresses'].apply(lambda x: expand_abbreviations(get_address_text(x)))
    return df

def main():
    print("Loading data...")
    # Using pandas to read parquet
    try:
        df = read_parquet_safe('data/project_a_samples.parquet')
    except FileNotFoundError:
        print("Data file not found. Please ensure 'data/project_a_samples.parquet' exists.")
        return

    print(f"Loaded {len(df)} rows.")

    # 1. Normalization
    print("Normalizing data...")
    
    # Addresses
    df = process_addresses(df)
    
    # Phones
    df['norm_conflated_phone'] = df['phones'].apply(standardize_phone)
    df['norm_base_phone'] = df['base_phones'].apply(standardize_phone)
    
    # Websites
    df['norm_conflated_website'] = df['websites'].apply(normalize_website)
    df['norm_base_website'] = df['base_websites'].apply(normalize_website)

    # 2. Fuzzy Matching Baseline
    print("Computing fuzzy matching scores...")
    
    def compute_similarity(row, col1, col2, scorer=fuzz.ratio):
        val1 = row[col1]
        val2 = row[col2]
        if not val1 or not val2:
            return 0
        return scorer(val1, val2)

    # Address Similarity
    df['addr_similarity_ratio'] = df.apply(lambda x: compute_similarity(x, 'norm_conflated_addr', 'norm_base_addr', fuzz.ratio), axis=1)
    df['addr_token_sort'] = df.apply(lambda x: compute_similarity(x, 'norm_conflated_addr', 'norm_base_addr', fuzz.token_sort_ratio), axis=1)

    # Phone Similarity (Exact match usually, but ratio handles minor diffs)
    df['phone_similarity'] = df.apply(lambda x: compute_similarity(x, 'norm_conflated_phone', 'norm_base_phone', fuzz.ratio), axis=1)

    # Website Similarity
    df['website_similarity'] = df.apply(lambda x: compute_similarity(x, 'norm_conflated_website', 'norm_base_website', fuzz.ratio), axis=1)

    # Save processed data
    output_path = 'data/phase1_processed.parquet'
    df.to_parquet(output_path)
    print(f"Processed data saved to {output_path}")
    
    # Print sample
    print("\nSample Results (first 5 rows):")
    cols = ['id', 'norm_conflated_addr', 'norm_base_addr', 'addr_similarity_ratio', 'phone_similarity', 'website_similarity']
    print(df[cols].head().to_string())

if __name__ == "__main__":
    main()
