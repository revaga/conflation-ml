import pandas as pd
import numpy as np
import re
from rapidfuzz import fuzz, distance

# NOTE: scikit-learn usage removed due to DLL load issues on the current environment.
# We will use RapidFuzz's token_set_ratio and JaroWinkler to approximate the requested metrics.
# token_set_ratio handles the "bag of words" overlap logic similar to TF-IDF for short strings.

def calculate_jaro_winkler(s1, s2):
    if pd.isna(s1) or pd.isna(s2):
        return 0.0
    return distance.JaroWinkler.similarity(str(s1), str(s2))

def parse_address_custom(addr):
    if not isinstance(addr, str):
        return {"num": "", "street": "", "unit": ""}
    
    # Basic robust parsing
    # Extract number at start
    match = re.search(r"^(\d+)\s+(.*)", addr)
    if match:
        num, rest = match.groups()
    else:
        num, rest = "", addr

    # Extract unit (apt, ste, unit, #)
    unit_match = re.search(r"\b(ste|suite|apt|unit|#)\s*([\w-]+)", rest)
    unit = ""
    if unit_match:
        unit = f"{unit_match.group(1)} {unit_match.group(2)}"
        rest = re.sub(r"\b(ste|suite|apt|unit|#)\s*[\w-]+", "", rest).strip()
    
    return {"num": num, "street": rest, "unit": unit}

def score_address_components(row):
    addr1 = parse_address_custom(row['norm_conflated_addr'])
    addr2 = parse_address_custom(row['norm_base_addr'])
    
    # Strict number match
    num_match = 1.0 if addr1['num'] == addr2['num'] and addr1['num'] else 0.0
    if not addr1['num'] and not addr2['num']:
        num_match = 0.5 # Neutral if both missing
        
    # Street name fuzzy
    street_sim = fuzz.token_sort_ratio(addr1['street'], addr2['street']) / 100.0
    
    # Unit match (strict but handles missing)
    if not addr1['unit'] and not addr2['unit']:
        unit_match = 1.0
    elif addr1['unit'] == addr2['unit']:
        unit_match = 1.0
    else:
        unit_match = 0.0
        
    # Weighted address score
    # Number is critical (40%), Street (40%), Unit (20%)
    return (num_match * 0.4) + (street_sim * 0.4) + (unit_match * 0.2)

def main():
    print("Loading data...")
    df = pd.read_parquet('data/phase1_processed.parquet')
    
    # 1. Name Similarity
    print("Calculating Name Similarities...")
    # TF-IDF approximation: Token Set Ratio (good for "bag of words" similarity)
    df['name_tfidf_proxy'] = df.apply(lambda x: fuzz.token_set_ratio(str(x['names']), str(x['base_names'])) / 100.0, axis=1)
    
    # Jaro-Winkler
    df['name_jaro'] = df.apply(lambda x: calculate_jaro_winkler(x['names'], x['base_names']), axis=1)
    
    # Standard Ratio
    df['name_fuzz'] = df.apply(lambda x: fuzz.ratio(str(x['names']), str(x['base_names'])) / 100.0, axis=1)
    
    # Aggregate Name Score
    # Adjusted weights to balance the lack of true TF-IDF
    df['score_name'] = (df['name_tfidf_proxy'] * 0.3) + (df['name_jaro'] * 0.3) + (df['name_fuzz'] * 0.4)
    
    # 2. Address Similarity
    print("Calculating Address Similarities...")
    df['score_address'] = df.apply(score_address_components, axis=1)
    
    # 3. Phone/Website (Exact or high fuzzy)
    print("Calculating Phone/Website Similarities...")
    df['score_phone'] = df['phone_similarity'] / 100.0
    df['score_website'] = df['website_similarity'] / 100.0
    
    # 4. Overall Match Score
    # Weights: Name(30%), Address(40%), Phone(20%), Website(10%)
    df['match_score'] = (
        (df['score_name'] * 0.3) + 
        (df['score_address'] * 0.4) + 
        (df['score_phone'] * 0.2) + 
        (df['score_website'] * 0.1)
    )
    
    # 5. Prediction Label (Threshold)
    # Simple threshold 0.7
    df['pred_label'] = (df['match_score'] >= 0.7).astype(int)
    
    print("Saving results...")
    output_path = 'data/phase2_scored.parquet'
    df.to_parquet(output_path)
    print(f"Saved scored data to {output_path}")
    
    # Export CSV as requested
    csv_path = 'output.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")
    
    # Validation stats
    print("\n--- Scoring Summary ---")
    print(df[['match_score', 'pred_label']].describe())
    print(f"\nPredicted Matches: {df['pred_label'].sum()} / {len(df)}")

if __name__ == "__main__":
    main()
