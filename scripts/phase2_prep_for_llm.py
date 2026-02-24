import pandas as pd

def prepare_llm_candidates():
    print("Loading processed data...")
    # Read the parquet file generated in Phase 1
    df = pd.read_parquet('data/phase1_processed.parquet')
    
    # Define "Hard Cases" for LLM evaluation
    # These are cases where fuzzy matching is inconclusive (e.g., 30-80% similarity).
    # - If similarity > 80%, we can likely trust the match.
    # - If similarity < 30%, it's likely a mismatch (or different location).
    # - The 30-80% range is where "Intelligent" arbitration is most valuable.
    
    mask = (df['addr_similarity_ratio'] >= 30) & (df['addr_similarity_ratio'] <= 80)
    hard_cases = df[mask]
    
    print(f"Total rows: {len(df)}")
    print(f"Hard cases (30 <= address sim <= 80): {len(hard_cases)}")
    
    if len(hard_cases) == 0:
        print("No hard cases found. Using random sample.")
        sample = df.sample(5)
    else:
        sample = hard_cases.head(5)
    
    print("\n--- Example LLM Prompt (Prototype) ---")
    row = sample.iloc[0]
    
    prompt = f"""
You are an expert Data Steward responsible for maintaining a Golden Dataset of places.
Your task is to determine if the following two records represent the SAME physical place or DIFFERENT places.

Record A (Incoming Source):
- Name: {row.get('names', 'N/A')}
- Address: {row.get('norm_conflated_addr', 'N/A')}
- Phone: {row.get('norm_conflated_phone', 'N/A')}
- Website: {row.get('norm_conflated_website', 'N/A')}
- Categories: {row.get('categories', 'N/A')}

Record B (Base/Golden Record):
- Name: {row.get('base_names', 'N/A')}
- Address: {row.get('norm_base_addr', 'N/A')}
- Phone: {row.get('norm_base_phone', 'N/A')}
- Website: {row.get('norm_base_website', 'N/A')}
- Categories: {row.get('base_categories', 'N/A')}

Address Similarity Score: {row.get('addr_similarity_ratio', 0):.1f}%

Instructions:
1. Analyze the differences in Name, Address, and Phone.
2. Consider if the address difference implies a sub-unit (e.g., Suite 100 vs Suite 200), a nearby building, or a completely different location.
3. Determine if this is a "Match" (Same Place), "No Match" (Different Place), or "Ambiguous".
4. Provide a short reasoning.

Output Format: JSON with keys "judgment" (Enum: MATCH, NO_MATCH, AMBIGUOUS) and "reasoning" (string).
"""
    print(prompt)
    print("-" * 40)
    
    output_path = 'data/llm_candidates.csv'
    sample.to_csv(output_path, index=False)
    print(f"Saved {len(sample)} candidate pairs to {output_path} for testing.")

if __name__ == "__main__":
    prepare_llm_candidates()
