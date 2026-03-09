import os
import json
import pandas as pd
from openai import OpenAI
from parquet_io import read_parquet_safe

# Configuration
# This looks for standard OpenAI API key, or explicit GROQ_API_KEY if using Groq
API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1") if os.getenv("GROQ_API_KEY") else None

MODEL_NAME = "llama-3.3-70b-versatile" # Default as requested

def get_client():
    if not API_KEY:
        print("WARNING: No API Key found. Set GROQ_API_KEY or OPENAI_API_KEY environment variable.")
        return None
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)

def construct_prompt(row):
    prompt = f"""
You are an expert Data Steward responsible for resolving Place Conflation issues.
Your goal is to determine if two records represent the SAME physical place.

Input Data:
- Similarity Score: {row['match_score']:.2f} (0.0 to 1.0)
- Baseline Prediction: {'MATCH' if row['pred_label'] == 1 else 'NO MATCH'}

Record A (Conflated Source):
- Name: {row['names']}
- Address: {row['norm_conflated_addr']}
- Phone: {row['norm_conflated_phone']}
- Website: {row['norm_conflated_website']}
- Category: {row['categories']}

Record B (Base Record):
- Name: {row['base_names']}
- Address: {row['norm_base_addr']}
- Phone: {row['norm_base_phone']}
- Website: {row['norm_base_website']}
- Category: {row['base_categories']}

Instructions:
1. Analyze the similarity score and baseline prediction.
2. Verify the addresses. Use your knowledge of address normalization (e.g. 'St' vs 'Street', unit numbers).
3. If the baseline prediction seems wrong based on the address details (e.g. valid match but low score due to typo, or high score but different suite numbers), OVERRIDE it.
4. If addresses disagree significantly (different building number or street), it is likely NO MATCH.
5. If phone or website matches exactly, weight this heavily towards MATCH.

Output JSON Format:
{{
  "label": alt/base/both/none,
  "confidence": 0.0 to 1.0,
  "reason": "One sentence explanation referencing specific fields."
}}
"""
    return prompt

def call_slm(client, prompt, model=MODEL_NAME):
    if not client:
        # Mock response for testing without API key
        return {
            "label": "none",
            "confidence": 0.0,
            "reason": "API Key missing, mock response."
        }
        
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"Error calling SLM: {e}")
        return {
            "same_place": False,
            "confidence": 0.0,
            "reason": f"Error: {str(e)}"
        }

def main():
    print(f"Model: {MODEL_NAME}")
    print("Loading scored data...")
    try:
        df = read_parquet_safe('data/phase2_scored.parquet')
    except FileNotFoundError:
        print("Error: data/phase2_scored.parquet not found. Run phase2_similarity.py first.")
        return

    # Filter for interesting cases (e.g., mismatch between high score and low score?)
    # Or just take a sample of 20 for demonstration
    sample_size = 20
    print(f"Processing a sample of {sample_size} records...")
    
    # We prioritize "Ambiguous" cases (score 0.4 to 0.8) if possible
    ambiguous = df[(df['match_score'] >= 0.4) & (df['match_score'] <= 0.8)]
    if len(ambiguous) > 0:
        sample = ambiguous.head(sample_size)
    else:
        sample = df.head(sample_size)
        
    client = get_client()
    
    results = []
    
    for idx, row in sample.iterrows():
        prompt = construct_prompt(row)
        result = call_slm(client, prompt)
        
        results.append({
            "id": row['id'],
            "base_id": row['base_id'],
            "match_score": row['match_score'],
            "pred_label": row['pred_label'],
            "SLM_same_place": result.get('same_place'),
            "SLM_confidence": result.get('confidence'),
            "SLM_reason": result.get('reason')
        })
        print(f"Processed {row['id'][:8]}... SLM: {result.get('same_place')}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save output
    output_path = 'SLM_predictions.csv'
    results_df.to_csv(output_path, index=False)
    print(f"Saved SLM predictions to {output_path}")
    
    # Analyze overrides
    overrides = results_df[results_df['pred_label'] != results_df['SLM_same_place'].astype(int)]
    print(f"\nSLM Overrides: {len(overrides)}")
    if len(overrides) > 0:
        print(overrides[['id', 'match_score', 'pred_label', 'SLM_same_place', 'SLM_reason']].to_string())

if __name__ == "__main__":
    main()
