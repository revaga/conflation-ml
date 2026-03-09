import pandas as pd
import json
from parquet_io import read_parquet_safe
import os
import sys

def safe_json_parse(val):
    if pd.isna(val) or not val:
        return {}
    if isinstance(val, dict):
        return val
    try:
        return json.loads(val)
    except:
        return {"raw": str(val)}

def get_primary_name(val):
    data = safe_json_parse(val)
    return data.get('primary', 'N/A')

def get_primary_category(val):
    data = safe_json_parse(val)
    return data.get('primary', 'N/A')

def display_record(row, current_count, total_count):
    print("\n" + "="*90)
    print(f" RECORD {current_count} / {total_count} ")
    print(f" ID: {row['id']} | Base ID: {row['base_id']}")
    print("-" * 90)
    
    # Extracted data
    name_b = get_primary_name(row.get('base_names'))
    name_m = get_primary_name(row.get('names'))
    
    addr_b = row.get('norm_base_addr', 'N/A')
    addr_m = row.get('norm_conflated_addr', 'N/A')
    
    conf_b = row.get('base_confidence', 0)
    conf_m = row.get('confidence', 0)
    
    cat_b = get_primary_category(row.get('base_categories'))
    cat_m = get_primary_category(row.get('categories'))
    
    phone_b = row.get('norm_base_phone', 'N/A')
    phone_m = row.get('norm_conflated_phone', 'N/A')
    
    web_b = row.get('norm_base_website', 'N/A')
    web_m = row.get('norm_conflated_website', 'N/A')

    # Formatting Table
    print(f"{'ATTRIBUTE':<15} | {'BASE RECORD':<35} | {'MATCH CANDIDATE':<35}")
    print("-" * 90)
    print(f"{'Name':<15} | {str(name_b)[:35]:<35} | {str(name_m)[:35]:<35}")
    print(f"{'Address':<15} | {str(addr_b)[:35]:<35} | {str(addr_m)[:35]:<35}")
    print(f"{'Category':<15} | {str(cat_b)[:35]:<35} | {str(cat_m)[:35]:<35}")
    print(f"{'Phone':<15} | {str(phone_b)[:35]:<35} | {str(phone_m)[:35]:<35}")
    print(f"{'Website':<15} | {str(web_b)[:35]:<35} | {str(web_m)[:35]:<35}")
    print(f"{'Confidence':<15} | {conf_b:<35.2f} | {conf_m:<35.2f}")
    
    print("-" * 90)
    print(f"Phase 1 Comparison Score: {row.get('addr_similarity_ratio', 0):.1f}% Address Similarity")
    print("="*90)

def main():
    input_file = 'data/phase1_processed.parquet'
    output_file = 'data/golden_labels.csv'
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Run phase1_data_prep.py first.")
        return

    print("Loading data...")
    df = read_parquet_safe(input_file)
    
    # Load existing labels if any
    if os.path.exists(output_file):
        try:
            labels_df = pd.read_csv(output_file)
            labeled_ids = set(labels_df['id'].tolist())
        except:
            labels_df = pd.DataFrame(columns=['id', 'base_id', 'label', 'notes'])
            labeled_ids = set()
    else:
        labels_df = pd.DataFrame(columns=['id', 'base_id', 'label', 'notes'])
        labeled_ids = set()

    # Filter out already labeled records
    remaining_df = df[~df['id'].isin(labeled_ids)]
    
    total_to_label = len(remaining_df)
    already_labeled = len(labeled_ids)
    
    print(f"\nTotal Dataset: {len(df)}")
    print(f"Already Labeled: {already_labeled}")
    print(f"Remaining: {total_to_label}")

    if total_to_label == 0:
        print("Everything is labeled!")
        return

    # Shuffle for variety
    remaining_df = remaining_df.sample(frac=1, random_state=42)

    new_labels = []
    session_count = 0

    try:
        for _, row in remaining_df.iterrows():
            session_count += 1
            display_record(row, already_labeled + session_count, len(df))
            
            while True:
                choice = input("\n[b]ase better, [m]atch better, [s]kip, [q]uit: ").lower().strip()
                if choice in ['b', 'm', 's', 'q']:
                    break
                print("Invalid choice. Enter b, m, s, or q.")
            
            if choice == 'q':
                break
            if choice == 's':
                print("Skipped.")
                continue
                
            label = 1 if choice == 'm' else 0
            note = input("Why? (Reason/Notes): ").strip()
            
            new_labels.append({
                'id': row['id'],
                'base_id': row['base_id'],
                'label': label,
                'notes': note
            })
            
            # Periodically save
            if len(new_labels) >= 3:
                temp_df = pd.DataFrame(new_labels)
                labels_df = pd.concat([labels_df, temp_df], ignore_index=True)
                labels_df.to_csv(output_file, index=False)
                new_labels = []
                print(f"\n>>> Progress auto-saved to {output_file}")

    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving current session...")

    # Final save
    if new_labels:
        temp_df = pd.DataFrame(new_labels)
        labels_df = pd.concat([labels_df, temp_df], ignore_index=True)
        labels_df.to_csv(output_file, index=False)
        print(f"Saved {len(new_labels)} new entries to {output_file}.")

    print(f"\nFinished session. You have labeled {already_labeled + session_count - (1 if choice=='q' or choice=='s' else 0)} total records.")
    print("Run this script again to continue labeling.")

if __name__ == "__main__":
    main()
