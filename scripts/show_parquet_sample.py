
import pandas as pd
import sys
import os

def show_parquet_sample(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    try:
        df = pd.read_parquet(file_path)
        
        # Configure pandas to show all columns and not truncate content too aggressively
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_rows', 20)
        pd.set_option('display.max_colwidth', 50) # Limit column width for readability


        print(f"\n--- FILE INFO: {file_path} ---")
        print(f"Shape: {df.shape}")
        
        print("\n--- COLUMNS ---")
        for col in df.columns:
            print(f"- {col}")
            


        print("\n--- FIRST ROW (Detailed View) ---")
        if not df.empty:
            print(df.iloc[0].to_json(indent=4))
        
        print("\n--- SAMPLE VIEW (First 5 rows, limited columns) ---")
        # limit columns if too many (e.g. > 5)
        cols_to_show = df.columns[:5] if len(df.columns) > 5 else df.columns
        print(df[cols_to_show].head(10))

    except Exception as e:
        print(f"Error reading parquet file: {e}")

if __name__ == "__main__":
    # check for command line arg, otherwise default
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        # Default to the file requested by the user relative to project root
        # Assuming script is run from project root
        target_file = "data/project_a_samples.parquet"
    
    show_parquet_sample(target_file)
