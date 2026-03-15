import pandas as pd
import os

data_dir = "data"
files_to_update = []
for f in os.listdir(data_dir):
    path = os.path.join(data_dir, f)
    if f.endswith('.parquet'):
        try:
            df = pd.read_parquet(path)
            for col in df.columns:
                if df[col].dtype == 'object' or df[col].dtype == 'string':
                    if (df[col] == 'match').any():
                        files_to_update.append((f, col))
        except Exception as e:
            print(f"Error reading {f}: {e}")
    elif f.endswith('.csv'):
        try:
            df = pd.read_csv(path)
            for col in df.columns:
                if df[col].dtype == 'object' or df[col].dtype == 'string':
                    if (df[col] == 'match').any():
                        files_to_update.append((f, col))
        except Exception as e:
            pass

print("Files to update:")
for f, col in set(files_to_update):
    print(f"- {f}: col '{col}'")
