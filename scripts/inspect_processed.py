"""
Inspect the processed data from phase 1.
Run from project root: python scripts/inspect_processed.py
"""
import duckdb
from pathlib import Path

# Use path relative to script location so it works from any cwd
DATA_PATH = Path(__file__).parent.parent / "data" / "phase1_processed.parquet"

con = duckdb.connect()

print("=" * 60)
print("PHASE 1 PROCESSED DATA — Inspection")
print("=" * 60)

# --- Schema ---
print("\nSCHEMA (columns & types)")
print("-" * 40)
schema = con.execute(f"DESCRIBE SELECT * FROM '{DATA_PATH}'").fetchdf()
for _, row in schema.iterrows():
    print(f"  {row['column_name']:<25} {row['column_type']}")

# --- Row count ---
row_count = con.execute(f"SELECT COUNT(*) FROM '{DATA_PATH}'").fetchone()[0]
print(f"\nROW COUNT: {row_count:,}")

# --- Similarity Distribution ---
print("\nSIMILARITY SCORES (Average)")
print("-" * 40)
avg_scores = con.execute(f"""
    SELECT 
        AVG(addr_similarity_ratio) as avg_addr,
        AVG(phone_similarity) as avg_phone,
        AVG(website_similarity) as avg_website
    FROM '{DATA_PATH}'
""").fetchone()
print(f"  Address Similarity: {avg_scores[0]:.2f}")
print(f"  Phone Similarity:   {avg_scores[1]:.2f}")
print(f"  Website Similarity: {avg_scores[2]:.2f}")

# --- Distribution of Address Similarity ---
print("\nADDRESS SIMILARITY DISTRIBUTION")
print("-" * 40)
hist = con.execute(f"""
    SELECT 
        FLOOR(addr_similarity_ratio / 10) * 10 as bucket,
        COUNT(*) as count
    FROM '{DATA_PATH}'
    GROUP BY bucket
    ORDER BY bucket DESC
""").fetchdf()
print(hist.to_string(index=False))

# --- Sample High/Low Similarity ---
print("\nSAMPLE: High Address Similarity (> 90)")
print("-" * 40)
high_sim = con.execute(f"""
    SELECT id, norm_conflated_addr, norm_base_addr, addr_similarity_ratio
    FROM '{DATA_PATH}'
    WHERE addr_similarity_ratio > 90
    LIMIT 3
""").fetchdf()
print(high_sim.to_string(index=False))

print("\nSAMPLE: Low Address Similarity (< 50)")
print("-" * 40)
low_sim = con.execute(f"""
    SELECT id, norm_conflated_addr, norm_base_addr, addr_similarity_ratio
    FROM '{DATA_PATH}'
    WHERE addr_similarity_ratio < 50
    LIMIT 3
""").fetchdf()
print(low_sim.to_string(index=False))

print("\n" + "=" * 60)
con.close()
