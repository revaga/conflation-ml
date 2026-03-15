"""Analyze feature discriminative power on golden-100."""
import pandas as pd, numpy as np, sys
sys.path.insert(0, 'scripts')
from parquet_io import read_parquet_safe

df = read_parquet_safe('data/phase1_processed.parquet')
golden = read_parquet_safe('data/golden_dataset_100.parquet')
g = df[df['id'].isin(golden['id'])].copy()
label_map = dict(zip(golden['id'], golden['xgboost_testlabels']))
g['truth'] = g['id'].map(label_map)
g['binary'] = g['truth'].map({'match': 1, 'both': 1, 'base': 0})
y = g['binary'].values

feats = {
    'conf_delta': (g['confidence'] - g['base_confidence']).fillna(0).values,
    'confidence': g['confidence'].fillna(0).values,
    'base_confidence': g['base_confidence'].fillna(0).values,
    'addr_sim': g['addr_similarity_ratio'].fillna(0).values,
    'phone_sim': g['phone_similarity'].fillna(0).values,
    'web_sim': g['website_similarity'].fillna(0).values,
    'addr_len': g['norm_conflated_addr'].fillna('').str.len().values,
    'base_addr_len': g['norm_base_addr'].fillna('').str.len().values,
    'addr_len_delta': (g['norm_conflated_addr'].fillna('').str.len() - g['norm_base_addr'].fillna('').str.len()).values,
    'has_phone': (g['norm_conflated_phone'].fillna('') != '').astype(int).values,
    'has_base_phone': (g['norm_base_phone'].fillna('') != '').astype(int).values,
    'has_web': (g['norm_conflated_website'].fillna('') != '').astype(int).values,
    'has_base_web': (g['norm_base_website'].fillna('') != '').astype(int).values,
}

results = []
for name, v in feats.items():
    corr = np.corrcoef(v, y)[0, 1] if v.std() > 0 else 0
    accs = []
    for pct in range(5, 100, 5):
        t = np.percentile(v, pct)
        a1 = ((v >= t).astype(int) == y).mean()
        a2 = ((v < t).astype(int) == y).mean()
        accs.append(max(a1, a2))
    best = max(accs)
    results.append((name, corr, best, v[y==1].mean(), v[y==0].mean()))

results.sort(key=lambda x: -x[2])
with open('data/feature_analysis.txt', 'w') as f:
    f.write(f"{'Feature':<20} {'Corr':>6} {'BestAcc':>8} {'PosMean':>8} {'NegMean':>8}\n")
    f.write("-"*55 + "\n")
    for name, corr, best, pm, nm in results:
        f.write(f"{name:<20} {corr:>6.3f} {best:>8.3f} {pm:>8.3f} {nm:>8.3f}\n")

# Also check 3-class separability
print("3-class analysis:")
for label in ['match', 'both', 'base']:
    sub = g[g['truth'] == label]
    print(f"  {label} (n={len(sub)}):")
    print(f"    addr_sim: {sub['addr_similarity_ratio'].mean():.1f}")
    print(f"    phone_sim: {sub['phone_similarity'].mean():.1f}")
    print(f"    web_sim: {sub['website_similarity'].mean():.1f}")
    print(f"    conf_delta: {(sub['confidence'] - sub['base_confidence']).mean():.3f}")
    print(f"    addr_len_delta: {(sub['norm_conflated_addr'].fillna('').str.len() - sub['norm_base_addr'].fillna('').str.len()).mean():.1f}")

print("\nFeature analysis saved to data/feature_analysis.txt")
