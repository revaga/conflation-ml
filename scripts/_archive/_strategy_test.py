"""Quick test: try different approaches without class balancing."""
import pandas as pd, numpy as np, sys, json
sys.path.insert(0, 'scripts')
from rapidfuzz import fuzz
from datetime import datetime, timezone
from parquet_io import read_parquet_safe

df = read_parquet_safe('data/phase1_processed.parquet')
golden = read_parquet_safe('data/golden_dataset_100.parquet')

def safe_json(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return {}
    if isinstance(x, (dict, list)):
        return x
    try: return json.loads(str(x))
    except: return {}

def extract_primary(val):
    obj = safe_json(val)
    return obj.get("primary", "") if isinstance(obj, dict) else ""

def extract_sources_info(val):
    arr = safe_json(val)
    if not isinstance(arr, list): return 0, set()
    count = len(arr)
    datasets = set()
    for item in arr:
        if not isinstance(item, dict): continue
        ds = item.get("dataset", "")
        if ds: datasets.add(ds.lower())
    return count, datasets

print("Engineering features...")
df["_name"] = df["names"].apply(extract_primary)
df["_base_name"] = df["base_names"].apply(extract_primary)
df["_category"] = df["categories"].apply(extract_primary)
df["_base_category"] = df["base_categories"].apply(extract_primary)

src_info = df["sources"].apply(extract_sources_info)
df["_src_count"] = src_info.apply(lambda x: x[0])
df["_src_datasets"] = src_info.apply(lambda x: x[1])
base_src = df["base_sources"].apply(extract_sources_info)
df["_base_src_count"] = base_src.apply(lambda x: x[0])

df["f_conf_delta"] = df["confidence"] - df["base_confidence"]
df["f_conf"] = df["confidence"]
df["f_base_conf"] = df["base_confidence"]
df["f_addr_sim"] = df["addr_similarity_ratio"] / 100.0
df["f_phone_sim"] = df["phone_similarity"].fillna(0) / 100.0
df["f_web_sim"] = df["website_similarity"].fillna(0) / 100.0
df["f_name_sim"] = df.apply(
    lambda r: fuzz.token_sort_ratio(r["_name"], r["_base_name"]) / 100.0
    if r["_name"] and r["_base_name"] else 0.0, axis=1
)
df["f_cat_sim"] = df.apply(
    lambda r: fuzz.token_sort_ratio(r["_category"], r["_base_category"]) / 100.0
    if r["_category"] and r["_base_category"] else 0.0, axis=1
)
df["f_addr_delta"] = df["norm_conflated_addr"].fillna("").str.len() - df["norm_base_addr"].fillna("").str.len()
df["f_has_phone"] = (df["norm_conflated_phone"].fillna("") != "").astype(int)
df["f_has_base_phone"] = (df["norm_base_phone"].fillna("") != "").astype(int)
df["f_has_web"] = (df["norm_conflated_website"].fillna("") != "").astype(int)
df["f_has_base_web"] = (df["norm_base_website"].fillna("") != "").astype(int)
df["f_phone_delta"] = df["f_has_phone"] - df["f_has_base_phone"]
df["f_web_delta"] = df["f_has_web"] - df["f_has_base_web"]
df["f_src_delta"] = df["_src_count"] - df["_base_src_count"]
df["f_is_msft"] = df["_src_datasets"].apply(lambda s: int("msft" in s) if isinstance(s, set) else 0)
df["f_name_addr_prod"] = df["f_name_sim"] * df["f_addr_sim"]
df["f_avg_sim"] = (df["f_name_sim"] + df["f_addr_sim"] + df["f_phone_sim"] + df["f_web_sim"]) / 4.0
df["f_phone_exact"] = (
    (df["norm_conflated_phone"].fillna("").astype(str) != "")
    & (df["norm_base_phone"].fillna("").astype(str) != "")
    & (df["norm_conflated_phone"].fillna("").astype(str) == df["norm_base_phone"].fillna("").astype(str))
).astype(int)
df["f_cat_exact"] = (
    (df["_category"].fillna("") != "")
    & (df["_base_category"].fillna("") != "")
    & (df["_category"].fillna("") == df["_base_category"].fillna(""))
).astype(int)

ALL_FEATURES = [c for c in df.columns if c.startswith("f_")]

# TOP features only (reduce overfitting)
TOP_FEATURES = [
    "f_conf_delta", "f_addr_sim", "f_addr_delta", "f_name_sim",
    "f_phone_sim", "f_web_sim", "f_avg_sim", "f_name_addr_prod",
    "f_conf", "f_cat_sim",
]

# Labels
label_map = dict(zip(golden['id'], golden['xgboost_testlabels']))
df['_gold'] = df['id'].map(label_map)
df['_binary'] = df['_gold'].map({'match': 1, 'both': 1, 'base': 0})
df['is_golden'] = df['_binary'].notna()

g_df = df[df['is_golden']].copy()

# Heuristic labels
ng = df[~df['is_golden']].copy()
t_name = 0.80; t_addr = 0.78
cond_pos = (
    (ng["f_name_sim"] > t_name) |
    ((ng["f_name_sim"] > t_name - 0.12) & (ng["f_addr_sim"] > t_addr))
) & (
    (ng["f_conf_delta"] > 0) | (ng["f_addr_delta"] > 3) |
    (ng["f_phone_delta"] > 0) | (ng["f_web_delta"] > 0)
)
cond_neg = (
    (ng["f_name_sim"] < t_name - 0.40) |
    ((ng["f_avg_sim"] < 0.45) & (ng["f_conf_delta"] <= 0))
)
ng['_label'] = np.nan
ng.loc[cond_pos, '_label'] = 1.0
ng.loc[cond_neg & ~cond_pos, '_label'] = 0.0
h_df = ng.dropna(subset=['_label'])
X_h = h_df[ALL_FEATURES].fillna(0).values
y_h = h_df['_label'].astype(int).values
print(f"Heuristic: {len(h_df)} rows (pos={int((y_h==1).sum())}, neg={int((y_h==0).sum())})")

# Simple GBC
class Stump:
    def __init__(self):
        self.feat = 0; self.thr = 0; self.lv = 0; self.rv = 0
    def fit(self, X, r, w=None):
        if w is None: w = np.ones(len(r))
        best = np.inf
        for fi in range(X.shape[1]):
            col = X[:, fi]
            for thr in np.unique(np.percentile(col, np.arange(5, 100, 5))):
                lm = col <= thr; rm = ~lm
                if w[lm].sum() < 2 or w[rm].sum() < 2: continue
                lv = np.average(r[lm], weights=w[lm])
                rv = np.average(r[rm], weights=w[rm])
                loss = np.average((r - np.where(lm, lv, rv))**2, weights=w)
                if loss < best:
                    best = loss; self.feat = fi; self.thr = thr; self.lv = lv; self.rv = rv
    def predict(self, X):
        return np.where(X[:, self.feat] <= self.thr, self.lv, self.rv)

class GBC:
    def __init__(self, n_est=100, lr=0.1, sub=0.8):
        self.n_est = n_est; self.lr = lr; self.sub = sub
        self.trees = []; self.init_pred = 0.0
    def fit(self, X, y, w=None):
        n = len(y)
        if w is None: w = np.ones(n)
        p = np.clip(np.average(y, weights=w), 1e-6, 1-1e-6)
        self.init_pred = np.log(p/(1-p))
        raw = np.full(n, self.init_pred)
        rng = np.random.RandomState(42)
        for _ in range(self.n_est):
            probs = 1/(1+np.exp(-np.clip(raw, -500, 500)))
            resid = y - probs
            if self.sub < 1:
                idx = rng.choice(n, max(1, int(n*self.sub)), replace=False)
            else:
                idx = np.arange(n)
            t = Stump()
            t.fit(X[idx], resid[idx], w[idx])
            self.trees.append(t)
            raw += self.lr * t.predict(X)
    def predict_proba(self, X):
        raw = np.full(X.shape[0], self.init_pred)
        for t in self.trees:
            raw += self.lr * t.predict(X)
        return 1/(1+np.exp(-np.clip(raw, -500, 500)))

# Test strategies
print("\n=== Strategy Comparison ===\n")

# Strategy 1: Train on heuristic, test on golden, threshold = 0.5
for feat_set_name, feat_set in [("ALL", ALL_FEATURES), ("TOP10", TOP_FEATURES)]:
    X_g = g_df[feat_set].fillna(0).values
    y_g = g_df['_binary'].astype(int).values
    X_h_fs = h_df[feat_set].fillna(0).values
    
    for n_est, lr in [(80, 0.1), (150, 0.05), (200, 0.03)]:
        model = GBC(n_est, lr, 0.8)
        model.fit(X_h_fs, y_h)
        proba = model.predict_proba(X_g)
        
        best_acc = 0; best_thr = 0.5
        for thr in np.arange(0.3, 0.75, 0.01):
            acc = ((proba >= thr).astype(int) == y_g).mean()
            if acc > best_acc:
                best_acc = acc; best_thr = thr
        
        p05 = (proba >= 0.5).astype(int)
        acc05 = (p05 == y_g).mean()
        
        print(f"  Heuristic->{feat_set_name} n={n_est} lr={lr}: acc@0.5={acc05:.2%}  best_acc={best_acc:.2%}@{best_thr:.2f}")

# Strategy 2: Train on heuristic+golden, 5-fold CV (NO class balance)
print()
for feat_set_name, feat_set in [("ALL", ALL_FEATURES), ("TOP10", TOP_FEATURES)]:
    X_g = g_df[feat_set].fillna(0).values
    y_g = g_df['_binary'].astype(int).values
    X_h_fs = h_df[feat_set].fillna(0).values
    
    for n_est, lr, gw in [(80, 0.1, 5), (80, 0.1, 10), (150, 0.05, 5), (150, 0.05, 10)]:
        rng = np.random.RandomState(42)
        pos_idx = np.where(y_g == 1)[0]; neg_idx = np.where(y_g == 0)[0]
        rng.shuffle(pos_idx); rng.shuffle(neg_idx)
        folds = [[] for _ in range(5)]
        for i, idx in enumerate(pos_idx): folds[i%5].append(idx)
        for i, idx in enumerate(neg_idx): folds[i%5].append(idx)
        
        oof = np.zeros(len(y_g))
        for fold in range(5):
            val_idx = np.array(folds[fold])
            tr_idx = np.concatenate([np.array(folds[f]) for f in range(5) if f != fold])
            X_t = np.vstack([X_h_fs, X_g[tr_idx]])
            y_t = np.concatenate([y_h, y_g[tr_idx]])
            w_t = np.concatenate([np.ones(len(y_h)), np.full(len(tr_idx), gw)])
            m = GBC(n_est, lr, 0.8)
            m.fit(X_t, y_t, w_t)
            oof[val_idx] = m.predict_proba(X_g[val_idx])
        
        best_acc = 0; best_thr = 0.5
        for thr in np.arange(0.3, 0.75, 0.01):
            acc = ((oof >= thr).astype(int) == y_g).mean()
            if acc > best_acc:
                best_acc = acc; best_thr = thr
        
        acc05 = ((oof >= 0.5).astype(int) == y_g).mean()
        print(f"  Blend+CV {feat_set_name} n={n_est} lr={lr} gw={gw}: acc@0.5={acc05:.2%}  best_acc={best_acc:.2%}@{best_thr:.2f}")

# Strategy 3: Simple logistic regression (manual gradient descent)
print("\n--- Logistic Regression (golden-only, 5-fold CV) ---")
for feat_set_name, feat_set in [("ALL", ALL_FEATURES), ("TOP10", TOP_FEATURES)]:
    X_g = g_df[feat_set].fillna(0).values
    y_g = g_df['_binary'].astype(int).values
    
    # Standardize
    mu = X_g.mean(axis=0); sd = X_g.std(axis=0) + 1e-8
    X_g_std = (X_g - mu) / sd
    
    rng = np.random.RandomState(42)
    pos_idx = np.where(y_g == 1)[0]; neg_idx = np.where(y_g == 0)[0]
    rng.shuffle(pos_idx); rng.shuffle(neg_idx)
    folds = [[] for _ in range(5)]
    for i, idx in enumerate(pos_idx): folds[i%5].append(idx)
    for i, idx in enumerate(neg_idx): folds[i%5].append(idx)
    
    oof = np.zeros(len(y_g))
    for fold in range(5):
        val_idx = np.array(folds[fold])
        tr_idx = np.concatenate([np.array(folds[f]) for f in range(5) if f != fold])
        Xt = X_g_std[tr_idx]; yt = y_g[tr_idx]
        
        # L2 regularized logistic regression
        n_feat = Xt.shape[1]
        w = np.zeros(n_feat); b = 0.0
        lr_val = 0.01; lam = 1.0
        for epoch in range(500):
            z = Xt @ w + b
            p = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            grad_w = Xt.T @ (p - yt) / len(yt) + lam * w
            grad_b = (p - yt).mean()
            w -= lr_val * grad_w
            b -= lr_val * grad_b
        
        Xv = X_g_std[val_idx]
        z = Xv @ w + b
        oof[val_idx] = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    best_acc = 0; best_thr = 0.5
    for thr in np.arange(0.3, 0.75, 0.01):
        acc = ((oof >= thr).astype(int) == y_g).mean()
        if acc > best_acc:
            best_acc = acc; best_thr = thr
    acc05 = ((oof >= 0.5).astype(int) == y_g).mean()
    print(f"  LogReg {feat_set_name} L2=1.0: acc@0.5={acc05:.2%}  best_acc={best_acc:.2%}@{best_thr:.2f}")

# Strategy 4: k-NN
print("\n--- k-NN (golden-only, 5-fold CV) ---")
for feat_set_name, feat_set in [("TOP10", TOP_FEATURES)]:
    X_g = g_df[feat_set].fillna(0).values
    y_g = g_df['_binary'].astype(int).values
    mu = X_g.mean(axis=0); sd = X_g.std(axis=0) + 1e-8
    X_g_std = (X_g - mu) / sd
    
    rng = np.random.RandomState(42)
    pos_idx = np.where(y_g == 1)[0]; neg_idx = np.where(y_g == 0)[0]
    rng.shuffle(pos_idx); rng.shuffle(neg_idx)
    folds = [[] for _ in range(5)]
    for i, idx in enumerate(pos_idx): folds[i%5].append(idx)
    for i, idx in enumerate(neg_idx): folds[i%5].append(idx)
    
    for k in [3, 5, 7, 11, 15]:
        oof = np.zeros(len(y_g))
        for fold in range(5):
            val_idx = np.array(folds[fold])
            tr_idx = np.concatenate([np.array(folds[f]) for f in range(5) if f != fold])
            for vi in val_idx:
                dists = np.sqrt(((X_g_std[tr_idx] - X_g_std[vi])**2).sum(axis=1))
                nn = tr_idx[np.argsort(dists)[:k]]
                oof[vi] = y_g[nn].mean()
        
        best_acc = 0; best_thr = 0.5
        for thr in np.arange(0.3, 0.75, 0.01):
            acc = ((oof >= thr).astype(int) == y_g).mean()
            if acc > best_acc:
                best_acc = acc; best_thr = thr
        acc05 = ((oof >= 0.5).astype(int) == y_g).mean()
        print(f"  k-NN k={k}: acc@0.5={acc05:.2%}  best_acc={best_acc:.2%}@{best_thr:.2f}")

print("\nDone!")
