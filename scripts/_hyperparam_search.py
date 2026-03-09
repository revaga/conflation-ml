"""Quick hyperparameter search for best golden-100 accuracy."""
import pandas as pd, numpy as np, sys, json
sys.path.insert(0, 'scripts')
from rapidfuzz import fuzz
from datetime import datetime, timezone
from parquet_io import read_parquet_safe

# Load and feature-engineer
df = read_parquet_safe('data/phase1_processed.parquet')
golden = read_parquet_safe('data/golden_dataset_100.parquet')

def safe_json(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return {}
    if isinstance(x, (dict, list)):
        return x
    try:
        return json.loads(str(x))
    except:
        return {}

def extract_primary(val):
    obj = safe_json(val)
    return obj.get("primary", "") if isinstance(obj, dict) else ""

def extract_sources_info(val):
    arr = safe_json(val)
    if not isinstance(arr, list):
        return 0, None, set()
    count = len(arr)
    datasets = set()
    latest_dt = None
    for item in arr:
        if not isinstance(item, dict): continue
        ds = item.get("dataset", "")
        if ds: datasets.add(ds.lower())
        ut = item.get("update_time")
        if ut:
            try:
                dt = datetime.fromisoformat(ut.replace("Z", "+00:00"))
                if latest_dt is None or dt > latest_dt: latest_dt = dt
            except: pass
    return count, latest_dt, datasets

print("Engineering features...")
df["_name"] = df["names"].apply(extract_primary)
df["_base_name"] = df["base_names"].apply(extract_primary)
df["_category"] = df["categories"].apply(extract_primary)
df["_base_category"] = df["base_categories"].apply(extract_primary)

src_info = df["sources"].apply(extract_sources_info)
df["_src_count"] = src_info.apply(lambda x: x[0])
df["_src_datasets"] = src_info.apply(lambda x: x[2])

base_src_info = df["base_sources"].apply(extract_sources_info)
df["_base_src_count"] = base_src_info.apply(lambda x: x[0])

# Features (no web/phone validation to keep it fast)
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
df["f_addr_len"] = df["norm_conflated_addr"].fillna("").str.len()
df["f_base_addr_len"] = df["norm_base_addr"].fillna("").str.len()
df["f_addr_delta"] = df["f_addr_len"] - df["f_base_addr_len"]
df["f_has_phone"] = (df["norm_conflated_phone"].fillna("") != "").astype(int)
df["f_has_base_phone"] = (df["norm_base_phone"].fillna("") != "").astype(int)
df["f_has_web"] = (df["norm_conflated_website"].fillna("") != "").astype(int)
df["f_has_base_web"] = (df["norm_base_website"].fillna("") != "").astype(int)
df["f_phone_delta"] = df["f_has_phone"] - df["f_has_base_phone"]
df["f_web_delta"] = df["f_has_web"] - df["f_has_base_web"]
df["f_src_delta"] = df["_src_count"] - df["_base_src_count"]
df["f_is_msft"] = df["_src_datasets"].apply(lambda s: int("msft" in s) if isinstance(s, set) else 0)
df["f_is_meta"] = df["_src_datasets"].apply(lambda s: int("meta" in s) if isinstance(s, set) else 0)
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

# Completeness
df["f_completeness"] = df["f_has_phone"] + df["f_has_web"] + (df["f_addr_len"] > 0).astype(int)
df["f_base_completeness"] = df["f_has_base_phone"] + df["f_has_base_web"] + (df["f_base_addr_len"] > 0).astype(int)
df["f_completeness_delta"] = df["f_completeness"] - df["f_base_completeness"]

FEATURE_COLS = [c for c in df.columns if c.startswith("f_")]
print(f"Features: {len(FEATURE_COLS)}")

# Labels
label_map = dict(zip(golden['id'], golden['xgboost_testlabels']))
df['_gold_label'] = df['id'].map(label_map)
df['_binary'] = df['_gold_label'].map({'match': 1, 'both': 1, 'base': 0})
df['is_golden'] = df['_binary'].notna()

# Get golden and heuristic sets
golden_mask = df['is_golden']
g_df = df[golden_mask].copy()
X_golden = g_df[FEATURE_COLS].fillna(0).values
y_golden = g_df['_binary'].astype(int).values

# Simple Decision Tree implementation for speed
class Stump:
    def __init__(self):
        self.feat = 0
        self.thr = 0.0
        self.lv = 0.0
        self.rv = 0.0
    def fit(self, X, r, w=None):
        if w is None: w = np.ones(len(r))
        best = np.inf
        for fi in range(X.shape[1]):
            col = X[:, fi]
            for thr in np.unique(np.percentile(col, np.arange(10, 100, 10))):
                lm = col <= thr; rm = ~lm
                if w[lm].sum() < 2 or w[rm].sum() < 2: continue
                lv = np.average(r[lm], weights=w[lm])
                rv = np.average(r[rm], weights=w[rm])
                loss = np.average((r - np.where(lm, lv, rv))**2, weights=w)
                if loss < best:
                    best = loss; self.feat = fi; self.thr = thr; self.lv = lv; self.rv = rv
    def predict(self, X):
        return np.where(X[:, self.feat] <= self.thr, self.lv, self.rv)

class DTree:
    def __init__(self, max_depth=2, min_leaf=3):
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.tree = None
    def _split(self, X, r, w, depth):
        node = {"value": np.average(r, weights=w) if w.sum() > 0 else 0.0}
        if depth >= self.max_depth or len(r) < 2 * self.min_leaf:
            return node
        best = np.inf; bf = None; bt = None
        for fi in range(X.shape[1]):
            col = X[:, fi]
            uv = np.unique(col)
            thrs = uv if len(uv) <= 15 else np.unique(np.percentile(col, np.arange(10, 100, 10)))
            for thr in thrs:
                lm = col <= thr; rm = ~lm
                if w[lm].sum() < self.min_leaf or w[rm].sum() < self.min_leaf: continue
                lv = np.average(r[lm], weights=w[lm])
                rv = np.average(r[rm], weights=w[rm])
                loss = np.average((r - np.where(lm, lv, rv))**2, weights=w)
                if loss < best:
                    best = loss; bf = fi; bt = thr
        if bf is None:
            return node
        lm = X[:, bf] <= bt
        node["f"] = bf; node["t"] = bt
        node["l"] = self._split(X[lm], r[lm], w[lm], depth+1)
        node["r"] = self._split(X[~lm], r[~lm], w[~lm], depth+1)
        return node
    def fit(self, X, r, w=None):
        if w is None: w = np.ones(len(r))
        self.tree = self._split(X, r, w, 0)
    def _pred1(self, n, x):
        if "f" not in n: return n["value"]
        return self._pred1(n["l"] if x[n["f"]] <= n["t"] else n["r"], x)
    def predict(self, X):
        return np.array([self._pred1(self.tree, x) for x in X])

class GBC:
    def __init__(self, n_est=100, lr=0.1, max_depth=2, min_leaf=3, sub=0.8, tree_type='dtree'):
        self.n_est = n_est; self.lr = lr; self.max_depth = max_depth
        self.min_leaf = min_leaf; self.sub = sub; self.tree_type = tree_type
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
            if self.tree_type == 'stump':
                t = Stump()
            else:
                t = DTree(max_depth=self.max_depth, min_leaf=self.min_leaf)
            t.fit(X[idx], resid[idx], w[idx])
            self.trees.append(t)
            raw += self.lr * t.predict(X)
    def predict_proba(self, X):
        raw = np.full(X.shape[0], self.init_pred)
        for t in self.trees:
            raw += self.lr * t.predict(X)
        return 1/(1+np.exp(-np.clip(raw, -500, 500)))

# Try different configurations with stratified 5-fold CV
def stratified_kfold_cv(X, y, model_fn, n_folds=5, seed=42):
    """Returns (best_threshold, best_balanced_acc, best_raw_acc, oof_probas)."""
    rng = np.random.RandomState(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    
    oof = np.zeros(len(y))
    folds = [[] for _ in range(n_folds)]
    for i, idx in enumerate(pos_idx):
        folds[i % n_folds].append(idx)
    for i, idx in enumerate(neg_idx):
        folds[i % n_folds].append(idx)
    
    for fold in range(n_folds):
        val_idx = np.array(folds[fold])
        train_idx = np.concatenate([np.array(folds[f]) for f in range(n_folds) if f != fold])
        model = model_fn()
        # Class balanced weights
        w = np.ones(len(train_idx))
        yt = y[train_idx]
        np_t = (yt == 1).sum()
        nn_t = (yt == 0).sum()
        if np_t > 0 and nn_t > 0:
            w[yt == 1] = len(yt) / (2 * np_t)
            w[yt == 0] = len(yt) / (2 * nn_t)
        model.fit(X[train_idx], yt, w)
        oof[val_idx] = model.predict_proba(X[val_idx])
    
    best_thr = 0.5; best_bal = 0; best_raw = 0
    for thr in np.arange(0.2, 0.8, 0.01):
        p = (oof >= thr).astype(int)
        tpr = ((p==1)&(y==1)).sum() / max(1,(y==1).sum())
        tnr = ((p==0)&(y==0)).sum() / max(1,(y==0).sum())
        bal = (tpr+tnr)/2
        raw = (p==y).mean()
        if bal > best_bal:
            best_bal = bal; best_thr = thr; best_raw = raw
    return best_thr, best_bal, best_raw, oof

print("\n=== Hyperparameter Search (Golden-100 only, stratified 5-fold CV) ===")
print(f"{'Config':<45s} {'BalAcc':>7} {'RawAcc':>7} {'Thr':>5}")
print("-"*70)

configs = [
    ("stump_100_lr01", lambda: GBC(100, 0.1, 1, 3, 0.8, 'stump')),
    ("stump_200_lr005", lambda: GBC(200, 0.05, 1, 3, 0.8, 'stump')),
    ("stump_150_lr008", lambda: GBC(150, 0.08, 1, 3, 0.8, 'stump')),
    ("depth2_80_lr01", lambda: GBC(80, 0.1, 2, 3, 0.8)),
    ("depth2_120_lr005", lambda: GBC(120, 0.05, 2, 3, 0.8)),
    ("depth2_150_lr005", lambda: GBC(150, 0.05, 2, 3, 0.8)),
    ("depth2_200_lr003", lambda: GBC(200, 0.03, 2, 3, 0.8)),
    ("depth3_80_lr005", lambda: GBC(80, 0.05, 3, 3, 0.8)),
    ("depth3_120_lr005", lambda: GBC(120, 0.05, 3, 3, 0.8)),
    ("depth3_80_lr01", lambda: GBC(80, 0.1, 3, 3, 0.8)),
    ("depth2_120_lr005_sub1", lambda: GBC(120, 0.05, 2, 3, 1.0)),
    ("depth2_80_lr01_leaf5", lambda: GBC(80, 0.1, 2, 5, 0.8)),
    ("depth1_300_lr005", lambda: GBC(300, 0.05, 1, 3, 0.8, 'dtree')),
    ("depth2_200_lr005_sub1", lambda: GBC(200, 0.05, 2, 3, 1.0)),
]

best_overall = None
for name, model_fn in configs:
    thr, bal, raw, oof = stratified_kfold_cv(X_golden, y_golden, model_fn)
    print(f"  {name:<43s} {bal:>7.4f} {raw:>7.4f} {thr:>5.2f}")
    if best_overall is None or bal > best_overall[1]:
        best_overall = (name, bal, raw, thr, oof)

print(f"\nBest: {best_overall[0]} bal_acc={best_overall[1]:.4f} raw_acc={best_overall[2]:.4f} thr={best_overall[3]:.2f}")

# Now try with blended heuristic data
print("\n=== With Heuristic Data Blended In ===")

# Heuristic labels for non-golden rows
ng_df = df[~df['is_golden']].copy()
# Simple heuristics
t_name = 0.80
t_addr = 0.78
cond_pos = (
    (ng_df["f_name_sim"] > t_name) |
    ((ng_df["f_name_sim"] > t_name - 0.12) & (ng_df["f_addr_sim"] > t_addr))
) & (
    (ng_df["f_conf_delta"] > 0) |
    (ng_df["f_completeness_delta"] > 0) |
    (ng_df["f_addr_delta"] > 3)
)
cond_neg = (
    (ng_df["f_name_sim"] < t_name - 0.40) |
    ((ng_df["f_avg_sim"] < 0.45) & (ng_df["f_conf_delta"] <= 0))
)
ng_df['_label'] = np.nan
ng_df.loc[cond_pos, '_label'] = 1.0
ng_df.loc[cond_neg & ~cond_pos, '_label'] = 0.0
h_df = ng_df.dropna(subset=['_label'])
X_heuristic = h_df[FEATURE_COLS].fillna(0).values
y_heuristic = h_df['_label'].astype(int).values
print(f"Heuristic training rows: {len(h_df)} (pos={int((y_heuristic==1).sum())}, neg={int((y_heuristic==0).sum())})")

def blended_cv(X_g, y_g, X_h, y_h, model_fn, gw=5.0, n_folds=5, seed=42):
    rng = np.random.RandomState(seed)
    pos_idx = np.where(y_g == 1)[0]
    neg_idx = np.where(y_g == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    
    oof = np.zeros(len(y_g))
    folds = [[] for _ in range(n_folds)]
    for i, idx in enumerate(pos_idx):
        folds[i % n_folds].append(idx)
    for i, idx in enumerate(neg_idx):
        folds[i % n_folds].append(idx)
    
    # Heuristic weights (class balanced)
    hw = np.ones(len(y_h))
    np_h = (y_h == 1).sum()
    nn_h = (y_h == 0).sum()
    if np_h > 0 and nn_h > 0:
        hw[y_h == 1] = len(y_h) / (2 * np_h)
        hw[y_h == 0] = len(y_h) / (2 * nn_h)
    
    for fold in range(n_folds):
        val_idx = np.array(folds[fold])
        train_g_idx = np.concatenate([np.array(folds[f]) for f in range(n_folds) if f != fold])
        
        X_t = np.vstack([X_h, X_g[train_g_idx]])
        y_t = np.concatenate([y_h, y_g[train_g_idx]])
        
        # Golden weights (class balanced, weighted higher)
        gw_arr = np.full(len(train_g_idx), gw)
        yt_g = y_g[train_g_idx]
        np_g = (yt_g == 1).sum()
        nn_g = (yt_g == 0).sum()
        if np_g > 0 and nn_g > 0:
            gw_arr[yt_g == 1] = len(yt_g) / (2 * np_g) * gw
            gw_arr[yt_g == 0] = len(yt_g) / (2 * nn_g) * gw
        
        w_t = np.concatenate([hw, gw_arr])
        
        model = model_fn()
        model.fit(X_t, y_t, w_t)
        oof[val_idx] = model.predict_proba(X_g[val_idx])
    
    best_thr = 0.5; best_bal = 0; best_raw = 0
    for thr in np.arange(0.2, 0.8, 0.01):
        p = (oof >= thr).astype(int)
        tpr = ((p==1)&(y_g==1)).sum() / max(1,(y_g==1).sum())
        tnr = ((p==0)&(y_g==0)).sum() / max(1,(y_g==0).sum())
        bal = (tpr+tnr)/2
        raw = (p==y_g).mean()
        if bal > best_bal:
            best_bal = bal; best_thr = thr; best_raw = raw
    return best_thr, best_bal, best_raw, oof

blend_configs = [
    ("blend_stump100_lr01_gw3", lambda: GBC(100, 0.1, 1, 3, 0.8, 'stump'), 3.0),
    ("blend_stump100_lr01_gw5", lambda: GBC(100, 0.1, 1, 3, 0.8, 'stump'), 5.0),
    ("blend_stump100_lr01_gw10", lambda: GBC(100, 0.1, 1, 3, 0.8, 'stump'), 10.0),
    ("blend_d2_120_lr005_gw3", lambda: GBC(120, 0.05, 2, 3, 0.8), 3.0),
    ("blend_d2_120_lr005_gw5", lambda: GBC(120, 0.05, 2, 3, 0.8), 5.0),
    ("blend_d2_120_lr005_gw10", lambda: GBC(120, 0.05, 2, 3, 0.8), 10.0),
    ("blend_d2_80_lr01_gw5", lambda: GBC(80, 0.1, 2, 3, 0.8), 5.0),
    ("blend_d2_200_lr003_gw5", lambda: GBC(200, 0.03, 2, 3, 0.8), 5.0),
]

print(f"\n{'Config':<45s} {'BalAcc':>7} {'RawAcc':>7} {'Thr':>5}")
print("-"*70)

best_blend = None
for name, model_fn, gw in blend_configs:
    thr, bal, raw, oof = blended_cv(X_golden, y_golden, X_heuristic, y_heuristic, model_fn, gw)
    print(f"  {name:<43s} {bal:>7.4f} {raw:>7.4f} {thr:>5.2f}")
    if best_blend is None or bal > best_blend[1]:
        best_blend = (name, bal, raw, thr, oof)

print(f"\nBest blended: {best_blend[0]} bal_acc={best_blend[1]:.4f} raw_acc={best_blend[2]:.4f} thr={best_blend[3]:.2f}")
