"""G28 (Gold Pass 1/3) — fusion + learned-ranker layer (PulseDiscover V2).

Heuristic fusion policies and learned ranker wrappers over the G28 candidate feature table
(ALS + semantic + popularity candidates). Learned models score candidates; we rerank per user
and evaluate ranking metrics by cohort / gold-tier. NOT a claim that learned beats ALS unless
test metrics prove it; NOT online; served-c2 protocol only.
"""
from __future__ import annotations
import numpy as np

FEATURES = ["from_als", "from_semantic", "from_popularity", "als_score", "als_rank",
            "sem_score", "sem_rank", "pop_rank_pct", "is_head", "is_mid", "is_long_tail",
            "is_coldstart", "in_semantic_index", "n_sources"]


# ---------------- heuristic fusion (return ordered item_ids per user group) ----------------
def _order_by(df_u, key, asc=False):
    return df_u.sort_values(key, ascending=asc)["item_id"].tolist()


def fuse_als_only(df_u):
    a = df_u[df_u.from_als == 1]
    return a.sort_values("als_rank")["item_id"].tolist()


def fuse_semantic_only(df_u):
    s = df_u[df_u.from_semantic == 1]
    return s.sort_values("sem_rank")["item_id"].tolist()


def fuse_popularity_only(df_u):
    p = df_u[df_u.from_popularity == 1]
    return p.sort_values("pop_rank_pct", ascending=False)["item_id"].tolist()


def fuse_als_first(df_u, k=20):
    als = fuse_als_only(df_u); sem = fuse_semantic_only(df_u)
    out = list(als)
    for b in sem:
        if b not in set(out):
            out.append(b)
    return out


def fuse_semantic_fallback(df_u, k=20, thin=10):
    """ALS primary; only inject semantic if ALS pool is thin (< thin candidates)."""
    als = fuse_als_only(df_u)
    if len(als) >= thin:
        out = list(als)
        for b in fuse_semantic_only(df_u):
            if b not in set(out):
                out.append(b)
        return out
    # thin ALS -> prioritize semantic to recover coverage/cold-start
    sem = fuse_semantic_only(df_u); out = list(als)
    for b in sem:
        if b not in set(out):
            out.append(b)
    return out


def fuse_source_balanced(df_u, k=20):
    """Interleave ALS and semantic ranks (round-robin) to balance relevance + coverage."""
    als = fuse_als_only(df_u); sem = fuse_semantic_only(df_u)
    out, seen, i = [], set(), 0
    while (i < len(als) or i < len(sem)) and len(out) < max(k, 50):
        if i < len(als) and als[i] not in seen:
            out.append(als[i]); seen.add(als[i])
        if i < len(sem) and sem[i] not in seen:
            out.append(sem[i]); seen.add(sem[i])
        i += 1
    return out


def fuse_rrf(df_u, k=20, c=60):
    """Reciprocal Rank Fusion over ALS + semantic ranks."""
    score = {}
    for b, r in zip(df_u.item_id, df_u.als_rank):
        if r > 0:
            score[b] = score.get(b, 0.0) + 1.0 / (c + r)
    for b, r in zip(df_u.item_id, df_u.sem_rank):
        if r > 0:
            score[b] = score.get(b, 0.0) + 1.0 / (c + r)
    return [b for b, _ in sorted(score.items(), key=lambda kv: -kv[1])]


def fuse_weighted_score(df_u, k=20, w_als=0.7, w_sem=0.3):
    """Weighted fusion of min-max-normalized ALS + semantic scores."""
    d = df_u.copy()
    def nz(x):
        x = x.astype(float); r = x.max() - x.min()
        return (x - x.min()) / r if r > 1e-9 else x * 0
    d["_a"] = np.where(d.from_als == 1, nz(d.als_score), 0.0)
    d["_s"] = np.where(d.from_semantic == 1, nz(d.sem_score), 0.0)
    d["_f"] = w_als * d["_a"] + w_sem * d["_s"]
    return d.sort_values("_f", ascending=False)["item_id"].tolist()


HEURISTICS = {
    "als_only": fuse_als_only, "semantic_only": fuse_semantic_only, "popularity_only": fuse_popularity_only,
    "als_first": fuse_als_first, "semantic_fallback": fuse_semantic_fallback,
    "source_balanced": fuse_source_balanced, "rrf": fuse_rrf, "weighted_score": fuse_weighted_score,
}


# ---------------- learned models ----------------
def train_lightgbm_ranker(Xtr, ytr, grp_tr, Xva, yva, grp_va, seed=42):
    import lightgbm as lgb
    dtr = lgb.Dataset(Xtr, label=ytr, group=grp_tr)
    dva = lgb.Dataset(Xva, label=yva, group=grp_va, reference=dtr)
    params = {"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [20],
              "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 50,
              "seed": seed, "verbose": -1}
    m = lgb.train(params, dtr, num_boost_round=300, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
    return m, ("lambdarank", m.num_trees())


def downsample_negatives(X, y, ratio=30, seed=42):
    """Keep all positives + `ratio` negatives per positive (handles extreme imbalance + speed)."""
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    keep_neg = rng.choice(neg, min(len(neg), max(len(pos) * ratio, 2000)), replace=False)
    idx = np.concatenate([pos, keep_neg]); rng.shuffle(idx)
    return X[idx], y[idx]


def train_sklearn(model_name, Xtr, ytr, seed=42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    Xs, ys = downsample_negatives(Xtr, ytr, seed=seed)  # speed + imbalance
    if model_name == "logistic":
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    elif model_name == "random_forest":
        m = RandomForestClassifier(n_estimators=80, max_depth=8, class_weight="balanced",
                                   random_state=seed, n_jobs=-1)
    elif model_name == "sklearn_gbm":
        m = HistGradientBoostingClassifier(max_iter=150, max_depth=6, random_state=seed)
    else:
        raise ValueError(model_name)
    m.fit(Xs, ys)
    return m


def model_scores(m, kind, Xte):
    if kind == "ranker":
        return m.predict(Xte)
    if hasattr(m, "predict_proba"):
        return m.predict_proba(Xte)[:, 1]
    return m.decision_function(Xte)
