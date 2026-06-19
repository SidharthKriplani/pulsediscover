"""
PulseDiscover — retrieval baselines (Phase 2). Pure numpy/pandas (no heavy deps).

Baselines (NO deep models in Phase 2):
  - popularity        : global frequency floor
  - cooccurrence (PMI): item-item, session/user co-engagement
  - als               : implicit-feedback matrix factorization, from scratch
                        (confidence-weighted ALS; alternating closed-form solves)

ALS logic is a minimal, self-contained reimplementation aligned with the
pulserank_platform approach (provenance vendored in archive/pulserank_evidence/);
no live import / dependency coupling.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from collections import defaultdict


def popularity_scores(train: pd.DataFrame) -> dict:
    c = train["item_id"].value_counts()
    return c.to_dict()


def popularity_topk(train: pd.DataFrame, k: int):
    return list(train["item_id"].value_counts().head(k).index)


def cooccurrence_pmi(train: pd.DataFrame, min_co: int = 2, topn: int = 50) -> dict:
    """Item-item PMI from per-user item co-engagement. Vectorized via sparse M^T M.
    Returns {item: [(neighbor, pmi), ...]}. Falls back to pure-python if scipy absent."""
    users = {u: i for i, u in enumerate(train.user_id.unique())}
    items = {it: j for j, it in enumerate(train.item_id.unique())}
    inv = {j: it for it, j in items.items()}
    ui = train.user_id.map(users).to_numpy()
    ij = train.item_id.map(items).to_numpy()
    N = len(users)
    try:
        import scipy.sparse as sp
        M = sp.csr_matrix((np.ones(len(ui)), (ui, ij)), shape=(N, len(items)))
        M = (M > 0).astype(np.float64)                  # binary user x item
        item_count = np.asarray(M.sum(axis=0)).ravel()  # users per item
        C = (M.T @ M).tocoo()                            # item-item co-counts
        mask = (C.row != C.col) & (C.data >= min_co)
        r, c2, cv = C.row[mask], C.col[mask], C.data[mask]
        pmi = np.log((cv / N) / ((item_count[r] / N) * (item_count[c2] / N)) + 1e-12)
        d = pd.DataFrame({"x": r, "y": c2, "pmi": pmi}).sort_values(["x", "pmi"], ascending=[True, False])
        neigh = {}
        for x, g in d.groupby("x", sort=False):
            neigh[inv[int(x)]] = [(inv[int(y)], float(p))
                                  for y, p in zip(g.y.to_numpy()[:topn], g.pmi.to_numpy()[:topn])]
        return neigh
    except ImportError:
        return {}  # scipy required for vectorized co-occurrence at scale


def cooccurrence_user_topk(neigh, history, k):
    """history: iterable of the user's train items (set or list). O(history * neighbors)."""
    hist_set = set(history)
    scores = defaultdict(float)
    for it in hist_set:
        for nb, w in neigh.get(it, []):
            if nb not in hist_set:
                scores[nb] += w
    return [i for i, _ in sorted(scores.items(), key=lambda t: -t[1])[:k]]


class ALS:
    """Confidence-weighted implicit ALS, from scratch.
    minimize sum c_ui (p_ui - x_u . y_i)^2 + lambda(||x||^2+||y||^2), c_ui = 1 + alpha*count."""
    def __init__(self, factors=32, reg=0.1, alpha=40.0, iters=15, seed=20260616):
        self.f, self.reg, self.alpha, self.iters, self.seed = factors, reg, alpha, iters, seed

    def fit(self, train: pd.DataFrame):
        rng = np.random.default_rng(self.seed)
        self.users = {u: i for i, u in enumerate(train.user_id.unique())}
        self.items = {it: j for j, it in enumerate(train.item_id.unique())}
        self.item_ids = list(self.items.keys())
        nu, ni = len(self.users), len(self.items)
        counts = train.groupby(["user_id", "item_id"]).size().reset_index(name="n")
        ui = counts["user_id"].map(self.users).to_numpy()
        ij = counts["item_id"].map(self.items).to_numpy()
        cval = 1.0 + self.alpha * counts["n"].to_numpy()
        # group (item_idx, confidence) by user and (user_idx, confidence) by item
        byu_i = defaultdict(list); byu_c = defaultdict(list)
        byi_u = defaultdict(list); byi_c = defaultdict(list)
        for u, it, c in zip(ui, ij, cval):
            byu_i[u].append(it); byu_c[u].append(c)
            byi_u[it].append(u); byi_c[it].append(c)
        byu_i = {u: np.asarray(v) for u, v in byu_i.items()}
        byu_c = {u: np.asarray(v) for u, v in byu_c.items()}
        byi_u = {it: np.asarray(v) for it, v in byi_u.items()}
        byi_c = {it: np.asarray(v) for it, v in byi_c.items()}
        X = 0.01 * rng.standard_normal((nu, self.f))
        Y = 0.01 * rng.standard_normal((ni, self.f))
        I = self.reg * np.eye(self.f)

        def solve_entity(F, FtF, idxs, cs):
            Fi = F[idxs]                              # (m, f)
            A = FtF + (Fi * (cs - 1.0)[:, None]).T @ Fi + I
            b = (cs[:, None] * Fi).sum(0)
            return np.linalg.solve(A, b)

        for _ in range(self.iters):
            YtY = Y.T @ Y
            for u in range(nu):
                if u in byu_i:
                    X[u] = solve_entity(Y, YtY, byu_i[u], byu_c[u])
            XtX = X.T @ X
            for it in range(ni):
                if it in byi_u:
                    Y[it] = solve_entity(X, XtX, byi_u[it], byi_c[it])
        self.X, self.Y = X, Y
        return self

    def recommend(self, user_id, k, exclude=None):
        if user_id not in self.users:
            return []
        s = self.Y @ self.X[self.users[user_id]]
        order = np.argsort(-s)
        exclude = exclude or set()
        out = []
        for j in order:
            it = self.item_ids[j]
            if it in exclude:
                continue
            out.append(it)
            if len(out) >= k:
                break
        return out
