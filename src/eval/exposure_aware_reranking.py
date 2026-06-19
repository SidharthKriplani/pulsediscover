"""G26 — Exposure-aware reranking policies (PulseDiscover V2).

Mitigation policies that rerank an ALS relevance-ordered candidate POOL into a
top-K slate, trading a measured amount of relevance for improved catalog
exposure. Pure functions over a single user's pool — the run script aggregates
relevance, exposure-governance and latency metrics across users.

Each reranker takes:
  pool_items  : list[str]      candidate item_ids, ALS-relevance-descending
  pool_scores : np.ndarray     ALS scores aligned to pool_items
  ctx         : dict with
                  'pop_pct'  item_id -> popularity percentile in [0,1] (1=head)
                  'emb'      item_id -> unit-norm embedding (np.ndarray) [MMR]
                  'genre'    item_id -> genre token (or None)
                  'is_head'  item_id -> bool
and returns a list[str] of length<=k (the reranked slate).

Nothing here claims fairness; these are catalog-exposure mitigation heuristics.
"""
from __future__ import annotations
import numpy as np


def baseline(pool_items, pool_scores, ctx, k):
    """Pure ALS relevance order (the G25 served ordering)."""
    return list(pool_items[:k])


def long_tail_boost(pool_items, pool_scores, ctx, k, alpha=0.3):
    """Add a popularity-inverse bonus: score' = z(score) + alpha * (1 - pop_pct).

    alpha=0 -> baseline; larger alpha pushes long-tail items up.
    """
    s = np.asarray(pool_scores, dtype=np.float64)
    z = (s - s.mean()) / (s.std() + 1e-9)
    pp = np.array([ctx["pop_pct"].get(b, 0.0) for b in pool_items])
    adj = z + alpha * (1.0 - pp)
    order = np.argsort(-adj)
    return [pool_items[i] for i in order[:k]]


def novelty_boost(pool_items, pool_scores, ctx, k, alpha=0.3):
    """Novelty = -log2(pop_share) proxy via (1-pop_pct)^2; rewards rarer items more steeply."""
    s = np.asarray(pool_scores, dtype=np.float64)
    z = (s - s.mean()) / (s.std() + 1e-9)
    pp = np.array([ctx["pop_pct"].get(b, 0.0) for b in pool_items])
    adj = z + alpha * (1.0 - pp) ** 2
    order = np.argsort(-adj)
    return [pool_items[i] for i in order[:k]]


def head_cap(pool_items, pool_scores, ctx, k, cap_frac=0.6):
    """Cap the number of head items in the slate at cap_frac*k; fill with non-head."""
    cap = int(round(cap_frac * k))
    out, n_head = [], 0
    for b in pool_items:
        if ctx["is_head"].get(b, False):
            if n_head >= cap:
                continue
            n_head += 1
        out.append(b)
        if len(out) >= k:
            break
    if len(out) < k:  # backfill with remaining (incl. head) to never under-serve
        for b in pool_items:
            if b not in out:
                out.append(b)
            if len(out) >= k:
                break
    return out[:k]


def genre_diversify(pool_items, pool_scores, ctx, k, max_per_genre=4):
    """Limit items per genre/shelf in the slate; backfill if metadata sparse."""
    out, counts = [], {}
    for b in pool_items:
        g = ctx["genre"].get(b)
        if g is not None and counts.get(g, 0) >= max_per_genre:
            continue
        out.append(b)
        if g is not None:
            counts[g] = counts.get(g, 0) + 1
        if len(out) >= k:
            break
    if len(out) < k:
        for b in pool_items:
            if b not in out:
                out.append(b)
            if len(out) >= k:
                break
    return out[:k]


def mmr(pool_items, pool_scores, ctx, k, lam=0.7, pool_cap=120):
    """Maximal Marginal Relevance: lam*relevance - (1-lam)*max_sim_to_selected.

    lam=1 -> pure relevance (baseline); lower lam -> more intra-slate diversity.
    Uses item embeddings (unit-norm) for similarity; pool capped for latency.
    """
    items = list(pool_items[:pool_cap])
    s = np.asarray(pool_scores[:pool_cap], dtype=np.float64)
    rel = (s - s.min()) / (s.max() - s.min() + 1e-9)
    emb = np.stack([ctx["emb"][b] for b in items])  # (P, d) unit-norm
    selected, sel_idx = [], []
    cand = set(range(len(items)))
    # seed with top relevance
    first = int(np.argmax(rel)); selected.append(items[first]); sel_idx.append(first); cand.discard(first)
    while len(selected) < min(k, len(items)) and cand:
        cidx = np.array(sorted(cand))
        sims = emb[cidx] @ emb[sel_idx].T          # (|cand|, |sel|)
        max_sim = sims.max(axis=1)
        mmr_score = lam * rel[cidx] - (1.0 - lam) * max_sim
        pick = cidx[int(np.argmax(mmr_score))]
        selected.append(items[pick]); sel_idx.append(pick); cand.discard(pick)
    return selected[:k]


# Registry of (name, fn, kwargs) — the run script may sweep params to trace a frontier.
def policy_grid():
    grid = [("baseline", baseline, {})]
    for a in (0.15, 0.3, 0.6, 1.0):
        grid.append((f"tail_boost_a{a}", long_tail_boost, {"alpha": a}))
    for a in (0.3, 0.6):
        grid.append((f"novelty_a{a}", novelty_boost, {"alpha": a}))
    for cf in (0.7, 0.5, 0.3):
        grid.append((f"head_cap_{cf}", head_cap, {"cap_frac": cf}))
    for mg in (4, 2):
        grid.append((f"genre_div_{mg}", genre_diversify, {"max_per_genre": mg}))
    for lam in (0.85, 0.7, 0.5):
        grid.append((f"mmr_lam{lam}", mmr, {"lam": lam}))
    return grid
