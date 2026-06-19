"""
PulseDiscover — evaluation + confidence intervals (Phase 2). Pure numpy.

Ranking metrics: Recall@K, NDCG@K (leave-last-out style: one held-out item per user).
CIs:
  - bootstrap_ci : per-user bootstrap (the headline method for Recall@K/NDCG@K,
                   since these are means over users — distribution-free, captures
                   user-level variance).
  - wilson_ci    : Wilson score interval for a single proportion (used when a metric
                   is a clean 0/1 hit-rate; narrower/correct at small n vs normal approx).
"""
from __future__ import annotations
import numpy as np


def recall_at_k(recommended, holdout_item, k) -> float:
    return 1.0 if holdout_item in recommended[:k] else 0.0


def ndcg_at_k(recommended, holdout_item, k) -> float:
    topk = recommended[:k]
    if holdout_item in topk:
        rank = topk.index(holdout_item)  # 0-based
        return 1.0 / np.log2(rank + 2)
    return 0.0


def evaluate_users(per_user_hits: list[float]) -> float:
    return float(np.mean(per_user_hits)) if per_user_hits else 0.0


def bootstrap_ci(per_user_values, n_boot=2000, alpha=0.05, seed=20260616):
    """95% bootstrap CI of the mean over users. Returns (mean, lo, hi, n)."""
    v = np.asarray(per_user_values, dtype=float)
    n = len(v)
    if n == 0:
        return (0.0, 0.0, 0.0, 0)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(v.mean()), float(lo), float(hi), int(n))


def wilson_ci(successes, n, z=1.96):
    """Wilson score interval for a proportion. Returns (phat, lo, hi)."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    phat = successes / n
    den = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    adj = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return (float(phat), float((centre - adj) / den), float((centre + adj) / den))
