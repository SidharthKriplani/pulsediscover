"""
PulseDiscover — feedback-loop / catalog-health audit (Phase 7). MovieLens smoke, simulator-scoped.

Models the recommender feedback loop at the catalog level:
  * popularity = TRAIN interaction count per item.
  * quality    = TRAIN mean rating per item, normalized to [0,1]  (quality != popularity, so
                 some high-quality items have low initial popularity -> a ratchet can bury them).
Each round, a policy assigns exposure ~ score; expected clicks = exposure * quality;
clicks accumulate and feed the next round's score (the loop).

Two policies:
  * ratchet     : rank by cumulative CLICK VOLUME (rich-get-richer)  -> concentration grows.
  * ctr_explore : rank by cumulative CTR (exposure-normalized) + exploration -> breaks the loop.

Metrics per round on the exposure distribution: Gini, entropy, long-tail share.
All claims are simulator-scoped (no real online lift).
"""
from __future__ import annotations
import numpy as np


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x); s = x.sum()
    if s <= 0:
        return 0.0
    return float((2.0 * np.sum((np.arange(1, n + 1)) * x)) / (n * s) - (n + 1.0) / n)


def entropy(p):
    p = np.asarray(p, dtype=float); p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def long_tail_share(expo, longtail_mask):
    """Share of this round's exposure going to long-tail items (bottom-by-initial-popularity)."""
    tot = expo.sum()
    return float(expo[longtail_mask].sum() / tot) if tot > 0 else 0.0


def run_loop(quality, popularity, longtail_mask, rounds=10, T=1_000_000.0, eps=0.02, mode="ratchet"):
    M = len(quality)
    C = np.zeros(M)        # cumulative clicks
    E = np.zeros(M)        # cumulative exposures
    score = popularity.astype(float).copy()   # round-0 ranking = popularity (both policies)
    metrics, expo_hist = [], []
    for t in range(rounds):
        p = (1 - eps) * (score / score.sum()) + eps / M
        expo = T * p                          # expected exposures this round (flow)
        clicks = expo * quality               # expected clicks (quality drives CTR)
        C += clicks; E += expo
        expo_hist.append(expo.copy())
        metrics.append({"round": t, "gini": round(gini(expo), 4),
                        "entropy": round(entropy(expo / expo.sum()), 4),
                        "long_tail_share": round(long_tail_share(expo, longtail_mask), 4)})
        if mode == "ratchet":
            score = C + 1.0                   # volume of clicks -> rich get richer
        elif mode == "ctr_explore":
            score = C / (E + 1.0) + 1e-4      # exposure-normalized rate (+eps exploration via p)
    return metrics, expo_hist, C, E
