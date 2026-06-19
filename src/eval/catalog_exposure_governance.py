"""G25 — Catalog Exposure Governance metrics (PulseDiscover V2).

Pure metric helpers for auditing how retrieval/fallback policies distribute
EXPOSURE across the catalog. This is *catalog exposure governance* — concentration,
coverage and head/tail opportunity under the current offline protocol — and is
explicitly NOT legally protected-class fairness and NOT a fairness certification.

All functions take an `exposure` mapping (item_id -> times recommended across the
audited user population at top-K) and return descriptive statistics only.
"""
from __future__ import annotations
import numpy as np


def gini(counts: np.ndarray) -> float:
    """Gini coefficient of an exposure-count vector over the FULL catalog
    (zero-exposure items included). 0 = perfectly even, ->1 = fully concentrated.
    """
    x = np.sort(np.asarray(counts, dtype=np.float64))
    n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * x) / (n * x.sum())) - (n + 1.0) / n)


def lorenz_curve(counts: np.ndarray, points: int = 101):
    """Return (cum_pop_fraction, cum_exposure_fraction) for a Lorenz curve."""
    x = np.sort(np.asarray(counts, dtype=np.float64))
    n = x.size
    cum = np.cumsum(x)
    if cum[-1] == 0:
        cum_frac = np.zeros(n)
    else:
        cum_frac = cum / cum[-1]
    pop = np.linspace(0, 1, n)
    grid = np.linspace(0, 1, points)
    return grid.tolist(), np.interp(grid, pop, cum_frac).tolist()


def top_share(counts: np.ndarray, frac: float) -> float:
    """Share of total exposure captured by the top `frac` of items by exposure."""
    x = np.sort(np.asarray(counts, dtype=np.float64))[::-1]
    tot = x.sum()
    if tot == 0:
        return 0.0
    k = max(1, int(np.ceil(len(x) * frac)))
    return float(x[:k].sum() / tot)


def exposure_stats(exposure: dict, catalog: list, item_tier: dict, k: int) -> dict:
    """Full exposure-governance stats for one policy.

    exposure  : item_id -> exposure count (only items with >0 may appear)
    catalog   : list of all catalog item_ids (defines zero-exposure universe)
    item_tier : item_id -> 'head'|'mid'|'long_tail'
    """
    cat = list(catalog)
    counts = np.array([exposure.get(b, 0) for b in cat], dtype=np.float64)
    total = float(counts.sum())
    nonzero = int((counts > 0).sum())
    n = len(cat)

    # tier exposure shares + averages + base-rate lift
    tiers = ["head", "mid", "long_tail"]
    tier_exposure = {t: 0.0 for t in tiers}
    tier_items = {t: 0 for t in tiers}
    tier_nonzero = {t: 0 for t in tiers}
    for b, c in zip(cat, counts):
        t = item_tier.get(b, "long_tail")
        if t not in tier_exposure:
            t = "long_tail"
        tier_exposure[t] += c
        tier_items[t] += 1
        if c > 0:
            tier_nonzero[t] += 1

    tier_share = {t: round(tier_exposure[t] / total, 5) if total else 0.0 for t in tiers}
    tier_catalog_share = {t: round(tier_items[t] / n, 5) for t in tiers}
    # exposure lift vs catalog base rate (share of exposure / share of catalog)
    tier_lift = {t: round((tier_share[t] / tier_catalog_share[t]), 3)
                 if tier_catalog_share[t] > 0 else None for t in tiers}
    tier_avg_exposure = {t: round(tier_exposure[t] / max(tier_items[t], 1), 3) for t in tiers}
    tier_coverage = {t: round(tier_nonzero[t] / max(tier_items[t], 1), 4) for t in tiers}

    return {
        "k": k,
        "total_exposure_slots": int(total),
        "unique_items_recommended": nonzero,
        "catalog_coverage": round(nonzero / n, 5),
        "zero_exposure_items": n - nonzero,
        "zero_exposure_share": round((n - nonzero) / n, 5),
        "gini": round(gini(counts), 4),
        "top_1pct_share": round(top_share(counts, 0.01), 4),
        "top_5pct_share": round(top_share(counts, 0.05), 4),
        "top_10pct_share": round(top_share(counts, 0.10), 4),
        "tier_exposure_share": tier_share,
        "tier_catalog_share": tier_catalog_share,
        "tier_exposure_lift_vs_base": tier_lift,
        "tier_avg_exposure": tier_avg_exposure,
        "tier_coverage": tier_coverage,
    }
