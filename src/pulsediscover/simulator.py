"""
PulseDiscover — known-propensity semi-synthetic OPE simulator (Phase 4A).

Purpose: manufacture the ONE thing real datasets lack — KNOWN logging propensities —
so that Phase-5 off-policy estimators (IPS/SNIPS/DR) can be validated against ground
truth. This module ONLY generates logged data; it does NOT estimate any policy value.

Design (all components built from TRAIN only — no test-label leakage):
  * mu  (logging policy)      : popularity-softmax over a per-user candidate pool.
  * truth (reward model)      : sigmoid of standardized ALS score (the "real" preference).
  * pi  (target policy)       : softmax over ALS score (the "new" policy to be evaluated).
  mu != truth != pi  -> avoids trivial circularity.

Per impression we build a slate of L slots by SEQUENTIAL sampling without replacement.
At each slot, among the remaining candidates R:
    p_slot(a) = (1-eps) * softmax_mu(a | R)  +  eps * (1/|R|)        # EXACT mixture
  -> positivity guaranteed: every remaining item has p_slot >= eps/|R| > 0.
Examination ~ Bernoulli(PBM(rank)); reward = examined AND Bernoulli(truth(u,a)).

Logged per (impression, slot): user_id, context_len, item_id, rank, position,
propensity (mu per-slot, exact), exploration_flag, examined, reward,
pi_propensity (pi per-slot, for Phase-5 IPS overlap), true_rel (ORACLE/debug only).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def pbm_examination_curve(L, base=0.95, gamma=0.7):
    r = np.arange(L)
    return base * (1.0 / (r + 1.0)) ** gamma            # declining with rank


def _softmax(z, tau):
    z = (z - z.mean()) / (z.std() + 1e-9)
    e = np.exp(z / tau)
    return e / e.sum()


def simulate(train, als, cfg, seed=20260616):
    rng = np.random.default_rng(seed)
    eps, L, N = cfg["epsilon"], cfg["slate_L"], cfg["pool_N"]
    tau_mu, tau_pi = cfg["tau_mu"], cfg["tau_pi"]
    exam = pbm_examination_curve(L, cfg["pbm_base"], cfg["pbm_gamma"])
    pop = train.groupby("item_id").size()
    pop_score = pop.to_dict()
    top_items = list(pop.sort_values(ascending=False).index)             # TRAIN popularity only
    hist = train.groupby("user_id")["item_id"].apply(set).to_dict()

    rows = []
    slot_sum_checks = []
    for u in hist:
        h = hist[u]
        pool = [it for it in top_items if it not in h][:N]
        if len(pool) < L:
            continue
        pool = np.array(pool)
        mu_raw = np.array([np.log(pop_score[it] + 1.0) for it in pool])
        if u in als.users:
            xu = als.X[als.users[u]]
            als_sc = np.array([(als.Y[als.items[it]] @ xu) if it in als.items else 0.0 for it in pool])
        else:
            als_sc = np.zeros(len(pool))
        zt = (als_sc - als_sc.mean()) / (als_sc.std() + 1e-9)
        true_rel = 1.0 / (1.0 + np.exp(-(cfg["rel_alpha"] * zt + cfg["rel_bias"])))

        remaining = list(range(len(pool)))
        for r in range(L):
            rem = np.array(remaining)
            q = _softmax(mu_raw[rem], tau_mu)                 # exploit dist over remaining
            unif = np.ones(len(rem)) / len(rem)
            p_slot = (1 - eps) * q + eps * unif               # EXACT logging prob this slot
            if len(slot_sum_checks) < 200:
                slot_sum_checks.append(float(p_slot.sum()))   # sanity: must be ~1
            explore = rng.random() < eps
            loc = rng.integers(len(rem)) if explore else rng.choice(len(rem), p=q)
            chosen = int(rem[loc])
            pi_slot = _softmax(als_sc[rem], tau_pi)           # target policy this slot
            examined = rng.random() < exam[r]
            reward = 1 if (examined and rng.random() < true_rel[chosen]) else 0
            rows.append((u, len(h), pool[chosen], r, r + 1,
                         float(p_slot[loc]), int(explore), int(examined), int(reward),
                         float(pi_slot[loc]), float(true_rel[chosen])))
            remaining.remove(chosen)

    cols = ["user_id", "context_len", "item_id", "rank", "position",
            "propensity", "exploration_flag", "examined", "reward",
            "pi_propensity", "true_rel"]
    return pd.DataFrame(rows, columns=cols), np.array(slot_sum_checks)
