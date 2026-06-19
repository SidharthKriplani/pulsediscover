"""
PulseDiscover — off-policy estimators + OPE validation harness (Phase 5).

Estimators of the target policy value V(pi), from mu-logged data only:
  IPS   = mean( w * r ),                       w = pi/mu
  SNIPS = sum(w*r) / sum(w)
  DM    = mean( qhat_pi )                       qhat_pi = E_{a~pi}[qhat(x,a)]
  DR    = mean( qhat_pi + w*(r - qhat_ai) )

Ground truth V(pi) is computed SEPARATELY from the oracle reward model (true_rel),
and is NEVER given to the estimators (they use only w, r, and qhat). qhat is a fitted
reward model with a 'good' (mildly misspecified) and a 'bad' (severely misspecified)
variant, so DR's double-robustness is actually stress-tested.

All per-user slates have exactly L events -> arrays reshape to (n_users, L), enabling
fast user-CLUSTER bootstrap (resample users, not individual slots).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _softmax(z, tau):
    z = (z - z.mean()) / (z.std() + 1e-9)
    e = np.exp(z / tau)
    return e / e.sum()


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def augmented_log(train, als, cfg, seed, pi_raw_for_pool=None):
    """Replays the Phase-4A slate logic (same seed/config -> same events) and ALSO emits,
    per event: w-inputs (mu,pi), reward, ground-truth pi-reward g, and qhat terms for a
    'good' (mild) and 'bad' (severe) reward model. Returns dict of (n_users, L) arrays."""
    rng = np.random.default_rng(seed)
    eps, L, N = cfg["epsilon"], cfg["slate_L"], cfg["pool_N"]
    tau_mu, tau_pi = cfg["tau_mu"], cfg["tau_pi"]
    rA, rB = cfg["rel_alpha"], cfg["rel_bias"]                 # truth params
    gA, gB = cfg["qhat_good_alpha"], cfg["qhat_good_bias"]     # good qhat params (off-truth)
    exam = cfg["pbm_base"] * (1.0 / (np.arange(L) + 1.0)) ** cfg["pbm_gamma"]
    pop = train.groupby("item_id").size()
    pop_score = pop.to_dict()
    top_items = list(pop.sort_values(ascending=False).index)
    hist = train.groupby("user_id")["item_id"].apply(set).to_dict()
    maxpop = float(max(pop_score.values()))

    cols = {k: [] for k in ["mu", "pi", "r", "g", "qai_g", "qpi_g", "qai_b", "qpi_b"]}
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
        true_rel = _sigmoid(rA * zt + rB)                       # ORACLE truth (not given to estimators)
        qrel_good = _sigmoid(gA * zt + gB)                      # mildly misspecified reward model
        qrel_bad = np.clip(qrel_good + 0.4, 0.0, 1.0)           # SEVERELY misspecified: +0.4 overconfident
        # target policy π raw scores over the pool: ALS by default, or an external source (e.g. SASRec)
        pi_base = als_sc if pi_raw_for_pool is None else pi_raw_for_pool(u, pool)
        remaining = list(range(len(pool)))
        for r in range(L):
            rem = np.array(remaining)
            q = _softmax(mu_raw[rem], tau_mu)
            p_slot = (1 - eps) * q + eps * (1.0 / len(rem))
            explore = rng.random() < eps
            loc = rng.integers(len(rem)) if explore else rng.choice(len(rem), p=q)
            chosen = int(rem[loc])
            pi_slot = _softmax(pi_base[rem], tau_pi)
            examined = rng.random() < exam[r]
            reward = 1 if (examined and rng.random() < true_rel[chosen]) else 0
            # ground truth expected reward under pi at this (user,rank,remaining-set) context:
            g = exam[r] * float(np.sum(pi_slot * true_rel[rem]))
            # qhat terms (good / bad), per-slot expectation under pi + value at chosen action:
            qg_rem = exam[r] * qrel_good[rem]
            qb_rem = exam[r] * qrel_bad[rem]
            cols["mu"].append(float(p_slot[loc])); cols["pi"].append(float(pi_slot[loc]))
            cols["r"].append(reward); cols["g"].append(g)
            cols["qai_g"].append(float(qg_rem[loc])); cols["qpi_g"].append(float(np.sum(pi_slot * qg_rem)))
            cols["qai_b"].append(float(qb_rem[loc])); cols["qpi_b"].append(float(np.sum(pi_slot * qb_rem)))
            remaining.remove(chosen)
    nu = len(cols["mu"]) // L
    return {k: np.asarray(v, dtype=float).reshape(nu, L) for k, v in cols.items()}, nu, L


def _clip(w, c):
    return w if c is None else np.minimum(w, c)


def estimators(mu, pi, r, qai, qpi, clip=None):
    w = _clip(pi / mu, clip)
    ips = float(np.mean(w * r))
    sw = np.sum(w)
    snips = float(np.sum(w * r) / sw) if sw > 0 else 0.0
    dm = float(np.mean(qpi))
    dr = float(np.mean(qpi + w * (r - qai)))
    ess = float((sw ** 2) / np.sum(w ** 2)) if np.sum(w ** 2) > 0 else 0.0
    return {"IPS": ips, "SNIPS": snips, "DM": dm, "DR": dr, "ESS": ess,
            "w_max": float(w.max()), "w_mean": float(w.mean())}


def cluster_bootstrap(A, qkey, clip, B=500, seed=0):
    """User-cluster bootstrap (resample the (nu,L) rows). Returns {est: (lo,hi,std)} for
    IPS/SNIPS/DM/DR using qhat variant qkey ('g' or 'b')."""
    rng = np.random.default_rng(seed)
    nu = A["mu"].shape[0]
    qai, qpi = A[f"qai_{qkey}"], A[f"qpi_{qkey}"]
    keys = ["IPS", "SNIPS", "DM", "DR"]
    samples = {k: [] for k in keys}
    for _ in range(B):
        idx = rng.integers(0, nu, nu)
        e = estimators(A["mu"][idx].ravel(), A["pi"][idx].ravel(), A["r"][idx].ravel(),
                       qai[idx].ravel(), qpi[idx].ravel(), clip=clip)
        for k in keys:
            samples[k].append(e[k])
    out = {}
    for k in keys:
        s = np.array(samples[k])
        out[k] = {"ci95": [round(float(np.quantile(s, 0.025)), 5), round(float(np.quantile(s, 0.975)), 5)],
                  "boot_std": round(float(s.std()), 6)}
    return out
