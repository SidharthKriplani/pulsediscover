"""G29 (Gold Pass 2/3) — Off-Policy Evaluation: logging policy + IPS/SNIPS/DR estimators.

Demonstrates VALID off-policy-evaluation methodology on PulseDiscover's served-c2 candidates.
Honesty boundary: this is OPE METHODOLOGY DEMONSTRATED OFFLINE — propensities come from a
synthetic stochastic logging policy we control (not real traffic), and rewards are a held-out
gold PROXY (not real engagement). It validates that IPS/SNIPS/DR recover a known policy value;
it does NOT estimate real online lift.

Single-action contextual-bandit OPE:
  context x = user; action a = one recommended item from the candidate pool.
  logging policy pi0 stochastic with epsilon-uniform exploration -> pi0(a|x) > 0 (overlap).
  reward r = 1 if a == held-out gold else 0 (proxy).
  target policy pie deterministic (top-1 of ALS / RRF-fusion).
  V_IPS  = mean_i [ 1{a_i = pie(x_i)} / pi0(a_i|x_i) ] r_i
  V_SNIPS= sum_i w_i r_i / sum_i w_i,  w_i = 1{a_i=pie(x_i)}/pi0(a_i|x_i)
  V_DR   = mean_i [ rhat(x_i,pie(x_i)) + w_i (r_i - rhat(x_i,a_i)) ]
"""
from __future__ import annotations
import numpy as np


def softmax(x, temp=1.0):
    z = (np.asarray(x, dtype=np.float64) - np.max(x)) / max(temp, 1e-9)
    e = np.exp(z)
    return e / e.sum()


def logging_propensities(base_scores, epsilon=0.1, temp=1.0):
    """pi0 over a candidate pool: (1-eps)*softmax(base) + eps*uniform. Guarantees overlap."""
    n = len(base_scores)
    p = (1.0 - epsilon) * softmax(base_scores, temp) + epsilon / n
    return p / p.sum()


def sample_logged(rng, pool_items, p, m):
    """Draw m logged actions ~ pi0; return list of (item, propensity)."""
    idx = rng.choice(len(pool_items), size=m, replace=True, p=p)
    return [(pool_items[i], float(p[i])) for i in idx]


def ips(weights, rewards):
    return float(np.mean(weights * rewards))


def snips(weights, rewards):
    s = weights.sum()
    return float((weights * rewards).sum() / s) if s > 0 else 0.0


def doubly_robust(weights, rewards, rhat_action, rhat_target):
    """DR = mean(rhat_target) + mean(w*(r - rhat_action))."""
    return float(np.mean(rhat_target) + np.mean(weights * (rewards - rhat_action)))


def effective_sample_size(weights):
    s1 = weights.sum(); s2 = (weights ** 2).sum()
    return float(s1 * s1 / s2) if s2 > 0 else 0.0
