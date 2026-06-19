"""
PulseDiscover — position-bias correction (Phase 6). Uses Phase-4A simulator logs.

Three estimators of per-item relevance (aggregated over the impressions of each item):
  * naive : mean(click)                         -- position-confounded (top slots inflate clicks)
  * IPS   : mean(click / P(examined|rank))      -- examination-reweighted (Route A)
  * PAL   : logit(click) = theta_item + beta_pos, learned jointly; theta = debiased
            relevance, beta dropped at serving (Route B). (Route B, if torch available.)

Ground truth = oracle true_rel (logged), used ONLY for comparison, never by the estimators.
Examination propensity P(examined|rank) is ESTIMATED empirically from the logged
examination rate by rank (in reality this comes from randomization/PBM-EM; here the sim
provides the examination signal). Per-item relevance is the simulator-scoped quantity.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr


def empirical_examination(events: pd.DataFrame):
    """P(examined | rank) estimated from logged examination rates."""
    return events.groupby("rank")["examined"].mean()


def per_item_estimates(events: pd.DataFrame, exam_prop: pd.Series, min_impr=20):
    ev = events.copy()
    ev["w_exam"] = 1.0 / ev["rank"].map(exam_prop).clip(lower=1e-6)
    ev["click_ips"] = ev["reward"] * ev["w_exam"]
    g = ev.groupby("item_id")
    tab = pd.DataFrame({
        "n_impr": g.size(),
        "avg_rank": g["rank"].mean(),
        "naive_rel": g["reward"].mean(),
        "ips_rel": g["click_ips"].mean(),
        "true_rel": g["true_rel"].mean(),
    })
    return tab[tab["n_impr"] >= min_impr].copy()


def fit_pal(events: pd.DataFrame, n_steps=400, lr=0.05, seed=20260616):
    """PAL additive tower (Route B): logit(click)=theta_item+beta_pos. Returns {item_id: theta}.
    Returns None if torch unavailable (-> mark PAL as V2, do not claim)."""
    try:
        import torch, torch.nn as nn
    except Exception:
        return None
    torch.manual_seed(seed)
    items = {it: i for i, it in enumerate(events["item_id"].unique())}
    L = int(events["rank"].max()) + 1
    it_idx = torch.tensor(events["item_id"].map(items).to_numpy(), dtype=torch.long)
    rk_idx = torch.tensor(events["rank"].to_numpy(), dtype=torch.long)
    y = torch.tensor(events["reward"].to_numpy(), dtype=torch.float32)
    theta = nn.Embedding(len(items), 1); beta = nn.Embedding(L, 1)
    nn.init.zeros_(theta.weight); nn.init.zeros_(beta.weight)
    opt = torch.optim.Adam(list(theta.parameters()) + list(beta.parameters()), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(n_steps):
        logit = (theta(it_idx) + beta(rk_idx)).squeeze(-1)
        loss = lossf(logit, y)
        opt.zero_grad(); loss.backward(); opt.step()
    th = theta.weight.detach().numpy().ravel()
    return {it: float(th[i]) for it, i in items.items()}, float(loss.detach())


def comparison_metrics(tab: pd.DataFrame):
    t = tab["true_rel"].to_numpy()
    out = {}
    for name in ["naive_rel", "ips_rel"]:
        e = tab[name].to_numpy()
        out[name] = {
            "mse_vs_true": round(float(np.mean((e - t) ** 2)), 6),
            "pearson_vs_true": round(float(pearsonr(e, t)[0]), 4),
            "spearman_vs_true": round(float(spearmanr(e, t)[0]), 4),
            "corr_with_avg_rank": round(float(pearsonr(e, tab["avg_rank"].to_numpy())[0]), 4),
        }
    return out
