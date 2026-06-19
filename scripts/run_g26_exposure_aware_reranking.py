"""G26 — Exposure-aware reranking frontier (PulseDiscover V2).

Builds mitigation policies on top of the G25 baseline and quantifies the
RELEVANCE vs EXPOSURE tradeoff: for each reranker we measure Recall@20/@50,
catalog coverage, Gini, top-x% exposure, head/mid/tail share, zero-exposure,
long-tail lift vs baseline, per-slate rerank latency, and empty-response rate.

Scope guardrails:
  * Reranking is applied to the WARM ALS candidate pool (served c2_als.pkl,
    ALS f64) — the cohort where a pool exists to rerank. Cold/unknown users still
    receive the popularity fallback (unchanged); reranking does not "fix" them.
  * Catalog EXPOSURE mitigation under the offline protocol. NOT fairness solved,
    NOT certified, NOT online diversity, NOT production deployment.

Outputs:
  outputs/evidence/g26_exposure_reranking_frontier_report.json
  outputs/plots/g26_relevance_exposure_frontier.png
  outputs/plots/g26_tail_exposure_vs_recall_tradeoff.png
"""
from __future__ import annotations
import os, sys, json, time
from collections import defaultdict
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from eval.cold_start_cohorts import load_als, load_eval_samples, build_cohorts, build_genre_map
from eval.catalog_exposure_governance import exposure_stats
from eval.exposure_aware_reranking import policy_grid

DATA = os.environ.get("PD_DATADIR", "data/interim")
SPLITS = os.path.join(DATA, "domain_splits")
EVID, PLOTS = "outputs/evidence", "outputs/plots"
os.makedirs(EVID, exist_ok=True); os.makedirs(PLOTS, exist_ok=True)
K, POOL = 20, 200
SAMPLE_WARM = 4000
RNG = np.random.default_rng(7)
t0 = time.time()

print("[1/5] load data + cohorts (served c2_als.pkl) ...")
train = pd.read_csv(os.path.join(SPLITS, "train.csv"),
                    usecols=["user_id", "book_id", "event_ts"], dtype={"book_id": str})
als = load_als(); ev = load_eval_samples()
coh, ctx = build_cohorts(train, als, ev)
Y = als["Y"].astype(np.float32); X = als["X"].astype(np.float32)
i_uniq = als["i_uniq"]; iidx = als["iidx"]; uidx = als["uidx"]
pop = ctx["pop"]; item_tier = ctx["item_tier"]
catalog = list(map(str, i_uniq)); catalog_n = len(catalog)

# context maps for rerankers
ranks = pop.rank(method="average")  # higher = more popular
pop_pct = {str(b): float(r / len(pop)) for b, r in ranks.items()}
pop_pct = defaultdict(float, pop_pct)
is_head = {b: (item_tier.get(b) == "head") for b in catalog}
Ynorm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9)
emb = {str(i_uniq[i]): Ynorm[i] for i in range(len(i_uniq))}
gmap = build_genre_map()
rerank_ctx = {"pop_pct": pop_pct, "emb": emb, "genre": gmap, "is_head": is_head}

# warm users + seen sets + gold
warm = coh[coh["cohort"] == "warm_6plus"].copy()
warm = warm[warm["has_factor"]].reset_index(drop=True)
if len(warm) > SAMPLE_WARM:
    warm = warm.iloc[RNG.choice(len(warm), SAMPLE_WARM, replace=False)].reset_index(drop=True)
warm_uset = set(warm["user_id"])
seen = train[train["user_id"].isin(warm_uset)].groupby("user_id")["book_id"].apply(set).to_dict()
gold_by_user = dict(zip(warm["user_id"], warm["gold"]))
print(f"   warm sample={len(warm)} catalog={catalog_n}  ({time.time()-t0:.1f}s)")

print("[2/5] building ALS candidate pools (top-200) ...")
pools = {}  # user -> (items, scores)
rows = np.array([uidx[u] for u in warm["user_id"]])
users = warm["user_id"].values
BATCH = 500
for s0 in range(0, len(users), BATCH):
    chunk = users[s0:s0 + BATCH]; rr = rows[s0:s0 + BATCH]
    sc = X[rr] @ Y.T  # (B, nI)
    part = np.argpartition(-sc, POOL + 60, axis=1)[:, :POOL + 60]
    for ci, u in enumerate(chunk):
        idxs = part[ci][np.argsort(-sc[ci, part[ci]])]
        s = seen.get(u, set()); items, scores = [], []
        for j in idxs:
            b = str(i_uniq[j])
            if b in s:
                continue
            items.append(b); scores.append(float(sc[ci, j]))
            if len(items) >= POOL:
                break
        pools[u] = (items, np.array(scores))
print(f"   pools built {len(pools)}  ({time.time()-t0:.1f}s)")

print("[3/5] running reranking policies + metrics ...")
grid = policy_grid()
rows_out = []
for name, fn, kw in grid:
    exposure = defaultdict(int)
    hit20 = hit50 = empty = 0
    lat = []
    for u in users:
        items, scores = pools[u]
        if not items:
            empty += 1
            continue
        ts = time.perf_counter()
        slate = fn(items, scores, rerank_ctx, K, **kw)
        lat.append((time.perf_counter() - ts) * 1000.0)
        if not slate:
            empty += 1
            continue
        gold = gold_by_user[u]
        if gold in slate[:K]:
            hit20 += 1
        # recall@50 uses the top-50 of the SAME reranked order if available
        slate50 = fn(items, scores, rerank_ctx, 50, **kw) if K < 50 else slate
        if gold in slate50[:50]:
            hit50 += 1
        for b in slate[:K]:
            exposure[b] += 1
    n = len(users)
    es = exposure_stats(exposure, catalog, item_tier, K)
    rows_out.append({
        "policy": name,
        "recall@20": round(hit20 / n, 4),
        "recall@50": round(hit50 / n, 4),
        "catalog_coverage": es["catalog_coverage"],
        "unique_items": es["unique_items_recommended"],
        "gini": es["gini"],
        "top_1pct_share": es["top_1pct_share"],
        "top_5pct_share": es["top_5pct_share"],
        "top_10pct_share": es["top_10pct_share"],
        "head_share": es["tier_exposure_share"]["head"],
        "mid_share": es["tier_exposure_share"]["mid"],
        "long_tail_share": es["tier_exposure_share"]["long_tail"],
        "zero_exposure_share": es["zero_exposure_share"],
        "long_tail_lift_vs_base": es["tier_exposure_lift_vs_base"]["long_tail"],
        "rerank_latency_ms_p50": round(float(np.percentile(lat, 50)), 4) if lat else None,
        "rerank_latency_ms_p95": round(float(np.percentile(lat, 95)), 4) if lat else None,
        "empty_rate": round(empty / n, 4),
    })
    print(f"  {name:16s} R@20={rows_out[-1]['recall@20']:.4f} "
          f"cov={rows_out[-1]['catalog_coverage']:.4f} gini={rows_out[-1]['gini']:.3f} "
          f"tail={rows_out[-1]['long_tail_share']:.4f} lat_p95={rows_out[-1]['rerank_latency_ms_p95']}")

# ---- frontier analysis vs baseline
base = next(r for r in rows_out if r["policy"] == "baseline")
for r in rows_out:
    r["recall_delta_vs_base"] = round(r["recall@20"] - base["recall@20"], 4)
    r["recall_retention"] = round(r["recall@20"] / base["recall@20"], 4) if base["recall@20"] else None
    r["coverage_gain_vs_base"] = round(r["catalog_coverage"] - base["catalog_coverage"], 5)
    r["tail_share_gain_vs_base"] = round(r["long_tail_share"] - base["long_tail_share"], 5)

# Pareto frontier: maximize recall, maximize long_tail_share (non-dominated set)
def dominated(a, b):
    return (b["recall@20"] >= a["recall@20"] and b["long_tail_share"] >= a["long_tail_share"]
            and (b["recall@20"] > a["recall@20"] or b["long_tail_share"] > a["long_tail_share"]))
pareto = [r["policy"] for r in rows_out if not any(dominated(r, o) for o in rows_out if o is not r)]

# Recommendation: best long-tail/coverage gain while retaining >=90% baseline recall
eligible = [r for r in rows_out if r["recall_retention"] is not None and r["recall_retention"] >= 0.90
            and r["policy"] != "baseline"]
eligible.sort(key=lambda r: (r["long_tail_share"], r["catalog_coverage"]), reverse=True)
recommended = eligible[0]["policy"] if eligible else "baseline"

report = {
    "gate": "G26",
    "title": "Exposure-Aware Reranking / Mitigation Policy",
    "lane": "V2",
    "provenance": "computed_offline_real_data",
    "scope_guardrails": [
        "Reranking applied to the WARM ALS candidate pool (served c2_als.pkl, ALS f64); cold/unknown users keep the popularity fallback (unchanged).",
        "Catalog EXPOSURE mitigation under the offline protocol; NOT fairness solved/certified, NOT online diversity, NOT production deployment, NOT protected-class analysis.",
        "Recall is full-catalog single-held-out-gold on the served model; comparable WITHIN G26, not to the V1 d3aplus headline.",
    ],
    "config": {"k": K, "pool_size": POOL, "warm_sample_users": len(warm),
               "catalog_size": catalog_n, "recall_protocol": "single held-out gold, ALS pool top-200"},
    "baseline": {"recall@20": base["recall@20"], "catalog_coverage": base["catalog_coverage"],
                 "gini": base["gini"], "long_tail_share": base["long_tail_share"]},
    "frontier_table": rows_out,
    "pareto_optimal_policies": pareto,
    "recommended_policy": recommended,
    "recommendation_rule": "max long-tail/coverage gain while retaining >=90% of baseline Recall@20",
    "limitations": [
        "Warm cohort only — cold/unknown users are out of scope for reranking (they have no ALS pool).",
        "Recall is single held-out gold; NDCG not computed (one gold per user makes it ~ proportional to recall).",
        "Latency is in-process rerank compute (CPU), not end-to-end serving latency.",
        "Genre diversification uses noisy Goodreads-shelf genres; partial coverage.",
        "Exposure mitigation is a heuristic tradeoff, NOT a fairness guarantee or certification.",
    ],
    "claim_boundary": {
        "relevance_exposure_tradeoff_measured": True,
        "mitigation_attempted": True,
        "fairness_solved": False,
        "creator_fairness_solved": False,
        "marketplace_fairness_certified": False,
        "long_tail_discovery_solved": False,
        "online_diversity_improved": False,
        "production_exposure_governance_deployed": False,
        "protected_class_analysis": False,
        "safe_claim": ("PulseDiscovery G26 evaluates exposure-aware reranking policies and "
                       "quantifies the relevance/exposure tradeoff, showing which mitigation "
                       "strategies improve catalog coverage or long-tail exposure without "
                       "unacceptable retrieval-quality degradation."),
    },
    "runtime_sec": round(time.time() - t0, 1),
}
json.dump(report, open(os.path.join(EVID, "g26_exposure_reranking_frontier_report.json"), "w"), indent=2)

print("[4/5] plots ...")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Plot 1: relevance vs exposure frontier (recall@20 vs catalog coverage), Pareto highlighted
fig, ax = plt.subplots(figsize=(8.5, 6))
for r in rows_out:
    isp = r["policy"] in pareto
    ax.scatter(r["catalog_coverage"], r["recall@20"],
               s=90 if isp else 45, c=("#c0392b" if isp else "#7f8c8d"),
               edgecolor="k" if r["policy"] == "baseline" else "none", zorder=3 if isp else 2)
    ax.annotate(r["policy"], (r["catalog_coverage"], r["recall@20"]),
                fontsize=6.5, xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("catalog coverage@20 (exposure breadth →)")
ax.set_ylabel("Recall@20 (relevance ↑)")
ax.set_title("G26 Relevance–Exposure frontier\n(red = Pareto-optimal; black edge = baseline)")
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "g26_relevance_exposure_frontier.png"), dpi=130); plt.close(fig)

# Plot 2: long-tail exposure share vs recall@20 tradeoff
fig, ax = plt.subplots(figsize=(8.5, 6))
xs = [r["long_tail_share"] for r in rows_out]
ys = [r["recall@20"] for r in rows_out]
ax.axhline(base["recall@20"] * 0.9, ls="--", c="#e67e22", lw=1, label="90% baseline recall")
ax.axvline(base["long_tail_share"], ls=":", c="#2980b9", lw=1, label="baseline tail share")
for r in rows_out:
    ax.scatter(r["long_tail_share"], r["recall@20"], s=55,
               c=("#27ae60" if r["policy"] == recommended else "#34495e"), zorder=3)
    ax.annotate(r["policy"], (r["long_tail_share"], r["recall@20"]),
                fontsize=6.5, xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("long-tail exposure share (↑ healthier tail)")
ax.set_ylabel("Recall@20 (relevance ↑)")
ax.set_title(f"G26 Long-tail exposure vs recall tradeoff\n(green = recommended: {recommended})")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "g26_tail_exposure_vs_recall_tradeoff.png"), dpi=130); plt.close(fig)

print(f"[5/5] DONE in {time.time()-t0:.1f}s")
print("baseline R@20:", base["recall@20"], "cov:", base["catalog_coverage"], "tail:", base["long_tail_share"])
print("pareto:", pareto)
print("recommended:", recommended)
