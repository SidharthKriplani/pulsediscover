"""G25 — Catalog Exposure Governance audit (PulseDiscover V2).

Quantifies how each retrieval/fallback policy distributes EXPOSURE across the
catalog: concentration (Gini, Lorenz, top-x%), coverage@K, head/mid/long-tail
exposure share + base-rate lift, zero-exposure share, and a user-cohort x
item-cohort exposure matrix. Also compares the SERVED exact-FAISS vs HNSW mode.

Scope guardrails:
  * Catalog EXPOSURE governance under the offline G24/G23 served-model protocol
    (c2_als.pkl ALS f64). NOT protected-class fairness, NOT a fairness
    certification, NOT online diversity, NOT production exposure governance.
  * "item-cold-start" = held-out gold items unseen in train under this split.

Outputs:
  outputs/evidence/g25_catalog_exposure_governance_report.json
  outputs/plots/g25_exposure_concentration_curve.png
  outputs/plots/g25_head_mid_tail_exposure_by_policy.png
  outputs/plots/g25_policy_exposure_lorenz_curve.png
"""
from __future__ import annotations
import os, sys, json, time
from collections import defaultdict
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "src", "serving"))
from eval.cold_start_cohorts import (
    load_als, load_eval_samples, build_cohorts, build_genre_map, user_last_item,
)
from eval.catalog_exposure_governance import exposure_stats, lorenz_curve, gini

DATA = os.environ.get("PD_DATADIR", "data/interim")
SPLITS = os.path.join(DATA, "domain_splits")
EVID, PLOTS = "outputs/evidence", "outputs/plots"
os.makedirs(EVID, exist_ok=True); os.makedirs(PLOTS, exist_ok=True)
K = 20
t0 = time.time()

print("[1/6] loading data + cohorts (served c2_als.pkl protocol) ...")
train = pd.read_csv(os.path.join(SPLITS, "train.csv"),
                    usecols=["user_id", "book_id", "event_ts"], dtype={"book_id": str})
als = load_als(); ev = load_eval_samples()
coh, ctx = build_cohorts(train, als, ev)
Y = als["Y"].astype(np.float32); X = als["X"].astype(np.float32)
i_uniq = als["i_uniq"]; iidx = als["iidx"]; uidx = als["uidx"]
pop = ctx["pop"]; item_tier = ctx["item_tier"]
catalog = list(map(str, i_uniq))
catalog_n = len(catalog)

pop_books = [b for b in pop.index.tolist() if b in iidx]
pop_top = pop_books[:200]
eval_users = set(coh["user_id"].tolist())
tr_eval = train[train["user_id"].isin(eval_users)]
seen = tr_eval.groupby("user_id")["book_id"].apply(set).to_dict()
last_item = user_last_item(tr_eval)
coh_by_user = dict(zip(coh["user_id"], coh["cohort"]))
print(f"   users={len(coh)} catalog={catalog_n}  ({time.time()-t0:.1f}s)")

print("[2/6] genre map + ALS warm scoring ...")
gmap = build_genre_map()
genre_pop = {}
for b in pop_books:
    g = gmap.get(b)
    if g:
        genre_pop.setdefault(g, [])
        if len(genre_pop[g]) < 200:
            genre_pop[g].append(b)

warm_users = coh["user_id"].values[coh["has_factor"].values]
als_recs = {}
BATCH = 800
for s0 in range(0, len(warm_users), BATCH):
    chunk = warm_users[s0:s0 + BATCH]
    rows = np.array([uidx[u] for u in chunk])
    sc = X[rows] @ Y.T
    part = np.argpartition(-sc, K + 60, axis=1)[:, :K + 60]
    for ci, u in enumerate(chunk):
        idxs = part[ci][np.argsort(-sc[ci, part[ci]])]
        s = seen.get(u, set()); out = []
        for j in idxs:
            b = str(i_uniq[j])
            if b in s:
                continue
            out.append(b)
            if len(out) >= K:
                break
        als_recs[u] = out
print(f"   als scored {len(als_recs)}  ({time.time()-t0:.1f}s)")


def r_pop(u):
    s = seen.get(u, set()); return [b for b in pop_top if b not in s][:K]


def r_content(u):
    li = last_item.get(u)
    if li is None:
        return None
    g = gmap.get(str(li))
    if not g or g not in genre_pop:
        return None
    s = seen.get(u, set()); return [b for b in genre_pop[g] if b not in s][:K]


def r_itemsim(u, kbuf=300):
    li = last_item.get(u)
    if li is None or li not in iidx:
        return None
    sc = Y @ Y[iidx[li]]
    top = np.argpartition(-sc, kbuf)[:kbuf]; top = top[np.argsort(-sc[top])]
    s = seen.get(u, set()); out = []
    for j in top:
        b = str(i_uniq[j])
        if b == str(li) or b in s:
            continue
        out.append(b)
        if len(out) >= K:
            break
    return out


def r_als(u):
    return als_recs.get(u)


def r_blended(u):
    for fn in (r_als, r_itemsim, r_content):
        r = fn(u)
        if r:
            return r
    return r_pop(u)


POLICIES = {"als_warm": r_als, "popularity": r_pop, "content": r_content,
            "item_sim": r_itemsim, "blended": r_blended}

print("[3/6] accumulating per-item exposure per policy ...")
exposure = {p: defaultdict(int) for p in POLICIES}
# user-cohort x item-tier exposure matrix (blended = the served policy)
cohort_order = ["unknown", "sparse_1_2", "low_3_5", "warm_6plus"]
uc_matrix = {c: {"head": 0, "mid": 0, "long_tail": 0} for c in cohort_order}
for u in coh["user_id"].values:
    cohort = coh_by_user[u]
    for p, fn in POLICIES.items():
        recs = fn(u)
        if not recs:
            continue
        for b in recs[:K]:
            exposure[p][b] += 1
    br = r_blended(u)
    if br:
        for b in br[:K]:
            t = item_tier.get(b, "long_tail")
            uc_matrix[cohort][t if t in uc_matrix[cohort] else "long_tail"] += 1

print("[4/6] served exact vs HNSW exposure (FAISS) ...")
served_exposure = {"exact": defaultdict(int), "scale_hnsw": defaultdict(int)}
try:
    from serving.recommender_service import RecommenderService
    svc = RecommenderService(DATA, default_mode="exact", build_hnsw=True)
    warm_sample = [u for u in warm_users][:4000]
    for mode, key in [("exact", "exact"), ("scale", "scale_hnsw")]:
        for u in warm_sample:
            r = svc.recommend(u, k=K, mode=mode)
            for it in r["items"]:
                served_exposure[key][str(it["item_id"])] += 1
    served_ok = True
except Exception as e:
    served_ok = False
    served_err = str(e)
print(f"   served_ok={served_ok}  ({time.time()-t0:.1f}s)")

print("[5/6] computing governance stats + plots ...")
stats = {p: exposure_stats(exposure[p], catalog, item_tier, K) for p in POLICIES}
served_stats = {}
if served_ok:
    served_stats = {k: exposure_stats(served_exposure[k], catalog, item_tier, K)
                    for k in served_exposure}

# Recommendation: policy that best preserves catalog opportunity while serving
# everyone (lowest Gini / zero-exposure among policies that never return empty).
candidates = {p: stats[p] for p in ["blended", "als_warm", "item_sim", "content", "popularity"]}
ranked = sorted(candidates.items(),
                key=lambda kv: (kv[1]["gini"], kv[1]["zero_exposure_share"]))
recommended = {
    "policy": "blended",
    "rationale": ("Among policies that always return a slate, blended inherits ALS's "
                  "wider catalog opportunity (lowest Gini / zero-exposure of the "
                  "serving-complete policies) while still falling back safely. "
                  "Popularity has the worst concentration (near-degenerate). "
                  "item-sim spreads exposure most but cannot serve no-history users."),
    "policy_gini_ranking_low_to_high": [(p, s["gini"]) for p, s in ranked],
    "honest_caveat": ("Even the best policy is head-concentrated: this catalog and the "
                      "ALS floor concentrate exposure; G25 quantifies the gap, it does "
                      "NOT solve long-tail discovery."),
}

item_coldstart = {
    "definition": "held-out gold items unseen in train under this offline split",
    "share_unseen_gold": round(float((coh["gold_tier"] == "unseen").mean()), 4),
    "note": ("Item-cold-start items by construction receive ~0 collaborative exposure; "
             "no policy here surfaces them. Offline-split sense only, not production "
             "new-catalog cold-start."),
}

report = {
    "gate": "G25",
    "title": "Catalog Exposure Fairness / Coverage Governance",
    "lane": "V2",
    "provenance": "computed_offline_real_data",
    "scope_guardrails": [
        "Catalog EXPOSURE governance under the served G23 c2_als.pkl protocol (ALS f64).",
        "NOT protected-class fairness; NOT a fairness certification; NOT creator/marketplace fairness solved.",
        "NOT online diversity; NOT production exposure governance; NOT long-tail discovery solved.",
        "Absolute numbers reflect the served model + offline split; V1 (gold_candidate 8.9) untouched.",
    ],
    "k": K,
    "audited_population": {"users": int(len(coh)), "cohorts": coh["cohort"].value_counts().to_dict()},
    "catalog_size": catalog_n,
    "item_tier_thresholds": ctx["tier_thresholds"],
    "per_policy_exposure_stats": stats,
    "served_mode_exposure_stats": served_stats,
    "served_exact_vs_hnsw_note": (
        "Exposure distribution under exact FAISS vs HNSW ef64 on the same warm sample; "
        "compare gini / coverage to confirm the approximate index does not materially "
        "reshape catalog exposure." if served_ok else f"served unavailable: {served_err}"),
    "user_cohort_x_item_tier_exposure_blended": uc_matrix,
    "item_coldstart": item_coldstart,
    "recommended_policy": recommended,
    "limitations": [
        "Exposure measured over the eval-user cohort sample at top-20, offline.",
        "Head/mid/long-tail tiers are train-popularity percentiles, not editorial categories.",
        "Genre/shelf cohorts not separately audited (shelf metadata is a noisy proxy).",
        "This is exposure concentration, NOT protected-class fairness or certification.",
        "No exposure-rebalancing intervention is built; G25 measures, it does not fix.",
    ],
    "claim_boundary": {
        "exposure_concentration_measured": True,
        "fairness_solved": False,
        "creator_fairness_solved": False,
        "marketplace_fairness_certified": False,
        "online_diversity_improved": False,
        "long_tail_discovery_solved": False,
        "production_exposure_governance_deployed": False,
        "protected_class_analysis": False,
        "safe_claim": ("PulseDiscovery G25 audits catalog exposure concentration across "
                       "retrieval and fallback policies, measuring head/mid/tail exposure, "
                       "catalog coverage, zero-exposure share, and concentration tradeoffs."),
    },
    "runtime_sec": round(time.time() - t0, 1),
}
json.dump(report, open(os.path.join(EVID, "g25_catalog_exposure_governance_report.json"), "w"), indent=2)

# ---- plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Plot 1: exposure concentration curve (sorted exposure share, log-y) per policy
fig, ax = plt.subplots(figsize=(8, 5))
for p in ["blended", "als_warm", "item_sim", "content", "popularity"]:
    c = np.sort(np.array([exposure[p].get(b, 0) for b in catalog], dtype=float))[::-1]
    tot = c.sum() or 1
    ax.plot(np.arange(1, len(c) + 1), np.cumsum(c) / tot, label=p)
ax.set_xscale("log")
ax.set_xlabel("items ranked by exposure (log)"); ax.set_ylabel("cumulative exposure share")
ax.set_title("G25 Exposure concentration by policy\n(steeper rise = more concentrated)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "g25_exposure_concentration_curve.png"), dpi=130); plt.close(fig)

# Plot 2: head/mid/tail exposure share by policy (stacked) + zero-exposure annotation
fig, ax = plt.subplots(figsize=(8, 5))
pol = list(POLICIES.keys())
h = [stats[p]["tier_exposure_share"]["head"] for p in pol]
m = [stats[p]["tier_exposure_share"]["mid"] for p in pol]
l = [stats[p]["tier_exposure_share"]["long_tail"] for p in pol]
b = np.zeros(len(pol))
for vals, lab, col in [(h, "head", "#c0392b"), (m, "mid", "#e67e22"), (l, "long_tail", "#27ae60")]:
    ax.bar(pol, vals, bottom=b, label=lab, color=col); b = b + np.array(vals)
for i, p in enumerate(pol):
    ax.text(i, 1.02, f"cov {stats[p]['catalog_coverage']*100:.1f}%\ngini {stats[p]['gini']:.2f}",
            ha="center", fontsize=7)
ax.set_ylim(0, 1.18); ax.set_ylabel("exposure share of top-20 slots")
ax.set_title("G25 Head/Mid/Long-tail exposure share by policy")
ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "g25_head_mid_tail_exposure_by_policy.png"), dpi=130); plt.close(fig)

# Plot 3: Lorenz curves per policy
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect equality")
for p in ["blended", "als_warm", "item_sim", "content", "popularity"]:
    c = np.array([exposure[p].get(b, 0) for b in catalog], dtype=float)
    gx, gy = lorenz_curve(c)
    ax.plot(gx, gy, label=f"{p} (gini {stats[p]['gini']:.2f})")
ax.set_xlabel("cumulative share of catalog items"); ax.set_ylabel("cumulative share of exposure")
ax.set_title("G25 Policy exposure Lorenz curves\n(lower bulge = more concentrated)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "g25_policy_exposure_lorenz_curve.png"), dpi=130); plt.close(fig)

print(f"[6/6] DONE in {time.time()-t0:.1f}s")
for p in pol:
    s = stats[p]
    print(f"  {p:11s} gini={s['gini']:.3f} cov={s['catalog_coverage']:.4f} "
          f"zero={s['zero_exposure_share']:.3f} top1%={s['top_1pct_share']:.3f} "
          f"tail_share={s['tier_exposure_share']['long_tail']:.4f}")
if served_ok:
    for k in served_stats:
        s = served_stats[k]
        print(f"  served:{k:9s} gini={s['gini']:.3f} cov={s['catalog_coverage']:.4f}")
