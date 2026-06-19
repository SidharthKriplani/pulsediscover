"""G28 (Gold Pass 1/3) — Candidate feature-table builder (PulseDiscover V2).

One row per (user, item candidate) from the union of ALS (served c2) + semantic (G27) +
popularity candidate sources, with source flags, scores/ranks, item tier, label (held-out
gold), and a leakage-safe train/val/test split BY USER. Persisted for the fusion baselines
and the learned ranker tournament.

Leakage rules: split by user (a user is wholly in one split); test labels never used for
training/selection; served-c2 protocol only (never mixed with V1 d3aplus).

Outputs:
  data/interim/g28_candidate_table.parquet
  outputs/evidence/g28_candidate_feature_table_summary.json
"""
from __future__ import annotations
import os, sys, json, time, hashlib
from collections import defaultdict
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from eval.cold_start_cohorts import load_als, load_eval_samples, build_cohorts, user_last_item
from retrieval.semantic_content_index import SemanticContentIndex

DATA = os.environ.get("PD_DATADIR", "data/interim")
SPLITS = os.path.join(DATA, "domain_splits")
EVID = "outputs/evidence"; os.makedirs(EVID, exist_ok=True)
N_ALS, N_SEM, N_POP = 50, 50, 20   # candidates per source per user
t0 = time.time()

print("[1/4] load cohorts + ALS + semantic ...")
train = pd.read_csv(os.path.join(SPLITS, "train.csv"), usecols=["user_id", "book_id", "event_ts"], dtype={"book_id": str})
als = load_als(); ev = load_eval_samples()
coh, ctx = build_cohorts(train, als, ev)
X = als["X"].astype(np.float32); Y = als["Y"].astype(np.float32)
i_uniq = list(map(str, als["i_uniq"])); iidx = als["iidx"]; uidx = als["uidx"]
als_catalog = set(i_uniq); item_tier = ctx["item_tier"]; pop = ctx["pop"]
pop_books = [b for b in pop.index.tolist() if b in iidx]; pop_top = pop_books[:N_POP]
pop_rank = {b: (len(pop) - i) / len(pop) for i, b in enumerate(pop.index)}

tr_eval = train[train["user_id"].isin(set(coh["user_id"]))]
seen = tr_eval.groupby("user_id")["book_id"].apply(set).to_dict()
last = user_last_item(tr_eval)
sci = SemanticContentIndex.load(os.path.join(DATA, "g28_content_index")) if os.path.exists(os.path.join(DATA, "g28_content_index_meta.json")) else SemanticContentIndex.load(os.path.join(DATA, "g27_content_index"))
sem_row = sci.id_to_row(); sem_ids = sci.ids; sem_cat = set(sem_ids)

# restrict to users with a last item (semantic-queryable) — the fusion-relevant population
coh = coh[coh["user_id"].isin(last.keys())].reset_index(drop=True)
gold_by = dict(zip(coh["user_id"], coh["gold"])); tier_by = dict(zip(coh["user_id"], coh["gold_tier"]))
cohort_by = dict(zip(coh["user_id"], coh["cohort"]))
users = coh["user_id"].values
print(f"   users={len(users)}  ({time.time()-t0:.1f}s)")

# ---- ALS candidates + scores (batch)
print("[2/4] ALS + semantic candidates ...")
als_cand = {}
warm_u = coh[coh["has_factor"]]["user_id"].values
for s0 in range(0, len(warm_u), 800):
    ch = warm_u[s0:s0 + 800]; rows = np.array([uidx[u] for u in ch])
    sc = X[rows] @ Y.T; part = np.argpartition(-sc, N_ALS + 40, axis=1)[:, :N_ALS + 40]
    for ci, u in enumerate(ch):
        order = part[ci][np.argsort(-sc[ci, part[ci]])]; s = seen.get(u, set()); lst = []
        for j in order:
            b = i_uniq[j]
            if b in s:
                continue
            lst.append((b, float(sc[ci, j])))
            if len(lst) >= N_ALS:
                break
        als_cand[u] = lst

# ---- semantic candidates (batched) + scores
q_users = [u for u in users if last.get(u) in sem_row]
q_rows = np.array([sem_row[last[u]] for u in q_users])
S, I = sci.index.search(np.ascontiguousarray(sci.emb[q_rows], dtype=np.float32), N_SEM + 40)
sem_cand = {}
for qi, u in enumerate(q_users):
    li = last[u]; s = seen.get(u, set()); lst = []
    for rank, j in enumerate(I[qi]):
        b = sem_ids[j]
        if b == li or b in s:
            continue
        lst.append((b, float(S[qi][rank])))
        if len(lst) >= N_SEM:
            break
    sem_cand[u] = lst
print(f"   als_users={len(als_cand)} sem_users={len(sem_cand)}  ({time.time()-t0:.1f}s)")

# ---- assemble candidate rows
print("[3/4] assembling candidate table ...")
# leakage-safe split by user hash
def split_of(u):
    h = int(hashlib.md5(u.encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else ("val" if h < 85 else "test")

rows = []
for u in users:
    g = gold_by[u]; cohort = cohort_by[u]; sp = split_of(u)
    a = dict(als_cand.get(u, [])); sm = dict(sem_cand.get(u, []))
    a_rank = {b: r + 1 for r, (b, _) in enumerate(als_cand.get(u, []))}
    s_rank = {b: r + 1 for r, (b, _) in enumerate(sem_cand.get(u, []))}
    cand = set(a) | set(sm) | set(pop_top)
    for b in cand:
        tier = item_tier.get(b, "unseen")
        rows.append({
            "user_id": u, "item_id": b, "cohort": cohort, "split": sp,
            "from_als": int(b in a), "from_semantic": int(b in sm), "from_popularity": int(b in pop_top),
            "als_score": a.get(b, 0.0), "als_rank": a_rank.get(b, 0),
            "sem_score": sm.get(b, 0.0), "sem_rank": s_rank.get(b, 0),
            "pop_rank_pct": pop_rank.get(b, 0.0),
            "is_head": int(tier == "head"), "is_mid": int(tier == "mid"),
            "is_long_tail": int(tier == "long_tail"), "is_coldstart": int(b not in als_catalog),
            "in_semantic_index": int(b in sem_cat),
            "n_sources": int(b in a) + int(b in sm) + int(b in pop_top),
            "gold_tier": tier_by[u],
            "label": int(b == g),
        })
df = pd.DataFrame(rows)
df.to_parquet(os.path.join(DATA, "g28_candidate_table.parquet"))

# ---- summary
def src_bucket(r):
    if r["from_als"] and r["from_semantic"]:
        return "both"
    if r["from_als"]:
        return "als_only"
    if r["from_semantic"]:
        return "semantic_only"
    return "popularity_only"
df["_bucket"] = df.apply(src_bucket, axis=1)
cpu = df.groupby("user_id").size()
summary = {
    "gate": "G28", "pass": "Gold Pass 1/3", "provenance": "served_c2_offline",
    "candidate_rows": int(len(df)), "users": int(df["user_id"].nunique()),
    "candidates_per_user": {"mean": round(float(cpu.mean()), 1), "p50": int(cpu.median()),
                            "p90": int(cpu.quantile(0.9)), "max": int(cpu.max())},
    "source_overlap": {k: int(v) for k, v in df["_bucket"].value_counts().items()},
    "label_positives": int(df["label"].sum()), "label_sparsity": round(float(df["label"].mean()), 6),
    "users_with_a_positive_candidate": int(df.groupby("user_id")["label"].max().sum()),
    "coldstart_candidate_share": round(float(df["is_coldstart"].mean()), 4),
    "tier_distribution": {k: int(v) for k, v in df.assign(t=np.where(df.is_coldstart==1,"coldstart",
        np.where(df.is_head==1,"head",np.where(df.is_mid==1,"mid","long_tail"))))["t"].value_counts().items()},
    "split_users": {s: int(df[df.split==s]["user_id"].nunique()) for s in ["train","val","test"]},
    "split_rows": {s: int((df.split==s).sum()) for s in ["train","val","test"]},
    "leakage_controls": ["split by user-hash (user wholly in one split)", "test labels unused for train/selection",
                         "served-c2 protocol only; not mixed with V1 d3aplus"],
    "n_per_source": {"als": N_ALS, "semantic": N_SEM, "popularity": N_POP},
    "runtime_sec": round(time.time() - t0, 1),
}
json.dump(summary, open(os.path.join(EVID, "g28_candidate_feature_table_summary.json"), "w"), indent=2)
print(f"[4/4] DONE rows={len(df)} users={df['user_id'].nunique()} pos={df['label'].sum()} "
      f"users_w_pos={summary['users_with_a_positive_candidate']}  ({time.time()-t0:.1f}s)")
print("overlap:", summary["source_overlap"], "| split_users:", summary["split_users"])
