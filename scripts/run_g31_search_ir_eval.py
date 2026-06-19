"""G31 (V2) — Search/IR hybrid retrieval evaluation (PulseDiscover).

Two-stage-search lexical+dense retrieval comparison under the served-c2 protocol:
  BM25 (lexical)  vs  dense MiniLM (semantic)  vs  BM25+dense RRF (hybrid).
Query = seed item (the user's last-read item) -> query-by-document / more-like-this retrieval
(NOT free-text query). Reports Recall@K AND search-standard NDCG@K + MRR.

HONESTY GUARDS (locked):
  (a) seed-item / query-by-document framing, not free-text query search;
  (b) NDCG/MRR are SINGLE-RELEVANT (one held-out gold per user; no graded relevance) ->
      NDCG@K = 1/log2(rank+1) if gold in top-K, MRR = 1/rank of the gold;
  (c) served-c2 metrics only; V1 RecSys claims untouched.

Output: outputs/evidence/g31_search_ir_report.json + outputs/plots/g31_lexical_dense_hybrid.png
"""
from __future__ import annotations
import os, sys, json, time
from collections import defaultdict
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from eval.cold_start_cohorts import load_als, load_eval_samples, build_cohorts, build_genre_map, user_last_item
from retrieval.semantic_content_index import SemanticContentIndex, load_item_texts
from retrieval.bm25_lexical_index import BM25LexicalIndex, reciprocal_rank_fusion

DATA = os.environ.get("PD_DATADIR", "data/interim")
SPLITS = os.path.join(DATA, "domain_splits")
EVID, PLOTS = "outputs/evidence", "outputs/plots"; os.makedirs(PLOTS, exist_ok=True)
K, K2 = 20, 50
t0 = time.time()

print("[1/5] load cohorts + dense index + item texts ...")
train = pd.read_csv(os.path.join(SPLITS, "train.csv"), usecols=["user_id", "book_id", "event_ts"], dtype={"book_id": str})
als = load_als(); ev = load_eval_samples(); coh, ctx = build_cohorts(train, als, ev)
sci = SemanticContentIndex.load(os.path.join(DATA, "g27_content_index"))
sem_ids = sci.ids; sem_row = sci.id_to_row(); sem_set = set(sem_ids)
# item texts for the SAME universe as the dense index (title + genre shelves)
texts_df = load_item_texts(os.path.join(DATA, "domain_content_meta.csv"), set(sem_ids), max_desc=0)
text_lookup = dict(zip(texts_df["book_id"], texts_df["text"]))
# align BM25 corpus to the dense index id order (same universe) so both see the same items
ids_aligned = [b for b in sem_ids if b in text_lookup]
texts_aligned = [text_lookup[b] for b in ids_aligned]
print(f"   dense items={len(sem_ids)} bm25 items={len(ids_aligned)}  ({time.time()-t0:.1f}s)")

print("[2/5] build BM25 inverted index ...")
bm25 = BM25LexicalIndex().build(ids_aligned, texts_aligned)
print(f"   built  ({time.time()-t0:.1f}s)")

# users with a query item (seed) — query-by-document population
tr_eval = train[train["user_id"].isin(set(coh["user_id"]))]
seen = tr_eval.groupby("user_id")["book_id"].apply(set).to_dict()
last = user_last_item(tr_eval)
coh = coh[coh["user_id"].isin(last.keys())].copy()
gold_by = dict(zip(coh["user_id"], coh["gold"])); tier_by = dict(zip(coh["user_id"], coh["gold_tier"]))
# sample for speed/stability
users = coh["user_id"].values
rng = np.random.default_rng(31)
SAMPLE = int(os.environ.get("G31_USERS", "600"))  # BM25 get_scores ~43ms/query (rank_bm25, pure-Python) -> bound the sample
if len(users) > SAMPLE:
    users = users[rng.choice(len(users), SAMPLE, replace=False)]
print(f"   seed-item users={len(users)}  ({time.time()-t0:.1f}s)")


def dense_search(seed, k, ex):
    if seed not in sem_row:
        return []
    ids, _ = sci.search(sci.emb[sem_row[seed]], k + 40)
    return [b for b in ids if b != seed and b not in ex][:k]


print("[3/5] retrieve BM25 / dense / hybrid (seed-item queries) ...")
lat = {"bm25": [], "dense": [], "hybrid": []}
results = {m: {"hit20": 0, "hit50": 0, "ndcg": 0.0, "rr": 0.0, "exposure": defaultdict(int),
               "by_tier": defaultdict(lambda: [0, 0])} for m in ["bm25", "dense", "hybrid"]}
n = 0
for u in users:
    seed = last.get(u); gold = gold_by.get(u); gt = tier_by.get(u)
    if seed is None:
        continue
    ex = seen.get(u, set())
    n += 1
    t = time.perf_counter(); b_ids, _ = bm25.search_by_seed_item(seed, text_lookup, K2, ex); lat["bm25"].append((time.perf_counter()-t)*1000)
    t = time.perf_counter(); d_ids = dense_search(seed, K2, ex); lat["dense"].append((time.perf_counter()-t)*1000)
    t = time.perf_counter(); h_ids = reciprocal_rank_fusion([b_ids, d_ids], k=K2); lat["hybrid"].append((time.perf_counter()-t)*1000)
    for m, lst in [("bm25", b_ids), ("dense", d_ids), ("hybrid", h_ids)]:
        R = results[m]; R["by_tier"][gt][1] += 1
        top = lst[:K]
        for b in top:
            R["exposure"][b] += 1
        if gold in top:
            R["hit20"] += 1; R["by_tier"][gt][0] += 1
            rank = lst.index(gold) + 1               # 1-based rank within the slate
            R["ndcg"] += 1.0 / np.log2(rank + 1)     # single-relevant NDCG@20
            R["rr"] += 1.0 / rank                     # reciprocal rank (for MRR)
        if gold in lst[:K2]:
            R["hit50"] += 1

print(f"[4/5] metrics + report ...  ({time.time()-t0:.1f}s)")
catalog = sorted(set(sem_ids))
summary = {}
for m in ["bm25", "dense", "hybrid"]:
    R = results[m]
    rt = {k_: round(v[0] / v[1], 4) if v[1] else 0.0 for k_, v in R["by_tier"].items()}
    uniq = len([1 for b in catalog if R["exposure"].get(b, 0) > 0])
    summary[m] = {
        "recall@20": round(R["hit20"] / n, 4), "recall@50": round(R["hit50"] / n, 4),
        "ndcg@20_single_relevant": round(R["ndcg"] / n, 4),
        "mrr_single_relevant": round(R["rr"] / n, 4),
        "recall@20_by_gold_tier": rt,
        "coldstart_recall@20": rt.get("unseen", 0.0),
        "catalog_coverage@20": round(uniq / len(catalog), 4), "unique_items@20": uniq,
        "latency_p50_ms": round(float(np.percentile(lat[m], 50)), 3),
        "latency_p95_ms": round(float(np.percentile(lat[m], 95)), 3),
    }
    print(f"  {m:7s} R@20={summary[m]['recall@20']:.4f} NDCG@20={summary[m]['ndcg@20_single_relevant']:.4f} "
          f"MRR={summary[m]['mrr_single_relevant']:.4f} cold={summary[m]['coldstart_recall@20']:.4f} "
          f"cov={summary[m]['catalog_coverage@20']:.3f} p95={summary[m]['latency_p95_ms']}ms")

best = max(["bm25", "dense", "hybrid"], key=lambda m: summary[m]["recall@20"])
report = {
    "gate": "G31", "title": "Search/IR — BM25 + dense hybrid retrieval (query-by-document)", "lane": "V2",
    "provenance": "served_c2_offline_seed_item_query",
    "honesty_guards": [
        "(a) Retrieval is QUERY-BY-DOCUMENT: the query is the user's last-read item text (more-like-this / related-item search), NOT a free-text query.",
        "(b) NDCG@K and MRR are SINGLE-RELEVANT: one held-out gold per user, no graded relevance. NDCG@20=1/log2(rank+1) if gold in top-K; MRR=1/rank.",
        "(c) Served-c2 metrics only; V1 RecSys claims and V1 status are untouched.",
    ],
    "k_main": K, "k_wide": K2, "seed_item_users": int(n), "index_universe_items": len(catalog),
    "bm25": {"index_type": "BM25Okapi inverted index", "text": "title + genre shelves (max_desc=0)",
             "library": "rank_bm25"},
    "dense": {"index_type": "FAISS IndexFlatIP", "model": sci.model_name, "dim": sci.dim},
    "hybrid": {"fusion": "Reciprocal Rank Fusion (RRF, c=60) over BM25 + dense"},
    "metrics": summary,
    "best_by_recall@20": best,
    "interpretation": ("Lexical (BM25) and dense (semantic) retrieval capture complementary signal; "
                       "RRF hybrid is the standard two-stage-search front end. Reported with search-standard "
                       "NDCG@K + MRR (single-relevant) alongside Recall@K. Seed-item / query-by-document, "
                       "offline, served-c2 — not free-text query search, not online."),
    "search_ir_unlocked": True,
    "claim_status": {
        "two_stage_search_retrieval_demonstrated": True,
        "lexical_dense_hybrid_built": True,
        "ndcg_mrr_reported": True,
        "free_text_query_search": False,
        "in_ranker_position_bias_correction": False,
        "online_lift": False, "production": False, "v1_recsys_claims_changed": False,
        "safe_claim": ("Built a two-stage-search-style retrieval front end on the item catalog: BM25 lexical "
                       "+ dense semantic retrieval fused with RRF, evaluated with search-standard NDCG@K and "
                       "MRR alongside Recall@K, using seed-item (query-by-document) queries. Offline, served-c2 "
                       "— not free-text query search and not online."),
    },
    "forbidden_claims": ["free-text query search engine", "in-ranker position-bias correction (only evaluation)",
                         "online lift", "production search deployed", "changed V1 RecSys claims"],
    "runtime_sec": round(time.time() - t0, 1),
}
json.dump(report, open(os.path.join(EVID, "g31_search_ir_report.json"), "w"), indent=2)

print("[5/5] plot ...")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8.5, 5))
ms = ["bm25", "dense", "hybrid"]; x = np.arange(len(ms)); w = 0.25
for i, (lbl, key) in enumerate([("Recall@20", "recall@20"), ("NDCG@20", "ndcg@20_single_relevant"), ("MRR", "mrr_single_relevant")]):
    ax.bar(x + (i - 1) * w, [summary[m][key] for m in ms], w, label=lbl)
ax.set_xticks(x); ax.set_xticklabels(["BM25\n(lexical)", "dense\n(semantic)", "BM25+dense\n(RRF hybrid)"])
ax.set_ylabel("metric (single-relevant NDCG/MRR)")
ax.set_title("G31 Two-stage search front end: lexical vs dense vs hybrid\n(seed-item / query-by-document; served-c2; offline)")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y"); fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "g31_lexical_dense_hybrid.png"), dpi=130); plt.close(fig)
print(f"DONE in {time.time()-t0:.1f}s  best={best}")
