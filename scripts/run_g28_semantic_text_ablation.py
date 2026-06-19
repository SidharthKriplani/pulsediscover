"""G28 — Semantic text ablation (cold-start subset, restricted-pool A/B).

G27 embedded title+shelves only (build-time budget). This tests whether richer text
(title+shelves+description) improves cold-start retrieval, on a bounded cold-start subset to
stay within the sandbox. Restricted-pool A/B: same item set under both recipes; recipe A reuses
the existing G27 title+shelves embeddings (sliced); recipe B re-embeds with description.

Output: outputs/evidence/g28_semantic_text_ablation.json
"""
from __future__ import annotations
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "8"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from eval.cold_start_cohorts import load_als, load_eval_samples, build_cohorts, user_last_item
from retrieval.semantic_content_index import SemanticContentIndex, load_item_texts, build_item_text
DATA = os.environ.get("PD_DATADIR", "data/interim"); SPLITS = os.path.join(DATA, "domain_splits")
EVID = "outputs/evidence"; K = 20; MAX_ITEMS = int(os.environ.get("G28_ABL_ITEMS", "1200")); t0 = time.time()

print("[1/4] gather cold-start (query,last,gold) triples ...")
train = pd.read_csv(os.path.join(SPLITS, "train.csv"), usecols=["user_id", "book_id", "event_ts"], dtype={"book_id": str})
als = load_als(); ev = load_eval_samples(); coh, ctx = build_cohorts(train, als, ev)
als_cat = set(map(str, als["i_uniq"]))
tr_eval = train[train["user_id"].isin(set(coh["user_id"]))]
last = user_last_item(tr_eval)
coh = coh[(coh["gold_tier"] == "unseen") & (coh["user_id"].isin(last.keys()))].drop_duplicates("user_id")
gold_by = dict(zip(coh["user_id"], coh["gold"]))
triples = [(u, last[u], gold_by[u]) for u in coh["user_id"] if last.get(u) is not None]
# cap by number of triples so BOTH query item and gold stay in the pool (budget)
rng = np.random.default_rng(0)
if len(triples) * 2 > MAX_ITEMS:
    rng.shuffle(triples); triples = triples[:MAX_ITEMS // 2]
items = sorted({li for _, li, _ in triples} | {g for _, _, g in triples})
itemset = set(items)
triples = [(u, li, g) for (u, li, g) in triples if li in itemset and g in itemset]
print(f"   triples={len(triples)} items={len(items)}  ({time.time()-t0:.1f}s)")

# recipe A: slice existing g27 (title+shelves) embeddings
sci = SemanticContentIndex.load(os.path.join(DATA, "g27_content_index"))
row = sci.id_to_row(); have = [b for b in items if b in row]
embA = np.stack([sci.emb[row[b]] for b in have]); idxA = {b: i for i, b in enumerate(have)}

# recipe B: re-embed same items with title+shelves+desc[:200]
print("[2/4] embedding recipe B (title+shelves+desc) ...")
cm = load_item_texts(os.path.join(DATA, "domain_content_meta.csv"), set(have), max_desc=200).set_index("book_id")
textsB = [build_item_text(*( ["", "", ""])) if b not in cm.index else cm.loc[b, "text"] for b in have]
embB, modelB, dimB = SemanticContentIndex._embed_transformer(textsB)
print(f"   embedded {len(have)} items recipe B dim={dimB}  ({time.time()-t0:.1f}s)")


def recall_at_k(emb, idx):
    import faiss
    index = faiss.IndexFlatIP(emb.shape[1]); index.add(np.ascontiguousarray(emb, dtype=np.float32))
    hit = 0; n = 0
    for u, li, g in triples:
        if li not in idx or g not in idx:
            continue
        n += 1
        q = emb[idx[li]:idx[li] + 1]
        _, I = index.search(np.ascontiguousarray(q, dtype=np.float32), K + 1)
        got = [have[j] for j in I[0] if 0 <= j < len(have) and have[j] != li][:K]
        if g in got:
            hit += 1
    return round(hit / max(n, 1), 4), n


print("[3/4] restricted-pool recall A vs B ...")
rA, nA = recall_at_k(embA, idxA)
rB, nB = recall_at_k(embB, idxA)  # same idx mapping/order

report = {
    "gate": "G28", "section": "semantic_text_ablation", "type": "restricted_pool_coldstart_AB",
    "note": ("Restricted-pool A/B on a bounded cold-start subset (sandbox budget). Recall is within "
             "this item pool, not full-catalog; measures the TEXT-RECIPE effect, not absolute recall."),
    "subset": {"triples_evaluated": nA, "item_pool": len(have), "cap": MAX_ITEMS},
    "recipe_A_title_shelves": {"text": "title + genre shelves", "coldstart_recall@20_restricted": rA},
    "recipe_B_title_shelves_desc": {"text": "title + genre shelves + desc[:200]", "embedding_model": modelB,
                                     "coldstart_recall@20_restricted": rB},
    "delta_recall": round(rB - rA, 4),
    "verdict": ("description_helps" if rB > rA + 0.005 else
                "description_neutral_or_marginal" if abs(rB - rA) <= 0.005 else "description_hurts"),
    "recommendation": ("Patch G27 to title+shelves+desc in Pass 2 if delta is positive and worth the ~3x "
                       "embedding cost; otherwise keep the cheaper title+shelves recipe."),
    "runtime_sec": round(time.time() - t0, 1),
}
json.dump(report, open(os.path.join(EVID, "g28_semantic_text_ablation.json"), "w"), indent=2)
print(f"[4/4] DONE A={rA} B={rB} delta={report['delta_recall']} verdict={report['verdict']}  ({time.time()-t0:.1f}s)")
