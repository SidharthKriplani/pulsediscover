"""G32 prep — build two-tower training/eval arrays that reproduce the EXACT ALS-floor
protocol (src/d3b_als_floor.py): train on domain_core_interactions minus heldout golds,
evaluate leave-last-out on the 5,000-user eval_sample.csv (warm) and cold_eval_sample.csv (cold).

Item universe = unique books in the training interactions (same set ALS factorizes over),
optionally augmented with cold gold items that have a MiniLM content embedding (so the content
tower can retrieve cold items ALS/pop can never reach — the cold-lane comparison vs V2's 0.241).

Pure numpy/pandas — no torch. Outputs data/interim/g32_prep.npz + g32_prep_meta.json.
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np, pandas as pd

DATA = "data/interim"; SP = os.path.join(DATA, "domain_splits")
t0 = time.time()

# ---- 1. training interactions = core minus heldout golds (identical to ALS floor) ----
core = pd.read_csv(os.path.join(DATA, "domain_core_interactions.csv"),
                   usecols=["user_id", "book_id"], dtype=str)
held = pd.read_csv(os.path.join(SP, "heldout_eval.csv"), dtype=str)
mask = (core.user_id + "|" + core.book_id).isin(set(held.user_id + "|" + held.gold))
tr = core[~mask][["user_id", "book_id"]].reset_index(drop=True)
del core
print(f"[1] train interactions={len(tr):,}  after removing {int(mask.sum()):,} heldout golds")

# ---- 2. content embeddings (MiniLM, g27) keyed by book_id ----
c_emb = np.load(os.path.join(DATA, "g27_content_index_emb.npy"))          # (41866, 384)
c_meta = json.load(open(os.path.join(DATA, "g27_content_index_meta.json")))
c_ids = [str(x) for x in c_meta["ids"]]
cid2row = {b: r for r, b in enumerate(c_ids)}
CDIM = int(c_meta["dim"])
print(f"[2] content emb {c_emb.shape}  dim={CDIM}")

# ---- 3. item universe: training items (+ cold golds that have content) ----
train_items = pd.unique(tr.book_id)
cold = pd.read_csv(os.path.join(SP, "cold_eval_sample.csv"), dtype=str)
cold_golds_with_content = [g for g in pd.unique(cold.gold) if g in cid2row and g not in set(train_items)]
# universe order: warm training items first (ids 0..nWarm-1), then cold-only content items
universe = list(train_items) + list(cold_golds_with_content)
i2c = {b: i for i, b in enumerate(universe)}
nI = len(universe); nWarm = len(train_items)
print(f"[3] item universe={nI:,} (warm train items={nWarm:,}, cold-only w/ content={len(cold_golds_with_content):,})")

# content matrix aligned to universe (zeros where no embedding)
CMAT = np.zeros((nI, CDIM), dtype=np.float32)
have_content = np.zeros(nI, dtype=bool)
for b, i in i2c.items():
    r = cid2row.get(b)
    if r is not None:
        CMAT[i] = c_emb[r]; have_content[i] = True
print(f"    items with content vector: {int(have_content.sum()):,} / {nI:,}")

# ---- 4. user factorization + histories (training item codes per user) ----
u_uniq, u_codes = np.unique(tr.user_id.values, return_inverse=True)
uidx = {u: i for i, u in enumerate(u_uniq)}
item_codes = np.array([i2c[b] for b in tr.book_id.values], dtype=np.int64)
nU = len(u_uniq)
order = np.argsort(u_codes, kind="stable")
hist_split = np.split(item_codes[order], np.flatnonzero(np.diff(u_codes[order])) + 1)
# ragged -> object array of int32 arrays
hist = np.empty(nU, dtype=object)
for uc in range(nU):
    hist[uc] = hist_split[uc].astype(np.int32)
print(f"[4] users={nU:,}  interactions(pairs)={len(item_codes):,}")

# ---- 5. training pairs (user_code, pos_item_code) ----
pairs = np.stack([u_codes[order].astype(np.int64), item_codes[order].astype(np.int64)], axis=1)

# ---- 6. eval rows: map eval users -> user_code, gold -> item_code (may be -1 if OOV) ----
def build_eval(fname):
    df = pd.read_csv(os.path.join(SP, fname), dtype=str)
    urows = np.array([uidx.get(u, -1) for u in df.user_id], dtype=np.int64)
    grows = np.array([i2c.get(g, -1) for g in df.gold], dtype=np.int64)
    return urows, grows

warm_u, warm_g = build_eval("eval_sample.csv")
cold_u, cold_g = build_eval("cold_eval_sample.csv")
print(f"[6] warm eval: users_found={int((warm_u>=0).sum())}/{len(warm_u)}, gold_in_universe={int((warm_g>=0).sum())}")
print(f"    cold eval: users_found={int((cold_u>=0).sum())}/{len(cold_u)}, gold_in_universe={int((cold_g>=0).sum())} (of which cold-only={int((cold_g>=nWarm).sum())})")

# ---- 7. save ----
np.savez_compressed(os.path.join(DATA, "g32_prep.npz"),
                    pairs=pairs, CMAT=CMAT, have_content=have_content,
                    warm_u=warm_u, warm_g=warm_g, cold_u=cold_u, cold_g=cold_g,
                    hist=hist,
                    universe=np.array(universe, dtype=object),          # item_code -> book_id
                    eval_users=np.array(u_uniq, dtype=object))          # user_code -> user_id
meta = {"nU": int(nU), "nI": int(nI), "nWarm": int(nWarm), "cdim": CDIM,
        "n_pairs": int(len(pairs)), "n_items_with_content": int(have_content.sum()),
        "als_floor_R@20": 0.0846, "protocol": "eval_sample.csv leave-last-out, top-200, history-masked",
        "sec": round(time.time() - t0, 1)}
json.dump(meta, open(os.path.join(DATA, "g32_prep_meta.json"), "w"), indent=2)
print("[7] saved g32_prep.npz + meta:", meta)
