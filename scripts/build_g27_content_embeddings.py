"""G27 — Resumable dense content-embedding builder (PulseDiscover V2).

Embeds the item universe (served ALS catalog ∪ eval gold items) from metadata text using
sentence-transformers MiniLM, in resumable chunks (sandbox-friendly). Each run embeds the
next missing chunks for ~CHUNK_BUDGET items, persists them, and assembles the final index
once all chunks exist. Falls back to TF-IDF+SVD only on a documented transformer failure.

Run repeatedly until it prints ASSEMBLED. Outputs:
  data/interim/g27_emb/part_*.npy           (chunk embeddings)
  data/interim/g27_content_index_{emb.npy,_meta.json}  (final index)
  data/interim/g27_item_text_audit.json     (metadata coverage audit)
"""
from __future__ import annotations
import os, sys, json, time, glob
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")        # model is cached; skip network on load
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from eval.cold_start_cohorts import load_als, load_eval_samples
from retrieval.semantic_content_index import load_item_texts, SemanticContentIndex

DATA = os.environ.get("PD_DATADIR", "data/interim")
EMB_DIR = os.path.join(DATA, "g27_emb"); os.makedirs(EMB_DIR, exist_ok=True)
PREFIX = os.path.join(DATA, "g27_content_index")
CHUNK = 3000          # items per part file
TIME_BUDGET = 30      # seconds of embedding per run, then exit (resume next call)
MAX_DESC = 0          # title + genre shelves only (fast, dominant cold-start signal)
METHOD = os.environ.get("G27_METHOD", "transformer")
t0 = time.time()

# 1. universe + texts (deterministic order)
als = load_als(); catalog = list(map(str, als["i_uniq"]))
ev = load_eval_samples(); golds = sorted(set(ev["gold"].astype(int).astype(str)))
universe = sorted(set(catalog) | set(golds))
texts_df = load_item_texts(os.path.join(DATA, "domain_content_meta.csv"), set(universe), max_desc=MAX_DESC)
texts_df = texts_df.set_index("book_id").reindex(universe).dropna(subset=["text"]).reset_index()
texts_df = texts_df.rename(columns={"index": "book_id"})
ids_all = texts_df["book_id"].tolist(); texts_all = texts_df["text"].tolist()
n = len(ids_all)

# audit (write once)
audit_path = os.path.join(DATA, "g27_item_text_audit.json")
if not os.path.exists(audit_path):
    cold = sorted(set(golds) - set(catalog))
    cold_with_text = set(texts_df[texts_df["has_text"]]["book_id"]) & set(cold)
    json.dump({
        "item_count_total_universe": len(universe),
        "catalog_items": len(catalog),
        "eval_gold_items": len(golds),
        "item_count_with_text": int(texts_df["has_text"].sum()),
        "item_text_coverage_rate": round(float(texts_df["has_text"].mean()), 4),
        "cold_start_item_count": len(cold),
        "cold_start_items_with_text": len(cold_with_text),
        "cold_start_text_coverage_rate": round(len(cold_with_text) / max(len(cold), 1), 4),
        "text_recipe": "title + cleaned-genre-shelves(<=8) + desc[:200]",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2 (transformer) | tfidf_svd fallback",
    }, open(audit_path, "w"), indent=2)
    print("[audit] written", audit_path)

n_parts = (n + CHUNK - 1) // CHUNK
print(f"[plan] universe={n} parts={n_parts} method={METHOD}  ({time.time()-t0:.1f}s)")

# 2. embed missing parts (resumable, time-budgeted)
sci = SemanticContentIndex()
done = 0
for p in range(n_parts):
    pf = os.path.join(EMB_DIR, f"part_{p:04d}.npy")
    if os.path.exists(pf):
        continue
    if time.time() - t0 > TIME_BUDGET and done > 0:
        rem = n_parts - len(glob.glob(EMB_DIR + "/part_*.npy"))
        print(f"[chunk] embedded {done} parts this run; {rem} parts remain. Re-run."); sys.exit(0)
    s0, s1 = p * CHUNK, min((p + 1) * CHUNK, n)
    sub = texts_all[s0:s1]
    try:
        emb, model_name, dim = (SemanticContentIndex._embed_transformer(sub) if METHOD == "transformer"
                                else SemanticContentIndex._embed_tfidf_svd(sub))
    except Exception as e:
        print(f"[HARD-FAIL transformer] {e}\n[fallback] switching to tfidf_svd for this build")
        emb, model_name, dim = SemanticContentIndex._embed_tfidf_svd(texts_all)  # whole-corpus SVD
        np.save(PREFIX + "_emb.npy", np.ascontiguousarray(emb, dtype=np.float32))
        json.dump({"ids": ids_all, "model": model_name, "dim": dim, "fallback": True},
                  open(PREFIX + "_meta.json", "w"))
        print("ASSEMBLED (fallback tfidf_svd)"); sys.exit(0)
    np.save(pf, emb.astype(np.float32))
    done += 1
    print(f"[chunk] part {p:04d} [{s0}:{s1}] model={model_name} dim={dim} ({time.time()-t0:.1f}s)")

# 3. assemble if all parts present
parts = sorted(glob.glob(EMB_DIR + "/part_*.npy"))
if len(parts) == n_parts:
    emb = np.concatenate([np.load(pp) for pp in parts], axis=0).astype(np.float32)
    assert emb.shape[0] == n, f"emb {emb.shape[0]} != universe {n}"
    np.save(PREFIX + "_emb.npy", emb)
    json.dump({"ids": ids_all, "model": "sentence-transformers/all-MiniLM-L6-v2",
               "dim": int(emb.shape[1])}, open(PREFIX + "_meta.json", "w"))
    print(f"ASSEMBLED {emb.shape} -> {PREFIX}_emb.npy  ({time.time()-t0:.1f}s)")
else:
    print(f"[status] {len(parts)}/{n_parts} parts done. Re-run to continue.")
