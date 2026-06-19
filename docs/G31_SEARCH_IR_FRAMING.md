# G31 — Search / IR Framing (BM25 + dense hybrid retrieval)

*PulseDiscover V2. Adds a **BM25 lexical retrieval lane**, fuses it with the existing dense semantic lane via **RRF**, reports **search-standard NDCG@K + MRR** alongside Recall@K, and re-narrates the existing LTR / FAISS / OPE work as a **two-stage search pipeline** — to make the project speak to **Search / Information-Retrieval** roles in addition to RecSys. Evidence: `outputs/evidence/g31_search_ir_report.json`; plot `g31_lexical_dense_hybrid.png`. Code: `src/retrieval/bm25_lexical_index.py`, `scripts/run_g31_search_ir_eval.py`.*

> **Three honesty guards (locked):**
> **(a) Query-by-document, NOT free-text query.** PulseDiscover has no text queries — it's recommendation. The "query" is the user's **last-read item's text** → *more-like-this / related-item* retrieval (a real IR pattern), not a free-text search engine.
> **(b) Position-bias-aware *evaluation*, NOT in-ranker correction.** The OPE / two-lane position-bias work is an *evaluation/diagnostic* lane; PulseDiscover does **not** bake a position-bias correction into LTR training. (No PAL this gate.)
> **(c) NDCG@K and MRR are SINGLE-RELEVANT.** One held-out gold per user, no graded relevance → NDCG@20 = 1/log₂(rank+1) if the gold is in the top-K; MRR = 1/rank. These are the standard search metrics in their single-relevant form, stated as such.
> Served-c2 metrics only. **V1 RecSys claims and V1 status are untouched.**

## 1. Why this gate
PulseDiscover already contains the modern two-stage retrieval+ranking machinery — FAISS ANN retrieval, dense semantic retrieval, LambdaMART LTR re-ranking, RRF fusion, and off-policy evaluation. Those are the *same* components a search stack uses; OPE and position-bias literally originate in the IR/LTR literature (Joachims, 2002). The only things missing to *speak* the search dialect were a **lexical (BM25) baseline** and **NDCG/MRR reporting**. G31 adds exactly those, at low cost, without changing the project's identity or its RecSys claims.

## 2. The two-stage search pipeline (re-narration of existing work)
```
seed item (query-by-document)
   → STAGE 1 RETRIEVAL: BM25 lexical  +  dense semantic (FAISS)  → RRF fusion
   → STAGE 2 RE-RANK: LambdaMART LTR (G12 / G28 fusion ranker)
   → evaluation: Recall@K · NDCG@K · MRR  +  off-policy (IPS/SNIPS/DR, G29)  +  position-bias-aware eval
```
Every box already existed; G31 adds the BM25 lexical arm and the NDCG/MRR lens. This is the front end used at Flipkart / LinkedIn / Amazon-search-style stacks — **with the honest caveat that our "query" is a seed item, not free text.**

## 3. Results — lexical vs dense vs hybrid (600 seed-item users, K=20)
| Retrieval | Recall@20 | NDCG@20 (single-rel) | MRR (single-rel) | cold-start R@20 | coverage@20 | latency p95 |
|---|---|---|---|---|---|---|
| **BM25 (lexical)** | **0.0133** | **0.0062** | **0.0042** | 0.0147 | 0.195 | 63 ms* |
| dense (semantic) | 0.0067 | 0.0022 | 0.0010 | 0.0074 | 0.188 | 3.2 ms |
| BM25 + dense (RRF hybrid) | 0.0117 | 0.0048 | 0.0030 | 0.0147 | 0.194 | 0.07 ms (fusion only) |

\* BM25 p95 ~63 ms is a **pure-Python `rank_bm25` artifact**; a production lexical index (Lucene / Elasticsearch / `bm25s`) serves this sub-millisecond. Not a production latency benchmark.

## 4. Honest reading (no overclaim)
- **BM25 lexical is the strongest single lane** on seed-item query-by-document — shared **series/genre tokens** on short title+shelves text are a powerful "more-like-this" signal (same-series books share title words).
- **Dense is weaker here** — MiniLM on very short text (title + genre, no description) underperforms exact lexical overlap for this task. (Consistent with the G28 ablation: description didn't help; the text is short.)
- **RRF hybrid is a robust middle, NOT a recall winner** — fusing the weaker dense lane *dilutes* the stronger lexical signal, so hybrid (0.0117) sits below BM25 alone (0.0133). The value of hybrid is **robustness** (it never collapses to either lane's weakness and reaches cold-start like BM25), not a headline lift. Reporting this honestly is the point — I do not claim "hybrid beats both."
- All three reach **item-cold-start** (lexical most), because content tokens exist for unseen items.

## 5. What this unlocks (and the boundary)
**Unlocks (safe):** the project can now be narrated as a **two-stage search / IR retrieval pipeline** — lexical + dense hybrid retrieval, RRF fusion, LTR re-ranking, NDCG/MRR + off-policy evaluation — which maps to search/ranking/IR roles (e-commerce search, LinkedIn/Amazon-style). **Does NOT unlock:** a free-text query search engine, in-ranker position-bias correction, online lift, or production search — and it changes **no** V1 RecSys claim.

## 6. Resume / interview lines (search-flavored, bounded)
- "Built a two-stage retrieval front end — **BM25 lexical + dense semantic + RRF hybrid** — with **LambdaMART LTR re-ranking** and **off-policy evaluation (IPS/SNIPS/DR)**, reported with **NDCG@K, MRR, and Recall@K**." *(query-by-document on an item catalog, offline.)*
- "Showed lexical retrieval is the strongest single lane for short-text more-like-this, and that **RRF hybrid buys robustness, not a recall win** — measured, not assumed."
- Hard Q: *"Is this a search engine?"* → "A two-stage *retrieval* pipeline evaluated with search metrics; the query is a seed item (more-like-this), not free text — I don't claim a free-text query engine."
- Hard Q: *"Do you correct for position bias?"* → "I do **position-bias-aware evaluation** (off-policy lens); I did not bake a position-bias correction into the ranker — that's the labelled next step."

## 7. Claim boundary
**Safe:** §6 lines + the report's `safe_claim`. **Forbidden:** free-text query search engine · in-ranker position-bias correction · online lift · production search · any change to V1 RecSys claims · "hybrid beats both lanes" (it doesn't here) · graded-NDCG (ours is single-relevant).

## 8. Status
✅ G31 done (V2). Search/IR role coverage unlocked; three honesty guards held; V1 untouched. Pricing/forecasting deliberately **not** attempted (no demand/elasticity or time-series signal). Optional future: PAL/IPS-weighted LTR training to upgrade position-bias from *evaluated* to *corrected*.

**STOP — G31 complete; awaiting review.**
