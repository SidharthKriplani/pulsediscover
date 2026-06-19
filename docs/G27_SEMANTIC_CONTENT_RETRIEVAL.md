# G27 — Semantic Content Retrieval (cold-start candidate lane)

*PulseDiscover V2 flagship modernization gate. Adds a **dense semantic content-retrieval lane** (sentence-transformers MiniLM item embeddings + FAISS content index) as a **candidate-generation source** for item-cold-start and catalog-coverage recovery. Evidence: `outputs/evidence/g27_semantic_retrieval_report.json`; plots `g27_coldstart_recovery.png`, `g27_semantic_vs_popularity_coverage.png`, `g27_head_tail_exposure.png`, `g27_latency_tradeoff.png`. Code: `src/retrieval/semantic_content_index.py`, `scripts/build_g27_content_embeddings.py`, `scripts/run_g27_semantic_coldstart_eval.py`.*

> **Scope guardrails:** dense content-embedding **candidate generation** — **NOT** chat, RAG, an LLM reranker, an "LLM recommender", or "semantic taste understanding"; **NOT** learned LTR, two-tower, GNN, or bandits. Served-c2 / content metrics only (never mixed with V1 `d3aplus` 0.0846). **Offline candidate-generation evidence; not online lift, not solved cold-start.**

## 1. Why G27 (and why G26B didn't solve this)
G26B reranks items **already inside the ALS candidate pool** — it cannot surface items with no collaborative signal. **57% of held-out gold items are item-cold-start** (unseen in train), so they are *not in the ALS catalog at all* and **no ALS-based reranker can ever reach them**. Closing that gap needs a **new candidate source** keyed on item *content*, not collaborative co-occurrence.

## 2. Architecture
Item metadata → canonical text (`title + cleaned genre shelves`; description dropped at build time for embedding-throughput budget) → MiniLM dense embedding (384-d, L2-normalized) → **FAISS IndexFlatIP content index** (cosine-equivalent). Candidate generation is **item-to-item**: a user's last-read item embedding queries the content index for nearest items → `source=semantic_content` candidates. Built resumably (14 chunks) on CPU; index is deterministic and reusable.

## 3. Metadata audit (acceptance: coverage measured)
Universe = served ALS catalog (40,541) ∪ eval gold items → **41,866 items, 100% with usable text**. Item-cold-start golds: **1,325, 100% with usable text**. No metadata gap blocks the gate.

## 4. Embedding / index details
Model `sentence-transformers/all-MiniLM-L6-v2` (dim **384**, deterministic, local, HF-offline after cache). Index `IndexFlatIP` over 41,866 normalized vectors. Retrieval latency **p50/p95 ≈ 0.23 ms** (batched FlatIP). Build was the cost (~CPU, chunked); retrieval is cheap.

## 5. Results — candidate source comparison (served-c2 protocol, K=20)
| Source | head | mid | long_tail | **cold-start (unseen)** | coverage@20 | Gini |
|---|---|---|---|---|---|---|
| popularity | 0.051 | 0 | 0 | **0** | 0.14% | 1.00 |
| ALS (c2) | **0.087** | 0.012 | 0 | **0** (by construction) | 8.1% | 0.983 |
| **semantic** | 0.010 | 0.013 | **0.005** | **0.011** | **62.6%** | **0.731** |
| fusion (ALS∪sem, ALS-first) | 0.087 | 0.012 | 0 | 0 | 8.2% | 0.982 |

**Cold-start recovery:** semantic reaches **54.6%** of item-cold-start golds (926/1,696) somewhere in top-20; ALS and popularity reach **0%** by construction. Semantic Recall@20 on cold-start golds is **0.011 — low in absolute terms, but the only non-zero source.**

## 6. The honest tradeoff (relevance vs coverage vs latency)
- **Coverage / catalog health: a large win.** Semantic covers **62.6%** of the catalog (vs ALS 8.1%) and is far less concentrated (Gini **0.731** vs 0.983) — content neighbours aren't popularity-biased.
- **Relevance on warm/head: a clear loss.** Semantic Recall@20 on head items is **0.010 vs ALS 0.087** — collaborative signal dominates for popular items. Semantic is comparable on mid, marginally better on long-tail, and uniquely non-zero on cold-start.
- **Latency: negligible** (~0.23 ms p95).
- **Net:** semantic is a **complement for the tail and cold-start, not a replacement for ALS on the warm/head path.**

## 7. Fusion finding (honest negative for the naive policy)
Naive **ALS-first union** fusion gives **0** cold-start recall — ALS's 20 candidates **saturate the top-20** and push semantic candidates out. **Implication:** semantic must be used as a **cold-start fallback / prioritized source** (when the ALS pool is empty/thin, or for unseen-item queries), **not** appended after a full ALS slate. Learned fusion is explicitly out of scope (no learned ranker this gate).

## 8. Failure cases
Zero-history users have no query item → semantic item-to-item N/A (popularity fallback retained); content neighbours can be series/near-duplicate rather than novel discovery; single-held-out-gold recall is low in absolute terms (relative comparison only); genre-noisy shelves can pull off-topic neighbours; description was dropped at build time (title+genre only) — adding it may lift relevance at higher embedding cost.

## 9. Ship decision
**Ship as a cold-start fallback + supplemental tail/coverage candidate source — NOT as a warm-path replacement.** Enable semantic candidates when the ALS pool is empty/thin or for item-cold-start queries; keep ALS as the warm/head primary. Default integration would be a *prioritized* fallback, not an ALS-first union (per §7).

## 10. Claim boundary
**Safe (strong):** "I added a semantic content-retrieval lane using dense item embeddings and a FAISS content index as a candidate source for item-cold-start and sparse-catalog scenarios. I evaluated it against popularity fallback and the served c2 baseline on cold/sparse/warm cohorts, measured relevance, coverage, long-tail exposure, concentration, and latency, and used that evidence to decide where semantic retrieval belongs in the recommender stack — a cold-start/tail complement, not a warm-path replacement. This is offline candidate-generation evidence, not online lift or solved cold-start." **Avoid:** "I added embeddings / used an LLM / improved cold-start / AI-powered / semantic recommender." **Unsafe:** "cold-start solved / LLM recommender / semantic taste understanding / online lift / production deployed / learned ranker / OPE executed / fairness certified."

## 11. Next gate
Semantic candidates create real catalog-diverse candidate variety, which a learned ranker could exploit → **G28 learned ranker tournament** (LightGBM/XGBoost vs V1 LambdaMART over ALS+semantic candidate features); **or G29 OPE-logging execution** if logging is prioritized. **Patch G27** if richer text (add description) is wanted to lift absolute semantic relevance.

## 12. Status
✅ G27 done (V2). `ship_decision: cold_start_fallback_and_supplemental_candidate_source`. V1 (`gold_candidate` 8.9) untouched.

**STOP — G27 complete; awaiting review.**
