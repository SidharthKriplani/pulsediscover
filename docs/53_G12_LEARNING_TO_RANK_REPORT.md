# 53 — G12: Learning-to-Rank (LambdaMART) Re-Ranker

*Builds the two-stage industry layer PulseDiscover was missing: many retrievers produce a candidate set, then **one LambdaMART ranker fuses their signals as features**. Tests whether a learned ranker beats the best single retriever on the **same** candidate sets. CPU (LightGBM 4.6), real data, 5k+5k LLO queries. Evidence: `outputs/evidence/g12_ltr_report.json`; plot `outputs/plots/g12_ltr.png`.*

> **Honest framing (critical):** this is **re-ranking recall *within* a candidate set** (warm top-60 ∪ cold top-30 per user), **not full-catalog retrieval.** Its numbers are comparable to the baseline rankers on identical candidates — **NOT** to ALS f64's full-40k retrieval R@20 (0.085). Different denominators.

## 1. Setup
- **Candidate set / query:** ALS warm top-60 ∪ content-hybrid cold top-30 (~90 items), deduped.
- **Features (11):** ALS score, ALS rank, content rank, RRF, log-popularity, cold flag, author match, series match, publication year, in-warm-list, in-cold-list.
- **Label:** the single held-out gold = 1 (binary relevance).
- **Split:** by **user** (hash, ~75/25) → 7,410 train / 2,590 test queries; no user leakage.
- **Model:** LightGBM `lambdarank`, NDCG@20, 300 trees, lr 0.1.

## 2. Results — Recall@20 (within candidates; same candidates for all rankers)
| Ranker | R@20 | NDCG@20 |
|---|---|---|
| **LambdaMART (LTR, fused)** | **0.1286** | **0.0860** |
| RRF (best single) | 0.0664 | 0.0469 |
| ALS score | 0.0274 | — |
| popularity | 0.0208 | — |
| *candidate recall ceiling* | *0.2019* | — |

**LTR lift vs best single retriever: +0.0622 R@20 — nearly 2× RRF**, and it captures **~64% of the candidate ceiling** (0.1286 / 0.2019).

## 3. Findings
1. **The two-stage architecture now exists and the ranker is the win.** Fusing all retriever signals nearly **doubles** the best single-retriever ranker (0.129 vs 0.066). This is the canonical industry shape — retrieve → rank — and it materially strengthens the technique-tournament depth.
2. **This resolves our "ALS beats every sequence model" tension.** In a two-stage system you don't pick one retriever — you feed all their scores to a ranker. The LTR's top features by gain are **ALS score, log-popularity, publication year, ALS rank, content rank** — i.e., *every family contributes*. A signal can lose head-to-head yet still earn importance in the ranker.
3. **Single-lane rankers fail on mixed traffic.** ALS-score alone scores 0.027 because it's blind to cold items (50% of queries are cold-gold). RRF (0.066) handles both lanes; LTR (0.129) fuses everything. The ranker's job is exactly this lane-blending.
4. **The retrieval stage is the bottleneck, not the ranker.** Candidate recall ceiling is only **0.20** — 80% of golds aren't in the 90-item candidate set, so no ranker can recover them. **Two-stage insight: to lift end-to-end recall you improve *candidate generation* (or widen the set), not just ranking.** LTR already extracts ~64% of what's recoverable.

## 4. Verdict
- **Technique-tournament depth (audit #3): further materially strengthened** — PulseDiscover now has retrieval **and** a learned ranking layer, with an interpretable feature-fusion story.
- **No overclaim:** re-ranking recall within candidates (not full-catalog retrieval); not comparable to ALS f64 0.085; candidate ceiling stated; single-gold binary labels.

## 5. Documented v2 lift (not yet done)
Add **SASRec and two-tower scores as features** once the GPU runs return — both are exactly the kind of complementary signal a ranker exploits, and likely raise LTR further. Widening the candidate set (warm top-120 ∪ cold top-60) would raise the 0.20 ceiling.

## 6. Status
- ✅ G12 PASS — learning-to-rank layer built; LTR ≈2× best single retriever within candidates; feature-fusion + two-stage-bottleneck insights recorded.
- ⏭️ Next stack layers (per plan): G11E FAISS retrieval-systems layer, G11F LightGCN graph recommender, G13 final-slate MMR reranker, G14 Deep Defense Kernel. (And fold G11C/D GPU results + their scores into LTR features.)

**STOP — awaiting G12 review.**
