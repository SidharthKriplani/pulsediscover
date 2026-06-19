# PulseDiscovery — Gold Pass 1 Defense Notes (G28)

*Interview-maximal, evidence-bounded. Use the "built X → measured Y → learned Z → decided W → next evidence Q" shape.*

## 60-second answer (updated post-G28)
"PulseDiscovery is an offline, honesty-gated recommender **decision system** on 2.56M real Goodreads interactions. ALS is the warm-retrieval floor — I earned that by training a *converged* SASRec that still lost. I then built a serving spine (FastAPI + FAISS, fallback, monitoring), measured cold-start and catalog-exposure honestly, and added a dense semantic content-retrieval lane that uniquely reaches item-cold-start items. In Pass 1 I fused ALS + semantic + popularity into a learned ranker: on the held-out test split a LightGBM LambdaMART fusion beat ALS-only on Recall@20 (0.036 vs 0.022) and recovered cold-start (0.032 vs 0), because ALS and semantic candidates are almost disjoint. The policy keeps ALS the warm-path primary and uses semantic as a measured cold-start/tail source. It's offline on a small positive set — the next evidence I need is propensity logging for valid off-policy evaluation, not an online claim I can't back."

## 2-minute architecture story
Candidate generation (ALS / semantic / popularity / co-occurrence) → FAISS retrieval (FlatIP default, HNSW scale, IVF rejected) → fusion + learned ranker (Pass 1) → config-flagged heuristic exposure rerank (G26B) → fallback tree → FastAPI serving with structured logging + monitoring → cohort/exposure evaluation → ship/no-ship decision with a claim ladder. Every stage is versioned on the served `c2` model; V1's `d3aplus` tuning number is quarantined in a registry so V1/V2 metrics never get conflated.

## What changed after G27
G27 said semantic is a complement, not a replacement (great cold-start/coverage, weak head relevance). G28 answered *how to combine*: a learned fusion ranker over the near-disjoint ALS+semantic candidate pool beats either alone on the offline test split, while a simple RRF is the best non-learned fusion if a model isn't available.

## Why ALS-first fusion failed
ALS fills all 20 slots with its in-catalog candidates, so semantic cold-start items never compete → cold-start recall stays 0. The fix is letting semantic candidates compete for slots (RRF / source-balanced / learned scoring), not appending them after a full ALS slate.

## Why a learned ranker/fusion was needed
ALS and semantic candidate sets overlap on only 6,749 of ~880k candidates — they're complementary, not redundant. A learned ranker can pick the best items across both sources per query, which is exactly what lifted test Recall@20 from 0.022 (ALS-only) to 0.036 and cold-start from 0 to 0.032.

## Champion policy
LightGBM LambdaMART fusion (proper ranking objective, best cold-start + coverage + NDCG among models, beats ALS on relevance) where a candidate-feature table exists; RRF as the non-learned fallback; ALS-only stays the warm/head primary; semantic is the prioritized cold-start/thin-pool fallback.

## What NOT to overclaim
Test positives are only 81 (single held-out gold) → "beats ALS **on the offline test split**, small-sample, directionally consistent across four model families," never "beats ALS in production." No online lift, no solved cold-start, no LLM recommender, no OPE number (not logged yet), no fairness certification. Semantic = candidate generation, not "semantic taste."

## 10 likely interview questions (short answers)
1. **Why does fusion help here?** ALS and semantic candidates are ~disjoint (overlap 6,749); fusion picks the best across both.
2. **Did learned beat ALS?** On the held-out test split, yes (0.036 vs 0.022) and it recovered cold-start — but on 81 positives, so I treat it as directional, pending online/OPE.
3. **Why LambdaMART over logistic?** Logistic edged raw recall but LambdaMART has a real ranking objective, best cold-start, coverage and NDCG — better governance.
4. **Why did ALS-first fail?** ALS saturates the top-20; semantic never competes for slots.
5. **Best non-learned policy?** RRF — ~95% of ALS relevance, recovers cold-start, ~4× coverage.
6. **Is cold-start solved?** No. Semantic *reaches* unseen items (offline-split) and fusion *recovers* some; live new-item launch evidence is the gap.
7. **Why not richer embedding text?** Ablation showed description marginally hurt cold-start retrieval; title+shelves is cheaper and at least as good.
8. **How did you prevent leakage?** Split by user-hash, test labels unused for training/selection, served-c2 protocol only.
9. **Why is recall so low in absolute terms?** Full-catalog single-held-out-gold on the served model — relative comparisons are the signal; not comparable to V1's 0.0846 (different protocol).
10. **What's next?** OPE-logging execution (propensities) to make IPS/SNIPS/DR valid and confirm the offline fusion gain — the binding gap before any value claim.
