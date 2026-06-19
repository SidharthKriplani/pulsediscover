# G28 — Learned Fusion + Ranker Tournament (Gold Pass 1/3)

*PulseDiscover V2, Gold Pass 1/3. Turns the ALS + semantic + popularity candidate lanes into a source-aware **fusion / learned-ranking** layer and asks: can combining ALS strength and semantic coverage beat either source alone? Evidence: `g28_candidate_feature_table_summary.json`, `g28_fusion_baseline_report.json`, `g28_learned_ranker_tournament.json`, `g28_source_policy_decision.json`, `g28_semantic_text_ablation.json`, `g28_final_ranker_fusion_decision.json`. Code: `src/ranking/g28_fusion_ranker.py`, `scripts/run_g28_*.py`.*

> **Boundaries:** offline served-c2 ranking/fusion evidence. **NOT** online lift, production, solved cold-start, LLM recommender, semantic taste, OPE/fairness. Learned-beats-ALS is claimed **only** on the held-out test split, **with a small-positive-count caveat**. Served-c2 metrics only (never mixed with V1 `d3aplus` 0.0846).

## 1. Why G28 (after G27)
G27 proved semantic retrieval uniquely reaches item-cold-start (reachability 54.6%) and expands coverage (62.6% vs ALS 8.1%) but **loses on head relevance** (0.010 vs 0.087) and that **naive ALS-first fusion fails** on cold-start (ALS saturates the top-20). So semantic is a *complement*, not a replacement — which raises the real question G28 answers: **how do we combine them?**

## 2. Candidate-source architecture & feature table
One row per (user, candidate) over the union of **ALS (served c2)**, **semantic (G27 MiniLM)**, and **popularity** sources. **1,033,793 rows · 6,213 users** (those with both an ALS factor and a query item — the fusion-relevant population). Features: source flags, ALS score/rank, semantic score/rank, popularity rank, item-tier (head/mid/long-tail/cold-start), source count; label = held-out gold. **Leakage-safe split by user-hash** (train 4,331 / val 938 / test 944 users); test labels never used for training/selection.

**Key structural finding — the sources are nearly disjoint:** als_only 438,140 · semantic_only 440,151 · **both 6,749** · popularity_only 148,753. ALS and semantic retrieve almost entirely different items, which is exactly why a fusion layer can add value. Label sparsity is extreme (515 positives / ~1.03M rows; single held-out gold per user).

## 3. Fusion baselines (test split, K=20)
| Policy | Recall@20 | cold-start R@20 | coverage | Gini |
|---|---|---|---|---|
| als_only | 0.0222 | 0.0000 | 4.5% | 0.986 |
| semantic_only | 0.0169 | 0.0209 | 23.3% | 0.852 |
| popularity_only | 0.0074 | 0 | 0.04% | 1.000 |
| als_first | 0.0222 | **0.0000** | 4.5% | 0.986 |
| semantic_fallback (thin=10) | 0.0222 | 0.0000 | 4.5% | 0.986 |
| **source_balanced** | 0.0201 | **0.0171** | 17.7% | 0.918 |
| **rrf** | **0.0212** | **0.0171** | 17.6% | 0.919 |
| weighted_score | 0.0212 | 0.0000 | 5.1% | 0.985 |

**Why ALS-first (and weighted_score, semantic_fallback) fail on cold-start:** ALS fills all 20 slots with in-catalog items, so semantic cold-start items never compete → cold-start recall 0. **RRF and source_balanced fix it** by letting semantic candidates compete for slots: they recover cold-start (0→0.017) and ~4× coverage while retaining ~95% of ALS relevance (0.0212 vs 0.0222). **RRF is the best non-learned policy.**

## 4. Learned ranker tournament (test split; train on train, select on val)
LightGBM 4.6 (LambdaMART) available; **xgboost not installed** (noted); sklearn used. Classifiers trained on negative-downsampled data (extreme imbalance); LambdaMART trained on full grouped data. **Test positives = 81** (small — see caveat).

| Model | objective | Recall@20 | cold-start | coverage | NDCG@20 | decision |
|---|---|---|---|---|---|---|
| **LightGBM LambdaMART** | lambdarank(ndcg@20) | 0.0360 | **0.0323** | **7.6%** | **0.0235** | **CHAMPION** |
| logistic | clf score | **0.0392** | 0.0323 | 5.4% | 0.0213 | contender |
| sklearn HistGBM | clf score | 0.0371 | 0.0323 | 7.0% | 0.0211 | contender |
| random_forest | clf score | 0.0307 | 0.0209 | 4.8% | 0.0180 | weaker |
| ALS-only (reference) | — | 0.0222 | 0.0000 | 4.5% | — | baseline |

**Result:** every learned fusion model **beats ALS-only on test Recall@20** (0.031–0.039 vs 0.022, +40–77%) **and recovers cold-start** (0.032 vs 0) — by exploiting the near-disjoint ALS+semantic candidate pool. The signal is **consistent across four independent model families**, which strengthens it despite the small positive count.

**Champion = LightGBM LambdaMART** (evidence-based, not AUC-only): best cold-start recall (0.0323), best coverage among learned (7.6%), best NDCG, a *proper ranking objective* with query groups, and it beats ALS on relevance. Logistic edges raw recall but is a classifier hack with worse coverage/NDCG. Top features (LambdaMART): ALS rank/score, semantic rank/score, source flags.

## 5. The honest caveat (don't overclaim)
**Test positives = 81** (944 users). Recall@20 0.036 ≈ 34 users with gold in top-20 vs ~21 for ALS — a ~13-user difference. Directionally robust (consistent across model families, plus the structural cold-start=0→0.032), but **small absolute counts, single held-out gold, no NDCG-rich labels, no online confirmation.** I claim "learned fusion beats ALS-only **on the offline test split**," not "in production."

## 6. Source-aware decision policy (champion)
- **ALS primary** for warm/head users (best head relevance).
- **Semantic prioritized fallback** for item-cold-start & thin/empty-ALS users (only source reaching unseen items).
- **Semantic injection** for coverage/long-tail only when head-relevance loss stays within 10% of ALS — via **RRF / source_balanced**, never ALS-first.
- **Learned fusion (LambdaMART)** is the champion ranking policy where a candidate-feature table is available; it dominates the relevance/cold-start/coverage tradeoff offline.
- **Popularity** only for true zero-history users (no factor, no query item).
- **Heuristic head-cap rerank** (G26B) stays the config-flagged exposure lever.

## 7. Semantic text ablation
Restricted-pool cold-start A/B (996 items): title+shelves R@20 **0.072** vs title+shelves+desc **0.062** → **description marginally hurts** (Δ −0.01). Validates G27's cheaper title+shelves recipe; **no description patch recommended** for Pass 2. (Restricted-pool recall — relative text-recipe effect only.)

## 8. Failure cases
Extreme label sparsity (81 test positives) → small-sample caution; ALS/semantic disjoint → fusion adds coverage but can't rescue relevance neither source has; als_first/weighted/semantic_fallback fail on cold-start; learned models risk overfitting tiny positive set (mitigated: val-selected, test-held-out, consistent across families); single-gold recall (no graded labels).

## 9. Claim boundary
**Safe (strong):** "I built a source-aware fusion/ranking layer over ALS, semantic content retrieval, and popularity candidates; evaluated heuristic fusion and learned rankers across warm/head, mid, long-tail and item-cold-start cohorts (relevance, coverage, concentration, cold-start reachability, latency); a LightGBM LambdaMART fusion ranker beat ALS-only on the held-out test split (Recall@20 0.036 vs 0.022) while recovering cold-start (0.032 vs 0). The shipping policy keeps ALS the warm-path primary and uses semantic as a measured cold-start/tail source. Offline ranking/fusion evidence on a small positive set — not online lift or solved cold-start." **Forbidden:** online lift, production, cold-start solved, LLM recommender, semantic taste, OPE executed, fairness certified, learned-beats-ALS beyond the offline test split.

## 10. Next pass
**Pass 2 = G29 OPE-logging execution** (propensities) — the binding gap before any *value* claim and the only way to confirm the offline learned-fusion gain. Alternatively patch with denser labels / multi-gold for stronger ranking signal. **Description text not recommended** (ablation negative).

## 11. Status
✅ G28 done (Gold Pass 1/3). Champion: LightGBM LambdaMART fusion; ALS stays warm primary; semantic = measured cold-start/tail source. V1 (`gold_candidate` 8.9) untouched.

**STOP — Gold Pass 1/3 complete; awaiting review.**
