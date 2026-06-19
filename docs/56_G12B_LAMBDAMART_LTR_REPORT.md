# 56 — G12-B: LambdaMART LTR on the GPU-Enriched Candidate Table

*LambdaMART re-ranker over the G12-A2 enriched candidate table, with **user-level train/val/test split**, explicit **missingness + history features**, and a **fair variant ladder**. Re-ranking recall **within the candidate set** (coverage ceiling ~0.157) — **NOT** full-catalog retrieval. CPU (LightGBM). No online/production/RiskFrame-gold claim. Evidence: `outputs/evidence/g12_ltr_metrics.json`; plots `g12_ltr_comparison.png`, `g12_ltr_feature_importance.png`.*

## 1. Setup (rules enforced)
- **User-level split** (deterministic md5 hash): 70/15/15 train/val/test by user — no user appears in two splits. **733 test users.** *(Fixed a non-reproducible `hash()` split mid-build; the deterministic hash makes results stable across runs.)*
- **Missingness/history features added:** `has_sasrec_canonical_gpu_score`, `has_sasrec_small_gpu_score`, `has_twotower_gpu_score`, `history_len`, `has_sequence_history`.
- **Variant ladder:** ALS-only ranking · local LTR (no GPU cols) · enriched LTR (+GPU) · enriched−TwoTower.
- Val set used for early stopping; eval = re-ranking within each user's candidate set.

## 2. Results (test, R@20 1.96·SE ≈ 0.019)
| Variant | R@20 | R@50 | NDCG@20 |
|---|---|---|---|
| ALS-only ranking | 0.0382 | 0.0737 | 0.0114 |
| **local LTR (no GPU)** | **0.0764** | 0.1105 | 0.0388 |
| **enriched LTR (+GPU)** | **0.0791** | 0.1146 | 0.0425 |
| enriched − TwoTower | 0.0791 | 0.1160 | 0.0413 |
| *candidate coverage ceiling* | *0.1569* | — | — |

- **LTR vs ALS-only ranking: +0.0409 R@20 (~2.0×, ≈4 SE) — material and robust.**
- **enriched vs local: +0.0027 R@20 (within noise, <0.2 of the 1.96·SE threshold); NDCG@20 +0.0037 (marginal).**
- **TwoTower effect: 0.0 R@20** (enriched = enriched−TwoTower).

## 3. Findings (honest)
1. **The ranker is the win.** Learning-to-rank over the candidate set **doubles** ALS-only ranking (0.076 → vs 0.038), capturing ~50% of the 0.157 candidate ceiling. The two-stage retrieve→rank architecture is validated.
2. **GPU SASRec/TwoTower features do NOT materially improve the ranker** beyond the local features (ALS/content/popularity/CPU-SASRec). The enriched lift is +0.0027 R@20 — within noise. **Honest negative for the "GPU signals lift the ranker" hypothesis at this scale.**
3. **Feature importance ≠ marginal contribution — the key nuance.** By gain, `sasrec_small_gpu_score` and `twotower_gpu_score` rank **2nd and 3rd** (the model *uses* them heavily, incl. the weak-standalone TwoTower). Yet removing all GPU columns barely changes recall — because those signals are **correlated/substitutable** with `als_score`, `sasrec_cpu_d3b_score`, and `author_match`. The model leans on them when present but recovers nearly the same performance from correlated local features. (`author_match` is the single top feature.)
4. **Canonical coverage is on the *harder* segment.** Test R@20 is **0.102 for short-history users (<30, canonical absent)** vs **0.033 for long-history users (≥30, canonical present)** — short-history golds (often series/popular continuations) are easier. So the canonical signal, available only for the harder long-history users, has little room to lift aggregate recall.

## 4. Canonical-NaN = model-coverage limitation (corrected)
`sasrec_canonical_gpu_score` is NaN for 3,397/5,000 users — **not** empty history (none are empty). The split is exact at **history_len < 30 (= maxlen)**: the **2-block** canonical SASRec produces NaN at the last position whenever the sequence is left-padded, because masked-attention NaN at padding positions propagates through the *second* attention block; the 1-block SASRec-small is immune. A real implementation limitation (fixable via padding-safe attention or right-padding), surfaced honestly via `has_sasrec_canonical_gpu_score` and handled natively by LightGBM (NaN = missing).

## 5. Verdict
- **LTR improves materially** over single-retriever ranking (≈2×, robust) — ✅.
- **Improvement is within the candidate ceiling** (0.157) — the retrieval stage, not ranking, caps end-to-end recall.
- **GPU features do NOT materially help** at this scale (+0.0027 R@20, within noise); they're used by the model but substitutable. **Honest negative** for the GPU-enrichment hypothesis — *not* a negative for LTR itself.
- TwoTower: high feature-importance, ~zero marginal recall — the "weak-standalone-but-ranker-uses-it" effect is real in *importance* but does **not** translate to a recall gain here.

## 6. Claim boundaries
- Re-ranking recall **within candidates** (ceiling 0.157), **not** full-catalog retrieval; not comparable to ALS f64's 0.085.
- User-level split, deterministic + reproducible; significance judged vs ≈0.019 (1.96·SE, n=733).
- No online lift, no production, no RiskFrame-gold-complete claim.

## 7. Final interpretation (ACCEPTED — gate passed)
1. **LTR is the material win.** ALS-only ranking R@20 = 0.0382 → local LTR 0.0764 → enriched LTR 0.0791; candidate ceiling 0.1569. The ranker captures **~half the reachable candidate-set ceiling** and **materially improves over a single-retriever rank order**. *The validated result is the two-stage retrieve→rank architecture — not "GPU features won."*
2. **GPU features are an honest marginal negative.** Enriched (+GPU) improves only **+0.0027 R@20** over local LTR — within noise. **TwoTower has zero marginal R@20 effect.** Do **not** present GPU features as a ranker lift.
3. **Nuance kept.** Feature importance shows GPU SASRec-small and TwoTower are used heavily, but **marginal contribution ≈ 0** because those signals are correlated/substitutable with ALS, CPU-SASRec, `author_match`, and popularity. **Feature importance ≠ causal contribution.**
4. **Corrected bug story kept.** The non-reproducible Python `hash()` split was a real bug, now fixed with a **deterministic md5 user split**. The canonical-SASRec NaN was also corrected: **not empty history** — it's `history_len < maxlen` causing **left-padding / 2-block masked-attention NaN propagation**.
- **Claim boundary:** re-ranking **within candidates**, not full-catalog retrieval; **no online lift, no production, no RiskFrame-gold-complete claim.**

## 8. Next implication
**The bottleneck is candidate coverage, not ranking.** Candidate ceiling is only ~0.157, and LTR already reaches ~half of it. The next technical lever is **better/wider candidate generation**, not more ranker features → **G13: Candidate Coverage Expansion + Retrieval Ceiling Lift** (FAISS stays deferred unless clearly labeled an ANN/serving demo).

**STOP — G12-B accepted; G13 proposed (see docs/57), awaiting approval.**
