# 21 — D3A+ Performance & Evaluation Gate Report

*Scope: strengthen the domain baseline/evaluation floor before sequence modeling — richer metrics, segment eval, tuned collaborative baselines, domain-aware heuristics, a hybrid generator, and an external-run pack for what exceeds the sandbox. Same 5,000-user leave-last-out sample as D3A for comparability. Domain-scoped (Goodreads fantasy/paranormal). No SASRec / OPE / position-bias / feedback / PRD / PDF.*

## 1. Richer metrics — candidate generators (5,000-user LLO, bootstrap CIs)
| Generator | R@10 | R@20 | R@50 | R@100 | R@200 | NDCG@10 |
|---|---|---|---|---|---|---|
| popularity | 0.0202 | 0.0350 | 0.0630 | 0.1196 | 0.1654 | 0.0085 |
| co-occurrence (5k) | 0.0344 | 0.0464 | 0.0762 | 0.1120 | 0.1594 | 0.0207 |
| **ALS (f16,i4)** | 0.0300 | 0.0570 | 0.1088 | 0.1742 | **0.2548** | 0.0152 |
| same-author | 0.0286 | 0.0420 | 0.0682 | 0.0994 | 0.1376 | — |
| same-series | 0.0004 | 0.0006 | 0.0010 | 0.0018 | 0.0022 | — |
| **hybrid (RRF)** | **0.0406** | **0.0614** | 0.1036 | 0.1596 | 0.2362 | **0.0214** |

(R@20 CIs: hybrid [0.055,0.068] · ALS [0.051,0.064] · co-occ [0.040,0.052] — hybrid edges ALS at @20; both clear of co-occ and popularity.)

## 2. Honest reading
- **ALS dominates deep recall** (R@200 0.255) — the right *candidate-generation* floor before a ranker.
- **Hybrid (RRF of co-occ+ALS+author+series, popularity fallback) wins the top** — best R@10/R@20 and best NDCG@10/20/50 — but **loses to pure ALS at R@50–200** (marginal lift vs best single: **+0.0044 @20, −0.005 @50, −0.015 @100, −0.019 @200**). Blending weaker generators sharpens the head and dilutes the deep tail. Reported as-is — not a "hybrid wins everything" claim.
- **same-series alone ≈ 0 recall** — a genuine weak result (a user's *next* book is rarely another book in a series they already read that we can surface this way). same-author is moderate.
- **Domain vs smoke:** co-occurrence is strong here (R@10 0.034) vs near-zero on MovieLens — dense fiction co-reading is informative.

## 3. Baseline tuning
**ALS factor ladder (iters=4)** — `outputs/plots/domain_als_factor_tuning.png`:
| factors | R@20 | R@50 |
|---|---|---|
| 16 | 0.0570 | 0.1088 |
| 32 | 0.0732 | 0.1246 |
| 64 | **0.0846** | **0.1466** |
- **Monotonic gains with factors** — f64 ALS (R@20 0.085) is now the strongest single generator, beating the f16-based hybrid (0.061). Iterations **4→8 gave no gain** (converged by iter 4). f128+ → external pack.

**Co-occurrence vocab sweep:**
| vocab | R@20 | note |
|---|---|---|
| 5k | 0.0464 | base |
| 10k | 0.0500 | `MᵀM` nnz 39.5M, 30.9 s |
| 20k | **FAILED** | OOM/timeout on 3.8 GB (`MᵀM` nnz ~150M+) — documented limit |

## 3b. Baseline-consistency note (D3A vs D3A+) — no bug
co-occurrence R@20 differs between gates: **D3A 0.0532 vs D3A+ 5k 0.0464 / 10k 0.0500**. Verified cause: **neighbor-count (TOPN) only** — D3A used TOPN=50, D3A+ used TOPN=80. Truncating the saved 80-neighbor lists back to 50 and re-evaluating reproduces D3A **exactly** (R@5/10/20 = 0.0252/0.0360/0.0532); at TOPN=80 it is 0.0224/0.0344/0.0464. Same vocab (5k), same min_co (5), same train, same 5,000-user sample, same eval/dedup path. More neighbors add lower-PMI mass to the score sum, mildly reshuffling the top-20 and *lowering* recall. **No metric/dedup/scorer bug** — a pure hyperparameter difference. (popularity and ALS f16 are identical across gates: pop R@20 0.0322→0.035 reflects D3A capping recs at 20 vs D3A+ generating 200 then scoring @20 — same items, no drift at @20; ALS f16 R@20 0.057 in both.)

## 4. Segment / cold-start breakdown (held-out golds)
- Items: **4,999 warm / 1 new**; Creators: **5,000 warm / 0 new**; Series: **3,899 warm / 1,101 new** (the held-out book is in a series the user hadn't read before, in 22% of cases).
- Hybrid R@20: warm 0.0614, new (n=1) 0.0 → leave-last-out golds are ~all warm; **the substantive cold-start is the D2 global-time test window** (3,378 new items, 23.5% of test interactions on new items) where retrieval baselines structurally score ~0.

## 5. Extended-protocol plan (designed, not run)
- **leave-last-5:** hold out each user's last 5 test-window interactions; Recall@K over the 5-item target set.
- **future-window recall:** target = *all* of a user's test-window interactions (not just the last). Both are larger evals → in the external pack (full-population, ≥16 GB).

## 6. Sandbox limits — reported honestly
- co-occurrence ≥20k: OOM/timeout (dense `MᵀM`). ALS ≥128 factors / ≥12 iters and full-population (42,683-user) eval: exceed the 45 s/3.8 GB cap. All routed to `external_runs/D3A_PLUS_EXTERNAL_RUN_INSTRUCTIONS.md` (free Colab/Kaggle; deterministic seeds; metrics schema + manifest included).

## 7. The improvement ladder + the D3B comparison floor
`popularity (R@20 0.035) → same-author (0.042) → co-occ 5k→10k (0.046→0.050) → ALS f16 (0.057) → hybrid-f16 (0.061) → ALS f32 (0.073) → ALS f64 (0.085)`.

**D3B comparison floor (locked):**
- **Primary floor: ALS f64** (R@20 0.0846, R@50 0.1466) — the best in-sandbox single generator.
- **Secondary context: f16 hybrid** (R@20 0.0614, best NDCG) — the best blended generator at the head.
- **Historical context: original D3A baselines** (popularity / co-occ / ALS-f16).
- **f64-hybrid: pending/external** — the hybrid was built on f16 ALS; recomputing it with f64 ALS needs a refit+remerge (>1 call), so it is deferred to the external pack unless cheaply recomputed.

## 8. Artifacts
- `outputs/evidence/domain_baselines_plus_report.json` (metrics + CIs + segments + cooc/ALS tuning + findings)
- `outputs/plots/domain_baselines_plus_recall.png`, `outputs/plots/domain_als_factor_tuning.png`
- `external_runs/D3A_PLUS_EXTERNAL_RUN_INSTRUCTIONS.md`

## 9. No-overclaim check
Domain-scoped; hybrid reported with its deep-K loss (not forced); same-series weakness disclosed; OOM/CPU limits documented; 5k-user-sample + bootstrap CIs stated; no SASRec/OPE/position/feedback; no online-lift. ✅

## 10. Gate status & next step
- ✅ Evaluation floor strengthened (R@5..200, NDCG, segments, CIs); collaborative baselines tuned (ALS f64 best single); domain heuristics + hybrid built; external pack prepared.
- ⏸️ **Next (on approval): D3B — SASRec** on the domain core (top-40k vocab), compared against the **f64 ALS / hybrid** floor on the same sample; or run the external ALS f128+/full-co-occ first to finalize the candidate-generation floor.

**STOP — awaiting D3A+ review.**
