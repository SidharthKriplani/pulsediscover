# 15 — Option B Gate Report: SASRec-π through the OPE harness

*Scope: replace the ALS-based target policy π with the **actual trained SASRec model's scores**, re-run IPS/SNIPS/DM/DR against the **same** known-propensity simulator. μ (popularity logging) and the ground-truth reward (ALS-based simulator oracle) are **unchanged** — only the evaluated policy changes. No new model trained (reuses `outputs/_models/sasrec_phase3a.pt`). No domain. No online-lift claims. Stops at gate.*

---

## 1. What changed (and what didn't)
- **Changed:** target policy **π = softmax(SASRec scores)** instead of softmax(ALS). SASRec is batch-scored over every user's history and looked up per candidate-pool item.
- **Unchanged:** μ = popularity-softmax logging policy; **ground-truth reward = ALS-based simulator oracle**; q̂ good/bad; ε-exploration; PBM; seed.
- **Truth boundary (honest):** the ground-truth reward **remains ALS/simulator-based**. This validates *whether the OPE harness can evaluate the SASRec policy* — it is **not** real online lift and **not** a recommender-quality verdict.

## 2. Headline — estimating V(π_SASRec), N = 60,400 / 6,040 users
**Known ground truth V(π_SASRec) = 0.1383** (user-cluster bootstrap CIs, B=400).

| Estimator | Estimate | Bias vs GT | 95% CI | Note |
|---|---|---|---|---|
| IPS | 0.1346 | −0.0037 | [0.125, 0.144] | unbiased; highest variance |
| SNIPS | 0.1359 | −0.0024 | [0.126, 0.145] | self-normalized |
| DM (good q̂) | 0.1487 | +0.0104 | [0.148, 0.149] | good model, small bias |
| **DR (good q̂)** | **0.1362** | **−0.0021** | [0.127, 0.145] | smallest bias; std 0.0045 |
| DM (**bad** q̂) | 0.2949 | **+0.1566** | [0.294, 0.295] | trusts wrong model → wild |
| **DR (bad** q̂**)** | **0.1382** | **−0.00008** | [0.128, 0.148] | **recovers GT almost exactly** |

## 3. Can DR evaluate the actual SASRec policy? — **YES**
- **Small bias on the real model:** DR(good) bias **−0.0021** vs GT 0.1383 (IPS −0.0037, SNIPS −0.0024). The estimators track the SASRec policy's true value, not just an ALS stand-in.
- **DR keeps its variance edge:** DR(good) bootstrap std **0.0045 < IPS 0.0051** (observed, as expected).
- **Double robustness, even sharper here:** under the badly-misspecified q̂, DM blows up to **+0.157** bias but **DR(bad) lands at −0.00008** — essentially exact, because the exact propensities carry it.
- Automated check `can_DR_evaluate_SASRec_policy = True` (DR bias ≤0.02, DR std ≤ IPS std, DR-bad bias ≤0.03).

## 4. Honest nuance — SASRec-π is *harder* to evaluate off-policy
- **ESS dropped to 4,474 / 60,400** (vs 12,325 for ALS-π) and **w_max rose to 120** (vs 66). Reason: the SASRec policy **diverges more from the popularity logging policy μ** than ALS did, so importance weights are larger and noisier. OPE still works, but with less effective sample — a real, reported cost, not hidden.
- **Clipping sweep:** clip=5 raises ESS to ~22.8k but worsens IPS bias to −0.037 and over-corrects DR(bad) to +0.044 (clipping injects bias); larger clips converge to no-clip. DR(good)/SNIPS stay stable across clips.

## 5. Comparison to ALS-π (prior Phase 5)
| | ALS-π (Phase 5) | SASRec-π (Option B) |
|---|---|---|
| GT V(π) | 0.2191 | 0.1383 |
| IPS bias | −0.0125 | −0.0037 |
| DR(good) bias | −0.0074 | −0.0021 |
| DR(bad) bias | −0.0058 | −0.00008 |
| DR(good) std | 0.0034 | 0.0045 |
| ESS / w_max | 12,325 / 66 | 4,474 / 120 |

**Read this correctly:** the two GTs are **not comparable as recommender quality** — `GT_SASRec < GT_ALS` only because the synthetic ground-truth reward is **ALS-derived**, so a policy aligned with ALS scores naturally scores higher on it. This run answers "can the harness evaluate SASRec?", **not** "is SASRec better than ALS?". The estimator *biases* are actually a touch smaller for SASRec-π, while ESS is lower (harder weights) — both reported as-is.

## 6. Artifacts
- `outputs/evidence/ope_comparison_sasrec_pi.json` (`[BUILT][SYNTHETIC — simulator]`).
- `outputs/plots/ope_sasrec_pi.png` — SASRec-π estimates vs GT (95% CI) + ALS-π vs SASRec-π bias comparison.

## 7. No-overclaim check
- Truth still ALS-based — **stated**; no real online lift claimed. ✅
- `GT_SASRec < GT_ALS` explicitly flagged as **not** a quality comparison. ✅
- Lower ESS / higher weights reported, not hidden. ✅
- No new model trained (reused the Phase-3A SASRec checkpoint); no domain. ✅

## 8. Gate status & decision
- ✅ DR (and IPS/SNIPS) now evaluate the **actual SASRec policy** with small bias, DR variance ≤ IPS, and double robustness — closing the audit's ALS-π gap (axis 9).
- ⏸️ Domain headline (A) and T3 (C) still pending; defense PDF still deferred.

**Recommended next (not executed):** Option A (domain headline) when you supply a Goodreads-genre / Amazon Books dataset + GPU — then the defense PDF, domain-grounded. The doc-consistency fixes (audit §3) are applied.

**STOP — awaiting review.**
