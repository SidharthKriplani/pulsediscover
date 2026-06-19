# 67 — G17: Interview Defense Bank

*Spoken-ready answers for staff-level cross-examination. Aligned to the accepted narrative; all numbers from prior gates. No claim upgrades.*

## 60-second system story
"PulseDiscover is an offline, evidence-backed two-stage recommender on 2.5M Goodreads interactions. Stage one is candidate generation — ALS f64 (the warm floor, R@20 0.085), a content-hybrid cold lane, SASRec-small, popularity, and co-occurrence. Stage two is a LambdaMART ranker that fuses those signals and roughly doubles single-retriever ranking within the candidate set. On top I built the measurement and serving layers — off-policy evaluation, position-bias correction, a FAISS serving index, and catalog-health analysis. The through-line is honesty: I tested deep models seriously and most were honest negatives, I proved the real bottleneck was candidate coverage (not ranking) and fixed *that*, and I kept every claim scope-tagged. Nothing is online-proven or deployed."

## The skeptical-interviewer attack bank (direct answers)
1. **Why did ALS beat SASRec?** Next-book choice is weakly sequence-predictable; the dominant signal is collaborative co-read, which MF captures directly. Attention had little extra structure to exploit on this catalog/compute.
2. **Was your sequence model undertrained?** I tested it: d64 at 30 GPU epochs *plateaued* (loss flat) at ~0.006 with sampled negatives; full-softmax lifted the same model 6.7× to 0.046. So the lever was the loss/objective, not epochs — and even corrected it didn't beat ALS (0.085).
3. **Why did TwoTower fail?** Item-ID-only pooled-history two-tower ≈ weak neural MF; mean-pooling short histories discards the structure ALS factorizes. R@20 0.0020 — I report it as an honest negative, not bury it.
4. **Is LTR really better if candidate coverage is limited?** Yes *within candidates* (~2× single-retriever). I never claim full-catalog from it — and G13 directly attacked the coverage ceiling once I'd shown ranking was good but capped.
5. **Are you confusing feature importance with marginal contribution?** No — explicitly separated: GPU features rank 2nd/3rd by gain yet add +0.0027 R@20 (within noise), because they're correlated/substitutable with ALS/content. High importance, ~zero marginal lift.
6. **Why not LightGCN?** Deferred honestly — graph-embedding training over 40k items isn't cheap on CPU. Documented as a future GPU-gate source, not silently skipped.
7. **Why use co-occ if it's not statistically significant?** It adds genuine marginal coverage (+3.76pp) and consistent directional lifts (R@20 +0.0177 ~1.5 SE, R@50, NDCG) at low cost. Adopted *conditionally* with the not-95%-significant caveat — re-confirm at larger N.
8. **Does FAISS improve recommendation quality?** No. Exact FAISS returns identical results (retains recall); approximate can only lose recall. Its value is latency/scale, full stop.
9. **Is this production?** No — production-*shaped* (two-stage pipeline, serving index, logging schema, guardrails, risk register) but not deployed.
10. **Are these online results?** No — offline only (recall / known-propensity OPE / simulation). No live A/B.
11. **Is your evaluation full-catalog or within-candidate?** Retrieval metrics (ALS R@20 0.085) are full-40k; LTR/slate metrics are within-candidate re-ranking with the coverage ceiling stated. I keep them separate.
12. **What would you do next with more compute/data?** Canonical full-softmax SASRec trained to convergence; LightGCN candidate source; full-population significance for co-occ; and a live A/B of the 90/10 policy (the only thing that proves lift).
13. **What's the strongest honest negative?** Two candidates: (a) full-softmax rescues SASRec 6.7× but it *still* loses to MF; (b) GPU sequence features are non-additive to the ranker. Both overturned an intuitive expectation with evidence.
14. **What's the most senior product decision?** Diagnosing that the bottleneck was candidate coverage, not ranking (G12-B), proving it (G13, +44% R@20), and then *declining* aggressive slate governance (G14) because the candidate stage already delivered discovery — picking the cheaper upstream lever over a flashier post-hoc one.

## Rapid answer bank (1–2 lines each)
- **Warm floor?** ALS f64, R@20 0.085 — best single generator.
- **Cold lane?** content-hybrid, cold R@20 0.241 where ALS is 0.
- **Best sequence model?** SASRec-small 0.0594 (GPU), still < ALS.
- **Full-softmax effect?** 6.7× over sampled-neg (0.007→0.046), not enough to beat MF.
- **Ranker win?** LTR ~2× single-retriever within candidates (0.038→0.079).
- **GPU features in LTR?** non-additive (+0.0027, noise) — correlated with existing signals.
- **Bottleneck?** candidate coverage; widening converted (+44% R@20), knee at mid (~317 cands).
- **Slate governance?** pure LTR default; aggressive policies not worth the relevance cost.
- **Co-occ?** conditional adopt; +3.76pp coverage, directional, not 95%-significant.
- **FAISS?** exact adopted (lossless, −39% p95); IVF a scale-demo only.
- **OPE?** estimators validated; positivity decisive; can't separate 90/10 from control.
- **Position bias?** raw CTR under-credits cold ~11×; correction required.
- **Creator fairness?** reach improved (+444 creators), de-concentration not — unsolved, needs training-time fix.
- **Production?** shaped, not deployed. **Online?** no. **Gold-complete?** not claimed.

## One-line claim ceiling
*"PulseDiscover is an offline, evidence-backed, production-shaped two-stage RecSys with a complete governance and serving story — every positive result scope-tagged, every deep-model negative documented, nothing presented as online lift, deployment, or solved fairness/cold-start."*

**STOP — interview defense bank complete.**
