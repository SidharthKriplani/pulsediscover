# 65 — G17: Deep Defense Kernel

*The first-principles technical defense layer for PulseDiscover. Makes the project interview-defensible at senior/staff level by tracing every important technique along one chain. Companion files: `66_G17_METHOD_DEFENSE_CARDS.md` (18 cards), `67_G17_INTERVIEW_DEFENSE_BANK.md` (Q&A), `outputs/evidence/g17_defense_traceability_matrix.json`. **No new models, no new metrics, no claim upgrades.***

## The defense chain (every method traces all seven links)
**theory → product decision → data/logging/evaluation design → implementation artifact → evidence → business consequence → interview-safe claim.**

A technique is "defended" only if it survives all seven: *why it should work (theory) → what decision it serves (product) → how we set up data/eval to test it honestly → what we actually built → what the numbers said → what it costs to get wrong → and the bounded claim we can state out loud.*

## The accepted project narrative (binding — preserved verbatim in all G17 artifacts)
1. **ALS f64 is the strongest standalone warm collaborative floor** (R@20 0.085, 5k LLO).
2. **Sequence models were tested seriously; SASRec-small improved (0.036 CPU → 0.0594 GPU) but still did not beat ALS.**
3. **Full-softmax SASRec showed the loss/objective mattered (0.0068 sampled-neg → 0.0456 full-softmax, 6.7×) but did not close the gap to ALS.**
4. **TwoTower was built and is an honest negative (R@20 0.0020), not a win.**
5. **LTR (LambdaMART) was the material ranking win *inside candidate sets*** (ALS-rank 0.0382 → LTR 0.0791; mid 0.1201).
6. **GPU sequence features were non-additive to LTR** (+0.0027 R@20, within noise).
7. **G13 showed candidate coverage was the binding lever; widening converted** (coverage 0.157→0.42; R@20 0.086→0.124, +44%, ~3.2 SE).
8. **G14 showed aggressive post-hoc slate governance was mostly not worth the relevance cost; pure LTR remains default.**
9. **G15 co-occurrence is conditional/adopted** (+3.76pp marginal coverage, directional LTR lift, **not 95%-significant at n=733**).
10. **G16 FAISS exact (FlatIP) adopted as the ALS-slice serving index; IVF is a scale-demonstration only.**
11. **No online lift, no production, no live A/B, no full-catalog claim where evaluation is within-candidate.**

## Why this layer exists (RiskFrame audit axis #4)
The G2/G3/G4 company-realism layer and the evidence ledger establish *what* was built and *that* it's honest. The Deep Defense Kernel adds the missing *why-at-first-principles* per technique — the binding gap from the RiskFrame audit (docs/49, axis #4 scored 4/10). It is the difference between "I ran these methods" and "I can defend every method's mechanism, assumptions, failure modes, and kill conditions under staff-level cross-examination."

---

## What a skeptical interviewer will attack
*(Full Q&A with traps in `67_G17_INTERVIEW_DEFENSE_BANK.md`; headline answers here.)*

- **"Why did ALS beat SASRec?"** Book next-item is weakly sequence-predictable; the dominant signal is collaborative co-read, which MF captures directly. Sequence models need stronger signal/data than this domain offers on CPU.
- **"Was your sequence model undertrained?"** Tested directly: the d64 model at 30 GPU epochs plateaued (loss flat) at ~0.006 with sampled negatives; full-softmax lifted it to 0.046. So the lever was the *loss/objective*, not epochs — and even fixed, it didn't beat ALS.
- **"Why did TwoTower fail?"** Item-ID-only pooled-history two-tower ≈ neural collaborative retrieval; on a co-occurrence-dominated catalog with short histories, mean-pooling loses the structure ALS factorizes. Built it, reported 0.0020 straight.
- **"Is LTR really better if candidate coverage is limited?"** Yes *within candidates* (~2× single-retriever), and I'm explicit it's re-ranking recall, not full-catalog. G13 then attacked the coverage ceiling directly.
- **"Confusing feature importance with marginal contribution?"** No — I separated them: GPU features rank 2nd/3rd by gain but add +0.0027 R@20 (within noise) because they're correlated/substitutable. That distinction is in the report.
- **"Why not LightGCN?"** Deferred honestly — graph-embedding training over 40k items isn't cheap on CPU; documented as a future GPU-gate source, not silently skipped.
- **"Why use co-occ if it's not significant?"** It adds genuine marginal coverage (+3.76pp) and consistent directional lifts (R@20/R@50/NDCG) at low cost; adopted *conditionally* with the explicit not-95%-significant caveat — neither overclaimed nor dismissed.
- **"Does FAISS improve recommendation quality?"** No. FAISS retains recall (exact = identical, approx ≤ exact); its value is latency/scale. Stated up front.
- **"Is this production?"** No — production-*shaped* (two-stage retrieve→rank→slate, serving index, logging schema, guardrails) but not deployed.
- **"Are these online results?"** No — all offline (recall / known-propensity OPE / simulation); no live A/B.
- **"Full-catalog or within-candidate?"** The LTR/slate numbers are within-candidate re-ranking (ceiling stated); the retrieval R@20 (ALS 0.085) is full-40k. I never conflate them.
- **"What next with more compute/data?"** Canonical full-softmax SASRec to convergence + LightGCN candidate source + a live A/B of the 90/10 policy; full-population significance for co-occ.
- **"Strongest honest negative?"** Either the full-softmax-rescues-but-still-loses SASRec finding, or that GPU features are non-additive to the ranker — both overturned an intuitive expectation with evidence.
- **"Most senior product decision?"** Diagnosing (G12-B) that the bottleneck was candidate coverage not ranking, then proving it (G13) and *declining* aggressive slate governance (G14) because the candidate stage already handled discovery — choosing the cheaper upstream lever over a flashier post-hoc one.

**STOP — Deep Defense Kernel overview; cards in docs/66, bank in docs/67, matrix in evidence.**
