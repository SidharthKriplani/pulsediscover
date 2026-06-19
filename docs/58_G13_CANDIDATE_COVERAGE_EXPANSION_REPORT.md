# 58 — G13: Candidate Coverage Expansion + Retrieval Ceiling Lift

*Tests whether widening/diversifying candidate generation raises the retrieval ceiling AND converts to higher LTR R@20. Motivated by G12-B (ranker captured ~half the 0.157 ceiling → bottleneck is candidate coverage, not ranking). Sources: ALS f64, SASRec-small (GPU ckpt), content-hybrid cold, popularity. Co-occurrence + LightGCN deferred (no cheap candidate artifact). CPU; deterministic md5 user split; re-ranking within candidates (not full-catalog). No online/production/gold claim. Evidence: `g13_candidate_coverage_report.json`; plot `g13_coverage_vs_budget.png`.*

## 1. Candidate coverage sweep (gold-in-candidate-set, 5,000 users)
| Config | avg cands/user | coverage ceiling |
|---|---|---|
| ALS-60 | 56 | 0.0808 |
| ALS-120 | 113 | 0.1308 |
| ALS-200 | 188 | 0.1804 |
| ALS-400 | 375 | 0.2516 |
| ALS-60 + SAS-60 + pop-30 *(≈G12)* | 119 | 0.1694 |
| ALS-200 + SAS-100 | 261 | 0.2668 |
| ALS-200 + SAS-100 + content-50 | 311 | 0.3206 |
| **WIDE: ALS-400 + SAS-200 + content-100 + pop-30** | 614 | **0.4242** |

**The ceiling rises sharply** — from ~0.157 (G12) to **0.42** (WIDE), a **2.7×** lift. ALS widening alone: 0.08→0.25 (60→400, diminishing per-candidate). Content adds cold-gold reach (0.267→0.321). SASRec-small adds collaborative breadth.

## 2. Does the higher ceiling convert to higher LTR R@20? (retrained local LambdaMART, test n=733, 1.96·SE≈0.024)
| Config | avg cands | coverage | **R@20** | R@50 | NDCG@20 | table rows |
|---|---|---|---|---|---|---|
| baseline (≈G12) | 119 | 0.169 | **0.0859** | 0.1296 | 0.0431 | 0.60M |
| **mid** (ALS200+SAS100+content50+pop30) | 317 | 0.323 | **0.1201** | 0.1678 | 0.0556 | 1.59M |
| **wide** (ALS400+SAS200+content100+pop30) | 614 | 0.424 | **0.1241** | 0.1869 | 0.0585 | 3.07M |
| *G12-B reference (enriched LTR)* | 123 | 0.157 | 0.0791 | — | — | — |

**YES — it converts.** R@20 **0.0859 → 0.1241 (+0.0382, +44%, ≈3.2 SE) — material and robust.** Unlike the G12-B GPU-feature result (within noise), widening candidates moves end-to-end recall well beyond noise.

## 3. The cost/coverage knee
- baseline→**mid**: +0.034 R@20 (0.086→0.120) for +198 cands.
- mid→wide: **only +0.004 R@20** (0.120→0.124) for +297 *more* cands.
- **Knee = mid (~317 cands/user): it captures 0.120 of the 0.124 max R@20 at half the WIDE serving cost.** WIDE buys more *coverage* (0.42 vs 0.32) but almost no extra *reachable* R@20.
- Within-set precision falls as candidates grow (R@20 / ceiling: 51% → 37% → 29%) — more distractors per slate — but absolute R@20 still rises and plateaus at mid.

## 4. Verdict — **ADOPT a wider candidate strategy (the mid config)**
Decision rule satisfied on both arms: **coverage rises materially** (0.169→0.42) **AND retrained LTR R@20 improves materially** (0.086→0.124, +44%, ≈3.2 SE). This is the **opposite of an honest negative** — it confirms G12-B's diagnosis: the bottleneck was candidate coverage, and lifting it lifts end-to-end recall.
- **Recommended default: the mid config (~317 cands/user)** — best R@20-per-serving-cost; WIDE only for coverage-maximizing surfaces.
- Feature importance at wide: `als_rank`, `author_match`, `sas_score`, `pub_year`, `als_score`, `log_pop` — ALS + content + sequence all contribute; consistent with G12-B (no single source dominates).

## 5. Honest boundaries
- Re-ranking recall **within candidates**, not full-catalog retrieval; not comparable to ALS f64 0.085.
- Deterministic md5 user split (reproducible); significance vs ≈0.024 (1.96·SE, n=733).
- LTR uses **local features** (G12-B showed GPU score columns non-additive) — so this isolates the *candidate-coverage* effect cleanly.
- **Co-occurrence and LightGCN candidate sources deferred** (no cheap candidate artifact) — a documented future lever that could raise coverage further.
- No FAISS (a serving/ANN demo, not a coverage lever). No online/production/RiskFrame-gold-complete claim.

## 6. Status & next
- ✅ G13 done. **Widening candidate generation raised the ceiling 2.7× and lifted LTR R@20 +44% (material); knee at the mid config (~317 cands).** Bottleneck confirmed as retrieval, not ranking.
- ⏭️ Open levers: (a) add **co-occurrence / LightGCN** candidate sources to push the ceiling further (G13b/G11F); (b) **G11E FAISS** as an ANN/serving-latency demo over the adopted candidate set; (c) **G13 MMR final-slate reranker** + **G14 Deep Defense Kernel**.

---

## 7. Acceptance addendum (ACCEPTED — gate passed)
**1. Final accepted default = the MID config** (ALS-200 + SASRec-small-100 + content-50 + pop-30; ~317 cands/user; coverage 0.323; LTR R@20 0.1201). Adopted as PulseDiscover's default candidate-generation stage.

**2. Why WIDE is not the default.** WIDE (614 cands, coverage 0.424) lifts R@20 only to 0.1241 — **+0.004 over mid for ~2× the candidates**. The reachable-recall gain has plateaued; the extra coverage is not worth the serving cost. WIDE is reserved for coverage-maximizing surfaces only.

**3. Serving-cost tradeoff.** Candidate count ≈ request-time retrieval/scoring cost (ties to the G1 latency budget). mid ≈ 317 cands sits at the knee: it captures 0.120 of the 0.124 max R@20 at ~half WIDE's cost. baseline→mid is the high-ROI move (+0.034 R@20 for +198 cands); mid→wide is low-ROI (+0.004 for +297).

**4. Safe interview line.** *"I showed the two-stage system was candidate-bound, then widened candidate generation: coverage went 0.16→0.42 and offline re-ranking R@20 rose ~44% (0.086→0.124). I adopted the knee config — ~half the candidates for ~all the gain — rather than the widest, because reachable recall had plateaued and candidate count is serving cost."*

**5. Updated claim boundary.** Still **offline re-ranking within candidate sets** — not full-catalog retrieval, **no online lift, no production deployment, no RiskFrame-gold-complete claim.** The adopted default is an offline candidate-generation choice backed by reproducible evidence, nothing more.

**STOP — G13 accepted (mid = default); G14 plan written (docs/59), awaiting approval.**
