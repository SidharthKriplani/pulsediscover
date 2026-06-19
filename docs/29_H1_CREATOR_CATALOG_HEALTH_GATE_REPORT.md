# 29 — H1 Gate Report: Creator / Catalog Health Simulation on the Two-Lane Policy

*Company question: **Does the 90/10 two-lane policy actually improve creator/catalog exposure *health* over repeated rounds, or does it merely improve cold-start recall while still concentrating exposure?** Serving-side complement to P1 (P1 fixed how we *measure* biased exposure; H1 tests whether the serving policy changes *who receives* it). Domain-scoped. Proxy reward — **no business-lift claim.** No new training / SASRec / PRD / final packaging.*

## 1. Setup (auditable)
- **Policies:** `als_only` (control), `q90_10` (=reserve-2, primary), `q80_20` & `rrf` (aggressive arms). Warm lane = ALS; cold lane = C1 content-hybrid reserve.
- **Population:** 2,400 users at **D2 prevalence (76.5% warm / 23.5% cold)**, each with a held-out gold.
- **Rounds:** 12. Exposure (impressions) accumulates over items and creators.
- **Feedback loop (simple, auditable):** rich-get-richer on the **warm lane only** — warm rank `= base ALS rank + γ·log1p(cumulative warm exposure)`, γ=2. **The cold reserve is protected from the boost.** Click = examined(PBM) AND item==gold.
- **Health metrics:** item & creator exposure Gini, effective #creators `exp(entropy)`, long-tail share (outside top-1% items), distinct items/creators surfaced, cold-item exposure share, distinct **cold-lane (new) creators**.

## 2. Structural health — final round (the cross-policy comparison)
| Policy | creator Gini | eff. #creators | distinct items | distinct creators | **new (cold) creators** | cold-exp share | long-tail share |
|---|---|---|---|---|---|---|---|
| als_only | 0.845 | 147 | 2,553 | 839 | **0** | 0.000 | 0.830 |
| **q90_10** | **0.861** | 156 | 2,994 | 1,054 | **444** | 0.100 | 0.813 |
| q80_20 | 0.858 | 170 | 3,115 | 1,179 | 614 | 0.200 | 0.806 |
| rrf | 0.830 | 214 | 2,842 | 1,247 | 799 | 0.500 | 0.810 |

## 3. Two findings that answer the company question directly

**(a) 90/10 broadens *reach* clearly.** vs ALS-only it surfaces **+441 distinct items (+17%)**, **+215 distinct creators (+26%)**, and **+444 brand-new creators** that are reachable *only* through the cold lane (ALS-only reaches 0). That is real catalog widening.

**(b) But 90/10 does NOT de-concentrate the head — creator Gini gets slightly *worse* (0.861 vs 0.845).** Adding hundreds of tiny-exposure cold creators lengthens the tail while the dominant warm creators keep their mass → *more* inequality by Gini, even as effective-#creators and distinct counts rise. **The reserve adds breadth, not redistribution.** Only **rrf** actually lowers Gini (0.830) — and it costs −44% warm recall (C2).

> **Answer:** 90/10 improves cold recall **and** catalog *reach/breadth* and protects a cold-exposure share — but it does **not** by itself fix head concentration. To lower the Gini you need a bigger reserve (rrf, too costly) or **creator-aware reranking** (a future lever).

## 4. The dynamic result — feedback amplification vs the protected reserve
Over the 12 rounds, the rich-get-richer loop **concentrates the warm lane for every policy**: creator Gini rises and effective-#creators falls monotonically.

| Policy | creator Gini r1→r12 | eff. #creators r1→r12 | cold-exp share (every round) |
|---|---|---|---|
| als_only | 0.831 → 0.845 (+0.015) | 159 → 147 | 0.000 |
| q90_10 | 0.848 → 0.861 (+0.013) | 169 → 156 | **0.100 (flat)** |
| rrf | 0.817 → 0.830 (+0.013) | 231 → 214 | 0.500 (flat) |

- **Amplification is policy-independent:** the warm head self-concentrates over time no matter which policy serves it (+~0.013–0.015 Gini, ~7–8% drop in effective creators in just 12 rounds).
- **The cold reserve is the one thing that holds firm:** cold-exposure share stays *exactly* at the reserve fraction (10/20/50%) **every round**. The reserve **guarantees a constant cold-discovery share even as the warm lane amplifies** — ALS-only's cold share is 0 forever. *This protected-share guarantee is 90/10's real structural value here.*
- **Honest caveat:** cold-exposure share is a **structural guarantee (= reserve fraction), not an emergent behavioral win.**

## 5. Relevance cost (source of truth = C2, not this sim)
H1's dynamic warm-hit rate is **too noisy to be a guardrail** (gold-hit ~1% over ~1,800 warm users → large relative noise). The relevance guardrail stays the **C2 static warm R@20 loss: q90_10 −8.1% (PASS ≤10%), q80_20 −17%, rrf −44%.** H1 does not override C2.

## 6. Decision logic
- **90/10 stays the default A/B candidate — now from a catalog-health view too, with a precise claim:** it widens reach (+26% creators, +444 new cold creators), **guarantees a protected 10% cold-exposure share that survives feedback amplification**, and stays within the C2 warm guardrail (−8.1%). ✅
- **It does NOT improve exposure *concentration* (Gini flat-to-worse).** If the company's catalog-health goal is *de-concentrating the head* (not just widening reach), **90/10 alone is insufficient → creator-aware reranking is recommended** as a follow-on lever. ✅ (clearly stated, not oversold)
- **Aggressive arms = exploration only.** rrf is the only policy that lowers Gini and maximizes new-creator reach (799), but at −44% warm cost → discovery-heavy A/B arm with warm-retention guardrails, not default. 80/20 in between.
- **Cross-cutting risk surfaced:** warm-lane exposure amplifies over rounds for *all* policies — a feedback-loop governance concern independent of the two-lane choice (motivates exposure-aware/anti-amplification serving).

## 7. No-overclaim check
- Proxy reward (examined gold-hit); **no engagement/business-lift claim**. ✅
- "90/10 improves health" qualified precisely: **reach yes, de-concentration no**; Gini reported even though it cuts against the discovery narrative. ✅
- Cold-share labeled a structural guarantee, not behavioral. ✅
- Noisy dynamic warm-hit explicitly *not* used as guardrail; C2 is the relevance source of truth. ✅
- Feedback loop labeled simple/auditable, not production dynamics. ✅

## 8. Artifacts
- `outputs/evidence/domain_creator_catalog_health_report.json` (final + full per-round trajectories, deltas vs ALS, config, relevance source-of-truth)
- `outputs/plots/domain_creator_catalog_health.png` (creator Gini, distinct creators, cold-exp share, long-tail share over rounds)

## 9. Gate status & next step
- ✅ H1 done. **90/10 widens catalog reach and protects cold-exposure share under feedback amplification, but does not de-concentrate the head** (Gini flat-to-worse); relevance stays within C2 guardrail; **creator-aware reranking recommended if de-concentration is the goal.** Aggressive arms = exploration only.
- ⏸️ **Options next:** (a) **creator-aware reranking** prototype (can we lower creator Gini at ≤10% warm cost — the lever H1 says is missing?); (b) SNIPS/clipping/PAL variance-reduction on the per-item position-bias estimator (P1 follow-on); (c) **package the full A/B-readiness spec** (control/arms/guardrails/logging/estimators across C2·O1·O1b·P1·H1); (d) external GPU canonical SASRec. No further step run.

**STOP — awaiting H1 review.**
