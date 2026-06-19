# 30 — H2 Gate Report: Creator-Aware Reranking Prototype

*Company question: **Can we reduce creator exposure concentration while keeping warm relevance loss within the same ≤10% guardrail (vs ALS-only)?** H2 adds a creator-aware reranking layer on top of the two-lane 90/10 policy and tests the relevance↔concentration tradeoff. Offline reranking simulation — **not** business lift; creator fairness is **not** "solved." No new training / SASRec / PRD.*

## 1. Setup (auditable)
- **Base:** rerank the **warm lane** of 90/10 over a deeper ALS pool (top-60), **keeping the 2-slot cold reserve** (so cold recall is unchanged across rerank strategies).
- **Strategies:** creator **cap** (max K items/creator: K=3/2/1), **MMR** creator penalty (λ=0.3/1.0), **cap2+cold-creator-boost**, and budget-reallocation **reserve-1 + cap** (1 cold slot, 19 warm capped).
- **Population:** 4,000 users at D2 prevalence (76.5% warm / 23.5% cold), held-out golds.
- **Guardrail:** warm R@20 loss vs **ALS-only** ≤ 10%. **Metrics:** warm/cold/prevalence Recall@20/50, **global** creator Gini, **warm-only** creator Gini, effective #creators, distinct (cold) creators, runtime.

## 2. The relevance ↔ creator-concentration tradeoff (R@20)
| Strategy | global creator Gini | warm-only Gini | warm R@20 | **warm loss vs ALS** | guardrail | distinct creators | new (cold) creators |
|---|---|---|---|---|---|---|---|
| als_only | 0.843 | 0.843 | 0.0461 | 0.0% | ✅ | 923 | 0 |
| **base_90_10** | 0.862 | 0.840 | 0.0415 | **−10.0%** | ✅ (at edge) | 1,192 | 519 |
| cap3 | 0.855 | 0.830 | 0.0376 | −18.4% | ❌ | 1,218 | 519 |
| cap3 + reserve-1 | 0.850 | — | 0.0395 | −14.3% | ❌ | 1,119 | 377 |
| cap2 | 0.848 | 0.823 | 0.0343 | −25.6% | ❌ | 1,253 | 519 |
| cap1 | **0.832** | **0.805** | 0.0297 | −35.6% | ❌ | 1,355 | 519 |
| mmr (λ=1.0) | 0.832 | 0.805 | 0.0297 | −35.6% | ❌ | 1,355 | 519 |
| cap2 + cold-boost | 0.873 ❌worse | — | 0.0392 | −15.0% | ❌ | 1,000 | 519 |

Plot: `outputs/plots/domain_creator_rerank_tradeoff.png` (green = passes guardrail).

## 3. Findings (honest)
1. **The reranking mechanism works on the warm head.** Warm-only creator Gini falls cleanly with stronger caps: 0.843 (als) → 0.840 (base) → 0.830 (cap3) → 0.823 (cap2) → **0.805 (cap1/MMR)**. Capping does exactly what it should — it removes repeat (often **series**) authors and spreads warm exposure.
2. **But you cannot buy de-concentration within the ≤10% budget — the cold reserve already spent it.** base_90_10 sits **exactly at −10%**, leaving *zero* headroom. The mildest cap that moves Gini (cap3) already costs **−18%**; reaching global Gini ≤ ALS-only requires cap1/MMR at **−35%**. Every de-concentrating strategy **fails the guardrail.**
3. **Budget reallocation doesn't rescue it.** Dropping to a 1-slot cold reserve (cap3+reserve-1) improves the loss to −14.3% — still failing, and it sacrifices cold reach (new creators 519→377). Trading cold discovery for warm-head room is a lateral move, not a win.
4. **Global Gini is inflated by 90/10's own cold reach** (consistent with H1): base_90_10 global Gini 0.862 > als 0.843 even though its *warm-only* Gini (0.840) is *below* als. The +519 tiny-exposure cold creators lengthen the tail. So "90/10 has worse creator Gini" is largely the cost of *reach*, not warm-head worsening.
5. **Latency is not the blocker.** Cap reranking is ~0.05–0.10 ms/user; MMR ~0.7 ms/user. Relevance, not runtime, is the binding constraint.

## 4. Decision
- **No SHIP/A-B candidate for creator *de-concentration* emerges within the ≤10% guardrail.** Answer to the company question: **No** — not via cheap post-hoc reranking on this candidate generator. The 90/10 cold reserve consumes the entire relevance budget, so any cap strong enough to lower global Gini breaks the guardrail. ✅ (honest negative)
- **90/10 should remain a cold-discovery / catalog-reach policy, not be positioned as a head-de-concentration tool.** (Reaffirms H1: reach ✅, de-concentration ✗.)
- **Creator de-concentration needs a stronger intervention than reranking:** (a) accept a larger relevance budget and run cap2/cap1 as an **exploration-only** arm (they genuinely lower warm-only Gini to ~0.805, but at −25–36% warm); (b) bake **creator-fairness into the warm retrieval model** (training-time), not a post-hoc rerank; or (c) use a **less creator-concentrated candidate generator** (the ALS gold sits high and is series-clustered, so caps displace relevant items). 
- **Exploration-only:** cap2, cap1, MMR — meaningful de-concentration, guardrail-breaking.
- **Reject:** cap2+cold-boost (raises Gini, the boost concentrates onto a few cold-owning creators).

## 5. No-overclaim check
- Offline reranking sim on held-out golds; **no business-lift claim**; creator fairness **not** "solved." ✅
- **Creator REACH (distinct creators) separated from creator DE-CONCENTRATION (Gini/effective creators)** throughout. ✅
- Warm-only vs global Gini reported so the cold-reach inflation isn't mistaken for warm-head worsening. ✅
- Honest negative: the guardrail-passing region contains no de-concentrating strategy. ✅

## 6. Artifacts
- `outputs/evidence/domain_creator_rerank_report.json` (all strategies × recalls, global+warm Gini, effective/distinct/cold creators, guardrail flags, runtime)
- `outputs/plots/domain_creator_rerank_tradeoff.png` (creator-Gini vs warm-loss, guardrail line)

## 7. Gate status & next step
- ✅ H2 done. **Creator-aware reranking cannot reduce creator concentration within the ≤10% warm guardrail** — the cold reserve already spends the budget; the warm-head mechanism works but is too recall-costly. 90/10 stays a cold-discovery/reach policy; de-concentration needs a training-time or budget-relaxed intervention.
- ⏸️ **Options next:** (a) **package the full A/B-readiness spec** (control = ALS-only; default arm = 90/10; exploration arms = cap2/rrf; guardrails, logging schema from P1, estimators from O1/O1b, health caveats from H1/H2); (b) creator-fairness at the warm-model/training stage (larger effort); (c) SNIPS/clipping/PAL variance reduction on the P1 estimator; (d) external GPU canonical SASRec. No further step run.

**STOP — awaiting H2 review.**
