# 59 — G14 Plan: Final Slate Policy + MMR / Catalog-Guardrail Reranker

*Proposal only — not executed. The third stage after candidate generation (G13, mid default) and LTR ranking (G12-B): turn the ranked candidate list into the **final top-20 slate** under product/governance constraints, and measure the relevance↔governance tradeoff.*

## Company question
**Once we have an LTR-ranked candidate list, how do we assemble the final 20-slot slate — balancing relevance, cold/new discovery, creator/catalog exposure, and diversity — without unacceptable warm-relevance loss?**

## Pipeline position
candidate generation (G13 mid: ~317 cands) → **LTR ranking (G12-B LambdaMART)** → **G14 final-slate policy (this gate)** → top-20.

## What to build (slate policies over the LTR-ranked list)
1. **Pure LTR slate** — top-20 by LTR score (the relevance-max reference).
2. **LTR + MMR diversity** — greedy MMR: `score = λ·LTR_rel − (1−λ)·max_sim(item, chosen)`; creator/series/content similarity for the penalty. Sweep λ.
3. **LTR + cold/discovery reserve** — reserve k slots (k∈{1,2,4}) for cold/new items (the 90/10-style reserve, now on the LTR list).
4. **LTR + creator/catalog cap** — max items per creator in the slate (cap∈{1,2,3}); spill to next-best different-creator item.
5. **Combined final policy** — LTR + reserve + creator cap + light MMR, tuned to hold the warm guardrail.

## Metrics (per policy, test users)
- **Relevance:** R@20 / R@50 / NDCG@20 (within-candidate, as G12-B/G13).
- **Discovery:** cold-item exposure share, distinct cold items/creators surfaced.
- **Catalog health:** creator/catalog exposure Gini, effective #creators, catalog coverage, long-tail share.
- **Diversity/novelty:** intra-slate creator/series diversity, novelty (inverse popularity).
- **Guardrail:** warm-recall loss vs pure-LTR slate; **policy-violation rate** (slates breaching cap/reserve).
- **Tradeoff curve:** relevance (R@20) vs governance (creator Gini / cold share) across λ, cap, reserve.

## Decision rule
- **Adopt** a final-slate policy only if it **improves discovery/catalog-health metrics without unacceptable warm-relevance loss** (reuse the ≤10% warm-loss guardrail from C2/AB1, applied here vs the pure-LTR slate).
- If **governance improves but relevance collapses** → honest negative; recommend a softer policy (smaller reserve / looser cap / higher λ).
- Report the **knee** on the relevance↔governance curve (best health gain per unit relevance lost).

## Honest boundaries (pre-committed)
- Offline re-ranking within candidates; **no online lift, no production, no RiskFrame-gold-complete claim.**
- Reuses H1/H2 finding: creator caps de-concentrate the head but cost relevance — G14 quantifies that tradeoff *on the LTR slate* (not just candidate lists).
- Deterministic user split; all numbers stamped N / source / reproducibility / noise threshold.

## Required outputs (on approval)
- `docs/60_G14_FINAL_SLATE_POLICY_REPORT.md`
- `outputs/evidence/g14_final_slate_policy_report.json`
- `outputs/plots/g14_relevance_governance_tradeoff.png`

## Scope guardrails
- CPU-feasible (re-ranks the existing LTR-scored mid candidate table; no new model training).
- One coherent policy sweep + tradeoff curve; no FAISS, no new candidate sources.

---

## Remaining path to gold
| Gate | Purpose | Status |
|---|---|---|
| **G14** | Final slate policy / MMR / catalog-guardrail reranker (this plan) | proposed |
| **G15** | One extra candidate-source depth gate — **LightGCN or co-occurrence** candidates (raise the ceiling further; G13 deferred these) | planned |
| **G16** | **FAISS / serving-latency demo** over the adopted (mid) candidate set — ANN systems demo, clearly labeled (not a quality lever) | planned |
| **G17** | **Deep Defense Kernel** — first-principles defense cards per technique (ALS/SASRec/GRU/two-tower/LTR/MMR/IPS/SNIPS/DR/PAL/ANN/neg-sampling); closes RiskFrame audit axis #4 | planned |
| **G18** | **Final Gold Audit** — re-run the RiskFrame 15-axis audit, finalize claim-boundary table, resume-safe closeout | planned |

*Sequence rationale:* G14 completes the serving pipeline (candidates→rank→slate); G15 pushes the binding retrieval ceiling once more; G16 adds the systems/serving layer; G17 closes the last binding RiskFrame gap (deep defense); G18 audits to gold and locks claims. Each remains a gated stop.

**STOP — G14 plan + remaining path written; awaiting approval to execute G14.**
