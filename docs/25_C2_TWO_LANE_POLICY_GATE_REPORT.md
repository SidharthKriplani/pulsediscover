# 25 — C2 Gate Report: Two-Lane Serving Policy & Cold-Start Exposure Tradeoff

*Company question: **If ALS f64 is best for warm items and content-hybrid is best for new/cold items, how should a platform allocate recommendation slots between warm relevance and new-content discovery?** This is a recommendation-slot-allocation experiment, not a model benchmark. Domain-scoped (Goodreads fantasy/paranormal). No SASRec / OPE / position-bias / PRD / new T3 model.*

## 1. Setup
- **ALS f64 refit on PRE-TEST only** (ts < t2) so cold items stay genuinely unseen (91,732 users × 40,541 warm items).
- Two disjoint candidate lanes per user: **warm = ALS top-200** (over pre-test items), **cold = content-hybrid top-200** (same-author + same-series + TF-IDF over the 1,420 new-in-test items).
- Eval: 5,000 **warm** test golds (warm items) + 5,000 **cold** test golds (new items); warm golds reachable only via warm slots, cold only via cold slots. **Overall = prevalence-weighted** (D2: warm 76.5%, cold 23.5%).

## 2. Policy results (@20; full @50/100/200 in JSON)
| Policy | Warm R@20 | Cold R@20 | **Overall R@20** (prev-wt) | Cold share (top-20) | Distinct cold surfaced | Creator Gini | Flag |
|---|---|---|---|---|---|---|---|
| ALS-only | **0.0520** | 0.0000 | 0.0398 | 0.00 | 0 | 0.861 | baseline |
| **90/10 quota** | 0.0478 | 0.0504 | 0.0484 | 0.10 | 735 | 0.880 | **SHIP-CANDIDATE** |
| **explore-2-slots** | 0.0478 | 0.0504 | 0.0484 | 0.10 | 735 | 0.880 | **SHIP-CANDIDATE** |
| 80/20 quota | 0.0430 | 0.0924 | 0.0546 | 0.20 | 1,020 | 0.875 | HOLD |
| 70/30 quota | 0.0388 | 0.1230 | 0.0586 | 0.30 | 1,163 | 0.866 | HOLD |
| RRF-blend | 0.0290 | 0.1714 | **0.0625** | 0.50 | 1,299 | **0.842** | HOLD |
| cold-only | 0.0000 | 0.2358 | 0.0554 | 1.00 | 1,391 | **0.770** | HOLD |

(Warm-recall guardrail: ≤10% relative loss vs ALS-only's 0.0520.)

## 3. The tradeoff (the company decision)
- **Adding cold slots monotonically *raises overall recall*** (0.0398 → 0.0484 → 0.0546 → 0.0586 → 0.0625) because ALS scores 0 on the 23.5% cold traffic — even a 10% cold quota lifts overall **+21%**.
- **…and monotonically *lowers warm recall*** (0.052 → 0.048 → 0.043 → 0.039 → 0.029 → 0).
- **More cold also improves creator diversity** (Gini 0.861 → 0.770 at cold-only) and **catalog reach** (distinct cold surfaced 0 → 1,391).
- So it's a genuine relevance-vs-discovery dial: heavy cold maximizes overall recall + diversity but destroys warm relevance.

## 4. Decision
- **SHIP-CANDIDATE (offline → A/B): 90/10 quota ≈ reserve-2-slots-in-top-20.** It lifts overall R@20 +21% (0.0398→0.0484), turns cold R@20 from **0 → 0.050**, surfaces **735 distinct new books**, at only **−8.1% warm recall** (inside the ≤10% guardrail). The reserve-2-slots variant is the simplest implementation and is equivalent at the top-20 feed.
- **HOLD: 80/20, 70/30, RRF-blend, cold-only.** They give more discovery / higher overall recall / better diversity, but breach the warm-recall guardrail (−17% to −44%). They are legitimate **aggressive-discovery A/B arms**, not the default.
- **RRF-blend is the overall-recall maximizer (0.0625)** and best diversity short of cold-only — worth an A/B arm if the platform prioritizes discovery over warm relevance, but it is **not** ship-safe under a warm guardrail.

## 5. Coverage / concentration / runtime
- Cold quota directly controls cold exposure share (≈quota by construction). The load-bearing numbers: **distinct cold items surfaced** (735 at 10% → 1,391 at cold-only) and **creator Gini** (drops with more cold) — i.e., the cold lane materially widens catalog + creator reach.
- Runtime: policy simulation over prebuilt lists ~9 s; list build ~17 s; ALS pre-fit ~34 s — **all CPU-feasible**.

## 6. Honest notes / no-overclaim
- **OFFLINE slot-allocation simulation — NOT A/B, NOT business lift.** "SHIP-CANDIDATE" means *promote to an online test*, not *proven*. ✅
- Warm & cold are separate eval sets; overall is prevalence-weighted (not a single measured stream). ✅
- Warm-recall base is low (0.052, strict next-item on global-time test) → relative losses are sensitive; reported as-is. ✅
- **Eval-setup label (not a regression):** C2 warm ALS R@20 = **0.052** is from the **C2 global-time serving-policy setup** (ALS f64 refit on pre-test ts<t2; warm golds = test-window warm items). This is a *different eval* from the earlier **5k leave-last-out f64 ALS floor R@20 = 0.0846** (D3B/D3A+: held-out *last* item per user, ALS trained on core-minus-held). Both are correct for their setup; the gap is the harder global-time warm-gold definition here, **not** a model regression. Keep both, labeled. ✅
- Cold exposure share at top-20 is ~quota by construction; diversity/coverage/Gini are the substantive deltas. ✅
- Lanes use disjoint pools (warm=pre-test items, cold=new items); no leakage between them. ✅

## 7. Artifacts
- `outputs/evidence/domain_two_lane_policy_report.json` (all policies × full metrics)
- `outputs/plots/domain_two_lane_tradeoff.png` (warm vs cold recall @20)
- lists/checkpoints: `data/interim/c2_lists.pkl`, `c2_als.pkl`

## 8. Gate status & next step
- ✅ Two-lane serving policy simulated; **90/10 (reserve-2-slots) is the ship-candidate**; aggressive arms documented as HOLD; cold lane materially improves catalog/creator reach at a quantified warm cost.
- ⏸️ **Options next:** (a) domain **OPE on the two-lane policy** (estimate the warm/cold/overall tradeoff under a known-propensity simulator — connects C2 to the OPE differentiator); (b) **creator-health / position-bias** governance on the chosen policy; (c) external GPU canonical SASRec. No further step run.

**STOP — awaiting C2 review.**
