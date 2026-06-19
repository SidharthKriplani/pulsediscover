# 28 — P1 Gate Report: Position-Bias Correction on the Two-Lane Served Policy

*Company question: **If we deploy or A/B-test the 90/10 two-lane policy, can we correctly read logged clicks when top-position exposure strongly biases what gets observed?** This is the measurement layer required before trusting A/B logs — **not** another model benchmark and **not** a business-lift claim. Domain-scoped. No creator-health sim / SASRec retry / new training / PRD.*

## 1. Setup (auditable)
- **Served policies (top-20 slates):** `q90_10` (=reserve-2, primary), `als_only` (control), `q80_20` (aggressive arm). Warm slots first, cold slots at the bottom (90/10 → cold at positions 18–19; 80/20 → 16–19).
- **Position-bias model:** PBM examination by rank, `exam(pos)=0.95·(1/(pos+1))^0.7` — **0.95 at pos 0 → 0.117 at pos 19** (an ~8× exposure gap top-to-bottom). This is the KNOWN propensity.
- **Logged reward:** `click = examined(pos) AND item == user's gold` (relevance proxy). **True relevance is kept known** (gold-hit, examination-free) so every estimator can be scored against ground truth.
- **Estimators:** (1) **naive** observed CTR by position/lane/item; (2) **IPS** examination-correction `click / exam(pos)`; (3) SNIPS-style self-normalization implicit in lane means. 10,000 instances (5k warm + 5k cold) × 20 slots = 200k logged records.

## 2. Headline — naive CTR inverts the truth at the cold slots (90/10)
| Position | naive CTR | IPS-corrected | **true relevance** |
|---|---|---|---|
| pos 0 (top warm) | 0.0018 | 0.0019 | 0.0021 |
| pos 18 (cold) | 0.0014 | 0.0116 | **0.0154** |
| pos 19 (cold) | 0.0009 | 0.0077 | **0.0098** |

**The cold slots have the *highest* true per-slot relevance of the entire slate, but naive CTR ranks them near the *bottom*.** Reason: by construction the cold lane targets each cold user's gold, so its slots are highly relevant — but they sit where examination is ~12% vs 95% at the top. Naive CTR is `exam(pos) × true_rel`, so it crushes the bottom slots. **IPS divides the bias back out and recovers ~the true profile** (pos 18: 0.0116 vs true 0.0154; pos 19: 0.0077 vs 0.0098). Plot: `outputs/plots/domain_position_bias_two_lane.png`.

## 3. Warm vs cold lane (the two-lane effect)
| Lane (90/10) | naive | IPS | **true** | naive vs true |
|---|---|---|---|---|
| warm lane (pos 0–17) | 0.00033 | 0.00095 | 0.00133 | −75% |
| **cold lane (pos 18–19)** | 0.00115 | 0.00964 | **0.01260** | **−91% (11× under-credit)** |

- **By true relevance the cold lane is 9.5× more relevant per slot than the warm lane** (0.0126 vs 0.0013) — cold users' gold lives in the cold slots, warm gold is spread thin over 18 warm slots.
- **Naive CTR under-credits the cold lane by 11×** (0.00115 vs true 0.0126) and compresses the warm↔cold gap from 9.5× down to ~3.5×, in absolute terms making both lanes look like near-zero engagement.
- **IPS recovers 77% of true cold-lane relevance** (0.00964 / 0.0126), restoring the correct lane ordering. Same pattern for 80/20 (cold lane: naive 0.0014 → IPS 0.0107 → true 0.0115).

## 4. The misleading naive conclusion this prevents
> *"Raw CTR shows the cold discovery slots get almost no clicks (~0.001) → discovery doesn't work → revert to ALS-only."*

This is **false and purely a position artifact.** Those slots carry the highest per-slot relevance in the slate; they look dead only because they're at the bottom of the page (12% examination). **Killing the 90/10 discovery lane on raw bottom-of-page CTR would be a measurement error, not a product signal.** Position correction is what stops PulseDiscover from sabotaging its own cold-start strategy with biased logs.

## 5. Bias / variance tradeoff (honest)
- **Aggregate (lane & position) bias reduction is large and reliable:** IPS moves cold-lane estimate from 11× under-true to within ~23% of true; position profile recovered.
- **Per-item, IPS trades bias for variance — and at this sparsity loses on MSE.** 90/10 items (≥20 impressions, n=1,355): correlation-to-true barely improves (naive 0.437 → IPS 0.443), but **MSE rises** (naive 1.03e-4 → IPS 5.16e-4). Most items get few low-position impressions, so the `1/exam` up-weighting of rare bottom clicks injects variance that dominates the per-item estimate. **Lesson: trust position-corrected estimates at the lane/position aggregate; per-item IPS needs far more impressions or shrinkage.**
- **Weights are bounded here** (1/exam ∈ [1.05, 8.57]) because the slate is only 20 deep — so variance is manageable at the aggregate. **Deeper slates or rarer bottom examination → larger weights → worse variance** (already visible per-item). SNIPS/clipping/PAL would be the next mitigations.

## 6. Exposure concentration — correction is *not* a de-concentration lever
Item credit Gini: clicks 0.985 → IPS-credit 0.990 (90/10); same direction for all policies. **Position correction slightly *increases* credit concentration**, because it amplifies the sparse clicks that occur at low-examination slots onto a few items. **De-concentrating creator/item exposure is a *serving-policy* lever (the 90/10 reserve, future creator-health gate), not a *measurement* lever.** P1 fixes how we *read* exposure; it does not change *who gets* exposure. (Keeps P1 cleanly separate from the creator-health gate.)

## 7. Decision logic
- **Raw CTR is unsafe for the two-lane policy** — it systematically over-credits top warm positions and under-credits cold slots by ~11×, inverting the true slot ranking. Do not rank items/lanes or make ship/kill calls on naive CTR. ✅ documented.
- **IPS examination-correction is required and effective at the lane/position level** (recovers ~77–90% of true cold relevance) **with a known/estimated examination curve.** ✅
- **Variance caveat:** IPS reduces bias but inflates variance; per-item it currently *loses* on MSE at this impression density → use aggregate-level correction, or add SNIPS/clipping/shrinkage before per-item decisions. ✅
- **Required A/B log schema** (so propensities are recoverable online): `timestamp, user_id, user_context, policy_id, slate_position, item_id, lane(warm/cold), creator_id, examination_propensity(position), logging_propensity(item), reward(click/engagement)`. Examination must be estimated via randomization / PBM-EM in production (it is only known here because this is a simulator).

## 8. No-overclaim check
- Examination propensity is KNOWN (simulator PBM); flagged that production must estimate it. ✅
- Reward is an examined-gold-hit **proxy, not engagement/business lift**; no lift claim. ✅
- IPS variance cost reported (per-item MSE *worse* than naive) — honest negative, not hidden. ✅
- Concentration result reported even though it cuts against an intuitive "correction spreads credit" story. ✅

## 9. Artifacts
- `outputs/evidence/domain_position_bias_two_lane_report.json` (per-position naive/IPS/true, lane means, item corr/MSE, weight range, credit Gini, examination curve, A/B log schema)
- `outputs/plots/domain_position_bias_two_lane.png` (90/10: naive vs IPS vs true by position; warm-vs-cold lane bars)

## 10. Gate status & next step
- ✅ Position-bias correction layer built for the two-lane served policy; **naive CTR shown to invert the cold lane by 11×; IPS recovers it at the aggregate; variance tradeoff and A/B log schema documented.** Governance trio (OPE value · prevalence · position bias) now covers the measurement layer the 90/10 A/B needs.
- ⏸️ **Options next:** (a) **creator-health** simulation (does 90/10 actually de-concentrate creators over rounds? — the serving-side complement to §6); (b) SNIPS/clipping/PAL variance-reduction pass on the per-item estimator; (c) external GPU canonical SASRec; (d) package the full A/B-readiness spec (control/arms/guardrails/logging/estimators). No further step run.

**STOP — awaiting P1 review.**
