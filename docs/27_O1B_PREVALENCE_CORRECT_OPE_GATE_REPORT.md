# 27 — O1b Gate Report: Prevalence-Correct Two-Lane OPE

*Company question: **Under realistic traffic (D2: ~76.5% warm / 23.5% cold), which two-lane policy is actually the best offline A/B candidate?** Re-weights the O1 OPE to real prevalence, on the cold-support logger (O1 showed the ALS-heavy logger has weak cold positivity). Offline policy-risk estimation — **not** business lift. No position-bias / creator-health sim / SASRec / PRD / new training.*

*(Cleanup applied first: `domain_two_lane_ope_report.json` now keeps only the two-logger source of truth + meta; the earlier preliminary top-level keys were removed.)*

## 1. Setup
- Cold-support logger μ (ε=0.35, cold pool Mc=10, warm Mw=20) — the reliable logger from O1 (ESS up to 7.6k, w_max ~50).
- Targets: als_only, q90_10 (=reserve-2), q80_20, rrf. Reward = examined-gold-hit proxy. 5,000 warm + 5,000 cold instances, **re-weighted to warm 0.765 / cold 0.235**. Stratified bootstrap (resample within each stratum).

## 2. Policy value: 50/50 vs realistic prevalence
| Policy | GT (50/50) | **GT (prevalence)** | DR (prev) | DR 95% CI | ESS | w_max |
|---|---|---|---|---|---|---|
| als_only | 0.00037 | 0.00057 | 0.00071 | [0.00019, 0.00132] | 2,017 | 372 |
| q90_10 | 0.00051 | **0.00062** | 0.00070 | [0.00018, 0.00125] | 2,915 | 250 |
| q80_20 | 0.00063 | 0.00067 | 0.00084 | [0.00037, 0.00158] | 3,933 | 167 |
| rrf | 0.00141 | 0.00091 | 0.00106 | [0.00060, 0.00155] | 7,595 | 50 |

Plot: `outputs/plots/domain_two_lane_prevalence.png`.

## 3. What prevalence correction changes
- **The discovery advantage shrinks and the field compresses.** Re-weighting from 50/50 to 76.5/23.5: **als rises** 0.00037→0.00057 (warm up-weighted), **rrf falls** 0.00141→0.00091 (cold down-weighted); q90 0.00051→0.00062, q80 0.00063→0.00067. The als↔q90↔q80 spread is now tiny (0.00057 / 0.00062 / 0.00067).
- **rrf still has the highest GT even prevalence-corrected** (0.00091) under this proxy reward — its large cold recall, even at 23.5% weight, plus its warm hits, outweigh. **But that ignores the warm guardrail** (see §4), which is exactly why value alone isn't the ship rule.

## 4. Guardrails from C2 (the binding constraint)
| Policy | warm R@20 | warm rel-loss vs als | cold R@20 | distinct cold (top-20) | creator Gini |
|---|---|---|---|---|---|
| als_only | 0.052 | 0% | 0.000 | 0 | 0.861 |
| **q90_10** | 0.048 | **−8.1%** (PASS ≤10%) | 0.050 | 735 | 0.880 |
| q80_20 | 0.043 | −17% (FAIL) | 0.092 | 1,020 | 0.875 |
| rrf | 0.029 | −44% (FAIL) | 0.171 | 1,299 | 0.842 |

## 5. Key honest finding — OPE cannot separate 90/10 from ALS-only
DR(q90_10)=0.00070 ≈ DR(als_only)=0.00071, with **fully overlapping 95% CIs** ([0.00018,0.00125] vs [0.00019,0.00132]). `dr_separates_90_10_from_als = False`. Their true prevalence values are also close (0.00062 vs 0.00057). **90/10's edge over ALS-only is NOT offline-provable at this scale/reward.**

## 6. Decision
- **Primary A/B candidate: 90/10 (reserve-2).** Among the **guardrail-passing** policies (only als and q90 keep warm loss ≤10%), q90 has the higher prevalence GT (0.00062 > 0.00057) and adds real cold discovery (cold R@20 0.050, 735 distinct new books) at −8.1% warm. **But OPE cannot offline-prove it beats ALS-only — it is a *product-rational* A/B candidate, not an offline-proven winner.**
- **Aggressive A/B arms (not default): 80/20 and rrf.** They have higher proxy value (rrf is the value-maximizer even prevalence-corrected) but **violate the warm guardrail** (−17%, −44%). Run them only as explicit discovery-heavy arms with warm-retention guardrails monitored.
- **ALS-only = control.**
- **OPE's role here is honest risk-reduction, not a verdict:** it confirms discovery arms carry real value and quantifies variance/positivity needs, but it cannot certify the 90/10 micro-edge — that's what the live A/B is for.

## 7. No-overclaim check
- Prevalence-weighted to D2's 76.5/23.5; offline proxy reward; **no business-lift claim**. ✅
- 90/10-vs-als indistinguishability stated plainly; 90/10 framed as product-rational candidate, not proven winner. ✅
- rrf "highest value" qualified by the warm-guardrail violation (value ≠ ship). ✅
- Cold-support logger used (ALS-heavy was unreliable per O1); stratified bootstrap. ✅

## 8. Artifacts
- `outputs/evidence/domain_two_lane_ope_prevalence_report.json` (50/50 + prevalence values, DR/IPS/SNIPS/DM, bias, CIs, ESS/weights, C2 guardrails)
- `outputs/plots/domain_two_lane_prevalence.png` (50/50 vs prevalence value)
- (cleaned) `outputs/evidence/domain_two_lane_ope_report.json`

## 9. Gate status & next step
- ✅ Prevalence-correct OPE done; **90/10 = primary A/B candidate (guardrail-passing, product-rational, not offline-proven)**; 80/20 & rrf = aggressive arms; OPE honestly cannot separate 90/10 from als.
- ⏸️ **Options next:** (a) **position-bias correction** on the served policy (PBM/IPS — completes the governance trio); (b) **creator-health** simulation (does 90/10 actually de-concentrate creators over rounds?); (c) external GPU canonical SASRec; (d) package the A/B-readiness spec. No further step run.

**STOP — awaiting O1b review.**
