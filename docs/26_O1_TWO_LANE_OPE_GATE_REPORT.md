# 26 — O1 Gate Report: Two-Lane Policy OPE / Offline A/B Readiness

*Company question: **Before touching users, can we evaluate the two-lane policies' value and the estimators' behavior under a known-propensity logged setting?** This is offline A/B-readiness + policy-risk estimation — **not** a claim of real online lift. Domain-scoped. No position-bias / creator-health sim / SASRec / PRD / new model training.*

## 1. Setup (auditable)
- **Logging policy μ (per-slot, exact propensity, positivity on warm+cold):** ALS-heavy mixture — `(1-ε)·softmax(ALS rank, τ=5) over warm[:Mw] + ε·uniform over cold[:Mc]`, sampled independently per slot. Two μ variants tested.
- **Target policies π (deterministic 20-slates):** `als_only`, `q90_10` (=reserve-2), `q80_20`, `rrf`.
- **Reward (offline proxy):** `examined(PBM by rank) AND item == user's gold` (warm or cold). Simple, auditable; **not** engagement/lift.
- **Eval population:** 5,000 warm + 5,000 cold instances (50/50). 10k instances × 20 slots = **200k logged records**.
- **q̂ (for DM/DR):** empirical reward-rate table by (lane, rank) from the logs.

## 2. Ground-truth policy value (known reward model, 50/50 population)
| Policy | GT value |
|---|---|
| rrf | **0.00141** |
| q80_20 | 0.00063 |
| q90_10 | 0.00051 |
| als_only | 0.00037 |
**GT ranking: rrf > 80/20 > 90/10 > als_only.** Under this proxy + 50/50 population, more cold = higher value (als scores 0 on the 5k cold instances).

## 3. OPE estimates vs GT, and the logger effect
| Policy | GT | DR (ALS-heavy μ) | ESS | w_max | DR (cold-support μ) | ESS | w_max |
|---|---|---|---|---|---|---|---|
| als_only | 0.00037 | 0.00032 | 2,523 | 290 | 0.00046 | 2,017 | 372 |
| q90_10 | 0.00051 | 0.00032 | 2,565 | 267 | 0.00045 | 2,915 | 250 |
| q80_20 | 0.00063 | 0.00167 | 2,305 | 267 | 0.00075 | 3,933 | 167 |
| rrf | 0.00141 | 0.00442 | 1,361 | 267 | 0.00159 | 7,595 | 50 |

(IPS/SNIPS ≈ DR here; full numbers + CIs in the JSON. Plot: `outputs/plots/domain_two_lane_ope.png`.)

## 4. Findings (honest)
1. **The logging policy makes or breaks OPE.** ALS-heavy μ gives **weak cold support** → high variance: DR over-estimates rrf **3×** (0.0044 vs GT 0.0014), ESS as low as 1,361, w_max ~290. A **higher-cold-support μ** (ε=0.35, focused cold pool Mc=10) collapses the bias (DR rrf 0.0016 ≈ GT 0.0014) and lifts ESS to **7,595** with w_max **50**. *Positivity/coverage of the cold lane is the gating factor.*
2. **OPE reliably ranks the discovery-heavy policies above als_only** (rrf, 80/20 > als under both loggers — top-2 correct).
3. **OPE CANNOT separate 90/10 from als_only.** Their true values are nearly tied (0.00051 vs 0.00037) and DR returns them within noise (≈0.00032 / ≈0.00046) → the specific "90/10 beats als_only" claim is **not** offline-identifiable at this scale/reward.
4. **The 50/50 eval population over-weights discovery.** GT here rewards cold heavily; under the **realistic 23.5% cold prevalence (D2)**, warm value dominates and rrf's −44% warm-recall cost (C2) would sink it — consistent with C2's HOLD on rrf. So the "rrf is best" GT is partly an artifact of the 50/50 population; **prevalence-correct OPE is a required refinement.**

## 5. Decision
- **HOLD on the specific offline "90/10 > als_only" claim** — true values are indistinguishable at this scale/reward; only the live A/B can settle it. (C2's 90/10 ship-candidate stands as an *A/B candidate*, now with the caveat that OPE can't pre-confirm its edge.)
- **A/B-READY-CONDITIONAL for the discovery arms (80/20, rrf):** OPE distinguishes them as higher-value than als (under 50/50) **only with a logging policy that has adequate cold exploration** (the cold-support μ). With the ALS-heavy μ they are **HOLD — unreliable (positivity failure).**
- **Logging requirement for A/B:** deploy the experiment behind a logger with **structured cold exploration** (≈30–40% cold or focused cold pool), so propensities are known and ESS/variance are acceptable — otherwise OPE/interleaving will be untrustworthy.
- **Prevalence requirement:** evaluate/serve under realistic warm/cold prevalence; the 50/50 here inflates discovery value.

## 6. No-overclaim check
- Offline known-propensity simulation; reward is an examined-gold-hit **proxy, not business lift**; "A/B-ready" = ready to *test*, not proven. ✅
- 50/50 eval over-weights cold — flagged; prevalence-correct OPE recommended. ✅
- Estimator instability under the ALS-heavy logger reported, not hidden; diagnosis = cold positivity/coverage. ✅
- Deterministic targets vs stochastic logger → matched-support IPS; ESS/weights reported. ✅

## 7. Artifacts
- `outputs/evidence/domain_two_lane_ope_report.json` (both loggers × all policies × GT/IPS/SNIPS/DM/DR/bias/ESS/weights/CIs)
- `outputs/plots/domain_two_lane_ope.png` (DR-vs-GT by logger; ESS by logger)

## 8. Gate status & next step
- ✅ Two-lane OPE built; estimators validated against known truth; **logger positivity shown to be the deciding factor**; honest HOLD on the 90/10-vs-als offline claim; discovery arms A/B-ready only under an adequately-exploring logger; prevalence caveat surfaced.
- ⏸️ **Options next:** (a) **prevalence-correct OPE** (re-weight to 23.5% cold) to get the realistic value ranking; (b) **position-bias correction** layer on the served policy; (c) **creator-health** simulation; (d) external GPU canonical SASRec. No further step run.

**STOP — awaiting O1 review.**
