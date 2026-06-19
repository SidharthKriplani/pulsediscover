# 42 — G4: Formal A/B Rollout Doc

*Company-grade online test plan for the 90/10 reserve-2 policy. Grounded in C1/C2/O1/O1b/P1/H1/H2/G1/G2/G3. **No new code/experiments. No live A/B has run; 90/10 is not proven; cold-start and creator fairness are not solved. This is a rollout plan, not evidence of lift.** All sample sizes are labeled design placeholders.*

---

## 1. Company question
**How should a content platform safely test the 90/10 discovery policy online — without overclaiming offline evidence or harming user trust?**

## 2. Experiment objective
- **Not** to prove offline recall — offline recall has been evaluated as candidate evidence.
- **To test** whether the 2 protected discovery slots create **measurable, position-corrected cold/new-content value** while preserving warm relevance, session quality, and business guardrails.
- **90/10 is an A/B candidate, not a proven winner** — OPE cannot separate it from control (O1b), so the live test is the arbiter.

## 3. Hypotheses
- **Primary (H0/H1):** the 90/10 treatment creates positive **position-corrected** cold/new-content value relative to control's new-item exposure baseline, while preserving overall/warm engagement guardrails.
- **Secondary:** discovery-engaged sessions and new-creator reach increase; reading depth on discovered items is non-trivial.
- **Guardrail hypotheses:** warm engagement loss ≤10%; session abandonment flat; monetization not materially worse; latency/fallback within budget.
- **Catalog/creator-health hypothesis:** 90/10 increases creator/catalog **reach** (not necessarily de-concentration — H1/H2).
- **Measurement hypothesis:** naive CTR under-credits low-position cold slots (P1); position correction is required to read true cold value.
- **The discovery flywheel (G2 §5) remains a hypothesis to test, not proven lift.**

## 4. Experiment arms
| Arm | Purpose | Expected benefit | Risk | When to stop |
|---|---|---|---|---|
| **Control — ALS-only** | Baseline warm recommender | Reference relevance | none (status quo) | n/a |
| **T1 — 90/10 reserve-2** | Default candidate | Cold discovery within ≤10% warm guardrail | mild relevance cost (C2 −8.1% offline) | kill if warm guardrail breaks or cold value null/neg |
| **T2 — 80/20 / structured exploration bucket** *(optional — measurement/exploration arm, NOT the default product candidate)* | Aggressive discovery + OPE positivity support | Higher cold reach; enables logger-policy eval | higher warm loss (C2 −17%) | run only if powered + guardrailed; stop if warm loss unacceptable |
| **RRF — aggressive (mention only)** | Not a default arm | Max discovery value | −44% warm loss (C2) | exploration-only; not in default rollout |

RRF is named only as aggressive exploration; it is **not** in the default rollout due to warm-loss risk.

## 5. Target population & segmentation
- **Initial eligible:** logged-in users with enough history for a stable warm lane (so the −8.1% is measured fairly).
- **Excluded:** brand-new users with negligible history (relevance misses costlier early — A1); users in other conflicting experiments.
- **Priority segments:** heavy readers; genre explorers; stale-recommendation users; author/series-affinity users; new-content launch windows. **Risk segment:** intent-driven sessions (may not want a discovery slot).
- **Why targeted first:** offline evidence is candidate-only (B1) and OPE can't pre-confirm the edge (B5); starting on responsive segments limits downside, sharpens signal, and de-risks before any broad exposure.

## 6. Randomization & exposure unit
- **Default: user-level randomization** (stable hash) — keeps a user in one arm across sessions.
- **Why not session-level:** the two lanes and the feedback/learning effects span sessions; session-level assignment contaminates within-user comparisons and muddies retention readouts.
- **Exposure rules:** a user counts as exposed once they receive a slate from their assigned arm.
- **Primary readout is intent-to-treat (user-randomized):** all assigned users are analyzed in their assigned arm, regardless of exposure depth — this is the unbiased decision readout.
- **Minimum-exposure / per-protocol analysis is secondary/diagnostic only:** filtering to users above a minimum number of exposed slates (placeholder — set by power analysis) may sharpen the mechanism view but is **not** the primary decision metric, since exposure-conditioning can introduce selection bias.

## 7. Metrics framework
- **Proposed online north-star candidate:** retained reading sessions (returning, engaged sessions over a horizon) — *candidate, to be validated online.*
- **Primary experiment metric:** position-corrected cold/new-item engagement (the thing the policy is for), read alongside overall prevalence-weighted engagement.
- **Secondary:** discovery-engaged sessions; reading depth on discovered items; new-creator reach; distinct items/creators exposed; catalog coverage.
- **Guardrails:** warm engagement/relevance loss ≤10%; session abandonment flat; monetization guardrail (not materially worse); latency/fallback-rate within budget; creator Gini monitored.
- **Diagnostics:** OPE positivity/ESS (if a logging-policy/exploration arm is run); position-corrected vs naive cold value; creator Gini trajectory; cold-item age; treatment/control balance by segment.

## 8. Logging requirements (per served slot)
`user_key (privacy-safe)` · `experiment_id` · `arm_id` · `request_id / session_id` · `timestamp` · `item_id` · `creator_id` · `lane (warm/cold)` · `slate_position` · `score / candidate_source` · `logging_propensity` · `examination/position_propensity (if available)` · `exposure` · `reward (click/save/read/rating/downstream engagement)` · `monetization_event (if relevant)` · `fallback_flag` · `latency`.

**Missing propensities invalidate OPE and position-bias analysis (E1, B4) — propensity logging is a hard pre-launch contract.** Logs must respect retention windows, PII minimization, and aggregate-safe derived tables (G1/E3).

## 9. Position-bias & measurement plan
- **Raw CTR cannot decide the test** — it conflates examination with relevance.
- **Cold slots sit at lower positions and are under-credited** — P1 showed an ~11× under-credit on the cold lane.
- **Estimate examination** via randomized position tests or PBM-style modeling where feasible.
- **Read cold-lane value with position-corrected aggregate metrics** (lane/position level), not raw CTR.
- **Do not overtrust per-item IPS** — P1 showed per-item variance inflates MSE; trust aggregate corrections.

## 10. Rollout phases *(sample sizes are design placeholders, not commitments)*
| Phase | Entry criteria | Exposure (placeholder) | Monitors | Exit criteria |
|---|---|---|---|---|
| **0 — Offline readiness / logging QA** | logging schema live; propensities non-null; position logging verified | internal traffic | propensity null rate, schema completeness | clean logs + position propensities captured |
| **1 — Internal / small canary** | Phase 0 passed | tiny % / dogfood | latency, fallback rate, crashes, obvious relevance breaks | no serving regressions; guardrail instrumentation works |
| **2 — Targeted segment A/B** | canary clean | powered sample on priority segments (placeholder) | warm loss, abandonment, position-corrected cold value, monetization, ESS | guardrails hold + primary metric readable |
| **3 — Wider ramp** | Phase 2 guardrails hold | gradual % ramp | all guardrails + segment balance + prevalence mix (B6) | guardrails hold at each ramp step |
| **4 — Long-horizon readout** | sustained exposure | long-window cohort | retained sessions, creator Gini trajectory, catalog coverage, monetization | retention/health readout complete |

## 11. Ship / hold / kill rules
- **SHIP** if: warm guardrail holds (≤10% loss) **and** session quality intact **and** position-corrected cold value significantly positive **and** monetization not materially worse **and** logging/propensities valid.
- **HOLD / EXTEND** if: CIs overlap but discovery diagnostics trend positive (B5); underpowered segments; weak logger positivity; prevalence mix unstable (B6).
- **KILL** if: warm guardrail breaks; session quality/abandonment breaks; position-corrected cold value null/negative; monetization materially breaks; propensities missing (test invalid).
- **SAFE-TO-RAMP** if: guardrails hold on a targeted segment, position-corrected cold value positive, monetization neutral-or-positive, monitoring green.
- **Creator reach is interpreted separately from creator fairness** — reach gains never count as a fairness win (D1/D3).

## 12. Risk controls (G3 → rollout)
| Risk | Rollout control |
|---|---|
| **B4 position bias** | mandatory position-corrected readout; raw CTR barred from the decision |
| **B1 offline→online gap** | candidate-only framing; A/B is the arbiter; phased ramp |
| **B5 overlapping CIs** | pre-registered decision rule + power analysis; HOLD path |
| **B6 prevalence drift** | stratified, prevalence-weighted readout; segment-level rules |
| **F3 monetization** | monetization as an explicit guardrail; separate monitoring |
| **D1/D4 reach vs fairness / amplification** | reach≠fairness language; creator Gini trajectory monitor |
| **E1 missing propensities** | hard pre-launch logging contract; block test if null |
| **A1/A2 relevance/session** | warm + abandonment guardrails; targeted rollout first |

## 13. Decision readout template
- **What moved:** primary (position-corrected cold value), north-star candidate, key secondaries — with CIs.
- **What did not move:** null/flat metrics.
- **Guardrails:** warm loss, abandonment, monetization, latency/fallback — pass/fail.
- **Segments:** where it worked / didn't (explorers vs intent-driven; prevalence mix).
- **Uncertainty:** CI overlap, power, ESS/positivity, prevalence stability.
- **Decision:** ship / hold / kill / continue-research (with the triggering rule).
- **Next action:** ramp, extend, iterate cold lane, or pursue training-time fairness.

## 14. Claim boundaries
**No online result yet. 90/10 not proven. Cold-start not solved (candidate lane only). Creator fairness not solved (reach ≠ de-concentration). This document is a rollout plan, not evidence of lift.**

## 15. Interview-safe summary
- "It's a **phased rollout** — logging QA → canary → targeted segment A/B → ramp → long-horizon readout — not a flip-the-switch launch."
- "The decision metric is **position-corrected cold value**, never raw CTR — P1 showed raw CTR under-credits the cold lane ~11×."
- "I randomize at the **user level** because the lanes and learning span sessions; session-level would contaminate retention."
- "**Propensity logging is a hard pre-launch contract** — missing propensities invalidate the whole OPE/position analysis."
- "Guardrails are explicit and include a **separate monetization guardrail**; ship needs warm-loss ≤10%, session quality intact, and position-corrected cold value positive."
- "I plan for **prevalence drift** (B6) with stratified, prevalence-weighted readouts — O1b showed the live mix can change apparent value."
- "Everything is framed candidate-only: this is a **plan to test 90/10, not a claim that it wins**."

**STOP — awaiting G4 review.**
