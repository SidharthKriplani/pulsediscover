# 45 — G8: PulseDiscover Research Dossier

*The flagship written synthesis: problem → data → system → evidence → decision → limits → future work. Research-grade and company-realistic. **All evidence is offline (recall / known-propensity OPE / simulation). No online A/B, no business lift, no deployment. PulseDiscover is not declared BeastMax-complete in this document.***

---

## 1. Executive abstract
PulseDiscover is an offline recommender **experimentation program** built on the Goodreads fantasy/paranormal domain (~2.56M interactions). It addresses a structural failure of relevance-only ranking: warm collaborative models over-serve known/head content and score brand-new items at exactly zero, so discovery and new-creator supply decay. The solution is a **two-lane policy** — a tuned ALS f64 warm generator (R@20 0.085) plus a content-hybrid cold lane (cold R@20 ≈0.21–0.24 where new items have no ALS collaborative score) — served as **"90/10 reserve-2"**: 18 warm slots + 2 protected discovery slots, holding warm-relevance loss within a ≤10% guardrail. Around the policy sits a full governance layer: off-policy evaluation (with a logger-positivity diagnosis), prevalence correction, position-bias correction, and a creator/catalog-health simulation. The distinctive contribution is **measurement and honesty**, not a model: OPE cannot prove 90/10 beats control, naive CTR under-credits the cold lane ~11×, and creator-aware reranking provably can't de-concentrate within budget — all reported as findings. Conclusions were re-tested at ~7× scale (G6) and held, with warm loss drifting to the guardrail edge (−9.08%). The claim ceiling is strict: **90/10 is an A/B-ready, scale-stable, product-rational candidate — not a proven winner, not deployed, not a solution to cold-start or creator fairness.**

## 2. Problem framing
- **Warm recommenders over-serve known/head content.** Collaborative models rank what already has interaction signal, so popular items compound and the head dominates exposure.
- **New/cold items need protected exposure.** A never-seen item has no ALS factor / no collaborative score, so it cannot be recommended by the ALS lane; without reserved exposure it can never accumulate the interactions required to ever rank — a structural cold-start trap for items *and* creators.
- **The company problem is a balance, not a metric.** Real platforms must trade off short-term relevance, discovery of new content, and long-term catalog/creator health. "Maximize offline recall" optimizes one corner and silently degrades the others.
- **Offline recall alone is insufficient.** Engagement logs are confounded by position; a discovery slot at the bottom of the page looks dead even when it's the most relevant slot. Without off-policy and position-bias correction, a team can ship — or kill — the wrong policy on biased data.

## 3. Dataset & evaluation setup
- **Domain:** Goodreads fantasy/paranormal (UCSD Book Graph subset).
- **Scale:** ~2.56M core interactions; train 2.05M / val 256k / test 256k; ALS factors span 91,732 users × 40,541 items (64 dims).
- **Split:** global-time (train before cutoff t1, test after t2) — the realistic "predict the future" setup, not random holdout.
- **Warm vs cold:** warm = items with pre-cutoff signal (in the ALS item set); cold = items first appearing after the cutoff (no collaborative signal).
- **Prevalence (D2):** realistic traffic ≈ 76.5% warm / 23.5% cold.
- **Sampled vs scale (G6):** most gates used 5,000-user samples (and 2,400 for health); G6 re-ran at the largest valid scale — **34,539 warm users (~7×)** and the **entire 5,929-user cold-gold population**.
- **Offline vs online:** everything here is offline (recall, known-propensity OPE, simulation). There is **no online A/B and no business-lift evidence.**

## 4. System design
- **Warm lane — ALS f64.** Implicit matrix factorization, tuned f16→f64 (converged by iter 4); the best single generator built (R@20 0.085), beating popularity (~0.035), co-occurrence (~0.05), and SASRec.
- **Cold lane — content-hybrid.** RRF of same-author + same-series + TF-IDF over title/shelves/description, restricted to post-cutoff items; provides cold recall where ALS is 0.
- **Policy — 90/10 reserve-2.** 18 warm + 2 protected cold slots; aggressive variants (80/20, RRF) exist as exploration arms only.
- **Serving — two-stage.** Heavy work offline (factor training, content indexing, candidate-list precompute); request-time = candidate fetch + a small merge/dedup/guardrail step. Serving latency is less risky than freshness, logging completeness, and training cadence under this design (latency design estimate ~120 ms p95; G1).
- **Logging & propensity.** Per-slot schema (position, logging + examination propensities, lane, creator, reward); propensity logging is a hard pre-launch contract — missing it invalidates OPE/position-bias analysis.
- **Position-bias correction.** Examination estimated offline (randomized position data or PBM-style modeling); decisions read from IPS-corrected aggregate metrics, never raw CTR.
- **Governance layer.** OPE + positivity diagnosis, prevalence correction, position-bias correction, creator/catalog-health monitoring, and a claim-boundary discipline.

## 5. Evidence ladder
**D1/D2 — domain spine.** *Q:* is there a real, time-honest domain to experiment on? *Method:* stream-extract Goodreads core, global-time split, compute prevalence. *Result:* 2.56M interactions, 76.5/23.5 warm/cold. *Interpretation:* a realistic substrate. *Boundary:* single domain, not multi-vertical.

**D3A/D3A+ — warm baselines.** *Q:* what's the best warm generator? *Method:* tune popularity/co-occurrence/ALS (f16→f64), Recall@K/NDCG with bootstrap CIs, segments. *Result:* ALS f64 R@20 0.085, best single; hybrid wins only the head/NDCG. *Interpretation:* ALS f64 is the warm floor. *Boundary:* f128/f64-hybrid deferred to external.

**D3B — SASRec honest negative.** *Q:* can a sequence model beat the floor? *Method:* SASRec on the domain under CPU/last-position training. *Result:* R@20 0.036, below every collaborative baseline. *Interpretation:* under-trained/CPU-limited — a documented negative, fully diagnosed. *Boundary:* no SASRec competitiveness claimed; canonical GPU run deferred.

**C1 — cold-start content lane.** *Q:* can we recommend brand-new items? *Method:* content-hybrid (author/series/TF-IDF via RRF), leave-last-out on new-in-test items. *Result:* cold R@20 0.241; **top-100 candidate lists surfaced all 1,420 cold items across users — catalog candidate coverage, not user-level recall**; new items have no ALS factor / no collaborative score, so they cannot be recommended by the ALS lane. *Interpretation:* a real cold candidate channel. *Boundary:* offline channel, not proven discovery lift.

**C2 — two-lane policy.** *Q:* how many slots to reserve for cold? *Method:* slate tradeoff sweep with a warm-loss guardrail. *Result:* 90/10 warm −8.1% (PASS), cold 0.050, 735 new books; 80/20 −17%, RRF −44%. *Interpretation:* 90/10 is the only discovery policy within the guardrail. *Boundary:* no proof it beats control on engagement.

**O1/O1b — OPE + prevalence.** *Q:* what's each policy worth offline, reliably? *Method:* known-propensity OPE (IPS/SNIPS/DM/DR); prevalence re-weighting to 76.5/23.5. *Result:* logger positivity decisive (ESS 1.4k→7.6k); **OPE cannot separate 90/10 from control** (`dr_separates=False`); prevalence compresses the field. *Interpretation:* OPE de-risks but cannot certify the micro-edge. *Boundary:* offline proxy reward, not lift.

**P1 — position-bias correction.** *Q:* can we trust logged clicks? *Method:* PBM examination + IPS correction on the served slate. *Result:* **naive CTR under-credits the cold lane ~11×**; IPS recovers ~77% at aggregate; per-item IPS variance-heavy. *Interpretation:* raw CTR would wrongly kill the cold lane. *Boundary:* examination known only in sim; trust aggregate not per-item.

**H1/H2 — creator/catalog health.** *Q:* does 90/10 improve creator health, and can reranking de-concentrate? *Method:* feedback-loop exposure simulation; creator-cap/MMR reranking. *Result:* 90/10 widens **reach** (+26% creators, +444 new **in the H1 offline simulation**) and holds a protected cold share, but **does not de-concentrate** (Gini flat-to-worse); reranking de-concentrates the warm head only at −18% to −36% recall — all guardrail-breaking. *Interpretation:* reach ≠ de-concentration; cheap reranking can't fix concentration. *Boundary:* creator fairness not solved.

**G1 — infra/cost/latency.** *Q:* can it be served? *Method:* architecture + cost/latency analysis. *Result:* conventional two-stage design; heavy work offline, request-time cheap; design estimate ~120 ms p95. *Interpretation:* production-plausible. *Boundary:* not deployed, not benchmarked.

**G2 — product/growth.** *Q:* why should a platform care? *Method:* stakeholder + discovery-flywheel + metrics strategy. *Result:* a conservative discovery bet with a north-star (retained sessions, *online candidate*) and monetization caution. *Interpretation:* a real product decision. *Boundary:* flywheel is a hypothesis, no lift.

**G3 — risk register.** *Q:* what could go wrong? *Method:* risk table tied to evidence, with detection/mitigation/ship-kill. *Result:* top risks are measurement (position bias, offline→online), monetization, ambiguous CIs, reach-vs-fairness/amplification. *Interpretation:* governable. *Boundary:* risks, not incidents.

**G4 — A/B rollout.** *Q:* how to test it safely? *Method:* phased rollout (logging QA → canary → targeted A/B → ramp → long-horizon), ITT readout, ship/hold/kill. *Result:* a runnable plan with position-corrected decision metric and propensity-logging contract. *Interpretation:* test-ready. *Boundary:* a plan, not a result.

**G5 — model-depth deferral.** *Q:* do we need a competitive advanced model? *Method:* option analysis. *Result:* documented deferral (no GPU here); ALS floor + diagnosed SASRec negative. *Interpretation:* depth is bounded and honest, **not closed at T3 level**. *Boundary:* documented deferral ≠ an advanced-model win.

**G6 — scale-stability.** *Q:* do the sampled conclusions survive scale? *Method:* re-run at 34,539 warm (largest valid set) + 5,929 cold (full cold pop). *Result:* **PASS** — ALS R@20 0.052→0.0507, 90/10 warm 0.048→0.0461, warm loss −8.1%→**−9.08% (holds, near edge)**, policy cold recall 0.050→0.065; warm-lane Gini als≈q90 (0.908 vs 0.907). *Interpretation:* conclusions are not a sampling artifact. *Boundary:* scale robustness, not online causality.

## 6. Main findings (synthesis)
1. **ALS f64 owns warm relevance** — the reliable backbone; advanced models did not beat it here.
2. **Content-hybrid owns cold/new discovery** — it converts a structural zero into ~0.21–0.24 recall; the two lanes are complementary, not competing.
3. **90/10 is product-rational but not proven superior** — it's the only guardrail-passing discovery policy, yet OPE cannot separate it from control. Honesty over hype.
4. **Measurement is the real contribution** — logger positivity gates OPE reliability, prevalence correction compresses apparent value, and position correction prevents an 11× misread that would kill the cold lane. The governance layer is what makes the policy *decidable*.
5. **Creator reach improves; de-concentration does not** — 90/10 widens the catalog and protects a cold share, but neither it nor post-hoc reranking lowers head concentration within budget. Fairness is unsolved and labeled so.
6. **Scale robustness passed (G6)** — conclusions held at ~7×, with the warm guardrail drifting to the edge (−9.08%), making warm relevance the binding online constraint.
7. **Advanced-model depth remains deferred, not solved** — a deliberate, documented boundary, not a hidden gap.

## 7. Company decision logic
- **Why 90/10 is an A/B candidate:** it adds cold discovery and catalog reach within the ≤10% warm guardrail, is conservative (2 of 20 slots, reversible), and OPE confirms discovery arms carry value even if it can't certify 90/10's specific edge.
- **Who benefits:** readers (fresh discovery), new/underexposed creators (reach), catalog ops (freshness), the platform (hypothesized discovery flywheel → retention).
- **Risks:** perceived-relevance dip, offline→online gap, ambiguous CIs, monetization, amplification, reach-mistaken-for-fairness (G3).
- **Rollout:** phased and targeted-first (explorers, stale-rec users, launch windows), user-level randomization, ITT primary readout, position-corrected cold value as decider (G4).
- **Ship/hold/kill:** **ship** if warm guardrail holds, session quality intact, position-corrected cold value significantly positive, monetization not worse, propensities valid; **hold** if CIs overlap but diagnostics trend up; **kill** if warm/session/monetization breaks or corrected cold value is null/negative. Given G6's −9.08%, **warm relevance is a primary kill/ramp guardrail.**

## 8. Limitations
Offline-only; no online A/B; no business-lift evidence; no deployment; no production latency benchmark (estimates only); no competitive GPU sequence model (G5 deferred); creator fairness not solved (reach ≠ de-concentration); offline reward is an examined-gold-hit **proxy**, not engagement; single domain (fantasy/paranormal) — cross-domain generality untested; warm eval excludes ~2,215 users lacking ALS factors (G6).

## 9. Future work (prioritized)
1. **Live A/B** of 90/10 (the only thing that can prove the edge) — highest priority.
2. **Canonical GPU SASRec** (or a two-tower) to close the model-depth deferral honestly.
3. **Better cold-lane quality modeling** — precision-gate the discovery slots; reduce same-author over-reliance.
4. **Training-time creator/catalog fairness** — the real de-concentration lever H2 showed reranking can't provide.
5. **Long-horizon retention / counterfactual evaluation** — beyond next-item recall.
6. **Production logging/monitoring implementation** — turn the schema + guardrails into live instrumentation.

## 10. Claim-boundary table
| Supported claim | Evidence | Unsupported overclaim | Safe interview wording |
|---|---|---|---|
| ALS f64 best warm generator (R@20 0.085) | D3A+/G6 | "SOTA recommender" | "My tuned warm floor; best single generator I built." |
| Content-hybrid gives cold recall where ALS is 0 (~0.21–0.24) | C1/G6 | "cold-start solved" | "A working cold candidate channel, offline." |
| 90/10 within ≤10% warm guardrail (−8.1%→−9.08%) | C2/G6 | "90/10 proven best" | "Product-rational A/B candidate; OPE can't certify the edge." |
| OPE reliability hinges on logger positivity | O1 | "OPE proved lift" | "OPE de-risks; positivity is the deciding factor." |
| Naive CTR under-credits cold ~11× | P1 | "we fixed engagement measurement" | "Position correction is required; raw CTR is unsafe." |
| Reach up, de-concentration not | H1/H2 | "creator fairness solved" | "Reach improved, not de-concentration." |
| Conclusions scale-stable at ~7× | G6 | "validated in production" | "Scale-robust offline, not online-proven." |
| Advanced-model depth deferred | G5 | "benchmarked a winning deep model" | "Documented deferral; not an advanced-model result." |

## 11. Interview-defense summary
**60-second:** "PulseDiscover treats recommendation as a company tradeoff — relevance vs discovery vs catalog health — on 2.5M Goodreads interactions. A tuned ALS warm model plus a content-based cold lane feed a two-lane policy that reserves 2 of 20 slots for new content, within a 10% relevance guardrail. The real work is the governance layer: off-policy evaluation, prevalence and position-bias correction, and catalog-health analysis — the tools that decide whether it's safe to A/B test. I'm strict on claims: it's an A/B-ready candidate, scale-tested at ~7×, with honest negatives on sequence modeling and creator fairness. Nothing is deployed or proven as lift."

**Hard questions:**
1. *"Is 90/10 better than control?"* — "Not proven. OPE can't separate them; the A/B settles it."
2. *"Did you solve cold-start?"* — "I built a working cold channel offline; not a proven business win."
3. *"Why is SASRec worse?"* — "CPU/last-position as I built it; diagnosed, GPU run deferred."
4. *"Did reach fix creator fairness?"* — "No — reach improved, concentration didn't; reranking can't fix it in budget."
5. *"Does scale change the story?"* — "It held at ~7×; warm loss crept to −9.08%, so warm relevance is the primary online guardrail."

## 12. Conclusion
**PulseDiscover is an offline, evidence-backed, scale-stable, A/B-ready recommender experimentation program — not a deployed or business-proven recommender system.** Its strength is the rigor and honesty of its measurement and governance layer; its boundaries (offline-only, deferred advanced-model depth, unsolved creator fairness) are documented, not hidden. It is ready for capstone packaging — and explicitly **not** declared BeastMax-complete here (the final PDF/deck and golden-template extraction remain).

**STOP — awaiting G8 review.**
