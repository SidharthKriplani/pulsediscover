# 46 — G9: Final Deck Storyboard

*Presentation storyboard for the PulseDiscover flagship deck (15 slides + executive page + scripts + appendix). **A storyboard/spec — not a generated PDF (deferred). All evidence offline; no online lift, no deployment, 90/10 not proven, cold-start/fairness not solved.***

---

## Slide deck (15)

### S1 — Title / positioning
- **Objective:** frame the project in one line.
- **Key message:** "PulseDiscover — an offline, A/B-ready recommender experimentation system balancing warm relevance, cold-start discovery, and catalog health."
- **Visual:** title + the two-lane glyph (warm lane + 2 cold slots).
- **Bullets:** domain (Goodreads, 2.5M); the tradeoff framing; "experimentation system, not a model."
- **Evidence:** README/G8. **Boundary:** offline, candidate-level.

### S2 — Business problem
- **Objective:** why a platform should care.
- **Key message:** relevance-only ranking over-serves the head and buries new content; engagement data is position-confounded.
- **Visual:** head-vs-tail exposure curve; "new item = no signal."
- **Bullets:** head compounding; new items have no collaborative score; catalog freshness decays; CTR misleads.
- **Evidence:** G2/G8 §2. **Boundary:** problem framing, not a measured loss.

### S3 — Dataset & evaluation setup
- **Objective:** establish rigor.
- **Key message:** real domain, time-honest split, realistic prevalence.
- **Visual:** split timeline (t1/t2) + 76.5/23.5 prevalence donut.
- **Bullets:** 2.56M interactions; global-time split; warm/cold defs; sampled + G6 scale (34.5k warm / 5.9k cold); offline vs online line.
- **Evidence:** D1/D2, G6. **Boundary:** offline only.

### S4 — Architecture
- **Objective:** show it's a system.
- **Key message:** two lanes → policy → evaluation → governance → A/B-readiness, with governance as a gate.
- **Visual:** the ARCH1 Mermaid diagram.
- **Bullets:** warm/cold retrieval; 90/10 assembly; offline-heavy serving; governance gate.
- **Evidence:** ARCH1/G1. **Boundary:** plausible, not deployed.

### S5 — Warm model evidence
- **Objective:** the warm floor.
- **Key message:** ALS f64 is the tuned warm floor (R@20 0.085); advanced models didn't beat it.
- **Visual:** generator ladder bar (pop → co-occ → ALS f16 → f64; SASRec marked negative).
- **Bullets:** f16→f64 tuning; beats popularity/co-occ; SASRec honest negative.
- **Evidence:** D3A+/D3B. **Boundary:** not SOTA; GPU SASRec deferred.

### S6 — Cold lane evidence
- **Objective:** discovery is possible.
- **Key message:** content-hybrid gives cold R@20 0.241 where the ALS lane has no score for new items.
- **Visual:** cold recall bars (ALS 0 vs hybrid 0.24); coverage note.
- **Bullets:** author/series/TF-IDF via RRF; top-100 lists surfaced all 1,420 cold items (catalog candidate coverage, not user recall).
- **Evidence:** C1/G6. **Boundary:** offline channel, not discovery lift.

### S7 — 90/10 policy tradeoff
- **Objective:** the product decision.
- **Key message:** 90/10 is the only discovery policy within the ≤10% warm guardrail.
- **Visual:** tradeoff scatter (warm loss vs cold reach; guardrail line).
- **Bullets:** 90/10 −8.1% (PASS); 80/20 −17%; RRF −44% (exploration only).
- **Evidence:** C2. **Boundary:** not proven > control.

### S8 — OPE / prevalence correction
- **Objective:** measurement rigor.
- **Key message:** logger positivity gates OPE; prevalence compresses value; OPE can't separate 90/10 from control.
- **Visual:** ESS 1.4k→7.6k bars; 50/50-vs-prevalence value bars.
- **Bullets:** IPS/SNIPS/DM/DR vs known truth; `dr_separates=False`; positivity decisive.
- **Evidence:** O1/O1b. **Boundary:** offline proxy reward.

### S9 — Position-bias lesson
- **Objective:** the headline measurement insight.
- **Key message:** naive CTR under-credits the cold lane ~11×; correction recovers it.
- **Visual:** naive vs IPS vs true by position (P1 plot).
- **Bullets:** PBM examination; cold slots at bottom; raw CTR would kill the right lane.
- **Evidence:** P1. **Boundary:** examination known in sim; per-item IPS variance-heavy.

### S10 — Creator / catalog health
- **Objective:** honest fairness story.
- **Key message:** 90/10 buys reach, not de-concentration.
- **Visual:** reach up (+creators) vs Gini flat; reranking tradeoff curve.
- **Bullets:** +26% creators, +444 new (H1 offline sim); Gini flat-to-worse; reranking costs −18–36%.
- **Evidence:** H1/H2. **Boundary:** fairness not solved.

### S11 — G6 scale stability
- **Objective:** robustness.
- **Key message:** conclusions held at ~7×; warm loss drifted to the guardrail edge.
- **Visual:** sampled-vs-G6 drift table.
- **Bullets:** ALS 0.052→0.0507; 90/10 0.048→0.0461; warm loss −8.1%→−9.08% (holds); cold stable/favorable.
- **Evidence:** G6. **Boundary:** scale robustness, not online causality.

### S12 — Company realism: infra / product / risk / rollout
- **Objective:** it's a real decision, not just a model.
- **Key message:** serving plausible; product thesis clear; risks named; rollout planned.
- **Visual:** 2×2 — infra (latency/cost), product (flywheel/metrics), risk (top-5), rollout (phases).
- **Bullets:** ~120ms p95 estimate; discovery flywheel hypothesis; top risks measurement+monetization; phased targeted rollout, warm as primary guardrail.
- **Evidence:** G1/G2/G3/G4. **Boundary:** plans/estimates, not live results.

### S13 — Claim boundaries
- **Objective:** demonstrate discipline.
- **Key message:** every positive claim is candidate-level; negatives documented.
- **Visual:** the CLAIM1 supported/not-supported table (condensed).
- **Bullets:** no lift; no deployment; 90/10 not proven; cold-start/fairness not solved; SASRec deferred.
- **Evidence:** CLAIM1/G8. **Boundary:** this slide *is* the boundary.

### S14 — Final decision
- **Objective:** the verdict.
- **Key message:** 90/10 is an A/B-ready, scale-stable, product-rational candidate — control = ALS-only, exploration = 80/20/RRF.
- **Visual:** ship/hold/kill decision card.
- **Bullets:** ship if warm guardrail holds + position-corrected cold value positive + monetization neutral; warm = primary kill/ramp guardrail.
- **Evidence:** AB1/G4/G6. **Boundary:** candidate, not winner.

### S15 — Appendix map
- **Objective:** point to depth.
- **Key message:** full evidence ladder + dossier behind the deck.
- **Visual:** doc index (docs/18–45) grouped by layer.
- **Bullets:** evidence gates; governance; company realism; dossier.
- **Evidence:** PACK1/G8. **Boundary:** n/a.

---

## One-page executive version
PulseDiscover is an offline, A/B-ready recommender experimentation program on 2.5M Goodreads interactions. A tuned ALS f64 warm model (R@20 0.085) plus a content-hybrid cold lane (cold R@20 0.24, where new items have no collaborative score) feed a **90/10 reserve-2** policy that adds discovery within a ≤10% warm guardrail (−8.1%, −9.08% at full scale). A governance layer — OPE (positivity decisive; can't separate 90/10 from control), prevalence correction, position-bias correction (naive CTR under-credits cold ~11×), and catalog-health analysis (reach up, de-concentration not) — makes the policy *decidable*. Conclusions held at ~7× scale. Honest negatives: CPU-limited SASRec below the MF floor; creator fairness unsolved; advanced-model depth deferred. **It is a product-rational A/B candidate — not proven, not deployed, no business lift.**

## Presentation scripts
**90-second:** S1 → S2 → S6 → S7 → S9 → S14. "Recommenders over-serve the head and can't see new items. I built a two-lane policy that reserves 2 of 20 slots for new content within a 10% relevance guardrail, plus the measurement layer that makes it safe to test — including catching that raw CTR under-credits the cold lane 11×. It's an A/B-ready candidate, not a proven winner."

**3-minute:** add S3 (rigor), S5 (warm floor), S8 (OPE can't separate 90/10 from control), S10 (reach not fairness), S11 (held at scale). Emphasize honesty and governance as the differentiator.

**5-minute:** full S1–S15. Spend extra time on S8–S9 (measurement) and S12 (company realism: infra/product/risk/rollout), and close on S13–S14 (claim discipline + decision).

## Appendix plan
- A1 evidence ladder (G8 §5) · A2 OPE/positivity detail (O1/O1b) · A3 position-bias method (P1) · A4 health sim + reranking (H1/H2) · A5 infra/cost (G1) · A6 risk register (G3) · A7 A/B rollout (G4) · A8 G6 scale tables · A9 claim-boundary table (CLAIM1) · A10 model-depth deferral (G5).

## Final claim checklist (must hold on every slide)
- [ ] No online lift claimed. [ ] No deployment claimed. [ ] 90/10 = candidate, not proven. [ ] Cold-start = channel, not solved. [ ] Creator fairness = reach, not solved. [ ] SASRec = honest negative, GPU deferred. [ ] G6 = scale robustness, not online causality. [ ] Latency/cost = design estimates. [ ] Creator gains labeled "H1 offline simulation."

**STOP — G9 storyboard complete (PDF generation deferred).**
