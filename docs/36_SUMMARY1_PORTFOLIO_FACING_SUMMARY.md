# 36 — SUMMARY1: Portfolio-Facing Summary

*Shareable summaries of PulseDiscover for different audiences. No new claims — all figures trace to prior gates. Offline candidate evidence only; no business-lift or deployment claim.*

---

## Executive summary (5 lines)
PulseDiscover is an offline recommender **experimentation system** built on 2.56M Goodreads interactions. It pairs a tuned warm collaborative model (ALS f64) with a content-hybrid cold-start lane, served through a two-lane "90/10 reserve-2" policy that adds discovery within a ≤10% relevance guardrail. Around it sits a full governance layer — off-policy evaluation, prevalence correction, position-bias correction, and catalog-health analysis — that decides whether the policy is safe to A/B test. The headline is honesty: every positive result is candidate-level, and the negatives (SASRec, creator de-concentration) are documented, not hidden. The deliverable is an A/B-ready policy with guardrails and a logging design, not a "trained model."

## Recruiter-facing (≈150 words)
PulseDiscover is a portfolio project that treats recommendation as a **company decision problem**, not a model-training exercise. Using a real 2.56M-interaction Goodreads dataset, I built a warm recommender (tuned matrix factorization), a separate content-based lane for brand-new items the warm model structurally can't rank, and a serving policy that reserves slots for discovery without hurting relevance beyond a fixed guardrail.

The distinctive part is the measurement layer: off-policy evaluation, realistic-traffic prevalence correction, and position-bias correction — the tools that stop a team from shipping on misleading click data. I was deliberately rigorous about claims: the discovery policy is an A/B *candidate*, not a proven winner, and I documented where a sequence model and a fairness intervention failed rather than overselling them. The end product is an A/B-readiness spec — arms, guardrails, logging schema, and ship/hold/kill criteria — plus a full interview defense pack.

## Technical reviewer (≈230 words)
PulseDiscover is an end-to-end RecSys experimentation program on the Goodreads fantasy/paranormal core (2.56M interactions, global-time split, 76.5/23.5 warm/cold prevalence).

**Retrieval.** ALS tuned f16→f64 is the warm floor (R@20 0.085), beating popularity (~0.035), co-occurrence (~0.05), and a CPU/last-position SASRec (0.036 — reported as an honest negative with full diagnosis). New items get 0 from ALS by construction, so a content-hybrid lane (same-author + same-series + TF-IDF via RRF) provides cold R@20 0.241 at 100% cold-pool coverage.

**Policy.** A two-lane "90/10 reserve-2" slate (18 warm + 2 cold) keeps warm-recall loss at −8.1% (≤10% guardrail) while adding 735 new books; 80/20 and RRF are higher-discovery but guardrail-breaking exploration arms.

**Governance.** Known-propensity OPE (IPS/SNIPS/DM/DR) shows logger positivity is decisive (ESS 1.4k→7.6k) and **cannot separate 90/10 from control**; prevalence correction confirms it. A PBM/IPS position-bias layer shows naive CTR under-credits the cold lane 11×. A feedback-loop health sim shows 90/10 improves creator *reach* (+26%, +444 new) but not de-concentration, and creator-aware reranking can't fix that within the relevance budget.

**Output.** An A/B-readiness spec (arms, metrics, guardrails, P1 logging schema, ship/hold/kill), an evidence ledger, a claim-boundary table, and an interview defense manual. Everything is offline candidate evidence — no executed A/B, no lift claim.

## LinkedIn / portfolio card
> **PulseDiscover — Two-Lane Recommender Experimentation System**
> Built a warm (ALS f64) + cold (content-hybrid) recommender on 2.5M Goodreads interactions, served via a "90/10 reserve-2" discovery policy held within a ≤10% relevance guardrail. Added the governance layer that actually decides whether to ship: off-policy evaluation, prevalence correction, and position-bias correction (raw CTR under-credited new content 11×). Treated it as an A/B-ready company decision — with documented honest negatives — not just a trained model.
> *Offline experimentation program; A/B-ready candidate, not proven lift.*

## Resume bullet options
- Built an end-to-end recommender **experimentation system** on 2.56M Goodreads interactions, pairing a tuned ALS f64 warm model (R@20 0.085) with a content-hybrid cold-start lane (cold R@20 0.241 where collaborative filtering is structurally 0).
- Designed a two-lane "90/10 reserve-2" serving policy that adds new-content discovery while keeping warm-relevance loss within a ≤10% guardrail (−8.1%).
- Implemented off-policy evaluation (IPS/SNIPS/DM/DR) and showed **logger positivity is the deciding factor** for reliability (effective sample 1.4k→7.6k).
- Built a position-bias correction layer revealing that **naive CTR under-credited new-content slots ~11×**, preventing a wrong "kill the discovery lane" decision.
- Ran a feedback-loop catalog-health simulation showing the policy improves creator **reach** (+26% creators, +444 new) but not head de-concentration — and documented why creator-aware reranking fails within the relevance budget.
- Authored a complete **A/B-readiness spec** (arms, guardrails, logging schema, ship/hold/kill) plus an evidence ledger and claim-boundary table.
- Reported **honest negatives** (CPU-limited SASRec below the MF baseline; unsolved creator fairness) rather than inflating results.

## Claim boundary line
*All results are offline (recall / known-propensity OPE / simulation): 90/10 is an A/B-ready candidate, not a proven winner; no business-lift, deployment, cold-start-win, SASRec-win, or fairness-solved claim is made.*
