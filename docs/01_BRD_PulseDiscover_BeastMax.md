# 01 — Business Requirements: PulseDiscover BeastMax

*Product framing for the BeastMax-core build. This is the "why" the build serves; the technical "how" is in `02_TECHNICAL_ARCHITECTURE.md` and `plans/01_BEASTMAX_BUILD_PLAN.md`.*

## Product framing
PulseDiscover is a **content-recommendation decision system** whose governed decision is: *"Should we promote recommender version X+1?"* The business case is that the standard way teams answer this — compare offline NDCG — is unreliable, because offline metrics are computed on data the *old* policy generated. PulseDiscover's job is to make that promote/hold decision **trustworthy**, via honest off-policy evaluation, position-bias correction, and feedback-loop governance.

## Who it's for (and why it's hireable)
- **User of the system (in-narrative):** a content-discovery DS/PM deciding what to ship and monitoring catalog health.
- **Buyer of the candidate (real goal):** Senior/Staff DS — RecSys/Ranking/Discovery at content platforms (Pratilipi, ShareChat, Spotify, YouTube, Netflix, Hotstar, Audible) and marketplaces.
- **Why it raises callback probability:** it demonstrates the scarce skill (counterfactual/off-policy evaluation + feedback-loop reasoning), not the commodity skill (train a two-tower).

## Business objectives (what success means)
1. **Decision validity:** a promote/hold decision backed by an unbiased policy-value estimate (DR), not by old-policy NDCG.
2. **Catalog health:** evidence that the recommender isn't silently concentrating exposure and starving long-tail creators.
3. **Defensibility:** every claim survives first-principles grilling and is backed by a reproducible artifact.

## Success metrics (build-level, not vanity)
- A reproducible bias/variance comparison showing **DR is more trustworthy than IPS and than raw NDCG** (on the simulator).
- Real-data Recall/NDCG for a sequential model beating baselines, **with CIs**.
- A catalog-health audit showing exposure concentration and a loop-breaking demonstration.
- Zero un-sourced numbers in the PRD and defense PDF.

## In scope (BeastMax-core)
Public-data retrieval + sequential model; known-propensity simulator; IPS/SNIPS/DR; IPS position debiasing (PAL if feasible); feedback-loop/catalog-health audit; CIs; evidence artifacts; PRD update; defense PDF.

## Out of scope (deferred to T3/later)
Production deployment + real traffic; LightGCN; DIN/DIEN; DLRM; multi-objective revenue-aware ranking; real-time feature store; cross-subsystem integration.

## Stakeholder value translation
- **Reader:** better "what to read next," sooner; new creators surfaced.
- **Creator:** guaranteed exploration budget; protection from exposure monopolization.
- **Business:** retention lift validated by an unbiased estimator, with a guardrail that blocks engagement wins that come at the cost of churn or catalog health.

## Constraints
- Solo, portfolio build — production-shaped, not production-deployed (stated everywhere).
- Free/cheap infra only (CPU + free GPU; ≤ ~₹150 optional).
- Honesty constraints in `plans/03_RISK_AND_OVERCLAIM_BOUNDARIES.md` are binding.
