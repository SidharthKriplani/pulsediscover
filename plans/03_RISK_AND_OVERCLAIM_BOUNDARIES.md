# 03 — Risk & Overclaim Boundaries

*Binding constraints for the PulseDiscover BeastMax build. Anything here overrides convenience. The purpose is that, after the build, no claim can be punctured in an interview.*

## Status taxonomy (what each tag is allowed to mean)
- `[BUILT]` — code runs, artifact committed, reproducible from seed.
- `[BUILT — real data]` — the above, on a public real dataset (MovieLens/Goodreads/Amazon).
- `[BUILT][SYNTHETIC]` — real code on synthetic/simulator data (e.g., OPE on the simulator). Proves estimator/mechanism validity, **not** real-world lift.
- `[VISION]` — designed, not built. No number.
- `[BUILD-TASK]` — scheduled work.
- `[V2/T3]` — deferred (LightGCN, DIN, DLRM, production infra, multi-objective revenue).
- `[OVERCLAIM→DELETE]` — must not be said.

## What must NEVER be claimed
1. **No production ownership.** Never "deployed," "served real users," "X QPS in production." Say "production-shaped solo build."
2. **No synthetic-as-real.** Simulator/OPE results prove estimator validity, never real online lift or real reader behavior.
3. **No real OPE on raw observational data.** IPS/DR validity is claimed **only** on the known-propensity simulator. Never "unbiased OPE on raw Goodreads."
4. **"0.134 → 0.522" is never an improvement.** Different eval frames on different subsets. Honest figure: IPS bias_delta −0.005.
5. **No unbuilt model claimed as built.** Two-tower, SASRec, DR, PAL, LightGCN are not "built" until their evidence JSON exists. Until then they are `[VISION]`/`[BUILD-TASK]`.
6. **PAL ≠ IPS.** PAL reparameterizes the model (additive examination tower); IPS reweights the loss. Never conflate; never call PAL "built" unless implemented.
7. **No company-name authority.** Industry-pattern phrasing only ("a mature feed-ranking org would likely…"), never "Netflix/Spotify does exactly this."
8. **No number without N + CI + source** once the CI phase is done.
9. **No fabricated or circular metric.** (Cautionary reference: NexusSupply's AUC 1.0 came from circular labels — the simulator must avoid drawing "relevance" from the same signal as exposure.)
10. **The e-commerce PulseRank repo is never re-skinned as a fiction recommender.** It is the inherited *evaluation engine*; PulseDiscover is a new content-domain build.

## What IS honestly claimable today (pre-build)
- IPS/SNIPS mechanics + clipping sweep `[BUILT][SYNTHETIC]`; bias_delta −0.005.
- LambdaRank beats popularity (0.192 vs 0.177) `[BUILT][SYNTHETIC]`.
- Guardrail-first HOLD decision `[BUILT][SYNTHETIC]`.
- Honest negatives: V1 lost to popularity; Gini worsened (0.404→0.582); ALS not wired into ranker.
- All defense-card derivations as **conceptual** (math correct regardless of build).

## Build-specific risks and mitigations
| Risk | Likelihood | Mitigation / fallback |
|---|---|---|
| Goodreads too large/awkward to ingest | medium | fall back to Amazon Books; MovieLens-only smoke if both fail (label headline gap honestly) |
| SASRec/two-tower don't beat baselines | medium | report it honestly (a real negative is still evidence); debug; do not fabricate |
| DR estimator unstable (q̂ misspecified) | medium | ship IPS/SNIPS solidly; mark DR `[BUILD-TASK]`; never report an unstable DR number as a result |
| Simulator too clean / accidentally circular | medium | realism audit (Zipf, PBM curve, ε bucket, injected cold items); ensure relevance ≠ exposure signal |
| PAL training cost overruns budget | medium | defer PAL to `[V2]`; IPS reweighting carries the position-bias story |
| Shapley feedback decomposition too complex | medium | ship simple mix/rate exposure decomposition first |
| Compute limits (GPU) | low-medium | free Colab/Kaggle GPU; cap sequence length; ≤ ~₹150 A100 hour only if needed |
| Scope creep into T3 | medium | strict gate: LightGCN/DIN/DLRM/revenue ranking are out of BeastMax-core |
| Overstating CIs on small N | low | report N alongside every CI; widen honestly |

## "Say instead" quick reference
- ✗ "Deployed in production" → ✓ "Production-shaped solo build on public + synthetic data."
- ✗ "DR proves online lift" → ✓ "DR is my best *offline* predictor of online lift, validated on a known-propensity simulator."
- ✗ "Unbiased OPE on Goodreads" → ✓ "OPE validity shown on the simulator; Goodreads gives real ranking quality, not propensities."
- ✗ "Built SASRec + DR + PAL" (pre-build) → ✓ "Designed; SASRec/DR build-now, PAL build-if-feasible — no numbers until the artifacts exist."
- ✗ "4× NDCG improvement" → ✓ "Two different evaluation frames; honest IPS delta is −0.005."

*These boundaries are binding through every build phase and into the PRD update and defense PDF.*
