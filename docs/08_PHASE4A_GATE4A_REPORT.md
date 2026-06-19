# 08 — Phase 4A Gate 4A Report (Known-Propensity Simulator + Logging Policy)

*Scope: build the known-propensity semi-synthetic simulator + logging-policy design on MovieLens-1M smoke; log events; run sanity checks. NO IPS/SNIPS/DR estimation (Phase 5). No two-tower, no Goodreads claims, no PDF. Stops at Gate 4A.*

---

## 1. What was built
- `src/pulsediscover/simulator.py` — μ (logging policy), ground-truth reward, π (target), PBM examination, exact per-slot propensity logging with ε-exploration positivity. `[BUILT]`
- `src/run_phase4.py` — Phase-4A driver: identical leakage-safe split, train-only μ/truth/π, event generation, sanity checks, manifest + plots. `[BUILT]`
- Artifacts: `outputs/evidence/simulator_manifest.json`, `outputs/evidence/simulator_sanity_report.json`, `data/interim/logged_events.*`, `outputs/plots/simulator_diagnostics.png`. Schema in `docs/07_SIMULATOR_SCHEMA.md`. Tag: `[BUILT][SYNTHETIC — known-propensity simulator]`.

## 2. Logging-policy design (summary; full schema in doc 07)
- **μ:** popularity-softmax over each user's top-50 pool (TRAIN popularity), ε-mixed to uniform → exact `propensity` logged per slot.
- **Candidate-pool / slate-level positivity** (not full-catalog): `p_slot(a) = (1-ε)·softmax_μ + ε/|R| ≥ ε/|R| > 0` for items in the per-user pool; items outside the pool have zero logging probability, so OPE validity is over the logged candidate-pool support.
- **PBM examination** declining with rank; **reward** = examined AND Bernoulli(ground-truth relevance).
- **truth** = sigmoid(ALS), **π** = softmax(ALS); `μ ≠ truth ≠ π`.

## 3. Sanity-check results — **ALL PASS** (`simulator_sanity_report.json`)
- **Events:** 60,400 across **6,040 impressions** (one 10-slot slate per train user).
- **Propensities sum correctly:** max abs deviation from 1 = **0.0** (exact).
- **No zero-propensity selected actions:** min logged propensity **0.00452 > 0**; zero-propensity count **0**.
- **Exploration share matches ε:** **0.0985** vs configured **0.10** (|Δ| < 0.02). ✅
- **Examination plausible + declining:** overall examination rate **0.379**, monotonically decreasing by rank (PBM holds). ✅
- **Reward plausible:** overall reward rate **0.166** (in 0.01–0.8). ✅
- **π overlap support:** min π-propensity **> 0** (every logged action has positive target-policy probability → IPS/DR will have overlap). ✅
- **No test-label leakage:** μ/truth/π built only from `train`; held-out labels/items never referenced (asserted in code).

Run time: ~16 s in-sandbox.

## 4. Plots
`outputs/plots/simulator_diagnostics.png` — (left) examination + reward rate by rank (PBM shape); (right) histogram of log10 logged propensities showing the strictly-positive floor.

## 5. No-overclaim check (per `plans/03`)
- No real online lift claimed; labeled known-propensity **synthetic** simulator. ✅
- No IPS/SNIPS/DR estimation performed (correctly deferred to Phase 5). ✅
- `true_rel` is oracle/debug only — flagged, not to be used by estimators. ✅
- No two-tower / Goodreads / PDF. ✅
- μ/truth/π distinct → avoids the circular-label trap (the NexusSupply lesson). ✅

## 6. What this sets up (Phase 5, on approval)
The logged events now carry everything IPS/SNIPS/DR need: a known logging `propensity`, a target `pi_propensity`, `reward`, `position`/`examined`, and a known ground-truth (`true_rel`) to validate against. Phase 5 = compute IPS/SNIPS/DM/DR estimates of π's value, compare to the simulator's ground-truth value, with bootstrap CIs. **DR is *expected* to reduce variance vs IPS and to stay ~unbiased under an imperfect q̂ — this is reported honestly as observed, not asserted in advance.**

## 7. Gate 4A status & decision
- ✅ Simulator + logging policy built; positivity + known propensities guaranteed.
- ✅ All sanity checks pass; manifest, schema, sanity report, plots produced.
- ⏸️ No estimators yet; domain headline (Goodreads/Amazon) still pending.

**Gate 4A decision for you:** approve **Phase 5 (IPS / SNIPS / DR estimator validation against the simulator's known ground truth, with CIs)** — the core differentiator — or request changes to the simulator first (e.g., different ε, PBM shape, or a SASRec-based π).

**STOP — awaiting Gate 4A review. No Phase 5 work begins until approved.**
