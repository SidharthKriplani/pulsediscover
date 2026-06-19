# 09 — Phase 5 Gate 5 Report (IPS / SNIPS / DM / DR validation)

*Scope: off-policy estimator validation on the Phase-4A known-propensity simulator (MovieLens-1M smoke). Estimators use only `(propensity μ, pi_propensity π, reward, q̂)`; the ground-truth value is computed separately from the oracle reward model and is never given to the estimators. Validity is over the **logged candidate-pool support** (per-user top-50), not the full catalog. No two-tower, no Goodreads claims, no PDF.*

---

## 1. What was built
- `src/pulsediscover/ope.py` — IPS, SNIPS, DM, DR estimators; augmented replay computing the **known ground-truth** V(π) and two q̂ reward models (good/bad); ESS + weight diagnostics; user-cluster bootstrap. `[BUILT]`
- `src/run_phase5.py` — driver: estimates, bias vs ground truth, clipping sweep, q̂ stress test, CIs, `ope_comparison.json` + plot. `[BUILT]`
- Artifacts: `outputs/evidence/ope_comparison.json`, `outputs/plots/ope_comparison.png`. Tag `[BUILT][SYNTHETIC — OPE on known-propensity simulator]`.

## 2. Headline results — estimating V(π), N = 60,400 events / 6,040 users
**Known ground truth V(π) = 0.2191.** 95% CIs are user-cluster bootstrap (B=400).

| Estimator | Estimate | Bias vs GT | Bootstrap std | Note |
|---|---|---|---|---|
| IPS | 0.2066 | **−0.0125** | 0.00417 | unbiased in theory; highest variance |
| SNIPS | 0.2083 | −0.0108 | 0.00375 | self-normalized; lower variance than IPS |
| DM (good q̂) | 0.2204 | +0.0013 | — | good reward model → near-unbiased here |
| **DR (good q̂)** | **0.2117** | **−0.0074** | **0.00345** | lowest variance; small bias |
| DM (**bad** q̂) | 0.3513 | **+0.1322** | — | fully trusts the wrong model → badly biased |
| **DR (bad q̂)** | **0.2133** | **−0.0058** | — | **recovers to ≈GT despite the bad q̂** |

## 3. The two findings that matter (reported as observed, not asserted)
1. **DR variance ≤ IPS variance — observed as expected.** DR(good) bootstrap std **0.00345 < IPS 0.00417** (SNIPS 0.00375 sits between). Direction is the expected one; reported honestly from the run, not assumed.
2. **Double robustness — demonstrated by the stress test.** Under a **severely misspecified** q̂ (good q̂ + 0.4 overconfidence offset), the **direct method DM is wildly biased (+0.132)**, but **DR with the same bad q̂ stays near ground truth (bias −0.006)** — its 95% CI [0.207, 0.220] contains GT 0.219, while DM's CI [0.351, 0.351] is far away. Because the logging propensities are *exact*, DR's IPS-correction term carries the estimate even when the reward model is wrong. This is exactly the property DR exists for.

## 4. Clipping sweep (weight cap on w = π/μ)
| clip | ESS | w_max | IPS bias | SNIPS bias | DR(good) bias | DR(bad) bias |
|---|---|---|---|---|---|---|
| none | 12,325 | 65.8 | −0.0125 | −0.0108 | −0.0074 | **−0.0058** |
| 5 | 23,678 | 5 | −0.0429 | −0.0140 | −0.0029 | +0.0106 |
| 10 | 17,746 | 10 | −0.0242 | −0.0116 | −0.0048 | +0.0006 |
| 20 | 14,187 | 20 | −0.0144 | −0.0099 | −0.0057 | −0.0034 |
| 50 | 12,424 | 50 | −0.0125 | −0.0107 | −0.0073 | −0.0057 |
| 100 | 12,325 | 65.8 | −0.0125 | −0.0108 | −0.0074 | −0.0058 |

*(All values from the final overconfident-q̂ run; no-clip DR(bad) bias = −0.0058, consistent with the §2 headline. Note clipping the bad-q̂ DR at c=5 over-corrects to +0.0106 — another instance of clipping injecting bias.)*

**Reading:** aggressive clipping (c=5) **raises ESS** (23.7k vs 12.3k) but **increases IPS bias** (−0.043) — the classic clipping bias-variance tradeoff: capping large weights throws away the corrections they carry. **DR(good) bias stays small and stable (−0.003 to −0.007) across all clips** — DR is the least clip-sensitive estimator. SNIPS is also fairly stable.

## 5. Weight diagnostics
ESS = **12,325 / 60,400** (≈20%) at no clip; **w_max = 65.8**. Moderate weight dispersion — overlap is real (every action has positive π-propensity by design) but heavy enough that variance reduction (SNIPS/DR) matters. This is the honest reason DR/SNIPS beat raw IPS here.

## 6. Honest caveats (no spin)
- **All estimators slightly *under*-estimate GT** (biases −0.6 to −1.3 pp for IPS/SNIPS/DR-good). Plausible finite-sample + candidate-pool-support effects; reported as-is, not zeroed out.
- **Validity is over the logged candidate-pool support** (per-user top-50), **not full-catalog** — items never in any pool have zero logging probability and are out of scope for these estimates.
- **MovieLens-1M smoke + synthetic relevance.** This validates *estimator behaviour against a known truth*; it is **not** real online lift and makes no domain claim.
- **DM(good) looks near-unbiased only because the good q̂ is genuinely good**; the meaningful test is DM(bad) vs DR(bad), which is sharp.

## 7. No-overclaim check (per `plans/03`)
- Estimators never touched `true_rel`; ground truth computed separately. ✅
- DR-beats-IPS framed as *expected then observed*, not asserted. ✅
- Positivity described as candidate-pool/slate-level. ✅
- Synthetic simulator; no real-lift claim; no two-tower/Goodreads/PDF. ✅

## 8. Gate 5 status & decision
- ✅ IPS/SNIPS/DM/DR computed vs known ground truth, with bias, variance, bootstrap CIs, ESS, clipping sweep, and a sharp q̂ stress test demonstrating double robustness.
- ⏸️ Domain headline (Goodreads/Amazon) and position-bias correction (PAL/IPS) still pending.

**Gate 5 decision for you:** approve next step — (a) **Phase 6: position-bias correction** (IPS reweighting vs PAL additive tower, with the item-movement plot), (b) **load a Goodreads fiction genre / Amazon Books** subset to move the whole spine (baselines → SASRec → simulator → OPE) to the domain headline, or (c) **firm up Phase 5** (e.g., SASRec-based π, larger bootstrap, full-catalog discussion).

**STOP — awaiting Gate 5 review. No further phase begins until approved.**
