# 10 — Phase 6 Gate 6 Report (Position-Bias Correction)

*Scope: position-bias correction on the Phase-4A simulator logs — naive vs IPS examination-reweighting (Route A) vs PAL additive tower (Route B). Per-item relevance aggregated over impressions; ground-truth `true_rel` used only for comparison, never by the estimators. Simulator-scoped; no real-lift claim. No two-tower, no Goodreads, no PDF.*

---

## 1. What was built
- `src/pulsediscover/position_bias.py` — naive / IPS examination-reweighting / **PAL additive-logit tower** (torch), comparison metrics (MSE, Pearson, Spearman, position-confounding). `[BUILT]`
- `src/run_phase6.py` — driver: empirical examination curve, per-item estimates, PAL fit, item-movement, plots, `position_bias_report.json`. `[BUILT]`
- Artifacts: `outputs/evidence/position_bias_report.json`, `outputs/plots/position_bias.png`. Tag `[BUILT][SYNTHETIC — position-bias correction on simulator logs]`.

## 2. Examination curve recovered (validates the PBM)
Empirical P(examined|rank) from the logs vs the known PBM curve:

| rank | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| empirical | 0.948 | 0.588 | 0.433 | 0.372 | 0.319 | 0.278 | 0.241 | 0.220 | 0.198 | 0.193 |

The empirical examined-rate tracks the known declining PBM curve — the examination model is recovered from logged exposure. (Plot panel 1.)

## 3. Estimators vs ground truth (124 items with ≥20 impressions)
| Estimator | MSE vs true ↓ | Pearson vs true ↑ | Spearman vs true ↑ | corr with avg_rank (position-confounding, →0 is better) |
|---|---|---|---|---|
| **naive** (mean click) | 0.0534 | 0.755 | 0.708 | **−0.471** |
| **IPS** (examination-reweighted) | **0.0115** | 0.628 | 0.633 | **−0.253** |
| **PAL** (additive tower, Route B) | — (logit scale) | — | 0.699 | — |

## 4. Honest reading (no forced win — both wins AND a cost reported)
- **IPS removes most of the position bias — the core claim, demonstrated.** Naive relevance is strongly position-confounded (corr with avg display rank **−0.471**: items in better/top slots look more relevant). IPS reweighting cuts that confounding roughly in half (**−0.253**) and slashes absolute error to ground truth **4.7×** (MSE 0.0534 → 0.0115). This is exactly what examination-reweighting is for.
- **The honest cost: IPS is noisier in *ranking*.** IPS's Pearson/Spearman correlation with truth (0.63) is **lower** than naive's (0.71). Dividing clicks by small examination propensities at low ranks injects variance, so while IPS is far less *biased*, its *ranking* of items is a bit noisier here. **Neither dominates on every metric** — IPS wins decisively on bias/de-confounding, naive edges it on rank correlation. Reported as-is.
- **PAL (Route B) is BUILT** (torch available): an additive-logit tower `logit(click)=θ_item+β_pos`, BCE loss 0.406; θ = position-debiased relevance, β dropped at serving. Its Spearman vs truth is **0.699** — comparable to naive, slightly below, and it achieves debiasing **without** IPS's high-variance weights. No clear win over naive on ranking here; reported honestly.
- **Item movement (plot panels 2–3):** worse-placed items (higher avg display rank) receive larger upward corrections under IPS — the items that were undervalued because they were shown in poor slots move up; well-placed items move down. Example up-movers: items 588, 750, 1089, 908, 1219.

## 5. Caveats (no spin)
- **Simulator-scoped**, per-item relevance aggregated over impressions; **MovieLens-1M smoke**.
- **Examination propensity is estimated from logged exposure** (the sim exposes `examined`); in a real system it would come from randomization / result-swap interventions or a PBM-EM fit — stated, not hidden.
- Only **124 items** clear the ≥20-impression bar → correlations are noisy; the bias/de-confounding result (MSE, corr-with-rank) is the more robust finding than the rank-correlation deltas.
- PAL's absolute probability isn't directly comparable to `true_rel` due to the additive-logit offset identifiability; PAL is judged on ranking (Spearman) only.

## 6. No-overclaim check (per `plans/03`)
- IPS reported with both its win (bias/de-confounding) and its cost (rank-correlation variance) — not forced. ✅
- PAL built and labeled accordingly; not claimed to beat naive when it doesn't. ✅
- Examination propensity described as estimated, not god-given. ✅
- Simulator-scoped; no real online lift; no two-tower/Goodreads/PDF. ✅

## 7. Gate 6 status & decision
- ✅ Naive, IPS (Route A), and PAL (Route B) implemented; examination curve recovered; item-movement + examination-curve plots produced; honest comparison documented.
- ⏸️ Domain headline (Goodreads/Amazon) still pending; feedback-loop audit (Phase 7) not started.

**Gate 6 decision for you:** approve next — (a) **Phase 7: feedback-loop / catalog-health audit** (exposure concentration, Gini/entropy over rounds, popularity-ratchet demo), (b) **load a Goodreads fiction genre / Amazon Books** subset to move the whole spine to the domain headline, or (c) **firm up Phase 6** (e.g., tune PAL, raise the impression threshold, add a known-propensity examination variant).

**STOP — awaiting Gate 6 review. No further phase begins until approved.**
