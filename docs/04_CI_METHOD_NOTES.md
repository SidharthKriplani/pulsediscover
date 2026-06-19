# 04 — Confidence-Interval Method Notes

*Every headline number in PulseDiscover carries N + a 95% CI. This documents how, so the numbers are defensible.*

## Ranking metrics (Recall@K, NDCG@K)
These are **means over held-out users** (one held-out item per user, leave-last-out). The correct uncertainty is **user-level**, so the headline interval is a **per-user bootstrap**:
- Resample users with replacement (2,000 resamples), recompute the mean each time, take the 2.5%/97.5% percentiles.
- Distribution-free; captures the real source of variance (users differ); no normality assumption.
- Implemented in `src/pulsediscover/eval.py::bootstrap_ci`.

## Recall@K as a proportion — Wilson interval (cross-check)
Recall@K is also a clean hit/miss proportion, so we additionally report a **Wilson score interval** (`eval.py::wilson_ci`):
- Correct at small N where the normal approximation fails (e.g., faithfulness-style 0.85 on tiny N has a huge interval).
- The bootstrap and Wilson intervals should roughly agree; **divergence is itself a small-N warning** we surface rather than hide.

## Off-policy estimators (Phase 5, later)
IPS/SNIPS/DR values will use **bootstrap over logged sessions** (not users), since the estimand is a session-weighted policy value; variance comparison across estimators uses the same resamples for a paired comparison.

## Rules (binding)
- **No metric is reported without N and a 95% CI.**
- N is always shown next to the interval (`n_eval_users` / `n_sessions`).
- Seed is pinned (`SEED = 20260616`) so intervals are reproducible.
- If N is small, the wide CI is reported honestly — never narrowed by switching to a normal approx.
- Code-validation fixture numbers are **never** reported as findings (they live in `outputs/_codecheck/`, flagged `is_evidence: false`).
