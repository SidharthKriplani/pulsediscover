# 07 — Logged-Event Schema & Simulator Design (Phase 4A)

*The known-propensity simulator on MovieLens-1M smoke. Its entire reason to exist: real datasets have no logged propensities, so off-policy estimators can't be validated on them. This manufactures known propensities so Phase 5 (IPS/SNIPS/DR) can be checked against ground truth. It does NOT estimate any policy value.*

## Logged-event schema (`data/interim/logged_events.{parquet|csv}`)
| Field | Type | Meaning |
|---|---|---|
| `user_id` | int | reader (train users only) |
| `context_len` | int | length of the user's train history (the context) |
| `item_id` | int | the item shown in this slot (the action) |
| `rank` | int | slot index, 0 = top |
| `position` | int | rank + 1 |
| `propensity` | float | **μ's exact per-slot probability of the shown item** (the logging propensity) |
| `exploration_flag` | 0/1 | 1 if this slot was drawn from the uniform exploration branch |
| `examined` | 0/1 | drawn from the PBM examination curve at this rank |
| `reward` | 0/1 | click = examined AND Bernoulli(ground-truth relevance) |
| `pi_propensity` | float | **π's per-slot probability of the same action** (for Phase-5 IPS/DR overlap) |
| `true_rel` | float | **ORACLE/debug only** — the ground-truth relevance prob; used for sanity, never for estimation |

## Components (all built from TRAIN only — no test leakage)
- **μ — logging policy:** popularity-softmax over each user's top-50 candidate pool (TRAIN popularity), ε-mixed to uniform. The realized per-slot probability is logged exactly as `propensity`.
- **truth — reward model:** `sigmoid(α·zscore(ALS_score) + bias)`, ALS trained on TRAIN. The simulator's known ground truth (what Phase-5 estimators must recover).
- **π — target policy:** `softmax(ALS_score, τ_π)` over remaining candidates (the "new" policy we'd evaluate). `μ ≠ truth ≠ π` to avoid trivial circularity.

## Slate generation (exact propensities + positivity)
Sequential sampling without replacement, L=10 slots. At slot *r* over remaining candidates *R*:
```
p_slot(a) = (1-ε)·softmax_μ(a | R)  +  ε·(1/|R|)
```
- **Candidate-pool / slate-level positivity:** within the logged candidate pool, every remaining item has `p_slot ≥ ε/|R| > 0` (no zero-propensity action can be selected). This is positivity **over the slate's candidate pool**, NOT full-catalog positivity — items outside the per-user top-50 pool have zero logging probability, so OPE validity holds over the **logged candidate-pool support**, not the entire catalog.
- **Per-slot probabilities sum to 1** by construction: `(1-ε)·1 + ε·1 = 1`.
- Examination: `examined ~ Bernoulli(PBM(rank))`, PBM declining with rank (base 0.95, γ 0.7).
- Reward: `examined AND Bernoulli(true_rel)`.

## Configuration (this run)
ε=0.10 · slate_L=10 · pool_N=50 · τ_μ=1.0 · τ_π=0.7 · PBM base=0.95, γ=0.7 · rel_α=1.2, rel_bias=−0.8 · seed=20260616.

## Leakage control
μ, truth, and π are computed **only** from `train` (the leave-last-out training partition). The held-out next-item labels and test items are **never** referenced in any of them; the test set is used solely to define the split boundary. Asserted in `run_phase4.py`.

## What this proves / does not prove
- **Proves (in Phase 5):** estimator validity — IPS/SNIPS/DR can recover the known policy value **over the logged candidate-pool support**; candidate-pool positivity + known propensities make this legitimate.
- **Does NOT prove:** real online lift, real reader behavior — the relevance is a known synthetic ground truth by construction. Labeled `[BUILT][SYNTHETIC — known-propensity simulator]`.
