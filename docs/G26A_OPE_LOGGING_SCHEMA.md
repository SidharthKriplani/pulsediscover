# G26A — OPE Logging Schema (off-policy evaluation readiness)

*Designs the per-impression log that makes IPS / SNIPS / Doubly-Robust off-policy evaluation **valid**. PulseDiscovery is currently **OPE-ready by design, not by data**: this schema is specified; the propensities are not yet logged. **No valid IPS/SNIPS/DR claim may be made until `pi_log` and `reward` are actually populated from a logging-policy serving run.***

## Why this is the highest-leverage missing wire
Every behavioural/value claim in PulseDiscovery is offline. The single thing standing between "offline frontiers" and "estimated policy value without a full A/B" is **logged propensities** — the probability the *logging* policy assigned to each shown item. Without `pi_log`, IPS weights `pi_new/pi_log` are undefined and any OPE number is fiction. With it, the existing rerankers (G26) become *evaluable* off-policy.

## Schema (one row per shown item per impression)

| Field | Type | Purpose | OPE role |
|---|---|---|---|
| `request_id` | str | one recommendation request | join key |
| `impression_id` | str | one shown slate | unit of exposure |
| `user_id` | str (hashed) | user | context |
| `item_id` | str | shown item | action |
| `rank` | int | slot index in slate | position |
| `position` | int | rendered on-screen position | position-bias correction (PAL) |
| `policy_id` | str | logging policy identity | which π_log |
| `model_version` | str | e.g. `als_f64_c2:<md5>` | reproducibility / registry join |
| `index_version` | str | e.g. `faiss_flatip:<md5>` | reproducibility |
| `candidate_source` | enum | als/content/item_sim/popularity/cooc | multi-source attribution |
| `retrieval_score` | float | stage-2 score | feature / reward model |
| `rerank_score` | float | stage-3 score | feature / reward model |
| `final_score` | float | score that set the rank | propensity model input |
| **`pi_log`** | float (0,1] | **prob logging policy showed this item at this slot** | **IPS/SNIPS/DR denominator — REQUIRED** |
| `exploration_bucket` | enum | control / explore-tail / explore-random | guarantees π_log overlap (no zeros) |
| `fallback_reason` | enum/null | unknown_user/invalid_k/faiss_error/null | excludes degenerate impressions from OPE |
| `served_at` | ts | serve time | latency / windowing |
| **`reward`** | float/null | click/read/rating/dwell | **OPE numerator — REQUIRED** |
| `reward_observed_at` | ts | reward time | attribution latency |
| `reward_window` | duration | attribution window | reward definition guard |

## How each estimator uses it
- **IPS:** `V̂ = (1/N) Σ_i (π_new(a_i|x_i)/pi_log_i) · reward_i`. Needs `pi_log` overlap (every shown item had `pi_log>0`) — guaranteed by `exploration_bucket`.
- **SNIPS:** IPS normalized by `Σ weights` — lower variance; **first estimator we'd report**.
- **Doubly Robust:** reward-model baseline + IPS correction on residual; needs `reward` + a model over (`retrieval_score`,`rerank_score`,context). Robust if either piece is right.
- **PAL / position-bias:** uses `position` to separate examination from relevance so click `reward` isn't position-confounded.

## Validity preconditions (all must hold before any OPE claim)
1. `pi_log` logged and strictly positive for every shown item (overlap / no deterministic policy without exploration).
2. `reward` populated within a fixed `reward_window`.
3. Logging policy `policy_id` and target policy clearly distinct and both reconstructable.
4. Fallback/degenerate impressions excluded (`fallback_reason` not null → drop from OPE).
5. Stochasticity present (`exploration_bucket`) so weights are bounded.

## Status
**OPE-ready by design. NOT valid yet — no `pi_log`/`reward` logged.** The G23 service already emits `request_id`, `model_id`, `index_id`, `fallback_reason`, `latency`; the missing fields are `pi_log`, `position`, `exploration_bucket`, and `reward`. Populating them is the gate after G26B. Until then: **"I won't report an IPS number I can't defend."**
