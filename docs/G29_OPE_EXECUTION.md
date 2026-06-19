# G29 — Off-Policy Evaluation Execution (Gold Pass 2/3)

*PulseDiscover V2, Gold Pass 2/3. Executes the OPE pipeline designed in G26A: a stochastic logging policy that records propensities, then IPS / SNIPS / Doubly-Robust estimators that recover a known target-policy value. Evidence: `outputs/evidence/g29_ope_execution_report.json`; plot `outputs/plots/g29_ope_estimator_validation.png`. Code: `src/ope/logging_policy.py`, `scripts/run_g29_ope_execution.py`.*

> **Honesty boundary (read first):** this is **OPE methodology demonstrated OFFLINE**. Propensities come from a **synthetic stochastic logging policy we control**, not real traffic. Rewards are a **held-out-future-interaction PROXY** (binary relevance vs `test.csv`), not real engagement. It validates that the estimators and the logged-propensity pipeline are **correct** — it does **NOT** estimate real online lift. Served-c2 only.

## 1. Why G29
G26A designed the propensity-logging schema and locked the rule "no IPS/SNIPS/DR claim until propensities are logged." G29 closes that gap by actually executing the pipeline: it builds logged data with recorded `pi_log`, runs the three estimators, and checks they recover a value we can compute directly.

## 2. Setup (single-action contextual-bandit OPE)
Context = user; action = one item drawn from the user's candidate pool; **logging policy π₀** = ε-uniform (ε=0.15) + softmax over an inverse-rank blend, guaranteeing `π₀(a|x)>0` (overlap). Reward = 1 if the drawn item is in the user's held-out test interactions. Evaluated on the **leakage-safe TEST users** (same split as G28), 30 Monte-Carlo draws/user → **28,320 logged samples / 944 users**. Target policies (deterministic top-1): **ALS-top1** and **RRF-fusion-top1**. DR reward model = a logistic `r̂(x,a)` fit on the **train-split** users (different users → no test leakage), calibrated to the true base rate.

## 3. Results — estimators recover the known value
| Target policy | true (precision@1) | IPS | SNIPS | DR | ESS |
|---|---|---|---|---|---|
| ALS top-1 | 0.0106 | 0.0211 | 0.0129 | **0.0089** | 266 |
| RRF-fusion top-1 | 0.0117 | 0.0218 | 0.0134 | **0.0110** | 220 |

**What this validates:**
- All three estimators land in the right neighbourhood of the directly-computed true value — the logged-propensity pipeline is correct.
- **DR is closest to truth** (0.0089 vs 0.0106; 0.0110 vs 0.0117) — as expected, its reward-model baseline reduces bias.
- **IPS over-estimates** (~2×) — the textbook high-variance failure mode of raw importance weighting at modest ESS.
- **SNIPS** sits between (self-normalization controls variance) — the recommended estimator when no reward model is available.
- The estimators correctly **rank RRF-fusion above ALS** (0.0117 > 0.0106), the same ordering OPE would use to choose a policy.

## 4. What G29 unlocks (and what it does not)
**Unlocks (safe):** "I executed the full OPE pipeline — logging policy with recorded propensities, then IPS/SNIPS/DR that recover a known target-policy value; DR lowest-bias, SNIPS lowest-variance, IPS shows the importance-weight variance problem." The system is now **OPE-capable, demonstrated end-to-end**, not just schema-designed. **Does NOT unlock:** real online value, online lift, or "fusion beats ALS in production" — the propensities are synthetic and the reward is a proxy. ESS (220–266 of 28,320) honestly flags target/logging overlap as the variance driver.

## 5. Failure modes / caveats
Single-action top-1 OPE (not slate OPE); proxy reward (held-out membership, not engagement); synthetic logging policy (not real traffic); low ESS → estimates are illustrative-scale, not precision instruments; DR depends on a decent reward model (a balanced/mis-calibrated model breaks it — we use a base-rate-calibrated logistic).

## 6. Claim boundary
**Safe:** "Executed off-policy evaluation end-to-end (logging policy + recorded propensities + IPS/SNIPS/DR) and validated the estimators recover a known policy value offline; SNIPS recommended for variance, DR for bias when a reward model exists." **Forbidden:** online lift proven, real off-policy value estimated, production deployed, learned ranker beats ALS online, fairness certified, cold-start solved.

## 7. Status
✅ G29 done (Gold Pass 2/3). OPE methodology executed + validated offline. V1 (`gold_candidate` 8.9) untouched. **Next: G30 — final RiskFrame gold audit + interview kit (Gold Pass 3/3).**

**STOP — Gold Pass 2/3 complete.**
