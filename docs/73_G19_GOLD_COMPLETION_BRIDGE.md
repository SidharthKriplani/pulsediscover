# 73 — G19: Gold Completion Bridge / Method-Depth Resolution

*Final template-alignment pass against RiskFrame Gold Standard v4. Question: can PulseDiscover move from `gold_candidate` (8.7) to `gold_complete` via methodological closure, or is a run required? **No new models run in this gate** — G19 is the decision. Parts: A = docs/75 (template audit), B+C = docs/74 (method-depth + achievement moments), D = below. Machine-readable: `outputs/evidence/g19_gold_completion_decision.json`. (docs/72 = locked final status; G19 occupies 73–75.)*

## The central question
Can "MF wins on this catalog" be defended as a **terminal** senior finding through documentation alone — or does it require a technical run?

## Part D — Final decision
### `requires_lightgcn_or_canonical_sasrec_run`
**Specifically: canonical full-softmax SASRec trained to convergence is the *necessary* run; LightGCN is *optional/strongly-desirable*.**

### Why this and not `upgrade_to_gold_complete_now`
The template audit (docs/75) shows **all sections COMPLETE or non-blocking except one**: §9 Technique Tournament. The method-depth argument (docs/74) is **strong** — serious tournament, the full-softmax confound controlled, domain priors aligned, and the real lever found (candidate coverage). But it is **not yet terminal**: the closest competitor (canonical full-softmax SASRec) was **provably still improving when stopped** (loss descending at 40 ep, R@20 0.046). 

Per the locked decision rules: *do not mark gold_complete if canonical SASRec is judged necessary, or if any claim boundary would need loosening.* Declaring "MF terminally wins" via documentation would assert an unrun result and loosen the boundary — which the project has refused to do throughout. So **methodological closure alone is insufficient**; one bounded convergence run closes the gap.

### Why not `requires_live_ab_only` or `remain_gold_candidate`
- Not `live_ab_only`: the live A/B is about *online lift* (a separate, deferred axis), **not** about the method-depth/model-family question that blocks gold here. Online lift is explicitly out of scope and not required for RiskFrame-gold (which is offline-methodology gold).
- Not `remain_gold_candidate` as a *terminal* answer: that's the current status, but G19's job is to name the **single bounded action** that unlocks gold — which it does.

### What the run must be (bounded, not scope expansion)
- **Canonical full-softmax SASRec, same config (d64/2blk/2head, full-softmax over 40k), trained to convergence** (loss plateau), re-evaluated on the same 5k LLO protocol vs ALS f64. Checkpoints already saved (`outputs/_models/gpu/ckpt_canonical.pt`) → resume, don't restart.
- **Optional:** LightGCN candidate/scorer for graph-family closure.
- **Outcome handling (pre-registered):** if converged canonical still < ALS → "MF wins" becomes **terminal** → eligible for `gold_complete`. If it contends/beats ALS → an advanced model finally moves the floor → also gold-eligible (different, also honest). Either way the model-family question closes.

### What is already gold-grade (so only §9 remains)
Deep defense kernel (G17), evidence ledger, evaluation rigor, claim safety, interview defensibility, operational realism (G1/G16), achievement moments (docs/74 §C), convergence rubric + acceptance test (docs/75) — all COMPLETE. The five non-blocking NEEDS-MEAT items (JD map, synthetic-realism audit, $ cost chain, rendered demo, infra-tier doc) are polish, not gates.

## Claim safety (unchanged — nothing loosened)
No online lift · no production · no live A/B · no deep-model-beats-ALS claim · no solved cold-start/fairness/health · no FAISS quality claim · no 90/10-proven · no full-catalog-for-within-candidate · **no gold_complete claim.** Status remains **`gold_candidate`** until the convergence run executes.

## Bottom line
PulseDiscover is **one bounded, honest GPU run away from gold_complete** — train the canonical SASRec to convergence (checkpoints exist), re-evaluate vs ALS, and either way the model-family question closes terminally. Until then it correctly stays a strong **gold candidate (8.7)**. No fabrication, no boundary loosening, no fake deep-model win.

**STOP — G19 decision: `requires_lightgcn_or_canonical_sasrec_run` (canonical-SASRec-to-convergence necessary; LightGCN optional). Do not start G20 / run models without approval.**
