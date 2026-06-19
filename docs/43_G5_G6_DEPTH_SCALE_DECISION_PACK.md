# 43 — G5/G6: Depth & Scale Decision Pack

*Resolves the remaining technical-depth/scale question before capstone synthesis: does PulseDiscover need one more advanced model, a documented deferral, a full-population evaluation, or both? **No new code/experiments in this doc — it is a decision pack. No GPU run is claimed; no advanced-model win is claimed.***

---

## Part A — G5: model-depth decision

### A.1 The model-depth gap
- **SASRec was an honest negative** under a CPU/last-position setup — R@20 0.036, below every collaborative baseline (D3B).
- **ALS f64 remains the warm floor** — R@20 0.085, best single generator built.
- **No competitive advanced sequence/two-tower/GNN model has beaten ALS** in this project. The depth gap is real and acknowledged (BM2-GAP scored retrieval depth 6/10).

### A.2 Options
| Option | Product/system question it answers | Expected value | Cost / feasibility | Model-chasing risk | Acceptance criteria | Kill condition |
|---|---|---|---|---|---|---|
| **Canonical GPU SASRec** | Can a *properly trained* sequence model beat the MF floor? | High — directly closes the gap or yields a clean comparative negative | **Not feasible in this sandbox** (no GPU); feasible on free Colab/Kaggle | Low — it's the model already attempted, finished honestly | full-position, d≥64, 2 blocks/heads, maxlen≥50, 100+ epochs, pop-corrected negatives; eval vs ALS f64 on the same sample | abandon if it can't be trained to convergence or eval isn't apples-to-apples |
| **Two-tower retrieval** | Can content+collaborative features fuse into a learned retriever? | Medium-High — natural fit for the warm+cold fusion story | Medium; CPU-trainable at small scale but a new build | Medium — risks a second under-powered model | beats or matches ALS f64 at equal eval, or documents why not | kill if it becomes another CPU-limited under-train |
| **LightGCN / graph recommender** | Does a graph propagation model help on this bipartite data? | Medium | Medium-High; heavier to build/tune | **High — classic model-chasing** | only if it answers a specific question ALS can't | reject by default |
| **f128 / f64-hybrid ALS** | Do more factors / a refit hybrid lift the floor further? | Low-Medium (f16→f64 already converged by iter 4; f64 hybrid needs refit+remerge) | Low-Medium; external pack | Low | measurable lift over f64 at equal eval | skip if lift is marginal |
| **Documented deferral** | Is the current depth story defensible as-is? | High (honesty + control) | Free | None | the gap is named, bounded, and not overclaimed | n/a |

### A.3 Recommendation — **defer G5 (documented), with an optional external GPU SASRec**
- **GPU is unavailable in this environment**, so a canonical SASRec **cannot** be run here. I will **not** pretend a competitive advanced model was built.
- **Defer G5 as a defensible, documented deferral.** The current depth story is bounded and honest, but **not fully closed at T3/model-tournament level**: a tuned MF floor (ALS f64) plus a *fully diagnosed* sequence-model negative (D3B). "A transformer under CPU/last-position loses to a well-tuned MF baseline, and here's exactly why" is itself a senior-level result. **Documented deferral satisfies the G5 claim-honesty requirement; it is not the same as a competitive advanced-model result.**
- **Optional external path:** if/when a free GPU is available, run canonical SASRec to the A.2 acceptance criteria and append the result (win *or* clean negative). Two-tower is the second choice; LightGCN is rejected as model-chasing.
- **This does not block capstone synthesis, provided the final dossier explicitly labels advanced-model depth as deferred/optional rather than solved.** Per BM2-GAP, one competitive advanced model is a **high-value upgrade, not a blocker**, and the stop condition explicitly accepts "one competitive model **or** an explicit documented deferral." This is that deferral, justified — not deferral disguised as completion.

## Part B — G6: full-population evaluation plan

### B.1 What "full-population" means here
The core domain has **train 2.05M / val 256k / test 256k** interactions and a **heldout-eval pool of 42,684 users**. Most gate results were computed on **5,000-user samples** (warm_eval_sample / cold_eval_sample) with the **6,973-user** precomputed two-lane lists. "Full-population" = re-running the headline metrics at the **largest feasible scale** — at minimum the full precomputed set, ideally regenerated lists across the heldout pool — to confirm the sampled conclusions are stable.

### B.2 What was sampled vs what to expand
| Result | Current scale | Expand to (if feasible) |
|---|---|---|
| C2 90/10 policy metrics (warm/cold R@20) | 5k warm + 5k cold | full precomputed set → heldout pool |
| Creator/catalog reach (distinct items/creators) | H1: 2,400 users | larger user set |
| Warm/cold segment metrics | 5k each | full segments |
| Prevalence-weighted metrics (O1b) | 5k, re-weighted | larger sample, same weighting |
| H1/H2 health metrics | 2,400 / 4,000 users | larger if runtime allows |

### B.3 Exact targets
1. **C2 90/10 policy** — warm R@20/50, cold R@20/50, distinct cold surfaced, warm rel-loss vs ALS, at full scale.
2. **Creator/catalog reach** — distinct items/creators, new-creator count, creator Gini, at full scale.
3. **Warm/cold segment + prevalence-weighted** recall at full scale.
4. **H1/H2 health** — creator Gini / effective creators / cold-share, larger user set if runtime permits (these are the most compute-heavy; expand only if cheap).

### B.4 Expected outputs
- Evidence JSON(s): `outputs/evidence/domain_fullpop_eval_report.json`.
- Metric tables: sampled vs full-scale side-by-side.
- Runtime notes (sandbox: 45s/call, checkpoint if needed).
- **Drift analysis:** delta of each headline metric vs the sampled value.
- A stability verdict: do the C2/O1b/H1 conclusions hold?

### B.5 Pass / hold criteria
- **PASS** if sampled conclusions are **directionally stable** — 90/10 stays within the ≤10% warm guardrail, cold value stays positive, rankings unchanged.
- **HOLD** if the 90/10 guardrail or cold value **materially changes** at scale (e.g., warm loss crosses 10%, or cold recall collapses).
- **HOLD** if data scale/runtime makes the eval unreliable (truncated, timed-out, or non-comparable).

## Part C — Decision matrix
| Question | Decision | Rationale |
|---|---|---|
| **Do G6 now?** | **YES** | Sandbox-feasible, cheap rigor upgrade; directly strengthens the headline claims with scale robustness |
| **Defer G5?** | **YES — documented deferral** | No GPU here; depth story bounded + honest (ALS floor + diagnosed SASRec negative), not closed at T3 level. Documented deferral satisfies claim-honesty; it is not an advanced-model win |
| **Run external GPU G5 later?** | **OPTIONAL** | Only against the A.2 acceptance criteria; append win or clean negative; not a capstone blocker |
| **Proceed to G8 dossier after G6?** | **YES** | Once G6 confirms scale stability, the evidence base is final and the dossier can synthesize it |

**Recommended sequence:** run **G6** (in-sandbox full/larger-scale eval) → confirm stability → **G8 dossier**, carrying G5 as a documented deferral (with the optional external SASRec noted as a future append).

## Claim boundaries
- **No GPU run claimed; no advanced-model win claimed.** SASRec stays an honest negative.
- **G5 deferral is explicit and justified — not deferral disguised as completion.**
- **G6 is a plan here; any numbers will come only from an actual run in the next gate.**

**STOP — awaiting G5/G6 decision review (recommended: approve G6 run next, G5 deferred).**
