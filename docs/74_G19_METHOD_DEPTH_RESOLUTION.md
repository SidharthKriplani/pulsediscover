# 74 — G19 Part B+C: Method-Depth Resolution + Achievement Moments

*Addresses the G18 blocker head-on: "no advanced sequence/graph/two-tower model beat ALS." Is this a gold-blocking weakness, or a senior achievement moment? Honest analysis, no fabrication.*

## Part B — Method-depth resolution

### Is "ALS wins" a weakness or a senior result?
**Mostly a senior result — but not yet a *terminal* one.** Both are true and must be separated:

**Why ALS winning can be a senior finding (strong):**
- **The tournament was serious, not token:** ALS tuned f16→f64; SASRec in three forms (last-position, full-position, **canonical full-softmax over 40k on GPU**); GRU4Rec; a neural two-tower; co-occurrence. Multiple families, real GPU runs, proper objectives.
- **The key confound was tested and ruled out:** the full-softmax experiment showed sampled negatives had crippled the CPU runs (6.7× lift, 0.0068→0.0456) — yet the model *still* lost to ALS. That control is what separates a defended "MF wins" from a lazy one.
- **It matches domain priors:** next-book on a sparse, short-history catalog is weakly sequence-predictable; the dominant signal is collaborative co-read, which MF captures directly. Sequence/graph models need denser sequential signal than this domain offers.
- **The project found where the lever actually was:** candidate coverage + LTR fusion (G13: +44%), not a bigger model. Diagnosing the real bottleneck is the senior move.

**Why it is not yet *terminal* (the honest gap):**
- **Canonical full-softmax SASRec was NOT trained to convergence** — at 40 epochs its loss was still descending (7.53) and R@20 was 0.0456. To assert "MF *terminally* beats it," the strongest competitor must be run to convergence. We stopped it mid-improvement.
- **LightGCN (graph family) was never run** — graph propagation sometimes beats MF on co-occurrence-rich data; the tournament has a graph-shaped hole.
- **Two-tower was item-ID-only** — a content-rich two-tower wasn't tried.

### Were the negatives sufficiently serious?
- **SASRec/full-softmax: serious but not converged** — the objective confound was controlled (good), but convergence wasn't reached (gap).
- **TwoTower: serious enough as a baseline** — built, in-batch negatives, honest negative; a content two-tower is a "nicer" variant, not a gap that blocks.
- **GRU4Rec: serious enough** — recurrent family represented.

### Does the LightGCN deferral still block gold?
**No, by itself** — LightGCN is a *second* family; its absence is a non-terminal hole. The binding hole is the **canonical-SASRec convergence**, because we have *direct evidence the closest competitor was still improving when stopped.* LightGCN is "strongly desirable," canonical-convergence is "necessary for terminal."

### Is "canonical SASRec to convergence" required or optional?
**Required to mark gold_complete.** We cannot claim "MF terminally wins" while the nearest competitor was provably mid-descent. The gap is large (ALS 0.085 vs canonical 0.046 ≈ 1.85×, so convergence very likely still loses) — but gold-grade rigor forbids asserting an unrun result. One bounded GPU run (same canonical config, more epochs to convergence) closes it.

### Can the project reach gold through methodological closure alone (no run)?
**No — not honestly.** Documentation can defend "MF wins among the models I trained" (true, senior) but **not** "MF wins terminally on this catalog" (requires the converged competitor). Declaring gold via documentation would loosen the claim boundary — exactly what we've refused to do. So: methodological closure is *insufficient*; one bounded run is needed.

## Part C — Achievement moments (template format)

**1. Deep models looked attractive, but ALS won (honest negative).**
- *Looked good:* SASRec/two-tower are the "exciting" modern choices; expected to beat MF.
- *Wrong/incomplete:* every variant underperformed ALS (best 0.059 vs 0.085); two-tower near-random.
- *How detected:* same-protocol LLO eval against the ALS floor; GPU runs.
- *Decision:* report as honest negatives + test the sampled-neg confound (full-softmax) rather than quietly drop them.
- *Proves:* I benchmark honestly and don't inflate trendy models.

**2. Feature importance looked high, but marginal contribution was ~zero.**
- *Looked good:* GPU SASRec/two-tower scores ranked 2nd–3rd by LightGBM gain — "the model loves them."
- *Wrong:* removing them barely changed R@20 (+0.0027, within noise).
- *How detected:* ablation (enriched vs local LTR), not importance alone.
- *Decision:* report importance ≠ marginal contribution; don't claim GPU features helped.
- *Proves:* I distinguish correlation/importance from causal/marginal value.

**3. Candidate coverage, not ranking, was the real bottleneck.**
- *Looked good:* LTR doubled single-retriever ranking — looked like the win.
- *Incomplete:* R@20 was capped because 85% of golds weren't in the candidate set (ceiling 0.157).
- *How detected:* measured candidate coverage ceiling and the ranker's fraction of it.
- *Decision:* attack retrieval (G13 widening, +44%) instead of piling on ranker features.
- *Proves:* I find the binding constraint instead of optimizing the visible layer.

**4. Slate governance sounded sophisticated, but upstream candidate design was the better lever.**
- *Looked good:* MMR/creator-cap/combined slate policies look advanced.
- *Wrong:* they cost relevance for little health gain (combined had a 20.7% self-violation rate); pure LTR was already discovery-rich because the candidate stage handled it.
- *How detected:* relevance↔governance tradeoff curve on the test split.
- *Decision:* keep pure LTR default; push discovery upstream / to training-time, not post-hoc.
- *Proves:* I prefer the cheaper correct lever over the flashier one, and reject my own policy when evidence says so.

**5. FAISS looked like an ANN quality-improvement risk, but was correctly scoped as latency only.**
- *Looked good (risk):* easy to present "added FAISS → better recs."
- *Wrong:* ANN can only *retain* recall (exact = identical; approx ≤ exact).
- *How detected:* recall-retention vs brute-force measurement (exact 1.0; IVF 0.872).
- *Decision:* scope FAISS as a latency/serving win; IVF as a scale-demo, not a current-size or quality win.
- *Proves:* I don't dress up a systems optimization as a modeling gain.

**6. A convergence run reported "converged" — but it was a monitoring bug (G20).**
- *Looked good:* the G20 v1 run reported "val-loss plateau → converged" and a tidy "canonical SASRec loses to ALS" verdict that *confirmed my prior*.
- *Wrong:* val loss was all-NaN (2-block masked-attention artifact), which falsely fired early-stop after 6 epochs — while train loss and eval R@20 were both still rising. Not converged.
- *How detected:* read the val_loss_curve (all NaN) and noticed R@20 rose 0.0456→0.0504 during the "converged" window.
- *Decision:* reject the convenient result as invalid (outcome C), keep gold_candidate, and re-engineer the monitor (validation-monitor set + R@20 plateau + exact epoch cap) before re-running.
- *Proves:* I don't bank a number that flatters my hypothesis when the procedure is broken — I trust the protocol over the prior.

## Senior-vs-naive judgment table
| Situation | Naive move | My move |
|---|---|---|
| Deep model loses to MF | bury it / overfit to force a win | report honest negative + test the confound |
| High feature importance | claim the feature "helped" | ablate; importance ≠ marginal |
| Ranker doubles a baseline | declare victory | check the candidate ceiling first |
| Fancy governance policy available | ship the sophisticated one | reject it when the tradeoff curve says pure LTR wins |
| Add an ANN index | claim quality gain | scope it as latency; measure recall retention |
| "MF wins" result | call it terminal to reach gold | run the strongest competitor to convergence before claiming terminal |

**→ Decision in docs/73.**
