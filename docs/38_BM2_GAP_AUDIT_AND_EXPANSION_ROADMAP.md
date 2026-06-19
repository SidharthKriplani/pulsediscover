# 38 — BM2-GAP: PulseDiscover BeastMax Phase-2 Gap Audit & Expansion Roadmap

*Framing: V1 packaging (FINALPACK, docs/37) is a **baseline checkpoint, not final closure.** This audit defines the distance from V1 to the **BeastMax / T3 end-state** — research-grade, company-realistic, T3-depth, defensible as a serious experimentation system, and reusable as a golden template. Strategic audit only — no new experiment, no code, no PDF.*

---

## 1. Current status
- **V1 package checkpoint: COMPLETE.** 11 evidence gates + 7 packaging artifacts, internally consistent, claim-bounded.
- **BeastMax / T3 end-state: NOT COMPLETE.** Missing the depth layers that make a project *research-grade* and *reusable as a template*.

**What V1 already proves:**
- A real domain RecSys with a tuned warm floor (ALS f64) and a working cold lane (content-hybrid).
- A guardrail-respecting two-lane policy (90/10) framed as an A/B candidate.
- A genuinely strong governance layer (OPE · positivity · prevalence · position-bias · catalog-health) — the differentiator.
- Disciplined honesty: documented negatives, bounded claims, stop-control.

**What V1 does NOT yet prove:**
- Depth beyond MF + content (no competitive sequence/GNN/two-tower result).
- **Engineering realism** — cost, latency, infra, serving footprint (entirely absent).
- **Product/growth realism** — who this helps, the business case, rollout plan beyond a spec.
- **Risk realism** — a structured risk register (feedback loops, fairness, data drift).
- **Research-dossier coherence** — the gates exist but aren't synthesized into one defensible narrative document.
- **Reusable template** — the method that produced this isn't extracted for reuse.
- **Visual/PDF assets** — nothing presentation-ready.
- **Statistical robustness at scale** — eval is on 5k-user samples, not full population.

## 2. Gold-standard target — what "BeastMax PulseDiscover" contains
1. **Research dossier depth** — one coherent, citation-style document synthesizing problem → method → evidence → limits → decision.
2. **Model tournament depth** — at least one *competitive* advanced model (canonical SASRec or two-tower) benchmarked honestly against the floor.
3. **Company realism** — the project reads as an internal experimentation program a real team would run.
4. **Growth/product strategy** — the business case: who benefits, the discovery flywheel, success metrics.
5. **Cost / latency / infra tradeoffs** — serving footprint, candidate-gen cost, online scoring budget, build-vs-buy.
6. **A/B & experimentation realism** — a formal rollout doc (ramp, power, duration, interleaving, monitoring).
7. **Risk register** — feedback-loop amplification, fairness, drift, positivity failure, gaming.
8. **Final PDF/deck/README package** — presentation-ready assets.
9. **Reusable golden template** — the gated-experimentation method extracted so future lanes reuse it.

## 3. Gap audit (current vs target)
| # | Category | Score | Gap | Verdict |
|---|---|---|---|---|
| 1 | Retrieval / model depth | 6/10 | Strong MF+content; no competitive advanced model | **High-value upgrade** |
| 2 | Cold-start system depth | 8/10 | Working lane + coverage; lacks freshness-decay / time-aware cold | Optional T3 |
| 3 | OPE / measurement rigor | 9.5/10 | Near research-grade; only variance-reduction (SNIPS/clip/PAL) missing | Optional T3 |
| 4 | Company realism | 8.5/10 | Strong gate framing; missing growth + infra context | **High-value upgrade** |
| 5 | Experiment design | 7.5/10 | AB1 spec solid; no formal rollout/power/ramp doc | **High-value upgrade** |
| 6 | Product / growth realism | 3/10 | Essentially absent | **Blocker for flagship** |
| 7 | Infra / cost / latency realism | 2/10 | Absent (only a per-user ms note in H2) | **Blocker for flagship** |
| 8 | Creator / catalog health realism | 8/10 | Reach + amplification shown; fairness fix not built | Optional T3 |
| 9 | Documentation / defense quality | 9/10 | Excellent; needs the dossier synthesis | **High-value upgrade** |
| 10 | Visual / PDF readiness | 3/10 | One Mermaid spec; no deck/PDF | **High-value upgrade** |
| 11 | Portfolio distinctiveness | 9/10 | Governance arc is distinctive; depth+infra would cement it | Maintain |

**Reading:** the *evidence/governance* half is near-elite (8–9.5); the *product/infra/presentation* half is the real gap (2–3). **The blockers are growth realism and infra/cost realism — not another model.** A model upgrade is high-value but secondary to making the project read like a real product decision with real serving constraints.

## 4. Ranked expansion roadmap (checkpoint → BeastMax endpoint)

| Step | Company question | Build / write | Artifact | Why it matters | Stop/continue |
|---|---|---|---|---|---|
| **G1 — Infra / cost / latency analysis** | Can we actually serve this two-lane policy, and what does it cost? | Candidate-gen + online scoring cost model; latency budget per lane; index/footprint; build-vs-buy | `docs/39` infra memo | Closes the #7 blocker; makes it a real system | continue (essential) |
| **G2 — Product / growth strategy memo** | Who benefits, and what's the discovery flywheel + success metric? | Stakeholder map, discovery flywheel, north-star + guardrail metrics, rollout narrative | `docs/40` growth memo | Closes the #6 blocker; gives the business spine | continue (essential) |
| **G3 — Risk register** | What can go wrong and how do we detect/mitigate it? | Feedback amplification, fairness, drift, positivity failure, gaming — likelihood/impact/mitigation | `docs/41` risk register | Senior-signal; ties H1/O1 risks together | continue (high ROI) |
| **G4 — Formal A/B rollout doc** | Exactly how do we run the experiment? | Power/MDE, ramp schedule, duration, interleaving, monitoring, stop rules | `docs/42` rollout doc | Turns AB1 spec into a runnable plan | continue (high ROI) |
| **G5 — One competitive advanced model** | Can a proper sequence/two-tower model beat the ALS floor? | Canonical full-position SASRec **or** two-tower (GPU/external), benchmarked vs f64 | `docs/43` model gate + evidence | Closes the #1 depth gap honestly (win or documented negative) | continue ONLY if a real GPU run is feasible; else mark deferred and skip |
| **G6 — Full-population evaluation** | Do the headline numbers hold beyond 5k samples? | Re-run core metrics on full eval population | `docs/44` eval-robustness note | Cheap rigor upgrade | continue (high ROI if cheap) |
| **G7 — Variance reduction (SNIPS/clip/PAL)** | Can we tighten the per-item OPE/position estimates? | Add SNIPS/clipping/PAL to P1/O1 estimators | extend P1/O1 evidence | Marginal; only if measurement is the headline | T3 optional |
| **G8 — Research dossier** | Is there one document that defends the whole thing? | Synthesize all gates into a single research-grade narrative | `docs/45` dossier | The capstone reviewers actually read | continue (essential) |
| **G9 — Final PDF / deck** | Is it presentation-ready? | Convert dossier + claim table + architecture into PDF/deck | PDF + deck | Presentation surface | continue (essential, last) |
| **G10 — Golden-template extraction** | Can future lanes reuse this method? | Extract the gated-experimentation + claim-discipline method as a template | `docs/46` golden template | Compounding portfolio value | continue (high ROI, final) |

## 5. Anti-random-overbuild filter
| Step | Tag |
|---|---|
| G1 Infra/cost/latency | **Essential** |
| G2 Product/growth | **Essential** |
| G3 Risk register | **High ROI** |
| G4 A/B rollout doc | **High ROI** |
| G5 One advanced model (SASRec OR two-tower) | **High ROI if GPU feasible, else T3 optional** |
| G6 Full-population eval | **High ROI** |
| G7 SNIPS/clip/PAL | **T3 optional** |
| G8 Research dossier | **Essential** |
| G9 Final PDF/deck | **Essential** |
| G10 Golden template | **High ROI** |
| f64-hybrid / ALS f128 | **T3 optional** |
| LightGCN / BERT4Rec / multiple new models | **Reject** (depth ≠ model count; one competitive model suffices) |
| Training-time creator fairness | **T3 optional** (real fix, but large; only if fairness becomes the headline) |
| Production API / dashboard | **Reject for flagship** (polish, not evidence) |

**Rule enforced:** no model added merely to add a model. G5 caps advanced modeling at **one** competitive entry; everything else in the modeling family is T3 or reject.

## 6. Recommended next 3 moves
1. **G1 — Infra / cost / latency analysis** (`docs/39`). Closes the hardest blocker and is pure analysis (no GPU). Makes PulseDiscover read as a serveable system.
2. **G2 — Product / growth strategy memo** (`docs/40`). Closes the second blocker and supplies the business spine the governance work currently lacks.
3. **G3 — Risk register** (`docs/41`). Cheap, high senior-signal, and unifies the feedback-loop/positivity/fairness risks already surfaced in H1/O1/H2.

*(These three are all writing/analysis — they take PulseDiscover from "rigorous experiment" to "rigorous product decision" with zero new modeling, and they de-risk whether G5's GPU model is even worth doing.)*

## 7. Final BeastMax stop condition
PulseDiscover is **BeastMax-complete** only when **all** of these are coherent and mutually consistent:
1. **Research dossier** (G8) synthesizes the full story.
2. **Company-realism layer** exists — infra/cost (G1), product/growth (G2), risk (G3), A/B rollout (G4).
3. **Selected T3 depth** is satisfied — either one competitive advanced model (G5) *or* an explicit, defensible deferral with the reason documented.
4. **Final defense/PDF assets** (G9) are presentation-ready.
5. **Golden template** (G10) is extracted for reuse.

Until all five hold, the lane stays open. Once they do, **stop** — further models/dashboards are explicitly out of scope. *Ambitious but controlled: depth where it strengthens the flagship, hard stop where it would only pad it.*

**STOP — awaiting BM2-GAP review and selection of the next step (recommended: G1).**
