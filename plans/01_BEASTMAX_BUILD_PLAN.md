# 01 — PulseDiscover BeastMax Build Plan

*Status: **Phases 0–9 complete (2026-06-16); correction pass applied.** This plan converts PulseDiscover from "gold dossier with `[VISION]` items" into "gold dossier + working evidence spine + (later) a defense PDF grounded in real outputs." T3 extras (LightGCN, DIN, production infra) are explicitly deferred.*

> **Correction note (post-Phase 9):** (a) **No deep model is required to beat baselines** — the binding standard is *implement → CI-evaluate → compare honestly → debug if weak → document* (a win is desirable, not mandatory). (b) **The defense PDF (Phase 10) is deferred** until after the next-build decision (domain dataset vs SASRec-π strengthening vs T3); it is **not** auto-next after the PRD update. See `docs/13_NEXT_BUILD_DECISION_MEMO.md`.

---

## Amendments (Gate 0 approval)

1. **Gate 2 is not a forced win.** The deep model requirement is: *implement, evaluate with CIs, compare honestly, debug if weak, document the result.* Beating baselines is **not** a hard success condition (reworded in §J).
2. **Stack priority (locked):** **SASRec = primary headline deep model (build first).** Retrieval baselines = popularity, co-occurrence, ALS. **Two-tower = build-if-feasible *after* the SASRec path is stable.** **PAL = build-if-feasible, else V2.** **LightGCN, DIN, DLRM, revenue-aware ranking = T3 only.**
3. **`pulserank_platform` = provenance/reference only.** Vendor a read-only copy of its `outputs/evidence/` into `archive/pulserank_evidence/`; port only the minimal reusable OPE (IPS/SNIPS) and retrieval (ALS) *logic* needed. **No messy dependency coupling** — no import from the live repo; copy/adapt the minimal functions into `src/`.
4. **Phase 1 scope (this turn):** dataset feasibility · MovieLens smoke path · Goodreads feasibility · Amazon Books fallback · schema/timestamp/sequence viability · leakage-safe split design · dataset decision report. **No training. No simulator data. No defense PDF. Stop at Gate 1.**

---

## A. Build Objective — what "PulseDiscover BeastMax core" means

A reproducible, public-data-grounded RecSys evidence spine where the **differentiator (honest off-policy evaluation) is real, not designed.** Concretely, BeastMax-core is DONE when all of the following exist as reproducible artifacts:

1. A **real public dataset** ingested with a temporal, leakage-safe split (MovieLens smoke + a domain-aligned dataset).
2. **Retrieval baselines** (popularity, co-occurrence, ALS) with Recall@K + CIs on real data.
3. **At least one deep model** (**SASRec** primary; two-tower if feasible) — **implemented, evaluated with CIs, and compared honestly against baselines. Beating the baselines is desirable, not mandatory**; a weak result is debugged and documented, never forced or fabricated.
4. A **known-propensity semi-synthetic simulator** (the only honest substrate for OPE).
5. **IPS vs SNIPS vs DR** computed against the simulator's ground-truth policy value, with **variance + bias comparison** and **CIs**.
6. **Position-bias correction**: IPS reweighting (build-now) and, if feasible, a **PAL additive tower** (else V2) — with an item-movement plot.
7. A **feedback-loop / catalog-health audit** (exposure concentration before/after; decomposition).
8. **Bootstrap/Wilson CIs on every headline number** (closes the #1 evidence gap).
9. **Evidence JSONs + plots + model/decision cards + seed/version manifest.**
10. An **updated PRD** where `[VISION]` rows become `[BUILT]`/`[BUILT][SYNTHETIC]` with sources.
11. A **defense PDF grounded in the actual outputs** (no claim without an evidence row) — **deferred until after the next-build decision (domain vs SASRec-π vs T3); NOT auto-next after the PRD update.**

**Non-goals for BeastMax-core:** production deployment, real online traffic, LightGCN, DIN/DIEN, DLRM, multi-objective revenue ranking — all T3/later.

**The single sentence of done:** *"On real public data I implement retrieval baselines and a sequential model, evaluate them with CIs and compare honestly (a win is desirable, not required), and on a known-propensity simulator I validate IPS/SNIPS/DM/DR against a known ground truth — where DR is **expected** to reduce variance vs IPS and stay ~unbiased under an imperfect reward model, reported honestly as observed — with every number reproducible from a seed."*

> **Standard for all models (binding):** implement → evaluate with CIs → compare honestly → debug if weak → document the result as-is. No model is required to beat baselines; results are never forced or fabricated.

---

## B. Dataset Strategy

Three tiers, each with an explicit truth boundary. **No dataset is allowed to prove something it can't.**

| Dataset | Role | Proves | Cannot prove | Maps to content rec? | Sequence? | Cold-start? | OPE? | Must be simulated | Never overclaim |
|---|---|---|---|---|---|---|---|---|---|
| **MovieLens-1M** | smoke test | model *correctness* (SASRec/two-tower implemented right; sane Recall/NDCG) | real fiction behavior, real lift | weakly (movies≈items) | yes (timestamps) | partial | no (no propensities) | exposure/propensity | "headline result" — it's a sanity check only |
| **Goodreads (UCSD Book Graph)** | **headline domain dataset** | sequential + cold-start recommendation on a *fiction-adjacent* corpus (books, series, authors) | online lift; Pratilipi-specific behavior | strongly (books≈stories, authors≈creators, series≈serialization) | yes (shelving/read events) | yes (long-tail authors/new books) | no (observational; no logged propensities) | exposure/propensity for OPE | "this is production" / "real OPE on raw Goodreads" |
| **Amazon Reviews (Books)** | fallback headline | same as Goodreads if Goodreads is impractical | same | strongly | yes | yes | no | exposure/propensity | same |
| **Semi-synthetic simulator** (built on top of a real dataset) | **OPE / position-bias test harness** | IPS/SNIPS/DR *estimator validity* against known ground truth; position-debiasing direction | real user taste, real lift | n/a (it's a harness) | reuses real sequences | yes (inject cold items) | **yes — the only thing that can** | n/a (it IS the simulation) | "synthetic results = real-world results" |

**Decisions:**
- **MovieLens-1M is the smoke test** (everyone knows it; fast; validates implementation). It is never a headline.
- **Goodreads is the headline domain dataset** because it is the closest public analog to serialized fiction (authors = creators, series = serialization, shelving = engagement). Fallback to Amazon Books only if Goodreads ingestion proves impractical (size/licensing). **Decide at Phase 1 after a feasibility probe.**
- **The simulator is the OPE harness only** — it sits on top of a real dataset to give *known propensities*; it never stands in for real-world proof. RetailRocket (used in PulseSignal) is e-commerce and is **not** used here.
- **Truth boundary stated everywhere:** real datasets prove model correctness and ranking quality; the simulator proves OPE estimator validity; neither proves online lift.

---

## C. Modeling Strategy — minimum credible stack

| Model | Decision | Why | Impl. complexity | Evidence needed | Interview value | Overreach risk |
|---|---|---|---|---|---|---|
| Popularity | **baseline** | the honest floor (V1 lost to it) | trivial | Recall@K/NDCG + CI | high (judgment) | none |
| Item-item / co-occurrence | **baseline** | cheap explainable CF | low | Recall@K | medium | none |
| **ALS** | **build-now (baseline)** | already built; port to real data | low (exists) | Recall@K + CI on real data | medium | none |
| **Two-tower retrieval** | **build-if-feasible (after SASRec stable)** | scalable, content cold-start, real DL | medium | Recall@K vs ALS + CI | high | medium (negatives done right) |
| **SASRec** | **build-now (PRIMARY headline)** | sequence is the fiction signal; the headline model | medium-high | Recall@K/NDCG vs baselines + CI | very high | medium (tuning, seq length) |
| LambdaMART/LambdaRank | **build-now (tabular ranker)** | already built; strong listwise | low (exists) | NDCG + CI | high | none |
| LightGCN | **V2/T3** | high-order CF on real graph; only after foundation | medium-high | Recall vs two-tower | medium | high (4th paradigm before foundation) |
| BERT4Rec | **discuss-only** | bidirectional, less honest for next-item | — | — | medium | high |
| DIN/DIEN | **V2/T3** | needs rich features; heavier | high | — | medium | high |
| DLRM | **exclude (discuss-only)** | needs feature crosses we won't have | — | — | low | high |

**Minimum credible stack for BeastMax-core (amended priority):** popularity + co-occurrence + ALS (baselines) → **SASRec (primary headline — build first)** → LambdaRank (tabular comparison, port). **Two-tower is build-if-feasible *after* the SASRec path is stable** (it adds retrieval cold-start but is not the floor). **PAL is build-if-feasible, else V2.** **LightGCN / DIN / DLRM / revenue-aware ranking are T3 only.** The floor for BeastMax-core is: baselines + SASRec (implemented, CI-evaluated, honestly compared — not required to win) + the OPE differentiator.

---

## D. OPE & Simulator Strategy — the core differentiator

**Why a simulator at all:** IPS/SNIPS/DR are unbiased only with *known* logging propensities. Public datasets are observational — no propensities — so OPE on them is circular. The simulator manufactures known propensities so the estimators can be validated against ground truth. This is standard practice (Open Bandit Pipeline–style).

**Plan:**
1. **Known logging policy μ:** fit a simple "production" policy on the real dataset (e.g., popularity or a small two-tower), softened to a stochastic policy `μ(a|x)` so every shown item has a recorded probability. Reserve a **5% uniform/ε exploration bucket** with exactly-known probability (guarantees positivity).
2. **Honest position bias:** impose a Position-Based Model examination curve `P(examined|rank)` (declining with rank), parameterized and logged — not hidden.
3. **Logging schema (must-log):** `user_id, item_id, rank/position, propensity μ(a|x), exploration_bucket_flag, examined, click/reward`.
4. **Target policy π:** the new model (e.g., SASRec ranker) whose value we want to estimate offline.
5. **Estimators:**
   - **IPS:** `V̂ = (1/n) Σ (π/μ) r`.
   - **SNIPS:** `Σ(π/μ)r / Σ(π/μ)` (self-normalized; lower variance) + clipping sweep.
   - **DR:** `V̂_DR = DM + (1/p)(r − q̂)` with a fitted reward model `q̂`; **correct formula only**.
6. **Comparison:** compute **ground-truth** `V(π)` directly from the simulator (we know the true reward model), then report **bias** (estimate − truth) and **variance** (across bootstrap/seed resamples) for IPS vs SNIPS vs DR. Expected/target direction: `Var(DR) < Var(IPS)`, DR bias ≈ 0 even when q̂ or μ is slightly off. **No number claimed until run.**
7. **CIs:** bootstrap over logged sessions for each estimator.
8. **Avoiding circularity:** propensities come from the *logging* policy μ (known by construction), never re-estimated from π's own outputs; the exploration bucket guarantees positivity. State this explicitly.
9. **What it proves / doesn't:** proves estimator validity + variance ordering + the "DR predicts online better than NDCG" claim *within the simulator*; does **not** prove real online lift.

---

## E. Position-Bias Strategy

| Technique | Decision | Notes |
|---|---|---|
| Naive "position as a feature" | **baseline (to show it fails)** | include only to demonstrate it doesn't debias in a deep net |
| **IPS reweighting on examination propensity** | **BUILD-NOW** | reweight the click loss by `1/P(examined|rank)`; propensities from the simulator |
| **PAL additive examination tower** | **BUILD-NOW IF FEASIBLE, else V2** | `logit = f(x) + g(position)`, drop `g` at serving; flagged V2 if the model-training budget is tight. **Will NOT be claimed as built unless implemented.** |
| Item-movement plot | **BUILD-NOW** | which items rise/fall after debiasing |
| Examination-curve plot | **BUILD-NOW** | the imposed PBM curve + recovered estimate |

**Explicit:** PAL is **build-now-if-feasible**; if deferred it is labeled `[V2]` and IPS reweighting carries the position-bias story alone. No PAL claim without code.

---

## F. Feedback-Loop / Catalog-Health Strategy

Start simple, escalate only if feasible:
1. **Exposure concentration (build-now):** creator/item exposure distribution; **Gini + entropy**; long-tail share; head-concentration over simulated rounds. (Gini already exists in PulseRank — port it.)
2. **Popularity ratchet demo (build-now):** run the recommender for N rounds feeding its own outputs back; show exposure concentration rising — the loop made visible.
3. **Before/after exposure distribution (build-now):** plot exposure at round 0 vs round N.
4. **Decomposition (escalate):** start with a **simple mix/rate exposure decomposition**; only add the **Shapley** cross-term allocation if the simpler version is solid. Report the cross-term as a *confidence indicator*, with the honest caveat that mix/rate aren't cleanly separable when one policy sets both.
5. **Break-the-loop demo (stretch):** retrain with IPS-corrected training and show concentration drop.

**Decision:** build exposure concentration + ratchet + before/after now; Shapley decomposition is **build-if-feasible, else simple decomposition first**. No "Shapley feedback-loop audit" claim unless implemented.

---

## G. Evidence Plan — exact artifacts

Every phase emits reproducible artifacts (seed-pinned). Target files:

**`outputs/evidence/`**
- `data_manifest.json` — dataset, version, split, sizes, seed.
- `retrieval_baselines.json` — popularity/co-occ/ALS Recall@K + bootstrap CI.
- `two_tower_report.json` — Recall@K vs baselines + CI + model card fields.
- `sasrec_report.json` — Recall@K/NDCG@K + CI + arch (heads/layers/max_seq_len).
- `lambdarank_report.json` — NDCG + CI (real-data port).
- `simulator_manifest.json` — μ policy, PBM curve params, exploration ε, propensity logging schema.
- `ope_comparison.json` — IPS/SNIPS/DR estimates, ground-truth value, bias, variance, CIs, clipping sweep.
- `position_bias_report.json` — naive vs IPS (vs PAL if built); item-movement deltas.
- `feedback_loop_report.json` — Gini/entropy over rounds; before/after; decomposition.
- `ci_summary.json` — every headline metric with N + 95% CI + method.
- `run_log.json` + `seed_version_manifest.json` — provenance.

**`outputs/plots/`**
- `recall_at_k.png`, `ope_variance.png` (IPS vs SNIPS vs DR), `examination_curve.png`, `item_movement.png`, `exposure_gini_over_rounds.png`, `before_after_exposure.png`, `calibration_of_ope.png`.

**`reports/`** — `model_card_*.md`, `decision_card_promote_or_hold.md`, `run_report.md`.

**Rule:** no number enters the PRD or defense PDF unless it has a row here with tag + N + CI + source.

---

## H. Defense PDF Plan — `defense/PulseDiscover_Defense.pdf`

**Do not generate until build evidence exists.** Outline (built from the gold PRD §10 + actual outputs):
1. Product thesis + the one insight.
2. Architecture + data flow.
3. Dataset choices + truth boundaries.
4. Technique tournaments (retrieval/ranking/OPE/debiasing).
5. Derivations by importance — full: IPS, SNIPS, **DR** (both robustness cases), SASRec; sketch: ALS, LambdaMART, two-tower, PAL, BM25-n/a.
6. **OPE proof** — simulator design + IPS/SNIPS/DR bias-variance table (from `ope_comparison.json`).
7. **DR proof** — derivation + the variance plot.
8. SASRec/two-tower explanation + real Recall/NDCG.
9. PAL vs IPS (or "IPS-only, PAL V2" if deferred).
10. Failure modes + feedback-loop evidence.
11. Evidence tables (claim · number · tag · N · CI · source).
12. Safe interview answers; never-say / say-instead boundaries.

Source format: build in Markdown → render to PDF via the `pdf` skill at the end. Every figure references an `outputs/` artifact.

---

## I. Implementation Phases

| Phase | Files to create | Expected outputs | Acceptance criteria | Risks | Fallback |
|---|---|---|---|---|---|
| **0 — workspace + audit** (this turn) | workspace, MANIFEST, audit, plans | folder tree + docs | plan approved | scope creep | — |
| **1 — data ingestion + MovieLens smoke** | `src/data/`, `notebooks/01_smoke.ipynb` | `data_manifest.json`; temporal split | MovieLens loads; leakage-safe split; sane stats | Goodreads too big/awkward | fall back to Amazon Books; MovieLens-only smoke if needed |
| **2 — retrieval baselines** | `src/retrieval/` | `retrieval_baselines.json` + Recall plot | popularity/co-occ/ALS Recall@K with CIs on real data | ALS port issues | reuse PulseRank ALS code directly |
| **3 — sequence model / two-tower** | `src/models/sasrec.py`, `two_tower.py` | `sasrec_report.json`, `two_tower_report.json` | SASRec implemented + CI-evaluated + honestly compared vs baselines (**win not required**) + documented | training time/compute | SASRec-first; two-tower V2; use free Colab/Kaggle GPU |
| **4 — known-propensity simulator** | `src/sim/` | `simulator_manifest.json` + logged events | propensities logged; positivity via ε bucket; PBM curve recovers | simulator too clean/circular | simplest PBM first; document realism audit |
| **5 — IPS/SNIPS/DR** | `src/ope/` | `ope_comparison.json` + variance plot | DR computed (correct formula); bias/variance vs ground truth; CIs | DR/q̂ instability | report IPS/SNIPS solidly even if DR needs iteration |
| **6 — position-bias correction** | `src/debias/` | `position_bias_report.json` + item-movement plot | IPS reweighting works; (PAL if feasible) | PAL training cost | PAL → V2; IPS-only carries the story |
| **7 — feedback-loop audit** | `src/audit/` | `feedback_loop_report.json` + exposure plots | Gini/entropy over rounds; before/after | Shapley complexity | simple mix/rate decomposition first |
| **8 — evidence JSONs + plots + CIs** | `src/eval/ci.py` | `ci_summary.json`, all plots | every headline number has N + 95% CI | — | bootstrap if Wilson inapplicable |
| **9 — PRD update** | `docs/PRD_PulseDiscover_v2.md` | updated PRD | `[VISION]`→`[BUILT]` with sources; self-score re-run | — | keep unbuilt items tagged |
| **10 — defense PDF** | `defense/` | `PulseDiscover_Defense.pdf` | every claim has an evidence row | — | ship without PAL section if deferred |

---

## J. Go/No-Go Checkpoints

- **Gate 0 (now):** plan approved by reviewer → proceed to Phase 1. **← we are here.**
- **Gate 1 (after Phase 1):** dataset decision locked (Goodreads vs Amazon Books) + leakage-safe split verified → proceed to modeling.
- **Gate 2 (after Phase 3) — amended, NOT a forced win:** SASRec is *implemented, evaluated with CIs, and compared honestly against baselines.* A win is **not** required. If the model is weak, debug and **document the result as-is**. Proceed to OPE once the model + honest comparison + CIs exist. Never fabricate or force a win.
- **Gate 3 (after Phase 5):** DR computed with correct formula and a defensible bias/variance story → proceed to debiasing/audit. *If DR unstable:* ship IPS/SNIPS honestly, mark DR `[BUILD-TASK]`, do not claim.
- **Gate 4 (after Phase 8):** every headline number carries N + CI → proceed to PRD update + PDF.
- **Gate 5 (after Phase 10):** defense PDF has zero claims without an evidence row → BeastMax-core DONE; convergence self-score re-run targeting 9+.

**No phase may invent results. Any failed phase falls back to the honest, lesser claim rather than a fabricated number.**

---

## Awaiting approval before implementation.
