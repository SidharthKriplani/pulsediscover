# PulseDiscover — Final Canonical PRD
### Content Recommendation / RecSys · PulseRank as the ranking + off-policy-evaluation engine
*RiskFrame gold-standard depth. Source of truth: the PulseOS architecture review + direct code audit of `pulserank_platform`. Philosophy: data is the gospel, technology is the guidance system.*

**Claim tags:** `[BUILT]` verified in code/outputs · `[SYNTHETIC]` real code on seeded/synthetic data · `[PLACEHOLDER]` number written but not reproducibly computed · `[FABRICATED]` hardcoded/self-fulfilling (must die) · `[VISION]` target architecture, not built · `[BUILD-TASK]` ordered work · `[PARKED]` · `[DELETE]`.

> **Honesty preamble (read first).** The shipped repo today is **`pulserank_platform`** — a real *ranking + offline-evaluation harness* on a **synthetic e-commerce corpus** (sellers, GMV, returns). It is `[BUILT][SYNTHETIC]`. **PulseDiscover** is the *content-domain* recommender that **absorbs PulseRank's evaluation engine** and re-homes it in serialized fiction (the Pratilipi domain). Most of PulseDiscover below is `[VISION]`/`[BUILD-TASK]`; PulseRank's evaluation engine is the `[BUILT]` core it inherits. I never retro-claim the e-commerce repo as a fiction recommender, and I never attach a number to anything unbuilt.

---

## 1. North-Star Thesis + One-Line Identity

**Identity:** *PulseDiscover is a sequential content recommender whose defining feature is not the model — it is the honesty of the evaluation.* It retrieves with a two-tower + content cold-start channel, ranks reading sequences with a self-attentive transformer, and then refuses to trust its own offline numbers until they survive doubly-robust off-policy correction, position-bias correction, and a feedback-loop audit.

**North-star metric:** **Long-horizon reader value per session, governed across three time horizons** — immediate engagement (click/open), satisfaction (length-normalized completion depth), and catalog health (creator-exposure equity over 30 days). A recommender that wins horizon 1 while silently losing horizon 3 is the failure this product exists to prevent.

**One line:** "I built a sequential recommender and then spent most of the effort proving its offline metrics actually predict online value — via doubly-robust evaluation on a simulator with known propensities, position-bias correction done two ways, and a feedback-loop decomposition."

---

## 2. Target Buyer / JD Archetype

- **Primary:** Senior/Staff DS or ML Engineer on **content-recommendation / discovery / feed-ranking** teams — Pratilipi, ShareChat, Glance/InMobi, Spotify, YouTube, Netflix, Meta, Hotstar, Audible. JD keywords: *learning-to-rank, candidate generation, sequential recommendation, off-policy evaluation, position bias, exploration, feedback loops.*
- **Secondary:** Marketplace/search-ranking roles (Flipkart, Amazon, Swiggy, Zomato, Uber) where ranking + counterfactual evaluation is the daily job.
- **The exact archetype this is built to beat:** the candidate who says "I built two-tower + DIN on MovieLens, Recall@10 was 0.12." PulseDiscover's answer to "tell me about a RecSys project" is a Staff-level answer because it leads with *evaluation validity*, not architecture.

---

## 3. Why This Combination Matters in the Current Market (2026)

Every content platform has the same unsolved-in-practice problem: **offline metrics don't predict online outcomes, and the recommender corrupts its own future training data.** Hiring teams at discovery companies are flooded with portfolios that train a two-tower and report NDCG. Almost none can answer "is that NDCG biased by the logging policy?" or "what does your model do to creator diversity in six months?" Off-policy evaluation (IPS/SNIPS/DR) and feedback-loop governance are exactly the skills that separate a Staff DS from a strong junior — they are *rare in Indian DS portfolios* and *daily reality* at the companies Avinash wants. The market is also shifting from "best model" to "trustworthy decision system," which is the entire PulseDiscover thesis.

---

## 4. The One Insight Nobody Else Brings

**"Your offline ranking metric is a measurement of your old logging policy, not of your new model — and if you don't correct for that, you will ship the model that best exploits past exposure, not the model that best serves readers."**

PulseRank already discovered this concretely and honestly: under naive evaluation the ranker looked one way; under IPS-corrected evaluation the picture changed, and the repo's own `INTERVIEW_SAFE_CLAIMS.md` forbids framing the two frames as a "0.134 → 0.522 improvement" because **they are different evaluation frames on different session subsets, not a before/after of the same system** `[BUILT][SYNTHETIC]`. That self-correction *is* the insight. PulseDiscover generalizes it: it treats every offline number as guilty until proven unbiased.

---

## 5. Layer 0 — Foundational Assumption (and When It Breaks)

**Assumption: the Position-Based click/read Model (PBM).** Observed engagement = `P(examined | position) × P(relevant | user, item)`. Examination depends only on position (and surface), relevance only on the user-item match; the two are independent given position.

**Why it's valid here:** it makes position-bias correction and off-policy evaluation *identifiable* — you can factor exposure out of the signal. It is the standard assumption behind unbiased LTR (Joachims 2017) and PAL.

**When it breaks:** (1) **trust/brand bias** — readers examine known creators regardless of position, violating position-only examination; (2) **serial dependence** — in serialized fiction, "next chapter" examination is driven by narrative continuation, not slot position, so PBM under-explains within-series clicks; (3) **multi-surface** — examination on an infinite-scroll feed differs from a fixed grid. **Mitigation:** use a cascade or a position-and-surface examination model as a V2 robustness check; state PBM as the working assumption and the conditions under which the OPE numbers degrade.

---

## 6. Component Map

| # | Component | Role | Status |
|---|---|---|---|
| C1 | **Two-tower retrieval** | user/item embeddings, ANN (FAISS) candidate generation | `[VISION/BUILD-TASK]` |
| C2 | **Cold-start content channel** | item-metadata → embedding; reserved exploration slots | `[VISION/BUILD-TASK]` |
| C3 | **ALS / co-occurrence channels** | collaborative baselines, hybrid candidate merge | `[BUILT][SYNTHETIC]` (from PulseRank) |
| C4 | **SASRec sequential ranker** | self-attentive transformer over reading sequence | `[VISION/BUILD-TASK]` (primary DL model) |
| C5 | **PulseRank LambdaRank ranker** | gradient-boosted listwise ranker + feature pipeline | `[BUILT][SYNTHETIC]` |
| C6 | **PulseRank OPE engine** | IPS / SNIPS / **DR** off-policy evaluation + clipping sweep | IPS/SNIPS `[BUILT][SYNTHETIC]`; **DR** `[BUILD-TASK]` |
| C7 | **Position-bias correction** | Route A: IPS on examination propensity · Route B: PAL additive tower | `[VISION/BUILD-TASK]` |
| C8 | **Multi-objective head + governed reranker** | engagement/completion/diversity; MMR + constrained selection | MMR `[BUILT][SYNTHETIC]`; multi-objective `[BUILD-TASK]` |
| C9 | **Feedback-loop auditor** | 30-day creator-exposure mix/rate/Shapley decomposition | `[BUILD-TASK]` (reuses MetricLens) |
| C10 | **Semi-synthetic OPE simulator** | known propensities, PBM curve, ε-exploration (OBP-style) | `[BUILD-TASK]` |
| C11 | **LightGCN** | graph CF on real user-item graph | `[VISION — V2]` |
| C12 | **Drift + offline A/B replay** | KL/PSI monitoring, guardrail-first SHIP/HOLD | `[BUILT][SYNTHETIC]` |

---

## 7. Data Flow — Each Component's Output → Next Component's Input

1. **Reading logs** (user, story, position, completion_depth, timestamp, logged exposure probability) → **C1/C2/C3** produce a candidate set (top-K via ANN + content cold-start slots + collaborative channels), merged & deduped → **candidate list with retrieval scores**.
2. Candidate list + **user reading sequence** → **C4 SASRec** emits sequence-aware relevance; in parallel **C5 LambdaRank** emits a listwise score from engineered features → **per-candidate scores**.
3. Scores + **position feature** → **C7** removes position bias (Route A reweights the loss using examination propensity; Route B's PAL tower is dropped at serving) → **debiased relevance**.
4. Debiased relevance → **C8** predicts engagement/completion/diversity heads → **constrained selection** (maximize engagement s.t. completion ≥ τ₁, creator-diversity ≥ τ₂) → **final top-N ranking**.
5. Ranking shown; **interaction logged with its propensity** → feeds **C6 OPE**: `1/propensity` weights IPS/SNIPS; the reward model q̂ + IPS residual feed **DR**. The **logged propensity comes from C10's known logging policy** (this is what makes OPE valid).
6. Daily/30-day exposure distribution → **C9** decomposes creator-exposure shift into preference-change vs **exposure-amplification** vs cross (Shapley) → if amplification dominates, trigger **IPS-corrected retraining**.
7. **C12** monitors KL/PSI drift and runs offline A/B replay → emits **SHIP / HOLD / INVALIDATE** with guardrail-first logic.

**The load-bearing arrow:** C10's *known propensity* → C6's *unbiased estimate*. Without it, every downstream number is a measurement of the old policy.

---

## 8. Premium-Skill Coverage

Recommender systems (S) · Sequential modeling / transformers — SASRec (S) · Deep learning — two-tower + SASRec (S) · Off-policy / counterfactual evaluation — IPS/SNIPS/DR (S, the differentiator) · Position-bias correction — PAL vs IPS (S) · Search/IR — ANN candidate generation (s) · Causal inference — DR + feedback-loop decomposition (S) · GraphML — LightGCN (V2) · Monitoring/governance — drift, guardrail-first A/B (s) · Multi-objective optimization — constrained ranking (s). **CNN: excluded (no honest fit). Dynamic pricing: parked (PricePulse).**

---

## 9. Data Layer — Signals, Labels, Datasets, Real-vs-Synthetic Boundary, Cold Start

**Signals (ranked by quality):** reading *sequence* (strongest on a serial platform) > completion depth (length-normalized) > explicit rating/like > dwell > click position (bias source, not pure signal) > genre/affinity > recency. **Negative sampling:** in-batch negatives for two-tower; popularity-corrected sampled softmax to avoid the model just learning popularity.

**Label construction — where it can be wrong:** click is exposure-confounded (position); completion confounds with chapter length and prior fandom — **length-normalize completion** and treat it as the satisfaction label, with click as the engagement label. State that label choice changes both density and bias.

**Datasets — the three-tier strategy (each claim tagged by tier):**
- **Tier 1 — MovieLens-1M:** smoke test only. Validates SASRec/two-tower correctness. **Never a headline number.** `[BUILD-TASK]`
- **Tier 2 — Goodreads (UCSD Book Graph):** the real, domain-relevant headline dataset — books, **series (sequential reading)**, **authors (long-tail creators)**, ratings, shelving events. Closest public analog to serialized fiction. Real held-out interactions = real labels, no circularity. (Fallback: Amazon Reviews "Books".) `[BUILD-TASK]`
- **Tier 3 — Semi-synthetic OPE simulator (OBP-style):** the *only* place OPE validity can be honestly shown, because **public datasets have no known logging propensities**. Take Tier-2 data, fit a logging policy, impose a PBM examination curve + ε-exploration, simulate clicks → now propensities are known and IPS/DR can be checked against ground truth. `[BUILD-TASK]`
- **Optional Pratilipi-flavored synthetic generator:** users w/ genre affinity, stories w/ chapters, Zipf popularity skew, long-tail creators, PBM position curve, ε-exploration bucket (this is what makes propensities *known*), repeat reading, completion depth, cold-start items, feedback-loop amplification knob. Every number `[SYNTHETIC]`.

**Real-vs-synthetic boundary (stated in interviews):** "Model *correctness* is shown on real data (MovieLens, Goodreads). OPE *validity* is shown on a semi-synthetic simulator with known propensities. I do **not** claim unbiased off-policy evaluation on raw observational data — nobody can, because the propensities aren't known."

**Cold start (two phases, both addressed):** *retrieval* cold start — new stories with no interactions get a **content-based embedding (C2)** and reserved **exploration slots**, because you cannot explore what retrieval never surfaces; *ranking* cold start — ε-greedy/UCB exposure budget until enough signal accrues. Honest note: full Thompson sampling needs a posterior the point-estimate ranker lacks — either add MC-dropout/ensemble or use ε-greedy/UCB and say so.

---

## 10. Model / Algorithm Layer — Alternatives Table

**Retrieval / candidate generation:**

| Approach | Recall (synthetic, PulseRank) | Cold start | Cost | Verdict |
|---|---|---|---|---|
| Popularity | Recall@100 0.587 `[BUILT][SYNTHETIC]` | none | trivial | baseline / floor |
| ALS (32 factors, from scratch) | Recall@100 0.347 `[BUILT][SYNTHETIC]` | poor | low | collaborative baseline |
| Item-item co-occurrence (PMI) | Recall@100 0.04 `[BUILT][SYNTHETIC]` | poor | low | weak alone |
| Content heuristic | Recall@100 0.647 `[BUILT][SYNTHETIC]` | **good** | low | cold-start channel |
| Hybrid V2 (merged) | Recall@100 0.627 `[BUILT][SYNTHETIC]` | good | med | shipped merge |
| **Two-tower (DL)** | *target > hybrid; no number until built* `[VISION]` | good w/ content tower | med | **primary retrieval** |

**Ranking model:**

| Approach | Metric | Why win / lose |
|---|---|---|
| Popularity sort | NDCG@10 0.177 `[BUILT][SYNTHETIC]` | the bar V1 *failed* to clear |
| V1 heuristic ranker | NDCG@10 0.136 `[BUILT][SYNTHETIC]` | **lost to popularity** — the honest origin story |
| **LightGBM LambdaRank (V2)** | NDCG@10 0.192 (+8.6% vs pop) `[BUILT][SYNTHETIC]` | real listwise trained ranker; PulseRank's shipped ranker |
| **SASRec (transformer)** | *target > LambdaRank on sequential data; no number until built* `[VISION]` | sequence is the strongest signal on serial fiction; primary DL ranker |
| DIN/DIEN | — | `[VISION — V2]` harder, less transferable than SASRec |
| DLRM | — | discuss-only; needs rich feature crosses you won't have |

**Why SASRec over DIN as primary:** unidirectional self-attention over the reading sequence is the strongest, most defensible signal for chapter→chapter→next-story behavior, simpler to implement and explain ("BERT for reading order"), trains on free GPU at this scale. DIN's attention-over-history-vs-candidate is harder and less transferable → V2.

---

## 11. Evaluation Layer — Metrics, Offline-Online Gap, Leakage, N, CI, Validity

- **Metrics:** NDCG@10 / Recall@K (ranking quality), **IPS/SNIPS/DR off-policy value** (unbiased policy value), per-position examination curves, creator-exposure Gini/entropy (catalog health). NDCG over MAP because graded relevance (completion depth) matters; off-policy value because NDCG alone measures the old policy.
- **The offline-online gap — the centerpiece:** vanilla offline NDCG measures the logging policy. The bridge is **DR** on the Tier-3 simulator: show `Var(DR) < Var(IPS)` and that **DR-ranked offline value predicts simulated online lift better than raw NDCG**. Correct estimator (the brief's earlier formula was wrong): **`V_DR = DM + (1/p)·(r − q̂)`** — direct-method estimate plus IPS-weighted residual of the reward model; unbiased if *either* p or q̂ is correct.
- **Leakage controls (already enforced in PulseRank):** `display_rank` excluded as a training feature (leakage control); temporal split enforced at the data level `[BUILT]`. State both — they are exactly the "did you leak position into the ranker?" answer.
- **N / CI discipline:** every offline number reported with N (sessions/queries) and a bootstrap/Wilson CI. Synthetic corpus N today: **4,132 sessions / 41,320 impressions** `[BUILT][SYNTHETIC]`. For Goodreads, report held-out N and CIs.
- **Known honest negatives (do not hide — they are the credibility):** IPS bias_delta is **−0.0053**, i.e. IPS shows a *smaller* gain than raw NDCG, not an inflated one `[BUILT][SYNTHETIC]`; ALS scores are **0.0 in the ranker's training features** (retrieval and ranker are not yet wired) `[BUILT]`; content embeddings are **8-dim random**, capturing collaborative not semantic signal `[BUILT][SYNTHETIC]`.

---

## 12. Production-Behavior Layer — Monitoring, Drift, Regression Gates, Feedback Loops

- **Monitoring:** KL-divergence + PSI histograms on score/feature distributions `[BUILT][SYNTHETIC]` (`monitoring/drift_report.py`). Leading signal = retrieval recall drop; lagging = engagement drop; name the detection gap between them.
- **Regression gate (offline A/B replay):** guardrail-first SHIP/HOLD. Real demonstrated decision: offline replay showed **+12.6% net revenue/session but +9.09pp return rate → HOLD_SIMULATED** `[BUILT][SYNTHETIC]`. The product *shipped a HOLD* — that is the governance point. For fiction, the analog guardrail is completion-without-regret / unsubscribe rate.
- **Feedback loop (the 12-month risk made operational):** C9 decomposes 30-day creator-exposure shift into preference-change vs **exposure-amplification** vs cross via Shapley. If amplification > threshold, retrain with IPS-corrected training to break the loop. **Honest caveat:** mix and rate are not cleanly separable when one policy sets both exposure and engagement; report descriptively under a stated reference, with the cross/Shapley term as a confidence indicator, not causal attribution.
- **Retraining cadence:** triggered (drift/amplification), not fixed-clock; justify by staleness cost.

---

## 13. Business Layer — Unit Economics, Cost of Not Having, Stakeholder Translation

**Unit economics (illustrative ₹ at 3 scales, clearly modeled not measured):** serving cost is dominated by ANN retrieval + a batched neural forward pass. At 100K / 1M / 10M ranking requests/day, batched GPU inference keeps per-request cost in the sub-paise range; the real cost is embedding refresh + FAISS rebuild (hours at 100K items, a partitioned/incremental job at 10M). **Break-even framing:** a 1% lift in length-normalized completion on a platform monetizing reading time pays for the entire serving stack many times over; the number to put in front of a PM is "extra readers retained per 1,000 sessions," not "ΔNDCG."

**Cost of NOT having each technique:**
- No **DR/IPS** → you ship the model that best exploits past exposure; popular-item monopolization; offline gains evaporate online.
- No **position debiasing** → you reward prime placement, not quality; good items in bad slots never recover.
- No **cold-start channel** → new creators are invisible; the catalog ossifies; supply-side churn.
- No **feedback-loop audit** → silent creator-diversity collapse over months (the failure that looks fine every single day).
- No **completion normalization** → bias toward short content; satisfaction proxy corrupted.

**Stakeholder translation:** *Reader* — "finds the next story you'll actually finish, sooner." *Creator* — "new and long-tail authors get a guaranteed exploration budget." *Business* — "retention lift with a guardrail that blocks engagement gains that come at the cost of churn or catalog health."

---

## 14. Senior-vs-Naive Decision Points

| Decision | Naive answer | Senior answer (PulseDiscover) |
|---|---|---|
| Offline metric | "NDCG@10 = X, ship it" | "NDCG measures the old policy; here's the DR-corrected value with CI" |
| Position in the model | "Add position as a feature" | "Position must enter an *additive* PAL tower or be IPS-reweighted — concatenating it into a deep net lets it interact and doesn't debias" |
| Multi-objective selection | "Enumerate the Pareto frontier, apply weights" | "For fixed linear weights, enumeration is redundant; use constrained optimization (max engagement s.t. completion/diversity floors)" |
| Cold start | "Thompson sampling handles it" | "TS lives in the ranker; it can't rescue an item retrieval never surfaces — need a content retrieval channel + reserved exploration slots" |
| New ranker beats old offline | "Promote it" | "Is the gain real or exposure-driven? Check with DR before promoting" |
| Diversity reranking | "MMR improved fairness" | "MMR raised category diversity but seller Gini *worsened* 0.404→0.582 — diversity ≠ equality; I report the tradeoff" `[BUILT][SYNTHETIC]` |

---

## 15. The Hidden Failure Mode Nobody Talks About

**Propensity circularity + positivity violation under a hard top-K retriever.** In production the logging policy *is* the current model, so a propensity model fit to its outputs estimates "what the model already did" — using the model to correct the model's own bias. Worse, a hard top-K (e.g., top-500) retriever assigns propensity ≈ 0 to every item outside the candidate set, so `1/p → ∞` (or those items simply never appear in training data). **Consequence:** your "unbiased" estimator is unbiased only over the *retrieved support*, not the catalog. **The fix that most portfolios miss:** you need *known* propensities from an exploration/randomization slice (ε-greedy with logged probabilities) — which is precisely why C10's simulator and the exploration bucket exist. State this explicitly; it is the single most sophisticated thing you can say about RecSys evaluation.

---

## 16. 12-Month System Risk and Mitigation

- **Creator-exposure collapse (feedback loop):** model shows X → X gets signal → shows X more. Mitigation: C9 Shapley amplification audit → IPS-corrected retraining trigger.
- **Embedding/index staleness:** a new model produces different embeddings than the FAISS index holds; transition period mismatches ranker and index. Mitigation: version + pin embeddings to ranker, atomic shadow-rebuild, incremental re-encode at scale.
- **Popularity ratchet:** offline-best policy concentrates on head items. Mitigation: DR-corrected training + diversity floors in constrained selection.
- **Distribution drift (new genres/seasonality):** Mitigation: PSI/KL triggers + retrain cadence tied to staleness cost.
- **Cold-start starvation:** Mitigation: guaranteed exploration slot budget for new/long-tail creators.

---

## 17. Evidence Table (claim · number · tag · N · CI · source file)

| Claim | Number | Tag | N | CI | Source |
|---|---|---|---|---|---|
| V2 LambdaRank beats popularity | NDCG@10 0.192 vs 0.177 (+8.6%) | `[BUILT][SYNTHETIC]` | 4,132 sessions | report CI (bootstrap TODO) | `outputs/evidence/ml_ranker_v2_report.json` |
| V1 heuristic lost to popularity | NDCG@10 0.136 < 0.177 | `[BUILT][SYNTHETIC]` | 4,132 | — | `ml_ranker_v2_report.json` |
| Retrieval recall (content best channel) | Recall@100 0.647 | `[BUILT][SYNTHETIC]` | 4,132 | — | `learned_retrieval_v2_report.json` |
| IPS shows *smaller* gain than raw | bias_delta −0.0053 | `[BUILT][SYNTHETIC]` | impressions 41,320 | — | `ips_snips_v2_report.json`, `bias_correction_report.json` |
| Guardrail HOLD decision | +12.6% rev/session, +9.09pp returns → HOLD | `[BUILT][SYNTHETIC]` | replay | — | `ab_simulation_results.json` |
| Diversity tradeoff (honest negative) | seller Gini 0.404→0.582 (worse) | `[BUILT][SYNTHETIC]` | — | — | `reranking_constraints_report.json` |
| Corpus scale | 4,132 sessions / 41,320 impressions / 900 users / 650 items | `[BUILT][SYNTHETIC]` | — | — | `scripts/seed_demo.py` (seed 20260505) |
| "0.134 → 0.522" as improvement | — | `[DELETE]` | — | — | eval-frame artifact per `INTERVIEW_SAFE_CLAIMS.md` |
| Two-tower / SASRec / DR / PAL numbers | — | `[VISION]` | — | — | not built — no number claimed |

---

## 18. Honest Gaps and No-Overclaim Boundaries

- "0.134 → 0.522" is **not** an improvement — different eval frames, different subsets. `[DELETE the improvement framing]`
- ALS retrieval is **not wired into** the ranker's training features (scores 0.0). `[BUILT — disclosed]`
- Content embeddings are **8-dim random**, not semantic. `[BUILT][SYNTHETIC]`
- All current numbers are **synthetic e-commerce**, not fiction, not production traffic. `[SYNTHETIC]`
- SASRec, two-tower, DR, PAL, LightGCN, feedback-loop audit are **not built yet**. `[VISION]`
- No claim of unbiased OPE on raw observational data — only on the known-propensity simulator.

---

## 19. Free Version vs Small-Paid-Infra Version

- **Free (build first):** SASRec + two-tower + IPS/SNIPS/DR + PAL + cold-start + simulator on MovieLens-1M/Goodreads — all CPU or free Colab/Kaggle GPU. ₹0.
- **Small-paid:** full-Goodreads SASRec on a rented A100 hour (~₹150) only if you scale past free-GPU limits. No paid spend is required to make the headline (DR) result real.

---

## 20. Current → BeastMax → T3 / Polished

- **Current:** PulseRank e-commerce eval harness only (ALS, LambdaRank, IPS/SNIPS, MMR, A/B replay) `[BUILT][SYNTHETIC]`.
- **BeastMax (target):** two-tower + SASRec on Goodreads with real Recall/NDCG; semi-synthetic simulator; IPS→SNIPS→**DR** with variance + online-prediction comparison; PAL-vs-IPS position debiasing; cold-start content channel; feedback-loop Shapley audit.
- **T3 / polished:** LightGCN (V2); DIN (V2); multi-objective revenue-aware ranking; Pratilipi-flavored synthetic demo; serving-architecture write-up (FAISS rebuild strategy, batched inference).

---

## 21. Interview Narrative (the spoken story)

"Most RecSys portfolios train a model and report NDCG. Mine started by failing: my V1 heuristic ranker *lost to a popularity baseline* — 0.136 vs 0.177 NDCG. That forced the real question: is my offline metric even measuring the model, or the exposure my logging policy already gave popular items? So the project became about evaluation honesty. I built IPS and SNIPS off-policy evaluation — and the IPS-corrected gain was actually *smaller* than the raw number, which is the honest direction. Then I generalized: a doubly-robust estimator, `V_DR = DM + (1/p)(r − q̂)`, evaluated on a simulator with *known* propensities, because you can't get unbiased OPE from observational data. On top of that, a self-attentive sequential model — SASRec — because on serialized fiction the reading *order* is the strongest signal; position-bias correction two ways, PAL versus IPS, because they're different tools; a content cold-start channel, because you can't explore what you can't retrieve; and a 30-day feedback-loop audit that decomposes creator-exposure shift into genuine preference change versus the model amplifying its own past choices. The thing I'm proudest of is the stuff I *held*: I shipped a HOLD on a +12.6% revenue change because it breached a returns guardrail, and I report that my diversity reranker actually made seller-Gini *worse*. That's the difference between a demo and a system."

---

## 22. Uncomfortable Questions and Safe Answers

- **"Your propensities — where do they come from, are they unbiased?"** → "From an exploration slice with logged probabilities in my simulator. On observational data they'd be circular — the logging policy is the model. That's exactly why I don't claim unbiased OPE on raw data."
- **"Write the DR estimator."** → `V_DR = DM + (1/p)(r − q̂)`; unbiased if either p or q̂ is correct; lower variance than vanilla IPS.
- **"You added position as a feature and zeroed it at serving — does that debias?"** → "Not in a deep net — position interacts nonlinearly. Clean debiasing needs an additive PAL tower or IPS reweighting. I compare both."
- **"How does a new creator ever get exposure if your ranker is good?"** → "Content cold-start retrieval channel + reserved exploration slots; ranking exploration via ε-greedy/UCB. TS alone can't help an item retrieval never surfaces."
- **"MovieLens Recall is tiny — why should I be impressed?"** → "MovieLens is my smoke test, not my headline. The headline is DR validity on Goodreads + simulator. The model is table stakes; the evaluation isn't."
- **"What does your ranker do to the catalog in 12 months?"** → "Without correction, popularity ratchet + creator collapse. I audit it with a Shapley exposure decomposition and break the loop with IPS-corrected retraining."

---

## 23. Resume / LinkedIn Bullets

- "Built a sequential content recommender (SASRec + two-tower) whose offline metrics are validated by doubly-robust off-policy evaluation on a known-propensity simulator — `V_DR = DM + (1/p)(r−q̂)` — because offline NDCG measures the logging policy, not the model." `[VISION until built]`
- "Implemented IPS/SNIPS off-policy evaluation with clipping-sensitivity analysis on a 4,132-session corpus; honestly reported that IPS correction *reduced* the apparent gain (bias_delta −0.005), not inflated it." `[BUILT][SYNTHETIC]`
- "Shipped a guardrail-first HOLD on a +12.6% revenue change that breached a 9pp returns guardrail — governance over vanity metrics." `[BUILT][SYNTHETIC]`
- "Corrected position bias two ways (IPS reweighting vs PAL additive tower) and audited recommendation feedback loops via a Shapley exposure decomposition." `[VISION until built]`

---

## 24. Build Checklist (with done-criteria)

1. `[BUILD-TASK]` Two-tower + SASRec on MovieLens-1M (smoke) → Goodreads. **Done:** real Recall@10/NDCG@10, SASRec > two-tower > popularity, model card (heads/layers/max_seq_len).
2. `[BUILD-TASK]` OBP-style simulator (known propensities, PBM curve, ε-exploration). **Done:** propensities logged; replay reproducible.
3. `[BUILD-TASK]` IPS→SNIPS→**DR** with correct formula. **Done:** JSON shows Var(DR) < Var(IPS) and DR predicts simulated online lift better than NDCG.
4. `[BUILD-TASK]` PAL tower vs IPS reweighting. **Done:** NDCG distribution-shift plot; precise terminology in doc.
5. `[BUILD-TASK]` Cold-start content channel + exploration slots. **Done:** new-item Recall@5 reported.
6. `[BUILD-TASK]` Feedback-loop Shapley audit. **Done:** amplification-share computed with cross-term-as-confidence caveat.
7. `[VISION/V2]` LightGCN on real user-item graph; DIN; multi-objective revenue-aware ranking.
8. `[BUILD-TASK]` Evidence Table kept current; every number tagged + N + CI + source file.

---

## 25. What Must Be Deleted / Relabelled / Postponed From Old Docs

- **DELETE:** "0.134 → 0.522 improvement" framing everywhere (eval-frame artifact). Any implication the e-commerce repo is a fiction recommender.
- **RELABEL:** all PulseRank numbers as `[BUILT][SYNTHETIC]`; "embeddings" as 8-dim collaborative (not semantic); MMR result as "diversity up, seller-Gini worse" (honest negative).
- **POSTPONE:** LightGCN, DIN, DLRM, full Thompson sampling, multi-objective revenue ranking, production serving infra → V2/discussion-only.
- **RESOLVE NAMING:** PulseDiscover = product, PulseRank = ranking+OPE engine, RecSys = domain. Stop interchanging.

---
*Caveat: PulseRank numbers are code-grounded (`outputs/evidence/*.json`, verified); "tests green" inferred from committed caches/CI, not a live run. Academic methods (Joachims 2017 unbiased LTR; PAL/Guo 2019; YouTube additive tower/Zhao 2019; Open Bandit Pipeline) are stable pre-2025 knowledge, not re-verified by search this session — citation-check before quoting papers by name.*
