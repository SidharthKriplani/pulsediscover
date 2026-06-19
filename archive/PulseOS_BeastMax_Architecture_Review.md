# PulseOS — Final Architecture Review & BeastMax Build Plan

*Reviewer role: final architecture reviewer + BeastMax architect. Method: I read the actual source code in the mounted repos, not just the PRDs. Where a claim came from docs only, it is tagged. Philosophy held throughout: data is the gospel, technology is the guidance system; richness and interview-defensibility over complexity-for-its-own-sake; no faked production ownership.*

**Claim tags used everywhere below:** `[BUILT]` verified in code/outputs · `[PLACEHOLDER]` number written in markdown but not reproducibly computed · `[SYNTHETIC]` real code on synthetic/seeded data · `[FABRICATED]` number hardcoded or self-fulfilling in code · `[VISION]` BeastMax aspiration, not built · `[BUILD-TASK]` work order · `[PARKED]` · `[DELETE]`.

---

## 1. Executive Verdict

The portfolio is **stronger in engineering than in evidence integrity**, and the gap between the two is the whole game. Reading the code changes the picture in three ways the docs hide:

1. **Real, defensible spine exists.** `riskframe_platform` is genuinely production-shaped ML on the *full* Home Credit dataset (307,512 rows) with a real 5-gate promote/hold, calibration (ECE 0.0046 `[BUILT]`), and PSI/KS drift. `pulserank_platform` is a real offline RecSys-evaluation harness: from-scratch ALS, LightGBM LambdaRank, real IPS/SNIPS with a clipping sweep, MMR governance, A/B replay — all code-driven `[SYNTHETIC]`. `metriclens`, `trialcheck`, `featureleakagelens`, `docingestqa`, `inferencelens` are real, tested, honestly-scoped libraries. This is a Senior-credible body of work.

2. **Several marquee numbers are landmines.** The code audit found that some of the exact figures the planning docs lead with are not real: DevPulse's conflict macro-F1 `0.966` and Recall@5 `0.94` are **hardcoded / self-fulfilling** `[FABRICATED]`; MetaSignal's CUPED treatment effect is **synthetically injected** via a SHA256 uplift function `[SYNTHETIC-INJECTED]`; "mSPRT sequential testing" and "Oaxaca-Blinder" are **not in any codebase** `[VISION]` (MetricLens actually does mix/rate/cross with *Shapley*, not OB); NexusSupply's AUC `1.0` is **circular by construction** `[FABRICATED]`; and the famous PulseRank "0.134 → 0.522" is an **evaluation-frame artifact, not an improvement** (the repo's own `INTERVIEW_SAFE_CLAIMS.md` says so). An interviewer who pulls any one of these thread ends finds a hole. These must be relabeled or rebuilt before they are spoken aloud.

3. **The 4-combination pivot is correct, and Sonnet's audit is ~80% right** — but Sonnet over-credited the fabricated/not-built items above because it read the docs, not the code. My job here is to keep Sonnet's structural moves and correct the evidence layer underneath them.

**Verdict:** Proceed with the 4-combination architecture (PulseSignal / PulseGuard / PulseKnowledge / PulseDiscover). It is a **Senior-level portfolio with genuine Staff-level *judgment* signals** (calibration-over-discrimination gate, guardrail-first decisions, leakage pre-flight, honest off-policy evaluation). It is **not yet** a Staff-level *evidence* portfolio, and the fastest path to closing that is not more components — it is (a) deleting/relabeling ~6 fabricated numbers, and (b) building the 3–4 differentiators that are currently claimed-but-absent (real mSPRT, real DR off-policy eval, SASRec, FDR). Do that and PulseDiscover + PulseSignal become genuinely rare.

---

## 2. Files / Repos Actually Inspected vs Docs-Only Sources

**Inspected in source code (`src/`, `tests/`, `outputs/`, notebooks, configs) — claims here are code-grounded:**

- `pulserank_platform` — full src read: ALS (`ranking/learned_retrieval_v2.py`), LightGBM LambdaRank (`ranking/ml_ranker_v2.py`), IPS/SNIPS + clipping sweep (`evaluation/position_bias.py`, `ips_snips_v2.py`), MMR reranking, drift, A/B sim; `portfolio_kb/` decision logs; 61 test fns; evidence JSONs in `outputs/`.
- `riskframe_platform` — XGBoost + calibration (`training/`), 5-gate compare (`challenger/compare_models.py`), PSI/KS drift, policy engine; `artifacts/model_registry.json`, `calibration_report.json`. Real Home Credit data present.
- `featureleakagelens_v0` — `core.py` rule engine, 23 tests.
- `lendflow` — LangGraph DAG (`pipeline/graph.py`), hybrid RAG (`rag/retriever.py`), FOIR nodes, 40 tests.
- `nexussupply` — `data/generator.py`, `models/financial_model.py`, `pipeline/nodes/graph_risk.py`; confirmed circular labels in code.
- `agentreliabilitylab` — `rag/retriever.py`, `pipeline/nodes/`, `eval/ragas_eval.py`, `eval/ragas_results.json` (N=5 static).
- `docingestqa` — `checks.py` (11 checks), pyproject; `goldensetauditor_v0` — `core.py`, validation_summary; `inferencelens` — `pipeline/router.py`, `pareto.py`, 54 tests.
- `devpulse_platform` — `core/query_mode.py` (real conflict/version logic), `run_rag_eval_hardening_v35.py` (fabricated F1 cells), `hybrid_retrieval_report.json` (4-query), traffic backtest script.
- `metasignal_platform` — CUPED/SRM/guardrail/A-A scripts, `deterministic_uplift()` injection, streaming sim; README metrics not backed by committed JSONs.
- `trialcheck_v0` — `checks.py`/`stats.py`, 27 tests, publish workflow; `metriclens` — `decomposition.py` (Shapley), 38 tests, committed `dist/`.
- `ml-flagship-home_credit_risk` — notebook only; RandomForest AUC 0.717; `src/` empty.

**Read as documents only (treated as intent/claims, not implementation):** `CURRENT_STATE.md`, `combination-audit-final.md` (= Sonnet's audit), `BEASTING_STANDARD.md`, `INTERVIEW_SAFE_CLAIMS.md`, `riskframe-full-beast.md` (gold-standard reference), the three parking PRDs (`experimentiq/riskops/ragreliability-prd.md`), the `*-beastmax.md` and `*-overscope.md` docs, `opus-battletest-brief.md`, the Pratilipi defense docs.

**Not inspected (out of scope or redundant):** `artha`, `india-wealth-architecture`, `genai-systems-lab`, `ml-systems-lab`, `product-analytics-lab`, `production-systems-lab`, the `*.zip` archives, `AI-Flagship-*` (appears superseded by the named repos). Flag me if any of these is load-bearing.

**Verification note:** I could not execute the test suites live (sandbox lacked the deps / timed out). "Tests pass" rests on committed pytest caches with empty `lastfailed` and CI configs — not a fresh run I observed. Treat test counts as real, "green" as inferred.

---

## 3. State Reconciliation — Old 3-Combo vs New 4-Combo

There is a genuine, unresolved conflict between two documents you handed me, and silently mixing them is exactly what you told me not to do:

| Dimension | OLD truth (`CURRENT_STATE.md`, marked "LOCKED" 2026-06-12) | NEW scope (your brief + `combination-audit-final.md`) | Resolution |
|---|---|---|---|
| Combinations | ExperimentIQ, RiskOps, RAGReliability **+ Crucible as the 4th** | PulseSignal, PulseGuard, PulseKnowledge, **PulseDiscover (RecSys) as 4th** | **NEW wins.** RecSys replaces Crucible as the 4th active combination. |
| PulseRank home | Inside **ExperimentIQ** | Moves to **PulseDiscover** | **NEW wins.** PulseRank is a ranking/OPE engine; its home is RecSys. |
| Crucible / PulseTrain | "Confirmed fourth project," full PRD written | **PARKED** | **PARKED.** Future training-governance expansion only. |
| NexusSupply | Active component of RiskOps (AUC 1.0 flagged) | Remove / relabel | **RELABEL** (see §11, §14) — keep the contagion *graph*, kill the AUC. |
| Naming | ExperimentIQ / RiskOps / RAGReliability | Pulse* product names | Use Pulse* as product names; keep old names as the "skills" subtitle. |

**What is STALE and should stop being referenced as truth:** `combo-ab-lifecycle.md`, `combo-credit-platform.md`, `combo-rag-trilogy.md` (CURRENT_STATE itself marks these stale); the "4th project = Crucible" sections of `CURRENT_STATE.md`; `crucible-prd.md` / `crucible-beastmax.md` as *active* (reclassify to parked); the line in `CURRENT_STATE.md` that lists PulseRank under ExperimentIQ's components.

**What is PARKED:** Crucible/PulseTrain entirely; the RAGReliability "T3 GCP/Terraform/NeMo production deployment" layer (infra-for-show, not what interviews test — keep as a discussion-only roadmap).

**What is STILL VALID:** the `BEASTING_STANDARD.md` rubric (this is your real gold standard — keep it verbatim as the PRD bar); `riskframe-full-beast.md` as the gold-standard exemplar; `INTERVIEW_SAFE_CLAIMS.md` (this is the most valuable doc you have — it already does honest relabeling and should be expanded to every component); the real build numbers in §2-inspected repos.

**Conflicts that must be resolved out loud (not silently):** (1) PulseRank's "0.134→0.522" appears as an *improvement* in `combination-audit-final.md`'s resume-line examples but as an *evaluation-frame artifact* in `INTERVIEW_SAFE_CLAIMS.md` — the safe-claims doc is correct; purge the improvement framing everywhere. (2) "Oaxaca-Blinder" (docs) vs *Shapley mix/rate/cross* (actual MetricLens code) — code is correct; fix the term. (3) "mSPRT / sequential testing" claimed in audit, absent in code — build it or stop claiming it.

---

## 4. Sonnet Review — Accept / Modify / Reject

Sonnet's `combination-audit-final.md` is a strong, mostly-correct audit. The pattern in its errors is consistent: **it praised numbers that the code shows are not real**, because it worked from docs. Below, every Sonnet position is classified, and each weakness it raised is converted into an action (per your 7-category scheme: 1 conceptual flaw · 2 doc/ownership mismatch · 3 placeholder-number · 4 implementation gap · 5 fake blocker · 6 build task · 7 relabel/delete/postpone).

| # | Sonnet's claim / recommendation | Verdict | Correction / action (category) |
|---|---|---|---|
| 1 | Move PulseRank out of PulseSignal into RecSys | **ACCEPT** | Code confirms PulseRank is a ranking+OPE harness, not an experimentation tool. (cat 7) |
| 2 | Remove NexusSupply entirely | **MODIFY** | Don't delete the asset; **relabel**. Kill the AUC=1.0 (circular `[FABRICATED]`), keep the contagion graph + Altman-Z as a "graph-based supplier-risk *scoring* demo, no discrimination metric." (cat 3+7) |
| 3 | "mSPRT is rare, you can derive the mixing integral — strongest technique" | **REJECT (factual)** | **mSPRT is not in any codebase.** It's `[VISION]`. Sonnet credited a doc claim. Either build a real always-valid-p-value module (high value) or stop calling it owned. (cat 2→6) |
| 4 | "CUPED 26.46% with real theta and real experiment ID survives adversarial Qs" | **MODIFY** | CUPED *code* is real, but the treatment effect is **synthetically injected** (`deterministic_uplift()` SHA256) and the headline metrics are **markdown-only** (evidence JSONs not committed). Relabel as "CUPED on RetailRocket covariates with an *injected* treatment effect — variance-reduction mechanics are real, the lift is constructed." (cat 3) |
| 5 | "Oaxaca-Blinder decomposition on MetricLens (used at Lyft/Airbnb/Netflix)" | **MODIFY (terminology)** | MetricLens implements **mix/rate/cross with Shapley cross-term allocation**, *not* Oaxaca-Blinder, and has **no Simpson's detector** despite the example. Rename to "mix/rate/cross decomposition (Shapley)"; either build the Simpson's detector or drop the example. (cat 2+6) |
| 6 | "DevPulse uniquely memorable: macro-F1 0.966, Recall@5 0.94, 2,479 queries" | **REJECT (integrity)** | These are **fabricated/self-fulfilling in code**: F1 cells hardcoded; "hybrid" retrieval has no embeddings (vector = lexical×0.85); 2,479 = 37-day replay of 12 queries; wrong-version-rate 0.0 is tautological. The *version/conflict logic itself is real and good*. **Delete the fabricated metrics; recompute on a real corpus or claim only the deterministic logic.** Highest-priority credibility fix. (cat 3→6) |
| 7 | RAGAS N=5 must go to N=100 with CIs | **ACCEPT + escalate** | Correct, and worse than stated: the N=5 scores are a **frozen static JSON**, not regenerated by the harness. Rebuild as N≥100 with Wilson CIs, computed by code. (cat 3→6) |
| 8 | Add GNN fraud detection to PulseGuard (not overreach) | **REJECT / MODIFY** | Synthetic fraud-graph GNN repeats NexusSupply's circular-label mistake and earns *low* respect. Put real GraphML (LightGCN) in **PulseDiscover** on the real user-item graph instead; if PulseGuard wants GraphML, use a **public** AML/Elliptic graph, never synthetic. (cat 1+7) — see §14. |
| 9 | TabNet "run it, it's free on CPU"; numbers suspect | **ACCEPT** | Code audit: no real TabNet run found in `riskframe_platform`; ECE 0.018/0.024 is `[PLACEHOLDER]`. Either run it for real (CPU) or remove the challenger comparison. (cat 3→6) |
| 10 | LendFlow numbers unvalidated | **MODIFY** | routing_accuracy 0.95 *is* code-computed but on **20 synthetic apps with flags reconstructed from ground truth** (mildly self-fulfilling); field_extraction_f1 0.688 not found in code. Relabel as synthetic-fixture eval; validate F1 on a real doc set. (cat 3) |
| 11 | Rebrand PulseKnowledge as "Enterprise Search + RAG Eval"; absorb InferenceLens | **ACCEPT** | Matches code (real BM25+dense+cross-encoder exists). InferenceLens is real but thin as a standalone — fold into the eval story. (cat 7) |
| 12 | SASRec replaces DIN as primary sequential model | **ACCEPT** | Correct: simpler, more defensible, directly relevant to serialized fiction. (cat 6) |
| 13 | DR offline evaluation is the RecSys differentiator | **ACCEPT + fix formula** | Yes — but the brief's earlier DR formula was wrong. Correct form: `V_DR = DM + (1/p)(r − q̂)`. Build it on a semi-synthetic simulator (known propensities). (cat 6) |
| 14 | Exclude CNN and Pricing permanently | **ACCEPT (CNN) / MODIFY (Pricing)** | CNN: exclude, no honest fit. Pricing: don't build a 5th combo, but a thin revenue/LTV objective belongs inside PulseDiscover's multi-objective ranking + a revenue-guardrail in PulseSignal. (cat 7) — see §15. |
| 15 | Build Causal Forest HTE in PulseSignal | **MODIFY → V2** | Good differentiator but synthetic-only HTE is weak evidence. Make it V2; prioritize real mSPRT + FDR first (rarer, currently fabricated-as-claimed). (cat 6) |

**Net:** Accept Sonnet's restructuring (PulseRank→RecSys, rebrand PulseKnowledge, SASRec, DR, exclude CNN/pricing-as-combo). Reject/modify wherever Sonnet trusted a doc number the code doesn't support (mSPRT, CUPED lift, DevPulse metrics, GNN-fraud, TabNet). The corrected portfolio is *more honest and equally impressive*.

---

## 5. Final Active Scope (LOCKED)

Four combinations. Product name / skills subtitle / one-line thesis / status:

1. **PulseSignal** — *Experimentation & Decision Validity* — "Catches the four ways a decision lies: before (power, A/A, SRM), during (always-valid sequential testing), at readout (guardrails, FDR), and after the metric moves (mix/rate/Shapley decomposition)." Spine: MetaSignal + TrialCheck + MetricLens. `[BUILT core, 3 build-tasks]`
2. **PulseGuard** — *ML Risk Governance* — "Deterministic policy first, ML for scoring, LLM only for synthesis; leakage caught pre-flight, calibration gated over discrimination." Spine: FeatureLeakageLens + RiskFrame + LendFlow (+ NexusSupply relabeled). `[BUILT spine on real Home Credit]`
3. **PulseKnowledge** — *Enterprise Search + RAG Reliability* — "Five failure points across the AI lifecycle: ingestion QA, golden-set quality, cost/quality routing, agent termination, version-safe retrieval." Spine: DocIngestQA + GoldenSetAuditor + InferenceLens(absorbed) + AgentReliabilityLab + DevPulse. `[BUILT, 2 integrity fixes required]`
4. **PulseDiscover** — *Content Recommendation* — "A sequential recommender with honest off-policy evaluation, position-bias correction, cold-start, and feedback-loop governance." Engine: PulseRank (ranking + OPE). `[mostly VISION — the priority build]`

---

## 6. Parked Scope

- **Crucible / PulseTrain** — QLoRA/Unsloth fine-tuning + 5-dim eval harness + regression gate + RCA. `[PARKED]`. Mention only as "the training-governance expansion: the same pre-flight→gate→decomposition philosophy applied to fine-tuned checkpoints." Do not build pre-Pratilipi; do not claim.
- **RAGReliability T3 production deployment** (GCP Cloud Run + Qdrant + Portkey + NeMo + Terraform) — `[PARKED / discussion-only]`. Infrastructure does not move a DS interview; keep as a spoken roadmap, not a build.
- **PricePulse** (dynamic pricing / demand forecasting / discount experimentation) — `[PARKED]`. The honest "5th-combination gap"; do not force it into the current four.
- **GNN fraud on synthetic graph** — `[DELETE from plan]` (circular-label trap). Real GraphML lives in PulseDiscover (§14).

---

## 7. PulseRank / PulseDiscover Ownership Decision

**Decision: PulseDiscover is the product/beast; PulseRank is the ranking + off-policy-evaluation engine inside it.** This matches your preference *and* the code.

Reasoning grounded in source: the shipped repo is literally "PulseRank Platform," and what it contains is a *ranking and offline-evaluation stack* — ALS/co-occurrence candidate generation, a LightGBM LambdaRank ranker, IPS/SNIPS off-policy evaluation with a clipping sweep, MMR governance, delayed attribution, and offline A/B replay. That is precisely "the ranking/evaluation engine," not a full discovery product. "PulseDiscover" exists today only as an aspirational name in `portfolio_kb/decisions/` with no implementation.

So: **PulseDiscover** = the end-to-end recommender (retrieval → sequential model → ranking → governance → feedback-loop audit). **PulseRank** = the component that scores and *honestly evaluates* rankings (LambdaRank + IPS/SNIPS/DR + NDCG). One nuance to state in interviews: the current PulseRank repo is e-commerce/marketplace-flavored (sellers, GMV, returns). PulseDiscover is a *new build in the content/fiction domain* that **absorbs PulseRank's evaluation engine** and replaces the e-commerce framing with reading-sequence framing. Do not retro-claim the e-commerce repo as a fiction recommender.

**Stop doing:** using PulseRank, PulseDiscover, and "RecSys" interchangeably. RecSys = domain (for conversation), PulseDiscover = product, PulseRank = engine.

---

## 8. Missing PulseDiscover / PulseRank RecSys PRD — Architecture Spec

This is the combination with the most upside and the least code. Below is the architecture to build, with explicit *now / V2 / overkill* calls. No name-dropping — every choice is justified or cut.

### 8.0 Foundational assumption (Layer 0)
Click/read model: **Position-Based Model (PBM)** — observed engagement = P(examined | position) × P(relevant | user, item). This is the assumption that makes position-bias correction and off-policy evaluation *meaningful*. State it first; it is the thing most RecSys portfolios never name.

### 8.1 Candidate generation (retrieval) — what belongs NOW
- **Two-tower retrieval** (user tower / item tower, dot-product, ANN via FAISS). NOW. It is table stakes, but it is the honest backbone and the substrate for cold-start content features. Don't oversell it.
- **Cold-start retrieval channel** — a *separate* content-based channel (item metadata → embedding) reserved as exploration slots, because Thompson sampling in the ranker cannot rescue an item retrieval never surfaces (the "you can't explore what you can't retrieve" point). NOW — it is one day of work and closes the #1 RecSys interview question.
- **ALS / co-occurrence** — already built in PulseRank; keep as a baseline channel and as the honest "collaborative signal" comparison.

### 8.2 Ranking / sequential model — what belongs NOW vs V2
- **SASRec (self-attentive sequential recommendation)** — NOW, as the primary deep model. Justification over alternatives: unidirectional transformer over the user's reading sequence is the strongest signal on a serialized-fiction platform (chapter→chapter→next-story), it is simpler and more defensible than DIN/DIEN, and it is free to train on CPU/small-GPU for MovieLens/Goodreads scale.
- **DLRM-style feature-interaction model** — V2/overkill for a portfolio; it shines with rich tabular/cross features you won't have. Discuss, don't build.
- **DIN / DIEN** — V2. Attention-over-history against a candidate is harder to implement and less transferable than SASRec. Replace DIN-as-primary with SASRec-as-primary (agrees with Sonnet).
- **LightGCN / GraphSAGE on the user-item graph** — V2 (see §14). Natural and respected *because the graph is real interactions*; build after two-tower + SASRec + DR work.
- **Multi-objective ranking head** — NOW but small: predict engagement + completion + a diversity/creator-exposure term; select with **constrained optimization** ("maximize engagement s.t. completion ≥ τ₁, creator-diversity ≥ τ₂"), *not* "enumerate a Pareto frontier then apply fixed linear weights" (that enumeration is redundant for fixed linear weights — a known trap). The constraint thresholds are what PulseSignal A/B-tests.

### 8.3 Evaluation — the differentiator (this is where the portfolio wins)
- **IPS / SNIPS off-policy evaluation** — already built `[SYNTHETIC]`; keep and clip.
- **Doubly-Robust (DR) off-policy evaluation** — NOW, the headline. Correct estimator: `V_DR = DM + (1/p)·(r − q̂)` (direct-method estimate plus IPS-weighted residual of the reward model). Show that DR has lower variance than vanilla IPS and that DR-corrected offline ranking predicts simulated online lift better than raw NDCG. **This single result is the lean-forward moment** and almost no Indian DS portfolio has it.
- **Position-bias correction, two routes, compared** — NOW (see §8.5 terminology).
- **Honest framing:** offline metrics on public data are not online truth; you bridge the gap with DR on a semi-synthetic simulator with *known* propensities. Say this explicitly.

### 8.4 Feedback-loop governance (Layer for "12-month system risk")
- Every N days, decompose the shift in creator-exposure distribution into preference-change vs **exposure-amplification** vs cross, using MetricLens's mix/rate/Shapley engine. If amplification dominates, retrain with IPS-corrected training to break the loop. **Honest caveat to state:** mix and rate are not cleanly separable when the same policy sets both exposure and engagement; report this as descriptive-under-a-stated-reference, with the cross/Shapley term as a confidence indicator, not a causal attribution.

### 8.5 PAL vs IPS — the exact terminology question, answered

**What is PAL?** PAL = **Position-bias Aware Learning** (Huawei, RecSys 2019). It is an *architecture*: model predicted CTR as `P(seen | position) × P(relevant | item, user)`, where a small **position module** learns the examination probability as part of the network, trained jointly with the relevance model. At serving you drop/fix the position module and score pure relevance. YouTube's "Watch Next" paper (Zhao et al., RecSys 2019) does the same idea as an **additive** shallow position tower summed at the logit. Key property: position must enter **separately/additively** so it does not interact with the relevance features — simply concatenating position as one more input feature into a deep net does *not* cleanly debias, because position then interacts nonlinearly with everything.

**What is IPS?** **Inverse Propensity Scoring** is a *reweighting* technique, not an architecture. You estimate the propensity (here, the examination/position probability) and weight each training/eval example by `1/propensity`, so under-examined-but-relevant items count more. For position bias specifically this is the Joachims et al. (2017) "Unbiased Learning-to-Rank with Biased Feedback" line, where examination propensities are estimated from a **randomization intervention** (RandPair / result-swap experiments) or a Position-Based-Model EM fit.

**Is "IPS position debiasing" wrong terminology?** It is **imprecise and conflates two different families.** IPS is a general reweighting tool; "position debiasing" is one *application* of it. And PAL is *not* IPS — PAL reparameterizes the model, IPS reweights the loss. Calling PAL "IPS position debiasing," or calling any position-debiasing "IPS," will get corrected by a RecSys interviewer.

**The precise language to use (so you sound exact):**
> "I correct position bias two complementary ways and compare them. Route 1 is **propensity-based**: IPS/IPW on examination propensities, estimated from a randomization slice (Joachims 2017) — this reweights the click loss. Route 2 is **model-based**: a PAL-style additive position tower (Guo 2019; YouTube's additive tower, Zhao 2019) learned jointly and dropped at serving — this reparameterizes the model. IPS reweights; PAL reparameterizes. They're complementary, and I show the NDCG distribution shift each produces."

Never say "IPS position debiasing" as if it were one named method, and never call PAL a form of IPS.

### 8.6 Data strategy — the realistic path
MovieLens alone *is* too toy-like for a headline. Use a **three-tier hybrid**, and be explicit about which tier each claim comes from:

1. **MovieLens-1M → bootstrap/smoke test only.** Validate that SASRec and two-tower are implemented correctly (sane Recall@10/NDCG). Never a headline number.
2. **Goodreads (UCSD Book Graph) → the real, domain-relevant headline dataset.** Strongly preferred over Amazon Reviews for a *fiction* platform: it has books, **series (sequential reading)**, **authors (long-tail creators)**, ratings and shelving events — the closest public analog to Pratilipi. (Amazon Reviews "Books" is the fallback if Goodreads licensing/size is a problem; RetailRocket is e-commerce, wrong domain — skip for PulseDiscover.) Real held-out interactions are your real labels; no circularity.
3. **Semi-synthetic simulation layer (Open Bandit Pipeline–style) → for IPS/SNIPS/DR, position bias, exploration, feedback loops.** This is non-optional and is the senior move: public datasets have **no known logging propensities**, so true off-policy evaluation is impossible on them directly. You take the real data, fit a "production" logging policy, impose a PBM examination curve + exploration epsilon, simulate clicks, and now have *known* propensities to evaluate IPS/DR against ground truth. Academic-standard (this is literally what OBP exists for). **State the boundary: "OPE validity is demonstrated on a semi-synthetic simulator with known propensities; I do not claim unbiased OPE on raw observational data."**

**If you also want the Pratilipi-flavored synthetic demo** (optional, for narrative color), generate it realistically with: users with genre affinities; stories with chapters; **Zipf/long-tail popularity skew**; creators with varying catalog age (long-tail authors); a PBM **position-bias curve**; an **exploration bucket** (ε-greedy with logged probabilities — this is what makes the simulator's propensities *known*); **repeat reading** and **completion-depth** signals (fraction of chapters finished, the satisfaction proxy — but length-normalize it, since completion confounds with chapter count); cold-start items with no history; and a **feedback-loop knob** (exposure amplification factor) so you can *demonstrate* the loop and then break it. Label every number from this generator `[SYNTHETIC]`.

### 8.7 What is overkill for this PRD (cut or V2)
Real-time feature store / serving infra (discuss, don't build); DIN/DIEN (V2); DLRM (discuss); GNN/LightGCN (V2, after foundation); contextual bandits with full Thompson sampling (needs a posterior the point-estimate ranker doesn't produce — either add MC-dropout/ensemble or downgrade to ε-greedy/UCB, and say so); multi-task DLRM-scale feature crosses.

---

## 9. PRD Upgrade Gap Table vs RiskFrame BeastMax Standard

The gold standard (`riskframe-full-beast.md` + `BEASTING_STANDARD.md`) requires, per component: Layer 0 foundational assumption · full alternatives comparison table with numbers · unit economics at 3 scales · customer-experience translation · short-vs-long-term tension · the unspoken failure mode · the regulator/PM question · 12-month system risk · cost-of-not-having · domain-evolution rungs · interview narrative. The three parking PRDs are good *build specs* but are not yet *beast docs* on these axes.

| BeastMax axis | PulseSignal PRD | PulseGuard PRD | PulseKnowledge PRD | PulseDiscover (none yet) |
|---|---|---|---|---|
| North-star thesis | ✅ has | ✅ has | ✅ has | ❌ build |
| Target buyer / JD archetype | ⚠️ thin | ⚠️ thin | ⚠️ thin | ❌ |
| Layer-0 assumption stated | ❌ add (which validity model) | ⚠️ partial (survivorship) | ❌ add | ❌ add (PBM) |
| Full alternatives table w/ numbers | ⚠️ partial | ✅ strong (7 models) | ⚠️ partial | ❌ build |
| Output→input data flow across components | ⚠️ implied | ✅ | ⚠️ | ❌ |
| Senior-vs-naive decision points | ✅ (guardrail HOLD) | ✅ (calibration gate) | ⚠️ | ❌ |
| Unit economics @ 3 scales (₹) | ⚠️ summary only | ✅ | ⚠️ | ❌ |
| Cost-of-not-having each technique | ❌ add | ⚠️ partial | ❌ add | ❌ |
| Monitoring + regression gates | ⚠️ (A/A only) | ✅ (PSI/challenger) | ⚠️ | ❌ |
| Honest gaps / no-overclaim | ⚠️ **must add CUPED-injection, mSPRT-not-built** | ⚠️ **must add TabNet/LendFlow/Nexus labels** | ⚠️ **must add DevPulse/RAGAS labels** | ❌ |
| Evidence requirements (N, CI, real-vs-synthetic) | ❌ **biggest gap** | ⚠️ | ❌ **biggest gap (N=5)** | ❌ |
| Interview story + drill-downs | ✅ has Q&A | ✅ has Q&A | ✅ has Q&A | ❌ build |

**Pattern:** the PRDs are strong on narrative and weak on the **evidence layer** (N, CI, real-vs-synthetic labeling) and on **cost-of-not-having / Layer-0**. The single highest-leverage upgrade across all four is to add an **Evidence Table** to every component: claim · number · `[BUILT/SYNTHETIC/PLACEHOLDER/...]` tag · N · CI · source file. `INTERVIEW_SAFE_CLAIMS.md` already does this for PulseRank — clone its format into all four.

---

## 10. PulseSignal Depth Decision (Objective 5 audit)

Current code reality: CUPED, SRM (χ²), guardrail-first SHIP/HOLD, 1000-run A/A with KS-uniformity, streaming early-warning — all `[BUILT]` in `metasignal_platform`; SRM/z-test/Welch/peeking/MDE/SMD in `trialcheck_v0` `[BUILT]`; mix/rate/cross+Shapley in `metriclens` `[BUILT]`. **Not present:** sequential/mSPRT, Simpson's, FDR, bandits, geo/switchback/cluster, CUPAC, HTE, OB.

| Technique | Decision | Why |
|---|---|---|
| A/A, SRM, power/MDE, guardrails, peeking detection | **ESSENTIAL — already built** | Keep; these are the spine. Fix the A/A FPR story (5.5% vs 5% is a real teachable point). |
| CUPED | **ESSENTIAL — but relabel** | Mechanics real; treatment effect injected. State it. |
| **mSPRT / always-valid p-values** | **BUILD NOW (highest value)** | The rare differentiator, currently *claimed but absent*. A real always-valid-p module + a demo that fixed-horizon testing inflates FPR under continuous peeking while mSPRT controls it = a genuine Staff-signal. ~3–4 days. |
| **Multiple testing / FDR (Benjamini-Hochberg)** | **BUILD NOW** | Directly answers the FWER landmine (5 experiments @0.05 → 22.6% family-wise error). Cheap, high-credibility. Layer over both many-metrics and many-experiments. |
| **Simpson's paradox detector** | **BUILD NOW (cheap)** | Makes the "+3.1% aggregate hides −1.2% India" example *real* instead of illustrative; add to MetricLens. ~1 day. |
| CUPAC (ML-based CUPED) | **V2** | Natural extension once CUPED is honest. |
| Heterogeneous treatment effects (Causal Forest) | **V2** | Good, but synthetic-only HTE is weak evidence; after mSPRT/FDR. |
| Long-term holdouts, novelty/primacy effects | **DISCUSS now, simulate in V2** | Maturity signal; cheap to simulate one. |
| Cluster-randomized, network/SUTVA interference | **DISCUSS only** | Conceptually critical for a creator marketplace (treating a creator affects their readers) — *know it cold*, but don't fake a build. |
| Geo / switchback experiments | **OVERKILL now** | Need the right temporal/marketplace data; one switchback sim is a stretch-V2 at most. |
| Experiment meta-analysis | **V2** | Pairs with FDR. |

**Minimum build proof to make PulseSignal genuinely deep:** real mSPRT (with the peeking-FPR demo) + FDR layer + Simpson's detector + honest CUPED relabel. That converts PulseSignal from "shallow + one fabricated claim" into "the rare experimentation portfolio that actually has sequential testing and multiplicity control."

---

## 11. PulseGuard Depth Decision

Spine is the most *real* in the portfolio (RiskFrame on full Home Credit). Decisions:

- **RiskFrame — keep as flagship, fix two doc/code gaps.** `[BUILT]`: XGBoost champion AUC 0.766 / PR-AUC 0.261 / ECE 0.0046, real 5-gate compare, PSI/KS. **But:** (a) **WOE/IV is in the docs, not the code** — either implement the binning (it's standard, ~1 day) or stop claiming "183→47 via WOE/IV"; (b) **reject inference is claimed but the code does forward delayed-label tracking, not reject inference** — relabel honestly or build a real reweighting/Heckman approach; (c) the **fairness DI gate is hardcoded 0.003** — compute it. (cat 2/6)
- **TabNet challenger — run it for real or cut it.** No real TabNet run found. The "held TabNet on ECE despite better AUC" story is a *great* Staff narrative *if real*; run TabNet on the real test set (CPU-fine) and, importantly, **try isotonic/Platt recalibration before holding it** — calibration is the most fixable metric, and holding the better-discriminating model on raw ECE without recalibrating is a process gap an interviewer will probe.
- **LendFlow — keep, relabel.** Real LangGraph + hybrid RAG + FOIR `[BUILT]`; routing 0.95 is on 20 synthetic apps with flags reconstructed from ground truth `[SYNTHETIC, mildly self-fulfilling]`. Validate field-extraction F1 on a small real-PDF set; otherwise frame as a deterministic-policy + agentic-escalation demo, which it genuinely is.
- **NexusSupply — relabel, don't delete.** Kill AUC 1.0 (circular `[FABRICATED]`). Keep Altman-Z + sentiment + **contagion graph (2-hop BFS + betweenness)** as a *scoring/architecture* demo with **no discrimination metric**. This becomes your honest "graph on financial networks" story (classical graph analytics, not GraphML). (cat 3+7)
- **GraphML in PulseGuard?** See §14 — only on a **public** AML/Elliptic graph; never synthetic fraud.

Depth verdict: PulseGuard is already deep; its work is **integrity cleanup, not new components**. After cleanup it is the most interview-bulletproof combination.

---

## 12. PulseKnowledge Depth Decision

- **Rebrand to "Enterprise Search + RAG Reliability"** (accept Sonnet). The BM25 + dense + cross-encoder reranker stack in `lendflow`/`agentreliabilitylab` is a real search system — claim it.
- **AgentReliabilityLab — fix the evidence (top priority).** Pipeline (hybrid retrieval, confidence gating, MITRE hard-overrides) is `[BUILT]`; the RAGAS scores are a **frozen N=5 static JSON** `[PLACEHOLDER]`. Rebuild: N≥100 real MITRE Q&A, RAGAS computed by code each run, **Wilson 95% CIs on every metric** (faithfulness 0.847 on N=5 has a CI of roughly [0.42, 0.98] — unusable). (cat 3→6)
- **DevPulse — the biggest integrity problem in the portfolio.** Real deterministic version/conflict logic `[BUILT]`; but macro-F1 0.966 is **hardcoded**, "hybrid retrieval" has **no embeddings** (vector = lexical×0.85), Recall@5 0.94 is from **4 queries**, the 2,479-query backtest is a **synthetic replay of 12 queries**, wrong-version-rate 0.0 is **tautological**. **Action:** delete every fabricated metric immediately; either (a) build a *real* hybrid retriever and recompute conflict-F1 against hand-labeled conflicts on a real multi-version corpus (e.g., `requests`/`boto3`/Pandas version docs), or (b) claim only the deterministic version-gate logic (which is genuinely good) and drop the RAG-eval numbers. This is the #1 "thread that unravels in an interview." (cat 3→6)
- **DocIngestQA, GoldenSetAuditor, InferenceLens — keep, honest.** All real and well-scoped. **Absorb InferenceLens** into the eval story as the cost/quality-routing module rather than a standalone product (it's real but thin alone). Specify DocIngestQA's gates concretely (token-length thresholds, Jaccard dedup cutoff, OCR-noise rule, PII regex) — "quality gates" is too vague.
- **Boundary cleanup:** GoldenSetAuditor = the *data-quality* auditor; AgentReliabilityLab = the *runtime/eval* pipeline. State the distinction so they don't read as duplicates.

Depth verdict: PulseKnowledge has the strongest *unique* concept (version-safe retrieval) attached to the **weakest evidence**. Fixing DevPulse + RAGAS is worth more than any new component.

---

## 13. PulseDiscover Depth Decision

Covered architecturally in §8. Depth verdict: this is the **priority build**, because it is the only combination whose differentiators (DR off-policy evaluation, SASRec, PAL-vs-IPS comparison, feedback-loop audit) are simultaneously *rare*, *directly relevant to Pratilipi*, and *currently unbuilt*. Build order within it: (1) two-tower + SASRec on Goodreads with real Recall/NDCG; (2) semi-synthetic simulator with known propensities; (3) IPS→SNIPS→**DR** with the variance + online-prediction comparison; (4) PAL vs IPS position-debiasing comparison; (5) cold-start content channel; (6) feedback-loop Shapley audit; (7) LightGCN as V2. The first real DR number is the milestone that turns this from "designed" into "built."

---

## 14. GraphML / GNN Decision

**Decision: real GraphML belongs in PulseDiscover (LightGCN on the real user-item interaction graph), as V2 after the SASRec/DR foundation. NOT synthetic fraud-GNN in PulseGuard.**

Reasoning:
- **Synthetic fraud-graph GNN would repeat NexusSupply's exact mistake.** If you plant fraud rings in a synthetic graph and then "detect" them, the labels are circular and a sophisticated interviewer discounts it. High effort, low respect. Reject Sonnet's "add fraud GNN to PulseGuard (synthetic)."
- **Would synthetic graph fraud be respected?** No — same circularity smell as AUC=1.0. **Would real graph fraud be respected?** Yes. So if PulseGuard wants GraphML, use a **public** graph-fraud dataset (Elliptic Bitcoin, IBM AML, or DGL's fraud-amazon/yelp) — real labels, real graph. That is a legitimate optional V2 for PulseGuard, clearly labeled as a public benchmark.
- **Is LightGCN on the user-item graph more natural?** Yes — it is collaborative filtering on a *real* bipartite interaction graph; the "labels" are real held-out interactions, no circularity. It belongs in PulseDiscover and compares cleanly against two-tower/ALS.
- **Minimum defensible GraphML build:** LightGCN on Goodreads/MovieLens user-item graph, Recall@K vs two-tower baseline, with an honest note on over-smoothing and why depth ≤ 3 layers. 
- **Should GraphML wait until RecSys foundation is done?** Yes — V2. Don't add a fourth modeling paradigm before two-tower + SASRec + DR produce real numbers.

PulseGuard's *graph* story in the meantime = NexusSupply's contagion/centrality, relabeled as **classical graph analytics** (not representation learning), no AUC.

---

## 15. Pricing / Revenue-Science Decision

**Decision: no 5th combination. A thin revenue layer is split across two existing combinations; dynamic pricing is parked.**

- **Into PulseDiscover:** a **revenue/LTV-proxy objective inside multi-objective ranking** (engagement vs completion vs creator-diversity vs a revenue/retention proxy), selected by constrained optimization. This is "revenue-aware ranking," genuinely senior, and reuses the multi-objective head you're already building. ACCEPT.
- **Into PulseSignal:** a **revenue guardrail + long-term holdout** framing on top of the existing guardrail engine (a positive engagement metric held when a revenue/retention guardrail breaches). This is maturity on what's already built. ACCEPT.
- **Excluded (overreach / future PricePulse):** dynamic pricing, discount optimization, price-elasticity modeling, ad-load tradeoffs. These are a real separate combination (`[PARKED]`); forcing them in now adds complexity without credibility. If asked: "pricing isn't in this portfolio; experimentation + causal inference + multi-objective ranking are the foundation I'd build it on."

---

## 16. Premium-Skill Coverage Table (corrected for code reality)

`S`=strong/built · `s`=present/needs-emphasis · `V2`=planned · `—`=excluded · `✗`=claimed-but-not-built

| Premium skill | PulseSignal | PulseGuard | PulseKnowledge | PulseDiscover |
|---|---|---|---|---|
| Experimentation | **S** (after mSPRT+FDR) | — | — | s (offline A/B) |
| Causal inference | **S** (decomp; HTE V2) | — | — | **S** (DR/IPS) |
| Sequential testing / mSPRT | ✗→**BUILD** | — | — | — |
| Multiplicity / FDR | ✗→**BUILD** | — | — | — |
| Recommender systems | — | — | — | **S** (build) |
| Search / IR | — | — | **S** (claim it) | s (candidate gen) |
| Deep learning | — | s (TabNet if run) | s (BERT encoders) | **S** (SASRec, two-tower) |
| Sequential / transformers | — | — | — | **S** (SASRec) |
| GNN / GraphML | — | V2 (public AML only) | — | V2 (LightGCN) |
| Off-policy evaluation | s (concept) | — | — | **S** (IPS/SNIPS/DR) |
| Position-bias / debiasing | — | — | — | **S** (PAL vs IPS) |
| Risk / credit / fraud | — | **S** (real Home Credit) | — | — |
| Calibration / monitoring / governance | **S** | **S** (PSI, 5-gate) | s | s (feedback-loop audit) |
| RAG / retrieval eval | — | s (LendFlow RAG) | **S** | — |
| Agentic systems (LangGraph) | — | **S** (LendFlow) | **S** (AgentReliabilityLab) | — |
| Cost / quality / unit economics | s | **S** | **S** (InferenceLens) | s |
| Revenue-aware ranking | s (guardrail) | — | — | s (multi-objective) |
| CNN | — | — | — | — (excluded, no honest fit) |
| Pricing / demand | — | — | — | — (PARKED: PricePulse) |

Coverage is broad and, post-cleanup, *honest*. The two genuinely empty cells worth filling are **sequential testing/FDR** (PulseSignal) and **DR off-policy eval** (PulseDiscover) — both currently claimed-but-absent, both rare, both buildable.

---

## 17. Free Version vs Small-Paid-Infra Version (per combination)

"Free" = laptop/CPU + free-tier APIs + public data, ₹0–500. "Small-paid" = a few hundred ₹ of GPU/API to make one headline number real.

| Combination | Free version (build this first) | Small-paid-infra version (the headline upgrade) |
|---|---|---|
| **PulseSignal** | All of it. CUPED/SRM/A-A/guardrail/MetricLens + new mSPRT/FDR/Simpson's are pure NumPy/SciPy on synthetic + RetailRocket. ₹0. | Optional: one LLM-eval experiment (CUPED-analogy on summarization) ~₹400 of API to add a real "experimentation on LLM outputs" story. |
| **PulseGuard** | RiskFrame (real Home Credit), FeatureLeakageLens, NexusSupply graph, **TabNet** (CPU) — all free. | LendFlow real-PDF field-extraction eval with a real LLM (~₹300); optional public-AML LightGCN on Colab free GPU. |
| **PulseKnowledge** | DocIngestQA/GoldenSetAuditor/InferenceLens free. Local embeddings (sentence-transformers) + BM25 + cross-encoder run on CPU → real hybrid retrieval for DevPulse rebuild, ₹0. | RAGAS N≥100 needs an LLM judge: ~₹500–800 of API for the AgentReliabilityLab rebuild with CIs. This is the one paid spend worth making. |
| **PulseDiscover** | SASRec + two-tower + IPS/SNIPS/DR + PAL on MovieLens/Goodreads + semi-synthetic simulator — all CPU/Colab-free. ₹0. | Goodreads-scale SASRec training on a Colab/Kaggle free GPU (still ₹0); a rented A100 hour (~₹150) only if you go to full Goodreads. |

The portfolio is **almost entirely free to build**. The only spend worth making is the RAGAS N≥100 LLM-judge run (kills the single worst evidence liability). Everything else is CPU or free-tier GPU.

---

## 18. Current → BeastMax → T3/Polished Evolution (per combination)

| Combination | Current (verified) | BeastMax (honest + deep, the target) | T3 / Polished (stretch) |
|---|---|---|---|
| **PulseSignal** | CUPED/SRM/A-A/guardrail/decomp built; CUPED lift injected; no mSPRT/FDR | + real mSPRT (peeking-FPR demo) + FDR + Simpson's; CUPED relabeled; evidence table | + CUPAC + Causal-Forest HTE + one LLM-eval experiment |
| **PulseGuard** | RiskFrame real on Home Credit; TabNet/LendFlow/Nexus numbers unvalidated/circular | + real TabNet run w/ recalibration; WOE/IV or claim dropped; Nexus relabeled (no AUC); LendFlow F1 on real docs | + reject-inference (real method) + public-AML LightGCN + RBI/Screener real corpus |
| **PulseKnowledge** | Pipelines real; RAGAS N=5 frozen; DevPulse metrics fabricated | + real hybrid retriever in DevPulse, conflict-F1 recomputed on real multi-version docs; RAGAS N≥100 + CIs; InferenceLens absorbed; gates specified | + production RAG (GCP/Qdrant/Portkey/NeMo) as discussion-only roadmap |
| **PulseDiscover** | PulseRank e-commerce eval harness only; RecSys unbuilt | two-tower + SASRec on Goodreads (real Recall/NDCG); semi-synthetic sim; IPS→SNIPS→**DR**; PAL vs IPS; cold-start; feedback-loop Shapley audit | + LightGCN (V2) + DIN (V2) + multi-objective revenue-aware ranking + Pratilipi-flavored synthetic demo |

---

## 19. What to Build First (ranked work plan)

**Tier 0 — Integrity (do before any interview; mostly deletion, ~2 days):**
1. Delete/relabel the fabricated numbers: DevPulse macro-F1 0.966 + Recall 0.94 + 2,479-backtest; NexusSupply AUC 1.0; MetaSignal CUPED "lift"; PulseRank "0.134→0.522" improvement framing; "mSPRT/Oaxaca-Blinder owned." Replace each with an honest label per `INTERVIEW_SAFE_CLAIMS.md` format.
2. Add an Evidence Table (claim · number · tag · N · CI · source file) to all four combinations.

**Tier 1 — The differentiators that are claimed-but-absent (the real ROI, ~2–3 weeks):**
3. **PulseDiscover: two-tower + SASRec on Goodreads** → first real RecSys numbers.
4. **PulseDiscover: semi-synthetic simulator + IPS/SNIPS/DR** → the lean-forward result (DR lower variance than IPS; DR predicts simulated online lift better than NDCG). Correct DR formula throughout.
5. **PulseSignal: real mSPRT + FDR + Simpson's** → converts the experimentation story from partly-fabricated to genuinely rare.
6. **PulseKnowledge: RAGAS N≥100 + CIs; DevPulse real hybrid retriever + recomputed conflict-F1** → removes the worst liabilities.

**Tier 2 — Cleanup builds (~1 week):**
7. PulseGuard: real TabNet run + recalibration; decide WOE/IV (build or drop); compute the fairness gate.
8. PulseDiscover: PAL-vs-IPS comparison + cold-start content channel + feedback-loop Shapley audit.

**Tier 3 — V2 elevation (post-Pratilipi or if time):**
9. LightGCN (PulseDiscover), Causal-Forest HTE (PulseSignal), public-AML GraphML (PulseGuard optional), multi-objective revenue-aware ranking.

---

## 20. What to Delete From Docs Immediately

- "mSPRT / sequential testing" and "Oaxaca-Blinder" as **owned/built** capabilities (not in code). Replace with "Shapley mix/rate/cross decomposition (built)" + "mSPRT (planned)".
- DevPulse "conflict macro-F1 0.966", "hybrid Recall@5 0.94/0.97", "2,479 queries / 37-day backtest", "wrong-version rate 0.0" as **evidence**. (Keep the deterministic version-gate concept.)
- NexusSupply "CV ROC-AUC 1.00" everywhere except one honest line.
- PulseRank "0.134 → 0.522" as an **improvement** (it's an eval-frame artifact).
- MetaSignal "26.5% variance reduction" stated without the "treatment effect is injected" caveat.
- "Causal Forest HTE" as built (it isn't).
- Stale combo docs as truth: `combo-ab-lifecycle.md`, `combo-credit-platform.md`, `combo-rag-trilogy.md`, and the "Crucible = 4th project" sections of `CURRENT_STATE.md`.

---

## 21. What to Relabel as Placeholder / Synthetic / Future

| Item | Relabel to |
|---|---|
| RiskFrame numbers (AUC 0.766 / ECE 0.0046) | `[BUILT — real Home Credit]` (these are your good real numbers; keep prominent) |
| PulseRank V2 NDCG 0.192 vs pop 0.177; IPS Δ −0.005 | `[SYNTHETIC — real code, seeded e-commerce corpus]` (defensible) |
| CUPED 26.5% / θ | `[SYNTHETIC — real mechanics, injected treatment effect]` |
| LendFlow routing 0.95 | `[SYNTHETIC — 20 fixtures, flags reconstructed from ground truth]` |
| LendFlow field-extraction F1 0.688 | `[PLACEHOLDER — validate on real PDFs]` |
| TabNet ECE 0.018/0.024 | `[PLACEHOLDER — run for real or remove]` |
| RAGAS faithfulness 0.847 etc. | `[PLACEHOLDER — N=5 frozen; rebuild N≥100 + CI]` |
| NexusSupply contagion/centrality | `[SYNTHETIC architecture demo — no discrimination metric]` |
| All PulseDiscover model numbers | `[VISION until built]` |
| Crucible, GCP production RAG, PricePulse | `[FUTURE / PARKED]` |

---

## 22. What to Postpone

- Crucible/PulseTrain (all). RAGReliability production-infra (GCP/Terraform/NeMo) → discussion-only. DIN/DIEN, DLRM, LightGCN, contextual-bandit Thompson sampling → PulseDiscover V2. Causal-Forest HTE, CUPAC, meta-analysis → PulseSignal V2. Geo/switchback/cluster experiments → discuss-only. Real-time PulseGuard→PulseDiscover feature integration → postpone (latency + fairness hazard; batch-precompute is the right answer to *describe*, not build). PricePulse / pricing science → future 5th combination.

---

## 23. Final RiskFrame-Standard PRD Table of Contents (template for all four)

Use this exact skeleton for each combination's upgraded PRD (it merges `BEASTING_STANDARD.md`'s 8 questions + 6 layers with the evidence discipline this review adds):

```
0.  North-star thesis + one-line identity
1.  Target buyer / JD archetype (which roles, which companies, why now)
2.  The ONE insight nobody else brings
3.  Layer 0 — Foundational assumption (world-model; when it breaks)
4.  Component map + data flow (each component's output → next component's input)
5.  Premium-skill coverage (what this combo proves)
6.  Layer 1 Data — signals, labels, cold start, real-vs-synthetic boundary
7.  Layer 2 Model — full alternatives table (metric / ₹-per-1M / latency / cold-start / interpretability / min-data), winner + "would switch IF…"
8.  Layer 3 Evaluation — metric choice justified; offline↔online gap; validity (leakage, N, CI)
9.  Layer 4 Production behavior — monitoring, drift triggers, feedback loops, retrain cadence
10. Layer 5 Business + unit economics @ 3 scales (₹/day), break-even, 3 stakeholder views
11. Senior-vs-naive decision points (the calls you'd argue about)
12. Cost of NOT having each major technique
13. The failure mode nobody talks about (1 real one, explained)
14. 12-month system risk + the fix
15. ** Evidence Table ** — claim · number · [tag] · N · CI · source file   ← the new mandatory section
16. Honest gaps / no-overclaim boundaries
17. Domain-evolution rungs (how this grows over 5 years)
18. Interview narrative + pre-loaded uncomfortable Q&A (with pass/fail answers)
19. Resume lines (Outcome by Method, beating Alternative by Δ, at Cost, despite Constraint)
20. Build checklist (ordered, with done-criteria + time/₹)
```

Section 15 (Evidence Table) is the upgrade that separates a beast doc from a build spec, and it is exactly the layer this review found missing.

---

## 24. Exact Next Work Orders for the Builder Model

Hand these to Sonnet/the builder as discrete, done-criteria'd tasks. Ordered by the §19 tiers.

**WO-1 (Tier 0, integrity): Purge + relabel.** Across all PRDs/READMEs/`portfolio_kb`, apply the §20 deletions and §21 relabels. *Done when:* no fabricated/placeholder number appears without a tag; `INTERVIEW_SAFE_CLAIMS.md` extended to cover DevPulse, MetaSignal CUPED, NexusSupply, RAGAS.

**WO-2 (Tier 0): Evidence Tables.** Add §23-section-15 Evidence Table to each of the four combinations. *Done when:* every headline number has tag + N + CI + source file.

**WO-3 (Tier 1, PulseDiscover): retrieval + sequential.** Implement two-tower + SASRec; train on MovieLens-1M (smoke) then Goodreads. *Done when:* `outputs/` has real Recall@10/NDCG@10 for SASRec > two-tower > popularity, with model card (heads/layers/max_seq_len).

**WO-4 (Tier 1, PulseDiscover): OPE.** Build the OBP-style semi-synthetic simulator (known propensities, PBM curve, ε-exploration) and implement IPS→SNIPS→DR with the **correct** estimator `V_DR = DM + (1/p)(r−q̂)`. *Done when:* a JSON shows Var(DR) < Var(IPS) and DR-ranked offline metric correlates with simulated online lift better than raw NDCG.

**WO-5 (Tier 1, PulseDiscover): position debiasing.** Implement both routes — IPS-on-examination-propensity and PAL additive tower — and compare. *Done when:* NDCG distribution-shift plot shows which items move under each; doc uses the precise §8.5 terminology.

**WO-6 (Tier 1, PulseSignal): sequential + multiplicity.** Implement mSPRT always-valid p-values + Benjamini-Hochberg FDR + a Simpson's detector in MetricLens. *Done when:* an A/A simulation shows fixed-horizon FPR inflates under continuous peeking while mSPRT holds ~5%; FDR demo over ≥5 metrics; Simpson's flags the aggregate-vs-segment reversal.

**WO-7 (Tier 1, PulseKnowledge): RAGAS + DevPulse.** Rebuild AgentReliabilityLab RAGAS at N≥100 with Wilson CIs computed in code; replace DevPulse's fake hybrid with a real BM25+dense+rerank retriever and recompute conflict macro-F1 against hand-labeled conflicts on a real multi-version doc corpus. *Done when:* RAGAS JSON shows N≥100 + CIs; DevPulse F1 is detector-vs-labels (not hardcoded) on real docs.

**WO-8 (Tier 2, PulseGuard): TabNet + gates.** Run TabNet on the real Home Credit test set; apply isotonic recalibration before the hold decision; compute the fairness DI gate; decide WOE/IV (implement or drop the claim). *Done when:* MLflow run exists with real ECE pre/post recalibration; gate decision reproduced from real numbers.

**WO-9 (Tier 2, PulseDiscover): cold-start + feedback loop.** Add the content-based cold-start retrieval channel and the 30-day creator-exposure Shapley decomposition (preference vs amplification). *Done when:* new-item Recall@5 reported; amplification-share computed with cross-term-as-confidence caveat.

**WO-10 (Tier 3, V2): elevation.** LightGCN (PulseDiscover), Causal-Forest HTE (PulseSignal), optional public-AML GraphML (PulseGuard), multi-objective revenue-aware ranking. Only after WO-1…9 are green.

---

*One caveat on this review's own evidence: repo claims are grounded in source I read directly; the academic terminology (Joachims 2017 unbiased LTR, PAL/Guo 2019, YouTube additive tower/Zhao 2019, Open Bandit Pipeline) is from stable pre-2025 knowledge, not re-verified via search this session — worth a citation check before you quote papers by name in a room. Test-suite "green" status is inferred from committed caches/CI, not a live run.*
