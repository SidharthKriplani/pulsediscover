# 12 — Consolidated Evidence Ledger (Phases 2–7)

*Single source of truth for every built number in the PulseDiscover BeastMax-core spine. Every row: claim · number · tag · N · CI/uncertainty · source file · reproducible? · safe interview line. Tags: `[BUILT — real data (smoke)]` (MovieLens-1M), `[BUILT][SYNTHETIC — simulator]` (known-propensity simulator), `[BUILT][SYNTHETIC — e-commerce]` (vendored PulseRank provenance). Seed 20260616 throughout; "tests green" inferred from CI/caches, not a live run.*

> **Scope boundaries (binding):** MovieLens-1M is a **smoke test**, never the domain headline. Simulator results are **estimator-validation / mechanism proof**, never real online lift. The T3 deep-learning models (Two-tower, LightGCN, GraphSAGE/PinSage/PyG, Wide&Deep, DeepFM/xDeepFM, DIN/DIEN, BERT4Rec, DLRM, Transformers4Rec) are **roadmap only — NOT built**.

## Phase 2 — Retrieval baselines (MovieLens-1M, real data, smoke)
| Claim | Number | Tag | N | CI (boot95) | Source | Repro | Safe interview line |
|---|---|---|---|---|---|---|---|
| ALS best baseline @20 | Recall@20 0.0761 | `[BUILT — real data (smoke)]` | 1,209 | [0.0612, 0.0910] | `outputs/evidence/retrieval_baselines.json` | yes | "ALS beats popularity/co-occ on the smoke, CIs separate at @20" |
| ALS @10 | Recall@10 0.0339 | `[BUILT — real data (smoke)]` | 1,209 | [0.0240, 0.0447] | same | yes | — |
| Popularity @10 | Recall@10 0.0182 | `[BUILT — real data (smoke)]` | 1,209 | [0.0116, 0.0256] | same | yes | "the honest floor" |
| Co-occurrence @10 | Recall@10 0.0033 | `[BUILT — real data (smoke)]` | 1,209 | [0.0008, 0.0066] | same | yes | "weak at next-item; reported as-is" |
| Split | 999,000 train / 1,209 held-out | `[BUILT — real data (smoke)]` | — | — | `data_manifest.json` | yes | "global-time split + constrained leave-last-out" |

## Phase 3A — SASRec (MovieLens-1M, real data, smoke; last-position training)
| Claim | Number | Tag | N | CI (boot95) | Source | Repro | Safe interview line |
|---|---|---|---|---|---|---|---|
| SASRec beats ALS @10 | Recall@10 0.0662 | `[BUILT — real data (smoke)]` | 1,209 | [0.0521, 0.0811] | `outputs/evidence/sasrec_report.json` | yes | "sequential model beats ALS on the smoke; CIs vs ALS [0.024,0.045] don't overlap at @10" |
| SASRec @20 | Recall@20 0.1026 | `[BUILT — real data (smoke)]` | 1,209 | [0.0852, 0.1199] | same | yes | — |
| SASRec @5 | Recall@5 0.0405 | `[BUILT — real data (smoke)]` | 1,209 | [0.0298, 0.0521] | same | yes | — |
| Training | 30 epochs, d=48/2 blocks/1 head/maxlen=30 | `[BUILT — real data (smoke)]` | — | loss 7.43 (still falling) | same | yes | "under-trained floor; last-position training, NOT canonical full-position SASRec" |

## Phase 4A — Known-propensity simulator (sanity)
| Claim | Number | Tag | N | Uncertainty | Source | Repro | Safe interview line |
|---|---|---|---|---|---|---|---|
| Logged events | 60,400 / 6,040 impressions | `[BUILT][SYNTHETIC — simulator]` | 60,400 | — | `outputs/evidence/simulator_manifest.json` | yes | "known μ, known propensities — the substrate OPE needs" |
| Propensity sums | max dev from 1 = 0.0 | `[BUILT][SYNTHETIC — simulator]` | — | exact | `simulator_sanity_report.json` | yes | "per-slot mixture sums to 1 exactly" |
| Positivity | min propensity 0.0045 > 0 | `[BUILT][SYNTHETIC — simulator]` | — | — | same | yes | "candidate-pool/slate-level positivity, not full-catalog" |
| Exploration share | 0.0985 vs ε=0.10 | `[BUILT][SYNTHETIC — simulator]` | — | — | same | yes | "ε-exploration guarantees positivity" |
| Examination / reward | 0.379 / 0.166 | `[BUILT][SYNTHETIC — simulator]` | — | — | same | yes | "PBM examination declining by rank; plausible reward" |

## Phase 5 — IPS / SNIPS / DM / DR validation (simulator)
**Known ground-truth V(π) = 0.2191.** CIs: user-cluster bootstrap, B=400.
| Estimator | Estimate | Bias | Tag | CI95 / std | Source | Safe interview line |
|---|---|---|---|---|---|---|
| IPS | 0.2066 | −0.0125 | `[BUILT][SYNTHETIC — simulator]` | [0.1984,0.2145], std 0.00417 | `outputs/evidence/ope_comparison.json` | "unbiased in theory; highest variance" |
| SNIPS | 0.2083 | −0.0108 | `[BUILT][SYNTHETIC — simulator]` | [0.2013,0.2158], std 0.00375 | same | "self-normalized; lower variance" |
| DR (good q̂) | 0.2117 | −0.0074 | `[BUILT][SYNTHETIC — simulator]` | [0.2048,0.2184], std 0.00345 | same | "lowest variance, small bias — expected, observed" |
| DM (bad q̂) | 0.3513 | **+0.1322** | `[BUILT][SYNTHETIC — simulator]` | [0.3512,0.3514] | same | "trusts the wrong model → badly biased" |
| DR (bad q̂) | 0.2133 | **−0.0058** | `[BUILT][SYNTHETIC — simulator]` | [0.2066,0.2199] | same | "recovers to ≈GT despite bad q̂ — double robustness, propensities exact" |
| ESS / w_max | 12,325 / 60,400 ; 65.8 | `[BUILT][SYNTHETIC — simulator]` | — | same | "moderate weight dispersion → variance reduction matters" |

> **Phase 5 scope note:** the target policy **π and the ground-truth reward are ALS-based** for this estimator-validation run (μ = popularity, truth = sigmoid(ALS), π = softmax(ALS)). This validates estimator *behaviour* against a known ground truth. A **SASRec-based π** — running the actual headline model through the OPE harness — is **now built (Option B, below)**; the ground-truth reward remains ALS-based.

## Phase 6 — Position-bias correction (simulator, 124 items ≥20 impr)
| Estimator | MSE vs true | Spearman vs true | corr w/ avg-rank | Tag | Source | Safe interview line |
|---|---|---|---|---|---|---|
| naive | 0.0534 | 0.708 | −0.471 | `[BUILT][SYNTHETIC — simulator]` | `outputs/evidence/position_bias_report.json` | "position-confounded" |
| IPS reweight | **0.0115** | 0.633 | **−0.253** | `[BUILT][SYNTHETIC — simulator]` | same | "removes most position bias (4.7× lower MSE) at a rank-correlation/variance cost — honest tradeoff" |
| PAL (Route B, built) | — (logit) | 0.699 | — | `[BUILT][SYNTHETIC — simulator]` | same | "additive tower, β dropped at serving; debiases without high-variance weights; no ranking win here" |
| Examination curve | empirical 0.95→0.19 ≈ known PBM | `[BUILT][SYNTHETIC — simulator]` | — | same | "examination curve recovered from logged exposure" |

## Phase 7 — Feedback-loop / catalog-health audit (simulator, 10 rounds)
| Claim | Number | Tag | Source | Safe interview line |
|---|---|---|---|---|
| Ratchet concentrates | Gini 0.621→0.700; long-tail 0.357→0.289 | `[BUILT][SYNTHETIC — simulator]` | `outputs/evidence/feedback_loop_report.json` | "volume-ranking amplifies what it showed — catalog ossifies" |
| Mitigation | CTR+explore: Gini 0.151; long-tail 0.766 | `[BUILT][SYNTHETIC — simulator]` | same | "rate+exploration breaks the loop — but magnitude is idealized; relevance cost not modeled" |
| Buried quality surfaced | 0.0204 → 0.0976 (×4.8) | `[BUILT][SYNTHETIC — simulator]` | same | "high-quality low-popularity items surfaced" |
| Genre proxy | Gini 0.675 vs 0.657 (barely moves) | `[BUILT][SYNTHETIC — simulator]` | same | "loop bites at item level; creator-level pending Goodreads" |

## Option B — SASRec-π through the OPE harness (simulator)
*π = trained SASRec scores (μ and ALS-based ground-truth reward unchanged). **GT V(π_SASRec) = 0.1383** — NOT comparable to ALS-π's 0.219 as recommender quality (truth is ALS-derived).*
| Estimator | Estimate | Bias | Tag | CI95 / std | Source | Safe interview line |
|---|---|---|---|---|---|---|
| IPS | 0.1346 | −0.0037 | `[BUILT][SYNTHETIC — simulator]` | [0.125, 0.144], std 0.0051 | `outputs/evidence/ope_comparison_sasrec_pi.json` | "OPE now evaluates the *actual* SASRec policy" |
| SNIPS | 0.1359 | −0.0024 | `[BUILT][SYNTHETIC — simulator]` | [0.126, 0.145] | same | — |
| DR (good q̂) | 0.1362 | −0.0021 | `[BUILT][SYNTHETIC — simulator]` | [0.127, 0.145], std 0.0045 < IPS | same | "DR evaluates SASRec with low bias + variance" |
| DM (bad q̂) | 0.2949 | +0.1566 | `[BUILT][SYNTHETIC — simulator]` | [0.294, 0.295] | same | "trusts wrong model → wild" |
| DR (bad q̂) | 0.1382 | −0.00008 | `[BUILT][SYNTHETIC — simulator]` | [0.128, 0.148] | same | "DR recovers GT almost exactly — double robustness" |
| ESS / w_max | 4,474 / 120.1 | — | `[BUILT][SYNTHETIC — simulator]` | — | same | "harder than ALS-π (SASRec diverges more from μ) — reported, not hidden" |

*Answer to "can DR evaluate the actual SASRec policy?": **YES** — small bias, DR variance ≤ IPS, double-robust; at the honest cost of lower ESS. Truth still ALS-based; no real online lift.*

## Vendored provenance — PulseRank e-commerce harness (reference only)
| Claim | Number | Tag | Source | Safe interview line |
|---|---|---|---|---|
| LambdaRank vs popularity | NDCG@10 0.192 vs 0.177 | `[BUILT][SYNTHETIC — e-commerce]` | `archive/pulserank_evidence/ml_ranker_v2_report.json` | "earlier e-commerce harness; not the fiction build" |
| IPS smaller gain | bias_delta −0.0053 | `[BUILT][SYNTHETIC — e-commerce]` | `archive/pulserank_evidence/ips_snips_v2_report.json` | "IPS corrected *down* — honest direction" |
| Guardrail HOLD | +12.6% rev, +9.09pp returns → HOLD | `[BUILT][SYNTHETIC — e-commerce]` | `archive/pulserank_evidence/ab_simulation_results.json` | "shipped a HOLD" |
| Diversity tradeoff | seller Gini 0.404→0.582 (worse) | `[BUILT][SYNTHETIC — e-commerce]` | `archive/pulserank_evidence/reranking_constraints_report.json` | "diversity ≠ equality" |

## G-series — domain deep models + two-stage system (Goodreads fantasy/paranormal, real data)
| Claim | Number | Tag | N | Source | Repro | Safe interview line |
|---|---|---|---|---|---|---|
| Sequence tournament (CPU) | best seq SASRec-small R@20 0.036; all < ALS f64 0.085 | `[BUILT — real data]` | 5k LLO | `g11_sequence_tournament_metrics.json` | yes | "honest negative — seq models lose to MF on CPU" |
| Full-softmax SASRec (GPU) | R@20 0.0456 vs sampled-neg 0.0068 (6.7×); small d48 0.0594 | `[BUILT — real data, T4]` | 5k LLO | `g11cd_colab_metrics.json` | yes | "full softmax is the lever; ALS 0.085 still wins" |
| LTR (LambdaMART) materially beats single retriever | R@20 0.0382 (ALS-rank) → 0.0791 (LTR), within-candidate | `[BUILT — real data]` | 733 test | `g12_ltr_metrics.json` | yes (md5 split) | "the two-stage ranker is the win, ~2×" |
| GPU features non-additive to LTR | +0.0027 R@20, within noise | `[BUILT — real data]` | 733 | `g12_ltr_metrics.json` | yes | "importance ≠ marginal contribution" |
| G13 candidate widening converts | coverage 0.157→0.42; LTR R@20 0.086→0.124 (+44%, ~3.2 SE) | `[BUILT — real data]` | 733 | `g13_candidate_coverage_report.json` | yes | "bottleneck was retrieval; mid config = knee" |
| G14 slate governance | honest negative — pure LTR already discovery-rich; combined policy 20.7% self-violation | `[BUILT — real data]` | 733 | `g14_final_slate_policy_report.json` | yes | "governance best handled upstream, not post-hoc slate" |
| **G15 co-occurrence = conditional adopt** | **+3.76pp marginal coverage and directional LTR lift (R@20 +0.0177 ~1.5 SE), but NOT 95%-significant at n=733** | `[BUILT — real data]` | 733 | `g15_candidate_source_report.json` | yes (md5 split) | "low-cost complementary source; directional, not statistically established; revalidate at larger N" |

| **G20 canonical SASRec to convergence (terminal)** | converged (monitor-R@20 plateau, ~95 ep) R@20 **0.0650 < ALS 0.0846** (ratio 0.768) → clearly_loses | `[BUILT — real data, T4; v3 JSON plot-reconstructed]` | 5k LLO final (used once) | `g20_canonical_sasrec_convergence_report_v3.json` | yes (md5 split, monitor set separate) | "canonical full-softmax SASRec trained to convergence still didn't beat ALS; MF is the right warm floor — terminal, offline" |

**Adopted default candidate strategy:** ALS-200 + SASRec-small-100 + content-50 + pop-30 + co-occurrence-100. LightGCN deferred (not cheap on CPU). FAISS = G16 (serving demo).

## V2 expansion lane (new evidence; V1 unchanged)
| Claim | Number | Tag | Source | Safe interview line |
|---|---|---|---|---|
| **G22 FAISS serving latency-quality frontier** | FlatIP exact p95 0.386 ms (~47% < brute 0.723), lossless; **HNSW ef64 overlap@20 0.992, rec-R@20 deg 0.0, p95 0.160 (~2.4× < FlatIP)** | `[BUILT — real data, CPU in-sandbox latency, NOT production]` | `g22_faiss_latency_quality_report.json` | "FAISS retains exact recall and cuts p95 latency (measured); HNSW near-lossless at 2.4× speedup — latency, not quality" |

| **G23 production-shaped serving API + load test** | warm p95 3.34 ms, cold→popularity 100% fallback / 0% empty, 276 qps; failure modes → 422/graceful; metrics+health live | `[BUILT — real data, CPU in-sandbox, NOT production]` | `g23_serving_api_report.json` | "built + load-tested a production-like recommender API: FAISS retrieval, versioned loader, fallback tree, structured logging, health, monitoring — production-shaped, not deployed" |

| **G24 cold-start + sparse-cohort fallback quality** | popularity fallback collapses catalog coverage ~59× (3532→60 unique items) & personalization 0.68→0.001 for cold users; recall non-monotonic (cold gold more popularity-predictable); 57% gold items item-cold-start; 7/7 serving cases non-empty | `[COMPUTED — served c2 ALS f64, offline]` | `g24_cold_start_sparse_cohort_report.json`, `g24_cold_start_degradation_curve.png`, `g24_head_tail_exposure.png` | "measured cold/sparse fallback quality, degradation, coverage & head/tail exposure across cohorts — cold-start NOT solved" |

| **G25 catalog exposure / coverage governance** | all policies Gini>0.97; popularity near-degenerate (Gini 1.0, 99.9% zero-exposure, ~60 unique items); head exposure ~10× catalog base rate, long-tail ~0.02× (suppressed ~50×); blended best serving-complete policy; HNSW≈exact exposure (Gini 0.986 vs 0.986) | `[COMPUTED — served c2 ALS f64, offline, top-20]` | `g25_catalog_exposure_governance_report.json`, `g25_exposure_concentration_curve.png`, `g25_head_mid_tail_exposure_by_policy.png`, `g25_policy_exposure_lorenz_curve.png` | "audited catalog exposure concentration (Gini, coverage, zero-exposure, head/tail lift) across policies — concentration measured, NOT fairness certified" |

| **G26 exposure-aware reranking / mitigation** | head-cap rerank is Pareto-dominant: Recall@20 +40% (0.0175→0.0245) AND coverage 7.3%→12.3% AND long-tail exposure ~8× AND Gini 0.982→0.968 (head inflation crowded out niche golds); tail-boost = mild clean lever; MMR/genre-div honest negatives; all rerankers ≤~1.3 ms p95 | `[COMPUTED — served c2 ALS f64, warm pool, offline]` | `g26_exposure_reranking_frontier_report.json`, `g26_relevance_exposure_frontier.png`, `g26_tail_exposure_vs_recall_tradeoff.png` | "evaluated exposure-aware rerankers and quantified the relevance/exposure tradeoff frontier — mitigation attempted, NOT fairness certified" |

| **RiskFrame-Gold Ruthless Audit** | verdict PATCH (mean 8.3, band: strong/needs-consolidation); component-coherence + OPE-logging gap; next = G26A→G26B; G26 BLOCKED | `[AUDIT]` | `pulsediscovery_riskframe_gold_audit.json`, `docs/PULSEDISCOVERY_RISKFRAME_GOLD_AUDIT.md` | "honest, gated recommender DECISION dossier; not a pile of components; one consolidation spine from cohesive" |
| **G26A decision-architecture + claim-unlock spine** | unified decision flow + model/protocol registry (d3aplus 0.0846 vs served c2 ~0.0375 = NOT_COMPARABLE) + claim ladder + interview wording bank + OPE logging schema; G26 quarantined as G26B | `[CONSOLIDATION]` | `docs/G26A_*` (5 docs), `g26a_decision_architecture_summary.json` | "unified recommender DECISION architecture: retrieval→serving→fallback→exposure→claim-boundaries→future-OPE, interview-maximal but evidence-bounded" |
| **G26B heuristic reranker integration (in-service)** | config-flagged default-OFF head-cap rerank in served `c2` path; coverage 6.9%→9.8%, long-tail 0.68%→2.93% (~4.3×), Gini 0.983→0.977, R@20 +5% (no loss), +0.3ms p95; **PASS — quarantine CLOSED**, ship=enable-as-config-default-off | `[BUILT — served c2, in-service offline]` | `g26b_rerank_integration_report.json`, `docs/G26B_RERANK_INTEGRATION.md` | "integrated a config-flagged heuristic reranker into the served path and measured the relevance/exposure/latency tradeoff — interpretable exposure control, NOT learned ranking" |
| **G31 search/IR framing (BM25 + dense hybrid)** | BM25 lexical lane + RRF hybrid with dense; search-standard NDCG@K + MRR (single-relevant) + Recall@K on seed-item (query-by-document) queries (600 users); BM25 strongest single lane (R@20 0.0133 vs dense 0.0067); **hybrid robust-middle, NOT a recall winner**; re-narrates LTR/FAISS/OPE as two-stage search | `[BUILT — served c2, seed-item, offline]` | `g31_search_ir_report.json`, `g31_lexical_dense_hybrid.png`, `docs/G31_SEARCH_IR_FRAMING.md` | "built a two-stage-search retrieval front end (BM25 lexical + dense + RRF) with NDCG/MRR reporting; query-by-document not free-text, position-bias *evaluation* not correction, single-relevant NDCG/MRR — V1 RecSys claims untouched" |
| **G30 final gold audit + interview kit (Gold Pass 3/3)** | RiskFrame re-score 9.1/10 (mid-audit 8.3); verdict OFFLINE GOLD-COMPLETE / interview-ready; V1 stays gold_candidate 8.9 (locked); online/production permanently out of scope; interview kit + claim ladder locked | `[AUDIT/CLOSEOUT]` | `g30_final_gold_audit.json`, `docs/G30_FINAL_GOLD_AUDIT.md`, `docs/PULSEDISCOVERY_INTERVIEW_KIT.md` | "cohesive offline recommender DECISION dossier, interview-ready; every claim artifact-backed & bounded; online evidence out of scope by design" |
| **G29 OPE execution (Gold Pass 2/3)** | logging policy (ε+softmax, overlap-safe) + recorded propensities; IPS/SNIPS/DR recover known target-policy value (DR 0.0089 vs true 0.0106; IPS over-estimates ~2×; SNIPS stable), rank RRF-fusion > ALS; ESS 220-266 | `[METHODOLOGY — offline self-logged, proxy reward]` | `g29_ope_execution_report.json`, `g29_ope_estimator_validation.png`, `docs/G29_OPE_EXECUTION.md` | "executed OPE end-to-end (logging+propensities+IPS/SNIPS/DR) and validated estimators recover a known value offline — methodology demonstrated, NOT real online lift" |
| **G28 learned fusion + ranker tournament (Gold Pass 1/3)** | candidate table 1.03M rows/6,213 users (ALS & semantic ~disjoint, overlap 6,749); RRF best non-learned; **LightGBM LambdaMART fusion beats ALS-only on test R@20 0.036 vs 0.022 + recovers cold-start 0.032 vs 0** (81 test positives — directional, offline); text ablation: desc marginally hurts; champion=LambdaMART, ALS warm primary, semantic cold-start/tail source | `[BUILT — served c2, offline test split, small positive set]` | `g28_candidate_feature_table_summary.json`, `g28_fusion_baseline_report.json`, `g28_learned_ranker_tournament.json`, `g28_source_policy_decision.json`, `g28_semantic_text_ablation.json`, `g28_final_ranker_fusion_decision.json`, `docs/G28_LEARNED_FUSION_RANKER_TOURNAMENT.md`, `docs/PULSEDISCOVERY_GOLD_PASS_1_DEFENSE_NOTES.md` | "built a source-aware fusion/learned-ranking layer over ALS+semantic+popularity; learned fusion beat ALS-only on the offline test split & recovered cold-start — offline ranking evidence, NOT online lift or solved cold-start" |
| **G27 semantic content retrieval (cold-start candidate lane)** | MiniLM 384-d embeddings, 41,866 items (100% text), FAISS FlatIP content index; semantic ONLY source reaching item-cold-start golds (reachability 54.6%, ALS/pop=0), coverage 62.6% vs ALS 8.1%, Gini 0.73 vs 0.98, p95 ~0.23ms; loses on head relevance (0.010 vs 0.087) → cold-start/tail COMPLEMENT not replacement; naive ALS-first fusion fails (honest negative). ship=cold_start_fallback+supplemental | `[BUILT — dense MiniLM, served-c2 protocol, offline]` | `g27_semantic_retrieval_report.json`, `g27_coldstart_recovery.png`, `g27_semantic_vs_popularity_coverage.png`, `g27_head_tail_exposure.png`, `g27_latency_tradeoff.png`, `docs/G27_SEMANTIC_CONTENT_RETRIEVAL.md` | "added a dense semantic content-retrieval candidate lane for item-cold-start/sparse-catalog; measured recall/coverage/long-tail/latency vs popularity & served c2 — offline candidate-generation evidence, NOT online lift or solved cold-start" |

**V2 boundary:** FAISS does NOT improve recommendation quality (best case = retains); serving is production-**shaped** (FastAPI+Docker+load test), **not** deployed (no real users/online lift); CPU in-sandbox latency, not a production benchmark; G24 measures fallback QUALITY but does **not** solve cold-start (user or item) and claims **no** fairness/personalization for unknown users; G25 measures EXPOSURE concentration but is **not** protected-class fairness, **not** a fairness certification, and does **not** solve long-tail discovery or deploy exposure governance; G26 **attempts mitigation** (rerankers) and reports the relevance/exposure tradeoff but is **not** fairness-solved/certified, **not** online diversity, **not** production-deployed, and the head-cap win is protocol-specific (warm cohort, single-held-out-gold); does NOT change the V1 ALS-warm-floor conclusion. V1 status (`gold_candidate` 8.9) unchanged.

**G20 evidence note:** v1 JSON was stale/invalid (val-loss NaN false early-stop) and **rejected**; the **v3 plot** (script-generated) is the source for the headline values; the repaired JSON `g20_canonical_sasrec_convergence_report_v3.json` is **provenance-tagged `plot_reconstructed`** with secondary metrics (R@50/R@200/NDCG) `null / pending_correct_v3_json` (NOT invented). G20 closes the method-depth blocker and justifies G21 — it does **not** unlock online/production/gold_complete.

## Must-not-claim (carried forward)
- "0.134 → 0.522 improvement" — `[DELETE]`, eval-frame artifact, never an improvement.
- DLRM / DIN / DIEN / Wide&Deep / DeepFM / BERT4Rec / Transformers4Rec — `[ROADMAP — NOT BUILT]`. Two-tower BUILT (G11D, honest negative R@20 0.002); LightGCN `[DEFERRED — not cheap on CPU]`.
- G-series numbers are **offline re-ranking within candidates** (not full-catalog) where noted; no real online lift, no production deployment, no RiskFrame-gold-complete claim.
- Co-occurrence (G15): **not** "significantly improved recall"; graph retrieval **not** proven.
