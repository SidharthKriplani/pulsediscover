# 69 — G18: Claim-Boundary Lock

*The locked, final set of what may and may not be said about PulseDiscover, by surface. Every safe claim maps to an artifact. Status: GOLD CANDIDATE (not gold_complete).*

## Resume-safe claims (each maps to an artifact)
- "Built an offline two-stage recommender (candidate generation → learning-to-rank) on a 2.56M-interaction Goodreads dataset." — G12–G14
- "Tuned ALS matrix factorization to a R@20 0.085 warm floor; built a content-hybrid cold-start candidate lane (cold R@20 0.241)." — D3A+/C1
- "Showed a LambdaMART ranker ~doubled single-retriever ranking within candidate sets (0.038→0.079)." — G12-B
- "Diagnosed candidate coverage as the binding bottleneck and lifted offline R@20 ~44% by widening candidate generation." — G12-B/G13
- "Built off-policy evaluation, position-bias correction, and a FAISS serving index; documented honest negatives (CPU sequence models, two-tower)." — O1/P1/G16/G11

## LinkedIn-safe claims
- "An offline, evidence-backed RecSys experimentation program: warm + cold candidate generation, learning-to-rank, OPE/position-bias governance, and a FAISS serving layer — built with disciplined claim boundaries and documented honest negatives."

## Interview-safe claims
- All resume/LinkedIn lines above, plus the G17 bank: ALS-beats-sequence reasoning, full-softmax 6.7× lever (still < MF), GPU-features-non-additive, coverage-is-the-bottleneck, governance-best-handled-upstream, FAISS-retains-not-improves, co-occ conditional.

## Forbidden claims (never say, even if asked directly)
- ❌ "Improved business metrics / drove engagement / lift." (no online A/B)
- ❌ "Deployed / in production / serving live traffic." (production-shaped only)
- ❌ "Beat ALS with a deep / sequence / graph / two-tower model." (none did)
- ❌ "Solved cold-start / creator fairness / catalog health." (candidate lane / reach, not solved)
- ❌ "FAISS improved recommendation quality." (retains recall; latency only)
- ❌ "90/10 is the proven best policy." (OPE can't separate it from control)
- ❌ "Full-catalog recall of 0.12+." (that's within-candidate re-ranking)
- ❌ "RiskFrame-gold complete." (gold candidate)

## Deferred claims (true-but-not-done; say "planned/next", never "done")
- Canonical full-softmax SASRec to convergence (GPU). — deferred
- LightGCN candidate source. — deferred (not cheap on CPU)
- Live A/B of the 90/10 policy. — specced (G4), not run
- Co-occurrence significance at larger N. — conditional
- Training-time creator fairness. — recommended, not built

## "How to say this under pressure" (compression lines)
- *Pushed on impact:* "It's offline and rigorous — I proved the bottleneck and fixed it in offline recall; I never claim online lift because I didn't run a live A/B."
- *Pushed on deep models:* "I tested them seriously; on this catalog MF won and the deep models were honest negatives — and I can show exactly why."
- *Pushed on 'is it real':* "Real 2.5M-interaction domain, real two-stage system, real serving index, documented negatives — offline by scope, not a toy."
- *Pushed on FAISS:* "Latency, not quality — exact FAISS returns identical results ~39% faster."

## "What not to say even if asked directly"
- Don't invent an online/business number. If asked "what was the lift?": "No online lift — offline only; the A/B is specced but not run."
- Don't claim a model beat ALS to sound impressive. The honest negative *is* the senior signal.
- Don't call within-candidate R@20 a catalog recall.
- Don't say "solved" for cold-start, fairness, or health.
- Don't upgrade "gold candidate" to "gold complete."

## G20 update (terminal method-depth closure)
- **NEW safe claim (artifact-backed by `g20_canonical_sasrec_convergence_report_v3.json` + plot):** *"Canonical full-softmax SASRec was trained to convergence and still did not beat ALS on the final offline evaluation (R@20 0.065 vs 0.085); matrix factorization is the right warm-retrieval floor on this catalog under the offline protocol."*
- **Deferred → now closed:** "canonical SASRec to convergence" moves from *deferred* to **done (terminal, outcome A)**. The method-depth blocker is closed; **G21 final audit is justified.**
- **Still forbidden (G20 unlocks none of these):** online lift · production · live A/B · gold_complete (pending G21) · "a deep model beat ALS" (it did not). LightGCN remains optional/deferred (second family; the terminal sequence result stands without it).
- **Evidence-cleanliness caveat:** the v3 JSON is `plot_reconstructed` (stale v1 JSON was rejected); secondary metrics pending the direct v3 JSON. Headline (R@20 0.065, valid convergence, clearly_loses) is artifact-backed.

## G21 final lock
- **Verdict: PATCH → `gold_candidate` (8.9).** Method-depth blocker closed; all 15 axes strong; all 10 G21 checks pass. Held from `gold_complete` by one evidence-packaging gap: the **G20 v3 artifact is `plot_reconstructed`** (secondary metrics pending), not the direct v3 JSON.
- **Single bounded action to `gold_complete`:** recover the direct v3 JSON (exact secondary metrics + direct provenance).
- **Still forbidden:** `gold_complete` (until that JSON), online lift, production, live A/B, deployed quality, deep-model-beats-ALS, solved cold-start/fairness/health, full-catalog for within-candidate.

**Lock status: post-G21. `gold_candidate` (8.9); method-depth CLOSED; `gold_complete` one artifact away. All safe claims artifact-backed.**

## V2 lane claim additions (G22; V1 boundary unchanged)
- **G31 NEW safe (proven, `g31_search_ir_report.json`):** *"Built a two-stage-search-style retrieval front end on the item catalog — BM25 lexical + dense semantic retrieval fused with RRF — evaluated with search-standard NDCG@K and MRR alongside Recall@K, using seed-item (query-by-document) queries; BM25 was the strongest single lane (R@20 0.0133 vs dense 0.0067) and RRF hybrid bought robustness, not a recall win. Re-narrates the existing LambdaMART LTR / FAISS / OPE work as a search ranking pipeline. Offline, served-c2 — not free-text query search, not online."*
- **G31 forbidden / honesty guards:** free-text query search engine (it's query-by-document) · in-ranker position-bias correction (only position-bias-aware *evaluation*) · graded NDCG (ours is single-relevant: one held-out gold) · "hybrid beats both lanes" (it doesn't here) · online lift · production search · any change to V1 RecSys claims. Pricing/forecasting deliberately not attempted.

- **NEW safe (proven, `g22_faiss_latency_quality_report.json`):** *"FAISS HNSW ef64 preserved candidate retrieval quality within measured tolerance (overlap@20 0.992, rec-Recall@20 degradation 0.0) while reducing p95 latency ~2.4× vs exact FlatIP and ~4.5× vs numpy brute-force (CPU in-sandbox); FlatIP exact reduces p95 ~47% at zero quality loss."*
- **Still forbidden (V2 too):** FAISS improved recommendation quality · production deployment · online lift · any change to the V1 ALS-warm-floor conclusion. CPU latency ≠ production benchmark.
- V1 status and all V1 claims are **unchanged** by V2.
- **G23 NEW safe (proven, `g23_serving_api_report.json`):** *"Built and load-tested a production-like recommender serving API with FAISS retrieval, versioned model/index loading, fallback behavior, structured logging, health checks, and latency/error monitoring hooks (warm p95 ~3.3 ms, 0% empty-response, graceful cold-user fallback; CPU in-sandbox)."*
- **G23 forbidden:** "deployed a production recommender system" · real users · online lift · HNSW as silent default. `production_claim_allowed: false`, `production_like_claim_allowed: true`.
- **G24 NEW safe (proven, `g24_cold_start_sparse_cohort_report.json`):** *"Evaluated cold-start and sparse-user fallback policies across defined cohorts, measuring quality degradation, coverage, fallback hit rate, and head/tail exposure — falling back to popularity collapses catalog coverage ~59× (3532→60 unique items) and personalization to ~0 for cold users, while ~57% of held-out items are item-cold-start with no collaborative signal."*
- **G24 honest nuance:** per-cohort Recall@20 is non-monotonic (cold gold is more popularity-predictable); recall UNDERSTATES the cold-start cost — coverage/personalization collapse is the cost.
- **G24 forbidden:** "solved cold-start" · "personalizes for unknown users" (it's popularity) · mixing warm-ALS + cold-fallback into one headline · "solved item cold-start" · any fairness claim · online/production user behaviour. `cold_start_solved:false`, `fairness_claimed:false`.
- **G25 NEW safe (proven, `g25_catalog_exposure_governance_report.json`):** *"PulseDiscovery G25 audits catalog exposure concentration across retrieval and fallback policies, measuring head/mid/tail exposure, catalog coverage, zero-exposure share, and concentration tradeoffs — all policies are head-concentrated (Gini>0.97), popularity fallback is near-degenerate (Gini 1.0, 99.9% zero-exposure), head items get ~10× their catalog base rate while long-tail get ~0.02×, and HNSW exposure ≈ exact."*
- **G25 forbidden:** fairness solved · creator fairness solved · marketplace fairness certified · online diversity improved · long-tail discovery solved · production exposure governance deployed · protected-class fairness analysis. This is exposure *concentration measurement*, not certification.
- **G26 NEW safe (proven, `g26_exposure_reranking_frontier_report.json`):** *"PulseDiscovery G26 evaluates exposure-aware reranking policies and quantifies the relevance/exposure tradeoff, showing which mitigation strategies improve catalog coverage or long-tail exposure without unacceptable retrieval-quality degradation — a head-item exposure cap was Pareto-dominant (Recall@20 +40%, coverage 7.3%→12.3%, long-tail exposure ~8×, Gini 0.982→0.968) on the warm cohort, while MMR/genre-diversification were honest negatives."*
- **G26 honest caveat:** the head-cap win is protocol-specific (warm cohort, single-held-out-gold where the true next-read is niche-skewed); it must be tuned, not assumed universal.
- **G26 forbidden:** fairness solved · creator fairness solved · marketplace fairness certified · long-tail discovery solved · online diversity improved · production exposure governance deployed · protected-class fairness analyzed. Mitigation *attempted and measured*, not certified/deployed.
- **RiskFrame-Gold Audit lock (verdict PATCH):** G26 is **BLOCKED** until G26A (architecture consolidation + OPE logging design) is accepted; then unblocks as G26B. **No OPE/IPS/SNIPS/DR number may be claimed until propensities are logged.** Recall is split-protocol: V1 `d3aplus` R@20 0.0846 vs served `c2` ~0.0375 are `NOT_COMPARABLE_ACROSS_PROTOCOLS` — never substitute one for the other. G26 rerankers are **heuristic**, not a learned LTR ranker. Head-cap win is **protocol-specific** (warm cohort, single-held-out-gold). V1 stays `gold_candidate` 8.9. Full claim register: `docs/PULSEDISCOVERY_RISKFRAME_GOLD_AUDIT.md` + `outputs/evidence/pulsediscovery_riskframe_gold_audit.json`.
- **G26B NEW safe (proven, `g26b_rerank_integration_report.json`):** *"I integrated a config-flagged heuristic reranker into the served recommender path and measured the relevance/exposure/latency tradeoff against the served c2 baseline (coverage 6.9%→9.8%, long-tail exposure ~4.3×, Gini 0.983→0.977, no relevance loss, +0.3ms p95). This gives interpretable exposure control, not learned ranking. The next modernization gate is semantic content retrieval for cold-start recovery."* G26 quarantine **CLOSED**; rerank ships as a config option, **default OFF**.
- **G26B forbidden:** "I built a ranker / improved recommendations / solved long-tail / added AI" (weak); "learned LTR deployed / cold-start solved / online lift proven / OPE executed / fairness certified" (unsafe). Served `c2` metrics only — never mixed with V1 `d3aplus` 0.0846.
- **G27 NEW safe (proven, `g27_semantic_retrieval_report.json`):** *"I added a semantic content-retrieval lane (dense MiniLM item embeddings + FAISS content index) as a candidate source for item-cold-start and sparse-catalog scenarios; evaluated it vs popularity and the served c2 baseline on cold/sparse/warm cohorts; measured relevance, coverage, long-tail exposure, concentration, and latency; and decided it belongs as a cold-start/tail COMPLEMENT (it uniquely reaches unseen items — reachability 54.6%, coverage 62.6% vs ALS 8.1%, Gini 0.73 vs 0.98 — but loses on head relevance, so NOT a warm-path replacement). Offline candidate-generation evidence, not online lift or solved cold-start."*
- **G27 forbidden:** "cold-start solved" · "LLM recommender" · "semantic taste understanding" · "AI-powered recommendations" · "semantic recommender" · online lift · production deployed · learned ranker · OPE executed · fairness certified · two-tower/GNN/bandit. Weak wording to avoid: "I added embeddings / used an LLM / improved cold-start." Served-c2/content metrics only — never mixed with V1 `d3aplus` 0.0846.
- **G28 NEW safe (proven, `g28_final_ranker_fusion_decision.json`):** *"I built a source-aware fusion/ranking layer over ALS, semantic content retrieval, and popularity candidates; evaluated heuristic fusion and learned rankers across warm/head, mid, long-tail and item-cold-start cohorts (relevance, coverage, concentration, cold-start reachability, latency). A LightGBM LambdaMART fusion ranker beat ALS-only on the held-out test split (Recall@20 0.036 vs 0.022) while recovering cold-start (0.032 vs 0); the shipping policy keeps ALS the warm-path primary and uses semantic as a measured cold-start/tail source. Offline ranking/fusion evidence on a small positive set (81 test positives) — not online lift or solved cold-start."*
- **G28 caveat (must attach):** learned-beats-ALS is claimed ONLY on the offline test split, small positive count (directional, consistent across 4 model families); RRF is the best non-learned fusion; ALS-first fusion fails on cold-start; description text marginally hurts (title+shelves kept). Served-c2 only — never mixed with V1 `d3aplus` 0.0846.
- **G28 forbidden:** online lift · production · cold-start solved · LLM recommender · semantic taste · OPE executed · fairness certified · "learned ranker beats ALS" without the offline-test-split qualifier. Weak wording to avoid: "I built a modern recommender / used embeddings and ranking / improved recommendations / AI-powered."
- **G29 NEW safe (proven, `g29_ope_execution_report.json`):** *"I executed the off-policy-evaluation pipeline end-to-end — a stochastic logging policy with recorded propensities, then IPS/SNIPS/DR estimators that recover a known target-policy value on held-out data (DR lowest-bias, SNIPS lowest-variance, IPS showing the importance-weight variance problem); the estimators correctly rank RRF-fusion above ALS. OPE methodology demonstrated offline with synthetic propensities and a proxy reward — not a real online-lift estimate."* This moves OPE from `BLOCKED` (G26A) to `DEMONSTRATED-OFFLINE`.
- **G29 forbidden:** online lift proven · real off-policy value estimated · production · learned-ranker-beats-ALS-online · fairness certified · cold-start solved. Propensities are synthetic, reward is a held-out proxy.
- **G30 FINAL LOCK (Gold Pass 3/3):** PulseDiscover is **OFFLINE GOLD-COMPLETE / INTERVIEW-READY** (RiskFrame 9.1). The canonical interview wording is `docs/PULSEDISCOVERY_INTERVIEW_KIT.md`; the claim ladder `docs/G26A_CLAIM_UNLOCK_TABLE.md`; registry `docs/G26A_MODEL_PROTOCOL_REGISTRY.md`. V1 stays `gold_candidate` 8.9 (locked, not upgraded). **Permanently forbidden (out of scope by design):** online lift · real users · real-traffic/real-reward OPE · production deployment · learned-ranker-beats-ALS online · fairness certification · gold_complete-with-online-evidence. Everything claimable is in the Interview Kit and is artifact-backed and bounded.
- **G26A claim-ladder lock (the canonical claim register going forward):** the authoritative, interview-maximal-but-bounded claim ladder lives in `docs/G26A_CLAIM_UNLOCK_TABLE.md`; model/metric pin-board in `docs/G26A_MODEL_PROTOCOL_REGISTRY.md`; OPE validity rule in `docs/G26A_OPE_LOGGING_SCHEMA.md`; interview wording in `docs/G26A_INTERVIEW_DEFENSE_LINES.md`. **Cold-start, long-tail, and exposure claims must use the *system-level* maximal wording (e.g. "handled cold-start at the offline/serving-system level…"), NOT apologetic "did not solve" phrasing — while keeping the boundaries.** G26 = `QUARANTINED_G26B_PENDING_G26A`.
