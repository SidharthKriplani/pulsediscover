# PulseDiscover — Unified Deep Defense Kernel (V1 + V2)

*The single study deck. Consolidates every method's first-principles defense across V1 (G11–G18) and the V2 expansion (G22–G29) in one place. Format per method: **problem → core idea/objective → assumptions → failure modes → why-over-alternatives → when-alternatives-win → in-PulseDiscover (artifact + number) → hard interview Q → safe answer.** V1 cards remain in `66_G17_METHOD_DEFENSE_CARDS.md` / `67_G17_INTERVIEW_DEFENSE_BANK.md`; this deck restates the V1 headline cards briefly and adds the V2 methods in full. Claim wording: `PULSEDISCOVERY_INTERVIEW_KIT.md`. Registry: `G26A_MODEL_PROTOCOL_REGISTRY.md`.*

## Defense chain (every method survives all seven)
theory → product decision → data/eval design → implementation artifact → evidence → business consequence → interview-safe claim.

## Unified narrative (V1 + V2, binding)
1. **ALS f64 is the warm-retrieval floor** (V1 d3aplus R@20 0.085; served c2 ~0.0375 — different protocol, never conflate).
2. **Sequence/graph/two-tower were tested and lost** — canonical full-softmax SASRec converged and still lost (0.065 < 0.085); TwoTower honest negative.
3. **Candidate coverage was the binding lever** — widening lifted offline recall ~44%.
4. **LambdaMART LTR was the V1 ranking win within candidates** (0.038→0.079).
5. **FAISS = latency, not quality** — FlatIP exact lossless ~47% faster; HNSW ef64 near-lossless ~2.4×; IVF rejected (overlap collapse).
6. **Serving is production-shaped, not deployed** — FastAPI + fallback + monitoring, warm p95 ~3.3 ms local, 0% empty.
7. **Cold-start handled at system level, not solved** — popularity fallback collapses coverage ~59×; 57% of held-out items are item-cold-start (offline-split sense).
8. **Exposure is concentration governance, not fairness** — all policies Gini>0.97; popularity degenerate (1.0); head ~10× base rate.
9. **Semantic content retrieval is the cold-start/tail complement** — MiniLM+FAISS uniquely reaches unseen items (reachability 54.6%), coverage 62.6% vs 8.1%, but loses head relevance (0.010 vs 0.087).
10. **Learned ALS+semantic fusion beat ALS-only on held-out test** (0.036 vs 0.022) + recovered cold-start (0.032 vs 0); ALS & semantic candidates ~disjoint (overlap 6,749).
11. **OPE executed offline** — IPS/SNIPS/DR recover a known value (DR lowest-bias, SNIPS lowest-variance); synthetic propensities + proxy reward, not online.
12. **No online lift, no production, no fairness certification, no real-traffic OPE** — permanently out of scope by design.

---

# V1 headline cards (brief — full versions in 66/67)

**ALS (implicit MF).** Obj `min Σ c_ui(p_ui − xᵤᵀyᵢ)² + λ‖·‖²`, c_ui=1+αr. Alternating closed-form ridge solves. Fails on cold users/niche tail. Warm floor on this catalog. *Q: why MF over a transformer? A: a converged full-softmax SASRec still lost — depth isn't the lever on sparse book histories.*

**SASRec (self-attentive next-item).** Causal self-attention; full-softmax vs sampled-softmax was the 6.7× lever; still < ALS. *Q: undertrained? A: trained to convergence (~95 ep, eval-R@20 plateau); the loss objective was the lever, not epochs.*

**TwoTower / LightGCN.** Two-tower honest negative (0.0020); LightGCN deferred (CPU graph cost). *Q: why did two-tower fail? A: item-ID mean-pooling loses the co-occurrence structure ALS factorizes on short histories.*

**LambdaMART LTR (V1).** Pairwise/listwise NDCG-surrogate gradient boosting over candidate features; ~2× within-candidate ranking. *Q: is that catalog recall? A: no — within-candidate re-ranking; I attacked the coverage ceiling separately (G13).*

**FAISS family.** FlatIP exact / HNSW navigable-small-world / IVF coarse-quantizer. *Q: exact or approx in prod? A: FlatIP exact default (lossless, ~47% faster); HNSW ef64 explicit scale option; IVF rejected for overlap collapse.*

---

# V2 method cards (full)

## 1. Dense semantic content retrieval (G27)
- **Problem:** item-cold-start — 57% of held-out items have zero collaborative signal, so ALS/popularity reach them with probability ~0.
- **Core idea:** embed each item's *content* (title + genre shelves) with sentence-transformers MiniLM (384-d, L2-normalized); FAISS IndexFlatIP (cosine-equiv); item-to-item retrieval from the user's last-read item.
- **Assumptions:** content similarity ≈ taste similarity at the genre/topic level; metadata text is informative (100% coverage here).
- **Failure modes:** content neighbours can be series/near-duplicates not novel discovery; needs a query item (zero-history users excluded); genre-noisy shelves pull off-topic items; weak on head relevance (collaborative signal dominates there).
- **Why over alternatives:** the only candidate source that *structurally can* reach unseen items; cheap, deterministic, local, sub-ms retrieval.
- **When alternatives win:** warm/head users — ALS is far stronger (0.087 vs 0.010).
- **In-PulseDiscover:** `g27_semantic_retrieval_report.json` — reachability 54.6% (ALS/pop 0), coverage 62.6% vs 8.1%, Gini 0.73 vs 0.98, p95 ~0.23 ms.
- **Hard Q:** "Is this an LLM recommender?" **A:** "No — embedding-based candidate generation. No generation, no chat, no 'semantic taste'. It's a content lane that reaches items collaborative filtering can't."

## 2. Candidate fusion — RRF + the ALS-first failure (G28)
- **Problem:** how to combine two nearly-disjoint candidate sets (ALS overlap with semantic = 6,749 of ~880k) without losing ALS relevance or starving cold-start.
- **Core idea:** Reciprocal Rank Fusion `score(i)=Σ_s 1/(c+rank_s(i))` over ALS+semantic; rank by fused score.
- **Assumptions:** rank position is a comparable cross-source signal (scores aren't directly comparable).
- **Failure modes:** **ALS-first union fails** — ALS saturates the top-20 so semantic cold-start items never compete (cold-start recall stays 0); weighted-score fusion fails the same way unless scores are normalized and balanced.
- **Why over alternatives:** RRF is score-scale-free, needs no training, recovers cold-start (0→0.017) and ~4× coverage at ~95% of ALS relevance — the best *non-learned* fusion.
- **When alternatives win:** a learned ranker (below) beats RRF when a candidate-feature table + labels exist.
- **In-PulseDiscover:** `g28_fusion_baseline_report.json`.
- **Hard Q:** "Why did ALS-first fail?" **A:** "ALS fills all 20 slots with in-catalog items; semantic never competes. Any fusion that doesn't let semantic items compete for slots inherits that — RRF/source-balanced fix it."

## 3. Learned fusion ranker — LightGBM LambdaMART (G28)
- **Problem:** can a learned ranker exploit the disjoint ALS+semantic pool to beat either source alone?
- **Core idea:** LambdaMART (gradient-boosted trees, lambdarank/NDCG objective, query groups = users) over candidate features (ALS rank/score, semantic rank/score, popularity, item-tier, source flags).
- **Assumptions:** enough positive labels per query to learn a ranking; features carry cross-source signal.
- **Failure modes:** extreme label sparsity (81 test positives, single held-out gold) → overfit risk and small-sample noise; needs leakage-safe split.
- **Why over alternatives:** proper ranking objective + query structure; beats classifiers on cold-start/coverage/NDCG; champion over logistic/RF/HistGBM.
- **When alternatives win:** RRF if no labels/feature table; logistic edged raw recall but lost coverage/NDCG.
- **In-PulseDiscover:** `g28_learned_ranker_tournament.json` — test R@20 0.036 vs ALS 0.022, cold-start 0.032 vs 0; consistent across 4 model families (split by user-hash, val-selected, test-held-out).
- **Hard Q:** "You said learned beats ALS — prove it." **A:** "On the held-out test split only, 0.036 vs 0.022, directionally consistent across four model families — but on 81 positives, single-gold, offline. I don't claim it beats ALS in production; that needs the OPE/online evidence."

## 4. Exposure governance — Gini / Lorenz / base-rate lift (G25)
- **Problem:** a recall win can quietly wreck catalog health; need to measure concentration.
- **Core idea:** over the *full* catalog (zeros included) compute exposure Gini `2Σi·xᵢ/(nΣxᵢ)−(n+1)/n`, Lorenz curve, top-x% share, and tier exposure-lift vs catalog base rate.
- **Assumptions:** exposure counts over a representative user sample reflect serving exposure; tiers (head/mid/tail) from train-popularity percentiles.
- **Failure modes:** Gini alone hides *which* tail — pair with tier-lift; tiers aren't editorial categories.
- **Why over alternatives:** concentration + coverage are the catalog-health signals recall ignores.
- **When alternatives win:** for relevance you still need recall — these are complementary, not substitutes.
- **In-PulseDiscover:** `g25_catalog_exposure_governance_report.json` — all policies Gini>0.97, popularity 1.0 (99.9% zero-exposure), head ~10× base rate, tail ~0.02×; HNSW exposure-neutral (0.986≈0.986).
- **Hard Q:** "Is this fairness?" **A:** "No — catalog *exposure concentration* governance, explicitly not protected-class fairness. I don't certify fairness; I measure concentration as a ship-gate metric."

## 5. Cold-start cohorts + fallback quality (G24)
- **Problem:** "fallback exists" ≠ "fallback is good"; measure degradation by cohort.
- **Core idea:** cohort users (unknown/sparse 1-2/low 3-5/warm 6+); evaluate popularity/content/item-sim/blended; report recall + coverage + personalization + exposure separately (never mix warm-ALS with cold-fallback).
- **Failure modes / honest nuance:** per-cohort recall is *non-monotonic* (cold gold is more popularity-predictable) → recall *understates* cold-start cost; the real cost is coverage/personalization collapse.
- **In-PulseDiscover:** `g24_cold_start_sparse_cohort_report.json` — popularity fallback collapses coverage ~59× (3532→60 unique) and personalization 0.68→0.001.
- **Hard Q:** "Did you solve cold-start?" **A:** "I handled it at the offline/serving-system level and named the gap — live new-item-launch data. Popularity fallback is non-empty but non-personalized; I measured exactly how much it costs."

## 6. Off-policy evaluation — IPS / SNIPS / DR (executed, G29)
- **Problem:** estimate a target policy's value from data logged under a different (logging) policy, without an online test.
- **Core idea:** stochastic logging policy π₀ with recorded propensities (ε-uniform + softmax → overlap). `IPS = mean[1{a=πe(x)}/π0]·r`; `SNIPS` self-normalizes by Σweights; `DR = mean(r̂(x,πe)) + mean(w·(r−r̂(x,a)))` with a reward model r̂.
- **Assumptions:** overlap (π0>0 on πe's support); correct propensities; for DR, a calibrated reward model.
- **Failure modes:** **IPS high variance** (over-estimated ~2× here at modest ESS); **DR breaks with a mis-calibrated reward model** (a `class_weight=balanced` r̂ predicting ~0.5 vs true ~0.01 base rate drove DR negative — fixed with a base-rate-calibrated logistic); low ESS → noisy estimates.
- **Why over alternatives:** SNIPS lowest-variance when no reward model; DR lowest-bias when a good r̂ exists.
- **When alternatives win:** a real online A/B beats all OPE — OPE is the offline proxy.
- **In-PulseDiscover:** `g29_ope_execution_report.json` — estimators recover the known precision@1 (DR 0.0089 vs true 0.0106; IPS over-estimates; SNIPS stable) and rank RRF-fusion > ALS.
- **Hard Q:** "What's your off-policy value?" **A:** "I validated that IPS/SNIPS/DR recover a *known* value on offline self-logged data with a proxy reward — methodology demonstrated. A real off-policy number needs logged real-traffic propensities, which I don't fabricate."

---

## The 12 attacks a skeptical interviewer will run (one-line answers)
1. *Impact/lift?* → "Offline by scope; no online lift; A/B specced not run."
2. *Deep model beat ALS?* → "No — converged SASRec lost; honest negative."
3. *In production?* → "Production-like, load-tested locally; not deployed."
4. *Cold-start solved?* → "Handled at system level; gap is live launch data."
5. *Fairness?* → "Exposure concentration, not protected-class fairness; uncertified."
6. *Recall is low?* → "Which protocol? d3aplus 0.085 vs served c2 ~0.038 — different builds, registry-separated; relative structure is the signal."
7. *LLM recommender?* → "Embedding candidate generation, not chat/RAG/LLM-rerank."
8. *Why fusion?* → "ALS & semantic overlap on ~6,749 of ~880k — complementary; learned ranker picks best across both."
9. *Learned beats ALS — real?* → "Held-out test split only, 81 positives, offline, cross-family consistent; not a production claim."
10. *Your OPE number?* → "Estimators validated to recover a known value offline; real number needs real propensities."
11. *DR went negative once?* → "Mis-calibrated reward model; I caught it and fixed with a base-rate-calibrated r̂ — DR is then lowest-bias."
12. *Why not GNN/two-tower/bandits?* → "T3/deferred — on a catalog where converged SASRec lost to MF they're unlikely to pay and expensive to defend offline; I built where it's measured to matter (cold-start, catalog health)."

## Kill conditions (what would falsify each claim — shows I know the limits)
ALS-floor: a properly-tuned sequence/graph model beating it on this exact protocol. Fusion-win: it not holding on a larger positive set or a different relevance target. Semantic value: cold-start reachability not translating to any online engagement. OPE: ESS too low / no overlap → estimates uninformative. Each is stated, not hidden.

## Failures I'm proud of + serving stress test
The catch-and-correct stories (SASRec false convergence, eval-leakage, stale-script contamination, IVF "metric can lie", BM25-beats-dense-on-short-text) and the empirical serving edge-case stress test (with two load-path fixes + Cloud Run memory sizing) live in **`docs/PULSEDISCOVERY_FAILURES_AND_HARDENING.md`** (`outputs/evidence/g_serving_stress_test.json`). These are often the strongest interview material — lead with them when asked "tell me about a time something went wrong."

**Status: unified defense built — all V1 + V2 methods defended in one deck, claim-bounded, with kill conditions. Interview-ready.**
