# PulseDiscover — Interview Kit (locked at G30)

*The single ready-to-use package: resume bullets, the 60-second pitch, the architecture story, the strongest-safe claims, and the answers to the hard questions. Everything here is artifact-backed and claim-bounded. Offline by scope; online/production explicitly out of scope.*

---

## One-line identity
**PulseDiscover is an offline, honesty-gated recommender decision system on 2.56M Goodreads interactions: it decides which retrieval, serving, fallback, fusion and exposure-governance policy should ship — and proves which must not — by catching offline-metric bias, candidate-coverage gaps, latency–quality tradeoffs, cold-start degradation, and catalog concentration before they reach users.**

## Resume bullets (safe, artifact-backed)
- Built an offline two-stage recommender **decision system** on a real **2.56M-interaction** Goodreads dataset (candidate generation → FAISS retrieval → fusion/ranking → fallback → serving → logging → evaluation).
- Established **ALS matrix factorization as the warm-retrieval floor** and *earned* it: trained a **canonical full-softmax SASRec to convergence** that still lost (R@20 0.065 vs 0.085) — documented as an honest negative.
- Diagnosed **candidate coverage** as the binding bottleneck and lifted offline recall **~44%** by widening candidate generation.
- Built a **FAISS latency–quality frontier**: FlatIP exact lossless and ~47% faster; HNSW ef64 near-lossless (overlap 0.992) ~2.4×; **rejected IVF** for candidate-overlap collapse.
- Built and **load-tested a production-like FastAPI + FAISS serving API** with versioned model/index loading, a full fallback tree, structured logging, health and latency/error monitoring (warm p95 ~3.3 ms local, **0% empty**).
- Quantified **cold-start fallback quality** and **catalog exposure governance** (Gini, coverage, head/tail lift) — showing popularity fallback collapses catalog coverage **~59×**.
- Added a **dense semantic content-retrieval lane** (MiniLM + FAISS, 41,866 items) that uniquely reaches **item-cold-start** items (reachability 54.6%) and expands coverage **62.6% vs 8.1%**.
- Built a **source-aware fusion + learned-ranking layer**; a **LightGBM LambdaMART fusion beat ALS-only on the held-out test split** (Recall@20 0.036 vs 0.022) and recovered cold-start (0.032 vs 0).
- **Executed off-policy evaluation** end-to-end (logging policy + recorded propensities + **IPS/SNIPS/DR**), validating the estimators recover a known policy value (DR lowest-bias, SNIPS lowest-variance).

## 60-second pitch
"PulseDiscover is an offline, honesty-gated recommender decision system on 2.56M real Goodreads interactions. ALS is the warm-retrieval floor — I earned that by training a converged SASRec that still lost. I built a production-like serving spine (FastAPI + FAISS, fallback, monitoring), then measured the things that actually break recommenders: cold-start degradation, catalog concentration, and latency–quality. I added a semantic content lane that uniquely reaches cold-start items, fused it with ALS into a learned ranker that beat ALS-only on the held-out test split, and executed off-policy evaluation to show I can estimate policy value the right way. It's offline by scope — I never claim online lift; the next step with production access is logged propensities from real traffic and a live A/B."

## 2-minute architecture story
Candidate generation (ALS / semantic / popularity / co-occurrence) → FAISS retrieval (FlatIP default, HNSW scale, IVF rejected) → fusion + learned ranker (RRF / LambdaMART) → config-flagged heuristic exposure rerank (head-cap) → fallback tree (never empty) → FastAPI serving with versioned loader + structured logging + monitoring → leakage-safe cohort/exposure/OPE evaluation → ship/no-ship decision behind a claim ladder. Everything runs on the served `c2` ALS model; V1's `d3aplus` tuning number is quarantined in a registry so V1/V2 metrics are never conflated.

## Strongest-safe claims (with the boundary attached, never apologetic)
- **Cold-start:** "I handled cold-start at the offline/serving-system level — cohorted unknown/sparse/low/warm users, measured fallback quality (popularity collapses personalization 0.68→0.001 and coverage ~59×), and added a semantic lane that uniquely reaches unseen items. The gap is live new-item-launch data, not understanding."
- **Fusion/ranking:** "ALS and semantic candidates are nearly disjoint, so I fused them; a LambdaMART fusion beat ALS-only on the held-out test split and recovered cold-start — offline, small positive set, directionally consistent across four model families."
- **Exposure:** "Catalog **exposure concentration** governance — explicitly not protected-class fairness — Gini/coverage/tail-lift across policies, and I showed HNSW is exposure-neutral."
- **OPE:** "I executed the OPE pipeline and validated IPS/SNIPS/DR recover a known value; SNIPS for variance, DR for bias. I won't report a real off-policy number without logged real-traffic propensities."
- **Serving:** "Production-*like*, load-tested locally — not deployed."

## Hard questions → short answers
1. *Impact / lift?* "Offline by scope — no online lift claimed; the A/B is specced, not run."
2. *Did a deep model beat ALS?* "No — a converged SASRec lost. The honest negative is the senior signal."
3. *Is it in production?* "Production-like, load-tested locally. I don't claim deployment I didn't do."
4. *Solved cold-start?* "No — handled at the system level and named the one missing piece: live launch data."
5. *Fairness?* "Exposure-concentration governance, not protected-class fairness; I don't certify fairness."
6. *Recall looks low?* "Which protocol? V1 d3aplus 0.085 vs served c2 ~0.038 are different builds — I keep a registry so they're never conflated. Relative cross-policy structure is the signal."
7. *LLM in it?* "Embedding-based content retrieval for cold-start — not a chatbot, not an LLM reranker, not 'semantic taste'."
8. *Why fusion?* "ALS and semantic candidates overlap on only ~6,749 of ~880k — complementary, so a learned ranker picks the best across both."
9. *Your OPE number?* "I validated the estimators recover a known value offline; a real off-policy estimate needs logged real-traffic propensities."
10. *What would you build next with production access?* "Propensity logging from real traffic → real IPS/SNIPS/DR → a canary + live A/B of the fusion policy and the exposure rerank, with the G3 risk-register monitors wired to alerts."

## Boundaries (never say)
Online lift · production deployment · real users · cold-start solved · long-tail discovery solved · fairness certified · LLM recommender · semantic taste · learned-ranker-beats-ALS-online · real off-policy value · gold_complete-with-online-evidence.

## Evidence map (where each claim is backed)
ALS floor & SASRec negative → G20/`d3aplus_als_tune.json`,`g20_v3.json` · FAISS frontier → `g22_*` · serving → `g23_*` · cold-start → `g24_*` · exposure → `g25_*` · rerank → `g26b_*` · semantic → `g27_*` · fusion → `g28_*` · OPE → `g29_*` · audit → `g30_final_gold_audit.json`. Claim ladder: `docs/G26A_CLAIM_UNLOCK_TABLE.md`. Registry: `docs/G26A_MODEL_PROTOCOL_REGISTRY.md`.

## Technical defense (study deck)
For mechanism-level cross-examination (algorithm, assumptions, failure modes, kill conditions) on every method V1+V2, study `docs/PULSEDISCOVERY_UNIFIED_DEFENSE_KERNEL.md` — the single consolidated deck (ALS, SASRec, two-tower, LambdaMART, FAISS family, semantic retrieval, RRF/learned fusion, exposure governance, cold-start cohorts, IPS/SNIPS/DR). V1 detail also in `66/67_G17_*`.

**Status: interview-ready (offline gold-complete). Use freely within these boundaries.**
