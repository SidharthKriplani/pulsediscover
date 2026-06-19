# G26A — Claim-Unlock Table (the claim ladder)

*Every PulseDiscovery claim, classified, with the **strongest interview-safe wording** (not the weakest cautious wording). Status legend: **LOCKED SAFE** (say freely) · **SAFE WITH BOUNDARY** (say with the boundary attached) · **BUILT NOT LICENSED** (built, don't claim as shipped/valid yet) · **PLANNED** (not built) · **BLOCKED** (needs a specific unlock) · **DELETE** (never say).*

> Principle: a safe claim is **maximal and bounded**, never apologetic. Do not say "I didn't solve cold-start." Say "I handled cold-start at the offline/serving-system level and named the exact production gap."

---

### 1. Offline recommender decision dossier — **LOCKED SAFE**
- **Strongest wording:** "I built an offline, honesty-gated recommender **decision system** on 2.56M real Goodreads interactions — it decides which retrieval/serving/fallback/exposure policy ships and proves which must not."
- Weak to avoid: "I did some recommender experiments." · Unsafe: "I built a production recommender." · Evidence: all gate reports. · Next unlock: — (already strong).

### 2. ALS warm floor — **LOCKED SAFE**
- **Strongest:** "ALS is the warm-retrieval floor on this catalog — I tuned it (V1 R@20 0.0846) and *earned* that conclusion by beating deeper models, not by skipping them."
- Weak: "ALS worked okay." · Unsafe: "ALS is the best possible model." · Evidence: `d3aplus_als_tune.json`. · Boundary: V1 tuning protocol.

### 3. SASRec converged and lost — **SAFE WITH BOUNDARY**
- **Strongest:** "I trained a canonical full-softmax SASRec **to convergence** and it still lost to MF (R@20 0.065 vs 0.0846) — depth is not the lever on this sparse-history catalog, and I can show exactly why."
- Boundary: headline is plot_reconstructed (secondary metrics lost). · Weak: "SASRec didn't work." · Unsafe: "SASRec never works." · Evidence: `g20_v3.json`. · Next unlock: recover direct v3 JSON → gold_complete packaging.

### 4. FAISS serving frontier — **LOCKED SAFE**
- **Strongest:** "I built the latency–quality frontier for serving retrieval: FlatIP exact is lossless and ~47% faster, HNSW ef64 is near-lossless (overlap 0.992) at ~2.4×, and I **rejected IVF** because aggressive nprobe collapsed candidate overlap — speed that destroys recall is a bug."
- Weak: "I used FAISS." · Unsafe: "FAISS improved recommendation quality." · Evidence: `g22.json`. · Boundary: latency, not quality.

### 5. Production-LIKE API — **SAFE WITH BOUNDARY**
- **Strongest:** "I built and load-tested a production-**like** serving API — versioned model/index loading, a full fallback tree, structured logging, health and latency/error monitoring (warm p95 ~3.3ms local, 0% empty across failure modes)."
- Boundary: local TestClient, not deployed. · Weak: "I made an endpoint." · Unsafe: "I deployed to production / served real users." · Evidence: `g23.json`.

### 6. Cold-start handling — **SAFE WITH BOUNDARY** *(the rewrite the user asked for)*
- **Strongest:** "I handled cold-start at the offline/serving-system level: I separated unknown, sparse, low-history, and warm cohorts; evaluated popularity/content/item-sim/blended fallback policies; tested serving behavior across all of them; and found popularity fallback returns non-empty results but **collapses personalization (0.68→0.001) and catalog coverage (~59×)**. The production gap is live-user / new-item-launch evidence — not conceptual understanding."
- Weak/forbidden to *lead* with: "I did not solve cold-start." · Unsafe: "cold-start solved." · Evidence: `g24.json`.

### 7. Cold-start SOLVED — **DELETE**
- Never claim. Replace with #6.

### 8. Fallback quality evaluated — **LOCKED SAFE**
- **Strongest:** "I treated fallback **existence** and fallback **quality** as different questions — 0% empty is table stakes; I measured what the user actually gets, and showed popularity fallback is an emergency floor, not a healthy default."
- Evidence: `g24.json`, `g23.json`.

### 9. Catalog exposure governance — **LOCKED SAFE**
- **Strongest:** "I audited catalog exposure as a first-class concern: full-catalog Gini, coverage, zero-exposure share and head/tail lift across every policy — popularity fallback is near-degenerate (Gini 1.0, 99.9% zero-exposure), head items get ~10× their catalog share while the long tail gets ~0.02×, and HNSW is exposure-neutral vs exact."
- Weak: "I looked at diversity." · Unsafe: "fairness certified." · Evidence: `g25.json`.

### 10. Fairness SOLVED — **DELETE**
- Never. Say "catalog **exposure concentration** governance," explicitly not protected-class fairness.

### 11. Long-tail exposure mitigation — **BUILT NOT LICENSED (QUARANTINED_G26B)**
- **Strongest (allowed now):** "I built exposure-aware rerankers and mapped the relevance/exposure frontier — a head-item cap was Pareto-dominant on the warm cohort (Recall@20 +40% **and** long-tail exposure ~8×), while MMR and genre-diversification were honest negatives. It's quarantined pending architecture consolidation before I license it."
- Unsafe: "I improved long-tail discovery." · Evidence: `g26.json` (quarantined). · Next unlock: **G26B**.

### 12. Long-tail discovery SOLVED — **DELETE**

### 13. Heuristic reranker — **LOCKED SAFE (named honestly)**
- **Strongest:** "My V2 rerankers are **heuristic** (head-cap, tail/novelty boost, MMR) — deliberately interpretable levers to trace the tradeoff before investing in a learned ranker."
- Unsafe: calling them a learned LTR. · Evidence: `exposure_aware_reranking.py`.

### 14. Learned LTR — **PLANNED / NOT BUILT**
- **Strongest:** "A LambdaMART/LambdaRank lane is specced as the next ranking investment; the V2 reranking is heuristic by design." · Unsafe: "I shipped a learned ranker in V2." · Next unlock: G26B/BeastMax build.

### 15. OPE readiness — **BLOCKED (design done in G26A, data not logged)**
- **Strongest:** "I designed the propensity-logging schema that makes IPS/SNIPS/DR valid — request/impression/position/π_log/reward — so the system is **OPE-ready by design**, and I refuse to report an OPE number until those propensities are actually logged."
- Unsafe: any IPS/SNIPS/DR value now. · Evidence: `G26A_OPE_LOGGING_SCHEMA.md`. · Next unlock: log propensities in a serving run.

### 16. Valid IPS / SNIPS / DR — **BLOCKED**
- **Strongest:** "Not yet valid — no logged propensities. SNIPS would be my first estimator for variance control once logging exists; DR after I also have a reward model." · Unsafe: any number. · Next unlock: logging + a counterfactual run.

### 17. Online experiment readiness — **PLANNED (card in G26A)**
- **Strongest:** "I have an online-experiment-readiness view: the guardrail metrics (relevance, coverage, Gini, fallback-rate, p95) and the logging needed to run a real A/B." · Unsafe: "I ran an A/B." · Next unlock: live traffic.

### 18. Online lift — **DELETE (until a real A/B)**
- Never claim. "No online lift — offline by scope; the A/B is specced, not run."

### 19. Production deployment — **DELETE (production-LIKE only)**
- Never claim "deployed." Always "production-like, load-tested locally."

---

## Claim ladder (how strong I can go, today → after next gates)
1. **Today (LOCKED/BOUNDED):** decision dossier · ALS floor · SASRec-converged-and-lost · FAISS frontier · production-like API · cold-start *handled at system level* · fallback quality · exposure governance · heuristic rerank frontier · **OPE-ready by design**.
2. **After G26B:** licensed exposure-mitigation policy in the serving path (offline-shaped) · head-cap/tail-boost as a shippable lever.
3. **After OPE logging + counterfactual run:** valid SNIPS/DR off-policy value estimate.
4. **After live A/B:** online lift · production deployment · real-user behaviour.

**Acceptance check:** this table gives a *ladder* and *maximal-but-bounded* language — not legal-safe hedging.
