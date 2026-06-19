# G26A — Interview-Maximal Defense Lines

*Strong, evidence-bounded answers. Every answer follows: **"I built X, measured Y, learned Z, decided W, and the next evidence needed for the production claim is Q."** Lead with the strength; attach the boundary as a sign of seniority, never as an apology.*

---

## Cold-start
"I handled cold-start at the offline/serving-system level. I **built** cohorts — unknown, sparse (1–2), low-history (3–5), warm (6+) — and **measured** popularity/content/item-similarity/blended fallback policies, then tested all of them through the live service. I **learned** that popularity fallback returns non-empty results but collapses personalization from 0.68 to ~0.001 and catalog coverage ~59× (3,532→60 unique items), and that recall is actually non-monotonic because cold users' next item is more popularity-predictable. I **decided** on a blended fallback tree that never returns empty while preserving warm personalization. The **next evidence** for a production claim is live new-user / new-item-launch data — that's the gap, not the concept."

## Long-tail
"I **built** exposure-aware rerankers and **measured** a relevance/exposure frontier: a head-item cap was Pareto-dominant on the warm cohort — Recall@20 +40% *and* long-tail exposure ~8× — because popular items were crowding the user's actual niche next-read out of the top-20. MMR and genre-diversification were honest negatives. I **learned** that 'diversify' is not free and must be measured. I **decided** to quarantine the win pending architecture consolidation rather than over-claim it. The **next evidence** is an offline counterfactual with logged propensities, then a live A/B — I don't say 'long-tail discovery solved.'"

## Catalog exposure / fairness-like governance
"I **built** a full-catalog exposure audit — Gini, coverage, zero-exposure share, top-1/5/10%, head/mid/tail lift — and **measured** that every policy is concentrated (Gini>0.97), popularity fallback is near-degenerate (Gini 1.0, 99.9% zero-exposure), head items get ~10× their catalog share while the long tail gets ~0.02×. I also **measured** that HNSW is exposure-neutral vs exact. I **learned** this is **catalog-exposure governance, explicitly not protected-class fairness**. I **decided** it's a first-class ship-gate metric alongside recall. The **next evidence** for a fairness claim would be protected-attribute data and a formal fairness criterion — which I deliberately don't claim."

## FAISS serving
"I **built** the latency–quality frontier across FlatIP, HNSW and IVF and **measured** that FlatIP exact is lossless and ~47% faster, HNSW ef64 is near-lossless (overlap 0.992) at ~2.4×, and aggressive IVF collapses candidate overlap to 0.28–0.78. I **learned** that ANN speed that silently shreds the candidate pool is a recall bug, not a win. I **decided** FlatIP exact is the default and HNSW ef64 an explicit scale option; IVF rejected. The **next evidence** for a production latency claim is a real multi-worker benchmark over the network — mine is offline/in-process."

## ALS vs SASRec
"I **built** a canonical full-softmax SASRec and trained it **to convergence**, and **measured** R@20 0.065 vs ALS 0.0846 — it lost. I **learned** that on this sparse-history catalog depth isn't the lever, and that the full-softmax-vs-sampled-softmax gap (~6.7×) still landed under MF. I **decided** MF is the warm-retrieval floor and documented the negative rather than burying it. The **next evidence** to revisit would be richer/denser sequences or session context; the convergence headline is plot-reconstructed, which is why V1 is gold_candidate, not gold_complete."

## Production-like API
"I **built** a FastAPI service with versioned model/index loading, a full fallback tree, structured per-request logging, health and monitoring endpoints, plus Docker, and **load-tested** it: warm p95 ~3.3ms, 0% empty across failure modes, invalid-k → 422, unknown user → graceful popularity. I **learned** how to make a recommender *serving-shaped* — observability and fallback, not just a model. I **decided** to call it production-**like**, never deployed. The **next evidence** for a production claim is a real deployment with multi-worker concurrency, a feature store, and network-level p95."

## OPE / propensities
"I **designed** the propensity-logging schema — request/impression/position/policy_id/π_log/exploration_bucket/reward — that makes IPS, SNIPS and DR valid, so the system is **OPE-ready by design**. I **learned** exactly why my current rerankers can't be value-estimated: no logged propensities, so IPS weights are undefined. I **decided** to refuse any OPE number until π_log and reward are logged — SNIPS first for variance control, DR once I have a reward model. The **next evidence** is one logging-policy serving run with exploration."

## Offline vs online limitations
"Everything I claim is offline by scope, and I'm precise about it: no online lift, no real users, no logged propensities yet. What I **do** have is a versioned, gated decision system that catches offline-metric bias, candidate-coverage gaps, ANN recall loss, cold-start degradation and catalog concentration *before* they'd reach users. The **next evidence** for any online claim is a live A/B — which I've specced, with guardrail metrics and the logging to run it, but not run."

---

## Pressure-test one-liners
- *"So you didn't really do cold-start?"* → "I did cold-start at the system level and named the one missing piece — live launch data. That precision is the point."
- *"Your recall is low."* → "Which protocol? V1 tuning ALS is 0.0846; the V2 *served* model under full-catalog single-gold is ~0.0375 — different builds, not comparable. I keep them in a registry so they're never conflated."
- *"Did you beat ALS with a deep model?"* → "No — and that honest negative, with a converged SASRec, is the senior signal."
- *"Is it in production?"* → "Production-*like* and load-tested locally. I don't claim deployment I didn't do."
- *"What was the off-policy value?"* → "Undefined until I log propensities — I designed the schema; I won't fabricate an IPS number."
