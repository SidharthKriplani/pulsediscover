# G26A — Recommender Decision Architecture + Claim-Unlock Spine

*PulseDiscovery V2. Converts strong components into one interview-defensible **decision system**. This is the spine that ties candidate generation → retrieval → ranking/reranking → fallback → serving → logging → evaluation → decision, names the model/index version at each stage, and states the claim each stage **licenses** vs **does not**. Companion docs: `G26A_MODEL_PROTOCOL_REGISTRY.md`, `G26A_CLAIM_UNLOCK_TABLE.md`, `G26A_OPE_LOGGING_SCHEMA.md`, `G26A_INTERVIEW_DEFENSE_LINES.md`. Evidence: `outputs/evidence/g26a_decision_architecture_summary.json`.*

## The one-line system identity
**PulseDiscovery is an offline, honesty-gated decision system that picks which retrieval, serving, fallback, and exposure-governance policy should ship on a 2.56M-interaction Goodreads catalog — and proves which ones must not — by catching offline-metric bias, candidate-coverage gaps, latency–quality tradeoffs, cold-start degradation, and catalog concentration before they reach users.**

The point of G26A: every box below is wired to the next, every box has a **version**, and every box has a **claim license**. No orphan components.

## End-to-end decision flow

```
[interactions] → (1) CANDIDATE GENERATION → (2) FAISS RETRIEVAL → (3) RANK / RERANK
                       │                          │                     │
                       └──────────────→ (4) FALLBACK ←──────────────────┘
                                              │
                                     (5) SERVING API
                                              │
                                     (6) LOGGING  ──►  (7) EVALUATION  ──►  (8) DECISION
```

| # | Stage | Input | Output | Artifact | Model/Index version | Failure mode | Metric | Claim ENABLED | Claim NOT enabled |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Candidate generation | train interactions | candidate item pool | ALS factors / popularity / content / co-occ | **`c2_als.pkl` (ALS f64)** served; d3aplus = V1 tuning only | coverage bottleneck; rare items never enter pool | candidate Recall, coverage | "coverage is the binding constraint; widening lifted offline recall ~44%" | "best-possible recall"; online impact |
| 2 | FAISS retrieval | candidate embeddings | top-K by IP | `src/serving/faiss_retriever.py` | FlatIP (default) / HNSW ef64 (scale) / IVF rejected | ANN silently shreds pool (IVF low-nprobe) | overlap@K, p95 latency | "FlatIP lossless ~47% faster; HNSW ef64 near-lossless (overlap 0.992) ~2.4×" | "FAISS improves recommendation quality" |
| 3 | Rank / rerank | candidate pool + scores | ordered slate | `src/eval/exposure_aware_reranking.py` **(QUARANTINED — G26B)** | heuristic rerankers on `c2` pool | over-diversify → recall collapse (MMR did) | Recall@20, Gini, coverage, tail share | "heuristic exposure rerankers map the relevance/exposure frontier" | "learned LTR"; "long-tail discovery solved" |
| 4 | Fallback | any request | always-nonempty slate | `recommender_service.py` fallback tree | popularity/content/item-sim/blended on `c2` | popularity fallback becomes dominant & non-personalized | fallback_rate, empty_rate, personalization | "every request returns a slate (0% empty); fallback QUALITY measured, not assumed" | "cold-start solved"; "unknown-user personalization" |
| 5 | Serving API | user_id, k, mode | JSON slate + headers | `src/serving/api.py` (FastAPI) | `model_id`/`index_id` md5-versioned | dim mismatch → degraded; load failure | warm p95 (local), error_rate | "production-LIKE API, load-tested, versioned, monitored" | "production deployment"; "prod p95"; "real users" |
| 6 | Logging | request stream | structured per-request record | G23 JSON logs **+ planned OPE schema** | logs `model_id`,`index_id`,`fallback`,`latency` | **no propensities logged → OPE invalid** | log completeness | "structured per-request logging + monitoring hooks" | "valid IPS/SNIPS/DR" (until propensities logged) |
| 7 | Evaluation | logs / offline splits | metrics by cohort/policy | G24/G25/G26 reports | served `c2`, full-catalog single-gold | offline metric ≠ catalog health; protocol-incomparable recall | Recall, coverage, Gini, exposure lift | "cohorted, exposure-aware offline evaluation" | "online lift"; cross-protocol recall comparison |
| 8 | Decision | metric frontiers | ship/keep/reject policy | gate docs + claim boundary | — | shipping a recall win that worsens catalog health | decision record | "I decide which retrieval/serving/exposure policy ships — and which must not" | "deployed to production / online-proven" |

## Why this is a *system*, not a pile
- **Single served model spine.** Stages 1–5 all run on **`c2_als.pkl`**; V1's `d3aplus` is explicitly quarantined to V1 tuning (see registry). This kills the "which ALS?" confusion.
- **Every stage has a failure-mode + metric + claim license.** The decision at stage 8 is only as strong as the weakest licensed claim upstream — and the audit named exactly where that is (logging → OPE).
- **The one missing wire is logging→OPE.** Stages 6→7 currently carry offline splits, not logged propensities. That is the single highest-leverage build (G26A schema below; execution is the gate after).

## What G26A licenses
A unified, versioned, claim-bounded recommender **decision architecture** with interview-maximal wording (companion docs) and a designed-but-not-yet-populated OPE logging path. It does **not** license any online or OPE-value claim, and it keeps G26 mitigation **quarantined** until G26B.

**STOP — architecture spine defined; see claim ladder + interview bank.**
