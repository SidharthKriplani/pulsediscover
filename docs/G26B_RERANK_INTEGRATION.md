# G26B — Heuristic Reranker Integration (in-service)

*PulseDiscover V2. Closes the G26 quarantine by integrating the existing heuristic reranker into the **served** recommender spine (`c2_als.pkl`, ALS f64) as a small, **config-flagged, default-OFF** rerank stage, and measuring the relevance/exposure/latency tradeoff **end-to-end through the live service**. Evidence: `outputs/evidence/g26b_rerank_integration_report.json`. Code: `src/serving/recommender_service.py` (rerank stage), `src/serving/api.py` (flag + query param), `scripts/run_g26b_rerank_integration.py`.*

> **Boundaries:** heuristic exposure control — **NOT learned ranking, NOT LTR**. Served `c2` metrics only (never mixed with V1 `d3aplus` 0.0846). No long-tail-solved, no cold-start-solved, no online lift, no OPE execution.

## 1. What was integrated
A `_rerank_select` stage on the ALS retrieval path: when enabled, the service retrieves a **wider candidate pool** (`k × pool_mult`, default ×4) from FAISS, then reranks to `k` with a heuristic (`head_cap`, default — caps head items at `cap_frac×k` and fills with non-head; `tail_boost` available). It **preserves the base candidate IDs** (`base_candidate_ids` in the response/log) for audit and emits `rerank_applied`/`rerank_policy`. Controls: constructor flags, env (`PD_ENABLE_RERANK`, `PD_RERANK_POLICY`), and a per-request `?rerank=` override. **Default is OFF** — the baseline served path is unchanged unless explicitly enabled.

## 2. Why heuristic, not learned
`head_cap`/`tail_boost` are transparent, parameter-light rules over popularity tiers — no training, no labels, no model. They are deliberately interpretable levers to control catalog exposure and to *trace the tradeoff* before investing in a learned ranker. Calling this "LTR" would be false; a learned LambdaMART/LightGBM ranker is a separate, later investment.

## 3. Before / after (in-service, 4,000 warm users, served c2, K=20)
| Metric | Baseline (rerank OFF) | Reranked (head_cap ON) | Δ |
|---|---|---|---|
| Recall@20 | 0.0095 | 0.0100 | **+5.3%** (no loss) |
| Catalog coverage@20 | 6.90% | **9.79%** | +2.9 pp |
| Unique items surfaced | (base) | **+** | wider |
| Long-tail exposure share | 0.68% | **2.93%** | ~4.3× |
| Gini | 0.983 | **0.977** | −0.006 |
| Latency p50 / p95 | 0.40 ms p95 | 0.71 ms p95 | **+0.3 ms** |
| Slates materially changed | — | **2,849 / 4,000** | rerank is active |

NDCG is not separately reported (single held-out gold → NDCG≈monotone in recall). Cohort note: rerank only affects the **warm ALS path**; cold/unknown users keep the popularity fallback unchanged (verified — rerank does not touch them).

## 4. Relevance / exposure / latency tradeoff
On the served spine the head-cap rerank is a **clean win on this protocol**: it *slightly improves* relevance (+5%, because popular items were crowding niche relevant items out of the top-20), **materially improves** catalog coverage (+2.9 pp) and long-tail exposure (~4.3×), and **lowers** concentration (Gini −0.006) — at a **sub-millisecond latency cost** (+0.3 ms p95, from widening the pool). The effect is milder than the G26 offline frontier (which used a 200-item pool vs the service's ×4 pool) but directionally identical.

## 5. When to enable / keep off
- **Enable** (as a config option) when catalog-health / long-tail exposure is a goal and the small latency cost is acceptable — it does not degrade relevance on this protocol.
- **Keep OFF (default)** for the pure relevance-maximal baseline, latency-critical paths, or until validated on a different relevance target (the win is protocol-specific: warm cohort, single-held-out-gold).

## 6. Failure modes
Pool smaller than `k` (tiny catalog) → backfill guarantees `k` items, never empty. Cold/unknown user with rerank flag on → rerank is bypassed (no ALS pool), popularity fallback unchanged. Mis-set `cap_frac` near 0 → over-suppresses head, could hurt relevance — sweep on the G26 frontier before changing. Latency budget → pool widening adds ~0.3 ms; bound `pool_mult`.

## 7. Claim boundary
**Safe (strong):** "I integrated a config-flagged heuristic reranker into the served recommender path and measured the relevance/exposure/latency tradeoff against the served c2 baseline — interpretable exposure control, not learned ranking." **Avoid (weak):** "I built a ranker / improved recommendations / solved long-tail / added AI." **Unsafe:** "learned LTR deployed / cold-start solved / online lift proven / OPE executed / fairness certified."

## 8. Why G27 is still needed
G26B controls *exposure within the existing ALS candidate pool* — it reshuffles items the collaborative model already retrieves. It does **nothing** for the **57.2% item-cold-start** (items with no collaborative signal never enter the pool, so no reranker can surface them). Closing that gap requires a **new candidate source**: **G27 — Semantic Content Retrieval** (dense item-content embeddings → FAISS), the next flagship modernization gate.

## 9. Status
✅ G26B done. `ship_decision: enable_as_config_option_default_off`. **G26 quarantine CLOSED.** V1 (`gold_candidate` 8.9) untouched. Next: **G27 Semantic Content Retrieval.**

**STOP — G26B complete; awaiting review.**
