# G26 — Exposure-Aware Reranking / Mitigation Policy (V2)

*PulseDiscover **V2** lane. Goes beyond G25's measurement: it **builds mitigation rerankers** and quantifies the **relevance vs exposure tradeoff**. Evidence: `outputs/evidence/g26_exposure_reranking_frontier_report.json`; plots `g26_relevance_exposure_frontier.png`, `g26_tail_exposure_vs_recall_tradeoff.png`; code `src/eval/exposure_aware_reranking.py`, `scripts/run_g26_exposure_aware_reranking.py`.*

> **Scope guardrails (read first):**
> - Reranking is applied to the **warm ALS candidate pool** (served `c2_als.pkl`, ALS f64, top-200) — the cohort that has a pool to rerank. **Cold/unknown users keep the popularity fallback, unchanged**; reranking does not "fix" them.
> - Catalog **EXPOSURE mitigation** under the offline protocol. **NOT** fairness solved/certified, **NOT** creator/marketplace fairness, **NOT** online diversity, **NOT** production deployment, **NOT** protected-class analysis.
> - Recall is full-catalog single-held-out-gold on the served model — comparable **within G26**, not to the V1 `d3aplus` headline. V1 (`gold_candidate` 8.9) untouched.

## 1. Setup
4,000-user warm sample; for each user an ALS top-200 candidate pool (relevance-ordered, seen-filtered) is reranked to a top-20 slate by each policy. Metrics per policy: Recall@20/@50, catalog coverage@20, Gini, top-1/5/10% exposure share, head/mid/long-tail share, zero-exposure share, long-tail lift vs baseline, per-slate rerank latency, empty-rate.

## 2. Policies
baseline (pure ALS) · long-tail boost (α sweep) · novelty boost · head-item exposure cap (cap-fraction sweep) · genre/shelf diversification · MMR (λ sweep). xQuAD / exploration-bucket left as noted future options.

## 3. Frontier table (key rows)
| Policy | Recall@20 | Δrecall | Coverage@20 | Gini | Long-tail share | Tail lift | Rerank p95 |
|---|---|---|---|---|---|---|---|
| baseline | 0.0175 | — | 7.3% | 0.982 | 0.87% | 0.018× | 0.001 ms |
| tail_boost α1.0 | 0.0180 | +0.0005 | 7.7% | 0.981 | 3.45% | — | 0.06 ms |
| **head_cap 0.3** | **0.0245** | **+0.0070** | **12.3%** | **0.968** | **7.03%** | — | 0.02 ms |
| head_cap 0.5 | 0.0243 | +0.0068 | 11.6% | 0.971 | 5.39% | — | 0.02 ms |
| genre_div 2 | 0.0110 | −0.0065 | 6.7% | 0.985 | 2.27% | — | 0.08 ms |
| mmr λ0.5 | 0.0160 | −0.0015 | 6.0% | 0.986 | 0.62% | — | 1.0 ms |

## 4. Headline finding (and why it's honest, not too-good-to-be-true)
**The head-item exposure cap is Pareto-dominant: it improves Recall@20 (+40%, 0.0175→0.0245) AND catalog coverage (7.3%→12.3%) AND long-tail exposure (~8×) AND lowers Gini — simultaneously.** This is explainable, not a bug: a warm user's actual held-out next-read is frequently a **niche** item that ALS ranks just outside the top-20 because popular "head" items inflate the head of the list. Capping head slots promotes those niche-but-relevant items into the slate — so on this catalog, head inflation was costing *both* relevance and catalog health. **Caveat:** this depends on the niche-skewed single-held-out-gold structure of this offline protocol; under a different relevance target (e.g. dwell/rating-weighted) the tradeoff could shift, and the cap should be tuned, not assumed.

By contrast, **MMR and genre diversification are honest negatives here** — they reduce recall *and* coverage (intra-slate de-duplication throws away relevant near-neighbours without surfacing better tail items). **Long-tail/novelty boosts** cleanly trade a tiny recall change for a few-× increase in tail exposure — a usable mild lever.

## 5. Recommendation
**`head_cap_0.3`** (cap head items at ~30% of the slate) under the rule *"max long-tail/coverage gain while retaining ≥90% of baseline Recall@20."* It is the single Pareto-optimal policy on the (recall, long-tail) frontier and is sub-millisecond. **`tail_boost`** is the recommended *mild* lever when a softer, monotone exposure nudge is wanted. MMR/genre-diversification are **not** recommended on this catalog/protocol. Latency impact of all rerankers is negligible (≤~1.3 ms p95, MMR being the heaviest); none increase empty-rate.

## 6. Safe / forbidden claims
**Safe (proven):** *"PulseDiscovery G26 evaluates exposure-aware reranking policies and quantifies the relevance/exposure tradeoff, showing which mitigation strategies improve catalog coverage or long-tail exposure without unacceptable retrieval-quality degradation."*
**Forbidden:** fairness solved · creator fairness solved · marketplace fairness certified · long-tail discovery solved · online diversity improved · production exposure governance deployed · protected-class fairness analyzed.

## 7. Limitations
Warm cohort only (cold/unknown have no pool to rerank); recall is single held-out gold (NDCG ≈ proportional, not separately computed); latency is in-process rerank compute, not end-to-end serving; genre diversification uses noisy shelf genres; the head-cap win is protocol-specific and must be tuned, not treated as universal; mitigation is a heuristic tradeoff, not a fairness guarantee.

## 8. Status
✅ G26 done (V2). `mitigation_attempted:true`, `relevance_exposure_tradeoff_measured:true`, `fairness_solved:false`, `long_tail_discovery_solved:false`, `production_exposure_governance_deployed:false`. V1 unchanged.

**STOP — G26 complete; awaiting review.**
