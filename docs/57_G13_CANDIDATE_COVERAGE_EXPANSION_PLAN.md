# 57 — G13 Plan: Candidate Coverage Expansion + Retrieval Ceiling Lift

*Proposal only — awaiting approval. No execution yet. Motivated by G12-B: the ranker already captures ~half the candidate ceiling (0.0791 / 0.1569), so end-to-end recall is **bottlenecked by candidate generation, not ranking.** G13 attacks the ceiling.*

## Company question
**Can we raise the candidate-set recall ceiling (currently ~0.157 at ~123 cands/user) by widening/diversifying candidate generation — and does that translate into higher end-to-end ranker R@20 at acceptable table size/runtime?**

## What to build / test
1. **Widen ALS top-N** — sweep ALS candidates per user (e.g., 60 → 120 → 200 → 400).
2. **Widen the collaborative/SASRec candidate pool** — sweep SASRec(-small, the best sequence model) top-N; optionally add co-occurrence candidates.
3. **Add content/cold candidates** — include content-hybrid (same-author/series/TF-IDF) candidates so cold/new-item golds become reachable (currently absent for warm-only candidate sets).
4. *(optional, if feasible)* **LightGCN candidate source** — graph-propagation retrieval as an additional generator; include only if it trains cheaply (CPU small-scale or a quick GPU pass) — otherwise defer to a later gate.

## Metrics (before vs after)
- **candidate coverage@K** (fraction of users whose gold is in the candidate set) — the headline ceiling, per source and combined.
- **table size** (rows, avg candidates/user) and **build runtime/cost**.
- **ranker R@20 / R@50 / NDCG@20** on the widened table (retrain LambdaMART, same user-level split) — does a higher ceiling convert to higher end-to-end recall?
- **per-source contribution** to coverage (how much each generator adds) + **marginal coverage** (overlap-adjusted).
- **coverage vs cost curve** — coverage gain per extra candidate (diminishing returns point).

## Comparison
- Retrieval ceiling **before** (G12-A: ~0.157 at warm-only ALS60+SASRec60+pop30) vs **after** (widened + content).
- End-to-end ranker R@20 before (0.0791) vs after.
- Identify the **knee**: the candidate budget where coverage gains flatten (ties back to G1 latency/cost — bigger candidate sets cost serving time).

## Decision logic
- If widening **materially raises coverage AND ranker R@20** at acceptable cost → adopt the wider candidate set as the default; record the coverage/cost knee.
- If coverage rises but **ranker R@20 doesn't follow** → the added candidates are low-quality/hard; report as honest negative (ceiling ≠ achievable).
- If gains are marginal → document that retrieval is near-saturated for these generators and the real lever is a **better generator** (e.g., LightGCN/two-tower done right), not just wider N.

## Honest boundaries (pre-committed)
- Re-ranking recall within candidates; **not** full-catalog retrieval; no online/production/gold claim.
- Coverage ceiling caps any ranker — widening N raises the ceiling but also table size and serving cost (G1).
- FAISS stays **deferred** — it's an ANN/serving-latency demo (approximates brute-force retrieval), **not** a coverage/quality lever; only build it later, clearly labeled as a systems demo.
- All numbers stamped: tag / N / CI-or-noise / source / deterministic-reproducible split.

## Required outputs (on approval)
- `outputs/evidence/g13_candidate_coverage_report.json` (coverage@K per source + combined, table sizes, runtimes, ranker R@20 before/after)
- `outputs/plots/g13_coverage_vs_budget.png` (coverage & ranker R@20 vs candidate budget)
- `docs/58_G13_CANDIDATE_COVERAGE_EXPANSION_REPORT.md`

## Scope guardrails
- CPU-feasible by default (ALS/SASRec-small/content/co-occ candidate generation + LightGBM retrain). LightGCN only if a cheap pass is feasible, else explicitly deferred.
- One coherent sweep + retrain; no FAISS, no new heavy model training beyond optional LightGCN.

**STOP — G13 plan proposed; awaiting approval to execute.**
