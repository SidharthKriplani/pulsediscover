# 61 — G15 Plan: Additional Candidate-Source Depth (Co-occurrence first, LightGCN if feasible)

*Proposal only — not executed. G13 proved the binding lever is candidate coverage; G14 confirmed discovery is best handled upstream. G15 tries to **raise the candidate ceiling beyond G13-mid** with one additional graph/collaborative retrieval source — co-occurrence first (cheap, CPU), LightGCN only if genuinely cheap.*

## Company question
**Can an additional collaborative/graph candidate source (co-occurrence; optionally LightGCN) raise candidate coverage beyond G13-mid — and does the higher ceiling convert to higher LTR R@20 — at acceptable cost?**

## Why co-occurrence first
- **Cheap & CPU-feasible:** item-item co-occurrence (PMI/normalized counts) from the train interactions; no GPU, no heavy training. We already have co-occ tooling (`pulsediscover` modules + prior D3A co-occ work).
- **Complementary signal:** co-occurrence captures "bought/read together" structure that ALS factorization may smooth over — plausibly adds *marginal* coverage (golds ALS/SAS miss).
- **LightGCN** is the graph-model option but needs training (CPU-small or a quick GPU pass); include **only if cheap**, else defer with a documented reason.

## What to build / test
1. **Co-occurrence candidate generator** — build item-item co-occ (windowed/basket co-counts → PMI or normalized), top-N co-occ neighbors of each user's recent history items → candidate list.
2. *(optional, if cheap)* **LightGCN candidate generator** — light graph-propagation embeddings on the user-item bipartite graph; top-N by score.
3. **Augment G13-mid** with the new source(s); rebuild candidate sets + retrain LTR (deterministic split, local features — GPU columns shown non-additive in G12-B).

## Comparisons
- G13-mid baseline (coverage 0.323, LTR R@20 0.1201).
- G13-mid + co-occurrence.
- G13-mid + LightGCN (if feasible).
- G13-mid + both (if feasible).

## Metrics
- **candidate coverage** (combined) and **marginal coverage contribution** of the new source (golds it adds that ALS/SAS/content miss).
- **overlap** of co-occ/LightGCN candidates with ALS/SAS/content (how redundant vs complementary).
- **LTR R@20 / R@50 / NDCG@20** after retrain (does the new ceiling convert?).
- **table size, build/train runtime/cost**; coverage-per-candidate (knee check vs G13).

## Decision rule
- **Adopt** the new source only if it **adds marginal coverage AND lifts LTR R@20 materially** (vs ≈0.02 noise threshold, n≈733) at acceptable cost.
- If it **adds coverage but is redundant** (high overlap, no R@20 lift) → honest negative; report the source as non-additive on this data.
- If LightGCN isn't cheap → **defer it explicitly** (documented), keep co-occurrence as the G15 deliverable.

## Honest boundaries (pre-committed)
- Offline re-ranking within candidates; **no online lift, no production, no RiskFrame-gold-complete claim.**
- Deterministic md5 split; all numbers stamped N / source / noise threshold / reproducibility.
- No FAISS in this gate (that's G16, a serving/ANN demo).

## Required outputs (on approval)
- `docs/62_G15_CANDIDATE_SOURCE_DEPTH_REPORT.md`
- `outputs/evidence/g15_candidate_source_report.json`
- `outputs/plots/g15_coverage_contribution.png` (marginal coverage + R@20 by source config)

## Scope guardrails
- CPU-feasible by default (co-occurrence + LTR retrain). LightGCN only if a cheap pass fits; otherwise deferred.
- One coherent build + retrain; no new ranker features, no slate-policy changes (G14 default = pure LTR stands).

**STOP — G15 plan written; awaiting approval to execute.**
