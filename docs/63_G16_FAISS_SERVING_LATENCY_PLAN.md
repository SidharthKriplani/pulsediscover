# 63 — G16 Plan: FAISS / ANN Serving-Latency Demo

*Proposal only — not executed. A **systems/latency gate, not a ranking-quality gate.** Demonstrates serving realism for the adopted candidate strategy: can approximate nearest-neighbor retrieval (FAISS) reproduce brute-force candidate generation at much lower latency, with acceptable recall retention? **FAISS is not a quality-improvement claim** — it's a serving/ANN systems demo.*

## Company question
**For the adopted candidate strategy, can an ANN index (FAISS) serve the embedding-based retrieval stage (ALS, and where appropriate SASRec/two-tower embeddings) at production-like latency while retaining ~all of brute-force's candidate recall?**

## Scope (what FAISS can and can't index)
- **ALS f64 item factors** → FAISS index (dot-product/IP); user vector queries top-K. This is the clean ANN demo (embedding retrieval).
- *(optional)* SASRec-small / two-tower item embeddings → separate indexes, same protocol.
- **Not ANN-indexable, handled honestly:** content-hybrid (sparse TF-IDF + author/series rules), co-occurrence (lookup tables), popularity (static list) — these are *not* dense-embedding retrieval, so FAISS doesn't apply; they stay as-is. The demo covers the **embedding-retrieval slice** of the candidate stage and says so.

## What to test
1. **Brute-force baseline** — exact top-K from full `X·Yᵀ` (what we've been doing).
2. **FAISS index** over item embeddings — `IndexFlatIP` (exact, baseline ANN) and an approximate index (`IVFFlat` / `HNSW`) with tunable nlist/nprobe or efSearch.
3. **Exact vs approximate top-K agreement** + recall retention vs brute-force, swept over the approximation knob.

## Metrics
- **top-K agreement / recall retention** vs brute-force (e.g., fraction of brute-force top-K recovered) at K=100/200/400.
- **p50 / p95 query latency** (per user) — brute-force vs FAISS exact vs FAISS approx.
- **index build time**, **index size / memory**.
- **query throughput** (queries/sec) if feasible.
- **failure/edge cases:** cold items (no embedding → not ANN-servable, fall back to content lane), users without an ALS factor.
- **speed↔recall-retention tradeoff curve** (the headline plot).

## Decision rule
- **Include FAISS as a portfolio serving layer** if it gives a real latency/throughput win at high recall retention (e.g., ≥0.95 retention at materially lower p95) — i.e., it *demonstrates* the standard retrieve-at-scale pattern.
- If on this dataset size brute-force is already fast enough that FAISS adds no latency win → report that honestly: FAISS is a **scale-demonstration** (would matter at 10–100× catalog), not a win here.

## Honest boundaries (pre-committed)
- **FAISS is NOT a quality/recall improvement** — at best it *retains* brute-force recall; ANN can only lose recall vs exact. The value is latency/scale.
- Covers the **embedding-retrieval slice** only; content/co-occ/pop candidates are not dense-ANN and are excluded from the FAISS path (stated).
- No online lift, no production deployment, no RiskFrame-gold-complete claim. Deterministic; reproducible.

## Required outputs (on approval)
- `docs/64_G16_FAISS_SERVING_LATENCY_REPORT.md`
- `outputs/evidence/g16_faiss_serving_report.json`
- `outputs/plots/g16_latency_recall_tradeoff.png`

## Scope guardrails
- CPU `faiss-cpu` (pip). One index-family sweep (IVF or HNSW) + exact baseline; no new candidate sources, no ranking/slate changes.
- Latency reported as **in-sandbox CPU measurements, labeled as such** (not a production benchmark) — honest like the G1 estimates.

**STOP — G16 plan written; awaiting approval to execute.**
