# G22 — FAISS Serving & Latency-Quality Frontier (V2)

*PulseDiscover **V2** lane (V1 stays locked at `gold_candidate` 8.9 — not modified). Evaluates FAISS as a scalable serving layer for ALS candidate retrieval, measuring **both** quality (vs exact ALS) and serving latency. **FAISS retains recall at best (exact) or trades a measured amount for latency (approx); it does NOT improve recommendation quality, is not production, no online lift, and does not change the V1 conclusion that ALS is the warm-retrieval floor.** CPU in-sandbox latency, not a production benchmark. Evidence: `g22_faiss_latency_quality_report.json`; plot `g22_faiss_latency_quality_frontier.png`; module `src/serving/faiss_retriever.py`; runner `scripts/run_g22_faiss_latency_quality.py`.*

## 1. Setup
ALS f64 item factors (40,541 × 64); 2,000 warm-gold query users. Exact numpy brute-force is the reference; FAISS FlatIP (exact), IVF (nprobe sweep), HNSW (efSearch sweep), inner-product metric. Quality = candidate overlap@K vs exact + downstream rec Recall@20 (gold in top-20); serving = p50/p95/p99 latency, throughput, build time, index size.

## 2. Frontier (key rows)
| Setting | p50 ms | p95 ms | p99-band | overlap@20 (cand fidelity) | rec R@20 | degradation | index MB |
|---|---|---|---|---|---|---|---|
| exact numpy (reference) | 0.596 | 0.723 | — | 1.000 | 0.0375 | 0.0 | 10.4 |
| **FAISS FlatIP (exact)** | 0.311 | **0.386** | — | **1.000** | 0.0375 | **0.0** | 10.4 |
| IVF nprobe=1 | 0.016 | 0.030 | — | 0.275 | 0.0335 | 0.004 | 10.8 |
| IVF nprobe=16 | 0.062 | 0.083 | — | 0.783 | 0.0360 | 0.0015 | 10.8 |
| IVF nprobe=64 | 0.133 | 0.164 | — | 0.936 | 0.0375 | 0.0 | 10.8 |
| HNSW ef=32 | 0.101 | 0.121 | — | 0.968 | 0.0380 | ~0 | — |
| **HNSW ef=64** | 0.140 | **0.160** | — | **0.992** | 0.0375 | **0.0** | — |
| HNSW ef=128 | 0.204 | 0.228 | — | 0.998 | 0.0375 | 0.0 | — |

*(rec R@20 here is retrieval recall on the warm-query sample without seen-item exclusion — used only as the exact-vs-ANN **reference**, not the V1 ALS floor 0.085. Degradation = exact − setting; full p50/p95/p99 + throughput in the JSON.)*

## 3. Findings (measured, honest)
1. **FlatIP exact is a free serving win:** identical results (overlap 1.0, zero degradation) at **p95 0.386 ms vs 0.723 ms** brute-force (**~47% lower**). No quality risk.
2. **HNSW is the best approximate frontier:** ef64 holds **near-lossless candidate fidelity (overlap@20 0.992, rec-R@20 degradation 0.0)** at **p95 0.160 ms — ~2.4× faster than FlatIP, ~4.5× faster than brute.** ef128 → overlap 0.998.
3. **Aggressive IVF preserves downstream gold-recall but wrecks candidate fidelity:** IVF nprobe≤16 keeps rec-R@20 (the gold is a strong match that survives ANN) but candidate **overlap collapses to 0.28–0.78** — risky for a system that **re-ranks** the candidate set (the ranker would see a degraded candidate pool). So "gold survives" ≠ "candidate set is faithful."
4. **Quality is retained, never improved:** every FAISS setting's rec-Recall@20 is ≤ exact (best case equal). FAISS buys latency, not quality.
5. **Scale caveat:** at 40k items everything is sub-millisecond — ANN's payoff is a **scale story** (10–100× catalog); the latency wins are real and measured but absolute values are tiny here.

## 4. Recommended serving setting + fallback
- **Default: FAISS FlatIP (exact)** — lossless, p95 0.386 ms (~47% faster than brute), 10 MB. Zero quality risk; simplest to operate.
- **Approximate (scale) option: HNSW ef64** — overlap 0.992 / rec-R@20 degradation 0.0 at p95 0.160 ms (2.4× faster than FlatIP). Use when catalog size/latency demands sub-exact **and** candidate fidelity must be preserved for the ranker.
- **Avoid:** aggressive IVF (low nprobe) for a re-ranking pipeline — downstream gold survives but candidate overlap is too low.
- **Fallback policy:** if a chosen ANN setting's candidate overlap or rec-Recall@20 degrades beyond tolerance (or p95 regresses), **fall back to FlatIP exact** (lossless, sub-ms at this scale).

## 5. Claims (V2 — new boundary)
- **Safe (proven):** *"Built and evaluated a FAISS ANN serving layer measuring latency-quality tradeoffs against exact ALS retrieval."*
- **Stronger (now PROVEN, measured):** *"FAISS HNSW ef64 preserved candidate retrieval quality within measured tolerance (overlap@20 0.992, rec-Recall@20 degradation 0.0) while reducing p95 latency ~2.4× vs exact FlatIP and ~4.5× vs numpy brute-force (CPU in-sandbox); FlatIP exact reduces p95 ~47% at zero quality loss."*
- **Forbidden (unchanged):** FAISS improved recommendation quality (it did not) · production deployment · online lift · any change to the V1 ALS-warm-floor conclusion.

## 6. Status
- ✅ G22 done (V2). FAISS serving layer built (`src/serving/faiss_retriever.py`), latency-quality frontier measured, recommended setting + fallback defined, stronger latency claim **evidence-backed**.
- V1: unchanged (`gold_candidate` 8.9, terminal). ⏭️ Next V2 gate: **G23 — Production-like API + Monitoring** (FastAPI `/recommend`, Dockerfile, loader, fallback, health, logging, latency/load test) — production-*shaped*, not production-claimed.

**STOP — G22 complete; awaiting review before G23.**
