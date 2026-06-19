# 64 — G16: FAISS / ANN Serving-Latency Demo

*Systems/serving gate **only** — not a ranking-quality gate. FAISS over the **ALS f64 embedding slice** (content/co-occ/popularity are non-dense lanes, excluded by design). Brute-force vs FAISS-exact vs FAISS-IVF-approx. **FAISS can only retain brute-force recall, never improve ranking.** Latency = CPU in-sandbox, **not** a production benchmark. Evidence: `g16_faiss_serving_report.json`; plot `g16_latency_recall_tradeoff.png`.*

## 1. Setup
- Index: **ALS f64 item factors** — 40,541 items × 64-d. Queries: 2,000 sampled eval users' ALS vectors. Top-K at K=100/200/400. Metric: inner product (matches ALS scoring).

## 2. Results (per-query latency, recall retention vs brute-force)
| Method | p50 ms | p95 ms | retention@200 | build | index MB | throughput (qps, batch) |
|---|---|---|---|---|---|---|
| brute-force (numpy, exact) | 0.641 | 0.781 | 1.000 (ref) | — | — | — |
| **FAISS IndexFlatIP (exact)** | **0.377** | **0.475** | **1.000** | 0.005s | 10.4 | 11,581 |
| FAISS IVF nprobe=8 (approx) | 0.071 | 0.100 | 0.474 | 0.34s | 10.8 | 56,473 |
| FAISS IVF nprobe=32 | 0.136 | 0.191 | 0.760 | — | — | — |
| FAISS IVF nprobe=64 | 0.188 | 0.254 | 0.872 | — | — | — |

IVF nprobe sweep (retention@200): 1→0.130, 4→0.334, 8→0.474, 16→0.622, 32→0.760, **64→0.872** (still < 0.95).

## 3. Findings
1. **FAISS exact (IndexFlatIP) is a clean win: faster *and* lossless.** p95 **0.475 ms vs 0.781 ms** brute-force (~39% lower) at **100% recall retention** (it's exact), 10 MB index, 11.6k qps. The optimized exact index beats naive numpy brute-force with zero recall cost.
2. **IVF approximation does NOT pay off at this catalog size.** Even nprobe=64 reaches only **0.872 retention@200** — below the 0.95 target — and to push retention near-exact you'd raise nprobe until the speed advantage erodes. On 40k items, exact search is already sub-millisecond, so trading recall for ANN speed is the wrong trade *here*.
3. **Small-catalog caveat is the honest headline.** With only 40k items × 64-d, brute-force is already <1 ms. **IVF/ANN approximation is a scale-demonstration — it pays off at 10–100× catalog (millions of items), not at the current size.** Reported as such, not as a measured win.

## 4. Failure / edge cases
- **Cold items (no ALS embedding):** not ANN-retrievable by construction → served by the content lane (by design; the FAISS path covers only the dense ALS slice).
- **Users without an ALS factor:** **310** of 5,000 eval users → no query vector → fall back (popularity/content).
- **Small catalog:** brute-force p50 0.64 ms already fast → FAISS-approx unnecessary at this size (scale-demo).
- **Aggressive approximation:** low nprobe (1) collapses retention to 0.13 — a real recall cost; the sweep makes the speed↔recall tradeoff explicit.

## 5. Decision — adopt FAISS *exact* (FlatIP) as the serving index; IVF = scale-demonstration only
Decision rule: include FAISS if it shows a real latency/throughput win at ≥0.95 retention, esp. at p95.
- **FAISS IndexFlatIP qualifies** — ~39% lower p95 than naive brute-force at **1.0 retention** (lossless). Adopt it as the embedding-retrieval serving index for the ALS slice.
- **FAISS IVF (approx) does NOT qualify at current scale** — can't reach 0.95 retention at useful nprobe. **Honest call: IVF is a serving-pattern / scale-demonstration, not a measured win at 40k items.** Revisit at much larger catalogs.

## 6. Claim boundaries (preserved)
- **FAISS retains recall vs brute-force; it cannot improve ranking quality** (exact = identical; approx ≤ exact). No recommendation-quality claim.
- Latency numbers are **CPU in-sandbox, not production benchmarks.**
- Candidate strategy, ranker, slate policy, and the **G15 conditional-adopt interpretation are unchanged.**
- No online lift, no production deployment, no RiskFrame-gold-complete claim.

## 7. Status & next
- ✅ G16 done. **FAISS exact (FlatIP) adopted as the ALS-slice serving index (lossless, ~39% lower p95); IVF approximation is a documented scale-demonstration, not a current-size win.** Demonstrates the standard two-stage serving pattern honestly.
- ⏭️ Remaining path to gold: **G17** Deep Defense Kernel (closes RiskFrame audit axis #4); **G18** Final Gold Audit + resume-safe closeout.

---

## 8. Acceptance addendum (ACCEPTED — G16 passes)
- **FAISS demo run only on the ALS f64 embedding slice.**
- **FAISS IndexFlatIP (exact) ADOPTED as the ALS-slice serving index:** brute-force numpy p95 **0.781 ms** → FAISS FlatIP p95 **0.475 ms**; recall retention@100/@200/@400 = **1.000**; index ~**10.4 MB**; build **0.005 s**.
- **IVF approximation NOT adopted at current scale:** nprobe=64 retention@200 only **0.872** (< 0.95 target) → IVF is a **scale-demonstration for much larger catalogs, not a measured win at 40k items.**
- **Boundaries preserved:** FAISS retains brute-force recall, does not improve ranking quality; latency = CPU in-sandbox, not production; no online lift; no production deployment; no RiskFrame-gold-complete claim; **G15 co-occurrence remains conditional/adopted (directional, not statistically established).**

**STOP — G16 accepted; G17 Deep Defense Kernel built next (docs/65–67 + matrix).**
