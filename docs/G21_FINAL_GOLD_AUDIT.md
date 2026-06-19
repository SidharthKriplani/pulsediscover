# G21 — Final 15-Axis RiskFrame-Gold Audit

*Decision gate: `gold_complete` vs `gold_candidate`. Audit/consistency only — no new models. The G19 method-depth blocker is now CLOSED by G20 v3 (canonical full-softmax SASRec converged, R@20 0.065 < ALS 0.085, terminal). Machine-readable: `outputs/evidence/g21_gold_audit_scorecard.json`.*

## Verdict (up front)
**PATCH → `gold_candidate` (score 8.9).** Substantively gold: the binding method-depth blocker is terminally closed, all 15 axes are strong, every claim is artifact-backed and within boundary, no integrity violation. **Held one notch from `gold_complete` by a single evidence-packaging gap:** the G20 v3 artifact is **`plot_reconstructed`** (not the direct v3 machine-readable JSON) with secondary metrics pending. Per the rule "PATCH / gold_candidate if … evidence/packaging gaps remain," that triggers PATCH. **Single bounded action to gold_complete:** recover the direct v3 JSON (exact R@50/R@200/NDCG, direct provenance).

## 1. 15-axis scorecard
| # | Axis | Score | Evidence | Limitation |
|---|---|---|---|---|
| 1 | Product thesis clarity | 9.5 | docs/45 dossier; "coverage is the bottleneck" insight | — |
| 2 | Market / JD relevance | 8.5 | DEF1, G17 bank (retrieval/rank/OPE/serving roles) | no explicit one-page JD map |
| 3 | **Technique tournament depth** | **9** | ALS ladder + SASRec(last/full/canonical-to-convergence) + GRU + two-tower + co-occ + LTR + FAISS; **terminally resolved (G20)** | MF wins (terminal) — by design, not a gap |
| 4 | Deep Defense Kernel | 9.5 | G17 (18 cards + bank + traceability matrix) | — |
| 5 | Product Reasoning Kernel | 9 | G2/G4 ship-hold-kill; decision logic throughout | — |
| 6 | Data realism / feature eng | 8 | D1/D2 (2.56M, global-time, prevalence) | features moderate; single domain |
| 7 | Synthetic realism audit | 7.5 | OPE/PBM/feedback sims labeled + honest-noted | no dedicated realism-audit doc (non-blocking) |
| 8 | Evidence honesty | 10 | tags/CIs/negatives; **G20 monitoring-bug caught**; plot-reconstructed provenance flagged; stale-JSON rejected | — |
| 9 | Evaluation validity | 9 | LLO+global-time, CIs, G6 scale, coverage ceiling, within-vs-full-catalog + monitor/eval separation | — |
| 10 | Decision economics | 8 | G1 cost/infra, monetization caution, knee/cost tradeoffs | qualitative, no $ model |
| 11 | Industry-pattern awareness | 9 | two-stage, ANN serving, OPE, PBM, MMR, LTR | — |
| 12 | Hairy failure modes | 9.5 | G3 register + per-card + 2-block NaN + stale-script + monitoring bug | — |
| 13 | Achievement moments | 9.5 | 6 moments incl. the convergence-monitor catch | — |
| 14 | Tradeoff density | 9 | C2/O1b/H2/G13/G14/G16 | — |
| 15 | Interview dominance | 9.5 | DEF1 + G17 bank + claim lock | — |

**Mean ≈ 9.0** on substance; **recorded 8.9** because the residual evidence-packaging gap (G20 plot-reconstructed JSON) caps it at PATCH per the rules.

## 2. The 10 mandated G21 checks
1. Every headline metric has tag / N / source / limitation — **PASS** (`docs/12` ledger).
2. G20 plot-reconstructed provenance acknowledged — **PASS** (`provenance: plot_reconstructed`, source_note).
3. Secondary SASRec metrics unavailable unless direct JSON — **PASS** (`null` / `pending_correct_v3_json`, not invented) — *this is the packaging gap → PATCH*.
4. ALS/MF terminal warm-floor claim framed offline/protocol-specific — **PASS** ("under the offline protocol").
5. Cold-start not overclaimed — **PASS** ("candidate lane," not "solved").
6. FAISS not claimed to improve quality — **PASS** (latency/retention only).
7. LTR within-candidate not mixed with full-catalog — **PASS** (kept separate throughout).
8. No production / online lift / live A/B / deployed claim — **PASS**.
9. LightGCN optional/deferred unless required — **PASS** (G21 does NOT require it; terminal sequence result stands without it).
10. Resume bullets use only ledger-surviving claims — **PASS** (table §4).

All 10 pass; check #3 is the acknowledged packaging gap (not a violation) that keeps it at gold_candidate.

## 3. Evidence-ledger consistency check
`docs/12` G-series rows reconcile with `outputs/evidence/` artifacts: ALS 0.085, content cold 0.241, SASRec-small 0.0594, full-softmax 0.0456→canonical-converged 0.065, two-tower 0.0020, LTR 0.0382→0.0791 / mid 0.1201, G13 +44%, G14 pure-LTR default, G15 co-occ conditional, G16 FAISS exact, **G20 terminal 0.065<0.085**. No contradictions. G20 row carries the plot-reconstructed/pending caveat. ✅

## 4. Final resume / LinkedIn-safe claims (each maps to a ledger artifact)
| Safe claim | Artifact |
|---|---|
| Offline two-stage recommender (candidate gen → LambdaMART) on 2.56M Goodreads interactions | G12/G13 |
| ALS f64 warm floor R@20 0.085; content-hybrid cold lane 0.241 | D3A+/C1 |
| **Trained canonical full-softmax SASRec to convergence; it still didn't beat ALS — MF is the right warm floor (offline)** | **G20 v3** |
| LTR ~2× single-retriever within candidates; diagnosed coverage as bottleneck, +44% R@20 by widening retrieval | G12-B/G13 |
| OPE + position-bias correction + FAISS serving index; documented honest negatives (sequence models, two-tower) | O1/P1/G16/G11 |

## 5. Decision
- **`gold_candidate` (PATCH), 8.9.** Method-depth blocker closed; substance is gold-grade; one evidence-packaging gap (G20 plot-reconstructed) holds it from `gold_complete`.
- **To `gold_complete`:** recover the **direct v3 JSON** (exact secondary metrics + direct provenance). That single artifact swap clears check #3 and lifts the score over 9 with no blocker → then declare `gold_complete`.
- **Not declared:** `gold_complete`, online lift, production, live A/B, deployed quality, deep-model-beats-ALS.

## 6. Safe target claim (licensed)
*"PulseDiscover is an offline recommender-systems decision dossier showing that ALS remains the strongest warm-retrieval floor after a valid convergence test of canonical full-softmax SASRec, with cold-start and serving layers evaluated under explicit claim boundaries."*

**STOP — G21 verdict: PATCH / gold_candidate (8.9). One bounded action (direct v3 JSON) from gold_complete. No gold_complete declared.**
