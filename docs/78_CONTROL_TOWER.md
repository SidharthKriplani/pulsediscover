# 78 — Control Tower (single status index)

*Live status index for PulseDiscover. (Repo note: this is the control-tower doc the workflow refers to as "00_CONTROL_TOWER"; docs/00 is already the RiskFrame Gold Standard, so the control tower lives here. The "evidence ledger" is `docs/12`; the "claim boundary" lock is `docs/69` + table `docs/33`.)*

## Current status — LOCKED (terminal)
- **Project status:** `gold_candidate` (**G21 score 8.9, LOCKED terminal**) — method-depth blocker CLOSED (G20 v3 terminal MF win); all 10 G21 checks pass; substance ≈9.0.
- **gold_complete: DEFERRED/CLOSED.** The Colab session producing the v3 run was lost; the direct v3 JSON + v3 checkpoint are unrecoverable without a 30–80 min re-run. The pending secondary metrics (R@50/R@200/NDCG) are **immaterial** — of a model that loses to ALS anyway — and change no claim or conclusion. So the project is **locked at `gold_candidate`** as the honest terminal status (the one evidence-packaging gap is the only thing between it and gold_complete; closing it requires a re-run that isn't worth it for immaterial numbers).
- **Re-run optional:** if a clean gold_complete artifact is ever wanted, re-run the hardened v3 notebook and paste the printed JSON.
- **Artifacts:** `docs/G21_FINAL_GOLD_AUDIT.md`, `outputs/evidence/g21_gold_audit_scorecard.json` (`final_status: gold_candidate (LOCKED, terminal)`).

## V2 expansion lane (V1 locked, untouched)
- **V1:** `gold_candidate` 8.9 (terminal, interview-ready) — **frozen.**
- **V2:** `PulseDiscover V2 — Serving, Cold-Start, Fairness, Launch-Readiness`. Builds NEW evidence for previously-forbidden-but-not-impossible claims, under NEW boundaries. No retroactive V1 upgrades.
  - **G22 — FAISS Serving & Latency-Quality Frontier: ✅ DONE.** FlatIP exact lossless ~47% lower p95; HNSW ef64 near-lossless (overlap 0.992) at ~2.4× speedup. Module `src/serving/faiss_retriever.py`. Latency claim evidence-backed. (`docs/G22_FAISS_SERVING_LATENCY_QUALITY.md`)
  - **G23 — Production-like API + Monitoring: ✅ DONE.** FastAPI (`/recommend`,`/health`,`/metadata`,`/metrics`), versioned loader, full fallback tree, structured logging, Docker + load test. warm p95 ~3.3 ms, 0% empty. Production-**shaped**, not deployed. (`docs/G23_PRODUCTION_LIKE_SERVING_API.md`)
  - **G24 — Cold-Start + Sparse-Cohort Fallback Quality: ✅ DONE.** Cohort + policy eval on the served c2 ALS f64: popularity fallback collapses catalog coverage ~59× (3532→60 unique) & personalization 0.68→0.001 for cold users; recall non-monotonic (cold gold more popularity-predictable, reported honestly); 57% gold = item-cold-start; 7/7 serving cases non-empty. **Cold-start NOT solved; no fairness claim.** (`docs/G24_COLD_START_SPARSE_COHORT_POLICY.md`)
  - **G25 — Catalog Exposure / Coverage Governance: ✅ DONE.** Per-item exposure audit across policies: all Gini>0.97; popularity near-degenerate (Gini 1.0, 99.9% zero-exposure); head exposure ~10× catalog base rate, long-tail ~0.02× (suppressed ~50×); blended = best serving-complete policy; HNSW≈exact exposure (governance-neutral). **Exposure concentration measured; NOT fairness certified, NOT long-tail solved.** (`docs/G25_CATALOG_EXPOSURE_GOVERNANCE.md`)
  - **G26 — Exposure-Aware Reranking / Mitigation: ⛔ BUILT BUT BLOCKED.** Rerankers built & evaluated (head-cap Pareto-dominant on warm cohort; MMR/genre-div honest negatives) but **BLOCKED pending the RiskFrame-Gold ruthless audit verdict.** Will ship as **G26B** after G26A. (`docs/G26_EXPOSURE_AWARE_RERANKING_POLICY.md`)
  - **RiskFrame-Gold Ruthless Audit — ✅ DONE (verdict PATCH, mean 8.3).** Found a real component-coherence + OPE-logging gap, NOT a pile of components. **Next gate = Option C: G26A (Recommender Decision Architecture Consolidation + OPE Logging Design) → then G26B (unblock built G26).** Key risks: two ALS models (d3aplus 0.0846 vs served c2 ~0.0375) unreconciled; no propensity logging (OPE invalid); decision flow undrawn. (`docs/PULSEDISCOVERY_RISKFRAME_GOLD_AUDIT.md`, `outputs/evidence/pulsediscovery_riskframe_gold_audit.json`)
  - **G26A — Recommender Decision Architecture + Claim-Unlock Spine: ✅ DONE.** Unified decision flow (candgen→FAISS→rank/rerank→fallback→serve→log→eval→decide) with per-stage claim licenses; model/protocol registry reconciling V1 `d3aplus` (0.0846) vs served `c2` (~0.0375) as NOT_COMPARABLE; claim-ladder + interview-maximal wording bank; OPE logging schema (OPE-ready by design, not valid until propensities logged). G26 flagged `QUARANTINED_G26B_PENDING_G26A`. (`docs/G26A_*`, `outputs/evidence/g26a_decision_architecture_summary.json`)
  - **G26B — Heuristic Reranker Integration: ✅ DONE (quarantine CLOSED).** Config-flagged, default-OFF head-cap rerank stage wired into the served `c2` path. In-service (4k warm, K=20): coverage 6.9%→9.8%, long-tail 0.68%→2.93% (~4.3×), Gini 0.983→0.977, **no relevance loss** (R@20 +5%), +0.3 ms p95; 2,849/4,000 slates changed. `ship_decision: enable_as_config_option_default_off`. **Heuristic exposure control, NOT learned LTR.** (`docs/G26B_RERANK_INTEGRATION.md`)
  - **G27 — Semantic Content Retrieval: ✅ DONE (flagship modernization gate).** Dense MiniLM item embeddings (384-d, 41,866 items, 100% metadata text coverage) + FAISS content index as a candidate source. Targets item-cold-start + popularity-fallback coverage collapse. Semantic is the **only** source reaching item-cold-start golds (reachability 54.6%; ALS/pop = 0 by construction), massively expands coverage (62.6% vs ALS 8.1%) and lowers concentration (Gini 0.73 vs 0.98), at ~0.23 ms p95 — but **loses on head relevance** (R@20 0.010 vs ALS 0.087), so it's a **cold-start/tail complement, NOT a warm-path replacement**. Naive ALS-first fusion fails on cold-start (honest negative). **Does NOT claim LLM recommender, online lift, or solved cold-start.** `ship: cold_start_fallback_and_supplemental_candidate_source`. (`docs/G27_SEMANTIC_CONTENT_RETRIEVAL.md`)
  - **G28 — Learned Fusion + Ranker Tournament (GOLD PASS 1/3): ✅ DONE / PASS.** Candidate table (1.03M rows, 6,213 users; ALS & semantic ~disjoint — overlap 6,749). Fusion baselines: RRF best non-learned (~95% ALS relevance + recovers cold-start + ~4× coverage; ALS-first fails on cold-start). Learned tournament (LightGBM LambdaMART champion): **beats ALS-only on held-out test Recall@20 (0.036 vs 0.022) AND recovers cold-start (0.032 vs 0)**, consistent across 4 model families — **caveat: 81 test positives, offline, directional**. Champion policy: ALS warm primary + semantic prioritized cold-start/tail source + LambdaMART fusion where features exist. Text ablation: description marginally hurts (keep title+shelves). **Gold-readiness: strong Pass 1; ready for Pass 2.** (`docs/G28_LEARNED_FUSION_RANKER_TOURNAMENT.md`, `docs/PULSEDISCOVERY_GOLD_PASS_1_DEFENSE_NOTES.md`)
  - **G29 — OPE Execution (GOLD PASS 2/3): ✅ DONE.** Executed the OPE pipeline: stochastic logging policy with recorded propensities → IPS/SNIPS/DR. Estimators recover the known target-policy value (DR closest 0.0089 vs true 0.0106; IPS over-estimates ~2×; SNIPS stable), and correctly rank RRF-fusion > ALS. **OPE methodology demonstrated OFFLINE (synthetic propensities, proxy reward) — NOT real online lift.** (`docs/G29_OPE_EXECUTION.md`)
  - **G30 — Final Gold Audit + Interview Kit (GOLD PASS 3/3): ✅ DONE.** Re-score **9.1/10** (mid-audit 8.3). Verdict: **OFFLINE GOLD-COMPLETE / INTERVIEW-READY.** Mid-audit risks resolved (component-coherence via G26A, OPE via G29, pile-of-components via G27/G28). V1 stays `gold_candidate` 8.9 (locked, not upgraded); V2 lane offline gold-complete. Online/production evidence permanently out of scope. Interview package locked: `docs/PULSEDISCOVERY_INTERVIEW_KIT.md`. (`docs/G30_FINAL_GOLD_AUDIT.md`, `outputs/evidence/g30_final_gold_audit.json`)

  - **G31 — Search / IR Framing (BM25 + dense hybrid): ✅ DONE (role-coverage add-on).** Added a BM25 lexical lane + RRF hybrid with the dense lane; reported search-standard NDCG@K + MRR (single-relevant) + Recall@K on seed-item (query-by-document) queries. Honest finding: **BM25 lexical strongest single lane** (R@20 0.0133 vs dense 0.0067); **RRF hybrid is robust-middle, not a recall winner**. Re-narrates LTR/FAISS/OPE as a two-stage search pipeline. **Three honesty guards held: query-by-document (not free-text), position-bias *evaluation* (not in-ranker correction), single-relevant NDCG/MRR. V1 RecSys claims untouched.** Unlocks Search/IR role coverage. (`docs/G31_SEARCH_IR_FRAMING.md`)

### ✅ PROJECT STATUS: INTERVIEW-READY (offline gold-complete) + Search/IR role coverage (G31). Gold Passes complete. Use the Interview Kit + Claim Ladder within their boundaries.

## Control-doc map
| Role | File |
|---|---|
| RiskFrame gold standard | `docs/00_RISKFRAME_GOLD_STANDARD.md` |
| Control tower (this) | `docs/78_CONTROL_TOWER.md` |
| Evidence ledger | `docs/12_EVIDENCE_LEDGER.md` |
| Claim boundary lock / table | `docs/69_G18_CLAIM_BOUNDARY_LOCK.md` · `docs/33_CLAIM1_FINAL_CLAIM_BOUNDARY_TABLE.md` |
| Final status (pre-G19 lock) | `docs/72_PULSEDISCOVER_FINAL_STATUS.md` |

## Gate ledger (D-series → G-series)
- D1–D3B, C1, C2, O1/O1b, P1, H1/H2, AB1: core spine + governance (offline).
- PACK1/CLAIM1/DEF1/README/ARCH1/SUMMARY/FINALPACK: V1 packaging.
- BM2-GAP → G1 infra · G2 product · G3 risk · G4 rollout: company-realism layer.
- G5/G6 depth-scale; G8 dossier; G9 deck; G10 template; G11/G11B/G11C-D sequence tournament; G12/G12-A/A2/B LTR; G13 coverage; G14 slate; G15 co-occ; G16 FAISS; G17 deep defense; G18 final audit (gold_candidate, 8.7); G19 gold bridge (requires canonical run); **G20 canonical convergence — v1 invalid (caught), v3 VALID terminal (outcome A).**

## G20 evidence-cleanliness status
- Clean artifact: `outputs/evidence/g20_canonical_sasrec_convergence_report_v3.json` — **`provenance: plot_reconstructed`**, `VALID_CONVERGENCE: true`, R@20 0.065, verdict clearly_loses; secondary metrics `null / pending_correct_v3_json` (not invented).
- Stale v1 JSON **rejected**; v3 plot = source of headline; checkpoint `ckpt_canonical_converged_v3.pt` saved.

## Locked claim boundary (top-level)
No online lift · no production · no live A/B · no "deep model beat ALS" · no cold-start/fairness/health "solved" · no full-catalog for within-candidate · **no `gold_complete` until G21 passes.** Safe G20 line: *"canonical full-softmax SASRec trained to convergence still did not beat ALS; MF is the right warm floor under the offline protocol — closes the method-depth blocker, justifies G21."*

## Open / deferred
- **G21** final gold audit (justified, not started).
- LightGCN candidate source — optional/deferred (second family; terminal sequence result stands without it).
- Live A/B of the 90/10 policy — specced (G4), deferred (separate online axis, not required for offline RiskFrame-gold).
- Correct v3 JSON re-upload → lock exact G20 secondary metrics (R@50/R@200/NDCG).

**Status as of G20 v3: gold_candidate; method-depth blocker closed; G21 justified; no gold_complete claimed.**
