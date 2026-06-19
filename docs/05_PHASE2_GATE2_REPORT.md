# 05 — Phase 2 Gate 2 Report (Retrieval Baselines)

*Phase 2 scope: retrieval baselines only. No SASRec, no two-tower, no simulator, no PDF. Stops at Gate 2 for review.*

---

## 1. What was built (real, reviewable code in `src/`)
| Module | Contents | Status |
|---|---|---|
| `src/pulsediscover/data.py` | canonical schema; loaders for MovieLens-1M / Goodreads-genre JSON / Amazon Books JSONL; `global_time_split`; `leave_last_out_constrained`; `cold_item_flags` | `[BUILT]` |
| `src/pulsediscover/baselines.py` | popularity; item-item co-occurrence (PMI); from-scratch confidence-weighted implicit **ALS** | `[BUILT]` |
| `src/pulsediscover/eval.py` | Recall@K, NDCG@K; **per-user bootstrap CI**; **Wilson CI** | `[BUILT]` |
| `src/run_phase2.py` | driver: detect data → split → baselines → metrics+CIs → evidence JSON + plot | `[BUILT]` |

Dependencies: numpy/pandas/matplotlib only (already in sandbox). No heavy deps; fully in-sandbox.

## 2. Canonical interaction schema (created)
`user_id, item_id, creator_id?, series_id?, event_ts, rating?/event_type` — exactly as specified. `creator_id`/`series_id` populate from Goodreads (authors/series); `None` for MovieLens. **Position and propensity are deliberately NOT in the schema** — they are manufactured only by the Phase-4 simulator, never claimed as real.

## 3. Leakage-safe splits (created)
- **Primary — `global_time_split` (0.8/0.9 quantiles):** no test interaction precedes any train interaction in wall-clock time.
- **Secondary — `leave_last_out_constrained`:** each user's last item held out **only if** it falls at/after the global cutoff (prevents cross-user future leakage — the honest version of leave-last-out).
- Cold-item fraction computed automatically.

## 4. Baselines implemented (no deep models)
Popularity (frequency floor) · co-occurrence (item-item PMI, min_co=2, top-50 neighbours) · ALS (from-scratch, confidence-weighted, alternating closed-form solves; aligned with the vendored `pulserank` approach, no live coupling).

## 5. Code validation (NOT evidence)
With no real dataset present, the pipeline was executed on a tiny, deterministic **code-validation fixture** to prove correctness. Output is quarantined in `outputs/_codecheck/` and flagged `is_evidence: false`, `tag: [CODE-CHECK ONLY]`. **Its numbers are not findings and are not reported as results.** It confirms: loaders, both splits, all three baselines, both CI methods, and plotting run end-to-end.

## 6. Data received — **REAL RUN COMPLETE** (MovieLens-1M smoke)
You placed MovieLens-1M in `data/raw/ml-1m/`. The driver ran on the **full 1,000,209 interactions** (6,040 users × 3,706 items) in ~18 s in-sandbox. `outputs/evidence/retrieval_baselines.json` is tagged `[BUILT — real data]`.

- **Split:** global-time cutoffs at the 0.8/0.9 timestamp quantiles → **999,000 train interactions, 1,209 held-out test users** (constrained leave-last-out). Cold-item fraction 0.1% (expected — MovieLens items have long histories).

### Real results — Recall@K / NDCG@K with 95% CIs (N = 1,209 held-out users)
| Baseline | Recall@10 (boot 95%) | Recall@20 (boot 95%) | NDCG@10 (boot 95%) |
|---|---|---|---|
| Popularity | 0.0182 [0.0116, 0.0256] | 0.0207 [0.0132, 0.0289] | 0.0113 [0.0066, 0.0166] |
| Co-occurrence (PMI) | 0.0033 [0.0008, 0.0066] | 0.0033 [0.0008, 0.0066] | 0.0012 [0.0002, 0.0027] |
| **ALS** | **0.0339 [0.0240, 0.0447]** | **0.0761 [0.0612, 0.0910]** | **0.0158 [0.0109, 0.0214]** |

**Honest reading (per the "no forced win" standard):** **ALS > popularity > co-occurrence**, and the ALS↔popularity gap is real — their 95% CIs separate at Recall@20 (ALS [0.061, 0.091] vs popularity [0.013, 0.029]). Absolute numbers are **low**, which is expected and honest: this is a strict *next-item leave-last-out under a global temporal split*, with *retrieval-only* baselines (no personalized ranker yet), on a hard cutoff. Co-occurrence (unordered PMI, min_co=2) is genuinely weak at next-item prediction here — documented, not hidden. MovieLens-1M is the **smoke test** (validates the pipeline + establishes baselines); the domain headline remains a Goodreads fiction genre (Phase 2b / next).

**Acquisition note:** I did not auto-download the data (platform policy blocks me fetching URLs via shell/Python); you provided it, which is the intended path. Re-running on a Goodreads genre or Amazon Books is one command once that file is in `data/raw/`.

## 7. Artifacts produced this phase
- `outputs/evidence/data_manifest.json` — status **`DATA_PRESENT`**, source `movielens-1m`, 1,000,209 interactions / 6,040 users / 3,706 items, schema, split design, seed.
- `outputs/evidence/retrieval_baselines.json` — **real MovieLens-1M baselines**, `is_evidence: true`, tag `[BUILT — real data]`, with N + bootstrap + Wilson CIs.
- `outputs/plots/retrieval_baselines_recall.png` — real Recall@K curve.
- `outputs/_codecheck/*` — earlier code-validation fixture only (`is_evidence: false`), retained for provenance; not a finding.
- `docs/04_CI_METHOD_NOTES.md` — bootstrap + Wilson methodology.

## 8. Honesty / no-overclaim check (per `plans/03`)
- Real metrics reported **only** for MovieLens-1M (the smoke dataset), each with N + 95% CI; framed as smoke baselines, not the domain headline. ✅
- Code-check fixture numbers remain explicitly non-evidence (`is_evidence: false`). ✅
- No SASRec / two-tower / simulator / PDF built in Phase 2. ✅
- No Goodreads/Amazon (domain) claims — those remain pending data load. ✅
- ALS reuses *logic* aligned with vendored pulserank provenance; no live dependency coupling. ✅

## 9. Gate 2 status & decision needed
- ✅ Baseline code + splits + CIs implemented and validated.
- ✅ Canonical schema + leakage-safe split design realized in code.
- ✅ **Real baselines computed on full MovieLens-1M with N + 95% CIs** (`retrieval_baselines.json`, `[BUILT — real data]`); recall plot in `outputs/plots/`.
- ✅ Honest comparison documented (ALS > popularity > co-occurrence; low absolute numbers explained; no forced win).
- ⏸️ No SASRec / two-tower / simulator / PDF (correctly out of Phase 2).

**Gate 2 decision for you:**
- (a) Accept the MovieLens-1M smoke baselines as the Phase-2 deliverable, and/or
- (b) provide a **Goodreads fiction genre** file (default `fantasy_paranormal`) or **Amazon Books** in `data/raw/` for the *domain* baselines (one command re-runs), then
- (c) approve **Phase 3 (SASRec)** — built to the binding standard: implement → CI-evaluate → compare honestly → debug if weak → document (a win is not required).

**STOP — awaiting Gate 2 review. No Phase 3 work begins until approved.**
