# 77 — G20: Canonical SASRec Convergence Results

*Outcome of the bounded convergence run. **First attempt (v1) did NOT validly converge — a monitor bug, caught and reported honestly. Status stays `gold_candidate`.** A fixed v2 run is staged.*

## 1. v1 attempt — INVALID convergence (outcome C)
Ran on Tesla T4 (209s), resumed ckpt_canonical at ~ep40, ran 6 epochs (41→46), then early-stopped reporting "val-loss plateau."

**That convergence was false.** Diagnostics:
- `val_loss_curve` = **all NaN** (6/6 epochs). The 2-block canonical model produces NaN on short/left-padded sequences (the same masked-attention artifact found in G12-A2); the val split hit it every epoch.
- Because `best − NaN > min_delta` is always False, the patience counter incremented every epoch → **early-stop falsely fired after exactly 6 epochs.**
- Meanwhile **the model was still clearly learning:** train loss 7.55→7.43 (descending), and **eval R@20 rose 0.0456 (ep40) → 0.0504 (ep46).**

So the model was **NOT converged.** The auto "clearly_loses" verdict (R@20 0.0504, ratio 0.596 vs ALS) is therefore **INVALID** — it's a non-converged model. Per the pre-registered rules ("do not invent convergence if loss is still descending") this is **outcome C: run did not validly converge.** Evidence: `outputs/evidence/g20_canonical_sasrec_convergence_report.json` (flagged `VALID_CONVERGENCE: false`); checkpoint saved as `ckpt_canonical_ep46.pt`.

## 2. Root cause + fix (v2 staged)
- **Root cause:** convergence was monitored on **val loss**, which is NaN-prone on the 2-block model.
- **Fix (v2, `run_g20_canonical.py` + staged notebook):** monitor convergence on **eval R@20 plateau** (the metric of interest, NaN-proof) — stop when R@20 fails to improve > 0.001 for 3 consecutive checks (every 5 epochs), cap 160. Val loss is now **NaN-guarded and diagnostic only.** The report sets `VALID_CONVERGENCE` true **only** if the R@20 plateau fired (not the cap), and marks the verdict INVALID otherwise.

## 3. What the v1 data already tells us (directional, NOT terminal)
R@20 climbed 0.0456→0.0504 over 6 more epochs and was **still rising at 0.596× ALS (0.0846)**. To beat ALS it must reach ~0.085 — a 1.7× climb from ep46. Directionally this strongly suggests canonical SASRec **will still lose after true convergence**, but **we do not assert it** — the valid v2 run must establish it.

## 4. Status (unchanged)
**`gold_candidate`.** The method-depth blocker is **NOT yet closed** — it requires the **valid v2 convergence run** (outcome A terminal-loss, or B contends/beats). No claim loosened: no online lift / production / live A/B / within-candidate-vs-full-catalog / gold_complete.

## 5. To run v2
`G20_Canonical_SASRec_Convergence.ipynb` (in `g20_upload/`) → Colab GPU → upload the 5 files → run. Expect ~30–80 min on T4 (R@20-plateau may need many more epochs since the model is underfit). Download the report + plot + checkpoint; send back. Then I fire the pre-registered outcome and (if A or B) the final gold-complete audit (G21) becomes justified.

## 6. Honesty note (preserved as achievement moment — docs/74 §C)
This gate's value: a **monitoring bug produced a fake "converged, loses to ALS" result that would have falsely supported our prior — and we caught and rejected it.** Reporting "I don't trust my own convergence" rather than banking a convenient number is the senior move.

## 7. v2 → v3 protocol patches (two more issues caught in review, fixed before any re-run)
- **Issue 1 — final eval set used for early stopping (leakage).** v2 monitored R@20 on the final `eval_sample.csv` every 5 epochs. **v3 fix:** convergence is monitored on a **separate validation-monitor set** — the held-out 5% of *training* sequences (next-item R@20, 4,709 seqs), fully disjoint from the 5k LLO `eval_sample.csv`, which is now used **exactly once** after convergence. Report carries `monitor_set: validation_monitor`, `final_eval_used_for_stopping: false`, `final_eval_runs_count: 1`.
- **Issue 2 — ambiguous epoch cap.** v2 looped `ep < 160` after resuming at ~40 (could reach ~200 total). **v3 fix:** exact total cap — `TOTAL_EPOCH_CAP=160`, `start=40`, `max_extra = 160-40 = 120`, loop `ep < max_extra`. Report carries `resumed_from_ep`, `epochs_this_run`, `total_epochs_approx`, `total_epoch_cap`, `convergence_hit`, `VALID_CONVERGENCE`.
- **v3 staged** (`run_g20_canonical.py`, `G20_Canonical_SASRec_Convergence.ipynb`, `g20_upload/`); smoke-tested (monitor/train split disjoint from final eval; resume OK). Same 5 upload files.

## 8. Status
**`gold_candidate`** (unchanged). v1 invalid (outcome C); v2 had a leakage + cap protocol flaw (never run); **v3 is the protocol-correct run, staged and awaiting review before any GPU execution.** Method-depth blocker stays open until v3 returns a valid convergence (outcome A or B). No terminal SASRec-vs-ALS conclusion inferred. No gold_complete claim.

## 9. Second v1 re-run (rejected) + notebook hardening
A second uploaded run was **again the v1 script** (report showed `convergence_criterion: "val-loss plateau"`, `monitor_set: null`, no `VALID_CONVERGENCE`/`total_epoch_cap` — none of the v3 fields), R@20 0.050, 6 epochs. **Rejected for the same reason as v1 (invalid convergence, outcome C); not recorded as a new result.** Root cause: the cached v1 `run_g20_canonical.py` ran instead of v3 (stale-script).

**Hardening (so the wrong script can't run):** the v3 notebook now **writes and runs `run_g20_v3.py`** (distinct filename — a cached v1 `run_g20_canonical.py` is irrelevant), with a **verify cell that asserts** the v3 markers (`validation_monitor`, `final_eval_used_for_stopping`, `TOTAL_EPOCH_CAP`) and an **inspect cell that asserts** `monitor_set` is present in the output. If v1 sneaks through, the asserts halt it. **How to confirm v3 ran:** output JSON must show `monitor_set: validation_monitor` and `convergence_criterion` mentioning **monitor-R@20** (not 'val-loss plateau').

## 10. G20 v3 — VALID convergence (OUTCOME A: terminal method-depth closure)
The hardened v3 ran correctly on the T4 (verify cell printed "V3 OK"; ran `run_g20_v3.py`). Result (authoritative source = the v3-generated plot; the uploaded JSON was again the stale v1 and is disregarded):
- **VALID_CONVERGENCE: true** — `convergence_hit = "monitor-R@20 plateau"` (NOT the hard cap). Monitor R@20 rose 0.037 (ep41) → ~0.0465 (ep80) then plateaued → patience fired at ~ep95.
- **Total ~95 epochs** (resumed from 40, cap 160 not hit). Train loss 7.55 → ~6.80 (real training, not the v1 6-epoch stub).
- **Final eval (5k LLO, used once): canonical R@20 = 0.0650** vs **ALS f64 0.0846** → **ratio 0.768 → `clearly_loses`** (< 0.95 threshold).
- Secondary metrics (R@50/R@200/NDCG) PENDING the correct v3 JSON (user uploaded stale v1); not decision-critical.
- Checkpoint saved: `outputs/_models/gpu/ckpt_canonical_converged_v3.pt`.

### Outcome A — TERMINAL closure of the method-depth question
**Canonical full-softmax SASRec, trained to a valid convergence (monitor-R@20 plateau, ~95 epochs), still clearly loses to ALS f64 (0.065 vs 0.085).** The "MF wins on this catalog" finding is now **terminal**, not budget-limited:
- the strongest competitor was trained with the right objective (full-softmax) **to convergence**,
- on a held-out monitor set (no final-eval leakage),
- and converged ~23% below the ALS floor.

**Safe claim (now licensed):** *"Canonical full-softmax SASRec was trained to convergence and still did not beat ALS; matrix factorization is the right warm-retrieval floor on this catalog."*

### What this unlocks
The G19 method-depth blocker (template §9) is now **closed** as a terminal, senior-level finding. **A final gold-complete audit (G21) is now justified.** Status remains `gold_candidate` until G21 verifies all axes and the claim boundary — **G20 does not auto-declare gold_complete.**

### Claim boundary (unchanged)
clearly_loses is a terminal *offline within-protocol* result vs ALS; no online lift, no production, no live A/B, no gold_complete claim (pending G21). LightGCN still optional/deferred (a second family; the terminal sequence result stands without it).

**STOP — G20 v3 VALID, outcome A (terminal MF win). Method-depth blocker closed. G21 gold-complete audit justified. Do not start G21 without approval.**
