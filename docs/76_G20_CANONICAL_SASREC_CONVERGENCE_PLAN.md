# 76 — G20 Plan: Canonical SASRec Convergence Run

*Bounded gold-completion run (not scope expansion): train the canonical full-softmax SASRec competitor **to convergence**, then compare honestly to ALS f64 on the same 5k LLO protocol. **This is a GPU run** — full-softmax over 40k items is infeasible to converge on the sandbox CPU (no GPU, 45s/call), so it executes via the self-writing Colab notebook `G20_Canonical_SASRec_Convergence.ipynb` (mirror script `src/run_g20_canonical.py`). Results PENDING the run — nothing inferred.*

## Run specification (frozen — no protocol changes to help SASRec)
- **Model:** canonical full-softmax SASRec, **d64 / 2 blocks / 2 heads**, full-softmax over the 40,001-item vocab, maxlen 30.
- **Data:** same `g11_fullpos_arrays.npz` (full-position targets); **95/5 train/val split (deterministic, seed 20260616)** for convergence monitoring — does NOT touch the held-out 5k LLO eval.
- **Resume:** `outputs/_models/gpu/ckpt_canonical.pt` (model weights; optimizer re-initialized — noted). Start ≈ epoch 40 where the prior run stopped (loss still descending at 7.53).
- **Eval (unchanged):** 5,000-user leave-last-out, full 40k scoring — R@20, R@50, R@200, NDCG@10, NDCG@20 — vs ALS f64 (R@20 0.085).

## Convergence criterion (DEFINED BEFORE TRAINING)
Stop when **validation full-softmax loss fails to improve by ≥ `min_delta=0.005` for `patience=6` consecutive epochs** (plateau), OR a hard cap of **120 total epochs**. The run records which criterion fired. **Training will NOT stop while val loss is still descending faster than min_delta.** If neither fires (still clearly descending at 120), that is logged as "not converged — needs more epochs," not a result.

## Metrics reported
R@20/R@50/R@200, NDCG@10/NDCG@20, full loss curve (train+val), epoch count, convergence-criterion-hit (yes/which), and the verdict vs ALS: **beats / contends / still loses**.

## Pre-registered outcome handling (no post-hoc goalpost moving)
- **A — converged & still clearly loses to ALS:** terminal method-depth closure. Safe claim: *"Canonical full-softmax SASRec was trained to convergence and still did not beat ALS; MF is the right warm-retrieval floor on this catalog."* → likely unlocks a final gold-complete audit.
- **B — converged & contends/beats ALS:** advanced-model method depth satisfied; update candidate strategy only if evidence supports. → also likely unlocks gold-complete, different positioning.
- **C — run fails / cannot converge (compute/runtime):** do NOT infer; keep `gold_candidate`; document the compute failure + exact missing requirement.

## Hard rules (enforced)
No LightGCN in G20. No eval-protocol changes to help SASRec. No within-candidate vs full-catalog mixing. No automatic gold_complete after G20. No online/production claim. No hiding a losing result. No inventing convergence while loss descends.

## Definition of "clearly loses / contends / beats" (pre-set)
- **clearly loses:** converged R@20 < 0.95 × ALS R@20 (i.e., < ~0.081).
- **contends:** within ±5% of ALS R@20 (≈ 0.081–0.089).
- **beats:** converged R@20 > 1.05 × ALS (> ~0.089).
*(Prior: at 40 ep canonical was 0.0456 ≈ 0.54× ALS, still descending — so "clearly loses after convergence" is the expected-but-not-asserted outcome.)*

## Outputs
`docs/77_G20_CANONICAL_SASREC_RESULTS.md` (PENDING → filled from run), `outputs/evidence/g20_canonical_sasrec_convergence_report.json` (PENDING stub), `outputs/plots/g20_canonical_sasrec_loss_curve.png` (produced by run).

**STOP — plan frozen; convergence run executes on GPU via the staged notebook.**
