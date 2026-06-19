# 22 — D3B Gate Report (Domain SASRec)

*Scope: train + evaluate SASRec on the Goodreads fantasy/paranormal core (top-40k vocab), same train/val/test split + same 5,000-user leave-last-out sample as D3A/D3A+. Compare vs **ALS f64 (primary floor)**, **f16 hybrid (secondary)**, **original baselines (historical)**. Domain-scoped. No OPE / position-bias / feedback / cold-start channel / PRD / PDF.*

## 1. Training config & curve
| Param | Value |
|---|---|
| embedding dim | 48 |
| max sequence length | 30 |
| blocks / heads | 1 / 1 |
| dropout | 0.2 |
| batch size | 2048 |
| loss | sampled-softmax (negative BCE), 50 uniform negatives |
| optimizer / lr | Adam / 1e-3 |
| epochs | **20** (resumable, time-budgeted across calls) |
| epoch time | ~8.3 s (CPU, 4 threads) |
| early stopping | none — fixed time budget; **loss still mildly decreasing at stop** |
| training scheme | **last-position next-item** (1 example/user = 94,185), CPU-budget choice |

**Loss curve (per epoch):** 5.29 → 4.27 → 3.41 → 2.77 → 2.28 → 1.89 → 1.62 → 1.43 → 1.29 → 1.20 → 1.13 → 1.08 → 1.04 → 1.01 → 0.98 → 0.96 → 0.94 → 0.92 → 0.90 → **0.889** (not plateaued).

## 2. Results — SASRec vs floor (5,000-user LLO, gold-in-vocab 99.0%)
| Model | R@10 | R@20 | R@50 | R@100 | R@200 | NDCG@10 |
|---|---|---|---|---|---|---|
| **SASRec (20 ep)** | 0.0188 | 0.0362 | 0.0804 | 0.1292 | 0.1872 | 0.0078 |
| **ALS f64 (primary floor)** | **0.0508** | **0.0846** | **0.1466** | **0.2154** | **0.2978** | **0.0259** |
| hybrid f16 (secondary) | 0.0406 | 0.0614 | 0.1036 | 0.1596 | 0.2362 | 0.0214 |
| ALS f16 (historical) | 0.0300 | 0.0570 | 0.1088 | 0.1742 | 0.2548 | 0.0152 |
| co-occurrence 5k | 0.0344 | 0.0464 | 0.0762 | 0.1120 | 0.1594 | 0.0207 |
| popularity | 0.0202 | 0.0350 | 0.0630 | 0.1196 | 0.1654 | 0.0085 |

**SASRec vs ALS f64 deltas:** R@10 −0.032 · R@20 −0.048 · R@50 −0.066 · R@100 −0.086 · R@200 −0.111. Plot: `outputs/plots/domain_sasrec_vs_floor.png`.

## 3. Verdict — honest
**SASRec underperforms every collaborative baseline on the domain** at this setting — below ALS f64, the f16 hybrid, f16 ALS, and even co-occurrence; roughly popularity-level at R@10. **Not hidden, not forced.** This is the anticipated CPU-limited outcome.

## 4. Diagnosis (likely causes, ranked)
1. **Last-position training (prime suspect):** I trained on next-item-at-sequence-end — **one example per user (94k)** — not canonical full-position SASRec, which supervises **every** position (millions of examples). That's ~30× less training signal, a deliberate CPU-budget choice that cripples SASRec relative to ALS.
2. **Under-training:** 20 epochs, loss still descending (0.889, not plateaued). Canonical SASRec trains hundreds of epochs.
3. **Small CPU config:** d=48, 1 block, 1 head, maxlen=30 — well below typical SASRec (d 64–128, 2 blocks/heads, maxlen 50–200) → capacity-limited.
4. **Uniform negatives (50):** popularity-corrected negatives often help on skewed catalogs.
5. **Sequence predictability:** next-book in fiction is weakly sequence-predictable vs the strong collaborative/popularity signal ALS captures.
6. **NOT cold-start:** 99.0% of golds are in the top-40k vocab — vocab coverage is not the cause. *(No cold-start progress is claimed from SASRec.)*
7. **CPU/45s-call limits** prevented the canonical setup (full-position + more epochs + bigger model).

## 5. Fix path (external pack)
Canonical **full-position** SASRec, d≥64, 2 blocks/2 heads, maxlen≥50, 100+ epochs, popularity-corrected negatives, on a free Colab/Kaggle **GPU**; re-evaluate vs the ALS f64 floor on the same sample. Added to `external_runs/D3A_PLUS_EXTERNAL_RUN_INSTRUCTIONS.md` scope.

## 6. What this means for the ladder
The **candidate-generation floor remains ALS f64** (R@20 0.085). SASRec as built does **not** beat it — so the honest current best is ALS f64; SASRec needs the canonical full-position GPU setup before it can be claimed competitive. This is a genuine result: a transformer sequence model **under CPU constraints + last-position training loses to a well-tuned MF baseline**, and saying so is the point.

## 7. No-overclaim check
Domain-scoped; SASRec underperformance reported with full deltas + diagnosis; no cold-start claim; last-position/under-training/CPU limits disclosed; gold-in-vocab stated; no OPE/position/feedback/PRD/PDF. ✅

## 8. Artifacts
- `outputs/evidence/domain_sasrec_report.json` (config, loss curve, metrics+CIs, comparison, diagnosis)
- `outputs/plots/domain_sasrec_vs_floor.png`
- checkpoint `outputs/_models/sasrec_domain.pt` (resumable)

## 9. Gate status & next step
- ✅ SASRec trained (20 ep, resumable, full config + loss curve) and evaluated vs the floor with CIs; underperformance honestly diagnosed.
- ⏸️ **Options next:** (a) external GPU run of canonical full-position SASRec to give it a fair shot vs ALS f64; (b) proceed to domain OPE/position-bias/feedback on the **ALS f64** generator (current best); (c) finalize ladder. No further step run.

**STOP — awaiting D3B review.**
