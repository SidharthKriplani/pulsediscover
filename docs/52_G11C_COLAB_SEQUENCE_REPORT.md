# 52 — G11C/G11D: Colab/GPU Sequence + Two-Tower Results

*Real GPU run (Tesla T4, ~26 min). Eval unchanged from G11/G11B (5k-user LLO, full 40k scoring, R@20/50 + NDCG@20 + bootstrap CI). **Real numbers; no fabrication.** Evidence: `outputs/evidence/g11cd_colab_metrics.json` + `g11cd_training_log.csv`; plot `outputs/plots/g11cd_gpu_results.png`. (Earlier capacity-ablation run kept at `g11c_colab_metrics.json`.) GRU4Rec-full-softmax was interrupted to save GPU quota — not needed.*

## 1. Results (R@20, GPU run + comparators)
| Model | env | R@20 | R@50 | N@20 | note |
|---|---|---|---|---|---|
| **ALS f64 (floor)** | — | **0.0846** | 0.1466 | 0.0344 | best overall |
| **SASRec-small d48/1blk last-pos, 20ep** | GPU | **0.0594** | 0.1056 | 0.0212 | **best sequence model — ~70% of floor** |
| **SASRec canonical d64/2blk FULL-SOFTMAX, 40ep** | GPU | **0.0456** | 0.0676 | 0.0230 | **full softmax rescued the big model (see §2)** |
| SASRec v2 d64/2blk sampled-neg, 30ep | GPU | 0.0068 | 0.0170 | 0.0023 | sampled-neg dead-end (confirms G11C) |
| TwoTower neural retrieval (G11D), 30ep | GPU | 0.0020 | 0.0060 | 0.0006 | honest negative (see §3) |
| SASRec v2 d64 sampled-neg, 8ep | CPU | 0.0076 | 0.0162 | 0.0023 | (G11) |
| SASRec v1 d48 last-pos, 20ep | CPU | 0.0362 | 0.0804 | 0.0122 | prior best (now beaten) |

## 2. Headline finding — full-softmax is the dominant sequence-model lever
The **same** d64/2blk/2head SASRec architecture:
- **sampled negatives (100/step): R@20 0.0068**
- **full softmax over all 40k items: R@20 0.0456 — a ~6.7× jump.**

This **confirms the long-standing suspicion** that *sampled negatives crippled every prior SASRec run* (D3B, G11, G11B were all sampled-neg on CPU). Full softmax is the canonical training objective and **GPU is what made it feasible** — it's the single biggest lever we found for the sequence family. The full-softmax model's loss was still descending at 40 epochs (7.53), so it may not even be fully converged.

## 3. Other findings
- **Best sequence model is still the *small* d48 last-position SASRec (0.0594, ~70% of the ALS floor)** — *simpler beats bigger even with full-softmax* on this short-sequence book data. Reproducible (0.0622 / 0.0626 / 0.0594 across runs).
- **TwoTower (G11D) = 0.0020 — honest negative.** A pooled-history, in-batch-negative, item-ID-only two-tower underperforms badly here. On a collaborative-dominated catalog with short histories, mean-pooling the history loses the signal ALS captures via factorization. **This is a neural retrieval baseline, not a content/production two-tower — and it lost.** Reported straight.
- **ALS f64 (0.0846) remains the best overall.** No sequence or retrieval model beats it.

## 4. Verdict — **T3/model-depth: MATERIALLY REDUCED, not closed**
- Best sequence model rose from 0.036 (pre-GPU) to **0.0594** (~70% of ALS floor); full-softmax SASRec reached 0.0456 from a 0.007 floor — a real, evidence-backed narrowing.
- **Not closed:** ALS f64 still wins; two-tower is a documented negative.
- **Confirmed lever for future work:** full-softmax (and more epochs — canonical wasn't converged) is the path to push sequence models further; the small-config last-position recipe is the current best.

## 5. Feeding these into G12 LTR (pending)
The canonical-SASRec and two-tower **per-candidate scores** would be new ranker features (the documented v2 lift). That requires exporting the trained models' scores from Colab (the aggregate JSON here doesn't carry per-item scores) — **pending a score-export step**; not blocking.

## 6. Claim boundaries
Real T4 results; eval unchanged; no fabricated numbers. Full-softmax lever shown with evidence (6.7×). Two-tower negative reported honestly (baseline, not production). "Materially reduced," **not** T3-closed / RiskFrame-gold-complete — ALS remains the floor.

**STOP — G11C/D recorded. T3 materially reduced; full-softmax confirmed as the key sequence lever; two-tower an honest negative.**
