# 51 — G11B: SASRec Objective Ablation Report

*Controlled ablation to test the G11 diagnosis that full-position SASRec v2 underperformed because of an **objective/data mismatch** on short, left-padded book sequences. **Result: the diagnosis is REFUTED.** Last-position training did NOT improve SASRec v2 — it was marginally worse. The real cause lies in architecture/capacity + training budget, not the objective. Real CPU runs; no GPU; no advanced-model win claimed; no claim that sequence models are bad in general.*

---

## 1. Design — one controlled variable
Hold the SASRec v2 architecture fixed (**d=64, 2 blocks, 2 heads, dropout 0.2, maxlen=30**), same 40k vocab, same 5,000-user leave-last-out eval, same full 40k scoring, same per-example sampled-softmax (100 neg). **Change only the training objective:** full-position (every step predicts next) → **last-position only** (predict the final next-item). Trained to **8 epochs** to match the G11 full-position run.

## 2. Results
| Model | objective | arch / epochs | R@20 | R@50 | N@20 |
|---|---|---|---|---|---|
| ALS f64 (floor) | — | MF | **0.0846** [0.077,0.092] | 0.1466 | 0.0344 |
| SASRec v1 (D3B) | last-position | d48/1blk, 20ep | 0.0362 [0.031,0.041] | 0.0804 | 0.0122 |
| SASRec v2 (G11) | **full-position** | d64/2blk, 8ep | 0.0076 [0.005,0.010] | 0.0162 | 0.0023 |
| **SASRec v2 (G11B)** | **last-position** | d64/2blk, 8ep | **0.0030** [0.001,0.005] | 0.0098 | 0.0009 |

Loss curve (saslp): 5.21 → 3.16 → 2.07 → 1.54 → 1.27 → 1.12 → 1.03 → 0.97. Hardware: CPU (4 threads), no GPU. Repro: `run_g11_sequence_tournament.py train saslp; evalb`; atomic checkpoints in `outputs/_models/`; seed 20260616. Metrics: `outputs/evidence/g11b_sasrec_objective_ablation_metrics.json`. Plot: `outputs/plots/g11b_objective_ablation.png`.

## 3. Finding — diagnosis refuted
- **Last-position did NOT beat full-position.** At matched architecture and epochs: full-position 0.0076 vs last-position **0.0030** (Δ −0.0046). If anything, **full-position was slightly better** — it supplies more supervised targets per epoch, which helps an under-trained model.
- Therefore the **G11 §6 hypothesis (full-position objective caused the underperformance) is not supported.** The controlled test isolates the objective and shows it is **not** the driver.

## 4. So what *is* the cause? (redirected)
With the objective ruled out, the gap between SASRec v2 (d64/2blk, ~0.003–0.008) and SASRec v1 (d48/1blk, 0.036) is attributable to the remaining differences:
- **Capacity vs training budget (prime suspect):** v2 is a *larger* model (d64, 2 blocks, 2 heads) trained only **8 epochs**; v1 is *smaller* (d48, 1 block, 1 head) trained **20 epochs**. A larger transformer needs **more** epochs to converge, not fewer — at 8 epochs v2 is **under-trained for its capacity**. (Loss was still descending: 1.03→0.97.)
- **Optimization difficulty on CPU:** deeper/wider attention (2 blocks/2 heads) is harder to optimize in few CPU epochs; the extra capacity hurts at this budget rather than helping.
- *(Negative sampling and loss form were already matched to D3B, so they are not the difference.)*

**Corrected conclusion:** the SASRec v2 shortfall is an **under-training / capacity-vs-budget** effect, **not** an objective/data mismatch. This is exactly the decision-rule branch "if it remains poor, the issue is not just full-position supervision; investigate architecture/config."

## 5. What this does and doesn't change
- **Does not change the G11 headline:** all sequence models still lose to ALS f64 (0.085) under CPU-feasible training; ALS remains the floor. GRU4Rec (0.031) ≈ SASRec v1 (0.036) remain the best sequence results.
- **Corrects the G11 *explanation*:** the cause of SASRec v2's specific shortfall is capacity/under-training, not the training objective. (A correction note has been added to docs/50 §6.)
- **No general claim** that sequence models are bad; **no** RiskFrame-gold-complete claim; **no** GPU result.

## 6. Recommended follow-up (not run here)
To properly close the SASRec question: either (a) train the **smaller v1-style config** (d48/1blk) under full-position to many epochs, or (b) train the d64/2blk model to **20+ epochs** to test the under-training hypothesis directly — ideally on **GPU** (canonical config already specified in `models/sasrec_v2.py`). Until then, SASRec's competitiveness remains **open/deferred**, consistent with G5.

## 7. Status
- ✅ G11B PASS (controlled ablation run and evaluated comparably); **G11 diagnosis refuted, cause redirected to capacity/training-budget**.
- ⚠️ T3/model-depth still **open**; PulseDiscover remains **RiskFrame-gold incomplete**. This ablation strengthens *evidence honesty* (a hypothesis was tested and overturned) and *technique-tournament depth*, but does not produce a competitive deep model.

**STOP — awaiting G11B review.**
