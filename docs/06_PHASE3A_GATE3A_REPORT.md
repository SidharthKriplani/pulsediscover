# 06 — Phase 3A Gate 3A Report (SASRec, MovieLens-1M smoke)

*Scope: SASRec on MovieLens-1M smoke only, CPU-safe. Same leakage-safe split + same held-out set + same CIs as the Phase-2 baselines. Standard: implement → CI-evaluate → compare honestly → debug if weak → document; a win is not required. No two-tower, no simulator, no Goodreads claims, no PDF.*

---

## 1. What was built
- `src/pulsediscover/sasrec.py` — small self-attentive sequential recommender: item + positional embeddings → **causal** self-attention blocks → next-item logits. Trainable, CPU-safe, checkpoint/resumable. `[BUILT]`
- `src/run_phase3.py` — Phase-3A driver: identical split, time-budgeted resumable training, identical eval + CIs, honest baseline comparison, evidence + plot. `[BUILT]`
- Evidence: `outputs/evidence/sasrec_report.json` (`is_evidence: true`, `[BUILT — real data]`); plot `outputs/plots/sasrec_vs_baselines_recall10.png`.

## 2. Model + training (CPU-safe, honest about the choices)
- **Config:** d=48, 2 attention blocks, 1 head, maxlen=30, dropout=0.2, batch=256, Adam lr=1e-3. (~deliberately small for CPU.)
- **Training scheme (documented deviation):** next-item-**at-sequence-end** cross-entropy — predict `items[-1]` from the full left-padded history `items[:-1]`, computing logits only at the final position. This is a **CPU-budget simplification** of canonical SASRec (which trains on *all* positions with sampled negatives). It is a legitimate next-item objective and matches the serving/eval objective, but I do **not** claim canonical full-position SASRec.
- **Epochs:** 30 (fresh, ~0.9 s/epoch within a 28 s budget). **Training loss was still decreasing (7.43 at epoch 30)** — the result is a *floor*, not converged; training is resumable (re-run continues) and would likely improve with more epochs/full-position training.
- Causal mask = no future leakage; padding excluded from loss.

## 3. Results — same held-out set (N = 1,209), same metrics + 95% CIs
| Model | Recall@5 (boot95) | Recall@10 (boot95) | Recall@20 (boot95) | NDCG@10 |
|---|---|---|---|---|
| Co-occurrence | 0.0017 | 0.0033 [0.0008, 0.0066] | 0.0033 | 0.0012 |
| Popularity | 0.0132 | 0.0182 [0.0116, 0.0256] | 0.0207 | 0.0113 |
| ALS | 0.0182 | 0.0339 [0.0240, 0.0447] | 0.0761 [0.0612, 0.0910] | 0.0158 |
| **SASRec** | **0.0405 [0.0298, 0.0521]** | **0.0662 [0.0521, 0.0811]** | **0.1026 [0.0852, 0.1199]** | **0.0350** |

## 4. Honest reading (per the "no forced win" standard)
- **SASRec is the best model on the smoke**, and the win is statistically real at @5 and @10: SASRec Recall@10 [0.0521, 0.0811] vs ALS [0.0240, 0.0447] — **the 95% CIs do not overlap**. At @20 the CIs touch (SASRec [0.085, 0.120] vs ALS [0.061, 0.091]) — a narrower margin, reported as-is.
- This is **not forced**: it came from a small, under-trained model (loss still dropping). It is a genuine result, but a *floor*, and I label it that way.
- **Caveats kept visible:** (a) MovieLens-1M is the **smoke test**, not the domain headline; (b) last-position training is a documented simplification of canonical SASRec; (c) absolute numbers remain modest — expected for strict next-item leave-last-out under a global temporal split.
- Sequence-aware modeling helping over collaborative-only ALS is the *expected* direction and the reason SASRec is the headline model for serialized fiction — but that claim only becomes a *domain* claim once run on Goodreads/Amazon.

## 5. Artifacts produced
- `outputs/evidence/sasrec_report.json` — config, epochs, loss history, SASRec metrics + CIs, baselines for comparison.
- `outputs/plots/sasrec_vs_baselines_recall10.png` — Recall@10 with bootstrap CI error bars.
- `outputs/_models/sasrec_phase3a.pt` — checkpoint (resumable).

## 6. No-overclaim check (per `plans/03`)
- Real metrics for MovieLens-1M only, each with N + 95% CI; framed as smoke, not domain headline. ✅
- SASRec win reported with CI overlap honesty; not forced. ✅
- Training simplification (last-position) documented, not hidden; canonical SASRec not claimed. ✅
- No two-tower / simulator / Goodreads / PDF. ✅

## 7. Gate 3A status & decision
- ✅ SASRec implemented, CI-evaluated on the identical held-out set, honestly compared.
- ✅ Result documented as-is (best model on smoke, CIs separate at @10; floor, under-trained).
- ⏸️ Domain headline (Goodreads/Amazon) still pending data load + profiling.

**Gate 3A decision for you:** (a) accept the SASRec smoke result; and choose next — (b) **more SASRec epochs / full-position training** to firm up the smoke number, (c) **load a Goodreads fiction genre / Amazon Books** subset to move to the domain headline, or (d) proceed to **Phase 4 (known-propensity simulator + IPS/SNIPS/DR)** — the core differentiator.

**STOP — awaiting Gate 3A review. No Phase 3B/4 work begins until approved.**
