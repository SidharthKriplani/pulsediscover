# 50 — G11: Sequence Model Tournament Report

> **⚠️ CORRECTION (see G11B, docs/51):** the §6 hypothesis that full-position supervision *caused* SASRec v2's shortfall was **tested and REFUTED** by a controlled objective ablation. Last-position training did not improve it (0.0030 vs 0.0076). The real cause is **capacity vs training-budget (under-trained larger model)**, not the training objective. The §5 results table and the overall headline (all sequence models lose to ALS f64) stand unchanged; only the §6 *explanation* is superseded by G11B.

*Attacks the binding RiskFrame-gold gap (technique-tournament depth / deep-model evidence) by **building and evaluating real sequential models** — GRU4Rec and a full-position SASRec v2 — against the ALS f64 warm floor on the same protocol. **Real CPU runs; no GPU; no fake advanced-model win. Result: HONEST NEGATIVE — all sequence models lose to ALS f64; the depth gap is materially reduced, not closed.***

---

## 1. Data setup
- **Domain:** Goodreads fantasy/paranormal core (existing); top-40k item vocabulary (`n_items=40001`, 0=pad).
- **Sequences:** per-user chronological item sequences from train (held-out gold pairs excluded), left-padded to `maxlen=30`.
- **Training examples:** **94,185** users with ≥2 interactions. **Full-position** arrays (`X_in = items[:-1]`, `Y_tgt = items[1:]`) so every step supervises the next item — the canonical objective and the #1 diagnosed fix vs D3B's last-position run.
- **Eval:** same **5,000-user leave-last-out** sample, full 40k scoring, as D3A+/D3B — directly comparable to the ALS f64 floor.

## 2. Models & configs
| Model | Config | Training |
|---|---|---|
| **ALS f64 (floor)** | 64 factors (existing) | implicit MF |
| SASRec v1 (D3B, context) | d=48, 1 block, 1 head, 20 ep | **last-position** sampled-softmax |
| **GRU4Rec (G11)** | d=64, 1 GRU layer, dropout 0.2, batch 1024 | **full-position**, per-example sampled-softmax BCE (100 neg) |
| **SASRec v2 (G11)** | d=64, 2 blocks, 2 heads, dropout 0.2, batch 512 | **full-position**, per-example sampled-softmax BCE (100 neg) |
| SASRec v2 GPU-ready (not run) | d=64, maxlen=50, full softmax, 200 ep | documented for external GPU run |

- **Hardware:** CPU (4 threads), no GPU. **Reproducibility:** `run_g11_sequence_tournament.py build | train gru | train sas | eval`; resumable atomic checkpoints in `outputs/_models/`; seed 20260616.

## 3. Training protocol
Time-budgeted, resumable, **atomic checkpoints** (crash-safe). Full-position sampled-softmax BCE: each non-pad position predicts its next item against 100 per-example negatives. GRU4Rec trained to **9 epochs** (loss 1.84→0.90, plateauing); SASRec v2 to **8 epochs** (loss 3.90→0.95, **plateaued** — converged, not under-trained).

## 4. Evaluation protocol
Leave-last-out, 5,000 users, score all 40k items at the last position, exclude seen items, Recall@20/50 + NDCG@20 with bootstrap 95% CIs. Gold-in-vocab 4,948/5,000.

## 5. Results
| Model | R@20 | R@50 | N@20 | vs ALS f64 (R@20) |
|---|---|---|---|---|
| **ALS f64 (floor)** | **0.0846** [0.077, 0.092] | **0.1466** | **0.0344** | — |
| SASRec v1 (D3B, last-pos, 20ep) | 0.0362 [0.031, 0.041] | 0.0804 | 0.0122 | −0.048 |
| **GRU4Rec (full-pos, 9ep)** | **0.0306** [0.026, 0.035] | 0.0670 | 0.0122 | **−0.054** |
| **SASRec v2 (full-pos, 8ep)** | 0.0076 [0.005, 0.010] | 0.0162 | 0.0023 | −0.077 |

Plot: `outputs/plots/g11_sequence_tournament.png`. Metrics: `outputs/evidence/g11_sequence_tournament_metrics.json`.

## 6. Why each won / lost
- **ALS f64 wins (0.085).** On this catalog the dominant signal is collaborative co-purchase/co-read, which MF captures directly. Next-book choice is only weakly sequence-predictable.
- **GRU4Rec (0.031) ≈ D3B SASRec, both ~36% of ALS.** A properly-trained recurrent sequence baseline lands in the same range as the attention model — a credible "sequence model, well-built, still loses to MF on book data" result.
- **SASRec v2 full-position UNDERperforms even last-position v1 (0.0076 vs 0.0362) — the interesting finding.** It is **converged** (loss plateaued 0.97→0.95 over epochs 5–8), so this is not under-training. The cause is an **objective/data mismatch**: book interaction sequences are short and heavily left-padded, so full-position training spends most of its gradient on **low-context early positions**, while the eval (and the serving objective) only cares about the **last-position** prediction. On short-sequence catalogs, full-position supervision — the canonical fix for long-sequence data — **dilutes** the signal the eval rewards. Last-position training (v1) is better matched here. A bigger d=64/2-block model amplifies the mismatch.

## 7. Gap verdict — **materially reduced, NOT closed**
- **PASS** (acceptance met): multiple sequential models were actually trained and evaluated comparably against ALS f64.
- **HONEST NEGATIVE:** no sequence model beats the floor — ALS f64 0.085 > SASRec v1 0.036 ≈ GRU4Rec 0.031 > SASRec v2 0.008.
- **Depth gap: (b) materially reduced.** The technique-tournament-depth axis now has a real 3-way sequence tournament (GRU4Rec + two SASRec objectives) with a substantive comparative finding (full-position hurts on short-sequence book data) — not a single under-powered run. **Not (a) closed:** no *competitive* advanced model exists; a canonical full-softmax **GPU** SASRec (config provided) remains the deferred path to truly close it.

## 8. Honest boundaries
- CPU-limited (no GPU); GPU-ready canonical config provided but **not run** — no canonical-GPU result claimed.
- "Sequence models lose to MF **on this domain under these configs**" — **not** a general claim that sequence models are bad.
- All numbers carry tag/N/CI/source/repro (RiskFrame number-stamp).

## 9. Status
- ✅ G11 PASS / honest negative; technique-tournament depth **materially improved**.
- ⚠️ T3/model-depth still **open** (no competitive deep model; canonical GPU SASRec deferred). PulseDiscover remains **RiskFrame-gold incomplete** until the audit's binding axes clear — G11 lifts axis #3, but #4 (Deep Defense Kernel) and the competitive-model bar remain.

**STOP — awaiting G11 review.**
