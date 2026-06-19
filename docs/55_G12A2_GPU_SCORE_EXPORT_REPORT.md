# 55 — G12-A2: GPU Score Export for LTR (run template)

*Fills the three pending GPU columns in the G12-A candidate table by re-training the GPU models (saving checkpoints this time) and scoring the exact candidate rows. **This sandbox has no GPU — GPU values are PENDING the Colab run.** No fabricated scores; eval unchanged; no LTR training here.*

*Artifacts to produce (Colab): `g12_candidate_feature_table_enriched.csv.gz`, `g12_gpu_score_export_summary.json`, `ckpt_canonical.pt` / `ckpt_small.pt` / `ckpt_twotower.pt`.*

## 1. How to run
Open **`G12A2_GPU_Score_Export.ipynb`** (in `g11c_upload/`) in Colab → Runtime ▸ GPU → run cells. Cell 2 **writes the export script itself** (no stale-script risk; verify cell prints ≥3). Upload **3 files** (Cell 4): `g11_fullpos_arrays.npz`, `d3b_seq_meta.pkl`, `g12_candidate_feature_table.csv.gz` (all staged in `g11c_upload/`). Mirror script: `src/run_g12a2_export.py`. **Scoring pipeline smoke-tested locally** (random-init model filled all 24,439 test rows, 0 NaN) — the fill mechanics are validated; the Colab run supplies real GPU weights.

## 2. What it does
1. **Re-trains and SAVES checkpoints** (the fix for last time): canonical full-softmax SASRec (d64/2blk/2head, 40ep), SASRec-small (d48/1blk/1head, last-pos, 20ep), TwoTower (pooled-history in-batch-neg, d64, 30ep).
2. **Scores the exact G12-A candidate rows** via each model's `score_last(history)` → fills `sasrec_canonical_gpu_score`, `sasrec_small_gpu_score`, `twotower_gpu_score`. NaN only if an item is outside the 40k SASRec vocab (none expected — all candidates were in-vocab in the smoke test).

## 3. Results to record *(PENDING)*
| Field | Value |
|---|---|
| GPU type / runtime | **PENDING** |
| GPU / runtime | **Tesla T4 / 1,516 s** |
| Candidate rows scored | **613,896** |
| Users scored | **5,000** |
| Missing `sasrec_canonical_gpu_score` | **411,880 (3,397 users — §3a)** |
| Missing `sasrec_small_gpu_score` | **0** |
| Missing `twotower_gpu_score` | **0** |
| Checkpoints saved | ✅ ckpt_canonical / ckpt_small / ckpt_twotower (`outputs/_models/gpu/`) |
| Leakage check (`gold_in_seen` == 0) | ✅ **0** |

### 3a. Why canonical is NaN for 3,397 users (corrected — verified in G12-B)
Diagnosed: 3,397 users **fully** NaN, 1,603 **fully** filled — zero partial. The split is **exact at history_len < 30 (= maxlen)**; NaN users have history 2–29 (median 10), OK users have ≥30. **Not empty history** (none are empty). Cause: the **2-block** canonical SASRec produces NaN at the last position whenever the sequence is **left-padded** — masked self-attention over the padding positions yields NaN, which propagates to the last position through the *second* attention block. The **1-block** SASRec-small is immune (no second block to leak padding-NaN), which is why it scored all users. A real, fixable implementation limitation (padding-safe attention / right-padding), surfaced honestly via `has_sasrec_canonical_gpu_score`; LightGBM treats NaN as missing.

## 4. Model configs (fixed, = G11C/D)
- **canonical:** SASRec d64/2blk/2head, full-softmax over 40k, 40 ep (R@20 0.0456 in G11C/D)
- **small:** SASRec d48/1blk/1head, last-position sampled-neg, 20 ep (R@20 0.0594 — best sequence)
- **twotower:** pooled-history user tower + item tower, in-batch negatives, d64, 30 ep (R@20 0.0020)

## 5. Honest notes
- **TwoTower was weak standalone (R@20 ≈ 0.0020)** but is **still included as a candidate feature** — a learned ranker can down-weight or exploit even a weak signal; excluding it pre-judges. We let the LTR decide its importance.
- Scores are `dot(user_repr, item_emb)` (raw logits, not probabilities) — fine as ranker features.
- Re-training is required because the original G11C/D run didn't checkpoint the models; this run **saves them** so future passes skip training.
- Eval protocol unchanged; **no LTR trained in this gate.**

## 6. After the run
Drop `g12_candidate_feature_table_enriched.csv.gz` + `g12_gpu_score_export_summary.json` (and the 3 checkpoints) back into the repo. Then **G12-B**: retrain LambdaMART on the enriched table and measure the lift from adding the strong GPU SASRec signals (the real question). FAISS still deferred.

## 7. Status
- ✅ G12-A2 **artifacts prepared + scoring smoke-tested**: notebook, export script, PENDING summary stub, this report.
- ⏸️ Awaiting the Colab/GPU run to fill the 3 columns and save checkpoints.

**STOP — score-export artifacts ready; GPU run is the user's to execute.**
