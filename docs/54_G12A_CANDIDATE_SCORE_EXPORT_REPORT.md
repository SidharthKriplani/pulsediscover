# 54 — G12-A: Candidate Score Export for LTR

*Builds the per-(user,candidate) feature table the ranker consumes, on the **5,000 LLO eval_sample users** (same set the SASRec models were evaluated on). All locally-computable signals are filled now; the **GPU canonical / SASRec-small / two-tower score columns are NaN placeholders** (those weights live in Colab) — to be filled by a score-export run. **No LTR training here; no FAISS; no fake/online claims.** (Numbered 54 — G12's LTR report already occupies docs/53.)*

*Artifacts: `outputs/evidence/g12_candidate_feature_table.csv.gz`, `g12_candidate_feature_schema.json`, `g12_score_export_summary.json`.*

## 1. Table summary
| Quantity | Value |
|---|---|
| Users | **5,000** (eval_sample LLO) |
| Candidate rows | **613,896** |
| Avg candidates / user | **122.8** |
| Positives (gold in candidates) | 754 |
| **Positive coverage rate** | **0.1508** (15% of golds are in the candidate set → ranker recall ceiling) |
| Warm-gold users | 4,346 / 5,000 |
| Build time | 26.7 s (CPU) |

## 2. Candidate construction
Per user, union (deduped, source-flagged) of:
- **ALS f64 top-60** (`came_from_als`) — 281,400 rows
- **D3B SASRec (CPU) top-60** (`came_from_sasrec`) — 300,000 rows
- **Popularity top-30** (`came_from_pop`, hard/popularity negatives) — 150,000 rows

Source flags preserved so the ranker (and we) can see provenance. Dedupe keeps first-seen order.

## 3. Feature list
**Local (filled now):** `als_score`, `als_rank`, `sasrec_cpu_d3b_score`, `sasrec_rank`, `log_pop`, `pub_year` (recency), `warm_flag`, `seen_flag`, `author_match`, `series_match`, `came_from_als`, `came_from_sasrec`, `came_from_pop`.
**Pending GPU export (NaN placeholders):** `sasrec_canonical_gpu_score`, `sasrec_small_gpu_score`, `twotower_gpu_score`.
**Label:** `1` if candidate == held-out gold else `0`.

## 4. Missing-feature handling
- `als_score` = NaN for cold / no-factor items (no ALS embedding) — and for the ~310 users without an ALS user-factor.
- `sasrec_cpu_d3b_score` = NaN if the item is outside the 40k SASRec vocab.
- GPU columns = NaN for all rows (weights in Colab) — explicit placeholders, not silently dropped.
- Ranker handling: LightGBM consumes NaN natively (learns a default direction), so missing-as-signal is preserved.

## 5. Leakage checks
- **`gold_in_seen_history_rows = 0`** ✅ — the held-out gold never appears as a `seen_flag=1` row (LLO integrity confirmed).
- All candidate generators and features are **train-time only** (ALS factors, D3B SASRec, train-popularity, pre-cutoff history) — no post-cutoff/future signal.
- `seen_flag` marks the user's own history items so the ranker/serving layer can exclude already-seen items.

## 6. Source contribution counts
ALS 281,400 · SASRec 300,000 · popularity 150,000 candidate-rows (with overlap deduped at the union level). SASRec and ALS contribute comparably; popularity adds a fixed hard-negative floor.

## 7. Honest limitations
- **The strongest signals (GPU canonical full-softmax 0.046, SASRec-small 0.059) are NOT yet in the table** — their weights weren't saved from the Colab run. Filling them needs a score-export pass (see §8). The current local SASRec column is the weaker D3B model (~0.036).
- **Positive coverage is only 15%** — 85% of golds aren't in the 123-item candidate set, capping any ranker. Widening candidate top-N (e.g., ALS top-200) raises the ceiling at the cost of table size; deferred.
- **No content/cold-lane candidate source** added (eval_sample is ~87% warm-gold); `author_match`/`series_match` carry the content signal instead.
- Single held-out gold per user (binary relevance).

## 8. Next step — fill the GPU score columns (then train LTR)
A Colab score-export pass: in the G11D batch, **(a) save the trained canonical-SASRec and SASRec-small checkpoints**, then **(b) score every row of `g12_candidate_feature_table.csv.gz`** (each user's candidates via `score_last(history)`) and write the three GPU columns. Merge back, then **G12-B**: retrain LambdaMART on the enriched table and measure the lift from adding the strong GPU signals. (FAISS still deferred until an embedding worth serving exists.)

## 9. Status
- ✅ G12-A done — candidate feature table + schema + summary built and validated (leakage-clean); local features filled, GPU columns staged as placeholders.
- ⏭️ Pending: GPU score-export pass → G12-B LTR retrain with full signal set.

**STOP — score export complete; awaiting review (next: GPU score-export pass, then LTR retrain).**
