# 19 — D2 Working-Set Report (Goodreads fantasy_paranormal core)

*D2 only: iterative k-core + CPU-safe working set + leakage-safe **split metadata** (no train/val/test files written, no modeling). Config: users≥5 & items≥10 (iterative); all languages kept; rating-0 kept + flagged. No baselines / SASRec / OPE / position-bias / feedback / PDF / T3. Source: `data/interim/d2_stats.json`.*

## 1. Working-set files produced (`data/interim/`)
| File | Rows | Size | Purpose |
|---|---|---|---|
| `domain_core_interactions.csv` | 2,557,588 | 143.7 MB | core interactions (`user_id, book_id, rating, rating0_flag, event_ts`) |
| `domain_core_books_meta.csv` | (core items) | 2.6 MB | `book_id, creator_id, series_id, title, language_code` |
| `domain_sasrec_item_vocab_top40k.csv` | 40,000 | 0.46 MB | top-40k item vocab view (SASRec feasibility; **not trained**) |

## 2. k-core result (users≥5 & items≥10, iterative)
- Converged in **10 iterations**.
- **Interactions: 2,557,588** (from 3,424,641 → 74.7% retained)
- **Users: 94,185** (from 256,088) · **Items: 41,961** (from 258,212)
- **Creators (authors): 10,389** · **Series: 25,042**
- **Density after k-core:** mean **27.2**, median **12**, p90 **60**, max 1,983 (median jumped 3 → 12 — the long tail is gone).
- **Date range (core):** 2001-07-01 → 2017-11-03.

## 3. Rating & language
- **rating-0 (shelved-not-rated): 79,354 (3.1%)** — kept + `rating0_flag`. Dist: 0:79,354 · 1:67,682 · 2:168,823 · 3:485,264 · 4:861,400 · 5:895,065.
- **Language (by core item, top):** eng 25,849 · (none) 6,191 · en-US 5,109 · en-GB 2,280 · spa 699 · en-CA 407 · ita 338 · ger 311 · ind 306 · por 93. (English ≈ 80% of language-tagged items; all languages kept.)

## 4. Top-40k item-vocabulary view (for later SASRec)
- 40,000 items covering **99.23%** of core interactions → SASRec can use a ~40k softmax vocab and lose almost nothing. (Core has 41,961 items total, so the cap is barely binding — SASRec is CPU-marginal-but-feasible here, or use a GPU.)

## 5. Leakage-safe split metadata (global-time; NO split files written)
Cutoffs (80/90 percentile of `event_ts`): **t1 = 2016-02-17**, **t2 = 2016-11-02**.

| Split | Interactions | Users | Items | Creators | Series |
|---|---|---|---|---|---|
| train (`ts<t1`) | 2,046,070 | 87,667 | 38,389 | 9,497 | 22,756 |
| val (`t1≤ts<t2`) | 255,759 | 45,095 | 27,179 | 6,886 | 16,517 |
| test (`ts≥t2`) | 255,759 | 42,683 | 26,951 | 6,690 | 16,298 |

**Constrained leave-last-out feasibility:** users whose **last** interaction falls in the test window = **42,683** (eligible held-out eval set — ~35× larger than the MovieLens smoke's 1,209); val-window = 13,551.

## 6. Cold-start (new vs train) — measurable and substantial
| Window | New items | New users | New creators | New series | % interactions on new items |
|---|---|---|---|---|---|
| val vs train | 2,152 | 4,065 | 562 | 1,388 | 10.2% |
| test vs train | 3,378 | 5,914 | 852 | 2,151 | **23.5%** |

→ real cold-start at the item, **creator (author)**, and series level — the thing MovieLens couldn't provide.

## 7. Memory / size estimates (CPU-safe confirmed)
- On-disk: core interactions 143.7 MB + meta 2.6 MB + vocab 0.46 MB.
- In-memory for modeling: interactions as int32 codes ≈ **30.7 MB**; SASRec item embeddings (d=48, 40k vocab) ≈ **7.7 MB**; ALS factors (16, 94k users + 42k items) ≈ **17.4 MB**. Comfortably within the 3.8 GB sandbox.

## 8. Readiness verdict
The CPU-safe domain working set is built: **2.56M interactions / 94k users / 42k items / 10.4k creators / 25k series**, dense (median 12), with real timestamps, **measurable cold-start** (incl. new authors/series), and a 42.7k-user leave-last-out eval set. Memory is tiny; only SASRec's softmax is mildly scale-sensitive (handled by the 40k vocab view or a GPU). This is a genuine fiction-domain headline substrate.

## 9. Status & next step (not executed)
- ✅ D2 working set + split metadata complete; `d2_stats.json` + `D2_DONE` written.
- ⏸️ No split files materialized, no modeling. **Next (on approval):** D3 — materialize train/val/test (global-time + constrained leave-last-out) and run the domain spine gate-by-gate (baselines → SASRec → SASRec-π OPE → position-bias → creator-level feedback-loop), reusing the existing `src/` modules.

*Stop after D2 report, as instructed.*
