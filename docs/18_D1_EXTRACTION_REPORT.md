# 18 — D1 Streaming-Extraction Report (Goodreads fantasy_paranormal)

*D1 only: streaming extraction of compact files from the gz dumps. The 3.88 GB reviews payload was streamed via `gzip -dc` and never fully loaded; `review_text` and all heavy fields were dropped. No k-core, no English filter, rating-0 rows kept (counted). No baselines/SASRec/OPE/position-bias/feedback. Source of numbers: `data/interim/d1_stats.json`.*

## 1. Compact files produced (`data/interim/`)
| File | Rows | Size | Columns |
|---|---|---|---|
| `domain_interactions.csv` | 3,424,641 | 186 MB | `user_id, book_id, rating, event_ts` |
| `domain_books_meta.csv` | 258,585 | 15.6 MB | `book_id, creator_id, series_id, title, language_code` |

Both fit in RAM comfortably (the 20× shrink from dropping `review_text` worked as planned).

## 2. Exact counts
- **Interactions:** 3,424,641 · **Unique users:** 256,088 · **Unique items (books seen):** 258,212 · **Catalog (books meta):** 258,585.
- **Density:** interactions/user mean **13.37**, median **3**, p90 **29**.

## 3. Data quality
| Check | Result |
|---|---|
| Missing user_id / book_id / event_ts | **0 / 0 / 0** (every row complete) |
| Duplicate (user, book) rows | **0** (0.000%) — unique_pairs = rows |
| rating-0 (shelved-not-rated) | **113,762 (3.32%)** — kept, flagged |
| Rating distribution | 0:113,762 · 1:93,975 · 2:228,683 · 3:646,191 · 4:1,133,818 · 5:1,208,212 (skewed to 4–5) |
| Date range (`date_added`) | **2001-05-15 → 2017-11-04** (~16.5 years) |

Zero missingness, zero duplicates, full timestamp coverage — unusually clean.

## 4. Metadata coverage (books)
- **Author (creator_id): 100.0%** → creator-level catalog-health is fully supported.
- **Series: 72.39%** → strong serialization signal.
- **Language present: 72.67%**; English (`eng`/`en-US`/`en-GB`) ≈ **145.7k books (~56%)**; top non-English: spa, ger, ita, fre, por.

## 5. Post-filter estimate (one-pass, non-iterative)
- **users ≥ 5 interactions:** 103,845 (of 256,088 → drops the median-3 long tail).
- **items ≥ 10 interactions:** 45,301 (of 258,212).
- **interactions retained if user≥5 AND item≥10:** **2,615,837 (~76% of all interactions)**.

*(One-pass approximation; true iterative k-core will differ slightly. Net: a dense ~2.6M-interaction / ~104k-user / ~45k-item core is available — a strong working set.)*

## 6. Readiness verdict
**Clean and rich — ready for the domain build.** Real timestamps (16-yr span → cold-start measurable), zero missingness/dups, 100% authors (creator-health), 72% series (serialization). The only scale consideration is the item vocabulary for SASRec's softmax (≈45k after k-core, ≈258k full) — CPU-marginal, so cap top-N items and/or use a GPU for SASRec; everything else (baselines, simulator, SASRec-π OPE, position-bias, creator-health audit) runs on CPU at the working-set scale.

## 7. Proposed working set for the domain build (for approval — NOT executed)
- **Filter:** iterative k-core users≥5 / items≥10 → ~2.6M interactions / ~104k users / ~45k items (decide English-only vs keep-all at D2; default keep-all, flag multilingual).
- **SASRec:** cap item vocab to top ~40k by popularity (covers the ≥10 set) for CPU feasibility with resumable checkpointing; or run on free Colab/Kaggle GPU for full vocab.
- **Split:** global-time (2001–2017) primary + constrained leave-last-out; cold-start = books first seen in the test window.

## 8. Status
- ✅ D1 extraction + stats complete; compact files + `d1_stats.json` written; `D1_DONE` marker set.
- ⏸️ No modeling run. Awaiting approval of the D2 filter/working-set config before any baselines/SASRec/OPE.

*Stop after D1 report, as instructed.*
