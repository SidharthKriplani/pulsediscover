# 17 — Domain Ingestion Inspection (Goodreads fantasy_paranormal)

*Inspection only — no full load, no build. Done via a bounded sample pass (first 300k reviews + 100k books streamed to a temp file; the 3.88 GB reviews payload was never fully loaded). Totals below are estimates from byte-size + the sample; they will be made exact during capped ingestion.*

## 1. File facts
| File | Compressed | Uncompressed | Est. records |
|---|---|---|---|
| `goodreads_reviews_fantasy_paranormal.json.gz` | 1.26 GB | **3.88 GB** | **~3.55M reviews** (≈1,093 B/line) |
| `goodreads_books_fantasy_paranormal.json.gz` | 279 MB | 1.29 GB | **~258k books** (≈4,973 B/line) |

The reviews payload is dominated by `review_text` (which the build does **not** need) — so streaming-extract a few columns and the working interaction table shrinks ~20×.

## 2. Schema (confirmed)
- **Reviews:** `user_id, book_id, review_id, rating, review_text, date_added, date_updated, read_at, started_at, n_votes, n_comments`.
- **Books:** `book_id, work_id, title, authors[ {author_id,role} ], series[], language_code, popular_shelves, publication_year, num_pages, average_rating, ratings_count, description, isbn/asin, …`.

## 3. Inspection findings (from the sample)
| Property | Finding | Implication |
|---|---|---|
| **File ordering** | **User-ordered** (93.7% consecutive same-user; 0% same-book) | the 300k sample = ~18.8k *complete* user histories → density estimates are trustworthy; **and a user-level subsample is trivial** (take the first K users) |
| **Timestamps** | `date_added` **100%**, `read_at` 84%, `started_at` 59% | use **`date_added`** as `event_ts` (fallback `read_at`) → temporal split + sequences fully supported |
| **Ratings** | rating>0 in **97%** (3% are shelved-not-rated, rating 0) | reliable engagement signal; keep rating 0 as an implicit "shelved" event or drop |
| **Density** | interactions/user: **mean 15.9, median 4, max 1365** | enough for sequences; the median-4 tail will be trimmed by k-core |
| **Authors (creator)** | **100%** of books have ≥1 author; avg **1.47 authors/book** | **real creator-health** audit unlocked (the upgrade over the smoke's item/genre proxy) |
| **Series** | **72.4%** of books in a series | strong **serialization** story for fiction |
| **Language** | `language_code` English ~**56%** (44% non-Eng/blank) | optional English-only filter; or keep all and note multilingual |

## 4. Estimated totals (to be confirmed during ingestion)
- **Interactions:** ~3.55M · **Users:** ~223k (sample ratio 18.8k users / 300k reviews, file is user-ordered → reliable) · **Books:** up to ~258k catalog (interactions likely touch ~150–250k).
- These comfortably clear the §2.3 minimum-viable bar in the readiness pack (≥200k interactions / ≥5k users / ≥2k items / ≥1-yr span).

## 5. Suitability verdict — **strong fit for PulseDiscover**
Real timestamps, real authors (creator_id), real series, good density. This dataset upgrades the spine in exactly the ways the smoke couldn't: **creator-level catalog-health**, **series/serialization**, and **new-author/new-book cold-start**.

## 6. Recommended safe subsampling / ingestion plan (proposed — for approval)
**D1 — Streaming extraction (never full-load):**
- Stream `reviews.gz` line-by-line; keep only `user_id, book_id, rating, date_added(→unix ts)`; **drop `review_text`** → compact interactions (~3.55M rows ≈ ~180 MB parquet/csv). Fits RAM easily.
- Stream `books.gz`; keep `book_id → primary author_id (creator_id), series_id (first series), title, language_code` → books_meta (~258k rows, small).

**D2 — Filter to a CPU-safe core (the subsample):**
- **k-core:** users with ≥5 interactions, books with ≥10 interactions (drops the median-4 long tail; densifies for SASRec).
- **Optional** English-only (`language_code∈{eng,en-*}`) — decide at D2; default **keep all**, flag multilingual.
- **Target working set:** ~1.5–2M interactions / ~60–90k users / ~30–50k books after k-core. Exact numbers reported at D2.

**D3 — SASRec scale control (CPU vs GPU):**
- The SASRec next-item softmax is over the item vocabulary; at ~50–250k items it is **CPU-marginal**. Mitigation for a CPU headline: **cap the item vocabulary to the top-N popular books (e.g., 40k)** and/or **subsample to ~50k users**, with **resumable per-epoch checkpointing** (as in the smoke). Full-catalog / canonical SASRec → **free Colab/Kaggle GPU** (you run it, hand back the checkpoint).
- **Everything else (baselines, simulator, SASRec-π OPE, position-bias, feedback/creator-health) runs at the working-set scale on CPU.**

**D4 — Build (gated, same spine):** baselines → SASRec → SASRec-π OPE → position-bias → **creator-level** feedback-loop audit → evidence-ledger update (promote domain numbers to headline, demote MovieLens to smoke).

## 7. Risks / notes
- **Item count large for SASRec softmax** → cap vocabulary or use GPU (above). Classical baselines/OPE unaffected.
- **ALS / co-occurrence memory** at full ~258k items: co-occurrence `MᵀM` can get large → run co-occurrence on the **k-cored** item set (≤~50k) to bound memory.
- **Totals are estimates** (sample + byte-size; user-ordered file makes the user estimate solid, the item/interaction totals approximate) — D1 will report exact counts.
- **3% rating-0** and **~44% non-English** — handled by explicit D2 choices, not silently.
- Temp samples live in `/tmp` (sandbox-ephemeral), not in your repo.

## 8. Next step (not executed)
Approve the D1→D2 ingestion config (k-core thresholds; English-only yes/no; SASRec item-cap / CPU-vs-GPU). On approval I run **D1 streaming extraction only** and report exact counts before any modeling.

*No full build run. Awaiting your go on the ingestion config.*
