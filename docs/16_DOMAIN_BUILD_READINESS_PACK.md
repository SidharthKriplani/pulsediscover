# 16 — Domain-Build Readiness Pack

*Preparation only — no code, no downloads, no experiments. This pack specifies exactly what is needed to carry the PulseDiscover spine (baselines → SASRec → SASRec-π OPE → position-bias → feedback-loop) from the MovieLens-1M **smoke** onto a real **fiction-domain** dataset, and exactly what you need to provide. Reminder: I cannot fetch datasets (platform policy) — the file must be placed by you.*

---

## 1. Recommended domain dataset

**Recommendation: a single Goodreads fiction-genre subset (primary); Amazon Books category (fallback).**

| Axis | **Goodreads (UCSD Book Graph) genre subset** | Amazon Reviews 2023 — Books |
|---|---|---|
| Creator / author | **Native `authors` field per book** → real creator-health audit | Author metadata inconsistent/sparse; often absent |
| Series / serialization | **Native `series` field** → directly models serialized fiction | No series concept |
| Cold-start | New books/authors over 2007–2017 span → measurable | New ASINs over time → measurable |
| Catalog-health story | Strong (authors + series + shelves) | Moderate (items only) |
| RecSys storytelling (Pratilipi-like) | **Best** — books≈stories, authors≈creators, series≈serialization, shelving/reading≈engagement | Books domain but e-commerce-shaped, not serialized fiction |
| Sequence modeling | Yes (with a timestamped file — see §2 caveat) | Yes (per-review `timestamp`) |
| Size manageability | **Per-genre subsets are CPU-feasible**; full set is 4–11 GB (too big) | Per-category via HF; Books still large, needs subsetting |
| Timestamp reliability | ⚠️ depends on the file chosen (reviews file has dates; compact interactions file may not) | ✅ every review has a `timestamp` |

**Why Goodreads wins for PulseDiscover:** the whole differentiator narrative is *creator equity, serialization, and catalog health on a fiction platform*. Only Goodreads gives **author_id (creator)** and **series_id (serialization)** natively, which upgrades the Phase-7 audit from item/genre-proxy to a **real creator-level** catalog-health story and makes cold-start about **new authors/series**. Amazon Books is the fallback if Goodreads is impractical to obtain — it has reliable timestamps and scale but weak creator/series structure.

**Suggested genre (smallest that still has rich series/author structure):** `fantasy_paranormal` or `romance` (both large, series-heavy). `poetry` / `comics_graphic` are smaller but thinner on series.

---

## 2. Exact dataset requirements

### 2.1 Required (the build cannot run without these)
A single fiction-genre **interaction file** mapping to the canonical schema, plus a **book-metadata file** for creators/series:

| Canonical field | Goodreads source | Amazon source | Required? |
|---|---|---|---|
| `user_id` | `user_id` | `user_id` | **Required** |
| `item_id` | `book_id` | `parent_asin`/`asin` | **Required** |
| `event_ts` | `read_at` / `date_added` / `started_at` | `timestamp` (ms) | **Required** (temporal split + sequences) |
| `rating` / `event_type` | `rating` / `is_read` | `rating` | **Required** (one of) |
| `creator_id` (author) | `authors[*].author_id` (from books meta) | author field (if present) | **Required for creator-health** |
| `series_id` | `series` (from books meta) | — | Strongly desired (serialization story) |

> ⚠️ **Timestamp caveat (important):** the compact Goodreads `goodreads_interactions_<genre>.json.gz` may **lack timestamps**. Provide a file that has a date field — the **`goodreads_reviews_<genre>.json.gz`** (has `date_added` / `read_at` / `started_at`) or the dated `goodreads_interactions_dedup` — so the leakage-safe temporal split and sequences are possible. No timestamps ⇒ no temporal split ⇒ blocks the build.

### 2.2 Required metadata file
`goodreads_books_<genre>.json.gz` (or equivalent) providing: `book_id` → `authors` (author_id), `series`, `title`, optionally `popular_shelves`/`genres`, `language_code`, `publication_year`.

### 2.3 Minimum viable scale (for a credible domain headline)
- **Interactions:** ≥ ~200k (ideal 1–3M).
- **Users:** ≥ ~5,000 (ideal 20k+).
- **Items (books):** ≥ ~2,000 (ideal 20k+).
- **Sequence density:** average ≥ ~8–10 interactions/user (SASRec needs sequences).
- **Temporal span:** ≥ ~1 year so some items appear only in the test window (cold-start measurable).
- **Creators:** ≥ ~500 distinct authors with multiple books (creator-health needs a tail).

### 2.4 Optional metadata (nice-to-have, not blocking)
Title, shelves/genres, language, publication year, review text, page count (for length-normalized "completion" proxy).

### 2.5 Formats & size
- **Accepted formats:** `.json` / `.jsonl` / `.csv`, optionally `.gz`-compressed, or `.parquet`.
- **Max safe size for the CPU sandbox:** raw interaction file **≤ ~300–500 MB** (must fit in 3.8 GB RAM via pandas with headroom); total `data/raw/` footprint **≤ ~4 GB** (6.8 GB free, leave room). If the only available file is larger, I will **subsample to ≤ ~3M interactions** after you place it — I won't ask you to pre-trim.
- **When a GPU (Colab/Kaggle) is needed:** **only SASRec training** if the subset exceeds ~2–3M interactions or ~50k items, or to train canonical full-position SASRec for more epochs. **Everything else — ingestion, baselines, the simulator, SASRec-π OPE, position-bias, feedback-loop — runs on the CPU sandbox** regardless. The classical/OPE spine is never GPU-bound.

---

## 3. Domain build plan (mirrors the smoke spine, with the creator upgrade)

| Step | What it does | New vs smoke |
|---|---|---|
| **Ingestion** | load genre interactions + book metadata, join on `book_id`, normalize to canonical schema (with `creator_id`, `series_id`) | reuses `src/pulsediscover/data.py` loaders (add a Goodreads/Amazon loader) |
| **Schema mapping** | map source fields → canonical; parse timestamps; dedup | new loader; same canonical schema |
| **Leakage-safe split** | global-time split (0.8/0.9) + constrained leave-last-out | reuses `data.py`; now on real dates |
| **Baselines** | popularity, co-occurrence, ALS — Recall@K/NDCG + CIs | reuses `baselines.py` + `eval.py` |
| **SASRec** | sequential model on real reading sequences; Recall@K/NDCG + CIs, honest comparison (win not required) | reuses `sasrec.py`; GPU only if large |
| **SASRec-π OPE** | known-propensity simulator on the domain data; IPS/SNIPS/DM/DR with π = SASRec; bias/CI/ESS/clip | reuses `simulator.py` + `ope.py` + Option-B driver |
| **Position-bias** | naive vs IPS vs PAL; examination curve; item-movement | reuses `position_bias.py` |
| **Feedback-loop / creator-health** | exposure Gini/entropy/long-tail **per author (creator)** + per series, over rounds; ratchet + mitigation | **upgrade:** real `creator_id` → creator-level concentration (no longer item/genre proxy) |
| **Evidence ledger update** | promote **domain numbers to headline**, demote MovieLens to smoke; tag `[BUILT — real data (domain)]` | update `docs/12` + PRD |

Each step ends at a gate, same as the smoke build. No step is required to "win."

---

## 4. Risk register

| Risk | Likelihood | Impact | Mitigation / fallback |
|---|---|---|---|
| **Dataset too large** (full Goodreads 4–11 GB) | High if full dump | Won't fit sandbox | Use a **single genre subset**; I subsample to ≤3M interactions after placement |
| **Missing timestamps** (compact interactions file) | Medium | Blocks temporal split + sequences | Provide the **reviews file** (has `read_at`/`date_added`) or dated dedup interactions; §2.1 caveat |
| **Missing author/creator fields** | Medium (Amazon) / Low (Goodreads) | No creator-health upgrade | Use Goodreads (native authors); if Amazon, fall back to **item/genre-level** audit (as in smoke) and state it |
| **Sparse interactions** (avg < ~5/user) | Medium | SASRec/sequences weak | Filter to users/items with ≥ k interactions; report density; SASRec may underperform (documented, not forced) |
| **Cold-start not measurable** (no items only in test window) | Low–Medium | Cold-start claim unsupported | Need ≥1-year span; else drop the cold-start claim honestly |
| **GPU unavailable** | Medium | SASRec slow at domain scale | Keep subset ≤2–3M interactions for CPU; or you run SASRec on free Colab/Kaggle and hand back the checkpoint |
| **License / availability** | Low | Can't obtain | Goodreads UCSD + Amazon-2023 are public research datasets (cite Wan & McAuley RecSys'18 / Amazon-Reviews-2023); if a mirror is down, use the other |

---

## 5. Clear user action

### What to provide (pick ONE track)
**Track A — Goodreads (recommended):** two files for one fiction genre —
1. a **timestamped interactions/reviews file**, e.g. `goodreads_reviews_fantasy_paranormal.json.gz` (must contain `user_id`, `book_id`, a date field, `rating`), and
2. the **book metadata** file, e.g. `goodreads_books_fantasy_paranormal.json.gz` (must contain `book_id`, `authors`, `series`, `title`).

**Track B — Amazon Books (fallback):** the per-category **Books reviews** file (e.g. `Books.jsonl.gz` with `user_id`, `parent_asin`, `rating`, `timestamp`) and the **`meta_Books.jsonl.gz`** metadata file.

### Where to place it
- Track A → `data/raw/goodreads/`
- Track B → `data/raw/amazon_books/`
(Create the folder; drop the file(s) in. I auto-detect and ingest from there — no path config needed.)

### Size guidance
- Aim for a **genre subset ≤ ~300–500 MB**. If only a larger file exists, place it anyway and I'll subsample to a CPU-safe size; do **not** spend effort trimming it yourself.

### What NOT to provide
- ❌ The full 4–11 GB Goodreads dump or the 275 GB Amazon full corpus (won't fit; a genre/category subset only).
- ❌ Any file **stripped of timestamps or author/series fields** (those are the load-bearing columns).
- ❌ Pre-aggregated/anonymized data that loses `user_id` sequences or `creator_id`.
- ❌ Any non-public or PII-bearing data — only the official public research datasets.

---

## 6. Status & next step
- **Ready to build the domain spine the moment a qualifying file lands** in `data/raw/`. Estimated effort: ingestion + re-run of all gates, CPU-feasible except possibly SASRec (GPU only if large).
- **Until then:** no domain experiments run; the MovieLens smoke + simulator spine stands as-is; T3 models and the defense PDF remain deferred.

*No code executed, no downloads attempted, no experiments run. Awaiting your dataset placement (or a decision to defer).*
