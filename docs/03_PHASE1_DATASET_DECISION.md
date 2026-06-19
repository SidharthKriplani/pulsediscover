# 03 — Phase 1: Dataset Feasibility & Decision Report (Gate 1)

*Phase 1 scope only: feasibility, schema/sequence viability, leakage-safe split design, dataset decision. **No training, no simulator data, no PDF.** This report is the Gate 1 deliverable; implementation pauses here for review.*

---

## 1. Sandbox environment feasibility (a real Gate 1 finding)

| Check | Result | Implication |
|---|---|---|
| Python | 3.10.12 | fine |
| Present libs | numpy 2.2.6, pandas 2.3.3, matplotlib 3.10.9 | data wrangling + plots OK |
| Missing libs | scipy, scikit-learn, lightgbm, **torch**, faiss, implicit | must `pip install` |
| PyPI reachable? | **Yes** — `pip install scipy` succeeded (1.15.3) | scipy/sklearn/lightgbm installable (CPU) |
| Compute | **4 CPU, 3.8 GB RAM, ~6.8 GB free disk, no GPU** | **constrained** |

**Consequence:** classical work (popularity/co-occurrence/ALS baselines, IPS/SNIPS/DR, the known-propensity simulator, CIs) is **fully feasible in-sandbox** — it's numpy/scipy-light. **SASRec (torch) training and full-Goodreads processing are marginal-to-infeasible here** (no GPU; 3.8 GB RAM; 6.8 GB disk). This is the single most important Gate-1 decision input — see §6 open questions.

---

## 2. Dataset feasibility (verified via web, June 2026)

| Dataset | Size | Records / span | Sequence? | Cold-start? | OPE? | Fiction-domain fit | In-sandbox? |
|---|---|---|---|---|---|---|---|
| **MovieLens-1M** | **6 MB** | 1M ratings, 6k users × 4k movies; `UserID::MovieID::Rating::Timestamp` (UTC s) | yes (timestamps) | partial | no (observational) | weak (movies) | **trivially yes** |
| MovieLens-25M | 262 MB | 25M ratings CSV, to Nov 2019 | yes | partial | no | weak | yes (fits disk) |
| **Goodreads UCSD (full)** | **~4.1 GB CSV / ~11 GB JSON** | 229M interactions, Jan 2007–Nov 2017; +2 GB book metadata; 15M reviews | yes (shelving/read events) | **yes** (new books/authors) | no (observational) | **strong** (books, **series=serialization**, **authors=creators**) | **NO — exceeds 6.8 GB disk / 3.8 GB RAM** |
| **Goodreads per-genre subset** (e.g., fantasy_paranormal, romance, mystery, young-adult) | **hundreds of MB each** | genre slice of the above | yes | yes | no | **strong + fiction-pure** | **yes (single genre)** |
| Amazon Reviews 2023 — Books (fallback) | full corpus 275 GB; **Books loadable per-category via HF `datasets`** | 571M total reviews, May 1996–Sep 2023; user-item graph + metadata | yes (timestamps) | yes | no | medium (books, not serialized) | only via streaming/subset |

**What each proves / cannot prove:** all are observational → **none can prove OPE validity directly** (no logged propensities); they prove model *correctness* and ranking quality. OPE validity comes only from the semi-synthetic simulator (Phase 4) built *on top of* one of these. None proves real online lift.

**Data-acquisition compliance note:** I did **not** download any dataset this turn — Phase 1 is feasibility, and per the environment's content-fetch rules I don't pull dataset URLs via shell/Python. Acquisition is a Gate-1 action item (see §6): either the user drops files in `data/raw/`, or we approve a specific acquisition mechanism (HuggingFace `datasets` for Amazon Books; direct genre files for Goodreads; the 6 MB MovieLens-1M for smoke) at the start of Phase 2.

---

## 3. Schema / timestamp / sequence viability

All three candidates expose **per-event timestamps**, which is what the build needs:
- **Sequences:** order each user's events by timestamp → reading/viewing sequence → SASRec input. ✓ all three.
- **Temporal split:** a global time cutoff is possible on all three. ✓
- **Cold-start:** items first appearing after the cutoff = cold items → exercises the content/cold-start channel. ✓ (richest on Goodreads: new books/authors).
- **Creators/serialization (fiction signal):** **only Goodreads** has authors (creators) and series (serialization) natively — the reason it is the headline candidate over MovieLens/Amazon.

Minimal canonical schema the loader will normalize to (`src/data/`):
`user_id, item_id, creator_id?, series_id?, event_ts, rating?/event_type, position?(none in public data → simulated later)`.
Note: public datasets have **no position/propensity** — those are *manufactured by the simulator* (Phase 4), never claimed as real.

---

## 4. Leakage-safe split design (the Gate 1 design deliverable)

Two complementary splits, both anchored to a **global time cutoff** so no test interaction precedes any train interaction in wall-clock time:

1. **Primary — global time-based split** (`train: t < T1 | val: T1 ≤ t < T2 | test: t ≥ T2`).
   - Purpose: OPE honesty + ranking eval without lookahead.
   - Guards: no feature computed from data after `event_ts`; items/users appearing only in test are flagged **cold**; no `display_rank`/position feature leaks (none exists in public data anyway).
2. **Secondary — leave-last-N-out per user** (standard for sequential next-item eval of SASRec).
   - Purpose: report sequential Recall@K/NDCG@K in the conventional way.
   - **Documented caveat:** pure leave-last-out can leak *global* future context across users; we therefore constrain it to also respect the global cutoff (a user's held-out last item must fall in the test window). This hybrid is the honest version.

**Leakage checklist (enforced in `src/data/`):** temporal monotonicity per user; train/test time-disjointness; no post-event features; cold-item/cold-user tagging; reproducible split seed in `data_manifest.json`.

---

## 5. Dataset decision (recommendation — pending Gate 1 approval)

- **Smoke test:** **MovieLens-1M** (6 MB) — in-sandbox, validates SASRec/baseline correctness fast. *Never a headline number.*
- **Headline domain dataset:** **a single Goodreads fiction genre subset** (e.g., `fantasy_paranormal` or `romance`) — large enough to be real, small enough for the sandbox, and the only candidate with authors + series (the serialized-fiction signal). Exact genre chosen at Phase-2 start by size vs the 6.8 GB disk budget.
- **Fallback:** **Amazon Reviews 2023 — Books** via HuggingFace `datasets` per-category, if the Goodreads genre subset proves impractical to acquire/process.
- **Compute placement:** classical + OPE + simulator → **in-sandbox**. **SASRec training → run on a Goodreads-genre *or* MovieLens-1M-scale subset with a small model (short max-seq-len, few epochs, CPU)**, or on the user's local machine / free Colab/Kaggle GPU if the sandbox proves too tight. (Decision needed — §6.)

---

## 6. Open questions for Gate 1 (need your call before Phase 2)

1. **Where does the build run?** In-sandbox (CPU, 3.8 GB RAM — fine for everything except heavy SASRec) vs your local machine / Colab / Kaggle GPU for the SASRec training step. This sets SASRec scope and the Goodreads subset size.
2. **Dataset acquisition mechanism:** (a) you place files in `data/raw/`, or (b) approve a specific download step (HF `datasets` for Amazon Books; direct genre file for Goodreads; 6 MB MovieLens for smoke) to run at the start of Phase 2.
3. **Which Goodreads fiction genre** (fantasy_paranormal / romance / mystery_thriller_crime / young_adult) — or go straight to Amazon Books? (Recommend: smallest genre that still has rich series/author structure.)
4. **Sequential split preference:** global-time primary (recommended) with leave-last-out secondary — confirm acceptable.

---

## 7. Gate 1 status

- ✅ Environment feasibility assessed (libs installable; compute constrained; SASRec/full-Goodreads are the bottlenecks).
- ✅ MovieLens smoke path confirmed feasible in-sandbox.
- ✅ Goodreads feasibility checked (full too large; **genre subset is the viable path**).
- ✅ Amazon Books fallback checked (per-category via HF).
- ✅ Schema/timestamp/sequence viability confirmed for all three.
- ✅ Leakage-safe split designed (global-time primary + constrained leave-last-out secondary).
- ✅ Dataset decision drafted (MovieLens smoke → Goodreads genre headline → Amazon Books fallback).
- ✅ `pulserank` evidence vendored to `archive/pulserank_evidence/` (provenance only).
- ⏸️ **No models trained. No simulator data generated. No PDF.**

**STOP — awaiting Gate 1 review and answers to §6 before starting Phase 2 (retrieval baselines).**

---
*Sources (dataset facts, verified June 2026): [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) · [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/) · [UCSD Book Graph (Goodreads)](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html) · [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) · [McAuley-Lab/Amazon-Reviews-2023 (HF)](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023). Sandbox library/compute facts verified by direct probe this session.*
