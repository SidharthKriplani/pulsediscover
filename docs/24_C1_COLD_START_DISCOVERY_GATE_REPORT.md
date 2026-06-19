# 24 — C1 Gate Report: Cold-Start / New-Content Discovery Channel

*Company question: **Can we improve discovery of new or underexposed fiction content without relying only on historical interaction volume?** This builds the first content/metadata candidate channel for new books/authors/series, so PulseDiscover is not only a warm collaborative recommender. Domain-scoped (Goodreads fantasy/paranormal). No OPE / position-bias / creator-health sim / SASRec / PRD / PDF.*

## 1. Content features extracted
From the raw books file (258,585 books) → `domain_content_meta.csv`: `title, creator_id, series_id, language_code, publication_year, popular_shelves (top-5), description (trunc)`. (22 s, streaming — raw file never fully loaded.)

## 2. Cold-start evaluation set (the honest one)
Global-time split, **train = pre-test (ts < t2 = 2016-11-02), test = ts ≥ t2.** Cold = items in test **never seen pre-test**.
- **Cold items: 1,420** · **new creators: 330** · **new series: 898** · cold test interactions: **27,458** (~the 23.5%-new-item signal from D2).
- Eval: **5,000 cold test interactions** (user has pre-test history), gold = the new item. **Not** the warm 5k leave-last-out sample.
- **32.8%** of cold golds have a **warm author** (author the user already read) → the structural ceiling for same-author.

## 3. Cold-start recall (new-in-test items, 5,000 interactions, bootstrap CIs)
| Generator | R@20 | R@50 | R@100 | R@200 |
|---|---|---|---|---|
| **ALS f64** | **0.0000** | 0.0000 | 0.0000 | 0.0000 |
| same-author | 0.2084 [0.198, 0.221] | 0.2876 | 0.3158 | 0.3274 |
| same-series | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| TF-IDF (title+shelves+desc) | 0.1220 | 0.2104 | 0.3148 | 0.4500 [0.436, 0.464] |
| **content-hybrid (RRF)** | **0.2410 [0.229, 0.253]** | **0.3402** | **0.4406** | **0.5566 [0.543, 0.571]** |

## 4. The headline finding
**ALS f64 scores exactly 0 on new items — by construction** (a never-seen item has no factor vector). The content channel turns that into **R@20 0.24 / R@200 0.56**: a real new-content discovery capability that collaborative retrieval *structurally cannot* provide.
- **same-author** is the strongest single lane (R@20 0.21), recovering nearly all of its 32.8% author-warm ceiling by R@200 (0.327) — the "new book from an author you follow" mechanism.
- **TF-IDF** covers the harder case (new authors): R@200 0.45 — content similarity finds new items even when the author is new.
- **content-hybrid** (RRF of same-author + same-series + TF-IDF) is best at every K — author-following sharpens the head, content fills the tail.
- **same-series ≈ 0**: cold golds are rarely the next book in a series the user already read at this horizon — a genuine weak lane, reported.

## 5. Warm vs cold — division of labor (not the same metric)
| | ALS f64 (warm, LLO) | ALS f64 (cold) | content-hybrid (cold) |
|---|---|---|---|
| Recall@20 | 0.0846 | 0.0000 | **0.2410** |

Different eval sets/golds, shown to illustrate the **two-lane design**: ALS f64 owns warm collaborative recall; the content channel owns cold/new discovery. Plot: `outputs/plots/domain_coldstart_recall.png`.

## 6. Coverage & catalog reach
- same-author surfaces **1,025 / 1,420** distinct cold items (587 distinct authors) in top-200.
- **content-hybrid (top-100) surfaces all 1,420 cold items (100% cold-pool coverage)** across users — broad new-content reach, not a few head items.
- (Per-item exposure-concentration Gini for cold items is a light follow-up; coverage is the load-bearing catalog-health number here.)

## 7. Runtime / feasibility
same-author/series eval ~8 s; TF-IDF fit (20k vocab) + hybrid eval (5,000 users) ~10 s — **fully CPU-feasible** in-sandbox. No GPU needed.

## 8. Decision (per the gate's logic) — **KEEP**
The content channel gives **non-zero cold-start recall (hybrid R@20 0.24, R@200 0.56) with 100% cold-pool coverage and fast CPU runtime**, where ALS f64 is structurally 0. **Keep it as the cold-start lane.** same-series is weak → drop as a standalone lane (keep only inside the hybrid). The honest weak case (new authors) is carried by TF-IDF.

## 9. No-overclaim check
- ALS f64 cold = 0 is **structural** (contrast), not a tuning failure — stated. ✅
- same-author ceiling (32.8%) and same-series ≈ 0 disclosed. ✅
- **Offline cold-start *candidate channel* only — NOT proven business lift**; no A/B, no online exposure test, no cold-start "win" beyond offline recall. ✅
- Cold pool = core (k-cored) items new in the test window; sub-threshold brand-new items are out of core scope — stated. ✅
- No OPE / position-bias / creator-health sim / SASRec / PRD / PDF. ✅

## 10. Artifacts
- `outputs/evidence/domain_coldstart_report.json` (recall+CIs, coverage, warm context, decision, notes)
- `outputs/plots/domain_coldstart_recall.png`
- `data/interim/domain_content_meta.csv`, `domain_splits/cold_eval_sample.csv`

## 11. Gate status & next step
- ✅ PulseDiscover now has a **two-lane** generator story: **ALS f64 (warm)** + **content-hybrid (cold/new)** — the cold-start gap (~0 → 0.24@20) is filled offline.
- ⏸️ **Options next:** (a) domain OPE / position-bias / creator-health on ALS f64 (the differentiator, per the tournament plan's recommended gate B); (b) external GPU canonical SASRec; (c) cold-start exposure/creator-concentration simulation to connect this channel to catalog health. No further step run.

**STOP — awaiting C1 review.**
