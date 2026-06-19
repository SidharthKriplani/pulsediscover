# G24 — Cold-Start + Sparse-Cohort Fallback Quality (V2)

*PulseDiscover **V2** lane. Moves from "fallback exists" (G23) to **"fallback quality and degradation are measured"** across cold/sparse cohorts. Evidence: `outputs/evidence/g24_cold_start_sparse_cohort_report.json`; plots `outputs/plots/g24_cold_start_degradation_curve.png`, `g24_head_tail_exposure.png`; code `src/eval/cold_start_cohorts.py`, `scripts/run_g24_cold_start_eval.py`, `scripts/run_g24_serving_integration.py`.*

> **Scope guardrails (read first):**
> - **G24 uses the served G23 `c2_als.pkl` protocol** (ALS f64 — the exact model G23 serving loads), full-catalog ranking, single held-out gold per user.
> - **Absolute recall is NOT comparable to the V1 `d3aplus` headline** (R@20 0.0846). Different model build, user/item universe, and eval protocol.
> - **G24 measures serving / fallback cohort behaviour, NOT V1 model superiority.** It does not re-litigate or upgrade any V1 result.
> - **Cold-start is NOT solved; fairness is NOT claimed.** This gate measures the *cost* of falling back. V1 (`gold_candidate` 8.9) is untouched.

## 1. Cold-start taxonomy
**User cold-start** — too little signal to personalize: `unknown` (0 train history, n≈1062), `sparse_1_2` (1–2, n≈761), `low_3_5` (3–5, n≈817), vs `warm_6plus` (6+, n≈7360, the personalizable cohort). A `no_factor` cohort was defined but did not materialize (every train user with history has an ALS factor).
**Item cold-start (offline-split sense)** — the target item has no collaborative signal: **57.2% of held-out gold items are unseen in train under this offline split.** (This is an offline-split property — NOT a claim about real newly launched catalog items being solved/unsolved in production.) No policy here retrieves these unseen items by content, so recall on them is structurally near zero. Measured, **not** solved.

## 2. What the data supports vs what it cannot prove
**Supported:** per-cohort Recall@20/@50, fallback hit-rate, empty-response rate, catalog coverage, head/mid/long-tail exposure shares, and a slate-personalization measure — all on the served model, offline. **Cannot prove:** online behaviour, that unknown-user popularity is "personalization", item cold-start retrieval, or fairness (exposure is described, not optimized).

## 3. Headline finding (honest, and not the expected one)
Per-cohort **Recall@20 is non-monotonic** in history length: cold/sparse cohorts score **equal-or-higher** than warm (e.g. blended R@20 unknown ≈0.049 vs warm_6plus ≈0.014). Reason: a low-history user's next item is more **popularity-correlated** and easier to retrieve, while a warm user's single held-out gold is more niche among 40k items. **Recall alone therefore UNDERSTATES the cold-start cost.**

The real, measured cost of falling back to popularity for cold/unknown users:

| Metric (top-20 slate) | ALS warm (served) | Popularity fallback |
|---|---|---|
| Unique items recommended | **3,532** | **60** |
| Catalog coverage | 8.7% | 0.15% |
| Long-tail exposure share | 0.4% | 0.0% |
| Slate personalization (distinct slates / users) | **0.68** | **0.001** |

Falling back collapses catalog coverage **~59×** and serves **one near-identical popular slate to every cold user** (personalization ≈0). That is the cold-start cost — a personalization/coverage/long-tail collapse, not a recall drop. (Catalog is head-heavy overall: even ALS is ~93% head exposure; the sharp differentiator is unique-item coverage, not tier share.)

## 4. Policies evaluated
popularity (global top-N) · genre/content (popular items in the user's last-item shelf-genre; Goodreads `shelves` proxy) · item-similarity (ALS Y-space neighbours of last item) · ALS warm (served personalization) · **blended** = ALS → item-sim → content → popularity.

## 5. Fallback decision tree (selected policy = blended)
`has ALS factor` → **ALS** (personalized) · else `has last item` → **item-similarity** → else **genre/content** · else (no history) → **popularity**. Guarantees **0% empty responses**; popularity is the terminal floor. Selected because it gives warm users full personalization while degrading gracefully and never returning empty — at the honest cost that the popularity tail is non-personalized.

## 6. Business interpretation
Warm users get a wide, personalized, catalog-spanning slate. Cold/unknown users get a safe but **generic best-seller wall** — fine for first-session safety, but it does **not** personalize and **starves the long tail / new titles**. The growth levers this points to (not built here): content-to-embedding retrieval for item cold-start, onboarding signals to exit `unknown` faster, and an exposure/diversity constraint if catalog health is a goal.

## 7. API behaviour (G23 service, integration-tested)
warm/sparse known user → `als` (`fallback_used:false`); unknown / missing-factor → `fallback_popularity` (`unknown_user`); `mode=scale` → `scale_hnsw`; invalid k → graceful popularity; missing item metadata → `creator_id:null`, still served. **All 7 cohort cases returned non-empty; every integration assertion passed.**

## 8. Safe / unsafe claims
**Safe (proven):** *"Evaluated cold-start and sparse-user fallback policies across defined cohorts, measuring quality degradation, coverage, fallback hit rate, and head/tail exposure tradeoffs — showing that falling back to popularity collapses catalog coverage ~59× and personalization to near zero for cold users, while ~57% of held-out items are item-cold-start with no collaborative signal."*
**Unsafe (forbidden):** "solved cold-start" · "personalizes for unknown users" (it's popularity) · mixing warm-ALS and cold-fallback into one headline · "solved item cold-start" · any fairness claim · online/production user behaviour.

## 9. Limitations
Offline held-out recall (not online); recall non-monotonic → reported with coverage/personalization as the real cost; absolute recall on served c2 ALS f64 differs from the V1 d3aplus tuning number (different build/protocol — G24 does not re-litigate the V1 headline); content fallback uses shelves as a genre proxy (partial); item cold-start measured, not solved; diversity/novelty beyond tier shares not computed.

## 10. Status
✅ G24 done (V2). `cold_start_solved:false`, `fallback_quality_measured:true`, `degradation_quantified:true`, `item_coldstart_claimed_solved:false`, `fairness_claimed:false`. V1 (`gold_candidate` 8.9) unchanged.

**STOP — G24 complete; awaiting review.**
