# 20 — D3A Gate Report (Domain splits + retrieval baselines)

*Scope: materialize global-time splits + constrained leave-last-out eval; run **retrieval baselines only** (popularity, co-occurrence, ALS) on the Goodreads fantasy/paranormal core; Recall@5/10/20 + NDCG@10/20 with bootstrap CIs on held-out test users; cold-start breakdown. Domain-scoped. No SASRec / OPE / position-bias / feedback / PRD / PDF. Stops at this gate.*

## 1. Splits materialized (`data/interim/domain_splits/`)
Global-time cutoffs t1=2016-02-17, t2=2016-11-02:
- `train.csv` 2,046,070 · `val.csv` 255,759 · `test.csv` 255,759 interactions.
- `heldout_eval.csv` — **42,683** constrained leave-last-out users (last interaction in test window); `eval_sample.csv` — **5,000-user** seeded random sample used for metrics.

## 2. Evaluation setup
Leave-last-out next-item (consistent with the smoke): baselines fit on the core **minus** the 42,683 held-out last items; each held-out user's last book is the gold. Metrics on the **5,000-user sample** with per-user bootstrap 95% CIs.

## 3. Domain retrieval-baseline results (5,000-user sample)
| Baseline | Recall@5 | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |
|---|---|---|---|---|---|
| popularity | 0.0066 [0.0044, 0.0088] | 0.0202 [0.0164, 0.0240] | 0.0322 [0.0276, 0.0372] | 0.0085 | 0.0116 |
| **co-occurrence** | **0.0252 [0.0208, 0.0296]** | **0.0360 [0.0308, 0.0412]** | 0.0532 [0.0472, 0.0592] | **0.0207** | 0.0250 |
| **ALS** | 0.0178 [0.0144, 0.0216] | 0.0300 [0.0254, 0.0350] | **0.0570 [0.0506, 0.0638]** | 0.0152 | 0.0219 |

## 4. Honest reading
- **Both collaborative baselines beat popularity decisively** — co-occurrence and ALS CIs sit well above popularity at every K (e.g., R@10: co-occ [0.031,0.041], ALS [0.025,0.035] vs popularity [0.016,0.024]).
- **Co-occurrence vs ALS is a wash, K-dependent:** co-occurrence leads at @5/@10 and on NDCG@10; ALS leads at @20. Their CIs overlap at @10 and @20 → **no single clear winner**, reported as-is (not forced).
- **Contrast with the MovieLens smoke:** there co-occurrence was the *weakest* (R@10 0.003); here it is the *strongest at @10* (0.036). Dense book co-reading in a fiction genre makes item-item co-occurrence genuinely informative — a real domain-vs-smoke difference, not a bug.
- Absolute numbers are modest, as expected for strict leave-last-out next-item retrieval (no ranker/sequence model yet — that's D3B/SASRec).

## 5. Cold-start breakdown
- **Eval-gold level (5,000 sample):** 4,999 warm items / **1 new item**. Leave-last-out golds on a k-cored set are ~all warm (the user's last book was already seen via other users), so this eval is *not* where cold-start bites.
- **Substantive cold-start (D2 global-time test window):** **3,378 new items, 5,914 new users, 852 new creators, 2,151 new series; 23.5% of test interactions are on new items.** Retrieval baselines **structurally cannot** recommend a never-seen item → their recall on that 23.5% is ~0. This is the gap a content/cold-start channel (not built) would address, and is the honest limitation to state.

## 6. Config & CPU-scale limits (reported, not hidden)
- **Co-occurrence capped to top-5k items** with per-row PMI extraction: the all-pairs `MᵀM` at 20k items **OOM'd on the 3.8 GB sandbox** — a real scale limit, documented. (top-5k still covers the bulk of interactions.)
- **ALS:** constant confidence (all user-book pairs unique), factors=16, **bounded iters=4** for CPU; fit 10.8 s.
- **5,000-user sample** (seeded) of the 42,683 held-out users; bootstrap CIs reflect the sample.

## 7. No-overclaim check
- All claims **domain-scoped to Goodreads fantasy/paranormal**; tag `[BUILT — real data (domain)]`. ✅
- co-occ OOM cap + ALS iter cap + sample-based eval all disclosed. ✅
- No SASRec/OPE/position-bias/feedback; no online-lift claim. ✅

## 8. Artifacts
- `outputs/evidence/domain_baselines_report.json` · `outputs/plots/domain_baselines_recall.png`.
- Splits + eval sets in `data/interim/domain_splits/`.

## 9. Gate status & next step
- ✅ Domain splits + leave-last-out materialized; retrieval baselines with CIs; cold-start quantified.
- ⏸️ **Next (on approval): D3B — SASRec on the domain core** (top-40k vocab, resumable CPU training or GPU), evaluated against these baselines on the same 5,000-user sample. Then domain SASRec-π OPE, position-bias, creator-level feedback-loop.

**STOP — awaiting D3A review.**
