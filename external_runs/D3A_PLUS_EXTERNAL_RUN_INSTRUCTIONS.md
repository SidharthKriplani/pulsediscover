# D3A+ External-Run Pack

*For higher-quality runs that exceed the in-sandbox limits (3.8 GB RAM, 4 CPU, no GPU, 45 s/call). Run these on a local machine / Colab / Kaggle / any box with ≥16 GB RAM (and optionally a GPU). **No paid infrastructure required** — free Colab/Kaggle suffice. Deterministic seeds throughout.*

## What in-sandbox could NOT do (and why)
| Task | In-sandbox status | Limit hit |
|---|---|---|
| Co-occurrence vocab ≥ 20k | **FAILED** | `MᵀM` nnz ~150M+ → OOM/timeout on 3.8 GB |
| ALS factors ≥ 128 / iters ≥ 12 | not run | per-call 45 s CPU cap |
| Full hybrid with f64 ALS | not run | sequencing/time |
| Eval on all 42,683 held-out users (vs 5k sample) | not run | ALS scoring memory (42k×42k) |
| leave-last-5 / future-window recall | designed, not run | time |

## Inputs (already produced in-sandbox; copy these)
- `data/interim/domain_core_interactions.csv` (2,557,588 rows: `user_id,book_id,rating,rating0_flag,event_ts`)
- `data/interim/domain_core_books_meta.csv` (`book_id,creator_id,series_id,title,language_code`)
- `data/interim/domain_splits/{train,val,test,heldout_eval,eval_sample}.csv`
- `data/interim/d2_stats.json` (cutoffs t1=1455733286, t2=1478051029)

## Fixed config (use exactly)
```
SEED = 20260616
global-time cutoffs: t1 = 1455733286 (2016-02-17), t2 = 1478051029 (2016-11-02)
leave-last-out: held-out = each user's last interaction with event_ts >= t2
metric Ks: Recall@5,10,20,50,100,200 ; NDCG@10,20,50
eval: full 42,683 held-out users (drop the 5,000-sample cap)
```

## Runs to execute (priority order)
1. **ALS factor/iter grid:** factors ∈ {32,64,128,256}, iters ∈ {8,12,15}, alpha=40, reg=0.1, constant confidence (all user-book pairs unique). Use `implicit` (Cython, GPU-optional) for speed. Expected: monotonic Recall gains with factors (in-sandbox: f16→f32→f64 gave R@20 0.057→0.073→0.085).
2. **Co-occurrence vocab:** 20k, 40k, full (≈42k) with sparse top-k neighbor extraction (do **not** materialize dense `MᵀM`; iterate CSR rows or use `implicit`/`scipy` block-wise). Report R@10/20/50.
3. **Hybrid with f64+ ALS:** RRF union of {co-occ(full), best-ALS, same-author, same-series} + popularity fallback; re-measure marginal lift vs the new best single.
4. **Full-population eval:** all 42,683 held-out users, Recall/NDCG + bootstrap CIs.
5. **Extended protocols:** leave-last-5 per user; and a future-window recall (all test-window interactions per user, not just the last).

## Reproducible scripts (already in `src/`, parameterized)
- `python src/d3aplus_als_tune.py <F> <ITERS>` → appends to `d3aplus_als_tune.json`
- `python src/d3aplus_cooc_sweep.py <VOCAB>` → appends to `d3aplus_cooc_sweep.json`
- `python src/d3aplus_fit.py` / `d3aplus_als.py` / `d3aplus_maps.py` / `d3aplus_eval.py` → full pipeline (raise the 5k sample cap in `d3aplus_eval.py` to evaluate all held-out users)

## Logging
- stdout prints a one-line summary per run; redirect to a run log: `python ... | tee external_runs/logs/<run>.log`.
- Each script writes/append a metrics JSON (schema below). Keep the seed fixed for comparability.

## Metrics JSON schema (per method)
```json
{
  "method": "als|cooccurrence|hybrid|...",
  "config": {"factors": 64, "iters": 12, "vocab": 40000, "...": "..."},
  "n_eval_users": 42683,
  "metrics": {
     "R@5": {"mean": 0.0, "ci95": [0.0, 0.0]},
     "R@10": {"mean": 0.0, "ci95": [0.0, 0.0]},
     "...": "R@20,50,100,200",
     "N@10": {"mean": 0.0}, "N@20": {"mean": 0.0}, "N@50": {"mean": 0.0}
  },
  "segments": {"warm_items": 0, "new_items": 0, "warm_creators": 0, "new_creators": 0,
               "warm_series": 0, "new_series": 0},
  "runtime_sec": 0.0, "peak_mem_mb": 0.0, "seed": 20260616
}
```

## Artifact manifest (expected outputs)
- `outputs/evidence/domain_baselines_plus_report.json` (extended, full-population)
- `outputs/evidence/d3aplus_als_tune.json`, `d3aplus_cooc_sweep.json`
- `outputs/plots/domain_baselines_plus_recall.png`, `domain_als_factor_tuning.png`
- `external_runs/logs/*.log`

## Environment
- Python 3.10+, `numpy pandas scipy matplotlib`; optional `implicit` (fast ALS), `torch` (GPU). All free.
- ≥16 GB RAM recommended for full co-occurrence + full-population eval; a free Colab/Kaggle GPU only accelerates ALS/`implicit` and later SASRec — **not required** for these baseline/eval runs.
