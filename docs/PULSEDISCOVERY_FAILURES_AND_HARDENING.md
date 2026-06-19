# PulseDiscover — Failures I'm Proud Of + Serving Stress Test

*Two exercises before deployment. **Exercise 1** = failure archaeology across G20–G31 (the catches and course-corrections that show judgment). **Exercise 2** = an empirical edge-case stress test of `recommender_service.py` + the G31 BM25 module, with two fixes applied and the rest documented. Evidence: `outputs/evidence/g_serving_stress_test.json`.*

---

# Exercise 1 — Failures I'm Proud Of

*The interview value of a project isn't the wins — it's the failures you caught yourself. Each below: what happened → what it revealed → what I did → what it demonstrates.*

## 1. SASRec v1 false convergence (the NaN val-loss early-stop)
**What happened.** The SASRec training loop early-stopped at ~6 epochs on a patience trigger — looked like clean convergence. It wasn't: the validation-loss curve was **all-NaN** (a 2-block masked-attention NaN on short/padded sequences), and a NaN comparison silently satisfied the patience criterion while *training loss and eval-R@20 were still rising*.
**What it revealed.** A monitoring signal can fire "done" for the wrong reason. "Early stopping triggered" is not evidence of convergence — you have to look at *what* triggered it.
**What I did.** Rejected the run as an **invalid outcome (C)**, not a real result. Rebuilt the convergence monitor on an **eval-R@20 plateau** instead of val-loss, and only declared convergence (v3, ~95 epochs) once the *metric we actually care about* flattened.
**What it demonstrates.** I don't accept a convenient stop. I distinguish "the script halted" from "the model converged," and I'll throw away a result that looks good but is mechanically wrong.

## 2. The v2 eval-leakage bug (same set for stopping + final eval)
**What happened.** SASRec v2 used `eval_sample.csv` for **both** the early-stopping criterion **and** the final reported metric. The stopping decision was peeking at the test set.
**What it revealed.** Even with a "held-out" set, *reusing* it for model selection leaks — the reported number is optimistically biased because the model was tuned to stop exactly where that set looked best.
**What I did.** Split a **separate validation-monitor set** (held-out 5% of training sequences) for the stopping criterion, kept the final eval set untouched, and fixed the epoch cap to an exact value (`TOTAL_EPOCH_CAP=160`) so "convergence" wasn't a moving target.
**What it demonstrates.** I know the difference between a held-out set and a *clean* held-out set, and I treat model selection as a leakage surface — the thing most people miss.

## 3. Stale-script contamination on Colab
**What happened.** Twice, a Colab run reported old numbers because it executed a **cached v1 `run_g20_canonical.py`** instead of the v3 script — the notebook silently ran stale code.
**What it revealed.** "I updated the script" ≠ "the run used the updated script." Notebook/cache state is an invisible reproducibility hazard.
**What I did.** Hardened the notebook to **write and run a distinctly-named file** (`run_g20_v3.py`) with **assert guards** on v3-only markers, so a stale run fails loudly instead of reporting wrong numbers. When stale checkpoints/JSON kept arriving, I tagged the recovered headline `plot_reconstructed` rather than trust an unverifiable artifact.
**What it demonstrates.** Reproducibility discipline under messy real conditions — I make the pipeline fail loudly rather than silently lie, and I provenance-tag anything I can't independently verify.

## 4. The IVF nprobe finding — "the metric can lie"
**What happened.** FAISS IVF at low `nprobe` looked great on **gold Recall@20** — it survived. But the **candidate-overlap@K vs exact** had collapsed to **0.28–0.78**: IVF was returning a largely *different* candidate set that happened to still contain the single gold item.
**What it revealed.** A downstream metric (gold recall) can stay flat while the thing it depends on (candidate fidelity) silently degrades — because a single-gold metric is insensitive to *most* of the slate changing. Optimizing the visible metric would have shipped a broken retriever.
**What I did.** **Rejected IVF** despite its speed; kept FlatIP exact (default) and HNSW ef64 (overlap 0.992) as the only adopted indexes. Made candidate-overlap@K a first-class acceptance check, not just recall.
**What it demonstrates.** I instrument the *mechanism*, not just the headline metric, and I'll reject a faster option that passes the obvious test but fails the real one.

## 5. G31 BM25 beat dense on short text — what that says about embeddings
**What happened.** On seed-item (query-by-document) retrieval, **BM25 lexical beat dense MiniLM** (Recall@20 0.0133 vs 0.0067), and RRF hybrid (0.0117) didn't beat BM25 alone.
**What it revealed.** Dense embeddings are not free wins. On **short text** (title + genre shelves), exact lexical token overlap — shared series names, shared genre words — carries more signal than a 384-d semantic vector, which compresses and *blurs* exactly the rare tokens that disambiguate same-series/same-subgenre books. Embeddings shine on *long, paraphrase-heavy* text; they underperform lexical on short, keyword-dense, high-cardinality text.
**What I did.** Reported it straight — did **not** force a "hybrid wins" narrative. Framed hybrid as buying **robustness** (never collapses to either lane's weakness, reaches cold-start) rather than a recall lift; documented that the right text for dense would be longer descriptions (and the G28 ablation already showed description didn't help the *cold-start* slice).
**What it demonstrates.** I pick retrieval methods from measured behavior, not hype — and I can articulate *why* a "more modern" method lost, which is the senior signal.

## 6. The ID-dtype bug — a silent all-zero lookup
**What happened.** Early in G24 the cohort/eval numbers came back implausibly low — near-zero recall across the board. The cause: `book_id` is stored as a **string** in the ALS index (`iidx`, `i_uniq`), but I'd loaded the eval/train book_ids as **int64** from CSV. Every `b in iidx` / `gmap.get(b)` membership test compared `int` against `str` keys, so **every lookup missed** — the genre map built 0 entries, candidate membership was always false, recall silently floored.
**What it revealed.** A type mismatch on a join key doesn't raise — it returns "not found" forever. The pipeline *ran clean* and produced *plausible-shaped* (just wrong) numbers. The most dangerous bugs don't crash; they quietly degrade a metric to zero while everything looks healthy.
**What I did.** Caught it with an **overlap sanity check** (the genre map had 0 keys overlapping the catalog, which is impossible if types matched). Normalized `book_id` to `str` everywhere — load, lookup, and comparison — and made dtype an explicit, repo-wide convention; the serving loader now keeps `i_uniq` as strings end to end.
**What it demonstrates.** I sanity-check intermediate artifacts (does this set overlap at all?) instead of trusting a final number, and I know that silent type coercion is where real pipelines rot.

## 7. G24 recall non-monotonicity — refusing the expected story
**What happened.** The cold-start cohort analysis was *supposed* to show recall degrading from warm → cold. It showed the **opposite**: unknown/sparse cohorts scored **higher** Recall@20 than warm users (e.g. blended ~0.049 unknown vs ~0.014 warm).
**What it revealed.** A cold user's single held-out next item is more **popularity-predictable** (cold users consume popular items), so any model retrieves it easily — while a warm user's next read is more niche and harder. Recall@K, on a single-gold protocol, therefore *understates* the cold-start problem. The intuitive "cold = worse recall" framing is wrong on this data.
**What I did.** Refused to force the expected narrative or quietly bury the inversion. Reframed the **real** cold-start cost around what actually degrades — **catalog coverage (~59× collapse) and personalization (0.68 → 0.001)** — and documented the recall non-monotonicity explicitly as a finding, not an embarrassment.
**What it demonstrates.** I fit the story to the evidence, not the evidence to the story — and I can explain *why* a headline metric points the "wrong" way, which is exactly the judgment a metric-literate interviewer probes for.

## 8. G29 Doubly-Robust went negative — a mis-calibrated reward model
**What happened.** When I first ran the OPE estimators, IPS and SNIPS were sane but **Doubly-Robust returned a negative value estimate** (≈ −0.4) — nonsensical for a reward in [0,1].
**What it revealed.** The DR reward model `r̂(x,a)` was a `class_weight="balanced"` logistic, which calibrates predictions toward ~0.5 to handle imbalance. But the true reward base rate is ~0.01. So `r̂` massively **over-predicted**, the DR baseline + importance-weighted correction term went wildly off, and DR collapsed. DR is only "doubly robust" if at least one of (propensities, reward model) is *well-specified* — a mis-calibrated reward model actively breaks it.
**What I did.** Caught it from the impossible sign, diagnosed it to the reward-model calibration, and switched to a **plain logistic calibrated to the true base rate** (no class balancing). DR then recovered the known value **best of the three** (0.0089 vs true 0.0106) — lowest-bias, exactly as theory predicts.
**What it demonstrates.** I read an estimate's *plausibility* (a negative reward is impossible) before trusting it, and I understand the assumptions behind DR well enough to know that "balanced" calibration — usually a good default — is the wrong choice when you need calibrated probabilities for an estimator baseline.

---

# Exercise 2 — Serving Layer Stress Test (empirical)

*Probed `recommender_service.py` + `bm25_lexical_index.py` against the edge cases below. Each was **run**, not reasoned about. Two fixes applied; the rest handled or documented.*

| Edge case | Result | Status |
|---|---|---|
| BM25 empty / whitespace / punctuation-only query | tokenizes to ∅ → returns `[]` cleanly (guarded) | ✅ handled |
| BM25 seed item not in text lookup | returns `[]` cleanly | ✅ handled |
| `k = 0`, `k = -5` | → `invalid_k`, popularity fallback (20) | ✅ handled |
| `k = 201 … 99999` (> cap) | `k` capped at 200 → `invalid_k` fallback; **`k > n_items` can never reach FAISS** | ✅ handled by design |
| `k = 200` (cap) | served exactly (200) | ✅ handled |
| Unknown / missing-factor user | popularity fallback, non-empty | ✅ handled |
| FAISS query/index dim mismatch at request time | caught → `faiss_error` → popularity fallback | ✅ handled |
| **Corrupt / unreadable `c2_als.pkl`** | **was returning 0 items** (pop list never loaded) | 🔧 **FIXED** |
| **Missing keys in pickle (`'Y'` absent)** | clean `load_error`, but same empty-fallback gap | 🔧 **FIXED** |
| **X/Y factor-dim mismatch** | **passed startup (ready=True)**, only surfaced per-request | 🔧 **FIXED** |
| Memory footprint for Cloud Run | estimated below | ✅ documented |

## Fix 1 — degraded service now serves a non-empty fallback
**Break:** popularity (`pop_top`) was loaded *inside* the model-load `try`, so any `c2_als.pkl` failure left `pop_top` unset → the `index_load_failure` path returned **0 items**, silently breaking the "0% empty-response" guarantee exactly when you most need a fallback.
**Fix:** load popularity **independently of (and before) the model**, with `pop_top/_head/_pop_pct/b2c` initialized as safe defaults at the top of `__init__`. The not-ready path now serves `self._pop_fallback(k)`. **Verified:** corrupt pickle → `ready=False`, but `recommend` returns 5 popularity items (`source: popularity`), not 0.

## Fix 2 — startup dim validation was tautological
**Break:** the startup `assert self.exact.d == self.dim` can **never fail** — the FAISS index is built *from* `Y`, so its dim always equals `Y`'s. The real risk (user-factor `X` dim ≠ item-factor `Y` dim) **passed startup as healthy** and only blew up per-request as a caught `faiss_error` — false confidence in the health check.
**Fix:** added real startup validation — `assert X.shape[1] == Y.shape[1]` and `assert len(i_uniq) == n_items` — *before* building the index. **Verified:** an `X(8)/Y(4)` pickle now fails startup with a clear `load_error: "X/Y factor-dim mismatch: X=8 Y=4"`, `ready=False`, and the service degrades to popularity rather than serving wrong items.

## Memory footprint (Cloud Run sizing)
Measured on the real `c2_als.pkl`:

| Component | Size |
|---|---|
| ALS user factors `X` (91,732 × 64, f32) | 23.5 MB |
| ALS item factors `Y` (40,541 × 64, f32) | 10.4 MB |
| FAISS FlatIP index (from `Y`) | ~10.4 MB |
| FAISS HNSW index (optional scale mode) | ~16 MB |
| popularity + creator (`b2c`) dicts (~40k items) | ~30–80 MB |
| Python + numpy + faiss-cpu base | ~150–250 MB |
| **ALS service subtotal** | **~250–350 MB** |
| Semantic index (G27): 41,866 × 384 f32 emb + FAISS | +~128 MB (64 + 64) |
| **With semantic lane** | **~400–500 MB** |

### Minimum stable Cloud Run config for the ALS-only path (deployment target)

Measured startup (in-sandbox, `build_hnsw=False`): import numpy/faiss/pandas **0.8 s** + service load (`c2_als.pkl` unpickle + `train.csv` popularity + FlatIP index build) **2.3 s** = **~3.1 s to ready**; first request **1.8 ms**; FAISS default 4 threads. Resident memory ~**250–350 MB** (ALS-only, HNSW off).

| Setting | Value | Why |
|---|---|---|
| **Memory** | **512 MiB** | ALS-only RSS ~250–350 MB leaves ~150–260 MB headroom (incl. the transient `Y` copy during index build). This is the floor that runs **stably**; drop to 256 MiB and the FAISS build + Python base will OOM. Use 1 GiB only if the semantic lane is co-loaded. |
| **CPU** | **1 vCPU** | Needed for the numpy/FAISS index build at startup (Cloud Run gives full CPU during startup regardless) and for the inner-product search. FlatIP over 40k×64 is single-digit-ms on 1 vCPU. |
| **`PD_BUILD_HNSW`** | **`0`** | Skip the HNSW scale index — saves ~16 MB + build time; the minimal target serves exact FlatIP only. (New env flag in `api.py`; default is `1`.) |
| **min-instances** | **1** | Eliminates cold-start rebuild on the first real request **and** keeps the in-memory metrics/latency window warm. With min-instances=0 every cold start pays the ~3 s build and resets monitoring. **1 is required for "stable."** |
| **max-instances** | per traffic (start **3–5**) | The service is **single-process / GIL-bound**, so scale **horizontally** (more instances), not by raising container concurrency. |
| **container concurrency** | **8** | Conservative — one GIL-bound instance shouldn't fan out further; let autoscaling add instances. |
| **CPU allocation** | **CPU only during request processing** is fine **with min-instances=1** | The index lives in memory on the warm instance; you don't need always-allocated CPU. (If you set min-instances=0, switch to CPU-always-allocated or accept the cold-start build.) |
| **startup probe / timeout** | startup probe `timeoutSeconds` **30**, `failureThreshold` ~3 (≈ up to ~90 s grace); container **request timeout** default 300 s is ample | Real Cloud Run cold start = image pull + Python boot + the ~3 s build ≈ **5–15 s** typically. 30 s startup probe is comfortable; you do **not** need the default 240 s startup timeout, but leaving it is harmless. |
| **health check** | `GET /health` (returns `degraded` if `ready=False`) | Lets Cloud Run / a load balancer route away from a degraded instance — and after Fix 1, a degraded instance still serves the popularity fallback rather than empty. |
| **env** | `PD_DATADIR`, `PD_DEFAULT_MODE=exact`, `PD_BUILD_HNSW=0`, `PD_ENABLE_RERANK=0` | The minimal, exact, no-rerank ALS path. |

**Bottom line — minimum stable target:** **512 MiB / 1 vCPU / min-instances=1 / concurrency=8 / `PD_BUILD_HNSW=0` / 30 s startup probe.** That runs the ALS-only exact-FlatIP serving path stably with headroom. **Optional hardening:** persist the FAISS index to object storage and load it instead of rebuilding (cuts the ~2.3 s build), and mount `c2_als.pkl` + `train.csv` read-only — but neither is required for the minimum.

## Remaining documented limitations (not fixed this pass)
- **BM25 latency** ~63 ms/query is a pure-Python `rank_bm25` artifact — for production use `bm25s`, Lucene, or Elasticsearch (sub-ms). Documented in G31; not a serving-path component yet.
- **Single-process / GIL-bound** concurrency (noted in G23) — multi-worker (gunicorn) for real throughput.
- **In-memory metrics** are per-process and reset on restart — ship to Prometheus for real monitoring.
- **No persisted FAISS index** — rebuilt at startup; fine at this scale, persist for faster cold starts.
- **Degraded-service fallback depends on `train.csv`** being present — if both the model *and* `train.csv` are unreadable, the fallback list is empty (logged as `popularity_load_failure`); acceptable since that's a total-data-loss scenario.

## Net
The serving layer was already graceful on the request-path failure modes (bad k, unknown user, FAISS error). The stress test found two **load-path** gaps that gave false confidence — both now fixed and verified — and produced a concrete Cloud Run memory/sizing estimate. **Serving layer is deployment-sane within the documented limitations.**
