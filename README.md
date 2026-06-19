# PulseDiscover

An offline, honesty-gated **recommender decision system** on a real 2.56M-interaction Goodreads corpus. Two-stage retrieve→rank with four candidate sources (ALS · semantic content · BM25 · popularity), FAISS serving, a learned ALS+semantic fusion ranker, off-policy evaluation, and catalog-exposure governance — wrapped in a FastAPI service deployed on GCP Cloud Run.

Built to demonstrate production recommender/IR engineering: candidate generation, learned-to-rank fusion, ANN serving, cold-start retrieval, off-policy evaluation (IPS/SNIPS/DR), and ruthless claim discipline — every number is offline and protocol-tagged, and the honest negatives are kept, not buried.

**Live:** `https://pulsediscover-serving-98058433335.us-central1.run.app` · `/health` → `{"status":"ok","ready":true}`

---

## Architecture

```
                                user_id (+ optional seed item)
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  RecommenderService   (served on c2 ALS f64 — the single versioned spine) │
│                                                                           │
│   ① CANDIDATE GENERATION                                                  │
│      ┌──────────┐  ┌─────────────────┐  ┌───────────┐  ┌──────────────┐  │
│      │   ALS    │  │ semantic content│  │   BM25    │  │  popularity  │  │
│      │ (warm MF)│  │  MiniLM + FAISS │  │ (lexical) │  │  (fallback)  │  │
│      └────┬─────┘  └────────┬────────┘  └─────┬─────┘  └──────┬───────┘  │
│           │                 │                 │               │          │
│   ② RETRIEVAL — FAISS top-K (FlatIP exact ✓ · HNSW scale · ✗ IVF rejected)│
│           └──────────────┬──┴─────────────────┘               │          │
│                          ▼                                     │          │
│   ③ FUSION / RANK                                              │          │
│      RRF  ──or──  LightGBM LambdaMART  (over ALS+semantic features)       │
│                          │                                     │          │
│   ④ EXPOSURE RERANK  (config-flagged head-cap, default OFF)    │          │
│                          │                                     │          │
│   ⑤ FALLBACK TREE  ──unknown / sparse / error / empty──────────┘          │
│                          │   (never empty — 0% empty-response verified)   │
│                          ▼                                                 │
│   ⑥ SERVE   FastAPI:  /recommend  /health  /metadata  /metrics            │
│                          │                                                 │
│   ⑦ LOG  per-request JSON (versioned model/index ids, latency, fallback)  │
└──────────────────────────┼────────────────────────────────────────────────┘
                           ▼
   ⑧ EVALUATE  cohort recall · NDCG/MRR · Gini/coverage · IPS/SNIPS/DR
                           ▼
   ⑨ DECIDE    ship / hold / reject  (behind a locked claim ladder)
```

The whole system is a **gated experimentation program** (gates G11–G31): every stage was added, measured, and either adopted or rejected with an artifact. V1 set the offline modelling floor; the V2 lane (G22–G31) added serving, cold-start, exposure governance, learned fusion, executed OPE, and a search/IR front end — then the serving path shipped to Cloud Run.

---

## ⚠️ Metric protocol boundary (read before any number)

The V1 **`d3aplus`** ALS result (**R@20 0.0846**) and the V2 **served-`c2`** results (full-catalog, single-held-out-gold, **R@20 ~0.0375**) are **different model builds and different evaluation protocols — they are NOT comparable.** They're kept in a registry (`docs/G26A_MODEL_PROTOCOL_REGISTRY.md`) specifically so they're never conflated. V2 numbers are used for *relative* cross-policy structure, never as a headline that overturns V1.

---

## Eval results (offline, protocol-tagged)

| Result | Number | Protocol |
|--------|--------|----------|
| ALS warm floor | R@20 **0.085** | V1 d3aplus tuning |
| Canonical SASRec, converged | R@20 0.065 (**lost to ALS**) | V1 — honest negative |
| Content-hybrid cold lane | cold R@20 **0.241** (ALS = 0) | V1 cold-start |
| FAISS FlatIP (exact) | lossless, ~**47%** lower p95 | V2 latency |
| FAISS HNSW ef64 | overlap **0.992**, ~2.4× | V2 latency |
| FAISS IVF | **rejected** — overlap collapsed to 0.28–0.78 | V2 |
| Cold-start fallback cost | coverage **~59×** collapse; personalization 0.68→**0.001** | V2 served-c2 |
| Semantic cold-start reach | **54.6%** of unseen golds (ALS/pop = 0) | V2 served-c2 |
| Learned fusion vs ALS-only | test R@20 **0.036 vs 0.022**; cold 0.032 vs 0 | V2 — held-out test, 81 positives |
| Off-policy evaluation | DR **0.0089** vs true 0.0106; SNIPS stable; IPS over-estimates | V2 — offline, proxy reward |
| Exposure concentration | all policies Gini **> 0.97**; popularity = 1.0 | V2 — concentration, not fairness |
| Search/IR hybrid | BM25 **0.0133** > dense 0.0067; RRF hybrid 0.0117 | V2 — seed-item, NDCG/MRR |
| Serving | warm p95 **~3.3 ms** (local), **0%** empty | V2 load test |

Every cell is offline. Small-sample caveats (e.g. 81 test positives) are stated, not hidden.

---

## Stack

| Layer | Technology |
|-------|-----------|
| Warm retrieval | ALS implicit matrix factorization · 64 factors |
| Semantic (cold) retrieval | sentence-transformers `all-MiniLM-L6-v2` · 384-dim · FAISS FlatIP |
| Lexical retrieval | BM25Okapi (`rank_bm25`) · query-by-document |
| Fusion / rank | Reciprocal Rank Fusion (RRF, k=60) · LightGBM **LambdaMART** |
| Exposure rerank | heuristic head-cap (config-flagged, default off) |
| ANN serving | FAISS — IndexFlatIP (exact) · IndexHNSWFlat (scale) |
| Off-policy eval | IPS · SNIPS · Doubly-Robust |
| Evaluation | cohort Recall@K · NDCG@K · MRR · Gini / catalog coverage |
| API | FastAPI + uvicorn |
| Deployment | Docker + GCP Cloud Run (512 MiB / 1 vCPU, exact FlatIP) |
| Corpus | Goodreads fantasy/paranormal (UCSD book graph) · 2.56M interactions · 94k users · 42k items |

---

## Project structure

```
pulsediscover/
├── src/
│   ├── serving/                FastAPI app + serving spine
│   │   ├── api.py              /recommend /health /metadata /metrics
│   │   ├── recommender_service.py   versioned loader · fallback tree · logging
│   │   └── faiss_retriever.py  FlatIP / HNSW / IVF wrappers
│   ├── retrieval/
│   │   ├── semantic_content_index.py   MiniLM dense content lane (G27)
│   │   └── bm25_lexical_index.py       BM25 query-by-document (G31)
│   ├── ranking/
│   │   └── g28_fusion_ranker.py        RRF + LambdaMART fusion (G28)
│   ├── ope/
│   │   └── logging_policy.py            logging policy + IPS/SNIPS/DR (G29)
│   └── eval/
│       ├── cold_start_cohorts.py        cohort builder (G24)
│       ├── catalog_exposure_governance.py   Gini/coverage (G25)
│       └── exposure_aware_reranking.py  head-cap / tail-boost (G26)
├── scripts/                    every gate run script + embedding/asset builders
├── docs/                       gate reports · interview kit · defense kernel · registry · control tower
├── outputs/                    evidence JSONs + plots
├── deploy/                     Cloud Run deploy script · Dockerfile · cloudbuild
├── defense/                    interview defense PDF
└── README.md
```

---

## Setup

**Requirements:** Python 3.10+

```bash
pip install -r requirements.txt

# Run the serving API locally (exact FlatIP, HNSW off — the deployed config)
cd src && PD_DATADIR=../data/interim PD_BUILD_HNSW=0 uvicorn serving.api:app --port 8080

curl localhost:8080/health                                  # {"status":"ok","ready":true}
curl "localhost:8080/recommend?user_id=<known>&k=5"         # 5 items, source:als
```

> Large derived artifacts (ALS factors, embeddings, raw CSVs) are **not committed** (size / GitHub limits). `scripts/` shows how each is built from the public Goodreads dataset.

---

## Cloud deployment (GCP Cloud Run)

```bash
./deploy/cloudrun_deploy.sh <GCP_PROJECT_ID> us-central1
# precomputes a 4MB serving-assets bundle → builds a lean image (~42MB context)
# → deploys 512MiB / 1vCPU / min-instances=1 / concurrency=8 → curls /health + /recommend
```

The lean image bakes only the ALS factors + a precomputed popularity/creator bundle (no 114 MB train.csv), loads in ~0.8 s, and serves exact FlatIP over HTTPS. Full spec + teardown in `deploy/DEPLOY.md`.

---

## Failures I'm proud of

The interview value of a project is the failures you catch yourself.

- **SASRec false convergence** — training early-stopped at ~6 epochs on an all-NaN val-loss curve (masked-attention NaN); a NaN comparison silently satisfied patience while the model was still learning. Rejected it, rebuilt the monitor on an eval-R@20 plateau.
- **IVF — the metric can lie** — IVF looked fine on gold Recall@20 while candidate-overlap@K vs exact had collapsed to 0.28–0.78: a mostly *different* candidate set that happened to still contain the gold. Rejected IVF; made overlap@K a first-class acceptance check.
- **ID-dtype silent zeroing** — `book_id` was a string in the ALS index but ints in eval; every lookup missed and recall floored to ~0 *while the pipeline ran clean with plausible-shaped numbers*. Caught it with an overlap sanity check (zero catalog-overlapping keys — impossible if types matched).

Full set + the serving edge-case stress test: `docs/PULSEDISCOVERY_FAILURES_AND_HARDENING.md`.

---

## Key engineering decisions

**Why ALS over SASRec for the warm floor?**
I trained a canonical full-softmax SASRec *to convergence* (~95 epochs, eval-R@20 plateau) and it still scored below ALS (0.065 vs 0.085). On short, sparse book histories the dominant signal is collaborative co-read, which matrix factorization captures directly — depth wasn't the lever. The honest negative is the senior signal, so I kept it.

**Why FlatIP exact as the default instead of HNSW?**
FlatIP is lossless and ~47% faster than numpy brute force at this scale; HNSW ef64 is near-lossless (overlap 0.992, ~2.4×) but approximate. Correctness by default, opt into approximation explicitly — HNSW is a flagged scale mode, not the silent default.

**Why was IVF rejected if it was faster?**
At low `nprobe` it kept gold Recall@20 flat while candidate-overlap vs exact collapsed — it served a largely broken candidate set that a single-gold metric couldn't see. Speed that silently destroys candidate fidelity is a recall bug.

**Why call RRF "robustness, not a recall win"?**
RRF recovers cold-start and ~4× coverage at ~95% of ALS relevance, but does **not** beat ALS-only on warm recall. I report it as robustness/coverage, never as a headline lift — because it isn't one.

**Why a learned fusion ranker over the two retrievers?**
ALS and semantic candidate sets are almost disjoint (overlap 6,749 of ~880k). A LambdaMART ranker can pick the best across both complementary sources — which is what lifted held-out test recall (0.036 vs 0.022) and recovered cold-start (0.032 vs 0). Claimed only on the offline test split, with the 81-positive caveat attached.

**Why off-policy evaluation if there are no real users?**
To prove I can estimate a policy's value the right way before an A/B exists. The logging policy records propensities; IPS/SNIPS/DR recover a known value offline (DR lowest-bias, SNIPS lowest-variance, IPS shows the variance problem). It's a methodology demonstration — not a real online estimate, and I say so.

**Why keep V1 and V2 metrics in a registry?**
Because they're different model builds and protocols. Conflating a 0.0846 with a 0.0375 to look better would be the exact dishonesty this project is built to avoid.

---

## What this does *not* claim

- **Offline only** — recall / OPE / simulation; no live A/B.
- **Deployed ≠ improved engagement** — the API is live on Cloud Run, but there's no online-lift, real-user, or business-impact claim.
- **Cold-start handled, not solved** · **exposure measured, not fairness-certified** · the semantic lane is **embedding-based candidate generation, not an LLM recommender**.
- **No deep-model win** (the converged SASRec is a documented honest negative); **learned fusion beats ALS only on the offline test split**, not in production.

---

## Where to read more

`docs/78_CONTROL_TOWER.md` (status) · `docs/PULSEDISCOVERY_INTERVIEW_KIT.md` (pitch + claim ladder) · `docs/PULSEDISCOVERY_UNIFIED_DEFENSE_KERNEL.md` (method-by-method defense) · `defense/PulseDiscover_Interview_Defense.pdf`.

*If a result isn't backed by an artifact in `outputs/evidence/`, it isn't claimed.*
