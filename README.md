# PulseDiscover

**An offline, honesty-gated recommender *decision* system on 2.56M real Goodreads interactions** — it decides *which* retrieval, serving, fallback, fusion, and exposure policy should ship (and proves which must not) by catching offline-metric bias, candidate-coverage gaps, latency–quality trade-offs, cold-start degradation, and catalog concentration before they would reach users.

> **Status:** V1 `gold_candidate` (offline modelling floor, locked) · V2 lane *offline gold-complete* · the ALS+FAISS serving path is **deployed to Google Cloud Run** (live HTTPS).
> **Scope honesty:** everything here is **offline**. There is **no online-lift, real-user, or production-quality claim** anywhere — those are documented as explicit next steps, not results.

---

## The business problem

Recommenders over-serve safe, popular "warm" content, which buries new items (collaborative models can't rank a thing with no history) and lets raw engagement mislead (top-of-page items get examined more, so CTR measures *position* as much as *quality*). The real deliverable a company needs isn't a trained model — it's an **A/B-ready serving policy** with guardrails, a logging design, and an honest read on what the offline evidence does and doesn't prove.

## What I built

**V1 — the offline modelling floor**
- Real domain dataset: Goodreads fantasy/paranormal (2.56M interactions), global-time split, ~76.5% warm / 23.5% cold prevalence.
- **ALS (implicit MF, 64 factors)** as the warm retrieval floor — *earned* by training a canonical full-softmax **SASRec to convergence that still lost** (a documented honest negative).
- A **content-hybrid cold-start lane**, a **two-lane "90/10" serving policy**, and a governance layer: off-policy evaluation, prevalence + position-bias correction, and a creator/catalog-health simulation.

**V2 (gates G22–G31) — serving, governance, fusion, OPE, search**
- **FAISS latency–quality frontier** — FlatIP exact lossless & ~47% faster; HNSW ef64 near-lossless (overlap 0.992, ~2.4×); **IVF rejected** because aggressive `nprobe` collapsed candidate overlap while gold recall survived (the "metric can lie" finding).
- **Production-shaped FastAPI serving** — versioned loader, full fallback tree (0% empty), structured logging, health/metadata/metrics — now **deployed to Cloud Run**.
- **Cold-start fallback quality + exposure governance** — popularity fallback collapses coverage ~59× and personalization 0.68→0.001; all policies are head-concentrated (Gini > 0.97). Measured, not "solved."
- **Dense semantic content lane** (MiniLM + FAISS) — the *only* source that reaches item-cold-start items (ALS/popularity = 0 there).
- **Learned ALS+semantic fusion** (LightGBM LambdaMART) — beat ALS-only on the held-out test split (offline, small positive set) by exploiting the two sources' near-disjoint candidates.
- **Off-policy evaluation executed** — IPS/SNIPS/DR recover a known policy value offline (DR lowest-bias, SNIPS lowest-variance).
- **Search/IR front end** — BM25 lexical + dense + RRF hybrid with NDCG/MRR (query-by-document, not free-text search).

## Key results (all offline, all artifact-backed)

- **ALS f64 warm floor:** R@20 **0.085** (V1 protocol). Content-hybrid cold lane: cold R@20 **0.241** where ALS scores 0 by construction.
- **FAISS:** FlatIP exact lossless ~47% faster; HNSW ef64 overlap 0.992; IVF rejected (overlap collapse 0.28–0.78).
- **Cold-start fallback cost:** coverage collapse **~59×**, personalization **0.68 → 0.001**; semantic lane reaches **54.6%** of unseen golds.
- **Learned fusion:** test R@20 **0.036 vs 0.022** ALS-only, cold-start 0.032 vs 0 (offline, 81 positives — directional, cross-family consistent).
- **OPE:** estimators recover the known value (DR 0.0089 vs true 0.0106; SNIPS stable; IPS over-estimates — the variance problem).

## ⚠️ Metric protocol boundary (load-bearing)

The V1 `d3aplus` ALS result (**R@20 0.0846**) and the V2 served-`c2` results (full-catalog single-held-out-gold, **R@20 ~0.0375**) are **different model builds and different evaluation protocols — NOT comparable.** They're kept in a registry so they're never conflated (`docs/G26A_MODEL_PROTOCOL_REGISTRY.md`).

## Architecture

```
candidate generation (ALS · semantic content · popularity · co-occurrence)
   → FAISS retrieval (FlatIP exact default · HNSW scale · IVF rejected)
   → fusion / rank (RRF · LightGBM LambdaMART over ALS+semantic features)
   → exposure rerank (config-flagged head-cap, default OFF)
   → fallback tree (never empty)
   → FastAPI serving (/recommend /health /metadata /metrics)  → deployed to Cloud Run
   → logging → cohort / exposure / off-policy evaluation → ship/no-ship decision
```

## Repository map

| Path | What's there |
|---|---|
| `src/serving/` | FastAPI app, FAISS retriever, recommender service (loader + fallback + logging) |
| `src/retrieval/` | semantic content index (MiniLM), BM25 lexical index |
| `src/ranking/` | fusion ranker (RRF + LambdaMART) |
| `src/ope/` | off-policy evaluation (logging policy + IPS/SNIPS/DR) |
| `src/eval/` | cold-start cohorts, exposure governance, exposure-aware reranking |
| `scripts/` | all gate run scripts + embedding/asset builders |
| `docs/` | gate reports (G22–G31 + V1 D/O/P/H), interview kit, unified defense kernel, failures & hardening, model registry, control tower, evidence ledger, claim boundary |
| `outputs/evidence/`, `outputs/plots/` | per-gate evidence JSONs and figures |
| `deploy/` | Cloud Run deploy script, Dockerfile, cloudbuild config |
| `defense/` | interview defense PDF |

**Start here:** `docs/78_CONTROL_TOWER.md` (status) · `docs/PULSEDISCOVERY_INTERVIEW_KIT.md` (pitch + claims) · `docs/PULSEDISCOVERY_UNIFIED_DEFENSE_KERNEL.md` (method defenses) · `docs/PULSEDISCOVERY_FAILURES_AND_HARDENING.md` (failures caught + serving stress test).

## Claim boundaries (what this does *not* claim)

- **Offline only** — recall / known-propensity OPE / simulation; no live A/B.
- **No business-lift claim** — the serving API is **deployed and callable on Cloud Run, but "deployed" ≠ "improved engagement."** No online experiment has been run.
- **Cold-start handled at the system level, not solved**; **catalog exposure measured, not fairness-certified**; the semantic lane is **embedding-based candidate generation, not an LLM recommender**.
- **No deep-model win** — the converged SASRec is a documented honest negative.
- **Learned fusion beats ALS only on the offline test split** (small positive set) — not a production claim.

## Data & serving

Built on the public **Goodreads** book-graph dataset (fantasy/paranormal shelves). Raw dumps and large derived artifacts (ALS factors, embeddings, CSVs) are **not committed** (size / GitHub 100 MB limits) — `scripts/` shows how each is built. Deploy: see `deploy/DEPLOY.md`. Local run:

```bash
cd src && PD_DATADIR=../data/interim PD_BUILD_HNSW=0 uvicorn serving.api:app --port 8080
curl localhost:8080/health        # {"status":"ok","ready":true}
```

*A personal portfolio / research project. If a result isn't backed by an artifact in `outputs/evidence/`, it isn't claimed.*
