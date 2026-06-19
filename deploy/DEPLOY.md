# PulseDiscover — Cloud Run Deployment

> **Who runs this:** *you*, from a machine authenticated to your Google Cloud project. This session **cannot** deploy — it has no access to your GCP credentials, project, or billing, and won't fabricate a live URL. The package below is verified to build and serve locally; you run one script to put it on Cloud Run.

## What's verified (locally, in this repo)
- The serving app boots over **real HTTP** (`uvicorn serving.api:app`) and answers `/health → {"status":"ok","ready":true}`, `/metadata`, and `/recommend` (warm user → 5 `source: als` items; unknown user → non-empty popularity fallback).
- The **lean image** needs only `c2_als.pkl` (38 MB) + `c2_serving_assets.pkl` (4.9 MB) — no `train.csv` (114 MB), no `content_meta` — and loads in **~0.8 s** (+ FAISS build).
- Edge cases hardened (see `../docs/PULSEDISCOVERY_FAILURES_AND_HARDENING.md`): bad `k`, unknown user, FAISS errors, corrupt/missing pickle, X/Y dim mismatch all degrade gracefully; a degraded instance still serves the popularity fallback.

## Prerequisites (one time)
- `gcloud` CLI installed and `gcloud auth login` done.
- A GCP **project** with **billing enabled**.
- Roles on your account: `roles/run.admin`, `roles/artifactregistry.admin`, `roles/cloudbuild.builds.editor`, `roles/iam.serviceAccountUser`.

## Deploy (one command)
```bash
# from the repo root
./deploy/cloudrun_deploy.sh YOUR_GCP_PROJECT_ID us-central1
```
The script: precomputes `c2_serving_assets.pkl` → enables APIs → ensures an Artifact Registry repo → builds the image from `deploy/Dockerfile.cloudrun` via Cloud Build → deploys to Cloud Run with the **exact spec** below → prints the live URL → curls `/health`, `/metadata`, `/recommend`.

## The spec it applies (from the hardening doc)
| Setting | Value |
|---|---|
| memory | **512 MiB** |
| cpu | **1 vCPU** |
| min-instances | **1** |
| max-instances | **5** |
| concurrency | **8** |
| request timeout | 300 s |
| startup probe | `GET /health`, ~30 s grace (initialDelay 5s, timeout 5s, period 5s, failureThreshold 6) |
| env | `PD_BUILD_HNSW=0`, `PD_DEFAULT_MODE=exact`, `PD_ENABLE_RERANK=0` |
| port | 8080 (Cloud Run `$PORT`) |
| auth | `--allow-unauthenticated` (drop this for a private service) |

## Expected verification output
```
-- /health --
{"status":"ok","ready":true,"load_error":null}
-- /recommend (warm) --
{"items":[{"item_id":"...","creator_id":"...","source":"als"}, ... 5 items],
 "retrieval_mode":"exact","fallback_used":false,"response_count":5, ...}
```
A known warm `user_id` baked into `c2_als.pkl`: `8842281e1d1347389f2ab93d60773d4d`.

## Honesty / scope note
This puts the **offline-trained** ALS+FAISS serving path on managed infrastructure. It does **not** create online-lift, real-user, or A/B evidence — those remain out of scope and uncollected until real traffic flows. "Deployed the serving API to Cloud Run" is a true statement once you run this; "improved engagement / online lift" is **not** and must not be claimed.

## Cost / teardown
min-instances=1 keeps one 512 MiB / 1 vCPU instance always warm (small but non-zero cost). To tear down:
```bash
gcloud run services delete pulsediscover-serving --region us-central1 --project YOUR_GCP_PROJECT_ID
```
