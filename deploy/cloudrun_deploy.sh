#!/usr/bin/env bash
# PulseDiscover — one-command Cloud Run deploy (minimal ALS-only exact-FlatIP path).
# Run from the REPO ROOT on a machine with gcloud authenticated to your project.
#
#   ./deploy/cloudrun_deploy.sh YOUR_GCP_PROJECT_ID [REGION]
#
# Spec (from docs/PULSEDISCOVERY_FAILURES_AND_HARDENING.md):
#   512 MiB / 1 vCPU / min-instances=1 / max-instances=5 / concurrency=8 / startup probe 30s
#   env: PD_BUILD_HNSW=0, PD_DEFAULT_MODE=exact, PD_ENABLE_RERANK=0
set -euo pipefail

PROJECT="${1:?usage: cloudrun_deploy.sh PROJECT_ID [REGION]}"
REGION="${2:-us-central1}"
SERVICE="pulsediscover-serving"
REPO="pulsediscover"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${SERVICE}:$(date +%Y%m%d-%H%M%S)"

echo ">> 0. Precompute lean serving assets (so the image stays small)"
PYTHONPATH=src python3 scripts/build_serving_assets.py
test -f data/interim/c2_serving_assets.pkl

echo ">> 1. Enable APIs + ensure Artifact Registry repo exists"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com --project "$PROJECT"
gcloud artifacts repositories describe "$REPO" --location "$REGION" --project "$PROJECT" >/dev/null 2>&1 \
  || gcloud artifacts repositories create "$REPO" --repository-format=docker --location "$REGION" --project "$PROJECT"

echo ">> 2. Build + push image via Cloud Build (uses deploy/Dockerfile.cloudrun)"
gcloud builds submit --project "$PROJECT" --config=deploy/cloudbuild.yaml --substitutions=_IMAGE="$IMAGE" .

echo ">> 3. Deploy to Cloud Run with the exact spec"
gcloud run deploy "$SERVICE" \
  --project "$PROJECT" --region "$REGION" \
  --image "$IMAGE" \
  --platform managed --allow-unauthenticated \
  --memory 512Mi --cpu 1 \
  --min-instances 1 --max-instances 5 \
  --concurrency 8 \
  --timeout 300 \
  --port 8080 \
  --set-env-vars PD_BUILD_HNSW=0,PD_DEFAULT_MODE=exact,PD_ENABLE_RERANK=0 \
  --startup-probe httpGet.path=/health,initialDelaySeconds=5,timeoutSeconds=5,periodSeconds=5,failureThreshold=6

URL=$(gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" --format='value(status.url)')
echo ">> Live URL: $URL"

echo ">> 4. Verify"
echo "-- /health --";  curl -fsS "$URL/health" ; echo
echo "-- /metadata --"; curl -fsS "$URL/metadata" ; echo
WARM="8842281e1d1347389f2ab93d60773d4d"   # a known warm user_id from c2_als.pkl
echo "-- /recommend (warm) --"; curl -fsS "$URL/recommend?user_id=${WARM}&k=5" ; echo
echo ">> Done. Expect /health -> {\"status\":\"ok\",\"ready\":true} and /recommend -> 5 items source=als."
