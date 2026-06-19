"""G23 (V2) — FastAPI production-SHAPED recommender serving API (NOT production deployment).
Endpoints: GET /recommend, /health, /metadata, /metrics (+ JSON), /debug/retrieval.
Run: uvicorn serving.api:app --host 0.0.0.0 --port 8000  (from src/). Env: PD_DATADIR, PD_DEFAULT_MODE."""
from __future__ import annotations
import os, sys
from typing import Optional, List
from fastapi import FastAPI, Query, Response
from pydantic import BaseModel
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from recommender_service import RecommenderService  # type: ignore

DATADIR = os.environ.get("PD_DATADIR", "/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim")
DEFAULT_MODE = os.environ.get("PD_DEFAULT_MODE", "exact")  # exact (FlatIP) default; "scale" = HNSW ef64
# G26B: heuristic rerank is config-flagged and DEFAULT OFF (interpretable exposure control, NOT learned LTR)
ENABLE_RERANK = os.environ.get("PD_ENABLE_RERANK", "0") == "1"
RERANK_POLICY = os.environ.get("PD_RERANK_POLICY", "head_cap")  # head_cap | tail_boost
# Set PD_BUILD_HNSW=0 for the minimal ALS-only (exact FlatIP) deployment target — skips the
# HNSW scale index (smaller memory, faster cold start). Default ON (scale mode available).
BUILD_HNSW = os.environ.get("PD_BUILD_HNSW", "1") == "1"

app = FastAPI(title="PulseDiscover Serving API (V2, production-shaped)", version="0.2.0")
svc = RecommenderService(DATADIR, default_mode=DEFAULT_MODE, build_hnsw=BUILD_HNSW,
                         enable_rerank=ENABLE_RERANK, rerank_policy=RERANK_POLICY)


class Item(BaseModel):
    item_id: str; creator_id: Optional[str] = None; source: str
class RecommendResponse(BaseModel):
    items: List[Item]; retrieval_mode: Optional[str]; fallback_used: bool; candidate_count: int
    response_count: int; latency_ms: float; error_code: Optional[str]; request_id: str
    model_id: Optional[str]; index_id: Optional[str]
    rerank_applied: bool = False; rerank_policy: Optional[str] = None; base_candidate_ids: Optional[List[str]] = None


@app.get("/health")
def health():
    return {"status": "ok" if svc.ready else "degraded", "ready": svc.ready, "load_error": svc.load_error}

@app.get("/metadata")
def metadata():
    return svc.metadata()

@app.get("/recommend", response_model=RecommendResponse)
def recommend(user_id: str = Query(...), k: int = Query(20, ge=1, le=200), mode: Optional[str] = Query(None),
              rerank: Optional[bool] = Query(None, description="override config-flagged heuristic rerank (default off)")):
    return svc.recommend(user_id=user_id, k=k, mode=mode, rerank=rerank)

@app.get("/metrics.json")
def metrics_json():
    return svc.metrics()

@app.get("/metrics")
def metrics_prom():
    return Response(content=svc.prometheus(), media_type="text/plain")

@app.get("/debug/retrieval")
def debug_retrieval(user_id: str = Query(...), k: int = Query(20, ge=1, le=200), mode: Optional[str] = Query(None)):
    r = svc.recommend(user_id=user_id, k=k, mode=mode)
    return {"request": {"user_id_known": user_id in getattr(svc, "uidx", {}), "k": k, "mode": mode or svc.default_mode},
            "result": r, "metadata": svc.metadata()}
