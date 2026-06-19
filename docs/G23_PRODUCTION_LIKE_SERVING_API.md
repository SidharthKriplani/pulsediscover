# G23 — Production-Like Serving API + Monitoring (V2)

*PulseDiscover **V2** lane. Turns the offline recommender into a **production-SHAPED** local serving system — FastAPI + FAISS retrieval + versioned loader + fallback tree + structured logging + monitoring + load test + Docker. **This is production-shaped, NOT a real production deployment: no real users, no online lift, CPU in-sandbox latency, V1 conclusions untouched.** Evidence: `outputs/evidence/g23_serving_api_report.json`. Code: `src/serving/{api.py, recommender_service.py, faiss_retriever.py}`; load test `scripts/run_g23_serving_load_test.py`; `Dockerfile` + `docker-compose.yml`.*

## 1. Architecture
`GET /recommend` → **RecommenderService** → (versioned loader: ALS user/item factors + FAISS FlatIP exact [default] / HNSW ef64 [scale] + popularity fallback list + item metadata) → retrieve top-K → map to items + creator → structured log + metrics → JSON response. Startup validates index/vector dim compatibility; on load failure the service degrades (health=degraded) and serves popularity.

## 2. Endpoint contract
| Endpoint | Purpose |
|---|---|
| `GET /recommend?user_id=&k=&mode=` | top-K recommendations; `mode` ∈ {exact (default), scale} |
| `GET /health` | `{status: ok\|degraded, ready, load_error}` |
| `GET /metadata` | model_id, index_id, build_ts, n_items, embedding_dim, retrieval_modes, default_mode |
| `GET /metrics` | Prometheus-style text |
| `GET /metrics.json` | JSON metrics |
| `GET /debug/retrieval?user_id=&k=` | result + whether user known + metadata |

**Response schema:** `items[]{item_id, creator_id?, source}`, `retrieval_mode`, `fallback_used`, `candidate_count`, `response_count`, `latency_ms`, `error_code?`, `request_id`, `model_id`, `index_id`.

**Example (warm):** `GET /recommend?user_id=<known>&k=5` → `{"items":[{"item_id":"...","creator_id":"...","source":"als"},...],"retrieval_mode":"exact","fallback_used":false,"candidate_count":5,"response_count":5,"latency_ms":0.9,"error_code":null,...}`
**Example (cold/unknown):** → `{"items":[...popularity...],"retrieval_mode":"fallback_popularity","fallback_used":true,"error_code":"unknown_user","response_count":20,...}` (still returns results).

## 3. Fallback decision tree (nothing hidden)
unknown / sparse / missing-factor user → **popularity** · invalid k → **422** (pydantic) or popularity · empty candidates → **popularity** · FAISS error → **popularity** · index-load failure → **degraded health + popularity** · missing item metadata → `creator_id: null` (no crash). Every path returns a valid response — **0% empty-response rate measured.**

## 4. Monitoring plan
In-memory + `/metrics`: request_count, error_rate, fallback_rate, empty_response_rate, unknown_user_rate, latency p50/p95/p99, by-mode counts, model/index version. Production would ship these to Prometheus/Grafana + alert on fallback_rate, error_rate, p95, empty_rate, and a model/index-version change.

## 5. Load-test results (in-process TestClient: routing + serialization + service; CPU in-sandbox)
| Segment | p50 | p95 | p99 | fallback | error | empty |
|---|---|---|---|---|---|---|
| warm users (n=1500) | 2.70 ms | 3.34 ms | 4.40 ms | 0% | 0% | 0% |
| cold/unknown (n=500) | 2.37 ms | 2.90 ms | 5.16 ms | **100%** (→popularity) | 100% (handled) | **0%** |
| scale mode HNSW (n=500) | 2.77 ms | 3.19 ms | — | 0% | 0% | 0% |
- **Throughput:** 276 qps sequential; **245 qps** with 8 concurrent threads (GIL-bound; concurrent p95 ~46 ms under contention). Service-internal latency (excl. ASGI/serialization) p50 0.54 / p95 3.78 / p99 6.49 ms.
- **Failure modes verified:** invalid k (9999 / 0) → **422**; unknown user → graceful popularity; health ok; metadata/metrics/debug endpoints return 200. **No crash on any path.**
- **0% empty-response rate** across 3,802 requests — every request returned recommendations.

## 6. What is production-LIKE (and what is NOT)
**Production-like:** real FastAPI app, versioned model/index loader with startup validation, exact + approximate retrieval modes, complete fallback tree, structured per-request logging, monitoring endpoints (Prometheus + JSON), health check, a Docker image + compose, and a measured load/failure-mode test.
**NOT production:** no real users, no online lift, no live A/B; CPU in-sandbox latency (excludes real network, autoscaling, persistence, auth, rate-limiting); single-process (GIL-bound concurrency); data mounted read-only, not a feature store.

## 7. Known limitations
GIL-bound concurrency (would use gunicorn workers / async); popularity is the only built fallback (content/category fallback needs per-user history wired in); no auth/rate-limit; latency is in-process, not over real network; metrics are in-memory (not persisted).

## 8. How this changes for REAL deployment
Multi-worker (gunicorn+uvicorn) or async; Prometheus/Grafana + alerting; a feature/embedding store with versioned reload; canary + the G4 live A/B; auth + rate limiting; autoscaling; the G3 risk register's monitors wired to alerts; nearline factor refresh (G1).

## 9. Claims (V2)
- **Safe (proven):** *"Built and load-tested a production-like recommender serving API with FAISS retrieval, versioned model/index loading, fallback behavior, structured logging, health checks, and latency/error monitoring hooks."*
- **Forbidden (enforced):** deployed a production recommender · real users · online lift · HNSW as silent default (it's an explicit `mode=scale`; FlatIP exact is default).

## 10. Status
- ✅ G23 done (V2). `production_claim_allowed: false`, `production_like_claim_allowed: true`.
- Run locally: `cd src && PD_DATADIR=../data/interim uvicorn serving.api:app --port 8000` or `docker compose up`. V1 unchanged.

**STOP — G23 complete; awaiting review.**
