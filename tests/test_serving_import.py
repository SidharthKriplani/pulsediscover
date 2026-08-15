"""Fast CI smoke test for the serving app: import-check + a couple of endpoints.

Deliberately does NOT require the real ALS/FAISS model artifacts (data/interim/*.pkl) —
those are large, not committed, and built via scripts/build_serving_assets.py. Without
them, RecommenderService degrades gracefully (ready=False, load_error set — see
src/serving/recommender_service.py) rather than raising, so this test only proves the
app imports cleanly and the process-level endpoints (/health, /metadata) don't crash.
It is NOT a substitute for an end-to-end test against real model artifacts.
"""
import os

os.environ.setdefault("PD_DATADIR", "/tmp/pulsediscover-ci-no-data")
os.environ.setdefault("PD_BUILD_HNSW", "0")

from fastapi.testclient import TestClient  # noqa: E402


def test_import_serving_api():
    """The serving app must import without raising, even with no model data present."""
    import serving.api as api

    assert api.app is not None


def test_health_endpoint_responds():
    import serving.api as api

    client = TestClient(api.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body and "ready" in body


def test_metadata_endpoint_responds():
    import serving.api as api

    client = TestClient(api.app)
    resp = client.get("/metadata")
    assert resp.status_code == 200
