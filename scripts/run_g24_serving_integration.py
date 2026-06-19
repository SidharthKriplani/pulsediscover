"""G24 — Serving integration tests: exercise the LIVE G23 RecommenderService
across cohort-representative cases and verify fallback behaviour end-to-end.
Merges results into outputs/evidence/g24_cold_start_sparse_cohort_report.json.
"""
from __future__ import annotations
import os, sys, json
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "serving"))
from serving.recommender_service import RecommenderService
from eval.cold_start_cohorts import load_als, user_history

DATA = os.environ.get("PD_DATADIR", "data/interim")
SPLITS = os.path.join(DATA, "domain_splits")
REPORT = "outputs/evidence/g24_cold_start_sparse_cohort_report.json"

print("loading service (G23) ...")
svc = RecommenderService(DATA, default_mode="exact", build_hnsw=True)
assert svc.ready, f"service failed to load: {svc.load_error}"

# pick representative users per cohort from train history
train = pd.read_csv(os.path.join(SPLITS, "train.csv"),
                    usecols=["user_id", "book_id"], dtype={"book_id": str})
hist = user_history(train)
als = load_als(); uidx = als["uidx"]
warm = [u for u in hist[hist >= 6].index if u in uidx][:1]
sparse = [u for u in hist[(hist >= 1) & (hist <= 2)].index if u in uidx][:1]
# a user present in uidx but with no train history in this split (factor w/o history)
nofac_like = [u for u in uidx if hist.get(u, 0) == 0][:1]

cases = [
    ("warm_known_user", warm[0] if warm else next(iter(uidx)), "exact", 20),
    ("sparse_user", sparse[0] if sparse else next(iter(uidx)), "exact", 20),
    ("unknown_user", "THIS_USER_DOES_NOT_EXIST_xyz", "exact", 20),
    ("missing_factor_user", "another_unseen_user_42", "exact", 20),
    ("scale_mode_warm", warm[0] if warm else next(iter(uidx)), "scale", 20),
    ("invalid_k_zero", warm[0] if warm else next(iter(uidx)), "exact", 0),
    ("invalid_k_huge", warm[0] if warm else next(iter(uidx)), "exact", 9999),
]

results = []
for name, uid, mode, k in cases:
    r = svc.recommend(uid, k=k, mode=mode)
    # check item-metadata-missing resilience: any item with null creator still served
    null_creator = sum(1 for it in r["items"] if it.get("creator_id") is None)
    results.append({
        "case": name,
        "retrieval_mode": r["retrieval_mode"],
        "fallback_used": r["fallback_used"],
        "error_code": r["error_code"],
        "response_count": r["response_count"],
        "candidate_count": r["candidate_count"],
        "source0": r["items"][0]["source"] if r["items"] else None,
        "items_with_null_creator": null_creator,
        "nonempty": r["response_count"] > 0,
    })
    print(f"  {name:22s} mode={r['retrieval_mode']:18s} fb={r['fallback_used']!s:5s} "
          f"err={r['error_code']} n={r['response_count']}")

integration = {
    "service_ready": svc.ready,
    "metadata": svc.metadata(),
    "cases": results,
    "all_cases_nonempty": all(c["nonempty"] for c in results),
    "metrics_after": svc.metrics(),
    "assertions": {
        "warm_user_uses_als": any(c["case"] == "warm_known_user" and c["source0"] == "als" for c in results),
        "unknown_user_falls_back_to_popularity": any(
            c["case"] == "unknown_user" and c["retrieval_mode"] == "fallback_popularity" for c in results),
        "scale_mode_uses_hnsw": any(c["case"] == "scale_mode_warm" and c["retrieval_mode"] == "scale_hnsw" for c in results),
        "invalid_k_handled_gracefully": all(
            c["nonempty"] for c in results if c["case"].startswith("invalid_k")),
        "zero_empty_responses": all(c["nonempty"] for c in results),
    },
}

rep = json.load(open(REPORT))
rep["serving_integration"] = integration
json.dump(rep, open(REPORT, "w"), indent=2)
print("\nassertions:", json.dumps(integration["assertions"], indent=2))
print("all cases nonempty:", integration["all_cases_nonempty"])
