"""G26B — In-service heuristic rerank integration evaluation (PulseDiscover V2).

Drives the LIVE RecommenderService (the served c2 ALS f64 spine) twice over the same
warm-user sample: baseline (rerank OFF) vs heuristic rerank ON (config-flagged, head_cap).
Measures the relevance / exposure / latency tradeoff end-to-end through the serving path,
plus cohort behaviour. Heuristic exposure control — NOT learned ranking, NOT LTR.

Output: outputs/evidence/g26b_rerank_integration_report.json
"""
from __future__ import annotations
import os, sys, json, time
from collections import defaultdict
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "src", "serving"))
from serving.recommender_service import RecommenderService
from eval.cold_start_cohorts import load_als, load_eval_samples, build_cohorts
from eval.catalog_exposure_governance import exposure_stats

DATA = os.environ.get("PD_DATADIR", "data/interim")
SPLITS = os.path.join(DATA, "domain_splits")
EVID = "outputs/evidence"; os.makedirs(EVID, exist_ok=True)
K = 20
t0 = time.time()

print("[1/4] load service (rerank=head_cap, config-flagged default OFF) + cohorts ...")
svc = RecommenderService(DATA, default_mode="exact", build_hnsw=False,
                         enable_rerank=False, rerank_policy="head_cap", rerank_cap_frac=0.3, rerank_pool_mult=4)
assert svc.ready, svc.load_error

train = pd.read_csv(os.path.join(SPLITS, "train.csv"), usecols=["user_id", "book_id", "event_ts"], dtype={"book_id": str})
als = load_als(); ev = load_eval_samples()
coh, ctx = build_cohorts(train, als, ev)
item_tier = ctx["item_tier"]; catalog = list(map(str, als["i_uniq"])); catalog_n = len(catalog)
gold_by_user = dict(zip(coh["user_id"], coh["gold"]))
cohort_by_user = dict(zip(coh["user_id"], coh["cohort"]))

# warm users with factors (rerank only affects the ALS retrieval path)
warm = coh[(coh["cohort"] == "warm_6plus") & (coh["has_factor"])]["user_id"].tolist()
rng = np.random.default_rng(11)
if len(warm) > 4000:
    warm = list(np.array(warm)[rng.choice(len(warm), 4000, replace=False)])
print(f"   warm sample={len(warm)}  ({time.time()-t0:.1f}s)")


def run_pass(rerank_flag):
    exposure = defaultdict(int); hit = 0; lat = []; empty = 0
    cohort_hit = defaultdict(int); cohort_n = defaultdict(int); base_vs_final_changed = 0
    for u in warm:
        r = svc.recommend(u, k=K, mode="exact", rerank=rerank_flag)
        lat.append(r["latency_ms"])
        ids = [it["item_id"] for it in r["items"]]
        if not ids:
            empty += 1; continue
        g = gold_by_user[u]; c = cohort_by_user[u]; cohort_n[c] += 1
        if g in ids[:K]:
            hit += 1; cohort_hit[c] += 1
        for b in ids[:K]:
            exposure[b] += 1
        if rerank_flag and r.get("base_candidate_ids") and set(ids[:K]) != set(r["base_candidate_ids"][:K]):
            base_vs_final_changed += 1
    n = len(warm)
    es = exposure_stats(exposure, catalog, item_tier, K)
    return {
        "n": n, "recall@20": round(hit / n, 4), "empty_rate": round(empty / n, 4),
        "catalog_coverage": es["catalog_coverage"], "unique_items": es["unique_items_recommended"],
        "gini": es["gini"], "head_share": es["tier_exposure_share"]["head"],
        "mid_share": es["tier_exposure_share"]["mid"], "long_tail_share": es["tier_exposure_share"]["long_tail"],
        "zero_exposure_share": es["zero_exposure_share"], "top_1pct_share": es["top_1pct_share"],
        "latency_p50_ms": round(float(np.percentile(lat, 50)), 3),
        "latency_p95_ms": round(float(np.percentile(lat, 95)), 3),
        "slates_changed_by_rerank": base_vs_final_changed,
    }, cohort_hit, cohort_n


print("[2/4] baseline pass (rerank OFF) ...")
base, _, _ = run_pass(False)
print("[3/4] reranked pass (rerank ON, head_cap) ...")
rer, ch_hit, ch_n = run_pass(True)
cohort_results = {c: {"recall@20_reranked": round(ch_hit[c] / max(ch_n[c], 1), 4), "n": ch_n[c]} for c in ch_n}

relevance_delta = {"recall@20_abs": round(rer["recall@20"] - base["recall@20"], 4),
                   "recall@20_pct": round((rer["recall@20"] / base["recall@20"] - 1) * 100, 1) if base["recall@20"] else None}
exposure_delta = {"catalog_coverage_abs": round(rer["catalog_coverage"] - base["catalog_coverage"], 5),
                  "unique_items_abs": rer["unique_items"] - base["unique_items"],
                  "long_tail_share_abs": round(rer["long_tail_share"] - base["long_tail_share"], 5),
                  "gini_abs": round(rer["gini"] - base["gini"], 4)}
latency_delta = {"p50_abs_ms": round(rer["latency_p50_ms"] - base["latency_p50_ms"], 3),
                 "p95_abs_ms": round(rer["latency_p95_ms"] - base["latency_p95_ms"], 3)}

# ship decision: enable as a config option (default still OFF) since it does not degrade relevance
ship = ("enable_as_config_option_default_off" if relevance_delta["recall@20_abs"] >= 0
        and exposure_delta["long_tail_share_abs"] > 0 else "keep_off")

report = {
    "gate": "G26B",
    "title": "Heuristic Reranker Integration (in-service)",
    "lane": "V2",
    "reranker_type": "heuristic",
    "rerank_policy": "head_cap (cap_frac=0.3, pool_mult=4)",
    "config_default": "off",
    "served_model": "c2_als.pkl (ALS f64) — same served spine as G22-G26",
    "scope_guardrails": [
        "Heuristic exposure control (head-cap), NOT learned ranking / NOT LTR.",
        "Warm ALS retrieval path only; cold/unknown users keep popularity fallback (rerank does not touch them).",
        "Served c2 metrics only; NOT comparable to V1 d3aplus (0.0846).",
        "In-service offline eval; no online lift, no OPE execution.",
    ],
    "baseline_metrics": base,
    "reranked_metrics": rer,
    "relevance_delta": relevance_delta,
    "exposure_delta": exposure_delta,
    "latency_delta": latency_delta,
    "cohort_results_if_available": cohort_results,
    "ship_decision": ship,
    "claim_status": {
        "quarantine_closed": True,
        "reranker_is_heuristic_not_learned": True,
        "long_tail_solved": False,
        "cold_start_solved": False,
        "online_lift": False,
        "ope_executed": False,
        "v1_v2_metrics_mixed": False,
        "safe_claim": ("Integrated a config-flagged heuristic reranker into the served recommender "
                       "path and measured the relevance/exposure/latency tradeoff against the served "
                       "c2 baseline; interpretable exposure control, not learned ranking."),
    },
    "forbidden_claims": ["learned LTR deployed", "cold-start solved", "long-tail discovery solved",
                          "online lift proven", "OPE executed", "fairness certified",
                          "mixing V1 d3aplus with served c2 metrics"],
    "next_gate": "G27_SEMANTIC_CONTENT_RETRIEVAL",
    "runtime_sec": round(time.time() - t0, 1),
}
json.dump(report, open(os.path.join(EVID, "g26b_rerank_integration_report.json"), "w"), indent=2)
print(f"[4/4] DONE in {time.time()-t0:.1f}s")
print("baseline  R@20", base["recall@20"], "cov", base["catalog_coverage"], "tail", base["long_tail_share"], "gini", base["gini"], "p95", base["latency_p95_ms"])
print("reranked  R@20", rer["recall@20"], "cov", rer["catalog_coverage"], "tail", rer["long_tail_share"], "gini", rer["gini"], "p95", rer["latency_p95_ms"])
print("ship_decision:", ship, "| slates changed:", rer["slates_changed_by_rerank"])
