"""C1 step 4 — aggregate cold-start generators with bootstrap CIs, warm-vs-cold contrast,
coverage; write domain_coldstart_report.json + plot."""
import json, os, numpy as np, sys
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from pulsediscover import eval as E
OUT="/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim"; EVID="/sessions/festive-quirky-cannon/mnt/pulsediscover/outputs/evidence"; PLOTS="/sessions/festive-quirky-cannon/mnt/pulsediscover/outputs/plots"
KS=[20,50,100,200]; SEED=20260616
p1=json.load(open(os.path.join(OUT,"c1_partial.json"))); p2=json.load(open(os.path.join(OUT,"c1_tfidf_partial.json")))
methods={"als_f64":p1["res"]["als_f64"],"same_author":p1["res"]["same_author"],"same_series":p1["res"]["same_series"],
         "tfidf":p2["res"]["tfidf"],"content_hybrid":p2["res"]["content_hybrid"]}
agg={}
for m,r in methods.items():
    agg[m]={}
    for k in KS:
        rm,lo,hi,n=E.bootstrap_ci(r[str(k)],seed=SEED); agg[m][f"R@{k}"]={"mean":round(rm,4),"ci95":[round(lo,4),round(hi,4)]}
warm=json.load(open(os.path.join(EVID,"domain_als_f64_floor.json")))["metrics"]   # warm LLO recall
report={"dataset":"goodreads_fantasy_paranormal","scope":"DOMAIN cold-start (global-time test, new-vs-pre-test items)",
 "tag":"[BUILT — real data (domain)]","eval":"cold-start: 5,000 test interactions on items NEVER seen pre-test; users with pre-test history",
 "cold_pool":{"n_cold_items":p1["n_cold_items"],"n_new_creators":p1["n_cold_creators_new"],
              "n_new_series":p1["n_cold_series_new"],"n_cold_test_interactions":p1["n_cold_test_interactions"],
              "gold_author_warm_pct":p1["gold_author_warm_pct"]},
 "coldstart_recall":agg,
 "warm_context_ALS_f64_LLO":{f"R@{k}":warm[f"R@{k}"]["mean"] for k in KS},
 "coverage":{"cold_items_total":p1["n_cold_items"],
             "sameauthor_surfaced_cold_items":p1["surfaced_cold_items_sameauthor"],
             "sameauthor_surfaced_authors":p1["surfaced_cold_authors"],
             "hybrid_surfaced_cold_items_top100":p2["surfaced_cold_items_hybrid_top100"],
             "hybrid_cold_item_coverage_pct":round(100*p2["surfaced_cold_items_hybrid_top100"]/p1["n_cold_items"],1)},
 "runtime_sec":{"coldeval_sameauthor_series":p1["sec"],"tfidf_hybrid":p2["sec"]},
 "decision":"KEEP — content channel gives non-zero cold-start recall (hybrid R@20 0.24, R@200 0.56) with broad coverage and fast CPU runtime, where ALS f64 is structurally ~0.",
 "honest_notes":["ALS f64 cold recall is 0 BY CONSTRUCTION (new items absent from its factor space) — the contrast, not a tuning failure.",
   "same_author ceiling = 32.8% (gold's author warm); same_series ~0 (cold golds rarely in a previously-read series).",
   "OFFLINE cold-start candidate channel only — NOT proven business lift; no A/B, no online exposure test.",
   "cold pool is core (k-cored) items new in the test window; sub-threshold brand-new items are out of the core scope."]}
json.dump(report,open(os.path.join(EVID,"domain_coldstart_report.json"),"w"),indent=2)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(13,4.5))
    for m in ["content_hybrid","same_author","tfidf","same_series","als_f64"]:
        ax[0].plot(KS,[agg[m][f"R@{k}"]["mean"] for k in KS],marker="o",label=m)
    ax[0].set_xlabel("K"); ax[0].set_ylabel("cold-start Recall@K"); ax[0].set_title("Cold-start recall (new-in-test items)"); ax[0].legend()
    labels=["ALS f64\n(warm job)","ALS f64\n(cold)","content_hybrid\n(cold)"]
    vals=[warm["R@20"]["mean"],agg["als_f64"]["R@20"]["mean"],agg["content_hybrid"]["R@20"]["mean"]]
    ax[1].bar(labels,vals,color=["#4477aa","#cc6677","#228833"]); ax[1].set_ylabel("Recall@20"); ax[1].set_title("Division of labor: warm vs cold @20")
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS,"domain_coldstart_recall.png"),dpi=110); plt.close()
except Exception as ex: print("plot skipped:",ex)
for m in methods: print(m,{f"R@{k}":agg[m][f"R@{k}"]["mean"] for k in KS})
print("hybrid cold coverage %%:",report["coverage"]["hybrid_cold_item_coverage_pct"])
