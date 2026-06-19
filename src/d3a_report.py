"""D3A step 4 — aggregate popularity/co-occurrence/ALS metrics on the 5,000-user domain
leave-last-out sample with bootstrap CIs; cold-start breakdown; write domain_baselines_report.json
+ plot. Domain-scoped (Goodreads fantasy_paranormal)."""
import json, os, numpy as np, sys
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from pulsediscover import eval as E
OUT="/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim"; SP=os.path.join(OUT,"domain_splits")
PLOTS="/sessions/festive-quirky-cannon/mnt/pulsediscover/outputs/plots"; EVID="/sessions/festive-quirky-cannon/mnt/pulsediscover/outputs/evidence"
KS=[5,10,20]
pc=json.load(open(os.path.join(OUT,"d3a_popcooc_metrics.json")))
al=json.load(open(os.path.join(OUT,"d3a_als_metrics.json")))
warm=np.array(pc["gold_is_warm"])
methods={"popularity":pc["res"]["popularity"],"cooccurrence":pc["res"]["cooccurrence"],"als":al["res"]["als"]}
agg={}
for m,r in methods.items():
    agg[m]={}
    for k in KS:
        rm,rlo,rhi,n=E.bootstrap_ci(r[str(k)]["r"],seed=20260616)
        nm,nlo,nhi,_=E.bootstrap_ci(r[str(k)]["n"],seed=20260616)
        agg[m][f"@{k}"]={"recall_mean":round(rm,4),"recall_ci95":[round(rlo,4),round(rhi,4)],
                         "ndcg_mean":round(nm,4),"ndcg_ci95":[round(nlo,4),round(nhi,4)],"n":n}
# cold-start at eval-gold level (item warm vs new in train)
nwarm=int(warm.sum()); nnew=int((~warm).sum())
def recall_split(method,k,mask):
    arr=np.array(methods[method][k]["r"])[mask]
    return round(float(arr.mean()),4) if len(arr) else None, int(len(arr))
cold={"eval_gold_items_warm":nwarm,"eval_gold_items_new":nnew,
      "note":"leave-last-out golds on a k-cored set are ~all warm; the substantive cold-start is the D2 global-time test window (restated below)."}
d2=json.load(open(os.path.join(OUT,"d2_stats.json")))["coldstart_vs_train"]
report={
 "dataset":"goodreads_fantasy_paranormal","scope":"DOMAIN (Goodreads fantasy/paranormal)",
 "tag":"[BUILT — real data (domain)]","eval":"leave-last-out, 5,000-user sample of 42,683 held-out test users",
 "baselines":agg,
 "cooccurrence_config":{"vocab_top":pc["cooc_vocab"],"min_co":pc["min_co"]},
 "als_config":{"factors":al["factors"],"iters":al["iters"],"fit_sec":al["fit_sec"]},
 "coldstart_eval_gold":cold,
 "coldstart_global_time_D2":d2,
 "honest_notes":["co-occurrence capped to top-5k items (M^T M at 20k OOM'd on 3.8GB) — a real CPU-scale limit, reported.",
   "ALS uses constant confidence (all user-book pairs unique). bounded iters=%d for CPU."%al["iters"],
   "evaluated on a 5,000-user random sample (seeded) of the 42,683 leave-last-out users; bootstrap CIs reflect that.",
   "no SASRec/OPE/position-bias/feedback in D3A."],
}
json.dump(report,open(os.path.join(EVID,"domain_baselines_report.json"),"w"),indent=2)
# plot
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ms=["popularity","cooccurrence","als"]; fig,ax=plt.subplots(figsize=(7,4.5))
    for m in ms:
        ax.plot(KS,[agg[m][f"@{k}"]["recall_mean"] for k in KS],marker="o",label=m)
        for k in KS:
            lo,hi=agg[m][f"@{k}"]["recall_ci95"]; ax.plot([k,k],[lo,hi],color="gray",alpha=0.4)
    ax.set_xlabel("K");ax.set_ylabel("Recall@K");ax.set_title("Domain retrieval baselines — Goodreads fantasy/paranormal (5k-user LLO, 95% CI)");ax.legend()
    plt.tight_layout();plt.savefig(os.path.join(PLOTS,"domain_baselines_recall.png"),dpi=110);plt.close()
except Exception as ex: print("plot skipped:",ex)
for m in ms:
    print(m, {f"R@{k}":agg[m][f'@{k}']["recall_mean"] for k in KS}, "NDCG@10",agg[m]["@10"]["ndcg_mean"])
print("cold-start eval golds warm/new:",nwarm,"/",nnew)
