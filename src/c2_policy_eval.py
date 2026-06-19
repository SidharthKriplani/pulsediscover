"""C2 step 3 — simulate two-lane serving policies over the prebuilt warm(ALS)/cold(content) lists.
Warm/cold/overall recall, cold exposure share, distinct cold surfaced, catalog coverage, creator
Gini (over the warm-traffic top-20 feed). Writes domain_two_lane_policy_report.json + tradeoff plot."""
import pandas as pd, numpy as np, os, json, pickle, math, sys
from collections import defaultdict
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from pulsediscover.feedback_loop import gini
OUT="/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim"; SP=os.path.join(OUT,"domain_splits")
EVID="/sessions/festive-quirky-cannon/mnt/pulsediscover/outputs/evidence"; PLOTS="/sessions/festive-quirky-cannon/mnt/pulsediscover/outputs/plots"
KS=[20,50,100,200]; NK=[10,20]; WARM_PREV=0.765; COLD_PREV=0.235
L=pickle.load(open(os.path.join(OUT,"c2_lists.pkl"),"rb")); WL=L["warm_list"]; CL=L["cold_list"]; cold_set=L["cold_items"]; b2c=L["b2c"]
wsamp=pd.read_csv(os.path.join(SP,"warm_eval_sample.csv"),dtype=str); csamp=pd.read_csv(os.path.join(SP,"cold_eval_sample.csv"),dtype=str)
# RRF merged per user (precompute)
def rrf(u):
    rr=defaultdict(float)
    for r,it in enumerate(WL.get(u,[])): rr[it]+=1.0/(60+r)
    for r,it in enumerate(CL.get(u,[])): rr[it]+=1.0/(60+r)
    return [i for i,_ in sorted(rr.items(),key=lambda t:-t[1])]
POLICIES={"als_only":("quota",0.0),"cold_only":("quota",1.0),"q90_10":("quota",0.10),
          "q80_20":("quota",0.20),"q70_30":("quota",0.30),"rrf_blend":("rrf",None),"explore2":("explore",2)}
def served(u,K,kind,p):
    if kind=="rrf": return rrf(u)[:K]
    if kind=="explore": cn=min(p,K)
    else: cn=int(round(p*K))
    wn=K-cn
    return list(WL.get(u,[])[:wn])+list(CL.get(u,[])[:cn])
def rank_in(lst,g):
    return lst.index(g) if g in lst else -1
report={}
for name,(kind,p) in POLICIES.items():
    wr={k:[] for k in KS}; cr={k:[] for k in KS}; nd={k:[] for k in NK}
    # recall: warm golds via warm slots, cold via cold slots, overall via mixed
    for u,g in zip(wsamp.user_id,wsamp.gold):
        for k in KS:
            s=served(u,k,kind,p); wr[k].append(1.0 if g in s else 0.0)
        for k in NK:
            s=served(u,k,kind,p); r=rank_in(s,g); nd[k].append(1.0/math.log2(r+2) if r>=0 else 0.0)
    for u,g in zip(csamp.user_id,csamp.gold):
        for k in KS:
            s=served(u,k,kind,p); cr[k].append(1.0 if g in s else 0.0)
    # NDCG cold for overall
    ndc={k:[] for k in NK}
    for u,g in zip(csamp.user_id,csamp.gold):
        for k in NK:
            s=served(u,k,kind,p); r=rank_in(s,g); ndc[k].append(1.0/math.log2(r+2) if r>=0 else 0.0)
    # exposure / coverage / Gini over warm-traffic top-20 feed
    cold_share=[]; surfaced_cold=set(); surfaced_all=set(); cre=defaultdict(int)
    for u in wsamp.user_id:
        s=served(u,20,kind,p); nc=sum(1 for it in s if it in cold_set)
        cold_share.append(nc/max(1,len(s))); surfaced_all.update(s); surfaced_cold.update(it for it in s if it in cold_set)
        for it in s: cre[b2c.get(it,"")]+=1
    report[name]={
      "warm_recall":{f"R@{k}":round(float(np.mean(wr[k])),4) for k in KS},
      "cold_recall":{f"R@{k}":round(float(np.mean(cr[k])),4) for k in KS},
      "overall_recall_prevweighted":{f"R@{k}":round(WARM_PREV*float(np.mean(wr[k]))+COLD_PREV*float(np.mean(cr[k])),4) for k in KS},
      "ndcg_warm":{f"N@{k}":round(float(np.mean(nd[k])),4) for k in NK},
      "ndcg_cold":{f"N@{k}":round(float(np.mean(ndc[k])),4) for k in NK},
      "cold_exposure_share_top20":round(float(np.mean(cold_share)),4),
      "distinct_cold_surfaced_top20":len(surfaced_cold),
      "catalog_coverage_top20":len(surfaced_all),
      "creator_gini_top20":round(gini(np.array([v for v in cre.values()])),4)}
# decision: warm-recall guardrail = <=10% relative loss vs als_only @20
base=report["als_only"]["warm_recall"]["R@20"]
for name in report:
    wl=report[name]["warm_recall"]["R@20"]; loss=(base-wl)/base if base>0 else 0
    cold=report[name]["cold_recall"]["R@20"]
    report[name]["warm_recall_rel_loss_vs_alsonly@20"]=round(loss,4)
    report[name]["ship_flag"]=("SHIP-CANDIDATE" if (loss<=0.10 and cold>=0.05 and name not in("als_only","cold_only"))
                               else ("HOLD" if loss>0.10 else "baseline"))
out={"dataset":"goodreads_fantasy_paranormal","scope":"DOMAIN two-lane serving policy (offline)","tag":"[BUILT — real data (domain)]",
     "warm_prevalence":WARM_PREV,"cold_prevalence":COLD_PREV,"n_warm_eval":len(wsamp),"n_cold_eval":len(csamp),
     "policies":report,
     "honest_notes":["OFFLINE slot-allocation simulation — NOT A/B / business lift. Warm & cold golds are separate eval sets; overall is prevalence-weighted (D2: warm 76.5%, cold 23.5%).",
       "cold-item exposure share at top-20 is ~quota by construction; the load-bearing numbers are distinct-cold-surfaced, catalog coverage, creator Gini, and the warm-recall cost.",
       "warm golds reachable only via ALS slots, cold golds only via content slots (pools disjoint)."]}
json.dump(out,open(os.path.join(EVID,"domain_two_lane_policy_report.json"),"w"),indent=2)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    order=["als_only","q90_10","q80_20","q70_30","rrf_blend","explore2","cold_only"]
    warm=[report[p]["warm_recall"]["R@20"] for p in order]; cold=[report[p]["cold_recall"]["R@20"] for p in order]
    fig,ax=plt.subplots(figsize=(7.5,5))
    ax.scatter(warm,cold,s=60)
    for p,w,c in zip(order,warm,cold): ax.annotate(p,(w,c),fontsize=8,xytext=(4,4),textcoords="offset points")
    ax.set_xlabel("Warm Recall@20"); ax.set_ylabel("Cold Recall@20"); ax.set_title("Two-lane policy tradeoff: warm relevance vs cold discovery @20")
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS,"domain_two_lane_tradeoff.png"),dpi=110); plt.close()
except Exception as ex: print("plot skipped:",ex)
for p in ["als_only","q90_10","q80_20","q70_30","rrf_blend","explore2","cold_only"]:
    r=report[p]; print(p,"warmR@20",r["warm_recall"]["R@20"],"coldR@20",r["cold_recall"]["R@20"],"overall@20",r["overall_recall_prevweighted"]["R@20"],
          "coldshare",r["cold_exposure_share_top20"],"distinctcold",r["distinct_cold_surfaced_top20"],"Gini",r["creator_gini_top20"],r["ship_flag"])
