"""H1 — creator/catalog health over repeated rounds (serving-side complement to P1).
Simulate R rounds for als_only / 90-10 / 80-20 / rrf at D2 prevalence. Feedback loop:
examined-gold clicks boost a GLOBAL warm-lane popularity score (rich-get-richer); the cold
reserve is protected from the boost. Track item/creator exposure Gini, effective creators,
long-tail share, distinct items/creators, cold + new-creator exposure share, warm recall cost.
Proxy reward only; NO business-lift claim."""
import pandas as pd, numpy as np, os, json, pickle, time, math
from collections import defaultdict
HERE=os.path.dirname(os.path.abspath(__file__))
import sys; sys.path.insert(0,HERE)
from pulsediscover.feedback_loop import gini
OUT="/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim"; SP=os.path.join(OUT,"domain_splits")
EVID="/sessions/festive-quirky-cannon/mnt/pulsediscover/outputs/evidence"; PLOTS="/sessions/festive-quirky-cannon/mnt/pulsediscover/outputs/plots"
L=20; ROUNDS=12; N=2400; WP=0.765; GAMMA=2.0; SEED=20260616; t0=time.time(); rng=np.random.default_rng(SEED)
exam=0.95*(1.0/(np.arange(L)+1.0))**0.7
d=pickle.load(open(os.path.join(OUT,"c2_lists.pkl"),"rb")); WL=d["warm_list"]; CL=d["cold_list"]; cold_set=d["cold_items"]; b2c=d["b2c"]
cre=lambda it: b2c.get(it,"unk_"+str(it))
w=pd.read_csv(os.path.join(SP,"warm_eval_sample.csv"),dtype=str); c=pd.read_csv(os.path.join(SP,"cold_eval_sample.csv"),dtype=str)
nW=int(round(N*WP)); nC=N-nW
wu=list(zip(w.user_id,w.gold))[:nW]; cu=list(zip(c.user_id,c.gold))[:nC]
users=[(u,g,"warm") for u,g in wu]+[(u,g,"cold") for u,g in cu]
POL=["als_only","q90_10","q80_20","rrf"]
NW={"als_only":20,"q90_10":18,"q80_20":16,"rrf":10}; NC={"als_only":0,"q90_10":2,"q80_20":4,"rrf":10}
def eff(counts):
    a=np.array([v for v in counts.values() if v>0],float)
    if a.sum()==0: return 0.0
    p=a/a.sum(); H=-(p*np.log(p)).sum(); return float(np.exp(H))
def longtail(counts):  # exposure share NOT in top-1% most-exposed items
    a=np.array(sorted(counts.values(),reverse=True),float)
    if a.sum()==0: return 0.0
    k=max(1,int(0.01*len(a))); return float(1-a[:k].sum()/a.sum())
def warm_pool(u,pop):  # rich-get-richer: warm rank = base ALS rank + GAMMA*log1p(cumulative warm exposure)
    wl=WL.get(u,[])[:60]
    base={it:(len(wl)-r) for r,it in enumerate(wl)}
    return sorted(wl,key=lambda it:-(base[it]+GAMMA*math.log1p(pop.get(it,0))))
report={p:{} for p in POL}; traj={p:defaultdict(list) for p in POL}
for p in POL:
    pop=defaultdict(float)  # cumulative WARM-lane exposure (drives rich-get-richer); cold reserve NOT boosted
    item_exp=defaultdict(int); cre_exp=defaultdict(int); seen_items=set(); seen_cre=set(); seen_cold_cre=set()
    for rd in range(ROUNDS):
        pop_snap=dict(pop); warm_hit=warm_n=0; overall_hit=0
        for u,gold,seg in users:
            wpool=warm_pool(u,pop_snap); cl=CL.get(u,[])
            slate=(wpool[:NW[p]]+cl[:NC[p]])[:L] if p!="rrf" else None
            if p=="rrf":
                rr=defaultdict(float)
                for r,it in enumerate(wpool): rr[it]+=1.0/(60+r)
                for r,it in enumerate(cl): rr[it]+=1.0/(60+r)
                slate=[i for i,_ in sorted(rr.items(),key=lambda t:-t[1])][:L]
            for r,it in enumerate(slate):
                item_exp[it]+=1; cr=cre(it); cre_exp[cr]+=1; seen_items.add(it); seen_cre.add(cr)
                if it in cold_set: seen_cold_cre.add(cr)
                else: pop[it]+=1
                if rng.random()<exam[r] and it==gold:
                    overall_hit+=1
                    if seg=="warm": warm_hit+=1
            if seg=="warm": warm_n+=1
        tot=sum(item_exp.values()); coldexp=sum(v for it,v in item_exp.items() if it in cold_set)
        traj[p]["item_gini"].append(round(gini(np.array(list(item_exp.values()),float)),4))
        traj[p]["cre_gini"].append(round(gini(np.array(list(cre_exp.values()),float)),4))
        traj[p]["eff_cre"].append(round(eff(cre_exp),1))
        traj[p]["longtail"].append(round(longtail(item_exp),4))
        traj[p]["distinct_items"].append(len(seen_items))
        traj[p]["distinct_cre"].append(len(seen_cre))
        traj[p]["cold_exp_share"].append(round(coldexp/tot,4))
        traj[p]["distinct_cold_cre"].append(len(seen_cold_cre))
        traj[p]["warm_hit_rate"].append(round(warm_hit/max(warm_n,1),4))
    report[p]={"final":{k:v[-1] for k,v in traj[p].items()},"round1":{k:v[0] for k,v in traj[p].items()}}
base=report["als_only"]["final"]
out={"dataset":"goodreads_fantasy_paranormal","scope":"creator/catalog exposure health over repeated rounds","tag":"[BUILT][SYNTHETIC — exposure-health simulator]",
 "config":{"rounds":ROUNDS,"n_users":N,"prevalence_warm":WP,"feedback":"clicks boost global warm popularity (GAMMA=%.2f); cold reserve protected"%GAMMA,"longtail_def":"exposure share outside top-1% most-exposed items"},
 "policies_final":{p:report[p]["final"] for p in POL},
 "trajectories":{p:dict(traj[p]) for p in POL},
 "deltas_vs_als_final":{p:{"cold_exp_share":round(report[p]["final"]["cold_exp_share"]-base["cold_exp_share"],4),
    "distinct_items":report[p]["final"]["distinct_items"]-base["distinct_items"],
    "distinct_cre":report[p]["final"]["distinct_cre"]-base["distinct_cre"],
    "distinct_cold_cre":report[p]["final"]["distinct_cold_cre"]-base["distinct_cold_cre"],
    "cre_gini":round(report[p]["final"]["cre_gini"]-base["cre_gini"],4),
    "eff_cre":round(report[p]["final"]["eff_cre"]-base["eff_cre"],1),
    "longtail":round(report[p]["final"]["longtail"]-base["longtail"],4),
    "warm_hit_rel_loss":round((report[p]["final"]["warm_hit_rate"]-base["warm_hit_rate"])/base["warm_hit_rate"],4) if base["warm_hit_rate"]>0 else None} for p in POL},
 "relevance_source_of_truth":"C2 warm R@20 rel-loss (static): q90_10 -8.1% (PASS<=10%), q80_20 -17%, rrf -44%. H1 dynamic warm-hit is a noisy proxy (gold-hit ~1%), NOT used as the guardrail.",
 "honest_notes":["Proxy reward (examined gold-hit); NOT engagement/business lift.",
   "Feedback loop is a simple rich-get-richer on the WARM lane (rank += GAMMA*log1p(cumulative warm exposure)); the cold reserve is protected from the boost. Auditable, not production dynamics.",
   "Exposure health measured on cumulative exposure across rounds; D2 prevalence (76.5% warm).",
   "Dynamic warm-hit rate here is too noisy to use as a relevance guardrail (tiny gold-hit rates); the C2 static warm R@20 loss is the relevance source of truth.",
   "Cold-exposure share is a STRUCTURAL guarantee (= reserve fraction), not an emergent behavioral win."],
 "sec":round(time.time()-t0,1)}
json.dump(out,open(os.path.join(EVID,"domain_creator_catalog_health_report.json"),"w"),indent=2)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rounds=np.arange(1,ROUNDS+1); fig,ax=plt.subplots(2,2,figsize=(13,9)); col={"als_only":"#888","q90_10":"#228833","q80_20":"#ee7733","rrf":"#cc3311"}
    for p in POL:
        ax[0,0].plot(rounds,traj[p]["cre_gini"],"o-",color=col[p],label=p,ms=3)
        ax[0,1].plot(rounds,traj[p]["distinct_cre"],"o-",color=col[p],label=p,ms=3)
        ax[1,0].plot(rounds,traj[p]["cold_exp_share"],"o-",color=col[p],label=p,ms=3)
        ax[1,1].plot(rounds,traj[p]["longtail"],"o-",color=col[p],label=p,ms=3)
    ax[0,0].set_title("Creator exposure Gini (lower=healthier)"); ax[0,1].set_title("Distinct creators surfaced (cumulative)")
    ax[1,0].set_title("Cold-item exposure share"); ax[1,1].set_title("Long-tail exposure share (outside top-1%)")
    for a in ax.flat: a.set_xlabel("round"); a.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS,"domain_creator_catalog_health.png"),dpi=110); plt.close()
except Exception as ex: print("plot skipped:",ex)
for p in POL:
    f=report[p]["final"]; print(p,"creGini %.3f effCre %.0f distItems %d distCre %d coldShare %.3f distColdCre %d warmHit %.4f longtail %.3f"%(
      f["cre_gini"],f["eff_cre"],f["distinct_items"],f["distinct_cre"],f["cold_exp_share"],f["distinct_cold_cre"],f["warm_hit_rate"],f["longtail"]))
print("warm_hit rel-loss vs als:",{p:out["deltas_vs_als_final"][p]["warm_hit_rel_loss"] for p in POL},"| %.1fs"%(time.time()-t0))
