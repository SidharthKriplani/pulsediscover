"""H2 — creator-aware reranking on the two-lane policy. Rerank the warm lane (creator cap / MMR
creator penalty / cold-creator boost) over a deeper ALS pool while keeping the 2-slot cold reserve.
Eval vs ALS-only & base 90/10: warm/cold/prevalence Recall@20/50, creator Gini, effective creators,
distinct (cold) creators, item coverage, runtime. Offline reranking sim — NO business lift."""
import pandas as pd, numpy as np, os, json, pickle, time, math
from collections import defaultdict
import sys; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from pulsediscover.feedback_loop import gini
OUT="/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim"; SP=os.path.join(OUT,"domain_splits")
EVID="/sessions/festive-quirky-cannon/mnt/pulsediscover/outputs/evidence"; PLOTS="/sessions/festive-quirky-cannon/mnt/pulsediscover/outputs/plots"
WP=0.765; CP=0.235; N=4000; POOL=60; SEED=20260616; rng=np.random.default_rng(SEED)
d=pickle.load(open(os.path.join(OUT,"c2_lists.pkl"),"rb")); WL=d["warm_list"]; CL=d["cold_list"]; cold_set=d["cold_items"]; b2c=d["b2c"]
cre=lambda it: b2c.get(it,"unk_"+str(it))
w=pd.read_csv(os.path.join(SP,"warm_eval_sample.csv"),dtype=str); c=pd.read_csv(os.path.join(SP,"cold_eval_sample.csv"),dtype=str)
nW=int(round(N*WP)); nC=N-nW
wu=list(zip(w.user_id,w.gold))[:nW]; cu=list(zip(c.user_id,c.gold))[:nC]; wu_set=set(wu)
def sel_warm(wl,strat,n):
    if strat[0]=="base": return wl[:n]
    if strat[0]=="cap":
        K=strat[1]; ch=[]; cc=defaultdict(int)
        for it in wl:
            if cc[cre(it)]<K: ch.append(it); cc[cre(it)]+=1
            if len(ch)==n: break
        if len(ch)<n:
            for it in wl:
                if it not in ch: ch.append(it)
                if len(ch)==n: break
        return ch[:n]
    if strat[0]=="mmr":
        lam=strat[1]; rel={it:1.0/(r+1) for r,it in enumerate(wl)}; ch=[]; cc=defaultdict(int); rem=list(wl)
        while len(ch)<n and rem:
            best=max(rem,key=lambda it: rel[it]-lam*cc[cre(it)]); ch.append(best); cc[cre(best)]+=1; rem.remove(best)
        return ch
    if strat[0]=="capboost":   # cap2 + promote cold-lane creators' warm items earlier (cheap cold-creator boost)
        K=strat[1]; ch=[]; cc=defaultdict(int)
        wl2=sorted(wl,key=lambda it:(0 if cre(it) in strat[2] else 1))  # stable: cold-creators first, else ALS order
        for it in wl2:
            if cc[cre(it)]<K: ch.append(it); cc[cre(it)]+=1
            if len(ch)==n: break
        if len(ch)<n:
            for it in wl:
                if it not in ch: ch.append(it)
                if len(ch)==n: break
        return ch[:n]
STRATS={"als_only":("als",),"base_90_10":("base",),"cap3":("cap",3),"cap2":("cap",2),"cap1":("cap",1),
        "mmr_0.3":("mmr",0.3),"mmr_1.0":("mmr",1.0)}
# precompute the set of creators that own any cold item (for capboost)
cold_creators=set(cre(it) for it in cold_set)
STRATS["cap2_coldboost"]=("capboost",2,cold_creators)
STRATS["cap3_reserve1"]=("cap",3); STRATS["cap2_reserve1"]=("cap",2)
rep={}
for name,strat in STRATS.items():
    t0=time.time(); cre_exp=defaultdict(int); cre_exp_warm=defaultdict(int); seen_items=set(); seen_cre=set(); seen_cold_cre=set()
    wr20=wr50=0; cr20=cr50=0
    for (u,gold) in wu+cu:
        wl=WL.get(u,[])[:POOL]; cl=CL.get(u,[])
        if name=="als_only":
            s20=wl[:20]; s50=wl[:50]
        elif "reserve1" in name:                 # budget reallocation: 1 cold slot, 19 warm (capped)
            s20=sel_warm(wl,strat,19)+cl[:1]; s50=sel_warm(wl,strat,49)+cl[:1]
        else:
            s20=sel_warm(wl,strat,18)+cl[:2]; s50=sel_warm(wl,strat,48)+cl[:2]
        for it in s20:
            cr=cre(it); cre_exp[cr]+=1; seen_items.add(it); seen_cre.add(cr)
            if it in cold_set: seen_cold_cre.add(cr)
            else: cre_exp_warm[cr]+=1
        # recall (warm users -> warm gold, cold users -> cold gold)
        warm_user=(u,gold) in wu_set
        if warm_user:
            wr20+= 1 if gold in s20 else 0; wr50+= 1 if gold in s50 else 0
        else:
            cr20+= 1 if gold in s20 else 0; cr50+= 1 if gold in s50 else 0
    warmR20=wr20/nW; warmR50=wr50/nW; coldR20=cr20/nC; coldR50=cr50/nC
    rep[name]={"warm_R@20":round(warmR20,4),"warm_R@50":round(warmR50,4),
               "cold_R@20":round(coldR20,4),"cold_R@50":round(coldR50,4),
               "prev_R@20":round(WP*warmR20+CP*coldR20,4),
               "creator_gini":round(gini(np.array(list(cre_exp.values()),float)),4),
               "warm_creator_gini":round(gini(np.array(list(cre_exp_warm.values()),float)),4),
               "eff_creators":round(float(np.exp(-(lambda a:(a/a.sum()*np.log(a/a.sum())).sum())(np.array([v for v in cre_exp.values() if v>0],float)))),1),
               "distinct_items":len(seen_items),"distinct_creators":len(seen_cre),"distinct_cold_creators":len(seen_cold_cre),
               "ms_per_user":round(1000*(time.time()-t0)/N,3)}
als=rep["als_only"]; base=rep["base_90_10"]
def relloss(x): return round((x-als["warm_R@20"])/als["warm_R@20"],4)
for name in rep:
    rep[name]["warm_relloss_vs_als@20"]=relloss(rep[name]["warm_R@20"])
    rep[name]["creator_gini_delta_vs_als"]=round(rep[name]["creator_gini"]-als["creator_gini"],4)
    rep[name]["pass_warm_guardrail"]=bool(rep[name]["warm_relloss_vs_als@20"]>=-0.10)
out={"dataset":"goodreads_fantasy_paranormal","scope":"creator-aware reranking on two-lane policy","tag":"[BUILT][SYNTHETIC — offline reranking sim]",
 "config":{"n_users":N,"prevalence_warm":WP,"warm_pool":POOL,"cold_reserve":2,"guardrail":"warm R@20 loss vs ALS-only <= 10%"},
 "strategies":rep,
 "honest_notes":["Offline reranking simulation on held-out golds; NOT business lift; creator fairness NOT 'solved'.",
   "Reranking touches only the warm lane; the 2-slot cold reserve is preserved (cold recall unchanged across rerank strategies).",
   "Creator Gini is GLOBAL (cumulative exposure across all users); per-user creator caps reduce how often dominant (often series) creators repeat.",
   "Separate creator REACH (distinct creators) from creator DE-CONCENTRATION (Gini/effective creators)."],}
json.dump(out,open(os.path.join(EVID,"domain_creator_rerank_report.json"),"w"),indent=2)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    order=["als_only","base_90_10","cap3","cap2","cap2_coldboost","cap1","mmr_0.3","mmr_1.0"]
    gx=[rep[n]["creator_gini"] for n in order]; ry=[rep[n]["warm_relloss_vs_als@20"]*100 for n in order]
    fig,ax=plt.subplots(figsize=(9,6))
    for n in order:
        ok=rep[n]["pass_warm_guardrail"]; ax.scatter(rep[n]["creator_gini"],rep[n]["warm_relloss_vs_als@20"]*100,
            s=90,c=("#228833" if ok else "#cc3311"),edgecolor="k",zorder=3)
        ax.annotate(n,(rep[n]["creator_gini"],rep[n]["warm_relloss_vs_als@20"]*100),fontsize=8,xytext=(4,4),textcoords="offset points")
    ax.axhline(-10,ls="--",c="gray"); ax.text(min(gx),-10.3,"-10% warm guardrail",fontsize=8,color="gray")
    ax.axvline(als["creator_gini"],ls=":",c="gray"); ax.set_xlabel("creator exposure Gini (lower = healthier)"); ax.set_ylabel("warm R@20 loss vs ALS-only (%)")
    ax.set_title("H2: relevance vs creator-concentration tradeoff (green=passes guardrail)")
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS,"domain_creator_rerank_tradeoff.png"),dpi=110); plt.close()
except Exception as ex: print("plot skipped:",ex)
for n in ["als_only","base_90_10","cap3","cap2","cap2_coldboost","cap1","mmr_0.3","mmr_1.0","cap3_reserve1","cap2_reserve1"]:
    r=rep[n]; print("%-16s gini %.3f (d%+.3f) warmR20 %.4f (loss %+.1f%%) %s | distCre %d distColdCre %d effCre %.0f coldR20 %.4f %.2fms"%(
      n,r["creator_gini"],r["creator_gini_delta_vs_als"],r["warm_R@20"],100*r["warm_relloss_vs_als@20"],
      "PASS" if r["pass_warm_guardrail"] else "FAIL",r["distinct_creators"],r["distinct_cold_creators"],r["eff_creators"],r["cold_R@20"],r["ms_per_user"]))
print("warm-only creator Gini:",{n:rep[n]["warm_creator_gini"] for n in ["als_only","base_90_10","cap3","cap2","cap1","mmr_1.0"]})

