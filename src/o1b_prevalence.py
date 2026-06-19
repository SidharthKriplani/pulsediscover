"""O1b — prevalence-correct two-lane OPE. Cold-support logger (O1 showed ALS-heavy has weak cold
support). Re-weights value to realistic warm=76.5% / cold=23.5% (D2). Reports GT/IPS/SNIPS/DM/DR
(50/50 vs prevalence), bias, stratified bootstrap CIs, ESS/weights, ranking + C2 guardrails."""
import pandas as pd, numpy as np, os, json, pickle, time
from collections import defaultdict
OUT="/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim"; SP=os.path.join(OUT,"domain_splits")
EVID="/sessions/festive-quirky-cannon/mnt/pulsediscover/outputs/evidence"; PLOTS="/sessions/festive-quirky-cannon/mnt/pulsediscover/outputs/plots"
L=20; EPSC=0.35; Mc=10; Mw=20; TAU=5.0; SEED=20260616; B=300; WP=0.765; CP=0.235; t0=time.time(); rng=np.random.default_rng(SEED)
exam=0.95*(1.0/(np.arange(L)+1.0))**0.7
lists=pickle.load(open(os.path.join(OUT,"c2_lists.pkl"),"rb")); WL=lists["warm_list"]; CL=lists["cold_list"]; cold_set=lists["cold_items"]
w=pd.read_csv(os.path.join(SP,"warm_eval_sample.csv"),dtype=str); c=pd.read_csv(os.path.join(SP,"cold_eval_sample.csv"),dtype=str)
nW=len(w); nC=len(c); inst=[(u,g) for u,g in zip(w.user_id,w.gold)]+[(u,g) for u,g in zip(c.user_id,c.gold)]
def rrf_list(u):
    rr=defaultdict(float)
    for r,it in enumerate(WL.get(u,[])): rr[it]+=1.0/(60+r)
    for r,it in enumerate(CL.get(u,[])): rr[it]+=1.0/(60+r)
    return [i for i,_ in sorted(rr.items(),key=lambda t:-t[1])]
def lane(it): return 1 if it in cold_set else 0
def slate(u,name):
    wl=WL.get(u,[]); cl=CL.get(u,[])
    s={"als_only":wl[:L],"q90_10":wl[:18]+cl[:2],"q80_20":wl[:16]+cl[:4],"rrf":rrf_list(u)[:L]}[name]
    return (s+[None]*L)[:L]
POL=["als_only","q90_10","q80_20","rrf"]
LOG=[]
for u,gold in inst:
    wl=WL.get(u,[])[:Mw] or [None]; cl=CL.get(u,[])[:Mc]
    qw=np.exp(-np.arange(len(wl))/TAU); qw=qw/qw.sum()
    p=np.concatenate([(1-EPSC)*qw,(EPSC/len(cl))*np.ones(len(cl))]) if cl else qw; p=p/p.sum()
    pool=list(wl)+list(cl); lanes=np.array([0]*len(wl)+[1]*len(cl)); idx=rng.choice(len(pool),size=L,p=p)
    items=[pool[j] for j in idx]; mus=p[idx]; lns=lanes[idx]; ex=rng.random(L)<exam
    rew=np.array([1 if (ex[r] and items[r]==gold) else 0 for r in range(L)])
    LOG.append((u,gold,items,np.array(mus),lns,rew))
sums=np.zeros((2,L)); cnts=np.zeros((2,L))
for _,_,_,_,lns,rew in LOG:
    for r in range(L): sums[lns[r],r]+=rew[r]; cnts[lns[r],r]+=1
qhat=sums/np.maximum(cnts,1)
per={p:{"ips":[],"dr":[],"gt":[],"dm":[],"snn_w":0.,"snd_w":0.,"snn_c":0.,"snd_c":0.,"w":[]} for p in POL}
for i,(u,gold,items,mus,lns,rew) in enumerate(LOG):
    warm_stratum = i<nW
    for p in POL:
        s=slate(u,p); ips=dr=gt=dmv=0.0; ws=0.0
        for r in range(L):
            pit=s[r]; mw=(1.0/mus[r]) if items[r]==pit else 0.0; ips+=mw*rew[r]; ws+=mw
            dm=qhat[lane(pit) if pit is not None else 0,r]; dr+=dm+mw*(rew[r]-qhat[lns[r],r]); dmv+=dm
            gt+=exam[r]*(1.0 if pit==gold else 0.0)
            if mw>0: per[p]["w"].append(mw)
        per[p]["ips"].append(ips/L); per[p]["dr"].append(dr/L); per[p]["gt"].append(gt/L); per[p]["dm"].append(dmv/L)
        if warm_stratum: per[p]["snn_w"]+=ips; per[p]["snd_w"]+=ws
        else: per[p]["snn_c"]+=ips; per[p]["snd_c"]+=ws
def pw(arr): a=np.array(arr); return WP*a[:nW].mean()+CP*a[nW:].mean()
def f50(arr): return float(np.mean(arr))
def sboot(arr):
    a=np.array(arr); out=[]
    for _ in range(B):
        wi=rng.integers(0,nW,nW); ci=rng.integers(nW,nW+nC,nC)
        out.append(WP*a[wi].mean()+CP*a[ci].mean())
    s=np.array(out); return [round(float(np.quantile(s,.025)),6),round(float(np.quantile(s,.975)),6)],round(float(s.std()),6)
rep={}
for p in POL:
    gt_pw=pw(per[p]["gt"]); dr_pw=pw(per[p]["dr"]); ips_pw=pw(per[p]["ips"]); dm_pw=pw(per[p]["dm"])
    snips_pw=WP*(per[p]["snn_w"]/per[p]["snd_w"] if per[p]["snd_w"]>0 else 0)+CP*(per[p]["snn_c"]/per[p]["snd_c"] if per[p]["snd_c"]>0 else 0)
    ws=np.array(per[p]["w"]) if per[p]["w"] else np.array([0.])
    ess=(ws.sum()**2)/np.sum(ws**2) if np.sum(ws**2)>0 else 0
    drci,drsd=sboot(per[p]["dr"]); ipsci,_=sboot(per[p]["ips"])
    rep[p]={"GT_prevalence":round(gt_pw,5),"GT_50_50":round(f50(per[p]["gt"]),5),
            "IPS":round(ips_pw,5),"SNIPS":round(float(snips_pw),5),"DM":round(dm_pw,5),"DR":round(dr_pw,5),
            "bias_DR":round(dr_pw-gt_pw,5),"bias_IPS":round(ips_pw-gt_pw,5),
            "ESS":round(float(ess),1),"w_max":round(float(ws.max()),2),"w_mean":round(float(ws.mean()),3),
            "DR_ci95":drci,"DR_std":drsd,"IPS_ci95":ipsci}
gt_rank=sorted(POL,key=lambda p:-rep[p]["GT_prevalence"]); dr_rank=sorted(POL,key=lambda p:-rep[p]["DR"])
# C2 guardrails
c2=json.load(open(os.path.join(EVID,"domain_two_lane_policy_report.json")))["policies"]
namemap={"als_only":"als_only","q90_10":"q90_10","q80_20":"q80_20","rrf":"rrf_blend"}
guard={p:{"warm_R@20":c2[namemap[p]]["warm_recall"]["R@20"],"cold_R@20":c2[namemap[p]]["cold_recall"]["R@20"],
          "warm_rel_loss@20":c2[namemap[p]].get("warm_recall_rel_loss_vs_alsonly@20"),
          "distinct_cold_top20":c2[namemap[p]]["distinct_cold_surfaced_top20"],
          "creator_gini_top20":c2[namemap[p]]["creator_gini_top20"]} for p in POL}
out={"dataset":"goodreads_fantasy_paranormal","scope":"prevalence-correct two-lane OPE","tag":"[BUILT][SYNTHETIC — known-propensity OPE]",
 "prevalence":{"warm":WP,"cold":CP},"logger":"cold-support (eps=%.2f,Mc=%d,Mw=%d)"%(EPSC,Mc,Mw),
 "n_warm":nW,"n_cold":nC,"policies":rep,"GT_ranking_prevalence":gt_rank,"DR_ranking_prevalence":dr_rank,
 "ranking_match":gt_rank==dr_rank,
 "dr_separates_90_10_from_als":bool(abs(rep["q90_10"]["DR"]-rep["als_only"]["DR"])>(rep["q90_10"]["DR_std"]+rep["als_only"]["DR_std"])),
 "c2_guardrails":guard,
 "honest_notes":["Prevalence-weighted (warm 76.5% / cold 23.5%, from D2). Offline proxy reward; NOT business lift.",
   "Cold-support logger used (O1: ALS-heavy logger had weak cold positivity).",
   "Stratified bootstrap (resample within warm & cold strata)."],
 "sec":round(time.time()-t0,1)}
json.dump(out,open(os.path.join(EVID,"domain_two_lane_ope_prevalence_report.json"),"w"),indent=2)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    x=np.arange(len(POL)); wd=0.38; fig,ax=plt.subplots(figsize=(8,5))
    ax.bar(x-wd/2,[rep[p]["GT_50_50"] for p in POL],wd,label="GT (50/50)",color="#cc6677")
    ax.bar(x+wd/2,[rep[p]["GT_prevalence"] for p in POL],wd,label="GT (76.5/23.5 prevalence)",color="#228833")
    ax.set_xticks(x); ax.set_xticklabels(POL); ax.set_ylabel("policy value (per-slot)")
    ax.set_title("Two-lane policy value: 50/50 vs realistic prevalence"); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS,"domain_two_lane_prevalence.png"),dpi=110); plt.close()
except Exception as ex: print("plot skipped:",ex)
for p in POL: print(p,"GT_prev",rep[p]["GT_prevalence"],"GT_50",rep[p]["GT_50_50"],"DR",rep[p]["DR"],"DRci",rep[p]["DR_ci95"],"ESS",rep[p]["ESS"])
print("GT_prev rank",gt_rank,"| DR rank",dr_rank,"| DR separates 90/10 vs als:",out["dr_separates_90_10_from_als"],"| %.1fs"%(time.time()-t0))
