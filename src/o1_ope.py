"""O1 — two-lane policy OPE / offline A/B readiness (parametric logger).
Usage: python o1_ope.py TAG EPSC Mc Mw
Known-propensity per-slot logging mu; IPS/SNIPS/DM/DR for als_only/90-10/80-20/rrf vs known truth.
Offline proxy reward = examined gold-hit. Accumulates results per logger config. No business lift."""
import pandas as pd, numpy as np, os, json, pickle, sys, time
from collections import defaultdict
OUT="/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim"; SP=os.path.join(OUT,"domain_splits")
EVID="/sessions/festive-quirky-cannon/mnt/pulsediscover/outputs/evidence"; PLOTS="/sessions/festive-quirky-cannon/mnt/pulsediscover/outputs/plots"
TAG=sys.argv[1] if len(sys.argv)>1 else "alsheavy"; EPSC=float(sys.argv[2]) if len(sys.argv)>2 else 0.15
Mc=int(sys.argv[3]) if len(sys.argv)>3 else 40; Mw=int(sys.argv[4]) if len(sys.argv)>4 else 40
L=20; TAU=5.0; SEED=20260616; B=300; t0=time.time(); rng=np.random.default_rng(SEED)
exam=0.95*(1.0/(np.arange(L)+1.0))**0.7
lists=pickle.load(open(os.path.join(OUT,"c2_lists.pkl"),"rb")); WL=lists["warm_list"]; CL=lists["cold_list"]; cold_set=lists["cold_items"]
w=pd.read_csv(os.path.join(SP,"warm_eval_sample.csv"),dtype=str); c=pd.read_csv(os.path.join(SP,"cold_eval_sample.csv"),dtype=str)
inst=[(u,g) for u,g in zip(w.user_id,w.gold)]+[(u,g) for u,g in zip(c.user_id,c.gold)]
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
# log mu
LOG=[]
for u,gold in inst:
    wl=WL.get(u,[])[:Mw] or [None]; cl=CL.get(u,[])[:Mc]
    qw=np.exp(-np.arange(len(wl))/TAU); qw=qw/qw.sum()
    if cl: p=np.concatenate([(1-EPSC)*qw,(EPSC/len(cl))*np.ones(len(cl))])
    else: p=qw
    p=p/p.sum(); pool=list(wl)+list(cl); lanes=np.array([0]*len(wl)+[1]*len(cl))
    idx=rng.choice(len(pool),size=L,p=p)
    items=[pool[j] for j in idx]; mus=p[idx]; lns=lanes[idx]
    ex=rng.random(L)<exam; rew=np.array([1 if (ex[r] and items[r]==gold) else 0 for r in range(L)])
    LOG.append((u,gold,items,np.array(mus),lns,rew))
# qhat[lane][rank]
sums=np.zeros((2,L)); cnts=np.zeros((2,L))
for _,_,_,_,lns,rew in LOG:
    for r in range(L): sums[lns[r],r]+=rew[r]; cnts[lns[r],r]+=1
qhat=sums/np.maximum(cnts,1)
per={p:{"ips":[],"dr":[],"gt":[],"dm":[],"snn":0.0,"snd":0.0,"w":[]} for p in POL}
for u,gold,items,mus,lns,rew in LOG:
    for p in POL:
        s=slate(u,p); ips=dr=gt=dmv=0.0; ws=0.0
        for r in range(L):
            pit=s[r]; mw=(1.0/mus[r]) if items[r]==pit else 0.0
            ips+=mw*rew[r]; ws+=mw
            dm=qhat[lane(pit) if pit is not None else 0,r]
            dr+=dm+mw*(rew[r]-qhat[lns[r],r]); dmv+=dm; gt+=exam[r]*(1.0 if pit==gold else 0.0)
            if mw>0: per[p]["w"].append(mw)
        per[p]["ips"].append(ips/L); per[p]["dr"].append(dr/L); per[p]["gt"].append(gt/L); per[p]["dm"].append(dmv/L)
        per[p]["snn"]+=ips; per[p]["snd"]+=ws
rep={}
for p in POL:
    ips=np.array(per[p]["ips"]); dr=np.array(per[p]["dr"]); gt=float(np.mean(per[p]["gt"]))
    sn=per[p]["snn"]/per[p]["snd"] if per[p]["snd"]>0 else 0.0
    ws=np.array(per[p]["w"]) if per[p]["w"] else np.array([0.0])
    ess=(ws.sum()**2)/np.sum(ws**2) if np.sum(ws**2)>0 else 0.0
    def boot(a): s=np.array([a[rng.integers(0,len(a),len(a))].mean() for _ in range(B)]); return [round(float(np.quantile(s,.025)),6),round(float(np.quantile(s,.975)),6)],round(float(s.std()),6)
    drci,drsd=boot(dr); ipsci,_=boot(ips)
    rep[p]={"GT":round(gt,5),"IPS":round(float(ips.mean()),5),"SNIPS":round(float(sn),5),
            "DM":round(float(np.mean(per[p]["dm"])),5),
            "DR":round(float(dr.mean()),5),"bias_IPS":round(float(ips.mean())-gt,5),"bias_SNIPS":round(sn-gt,5),
            "bias_DR":round(float(dr.mean())-gt,5),"ESS":round(float(ess),1),"matched_slots":int(len(ws)),
            "w_max":round(float(ws.max()),2),"w_mean":round(float(ws.mean()),3),
            "DR_ci95":drci,"DR_std":drsd,"IPS_ci95":ipsci}
gt_rank=sorted(POL,key=lambda p:-rep[p]["GT"]); dr_rank=sorted(POL,key=lambda p:-rep[p]["DR"])
res={"config":{"epsc":EPSC,"Mc":Mc,"Mw":Mw,"tau":TAU},"policies":rep,
     "GT_ranking":gt_rank,"DR_ranking":dr_rank,"ranking_match":gt_rank==dr_rank,
     "dr_identifies_90_10_over_als":bool(rep["q90_10"]["DR"]>rep["als_only"]["DR"] and rep["q90_10"]["GT"]>rep["als_only"]["GT"])}
p_=os.path.join(EVID,"domain_two_lane_ope_report.json")
allr=json.load(open(p_)) if os.path.exists(p_) else {}
allr.setdefault("loggers",{})[TAG]=res
allr["meta"]={"dataset":"goodreads_fantasy_paranormal","scope":"two-lane OPE / offline A/B readiness",
 "tag":"[BUILT][SYNTHETIC — known-propensity OPE]","reward":"examined(PBM) AND item==user gold (warm/cold); offline proxy, NOT business lift",
 "n_instances":len(inst),"n_logged_slots":len(inst)*L,"eval_population":"5k warm + 5k cold (50/50)"}
json.dump(allr,open(p_,"w"),indent=2)
print(f"[{TAG} eps={EPSC} Mc={Mc} Mw={Mw}]")
for p in POL: print(" ",p,"GT",rep[p]["GT"],"IPS",rep[p]["IPS"],"SNIPS",rep[p]["SNIPS"],"DR",rep[p]["DR"],"ESS",rep[p]["ESS"],"wmax",rep[p]["w_max"])
print("  GT rank",gt_rank,"| DR rank",dr_rank,"| match",gt_rank==dr_rank,"| DR(90/10>ALS):",res["dr_identifies_90_10_over_als"],"| %.1fs"%(time.time()-t0))
