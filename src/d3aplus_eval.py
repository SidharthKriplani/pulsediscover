"""D3A+ step 2 — evaluate popularity / co-occurrence / ALS / same-author / same-series / hybrid
on the 5,000-user domain leave-last-out sample. Richer metrics (R@5..200, NDCG@10/20/50),
segment (warm/new), bootstrap CIs. Writes domain_baselines_plus_report.json + plot."""
import pandas as pd, numpy as np, os, json, pickle, time, math, sys
from collections import defaultdict
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from pulsediscover import eval as E
OUT="/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim"; SP=os.path.join(OUT,"domain_splits")
PLOTS="/sessions/festive-quirky-cannon/mnt/pulsediscover/outputs/plots"; EVID="/sessions/festive-quirky-cannon/mnt/pulsediscover/outputs/evidence"
KS=[5,10,20,50,100,200]; NK=[10,20,50]; SEED=20260616; t0=time.time()
A=pickle.load(open(os.path.join(OUT,"d3aplus_als.pkl"),"rb")); Mp=pickle.load(open(os.path.join(OUT,"d3aplus_maps.pkl"),"rb"))
X=A["X"];Y=A["Y"];i_uniq=A["i_uniq"];uidx=A["uidx"];iidx=A["iidx"];pop200=A["pop_top200"];train_items=A["train_items"]
neigh=Mp["neigh"];a2b=Mp["a2b"];s2b=Mp["s2b"];hist=Mp["hist"];ev_auth=Mp["ev_auth"];ev_ser=Mp["ev_ser"]
b2c=Mp["b2c"];b2s=Mp["b2s"];tr_cre=Mp["train_creators"];tr_ser=Mp["train_series"]
samp=pd.read_csv(os.path.join(SP,"eval_sample.csv"),dtype={"user_id":"str","gold":"str"})
users=samp.user_id.tolist(); golds=samp.gold.tolist()
# precompute ALS top-200 per user (batched)
als_top={}
rows=[uidx.get(u,-1) for u in users]
B=1000
for s in range(0,len(rows),B):
    rb=rows[s:s+B]
    sc=X[[r if r>=0 else 0 for r in rb]]@Y.T
    for bi,r in enumerate(rb):
        u=users[s+bi]
        if r<0: als_top[u]=[]; continue
        row=sc[bi].copy()
        for b in hist.get(u,()):
            j=iidx.get(b);
            if j is not None: row[j]=-1e9
        top=np.argpartition(-row,200)[:200]; top=top[np.argsort(-row[top])]
        als_top[u]=[i_uniq[t] for t in top]
def cooc_list(h):
    sc=defaultdict(float)
    for it in h:
        for nb,w in neigh.get(it,()):
            if nb not in h: sc[nb]+=w
    return [i for i,_ in sorted(sc.items(),key=lambda t:-t[1])[:200]]
def cand_from(mapping,keys,h):
    seen=set(); out=[]
    for k in keys:
        for b in mapping.get(k,()):
            if b not in h and b not in seen: out.append(b); seen.add(b)
            if len(out)>=200: return out
    return out
METHODS=["popularity","cooccurrence","als","same_author","same_series","hybrid"]
res={m:{f"R@{k}":[] for k in KS} for m in METHODS}
for m in METHODS:
    for k in NK: res[m][f"N@{k}"]=[]
seg={"item":{"warm":0,"new":0},"creator":{"warm":0,"new":0},"series":{"warm":0,"new":0}}
seg_recall={"warm":[],"new":[]}  # hybrid R@20 by item warm/new
for u,g in zip(users,golds):
    h=hist.get(u,set())
    L={"popularity":[i for i in pop200 if i not in h][:200],
       "cooccurrence":cooc_list(h),"als":als_top.get(u,[]),
       "same_author":cand_from(a2b,ev_auth.get(u,()),h),
       "same_series":cand_from(s2b,ev_ser.get(u,()),h)}
    sc=defaultdict(float)
    for nm in ("cooccurrence","als","same_author","same_series"):
        for r,it in enumerate(L[nm]): sc[it]+=1.0/(60+r)
    hyb=[i for i,_ in sorted(sc.items(),key=lambda t:-t[1]) if i not in h]
    if len(hyb)<200:
        for i in L["popularity"]:
            if i not in hyb: hyb.append(i)
            if len(hyb)>=200: break
    L["hybrid"]=hyb[:200]
    for m in METHODS:
        for k in KS: res[m][f"R@{k}"].append(E.recall_at_k(L[m],g,k))
        for k in NK: res[m][f"N@{k}"].append(E.ndcg_at_k(L[m],g,k))
    wi=g in train_items; seg["item"]["warm" if wi else "new"]+=1
    seg["creator"]["warm" if b2c.get(g,"") in tr_cre else "new"]+=1
    seg["series"]["warm" if b2s.get(g,"") in tr_ser else "new"]+=1
    seg_recall["warm" if wi else "new"].append(E.recall_at_k(L["hybrid"],g,20))
agg={}
for m in METHODS:
    agg[m]={}
    for k in KS:
        rm,lo,hi,n=E.bootstrap_ci(res[m][f"R@{k}"],seed=SEED); agg[m][f"R@{k}"]={"mean":round(rm,4),"ci95":[round(lo,4),round(hi,4)]}
    for k in NK: agg[m][f"N@{k}"]={"mean":round(float(np.mean(res[m][f"N@{k}"])),4)}
best_single=max(["popularity","cooccurrence","als","same_author","same_series"],key=lambda m:agg[m]["R@20"]["mean"])
lift={f"R@{k}":round(agg["hybrid"][f"R@{k}"]["mean"]-agg[best_single][f"R@{k}"]["mean"],4) for k in [20,50,100,200]}
report={"dataset":"goodreads_fantasy_paranormal","scope":"DOMAIN","tag":"[BUILT — real data (domain)]",
        "eval":"leave-last-out, 5,000-user sample","n_eval":len(users),
        "metrics":agg,"best_single_baseline_R@20":best_single,"hybrid_marginal_lift_vs_best_single":lift,
        "segment_counts":seg,"hybrid_R@20_by_segment":{"warm":round(float(np.mean(seg_recall["warm"])),4),
            "new":round(float(np.mean(seg_recall["new"])),4) if seg_recall["new"] else None,
            "n_new":len(seg_recall["new"])},
        "configs":{"als":json.load(open(os.path.join(OUT,"d3aplus_als_meta.json"))),
                   "cooc":json.load(open(os.path.join(OUT,"d3aplus_maps_meta.json")))},
        "sec":round(time.time()-t0,1)}
json.dump(report,open(os.path.join(EVID,"domain_baselines_plus_report.json"),"w"),indent=2)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5))
    for m in METHODS:
        ax.plot(KS,[agg[m][f"R@{k}"]["mean"] for k in KS],marker="o",label=m)
    ax.set_xscale("log"); ax.set_xlabel("K (log)"); ax.set_ylabel("Recall@K")
    ax.set_title("D3A+ domain candidate generators — Goodreads fantasy/paranormal (5k-user LLO)"); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS,"domain_baselines_plus_recall.png"),dpi=110); plt.close()
except Exception as ex: print("plot skipped:",ex)
for m in METHODS: print(m,{f"R@{k}":agg[m][f"R@{k}"]["mean"] for k in [10,20,50,100,200]})
print("best_single@20:",best_single,"hybrid lift:",lift,"sec",round(time.time()-t0,1))
