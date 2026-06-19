"""D3B floor — refit ALS f64 i4; full metrics (R@5..200, NDCG@10/20/50, bootstrap CIs) on the
5,000-user leave-last-out sample. Saves domain_als_f64_floor.json (primary D3B comparison floor)."""
import pandas as pd, numpy as np, os, json, time, sys
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from pulsediscover import eval as E
OUT="/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim"; SP=os.path.join(OUT,"domain_splits")
EVID="/sessions/festive-quirky-cannon/mnt/pulsediscover/outputs/evidence"
F=64; ITERS=4; ALPHA=40.0; REG=0.1; SEED=20260616; KS=[5,10,20,50,100,200]; NK=[10,20,50]; t0=time.time()
rng=np.random.default_rng(SEED); c=1.0+ALPHA
core=pd.read_csv(os.path.join(OUT,"domain_core_interactions.csv"),usecols=["user_id","book_id"],dtype=str)
held=pd.read_csv(os.path.join(SP,"heldout_eval.csv"),dtype=str)
tr=core[~(core.user_id+"|"+core.book_id).isin(set(held.user_id+"|"+held.gold))][["user_id","book_id"]]; del core
u_codes,u_uniq=pd.factorize(tr.user_id); i_codes,i_uniq=pd.factorize(tr.book_id); del tr
nU=len(u_uniq); nI=len(i_uniq)
iidx={it:i for i,it in enumerate(i_uniq)}; uidx={u:i for i,u in enumerate(u_uniq)}
o=np.argsort(u_codes,kind="stable"); user_items=np.split(i_codes[o],np.flatnonzero(np.diff(u_codes[o]))+1)
o2=np.argsort(i_codes,kind="stable"); item_users=np.split(u_codes[o2],np.flatnonzero(np.diff(i_codes[o2]))+1)
X=0.01*rng.standard_normal((nU,F)); Y=0.01*rng.standard_normal((nI,F)); I=REG*np.eye(F)
def solve(Fe,FtF,idx):
    Fi=Fe[idx]; return np.linalg.solve(FtF+(c-1.0)*(Fi.T@Fi)+I,c*Fi.sum(0))
for _ in range(ITERS):
    YtY=Y.T@Y
    for uc in range(nU): X[uc]=solve(Y,YtY,user_items[uc])
    XtX=X.T@X
    for ic in range(nI): Y[ic]=solve(X,XtX,item_users[ic])
fit_sec=round(time.time()-t0,1)
hist={u: user_items[uidx[u]] for u in u_uniq}   # code arrays
samp=pd.read_csv(os.path.join(SP,"eval_sample.csv"),dtype=str)
rows=[uidx.get(u,-1) for u in samp.user_id]; golds=samp.gold.tolist()
R={k:[] for k in KS}; Nm={k:[] for k in NK}; B=1000
for s in range(0,len(rows),B):
    rb=rows[s:s+B]; sc=X[[r if r>=0 else 0 for r in rb]]@Y.T
    for bi,r in enumerate(rb):
        u=samp.user_id.iloc[s+bi]; g=golds[s+bi]
        if r<0:
            for k in KS: R[k].append(0.0)
            for k in NK: Nm[k].append(0.0)
            continue
        row=sc[bi].copy(); row[hist[u]]=-1e9
        top=np.argpartition(-row,200)[:200]; top=top[np.argsort(-row[top])]; rk=[i_uniq[t] for t in top]
        for k in KS: R[k].append(E.recall_at_k(rk,g,k))
        for k in NK: Nm[k].append(E.ndcg_at_k(rk,g,k))
agg={}
for k in KS:
    rm,lo,hi,n=E.bootstrap_ci(R[k],seed=SEED); agg[f"R@{k}"]={"mean":round(rm,4),"ci95":[round(lo,4),round(hi,4)]}
for k in NK: agg[f"N@{k}"]={"mean":round(float(np.mean(Nm[k])),4)}
out={"method":"ALS","config":{"factors":F,"iters":ITERS,"alpha":ALPHA,"reg":REG},
     "n_eval":len(rows),"fit_sec":fit_sec,"metrics":agg,"sec":round(time.time()-t0,1)}
json.dump(out,open(os.path.join(EVID,"domain_als_f64_floor.json"),"w"),indent=2)
print("ALS_F64_FLOOR fit=%.1fs |"%fit_sec,{f"R@{k}":agg[f"R@{k}"]["mean"] for k in [10,20,50,100,200]},"N@10",agg["N@10"]["mean"])
