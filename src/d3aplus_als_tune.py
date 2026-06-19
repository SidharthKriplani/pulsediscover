"""D3A+ ALS tuning. Usage: python d3aplus_als_tune.py F ITERS
Fits ALS(F,ITERS) on domain eval-train, evals R@10/20/50 on the 5,000-user sample.
Appends to d3aplus_als_tune.json. f32/f64 + i12 exceed the CPU cap -> external pack."""
import pandas as pd, numpy as np, os, json, sys, time, pickle
OUT="/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim"; SP=os.path.join(OUT,"domain_splits")
F=int(sys.argv[1]); ITERS=int(sys.argv[2]); ALPHA=40.0; REG=0.1; SEED=20260616; t0=time.time()
rng=np.random.default_rng(SEED); c=1.0+ALPHA
core=pd.read_csv(os.path.join(OUT,"domain_core_interactions.csv"),usecols=["user_id","book_id"],dtype=str)
held=pd.read_csv(os.path.join(SP,"heldout_eval.csv"),dtype=str)
tr=core[~(core.user_id+"|"+core.book_id).isin(set(held.user_id+"|"+held.gold))][["user_id","book_id"]]; del core
u_codes,u_uniq=pd.factorize(tr.user_id); i_codes,i_uniq=pd.factorize(tr.book_id); del tr
nU=len(u_uniq); nI=len(i_uniq)
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
uidx={u:i for i,u in enumerate(u_uniq)}; iidx={it:i for i,it in enumerate(i_uniq)}
hist=pickle.load(open(os.path.join(OUT,"d3aplus_maps.pkl"),"rb"))["hist"]
samp=pd.read_csv(os.path.join(SP,"eval_sample.csv"),dtype=str)
R={10:[],20:[],50:[]}; rows=[uidx.get(u,-1) for u in samp.user_id]; golds=samp.gold.tolist()
B=1000
for s in range(0,len(rows),B):
    rb=rows[s:s+B]; sc=X[[r if r>=0 else 0 for r in rb]]@Y.T
    for bi,r in enumerate(rb):
        u=samp.user_id.iloc[s+bi]; g=golds[s+bi]
        if r<0:
            for k in R: R[k].append(0.0)
            continue
        row=sc[bi].copy()
        for b in hist.get(u,()):
            j=iidx.get(b)
            if j is not None: row[j]=-1e9
        top=np.argpartition(-row,50)[:50]; top=top[np.argsort(-row[top])]; rk=[i_uniq[t] for t in top]
        for k in R: R[k].append(1.0 if g in rk[:k] else 0.0)
out={"F":F,"iters":ITERS,"fit_sec":fit_sec,"R@10":round(float(np.mean(R[10])),4),
     "R@20":round(float(np.mean(R[20])),4),"R@50":round(float(np.mean(R[50])),4),"total_sec":round(time.time()-t0,1)}
p=os.path.join(OUT,"d3aplus_als_tune.json"); d=json.load(open(p)) if os.path.exists(p) else {}
d[f"f{F}_i{ITERS}"]=out; json.dump(d,open(p,"w"),indent=2)
print("ALS_TUNE",out)
