"""D3A step 3 — ALS (implicit, constant confidence since all (user,book) pairs are unique)
on the domain core; evaluate on the 5,000-user leave-last-out sample (batched scoring).
Bounded iters for the CPU sandbox. Domain-scoped."""
import pandas as pd, numpy as np, json, os, time
OUT="/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim"; SP=os.path.join(OUT,"domain_splits")
KS=[5,10,20]; F=16; ALPHA=40.0; REG=0.1; ITERS=4; SEED=20260616; t0=time.time()
rng=np.random.default_rng(SEED); c=1.0+ALPHA

core=pd.read_csv(os.path.join(OUT,"domain_core_interactions.csv"),
    usecols=["user_id","book_id"],dtype={"user_id":"str","book_id":"str"})
held=pd.read_csv(os.path.join(SP,"heldout_eval.csv"),dtype={"user_id":"str","gold":"str"})
heldset=set(held.user_id+"|"+held.gold)
tr=core[~(core.user_id+"|"+core.book_id).isin(heldset)]; del core
u_codes,u_uniq=pd.factorize(tr.user_id); i_codes,i_uniq=pd.factorize(tr.book_id)
del tr
nU=len(u_uniq); nI=len(i_uniq)
uidx={u:i for i,u in enumerate(u_uniq)}; iidx={it:i for i,it in enumerate(i_uniq)}
# group items by user and users by item (sorted-split)
o=np.argsort(u_codes,kind="stable"); us=u_codes[o]; itu=i_codes[o]
user_items=np.split(itu,np.flatnonzero(np.diff(us))+1)
o2=np.argsort(i_codes,kind="stable"); is_=i_codes[o2]; uti=u_codes[o2]
item_users=np.split(uti,np.flatnonzero(np.diff(is_))+1)
X=0.01*rng.standard_normal((nU,F)); Y=0.01*rng.standard_normal((nI,F)); I=REG*np.eye(F)
def solve(Femb,FtF,idxs):
    Fi=Femb[idxs]
    A=FtF+(c-1.0)*(Fi.T@Fi)+I
    b=c*Fi.sum(0)
    return np.linalg.solve(A,b)
for _ in range(ITERS):
    YtY=Y.T@Y
    for uc in range(nU): X[uc]=solve(Y,YtY,user_items[uc])
    XtX=X.T@X
    for ic in range(nI): Y[ic]=solve(X,XtX,item_users[ic])
log_fit=round(time.time()-t0,1)
# evaluate on sample (batched)
samp=pd.read_csv(os.path.join(SP,"eval_sample.csv"),dtype={"user_id":"str","gold":"str"})
su=[uidx.get(u,-1) for u in samp.user_id]; sg=[iidx.get(g,-1) for g in samp.gold]
res={"als":{k:{"r":[],"n":[]} for k in KS}}
import math
def recall_ndcg(ranked,gold,k):
    if gold in ranked[:k]:
        return 1.0, 1.0/math.log2(ranked.index(gold)+2)
    return 0.0,0.0
B=1000
for s in range(0,len(su),B):
    rows=su[s:s+B]; golds=sg[s:s+B]
    valid=[r for r in rows if r>=0]
    sc=X[[r if r>=0 else 0 for r in rows]] @ Y.T   # (b, nI)
    for bi,(uc,gc) in enumerate(zip(rows,golds)):
        if uc<0 or gc<0:
            for k in KS: res["als"][k]["r"].append(0.0); res["als"][k]["n"].append(0.0)
            continue
        row=sc[bi].copy(); row[user_items[uc]]=-1e9          # exclude history
        top=np.argpartition(-row,max(KS))[:max(KS)]; top=top[np.argsort(-row[top])]
        ranked=list(top)
        for k in KS:
            r,n=recall_ndcg(ranked,gc,k); res["als"][k]["r"].append(r); res["als"][k]["n"].append(n)
json.dump({"res":res,"fit_sec":log_fit,"iters":ITERS,"factors":F,"n_eval":int(len(samp)),
           "sec":round(time.time()-t0,1)}, open(os.path.join(OUT,"d3a_als_metrics.json"),"w"))
print("ALS_DONE fit=%.1fs total=%.1fs | ALS R@10=%.4f R@20=%.4f"%(
 log_fit,time.time()-t0,np.mean(res["als"][10]["r"]),np.mean(res["als"][20]["r"])))
