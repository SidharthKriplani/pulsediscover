"""D3A+ step 1a — ALS(f16,i4) fit on domain eval-train; save als.pkl for the eval step."""
import pandas as pd, numpy as np, os, time, pickle, json
OUT="/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim"; SP=os.path.join(OUT,"domain_splits")
F=16; ALPHA=40.0; REG=0.1; ITERS=4; SEED=20260616; t0=time.time()
rng=np.random.default_rng(SEED); c=1.0+ALPHA
core=pd.read_csv(os.path.join(OUT,"domain_core_interactions.csv"),usecols=["user_id","book_id"],
                 dtype={"user_id":"str","book_id":"str"})
held=pd.read_csv(os.path.join(SP,"heldout_eval.csv"),dtype={"user_id":"str","gold":"str"})
tr=core[~(core.user_id+"|"+core.book_id).isin(set(held.user_id+"|"+held.gold))][["user_id","book_id"]]; del core
popser=tr.book_id.value_counts()
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
art={"X":X.astype(np.float32),"Y":Y.astype(np.float32),"i_uniq":np.array(i_uniq),
     "uidx":{u:i for i,u in enumerate(u_uniq)},"iidx":{it:i for i,it in enumerate(i_uniq)},
     "item_pop":popser.to_dict(),"pop_top200":list(popser.head(200).index),
     "train_items":set(popser.index)}
pickle.dump(art,open(os.path.join(OUT,"d3aplus_als.pkl"),"wb"))
json.dump({"als_f":F,"als_iters":ITERS,"nU":nU,"nI":nI,"sec":round(time.time()-t0,1)},
          open(os.path.join(OUT,"d3aplus_als_meta.json"),"w"),indent=2)
print("ALSFIT_DONE nU=%d nI=%d sec=%.1f"%(nU,nI,time.time()-t0))
