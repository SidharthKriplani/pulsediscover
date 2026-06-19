"""C2 step 1 — fit ALS f64 on PRE-TEST (ts<t2) so cold items stay unseen; build warm eval set
(test interactions on warm items) + pre-test histories for warm+cold eval users. Saves pkls."""
import pandas as pd, numpy as np, os, json, pickle, time
OUT="/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim"; SP=os.path.join(OUT,"domain_splits")
F=64; ITERS=4; ALPHA=40.0; REG=0.1; SEED=20260616; SAMPLE=5000; t0=time.time()
rng=np.random.default_rng(SEED); c=1.0+ALPHA
d2=json.load(open(os.path.join(OUT,"d2_stats.json"))); t2=d2["split_global_time"]["cutoffs"]["t2"]
core=pd.read_csv(os.path.join(OUT,"domain_core_interactions.csv"),usecols=["user_id","book_id","event_ts"],
                 dtype={"user_id":"str","book_id":"str","event_ts":"int64"})
pre=core[core.event_ts<t2]; test=core[core.event_ts>=t2]
pre_items=set(pre.book_id)
# ALS f64 on pre-test
u_codes,u_uniq=pd.factorize(pre.user_id); i_codes,i_uniq=pd.factorize(pre.book_id)
nU=len(u_uniq); nI=len(i_uniq); uidx={u:i for i,u in enumerate(u_uniq)}; iidx={it:i for i,it in enumerate(i_uniq)}
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
# histories (pre-test) by user
hist=pre.groupby("user_id")["book_id"].apply(set).to_dict()
# warm eval: test interactions on warm items, user has pre history
warm=test[test.book_id.isin(pre_items)]; warm=warm[warm.user_id.isin(hist.keys())]
wsamp=warm.sample(min(SAMPLE,len(warm)),random_state=SEED)[["user_id","book_id"]].rename(columns={"book_id":"gold"})
wsamp.to_csv(os.path.join(SP,"warm_eval_sample.csv"),index=False)
cold=pd.read_csv(os.path.join(SP,"cold_eval_sample.csv"),dtype=str)
eval_users=set(wsamp.user_id)|set(cold.user_id)
eval_hist={u:hist.get(u,set()) for u in eval_users}
pickle.dump({"X":X.astype(np.float32),"Y":Y.astype(np.float32),"i_uniq":np.array(i_uniq),"iidx":iidx,"uidx":uidx},
            open(os.path.join(OUT,"c2_als.pkl"),"wb"))
pickle.dump({"pre_items":pre_items,"eval_hist":eval_hist},open(os.path.join(OUT,"c2_meta.pkl"),"wb"))
json.dump({"F":F,"iters":ITERS,"fit_sec":fit_sec,"nU":nU,"nI":nI,"n_pre_items":len(pre_items),
           "warm_eval":int(len(wsamp)),"n_warm_test":int(len(warm)),"sec":round(time.time()-t0,1)},
          open(os.path.join(OUT,"c2_setup_meta.json"),"w"),indent=2)
print("C2_SETUP nU=%d nI=%d warm_eval=%d fit=%.1fs total=%.1fs"%(nU,nI,len(wsamp),fit_sec,time.time()-t0))
