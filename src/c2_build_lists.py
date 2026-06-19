"""C2 step 2 — build per-eval-user candidate lists: ALS-warm top-200 + content-cold-hybrid top-200.
Saves c2_lists.pkl for the policy simulation."""
import pandas as pd, numpy as np, os, json, pickle, time
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
OUT="/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim"; SP=os.path.join(OUT,"domain_splits"); t0=time.time()
A=pickle.load(open(os.path.join(OUT,"c2_als.pkl"),"rb")); Mp=pickle.load(open(os.path.join(OUT,"c2_meta.pkl"),"rb"))
X=A["X"];Y=A["Y"];i_uniq=A["i_uniq"];iidx=A["iidx"];uidx=A["uidx"]; hist=Mp["eval_hist"]; pre_items=Mp["pre_items"]
d2=json.load(open(os.path.join(OUT,"d2_stats.json"))); t2=d2["split_global_time"]["cutoffs"]["t2"]
core=pd.read_csv(os.path.join(OUT,"domain_core_interactions.csv"),usecols=["user_id","book_id","event_ts"],dtype={"user_id":"str","book_id":"str","event_ts":"int64"})
cold_items=list(set(core[core.event_ts>=t2].book_id)-pre_items); del core
cm=pd.read_csv(os.path.join(OUT,"domain_content_meta.csv"),dtype=str).fillna("")
b2c=dict(zip(cm.book_id,cm.creator_id)); b2s=dict(zip(cm.book_id,cm.series_id)); b2y={b:(int(y) if y.isdigit() else 0) for b,y in zip(cm.book_id,cm.publication_year)}
text=dict(zip(cm.book_id,(cm.title+" "+cm.shelves+" "+cm.desc)))
wsamp=pd.read_csv(os.path.join(SP,"warm_eval_sample.csv"),dtype=str); csamp=pd.read_csv(os.path.join(SP,"cold_eval_sample.csv"),dtype=str)
users=list(dict.fromkeys(list(wsamp.user_id)+list(csamp.user_id)))
# WARM lists (ALS top-200, exclude history)
warm_list={}
rows=[uidx.get(u,-1) for u in users]; B=1000
for s in range(0,len(rows),B):
    rb=rows[s:s+B]; sc=X[[r if r>=0 else 0 for r in rb]]@Y.T
    for bi,r in enumerate(rb):
        u=users[s+bi]
        if r<0: warm_list[u]=[]; continue
        row=sc[bi].copy()
        for b in hist.get(u,()):
            j=iidx.get(b)
            if j is not None: row[j]=-1e9
        top=np.argpartition(-row,200)[:200]; top=top[np.argsort(-row[top])]; warm_list[u]=[i_uniq[t] for t in top]
warm_sec=round(time.time()-t0,1)
# COLD content-hybrid lists
a2cold=defaultdict(list); s2cold=defaultdict(list)
for it in cold_items:
    a=b2c.get(it,""); s=b2s.get(it,"")
    if a: a2cold[a].append(it)
    if s: s2cold[s].append(it)
for a in a2cold: a2cold[a].sort(key=lambda x:-b2y.get(x,0))
for s in s2cold: s2cold[s].sort(key=lambda x:-b2y.get(x,0))
hist_items=set(i for h in hist.values() for i in h); docs=list(set(cold_items)|hist_items); ridx={it:i for i,it in enumerate(docs)}
vec=TfidfVectorizer(max_features=20000,stop_words="english"); Mtx=vec.fit_transform([text.get(it,"") for it in docs])
cold_mat=Mtx[np.array([ridx[it] for it in cold_items])]
cold_list={}
for u in users:
    h=hist.get(u,set()); r=[ridx[i] for i in h if i in ridx]
    prof=sp.csr_matrix(Mtx[r].mean(axis=0)) if r else sp.csr_matrix((1,Mtx.shape[1]))
    sc=np.asarray(prof.dot(cold_mat.T).todense()).ravel(); order=np.argsort(-sc); tf=[cold_items[j] for j in order[:200]]
    auth=set(b2c.get(b,"") for b in h)-{""}; ser=set(b2s.get(b,"") for b in h)-{""}
    sa=[]; seen=set()
    for a in auth:
        for it in a2cold.get(a,()):
            if it not in seen: sa.append(it); seen.add(it)
    ss=[]; seen2=set()
    for s in ser:
        for it in s2cold.get(s,()):
            if it not in seen2: ss.append(it); seen2.add(it)
    rr=defaultdict(float)
    for lst in (sa,ss,tf):
        for rk,it in enumerate(lst): rr[it]+=1.0/(60+rk)
    cold_list[u]=[i for i,_ in sorted(rr.items(),key=lambda t:-t[1])][:200]
pickle.dump({"warm_list":warm_list,"cold_list":cold_list,"cold_items":set(cold_items),
             "b2c":{it:b2c.get(it,"") for it in set(cold_items)|pre_items}},open(os.path.join(OUT,"c2_lists.pkl"),"wb"))
print("C2_LISTS users=%d warm_sec=%.1f total=%.1f cold_items=%d"%(len(users),warm_sec,time.time()-t0,len(cold_items)))
