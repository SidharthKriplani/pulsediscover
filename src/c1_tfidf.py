"""C1 step 3 — TF-IDF content similarity cold-start generator + content-hybrid (RRF of
same-author/same-series/tfidf). Cold pool = new-in-test items. Evals cold-start Recall@K.
Saves c1_tfidf_partial.json."""
import pandas as pd, numpy as np, os, json, pickle, time
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
OUT="/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim"; SP=os.path.join(OUT,"domain_splits")
KS=[20,50,100,200]; t0=time.time()
d2=json.load(open(os.path.join(OUT,"d2_stats.json"))); t2=d2["split_global_time"]["cutoffs"]["t2"]
core=pd.read_csv(os.path.join(OUT,"domain_core_interactions.csv"),usecols=["user_id","book_id","event_ts"],dtype={"user_id":"str","book_id":"str","event_ts":"int64"})
pre_items=set(core[core.event_ts<t2].book_id); cold_items=list(set(core[core.event_ts>=t2].book_id)-pre_items); del core
cm=pd.read_csv(os.path.join(OUT,"domain_content_meta.csv"),dtype=str).fillna("")
b2c=dict(zip(cm.book_id,cm.creator_id)); b2s=dict(zip(cm.book_id,cm.series_id)); b2y={b:(int(y) if y.isdigit() else 0) for b,y in zip(cm.book_id,cm.publication_year)}
text=dict(zip(cm.book_id,(cm.title+" "+cm.shelves+" "+cm.desc)))
samp=pd.read_csv(os.path.join(SP,"cold_eval_sample.csv"),dtype=str)
hist=pickle.load(open(os.path.join(OUT,"c1_eval_hist.pkl"),"rb"))
hist_items=set(i for h in hist.values() for i in h)
docs_items=list(set(cold_items)|hist_items)
ridx={it:i for i,it in enumerate(docs_items)}
vec=TfidfVectorizer(max_features=20000,stop_words="english")
Mtx=vec.fit_transform([text.get(it,"") for it in docs_items])     # (docs x V)
cold_rows=np.array([ridx[it] for it in cold_items]); cold_mat=Mtx[cold_rows]   # (Ncold x V)
# same-author / same-series cold candidate maps
a2cold=defaultdict(list); s2cold=defaultdict(list)
for it in cold_items:
    a=b2c.get(it,""); s=b2s.get(it,"")
    if a: a2cold[a].append(it)
    if s: s2cold[s].append(it)
for a in a2cold: a2cold[a].sort(key=lambda x:-b2y.get(x,0))
for s in s2cold: s2cold[s].sort(key=lambda x:-b2y.get(x,0))
res={"tfidf":{k:[] for k in KS},"content_hybrid":{k:[] for k in KS}}
surfaced=set()
for u,g in zip(samp.user_id,samp.gold):
    h=hist[u]; rows=[ridx[i] for i in h if i in ridx]
    if rows: prof=Mtx[rows].mean(axis=0); prof=sp.csr_matrix(prof)
    else: prof=sp.csr_matrix((1,Mtx.shape[1]))
    sc=np.asarray(prof.dot(cold_mat.T).todense()).ravel()
    order=np.argsort(-sc); tfidf_list=[cold_items[j] for j in order[:200]]
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
    for lst in (sa,ss,tfidf_list):
        for r,it in enumerate(lst): rr[it]+=1.0/(60+r)
    hyb=[i for i,_ in sorted(rr.items(),key=lambda t:-t[1])][:200]
    surfaced.update(hyb[:100])
    for k in KS:
        res["tfidf"][k].append(1.0 if g in tfidf_list[:k] else 0.0)
        res["content_hybrid"][k].append(1.0 if g in hyb[:k] else 0.0)
out={"n_cold_items":len(cold_items),"tfidf_vocab":int(Mtx.shape[1]),
     "res":{m:{str(k):res[m][k] for k in KS} for m in res},
     "surfaced_cold_items_hybrid_top100":len(surfaced),"sec":round(time.time()-t0,1)}
json.dump(out,open(os.path.join(OUT,"c1_tfidf_partial.json"),"w"))
print("TFIDF cold_items=%d vocab=%d sec=%.1f | tfidf R@20=%.4f R@200=%.4f | hybrid R@20=%.4f R@50=%.4f R@200=%.4f"%(
 len(cold_items),Mtx.shape[1],time.time()-t0,
 np.mean(res["tfidf"][20]),np.mean(res["tfidf"][200]),np.mean(res["content_hybrid"][20]),np.mean(res["content_hybrid"][50]),np.mean(res["content_hybrid"][200])))
