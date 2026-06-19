"""D3A+ step 1 — fit shared artifacts: ALS(f16,i4), co-occurrence@5k neighbors, domain maps
(author/series), item popularity, and per-eval-user histories. Pickled for reuse by the eval step."""
import pandas as pd, numpy as np, json, os, time, pickle
import scipy.sparse as sp
OUT="/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim"; SP=os.path.join(OUT,"domain_splits")
F=16; ALPHA=40.0; REG=0.1; ITERS=4; SEED=20260616; COOC_VOCAB=5000; MIN_CO=5; TOPN=80; t0=time.time()
rng=np.random.default_rng(SEED); c=1.0+ALPHA
core=pd.read_csv(os.path.join(OUT,"domain_core_interactions.csv"),usecols=["user_id","book_id"],
                 dtype={"user_id":"str","book_id":"str"})
held=pd.read_csv(os.path.join(SP,"heldout_eval.csv"),dtype={"user_id":"str","gold":"str"})
heldset=set(held.user_id+"|"+held.gold)
tr=core[~(core.user_id+"|"+core.book_id).isin(heldset)][["user_id","book_id"]]; del core
# popularity
popser=tr.book_id.value_counts(); item_pop=popser.to_dict(); pop_top200=list(popser.head(200).index)
train_items=set(popser.index)
# ALS f16 i4
u_codes,u_uniq=pd.factorize(tr.user_id); i_codes,i_uniq=pd.factorize(tr.book_id)
nU=len(u_uniq); nI=len(i_uniq)
o=np.argsort(u_codes,kind="stable"); user_items=np.split(i_codes[o],np.flatnonzero(np.diff(u_codes[o]))+1)
o2=np.argsort(i_codes,kind="stable"); item_users=np.split(u_codes[o2],np.flatnonzero(np.diff(i_codes[o2]))+1)
X=0.01*rng.standard_normal((nU,F)); Y=0.01*rng.standard_normal((nI,F)); I=REG*np.eye(F)
def solve(Fe,FtF,idx):
    Fi=Fe[idx]; return np.linalg.solve(FtF+(c-1.0)*(Fi.T@Fi)+I, c*Fi.sum(0))
for _ in range(ITERS):
    YtY=Y.T@Y
    for uc in range(nU): X[uc]=solve(Y,YtY,user_items[uc])
    XtX=X.T@X
    for ic in range(nI): Y[ic]=solve(X,XtX,item_users[ic])
uidx={u:i for i,u in enumerate(u_uniq)}; iidx={it:i for i,it in enumerate(i_uniq)}
als_sec=round(time.time()-t0,1)
# co-occurrence @5k (frugal)
vocab=list(popser.head(COOC_VOCAB).index); vi={it:j for j,it in enumerate(vocab)}
sub=tr[tr.book_id.isin(set(vocab))]; uu={u:i for i,u in enumerate(sub.user_id.unique())}
M=sp.csr_matrix((np.ones(len(sub),dtype=np.float32),(sub.user_id.map(uu).to_numpy(),sub.book_id.map(vi).to_numpy())),
                shape=(len(uu),len(vocab))); M.data[:]=1.0
cc=np.asarray(M.sum(0)).ravel(); N=M.shape[0]; C=(M.T@M).tocsr(); neigh={}
for j in range(len(vocab)):
    s,e=C.indptr[j],C.indptr[j+1]
    best=[(vocab[int(k)],float(np.log((v/N)/((cc[j]/N)*(cc[int(k)]/N))+1e-12)))
          for k,v in zip(C.indices[s:e],C.data[s:e]) if k!=j and v>=MIN_CO]
    best.sort(key=lambda t:-t[1]); neigh[vocab[j]]=best[:TOPN]
del C,M,sub
# domain maps from core meta
meta=pd.read_csv(os.path.join(OUT,"domain_core_books_meta.csv"),
                 dtype={"book_id":"str","creator_id":"str","series_id":"str"}).fillna("")
b2c=dict(zip(meta.book_id,meta.creator_id)); b2s=dict(zip(meta.book_id,meta.series_id))
from collections import defaultdict
a2b=defaultdict(list); s2b=defaultdict(list)
for bid in train_items:
    a=b2c.get(bid,""); s=b2s.get(bid,"")
    if a: a2b[a].append(bid)
    if s: s2b[s].append(bid)
for a in a2b: a2b[a].sort(key=lambda x:-item_pop.get(x,0))
for s in s2b: s2b[s].sort(key=lambda x:-item_pop.get(x,0))
train_creators=set(a2b.keys()); train_series=set(s2b.keys())
# per-eval-user histories (full train history) + read authors/series
samp=pd.read_csv(os.path.join(SP,"eval_sample.csv"),dtype={"user_id":"str","gold":"str"})
ev_users=set(samp.user_id)
hsub=tr[tr.user_id.isin(ev_users)]
hist={u:set(g) for u,g in hsub.groupby("user_id")["book_id"]}
ev_auth={u:set(b2c.get(b,"") for b in h)-{""} for u,h in hist.items()}
ev_ser={u:set(b2s.get(b,"") for b in h)-{""} for u,h in hist.items()}
art={"X":X,"Y":Y,"i_uniq":np.array(i_uniq),"uidx":uidx,"iidx":iidx,
     "neigh":neigh,"item_pop":item_pop,"pop_top200":pop_top200,"train_items":train_items,
     "b2c":b2c,"b2s":b2s,"a2b":dict(a2b),"s2b":dict(s2b),
     "train_creators":train_creators,"train_series":train_series,
     "hist":hist,"ev_auth":ev_auth,"ev_ser":ev_ser,
     "user_items_codes":{u:set(i_uniq[ user_items[uidx[u]] ]) for u in ev_users if u in uidx}}
pickle.dump(art,open(os.path.join(OUT,"d3aplus_artifacts.pkl"),"wb"))
json.dump({"als_f":F,"als_iters":ITERS,"als_sec":als_sec,"cooc_vocab":COOC_VOCAB,
           "nU":nU,"nI":nI,"fit_sec":round(time.time()-t0,1)},
          open(os.path.join(OUT,"d3aplus_fit_meta.json"),"w"),indent=2)
print("FIT_DONE nU=%d nI=%d als=%.1fs total=%.1fs"%(nU,nI,als_sec,time.time()-t0))
