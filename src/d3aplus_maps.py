"""D3A+ step 1b — co-occurrence@5k neighbors + domain author/series maps + per-eval-user
histories/authors/series. Saves maps.pkl for the eval step."""
import pandas as pd, numpy as np, os, time, pickle, json
import scipy.sparse as sp
from collections import defaultdict
OUT="/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim"; SP=os.path.join(OUT,"domain_splits")
COOC_VOCAB=5000; MIN_CO=5; TOPN=80; t0=time.time()
core=pd.read_csv(os.path.join(OUT,"domain_core_interactions.csv"),usecols=["user_id","book_id"],
                 dtype={"user_id":"str","book_id":"str"})
held=pd.read_csv(os.path.join(SP,"heldout_eval.csv"),dtype={"user_id":"str","gold":"str"})
tr=core[~(core.user_id+"|"+core.book_id).isin(set(held.user_id+"|"+held.gold))][["user_id","book_id"]]; del core
popser=tr.book_id.value_counts(); item_pop=popser.to_dict(); train_items=set(popser.index)
# co-occurrence @5k
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
# domain maps
meta=pd.read_csv(os.path.join(OUT,"domain_core_books_meta.csv"),
                 dtype={"book_id":"str","creator_id":"str","series_id":"str"}).fillna("")
b2c=dict(zip(meta.book_id,meta.creator_id)); b2s=dict(zip(meta.book_id,meta.series_id))
a2b=defaultdict(list); s2b=defaultdict(list)
for bid in train_items:
    a=b2c.get(bid,""); s=b2s.get(bid,"")
    if a: a2b[a].append(bid)
    if s: s2b[s].append(bid)
for a in a2b: a2b[a].sort(key=lambda x:-item_pop.get(x,0))
for s in s2b: s2b[s].sort(key=lambda x:-item_pop.get(x,0))
# per-eval-user history/authors/series
samp=pd.read_csv(os.path.join(SP,"eval_sample.csv"),dtype={"user_id":"str","gold":"str"})
ev=set(samp.user_id); hsub=tr[tr.user_id.isin(ev)]
hist={u:set(g) for u,g in hsub.groupby("user_id")["book_id"]}
ev_auth={u:(set(b2c.get(b,"") for b in h)-{""}) for u,h in hist.items()}
ev_ser={u:(set(b2s.get(b,"") for b in h)-{""}) for u,h in hist.items()}
art={"neigh":neigh,"b2c":b2c,"b2s":b2s,"a2b":dict(a2b),"s2b":dict(s2b),
     "train_creators":set(a2b.keys()),"train_series":set(s2b.keys()),
     "hist":hist,"ev_auth":ev_auth,"ev_ser":ev_ser}
pickle.dump(art,open(os.path.join(OUT,"d3aplus_maps.pkl"),"wb"))
json.dump({"cooc_vocab":COOC_VOCAB,"min_co":MIN_CO,"topn":TOPN,
           "n_authors":len(a2b),"n_series":len(s2b),"sec":round(time.time()-t0,1)},
          open(os.path.join(OUT,"d3aplus_maps_meta.json"),"w"),indent=2)
print("MAPS_DONE authors=%d series=%d sec=%.1f"%(len(a2b),len(s2b),time.time()-t0))
