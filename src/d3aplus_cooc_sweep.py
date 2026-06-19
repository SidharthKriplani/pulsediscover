"""D3A+ co-occurrence vocab sweep (frugal CSR). Usage: python d3aplus_cooc_sweep.py VOCAB
Builds neighbors at VOCAB top-popular items, evals cooc R@10/20/50 on the 5,000-user sample.
Appends to d3aplus_cooc_sweep.json. Reports runtime; OOM at large VOCAB will kill the process
(recorded as a failure point)."""
import pandas as pd, numpy as np, os, json, sys, time, pickle
import scipy.sparse as sp
from collections import defaultdict
OUT="/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim"; SP=os.path.join(OUT,"domain_splits")
VOCAB=int(sys.argv[1]); MIN_CO=5; TOPN=80; t0=time.time()
core=pd.read_csv(os.path.join(OUT,"domain_core_interactions.csv"),usecols=["user_id","book_id"],dtype=str)
held=pd.read_csv(os.path.join(SP,"heldout_eval.csv"),dtype=str)
tr=core[~(core.user_id+"|"+core.book_id).isin(set(held.user_id+"|"+held.gold))][["user_id","book_id"]]; del core
popser=tr.book_id.value_counts(); vocab=list(popser.head(VOCAB).index); vi={it:j for j,it in enumerate(vocab)}
sub=tr[tr.book_id.isin(set(vocab))]; uu={u:i for i,u in enumerate(sub.user_id.unique())}
M=sp.csr_matrix((np.ones(len(sub),dtype=np.float32),(sub.user_id.map(uu).to_numpy(),sub.book_id.map(vi).to_numpy())),
                shape=(len(uu),len(vocab))); M.data[:]=1.0
cc=np.asarray(M.sum(0)).ravel(); N=M.shape[0]; C=(M.T@M).tocsr(); nnz=C.nnz
neigh={}
for j in range(len(vocab)):
    s,e=C.indptr[j],C.indptr[j+1]
    best=[(vocab[int(k)],float(np.log((v/N)/((cc[j]/N)*(cc[int(k)]/N))+1e-12)))
          for k,v in zip(C.indices[s:e],C.data[s:e]) if k!=j and v>=MIN_CO]
    best.sort(key=lambda t:-t[1]); neigh[vocab[j]]=best[:TOPN]
del C,M,sub,tr
hist=pickle.load(open(os.path.join(OUT,"d3aplus_maps.pkl"),"rb"))["hist"]
samp=pd.read_csv(os.path.join(SP,"eval_sample.csv"),dtype=str)
def cooc_list(h,k=50):
    sc=defaultdict(float)
    for it in h:
        for nb,w in neigh.get(it,()):
            if nb not in h: sc[nb]+=w
    return [i for i,_ in sorted(sc.items(),key=lambda t:-t[1])[:k]]
import math
R={10:[],20:[],50:[]}
for u,g in zip(samp.user_id,samp.gold):
    l=cooc_list(hist.get(u,set()),50)
    for k in R: R[k].append(1.0 if g in l[:k] else 0.0)
out={"vocab":VOCAB,"cooc_nnz_MtM":int(nnz),"R@10":round(float(np.mean(R[10])),4),
     "R@20":round(float(np.mean(R[20])),4),"R@50":round(float(np.mean(R[50])),4),"sec":round(time.time()-t0,1)}
p=os.path.join(OUT,"d3aplus_cooc_sweep.json"); d=json.load(open(p)) if os.path.exists(p) else {}
d[str(VOCAB)]=out; json.dump(d,open(p,"w"),indent=2)
print("COOC_SWEEP",out)
