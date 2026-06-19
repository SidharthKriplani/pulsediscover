"""D3A step 2 (memory-safe) — popularity + co-occurrence on the domain core, evaluated on the
5,000-user leave-last-out sample. Co-occurrence capped to top-5k items with per-row (frugal)
PMI-neighbor extraction (the all-pairs M^T M at 20k OOM'd on 3.8GB). Domain-scoped."""
import pandas as pd, numpy as np, json, os, sys, time
import scipy.sparse as sp
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from pulsediscover import eval as E
OUT="/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim"; SP=os.path.join(OUT,"domain_splits")
KS=[5,10,20]; COOC_VOCAB=5000; MIN_CO=5; TOPN=50; t0=time.time()

core=pd.read_csv(os.path.join(OUT,"domain_core_interactions.csv"),
    usecols=["user_id","book_id","event_ts"],dtype={"user_id":"str","book_id":"str","event_ts":"int64"})
held=pd.read_csv(os.path.join(SP,"heldout_eval.csv"),dtype={"user_id":"str","gold":"str"})
heldset=set(held.user_id+"|"+held.gold)
train=core[~(core.user_id+"|"+core.book_id).isin(heldset)][["user_id","book_id"]]
del core
popser=train.book_id.value_counts(); pop_top=list(popser.head(max(KS)).index)
train_items=set(popser.index)

# frugal co-occurrence neighbors on top-5k items
vocab=list(popser.head(COOC_VOCAB).index); vi={it:j for j,it in enumerate(vocab)}
sub=train[train.book_id.isin(set(vocab))]
uu={u:i for i,u in enumerate(sub.user_id.unique())}
rows=sub.user_id.map(uu).to_numpy(); cols=sub.book_id.map(vi).to_numpy()
M=sp.csr_matrix((np.ones(len(rows),dtype=np.float32),(rows,cols)),shape=(len(uu),len(vocab)))
M.data[:]=1.0
cc=np.asarray(M.sum(0)).ravel(); N=M.shape[0]
C=(M.T@M).tocsr()
neigh={}
for j in range(len(vocab)):
    s,e=C.indptr[j],C.indptr[j+1]; idx=C.indices[s:e]; dat=C.data[s:e]
    best=[(vocab[int(k2)],float(np.log((c/N)/((cc[j]/N)*(cc[int(k2)]/N))+1e-12)))
          for k2,c in zip(idx,dat) if k2!=j and c>=MIN_CO]
    best.sort(key=lambda t:-t[1]); neigh[vocab[j]]=best[:TOPN]
del C,M,sub

hist=train.groupby("user_id")["book_id"].apply(set).to_dict(); del train
def cooc_topk(h,k):
    sc={}
    for it in h:
        for nb,w in neigh.get(it,()):
            if nb not in h: sc[nb]=sc.get(nb,0.0)+w
    return [i for i,_ in sorted(sc.items(),key=lambda t:-t[1])[:k]]

samp=pd.read_csv(os.path.join(SP,"eval_sample.csv"),dtype={"user_id":"str","gold":"str"})
res={m:{k:{"r":[],"n":[]} for k in KS} for m in ["popularity","cooccurrence"]}
gold_is_warm=[]
for u,g in zip(samp.user_id,samp.gold):
    h=hist.get(u,set())
    pr=[i for i in pop_top if i not in h][:max(KS)]; cr=cooc_topk(h,max(KS))
    gold_is_warm.append(g in train_items)
    for k in KS:
        res["popularity"][k]["r"].append(E.recall_at_k(pr,g,k)); res["popularity"][k]["n"].append(E.ndcg_at_k(pr,g,k))
        res["cooccurrence"][k]["r"].append(E.recall_at_k(cr,g,k)); res["cooccurrence"][k]["n"].append(E.ndcg_at_k(cr,g,k))
json.dump({"res":res,"gold_is_warm":gold_is_warm,"cooc_vocab":COOC_VOCAB,"min_co":MIN_CO,
           "n_eval":int(len(samp)),"sec":round(time.time()-t0,1)},
          open(os.path.join(OUT,"d3a_popcooc_metrics.json"),"w"))
print("POPCOOC_DONE n=%d warm_gold=%d sec=%.1f | pop R@10=%.4f cooc R@10=%.4f"%(
 len(samp),sum(gold_is_warm),time.time()-t0,np.mean(res["popularity"][10]["r"]),np.mean(res["cooccurrence"][10]["r"])))
