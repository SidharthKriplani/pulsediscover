"""G15 step 1 — co-occurrence candidate generator + coverage/overlap analysis.
Item-item co-occurrence from train baskets (vocab capped to top-6k popular for memory safety on
3.8GB). For each eval user, aggregate co-occ neighbors of their history -> top-N co-occ candidates.
Measures coverage, MARGINAL coverage vs G13-mid, and overlap with ALS/SAS/content. Caches candidates.
LightGCN deferred (graph training not cheap on CPU at 40k items). No FAISS; deterministic; offline."""
import os, json, time, pickle, numpy as np, pandas as pd, scipy.sparse as sp
from collections import defaultdict
OUT="/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim"; SP=os.path.join(OUT,"domain_splits"); EVID="/sessions/festive-quirky-cannon/mnt/pulsediscover/outputs/evidence"
VOCAB=6000; TOPN_NBR=40; NCOOC=100; MINCO=5; t0=time.time()
tr=pd.read_csv(os.path.join(SP,"train.csv"),usecols=["user_id","book_id"],dtype=str)
popular=tr.book_id.value_counts(); vocab=list(popular.index[:VOCAB]); v2i={b:i for i,b in enumerate(vocab)}
trv=tr[tr.book_id.isin(v2i)]; uu=trv.user_id.unique(); u2i={u:i for i,u in enumerate(uu)}
rows=trv.user_id.map(u2i).to_numpy(); cols=trv.book_id.map(v2i).to_numpy()
R=sp.csr_matrix((np.ones(len(rows),np.float32),(rows,cols)),shape=(len(uu),VOCAB)); R.data[:]=1.0
C=(R.T@R).tocsr(); C.setdiag(0); C.eliminate_zeros()
print("cooc matrix built %dx%d nnz=%d (%.1fs)"%(VOCAB,VOCAB,C.nnz,time.time()-t0))
# top-N neighbors per vocab item (count >= MINCO)
nbr={}
for i in range(VOCAB):
    s,e=C.indptr[i],C.indptr[i+1]; idx=C.indices[s:e]; val=C.data[s:e]
    keep=val>=MINCO
    if keep.sum()==0: nbr[vocab[i]]=[]; continue
    idx=idx[keep]; val=val[keep]; top=np.argsort(-val)[:TOPN_NBR]; nbr[vocab[i]]=[(vocab[idx[t]],float(val[t])) for t in top]
# eval users + history
meta=pickle.load(open(os.path.join(OUT,"d3b_seq_meta.pkl"),"rb")); idx2item=meta["idx2item"]; eval_seq=meta["eval_seq"]
es=pd.read_csv(os.path.join(SP,"eval_sample.csv"),dtype=str); users=es.user_id.tolist(); golds=dict(zip(es.user_id,es.gold))
histb={u:[idx2item[i] for i in eval_seq.get(u,[]) if i<len(idx2item) and idx2item[i] is not None] for u in users}
cooc_cands={}
for u in users:
    agg=defaultdict(float); seen=set(histb[u])
    for b in histb[u]:
        for nb,w in nbr.get(b,[]):
            if nb not in seen: agg[nb]+=w
    cooc_cands[u]=[i for i,_ in sorted(agg.items(),key=lambda t:-t[1])][:NCOOC]
pickle.dump(cooc_cands,open(os.path.join(OUT,"g15_cooc_cands.pkl"),"wb"))
# coverage + marginal + overlap vs G13-mid
G=pickle.load(open(os.path.join(OUT,"g13_cands.pkl"),"rb")); cands=G["cands"]; poptop=G["poptop"]
def midset(u):
    c=cands[u]; s=set(c["als"][:200]+c["sas"][:100]+c["content"][:50]+poptop[:30]); return s
cooc_cov=mid_cov=both_cov=marginal=0; overlaps=[]
for u in users:
    g=golds[u]; mid=midset(u); cc=set(cooc_cands[u])
    if g in cc: cooc_cov+=1
    if g in mid: mid_cov+=1
    if g in (mid|cc): both_cov+=1
    if g in cc and g not in mid: marginal+=1
    overlaps.append(len(cc&mid)/max(len(cc),1))
N=len(users)
out={"scope":"G15 co-occurrence candidate source","tag":"[BUILT — real data, CPU]","cooc_vocab":VOCAB,"topN_neighbors":TOPN_NBR,"cooc_topN":NCOOC,"min_cooccount":MINCO,
 "g13_mid_coverage":round(mid_cov/N,4),"cooc_only_coverage":round(cooc_cov/N,4),
 "mid_plus_cooc_coverage":round(both_cov/N,4),
 "marginal_coverage_added_by_cooc":round(marginal/N,4),
 "mean_overlap_cooc_with_mid":round(float(np.mean(overlaps)),4),
 "lightgcn":"DEFERRED — graph-embedding training not cheap on CPU at 40k items; documented future source",
 "sec":round(time.time()-t0,1)}
json.dump(out,open(os.path.join(EVID,"g15_candidate_source_report.json"),"w"),indent=2)
print(json.dumps(out,indent=2))
