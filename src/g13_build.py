"""G13 step 1 — candidate generation + coverage sweep on the 5k LLO eval_sample users.
Sources: ALS f64 (top-400), SASRec-small GPU ckpt (top-200), content-hybrid cold (top-100),
popularity (top-30). Co-occurrence + LightGCN deferred (no cheap candidate artifact). Computes
candidate coverage@K per config; caches per-user candidate lists + scores for the LTR step."""
import os, json, time, pickle, math, numpy as np, pandas as pd, torch
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import sys; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from pulsediscover import sasrec as S
OUT="/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim"; SP=os.path.join(OUT,"domain_splits")
EVID="/sessions/festive-quirky-cannon/mnt/pulsediscover/outputs/evidence"; MOD="/sessions/festive-quirky-cannon/mnt/pulsediscover/outputs/_models/gpu"
MAXLEN=30; SEED=20260616; t0=time.time()
A=pickle.load(open(os.path.join(OUT,"c2_als.pkl"),"rb")); X=A["X"];Y=A["Y"];uidx=A["uidx"];iidx=A["iidx"];i_uniq=A["i_uniq"]
meta=pickle.load(open(os.path.join(OUT,"d3b_seq_meta.pkl"),"rb")); item2idx=meta["item2idx"]; idx2item=meta["idx2item"]; n_items=meta["n_items"]; eval_seq=meta["eval_seq"]
st=torch.load(os.path.join(MOD,"ckpt_small.pt"),weights_only=False,map_location="cpu")
sas=S.SASRec(n_items,d=48,maxlen=MAXLEN,nblocks=1,nheads=1,dropout=0.2); sas.load_state_dict(st["model"]); sas.eval()
cm=pd.read_csv(os.path.join(OUT,"domain_content_meta.csv"),dtype=str).fillna("")
b2c=dict(zip(cm.book_id,cm.creator_id)); b2s=dict(zip(cm.book_id,cm.series_id)); b2y={b:(int(y) if y.isdigit() else 0) for b,y in zip(cm.book_id,cm.publication_year)}
text=dict(zip(cm.book_id,(cm.title+" "+cm.shelves+" "+cm.desc)))
L=pickle.load(open(os.path.join(OUT,"c2_lists.pkl"),"rb")); cold_set=L["cold_items"]; cold_items=list(cold_set)
pop=pd.read_csv(os.path.join(SP,"train.csv"),usecols=["book_id"],dtype=str).book_id.value_counts(); poptop=list(pop.index[:30])
es=pd.read_csv(os.path.join(SP,"eval_sample.csv"),dtype=str); users=es.user_id.tolist(); golds=dict(zip(es.user_id,es.gold))
# content-hybrid setup (same-author/series/TFIDF over cold items)
a2c=defaultdict(list); s2c=defaultdict(list)
for it in cold_items:
    if b2c.get(it): a2c[b2c[it]].append(it)
    if b2s.get(it): s2c[b2s[it]].append(it)
for a in a2c: a2c[a].sort(key=lambda x:-b2y.get(x,0))
for s in s2c: s2c[s].sort(key=lambda x:-b2y.get(x,0))
hist_books={u:[idx2item[i] for i in eval_seq.get(u,[]) if i<len(idx2item) and idx2item[i] is not None] for u in users}
all_hist=set(b for bs in hist_books.values() for b in bs)
docs=list(cold_set|all_hist); ridx={it:i for i,it in enumerate(docs)}
vec=TfidfVectorizer(max_features=20000,stop_words="english"); Mtx=vec.fit_transform([text.get(it,"") for it in docs]); cold_mat=Mtx[np.array([ridx[it] for it in cold_items])]
def lp(s): s=s[-MAXLEN:]; return [0]*(MAXLEN-len(s))+list(s)
cands={}; B=500
for s0 in range(0,len(users),B):
    ub=users[s0:s0+B]; urows=[uidx.get(u,-1) for u in ub]
    alssc=X[[r if r>=0 else 0 for r in urows]]@Y.T
    xb=torch.tensor([lp(eval_seq.get(u,[])) for u in ub],dtype=torch.long)
    with torch.no_grad(): sassc=sas.score_last(xb).numpy()
    for bi,u in enumerate(ub):
        als=[]
        if urows[bi]>=0:
            sc=alssc[bi]; top=np.argpartition(-sc,400)[:400]; top=top[np.argsort(-sc[top])]; als=[i_uniq[t] for t in top]
        sv=sassc[bi]; sv[0]=-1e9; stop=np.argpartition(-sv,200)[:200]; stop=stop[np.argsort(-sv[stop])]; sas_c=[idx2item[t] for t in stop if idx2item[t] is not None]
        # content-hybrid top-100 (cold)
        h=hist_books[u]; r=[ridx[i] for i in h if i in ridx]
        prof=sp.csr_matrix(Mtx[r].mean(axis=0)) if r else sp.csr_matrix((1,Mtx.shape[1]))
        tf=np.asarray(prof.dot(cold_mat.T).todense()).ravel(); tford=[cold_items[j] for j in np.argsort(-tf)[:100]]
        auth=set(b2c.get(b,"") for b in h)-{""}; ser=set(b2s.get(b,"") for b in h)-{""}
        sa=[it for a in auth for it in a2c.get(a,())][:50]; ss=[it for sname in ser for it in s2c.get(sname,())][:50]
        rr=defaultdict(float)
        for lst in (sa,ss,tford):
            for rk,it in enumerate(lst): rr[it]+=1.0/(60+rk)
        content=[i for i,_ in sorted(rr.items(),key=lambda t:-t[1])][:100]
        cands[u]={"als":als,"sas":sas_c,"content":content}
pickle.dump({"cands":cands,"poptop":poptop},open(os.path.join(OUT,"g13_cands.pkl"),"wb"))
def cov(cfg):
    hit=0; sizes=[]
    for u in users:
        c=cands[u]; s=set()
        for src,k in cfg:
            if src=="pop": s.update(poptop[:k])
            else: s.update(c[src][:k])
        sizes.append(len(s));
        if golds[u] in s: hit+=1
    return round(hit/len(users),4), round(np.mean(sizes),1)
configs={
 "ALS-60 (G12 warm part)":[("als",60)],
 "ALS-120":[("als",120)],"ALS-200":[("als",200)],"ALS-400":[("als",400)],
 "ALS-60+SAS-60+pop-30 (G12-like)":[("als",60),("sas",60),("pop",30)],
 "ALS-200+SAS-100":[("als",200),("sas",100)],
 "ALS-200+SAS-100+content-50":[("als",200),("sas",100),("content",50)],
 "WIDE ALS-400+SAS-200+content-100+pop-30":[("als",400),("sas",200),("content",100),("pop",30)],
}
covres={}
for name,cfg in configs.items():
    c,sz=cov(cfg); covres[name]={"coverage":c,"avg_cands_per_user":sz}
out={"scope":"G13 candidate coverage sweep","tag":"[BUILT — real data, CPU]","n_users":len(users),
 "warm_gold_users":int(es.gold.isin(set(iidx)).sum()),"cold_gold_users":int((~es.gold.isin(set(iidx))).sum()),
 "g12_baseline_ceiling":0.157,"sources":"ALS f64, SASRec-small(GPU ckpt), content-hybrid cold, popularity; cooc+LightGCN deferred (no cheap candidate artifact)",
 "coverage_by_config":covres,"sec":round(time.time()-t0,1)}
json.dump(out,open(os.path.join(EVID,"g13_candidate_coverage_report.json"),"w"),indent=2)
for k,v in covres.items(): print("%-44s cov %.4f  avg_cands %.1f"%(k,v["coverage"],v["avg_cands_per_user"]))
print("sec",round(time.time()-t0,1))
