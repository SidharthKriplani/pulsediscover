"""G13 step 2 — LTR conversion test. Retrain local-feature LambdaMART (G12-B showed GPU features
non-additive) on a given candidate config; same deterministic md5 user split. Tests whether a higher
candidate ceiling converts to higher ranker R@20. Usage: python g13_ltr.py <config>
config in {baseline, wide}. Appends to g13_ltr_convert.json."""
import os, sys, json, time, pickle, math, hashlib, numpy as np, pandas as pd, torch, lightgbm as lgb
import sys as _s; HERE=os.path.dirname(os.path.abspath(__file__)); _s.path.insert(0,HERE)
from pulsediscover import sasrec as S
OUT="/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim"; SP=os.path.join(OUT,"domain_splits")
EVID="/sessions/festive-quirky-cannon/mnt/pulsediscover/outputs/evidence"; MOD="/sessions/festive-quirky-cannon/mnt/pulsediscover/outputs/_models/gpu"
CFG=sys.argv[1] if len(sys.argv)>1 else "baseline"; SEED=20260616; MAXLEN=30; t0=time.time()
def uhash(u): return int(hashlib.md5(u.encode()).hexdigest(),16)%100
CONF={"baseline":[("als",60),("sas",60),("pop",30)],
      "mid":[("als",200),("sas",100),("content",50),("pop",30)],
      "wide":[("als",400),("sas",200),("content",100),("pop",30)]}[CFG]
A=pickle.load(open(os.path.join(OUT,"c2_als.pkl"),"rb")); X=A["X"];Y=A["Y"];uidx=A["uidx"];iidx=A["iidx"]
meta=pickle.load(open(os.path.join(OUT,"d3b_seq_meta.pkl"),"rb")); item2idx=meta["item2idx"]; idx2item=meta["idx2item"]; n_items=meta["n_items"]; eval_seq=meta["eval_seq"]
st=torch.load(os.path.join(MOD,"ckpt_small.pt"),weights_only=False,map_location="cpu")
sas=S.SASRec(n_items,d=48,maxlen=MAXLEN,nblocks=1,nheads=1,dropout=0.2); sas.load_state_dict(st["model"]); sas.eval()
cm=pd.read_csv(os.path.join(OUT,"domain_content_meta.csv"),dtype=str).fillna("")
b2c=dict(zip(cm.book_id,cm.creator_id)); b2s=dict(zip(cm.book_id,cm.series_id)); b2y={b:(int(y) if y.isdigit() else 0) for b,y in zip(cm.book_id,cm.publication_year)}
pop=pd.read_csv(os.path.join(SP,"train.csv"),usecols=["book_id"],dtype=str).book_id.value_counts(); popd=pop.to_dict()
G=pickle.load(open(os.path.join(OUT,"g13_cands.pkl"),"rb")); cands=G["cands"]; poptop=G["poptop"]
es=pd.read_csv(os.path.join(SP,"eval_sample.csv"),dtype=str); users=es.user_id.tolist(); golds=dict(zip(es.user_id,es.gold))
warm=set(iidx); hlen={u:len(eval_seq.get(u,[])) for u in users}
hist_books={u:set(idx2item[i] for i in eval_seq.get(u,[]) if i<len(idx2item) and idx2item[i] is not None) for u in users}
uauth={u:set(b2c.get(b,"") for b in hist_books[u])-{""} for u in users}; user_ser={u:set(b2s.get(b,"") for b in hist_books[u])-{""} for u in users}
FN=["als_score","als_rank","sas_score","sas_rank","log_pop","pub_year","warm_flag","seen_flag","author_match","series_match","from_als","from_sas","from_content","from_pop","history_len"]
def lp(s): s=s[-MAXLEN:]; return [0]*(MAXLEN-len(s))+list(s)
rowsX=[]; rowsY=[]; grp=[]; split=[]; B=500; cov_hit=0
for s0 in range(0,len(users),B):
    ub=users[s0:s0+B]; urows=[uidx.get(u,-1) for u in ub]
    alssc=X[[r if r>=0 else 0 for r in urows]]@Y.T
    xb=torch.tensor([lp(eval_seq.get(u,[])) for u in ub],dtype=torch.long)
    with torch.no_grad(): sassc=sas.score_last(xb).numpy()
    for bi,u in enumerate(ub):
        c=cands[u]; cset=[]
        amap={}; srank={}
        for src,k in CONF:
            lst=poptop[:k] if src=="pop" else c[src][:k]
            cset+=lst
        cset=list(dict.fromkeys(cset))
        arank={b:r for r,b in enumerate(c["als"])}; sasrank={b:r for r,b in enumerate(c["sas"])}
        aset=set(c["als"][:dict(CONF).get("als",0)]); sset=set(c["sas"][:dict(CONF).get("sas",0)]); cont=set(c["content"][:dict(CONF).get("content",0)]); pset=set(poptop[:dict(CONF).get("pop",0)])
        sv=sassc[bi]; au=uauth[u]; se=user_ser[u]; hb=hist_books[u]; hl=hlen[u]; row=alssc[bi]
        if golds[u] in cset: cov_hit+=1
        for b in cset:
            f_als=float(row[iidx[b]]) if b in iidx else np.nan
            f_sas=float(sv[item2idx[b]]) if b in item2idx else np.nan
            rowsX.append([f_als,arank.get(b,999),f_sas,sasrank.get(b,999),math.log1p(popd.get(b,0)),b2y.get(b,0),
                1 if b in warm else 0,1 if b in hb else 0,1 if b2c.get(b,"") in au else 0,1 if b2s.get(b,"") in se else 0,
                1 if b in aset else 0,1 if b in sset else 0,1 if b in cont else 0,1 if b in pset else 0,hl])
            rowsY.append(1 if b==golds[u] else 0)
        grp.append(len(cset)); split.append("tr" if uhash(u)<70 else ("va" if uhash(u)<85 else "te"))
Xf=np.array(rowsX,np.float32); Yf=np.array(rowsY)
# expand split per row
sp_row=np.repeat(split,grp);
def mask(s): return sp_row==s
def ds(s):
    m=mask(s); gp=[g for g,sl in zip(grp,split) if sl==s]; return lgb.Dataset(Xf[m],label=Yf[m],group=gp,feature_name=FN)
params=dict(objective="lambdarank",metric="ndcg",ndcg_eval_at=[20],learning_rate=0.07,num_leaves=31,min_data_in_leaf=50,verbose=-1,seed=SEED,label_gain=[0,1])
m=lgb.train(params,ds("tr"),num_boost_round=500,valid_sets=[ds("va")],callbacks=[lgb.early_stopping(40,verbose=False)])
# eval on test: per user
te_users=[u for u,sl in zip(users,split) if sl=="te"]; nte=len(te_users)
# rebuild test rows per user for ranking
off=0; user_rows={}; idxptr=0
# recompute test predictions grouped: iterate rows with split
te_mask=mask("te"); preds=m.predict(Xf[te_mask]); te_lab=Yf[te_mask]
# need group boundaries within test
te_grp=[g for g,sl in zip(grp,split) if sl=="te"]
r20=r50=ndcg=0; p=0
for g in te_grp:
    pr=preds[p:p+g]; lb=te_lab[p:p+g]; p+=g
    order=np.argsort(-pr); ranked=lb[order]; pos=np.where(ranked==1)[0]; rk=pos[0] if len(pos) else 999
    if rk<20: r20+=1; ndcg+=1.0/math.log2(rk+2)
    if rk<50: r50+=1
res={"config":CFG,"candidate_spec":dict(CONF),"n_test_users":nte,"avg_cands_per_user":round(len(Xf)/len(users),1),
 "table_rows":int(len(Xf)),"candidate_coverage_all":round(cov_hit/len(users),4),
 "R@20":round(r20/nte,4),"R@50":round(r50/nte,4),"NDCG@20":round(ndcg/nte,4),
 "best_iter":m.best_iteration,"top_feats":dict(sorted(zip(FN,[int(v) for v in m.feature_importance('gain')]),key=lambda kv:-kv[1])[:6]),
 "sec":round(time.time()-t0,1)}
p_=os.path.join(EVID,"g13_ltr_convert.json"); allr=json.load(open(p_)) if os.path.exists(p_) else {}
allr[CFG]=res; json.dump(allr,open(p_,"w"),indent=2)
print(json.dumps(res,indent=2))
