"""G15 step 2 — LTR conversion: retrain on G13-mid + co-occurrence candidates; compare R@20 to the
mid baseline (0.1201). Tests whether the +3.76pp marginal coverage converts. Deterministic split,
local features + cooc rank/flag. Offline; no online/production/gold claim."""
import os, json, time, pickle, math, hashlib, numpy as np, pandas as pd, torch, lightgbm as lgb
import sys; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from pulsediscover import sasrec as S
OUT="/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim"; SP=os.path.join(OUT,"domain_splits"); EVID="/sessions/festive-quirky-cannon/mnt/pulsediscover/outputs/evidence"; MOD="/sessions/festive-quirky-cannon/mnt/pulsediscover/outputs/_models/gpu"
SEED=20260616; MAXLEN=30; t0=time.time()
def uh(u): return int(hashlib.md5(u.encode()).hexdigest(),16)%100
A=pickle.load(open(os.path.join(OUT,"c2_als.pkl"),"rb")); X=A["X"];Y=A["Y"];uidx=A["uidx"];iidx=A["iidx"]
meta=pickle.load(open(os.path.join(OUT,"d3b_seq_meta.pkl"),"rb")); item2idx=meta["item2idx"]; idx2item=meta["idx2item"]; n_items=meta["n_items"]; eval_seq=meta["eval_seq"]
st=torch.load(os.path.join(MOD,"ckpt_small.pt"),weights_only=False,map_location="cpu")
sas=S.SASRec(n_items,d=48,maxlen=MAXLEN,nblocks=1,nheads=1,dropout=0.2); sas.load_state_dict(st["model"]); sas.eval()
cm=pd.read_csv(os.path.join(OUT,"domain_content_meta.csv"),dtype=str).fillna("")
b2c=dict(zip(cm.book_id,cm.creator_id)); b2s=dict(zip(cm.book_id,cm.series_id)); b2y={b:(int(y) if y.isdigit() else 0) for b,y in zip(cm.book_id,cm.publication_year)}
popd=pd.read_csv(os.path.join(SP,"train.csv"),usecols=["book_id"],dtype=str).book_id.value_counts().to_dict()
G=pickle.load(open(os.path.join(OUT,"g13_cands.pkl"),"rb")); cands=G["cands"]; poptop=G["poptop"]
cooc=pickle.load(open(os.path.join(OUT,"g15_cooc_cands.pkl"),"rb"))
es=pd.read_csv(os.path.join(SP,"eval_sample.csv"),dtype=str); users=es.user_id.tolist(); golds=dict(zip(es.user_id,es.gold))
warm=set(iidx)
histb={u:set(idx2item[i] for i in eval_seq.get(u,[]) if i<len(idx2item) and idx2item[i] is not None) for u in users}
uauth={u:set(b2c.get(b,"") for b in histb[u])-{""} for u in users}; user_ser={u:set(b2s.get(b,"") for b in histb[u])-{""} for u in users}
FN=["als_score","als_rank","sas_score","sas_rank","log_pop","pub_year","warm_flag","seen_flag","author_match","series_match","from_als","from_sas","from_content","from_pop","from_cooc","cooc_rank","history_len"]
def lp(s): s=s[-MAXLEN:]; return [0]*(MAXLEN-len(s))+list(s)
def run(add_cooc):
    rowsX=[];rowsY=[];grp=[];split=[];B=500
    for s0 in range(0,len(users),B):
        ub=users[s0:s0+B]; urows=[uidx.get(u,-1) for u in ub]
        alssc=X[[r if r>=0 else 0 for r in urows]]@Y.T
        xb=torch.tensor([lp(eval_seq.get(u,[])) for u in ub],dtype=torch.long)
        with torch.no_grad(): sassc=sas.score_last(xb).numpy()
        for bi,u in enumerate(ub):
            c=cands[u]; cset=c["als"][:200]+c["sas"][:100]+c["content"][:50]+poptop[:30]
            coocl=cooc[u][:100] if add_cooc else []
            cset=list(dict.fromkeys(cset+coocl))
            arank={b:r for r,b in enumerate(c["als"])}; sasrank={b:r for r,b in enumerate(c["sas"])}; crank={b:r for r,b in enumerate(coocl)}
            aset=set(c["als"][:200]); sset=set(c["sas"][:100]); cont=set(c["content"][:50]); pset=set(poptop[:30]); ccset=set(coocl)
            sv=sassc[bi]; row=alssc[bi]; au=uauth[u]; se=user_ser[u]; hb=histb[u]; hl=len(eval_seq.get(u,[]))
            feats=[]
            for b in cset:
                feats.append([float(row[iidx[b]]) if b in iidx else np.nan,arank.get(b,999),
                    float(sv[item2idx[b]]) if b in item2idx else np.nan,sasrank.get(b,999),
                    math.log1p(popd.get(b,0)),b2y.get(b,0),1 if b in warm else 0,1 if b in hb else 0,
                    1 if b2c.get(b,"") in au else 0,1 if b2s.get(b,"") in se else 0,
                    1 if b in aset else 0,1 if b in sset else 0,1 if b in cont else 0,1 if b in pset else 0,
                    1 if b in ccset else 0,crank.get(b,999),hl])
            rowsX.append(np.array(feats,np.float32)); rowsY.append(np.array([1 if b==golds[u] else 0 for b in cset])); grp.append(len(cset)); split.append("tr" if uh(u)<70 else ("va" if uh(u)<85 else "te"))
    def block(s):
        sel=[i for i,x in enumerate(split) if x==s]; return lgb.Dataset(np.vstack([rowsX[i] for i in sel]),label=np.concatenate([rowsY[i] for i in sel]),group=[grp[i] for i in sel],feature_name=FN)
    params=dict(objective="lambdarank",metric="ndcg",ndcg_eval_at=[20],learning_rate=0.07,num_leaves=31,min_data_in_leaf=50,verbose=-1,seed=SEED,label_gain=[0,1])
    m=lgb.train(params,block("tr"),num_boost_round=500,valid_sets=[block("va")],callbacks=[lgb.early_stopping(40,verbose=False)])
    te=[i for i,x in enumerate(split) if x=="te"]; r20=r50=ndcg=0; cov=0; n=len(te); avgc=np.mean([grp[i] for i in te])
    for i in te:
        pr=m.predict(rowsX[i]); lb=rowsY[i]; o=np.argsort(-pr); ranked=lb[o]; pos=np.where(ranked==1)[0]; rk=pos[0] if len(pos) else 999
        if rk<20:r20+=1;ndcg+=1.0/math.log2(rk+2)
        if rk<50:r50+=1
        if lb.max()==1: cov+=1
    imp=dict(sorted(zip(FN,[int(v) for v in m.feature_importance('gain')]),key=lambda kv:-kv[1])[:6])
    return {"R@20":round(r20/n,4),"R@50":round(r50/n,4),"NDCG@20":round(ndcg/n,4),"coverage":round(cov/n,4),"avg_cands":round(float(avgc),1),"rows":int(sum(grp)),"top_feats":imp}
base=run(False); withc=run(True)
rep=json.load(open(os.path.join(EVID,"g15_candidate_source_report.json")))
rep["ltr_mid_baseline"]=base; rep["ltr_mid_plus_cooc"]=withc
rep["R@20_lift_cooc"]=round(withc["R@20"]-base["R@20"],4)
se=(0.12*0.88/ max(1,int(len(users)*0.15)))**0.5
rep["noise_threshold_1.96SE"]=round(1.96*se,4)
rep["verdict"]=("ADOPT" if (withc["coverage"]>base["coverage"]+0.005 and withc["R@20"]-base["R@20"]>1.96*se) else
   ("REDUNDANT/HONEST-NEGATIVE — adds marginal coverage but R@20 lift within noise" ))
json.dump(rep,open(os.path.join(EVID,"g15_candidate_source_report.json"),"w"),indent=2)
print("mid baseline:",base); print("mid+cooc:",withc); print("R@20 lift",rep["R@20_lift_cooc"],"| 1.96SE",rep["noise_threshold_1.96SE"],"| verdict",rep["verdict"])
print("sec",round(time.time()-t0,1))
