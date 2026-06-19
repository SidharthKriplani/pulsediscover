"""G12 — Learning-to-Rank (LambdaMART) re-ranker. The two-stage industry layer: many retrievers
produce a candidate set, then ONE ranker fuses their signals as features. Tests whether a learned
ranker beats the best single retriever on the SAME candidate sets. CPU (LightGBM). Real data.

Honest framing: this is RE-RANKING recall within a candidate set (not full-40k retrieval recall) —
comparable to the baseline rankers on identical candidates, NOT to ALS f64's full-catalog 0.085.
SASRec/two-tower scores can be added as features later (noted) once GPU results return."""
import os, sys, json, time, pickle, math, numpy as np, pandas as pd, lightgbm as lgb
OUT="/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim"; SP=os.path.join(OUT,"domain_splits")
EVID="/sessions/festive-quirky-cannon/mnt/pulsediscover/outputs/evidence"; PLOTS="/sessions/festive-quirky-cannon/mnt/pulsediscover/outputs/plots"
SEED=20260616; NW=60; NC=30; t0=time.time(); rng=np.random.default_rng(SEED)
A=pickle.load(open(os.path.join(OUT,"c2_als.pkl"),"rb")); X=A["X"];Y=A["Y"];uidx=A["uidx"];iidx=A["iidx"]
L=pickle.load(open(os.path.join(OUT,"c2_lists.pkl"),"rb")); WL=L["warm_list"]; CL=L["cold_list"]; cold_set=L["cold_items"]
M=pickle.load(open(os.path.join(OUT,"c2_meta.pkl"),"rb")); hist=M["eval_hist"]
cm=pd.read_csv(os.path.join(OUT,"domain_content_meta.csv"),dtype=str).fillna("")
b2c=dict(zip(cm.book_id,cm.creator_id)); b2s=dict(zip(cm.book_id,cm.series_id))
b2y={b:(int(y) if y.isdigit() else 0) for b,y in zip(cm.book_id,cm.publication_year)}
pop=pd.read_csv(os.path.join(SP,"train.csv"),usecols=["book_id"],dtype=str).book_id.value_counts().to_dict()
w=pd.read_csv(os.path.join(SP,"warm_eval_sample.csv"),dtype=str); c=pd.read_csv(os.path.join(SP,"cold_eval_sample.csv"),dtype=str)
queries=[(u,g,"warm") for u,g in zip(w.user_id,w.gold)]+[(u,g,"cold") for u,g in zip(c.user_id,c.gold)]
uauth={u:set(b2c.get(b,"") for b in hist.get(u,()))-{""} for u,_,_ in queries}
useries={u:set(b2s.get(b,"") for b in hist.get(u,()))-{""} for u,_,_ in queries}
FN=["als_score","als_rank","content_rank","rrf","log_pop","cold_flag","author_match","series_match","pub_year","in_warm","in_cold"]
def split(u): return (hash(u)%4!=0)   # ~75% train by USER (no leakage)
Xtr=[];ytr=[];gtr=[];Xte=[];yte=[];gte=[];te_meta=[]
gold_in_cand_tr=gold_in_cand_te=0
for u,gold,seg in queries:
    wl=WL.get(u,[])[:NW]; cl=CL.get(u,[])[:NC]; cands=list(dict.fromkeys(wl+cl))
    if not cands: continue
    arank={it:r for r,it in enumerate(wl)}; crank={it:r for r,it in enumerate(cl)}
    urow=X[uidx[u]] if u in uidx else None
    warm_c=[it for it in cands if it in iidx]
    alsmap={}
    if urow is not None and warm_c:
        rows=np.array([iidx[it] for it in warm_c]); sc=Y[rows]@urow
        alsmap={it:float(s) for it,s in zip(warm_c,sc)}
    feats=[]; labs=[]
    for it in cands:
        ar=arank.get(it,NW); cr=crank.get(it,NC)
        feats.append([alsmap.get(it,0.0),ar,cr,1.0/(60+ar)+1.0/(60+cr),math.log1p(pop.get(it,0)),
                      1 if it in cold_set else 0,1 if b2c.get(it,"") in uauth[u] else 0,
                      1 if b2s.get(it,"") in useries[u] else 0,b2y.get(it,0),1 if it in arank else 0,1 if it in crank else 0])
        labs.append(1 if it==gold else 0)
    has=1 if (gold in cands) else 0
    if split(u):
        Xtr+=feats; ytr+=labs; gtr.append(len(cands)); gold_in_cand_tr+=has
    else:
        Xte+=feats; yte+=labs; gte.append(len(cands)); gold_in_cand_te+=has
        te_meta.append((len(cands),feats,labs))
Xtr=np.array(Xtr,np.float32); ytr=np.array(ytr); Xte=np.array(Xte,np.float32); yte=np.array(yte)
ntr=len(gtr); nte=len(gte)
print("built tr_q=%d te_q=%d tr_rows=%d te_rows=%d cand_recall tr=%.3f te=%.3f (%.1fs)"%(
  ntr,nte,len(Xtr),len(Xte),gold_in_cand_tr/ntr,gold_in_cand_te/nte,time.time()-t0))
ds=lgb.Dataset(Xtr,label=ytr,group=gtr,feature_name=FN)
params=dict(objective="lambdarank",metric="ndcg",ndcg_eval_at=[20],learning_rate=0.1,num_leaves=31,
            min_data_in_leaf=50,n_estimators=300,verbose=-1,seed=SEED,label_gain=[0,1])
model=lgb.train(params,ds,num_boost_round=300)
# eval helper on test queries
def metrics_from_scores(scorer):
    r20=nd=0
    off=0
    for (n,feats,labs) in te_meta:
        f=np.array(feats,np.float32); s=scorer(f); order=np.argsort(-s)
        ranked=[labs[i] for i in order]
        try: pos=ranked.index(1)
        except ValueError: pos=999
        if pos<20: r20+=1; nd+=1.0/math.log2(pos+2)
    return r20/nte, nd/nte
ltr=metrics_from_scores(lambda f: model.predict(f))
als=metrics_from_scores(lambda f: f[:,0]*1.0 - f[:,1]*1e-6)             # ALS score (tie-break by rank)
alsrank=metrics_from_scores(lambda f: -f[:,1])                          # ALS rank only
rrf=metrics_from_scores(lambda f: f[:,3])                               # RRF
popb=metrics_from_scores(lambda f: f[:,4])                              # popularity
imp=dict(zip(FN,[int(v) for v in model.feature_importance(importance_type="gain")]))
imp=dict(sorted(imp.items(),key=lambda kv:-kv[1]))
out={"scope":"G12 Learning-to-Rank (LambdaMART) re-ranker over candidate sets","tag":"[BUILT — real data, CPU]",
 "framing":"RE-RANKING recall within candidate set (warm top-%d + cold top-%d), NOT full-catalog retrieval; comparable to baseline rankers on identical candidates only"%(NW,NC),
 "n_train_queries":ntr,"n_test_queries":nte,"candidate_recall_ceiling_test":round(gold_in_cand_te/nte,4),
 "features":FN,"feature_importance_gain":imp,
 "results_recall@20":{"LambdaMART_LTR":round(ltr[0],4),"ALS_score":round(als[0],4),"ALS_rank":round(alsrank[0],4),"RRF":round(rrf[0],4),"popularity":round(popb[0],4)},
 "results_ndcg@20":{"LambdaMART_LTR":round(ltr[1],4),"ALS_score":round(als[1],4),"RRF":round(rrf[1],4),"popularity":round(popb[1],4)},
 "lift_LTR_vs_best_single":round(ltr[0]-max(als[0],rrf[0],popb[0]),4),
 "honest_notes":["Re-ranking recall within candidate set; NOT comparable to ALS f64 full-catalog 0.085.",
   "Candidate recall ceiling caps any ranker; LTR optimizes ordering within that ceiling.",
   "SASRec/two-tower scores not yet added as features (GPU pending) — a documented v2 lift.",
   "Label = single held-out gold per query (binary relevance)."],
 "sec":round(time.time()-t0,1)}
json.dump(out,open(os.path.join(EVID,"g12_ltr_report.json"),"w"),indent=2)
pickle.dump(model,open(os.path.join(OUT,"g12_ltr_model.pkl"),"wb"))
print("LTR R@20 %.4f | ALS %.4f | RRF %.4f | pop %.4f | ceiling %.4f"%(ltr[0],als[0],rrf[0],popb[0],gold_in_cand_te/nte))
print("lift vs best single:",out["lift_LTR_vs_best_single"]); print("top feats:",list(imp.items())[:5])
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    names=["LambdaMART\nLTR","ALS\nscore","RRF","popularity"]; vals=[ltr[0],als[0],rrf[0],popb[0]]
    fig,ax=plt.subplots(1,2,figsize=(13,5))
    ax[0].bar(range(4),vals,color=["#ee7733","#4477aa","#888","#bbb"]); ax[0].axhline(gold_in_cand_te/nte,ls="--",c="green",label="candidate recall ceiling")
    for i,v in enumerate(vals): ax[0].text(i,v+0.005,"%.3f"%v,ha="center")
    ax[0].set_xticks(range(4)); ax[0].set_xticklabels(names); ax[0].set_ylabel("Recall@20 (within candidates)"); ax[0].legend(); ax[0].set_title("G12 LTR vs single-retriever rankers")
    fi=list(imp.items())[:8][::-1]; ax[1].barh([k for k,_ in fi],[v for _,v in fi],color="#ee7733"); ax[1].set_title("LambdaMART feature importance (gain)")
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS,"g12_ltr.png"),dpi=110); plt.close()
except Exception as e: print("plot skipped",e)
