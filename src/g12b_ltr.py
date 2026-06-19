"""G12-B — LambdaMART LTR on the GPU-enriched candidate table. USER-level train/val/test split;
explicit missingness + history features; fair variant ladder (ALS-only -> local LTR -> enriched LTR
-> enriched-no-TwoTower). Re-ranking recall WITHIN candidate set (coverage ceiling ~15%), NOT
full-catalog. No online/production/gold claims."""
import os, json, time, pickle, math, hashlib, numpy as np, pandas as pd, lightgbm as lgb
def uhash(u): return int(hashlib.md5(u.encode()).hexdigest(),16)%100   # deterministic, reproducible split
ROOT="/sessions/festive-quirky-cannon/mnt/pulsediscover"; OUT=ROOT+"/data/interim"; EVID=ROOT+"/outputs/evidence"; PLOTS=ROOT+"/outputs/plots"
SEED=20260616; t0=time.time(); rng=np.random.default_rng(SEED)
df=pd.read_csv(EVID+"/g12_candidate_feature_table_enriched.csv.gz",dtype={"user_id":str,"item_id":str})
meta=pickle.load(open(OUT+"/d3b_seq_meta.pkl","rb")); eval_seq=meta["eval_seq"]
hlen={u:len(s) for u,s in eval_seq.items()}
df["history_len"]=df.user_id.map(lambda u:hlen.get(u,0)); df["has_sequence_history"]=(df.history_len>0).astype(int)
df["has_sasrec_canonical_gpu_score"]=df.sasrec_canonical_gpu_score.notna().astype(int)
df["has_sasrec_small_gpu_score"]=df.sasrec_small_gpu_score.notna().astype(int)
df["has_twotower_gpu_score"]=df.twotower_gpu_score.notna().astype(int)
# USER-level split 70/15/15
uu=df.user_id.unique(); h=np.array([uhash(u) for u in uu]);
tr_u=set(uu[h<70]); va_u=set(uu[(h>=70)&(h<85)]); te_u=set(uu[h>=85])
df["split"]=df.user_id.map(lambda u:"tr" if u in tr_u else ("va" if u in va_u else "te"))
LOCAL=["als_score","als_rank","sasrec_cpu_d3b_score","sasrec_rank","log_pop","pub_year","warm_flag",
       "seen_flag","author_match","series_match","came_from_als","came_from_sasrec","came_from_pop",
       "history_len","has_sequence_history"]
GPU=["sasrec_canonical_gpu_score","sasrec_small_gpu_score","twotower_gpu_score","has_sasrec_canonical_gpu_score"]
VARIANTS={"local_no_gpu":LOCAL,"enriched_gpu":LOCAL+GPU,"enriched_no_twotower":LOCAL+[c for c in GPU if c!="twotower_gpu_score"]}
def groups(d): return d.groupby("user_id",sort=False).size().tolist()
def make_ds(d,feats):
    dd=d.sort_values("user_id"); return dd,lgb.Dataset(dd[feats].to_numpy(np.float32),label=dd.label.to_numpy(),group=groups(dd),feature_name=feats)
te=df[df.split=="te"].sort_values("user_id").reset_index(drop=True)
te_groups=te.groupby("user_id",sort=False)
nte=te.user_id.nunique()
cov=te.groupby("user_id").label.max().mean()                       # candidate coverage ceiling on test
def eval_scores(score_col_or_fn,is_model=False,feats=None):
    r20=r50=ndcg=0
    for u,g in te_groups:
        s = g[feats].to_numpy(np.float32) if is_model else None
        sc = score_col_or_fn.predict(s) if is_model else g[score_col_or_fn].fillna(-1e9).to_numpy()
        order=np.argsort(-sc); labs=g.label.to_numpy()[order]
        pos=np.where(labs==1)[0]; rk=pos[0] if len(pos) else 999
        if rk<20: r20+=1; ndcg+=1.0/math.log2(rk+2)
        if rk<50: r50+=1
    return r20/nte,r50/nte,ndcg/nte
res={}
# ALS-only baseline (rank by als_score)
res["ALS_only_ranking"]=dict(zip(["R@20","R@50","NDCG@20"],[round(x,4) for x in eval_scores("als_score")]))
models={}
params=dict(objective="lambdarank",metric="ndcg",ndcg_eval_at=[20],learning_rate=0.07,num_leaves=31,
            min_data_in_leaf=50,verbose=-1,seed=SEED,label_gain=[0,1])
for name,feats in VARIANTS.items():
    dtr,tr_ds=make_ds(df[df.split=="tr"],feats); dva,va_ds=make_ds(df[df.split=="va"],feats)
    m=lgb.train(params,tr_ds,num_boost_round=500,valid_sets=[va_ds],callbacks=[lgb.early_stopping(40,verbose=False)])
    models[name]=(m,feats)
    res[name]=dict(zip(["R@20","R@50","NDCG@20"],[round(x,4) for x in eval_scores(m,is_model=True,feats=feats)]))
    res[name]["best_iter"]=m.best_iteration
best=models["enriched_gpu"][0]; feats=models["enriched_gpu"][1]
imp=dict(sorted(zip(feats,[int(v) for v in best.feature_importance("gain")]),key=lambda kv:-kv[1]))
# segment: history vs no-history users (test)
def seg_eval(model,feats,mask_users):
    r20=0;n=0
    for u,g in te_groups:
        if u not in mask_users: continue
        n+=1; sc=model.predict(g[feats].to_numpy(np.float32)); order=np.argsort(-sc); labs=g.label.to_numpy()[order]
        pos=np.where(labs==1)[0]; rk=pos[0] if len(pos) else 999
        if rk<20: r20+=1
    return round(r20/max(n,1),4),n
# all eval LLO users have history>=2 (no zero-history segment); meaningful split = canonical-covered (len>=30) vs not
cov_u=set(te[te.has_sasrec_canonical_gpu_score==1].user_id); nocov_u=set(te[te.has_sasrec_canonical_gpu_score==0].user_id)
seg={"canonical_covered_hist_ge30":dict(zip(["R@20","n"],seg_eval(best,feats,cov_u))),
     "canonical_absent_hist_lt30":dict(zip(["R@20","n"],seg_eval(best,feats,nocov_u))),
     "note":"no zero-history users exist in eval LLO (all >=2 history items); split is by canonical coverage (history_len>=30)"}
als=res["ALS_only_ranking"]["R@20"]
out={"scope":"G12-B LambdaMART LTR on GPU-enriched candidate table","tag":"[BUILT — real data, CPU LightGBM]",
 "framing":"re-ranking recall WITHIN candidate set (coverage ceiling below); NOT full-catalog; user-level split",
 "n_test_users":int(nte),"candidate_coverage_ceiling_test":round(float(cov),4),
 "split":"user-level 70/15/15 (train/val/test)",
 "results":res,
 "lift_enriched_vs_ALSonly_R@20":round(res["enriched_gpu"]["R@20"]-als,4),
 "lift_enriched_vs_local_R@20":round(res["enriched_gpu"]["R@20"]-res["local_no_gpu"]["R@20"],4),
 "twotower_effect_R@20":round(res["enriched_gpu"]["R@20"]-res["enriched_no_twotower"]["R@20"],4),
 "feature_importance_gain":imp,"segment_enriched":seg,
 "honest_notes":["Candidate coverage ~%.1f%% caps achievable recall — ranker cannot recover golds absent from candidates."%(100*cov),
   "canonical GPU score present only for users with history (NaN else, surfaced via has_sasrec_canonical_gpu_score).",
   "Re-ranking recall, NOT full-catalog retrieval; no online/production/RiskFrame-gold claim."],
 "sec":round(time.time()-t0,1)}
json.dump(out,open(EVID+"/g12_ltr_metrics.json","w"),indent=2)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    order=["ALS_only_ranking","local_no_gpu","enriched_no_twotower","enriched_gpu"]
    vals=[res[n]["R@20"] for n in order]; cols=["#999","#4477aa","#ee9944","#ee7733"]
    fig,ax=plt.subplots(figsize=(9,5)); ax.bar(range(4),vals,color=cols)
    for i,v in enumerate(vals): ax.text(i,v+0.003,"%.3f"%v,ha="center")
    ax.axhline(cov,ls="--",c="green",label="candidate coverage ceiling %.3f"%cov)
    ax.set_xticks(range(4)); ax.set_xticklabels(["ALS-only\nrank","local LTR\n(no GPU)","enriched\n(no TwoTower)","enriched LTR\n(+GPU)"],fontsize=8)
    ax.set_ylabel("Recall@20 (within candidates)"); ax.legend(); ax.set_title("G12-B LTR variants (user-level test split)")
    plt.tight_layout(); plt.savefig(PLOTS+"/g12_ltr_comparison.png",dpi=110); plt.close()
    fi=list(imp.items())[:10][::-1]; fig,ax=plt.subplots(figsize=(9,5)); ax.barh([k for k,_ in fi],[v for _,v in fi],color="#ee7733")
    ax.set_title("G12-B enriched LTR feature importance (gain)"); plt.tight_layout(); plt.savefig(PLOTS+"/g12_ltr_feature_importance.png",dpi=110); plt.close()
except Exception as e: print("plot skipped",e)
print("coverage ceiling %.4f | ALS %.4f | local %.4f | enriched %.4f | enriched_noTT %.4f"%(
  cov,als,res["local_no_gpu"]["R@20"],res["enriched_gpu"]["R@20"],res["enriched_no_twotower"]["R@20"]))
print("lift enriched-vs-local",out["lift_enriched_vs_local_R@20"],"| TwoTower effect",out["twotower_effect_R@20"])
print("segment:",seg); print("top feats:",list(imp.items())[:6])
