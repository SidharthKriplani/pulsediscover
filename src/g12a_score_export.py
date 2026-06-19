"""G12-A — candidate-level feature table for LTR, on the 5k LLO eval_sample users.
Candidate sources: ALS f64 top-N, D3B SASRec(CPU) top-N, popularity top-N (+ hard negatives) — deduped
with source flags. Per-candidate features computed LOCALLY (ALS score/rank, CPU-SASRec score/rank,
popularity, recency, warm flag, seen flag, author/series match). GPU canonical/small/two-tower score
columns are written as NaN placeholders (those weights live in Colab) — to be filled by a Colab
score-export run. Label = 1 if candidate == held-out gold. No LTR training here; no fake claims."""
import os, json, time, pickle, math, numpy as np, pandas as pd, torch
import sys; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from pulsediscover import sasrec as S
OUT="/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim"; SP=os.path.join(OUT,"domain_splits")
EVID="/sessions/festive-quirky-cannon/mnt/pulsediscover/outputs/evidence"; MODELS="/sessions/festive-quirky-cannon/mnt/pulsediscover/outputs/_models"
NALS=60; NSAS=60; NPOP=30; MAXLEN=30; SEED=20260616; t0=time.time(); rng=np.random.default_rng(SEED)
A=pickle.load(open(os.path.join(OUT,"c2_als.pkl"),"rb")); X=A["X"];Y=A["Y"];uidx=A["uidx"];iidx=A["iidx"];i_uniq=A["i_uniq"]
meta=pickle.load(open(os.path.join(OUT,"d3b_seq_meta.pkl"),"rb")); item2idx=meta["item2idx"]; idx2item=meta["idx2item"]; n_items=meta["n_items"]; eval_seq=meta["eval_seq"]
st=torch.load(os.path.join(MODELS,"sasrec_domain.pt"),weights_only=False)
sas=S.SASRec(n_items,d=48,maxlen=MAXLEN,nblocks=1,nheads=1,dropout=0.2); sas.load_state_dict(st["model"]); sas.eval()
cm=pd.read_csv(os.path.join(OUT,"domain_content_meta.csv"),dtype=str).fillna("")
b2c=dict(zip(cm.book_id,cm.creator_id)); b2s=dict(zip(cm.book_id,cm.series_id)); b2y={b:(int(y) if y.isdigit() else 0) for b,y in zip(cm.book_id,cm.publication_year)}
pop=pd.read_csv(os.path.join(SP,"train.csv"),usecols=["book_id"],dtype=str).book_id.value_counts()
poptop=list(pop.index[:NPOP]); popd=pop.to_dict()
warm_set=set(iidx.keys())
es=pd.read_csv(os.path.join(SP,"eval_sample.csv"),dtype=str); users=es.user_id.tolist(); golds=dict(zip(es.user_id,es.gold))
def lp(s): s=s[-MAXLEN:]; return [0]*(MAXLEN-len(s))+list(s)
rows=[]; B=500; n_pos=0; src_counts={"als":0,"sasrec":0,"pop":0}
for s0 in range(0,len(users),B):
    ub=users[s0:s0+B]
    urows=[uidx.get(u,-1) for u in ub]
    alssc=X[[r if r>=0 else 0 for r in urows]]@Y.T                       # (b,n_warm_items) over i_uniq order
    xb=torch.tensor([lp(eval_seq.get(u,[])) for u in ub],dtype=torch.long)
    with torch.no_grad(): sassc=sas.score_last(xb).numpy()              # (b, n_items)
    for bi,u in enumerate(ub):
        gold=golds[u]; hist=set(eval_seq.get(u,[]))                      # item-idx history
        hist_books=set(idx2item[i] for i in hist if i< len(idx2item) and idx2item[i] is not None)
        uauth=set(b2c.get(b,"") for b in hist_books)-{""}; user_ser=set(b2s.get(b,"") for b in hist_books)-{""}
        # ALS candidates (warm)
        als_cands=[]
        if urows[bi]>=0:
            sc=alssc[bi]; top=np.argpartition(-sc,NALS)[:NALS]; top=top[np.argsort(-sc[top])]; als_cands=[i_uniq[t] for t in top]
            als_score_of={i_uniq[t]:float(sc[t]) for t in top}; als_rank_of={b:r for r,b in enumerate(als_cands)}
        else: als_score_of={}; als_rank_of={}
        # SASRec candidates
        sv=sassc[bi]; sv[0]=-1e9; stop=np.argpartition(-sv,NSAS)[:NSAS]; stop=stop[np.argsort(-sv[stop])]
        sas_cands=[idx2item[t] for t in stop if idx2item[t] is not None]
        sas_rank_of={b:r for r,b in enumerate(sas_cands)}
        cands=list(dict.fromkeys(als_cands+sas_cands+poptop))           # dedupe, preserve order
        for b in cands:
            f_als = als_score_of.get(b, float(alssc[bi][iidx[b]]) if (urows[bi]>=0 and b in iidx) else np.nan)
            f_sas = float(sv[item2idx[b]]) if b in item2idx else np.nan
            ca=1 if b in als_rank_of else 0; cs=1 if b in sas_rank_of else 0; cp=1 if b in poptop else 0
            src_counts["als"]+=ca; src_counts["sasrec"]+=cs; src_counts["pop"]+=cp
            lab=1 if b==gold else 0; n_pos+=lab
            rows.append((u,b,lab,
                round(f_als,5) if f_als==f_als else np.nan, als_rank_of.get(b,NALS),
                round(f_sas,5) if f_sas==f_sas else np.nan, sas_rank_of.get(b,NSAS),
                round(math.log1p(popd.get(b,0)),4), b2y.get(b,0),
                1 if b in warm_set else 0, 1 if b in hist_books else 0,
                1 if b2c.get(b,"") in uauth else 0, 1 if b2s.get(b,"") in user_ser else 0,
                ca,cs,cp,
                np.nan,np.nan,np.nan))                                   # GPU placeholders
cols=["user_id","item_id","label","als_score","als_rank","sasrec_cpu_d3b_score","sasrec_rank",
      "log_pop","pub_year","warm_flag","seen_flag","author_match","series_match",
      "came_from_als","came_from_sasrec","came_from_pop",
      "sasrec_canonical_gpu_score","sasrec_small_gpu_score","twotower_gpu_score"]
df=pd.DataFrame(rows,columns=cols)
seen_leak=int(((df.seen_flag==1)&(df.label==1)).sum())                  # gold should never be a seen item (LLO)
ncand=len(df); nuser=df.user_id.nunique(); cov=df.groupby("user_id").label.max().mean()
df.to_csv(os.path.join(EVID,"g12_candidate_feature_table.csv.gz"),index=False,compression="gzip")
schema={"columns":cols,"label":"1 if candidate==held-out gold else 0","local_features":[c for c in cols if "gpu" not in c],
 "pending_gpu_features":["sasrec_canonical_gpu_score","sasrec_small_gpu_score","twotower_gpu_score"],
 "missing_handling":"als_score NaN for cold/no-factor items; sasrec score NaN if item not in 40k vocab; GPU columns all NaN (pending Colab export)",
 "candidate_sources":{"als_topN":NALS,"sasrec_cpu_topN":NSAS,"pop_topN":NPOP}}
json.dump(schema,open(os.path.join(EVID,"g12_candidate_feature_schema.json"),"w"),indent=2)
summary={"n_users":int(nuser),"n_candidate_rows":int(ncand),"avg_candidates_per_user":round(ncand/nuser,1),
 "positive_coverage_rate":round(float(cov),4),"n_positives":int(n_pos),
 "source_contribution_counts":src_counts,
 "warm_gold_users":int(es.gold.isin(warm_set).sum()),
 "leakage_checks":{"gold_in_seen_history_rows":seen_leak,"expected":0,"note":"LLO holds out gold so it must not appear as a seen item"},
 "local_sasrec_model":"D3B sasrec_domain.pt (d48/1blk, R@20~0.036) — GPU canonical/small scores pending",
 "limitations":["GPU canonical/small/two-tower scores are NaN placeholders (weights in Colab) — fill via score-export run.",
   "Content/cold-lane candidate source not added (eval_sample is ~87% warm-gold); author/series match capture content signal.",
   "Positive coverage caps any ranker's achievable recall."],
 "sec":round(time.time()-t0,1)}
json.dump(summary,open(os.path.join(EVID,"g12_score_export_summary.json"),"w"),indent=2)
print(json.dumps(summary,indent=2))
