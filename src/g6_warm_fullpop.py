"""G6 step 1 — full-population WARM-lane scale test. Build per-user pre-cutoff history from train,
score ALS for ALL heldout users with a warm gold, exclude history, compute ALS-only vs 90/10
warm-lane recall@20/50 at full scale. Also size the cold-gold pool. Offline only; no new claims."""
import pandas as pd, numpy as np, os, json, pickle, time
OUT="/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim"; SP=os.path.join(OUT,"domain_splits"); t0=time.time()
A=pickle.load(open(os.path.join(OUT,"c2_als.pkl"),"rb")); X=A["X"];Y=A["Y"];i_uniq=A["i_uniq"];iidx=A["iidx"];uidx=A["uidx"]
he=pd.read_csv(os.path.join(SP,"heldout_eval.csv"),dtype={"user_id":"str","gold":"str","event_ts":"int64"})
# build pre-cutoff history from train (global-time split => train is all pre-cutoff)
tr=pd.read_csv(os.path.join(SP,"train.csv"),usecols=["user_id","book_id"],dtype=str)
hist=tr.groupby("user_id")["book_id"].apply(set).to_dict(); del tr
print("hist users=%d built %.1fs"%(len(hist),time.time()-t0))
warm=he[he.gold.isin(iidx)]; cold=he[~he.gold.isin(iidx)]
print("heldout warm-gold=%d cold-gold=%d total=%d"%(len(warm),len(cold),len(he)))
users=list(warm.user_id); golds=list(warm.gold); rows=[uidx.get(u,-1) for u in users]
hit_als20=hit_als50=hit_q18=hit_q48=valid=0; B=1000
for s in range(0,len(rows),B):
    rb=rows[s:s+B]; sc=X[[r if r>=0 else 0 for r in rb]]@Y.T
    for bi,r in enumerate(rb):
        u=users[s+bi]; g=golds[s+bi]; gj=iidx.get(g)
        if r<0 or gj is None: continue
        valid+=1; row=sc[bi]
        for b in hist.get(u,()):
            j=iidx.get(b)
            if j is not None and j!=gj: row[j]=-1e9   # exclude history (keep gold scorable)
        # top-50 by partial sort
        top50=np.argpartition(-row,50)[:50]; top50=top50[np.argsort(-row[top50])]
        pos=np.where(top50==gj)[0]
        rank=pos[0] if len(pos) else 999
        if rank<20: hit_als20+=1
        if rank<50: hit_als50+=1
        if rank<18: hit_q18+=1      # 90/10 warm lane occupies 18 slots in a 20-slate
        if rank<48: hit_q48+=1      # 90/10 warm lane occupies 48 slots in a 50-slate
als20=hit_als20/valid; als50=hit_als50/valid; q20=hit_q18/valid; q50=hit_q48/valid
out={"scale":{"heldout_total":len(he),"heldout_warm_gold":len(warm),"heldout_cold_gold":len(cold),
   "warm_eval_valid":valid,"sampled_warm_eval":5000,"expansion_x":round(valid/5000,2)},
 "warm_recall_fullpop":{"als_only_R@20":round(als20,4),"als_only_R@50":round(als50,4),
   "q90_10_warmlane_R@20":round(q20,4),"q90_10_warmlane_R@50":round(q50,4),
   "warm_rel_loss@20":round((q20-als20)/als20,4) if als20>0 else None,
   "warm_rel_loss@50":round((q50-als50)/als50,4) if als50>0 else None},
 "sec":round(time.time()-t0,1)}
json.dump(out,open(os.path.join(OUT,"g6_warm_fullpop.json"),"w"),indent=2)
print(json.dumps(out,indent=2))
