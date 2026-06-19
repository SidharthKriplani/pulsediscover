"""D2a — iterative k-core (users>=5, items>=10) on the compact interactions, keep all
languages, keep rating-0 (flagged). Writes the CPU-safe core working set + core books meta
+ top-40k item-vocab view + d2_partial.json. No modeling."""
import pandas as pd, numpy as np, json, os, time
OUT="/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim"
t0=time.time()
df=pd.read_csv(os.path.join(OUT,"domain_interactions.csv"),
               dtype={"user_id":"str","book_id":"str","rating":"int16","event_ts":"int64"})
u_codes,u_uniq=pd.factorize(df["user_id"]); i_codes,i_uniq=pd.factorize(df["book_id"])
rating=df["rating"].to_numpy(); ts=df["event_ts"].to_numpy(); del df
nU=len(u_uniq); nI=len(i_uniq); N=len(u_codes)
alive=np.ones(N,dtype=bool)
it=0
while True:
    it+=1
    uc=np.bincount(u_codes[alive],minlength=nU); ic=np.bincount(i_codes[alive],minlength=nI)
    na=alive & (uc[u_codes]>=5) & (ic[i_codes]>=10)
    if int(na.sum())==int(alive.sum()): break
    alive=na
cu=u_codes[alive]; ci=i_codes[alive]; cr=rating[alive]; ct=ts[alive]
core_users=np.unique(cu); core_items=np.unique(ci)
# write core interactions (original ids), rating0 flag
uid=u_uniq.to_numpy()[cu]; bid=i_uniq.to_numpy()[ci]
core=pd.DataFrame({"user_id":uid,"book_id":bid,"rating":cr,
                   "rating0_flag":(cr==0).astype("int8"),"event_ts":ct})
core.to_csv(os.path.join(OUT,"domain_core_interactions.csv"),index=False)
# core book ids
core_bids=set(i_uniq.to_numpy()[core_items].tolist())
# core books meta + creator/series/lang coverage
meta=pd.read_csv(os.path.join(OUT,"domain_books_meta.csv"),
                 dtype={"book_id":"str","creator_id":"str","series_id":"str",
                        "title":"str","language_code":"str"}).fillna("")
cmeta=meta[meta["book_id"].isin(core_bids)]
cmeta.to_csv(os.path.join(OUT,"domain_core_books_meta.csv"),index=False)
creators=set(cmeta.loc[cmeta["creator_id"]!="","creator_id"]);
series=set(cmeta.loc[cmeta["series_id"]!="","series_id"])
langcnt=cmeta["language_code"].replace("","(none)").value_counts().head(10).to_dict()
# top-40k item vocab by core interaction count
cnt=np.bincount(ci); order=np.argsort(-cnt)
top=order[:40000]; top=top[cnt[top]>0]
pd.DataFrame({"book_id":i_uniq.to_numpy()[top],"interactions":cnt[top]}).to_csv(
    os.path.join(OUT,"domain_sasrec_item_vocab_top40k.csv"),index=False)
percu=np.bincount(cu); percu=percu[percu>0]
partial={
 "kcore_iterations":it,"kcore_thresholds":"users>=5 & items>=10 (iterative)",
 "core_interactions":int(alive.sum()),
 "core_users":int(len(core_users)),"core_items":int(len(core_items)),
 "core_unique_creators":int(len(creators)),"core_unique_series":int(len(series)),
 "density_mean":round(float(percu.mean()),2),"density_median":int(np.median(percu)),
 "density_p90":int(np.quantile(percu,0.9)),"density_max":int(percu.max()),
 "rating0_core_rows":int((cr==0).sum()),"rating0_core_pct":round(100*float((cr==0).mean()),2),
 "rating_dist_core":{int(k):int(v) for k,v in zip(*np.unique(cr,return_counts=True))},
 "core_date_min":time.strftime("%Y-%m-%d",time.gmtime(int(ct.min()))),
 "core_date_max":time.strftime("%Y-%m-%d",time.gmtime(int(ct.max()))),
 "core_lang_dist_items_top10":{str(k):int(v) for k,v in langcnt.items()},
 "top40k_vocab_items":int(len(top)),
 "top40k_coverage_pct_of_core_interactions":round(100*float(cnt[top].sum())/float(alive.sum()),2),
 "d2a_sec":round(time.time()-t0,1),
}
json.dump(partial,open(os.path.join(OUT,"d2_partial.json"),"w"),indent=2)
print("D2A_DONE",partial["core_interactions"],"int",partial["core_users"],"u",partial["core_items"],"i",round(time.time()-t0,1),"s")
