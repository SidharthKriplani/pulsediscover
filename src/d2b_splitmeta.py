"""D2b — leakage-safe SPLIT METADATA ONLY on the k-cored core working set. Global-time split,
constrained leave-last-out feasibility, cold-start item/user/creator/series counts by split.
No modeling. Merges d2_partial.json -> d2_stats.json + D2_DONE."""
import pandas as pd, numpy as np, json, os, time
OUT="/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim"
t0=time.time()
df=pd.read_csv(os.path.join(OUT,"domain_core_interactions.csv"),
               dtype={"user_id":"str","book_id":"str","rating":"int16","rating0_flag":"int8","event_ts":"int64"})
meta=pd.read_csv(os.path.join(OUT,"domain_core_books_meta.csv"),
                 dtype={"book_id":"str","creator_id":"str","series_id":"str","title":"str","language_code":"str"}).fillna("")
cmap=dict(zip(meta.book_id,meta.creator_id)); smap=dict(zip(meta.book_id,meta.series_id)); lmap=dict(zip(meta.book_id,meta.language_code))
ts=df.event_ts.to_numpy()
t1=int(np.quantile(ts,0.8)); t2=int(np.quantile(ts,0.9))
train=ts<t1; val=(ts>=t1)&(ts<t2); test=ts>=t2
def d(x): return time.strftime("%Y-%m-%d",time.gmtime(int(x)))
def split_counts(m):
    s=df[m]
    return {"interactions":int(m.sum()),"users":int(s.user_id.nunique()),"items":int(s.book_id.nunique()),
            "creators":int(s.book_id.map(cmap).replace("",np.nan).nunique()),
            "series":int(s.book_id.map(smap).replace("",np.nan).nunique())}
splits={"train":split_counts(train),"val":split_counts(val),"test":split_counts(test),
        "cutoffs":{"t1":t1,"t2":t2,"t1_date":d(t1),"t2_date":d(t2)}}
# cold-start vs TRAIN
def setof(col,m): return set(df.loc[m,col])
tr_i=setof("book_id",train); tr_u=setof("user_id",train)
tr_c=set(df.loc[train,"book_id"].map(cmap))-{""}; tr_s=set(df.loc[train,"book_id"].map(smap))-{""}
def cold(m):
    vi=setof("book_id",m); vu=setof("user_id",m)
    vc=set(df.loc[m,"book_id"].map(cmap))-{""}; vs=set(df.loc[m,"book_id"].map(smap))-{""}
    return {"new_items":len(vi-tr_i),"new_users":len(vu-tr_u),
            "new_creators":len(vc-tr_c),"new_series":len(vs-tr_s),
            "pct_interactions_on_new_items":round(100*float(df.loc[m,"book_id"].isin(vi-tr_i).mean()),2)}
coldstart={"val_vs_train":cold(val),"test_vs_train":cold(test)}
# constrained leave-last-out feasibility
last=df.groupby("user_id").event_ts.max().to_numpy()
llo={"users_total":int(len(last)),
     "users_last_in_test(>=t2)":int((last>=t2).sum()),
     "users_last_in_val(t1..t2)":int(((last>=t1)&(last<t2)).sum())}
# language distribution by interaction
lang_by_int=df.book_id.map(lmap).replace("","(none)").value_counts().head(10).to_dict()
# size/memory estimates
files={f:os.path.getsize(os.path.join(OUT,f)) for f in
       ["domain_core_interactions.csv","domain_core_books_meta.csv","domain_sasrec_item_vocab_top40k.csv"]}
n=len(df); nU=df.user_id.nunique(); nI=df.book_id.nunique()
mem_est={"interactions_int32_codes_MB":round(n*3*4/1e6,1),
         "sasrec_item_emb_d48_MB":round(min(nI,40000)*48*4/1e6,1),
         "als_factors16_MB":round((nU+nI)*16*8/1e6,1)}
partial=json.load(open(os.path.join(OUT,"d2_partial.json")))
stats=dict(partial); stats.update({
 "split_global_time":splits,"coldstart_vs_train":coldstart,
 "leave_last_out_feasibility":llo,"core_lang_dist_by_interaction_top10":{str(k):int(v) for k,v in lang_by_int.items()},
 "core_file_sizes_bytes":files,"working_memory_estimate":mem_est,"d2b_sec":round(time.time()-t0,1)})
json.dump(stats,open(os.path.join(OUT,"d2_stats.json"),"w"),indent=2)
open(os.path.join(OUT,"D2_DONE"),"w").write("done")
print("D2B_DONE t1=%s t2=%s | train/val/test int=%d/%d/%d | llo_test=%d"%(
 d(t1),d(t2),splits["train"]["interactions"],splits["val"]["interactions"],splits["test"]["interactions"],
 llo["users_last_in_test(>=t2)"]))
