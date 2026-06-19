"""C1 step 2 — cold-start eval set (global-time test window, items NEW vs pre-test) + same-author
/ same-series content generators + ALS-f64 structural-zero contrast. Saves eval sample + histories
+ partial metrics for the TF-IDF/report steps."""
import pandas as pd, numpy as np, os, json, pickle, time
from collections import defaultdict
OUT="/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim"; SP=os.path.join(OUT,"domain_splits")
SEED=20260616; SAMPLE=5000; KS=[20,50,100,200]; t0=time.time()
d2=json.load(open(os.path.join(OUT,"d2_stats.json"))); t2=d2["split_global_time"]["cutoffs"]["t2"]
core=pd.read_csv(os.path.join(OUT,"domain_core_interactions.csv"),usecols=["user_id","book_id","event_ts"],
                 dtype={"user_id":"str","book_id":"str","event_ts":"int64"})
cm=pd.read_csv(os.path.join(OUT,"domain_content_meta.csv"),dtype=str).fillna("")
b2c=dict(zip(cm.book_id,cm.creator_id)); b2s=dict(zip(cm.book_id,cm.series_id))
def yr(x):
    try: return int(x)
    except: return 0
b2y={b:yr(y) for b,y in zip(cm.book_id,cm.publication_year)}
pre=core[core.event_ts<t2]; test=core[core.event_ts>=t2]
pre_items=set(pre.book_id); test_items=set(test.book_id)
cold_items=test_items-pre_items
pre_creators=set(b2c.get(b,"") for b in pre_items)-{""}; pre_series=set(b2s.get(b,"") for b in pre_items)-{""}
cold_creators=set(b2c.get(b,"") for b in cold_items)-{""}; cold_series=set(b2s.get(b,"") for b in cold_items)-{""}
# cold candidate maps (author/series -> cold items), ranked by pub_year desc
a2cold=defaultdict(list); s2cold=defaultdict(list)
for it in cold_items:
    a=b2c.get(it,""); s=b2s.get(it,"")
    if a: a2cold[a].append(it)
    if s: s2cold[s].append(it)
for a in a2cold: a2cold[a].sort(key=lambda x:-b2y.get(x,0))
for s in s2cold: s2cold[s].sort(key=lambda x:-b2y.get(x,0))
# pre-test histories
hsub=pre.groupby("user_id")["book_id"].apply(list).to_dict()
# cold eval interactions: test rows on cold items where user has pre-test history
ce=test[test.book_id.isin(cold_items)]; ce=ce[ce.user_id.isin(hsub.keys())]
samp=ce.sample(min(SAMPLE,len(ce)),random_state=SEED)[["user_id","book_id"]].rename(columns={"book_id":"gold"})
samp.to_csv(os.path.join(SP,"cold_eval_sample.csv"),index=False)
eval_hist={u:hsub[u] for u in samp.user_id}
pickle.dump(eval_hist,open(os.path.join(OUT,"c1_eval_hist.pkl"),"wb"))
# same-author / same-series generators + ALS cold (structural 0)
def author_cands(h):
    auth=set(b2c.get(b,"") for b in h)-{""}; seen=set(); out=[]
    for a in auth:
        for it in a2cold.get(a,()):
            if it not in seen: out.append(it); seen.add(it)
    return out
def series_cands(h):
    ser=set(b2s.get(b,"") for b in h)-{""}; seen=set(); out=[]
    for s in ser:
        for it in s2cold.get(s,()):
            if it not in seen: out.append(it); seen.add(it)
    return out
res={"same_author":{k:[] for k in KS},"same_series":{k:[] for k in KS},"als_f64":{k:[] for k in KS}}
gold_author_warm=0; surfaced_items=set(); surfaced_auth=set()
for u,g in zip(samp.user_id,samp.gold):
    h=hsub[u]
    sa=author_cands(h); ss=series_cands(h)
    surfaced_items.update(sa[:200]); surfaced_auth.update(b2c.get(i,"") for i in sa[:200])
    if b2c.get(g,"") in (set(b2c.get(b,"") for b in h)-{""}): gold_author_warm+=1
    for k in KS:
        res["same_author"][k].append(1.0 if g in sa[:k] else 0.0)
        res["same_series"][k].append(1.0 if g in ss[:k] else 0.0)
        res["als_f64"][k].append(0.0)  # cold items absent from ALS factor space -> structural 0
part={"t2":t2,"n_pre_items":len(pre_items),"n_test_items":len(test_items),
      "n_cold_items":len(cold_items),"n_cold_creators_new":len(cold_creators-pre_creators),
      "n_cold_series_new":len(cold_series-pre_series),
      "n_cold_test_interactions":int(len(ce)),"eval_sample":int(len(samp)),
      "gold_author_warm":gold_author_warm,"gold_author_warm_pct":round(100*gold_author_warm/len(samp),2),
      "res":{m:{str(k):res[m][k] for k in KS} for m in res},
      "surfaced_cold_items_sameauthor":len(surfaced_items),"surfaced_cold_authors":len(surfaced_auth-{""}),
      "sec":round(time.time()-t0,1)}
json.dump(part,open(os.path.join(OUT,"c1_partial.json"),"w"))
print("COLDEVAL cold_items=%d new_creators=%d new_series=%d cold_int=%d sample=%d gold_author_warm%%=%.1f"%(
 len(cold_items),len(cold_creators-pre_creators),len(cold_series-pre_series),len(ce),len(samp),part["gold_author_warm_pct"]))
print(" same_author R@20=%.4f R@50=%.4f R@200=%.4f | same_series R@20=%.4f"%(
 np.mean(res["same_author"][20]),np.mean(res["same_author"][50]),np.mean(res["same_author"][200]),np.mean(res["same_series"][20])))
