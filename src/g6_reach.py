"""G6 step 3 — creator/catalog reach + Gini at full warm scale. Build warm top-20 lists for all
heldout warm-gold users; accumulate creator/item exposure for ALS-only (top-20) vs 90/10 warm-lane
(top-18); compare distinct items/creators + creator Gini vs H1's 2.4k-user sample. Offline only."""
import pandas as pd, numpy as np, os, json, pickle, time
from collections import defaultdict
import sys; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from pulsediscover.feedback_loop import gini
OUT="/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim"; SP=os.path.join(OUT,"domain_splits"); t0=time.time()
A=pickle.load(open(os.path.join(OUT,"c2_als.pkl"),"rb")); X=A["X"];Y=A["Y"];i_uniq=A["i_uniq"];iidx=A["iidx"];uidx=A["uidx"]
cm=pd.read_csv(os.path.join(OUT,"domain_content_meta.csv"),dtype=str).fillna(""); b2c=dict(zip(cm.book_id,cm.creator_id))
he=pd.read_csv(os.path.join(SP,"heldout_eval.csv"),dtype=str); warm=he[he.gold.isin(iidx)]
tr=pd.read_csv(os.path.join(SP,"train.csv"),usecols=["user_id","book_id"],dtype=str)
hist=tr.groupby("user_id")["book_id"].apply(set).to_dict(); del tr
users=[u for u in warm.user_id if uidx.get(u,-1)>=0]; rows=[uidx[u] for u in users]
def eff(c):
    a=np.array([v for v in c.values() if v>0],float); p=a/a.sum(); return float(np.exp(-(p*np.log(p)).sum()))
ie_a=defaultdict(int); ce_a=defaultdict(int); ie_q=defaultdict(int); ce_q=defaultdict(int); B=1000
for s in range(0,len(rows),B):
    rb=rows[s:s+B]; sc=X[rb]@Y.T
    for bi,r in enumerate(rb):
        u=users[s+bi]; row=sc[bi]
        for b in hist.get(u,()):
            j=iidx.get(b)
            if j is not None: row[j]=-1e9
        top=np.argpartition(-row,20)[:20]; top=top[np.argsort(-row[top])]; items=[i_uniq[t] for t in top]
        for rk,it in enumerate(items):
            cr=b2c.get(it,"unk")
            ie_a[it]+=1; ce_a[cr]+=1                       # als-only top-20
            if rk<18: ie_q[it]+=1; ce_q[cr]+=1             # 90/10 warm lane = top-18
out={"scale":{"warm_users":len(users),"h1_sample_users":2400,"expansion_x":round(len(users)/2400,2)},
 "warm_reach_fullpop":{
   "als_only":{"distinct_items":len(ie_a),"distinct_creators":len(ce_a),"creator_gini":round(gini(np.array(list(ce_a.values()),float)),4),"eff_creators":round(eff(ce_a),1)},
   "q90_10_warmlane":{"distinct_items":len(ie_q),"distinct_creators":len(ce_q),"creator_gini":round(gini(np.array(list(ce_q.values()),float)),4),"eff_creators":round(eff(ce_q),1)}},
 "sec":round(time.time()-t0,1)}
json.dump(out,open(os.path.join(OUT,"g6_reach.json"),"w"),indent=2)
print(json.dumps(out,indent=2))
