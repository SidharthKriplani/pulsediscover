"""G6 step 2 — cold-lane recall at FULL cold population (all 5,929 heldout cold-gold users) +
creator reach/Gini at larger warm scale. Regenerates content-hybrid cold lists (same method as
c2_build_lists) for the full cold pop. Offline only; no new claims."""
import pandas as pd, numpy as np, os, json, pickle, time
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import sys; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from pulsediscover.feedback_loop import gini
OUT="/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim"; SP=os.path.join(OUT,"domain_splits"); t0=time.time()
A=pickle.load(open(os.path.join(OUT,"c2_als.pkl"),"rb")); X=A["X"];Y=A["Y"];i_uniq=A["i_uniq"];iidx=A["iidx"];uidx=A["uidx"]
Mp=pickle.load(open(os.path.join(OUT,"c2_meta.pkl"),"rb")); pre_items=Mp["pre_items"]
d2=json.load(open(os.path.join(OUT,"d2_stats.json"))); t2=d2["split_global_time"]["cutoffs"]["t2"]
he=pd.read_csv(os.path.join(SP,"heldout_eval.csv"),dtype={"user_id":"str","gold":"str","event_ts":"int64"})
cold_he=he[~he.gold.isin(iidx)]                                  # full cold-gold population
# pre-cutoff history for all needed users (cold users + a warm subset for reach)
tr=pd.read_csv(os.path.join(SP,"train.csv"),usecols=["user_id","book_id"],dtype=str)
hist=tr.groupby("user_id")["book_id"].apply(set).to_dict(); del tr
core=pd.read_csv(os.path.join(OUT,"domain_core_interactions.csv"),usecols=["user_id","book_id","event_ts"],dtype={"user_id":"str","book_id":"str","event_ts":"int64"})
cold_items=list(set(core[core.event_ts>=t2].book_id)-pre_items); del core
cm=pd.read_csv(os.path.join(OUT,"domain_content_meta.csv"),dtype=str).fillna("")
b2c=dict(zip(cm.book_id,cm.creator_id)); b2s=dict(zip(cm.book_id,cm.series_id)); b2y={b:(int(y) if y.isdigit() else 0) for b,y in zip(cm.book_id,cm.publication_year)}
text=dict(zip(cm.book_id,(cm.title+" "+cm.shelves+" "+cm.desc)))
a2cold=defaultdict(list); s2cold=defaultdict(list)
for it in cold_items:
    a=b2c.get(it,""); s=b2s.get(it,"")
    if a: a2cold[a].append(it)
    if s: s2cold[s].append(it)
for a in a2cold: a2cold[a].sort(key=lambda x:-b2y.get(x,0))
for s in s2cold: s2cold[s].sort(key=lambda x:-b2y.get(x,0))
cold_set=set(cold_items)
hist_items=set(i for u in cold_he.user_id for i in hist.get(u,()))
docs=list(cold_set|hist_items); ridx={it:i for i,it in enumerate(docs)}
vec=TfidfVectorizer(max_features=20000,stop_words="english"); Mtx=vec.fit_transform([text.get(it,"") for it in docs])
cold_mat=Mtx[np.array([ridx[it] for it in cold_items])]
print("setup %.1fs cold_items=%d cold_users=%d"%(time.time()-t0,len(cold_items),len(cold_he)))
# cold-lane eval at full cold population
lane20=lane50=pol90=pol80=valid=0
cre_exp=defaultdict(int); seen_cold=set(); seen_cold_cre=set()
for u,g in zip(cold_he.user_id,cold_he.gold):
    h=hist.get(u,set()); r=[ridx[i] for i in h if i in ridx]
    prof=sp.csr_matrix(Mtx[r].mean(axis=0)) if r else sp.csr_matrix((1,Mtx.shape[1]))
    sc=np.asarray(prof.dot(cold_mat.T).todense()).ravel(); order=np.argsort(-sc); tf=[cold_items[j] for j in order[:200]]
    auth=set(b2c.get(b,"") for b in h)-{""}; ser=set(b2s.get(b,"") for b in h)-{""}
    sa=[]; se=set()
    for a in auth:
        for it in a2cold.get(a,()):
            if it not in se: sa.append(it); se.add(it)
    ss=[]; se2=set()
    for s in ser:
        for it in s2cold.get(s,()):
            if it not in se2: ss.append(it); se2.add(it)
    rr=defaultdict(float)
    for lst in (sa,ss,tf):
        for rk,it in enumerate(lst): rr[it]+=1.0/(60+rk)
    cl=[i for i,_ in sorted(rr.items(),key=lambda t:-t[1])][:200]
    valid+=1
    if g in cl[:20]: lane20+=1
    if g in cl[:50]: lane50+=1
    if g in cl[:2]: pol90+=1
    if g in cl[:4]: pol80+=1
    for it in cl[:2]:
        cr=b2c.get(it,"unk"); cre_exp[cr]+=1; seen_cold.add(it); seen_cold_cre.add(cr)
out={"scale":{"full_cold_gold_users":int(len(cold_he)),"sampled_cold_eval":5000,
   "note":"cold-gold population is only %d; the 5,000 sample was ~%.0f%% of full cold population"%(len(cold_he),100*5000/len(cold_he))},
 "cold_lane_fullpop":{"content_lane_R@20":round(lane20/valid,4),"content_lane_R@50":round(lane50/valid,4),
   "q90_10_policy_cold_R@20(2 slots)":round(pol90/valid,4),"q80_20_policy_cold_R@20(4 slots)":round(pol80/valid,4)},
 "cold_creator_reach_q90":{"distinct_cold_items_in_2slots":len(seen_cold),"distinct_cold_creators":len(seen_cold_cre),
   "cold_slot_creator_gini":round(gini(np.array(list(cre_exp.values()),float)),4)},
 "sec":round(time.time()-t0,1)}
json.dump(out,open(os.path.join(OUT,"g6_cold_health.json"),"w"),indent=2)
print(json.dumps(out,indent=2))
