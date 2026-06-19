"""D3A step 1 — materialize global-time train/val/test split files + constrained
leave-last-out eval set (held-out = each user's LAST interaction when ts>=t2). Domain-scoped
(Goodreads fantasy_paranormal). No modeling."""
import pandas as pd, numpy as np, json, os
OUT="/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim"
SP=os.path.join(OUT,"domain_splits"); os.makedirs(SP, exist_ok=True)
SEED=20260616; EVAL_SAMPLE=5000
d2=json.load(open(os.path.join(OUT,"d2_stats.json")))
t1=d2["split_global_time"]["cutoffs"]["t1"]; t2=d2["split_global_time"]["cutoffs"]["t2"]
core=pd.read_csv(os.path.join(OUT,"domain_core_interactions.csv"),
    dtype={"user_id":"str","book_id":"str","rating":"int16","rating0_flag":"int8","event_ts":"int64"})
# global-time split files
core[core.event_ts<t1].to_csv(os.path.join(SP,"train.csv"),index=False)
core[(core.event_ts>=t1)&(core.event_ts<t2)].to_csv(os.path.join(SP,"val.csv"),index=False)
core[core.event_ts>=t2].to_csv(os.path.join(SP,"test.csv"),index=False)
# constrained leave-last-out: each user's last row; held-out if ts>=t2
last_idx=core.groupby("user_id").event_ts.idxmax()
last=core.loc[last_idx,["user_id","book_id","event_ts"]]
held=last[last.event_ts>=t2].rename(columns={"book_id":"gold"}).reset_index(drop=True)
held[["user_id","gold","event_ts"]].to_csv(os.path.join(SP,"heldout_eval.csv"),index=False)
samp=held.sample(min(EVAL_SAMPLE,len(held)),random_state=SEED).reset_index(drop=True)
samp[["user_id","gold","event_ts"]].to_csv(os.path.join(SP,"eval_sample.csv"),index=False)
out={"t1":t1,"t2":t2,"train_rows":int((core.event_ts<t1).sum()),
     "val_rows":int(((core.event_ts>=t1)&(core.event_ts<t2)).sum()),
     "test_rows":int((core.event_ts>=t2).sum()),
     "heldout_users":int(len(held)),"eval_sample":int(len(samp))}
json.dump(out,open(os.path.join(OUT,"d3a_split_meta.json"),"w"),indent=2)
print("SPLIT_DONE",out)
