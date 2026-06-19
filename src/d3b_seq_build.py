"""D3B — build domain SASRec sequences (top-40k vocab) from eval-train; cache training arrays
(last-position next-item) + eval-user sequences for resumable training/eval. No training here."""
import pandas as pd, numpy as np, os, pickle, json, time
OUT="/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim"; SP=os.path.join(OUT,"domain_splits")
MAXLEN=30; t0=time.time()
vocab=pd.read_csv(os.path.join(OUT,"domain_sasrec_item_vocab_top40k.csv"),dtype={"book_id":"str"})
item2idx={b:i+1 for i,b in enumerate(vocab.book_id)}   # 0 = pad
n_items=len(item2idx)+1
core=pd.read_csv(os.path.join(OUT,"domain_core_interactions.csv"),usecols=["user_id","book_id","event_ts"],
                 dtype={"user_id":"str","book_id":"str","event_ts":"int64"})
held=pd.read_csv(os.path.join(SP,"heldout_eval.csv"),dtype=str)
tr=core[~(core.user_id+"|"+core.book_id).isin(set(held.user_id+"|"+held.gold))]; del core
tr=tr[tr.book_id.isin(item2idx)].sort_values(["user_id","event_ts"])
seqs=tr.groupby("user_id")["book_id"].apply(lambda s:[item2idx[b] for b in s]).to_dict(); del tr
def leftpad(s): s=s[-MAXLEN:]; return [0]*(MAXLEN-len(s))+s
Xl=[]; Yl=[]
for u,s in seqs.items():
    if len(s)<2: continue
    Xl.append(leftpad(s[:-1])); Yl.append(s[-1])
X=np.array(Xl,dtype=np.int64); Yt=np.array(Yl,dtype=np.int64)
np.savez(os.path.join(OUT,"d3b_train_arrays.npz"),X=X,Yt=Yt)
samp=pd.read_csv(os.path.join(SP,"eval_sample.csv"),dtype=str)
eval_seq={u:seqs.get(u,[]) for u in samp.user_id}
pickle.dump({"item2idx":item2idx,"idx2item":np.array([None]+list(vocab.book_id)),
             "eval_seq":eval_seq,"n_items":n_items,"maxlen":MAXLEN},
            open(os.path.join(OUT,"d3b_seq_meta.pkl"),"wb"))
json.dump({"n_items":n_items,"n_train_examples":int(len(X)),"n_users_seq":len(seqs),
           "maxlen":MAXLEN,"sec":round(time.time()-t0,1)},open(os.path.join(OUT,"d3b_seq_meta.json"),"w"),indent=2)
print("SEQ_BUILD n_items=%d examples=%d users=%d sec=%.1f"%(n_items,len(X),len(seqs),time.time()-t0))
