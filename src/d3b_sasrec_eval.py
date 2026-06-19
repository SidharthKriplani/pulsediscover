"""D3B — evaluate the trained domain SASRec on the 5,000-user leave-last-out sample (full 40k
scoring), R@5..200 + NDCG@10/20/50 + bootstrap CIs; compare to f64 ALS floor / f16 hybrid /
baselines. Writes domain_sasrec_report.json + plot."""
import os, sys, json, pickle, numpy as np, torch, pandas as pd, math
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from pulsediscover import sasrec as S, eval as E
OUT="/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim"; SP=os.path.join(OUT,"domain_splits")
MODELS="/sessions/festive-quirky-cannon/mnt/pulsediscover/outputs/_models"; EVID="/sessions/festive-quirky-cannon/mnt/pulsediscover/outputs/evidence"; PLOTS="/sessions/festive-quirky-cannon/mnt/pulsediscover/outputs/plots"
KS=[5,10,20,50,100,200]; NK=[10,20,50]; SEED=20260616
meta=pickle.load(open(os.path.join(OUT,"d3b_seq_meta.pkl"),"rb"))
item2idx=meta["item2idx"]; n_items=meta["n_items"]; MAXLEN=meta["maxlen"]; eval_seq=meta["eval_seq"]
st=torch.load(os.path.join(MODELS,"sasrec_domain.pt"),weights_only=False)
model=S.SASRec(n_items,d=48,maxlen=MAXLEN,nblocks=1,nheads=1,dropout=0.2); model.load_state_dict(st["model"]); model.eval()
samp=pd.read_csv(os.path.join(SP,"eval_sample.csv"),dtype=str)
users=samp.user_id.tolist(); golds=samp.gold.tolist()
def leftpad(s): s=s[-MAXLEN:]; return [0]*(MAXLEN-len(s))+list(s)
R={k:[] for k in KS}; Nm={k:[] for k in NK}; gold_in_vocab=0; B=1000
with torch.no_grad():
    for s0 in range(0,len(users),B):
        ub=users[s0:s0+B]; xb=torch.tensor([leftpad(eval_seq.get(u,[])) for u in ub],dtype=torch.long)
        sc=model.score_last(xb).numpy()
        for bi,u in enumerate(ub):
            g=golds[s0+bi]; gi=item2idx.get(g)
            row=sc[bi].copy(); row[0]=-1e9
            for it in eval_seq.get(u,[]): row[it]=-1e9
            top=np.argpartition(-row,200)[:200]; top=top[np.argsort(-row[top])]; ranked=list(top)
            if gi is not None: gold_in_vocab+=1
            for k in KS: R[k].append(1.0 if (gi is not None and gi in ranked[:k]) else 0.0)
            for k in NK:
                if gi is not None and gi in ranked[:k]: Nm[k].append(1.0/math.log2(ranked.index(gi)+2))
                else: Nm[k].append(0.0)
agg={}
for k in KS:
    rm,lo,hi,n=E.bootstrap_ci(R[k],seed=SEED); agg[f"R@{k}"]={"mean":round(rm,4),"ci95":[round(lo,4),round(hi,4)]}
for k in NK: agg[f"N@{k}"]={"mean":round(float(np.mean(Nm[k])),4)}
# comparators
f64=json.load(open(os.path.join(EVID,"domain_als_f64_floor.json")))["metrics"]
plus=json.load(open(os.path.join(EVID,"domain_baselines_plus_report.json")))["metrics"]
hyb=plus["hybrid"]; pop=plus["popularity"]; cooc=plus["cooccurrence"]; als16=plus["als"]
tm=json.load(open(os.path.join(OUT,"d3b_train_meta.json")))
report={"method":"SASRec","scope":"DOMAIN (Goodreads fantasy/paranormal)","tag":"[BUILT — real data (domain)]",
 "eval":"leave-last-out, 5,000-user sample (full 40k scoring)","n_eval":len(users),
 "gold_in_vocab":gold_in_vocab,"gold_in_vocab_pct":round(100*gold_in_vocab/len(users),2),
 "training_config":tm["config"],"epochs_done":tm["epochs_done"],"loss_curve":tm["loss_curve"],
 "epoch_times_sec":tm["epoch_times_sec"],"early_stopping":"none (fixed time-budget, resumable; loss still mildly decreasing at stop)",
 "metrics_SASRec":agg,
 "comparison":{"ALS_f64_primary_floor":f64,"hybrid_f16_secondary":{k:hyb[k] for k in hyb},
               "popularity":{k:pop[k] for k in pop},"cooccurrence_5k":{k:cooc[k] for k in cooc},"ALS_f16":{k:als16[k] for k in als16}},
 "verdict_vs_floor":{f"R@{k}":{"sasrec":agg[f"R@{k}"]["mean"],"als_f64":f64[f"R@{k}"]["mean"],
                     "delta":round(agg[f"R@{k}"]["mean"]-f64[f"R@{k}"]["mean"],4)} for k in KS},
}
json.dump(report,open(os.path.join(EVID,"domain_sasrec_report.json"),"w"),indent=2)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5))
    series={"SASRec":agg,"ALS f64 (floor)":f64,"hybrid f16":hyb,"ALS f16":als16,"popularity":pop}
    for nm,mt in series.items(): ax.plot(KS,[mt[f"R@{k}"]["mean"] for k in KS],marker="o",label=nm)
    ax.set_xscale("log"); ax.set_xlabel("K (log)"); ax.set_ylabel("Recall@K")
    ax.set_title("D3B SASRec vs floor — Goodreads fantasy/paranormal (5k-user LLO)"); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS,"domain_sasrec_vs_floor.png"),dpi=110); plt.close()
except Exception as ex: print("plot skipped:",ex)
print("SASREC_EVAL gold_in_vocab%%=%.1f"%(100*gold_in_vocab/len(users)))
for k in [10,20,50,100,200]: print(f"  R@{k} SASRec={agg[f'R@{k}']['mean']} vs ALS_f64={f64[f'R@{k}']['mean']} (Δ{round(agg[f'R@{k}']['mean']-f64[f'R@{k}']['mean'],4)})")
print("  NDCG@10 SASRec=%s ALS_f64=%s"%(agg["N@10"]["mean"],f64["N@10"]["mean"]))
