"""G11 — sequence model tournament. Modes (argv[1]):
  build            -> full-position training arrays (X_in, Y_tgt) for all train users (resumable cache)
  train gru|sas    -> time-budgeted, resumable full-position training (checkpoints each epoch)
  eval             -> eval GRU4Rec + SASRec v2 vs ALS f64 on the same 5k LLO protocol; writes metrics+report json
Full-position sampled-softmax BCE (every position predicts next item) — the canonical fix vs D3B's
last-position run. CPU-feasible; GPU-ready config in models/sasrec_v2.py. Real numbers only."""
import os, sys, json, time, pickle, math, numpy as np, torch, torch.nn.functional as F
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from pulsediscover import sasrec as S, eval as E
from models.gru4rec import GRU4Rec
from models import sasrec_v2 as V2
OUT="/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim"; SP=os.path.join(OUT,"domain_splits")
MODELS="/sessions/festive-quirky-cannon/mnt/pulsediscover/outputs/_models"; EVID="/sessions/festive-quirky-cannon/mnt/pulsediscover/outputs/evidence"; PLOTS="/sessions/festive-quirky-cannon/mnt/pulsediscover/outputs/plots"
os.makedirs(MODELS,exist_ok=True)
MAXLEN=30; SEED=20260616; NNEG=100; LR=1e-3; BUDGET=15.0; MAX_EPOCHS=60
torch.manual_seed(SEED); np.random.seed(SEED); torch.set_num_threads(4)
meta=pickle.load(open(os.path.join(OUT,"d3b_seq_meta.pkl"),"rb")); item2idx=meta["item2idx"]; n_items=meta["n_items"]; eval_seq=meta["eval_seq"]
ARRP=os.path.join(OUT,"g11_fullpos_arrays.npz")

def build():
    import pandas as pd; t0=time.time()
    core=pd.read_csv(os.path.join(OUT,"domain_core_interactions.csv"),usecols=["user_id","book_id","event_ts"],
                     dtype={"user_id":"str","book_id":"str","event_ts":"int64"})
    held=pd.read_csv(os.path.join(SP,"heldout_eval.csv"),dtype=str)
    tr=core[~(core.user_id+"|"+core.book_id).isin(set(held.user_id+"|"+held.gold))]; del core
    tr=tr[tr.book_id.isin(item2idx)].sort_values(["user_id","event_ts"])
    seqs=tr.groupby("user_id")["book_id"].apply(lambda s:[item2idx[b] for b in s]).to_dict(); del tr
    def lp(s): s=s[-MAXLEN:]; return [0]*(MAXLEN-len(s))+s
    Xin=[]; Ytg=[]
    for u,s in seqs.items():
        if len(s)<2: continue
        Xin.append(lp(s[:-1])); Ytg.append(lp(s[1:]))      # full-position: target = next item at every step
    X=np.array(Xin,dtype=np.int64); Y=np.array(Ytg,dtype=np.int64)
    np.savez(ARRP,X=X,Y=Y)
    print("BUILD examples=%d shape=%s sec=%.1f"%(len(X),X.shape,time.time()-t0))

def fullpos_loss(model,xb,yb):
    h=model.seq_repr(xb)                                    # (B,L,d)
    pos=model.item_emb(yb)                                  # (B,L,d)
    pl=(h*pos).sum(-1)                                      # (B,L)
    B=xb.shape[0]
    neg=torch.randint(1,n_items,(B,NNEG))                   # PER-EXAMPLE negatives (B,NNEG)
    ne=model.item_emb(neg)                                  # (B,NNEG,d)
    nl=torch.einsum("bld,bnd->bln",h,ne)                   # (B,L,NNEG)
    mask=(yb!=0).float()                                    # supervise only real next-item positions
    lp=F.binary_cross_entropy_with_logits(pl,torch.ones_like(pl),reduction="none")*mask
    ln=(F.binary_cross_entropy_with_logits(nl,torch.zeros_like(nl),reduction="none").mean(-1))*mask
    denom=mask.sum().clamp(min=1)
    return (lp.sum()+ln.sum())/denom

def make(model_name):
    if model_name=="gru": return GRU4Rec(n_items,d=64,layers=1,dropout=0.2,maxlen=MAXLEN), os.path.join(MODELS,"g11_gru4rec_pe.pt")
    c=V2.CPU_CONFIG
    if model_name=="saslp":   # G11B: SAME SASRec v2 architecture, last-position objective only
        return V2.SASRec(n_items,d=c["d"],maxlen=MAXLEN,nblocks=c["nblocks"],nheads=c["nheads"],dropout=c["dropout"]), os.path.join(MODELS,"g11b_sasrec_v2_lastpos.pt")
    return V2.SASRec(n_items,d=c["d"],maxlen=MAXLEN,nblocks=c["nblocks"],nheads=c["nheads"],dropout=c["dropout"]), os.path.join(MODELS,"g11_sasrec_v2_pe2.pt")

def lastpos_loss(model,xb,yb):
    h=model.seq_repr(xb)[:,-1,:]                            # (B,d) last position
    tgt=yb[:,-1]                                            # (B,) final next-item
    pl=(h*model.item_emb(tgt)).sum(-1)                      # (B,)
    neg=torch.randint(1,n_items,(xb.shape[0],NNEG)); ne=model.item_emb(neg)  # (B,NNEG,d)
    nl=torch.einsum("bd,bnd->bn",h,ne)                     # (B,NNEG)
    return F.binary_cross_entropy_with_logits(pl,torch.ones_like(pl))+\
           F.binary_cross_entropy_with_logits(nl,torch.zeros_like(nl))

def train(model_name):
    z=np.load(ARRP); X=torch.tensor(z["X"]); Y=torch.tensor(z["Y"]); N=X.shape[0]
    BATCH=512 if model_name=="sas" else 1024
    lossfn=lastpos_loss if model_name=="saslp" else fullpos_loss
    model,ckpt=make(model_name); opt=torch.optim.Adam(model.parameters(),lr=LR)
    epochs=0; losses=[]; etimes=[]
    if os.path.exists(ckpt):
        try:
            st=torch.load(ckpt,weights_only=False)
            if st["n_items"]==n_items:
                model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); epochs=st["epochs"]; losses=st["losses"]; etimes=st.get("etimes",[]); print("resumed",model_name,"epoch",epochs)
        except Exception as ex:
            print("WARN corrupt ckpt, starting fresh:",ex)
    t0=time.time()
    while time.time()-t0<BUDGET and epochs<MAX_EPOCHS:
        te=time.time(); model.train(); perm=torch.randperm(N,generator=torch.Generator().manual_seed(SEED+epochs)); tot=nb=0
        for i in range(0,N,BATCH):
            idx=perm[i:i+BATCH]; loss=lossfn(model,X[idx],Y[idx])
            opt.zero_grad(); loss.backward(); opt.step(); tot+=float(loss.detach()); nb+=1
        epochs+=1; losses.append(round(tot/nb,4)); etimes.append(round(time.time()-te,1))
        torch.save({"model":model.state_dict(),"opt":opt.state_dict(),"epochs":epochs,"losses":losses,"etimes":etimes,"n_items":n_items},ckpt+".tmp")
        os.replace(ckpt+".tmp",ckpt)                        # atomic save (avoid corruption on timeout)
        print(f"{model_name} epoch {epochs} loss {losses[-1]} ({etimes[-1]}s)")
    print("TRAIN_DONE %s epochs=%d last=%s"%(model_name,epochs,losses[-1] if losses else None))

def eval_model(model_name):
    import pandas as pd
    model,ckpt=make(model_name); st=torch.load(ckpt,weights_only=False); model.load_state_dict(st["model"]); model.eval()
    samp=pd.read_csv(os.path.join(SP,"eval_sample.csv"),dtype=str); users=samp.user_id.tolist(); golds=samp.gold.tolist()
    def lp(s): s=s[-MAXLEN:]; return [0]*(MAXLEN-len(s))+list(s)
    KS=[20,50]; NK=[20]; R={k:[] for k in KS}; Nm={k:[] for k in NK}; giv=0; B=1000
    with torch.no_grad():
        for s0 in range(0,len(users),B):
            ub=users[s0:s0+B]; xb=torch.tensor([lp(eval_seq.get(u,[])) for u in ub],dtype=torch.long); sc=model.score_last(xb).numpy()
            for bi,u in enumerate(ub):
                g=golds[s0+bi]; gi=item2idx.get(g); row=sc[bi].copy(); row[0]=-1e9
                for it in eval_seq.get(u,[]): row[it]=-1e9
                top=np.argpartition(-row,50)[:50]; top=top[np.argsort(-row[top])]; ranked=list(top)
                if gi is not None: giv+=1
                for k in KS: R[k].append(1.0 if (gi is not None and gi in ranked[:k]) else 0.0)
                for k in NK: Nm[k].append((1.0/math.log2(ranked.index(gi)+2)) if (gi is not None and gi in ranked[:k]) else 0.0)
    agg={}
    for k in KS: rm,lo,hi,_=E.bootstrap_ci(R[k],seed=SEED); agg[f"R@{k}"]={"mean":round(rm,4),"ci95":[round(lo,4),round(hi,4)]}
    for k in NK: agg[f"N@{k}"]={"mean":round(float(np.mean(Nm[k])),4)}
    agg["epochs_done"]=st["epochs"]; agg["loss_curve"]=st["losses"]; agg["gold_in_vocab"]=giv; agg["n_eval"]=len(users)
    return agg

if __name__=="__main__":
    mode=sys.argv[1]
    if mode=="build": build()
    elif mode=="train": train(sys.argv[2])
    elif mode=="eval":
        f64=json.load(open(os.path.join(EVID,"domain_als_f64_floor.json")))["metrics"]
        sas_old=json.load(open(os.path.join(EVID,"domain_sasrec_report.json")))["metrics_SASRec"]
        gru=eval_model("gru"); sas=eval_model("sas")
        out={"scope":"DOMAIN Goodreads fantasy/paranormal","tag":"[BUILT — real data, CPU-limited]",
         "eval":"leave-last-out, 5,000-user sample, full 40k scoring","hardware":"CPU (4 threads), no GPU",
         "n_items":n_items,"maxlen":MAXLEN,"seed":SEED,"reproducibility":"run_g11_sequence_tournament.py build|train gru|train sas|eval; checkpoints in outputs/_models",
         "configs":{"GRU4Rec":{"d":64,"layers":1,"dropout":0.2,"batch":1024,"nneg":NNEG,"lr":LR,"training":"full-position sampled-softmax BCE"},
                    "SASRec_v2":V2.CPU_CONFIG,"SASRec_v2_gpu_ready":V2.GPU_CANONICAL_CONFIG},
         "results":{"ALS_f64_floor":{k:f64[k] for k in ["R@20","R@50","N@20"] if k in f64},
                    "SASRec_v1_d3b_lastpos":{k:sas_old[k] for k in ["R@20","R@50","N@20"] if k in sas_old},
                    "GRU4Rec":gru,"SASRec_v2_fullpos":sas},
         "deltas_vs_als_f64":{"GRU4Rec_R@20":round(gru["R@20"]["mean"]-f64["R@20"]["mean"],4),
                              "SASRec_v2_R@20":round(sas["R@20"]["mean"]-f64["R@20"]["mean"],4),
                              "SASRec_v2_vs_v1_R@20":round(sas["R@20"]["mean"]-sas_old["R@20"]["mean"],4)}}
        json.dump(out,open(os.path.join(EVID,"g11_sequence_tournament_metrics.json"),"w"),indent=2)
        print(json.dumps(out["results"],indent=2)); print("deltas",out["deltas_vs_als_f64"])
    elif mode=="evalb":   # G11B objective ablation: full-position vs last-position, same SASRec v2 arch
        f64=json.load(open(os.path.join(EVID,"domain_als_f64_floor.json")))["metrics"]
        g11=json.load(open(os.path.join(EVID,"g11_sequence_tournament_metrics.json")))["results"]
        sas_old=json.load(open(os.path.join(EVID,"domain_sasrec_report.json")))["metrics_SASRec"]
        saslp=eval_model("saslp")
        out={"scope":"DOMAIN Goodreads fantasy/paranormal — SASRec objective ablation","tag":"[BUILT — real data, CPU-limited]",
         "eval":"leave-last-out, 5,000-user sample, full 40k scoring","hardware":"CPU (4 threads), no GPU","seed":SEED,
         "controlled_variable":"training objective ONLY (full-position vs last-position); architecture d=64/2blocks/2heads/maxlen30 held fixed",
         "reproducibility":"run_g11_sequence_tournament.py train saslp; evalb",
         "results":{"ALS_f64_floor":{k:f64[k] for k in ["R@20","R@50","N@20"]},
                    "SASRec_v1_d3b_lastpos_d48":{k:sas_old[k] for k in ["R@20","R@50","N@20"]},
                    "SASRec_v2_fullpos_d64":g11["SASRec_v2_fullpos"],
                    "SASRec_v2_lastpos_d64":saslp},
         "ablation_delta":{"lastpos_minus_fullpos_R@20":round(saslp["R@20"]["mean"]-g11["SASRec_v2_fullpos"]["R@20"]["mean"],4),
                           "lastpos_vs_v1_R@20":round(saslp["R@20"]["mean"]-sas_old["R@20"]["mean"],4),
                           "lastpos_vs_als_R@20":round(saslp["R@20"]["mean"]-f64["R@20"]["mean"],4)}}
        json.dump(out,open(os.path.join(EVID,"g11b_sasrec_objective_ablation_metrics.json"),"w"),indent=2)
        print(json.dumps(out["results"],indent=2)); print("ablation",out["ablation_delta"])
