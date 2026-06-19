"""G20 (v3) — canonical full-softmax SASRec to CONVERGENCE vs ALS f64.
v3 patches over v2:
  - Issue 1: convergence is monitored on a SEPARATE validation-monitor set (held-out 5% of TRAINING
    sequences, next-item R@20). The final 5k LLO eval_sample.csv is reserved for ONE final eval after
    convergence (final_eval_used_for_stopping=false, final_eval_runs_count=1).
  - Issue 2: EXACT total epoch cap. TOTAL_EPOCH_CAP=160, start=40 -> max_extra=120; loop ep<max_extra.
(v1 was invalid: val-loss NaN falsely fired early-stop. v2 used the final eval set for stopping — fixed here.)
GPU run. No protocol changes to help SASRec. Resumes ckpt_canonical.pt. Real numbers only."""
import os, sys, json, time, math, pickle, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
DEV="cuda" if torch.cuda.is_available() else "cpu"; GPU=torch.cuda.get_device_name(0) if DEV=="cuda" else "CPU-only"
SEED=20260616; MAXLEN=30; EVAL_EVERY=5; PATIENCE_EVALS=3; R20_MIN_DELTA=0.001; TOTAL_EPOCH_CAP=160
torch.manual_seed(SEED); np.random.seed(SEED)

class SASBlock(nn.Module):
    def __init__(s,d,h,p):
        super().__init__(); s.attn=nn.MultiheadAttention(d,h,dropout=p,batch_first=True)
        s.ln1=nn.LayerNorm(d); s.ln2=nn.LayerNorm(d); s.ff=nn.Sequential(nn.Linear(d,d),nn.ReLU(),nn.Linear(d,d),nn.Dropout(p))
    def forward(s,x,m,kpm):
        a,_=s.attn(x,x,x,attn_mask=m,key_padding_mask=kpm,need_weights=False); x=s.ln1(x+a); return s.ln2(x+s.ff(x))
class SASRec(nn.Module):
    def __init__(s,n,d=64,maxlen=30,nblocks=2,nheads=2,dropout=0.2):
        super().__init__(); s.item_emb=nn.Embedding(n+1,d,padding_idx=0); s.pos_emb=nn.Embedding(maxlen,d)
        s.drop=nn.Dropout(dropout); s.blocks=nn.ModuleList([SASBlock(d,nheads,dropout) for _ in range(nblocks)]); s.ln=nn.LayerNorm(d)
    def seq_repr(s,seq):
        B,L=seq.shape; pos=torch.arange(L,device=seq.device).unsqueeze(0).expand(B,L)
        x=s.drop(s.item_emb(seq)+s.pos_emb(pos)); causal=torch.triu(torch.ones(L,L,device=seq.device),1).bool()
        for blk in s.blocks: x=blk(x,causal,seq==0)
        return s.ln(x)
    def forward(s,seq): h=s.seq_repr(seq); return h@s.item_emb.weight.T
    def score_last(s,seq): h=s.seq_repr(seq)[:,-1,:]; return h@s.item_emb.weight.T

def fullsoftmax_loss(model,xb,yb):
    lg=model.forward(xb); return F.cross_entropy(lg.reshape(-1,lg.size(-1)),yb.reshape(-1),ignore_index=0)

def main(datadir=".",outdir="."):
    print("Device",DEV,"GPU",GPU,flush=True)
    z=np.load(os.path.join(datadir,"g11_fullpos_arrays.npz")); X=torch.tensor(z["X"]); Y=torch.tensor(z["Y"]); N=X.shape[0]
    g=torch.Generator().manual_seed(SEED); perm=torch.randperm(N,generator=g); nval=N//20
    vidx=perm[:nval]; tidx=perm[nval:]                              # vidx = validation-MONITOR set (training seqs)
    Xv=X[vidx]; Yv=Y[vidx]
    meta=pickle.load(open(os.path.join(datadir,"d3b_seq_meta.pkl"),"rb")); item2idx=meta["item2idx"]; n_items=meta["n_items"]; eval_seq=meta["eval_seq"]
    samp=pd.read_csv(os.path.join(datadir,"eval_sample.csv"),dtype=str); users=samp.user_id.tolist(); golds=samp.gold.tolist()
    def lp(s): s=s[-MAXLEN:]; return [0]*(MAXLEN-len(s))+list(s)
    def monitor_r20(model):                                        # next-item R@20 on held-out TRAINING seqs (NOT the final eval set)
        model.eval(); hit=valid=0; n=Xv.shape[0]; B=512
        with torch.no_grad():
            for s0 in range(0,n,B):
                xb=Xv[s0:s0+B].to(DEV); sc=model.score_last(xb).cpu().numpy(); tg=Yv[s0:s0+B,-1].numpy(); hist=Xv[s0:s0+B].numpy()
                for bi in range(xb.shape[0]):
                    gt=int(tg[bi])
                    if gt==0: continue
                    valid+=1; row=sc[bi].copy(); row[0]=-1e9
                    for it in hist[bi]:
                        if it!=0 and it!=gt: row[int(it)]=-1e9
                    top=np.argpartition(-row,20)[:20]
                    if gt in top: hit+=1
        return hit/max(valid,1)
    def final_eval(model):                                          # the ONLY use of eval_sample.csv (5k LLO)
        model.eval(); KS=[20,50,200]; NK=[10,20]; R={k:0 for k in KS}; ND={k:0 for k in NK}; n=len(users); B=512
        with torch.no_grad():
            for s0 in range(0,n,B):
                ub=users[s0:s0+B]; xb=torch.tensor([lp(eval_seq.get(u,[])) for u in ub],dtype=torch.long,device=DEV)
                sc=model.score_last(xb).cpu().numpy()
                for bi,u in enumerate(ub):
                    gi=item2idx.get(golds[s0+bi]); row=sc[bi].copy(); row[0]=-1e9
                    for it in eval_seq.get(u,[]): row[it]=-1e9
                    top=np.argpartition(-row,200)[:200]; ranked=list(top[np.argsort(-row[top])])
                    for k in KS:
                        if gi is not None and gi in ranked[:k]: R[k]+=1
                    for k in NK:
                        if gi is not None and gi in ranked[:k]: ND[k]+=1.0/math.log2(ranked.index(gi)+2)
        out={f"R@{k}":round(R[k]/n,4) for k in KS}; out.update({f"NDCG@{k}":round(ND[k]/n,4) for k in NK}); return out
    model=SASRec(n_items,d=64,maxlen=MAXLEN,nblocks=2,nheads=2); start=0
    ck=os.path.join(datadir,"ckpt_canonical.pt")
    if os.path.exists(ck):
        st=torch.load(ck,weights_only=False,map_location=DEV)
        try: model.load_state_dict(st["model"]); start=40; print("resumed ckpt_canonical (~ep40)",flush=True)
        except Exception as e: print("resume failed, fresh:",e,flush=True)
    max_extra=max(0,TOTAL_EPOCH_CAP-start)
    model.to(DEV); opt=torch.optim.Adam(model.parameters(),lr=1e-3); BATCH=128
    tr=[]; mon=[]; best=-1; bad=0; conv=None; t0=time.time(); ep=0
    while ep<max_extra:
        model.train(); pr=torch.randperm(len(tidx)); tot=nb=0
        for i in range(0,len(tidx),BATCH):
            b=tidx[pr[i:i+BATCH]]; loss=fullsoftmax_loss(model,X[b].to(DEV),Y[b].to(DEV))
            opt.zero_grad(); loss.backward(); opt.step(); tot+=float(loss.detach()); nb+=1
        trl=tot/nb; tr.append(round(trl,4)); ep+=1; line=f"ep {start+ep} train {trl:.4f}"
        if ep%EVAL_EVERY==0 or ep==1:
            mr=monitor_r20(model); mon.append([start+ep,round(mr,4)]); line+=f" monitorR@20 {mr:.4f}"
            if mr-best>R20_MIN_DELTA: best=mr; bad=0
            else:
                bad+=1
                if bad>=PATIENCE_EVALS: print(line+" (monitor plateau)",flush=True); conv=f"monitor-R@20 plateau (min_delta {R20_MIN_DELTA}, patience {PATIENCE_EVALS} checks)"; break
        print(line+f" ({time.time()-t0:.0f}s)",flush=True)
    if conv is None: conv="HARD CAP total %d epochs reached — if monitor R@20 still rising, OUTCOME C (not converged)"%TOTAL_EPOCH_CAP
    torch.save({"model":model.state_dict(),"n_items":n_items},os.path.join(outdir,"ckpt_canonical_converged.pt"))
    res=final_eval(model)                                           # final eval: run ONCE
    als=json.load(open(os.path.join(datadir,"domain_als_f64_floor.json")))["metrics"]; als20=als["R@20"]["mean"]; ratio=res["R@20"]/als20
    converged = conv.startswith("monitor-R@20 plateau")
    verdict=("clearly_loses" if ratio<0.95 else ("contends" if ratio<=1.05 else "beats")) if converged else "INVALID_not_converged"
    out={"scope":"G20 v3 canonical full-softmax SASRec to convergence vs ALS f64","device":DEV,"gpu":GPU,
     "config":{"d":64,"nblocks":2,"nheads":2,"objective":"full-softmax over 40k","maxlen":MAXLEN,"batch":BATCH},
     "monitor_set":"validation_monitor (held-out 5% of TRAINING sequences, next-item R@20)",
     "final_eval_used_for_stopping":False,"final_eval_runs_count":1,
     "monitor_valid_examples":int((Yv[:,-1]!=0).sum()),"monitor_total_sequences":int(Xv.shape[0]),"final_eval_users":len(users),
     "protocol_note":"Monitor set used only for early stopping; final 5k LLO used once for ALS comparison.",
     "convergence_criterion":f"monitor-R@20 plateau (min_delta {R20_MIN_DELTA}, patience {PATIENCE_EVALS} checks every {EVAL_EVERY} ep)",
     "convergence_hit":conv,"VALID_CONVERGENCE":bool(converged),
     "resumed_from_ep":start,"epochs_this_run":ep,"total_epochs_approx":start+ep,"total_epoch_cap":TOTAL_EPOCH_CAP,
     "train_loss_curve":tr,"monitor_r20_history":mon,"canonical_metrics_final_eval":res,
     "als_f64_R@20":als20,"ratio_vs_als_R@20":round(ratio,3),"verdict":verdict,
     "eval":"final = 5k-user LLO eval_sample, full 40k scoring (used once)","seed":SEED,
     "honest_notes":["Convergence monitored on a SEPARATE validation set (held-out training seqs); final 5k LLO used once.",
       "verdict valid ONLY if VALID_CONVERGENCE true (monitor plateau, not hard cap).",
       "Optimizer re-init on resume. vs ALS full-catalog; no within-candidate mixing."],
     "sec":round(time.time()-t0,1)}
    json.dump(out,open(os.path.join(outdir,"g20_canonical_sasrec_convergence_report.json"),"w"),indent=2)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig,ax=plt.subplots(1,2,figsize=(13,5)); e=np.arange(1,len(tr)+1)+start
        ax[0].plot(e,tr,"o-"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("train full-softmax loss"); ax[0].set_title("Train loss")
        if mon:
            ax[1].plot([m[0] for m in mon],[m[1] for m in mon],"s-",label="monitor R@20 (validation-monitor set)")
        # NOTE: no ALS line here — monitor R@20 (val-monitor set) is NOT the same protocol as ALS R@20 (final 5k LLO). ALS comparison lives in the JSON after the single final eval.
        ax[1].set_xlabel("epoch"); ax[1].set_ylabel("monitor R@20 (val-monitor set)"); ax[1].legend(); ax[1].set_title("Monitor R@20 — %s (ALS comparison in JSON, final-eval only)"%("CONVERGED" if converged else "NOT converged"))
        plt.suptitle("G20 v3 — %s | final R@20 %.4f (%s)"%(conv,res["R@20"],verdict)); plt.tight_layout(); plt.savefig(os.path.join(outdir,"g20_canonical_sasrec_loss_curve.png"),dpi=110); plt.close()
    except Exception as e: print("plot skipped",e)
    print(json.dumps({"convergence":conv,"VALID":converged,"final_metrics":res,"als_R@20":als20,"ratio":round(ratio,3),"verdict":verdict,"monitor_set":out["monitor_set"],"final_eval_runs_count":1},indent=2),flush=True)

if __name__=="__main__":
    dd=sys.argv[1] if len(sys.argv)>1 else "."; od=sys.argv[2] if len(sys.argv)>2 else "."
    main(dd,od)
