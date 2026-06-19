"""G12-A2 — GPU score export. Re-trains canonical full-softmax SASRec, SASRec-small (last-pos),
and TwoTower; SAVES checkpoints this time; then scores the EXACT G12-A candidate rows to fill
sasrec_canonical_gpu_score / sasrec_small_gpu_score / twotower_gpu_score. No eval change; no LTR.

Required uploads: g11_fullpos_arrays.npz, d3b_seq_meta.pkl, g12_candidate_feature_table.csv.gz.
Outputs: g12_candidate_feature_table_enriched.csv.gz, g12_gpu_score_export_summary.json,
plus checkpoints ckpt_canonical.pt / ckpt_small.pt / ckpt_twotower.pt (download to avoid retrain)."""
import os, sys, json, time, pickle, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
DEV="cuda" if torch.cuda.is_available() else "cpu"
GPU_NAME=torch.cuda.get_device_name(0) if DEV=="cuda" else "CPU-only (no GPU)"
SEED=20260616; MAXLEN=30; NNEG=100; torch.manual_seed(SEED); np.random.seed(SEED)

class SASBlock(nn.Module):
    def __init__(s,d,h,p):
        super().__init__(); s.attn=nn.MultiheadAttention(d,h,dropout=p,batch_first=True)
        s.ln1=nn.LayerNorm(d); s.ln2=nn.LayerNorm(d); s.ff=nn.Sequential(nn.Linear(d,d),nn.ReLU(),nn.Linear(d,d),nn.Dropout(p))
    def forward(s,x,m,kpm):
        a,_=s.attn(x,x,x,attn_mask=m,key_padding_mask=kpm,need_weights=False); x=s.ln1(x+a); return s.ln2(x+s.ff(x))
class SASRec(nn.Module):
    def __init__(s,n,d=64,maxlen=30,nblocks=2,nheads=2,dropout=0.2):
        super().__init__(); s.maxlen=maxlen; s.item_emb=nn.Embedding(n+1,d,padding_idx=0); s.pos_emb=nn.Embedding(maxlen,d)
        s.drop=nn.Dropout(dropout); s.blocks=nn.ModuleList([SASBlock(d,nheads,dropout) for _ in range(nblocks)]); s.ln=nn.LayerNorm(d)
    def seq_repr(s,seq):
        B,L=seq.shape; pos=torch.arange(L,device=seq.device).unsqueeze(0).expand(B,L)
        x=s.drop(s.item_emb(seq)+s.pos_emb(pos)); causal=torch.triu(torch.ones(L,L,device=seq.device),1).bool()
        for blk in s.blocks: x=blk(x,causal,seq==0)
        return s.ln(x)
    def forward(s,seq): h=s.seq_repr(seq); return h@s.item_emb.weight.T
    def score_last(s,seq): h=s.seq_repr(seq)[:,-1,:]; return h@s.item_emb.weight.T
class TwoTower(nn.Module):
    def __init__(s,n,d=64,dropout=0.2,maxlen=30):
        super().__init__(); s.maxlen=maxlen; s.item_emb=nn.Embedding(n+1,d,padding_idx=0); s.drop=nn.Dropout(dropout)
        s.mlp=nn.Sequential(nn.Linear(d,d),nn.ReLU(),nn.Linear(d,d))
    def user_vec(s,seq):
        e=s.item_emb(seq); mask=(seq!=0).float().unsqueeze(-1); pooled=(e*mask).sum(1)/mask.sum(1).clamp(min=1); return s.mlp(s.drop(pooled))
    def score_last(s,seq): return s.user_vec(seq)@s.item_emb.weight.T

def loss_fullsoftmax(m,xb,yb,n): lg=m.forward(xb); return F.cross_entropy(lg.reshape(-1,lg.size(-1)),yb.reshape(-1),ignore_index=0)
def loss_lastpos(m,xb,yb,n):
    h=m.seq_repr(xb)[:,-1,:]; tgt=yb[:,-1]; pl=(h*m.item_emb(tgt)).sum(-1)
    neg=torch.randint(1,n,(xb.shape[0],NNEG),device=xb.device); nl=torch.einsum("bd,bnd->bn",h,m.item_emb(neg))
    return F.binary_cross_entropy_with_logits(pl,torch.ones_like(pl))+F.binary_cross_entropy_with_logits(nl,torch.zeros_like(nl))
def loss_twotower(m,xb,yb,n):
    uv=m.user_vec(xb); te=m.item_emb(yb[:,-1]); return F.cross_entropy(uv@te.T,torch.arange(xb.shape[0],device=xb.device))

def train_save(model,lossfn,epochs,batch,lr,X,Y,n,ckpt):
    model.to(DEV); opt=torch.optim.Adam(model.parameters(),lr=lr); N=X.shape[0]
    for ep in range(epochs):
        te=time.time(); model.train(); perm=torch.randperm(N); tot=nb=0
        for i in range(0,N,batch):
            idx=perm[i:i+batch]; loss=lossfn(model,X[idx].to(DEV),Y[idx].to(DEV),n)
            opt.zero_grad(); loss.backward(); opt.step(); tot+=float(loss.detach()); nb+=1
        print(f"  {os.path.basename(ckpt)} ep {ep+1}/{epochs} loss {tot/nb:.4f} ({time.time()-te:.1f}s)",flush=True)
    torch.save({"model":model.state_dict(),"n_items":n},ckpt); print("  saved",ckpt,flush=True)
    return model

def score_table(model,df,user_seq,item2idx,col,maxlen):
    model.eval()
    def lp(s): s=s[-maxlen:]; return [0]*(maxlen-len(s))+list(s)
    vidx=df["item_id"].map(lambda b: item2idx.get(b,-1)).to_numpy()
    scores=np.full(len(df),np.nan,dtype=np.float32)
    users=df["user_id"].to_numpy(); uniq=pd.unique(users)
    pos={};
    for i,u in enumerate(users): pos.setdefault(u,[]).append(i)
    B=256
    with torch.no_grad():
        for s0 in range(0,len(uniq),B):
            ub=uniq[s0:s0+B]; xb=torch.tensor([lp(user_seq.get(u,[])) for u in ub],dtype=torch.long,device=DEV)
            vec=model.score_last(xb).cpu().numpy()                      # (b,V)
            for bi,u in enumerate(ub):
                ri=pos[u]; vi=vidx[ri]
                for r,v in zip(ri,vi):
                    if v>=0 and v<vec.shape[1]: scores[r]=vec[bi,v]
    df[col]=scores; return int(np.isnan(scores).sum())

def main(datadir=".",outdir="."):
    print("Device:",DEV,"| GPU:",GPU_NAME,flush=True)
    z=np.load(os.path.join(datadir,"g11_fullpos_arrays.npz")); X=torch.tensor(z["X"]); Y=torch.tensor(z["Y"])
    meta=pickle.load(open(os.path.join(datadir,"d3b_seq_meta.pkl"),"rb")); item2idx=meta["item2idx"]; n=meta["n_items"]; eval_seq=meta["eval_seq"]
    df=pd.read_csv(os.path.join(datadir,"g12_candidate_feature_table.csv.gz"),dtype={"user_id":str,"item_id":str})
    t0=time.time(); miss={}
    print("[1] canonical full-softmax SASRec",flush=True)
    cano=train_save(SASRec(n,d=64,maxlen=MAXLEN,nblocks=2,nheads=2),loss_fullsoftmax,40,128,1e-3,X,Y,n,os.path.join(outdir,"ckpt_canonical.pt"))
    miss["sasrec_canonical_gpu_score"]=score_table(cano,df,eval_seq,item2idx,"sasrec_canonical_gpu_score",MAXLEN); del cano
    if DEV=="cuda": torch.cuda.empty_cache()
    print("[2] SASRec-small last-pos",flush=True)
    small=train_save(SASRec(n,d=48,maxlen=MAXLEN,nblocks=1,nheads=1),loss_lastpos,20,1024,1e-3,X,Y,n,os.path.join(outdir,"ckpt_small.pt"))
    miss["sasrec_small_gpu_score"]=score_table(small,df,eval_seq,item2idx,"sasrec_small_gpu_score",MAXLEN); del small
    if DEV=="cuda": torch.cuda.empty_cache()
    print("[3] TwoTower (weak standalone, included as feature)",flush=True)
    tt=train_save(TwoTower(n,d=64,maxlen=MAXLEN),loss_twotower,30,512,1e-3,X,Y,n,os.path.join(outdir,"ckpt_twotower.pt"))
    miss["twotower_gpu_score"]=score_table(tt,df,eval_seq,item2idx,"twotower_gpu_score",MAXLEN); del tt
    df.to_csv(os.path.join(outdir,"g12_candidate_feature_table_enriched.csv.gz"),index=False,compression="gzip")
    seen_leak=int(((df["seen_flag"]==1)&(df["label"]==1)).sum())
    summary={"n_candidate_rows":int(len(df)),"n_users":int(df.user_id.nunique()),"device":DEV,"gpu":GPU_NAME,
     "missing_score_counts":miss,"rows_total":int(len(df)),
     "model_configs":{"canonical":"SASRec d64/2blk/2head full-softmax 40ep","small":"SASRec d48/1blk/1head last-pos 20ep","twotower":"pooled-history in-batch-neg d64 30ep"},
     "checkpoints_saved":["ckpt_canonical.pt","ckpt_small.pt","ckpt_twotower.pt"],
     "leakage_check_gold_in_seen":seen_leak,
     "honest_notes":["TwoTower was weak standalone (R@20~0.002) but included as a candidate feature — the ranker can down-weight it.",
       "Scores are dot(user_repr, item_emb); NaN if item outside the 40k SASRec vocab.","Eval protocol unchanged; no LTR trained here."],
     "runtime_sec":round(time.time()-t0,1)}
    json.dump(summary,open(os.path.join(outdir,"g12_gpu_score_export_summary.json"),"w"),indent=2)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":
    dd=sys.argv[1] if len(sys.argv)>1 else "."; od=sys.argv[2] if len(sys.argv)>2 else "."
    main(dd,od)
