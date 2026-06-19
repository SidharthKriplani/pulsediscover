"""D3B — SASRec domain trainer (sampled-softmax / negative BCE; CPU-safe; resumable).
Time-budgeted per call; checkpoints each epoch to sasrec_domain.pt. Re-run to add epochs."""
import os, sys, json, time, numpy as np, torch, torch.nn.functional as F
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from pulsediscover import sasrec as S
OUT="/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim"; MODELS="/sessions/festive-quirky-cannon/mnt/pulsediscover/outputs/_models"
os.makedirs(MODELS,exist_ok=True)
D=48; MAXLEN=30; NBLOCKS=1; NHEADS=1; DROPOUT=0.2; BATCH=2048; NNEG=50; LR=1e-3
SEED=20260616; TRAIN_BUDGET=34.0; MAX_EPOCHS=40
torch.manual_seed(SEED); np.random.seed(SEED); torch.set_num_threads(4)
z=np.load(os.path.join(OUT,"d3b_train_arrays.npz")); X=torch.tensor(z["X"]); Yt=torch.tensor(z["Yt"])
import pickle; meta=pickle.load(open(os.path.join(OUT,"d3b_seq_meta.pkl"),"rb")); n_items=meta["n_items"]
model=S.SASRec(n_items,d=D,maxlen=MAXLEN,nblocks=NBLOCKS,nheads=NHEADS,dropout=DROPOUT)
opt=torch.optim.Adam(model.parameters(),lr=LR)
ckpt=os.path.join(MODELS,"sasrec_domain.pt"); epochs=0; losses=[]; etimes=[]
if os.path.exists(ckpt):
    st=torch.load(ckpt,weights_only=False)
    if st["n_items"]==n_items:
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); epochs=st["epochs"]; losses=st["losses"]; etimes=st.get("etimes",[])
        print("resumed epoch",epochs)
N=X.shape[0]; t0=time.time()
while time.time()-t0<TRAIN_BUDGET and epochs<MAX_EPOCHS:
    te=time.time(); model.train()
    perm=torch.randperm(N,generator=torch.Generator().manual_seed(SEED+epochs))
    tot=nb=0
    for i in range(0,N,BATCH):
        idx=perm[i:i+BATCH]; xb=X[idx]
        h=model.seq_repr(xb)[:,-1,:]                      # (B,d) with grad
        pos=Yt[idx]; pe=model.item_emb(pos); pl=(h*pe).sum(-1)
        neg=torch.randint(1,n_items,(xb.shape[0],NNEG))
        ne=model.item_emb(neg); nl=(h.unsqueeze(1)*ne).sum(-1)
        loss=F.binary_cross_entropy_with_logits(pl,torch.ones_like(pl))+\
             F.binary_cross_entropy_with_logits(nl,torch.zeros_like(nl))
        opt.zero_grad(); loss.backward(); opt.step(); tot+=float(loss.detach()); nb+=1
    epochs+=1; losses.append(round(tot/nb,4)); etimes.append(round(time.time()-te,1))
    torch.save({"model":model.state_dict(),"opt":opt.state_dict(),"epochs":epochs,"losses":losses,
                "etimes":etimes,"n_items":n_items},ckpt)
    print(f"epoch {epochs} loss {losses[-1]} ({etimes[-1]}s)")
json.dump({"config":{"d":D,"maxlen":MAXLEN,"nblocks":NBLOCKS,"nheads":NHEADS,"dropout":DROPOUT,
           "batch":BATCH,"nneg":NNEG,"lr":LR,"optimizer":"Adam","loss":"sampled-softmax BCE"},
           "epochs_done":epochs,"loss_curve":losses,"epoch_times_sec":etimes,
           "n_items":n_items,"n_train_examples":int(N)},
          open(os.path.join(OUT,"d3b_train_meta.json"),"w"),indent=2)
print("TRAIN_DONE epochs=%d last_loss=%s avg_epoch=%.1fs"%(epochs,losses[-1] if losses else None,np.mean(etimes) if etimes else 0))
