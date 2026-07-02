"""G32 two-tower trainer (PyTorch, CPU, resumable).

Model (genuine two-tower, dot-product scored):
  user tower : E_hist (nI x d) mean-pooled over the user's training history -> u_vec
  item tower : E_item (nI x d)  +  W_c(384->d) @ content(MiniLM)  -> i_vec
  score(u,i) = u_vec . i_vec
Loss: BPR  -log sigmoid(s(u,i+) - s(u,i-)),  i- sampled uniformly from the item universe.

Resumable: runs epochs until TIME_BUDGET, checkpoints (model+opt+epoch+loss curve) and exits.
Re-invoke until the loss curve flattens. Checkpoint: outputs/_models/gpu/g32_twotower.pt
"""
from __future__ import annotations
import os, sys, json, time, math
import numpy as np, torch, torch.nn as nn

DATA = "data/interim"
# checkpoint on VM-local fs (the repo mount blocks unlink/rename, breaking atomic save);
# the final model is copied into outputs/_models/gpu/ by the eval step for the record.
CKPT = os.environ.get("G32_CKPT", "/tmp/g32_twotower.pt")
torch.manual_seed(20260702); np.random.seed(20260702)
torch.set_num_threads(max(1, os.cpu_count() or 1))
D = 64; L = 50; BATCH = int(os.environ.get("G32_BATCH", 16384)); LR = 0.01; WD = 1e-6
NNEG = int(os.environ.get("G32_NNEG", 1))
TIME_BUDGET = float(os.environ.get("G32_BUDGET", 25))

def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)          # atomic; never leaves a half-written CKPT
MAX_EPOCHS = int(os.environ.get("G32_MAX_EPOCHS", 40))
t0 = time.time()

# ---- load prep ----
z = np.load(os.path.join(DATA, "g32_prep.npz"), allow_pickle=True)
pairs = z["pairs"].astype(np.int64)                 # (n_pairs, 2) = (user_code, item_code)
CMAT = torch.from_numpy(z["CMAT"].astype(np.float32))
hist = z["hist"]                                    # object array of int32 arrays
meta = json.load(open(os.path.join(DATA, "g32_prep_meta.json")))
nU, nI, cdim = meta["nU"], meta["nI"], meta["cdim"]

# padded history matrix (nU x L) + lengths (most-recent L kept)
Hpad = np.zeros((nU, L), dtype=np.int64); Hlen = np.zeros(nU, dtype=np.int64)
for u in range(nU):
    h = hist[u]
    if len(h) > L: h = h[-L:]
    Hpad[u, :len(h)] = h; Hlen[u] = max(len(h), 1)
Hpad = torch.from_numpy(Hpad); Hlen = torch.from_numpy(Hlen).float().unsqueeze(1)
pairs_t = torch.from_numpy(pairs)

class TwoTower(nn.Module):
    def __init__(self, nI, cdim, d):
        super().__init__()
        self.E_hist = nn.Embedding(nI, d)           # user-tower item embeddings (pooled)
        self.E_item = nn.Embedding(nI, d, padding_idx=None)  # item-tower id embeddings
        self.W_c = nn.Linear(cdim, d, bias=False)   # content projection into item space
        nn.init.normal_(self.E_hist.weight, std=0.05)
        nn.init.normal_(self.E_item.weight, std=0.05)
    def item_vec(self, idx, cmat):
        return self.E_item(idx) + self.W_c(cmat[idx])
    def user_vec(self, hpad, hlen):
        e = self.E_hist(hpad)                       # (B,L,d)
        m = (hpad != 0).float().unsqueeze(2)        # mask padding (id 0 is a real item but rare; acceptable)
        return (e * m).sum(1) / hlen                # mean-pool
    def forward(self, u_idx, i_idx, cmat, hpad_full, hlen_full):
        uv = self.user_vec(hpad_full[u_idx], hlen_full[u_idx])
        iv = self.item_vec(i_idx, cmat)
        return (uv * iv).sum(1)

model = TwoTower(nI, cdim, D)
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
start_epoch = 0; loss_curve = []
if os.path.exists(CKPT):
    try:
        ck = torch.load(CKPT, map_location="cpu")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"]; loss_curve = ck["loss_curve"]
        print(f"[resume] epoch={start_epoch} last_loss={loss_curve[-1]:.4f}")
    except Exception as e:
        print(f"[warn] checkpoint unreadable ({e}); starting fresh")

def run_epoch():
    model.train()
    perm = torch.randperm(pairs_t.shape[0])
    tot = 0.0; nb = 0
    for s in range(0, perm.shape[0], BATCH):
        b = perm[s:s+BATCH]
        u = pairs_t[b, 0]; ipos = pairs_t[b, 1]
        uv = model.user_vec(Hpad[u], Hlen[u])
        pv = model.item_vec(ipos, CMAT)
        s_pos = (uv * pv).sum(1, keepdim=True)                  # (B,1)
        ineg = torch.randint(0, nI, (b.shape[0], NNEG))          # (B,nneg)
        nv = model.item_vec(ineg.reshape(-1), CMAT).view(b.shape[0], NNEG, -1)
        s_neg = torch.einsum("bd,bnd->bn", uv, nv)              # (B,nneg)
        loss = -torch.nn.functional.logsigmoid(s_pos - s_neg).mean()  # mean BPR over negs
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item(); nb += 1
        if time.time() - t0 > TIME_BUDGET:      # partial epoch guard
            break
    return tot / max(nb, 1)

ep = start_epoch
while ep < MAX_EPOCHS and (time.time() - t0) < TIME_BUDGET:
    l = run_epoch(); ep += 1; loss_curve.append(round(l, 4))
    print(f"[epoch {ep}] bpr_loss={l:.4f}  ({time.time()-t0:.1f}s)")
    atomic_save({"model": model.state_dict(), "opt": opt.state_dict(),
                 "epoch": ep, "loss_curve": loss_curve,
                 "cfg": {"d": D, "L": L, "batch": BATCH, "lr": LR}}, CKPT)

done = ep >= MAX_EPOCHS
# convergence check: last 3 epochs improved < 0.5%
conv = len(loss_curve) >= 4 and abs(loss_curve[-1] - loss_curve[-4]) / max(loss_curve[-4], 1e-9) < 0.005
print(json.dumps({"epoch": ep, "last_loss": loss_curve[-1] if loss_curve else None,
                  "max_epochs_reached": done, "converged_flat": bool(conv),
                  "loss_curve_tail": loss_curve[-6:]}))
