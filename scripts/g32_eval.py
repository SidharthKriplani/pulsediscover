"""G32 eval — two-tower Recall@K on the EXACT ALS-floor protocol (eval_sample.csv, 5k users,
leave-last-out, history-masked, top-200). Reports warm headline vs ALS R@20=0.0846, a cold_eval
cohort, and a content-OFF ablation (zero the content projection) to isolate the content signal.

Writes outputs/evidence/G32_two_tower_eval.json. Also copies the final checkpoint into the repo.
"""
from __future__ import annotations
import os, sys, json, time, shutil
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, "src")
from pulsediscover import eval as E

DATA = "data/interim"; EVID = "outputs/evidence"
CKPT = os.environ.get("G32_CKPT", "/tmp/g32_twotower.pt")
REPO_CKPT = "outputs/_models/gpu/g32_twotower_final.pt"
KS = [5, 10, 20, 50, 100, 200]; NK = [10, 20, 50]; ALS_FLOOR = 0.0846
t0 = time.time()

z = np.load(os.path.join(DATA, "g32_prep.npz"), allow_pickle=True)
CMAT = torch.from_numpy(z["CMAT"].astype(np.float32))
hist = z["hist"]; meta = json.load(open(os.path.join(DATA, "g32_prep_meta.json")))
nU, nI, cdim, nWarm = meta["nU"], meta["nI"], meta["cdim"], meta["nWarm"]
L = 50
Hpad = np.zeros((nU, L), dtype=np.int64); Hlen = np.zeros(nU, dtype=np.float32)
for u in range(nU):
    h = hist[u]
    if len(h) > L: h = h[-L:]
    Hpad[u, :len(h)] = h; Hlen[u] = max(len(h), 1)
Hpad_t = torch.from_numpy(Hpad); Hlen_t = torch.from_numpy(Hlen).unsqueeze(1)

class TwoTower(nn.Module):
    def __init__(self, nI, cdim, d):
        super().__init__()
        self.E_hist = nn.Embedding(nI, d); self.E_item = nn.Embedding(nI, d)
        self.W_c = nn.Linear(cdim, d, bias=False)
ck = torch.load(CKPT, map_location="cpu"); D = ck["cfg"]["d"]
model = TwoTower(nI, cdim, D); model.load_state_dict(ck["model"]); model.eval()
print(f"[load] epoch={ck['epoch']} loss={ck['loss_curve'][-1]:.4f} d={D}")

@torch.no_grad()
def item_matrix(content_on=True):
    iv = model.E_item.weight.clone()
    if content_on: iv = iv + model.W_c(CMAT)
    return iv                                     # (nI, d)

@torch.no_grad()
def user_vecs(u_idx):
    e = model.E_hist(Hpad_t[u_idx]); m = (Hpad_t[u_idx] != 0).float().unsqueeze(2)
    return (e * m).sum(1) / Hlen_t[u_idx]

@torch.no_grad()
def evaluate(u_arr, g_arr, content_on=True, batch=400):
    IV = item_matrix(content_on)
    R = {k: [] for k in KS}; Nm = {k: [] for k in NK}
    for s in range(0, len(u_arr), batch):
        ub = u_arr[s:s+batch]; gb = g_arr[s:s+batch]
        valid = ub >= 0
        uv = user_vecs(torch.from_numpy(np.where(valid, ub, 0)))
        scores = uv @ IV.T                        # (B, nI)
        for bi in range(len(ub)):
            g = int(gb[bi])
            if not valid[bi] or g < 0:
                for k in KS: R[k].append(0.0)
                for k in NK: Nm[k].append(0.0);
                continue
            row = scores[bi].clone()
            h = hist[int(ub[bi])]
            if len(h): row[torch.from_numpy(h.astype(np.int64))] = -1e9
            top = torch.topk(row, 200).indices.tolist()
            for k in KS: R[k].append(E.recall_at_k(top, g, k))
            for k in NK: Nm[k].append(E.ndcg_at_k(top, g, k))
    out = {}
    for k in KS:
        rm, lo, hi, n = E.bootstrap_ci(R[k]); out[f"R@{k}"] = {"mean": round(rm, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    for k in NK: out[f"N@{k}"] = round(float(np.mean(Nm[k])), 4)
    out["n"] = len(u_arr)
    return out

warm = evaluate(z["warm_u"], z["warm_g"], content_on=True)
warm_off = evaluate(z["warm_u"], z["warm_g"], content_on=False)
cold = evaluate(z["cold_u"], z["cold_g"], content_on=True)
print(f"[warm] R@20={warm['R@20']['mean']} (content-off {warm_off['R@20']['mean']}) vs ALS {ALS_FLOOR}")

verdict = ("two_tower_wins" if warm["R@20"]["mean"] > ALS_FLOOR else
           "als_wins_honest_negative")
delta = round(warm["R@20"]["mean"] - ALS_FLOOR, 4)
report = {
    "gate": "G32", "model": "two_tower_bpr", "framework": "pytorch_cpu",
    "protocol": meta["protocol"], "n_eval_users": int(warm["n"]),
    "config": {"d": D, "L": L, "loss": "BPR", "neg": "uniform", "epochs": ck["epoch"]},
    "train_loss_curve": ck["loss_curve"],
    "als_floor_R@20": ALS_FLOOR,
    "two_tower_warm": warm, "two_tower_warm_content_off": warm_off,
    "two_tower_cold_eval": cold,
    "content_lift_R@20": round(warm["R@20"]["mean"] - warm_off["R@20"]["mean"], 4),
    "verdict": verdict, "delta_vs_als_R@20": delta,
    "v2_minilm_cold_lane_R@20_ref": 0.241,
    "truth_boundary": "offline leave-last-out; single held-out gold/user; no online lift claimed",
    "sec": round(time.time() - t0, 1),
}
os.makedirs(EVID, exist_ok=True)
json.dump(report, open(os.path.join(EVID, "G32_two_tower_eval.json"), "w"), indent=2)
try:
    with open(CKPT, "rb") as f, open(REPO_CKPT, "wb") as g: g.write(f.read())
except Exception as e:
    print("[warn] repo checkpoint copy:", e)
print(json.dumps({"verdict": verdict, "warm_R@20": warm["R@20"]["mean"],
                  "content_off_R@20": warm_off["R@20"]["mean"], "delta_vs_als": delta,
                  "cold_R@20": cold["R@20"]["mean"], "sec": report["sec"]}, indent=2))
