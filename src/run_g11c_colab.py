"""G11C/G11D — Colab/GPU consolidated sequence batch (self-contained, disconnect-resilient).
Runs several models in priority order, writing g11c_colab_metrics.json AFTER EACH model so a
disconnect never loses finished work. Keeps the EXACT G11/G11B eval (5k-user LLO, full 40k scoring,
R@20/50 + NDCG@20 + bootstrap CI). GPU if available, else CPU.

PRIORITY QUEUE (most valuable first):
  1. SASRec CANONICAL — FULL-SOFTMAX over all 40k items (impossible on CPU; the real T3 attempt)
  2. SASRec v2 d64 sampled-neg, 30 ep  (the under-training diagnostic vs CPU 8 ep)
  3. GRU4Rec d64 FULL-SOFTMAX, 30 ep
  4. SASRec-small d48 last-position, 20 ep  (clean-pipeline sanity ~= v1)

Required uploads (same 7 files as before): run_g11c_colab.py, g11_fullpos_arrays.npz,
d3b_seq_meta.pkl, eval_sample.csv, domain_als_f64_floor.json, domain_sasrec_report.json,
g11_sequence_tournament_metrics.json. NO eval changes; NO fabricated numbers."""
import os, sys, json, time, math, csv, pickle, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GPU_NAME = torch.cuda.get_device_name(0) if DEV == "cuda" else "CPU-only (no GPU)"
SEED = 20260616; MAXLEN = 30; NNEG = 100
torch.manual_seed(SEED); np.random.seed(SEED)

class SASBlock(nn.Module):
    def __init__(s, d, h, p):
        super().__init__(); s.attn = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        s.ln1 = nn.LayerNorm(d); s.ln2 = nn.LayerNorm(d)
        s.ff = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d), nn.Dropout(p))
    def forward(s, x, m, kpm):
        a, _ = s.attn(x, x, x, attn_mask=m, key_padding_mask=kpm, need_weights=False)
        x = s.ln1(x + a); return s.ln2(x + s.ff(x))

class SASRec(nn.Module):
    def __init__(s, n_items, d=64, maxlen=30, nblocks=2, nheads=2, dropout=0.2):
        super().__init__(); s.maxlen = maxlen
        s.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0); s.pos_emb = nn.Embedding(maxlen, d)
        s.drop = nn.Dropout(dropout); s.blocks = nn.ModuleList([SASBlock(d, nheads, dropout) for _ in range(nblocks)]); s.ln = nn.LayerNorm(d)
    def seq_repr(s, seq):
        B, L = seq.shape; pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = s.drop(s.item_emb(seq) + s.pos_emb(pos))
        causal = torch.triu(torch.ones(L, L, device=seq.device), 1).bool()
        for blk in s.blocks: x = blk(x, causal, seq == 0)
        return s.ln(x)
    def forward(s, seq): h = s.seq_repr(seq); return h @ s.item_emb.weight.T   # (B,L,V) full logits
    def score_last(s, seq): h = s.seq_repr(seq)[:, -1, :]; return h @ s.item_emb.weight.T

class GRU4Rec(nn.Module):
    def __init__(s, n_items, d=64, layers=1, dropout=0.2, maxlen=30):
        super().__init__(); s.maxlen = maxlen; s.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)
        s.drop = nn.Dropout(dropout); s.gru = nn.GRU(d, d, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
    def seq_repr(s, seq): x = s.drop(s.item_emb(seq)); o, _ = s.gru(x); return o
    def forward(s, seq): h = s.seq_repr(seq); return h @ s.item_emb.weight.T
    def score_last(s, seq): h = s.seq_repr(seq)[:, -1, :]; return h @ s.item_emb.weight.T

class TwoTower(nn.Module):
    """G11D — neural two-tower RETRIEVAL baseline (item-ID only => neural collaborative retrieval,
    NOT content cold-start, NOT production-scale). User tower = MLP over mean-pooled history item
    embeddings; item tower = same item-embedding table. Trained with in-batch negatives."""
    def __init__(s, n_items, d=64, dropout=0.2, maxlen=30):
        super().__init__(); s.maxlen = maxlen; s.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)
        s.drop = nn.Dropout(dropout); s.mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
    def user_vec(s, seq):
        e = s.item_emb(seq); mask = (seq != 0).float().unsqueeze(-1)
        pooled = (e * mask).sum(1) / mask.sum(1).clamp(min=1)        # mean over non-pad history
        return s.mlp(s.drop(pooled))                                  # (B,d)
    def score_last(s, seq): return s.user_vec(seq) @ s.item_emb.weight.T   # (B,V) over all items

def fullsoftmax_loss(model, xb, yb, n_items):
    logits = model.forward(xb)                                  # (B,L,V) — full softmax over all items
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1), ignore_index=0)

def sampledneg_fullpos_loss(model, xb, yb, n_items):
    h = model.seq_repr(xb); pl = (h * model.item_emb(yb)).sum(-1)
    neg = torch.randint(1, n_items, (xb.shape[0], NNEG), device=xb.device); ne = model.item_emb(neg)
    nl = torch.einsum("bld,bnd->bln", h, ne); mask = (yb != 0).float()
    lp = (F.binary_cross_entropy_with_logits(pl, torch.ones_like(pl), reduction="none") * mask).sum()
    ln = (F.binary_cross_entropy_with_logits(nl, torch.zeros_like(nl), reduction="none").mean(-1) * mask).sum()
    return (lp + ln) / mask.sum().clamp(min=1)

def sampledneg_lastpos_loss(model, xb, yb, n_items):
    h = model.seq_repr(xb)[:, -1, :]; tgt = yb[:, -1]; pl = (h * model.item_emb(tgt)).sum(-1)
    neg = torch.randint(1, n_items, (xb.shape[0], NNEG), device=xb.device); ne = model.item_emb(neg)
    nl = torch.einsum("bd,bnd->bn", h, ne)
    return F.binary_cross_entropy_with_logits(pl, torch.ones_like(pl)) + F.binary_cross_entropy_with_logits(nl, torch.zeros_like(nl))

def twotower_inbatch_loss(model, xb, yb, n_items):
    uv = model.user_vec(xb); tgt = yb[:, -1]; te = model.item_emb(tgt)   # (B,d),(B,d)
    logits = uv @ te.T                                                   # (B,B) in-batch negatives
    return F.cross_entropy(logits, torch.arange(xb.shape[0], device=xb.device))

LOSSES = {"fullsoftmax": fullsoftmax_loss, "sampledneg_fullpos": sampledneg_fullpos_loss,
          "sampledneg_lastpos": sampledneg_lastpos_loss, "twotower_inbatch": twotower_inbatch_loss}

def bootstrap_ci(vals, n=300, seed=SEED):
    a = np.array(vals); r = np.random.default_rng(seed)
    bs = [a[r.integers(0, len(a), len(a))].mean() for _ in range(n)]
    return float(a.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))

def train(model, X, Y, n_items, objective, epochs, batch, lr, log_rows, tag):
    model.to(DEV); opt = torch.optim.Adam(model.parameters(), lr=lr); N = X.shape[0]; lossfn = LOSSES[objective]; curve = []
    for ep in range(epochs):
        te = time.time(); model.train(); perm = torch.randperm(N); tot = nb = 0
        for i in range(0, N, batch):
            idx = perm[i:i + batch]; xb = X[idx].to(DEV); yb = Y[idx].to(DEV)
            loss = lossfn(model, xb, yb, n_items); opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
        curve.append(round(tot / nb, 4)); log_rows.append([tag, ep + 1, round(tot / nb, 4), round(time.time() - te, 1)])
        print(f"  {tag} ep {ep+1}/{epochs} loss {curve[-1]} ({log_rows[-1][3]}s)", flush=True)
    return curve

def evaluate(model, eval_seq, users, golds, item2idx, maxlen):
    model.eval()
    def lp(s): s = s[-maxlen:]; return [0] * (maxlen - len(s)) + list(s)
    KS = [20, 50]; R = {k: [] for k in KS}; nd = []; B = 1000
    with torch.no_grad():
        for s0 in range(0, len(users), B):
            ub = users[s0:s0 + B]; xb = torch.tensor([lp(eval_seq.get(u, [])) for u in ub], dtype=torch.long, device=DEV)
            sc = model.score_last(xb).cpu().numpy()
            for bi, u in enumerate(ub):
                g = golds[s0 + bi]; gi = item2idx.get(g); row = sc[bi].copy(); row[0] = -1e9
                for it in eval_seq.get(u, []): row[it] = -1e9
                top = np.argpartition(-row, 50)[:50]; top = top[np.argsort(-row[top])]; ranked = list(top)
                for k in KS: R[k].append(1.0 if (gi is not None and gi in ranked[:k]) else 0.0)
                nd.append((1.0 / math.log2(ranked.index(gi) + 2)) if (gi is not None and gi in ranked[:20]) else 0.0)
    out = {}
    for k in KS:
        m, lo, hi = bootstrap_ci(R[k]); out[f"R@{k}"] = {"mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    out["N@20"] = {"mean": round(float(np.mean(nd)), 4)}; return out

# priority queue: (tag, model_factory, objective, epochs, batch, lr). Most valuable first;
# optionals gated by env G11C_OPTIONAL (default on; set 0 to skip and save GPU time).
def QUEUE(n_items):
    core = [
        ("SASRec_canonical_fullsoftmax_d64_40ep", lambda: SASRec(n_items, d=64, maxlen=MAXLEN, nblocks=2, nheads=2, dropout=0.2), "fullsoftmax", 40, 128, 1e-3),
        ("SASRec_small_d48_lastpos_20ep",         lambda: SASRec(n_items, d=48, maxlen=MAXLEN, nblocks=1, nheads=1, dropout=0.2), "sampledneg_lastpos", 20, 1024, 1e-3),
        ("TwoTower_neural_retrieval_d64_30ep_G11D", lambda: TwoTower(n_items, d=64, dropout=0.2, maxlen=MAXLEN), "twotower_inbatch", 30, 512, 1e-3),
    ]
    optional = [
        ("SASRec_v2_sampledneg_d64_30ep",         lambda: SASRec(n_items, d=64, maxlen=MAXLEN, nblocks=2, nheads=2, dropout=0.2), "sampledneg_fullpos", 30, 512, 1e-3),
        ("GRU4Rec_fullsoftmax_d64_30ep",          lambda: GRU4Rec(n_items, d=64, layers=1, dropout=0.2, maxlen=MAXLEN), "fullsoftmax", 30, 256, 1e-3),
    ]
    return core + (optional if os.environ.get("G11C_OPTIONAL", "1") == "1" else [])

def main(datadir=".", outdir="."):
    print("Device:", DEV, "| GPU:", GPU_NAME, "| torch", torch.__version__, flush=True)
    z = np.load(os.path.join(datadir, "g11_fullpos_arrays.npz")); X = torch.tensor(z["X"]); Y = torch.tensor(z["Y"])
    meta = pickle.load(open(os.path.join(datadir, "d3b_seq_meta.pkl"), "rb")); item2idx = meta["item2idx"]; n_items = meta["n_items"]; eval_seq = meta["eval_seq"]
    samp = pd.read_csv(os.path.join(datadir, "eval_sample.csv"), dtype=str); users = samp.user_id.tolist(); golds = samp.gold.tolist()
    f64 = json.load(open(os.path.join(datadir, "domain_als_f64_floor.json")))["metrics"]
    v1 = json.load(open(os.path.join(datadir, "domain_sasrec_report.json")))["metrics_SASRec"]
    g11 = json.load(open(os.path.join(datadir, "g11_sequence_tournament_metrics.json")))["results"]
    comparators = {"ALS_f64_floor": {k: f64[k] for k in ["R@20", "R@50", "N@20"]},
                   "SASRec_v1_d48_lastpos_20ep_CPU": {k: v1[k] for k in ["R@20", "R@50", "N@20"]},
                   "GRU4Rec_cpu_9ep": g11["GRU4Rec"], "SASRec_v2_cpu_fullpos_8ep": g11["SASRec_v2_fullpos"]}
    log_rows = [["model", "epoch", "loss", "sec"]]; results = {}; t0 = time.time()
    metricsp = os.path.join(outdir, "g11c_colab_metrics.json"); logp = os.path.join(outdir, "g11c_training_log.csv")
    def flush_results():   # incremental write — disconnect-safe
        json.dump({"scope": "G11C/D GPU consolidated sequence batch", "device": DEV, "gpu": GPU_NAME, "torch": torch.__version__,
                   "seed": SEED, "maxlen": MAXLEN, "eval": "5k-user LLO, full 40k scoring (unchanged)",
                   "elapsed_sec": round(time.time() - t0, 1), "results_gpu": results, "comparators": comparators},
                  open(metricsp, "w"), indent=2)
        with open(logp, "w", newline="") as f: csv.writer(f).writerows(log_rows)
    for tag, factory, obj, eps, bs, lr in QUEUE(n_items):
        print(f"=== {tag} (objective={obj}, {eps} ep, batch {bs}) ===", flush=True)
        try:
            m = factory(); curve = train(m, X, Y, n_items, obj, eps, bs, lr, log_rows, tag)
            results[tag] = {**evaluate(m, eval_seq, users, golds, item2idx, MAXLEN), "epochs": eps, "objective": obj, "loss_curve": curve}
            print(f"  -> R@20 {results[tag]['R@20']['mean']:.4f}  R@50 {results[tag]['R@50']['mean']:.4f}", flush=True)
            del m
            if DEV == "cuda": torch.cuda.empty_cache()
        except Exception as e:
            results[tag] = {"ERROR": str(e)}; print("  !! FAILED:", e, flush=True)
        flush_results()                       # write after every model
    # final comparison plot
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        names = list(comparators) + [k for k in results if "ERROR" not in results[k]]
        vals = [comparators.get(n, results.get(n))["R@20"]["mean"] for n in names]
        plt.figure(figsize=(12, 5)); plt.bar(range(len(names)), vals)
        plt.axhline(comparators["ALS_f64_floor"]["R@20"]["mean"], ls="--", c="green", label="ALS f64 floor")
        plt.xticks(range(len(names)), names, rotation=30, ha="right", fontsize=7); plt.ylabel("Recall@20"); plt.legend()
        plt.title("G11C/D: GPU sequence models vs ALS f64 floor"); plt.tight_layout()
        plt.savefig(os.path.join(outdir, "g11c_model_comparison.png"), dpi=110); plt.close()
    except Exception as e: print("plot skipped", e)
    print("BATCH DONE. R@20:", {k: results[k].get("R@20", {}).get("mean") for k in results}, flush=True)

if __name__ == "__main__":
    dd = sys.argv[1] if len(sys.argv) > 1 else "."; od = sys.argv[2] if len(sys.argv) > 2 else "."
    main(dd, od)
