"""
PulseDiscover — Phase 3A driver: SASRec on MovieLens-1M (smoke), CPU-safe.

- Identical split to Phase 2 (global_time_split 0.8/0.9 -> leave_last_out_constrained).
- Identical eval (Recall@K / NDCG@K, per-user bootstrap CI) and identical held-out set.
- Time-budgeted, checkpoint/resumable training (respects the 45s shell cap): trains
  epoch-by-epoch until a wall-clock budget, checkpoints each epoch; re-run to add epochs.
- Honest comparison vs popularity/co-occurrence/ALS. A win is NOT required.

No two-tower. No simulator. No Goodreads claims. No PDF. Stops at Gate 3A.
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from pulsediscover import data as D, eval as E, sasrec as S  # noqa

RAW = os.path.join(ROOT, "data", "raw")
EVID = os.path.join(ROOT, "outputs", "evidence")
PLOTS = os.path.join(ROOT, "outputs", "plots")
MODELS = os.path.join(ROOT, "outputs", "_models")
SEED = 20260616
KS = [5, 10, 20]
MAXLEN, D_MODEL, NBLOCKS, NHEADS, DROPOUT = 30, 48, 2, 1, 0.2
BATCH, LR = 256, 1e-3
TRAIN_BUDGET_S, MAX_EPOCHS = 28.0, 60
torch.manual_seed(SEED); np.random.seed(SEED); torch.set_num_threads(4)


def main():
    os.makedirs(MODELS, exist_ok=True); os.makedirs(EVID, exist_ok=True); os.makedirs(PLOTS, exist_ok=True)
    df, source = D.autodetect_and_load(RAW)
    assert source == "movielens-1m", f"Phase 3A is MovieLens-1M smoke only; found {source}"
    _, _, _, (t1, t2) = D.global_time_split(df)
    train, test = D.leave_last_out_constrained(df, t_cutoff=t2)
    item2idx = {it: i + 1 for i, it in enumerate(train["item_id"].unique())}  # 0 = pad
    n_items = len(item2idx)
    user_seqs = S.build_user_sequences(train, item2idx)
    X, Y = S.make_training_tensors(user_seqs, MAXLEN)
    test_pairs = list(zip(test["user_id"].to_numpy(), test["item_id"].to_numpy()))

    model = S.SASRec(n_items, d=D_MODEL, maxlen=MAXLEN, nblocks=NBLOCKS, nheads=NHEADS, dropout=DROPOUT)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ckpt = os.path.join(MODELS, "sasrec_phase3a.pt")
    epochs_done, losses = 0, []
    if os.path.exists(ckpt):
        st = torch.load(ckpt, weights_only=False)
        if st.get("n_items") == n_items:
            model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
            epochs_done = st["epochs_done"]; losses = st.get("losses", [])
            print(f"resumed from epoch {epochs_done}")

    t0 = time.time()
    while (time.time() - t0) < TRAIN_BUDGET_S and epochs_done < MAX_EPOCHS:
        loss = S.train_one_epoch(model, X, Y, opt, batch=BATCH, seed=SEED + epochs_done)
        epochs_done += 1; losses.append(round(loss, 4))
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epochs_done": epochs_done, "losses": losses, "n_items": n_items}, ckpt)
        print(f"epoch {epochs_done} loss {loss:.4f} ({time.time()-t0:.1f}s)")

    # evaluate (same held-out set + metrics as baselines)
    res = S.evaluate(model, user_seqs, test_pairs, item2idx, MAXLEN, KS=tuple(KS))
    agg = {}
    for k in KS:
        rv = [d["r"] for d in res["sasrec"][k]]; nv = [d["n"] for d in res["sasrec"][k]]
        rm, rlo, rhi, n = E.bootstrap_ci(rv, seed=SEED)
        wm, wlo, whi = E.wilson_ci(int(sum(rv)), len(rv))
        nm, nlo, nhi, _ = E.bootstrap_ci(nv, seed=SEED)
        agg[f"@{k}"] = {"recall_mean": round(rm, 4), "recall_ci95_boot": [round(rlo, 4), round(rhi, 4)],
                        "recall_ci95_wilson": [round(wlo, 4), round(whi, 4)],
                        "ndcg_mean": round(nm, 4), "ndcg_ci95_boot": [round(nlo, 4), round(nhi, 4)],
                        "n_eval_users": n}

    # load baselines for honest comparison
    base = {}
    bpath = os.path.join(EVID, "retrieval_baselines.json")
    if os.path.exists(bpath):
        bj = json.load(open(bpath))
        if bj.get("is_evidence"):
            base = bj.get("baselines", {})

    payload = {
        "model": "SASRec", "dataset": source, "is_evidence": True, "tag": "[BUILT — real data]",
        "role": "SMOKE TEST (MovieLens-1M) — not the domain headline (Goodreads/Amazon pending)",
        "seed": SEED,
        "config": {"d": D_MODEL, "blocks": NBLOCKS, "heads": NHEADS, "maxlen": MAXLEN,
                    "dropout": DROPOUT, "batch": BATCH, "lr": LR},
        "train": {"epochs_done": epochs_done, "loss_history": losses,
                   "n_train_users": int(len(user_seqs)), "n_train_examples": int(X.shape[0])},
        "split": {"global_time_cutoffs": [int(t1), int(t2)], "n_test_heldout": int(len(test_pairs))},
        "sasrec": agg,
        "baselines_for_comparison": base,
        "honest_note": "A win is not required; result reported as-is with CIs. Epochs limited by CPU/time budget; re-run continues training.",
        "ci_methods": {"recall_ci95_boot": "per-user bootstrap (2000)", "recall_ci95_wilson": "Wilson"},
        "ts": int(time.time()),
    }
    with open(os.path.join(EVID, "sasrec_report.json"), "w") as f:
        json.dump(payload, f, indent=2)

    # comparison plot: Recall@10 with bootstrap CI error bars
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        models = ["popularity", "cooccurrence", "als", "sasrec"]
        vals, los, his = [], [], []
        for m in models:
            a = (agg if m == "sasrec" else base.get(m, {})).get("@10")
            if a:
                vals.append(a["recall_mean"]); los.append(a["recall_mean"] - a["recall_ci95_boot"][0])
                his.append(a["recall_ci95_boot"][1] - a["recall_mean"])
            else:
                vals.append(0); los.append(0); his.append(0)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(models, vals, yerr=[los, his], capsize=5,
               color=["#bbb", "#bbb", "#bbb", "#3b7"])
        ax.set_ylabel("Recall@10"); ax.set_title("SASRec vs baselines — MovieLens-1M smoke (95% boot CI)")
        plt.tight_layout(); plt.savefig(os.path.join(PLOTS, "sasrec_vs_baselines_recall10.png"), dpi=110); plt.close()
    except Exception as ex:
        print("plot skipped:", ex)

    print("epochs:", epochs_done, "| SASRec Recall@10:", agg["@10"]["recall_mean"],
          "| ALS Recall@10:", base.get("als", {}).get("@10", {}).get("recall_mean"))


if __name__ == "__main__":
    main()
