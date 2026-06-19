"""
PulseDiscover — Phase 2 driver (retrieval baselines only).

Behaviour:
  * If a real dataset is present in data/raw/  -> run baselines on it, write
    outputs/evidence/retrieval_baselines.json (+ plot)  [REAL EVIDENCE].
  * Else -> run a tiny, clearly-labelled CODE-VALIDATION fixture to prove the
    pipeline executes, writing to outputs/_codecheck/ ONLY. This fixture is NOT
    evidence, NOT the Phase-4 simulator, and its numbers are never reported as
    findings. retrieval_baselines.json is written with status AWAITING_DATA.

No deep models. No simulator. No network. Phase 2 stops at Gate 2.
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from pulsediscover import data as D          # noqa
from pulsediscover import baselines as B     # noqa
from pulsediscover import eval as E          # noqa

RAW = os.path.join(ROOT, "data", "raw")
EVID = os.path.join(ROOT, "outputs", "evidence")
PLOTS = os.path.join(ROOT, "outputs", "plots")
CODECHECK = os.path.join(ROOT, "outputs", "_codecheck")
SEED = 20260616
KS = [5, 10, 20]


def _make_codecheck_fixture(n_users=200, n_items=80, seed=SEED):
    """Tiny deterministic fixture to validate the code path. NOT EVIDENCE."""
    rng = np.random.default_rng(seed)
    pop = rng.zipf(1.4, n_items); pop = pop / pop.sum()
    rows, t0 = [], 1_600_000_000
    for u in range(n_users):
        seqlen = int(rng.integers(5, 20))
        items = rng.choice(n_items, size=seqlen, replace=False, p=pop / pop.sum())
        for s, it in enumerate(items):
            rows.append(dict(user_id=f"u{u}", item_id=f"i{int(it)}", creator_id=None,
                             series_id=None, event_ts=t0 + u * 1000 + s * 10,
                             rating=None, event_type="rating"))
    return D._finalize(pd.DataFrame(rows))


def run_baselines(df: pd.DataFrame, label: str, out_dir: str, is_evidence: bool):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(PLOTS if is_evidence else CODECHECK, exist_ok=True)
    # splits
    train_g, val_g, test_g, (t1, t2) = D.global_time_split(df)
    train, test = D.leave_last_out_constrained(df, t_cutoff=t2)
    cold = D.cold_item_flags(train, test)
    # fit baselines on train
    pop_top = B.popularity_topk(train, max(KS))
    neigh = B.cooccurrence_pmi(train)
    als = None
    try:
        als = B.ALS(factors=16, iters=8).fit(train)
    except Exception as ex:
        als_err = f"{type(ex).__name__}: {ex}"
    # evaluate per held-out user
    results = {m: {k: [] for k in KS} for m in ["popularity", "cooccurrence", "als"]}
    hist_by_user = train.groupby("user_id")["item_id"].apply(set).to_dict()
    for _, row in test.iterrows():
        u, gold = row["user_id"], row["item_id"]
        hist = hist_by_user.get(u, set())
        recs = {
            "popularity": [i for i in pop_top if i not in hist][:max(KS)],
            "cooccurrence": B.cooccurrence_user_topk(neigh, hist, max(KS)),
            "als": als.recommend(u, max(KS), exclude=hist) if als else [],
        }
        for m in results:
            for k in KS:
                results[m][k].append({"r": E.recall_at_k(recs[m], gold, k),
                                      "n": E.ndcg_at_k(recs[m], gold, k)})
    # aggregate with CIs
    agg = {}
    for m in results:
        agg[m] = {}
        for k in KS:
            rec_vals = [d["r"] for d in results[m][k]]
            ndcg_vals = [d["n"] for d in results[m][k]]
            rm, rlo, rhi, n = E.bootstrap_ci(rec_vals, seed=SEED)
            wm, wlo, whi = E.wilson_ci(int(sum(rec_vals)), len(rec_vals))
            nm, nlo, nhi, _ = E.bootstrap_ci(ndcg_vals, seed=SEED)
            agg[m][f"@{k}"] = {
                "recall_mean": round(rm, 4), "recall_ci95_boot": [round(rlo, 4), round(rhi, 4)],
                "recall_ci95_wilson": [round(wlo, 4), round(whi, 4)],
                "ndcg_mean": round(nm, 4), "ndcg_ci95_boot": [round(nlo, 4), round(nhi, 4)],
                "n_eval_users": n}
    payload = {
        "label": label,
        "is_evidence": is_evidence,
        "NOTE": "REAL EVIDENCE" if is_evidence else "CODE-VALIDATION FIXTURE — NOT EVIDENCE, NOT THE SIMULATOR",
        "tag": "[BUILT — real data]" if is_evidence else "[CODE-CHECK ONLY]",
        "seed": SEED,
        "split": {"global_time_cutoffs": [int(t1), int(t2)],
                  "n_train": int(len(train)), "n_test_heldout": int(len(test))},
        "cold_item_stats": cold,
        "baselines": agg,
        "methods": {"popularity": "global frequency", "cooccurrence": "item-item PMI, min_co=2",
                     "als": "from-scratch confidence-weighted implicit ALS (f=16, iters=8)"},
        "ci_methods": {"recall_ci95_boot": "per-user bootstrap (2000)", "recall_ci95_wilson": "Wilson score",
                        "ndcg_ci95_boot": "per-user bootstrap (2000)"},
        "ts": int(time.time()),
    }
    fn = os.path.join(out_dir, ("retrieval_baselines.json" if is_evidence else "toy_retrieval_baselines.json"))
    with open(fn, "w") as f:
        json.dump(payload, f, indent=2)
    # plot
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ks = KS; fig, ax = plt.subplots(figsize=(6, 4))
        for m in agg:
            ax.plot(ks, [agg[m][f"@{k}"]["recall_mean"] for k in ks], marker="o", label=m)
        ax.set_xlabel("K"); ax.set_ylabel("Recall@K"); ax.set_title(f"Retrieval baselines — {label}")
        ax.legend()
        pdir = PLOTS if is_evidence else CODECHECK
        plt.tight_layout(); plt.savefig(os.path.join(pdir, "retrieval_baselines_recall.png"), dpi=110)
        plt.close()
    except Exception as ex:
        print("plot skipped:", ex)
    return payload, fn


def main():
    os.makedirs(EVID, exist_ok=True)
    df, source = D.autodetect_and_load(RAW, max_rows=2_000_000)
    manifest = {"phase": 2, "seed": SEED, "schema": D.CANON_COLS,
                "split_design": {"primary": "global_time_split(0.8/0.9)",
                                  "secondary": "leave_last_out_constrained(t>=t2)"},
                "ci_method_notes": "docs/04_CI_METHOD_NOTES.md"}
    if df is not None:
        manifest.update({"status": "DATA_PRESENT", "source": source,
                         "n_interactions": int(len(df)),
                         "n_users": int(df.user_id.nunique()), "n_items": int(df.item_id.nunique()),
                         "ts_min": int(df.event_ts.min()), "ts_max": int(df.event_ts.max())})
        payload, fn = run_baselines(df, source, EVID, is_evidence=True)
        print("REAL run ->", fn)
    else:
        manifest.update({"status": "AWAITING_DATA",
                         "note": "No dataset in data/raw/. Drop MovieLens-1M (ratings.dat) or a "
                                 "Goodreads genre interactions json(.gz) or Amazon Books jsonl(.gz). "
                                 "Then re-run: python src/run_phase2.py"})
        # placeholder so downstream knows nothing real exists yet
        with open(os.path.join(EVID, "retrieval_baselines.json"), "w") as f:
            json.dump({"status": "AWAITING_DATA", "is_evidence": False,
                       "note": "Real baselines pending dataset in data/raw/."}, f, indent=2)
        payload, fn = run_baselines(_make_codecheck_fixture(), "CODECHECK_FIXTURE", CODECHECK, is_evidence=False)
        print("CODE-CHECK run ->", fn)
    with open(os.path.join(EVID, "data_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("manifest status:", manifest["status"])


if __name__ == "__main__":
    main()
