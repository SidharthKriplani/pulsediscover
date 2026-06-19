"""
PulseDiscover — Option B: SASRec-based target policy π through the existing OPE harness.

Replaces the ALS-based π with the ACTUAL trained SASRec model's scores, and re-runs
IPS / SNIPS / DM / DR against the SAME known-propensity simulator. Unchanged: μ (popularity
logging policy) and the ground-truth reward (ALS-based simulator oracle) — so the
simulator/truth boundary is identical; only the policy being evaluated changes.

Answers: "can DR evaluate the actual SASRec policy, not just an ALS stand-in?"

No new model trained (reuses outputs/_models/sasrec_phase3a.pt). No domain. No online-lift.
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from pulsediscover import data as D, baselines as B, ope as O, sasrec as S  # noqa

EVID = os.path.join(ROOT, "outputs", "evidence"); PLOTS = os.path.join(ROOT, "outputs", "plots")
MODELS = os.path.join(ROOT, "outputs", "_models")
SEED = 20260616; B_BOOT = 400
CFG = {"epsilon": 0.10, "slate_L": 10, "pool_N": 50, "tau_mu": 1.0, "tau_pi": 0.7,
       "pbm_base": 0.95, "pbm_gamma": 0.7, "rel_alpha": 1.2, "rel_bias": -0.8,
       "qhat_good_alpha": 1.0, "qhat_good_bias": -0.6}
CLIPS = [None, 5, 10, 20, 50, 100]
MAXLEN, D_MODEL, NBLOCKS, NHEADS, DROPOUT = 30, 48, 2, 1, 0.2
torch.manual_seed(SEED); np.random.seed(SEED); torch.set_num_threads(4)


def main():
    df, source = D.autodetect_and_load(os.path.join(ROOT, "data", "raw"))
    assert source == "movielens-1m"
    _, _, _, (t1, t2) = D.global_time_split(df)
    train, _ = D.leave_last_out_constrained(df, t_cutoff=t2)
    als = B.ALS(factors=16, iters=8).fit(train)                      # truth source (unchanged)

    # reconstruct item2idx EXACTLY as Phase 3A, load the trained SASRec
    item2idx = {it: i + 1 for i, it in enumerate(train["item_id"].unique())}
    n_items = len(item2idx)
    ckpt = torch.load(os.path.join(MODELS, "sasrec_phase3a.pt"), weights_only=False)
    assert ckpt["n_items"] == n_items, "item space mismatch — reconstruction differs from Phase 3A"
    model = S.SASRec(n_items, d=D_MODEL, maxlen=MAXLEN, nblocks=NBLOCKS, nheads=NHEADS, dropout=DROPOUT)
    model.load_state_dict(ckpt["model"]); model.eval()

    # batch-score every user's full history -> SASRec score over all items
    user_seqs = S.build_user_sequences(train, item2idx)
    users = list(train["user_id"].unique())
    user_row = {u: i for i, u in enumerate(users)}
    seqs = [S._left_pad(user_seqs.get(u, []), MAXLEN) for u in users]
    Xs = torch.tensor(seqs, dtype=torch.long)
    rows = []
    with torch.no_grad():
        for i in range(0, len(Xs), 512):
            rows.append(model.score_last(Xs[i:i + 512]).numpy().astype(np.float32))
    score_matrix = np.vstack(rows)                                   # (n_users, n_items+1)

    def pi_sasrec(u, pool):
        idxs = np.fromiter((item2idx.get(it, 0) for it in pool), dtype=np.int64, count=len(pool))
        return score_matrix[user_row[u]][idxs]

    # re-run the OPE replay with π = SASRec (μ, truth, q̂ unchanged)
    A, nu, L = O.augmented_log(train, als, CFG, seed=SEED, pi_raw_for_pool=pi_sasrec)
    GT = float(np.mean(A["g"]))
    flat = {k: A[k].ravel() for k in A}
    eg0 = O.estimators(flat["mu"], flat["pi"], flat["r"], flat["qai_g"], flat["qpi_g"], None)
    eb0 = O.estimators(flat["mu"], flat["pi"], flat["r"], flat["qai_b"], flat["qpi_b"], None)
    bias = lambda v: round(v - GT, 6)
    headline = {
        "ground_truth_V_pi_SASRec": round(GT, 6),
        "IPS": {"est": round(eg0["IPS"], 6), "bias": bias(eg0["IPS"])},
        "SNIPS": {"est": round(eg0["SNIPS"], 6), "bias": bias(eg0["SNIPS"])},
        "DM_good_qhat": {"est": round(eg0["DM"], 6), "bias": bias(eg0["DM"])},
        "DR_good_qhat": {"est": round(eg0["DR"], 6), "bias": bias(eg0["DR"])},
        "DM_bad_qhat": {"est": round(eb0["DM"], 6), "bias": bias(eb0["DM"])},
        "DR_bad_qhat": {"est": round(eb0["DR"], 6), "bias": bias(eb0["DR"])},
        "ESS": round(eg0["ESS"], 1), "w_max": round(eg0["w_max"], 3), "w_mean": round(eg0["w_mean"], 4),
        "n_events": int(nu * L), "n_users": int(nu),
    }
    ci_good = O.cluster_bootstrap(A, "g", clip=None, B=B_BOOT, seed=SEED)
    ci_bad = O.cluster_bootstrap(A, "b", clip=None, B=B_BOOT, seed=SEED + 1)
    sweep = []
    for c in CLIPS:
        eg = O.estimators(flat["mu"], flat["pi"], flat["r"], flat["qai_g"], flat["qpi_g"], c)
        eb = O.estimators(flat["mu"], flat["pi"], flat["r"], flat["qai_b"], flat["qpi_b"], c)
        sweep.append({"clip": c, "ESS": round(eg["ESS"], 1), "IPS_bias": bias(eg["IPS"]),
                      "SNIPS_bias": bias(eg["SNIPS"]), "DR_good_bias": bias(eg["DR"]), "DR_bad_bias": bias(eb["DR"])})

    # compare to prior ALS-π results
    als_pi = {}
    p = os.path.join(EVID, "ope_comparison.json")
    if os.path.exists(p):
        aj = json.load(open(p)); h = aj["headline_no_clip"]
        als_pi = {"ground_truth_V_pi_ALS": h["ground_truth_V_pi"],
                  "IPS_bias": h["IPS"]["bias"], "SNIPS_bias": h["SNIPS"]["bias"],
                  "DR_good_bias": h["DR_good_qhat"]["bias"], "DR_bad_bias": h["DR_bad_qhat"]["bias"],
                  "DR_good_boot_std": aj["bootstrap_ci95_no_clip"]["good_qhat"]["DR"]["boot_std"],
                  "IPS_boot_std": aj["bootstrap_ci95_no_clip"]["good_qhat"]["IPS"]["boot_std"], "ESS": h["ESS"]}

    can_dr_eval = (abs(headline["DR_good_qhat"]["bias"]) <= 0.02 and
                   ci_good["DR"]["boot_std"] <= ci_good["IPS"]["boot_std"] and
                   abs(headline["DR_bad_qhat"]["bias"]) <= 0.03)

    payload = {
        "phase": "Option B — SASRec-π OPE", "dataset": source, "is_evidence": True,
        "tag": "[BUILT][SYNTHETIC — simulator]",
        "role": "OPE with π = trained SASRec scores (smoke + simulator; NOT real online lift)",
        "what_changed": "target policy π = softmax(SASRec scores) instead of softmax(ALS). "
                        "μ (popularity logging) and ground-truth reward (ALS oracle) UNCHANGED.",
        "truth_boundary": "Ground-truth reward remains ALS/simulator-based; only the evaluated policy is SASRec. "
                          "No real online lift claimed.",
        "seed": SEED, "config": CFG,
        "headline_no_clip": headline,
        "bootstrap_ci95_no_clip": {"good_qhat": ci_good, "bad_qhat": ci_bad, "B": B_BOOT,
                                   "method": "user-cluster bootstrap"},
        "clipping_sweep": sweep,
        "comparison_to_ALS_pi": als_pi,
        "can_DR_evaluate_SASRec_policy": bool(can_dr_eval),
        "honesty": "DR expected to track GT and beat IPS variance; reported as observed. Truth still ALS-based.",
        "ts": int(time.time()),
    }
    json.dump(payload, open(os.path.join(EVID, "ope_comparison_sasrec_pi.json"), "w"), indent=2)
    _plot(headline, ci_good, ci_bad, GT, als_pi)
    print(f"SASRec-π GT={GT:.4f} | IPS={eg0['IPS']:.4f}(b{bias(eg0['IPS'])}) SNIPS={eg0['SNIPS']:.4f}"
          f"(b{bias(eg0['SNIPS'])}) DR_good={eg0['DR']:.4f}(b{bias(eg0['DR'])}) "
          f"DM_bad={eb0['DM']:.4f}(b{bias(eb0['DM'])}) DR_bad={eb0['DR']:.4f}(b{bias(eb0['DR'])})")
    print(f"ESS={eg0['ESS']:.0f}/{nu*L} w_max={eg0['w_max']:.1f} | DR_good std={ci_good['DR']['boot_std']} "
          f"IPS std={ci_good['IPS']['boot_std']} | CAN_DR_EVAL_SASREC={can_dr_eval}")
    print("ALS-π compare:", json.dumps(als_pi))


def _plot(h, cig, cib, GT, als_pi):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        names = ["IPS", "SNIPS", "DM_good_qhat", "DR_good_qhat", "DM_bad_qhat", "DR_bad_qhat"]
        ci = {"IPS": cig["IPS"], "SNIPS": cig["SNIPS"], "DM_good_qhat": cig["DM"],
              "DR_good_qhat": cig["DR"], "DM_bad_qhat": cib["DM"], "DR_bad_qhat": cib["DR"]}
        vals = [h[n]["est"] for n in names]
        los = [h[n]["est"] - ci[n]["ci95"][0] for n in names]; his = [ci[n]["ci95"][1] - h[n]["est"] for n in names]
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        cols = ["#4477aa", "#4477aa", "#ccaa44", "#228833", "#cc6677", "#aa3377"]
        ax[0].bar(range(len(names)), vals, yerr=[los, his], capsize=4, color=cols)
        ax[0].axhline(GT, color="k", ls="--", label=f"GT V(π_SASRec)={GT:.3f}")
        ax[0].set_xticks(range(len(names))); ax[0].set_xticklabels(names, rotation=30, ha="right")
        ax[0].set_ylabel("estimated V(π_SASRec)"); ax[0].set_title("SASRec-π OPE vs ground truth (95% boot CI)"); ax[0].legend()
        if als_pi:
            est = ["IPS", "SNIPS", "DR_good", "DR_bad"]
            als_b = [abs(als_pi["IPS_bias"]), abs(als_pi["SNIPS_bias"]), abs(als_pi["DR_good_bias"]), abs(als_pi["DR_bad_bias"])]
            sas_b = [abs(h["IPS"]["bias"]), abs(h["SNIPS"]["bias"]), abs(h["DR_good_qhat"]["bias"]), abs(h["DR_bad_qhat"]["bias"])]
            x = np.arange(len(est)); w = 0.38
            ax[1].bar(x - w/2, als_b, w, label="ALS-π |bias|", color="#999")
            ax[1].bar(x + w/2, sas_b, w, label="SASRec-π |bias|", color="#228833")
            ax[1].set_xticks(x); ax[1].set_xticklabels(est); ax[1].set_ylabel("|bias vs own GT|")
            ax[1].set_title("Estimator bias: ALS-π vs SASRec-π"); ax[1].legend()
        plt.tight_layout(); plt.savefig(os.path.join(PLOTS, "ope_sasrec_pi.png"), dpi=110); plt.close()
    except Exception as ex:
        print("plot skipped:", ex)


if __name__ == "__main__":
    main()
