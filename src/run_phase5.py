"""
PulseDiscover — Phase 5 driver: IPS / SNIPS / DM / DR validation on the Phase-4A simulator.
MovieLens-1M smoke only. Estimators use only (mu, pi, reward, qhat); ground truth (true_rel)
computed separately and never given to estimators. Clipping sweep + q-hat stress test +
cluster-bootstrap CIs + ESS. NO two-tower, NO Goodreads, NO PDF. Stop at Gate 5.
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from pulsediscover import data as D, baselines as B, ope as O  # noqa

RAW = os.path.join(ROOT, "data", "raw"); EVID = os.path.join(ROOT, "outputs", "evidence")
PLOTS = os.path.join(ROOT, "outputs", "plots")
SEED = 20260616; B_BOOT = 400
CFG = {"epsilon": 0.10, "slate_L": 10, "pool_N": 50, "tau_mu": 1.0, "tau_pi": 0.7,
       "pbm_base": 0.95, "pbm_gamma": 0.7, "rel_alpha": 1.2, "rel_bias": -0.8,
       "qhat_good_alpha": 1.0, "qhat_good_bias": -0.6}   # good qhat is close-but-off truth
CLIPS = [None, 5, 10, 20, 50, 100]


def main():
    os.makedirs(EVID, exist_ok=True); os.makedirs(PLOTS, exist_ok=True)
    df, source = D.autodetect_and_load(RAW)
    assert source == "movielens-1m", f"Phase 5 is MovieLens-1M smoke only; found {source}"
    _, _, _, (t1, t2) = D.global_time_split(df)
    train, _ = D.leave_last_out_constrained(df, t_cutoff=t2)
    als = B.ALS(factors=16, iters=8).fit(train)
    A, nu, L = O.augmented_log(train, als, CFG, seed=SEED)

    GT = float(np.mean(A["g"]))                       # known ground-truth V(pi) (oracle)
    flat = {k: A[k].ravel() for k in A}

    def point(clip):
        eg = O.estimators(flat["mu"], flat["pi"], flat["r"], flat["qai_g"], flat["qpi_g"], clip)
        eb = O.estimators(flat["mu"], flat["pi"], flat["r"], flat["qai_b"], flat["qpi_b"], clip)
        return eg, eb

    eg0, eb0 = point(None)
    # bias vs ground truth (no clip)
    def bias(v): return round(v - GT, 6)
    headline = {
        "ground_truth_V_pi": round(GT, 6),
        "IPS": {"est": round(eg0["IPS"], 6), "bias": bias(eg0["IPS"])},
        "SNIPS": {"est": round(eg0["SNIPS"], 6), "bias": bias(eg0["SNIPS"])},
        "DM_good_qhat": {"est": round(eg0["DM"], 6), "bias": bias(eg0["DM"])},
        "DR_good_qhat": {"est": round(eg0["DR"], 6), "bias": bias(eg0["DR"])},
        "DM_bad_qhat": {"est": round(eb0["DM"], 6), "bias": bias(eb0["DM"])},
        "DR_bad_qhat": {"est": round(eb0["DR"], 6), "bias": bias(eb0["DR"])},
        "ESS": round(eg0["ESS"], 1), "n_events": int(nu * L), "n_users": int(nu),
        "w_max": round(eg0["w_max"], 3), "w_mean": round(eg0["w_mean"], 4),
    }
    # bootstrap CIs (no clip), good & bad qhat
    ci_good = O.cluster_bootstrap(A, "g", clip=None, B=B_BOOT, seed=SEED)
    ci_bad = O.cluster_bootstrap(A, "b", clip=None, B=B_BOOT, seed=SEED + 1)

    # clipping sweep
    sweep = []
    for c in CLIPS:
        eg, eb = point(c)
        sweep.append({"clip": c, "ESS": round(eg["ESS"], 1), "w_max": round(eg["w_max"], 3),
                      "IPS_bias": bias(eg["IPS"]), "SNIPS_bias": bias(eg["SNIPS"]),
                      "DR_good_bias": bias(eg["DR"]), "DR_bad_bias": bias(eb["DR"])})

    payload = {
        "phase": 5, "dataset": source, "is_evidence": True,
        "tag": "[BUILT][SYNTHETIC — OPE on known-propensity simulator]",
        "role": "OPE estimator validation on MovieLens-1M smoke (NOT real online lift)",
        "seed": SEED, "config": CFG, "support_note":
            "Validity over the logged candidate-pool support (top-50 per user); not full-catalog.",
        "headline_no_clip": headline,
        "bootstrap_ci95_no_clip": {"good_qhat": ci_good, "bad_qhat": ci_bad, "B": B_BOOT,
                                    "method": "user-cluster bootstrap (resample the (nu,L) slate rows)"},
        "clipping_sweep": sweep,
        "qhat_variants": {"good": "sigmoid(1.0*z - 0.6) (mildly misspecified vs truth 1.2*z-0.8)",
                          "bad": "good qhat + 0.4 overconfidence offset (severely misspecified, biased high)"},
        "honesty": ("DR is EXPECTED to reduce variance vs IPS and stay ~unbiased even under the bad qhat "
                    "(double robustness, since propensities are exact); reported as OBSERVED, not asserted."),
        "ts": int(time.time()),
    }
    json.dump(payload, open(os.path.join(EVID, "ope_comparison.json"), "w"), indent=2)
    _plots(headline, ci_good, ci_bad, sweep, GT)

    print(f"GT={GT:.4f} | IPS={eg0['IPS']:.4f}(b{bias(eg0['IPS'])}) SNIPS={eg0['SNIPS']:.4f}"
          f"(b{bias(eg0['SNIPS'])}) DR_good={eg0['DR']:.4f}(b{bias(eg0['DR'])}) "
          f"DM_bad={eb0['DM']:.4f}(b{bias(eb0['DM'])}) DR_bad={eb0['DR']:.4f}(b{bias(eb0['DR'])})")
    print(f"ESS={eg0['ESS']:.0f}/{nu*L} | w_max={eg0['w_max']:.2f} | "
          f"IPS bootstd={ci_good['IPS']['boot_std']} DR_good bootstd={ci_good['DR']['boot_std']}")


def _plots(h, cig, cib, sweep, GT):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        names = ["IPS", "SNIPS", "DM_good_qhat", "DR_good_qhat", "DM_bad_qhat", "DR_bad_qhat"]
        ci = {"IPS": cig["IPS"], "SNIPS": cig["SNIPS"], "DM_good_qhat": cig["DM"],
              "DR_good_qhat": cig["DR"], "DM_bad_qhat": cib["DM"], "DR_bad_qhat": cib["DR"]}
        vals = [h[n]["est"] for n in names]
        los = [h[n]["est"] - ci[n]["ci95"][0] for n in names]
        his = [ci[n]["ci95"][1] - h[n]["est"] for n in names]
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        cols = ["#4477aa", "#4477aa", "#ccaa44", "#228833", "#cc6677", "#aa3377"]
        ax[0].bar(range(len(names)), vals, yerr=[los, his], capsize=4, color=cols)
        ax[0].axhline(GT, color="k", ls="--", label=f"ground truth V(π)={GT:.3f}")
        ax[0].set_xticks(range(len(names))); ax[0].set_xticklabels(names, rotation=30, ha="right")
        ax[0].set_ylabel("estimated V(π)"); ax[0].set_title("OPE estimates vs ground truth (95% boot CI)")
        ax[0].legend()
        cl = [str(s["clip"]) for s in sweep]
        ax[1].plot(cl, [abs(s["IPS_bias"]) for s in sweep], "o-", label="|IPS bias|")
        ax[1].plot(cl, [abs(s["DR_good_bias"]) for s in sweep], "s-", label="|DR good bias|")
        ax[1].plot(cl, [abs(s["DR_bad_bias"]) for s in sweep], "^-", label="|DR bad bias|")
        ax2 = ax[1].twinx(); ax2.plot(cl, [s["ESS"] for s in sweep], "d--", color="gray", label="ESS")
        ax[1].set_xlabel("weight clip"); ax[1].set_ylabel("|bias|"); ax2.set_ylabel("ESS (gray)")
        ax[1].set_title("Clipping sweep: bias vs clip (+ESS)"); ax[1].legend(loc="upper right")
        plt.tight_layout(); plt.savefig(os.path.join(PLOTS, "ope_comparison.png"), dpi=110); plt.close()
    except Exception as ex:
        print("plot skipped:", ex)


if __name__ == "__main__":
    main()
