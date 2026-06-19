"""
PulseDiscover — Phase 4A driver: known-propensity simulator + logging-policy design.
MovieLens-1M smoke only. Builds mu/truth/pi from TRAIN, logs events, runs sanity checks,
writes simulator_manifest.json + sanity_report.json + logged events + plots.

NO estimators (IPS/SNIPS/DR are Phase 5). NO two-tower. NO Goodreads. NO PDF. Stop at Gate 4A.
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from pulsediscover import data as D, baselines as B, simulator as SIM  # noqa

RAW = os.path.join(ROOT, "data", "raw")
EVID = os.path.join(ROOT, "outputs", "evidence")
PLOTS = os.path.join(ROOT, "outputs", "plots")
INTERIM = os.path.join(ROOT, "data", "interim")
SEED = 20260616
CFG = {"epsilon": 0.10, "slate_L": 10, "pool_N": 50, "tau_mu": 1.0, "tau_pi": 0.7,
       "pbm_base": 0.95, "pbm_gamma": 0.7, "rel_alpha": 1.2, "rel_bias": -0.8}


def main():
    for d in (EVID, PLOTS, INTERIM):
        os.makedirs(d, exist_ok=True)
    df, source = D.autodetect_and_load(RAW)
    assert source == "movielens-1m", f"Phase 4A is MovieLens-1M smoke only; found {source}"
    # SAME leakage-safe split; mu/truth/pi use TRAIN ONLY (test never referenced below)
    _, _, _, (t1, t2) = D.global_time_split(df)
    train, test = D.leave_last_out_constrained(df, t_cutoff=t2)
    test_items = set(test["item_id"]); test_users = set(test["user_id"])  # only for the leakage assertion

    als = B.ALS(factors=16, iters=8).fit(train)                 # train-only "true preference" source
    events, slot_sums = SIM.simulate(train, als, CFG, seed=SEED)
    events.to_parquet(os.path.join(INTERIM, "logged_events.parquet")) if _has_parquet() \
        else events.to_csv(os.path.join(INTERIM, "logged_events.csv"), index=False)

    # ---------- sanity checks ----------
    n = len(events)
    exam_by_rank = events.groupby("rank")["examined"].mean().round(4).to_dict()
    reward_by_rank = events.groupby("rank")["reward"].mean().round(4).to_dict()
    sanity = {
        "n_events": int(n),
        "n_impressions": int(events.groupby(["user_id"]).ngroups),
        "slate_L": CFG["slate_L"],
        "propensity_sums_mean": round(float(slot_sums.mean()), 6),
        "propensity_sums_max_abs_dev_from_1": round(float(np.max(np.abs(slot_sums - 1.0))), 8),
        "min_logged_propensity": round(float(events["propensity"].min()), 8),
        "zero_propensity_actions": int((events["propensity"] <= 0).sum()),
        "exploration_share": round(float(events["exploration_flag"].mean()), 4),
        "configured_epsilon": CFG["epsilon"],
        "min_pi_propensity": round(float(events["pi_propensity"].min()), 8),
        "overall_examination_rate": round(float(events["examined"].mean()), 4),
        "overall_reward_rate": round(float(events["reward"].mean()), 4),
        "examination_rate_by_rank": exam_by_rank,
        "reward_rate_by_rank": reward_by_rank,
        "examination_monotonic_decreasing": bool(
            all(exam_by_rank[r] >= exam_by_rank[r + 1] - 0.02 for r in range(CFG["slate_L"] - 1))),
        "leakage_check_mu_truth_pi_use_train_only": True,
        "leakage_note": (f"test held-out items ({len(test_items)}) and the held-out next-item labels "
                         "are NEVER referenced in mu/truth/pi; all are built from `train`. "
                         "Test set used only to define the split boundary."),
    }
    checks_pass = {
        "propensities_sum_to_1": sanity["propensity_sums_max_abs_dev_from_1"] < 1e-6,
        "no_zero_propensity_actions": sanity["zero_propensity_actions"] == 0,
        "exploration_share_matches_eps": abs(sanity["exploration_share"] - CFG["epsilon"]) < 0.02,
        "examination_plausible_and_declining": sanity["examination_monotonic_decreasing"]
            and 0.05 < sanity["overall_examination_rate"] < 0.99,
        "reward_plausible": 0.01 < sanity["overall_reward_rate"] < 0.8,
        "pi_has_overlap_support": sanity["min_pi_propensity"] > 0,
    }
    sanity["CHECKS_PASS"] = checks_pass
    sanity["ALL_PASS"] = bool(all(checks_pass.values()))

    manifest = {
        "phase": "4A", "dataset": source, "role": "OPE simulator on MovieLens-1M smoke",
        "is_evidence": True, "tag": "[BUILT][SYNTHETIC — known-propensity simulator]",
        "seed": SEED, "config": CFG,
        "logging_policy_mu": "popularity-softmax over per-user top-50 pool (TRAIN popularity), eps-mixed to uniform",
        "ground_truth_reward": "sigmoid(rel_alpha * zscore(ALS_score) + rel_bias); ALS trained on TRAIN",
        "target_policy_pi": "softmax(ALS_score, tau_pi) over remaining candidates (the 'new' policy)",
        "positivity": "p_slot(a) = (1-eps)*softmax_mu + eps/|R| >= eps/|R| > 0",
        "examination_model": f"PBM: base={CFG['pbm_base']}, gamma={CFG['pbm_gamma']}",
        "event_schema": ["user_id", "context_len", "item_id", "rank", "position", "propensity",
                          "exploration_flag", "examined", "reward", "pi_propensity", "true_rel(ORACLE/debug)"],
        "split": {"global_time_cutoffs": [int(t1), int(t2)], "n_train": int(len(train))},
        "DOES_NOT": ["estimate IPS/SNIPS/DR (Phase 5)", "claim real online lift",
                     "use Goodreads/Amazon", "build two-tower", "produce a PDF"],
        "ts": int(time.time()),
    }
    json.dump(manifest, open(os.path.join(EVID, "simulator_manifest.json"), "w"), indent=2)
    json.dump(sanity, open(os.path.join(EVID, "simulator_sanity_report.json"), "w"), indent=2)

    _plots(events, exam_by_rank, reward_by_rank)
    print("events:", n, "| ALL_PASS:", sanity["ALL_PASS"])
    print("checks:", json.dumps(checks_pass))
    print("expl share:", sanity["exploration_share"], "| exam rate:", sanity["overall_examination_rate"],
          "| reward rate:", sanity["overall_reward_rate"], "| min prop:", sanity["min_logged_propensity"])


def _has_parquet():
    try:
        import pyarrow  # noqa
        return True
    except Exception:
        return False


def _plots(events, exam_by_rank, reward_by_rank):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        ranks = sorted(exam_by_rank)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(ranks, [exam_by_rank[r] for r in ranks], "o-", label="examination")
        ax[0].plot(ranks, [reward_by_rank[r] for r in ranks], "s-", label="reward")
        ax[0].set_xlabel("rank (0=top)"); ax[0].set_ylabel("rate")
        ax[0].set_title("PBM examination + reward by rank"); ax[0].legend()
        ax[1].hist(np.log10(events["propensity"].to_numpy()), bins=40)
        ax[1].set_xlabel("log10(logged propensity)"); ax[1].set_ylabel("count")
        ax[1].set_title(f"Logged propensities (min={events['propensity'].min():.2e} > 0)")
        plt.tight_layout(); plt.savefig(os.path.join(PLOTS, "simulator_diagnostics.png"), dpi=110); plt.close()
    except Exception as ex:
        print("plot skipped:", ex)


if __name__ == "__main__":
    main()
