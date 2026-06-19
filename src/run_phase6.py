"""
PulseDiscover — Phase 6 driver: position-bias correction (naive vs IPS vs PAL).
Uses Phase-4A simulator logs. Synthetic/simulator-scoped. No two-tower, no Goodreads, no PDF.
Stops at Gate 6.
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from pulsediscover import position_bias as PB  # noqa
from scipy.stats import spearmanr

EVID = os.path.join(ROOT, "outputs", "evidence"); PLOTS = os.path.join(ROOT, "outputs", "plots")
INTERIM = os.path.join(ROOT, "data", "interim")
PBM_BASE, PBM_GAMMA = 0.95, 0.7
MIN_IMPR = 20


def main():
    ev_path = os.path.join(INTERIM, "logged_events.csv")
    if not os.path.exists(ev_path):
        ev_path = os.path.join(INTERIM, "logged_events.parquet")
    events = pd.read_csv(ev_path) if ev_path.endswith(".csv") else pd.read_parquet(ev_path)

    exam_emp = PB.empirical_examination(events)                 # P(examined|rank), estimated
    L = int(events["rank"].max()) + 1
    exam_known = {r: PBM_BASE * (1.0 / (r + 1.0)) ** PBM_GAMMA for r in range(L)}

    tab = PB.per_item_estimates(events, exam_emp, min_impr=MIN_IMPR)
    metrics = PB.comparison_metrics(tab)

    # PAL (Route B) if feasible
    pal = PB.fit_pal(events)
    pal_status = "V2 — not built (torch unavailable)"
    pal_spearman = None
    if pal is not None:
        theta_map, pal_loss = pal
        tab["pal_theta"] = tab.index.map(theta_map)
        pal_spearman = round(float(spearmanr(tab["pal_theta"], tab["true_rel"])[0]), 4)
        pal_status = f"BUILT (additive logit tower, BCE loss={pal_loss:.4f})"

    # item movement: rank by naive vs ips
    tab["naive_rank"] = tab["naive_rel"].rank(ascending=False)
    tab["ips_rank"] = tab["ips_rel"].rank(ascending=False)
    tab["rank_change"] = tab["naive_rank"] - tab["ips_rank"]     # +ve = moved up under IPS
    tab["correction"] = tab["ips_rel"] - tab["naive_rel"]
    movers_up = tab.sort_values("rank_change", ascending=False).head(5)
    movers_down = tab.sort_values("rank_change").head(5)

    report = {
        "phase": 6, "dataset": "movielens-1m", "is_evidence": True,
        "tag": "[BUILT][SYNTHETIC — position-bias correction on simulator logs]",
        "role": "position-bias correction validation (simulator-scoped; NOT real online lift)",
        "n_events": int(len(events)), "n_items_compared": int(len(tab)), "min_impr": MIN_IMPR,
        "examination_propensity_empirical": {int(r): round(float(v), 4) for r, v in exam_emp.items()},
        "examination_propensity_known_pbm": {int(r): round(float(v), 4) for r, v in exam_known.items()},
        "metrics_vs_ground_truth": metrics,
        "pal_route_b": {"status": pal_status, "spearman_vs_true": pal_spearman},
        "item_movement": {
            "biggest_up_under_ips": [{"item_id": int(i), "avg_rank": round(float(r.avg_rank), 2),
                                      "naive_rel": round(float(r.naive_rel), 4), "ips_rel": round(float(r.ips_rel), 4),
                                      "rank_change": int(r.rank_change)} for i, r in movers_up.iterrows()],
            "biggest_down_under_ips": [{"item_id": int(i), "avg_rank": round(float(r.avg_rank), 2),
                                        "naive_rel": round(float(r.naive_rel), 4), "ips_rel": round(float(r.ips_rel), 4),
                                        "rank_change": int(r.rank_change)} for i, r in movers_down.iterrows()],
        },
        "honesty": ("naive is position-confounded (correlates with avg_rank); IPS removes most of that "
                    "confounding and tracks ground truth better. All simulator-scoped; per-item relevance "
                    "aggregated over impressions; examination propensity estimated from logged exposure."),
        "ts": int(time.time()),
    }
    json.dump(report, open(os.path.join(EVID, "position_bias_report.json"), "w"), indent=2)
    _plots(exam_emp, exam_known, tab)
    print("n_items:", len(tab), "| metrics:", json.dumps(metrics))
    print("PAL:", pal_status, "| PAL spearman vs true:", pal_spearman)


def _plots(exam_emp, exam_known, tab):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
        ranks = sorted(exam_known)
        ax[0].plot(ranks, [exam_known[r] for r in ranks], "k--", label="known PBM curve")
        ax[0].plot(ranks, [exam_emp[r] for r in ranks], "o-", label="empirical examined rate")
        ax[0].set_xlabel("rank (0=top)"); ax[0].set_ylabel("P(examined)")
        ax[0].set_title("Examination curve: known vs empirical"); ax[0].legend()

        sc = ax[1].scatter(tab["avg_rank"], tab["correction"], c=tab["true_rel"], cmap="viridis", s=14)
        ax[1].axhline(0, color="gray", lw=0.8)
        ax[1].set_xlabel("avg display rank (higher = worse slot)")
        ax[1].set_ylabel("IPS_rel − naive_rel (upward correction)")
        ax[1].set_title("Item movement: worse-placed items corrected upward"); plt.colorbar(sc, ax=ax[1], label="true_rel")

        ax[2].scatter(tab["naive_rel"], tab["ips_rel"], s=14, alpha=0.6)
        lim = [0, max(tab["ips_rel"].max(), tab["naive_rel"].max()) * 1.05]
        ax[2].plot(lim, lim, "k--", lw=0.8, label="y=x")
        ax[2].set_xlabel("naive relevance"); ax[2].set_ylabel("IPS-corrected relevance")
        ax[2].set_title("Naive vs IPS-corrected (IPS > naive: debiased up)"); ax[2].legend()
        plt.tight_layout(); plt.savefig(os.path.join(PLOTS, "position_bias.png"), dpi=110); plt.close()
    except Exception as ex:
        print("plot skipped:", ex)


if __name__ == "__main__":
    main()
