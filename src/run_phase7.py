"""
PulseDiscover — Phase 7 driver: feedback-loop / catalog-health audit. MovieLens smoke.
ratchet vs ctr_explore over rounds; exposure Gini/entropy/long-tail; before/after; buried-quality.
Genre-level concentration as a category proxy (MovieLens has no real creators -> creator-level
is a Goodreads-domain feature, pending). Simulator-scoped. No Goodreads/two-tower/PDF. Stop at Gate 7.
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from pulsediscover import data as D, feedback_loop as FL  # noqa

RAW = os.path.join(ROOT, "data", "raw"); EVID = os.path.join(ROOT, "outputs", "evidence")
PLOTS = os.path.join(ROOT, "outputs", "plots")
SEED = 20260616; ROUNDS = 10; T = 1_000_000.0


def main():
    os.makedirs(EVID, exist_ok=True); os.makedirs(PLOTS, exist_ok=True)
    df, source = D.autodetect_and_load(RAW)
    assert source == "movielens-1m"
    _, _, _, (t1, t2) = D.global_time_split(df)
    train, _ = D.leave_last_out_constrained(df, t_cutoff=t2)

    g = train.groupby("item_id")
    pop = g.size()
    qual = ((g["rating"].mean() - 1.0) / 4.0).clip(0, 1)          # MovieLens quality = mean rating, normalized
    items = pop.index.to_numpy()
    popularity = pop.to_numpy().astype(float)
    quality = qual.reindex(pop.index).to_numpy()
    # long-tail = bottom 80% of items by initial popularity
    pop_rank = pd.Series(popularity, index=items).rank(pct=True)
    longtail_mask = (pop_rank <= 0.80).to_numpy()

    mA, hA, CA, EA = FL.run_loop(quality, popularity, longtail_mask, ROUNDS, T, eps=0.02, mode="ratchet")
    mB, hB, CB, EB = FL.run_loop(quality, popularity, longtail_mask, ROUNDS, T, eps=0.10, mode="ctr_explore")

    # buried-quality: high-quality (top 25%) AND low-popularity (bottom 50%) items
    q_hi = quality >= np.quantile(quality, 0.75)
    p_lo = popularity <= np.median(popularity)
    buried = q_hi & p_lo
    final_share = lambda E: float(E[buried].sum() / E.sum()) if E.sum() > 0 else 0.0
    base_pop_share = float(popularity[buried].sum() / popularity.sum())

    # genre-level (category proxy) concentration of FINAL exposure
    genre_gini = {}
    mv = os.path.join(RAW, "ml-1m", "movies.dat")
    if os.path.exists(mv):
        gmap = {}
        for line in open(mv, encoding="latin-1"):
            parts = line.rstrip("\n").split("::")
            if len(parts) == 3:
                gmap[int(parts[0])] = parts[2].split("|")[0]   # primary genre
        idx = {it: i for i, it in enumerate(items)}
        def genre_expo(E):
            agg = {}
            for it, i in idx.items():
                gg = gmap.get(int(it), "NA"); agg[gg] = agg.get(gg, 0.0) + E[i]
            return np.array(list(agg.values()))
        genre_gini = {"ratchet_final": round(FL.gini(genre_expo(EA)), 4),
                      "ctr_explore_final": round(FL.gini(genre_expo(EB)), 4)}

    report = {
        "phase": 7, "dataset": source, "is_evidence": True,
        "tag": "[BUILT][SYNTHETIC — catalog-health audit on simulator]",
        "role": "feedback-loop / catalog-health audit (simulator-scoped; NOT real online lift)",
        "config": {"rounds": ROUNDS, "exposures_per_round": T, "eps_ratchet": 0.02, "eps_ctr_explore": 0.10},
        "quality_def": "TRAIN mean rating normalized to [0,1]", "popularity_def": "TRAIN interaction count",
        "n_items": int(len(items)),
        "item_exposure_metrics_per_round": {"ratchet": mA, "ctr_explore": mB},
        "summary": {
            "gini_round0": mA[0]["gini"],
            "gini_final_ratchet": mA[-1]["gini"], "gini_final_ctr_explore": mB[-1]["gini"],
            "entropy_final_ratchet": mA[-1]["entropy"], "entropy_final_ctr_explore": mB[-1]["entropy"],
            "long_tail_share_round0": mA[0]["long_tail_share"],
            "long_tail_share_final_ratchet": mA[-1]["long_tail_share"],
            "long_tail_share_final_ctr_explore": mB[-1]["long_tail_share"],
        },
        "buried_quality_exposure_share": {
            "n_buried_items": int(buried.sum()),
            "baseline_popularity_share": round(base_pop_share, 4),
            "final_ratchet": round(final_share(EA), 4),
            "final_ctr_explore": round(final_share(EB), 4),
        },
        "genre_level_proxy_gini": genre_gini,
        "creator_level_note": "MovieLens has no creators/authors; creator-level concentration is a "
                              "Goodreads-domain feature (pending). Genre primary-category used as a proxy.",
        "honesty": ("ratchet (rank by click volume) concentrates exposure round over round; ctr_explore "
                    "(exposure-normalized rate + exploration) keeps concentration lower and surfaces "
                    "high-quality low-popularity items. Catalog-level abstraction; simulator-scoped."),
        "ts": int(time.time()),
    }
    json.dump(report, open(os.path.join(EVID, "feedback_loop_report.json"), "w"), indent=2)
    _plots(mA, mB, hA, hB, EA, EB, quality, popularity, buried)
    print("gini r0:", mA[0]["gini"], "| ratchet final:", mA[-1]["gini"], "| ctr_explore final:", mB[-1]["gini"])
    print("long-tail share r0:", mA[0]["long_tail_share"], "ratchet final:", mA[-1]["long_tail_share"],
          "ctr final:", mB[-1]["long_tail_share"])
    print("buried-quality share — baseline:", round(base_pop_share, 4),
          "ratchet:", round(final_share(EA), 4), "ctr_explore:", round(final_share(EB), 4))


def _plots(mA, mB, hA, hB, EA, EB, quality, popularity, buried):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        rounds = [m["round"] for m in mA]
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
        ax[0].plot(rounds, [m["gini"] for m in mA], "o-", color="#cc6677", label="ratchet Gini")
        ax[0].plot(rounds, [m["gini"] for m in mB], "s-", color="#228833", label="ctr+explore Gini")
        ax0b = ax[0].twinx()
        ax0b.plot(rounds, [m["long_tail_share"] for m in mA], "o--", color="#cc6677", alpha=0.5)
        ax0b.plot(rounds, [m["long_tail_share"] for m in mB], "s--", color="#228833", alpha=0.5)
        ax[0].set_xlabel("round"); ax[0].set_ylabel("exposure Gini (solid)")
        ax0b.set_ylabel("long-tail share (dashed)"); ax[0].set_title("Catalog concentration over rounds"); ax[0].legend(loc="center right")

        def lorenz(E):
            x = np.sort(E)[::-1]; c = np.cumsum(x) / x.sum(); return np.arange(1, len(x) + 1) / len(x), c
        fx, fyA = lorenz(hA[0]); _, fyAf = lorenz(hA[-1]); _, fyBf = lorenz(hB[-1])
        ax[1].plot(fx, fyA, "k--", label="round 0")
        ax[1].plot(fx, fyAf, color="#cc6677", label="ratchet final")
        ax[1].plot(fx, fyBf, color="#228833", label="ctr+explore final")
        ax[1].set_xlabel("top fraction of items"); ax[1].set_ylabel("cum. exposure share")
        ax[1].set_title("Exposure concentration (before/after)"); ax[1].legend()

        ax[2].scatter(np.log10(popularity + 1)[~buried], EA[~buried] / EA.sum(), s=8, alpha=0.3, color="gray", label="other")
        ax[2].scatter(np.log10(popularity + 1)[buried], EA[buried] / EA.sum(), s=14, color="#cc6677", label="buried (ratchet)")
        ax[2].scatter(np.log10(popularity + 1)[buried], EB[buried] / EB.sum(), s=14, color="#228833", label="buried (ctr+explore)")
        ax[2].set_xlabel("log10(initial popularity)"); ax[2].set_ylabel("final exposure share")
        ax[2].set_title("High-quality low-popularity items: buried vs surfaced"); ax[2].legend()
        plt.tight_layout(); plt.savefig(os.path.join(PLOTS, "feedback_loop.png"), dpi=110); plt.close()
    except Exception as ex:
        print("plot skipped:", ex)


if __name__ == "__main__":
    main()
