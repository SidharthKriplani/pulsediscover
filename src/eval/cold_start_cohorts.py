"""G24 — Cold-Start + Sparse-Cohort builder (PulseDiscover V2).

Defines user cohorts (by train-history length + ALS-factor presence) and item
tiers (head / mid / long-tail by train popularity) used to measure FALLBACK
QUALITY and degradation — NOT to claim cold-start is solved.

User cohorts
------------
  unknown        : user_id not in train and has no ALS factor
  no_factor      : in train but no ALS user factor (excluded at MF build)
  sparse_1_2     : 1-2 train interactions
  low_3_5        : 3-5 train interactions
  warm_6plus     : 6+ train interactions (the personalizable cohort)

Item tiers (by train interaction count, percentile thresholds)
  head           : >= p90
  mid            : p50..p90
  long_tail      : < p50

This module is pure data prep: it returns dataframes / dicts that the G24
evaluation script consumes. It never trains and never invents metrics.
"""
from __future__ import annotations
import os
import pickle
import numpy as np
import pandas as pd

DATA = os.environ.get("PD_DATADIR", "data/interim")
SPLITS = os.path.join(DATA, "domain_splits")

# Shelf tags that are NOT genres (organisational / status shelves on Goodreads).
_NON_GENRE = {
    "to-read", "owned", "books-i-own", "currently-reading", "read", "kindle",
    "hardcover", "paperback", "ebook", "audiobook", "audible", "library",
    "favorites", "favourites", "wish-list", "wishlist", "default", "books",
    "my-books", "to-buy", "dnf", "re-read", "reread", "netgalley", "arc",
    "kindle-unlimited", "calibre-list", "owned-books", "my-library",
}


def user_history(train: pd.DataFrame) -> pd.Series:
    """Return Series user_id -> train interaction count."""
    return train.groupby("user_id")["book_id"].size()


def user_last_item(train: pd.DataFrame) -> dict:
    """Most-recent (by event_ts if present, else last row) train item per user."""
    if "event_ts" in train.columns:
        idx = train.sort_values("event_ts").groupby("user_id").tail(1)
    else:
        idx = train.groupby("user_id").tail(1)
    return dict(zip(idx["user_id"].values, idx["book_id"].values))


def item_popularity(train: pd.DataFrame) -> pd.Series:
    """Return Series book_id -> train interaction count (descending)."""
    return train["book_id"].value_counts()


def item_tiers(pop: pd.Series) -> tuple[dict, dict]:
    """Classify items into head / mid / long_tail by popularity percentile.

    Returns (item_tier: book_id->str, thresholds: dict).
    """
    p50 = float(pop.quantile(0.50))
    p90 = float(pop.quantile(0.90))
    tier = {}
    for bid, c in pop.items():
        if c >= p90:
            tier[bid] = "head"
        elif c >= p50:
            tier[bid] = "mid"
        else:
            tier[bid] = "long_tail"
    return tier, {"p50": p50, "p90": p90}


def assign_user_cohort(hist_count: int, has_factor: bool) -> str:
    if hist_count == 0:
        return "unknown"
    if not has_factor:
        return "no_factor"
    if hist_count <= 2:
        return "sparse_1_2"
    if hist_count <= 5:
        return "low_3_5"
    return "warm_6plus"


def build_genre_map(content_csv: str | None = None, limit_rows: int | None = None) -> dict:
    """book_id -> primary genre shelf (first non-status shelf token). Best-effort."""
    if content_csv is None:
        content_csv = os.path.join(DATA, "domain_content_meta.csv")
    if not os.path.exists(content_csv):
        return {}
    cm = pd.read_csv(content_csv, usecols=["book_id", "shelves"], nrows=limit_rows)
    gmap = {}
    for bid, sh in zip(cm["book_id"].values, cm["shelves"].values):
        if not isinstance(sh, str):
            continue
        for tok in sh.split():
            if tok not in _NON_GENRE:
                gmap[str(bid)] = tok  # book_ids are strings in the ALS index
                break
    return gmap


def load_als(path: str | None = None) -> dict:
    if path is None:
        path = os.path.join(DATA, "c2_als.pkl")
    return pickle.load(open(path, "rb"))


def load_eval_samples() -> pd.DataFrame:
    """Concatenate warm + cold eval samples with a `source` tag.

    Each row: user_id, gold (held-out target book_id), source in {warm,cold}.
    """
    frames = []
    for name, src in [("warm_eval_sample.csv", "warm"), ("cold_eval_sample.csv", "cold")]:
        p = os.path.join(SPLITS, name)
        if os.path.exists(p):
            df = pd.read_csv(p)
            df["source"] = src
            frames.append(df)
    if not frames:
        # fall back to the generic eval sample
        df = pd.read_csv(os.path.join(SPLITS, "eval_sample.csv"))
        df["source"] = "mixed"
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_cohorts(train: pd.DataFrame, als: dict, evald: pd.DataFrame):
    """Attach cohort labels to each eval user. Returns enriched eval df + context."""
    hist = user_history(train)
    pop = item_popularity(train)
    tier, thr = item_tiers(pop)
    uidx = als["uidx"]

    rows = []
    for _, r in evald.iterrows():
        uid = r["user_id"]
        h = int(hist.get(uid, 0))
        has_factor = uid in uidx
        cohort = assign_user_cohort(h, has_factor)
        # book_ids are strings in the ALS index — keep gold as str consistently
        gold = str(int(r["gold"])) if not pd.isna(r["gold"]) else "-1"
        rows.append({
            "user_id": uid,
            "gold": gold,
            "source": r.get("source", "mixed"),
            "hist_count": h,
            "has_factor": has_factor,
            "cohort": cohort,
            "gold_tier": tier.get(gold, "unseen"),
        })
    out = pd.DataFrame(rows)
    ctx = {"hist": hist, "pop": pop, "item_tier": tier, "tier_thresholds": thr}
    return out, ctx
