"""
PulseDiscover — data layer (Phase 2).

Canonical interaction schema + dataset loaders + leakage-safe splits.

Canonical schema (one row per interaction):
    user_id    : str/int      — the reader
    item_id    : str/int      — the story/book/movie
    creator_id : str/int|None — author (Goodreads) ; None for MovieLens
    series_id  : str/int|None — series/serialization (Goodreads) ; None otherwise
    event_ts   : int          — unix seconds (sortable); enables sequences + temporal split
    rating     : float|None   — explicit rating if present
    event_type : str          — 'rating' | 'shelve' | 'read' | 'review' | 'purchase'

Public datasets carry NO position and NO logging propensity. Those are NOT in
this schema — they are manufactured only by the Phase-4 simulator, never claimed
as real here.

No network access in this module: loaders read files already present in data/raw/.
"""
from __future__ import annotations
import os, json, glob
import numpy as np
import pandas as pd

CANON_COLS = ["user_id", "item_id", "creator_id", "series_id", "event_ts", "rating", "event_type"]


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    for c in CANON_COLS:
        if c not in df.columns:
            df[c] = None
    df = df[CANON_COLS].copy()
    df = df.dropna(subset=["user_id", "item_id", "event_ts"])
    df["event_ts"] = df["event_ts"].astype("int64")
    return df.sort_values("event_ts").reset_index(drop=True)


# ---------- loaders (file-based; data must already be in data/raw/) ----------

def load_movielens_1m(path: str) -> pd.DataFrame:
    """ml-1m ratings.dat : UserID::MovieID::Rating::Timestamp"""
    df = pd.read_csv(path, sep="::", engine="python", header=None,
                     names=["user_id", "item_id", "rating", "event_ts"])
    df["event_type"] = "rating"
    return _finalize(df)


def load_goodreads_genre_json(path: str, max_rows: int | None = None) -> pd.DataFrame:
    """
    Goodreads UCSD per-genre interactions, gzipped JSON-lines.
    Fields vary; we map the common ones. 'date_added'/'read_at' -> event_ts.
    """
    import gzip, datetime as dt
    rows = []
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        for i, line in enumerate(fh):
            if max_rows and i >= max_rows:
                break
            try:
                r = json.loads(line)
            except Exception:
                continue
            ts = r.get("read_at") or r.get("date_updated") or r.get("date_added")
            try:
                event_ts = int(dt.datetime.strptime(ts[:24], "%a %b %d %H:%M:%S %z %Y").timestamp()) if ts else None
            except Exception:
                event_ts = None
            rows.append(dict(user_id=r.get("user_id"), item_id=r.get("book_id"),
                             creator_id=None, series_id=None, event_ts=event_ts,
                             rating=r.get("rating"), event_type="shelve"))
    return _finalize(pd.DataFrame(rows))


def load_amazon_books_jsonl(path: str, max_rows: int | None = None) -> pd.DataFrame:
    """Amazon Reviews 2023 Books review JSONL (fallback)."""
    import gzip
    rows = []
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        for i, line in enumerate(fh):
            if max_rows and i >= max_rows:
                break
            try:
                r = json.loads(line)
            except Exception:
                continue
            ts = r.get("timestamp")
            event_ts = int(ts / 1000) if isinstance(ts, (int, float)) and ts > 1e11 else (int(ts) if ts else None)
            rows.append(dict(user_id=r.get("user_id"), item_id=r.get("parent_asin") or r.get("asin"),
                             creator_id=None, series_id=None, event_ts=event_ts,
                             rating=r.get("rating"), event_type="review"))
    return _finalize(pd.DataFrame(rows))


def autodetect_and_load(raw_dir: str, max_rows: int | None = None):
    """Return (df, source_name) for whatever real dataset is present in data/raw/, else (None, None)."""
    ml = glob.glob(os.path.join(raw_dir, "**", "ratings.dat"), recursive=True)
    if ml:
        return load_movielens_1m(ml[0]), "movielens-1m"
    gr = glob.glob(os.path.join(raw_dir, "**", "*interactions*fantasy*"), recursive=True) or \
         glob.glob(os.path.join(raw_dir, "**", "goodreads_interactions_*.json*"), recursive=True)
    if gr:
        return load_goodreads_genre_json(gr[0], max_rows=max_rows), f"goodreads:{os.path.basename(gr[0])}"
    az = glob.glob(os.path.join(raw_dir, "**", "*Books*.jsonl*"), recursive=True)
    if az:
        return load_amazon_books_jsonl(az[0], max_rows=max_rows), f"amazon-books:{os.path.basename(az[0])}"
    return None, None


# ---------- leakage-safe splits ----------

def global_time_split(df: pd.DataFrame, q_train=0.8, q_val=0.9):
    """Primary split: by global wall-clock time. No test interaction precedes any train one.
    Returns (train, val, test) and the two cutoff timestamps."""
    t1 = int(df["event_ts"].quantile(q_train))
    t2 = int(df["event_ts"].quantile(q_val))
    train = df[df.event_ts < t1]
    val = df[(df.event_ts >= t1) & (df.event_ts < t2)]
    test = df[df.event_ts >= t2]
    return train, val, test, (t1, t2)


def leave_last_out_constrained(df: pd.DataFrame, t_cutoff: int):
    """Secondary split for sequential next-item eval: each user's LAST interaction is held out,
    but ONLY if it falls at/after the global cutoff (prevents cross-user future leakage).
    Returns (train, test_holdout)."""
    df = df.sort_values(["user_id", "event_ts"])
    last_idx = df.groupby("user_id").tail(1).index
    is_last = df.index.isin(last_idx)
    eligible = is_last & (df["event_ts"] >= t_cutoff)
    test = df[eligible]
    train = df[~eligible]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def cold_item_flags(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    seen = set(train["item_id"].unique())
    test_items = set(test["item_id"].unique())
    cold = test_items - seen
    return {"n_train_items": len(seen), "n_test_items": len(test_items),
            "n_cold_items": len(cold), "cold_item_fraction": (len(cold) / max(1, len(test_items)))}
