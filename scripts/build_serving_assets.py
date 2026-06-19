"""Precompute a small serving-assets bundle so the deploy image doesn't need the 114MB train.csv.

STANDARD-LIBRARY ONLY (no pandas/numpy) so it runs on any Python 3 without extra installs —
this runs locally during deploy, not inside the container.

Writes data/interim/c2_serving_assets.pkl with:
  pop_top : top-500 popular book_ids (popularity fallback)
  head    : set of head book_ids (>= p90 popularity) for the rerank lever
  pop_pct : book_id -> popularity percentile (~1.0 = most popular)
  b2c     : book_id -> creator_id (optional metadata; {} if content meta missing)
"""
from __future__ import annotations
import os, csv, pickle
from collections import Counter

DATA = os.environ.get("PD_DATADIR", "data/interim")

# 1. count item popularity from train.csv (book_id column), stdlib csv
counts: Counter = Counter()
with open(os.path.join(DATA, "domain_splits", "train.csv"), newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        counts[str(row["book_id"])] += 1

ranked = counts.most_common()              # [(book_id, count), ...] descending
n = len(ranked)
pop_top = [b for b, _ in ranked[:500]]

# p90 of the count distribution (head = top decile by popularity)
sorted_counts = sorted(counts.values())
p90 = sorted_counts[min(int(0.90 * n), n - 1)] if n else 0
head = {b for b, c in counts.items() if c >= p90}

# popularity percentile: ~1.0 = most popular (matches the serving loader)
pop_pct = {b: (n - i) / n for i, (b, _) in enumerate(ranked)}

# 2. optional creator metadata (book_id -> creator_id), stdlib csv; tolerate missing
b2c: dict = {}
meta_path = os.path.join(DATA, "domain_content_meta.csv")
if os.path.exists(meta_path):
    with open(meta_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            b2c[str(row.get("book_id", ""))] = row.get("creator_id", "") or None

assets = {"pop_top": pop_top, "head": head, "pop_pct": pop_pct, "b2c": b2c}
out = os.path.join(DATA, "c2_serving_assets.pkl")
with open(out, "wb") as f:
    pickle.dump(assets, f)
print(f"wrote {out}: pop_top={len(pop_top)} head={len(head)} pop_pct={len(pop_pct)} "
      f"b2c={len(b2c)} size={os.path.getsize(out)/1e6:.1f}MB")
