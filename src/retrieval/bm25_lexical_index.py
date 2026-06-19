"""G31 (V2) — BM25 lexical retrieval lane (PulseDiscover).

A BM25 inverted index over item text (title + genre shelves) used as a LEXICAL candidate
source, the lexical complement to the G27 dense semantic lane. PulseDiscover has no free-text
queries — it is recommendation — so retrieval is QUERY-BY-DOCUMENT: the user's last-read item's
text is the query ("more-like-this" / related-item search, a real IR pattern). This is the
classic lexical half of a modern two-stage search stack (lexical + dense -> RRF -> LTR rerank).

Honesty: seed-item / query-by-document, NOT free-text query search.
"""
from __future__ import annotations
import re
import numpy as np
from rank_bm25 import BM25Okapi

_TOK = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOK.findall((text or "").lower())


class BM25LexicalIndex:
    """BM25Okapi over item texts; query-by-document (seed item text) retrieval."""

    def __init__(self):
        self.ids: list[str] = []
        self.row: dict[str, int] = {}
        self.bm25: BM25Okapi | None = None
        self.corpus_tokens: list[list[str]] = []

    def build(self, ids, texts):
        self.ids = list(ids)
        self.row = {b: i for i, b in enumerate(self.ids)}
        self.corpus_tokens = [tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        return self

    def search_by_text(self, query_text: str, k: int, exclude: set | None = None):
        """Return (item_ids, scores) for the top-k lexical matches of query_text."""
        q = tokenize(query_text)
        if not q:
            return [], np.array([])
        scores = self.bm25.get_scores(q)
        exclude = exclude or set()
        order = np.argsort(-scores)
        out_ids, out_sc = [], []
        for j in order:
            b = self.ids[j]
            if b in exclude or scores[j] <= 0:
                continue
            out_ids.append(b); out_sc.append(float(scores[j]))
            if len(out_ids) >= k:
                break
        return out_ids, np.array(out_sc)

    def search_by_seed_item(self, seed_item_id: str, seed_text_lookup, k: int, exclude=None):
        """Query-by-document: use the seed item's own text as the query."""
        if seed_item_id not in seed_text_lookup:
            return [], np.array([])
        ex = set(exclude or set()); ex.add(seed_item_id)
        return self.search_by_text(seed_text_lookup[seed_item_id], k, ex)


def reciprocal_rank_fusion(rank_lists, k=20, c=60):
    """RRF over multiple ranked id-lists. rank_lists: list[list[item_id]] (best-first)."""
    score = {}
    for lst in rank_lists:
        for r, item in enumerate(lst):
            score[item] = score.get(item, 0.0) + 1.0 / (c + r + 1)
    return [it for it, _ in sorted(score.items(), key=lambda kv: -kv[1])][:k]
