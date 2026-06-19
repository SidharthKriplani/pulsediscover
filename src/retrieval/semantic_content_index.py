"""G27 (V2) — Semantic content retrieval lane (PulseDiscover).

Dense item-content embeddings + a FAISS content index used as a CANDIDATE-GENERATION
source for item-cold-start / sparse-catalog scenarios. This is candidate generation,
NOT chat, NOT RAG, NOT an LLM reranker, NOT a "semantic taste" model. Items are embedded
from their own metadata text (title + shelves/genres + short description); retrieval is
item-to-item / text-to-item nearest-neighbour by inner product on normalized vectors.

Embedding model: sentence-transformers MiniLM (deterministic, local, CPU). Falls back to
a documented TF-IDF+SVD ("LSA") dense embedding only if the transformer cannot be built.
"""
from __future__ import annotations
import os, json, time, hashlib
import numpy as np
import pandas as pd

NON_GENRE = {
    "to-read", "owned", "books-i-own", "currently-reading", "read", "kindle",
    "hardcover", "paperback", "ebook", "audiobook", "audible", "library",
    "favorites", "favourites", "wish-list", "wishlist", "default", "books",
    "my-books", "to-buy", "dnf", "re-read", "reread", "netgalley", "arc",
    "kindle-unlimited", "calibre-list", "owned-books", "my-library",
}
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_item_text(meta_row_title: str, shelves: str, desc: str, max_desc: int = 200) -> str:
    """Canonical item text = title + cleaned genre shelves + short description."""
    title = meta_row_title or ""
    shelf_tokens = [t for t in (shelves or "").split() if t not in NON_GENRE][:8]
    genres = " ".join(shelf_tokens)
    d = (desc or "")[:max_desc]
    return f"{title}. {genres}. {d}".strip()


def load_item_texts(content_csv: str, universe_ids: set | None = None, max_desc: int = 200) -> pd.DataFrame:
    """Return DataFrame[book_id, text, has_text] for items (optionally restricted to a universe).

    max_desc=0 drops the description (title+genre only) — ~3x faster to embed and the
    dominant cold-start signal (a new vampire-romance is near other vampire-romances by
    title+shelves); set >0 to include a description slice for richer semantics.
    """
    cols = ["book_id", "title", "shelves", "desc"]
    cm = pd.read_csv(content_csv, dtype=str, usecols=cols).fillna("")
    cm["book_id"] = cm["book_id"].astype(str)
    if universe_ids is not None:
        cm = cm[cm["book_id"].isin(universe_ids)]
    cm = cm.drop_duplicates("book_id", keep="first").reset_index(drop=True)
    cm["text"] = [build_item_text(t, s, d, max_desc) for t, s, d in zip(cm["title"], cm["shelves"], cm["desc"])]
    cm["has_text"] = cm["text"].str.len() > 3
    return cm[["book_id", "text", "has_text"]]


class SemanticContentIndex:
    """Dense content embeddings + FAISS FlatIP index over normalized vectors."""

    def __init__(self, dim: int | None = None):
        self.ids: list[str] = []
        self.emb: np.ndarray | None = None
        self.index = None
        self.dim = dim
        self.model_name = None
        self.index_version = None

    # ---- embedding ----
    @staticmethod
    def _embed_transformer(texts, batch_size=256, seed=42):
        import torch
        torch.manual_seed(seed); torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(EMB_MODEL)
        v = m.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(v, dtype=np.float32), EMB_MODEL, m.get_sentence_embedding_dimension()

    @staticmethod
    def _embed_tfidf_svd(texts, dim=256, seed=42):
        """Deterministic local fallback: TF-IDF -> TruncatedSVD (LSA) -> L2-normalize."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import normalize
        tf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2)
        X = tf.fit_transform(texts)
        svd = TruncatedSVD(n_components=min(dim, X.shape[1] - 1), random_state=seed)
        Z = svd.fit_transform(X)
        Z = normalize(Z.astype(np.float32))
        return Z, f"tfidf_svd_{Z.shape[1]}d", Z.shape[1]

    def build(self, ids, texts, method="transformer", **kw):
        if method == "transformer":
            self.emb, self.model_name, self.dim = self._embed_transformer(texts, **kw)
        else:
            self.emb, self.model_name, self.dim = self._embed_tfidf_svd(texts, **kw)
        self.ids = list(ids)
        self._build_faiss()
        return self

    def _build_faiss(self):
        import faiss
        self.emb = np.ascontiguousarray(self.emb, dtype=np.float32)
        self.index = faiss.IndexFlatIP(self.emb.shape[1])
        self.index.add(self.emb)
        self.index_version = "faiss_flatip_content_" + hashlib.md5(self.emb.tobytes()).hexdigest()[:8]

    # ---- persistence ----
    def save(self, path_prefix):
        np.save(path_prefix + "_emb.npy", self.emb)
        json.dump({"ids": self.ids, "model": self.model_name, "dim": self.dim,
                   "index_version": self.index_version}, open(path_prefix + "_meta.json", "w"))

    @classmethod
    def load(cls, path_prefix):
        meta = json.load(open(path_prefix + "_meta.json"))
        obj = cls(dim=meta["dim"]); obj.ids = meta["ids"]; obj.model_name = meta["model"]
        obj.emb = np.load(path_prefix + "_emb.npy").astype(np.float32)
        obj._build_faiss(); return obj

    # ---- retrieval ----
    def search(self, query_vec: np.ndarray, k: int) -> tuple[list[str], np.ndarray]:
        q = np.ascontiguousarray(np.asarray(query_vec, dtype=np.float32).reshape(1, -1))
        s, I = self.index.search(q, k)
        return [self.ids[i] for i in I[0] if 0 <= i < len(self.ids)], s[0]

    def id_to_row(self):
        return {b: i for i, b in enumerate(self.ids)}
