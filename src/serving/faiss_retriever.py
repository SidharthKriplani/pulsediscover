"""V2/G22 — FAISS ANN retriever for ALS candidate serving (reusable serving module).
Wraps ALS item factors in a FAISS index (exact FlatIP / approximate IVF / HNSW) with a uniform
search API. FAISS RETAINS exact recall at best (exact) or trades a measured amount for latency
(approx) — it is NOT a quality-improvement method. Inner-product metric (matches ALS scoring)."""
from __future__ import annotations
import time, numpy as np, faiss


class ExactRetriever:
    """Brute-force exact inner-product top-K over item factors (numpy reference)."""
    def __init__(self, item_factors: np.ndarray):
        self.Y = np.ascontiguousarray(item_factors, dtype=np.float32)
        self.build_sec = 0.0; self.kind = "exact_numpy"
    def search(self, q: np.ndarray, k: int) -> np.ndarray:
        s = self.Y @ np.asarray(q, dtype=np.float32); idx = np.argpartition(-s, k)[:k]; return idx[np.argsort(-s[idx])]
    def search_batch(self, Q: np.ndarray, k: int) -> np.ndarray:
        S = np.asarray(Q, dtype=np.float32) @ self.Y.T; out = np.argpartition(-S, k, axis=1)[:, :k]
        return np.take_along_axis(out, np.argsort(-np.take_along_axis(S, out, 1), axis=1), 1)


class FaissRetriever:
    """FAISS index over item factors. kind in {flatip (exact), ivf (approx), hnsw (approx)}."""
    def __init__(self, item_factors: np.ndarray, kind: str = "flatip",
                 nlist: int = 256, nprobe: int = 8, M: int = 32, efC: int = 200, efS: int = 64):
        self.Y = np.ascontiguousarray(item_factors, dtype=np.float32); self.d = self.Y.shape[1]
        self.kind = kind; self.params = {}
        t = time.perf_counter()
        if kind == "flatip":
            self.index = faiss.IndexFlatIP(self.d); self.index.add(self.Y)
        elif kind == "ivf":
            quant = faiss.IndexFlatIP(self.d)
            self.index = faiss.IndexIVFFlat(quant, self.d, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(self.Y); self.index.add(self.Y); self.index.nprobe = nprobe
            self.params = {"nlist": nlist, "nprobe": nprobe}
        elif kind == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.d, M, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = efC; self.index.add(self.Y); self.index.hnsw.efSearch = efS
            self.params = {"M": M, "efConstruction": efC, "efSearch": efS}
        else:
            raise ValueError(kind)
        self.build_sec = time.perf_counter() - t
    def set_nprobe(self, n): self.index.nprobe = n; self.params["nprobe"] = n
    def set_efsearch(self, e): self.index.hnsw.efSearch = e; self.params["efSearch"] = e
    def search(self, q: np.ndarray, k: int) -> np.ndarray:
        q = np.ascontiguousarray(np.asarray(q, dtype=np.float32).reshape(1, -1)); _, I = self.index.search(q, k); return I[0]
    def search_batch(self, Q: np.ndarray, k: int) -> np.ndarray:
        _, I = self.index.search(np.ascontiguousarray(Q, dtype=np.float32), k); return I
    def index_bytes(self) -> int:
        import tempfile, os
        f = tempfile.NamedTemporaryFile(suffix=".idx", delete=False).name
        faiss.write_index(self.index, f); n = os.path.getsize(f); os.remove(f); return n
