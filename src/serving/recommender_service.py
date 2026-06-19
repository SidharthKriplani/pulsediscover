"""G23 (V2) — RecommenderService: versioned model/index loader + retrieval + fallback tree +
structured logging + in-memory monitoring. Production-SHAPED, not production. ALS candidate retrieval
served via FAISS (FlatIP exact default; HNSW ef64 scale mode). All failure modes return a valid
response with fallback_used + error_code, never a crash."""
from __future__ import annotations
import os, time, json, pickle, hashlib, logging, threading, numpy as np
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
from faiss_retriever import FaissRetriever  # type: ignore

log = logging.getLogger("pulsediscover.serving")
if not log.handlers:
    h = logging.StreamHandler(); h.setFormatter(logging.Formatter("%(message)s")); log.addHandler(h); log.setLevel(logging.INFO)


class RecommenderService:
    def __init__(self, datadir: str, default_mode: str = "exact", build_hnsw: bool = True,
                 enable_rerank: bool = False, rerank_policy: str = "head_cap",
                 rerank_cap_frac: float = 0.3, rerank_pool_mult: int = 4):
        self.datadir = datadir; self.default_mode = default_mode; self._lock = threading.Lock()
        self._metrics = defaultdict(int); self._latencies = []; self._by_mode = defaultdict(int)
        self.ready = False; self.load_error = None
        # G26B: config-flagged heuristic rerank stage (DEFAULT OFF). Interpretable exposure
        # control (head-cap / tail-boost) — NOT learned ranking, NOT LTR.
        self.enable_rerank = enable_rerank; self.rerank_policy = rerank_policy
        self.rerank_cap_frac = rerank_cap_frac; self.rerank_pool_mult = max(2, rerank_pool_mult)
        # Defaults that must exist even if the model fails to load, so the fallback path always
        # has a popularity list to serve (otherwise index_load_failure returns 0 items).
        self.pop_top = []; self.b2c = {}; self._head = set(); self._pop_pct = {}
        import pandas as pd
        # load popularity/assets INDEPENDENTLY of the model so a model-load failure still serves a
        # non-empty fallback. Fast path: a precomputed c2_serving_assets.pkl (small, deploy-friendly);
        # else fall back to deriving from train.csv (114MB). Either way pop_top is populated.
        assets_path = os.path.join(datadir, "c2_serving_assets.pkl")
        try:
            if os.path.exists(assets_path):
                a = pickle.load(open(assets_path, "rb"))
                self.pop_top = list(a.get("pop_top", [])); self._head = set(a.get("head", set()))
                self._pop_pct = dict(a.get("pop_pct", {})); self.b2c = dict(a.get("b2c", {}))
                log.info(json.dumps({"event": "assets_loaded", "source": "c2_serving_assets.pkl",
                                     "pop_top": len(self.pop_top)}))
            else:
                pop = pd.read_csv(os.path.join(datadir, "domain_splits", "train.csv"), usecols=["book_id"], dtype=str).book_id.value_counts()
                self.pop_top = list(pop.index[:500])
                p90 = float(pop.quantile(0.90)); self._head = set(str(b) for b, c in pop.items() if c >= p90)
                _n = len(pop); self._pop_pct = {str(b): (_n - i) / _n for i, b in enumerate(pop.index)}  # ~1.0 = most popular (head)
        except Exception as pe:
            log.info(json.dumps({"event": "popularity_load_failure", "error": str(pe)}))
        try:
            A = pickle.load(open(os.path.join(datadir, "c2_als.pkl"), "rb"))
            self.X = np.ascontiguousarray(A["X"], dtype=np.float32); self.Y = np.ascontiguousarray(A["Y"], dtype=np.float32)
            self.uidx = A["uidx"]; self.i_uniq = list(A["i_uniq"])
            self.n_items, self.dim = self.Y.shape
            # startup validation: user-factor dim must match item-factor dim (NOT tautological — the
            # FAISS index is built from Y, so an X/Y mismatch would otherwise only surface per-request).
            assert self.X.shape[1] == self.Y.shape[1], (
                f"X/Y factor-dim mismatch: X={self.X.shape[1]} Y={self.Y.shape[1]}")
            assert len(self.i_uniq) == self.n_items, (
                f"i_uniq/Y length mismatch: {len(self.i_uniq)} vs {self.n_items}")
            # item metadata (creator) — tolerate missing; skip if already provided by the assets bundle
            if not self.b2c:
                try:
                    cm = pd.read_csv(os.path.join(datadir, "domain_content_meta.csv"), dtype=str).fillna("")
                    self.b2c = dict(zip(cm.book_id, cm.creator_id))
                except Exception:
                    self.b2c = {}
            # indexes (FAISS index dim must equal the item-factor dim it was built from)
            self.exact = FaissRetriever(self.Y, kind="flatip")
            assert self.exact.d == self.dim, "FAISS index/vector dim mismatch"
            self.hnsw = FaissRetriever(self.Y, kind="hnsw", M=32, efC=200, efS=64) if build_hnsw else None
            self.model_id = "als_f64_" + hashlib.md5(self.Y.tobytes()).hexdigest()[:10]
            self.index_id = "faiss_flatip_" + hashlib.md5(self.Y.tobytes()).hexdigest()[:8]
            self.build_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self.ready = True
        except Exception as e:
            self.load_error = str(e); log.info(json.dumps({"event": "load_failure", "error": str(e)}))

    def metadata(self):
        return {"model_id": getattr(self, "model_id", None), "index_id": getattr(self, "index_id", None),
                "build_ts": getattr(self, "build_ts", None), "n_items": getattr(self, "n_items", 0),
                "embedding_dim": getattr(self, "dim", 0), "retrieval_modes": ["exact", "scale"] if getattr(self, "hnsw", None) else ["exact"],
                "default_mode": self.default_mode, "ready": self.ready}

    def _pop_fallback(self, k):
        items = self.pop_top[:k]; return [{"item_id": b, "creator_id": self.b2c.get(b), "source": "popularity"} for b in items]

    # ---- G26B heuristic rerank (interpretable exposure control; NOT learned ranking) ----
    def _rerank_select(self, pool_idx, q, k):
        """Rerank a relevance-ordered candidate POOL (indices) to k item_ids using the
        configured heuristic. Returns selected item_ids. head_cap caps head items; tail_boost
        adds a popularity-inverse bonus. Pure heuristic — no learning, no labels."""
        ids = [str(self.i_uniq[i]) for i in pool_idx if 0 <= i < self.n_items]
        if self.rerank_policy == "tail_boost":
            sc = self.Y[pool_idx] @ np.asarray(q, dtype=np.float32)
            z = (sc - sc.mean()) / (sc.std() + 1e-9)
            adj = z + 0.6 * (1.0 - np.array([self._pop_pct.get(b, 0.0) for b in ids]))
            order = np.argsort(-adj); return [ids[i] for i in order[:k]]
        # default: head_cap — cap head items at cap_frac*k, fill with non-head, backfill
        cap = int(round(self.rerank_cap_frac * k)); out, n_head = [], 0
        for b in ids:
            if b in self._head:
                if n_head >= cap:
                    continue
                n_head += 1
            out.append(b)
            if len(out) >= k:
                break
        if len(out) < k:
            for b in ids:
                if b not in out:
                    out.append(b)
                if len(out) >= k:
                    break
        return out[:k]

    def recommend(self, user_id: str, k: int = 20, mode: str | None = None, request_id: str | None = None,
                  rerank: bool | None = None):
        t0 = time.perf_counter(); request_id = request_id or hashlib.md5((str(user_id)+str(t0)).encode()).hexdigest()[:12]
        do_rerank = self.enable_rerank if rerank is None else bool(rerank)
        rec = {"request_id": request_id, "user_id_hash": hashlib.md5(str(user_id).encode()).hexdigest()[:12],
               "timestamp": time.time(), "model_id": getattr(self, "model_id", None), "index_id": getattr(self, "index_id", None),
               "retrieval_mode": None, "fallback_used": False, "candidate_count": 0, "response_count": 0,
               "latency_ms": 0.0, "error_code": None,
               "rerank_applied": False, "rerank_policy": None, "base_candidate_ids": None}
        try:
            if not self.ready:
                rec.update(error_code="index_load_failure", fallback_used=True, retrieval_mode="fallback_popularity")
                kk = k if (isinstance(k, int) and 1 <= k <= 200) else 20
                items = self._pop_fallback(kk)  # degraded-but-serving: popularity loaded independently of the model
            elif not isinstance(k, int) or k < 1 or k > 200:
                rec.update(error_code="invalid_k", fallback_used=True, retrieval_mode="fallback_popularity")
                items = self._pop_fallback(20)
            else:
                m = mode or self.default_mode
                if user_id not in self.uidx:                                   # unknown / sparse / missing-factor user
                    rec.update(error_code="unknown_user", fallback_used=True, retrieval_mode="fallback_popularity")
                    items = self._pop_fallback(k)
                else:
                    try:
                        q = self.X[self.uidx[user_id]]
                        retr = self.hnsw if (m == "scale" and self.hnsw is not None) else self.exact
                        if do_rerank:                                           # G26B: widen pool, then heuristic rerank to k
                            pool_k = min(k * self.rerank_pool_mult, self.n_items)
                            pool = retr.search(q, pool_k); rec["candidate_count"] = int(len(pool))
                            base_ids = [str(self.i_uniq[i]) for i in pool[:k] if 0 <= i < self.n_items]
                            sel_ids = self._rerank_select(pool, q, k)
                            items = [{"item_id": b, "creator_id": self.b2c.get(b), "source": "als_rerank"} for b in sel_ids]
                            rec.update(rerank_applied=True, rerank_policy=self.rerank_policy, base_candidate_ids=base_ids)
                        else:
                            idx = retr.search(q, k); rec["candidate_count"] = int(len(idx))
                            items = [{"item_id": self.i_uniq[i], "creator_id": self.b2c.get(self.i_uniq[i]), "source": "als"} for i in idx if 0 <= i < self.n_items]
                        rec["retrieval_mode"] = "exact" if retr is self.exact else "scale_hnsw"
                        if not items:                                           # empty candidate set
                            rec.update(error_code="empty_candidates", fallback_used=True, retrieval_mode="fallback_popularity"); items = self._pop_fallback(k)
                    except Exception as fe:                                     # FAISS failure
                        rec.update(error_code="faiss_error", fallback_used=True, retrieval_mode="fallback_popularity"); items = self._pop_fallback(k)
            rec["response_count"] = len(items)
        except Exception as e:                                                  # catch-all: never crash
            rec.update(error_code="internal_error", fallback_used=True, retrieval_mode="fallback_popularity"); items = self._pop_fallback(20); rec["response_count"] = len(items)
        rec["latency_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        with self._lock:
            self._metrics["request_count"] += 1
            if rec["error_code"]: self._metrics["error_count"] += 1
            if rec["fallback_used"]: self._metrics["fallback_count"] += 1
            if rec["response_count"] == 0: self._metrics["empty_response_count"] += 1
            if rec["error_code"] == "unknown_user": self._metrics["unknown_user_count"] += 1
            self._by_mode[rec["retrieval_mode"] or "none"] += 1
            self._latencies.append(rec["latency_ms"])
            if len(self._latencies) > 50000: self._latencies = self._latencies[-50000:]
        log.info(json.dumps(rec))
        return {"items": items, **{kk: rec[kk] for kk in ("retrieval_mode", "fallback_used", "candidate_count", "response_count", "latency_ms", "error_code", "request_id", "model_id", "index_id", "rerank_applied", "rerank_policy", "base_candidate_ids")}}

    def metrics(self):
        with self._lock:
            lat = np.array(self._latencies) if self._latencies else np.array([0.0]); n = max(self._metrics["request_count"], 1)
            return {"request_count": self._metrics["request_count"], "error_count": self._metrics["error_count"],
                    "error_rate": round(self._metrics["error_count"]/n, 4), "fallback_count": self._metrics["fallback_count"],
                    "fallback_rate": round(self._metrics["fallback_count"]/n, 4), "empty_response_count": self._metrics["empty_response_count"],
                    "empty_response_rate": round(self._metrics["empty_response_count"]/n, 4),
                    "unknown_user_count": self._metrics["unknown_user_count"], "unknown_user_rate": round(self._metrics["unknown_user_count"]/n, 4),
                    "latency_p50_ms": round(float(np.percentile(lat, 50)), 3), "latency_p95_ms": round(float(np.percentile(lat, 95)), 3),
                    "latency_p99_ms": round(float(np.percentile(lat, 99)), 3), "by_mode": dict(self._by_mode),
                    "model_id": getattr(self, "model_id", None), "index_id": getattr(self, "index_id", None)}

    def prometheus(self):
        m = self.metrics(); lines = []
        for k in ["request_count", "error_count", "fallback_count", "empty_response_count", "unknown_user_count",
                  "error_rate", "fallback_rate", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms"]:
            lines.append(f"pulsediscover_{k} {m[k]}")
        return "\n".join(lines) + "\n"
