"""G34 (V2) — Kafka real-time streaming scorer for PulseDiscover.

Turns the request/response RecommenderService into an event-driven, real-time scoring
path: interaction events arrive on a Kafka topic, each one is scored through the SAME
RecommenderService.recommend() used by the FastAPI service, and the resulting
recommendation slate is published to an output topic. This is the streaming twin of the
G23 serving API — identical model, identical fallback tree, identical telemetry — so
online behaviour is consistent whether a caller hits REST or the event bus.

Design notes (honest scope):
  * Real Kafka client code (kafka-python). Broker is provided by deploy/docker-compose.kafka.yml
    (single-node Apache Kafka, KRaft mode). This is production-SHAPED, not a production cluster.
  * `offline` mode runs the exact per-event scoring callback WITHOUT a broker, so the streaming
    logic is unit-testable in CI (see scripts/run_g34_kafka_stream_offline.py).
  * At-least-once processing: the consumer commits offsets only after the recommendation is
    produced, so a crash mid-event re-processes rather than drops. Scoring is idempotent per event.

Topics
  pd.interactions   (in)   {"user_id": str, "item_id": str|None, "event_type": str, "ts": float}
  pd.recommendations(out)  {"user_id", "trigger", "items": [...], "latency_ms", "model_id",
                            "fallback_used", "error_code", "request_id", "ts"}

CLI
  python -m serving.kafka_stream offline  --n 500              # no broker; tests scoring path
  python -m serving.kafka_stream produce  --n 1000 --rate 200  # publish synthetic interactions
  python -m serving.kafka_stream consume  --k 20               # score events -> recommendations

Env
  PD_KAFKA_BOOTSTRAP   default "localhost:9092"
  PD_DATADIR           default (repo) data/interim
  PD_KAFKA_GROUP       default "pulsediscover-scorer"
"""
from __future__ import annotations
import os, sys, json, time, random, argparse, logging, signal
from typing import Optional, Dict, Any, List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from recommender_service import RecommenderService  # type: ignore

INTERACTIONS_TOPIC = os.environ.get("PD_KAFKA_IN_TOPIC", "pd.interactions")
RECOMMENDATIONS_TOPIC = os.environ.get("PD_KAFKA_OUT_TOPIC", "pd.recommendations")
BOOTSTRAP = os.environ.get("PD_KAFKA_BOOTSTRAP", "localhost:9092")
GROUP = os.environ.get("PD_KAFKA_GROUP", "pulsediscover-scorer")
_DEFAULT_DATADIR = os.path.join(os.path.dirname(HERE), "..", "data", "interim")
DATADIR = os.environ.get("PD_DATADIR", os.path.abspath(_DEFAULT_DATADIR))

log = logging.getLogger("pulsediscover.streaming")
if not log.handlers:
    _h = logging.StreamHandler(); _h.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(_h); log.setLevel(logging.INFO)


class RealTimeScorer:
    """Wraps a RecommenderService and turns a single interaction event into a recommendation
    record. Kept separate from the Kafka wiring so it can be unit-tested with no broker."""

    def __init__(self, datadir: str = DATADIR, k: int = 20, build_hnsw: bool = False):
        t0 = time.perf_counter()
        # build_hnsw off by default here: streaming path uses the exact FlatIP index (lower memory,
        # faster cold start). Flip on for the scale/HNSW lane if the event rate demands it.
        self.svc = RecommenderService(datadir, default_mode="exact", build_hnsw=build_hnsw)
        self.k = k
        self._events = 0
        self._lat_sum = 0.0
        log.info(json.dumps({"event": "scorer_ready", "ready": self.svc.ready,
                             "load_error": self.svc.load_error, "datadir": datadir,
                             "warmup_s": round(time.perf_counter() - t0, 3),
                             "model_id": getattr(self.svc, "model_id", None)}))

    def score_event(self, evt: Dict[str, Any]) -> Dict[str, Any]:
        """Score one interaction event -> recommendation record. Never raises: a malformed event
        yields an error record, matching the service's never-crash contract."""
        t0 = time.perf_counter()
        user_id = str(evt.get("user_id", "")) if isinstance(evt, dict) else ""
        if not user_id:
            rec = {"user_id": None, "trigger": None, "items": [], "error_code": "bad_event",
                   "fallback_used": True, "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
                   "model_id": getattr(self.svc, "model_id", None), "ts": time.time()}
            self._events += 1
            return rec
        r = self.svc.recommend(user_id=user_id, k=self.k)
        out = {
            "user_id": user_id,
            "trigger": {"item_id": evt.get("item_id"), "event_type": evt.get("event_type"),
                        "ts": evt.get("ts")},
            "items": r["items"],
            "response_count": r["response_count"],
            "retrieval_mode": r["retrieval_mode"],
            "fallback_used": r["fallback_used"],
            "error_code": r["error_code"],
            "request_id": r["request_id"],
            "model_id": r["model_id"],
            "latency_ms": r["latency_ms"],
            "ts": time.time(),
        }
        self._events += 1
        self._lat_sum += r["latency_ms"]
        return out

    def throughput(self) -> Dict[str, Any]:
        n = max(self._events, 1)
        return {"events_scored": self._events, "avg_scoring_latency_ms": round(self._lat_sum / n, 3),
                "service_metrics": self.svc.metrics()}


# ------------------------- Kafka wiring (lazy import) -------------------------
def _producer(bootstrap: str):
    from kafka import KafkaProducer  # lazy: offline mode needs no kafka-python
    return KafkaProducer(
        bootstrap_servers=bootstrap,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: (k or "").encode("utf-8"),
        acks="all", retries=3, linger_ms=20,
    )


def _consumer(bootstrap: str, topic: str, group: str):
    from kafka import KafkaConsumer
    return KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap,
        group_id=group,
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
        key_deserializer=lambda b: b.decode("utf-8") if b else None,
        enable_auto_commit=False,          # at-least-once: commit only after producing the rec
        auto_offset_reset="earliest",
        max_poll_records=200,
    )


def run_consumer(bootstrap: str = BOOTSTRAP, datadir: str = DATADIR, k: int = 20,
                 in_topic: str = INTERACTIONS_TOPIC, out_topic: str = RECOMMENDATIONS_TOPIC,
                 group: str = GROUP, max_events: Optional[int] = None) -> None:
    """Consume interaction events, score each in real time, produce recommendation slates.
    at-least-once: offsets are committed only after the output record is sent."""
    scorer = RealTimeScorer(datadir=datadir, k=k)
    consumer = _consumer(bootstrap, in_topic, group)
    producer = _producer(bootstrap)
    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    log.info(json.dumps({"event": "consumer_start", "bootstrap": bootstrap, "in": in_topic,
                         "out": out_topic, "group": group}))
    n = 0
    try:
        for msg in consumer:
            if stop["flag"]:
                break
            rec = scorer.score_event(msg.value)
            producer.send(out_topic, key=rec.get("user_id") or "na", value=rec)
            producer.flush()
            consumer.commit()               # commit AFTER the rec is durably sent (at-least-once)
            log.info(json.dumps({"event": "scored", "user_id": rec.get("user_id"),
                                 "response_count": rec.get("response_count"),
                                 "fallback_used": rec.get("fallback_used"),
                                 "latency_ms": rec.get("latency_ms"),
                                 "offset": msg.offset, "partition": msg.partition}))
            n += 1
            if max_events and n >= max_events:
                break
    finally:
        log.info(json.dumps({"event": "consumer_stop", **scorer.throughput()}))
        try: producer.flush(); producer.close(); consumer.close()
        except Exception: pass


def run_producer(bootstrap: str = BOOTSTRAP, datadir: str = DATADIR, n: int = 1000,
                 rate: float = 200.0, in_topic: str = INTERACTIONS_TOPIC,
                 known_frac: float = 0.7) -> None:
    """Publish synthetic interaction events to the interactions topic. Draws a `known_frac`
    fraction of user_ids from the real ALS user index so the stream produces genuine model
    hits (not only popularity fallback); the remainder are unknown users (cold-start path)."""
    producer = _producer(bootstrap)
    known_users = _sample_known_users(datadir, k=5000)
    interval = 1.0 / rate if rate > 0 else 0.0
    log.info(json.dumps({"event": "producer_start", "topic": in_topic, "n": n, "rate": rate,
                         "known_users_available": len(known_users)}))
    sent = 0
    for i in range(n):
        evt = _synth_event(known_users, known_frac)
        producer.send(in_topic, key=evt["user_id"], value=evt)
        sent += 1
        if sent % 500 == 0:
            producer.flush()
            log.info(json.dumps({"event": "produced", "sent": sent}))
        if interval:
            time.sleep(interval)
    producer.flush(); producer.close()
    log.info(json.dumps({"event": "producer_done", "sent": sent}))


def run_offline(datadir: str = DATADIR, n: int = 500, k: int = 20, known_frac: float = 0.7,
                evidence_path: Optional[str] = None) -> Dict[str, Any]:
    """No broker: synthesize n interaction events and push them through the SAME score_event
    callback the consumer uses. Verifies the streaming scoring path end-to-end and emits an
    evidence summary. This is what CI runs."""
    scorer = RealTimeScorer(datadir=datadir, k=k)
    known_users = _sample_known_users(datadir, k=5000)
    # Expected error_codes return a valid fallback response by design (cold-start / bad input);
    # only these signal an actual streaming defect.
    EXPECTED = {None, "unknown_user", "invalid_k", "empty_candidates"}
    hits = cold_start = unexpected_errors = 0
    for _ in range(n):
        evt = _synth_event(known_users, known_frac)
        rec = scorer.score_event(evt)
        ec = rec.get("error_code")
        if ec == "unknown_user":
            cold_start += 1
        if ec not in EXPECTED:
            unexpected_errors += 1
        if not rec.get("fallback_used"):
            hits += 1
    tp = scorer.throughput()
    empties = tp["service_metrics"]["empty_response_count"]
    summary = {
        "gate": "G34_kafka_realtime_streaming_offline",
        "events": n, "k": k, "known_frac": known_frac,
        "model_hits": hits, "cold_start_fallbacks": cold_start, "unexpected_errors": unexpected_errors,
        "empty_responses": empties,
        "avg_scoring_latency_ms": tp["avg_scoring_latency_ms"],
        "p95_scoring_latency_ms": tp["service_metrics"]["latency_p95_ms"],
        "service_ready": scorer.svc.ready,
        "model_id": getattr(scorer.svc, "model_id", None),
        # Healthy iff the model loaded, every event got a non-empty slate, and nothing failed
        # unexpectedly. Cold-start fallbacks are a feature, not a failure.
        "verdict": "PASS" if (scorer.svc.ready and empties == 0 and unexpected_errors == 0) else "FAIL",
    }
    log.info(json.dumps({"event": "offline_summary", **summary}))
    if evidence_path:
        os.makedirs(os.path.dirname(evidence_path), exist_ok=True)
        json.dump(summary, open(evidence_path, "w"), indent=2)
    return summary


# ------------------------------- helpers -------------------------------
def _sample_known_users(datadir: str, k: int = 5000) -> List[str]:
    """Pull real user_ids from the ALS user index (c2_als.pkl) so synthetic streams hit the
    actual model. Falls back to an empty list (all cold-start) if the index isn't available."""
    import pickle
    p = os.path.join(datadir, "c2_als.pkl")
    try:
        A = pickle.load(open(p, "rb"))
        users = list(A["uidx"].keys())
        random.shuffle(users)
        return [str(u) for u in users[:k]]
    except Exception as e:
        log.info(json.dumps({"event": "known_users_unavailable", "error": str(e)}))
        return []


def _synth_event(known_users: List[str], known_frac: float) -> Dict[str, Any]:
    if known_users and random.random() < known_frac:
        uid = random.choice(known_users)
    else:
        uid = "cold_" + str(random.randint(1, 10_000_000))   # unknown user -> cold-start path
    return {"user_id": uid, "item_id": str(random.randint(1, 2_000_000)),
            "event_type": random.choice(["view", "click", "add_to_shelf"]), "ts": time.time()}


def main(argv=None):
    ap = argparse.ArgumentParser(description="PulseDiscover Kafka real-time streaming scorer (G34)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("consume", help="score interaction events -> recommendations")
    c.add_argument("--k", type=int, default=20); c.add_argument("--max-events", type=int, default=None)
    p = sub.add_parser("produce", help="publish synthetic interaction events")
    p.add_argument("--n", type=int, default=1000); p.add_argument("--rate", type=float, default=200.0)
    p.add_argument("--known-frac", type=float, default=0.7)
    o = sub.add_parser("offline", help="run scoring path with no broker (CI)")
    o.add_argument("--n", type=int, default=500); o.add_argument("--k", type=int, default=20)
    o.add_argument("--known-frac", type=float, default=0.7); o.add_argument("--evidence", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "consume":
        run_consumer(k=args.k, max_events=args.max_events)
    elif args.cmd == "produce":
        run_producer(n=args.n, rate=args.rate, known_frac=args.known_frac)
    elif args.cmd == "offline":
        s = run_offline(n=args.n, k=args.k, known_frac=args.known_frac, evidence_path=args.evidence)
        sys.exit(0 if s["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
