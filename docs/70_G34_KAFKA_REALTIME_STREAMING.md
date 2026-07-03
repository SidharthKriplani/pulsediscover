# G34 — Kafka Real-Time Streaming Scorer

**Status:** implemented, offline-accepted (broker path runs locally via `docker-compose.kafka.yml`).
**Scope:** production-SHAPED, not a production cluster — single-node Apache Kafka (KRaft), same honesty boundary as the G23 serving API.

## Why

The G23 FastAPI service scores on request/response. G34 adds the **event-driven twin**: interaction
events arrive on a Kafka topic and are scored in real time through the **same** `RecommenderService`
— identical ALS+FAISS model, identical fallback tree, identical telemetry. Whether a caller hits REST
or the event bus, online behaviour is consistent. This is the pattern most "real-time / streaming /
event-driven" recommender roles ask for.

## Topics

| Topic | Direction | Schema |
|---|---|---|
| `pd.interactions` | in | `{"user_id": str, "item_id": str\|null, "event_type": str, "ts": float}` |
| `pd.recommendations` | out | `{"user_id", "trigger", "items": [...], "response_count", "retrieval_mode", "fallback_used", "error_code", "request_id", "model_id", "latency_ms", "ts"}` |

## Components (`src/serving/kafka_stream.py`)

- **`RealTimeScorer`** — wraps `RecommenderService`; `score_event(evt) -> rec`. Never raises (a malformed
  event yields a `bad_event` record), matching the service's never-crash contract. Broker-free, so it is
  unit-testable.
- **`run_consumer`** — subscribes to `pd.interactions`, scores each event, produces the slate to
  `pd.recommendations`. **At-least-once**: `enable_auto_commit=False` and offsets are committed only
  *after* the output record is durably sent (`acks="all"`), so a crash mid-event re-processes rather than
  drops.
- **`run_producer`** — publishes synthetic interactions; 70% of `user_id`s are drawn from the real ALS
  user index (`c2_als.pkl`) so the stream produces genuine model hits, 30% are unknown (cold-start path).
- **`run_offline`** — no broker; pushes N events through `score_event` and emits an evidence summary. CI entrypoint.

## Run

```bash
# 1) offline acceptance — no broker, this is what CI runs
python scripts/run_g34_kafka_stream_offline.py --n 1000

# 2) full broker path (local)
docker compose -f docker-compose.kafka.yml up -d kafka
docker compose -f docker-compose.kafka.yml up pulsediscover-stream          # consumer/scorer
docker compose -f docker-compose.kafka.yml run --rm producer --n 3000 --rate 300   # replay events
# inspect output:
#   docker exec -it pd-kafka /opt/kafka/bin/kafka-console-consumer.sh \
#     --bootstrap-server localhost:9092 --topic pd.recommendations --from-beginning
```

## Offline acceptance (evidence: `outputs/evidence/g34_kafka_stream_offline.json`)

| Metric | Value (n=1000, seed-varied) |
|---|---|
| Verdict | **PASS** |
| Model hits (real ALS retrieval) | ~65% of events |
| Cold-start fallbacks (correct, unknown user → popularity) | ~35% of events |
| Unexpected errors | **0** |
| Empty responses | **0** |
| Avg scoring latency | ~0.2 ms |
| p95 scoring latency | ~0.5 ms |

Cold-start fallbacks are a **feature**, not a failure — the acceptance gate only fails on unexpected
error codes or empty slates.

## Truth boundary

- ✅ Real Kafka client code (kafka-python), real broker via compose, real model scoring per event.
- ✅ At-least-once delivery semantics (commit-after-produce).
- ❌ Not a multi-broker production cluster; no schema registry / exactly-once transactions (documented exclusion).
- ❌ Offline acceptance uses synthetic interaction events, not a live user stream (same synthetic-vs-live boundary as the rest of PulseDiscover).
