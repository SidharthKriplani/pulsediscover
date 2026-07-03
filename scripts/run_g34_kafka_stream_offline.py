#!/usr/bin/env python3
"""G34 — offline acceptance for the Kafka real-time streaming scorer (no broker required).

Runs a batch of synthetic interaction events through the SAME per-event scoring callback the
Kafka consumer uses, and asserts the streaming path is healthy: model loaded, every event
returns a non-empty slate, and no unexpected errors. Writes evidence to
outputs/evidence/g34_kafka_stream_offline.json. Exit code 0 = PASS.

    python scripts/run_g34_kafka_stream_offline.py --n 1000
"""
import os, sys, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "serving"))
from serving.kafka_stream import run_offline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--known-frac", type=float, default=0.7)
    ap.add_argument("--datadir", default=os.path.join(ROOT, "data", "interim"))
    args = ap.parse_args()
    evidence = os.path.join(ROOT, "outputs", "evidence", "g34_kafka_stream_offline.json")
    summary = run_offline(datadir=args.datadir, n=args.n, k=args.k,
                          known_frac=args.known_frac, evidence_path=evidence)
    print("\nG34 offline acceptance:", summary["verdict"])
    print(f"  model_hits={summary['model_hits']}  cold_start_fallbacks={summary['cold_start_fallbacks']}"
          f"  unexpected_errors={summary['unexpected_errors']}  empty_responses={summary['empty_responses']}")
    print(f"  avg_latency={summary['avg_scoring_latency_ms']}ms  p95={summary['p95_scoring_latency_ms']}ms")
    print(f"  evidence -> {evidence}")
    sys.exit(0 if summary["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
