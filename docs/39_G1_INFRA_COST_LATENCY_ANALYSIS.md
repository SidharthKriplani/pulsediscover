# 39 — G1: Infra / Cost / Latency Analysis

*Engineering-realism memo for PulseDiscover's two-lane policy. **No new code, no new experiments, no measured production numbers.** All latency/cost figures are **labeled design estimates** based on the existing design and standard RecSys serving patterns — not benchmarks.*

---

## 1. Company question
**Can PulseDiscover's two-lane (90/10) policy be served in a realistic production recommendation stack, and what are the cost/latency tradeoffs?**

Short answer: yes, the architecture is production-plausible — it's a conventional two-stage (retrieval → light rerank) design where both lanes are precomputable. The interesting constraints are **cold-lane freshness**, **propensity logging completeness**, and **keeping exploration high enough for OPE** — not raw serving latency.

## 2. Serving architecture
- **Offline batch jobs** — ALS factor training; TF-IDF/content feature build; co-occurrence; cold candidate generation; OPE/health batch evaluation.
- **Model / artifact training** — ALS f64 (periodic full refit); content vectors (on metadata change); refreshed to artifact store.
- **Candidate indexes** — warm: ANN index over ALS item factors (or precomputed per-user top-N); cold: content-similarity index + same-author/same-series lookup tables.
- **Warm candidate retrieval** — per-user top-N from ALS (ANN or precomputed list).
- **Cold / content candidate retrieval** — content-hybrid (RRF of same-author + same-series + TF-IDF) over recently-added items.
- **Two-lane slate assembly** — merge 18 warm + 2 cold under the reserve-2 rule; dedup; apply guardrails.
- **Logging / propensity capture** — record slate, positions, per-slot propensities, lane, creator, reward (the P1 schema).
- **Monitoring** — warm-recall guardrail, cold-exposure share, creator Gini, ESS/positivity, latency, fallback rate.

## 3. Online vs offline work split
| Component | Offline / precomputed | Nearline | Online / request-time |
|---|---|---|---|
| ALS embeddings / factor scores | **Train factors; can precompute per-user top-N** | refresh on new interactions | optional ANN lookup |
| Content TF-IDF / metadata index | **Build vectors + index** | re-index on new items | similarity lookup |
| Content-hybrid cold lane | **Precompute cold candidate lists** | refresh as new items land | fetch top-k |
| 90/10 policy assembly | rule fixed offline | — | **assemble slate (merge/dedup/guardrail)** |
| OPE / logging | OPE runs as **batch**; logging schema fixed | — | **write log record** |
| Position-bias logging | curve **estimated offline through randomized position data or PBM-style modeling** | periodic re-estimate | **log position + propensity** |
| Health monitoring | **batch Gini/coverage/ESS** | — | emit counters |

Design principle: **push everything heavy offline.** At request time the work is a candidate fetch + a small merge/dedup + a log write — the policy itself is a rule, not a model.

## 4. Latency budget *(design estimates — not measured)*
Target for a single recommendation request (server-side, p95):

| Stage | Estimated budget (p95) |
|---|---|
| **Total** | **~120 ms** |
| Warm retrieval (ANN or precomputed fetch) | ~30–50 ms |
| Cold retrieval (content lane fetch) | ~15–30 ms |
| Reranking / policy assembly (merge, dedup, guardrail) | ~5–15 ms |
| Logging overhead (async write) | ~1–5 ms (off critical path) |
| **Fallback path** (cache/popularity if a lane times out) | ~5–10 ms |

Notes: if warm lists are **fully precomputed**, warm retrieval collapses toward a key-value read (~5–10 ms) and the budget is comfortable. The cold lane is the more variable cost (content similarity over a changing new-item set); cap its candidate set and time-box it. **These are planning estimates to size the system, explicitly not benchmarks.**

## 5. Cost drivers
- **Batch training** — periodic ALS refit (dominant compute; scales with users×items×factors×iterations) + content vector build. Cadence is the lever (daily vs hourly).
- **Embedding / index storage** — ALS factors (items×64 floats) + ANN index; modest.
- **Content index storage** — TF-IDF vectors + metadata lookups; grows with catalog, not traffic.
- **Request-time compute** — cheap (fetch + merge); scales with QPS.
- **Logging volume** — the real swing factor: full P1 schema × every served slot × QPS. Position + propensity per slot multiplies row size.
- **OPE / evaluation batch** — periodic; scales with log volume and number of candidate policies.
- **Monitoring dashboards** — counters/aggregations; minor.

**Cost shape:** training + logging dominate; request-time serving is cheap. Freshness cadence and logging completeness are the two biggest cost dials.

**Privacy / compliance note:** logging cost is not only storage — it carries **privacy and compliance constraints**. Propensity/reward logs reference users and must respect **retention windows** (time-bounded raw logs), **PII minimization** (log IDs/contexts, not raw personal data), and **aggregate-safe** derived tables for monitoring/OPE. These constraints can cap how much per-slot data is retained and for how long, which feeds back into both the logging-completeness tradeoff (§7) and OPE's data availability.

## 6. Scaling strategy
- **Small / portfolio scale** — precompute everything per user; serve from a flat store; full logging is trivially affordable; batch OPE on a laptop/single box. (This is PulseDiscover's actual regime.)
- **Medium content app** — move warm retrieval to an ANN service; re-index cold items nearline for freshness; sample logging if volume bites; scheduled OPE/health jobs.
- **Large consumer platform** — distributed ANN, streaming feature updates, real-time cold-item ingestion, **sampled + tiered logging** (full propensities on an exploration bucket, lighter logging elsewhere), continuous monitoring, and a dedicated experimentation platform. The 90/10 rule is unchanged; the infra around it is what scales. **Note:** at this scale the warm lane likely needs **nearline personalization freshness** — folding in recent user-session signals (last few interactions) between full ALS refits, so recommendations reflect in-session intent rather than only the last batch.

## 7. Tradeoffs
- **Precompute vs online scoring** — precompute = low latency, staler; online ANN = fresher, costlier. Two-lane policy works either way; start precomputed.
- **Freshness vs cost** — cold lane's whole value is new items, so it needs the freshest index; that's the most freshness-sensitive (and cost-sensitive) component.
- **Exploration vs latency** — exploration (for OPE positivity) is a logging/policy concern, not latency; it costs *relevance* and *log volume*, not ms.
- **Cold-lane quality vs retrieval cost** — richer content similarity (bigger TF-IDF, more neighbors) improves cold recall but raises index + fetch cost; cap candidate set.
- **Logging completeness vs storage** — full per-slot propensities enable trustworthy OPE/position-bias but multiply log volume; tier it (full on exploration bucket).
- **Position-bias estimation vs experimentation complexity** — estimating examination offline through randomized position data or PBM-style modeling adds experiment machinery but is required to read CTR safely (P1).
- **Creator-health monitoring vs serving simplicity** — Gini/coverage monitoring adds batch jobs but is cheap and catches amplification (H1); keep it off the serving path.

## 8. Failure modes
| Failure | Effect | Mitigation |
|---|---|---|
| Cold index stale | Discovery lane serves old "new" items; cold value decays | Freshness SLA + alert on cold-item age |
| ALS factors stale | Warm relevance drifts down | Refit cadence + drift monitor on warm recall |
| Logging propensities missing | OPE/position-bias become unusable | Treat propensity logging as a hard serving contract; alert on nulls |
| Latency timeout | Slate incomplete | Fallback to cached/popularity slate; count fallback rate |
| Popularity feedback loop | Head amplifies, catalog narrows (H1) | Monitor creator Gini/coverage; protected cold reserve already counters partially |
| Exploration support too low | OPE positivity fails (O1: ESS collapse) | Maintain a structured-exploration bucket (**~30–40% cold is a design estimate, tuned by experiment power and relevance guardrails**) |
| Storage / logging cost too high | Budget overrun | Tiered/sampled logging; full propensities only on exploration bucket |

## 9. Production-readiness verdict
- **Plausible architecture** — ✅ conventional two-stage retrieve-then-assemble; both lanes precomputable; request-time work is light.
- **Not actually deployed** — ❌ no live system.
- **No measured production latency** — ❌ §4 are design estimates only.
- **No real infra benchmark** — ❌ no load test, no QPS/cost measurement.
- **No online traffic** — ❌ no real users, no served logs.

**Verdict:** the design is *production-plausible and well-shaped* (heavy work offline, cheap online, clear failure handling), but everything here is **architecture and sizing analysis, not a deployed-and-measured system.**

## 10. Interview-safe summary
- "It's a standard two-stage design — retrieve warm + cold candidates, then assemble the slate with a rule — so the heavy work is offline and request-time is cheap."
- "My latency numbers are **design estimates** (~120 ms p95 total) to size the system, not benchmarks — I haven't deployed it."
- "The real cost dials are **training cadence and logging completeness**, not serving compute."
- "The cold lane is the most freshness-sensitive and cost-sensitive part, because its value *is* new items."
- "Exploration for OPE costs **relevance and log volume, not latency** — and you need it or OPE positivity fails."
- "Failure handling is explicit: stale-index SLAs, propensity-logging as a hard contract, and a popularity/cache fallback on timeout."
- "Honest boundary: plausible, well-shaped architecture — **not deployed, not benchmarked, no live traffic.**"

**STOP — awaiting G1 review.**
