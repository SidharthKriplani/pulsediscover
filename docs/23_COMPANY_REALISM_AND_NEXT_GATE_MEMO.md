# 23 — Company-Realism & Next-Gate Memo

*Decision memo only. No training, no OPE, no position-bias, no feedback-loop, no PRD/PDF. Aligns PulseDiscover with a realistic company recommender workflow and picks the next gate.*

## 1. Realistic company objective
For a fiction/content platform, the practical objective is **more meaningful content discovery and repeat engagement** — readers find stories they actually read and come back, while new and long-tail creators still get discovered. That objective decomposes by system layer into offline proxies + guardrails:
- **Candidate generation →** Recall@K (did the right item make the shortlist?).
- **Ranking →** NDCG@K (is it near the top?).
- **Guardrails (must not regress):** latency, **diversity / creator & catalog concentration** (Gini/entropy), **cold-start coverage** (new items/authors surfaced), and **user fatigue** (over-exposure / repetition).
No single offline number is the goal; the goal is discovery+retention, approximated by these per-layer metrics under guardrails.

## 2. Current model decision
- **Best generator: ALS f64** — Recall@20 0.085 / Recall@200 0.298 on the domain (5k-user LLO).
- **D3B SASRec: honest, *limited* negative** — a CPU-bounded, last-position, under-trained, small-config implementation. **Framing: this *implementation* was not strong enough to beat the current ALS candidate-generation floor — NOT "SASRec is bad."** A canonical full-position GPU SASRec is untested.
- **Ship-like decision:** **do not replace ALS with this SASRec.**
- **Offline decision:** **ALS f64 remains the candidate-generation floor.**

## 3. Why offline metrics still matter (and their limits)
- Companies use offline evaluation to **cheaply screen many strategies** before spending scarce A/B traffic — it's the filter, not the verdict.
- **Offline wins are not business wins.** A/B (online) is the moment of truth; offline-online gaps are real (which is exactly why PulseDiscover's OPE/DR layer exists — to make offline screening *less* misleading).
- **Match the metric to the layer:** candidate-gen = Recall@K; ranking = NDCG@K; reranking = diversity/guardrails; exploration = cold-start coverage + regret; governance = calibration of the offline estimate (IPS/SNIPS/DR), catalog-health, and drift. Reporting one global number across layers is the classic mistake.

## 4. Next-gate recommendation
**Option A** — one bounded canonical SASRec **external** run (full-position, d≥64, 100+ epochs, GPU) to test whether a *fair* sequence model can beat/complement ALS.
**Option B** — skip more sequence modeling; proceed to the **governance layers on ALS f64**: OPE (IPS/SNIPS/DR), position-bias (IPS vs PAL), creator/catalog-health feedback-loop.

**Recommendation: do B now; treat A as an optional, non-blocking external experiment.**
- The portfolio's **differentiator is the evaluation/governance layer**, not the model leaderboard. B advances that on a *real domain* with *real creators* — the highest-credibility remaining work, and it's in-sandbox.
- The candidate-gen floor (ALS f64) is already solid; B doesn't need a better generator to be valuable.
- A needs a GPU and only answers "can a fair SASRec beat ALS?" — useful but **not on the critical path** to a strong flagship. Run it in parallel via the external pack if/when convenient; let its result *complement* (a sequence signal in the hybrid), not gate the package.

## 5. Portfolio control — keep PulseDiscover finite
**Minimum remaining gates to package PulseDiscover as a flagship (then stop):**
1. **Domain OPE** — IPS/SNIPS/DM/DR on the ALS-f64 policy via the known-propensity simulator on the domain (the differentiator, now on real data).
2. **Domain position-bias** — IPS reweighting vs PAL, item-movement.
3. **Domain creator/catalog-health feedback-loop** — real creator-level Gini/entropy + ratchet (the upgrade real creators unlock).
4. **PRD + evidence-ledger update** — promote domain numbers to headline, demote MovieLens to smoke, fold in the honest SASRec negative.
5. **Defense PDF** — grounded in the domain outputs.

**Do NOT build (unless it fills a real, named gap):** more sequence models beyond the single optional external SASRec; LightGCN / two-tower / DeepFM / other T3 models; multi-objective revenue ranking; real-time serving infra; a cold-start *content channel* (discuss-only unless a gate explicitly needs it). These add surface area without strengthening the core differentiator.

## 6. Bottom line
ALS f64 is the floor; the SASRec attempt is a documented, bounded negative (implementation, not paradigm). The flagship is won by the **governance layers on real domain data**, not by chasing a better generator. **Recommend Gate B next (domain OPE first); SASRec-canonical stays an optional external side-quest.**

*No code run. Awaiting your pick: B (recommended) or A.*
