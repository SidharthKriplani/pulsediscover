# 31 — AB1: PulseDiscover Two-Lane A/B-Readiness Spec

*Internal company experiment spec. **This is not the final PRD/PDF.** It states what we would test, why, how we would log it, and what would make us ship / hold / kill — assembled from the offline evidence in C1, C2, O1, O1b, P1, H1, H2. Everything here is **offline candidate evidence; no business-lift, cold-start, fairness, or deployment claim.***

---

## 1. Current production / control policy
| Component | What it is | Status |
|---|---|---|
| **Control policy** | **ALS-only** top-20 (warm collaborative recall) | the A/B control |
| Best warm generator | **ALS f64** — R@20 0.085 (LLO 5k sample), strongest in-sandbox single generator; iters converged by 4, f128+ deferred to external | BUILT — real data |
| Sequence model | **D3B SASRec** — R@20 0.036, **below every collaborative baseline** (ALS f64 −0.048 @20) | BUILT — honest negative (CPU/last-position limited) |
| Cold/new-content lane | **content-hybrid** (RRF of same-author + same-series + TF-IDF) — cold R@20 0.241, 100% cold-pool coverage; ALS is structurally 0 on new items | BUILT — real data |

ALS f64 is the warm floor; content-hybrid is the cold lane; SASRec is a documented limited attempt, **not** part of any shipped/tested arm.

## 2. Default A/B candidate — 90/10 reserve-2
**Policy:** top-20 = 18 ALS f64 (warm) + **2 reserved cold/new slots** (content-hybrid).

**Why it's the default arm:** it adds real cold-discovery and catalog reach (cold R@20 0.050; **735 distinct new books**; +26% distinct creators; **+444 brand-new creators** reachable only via the cold lane) while keeping warm relevance loss **within the ≤10% guardrail** (warm R@20 0.048 vs ALS 0.052 → **−8.1%**).

**Caveat (binding):** OPE **cannot** prove 90/10 beats ALS-only — DR values are statistically indistinguishable with overlapping CIs (O1b). 90/10 is a **product-rational A/B candidate, not an offline-proven winner.** The live A/B settles the micro-edge.

## 3. Optional exploration arms (NOT default)
| Arm | Profile | Why exploration-only |
|---|---|---|
| **80/20 reserve-4** | cold R@20 0.092, 1,020 distinct cold | warm loss **−17%** (breaks ≤10% guardrail) |
| **RRF / aggressive discovery** | highest proxy value & cold reach (cold R@20 0.171, 1,299 distinct, 799 new creators) | warm loss **−44%**; value≠ship |
| **cap2 / cap1 / MMR** (creator de-concentration) | genuinely lowers warm-only creator Gini (0.84→0.805) | costs **−25% to −36%** warm recall (H2); only if the company relaxes the warm guardrail |

All three are discovery/health-heavy arms to run **with warm-retention guardrails monitored**, never as the default.

## 4. Offline evidence summary
- **C1 — cold-start channel exists.** Content-hybrid cold R@20 **0.241** / R@200 0.557, 100% cold-pool coverage, where ALS f64 is structurally 0. Establishes the cold lane. *(Offline candidate channel, not a cold-start business win.)*
- **C2 — two-lane slot tradeoff.** More cold slots → more discovery, less warm recall: als (warm 0.052 / cold 0) → 90/10 (0.048 / 0.050, −8.1%) → 80/20 (0.043 / 0.092, −17%) → rrf (0.029 / 0.171, −44%). **90/10 is the only discovery policy inside the ≤10% guardrail.**
- **O1 — OPE / A-B readiness.** Known-propensity logger; estimators (IPS/SNIPS/DM/DR) validated vs known truth. **Logger positivity is decisive:** an ALS-heavy logger gives unreliable OPE (ESS ~1.4k, w_max ~290, DR over-estimates rrf 3×); a cold-support logger (ε≈0.35, focused cold pool) fixes it (ESS up to 7.6k, w_max 50). OPE ranks discovery arms above als but **cannot separate 90/10 from als**.
- **O1b — prevalence correction.** Re-weighting to realistic 76.5% warm / 23.5% cold **compresses the field** (als 0.00037→0.00057; rrf 0.00141→0.00091) and confirms `dr_separates_90_10_from_als = False`. 90/10 = guardrail-passing product-rational candidate; rrf = value-max but guardrail-violating.
- **P1 — position-bias correction (measurement layer).** With PBM examination (95%→12% top-to-bottom), **naive CTR inverts the truth at the cold slots** (cold lane: naive 0.0011 vs true 0.0126 → **11× under-credit**); IPS recovers ~77%. **Raw bottom-of-page CTR would wrongly kill the cold lane.** Per-item IPS trades bias for variance (worse MSE at sparse impressions) → trust aggregate-level correction. Defines the required A/B log fields.
- **H1 — catalog reach vs concentration.** 90/10 **widens reach** (+26% creators, +444 new) and **holds a protected 10% cold-exposure share every round** under feedback amplification — but **does not de-concentrate the head** (creator Gini flat-to-worse). Warm-lane concentration amplifies over rounds for *all* policies.
- **H2 — reranking negative.** Creator-cap/MMR **works on the warm head** (warm-only Gini 0.843→0.805) but **cannot reduce concentration within the ≤10% guardrail** — base 90/10 already spends the budget (cap3 −18%, cap1 −36%). De-concentration needs a training-time or budget-relaxed intervention.

## 5. A/B design
- **Unit of randomization:** user (stable hash bucketing); user-level to avoid within-user contamination across the two lanes.
- **Treatment arms:** **Control** = ALS-only · **T1 (default)** = 90/10 reserve-2 · *(optional)* **T2** = 80/20 · **T3** = RRF — exploration arms behind their own guardrails.
- **Primary metric:** warm/overall engagement (e.g., click/save-through on served slate), prevalence-weighted. *(Offline proxy was examined-gold-hit; online primary is real engagement.)*
- **Secondary metrics:** cold/new-item engagement & discovery rate, distinct items/creators surfaced, **new-creator reach**, catalog coverage, long-tail exposure share.
- **Guardrails (must-hold):** warm relevance/engagement loss **≤10% vs control**; no significant latency regression; creator-exposure Gini **not materially worse** (monitor, don't gate hard given H1/H2); session abandonment flat.
- **Minimum logging schema (from P1):** `timestamp, user_id, user_context, policy_id, slate_position, item_id, lane(warm/cold), creator_id, examination_propensity(position), logging_propensity(item), reward(engagement)`.
- **Positivity / exploration requirement (from O1):** serve behind a logger with **structured cold exploration** (≈30–40% cold or a focused cold pool) so propensities are known and ESS/variance are acceptable; otherwise OPE/interleaving is untrustworthy.
- **Position-bias correction requirement (from P1):** estimate the examination curve online (randomization / PBM-EM) and report **IPS examination-corrected** lane/position metrics at the aggregate; **do not** rank items/lanes or make ship/kill calls on raw CTR.
- **Ship / Hold / Kill:**
  - **SHIP** T1 if warm guardrail holds (≤10% loss) **and** cold/new engagement improves significantly **and** position-corrected estimates confirm the cold lane carries real (non-position) value.
  - **HOLD** if warm loss is borderline or cold value is positive but not significant → extend / re-power.
  - **KILL** if warm guardrail breaks, or position-corrected cold value is null/negative, or latency/abandonment regress.
  - Exploration arms (T2/T3) ship **only** as opt-in discovery surfaces, never as the default ranker, and only if their warm loss is deemed acceptable for that surface.

## 6. Claim boundaries (explicit)
- **Offline candidate only** — all results are offline recall / known-propensity OPE / simulation.
- **No business-lift claim** — no online experiment has been run.
- **No cold-start business win** — C1 is an offline candidate channel, not proven discovery lift.
- **No creator fairness "solved"** — H1/H2 show reach without de-concentration; fairness remains open.
- **No SASRec win** — D3B underperforms ALS f64; honest negative.
- **No production-deployment claim** — this is a spec for an experiment, not a shipped system.

## 7. Portfolio packaging note — optional extensions (not blockers)
The A/B-readiness story is complete without these; each is an optional upgrade:
- **External canonical SASRec** — full-position, d≥64, 100+ epochs, GPU; re-evaluate vs the ALS f64 floor.
- **Training-time creator fairness** — bake creator de-concentration into the warm retrieval model (the lever H2 showed post-hoc reranking can't provide cheaply).
- **SNIPS / clipping / PAL** — variance reduction on the P1 per-item position-bias estimator.
- **Deeper T3 models** — two-tower / richer sequence models on the candidate-generation floor.

None of these block the 90/10 A/B; they are follow-ons that would strengthen specific components.

---
**Status:** AB1 spec complete. Control = ALS-only; default arm = 90/10 reserve-2 (product-rational, not offline-proven); exploration arms = 80/20 / RRF / creator-cap (guardrail-relaxed). Logging, positivity, and position-bias requirements specified; ship/hold/kill defined; claims bounded.

**STOP — awaiting AB1 review.**
