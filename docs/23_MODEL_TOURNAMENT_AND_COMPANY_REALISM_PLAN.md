# 23 — Model Tournament & Company-Realism Plan

*Planning only — no training, no PRD/PDF, no broad T3. Defines a company-realistic model tournament for PulseDiscover: every model is run because it answers a product/system question, gated on beating the current offline floor under guardrails, with A/B as the final arbiter. Portfolio can be more exhaustive than a company sprint, but the narrative stays realistic — not model decoration.*

## 1. Current state
- **Current best generator: ALS f64** — R@10 0.0508 · R@20 0.0846 · R@50 0.1466 · R@100 0.2154 · R@200 0.2978 · NDCG@10 0.0259 (Goodreads fantasy/paranormal core, 5k-user leave-last-out, bootstrap CIs).
- **Offline ladder so far:** popularity (R@20 0.035) → same-author (0.042) → co-occ 5k→10k (0.046→0.050) → ALS f16 (0.057) → f16 hybrid (0.061) → ALS f32 (0.073) → **ALS f64 (0.085)**.
- **D3B SASRec = honest *negative*, limited implementation.** CPU-bounded, **last-position** (not full-position) training, d48/1-block/maxlen30, 20 epochs (loss not plateaued) → R@20 0.036, below every collaborative baseline. **This does NOT mean SASRec is globally bad** — it was not given the canonical full-position/GPU setup. No conclusion about transformers in general.
- **No cold-start progress claimed.** Leave-last-out golds are ~all warm (99% in-vocab); the real cold-start signal (D2 global-time test: 23.5% of interactions on new items) is unaddressed by any model so far.

## 2. Model tournament ladder (grouped by company purpose)

### A. Retrieval baselines — "what's the cheap, robust floor?"
| Model | Status |
|---|---|
| popularity | done (floor anchor) |
| co-occurrence / item-kNN | done (5k/10k; 20k=OOM) |
| ALS / MF | done; **f64 = current best**; f128+ external |
| hybrid candidate generation (RRF) | done (f16); **f64-hybrid pending/external** |

### B. Sequential models — "does reading *order* add signal over collaborative?"
| Model | Status |
|---|---|
| canonical **full-position SASRec** | **not yet** (D3B was last-position/CPU) — the fair re-test |
| BERT4Rec / GRU4Rec | optional, **only if** they answer a distinct question (BERT4Rec = bidirectional/cloze; GRU4Rec = recurrent vs attention) — otherwise skip as decoration |

### C. Graph / content models — "can we cover the cold-start gap nothing else does?"
| Model | Status |
|---|---|
| LightGCN / graph CF retrieval | T3 — high-order CF on the real user-item graph |
| **content / title / tag / metadata cold-start channel** | **the one real gap** — baselines score ~0 on new items (23.5% of test) |
| two-tower | **only if** it fills the retrieval/cold-start gap (content tower) — not as another collaborative model |

### D. Reranking / governance — "ship-safety and catalog health" (the differentiator)
| Layer | Status |
|---|---|
| diversity-aware reranker (MMR) | built (smoke); port to domain |
| creator / catalog-health reranker | not yet (domain has real authors) |
| position-bias correction (IPS + PAL) | built (smoke + simulator); port to domain |
| OPE layer (IPS/SNIPS/DM/DR) | built (smoke + simulator, incl. SASRec-π); port to domain on ALS f64 |

## 3. Evaluation logic (per model type)
| Model type | Product/system question | Offline metric | Expected advantage | Failure mode | Kill when | External compute? | Essential vs T3 |
|---|---|---|---|---|---|---|---|
| Popularity/co-occ/ALS | cheap robust floor | Recall@K/NDCG | strong collaborative recall | head-bias, no cold-start | — (they're the floor) | f128+/20k=ext | essential (done) |
| Hybrid | does blending beat best single? | Recall@K/NDCG | head precision lift | dilutes deep tail (seen: f16-hybrid < f64 ALS deep) | if no lift over f64 ALS | f64-hybrid=ext | optional |
| Canonical SASRec | does order beat MF? | Recall@K/NDCG vs f64 ALS | sequence signal | needs scale/epochs; lost when crippled | if full-position+GPU still < f64 ALS | **yes (GPU)** | optional but credibility-important |
| BERT4Rec/GRU4Rec | distinct seq inductive bias? | Recall@K/NDCG | dense-sequence gains | ≈ SASRec, decoration risk | if no question beyond SASRec | yes | T3/optional |
| LightGCN | multi-hop CF > MF? | Recall@K | neighborhood propagation | over-smoothing, cost | if ≤ f64 ALS | yes | T3 |
| **Content cold-start channel** | recommend **new** items? | **cold-start Recall on new-item golds** (global-time test) | the only thing that scores >0 on new items | weak content features | if no cold-start recall lift | no (CPU) | **essential for the cold-start story** |
| Two-tower | scalable retrieval + content cold-start | Recall@K + cold-start | ANN scale + content tower | needs negatives/scale | if no gap filled vs ALS+content | maybe GPU | optional |
| Diversity/creator reranker | catalog health vs relevance | Gini/entropy/long-tail @ fixed Recall cost | protects creators | over-diversify → relevance loss | if relevance cost > guardrail | no | **differentiator** |
| Position-bias (IPS/PAL) | de-bias exposure | examination-corrected metrics | unbiased relevance | variance (IPS) | — (governance) | no | **differentiator** |
| OPE (IPS/SNIPS/DR) | trustworthy promote/hold | bias/variance vs known truth | offline decision validity | needs known propensities | — (governance) | no | **differentiator** |

## 4. Realistic company decision rule
1. **Offline improvement is a *gate*, not final truth.** A model must beat the floor (ALS f64) on Recall@K/NDCG, or *complement* it (e.g., cover cold-start, or lift diversity at acceptable relevance cost).
2. **A/B test is the final truth** — offline gains are necessary, not sufficient; promote only what wins (or holds neutral on) the online guardrails.
3. **Complexity must be justified:** a model that ties f64 ALS but is 10× heavier/slower is *rejected*. Marginal lift must exceed the added serving/maintenance cost.
4. **Guardrails (any breach → hold):** latency/serving cost, catalog **coverage**, **diversity / creator concentration** (Gini), **cold-start** exposure for new items/authors, user **fatigue/novelty** decay. (PulseSignal's guardrail-first SHIP/HOLD pattern applies.)
5. **Kill discipline:** if a model doesn't answer a distinct product question or doesn't clear the gate, it's documented as a negative result (like D3B SASRec) and dropped — no decoration.

## 5. Recommended next gate
**Primary recommendation: (B) move to the domain OPE / position-bias / creator-health layer on ALS f64 — now — and queue (A) a canonical full-position SASRec as an external GPU run in parallel.**

**Why B first:**
- **It's the actual differentiator.** PulseDiscover's edge is *trustworthy evaluation + catalog governance*, not which generator wins. That layer is fully **CPU-feasible on the domain today** and **generator-agnostic** (works on whatever the floor is).
- **ALS f64 is a legitimate, defensible current-best floor** to run OPE/governance against — we don't need SASRec to win first.
- **It advances the headline** (the gold PRD's centerpiece) and reuses already-built code (simulator, IPS/SNIPS/DR, PAL, exposure audit) ported from the smoke.
- **It doesn't block on GPU.** A fair canonical SASRec is GPU-bound (full-position ≈30× the CPU last-position cost) → it belongs in the external pack, run in parallel, not as the in-sandbox next step.

**Why not A as the immediate in-sandbox gate:** a *fair* SASRec needs full-position training + bigger model + 100+ epochs → not honestly doable on this CPU/45s sandbox. Forcing another crippled CPU attempt would repeat D3B. So A is **queued external** (it remains credibility-important and should run — just not as the in-sandbox blocker).

**Why C is second, not first:** the **content cold-start channel (C)** is the single highest-value *new model* gap (nothing scores >0 on the 23.5% new-item test), and it's CPU-feasible — so it's the **next in-sandbox build after B**. It's second because the OPE/governance layer is the project's defining differentiator and is generator-agnostic, whereas the cold-start channel is an additive capability.

**Sequence:** **B (domain OPE/position-bias/creator-health on ALS f64, in-sandbox) → C (content cold-start channel, in-sandbox) → A (canonical full-position SASRec, external GPU) → optional LightGCN/two-tower/BERT4Rec (T3, only if they answer a distinct question).**

*Stop after this plan. No model trained; awaiting approval of the next gate (recommended: B).*
