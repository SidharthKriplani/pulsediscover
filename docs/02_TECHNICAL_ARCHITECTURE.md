# 02 — Technical Architecture: PulseDiscover BeastMax

*End-to-end architecture, components, data flow, and artifact flow for the BeastMax-core build. Status tags: `[BUILT][SYNTHETIC]` (exists in `pulserank_platform`, e-commerce), `[PORT]` (port existing code to real data), `[BUILD-NOW]`, `[BUILD-IF-FEASIBLE]`, `[V2/T3]`.*

## System overview
```
                 ┌──────────────────────────────────────────────────────────┐
 real dataset →  │ INGEST (temporal split, leakage guard)         [BUILD-NOW]│
 (MovieLens/     └──────────────┬───────────────────────────────────────────┘
  Goodreads)                    │ sequences, interactions, item metadata
                                ▼
        ┌───────────────────────────────────────────────────────────────────┐
        │ RETRIEVAL                                                            │
        │  popularity / co-occurrence / ALS          [PORT]                    │
        │  two-tower (user/item towers, FAISS ANN)   [BUILD-NOW]               │
        │  content cold-start channel                [BUILD-NOW]               │
        └──────────────┬────────────────────────────────────────────────────┘
                       │ top-K candidates (+ retrieval scores)
                       ▼
        ┌───────────────────────────────────────────────────────────────────┐
        │ RANKING                                                              │
        │  SASRec (sequential transformer)           [BUILD-NOW, primary]      │
        │  LambdaRank (LightGBM, listwise)           [PORT]                     │
        └──────────────┬────────────────────────────────────────────────────┘
                       │ per-candidate relevance
                       ▼
        ┌───────────────────────────────────────────────────────────────────┐
        │ DEBIAS  IPS reweighting [BUILD-NOW] · PAL tower [BUILD-IF-FEASIBLE]  │
        └──────────────┬────────────────────────────────────────────────────┘
                       │ debiased relevance → final ranking
                       ▼
        ┌───────────────────────────────────────────────────────────────────┐
        │ EVALUATION                                                           │
        │  Simulator (known μ, PBM curve, ε bucket)  [BUILD-NOW]               │
        │  IPS / SNIPS / DR vs ground truth + CIs    [BUILD-NOW]               │
        │  Feedback-loop / catalog-health audit      [BUILD-NOW/IF-FEASIBLE]   │
        └──────────────┬────────────────────────────────────────────────────┘
                       │ evidence JSONs + plots + cards
                       ▼
            PRD update [BUILD-NOW]  →  Defense PDF [LAST]
```

## Components and responsibilities
| Component | Module (planned) | Status | Responsibility |
|---|---|---|---|
| Ingest | `src/data/` | `[BUILD-NOW]` | load dataset, temporal split, leakage guard, `data_manifest.json` |
| Retrieval baselines | `src/retrieval/` | `[PORT]` | popularity, co-occurrence, ALS (reuse PulseRank) |
| Two-tower | `src/models/two_tower.py` | `[BUILD-NOW]` | DL retrieval, in-batch negatives + logQ, FAISS index |
| Cold-start channel | `src/retrieval/coldstart.py` | `[BUILD-NOW]` | content embedding + reserved exploration slots |
| SASRec | `src/models/sasrec.py` | `[BUILD-NOW]` | causal self-attention next-item ranker |
| LambdaRank | `src/models/lambdarank.py` | `[PORT]` | LightGBM listwise ranker on real features |
| Simulator | `src/sim/` | `[BUILD-NOW]` | known μ, PBM examination curve, ε exploration, propensity logging |
| OPE | `src/ope/` | `[BUILD-NOW]` | IPS, SNIPS, DR; bias/variance vs ground truth |
| Debias | `src/debias/` | `[BUILD-NOW]` (IPS) / `[BUILD-IF-FEASIBLE]` (PAL) | position-bias correction |
| Feedback-loop audit | `src/audit/` | `[BUILD-NOW]`/`[IF-FEASIBLE]` (Shapley) | exposure concentration, ratchet, decomposition |
| CIs | `src/eval/ci.py` | `[BUILD-NOW]` | bootstrap/Wilson intervals on all metrics |
| Evidence/report | `src/eval/report.py` | `[BUILD-NOW]` | JSONs, plots, model/decision cards |

## Data flow → artifact flow (what each phase emits)
1. **Ingest** → `data_manifest.json` (sizes, split, seed).
2. **Retrieval** → `retrieval_baselines.json`, `two_tower_report.json`, `recall_at_k.png`.
3. **Ranking** → `sasrec_report.json`, `lambdarank_report.json`.
4. **Simulator** → `simulator_manifest.json` + logged events (propensity, position, exploration, reward).
5. **OPE** → `ope_comparison.json` (IPS/SNIPS/DR vs ground truth, bias, variance, CIs), `ope_variance.png`.
6. **Debias** → `position_bias_report.json`, `item_movement.png`, `examination_curve.png`.
7. **Audit** → `feedback_loop_report.json`, `exposure_gini_over_rounds.png`, `before_after_exposure.png`.
8. **CIs** → `ci_summary.json`.
9. **PRD update** consumes all of the above; **Defense PDF** consumes the PRD + plots.

## Key architectural rules (theory-driven, from the PRD's Product Reasoning Kernel)
- **Propensity is logged at decision time** (simulator) → this is what makes IPS/DR valid; it is the load-bearing logging requirement.
- **Position enters additively** (PAL) or via **loss reweighting** (IPS) — never as a concatenated deep-net feature.
- **`display_rank` is never a ranker feature** (leakage control; already enforced in PulseRank).
- **Embedding version pinned to model version** (avoids index/model staleness mismatch).
- **Every metric carries N + CI**; every artifact carries a seed + version stamp.

## Reproducibility
Single config with a fixed seed; `seed_version_manifest.json` records dataset version, code commit, library versions, seed. Re-running the pipeline reproduces every number in the evidence ledger.
