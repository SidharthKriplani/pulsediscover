# 14 — RiskFrame Gold v4 Template Audit (PulseDiscover)

*Audit only — no experiments, models, or PDF. Assesses the whole PulseDiscover project (PRD + evidence + plans + 8 evidence JSONs) against RiskFrame Gold v4. Honest scoring; residual gaps flagged and bucketed by milestone.*

## 1. Gold-axis scores (1–10) + status

| # | Axis | Score | Status | Note |
|---|---|---|---|---|
| 1 | Product thesis clarity | 9 | PASS | governs the promote/hold decision; sharp one-line |
| 2 | Market / JD relevance | 9 | PASS | RecSys + OPE pairing is the scarce hireable signal |
| 3 | Technique tournament depth | 9 | PASS | retrieval/ranking/OPE/debiasing + full T3 DL roadmap, all labeled |
| 4 | Deep Defense Kernel | 9 | PASS | full derivations for IPS/SNIPS/DR/SASRec/LambdaMART/ALS/two-tower/PAL; (no card for cold-start channel / LightGCN — not built) |
| 5 | Product Reasoning Kernel | 9 | PASS | theory→logging→eval→business traced; propensity-logging is the spine |
| 6 | **Data realism / feature eng.** | **7** | **WEAK** | **MovieLens smoke only; no domain dataset; cold-start channel not built; no real creators** |
| 7 | Synthetic realism audit | 9 | PASS | simulator audit explicit (can/can't-prove, circular-trap contrast) |
| 8 | Evidence honesty | 9 | PASS | every built number tagged + N + CI + source; overclaims patched; honest negatives kept |
| 9 | Evaluation validity | 8 | WEAK | DR/leakage/CIs strong; but **OPE π and truth are ALS-based, not SASRec** |
| 10 | Decision economics (₹ chain) | 8 | WEAK | negative-cost chain present but qualitative; ₹ illustrative, not domain-grounded |
| 11 | Industry-pattern awareness | 8 | WEAK | correct pattern phrasing, but thin/generic |
| 12 | Hairy failure modes | 9 | PASS | amplification, positivity, staleness, cold-start starvation, peeking |
| 13 | Achievement moments | 9 | PASS | 4 real, build-grounded (SASRec>ALS, DR robustness, IPS MSE-vs-rank, ratchet caveat) |
| 14 | Tradeoff density | 9 | PASS | tradeoffs at every layer |
| 15 | Interview dominance | 8 | WEAK | strong defense bank; weakened only by smoke + ALS-π until domain/B done |

**Average ≈ 8.5.** Gold threshold (9+) is met on the *document and methodology*; the sub-9 axes (6, 9, 10, 11, 15) are all gated on the **domain build** and the **SASRec-π** step — not on doc quality.

## 2. Mandatory-section status

PASS: thesis · buyer · why-now · one-insight · Layer 0 · component map · data flow · product-reasoning kernel · technique tournament · deep defense kernel · synthetic-realism audit · operational failure modes · senior-vs-naive · achievement moments · tradeoffs · production behavior · build path · no-overclaim · resume · convergence/acceptance.
WEAK: **data layer** (smoke only) · **evidence ledger** (vendored e-commerce rows lack CIs) · **evaluation layer** (ALS-π) · **negative-cost chain** (qualitative) · **industry-pattern** (generic) · **visual/demo plan** (lists unbuilt two-tower artifact) · **free-vs-paid** ("proves" row lists unbuilt two-tower/cold-start).
MISSING: none structurally — all 28 sections exist.

## 3. Residual overclaims / stale tags (specific)
1. **§9.1 tournament — "Content-only (cold-start channel) `[BUILT][SYNTHETIC]`"** is misleading: the Recall@100 0.647 is the **e-commerce** content channel (vendored), and Component Map **C2 (cold-start channel) is `[VISION/BUILD-TASK]`**. The two contradict. → retag the §9.1 row `[BUILT][SYNTHETIC — e-commerce]`; the fiction cold-start channel is NOT built.
2. **§25 free-version "What it proves" lists "SASRec/two-tower correctness … cold-start"** — two-tower and the cold-start channel are not built. → soften to "SASRec correctness (built); two-tower/cold-start when built."
3. **§22 visual plan** lists a retrieval table including **two-tower** and implies all artifacts exist. → split into *produced* (recall bar, IPS/SNIPS/DR variance, examination curve, PAL-vs-IPS item-movement, exposure/feedback plots — all in `outputs/plots/`) vs *planned* (two-tower).
4. **§16 north-star** says "creator-exposure equity over **30 days**" — the feedback audit ran **10 rounds** (not 30 days) and MovieLens has **no creators** (item-level + genre proxy only). → keep as aspirational metric but caveat "creator-level pending domain dataset."
5. **"Shapley" (4 mentions)** — now consistently qualified as **V2 / not built** (Phase 7 built Gini/entropy/long-tail). Verified no Shapley-as-built remains. OK; keep monitoring.

## 4. Targeted audits (the eight you asked for)
- **MovieLens smoke vs domain headline:** smoke is honestly labeled everywhere; the **domain headline does not exist** — the single biggest gap (axis 6). Goodreads/Amazon not ingested (blocked on user-provided data + likely GPU).
- **ALS-based π vs SASRec-based π:** Phase 5 OPE validates estimators with **π = softmax(ALS)** and truth = sigmoid(ALS); the **actual headline model (SASRec) is not yet the evaluated policy.** Now explicitly noted in §10.3 + ledger. This is Option B.
- **T3 roadmap not built:** Two-tower, LightGCN, GraphSAGE/PinSage/PyG, Wide&Deep, DeepFM/xDeepFM, DIN/DIEN, BERT4Rec, DLRM, Transformers4Rec are all labeled Use-now/V2/T3/Discuss/Exclude and tagged `[ROADMAP — NOT BUILT]`. Clean. (Two-tower has a conceptual defense card; LightGCN has none — acceptable since unbuilt.)
- **Synthetic simulator boundaries:** strong. Known propensities, candidate-pool (not full-catalog) positivity, μ≠truth≠π, oracle `true_rel` quarantined from estimators, "mechanism not online lift" stated. Best-audited part of the project.
- **Evidence ledger completeness:** consolidated `docs/12_EVIDENCE_LEDGER.md` covers Phases 2–7 with tag/N/CI/source/safe-line. Gap: **vendored e-commerce rows lack CIs** (provenance only — acceptable, labeled).
- **Business negative-cost chain:** present per technique (§15) but **qualitative**; ₹ figures illustrative, not modeled on a domain. WEAK — fine for a portfolio, but the thinnest "premium" section.
- **Technique tournament depth:** strong and now the deepest section (core + T3 roadmap with data-needs / when-it-wins / why-deferred).
- **Interview defense readiness:** defense bank + derivations are PASS; the only answers that currently carry a caveat are "what's your domain result?" (smoke) and "does DR evaluate your real model?" (ALS-π until Option B). Both honestly handled.

## 5. Gaps bucketed by milestone

**Must-fix before Option B (SASRec-π OPE):** *none blocking.* B is a clean in-sandbox increment. Recommended (cheap, optional): retag the §9.1 cold-start row (#3.1) so the doc is self-consistent before adding more.

**Must-fix before domain build (A):** (a) confirmed dataset acquisition (user supplies Goodreads-genre / Amazon Books — I can't fetch); (b) GPU plan for SASRec at domain scale; (c) define domain creator/series fields so creator-level concentration becomes real; (d) re-run the spine and **promote domain numbers to headline, demote MovieLens to smoke** in the PRD.

**Must-fix before T3 (C):** (a) evidence spine must already be on domain data; (b) richer features/graph + GPU available; (c) add a real **defense card + evidence row** for any T3 model actually built; (d) never claim before the evidence JSON exists.

**Must-fix before final defense PDF (Phase 10):** (a) clear all §3 residual overclaims (cold-start tag, free-version "proves", visual produced-vs-planned, 30-day/creator caveat); (b) every PDF figure cites an `outputs/` artifact; (c) resolve or clearly future-label SASRec-π; (d) ideally build the PDF **after** the domain headline (A) so it isn't smoke-only — else label it explicitly as a smoke + simulator methodology demo.

## 6. Recommendation (do not execute)
**Proceed with Option B (SASRec-based π through the OPE harness) next.** It has **no hard blockers** in this audit, is fully in-sandbox, and removes the most pointed evaluation caveat (axis 9 / ALS-π) by making DR evaluate the *actual* headline model. Apply the cheap §3 doc-consistency fixes opportunistically (they don't block B). Then **A (domain headline)** when you provide a dataset + GPU — that is what lifts axes 6/10/11/15 to 9. Defer **C (T3 models)** and the **defense PDF** until after A, so the PDF is domain-grounded.

*Sequence: B → (doc fixes) → A → C → defense PDF. No step executed; awaiting your go.*
