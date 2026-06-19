# 49 — PulseDiscover RiskFrame-Gold Audit

*Audits PulseDiscover against the RiskFrame Gold Standard v4 (docs/00). **Verdict up front: PulseDiscover is NOT RiskFrame-gold. It is a strong package + company-realism checkpoint with a real technical-depth gap.** Documentation cannot close that gap (Rule 1); the missing depth is build-required.*

---

## 1. Axis scores (1–10)
| # | Axis | Score | Current evidence | Gap | Required fix | Type |
|---|---|---|---|---|---|---|
| 1 | Product thesis clarity | **9** | G2, G8 abstract — discovery-vs-relevance-vs-health framing | minor | none material | doc ✅ |
| 2 | Market / JD relevance | **7** | RecSys role fit implicit (retrieval/OPE/A-B) | no explicit JD-mapping artifact | add a short JD/role-mapping note | doc |
| 3 | **Technique tournament depth** | **5** | ALS ladder + content-hybrid + **one failed** SASRec | no *competitive* deep model; no two-tower/LightGCN/GRU4Rec built | **G11 sequence tournament + technique expansion** | **build** |
| 4 | **Deep Defense Kernel** | **4** | techniques used but no per-technique first-principles defense cards | no defense cards (ALS/SASRec/IPS/SNIPS/DR/PAL/MMR/ANN/neg-sampling) | **build defense-card set grounded in real runs** | **both (build-anchored)** |
| 5 | Product Reasoning Kernel | **8** | G2 product, G4 rollout, ship/hold/kill | could tie reasoning to economics harder | minor | doc ✅ |
| 6 | Data realism / feature engineering | **7** | real Goodreads, global-time split, prevalence; features = TF-IDF/author/series | feature engineering is light | add richer features in tournament builds | both |
| 7 | Synthetic realism audit | **6** | simulators labeled SYNTHETIC (OPE/feedback/position-bias) | no formal realism audit of the simulators' assumptions | write a simulator-realism audit | doc |
| 8 | Evidence honesty | **9** | tagged claims, documented negatives, claim-boundary tables | exemplary | maintain | doc ✅ |
| 9 | Evaluation validity | **8** | LLO + global-time, R@K/NDCG, bootstrap CIs, segments, G6 scale | full-pop CIs not all computed | add CIs at scale where cheap | both |
| 10 | Decision economics | **6** | G1 cost/infra, G2 monetization caution | no quantified **negative-cost chain** per technique | **build negative-cost chain** | doc (build-informed) |
| 11 | Industry-pattern awareness | **7** | two-stage retrieval, OPE, PBM, RRF | patterns used, not catalogued | add pattern catalogue | doc |
| 12 | Hairy failure modes | **8** | G3 risk register + G1 failure modes | strong | maintain | doc ✅ |
| 13 | Achievement moments | **6** | exist (SASRec neg, 11×, positivity collapse) but not packaged | not surfaced as named "moments" | package achievement moments | doc |
| 14 | Tradeoff density | **8** | C2/O1b/H2/G1/G2 tradeoffs throughout | strong | maintain | doc ✅ |
| 15 | Interview dominance | **8** | DEF1 + dossier + scripts | depends on closing #3/#4 to be airtight | refresh after builds | doc |

**Mean ≈ 6.9 / 10.** Binding-axis failures: **#3 Technique tournament depth (5)** and **#4 Deep Defense Kernel (4)** — both **build-required** and both below the 9+ pass bar.

## 2. Why it fails the standard
- **Rule 1 (docs can't close technical gaps):** the two failing axes are technical-depth axes; no amount of writing closes them. They need built, measured models and run-anchored defense cards.
- **Rule 2 (no T3 without real evidence):** there is **no competitive advanced model** — SASRec is an honest negative and G5 is a *documented deferral*. That satisfies honesty, but it means **T3/model-depth is open**, not achieved.
- **Chain break:** for the advanced-model link, the chain stops at "first-principles → product decision → (artifact MISSING)". The cold/warm/OPE chains are complete; the deep-model chain is not.

## 3. Status classification (permitted labels only)
- ✅ **Package checkpoint complete**
- ✅ **Ready for next build gate**
- ⚠️ **RiskFrame-gold incomplete**
- ⚠️ **T3/model-depth open**

**Not** claimed (pass condition unmet): BeastMax complete / T3 complete / gold-standard complete / fully maxed.

## 4. Minimum fix plan (to reach RiskFrame-gold) — *plan only, not executed here*

**G11 — Sequence Model Tournament (BUILD).**
- Build **GRU4Rec** and a **canonical SASRec** (full-position, proper config); compare head-to-head against **ALS f64** on the same LLO sample, with tags/N/CI/source/reproducibility.
- Acceptance: a real comparative result (win *or* clean, properly-trained negative) — closes axis #3 and the deep-model chain link.

**Technique Tournament expansion (BUILD + CLASSIFY).**
- Cover: two-tower, LightGCN, LambdaRank, PAL, contextual bandits, MMR, IPS/SNIPS/DR.
- Classify each: **Use-now / Baseline / V2 / T3 / Exclude** with a one-line reason (no technique museum — Rule 4).

**Deep Defense Kernel (BUILD-ANCHORED CARDS).**
- One defense card each for: ALS, SASRec, GRU4Rec, two-tower, IPS, SNIPS, DR, PAL, MMR, negative sampling, ANN/top-K.
- Each card: first-principles why → when it wins/breaks → the decision it serves → grounded in an actual run where applicable.

**Business Negative-Cost Chain (DOC, build-informed).**
- Per technique: technique → bad decision it prevents → naive outcome → negative cost → senior decision. (e.g., position-bias → "cold lane looks dead" → kill discovery → lost catalog freshness → keep lane, correct CTR.)

**Achievement Moments (DOC).**
- Package the five: SASRec honest negative · OPE positivity collapse · position-bias 11× under-credit · reranking failed fairness within guardrail · G6 warm-loss drift toward edge.

### Priority order
1. **G11 sequence tournament** (the binding gap — unlocks #3 and the deep-model chain).
2. **Technique tournament classification** (frames #3, prevents museum).
3. **Deep Defense Kernel** (closes #4, anchored on G11 + existing runs).
4. **Negative-cost chain** + **Achievement moments** (lift #10/#13, mostly writing once builds exist).

## 5. Stop
Audit + minimum fix plan only. **Fixes not written. No new deck/closeout. No completion claimed.** PulseDiscover is **RiskFrame-gold incomplete / T3-open / ready for the next build gate (G11).**

**STOP — awaiting direction on the fix plan (recommended next: G11 sequence model tournament build).**
