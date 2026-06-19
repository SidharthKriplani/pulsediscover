# PulseDiscovery — RiskFrame-Gold Ruthless Audit

*PulseDiscovery only. V1 is terminal `gold_candidate` 8.9 (not reopened). V2 (G22–G25 accepted, G26 built-but-blocked-pending-this-audit). Verdict: **PATCH — strong dossier with a real component-coherence + OPE-logging gap; run G26A consolidation before G26B mitigation.** Evidence JSON: `outputs/evidence/pulsediscovery_riskframe_gold_audit.json`.*

> **Canonical control-doc map (the `00/04/06` names in the brief are generic RiskFrame labels; this repo's live files are):** Control tower = `docs/78_CONTROL_TOWER.md`; Evidence ledger = `docs/12_EVIDENCE_LEDGER.md`; Claim boundary = `docs/69_G18_CLAIM_BOUNDARY_LOCK.md`. All three are updated with this audit. No stray `00/04/06` files were created (that would fragment the control plane).

---

## A. North-Star Identity

**"PulseDiscovery helps ranking/recommendation teams decide which retrieval, serving, fallback, and exposure-governance policy should ship by catching offline-metric bias, candidate-coverage gaps, latency–quality tradeoffs, cold-start degradation, and catalog concentration before they damage user experience or catalog health."**

- **Decision governed:** which candidate-generation + serving + fallback + (now) reranking policy to ship on a Goodreads-scale fantasy/paranormal catalog, and which ones *not* to ship — under an explicit, honesty-disciplined offline protocol.
- **Who uses it:** recommender DS (protocol/metric validity), ranking ML engineer (retrieval/serving tradeoffs), search/discovery & marketplace PM (catalog health, cold-start UX), platform/serving eng (latency, fallback, monitoring).
- **Why wrongness is expensive:** a recall-looking win that worsens catalog health, an ANN setting that silently shreds the candidate pool, or a popularity fallback that quietly becomes the dominant policy all degrade discovery and creator opportunity at scale before any dashboard notices.
- **Why not a toy notebook:** real 2.56M-interaction k-core dataset, temporal split, a tuned MF floor with *documented deep-model negatives*, a production-shaped FastAPI+FAISS service with fallback/monitoring/load-test, and quantified exposure-governance + mitigation frontiers — each gated, claim-bounded, and provenance-tagged.
- **Why V1 is `gold_candidate`, not `gold_complete`:** the canonical G20 SASRec convergence headline is `plot_reconstructed` (direct v3 JSON + secondary metrics never recovered after a lost Colab session). The decision (MF beats canonical SASRec on this catalog) is artifact-backed; the *packaging* is one artifact short of clean.
- **Why V2 doesn't upgrade V1:** V2 builds **new** evidence under **new** boundaries on the **served** `c2_als.pkl` model; it never re-litigates V1's conclusions or its locked status.

**Forbidden identity (explicitly NOT):** production recommender serving real users · online-lift system · fairness-certified recommender · solved-cold-start system · solved-long-tail-discovery system · proof that SASRec never works.

---

## B. Current Component Map

| Component | Role | Input | Output | Tag | Evidence | Why it exists | Claim enabled | Claim NOT enabled |
|---|---|---|---|---|---|---|---|---|
| Goodreads k-core dataset | Domain substrate | raw GR shelves | 2.56M int / 94k u / 42k i | [BUILT — real/offline data] | `d2_stats.json` | Real scale, temporal | "real 2.56M-interaction domain" | "production traffic" |
| V1 ALS warm floor | Warm retrieval baseline | train interactions | ALS f64 factors | [OFFLINE EVAL] | `d3aplus_als_tune.json` (R@20 0.0846) | Strong simple floor | "MF is the warm floor under this protocol" | "best possible recall" |
| V1 SASRec convergence | Method-depth closure | train sequences | R@20 0.065 @~95 ep | [BOUNDARY-LIMITED] (`plot_reconstructed`) | `g20_..._v3.json` | Test if depth beats MF | "canonical SASRec converged & still lost" | "SASRec never works" |
| V1 candidate-coverage diagnosis | Bottleneck finding | retriever recall | coverage = binding constraint | [OFFLINE EVAL] | `12_EVIDENCE_LEDGER` G12-B/G13 | Locate leverage | "coverage > depth on this catalog" | "online impact" |
| G22 FAISS frontier | Latency–quality tradeoff | ALS item factors | FlatIP/HNSW/IVF frontier | [OFFLINE EVAL] | `g22_..._report.json` | Serving retrieval choice | "FlatIP lossless ~47% faster; HNSW ef64 near-lossless ~2.4×" | "FAISS improves quality" |
| G23 FastAPI serving | Production-shaped API | model+index | `/recommend` etc. | [LOCAL LOAD TEST] | `g23_..._report.json` | Serving realism | "production-like API, load-tested" | "deployed / real users" |
| G23 fallback tree | Graceful degradation | any request | always-nonempty slate | [LOCAL LOAD TEST] | `g23_..._report.json` | Never crash/empty | "0% empty across failure modes" | "fallback quality is good" |
| G23 monitoring hooks | Observability | request stream | metrics/Prometheus | [LOCAL LOAD TEST] | `g23_..._report.json` | Ops realism | "latency/error/fallback monitoring hooks" | "production SLO" |
| G24 cohort builder | Cold/sparse cohorts | train+factors | unknown/sparse/low/warm | [BUILT — real/offline data] | `g24_..._report.json` | Segment fallback | "cohorted fallback eval" | "real new-user behaviour" |
| G24 fallback policy eval | Fallback quality | cohorts×policies | coverage/personalization collapse | [OFFLINE EVAL] | `g24_..._report.json` | Quality not existence | "popularity fallback collapses coverage ~59×" | "cold-start solved" |
| G24 serving integration | End-to-end check | live service | 7/7 cases nonempty | [LOCAL LOAD TEST] | `g24_..._report.json` | Wire eval↔serving | "fallback paths verified in service" | "production behaviour" |
| G25 exposure governance | Concentration audit | policy exposure | Gini/coverage/tail lift | [OFFLINE EVAL] | `g25_..._report.json` | Catalog health | "exposure concentration measured" | "fairness certified" |
| G25 HNSW-vs-exact neutrality | Approx safety | served exact/HNSW | Gini 0.986≈0.986 | [OFFLINE EVAL] | `g25_..._report.json` | De-risk ANN | "HNSW exposure-neutral" | "HNSW improves diversity" |
| G26 exposure-aware reranking | Mitigation frontier | warm ALS pool | relevance/exposure frontier | [BUILT — real/offline data] **(BLOCKED pending audit)** | `g26_..._report.json` | Intervene not audit | (pending unblock) "head-cap Pareto-dominant on warm cohort" | "long-tail discovery solved" |
| LambdaMART/LTR ranker | Learned reranker | candidate features | re-scored slate | [PARKED] | G12 notes | Specced, not the V2 reranker | — | "learned reranker shipped in V2" |
| Logged propensities / OPE | Online decisioning prereq | serving logs | IPS/SNIPS/DR-valid logs | [BUILD-TASK] | none | Needed for OPE | — | "valid off-policy estimate" |
| Two-tower / LightGCN lanes | Alt retrieval families | interactions | embeddings | [VISION] | none | Coverage of families | — | "tested two-tower/LightGCN" |

**No untagged component survives.** The two [BUILD-TASK]/[VISION]/[PARKED] rows (OPE logging, LTR, alt-retrieval) are exactly the disconnected-component risk surfaced below.

---

## C. Data & Protocol Audit

### C1. Raw interaction data
Source Goodreads fantasy/paranormal shelves, iterative k-core (users≥5 & items≥10, 10 iters). **2,557,588** interactions · **94,185** users · **41,961** items · 10,389 creators · 25,042 series. Interaction = a shelving/rating event (ratings 0–5; 3.1% rating-0). Timestamps **present** (2001-07-01 → 2017-11-03). Split = **global temporal** (t1/t2 cutoffs): train 2,046,070 · val 255,759 · test 255,759 · 42,683 held-out users · 5,000 eval sample. **Leakage risk:** temporal split mitigates user-future leakage; G20 had a leakage bug (early-stop on eval set) — **caught and fixed v3**. **Duplicate risk:** k-core + per-user dedup applied. **ID-dtype risk:** **REAL and previously bit us** — `book_id` is a *string* in the ALS index; an int/str mismatch silently zeroed all lookups in G24 (caught & fixed). This is now a standing failure-mode check.

### C2. V1 protocol
ALS f64 warm floor R@20 **0.0846** (d3aplus tuning, `[OFFLINE EVAL]`). Canonical full-softmax SASRec converged (~95 ep) at R@20 **0.065** < ALS — `clearly_loses`, ratio 0.768. **Not `gold_complete`** because that headline is `[PLOT_RECONSTRUCTED]`: the direct v3 JSON + secondary metrics (NDCG, per-epoch) were lost with the Colab session and never recovered. Direct evidence that *does* exist: the v3 convergence plot + the rejected stale checkpoint. **Missing metrics (NDCG, exact secondary):** do **not** change the decision (the gap is large and monotone), but they block clean packaging.

### C3. V2 protocol
V2 serves **`c2_als.pkl`** (ALS f64, nU 91,732 / nI 40,541) — a **different build** from V1's `d3aplus`. **Consequence:** V2 absolute recall (G22 exact R@20 **0.0375**; G24/G26 baseline ~0.0175–0.038 under full-catalog single-held-out-gold) is **`[NOT_COMPARABLE_ACROSS_PROTOCOLS]`** to V1's 0.0846 — flagged in every V2 artifact. Denominators: **G22** n_query 2,000, n_items 40,541, dim 64, brute p95 0.723 ms. **G23** load test warm 1,500 / cold 500 / scale 500, monitoring 3,802 requests. **G24** 10,000 users (warm 7,360 / low 817 / sparse 761 / unknown 1,062). **G25** 10,000 users, **K=20**, catalog 40,541, ~200k exposure slots/policy; HNSW-vs-exact on 4,000 warm × K20 = 80,000 slots each (matched). **G26** 4,000 warm sample, pool 200, K=20.

### C4. Item cold-start phrasing
Locked as **"held-out gold items unseen in train under this offline split" (57.2%)** — explicitly **not** real production new-catalog cold-start.

### C5. Headline-metric register
| Claim | Number | Tag | N | Uncertainty | Source | Reproducible? | Protocol | Interview line | Boundary |
|---|---|---|---|---|---|---|---|---|---|
| ALS warm floor | R@20 0.0846 | REAL_OFFLINE | 5k eval | no CI | `d3aplus_als_tune.json` | yes | d3aplus tuning | "MF is the warm floor here" | not best-possible |
| SASRec converged & lost | R@20 0.065 | PLOT_RECONSTRUCTED | 5k | no CI | `g20_v3.json` | partial | canonical full-softmax | "depth converged, still lost" | not "never works" |
| FlatIP exact speedup | ~47% p95 ↓, lossless | REAL_OFFLINE | 2k q | overlap 1.0 | `g22.json` | yes | served factors | "lossless, faster" | not quality gain |
| HNSW ef64 | overlap 0.992, ~2.4× | REAL_OFFLINE | 2k q | overlap-based | `g22.json` | yes | served factors | "near-lossless scale option" | not default |
| Warm API p95 | 3.34 ms | LOCAL_LOAD_TEST | 1.5k | in-proc | `g23.json` | yes | TestClient | "production-like, load-tested" | not prod p95 |
| Empty-response rate | 0% | LOCAL_LOAD_TEST | 3,802 | exact | `g23.json` | yes | failure modes | "never empty" | not fallback quality |
| Popularity coverage collapse | ~59× (3532→60) | SERVED_MODEL | 10k | exact | `g24.json` | yes | served c2 | "fallback collapses catalog" | not cold-start solved |
| Exposure Gini (popularity) | 1.000 | SERVED_MODEL | 10k | exact | `g25.json` | yes | served c2 | "popularity near-degenerate" | not fairness |
| HNSW exposure-neutral | Gini 0.986≈0.986 | SERVED_MODEL | 4k | matched | `g25.json` | yes | served c2 | "ANN doesn't reshape exposure" | not diversity gain |
| Head-cap rerank | R@20 +40%, tail ~8× | SERVED_MODEL | 4k warm | no CI | `g26.json` | yes | warm pool | "mitigation Pareto win (protocol-specific)" | not long-tail solved |

**Systemic gaps:** no confidence intervals anywhere; no NDCG; recall not comparable across V1/V2; zero online/propensity evidence (`NO_ONLINE_EVIDENCE` on every behavioural claim).

---

## D. Technique Tournament

| Method | Category | Optimizes | Data needed | Strength | Failure mode | Cost/Latency | Interpretability | Decision | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| Popularity | non-personalized | hit-rate floor | counts | trivial, safe | zero personalization, degenerate exposure | ~0 | high | **Baseline only / emergency fallback** | proven (G24/G25) |
| Item-item co-occurrence | neighborhood | co-view | co-counts | cheap candidates | sparsity, popularity bias | low | high | **Baseline only** | proven (G15, conditional) |
| Content/shelf fallback | content | cold coverage | item meta | works w/o history | noisy shelves | low | med | **Use now (fallback)** | proven (G24) |
| ALS | MF | implicit recon | interactions | strong warm floor, fast | cold users, niche tail | low | med | **Use now (floor)** | proven (V1) |
| BPR-MF | MF (pairwise) | ranking AUC | interactions | ranking-aware | sampling sensitivity | low | med | **Discuss only** | not run |
| Two-tower | neural retrieval | in-batch softmax | interactions+features | scalable, feature-rich | needs scale/features | med | low | **T3 / V2 lane** | VISION |
| SASRec | sequential | next-item | sequences | captures order | sparse/noisy histories | high (GPU) | low | **Tested → lost (V1)** | proven negative |
| BERT4Rec | sequential (MLM) | masked next | sequences | bidirectional | data-hungry | high | low | **Discuss only** | not run |
| LightGCN | graph | propagated CF | bipartite graph | high-order signal | cost on CPU, tuning | high | low | **T3** | VISION |
| LambdaMART/LambdaRank | LTR | NDCG surrogate | labeled features | strong reranker | needs features+labels | med | med | **V2 (G26B)** | PARKED (specced) |
| DLRM | deep CTR | click prob | dense+sparse feats | industrial CTR | infra-heavy, overkill solo | high | low | **Exclude (overkill)** | n/a |
| FAISS FlatIP | ANN (exact) | exact IP | embeddings | lossless, fast | memory at scale | low | high | **Use now (default)** | proven (G22) |
| FAISS HNSW | ANN (graph) | approx recall | embeddings | near-lossless, fast | param sensitivity | low | med | **Use now (scale opt)** | proven (G22) |
| FAISS IVF | ANN (coarse) | approx recall | embeddings | memory-light | overlap collapse at low nprobe | low | med | **Rejected** | proven negative (G22) |
| MMR | rerank (diversity) | rel−redundancy | embeddings | intra-slate diversity | hurts recall here | low | med | **Discuss only (negative)** | proven negative (G26) |
| Constrained rerank (head-cap) | rerank | exposure cap | popularity tiers | Pareto win here | protocol-specific | ~0 | high | **V2 (G26B candidate)** | proven (G26, blocked) |
| Long-tail boost | rerank | tail exposure | popularity | clean mild lever | over-boost hurts rel | ~0 | high | **V2 (G26B)** | proven (G26) |
| xQuAD / category diversification | rerank | aspect coverage | categories+intent | query-aspect balance | needs intent model | med | med | **T3** | VISION |
| Contextual bandits | online learning | regret | live feedback+propensities | exploration | needs online loop | high | low | **T3** | VISION |
| Exploration buckets | online | exploration | traffic split | tail discovery | needs traffic | med | med | **T3** | VISION |
| IPS | OPE | unbiased value | logged propensities | unbiased | high variance, needs propensities | low | med | **BUILD-TASK (G26A)** | invalid w/o logs |
| SNIPS | OPE | self-normalized | propensities | lower variance | still needs propensities | low | med | **BUILD-TASK (G26A)** | invalid w/o logs |
| Doubly Robust OPE | OPE | model+IPS | propensities+reward model | robust | needs both | med | low | **T3 (after G26A)** | invalid w/o logs |
| PAL / position-bias learning | debias | examination-corrected rel | impression+position logs | corrects position bias | needs click logs | med | med | **T3** | specced (O1/P1), not productionized |
| Negative sampling | training detail | contrastive signal | interactions | enables implicit training | popularity-biased negatives | low | low | **Use now (inside ALS/seq)** | implicit |

**No name-dropping survives without a decision label.** The OPE family (IPS/SNIPS/DR) is uniformly **invalid until propensities are logged** — the central V2 gap.

---

## E. Deep Defense Kernel

Compressed cards (problem · objective · assumptions · movement · update · complexity · failure · why-over-alt · when-alt-wins · in-PulseDiscovery · decision-protected · hard-Q · safe-A · follow-up).

1. **ALS** — implicit-feedback MF. Obj: min Σ c_ui(p_ui − xᵤᵀyᵢ)² + λ(‖x‖²+‖y‖²); confidence c_ui=1+αr_ui. Assumes low-rank + missing≈weak-negative. Movement: alternate closed-form ridge solves per side. Update: xᵤ=(YᵀCᵤY+λI)⁻¹YᵀCᵤp. O(nnz·f²+ (U+I)f³). Fails on cold users/niche tail. Over BPR: closed-form, fast, stable floor. Alt wins when pairwise ranking or features matter. In-PD: the warm floor + served `c2`. Protects: "is the simple baseline actually strong?" HardQ: "why MF over a transformer here?" SafeA: "on this catalog MF beat a *converged* full-softmax SASRec; depth didn't pay." Follow-up: regularization/confidence sensitivity.
2. **SASRec** — self-attentive next-item. Obj: maximize next-item likelihood (full softmax). Assumes informative order + dense-enough sequences. Movement: causal self-attention over item-embedding sequence. Update: Adam on CE. O(L²d) per sequence. Fails on short/noisy histories (our NaN-mask bug). Over GRU/Markov: long-range order. Alt wins when histories are short/sparse — our case. In-PD: V1 method-depth closure (lost to ALS). Protects: "did you actually try deep models?" SafeA: "yes, to valid convergence; it lost honestly and I can show why." Follow-up: full-softmax vs sampled-softmax lever.
3. **FAISS ANN** — sublinear top-K. Obj: argmax IP under an index structure. Assumes embedding geometry preserved. FlatIP=exact scan; trades memory for losslessness. O(N) exact / sublinear approx. Fails when approx params shred recall (IVF low-nprobe). In-PD: G22 frontier. Protects: serving latency choice. HardQ: "exact or approximate in prod?" SafeA: "FlatIP exact default — lossless, ~47% faster; HNSW ef64 only as an explicit scale option."
4. **HNSW vs IVF vs FlatIP** — graph vs coarse-quantizer vs exact. HNSW: navigable small-world, efSearch controls recall/latency; IVF: nprobe controls cells scanned; FlatIP: exhaustive. In-PD: HNSW ef64 overlap 0.992 kept; IVF rejected (overlap collapse 0.28–0.78). Protects: "why not the fastest index?" SafeA: "speed that destroys candidate overlap is a downstream recall bug — IVF failed that test."
5. **CandGen vs Ranking vs Reranking** — three stages: recall-oriented retrieval → precision-oriented LTR → slate-level constraints. Assumes recall ceiling set by candidates. In-PD: coverage diagnosed as the binding constraint (G13). Protects: where to invest. SafeA: "I widened candidates before deepening the ranker because coverage was the ceiling."
6. **Recall@K / NDCG limits** — hit-rate ignores rank/utility; single-gold makes NDCG≈monotone in recall. In-PD: I report recall but flag it understates cold-start cost and isn't comparable across protocols. Protects: metric-gaming. SafeA: "recall improved while catalog health worsened — that's why I added exposure metrics."
7. **Coverage / exposure concentration** — catalog coverage@K and exposure share over the *full* catalog (zeros included). In-PD: G25. Protects: "did the catalog stay healthy?" SafeA: "popularity fallback covers 0.15% of catalog — coverage is a first-class metric here."
8. **Gini / Lorenz** — inequality of an exposure vector; Gini=2·Σi·xᵢ/(n·Σxᵢ)−(n+1)/n. Assumes meaningful zero-inclusion. In-PD: all policies Gini>0.97, popularity=1.0. Failure: Gini alone hides *which* tail; pair with tier-lift. Protects: concentration claim.
9. **Cold-start fallback policies** — popularity/content/item-sim/blended decision tree. In-PD: G24, blended selected. Protects: "what happens to a brand-new user?" SafeA: "graceful popularity — but I measured that it collapses personalization to ~0; it's emergency, not good."
10. **MMR / exposure-aware rerank** — λ·rel−(1−λ)·max-sim, or head-cap/tail-boost. In-PD: G26. Failure: MMR hurt recall here. Protects: mitigation-without-wrecking-relevance. SafeA: "MMR was an honest negative; a head-cap was Pareto-dominant on this protocol."
11. **IPS** — V̂=Σ (π_new/π_log)·r. Assumes logged propensities π_log>0 (overlap). Variance ∝ weight range. In-PD: **NOT valid — no propensities logged.** Protects: "what's the off-policy value?" SafeA: "I won't fake an IPS number; it's invalid until I log propensities — that's exactly G26A."
12. **SNIPS** — self-normalized IPS (÷Σweights). Lower variance, slight bias. Same propensity requirement. In-PD: same gap. SafeA: "SNIPS would be my first OPE estimator once logging exists, for variance control."
13. **Doubly Robust** — DR=reward-model + IPS-correction on residual. Robust if either model right. Needs propensities + reward model. In-PD: T3 after logging. SafeA: "DR after I have both a reward model and propensities."
14. **PAL / position-bias** — model examination·relevance to debias clicks. Assumes separable examination. In-PD: specced (O1/P1), not productionized. SafeA: "position bias is why naive click-recall lies; PAL/IPS correct it."
15. **Negative sampling** — sampled negatives approximate full softmax; popularity-biased samplers skew. In-PD: implicit in ALS/seq training; G20 lever was full-softmax vs sampled. SafeA: "sampled-softmax was the 6.7× lever — still under MF."

---

## F. Product Reasoning Kernel

| Product decision | First-principles driver | Alternative rejected | Data/logging consequence | Evaluation consequence | Business consequence | Evidence |
|---|---|---|---|---|---|---|
| ALS = warm floor | depth didn't beat MF at convergence | SASRec/two-tower | none new | recall ceiling set by candidates | ship simple, cheap floor | [OFFLINE EVAL] |
| SASRec loss ≠ seq useless | sparse/short histories here | "deep always better" | would need richer sequences | depends on history density | don't over-invest in depth now | [BOUNDARY-LIMITED] |
| Coverage > sophistication | recall ceiling is candidate-bound | deeper ranker first | log candidate sources | widening lifted recall ~44% | invest in candgen | [OFFLINE EVAL] |
| FAISS = latency evidence | ANN doesn't change relevance | "FAISS improves quality" | none | overlap, not recall gain | serving cost, not UX lift | [OFFLINE EVAL] |
| Fallback existence ≠ quality | always-nonempty ≠ good | "0% empty = solved" | log fallback rate | coverage/personalization collapse | cold UX is generic | [OFFLINE EVAL] |
| Exposure audit ≠ fairness | concentration ≠ protected-class | "fairness certified" | log impressions by item | Gini/coverage measured | catalog-health signal | [OFFLINE EVAL] |
| HNSW exposure-neutral matters | scale option mustn't reshape catalog | assume neutral | compare exact/HNSW exposure | Gini 0.986≈0.986 | safe to scale | [OFFLINE EVAL] |
| Popularity = emergency only | zero personalization, Gini 1.0 | "popularity is fine default" | track fallback dominance | degenerate exposure | creator/tail starvation | [OFFLINE EVAL] |
| G26 must intervene | measuring ≠ fixing | another audit | rerank logs | relevance/exposure frontier | actually move catalog health | [BUILT, blocked] |
| Logs for real OPE | IPS needs π_log overlap | "compute IPS now" | **log propensities + impressions + position** | enables IPS/SNIPS/DR | online decisioning unlocked | [BUILD-TASK] |
| Mature-org build beyond | solo ≠ org | "this is production" | feature store, exp platform, exposure SLOs | online A/B, guardrail metrics | sustained discovery quality | [VISION] |

---

## G. Business Operating Context

| Area | PulseDiscovery answer | Evidence status | Missing evidence |
|---|---|---|---|
| Business model | discovery/engagement on a large book catalog (marketplace-like) | inferred | real KPI tree |
| Decision owner | ranking/discovery DS + PM jointly | defined | org sign-off loop |
| Decision cadence | per candidate/serving/reranking policy change | defined | release calendar |
| KPI tree | engagement ← relevance × coverage × catalog health × latency | partial | online metric mapping |
| Cost of wrongness | silent catalog-health decay, ANN recall loss, fallback dominance | argued | $/retention quantification |
| Constraint envelope | CPU sandbox, offline only, p95 budget | explicit | prod infra budget |
| Rollout reality | offline → would need canary + A/B | honest | online experiment plan |
| Human ownership | solo, gated, claim-bounded | strong | team RACI |
| Startup version | ALS + FlatIP + popularity fallback + monitoring | shippable-shaped | real deploy |
| Mature-company version | multi-retrieval, LTR, OPE, exposure SLOs, exp platform | VISION | all of it |
| Interview punchline | "I decide which retrieval/serving/exposure policy ships — and prove which ones must not, offline and honestly." | strong | online proof |

---

## H. Operational Failure Modes

| # | Failure | Why it happens | Naive miss | Current detection | Evidence | Mitigation | Remaining gap |
|---|---|---|---|---|---|---|---|
| 1 | Recall↑ but catalog health↓ | optimize hit-rate only | trust recall | G25 exposure metrics | g25 | track Gini/coverage with recall | no online check |
| 2 | Old exposure policy contaminates offline eval | logged data reflects past policy | assume unbiased | flagged (OPE invalid) | audit | log propensities (G26A) | not yet logged |
| 3 | Long-tail zero exposure | head inflation | ignore tail | tier-lift 0.019× | g25 | head-cap/tail-boost | protocol-specific |
| 4 | Popularity fallback dominates | many cold users | ignore fallback rate | fallback_rate metric | g23/g24 | monitor + cap | no alert thresholds |
| 5 | ANN degrades pool silently | aggressive nprobe/ef | trust speed | overlap@K test | g22 | reject IVF, keep overlap≥0.99 | scale untested >40k |
| 6 | Cold users near-identical slates | popularity fallback | call it personalization | personalization 0.001 | g24 | content/item-sim fallback | still mostly head |
| 7 | Seq model underperforms (sparse history) | short noisy sequences | blame model | G20 convergence + caveat | g20 | recognize data regime | secondary metrics lost |
| 8 | Item ID dtype mismatch kills lookup | str vs int keys | silent zero recall | **caught in G24** | g24 | dtype assertions | add CI test |
| 9 | Coverage↑ but relevance collapses | over-diversify | celebrate coverage | recall tracked jointly | g26 | Pareto rule (≥90% recall) | single-gold recall |
| 10 | Exposure reranker overcorrects | too-strong tail boost | ship blindly | α/cap sweep | g26 | tune on frontier | no online value |
| 11 | No propensities → OPE invalid | never logged | report IPS anyway | audit refuses | audit | G26A logging | not built |
| 12 | Feedback loop amplifies popularity | recommend popular → more popular | ignore | exposure-share analysis | g25 | exploration/tail boost | no loop sim |
| 13 | New/rare creators never enter candidates | candgen popularity bias | ignore creators | coverage/zero-exposure | g25 | candgen widening | no creator-side metric |
| 14 | Candgen bottleneck hides reranker gains | reranker can't fix bad pool | tune ranker | coverage diagnosis | g13 | widen candidates first | — |
| 15 | Local p95 ≠ prod p95 | in-proc TestClient | quote local as prod | explicit boundary | g23 | label as local | no prod benchmark |

---

## I. Achievement Moments

1. **False convergence rejected.** *Looked good:* SASRec "early-stopped" at 6 epochs. *Wrong:* val-loss was all-NaN (masked-attention on short sequences) firing patience while train-loss + R@20 still rose. *Detected:* NaN curve inspection. *Decision:* reject as invalid outcome-C, rebuild monitor on eval-R@20 plateau. *Proves:* won't accept a convenient stop.
2. **Stale-script reruns rejected (twice).** *Looked good:* Colab "ran v3." *Wrong:* it executed cached v1. *Detected:* marker asserts. *Decision:* hardened notebook to write/run a distinct `run_g20_v3.py` with guards. *Proves:* reproducibility discipline.
3. **SASRec lost honestly.** *Looked good:* deep model should win. *Wrong/incomplete:* it converged and still scored R@20 0.065 < ALS 0.085. *Decision:* record terminal negative, don't bury it. *Proves:* negatives are senior signal, not failure to hide.
4. **Coverage > depth.** *Looked good:* tune the ranker. *Wrong:* recall ceiling was candidate-bound. *Detected:* coverage diagnosis. *Decision:* widen candgen (~44% recall lift). *Proves:* attacks the binding constraint, not the fashionable one.
5. **IVF rejected despite speed.** *Looked good:* fastest index. *Wrong:* low-nprobe collapsed candidate overlap to 0.28–0.78. *Decision:* keep FlatIP/HNSW, reject IVF. *Proves:* won't trade silent recall for latency.
6. **G24 refused a fake degradation story; G25 refused "fairness solved."** *Looked good:* expected cold<warm recall. *Wrong:* recall is non-monotonic (cold gold more popularity-predictable). *Decision:* reframe cost around coverage/personalization collapse; call G25 concentration, not fairness. *Proves:* fits evidence to reality, not narrative to expectation.
7. **ID dtype bug caught.** str/int `book_id` mismatch silently zeroed lookups; caught via overlap sanity check, fixed repo-wide. *Proves:* checks the plumbing, not just the model.

---

## J. Ambition Gap / Missing Meat

| # | Gap | Why it matters | Evidence needed | When | Claim unlocked |
|---|---|---|---|---|---|
| 1 | No online A/B | only offline | live experiment | T3 | "online lift" |
| 2 | No real user traffic | behaviour unproven | prod logs | T3 | "real users" |
| 3 | No logged propensities | OPE invalid | π_log per impression | **G26A (now)** | "valid IPS/SNIPS" |
| 4 | No valid IPS/SNIPS/DR | can't estimate off-policy value | propensities+reward | G26A→T3 | "off-policy value" |
| 5 | No real exposure intervention deployed | mitigation only measured | rerank in serving path | **G26B** | "exposure mitigation shipped (offline)" |
| 6 | No learned reranker (LTR) | G26 rerankers are heuristic | LambdaMART on features | G26B/BeastMax | "learned reranker" |
| 7 | No LambdaMART implementation | specced only | feature table + train | BeastMax | "LTR ranker" |
| 8 | No two-tower lane | family coverage | two-tower train | T3 | "tested two-tower" |
| 9 | No LightGCN lane | graph family | GCN train | T3 | "tested LightGCN" |
| 10 | No feedback-loop simulation | popularity amplification unmodeled | loop sim harness | BeastMax | "loop-aware analysis" |
| 11 | No creator/business metadata exploited | creator fairness blind | creator-side joins | BeastMax | "creator exposure analysis" |
| 12 | No online-experiment-readiness card | rollout vague | exp design doc | G26A | "experiment-ready" |
| 13 | No production infra benchmark | local p95 only | prod load test | T3 | "prod latency" |
| 14 | No real new-item cold-start sim | only offline-split unseen | new-item launch sim | BeastMax | "item cold-start study" |
| 15 | No retention/long-term metric | proxy metrics only | longitudinal data | T3 | "retention impact" |

---

## K. Next-Gate Decision

**Verdict: Option C — run G26A first, then G26B.**

- **Recommended next gate:** **G26A — Recommender Decision Architecture Consolidation + OPE Logging Design**, then **G26B — Exposure-Aware Reranking / Mitigation** (the already-built G26 work, unblocked and folded in).
- **Why:** the audit finds **real disconnected-component risk**, not incoherence-fatal, but enough to fix before more mitigation: (a) **two ALS models** float in the narrative (V1 `d3aplus` 0.0846 vs served `c2` ~0.0375) with recall not reconciled — an interview trap; (b) **no propensity logging**, so every reranker/OPE claim is offline-heuristic with no path to value estimation; (c) the **serving→evaluation→decision flow** is implied across G22–G26 but never drawn as one architecture. G26A directly **unlocks OPE/mitigation execution** (it produces the logging schema + reconciled model registry + decision map), so it clears the "no audit-only gate unless it unlocks OPE/mitigation" bar. G26 is already built and strong — it loses nothing by shipping as **G26B** on top of a consolidated spine.
- **What must NOT happen:** do not compute an IPS/SNIPS/DR number before propensities exist; do not silently swap V1's `d3aplus` recall for `c2`'s; do not let G26B claim long-tail discovery solved; do not reopen/upgrade V1; do not start another pure audit.
- **Evidence G26A must produce:** a one-page decision/data-flow architecture (candgen→FAISS→rank/rerank→fallback→serve→log→evaluate→decide); a **model/index registry** reconciling `d3aplus` vs `c2` with an explicit "which model backs which claim" table; a **propensity-logging schema** (impression, position, π_log, reward) that would make IPS/SNIPS/DR valid; an **online-experiment-readiness card**. Then G26B = unblock G26 + wire the recommended head-cap/tail-boost into the serving path (offline-shaped).

---

## RiskFrame score (15 axes, /10)

product_thesis_clarity 9 · market_jd_relevance 9 · technique_tournament_depth 8 · deep_defense_kernel 8 · product_reasoning_kernel 9 · data_realism_feature_engineering 8 · synthetic_realism_audit 7 · evidence_honesty 9 · evaluation_validity 7 · decision_economics 8 · industry_pattern_awareness 8 · hairy_failure_modes 9 · achievement_moments 9 · tradeoff_density 9 · interview_dominance 8 → **mean ≈ 8.3**.

**Band: STRONG, NEEDS A CONSOLIDATION/MEAT PASS — not a pile of components, but one spine (decision architecture + OPE logging) away from cohesive. V1 stays `gold_candidate` 8.9; V2 lane is strong-but-needs-G26A.**

**STOP — audit complete; G26 remains BLOCKED pending your G26A/G26B decision.**
