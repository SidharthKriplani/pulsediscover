# PulseDiscover — RiskFrame-Gold PRD (v2)
### Content Recommendation / RecSys · PulseRank as the ranking + off-policy-evaluation engine

> *Not a normal PRD. A RiskFrame-gold decision dossier: product thesis → research-grade technique tournament → first-principles defense → product-reasoning kernel → evidence ledger → synthetic-realism audit → business negative-cost chain → operational failure modes → achievement moments → interview defense. Every technical choice is traceable: theory → product decision → data/logging → implementation → evidence → business consequence → interview defense.*

**Claim tags:** `[BUILT]` verified in code/outputs · `[BUILT — real data]` · `[SYNTHETIC]` real code on seeded data · `[PLACEHOLDER]` written, not reproducibly computed · `[FABRICATED → DELETE]` hardcoded/circular · `[VISION]` target architecture, not built · `[BUILD-TASK]` ordered work · `[PARKED]` · `[DELETE]`.

> **Reality boundary (read first).** The shipped repo `pulserank_platform` is a real *ranking + offline-evaluation harness* on a **synthetic e-commerce corpus** (sellers, GMV, returns): from-scratch ALS, LightGBM LambdaRank, IPS/SNIPS with a clipping sweep, MMR governance, drift, offline A/B replay — all `[BUILT][SYNTHETIC]`. **PulseDiscover** is the *content-domain* recommender that **inherits PulseRank's evaluation engine** and re-homes it in serialized fiction. **Update — Phases 2–7 are now built on the MovieLens-1M smoke + a known-propensity simulator.** SASRec, the simulator, IPS/SNIPS/DM/DR, position-bias correction (IPS + PAL), and the feedback-loop audit are `[BUILT]` — on **MovieLens-1M (smoke, real data)** and the **simulator (estimator-validation)**, every headline number with N + CI in `docs/12_EVIDENCE_LEDGER.md`. **Still roadmap only (NOT built):** Two-tower, LightGCN, and the rest of the §9.5 T3 deep-learning roster. MovieLens is smoke (never the domain headline); simulator results are mechanism proof (never online lift); the e-commerce repo is never retro-claimed as a fiction recommender; no number is attached to anything unbuilt.

---

## 1. North-Star Thesis + One-Line Identity

**Identity:** *PulseDiscover helps a content-discovery team decide which recommender to ship by catching the failure that passes every offline test — an offline metric that measures the old logging policy instead of the new model — before it ships a feed that exploits past exposure and starves new creators.*

**North-star metric:** **long-horizon reader value per session, governed across three horizons** — immediate engagement (open/click), satisfaction (length-normalized completion depth), and catalog health (exposure equity). *Built so far: catalog health is measured as **item-level + genre-proxy exposure concentration over 10 simulated rounds** on MovieLens; **creator-level / 30-day equity is aspirational and pending the domain dataset** — MovieLens has no creators.* A recommender that wins horizon 1 while losing horizon 3 is the exact failure this system governs.

**One line:** "I built a sequential recommender and then spent most of the effort on **honest evaluation** — doubly-robust off-policy *estimator validation* against a known-propensity simulator's ground truth, position-bias correction done two ways, and a feedback-loop audit — because on a content platform the model is table stakes and the *evaluation honesty* is the job. (The simulator validates estimator **behaviour/mechanism**; it does **not** establish real online lift — that needs production traffic.)"

**Why this is not a toy:** the decision it governs (promote ranker vX+1 or not) has direct cost — a wrongly promoted ranker degrades retention and concentrates exposure on incumbents, and the damage is invisible to the metric that promoted it.

---

## 2. Target Buyer / JD Archetype

- **Primary role:** Senior/Staff DS or ML Engineer — Recommender Systems / Ranking / Discovery / Feed. **Target companies:** Pratilipi, ShareChat, Glance/InMobi, Spotify, YouTube, Netflix, Meta, Hotstar, Audible, Flipkart/Amazon search-ranking.
- **Secondary role:** Marketplace/search-ranking DS where counterfactual evaluation is daily work (Swiggy, Zomato, Uber, Meesho).
- **JD keywords this is built to hit:** learning-to-rank, candidate generation, two-tower retrieval, sequential recommendation, off-policy / counterfactual evaluation, IPS/SNIPS/doubly-robust, position bias, exploration/bandits, feedback loops, FAISS/ANN.
- **Premium skill clusters:** RecSys · sequential DL/transformers · off-policy evaluation · causal inference · search/IR.
- **Which interview this survives:** the "tell me about a RecSys project" round at a discovery company, where the differentiator question is *"is your offline metric even valid?"* — the question most candidates cannot answer.

---

## 3. Why This Matters Now

Training a recommender is a solved, commoditized skill — every portfolio has two-tower on MovieLens. The *hard* problems in 2026 are the ones that decide whether a recommender should ship:

- **Offline↔online divergence:** offline NDCG rises, online engagement doesn't, because the offline metric is computed on data the *old* policy generated. Counterfactual/off-policy evaluation (IPS/SNIPS/DR) is the scarce skill that addresses this, and it is daily reality at every content platform.
- **Feedback-loop pathology:** the recommender's outputs become its next training data, so it amplifies whatever it already exposed — quietly concentrating the catalog and starving long-tail creators. This is a creator-economy and regulatory concern now, not a research curiosity.
- **Position/exposure bias:** clicks are mostly a function of slot, not quality; uncorrected, the model rewards placement.

The market has moved from "best model" to "trustworthy decision system." PulseDiscover is built entirely on that axis, which is exactly where the scarce hiring signal sits.

---

## 4. The One Insight Nobody Else Brings

**"Your offline ranking metric is a measurement of your old logging policy, not of your new model — so if you don't correct for it, you will ship the model that best exploits past exposure, not the one that best serves readers."**

This is not borrowed; PulseRank *discovered it the hard way*: the V1 heuristic ranker **lost to a popularity baseline** (NDCG@10 0.136 vs 0.177) `[BUILT][SYNTHETIC]`, which forced the question "is my metric measuring the model or the exposure?" — and IPS-corrected evaluation then showed the gain was *smaller*, not larger, than the naive number (bias_delta −0.0053) `[BUILT][SYNTHETIC]`. The repo's own claim-boundary doc forbids framing the two evaluation frames as a "0.134 → 0.522 improvement" because they are different frames on different subsets, not a before/after. The insight drives the whole architecture: every offline number is treated as guilty until proven unbiased.

---

## 5. Layer 0 — Foundational Assumption

**Assumption — the Position-Based click/read Model (PBM):** observed engagement = `P(examined | position) × P(relevant | user, item)`, with examination depending only on position/surface and relevance only on the user-item match, independent given position.

- **Why it's needed:** it is the assumption that makes position-bias correction and off-policy evaluation *identifiable* — it lets you factor exposure out of the click signal (it underpins both IPS-for-LTR and PAL).
- **What becomes valid if it holds:** examination propensities are estimable (from randomization), so IPS/DR are unbiased over the logged support, and a PAL tower can absorb position.
- **What breaks it, and the monitoring/mitigation for each:**
  - *Trust/brand examination* — readers open a known creator regardless of slot → examination is not position-only. **Monitor:** per-creator click-through vs position curves; **mitigate:** add a creator/surface term to the examination model (V2).
  - *Serial continuation* — "next chapter" is driven by narrative, not slot → PBM under-explains within-series clicks. **Monitor:** within-series vs cross-story examination gap; **mitigate:** model continuation as a separate channel.
  - *Infinite scroll* — examination decays with scroll depth differently than a fixed grid. **Monitor:** examination-vs-rank curve drift; **mitigate:** cascade/scroll-depth examination model (V2).
- **Honest stance:** PBM is the working assumption; the OPE numbers are valid *conditional on it*, and I report the conditions under which they degrade rather than claiming universal validity.

---

## 6. Component Map

| # | Component | Role | Input | Output | Status | Why it exists |
|---|---|---|---|---|---|---|
| C1 | Two-tower retrieval | candidate generation | user history, item features | top-K candidates + retrieval score | `[VISION/BUILD-TASK]` | scalable ANN recall; substrate for cold-start |
| C2 | Cold-start content channel | retrieval for new items | item metadata | content embedding + reserved exploration slots | `[VISION/BUILD-TASK]` | you can't explore what retrieval never surfaces |
| C3 | ALS / co-occurrence channels | collaborative baselines | interaction matrix | candidate scores | `[BUILT][SYNTHETIC — e-commerce]` (vendored provenance) **+** `[BUILT — real data (smoke)]` (MovieLens-1M: ALS R@20 0.076 > popularity 0.021, CIs) | honest CF baselines — synthetic provenance and real-smoke |
| C4 | SASRec sequential ranker | primary DL ranker | user item-sequence | next-item relevance | `[BUILT — real data (smoke)]` | reading *order* is the strongest fiction signal |
| C5 | PulseRank LambdaRank ranker | listwise GBT ranker | engineered features | per-candidate score | `[BUILT][SYNTHETIC]` | strong tabular ranker; beat popularity |
| C6 | PulseRank OPE engine | off-policy evaluation | scores, rewards, propensities | IPS/SNIPS/DR value + CI | `[BUILT][SYNTHETIC — simulator]` (IPS/SNIPS/DM/DR validated vs known GT) | unbiased policy value vs old-policy NDCG |
| C7 | Position-bias correction | debiasing | clicks, position | debiased relevance | `[BUILT][SYNTHETIC — simulator]` (IPS reweight + PAL tower) | clicks reward slot, not quality |
| C8 | Multi-objective head + reranker | final selection | relevance, completion, diversity | top-N + audit | MMR `[BUILT][SYNTHETIC]`; multi-obj `[BUILD-TASK]` | govern engagement vs satisfaction vs catalog health |
| C9 | Feedback-loop auditor | catalog governance | exposure dist over rounds | Gini/entropy/long-tail + ratchet demo | `[BUILT][SYNTHETIC — simulator]` | detect exposure amplification before it ossifies catalog |
| C10 | Semi-synthetic OPE simulator | known-propensity data | real interactions | logged exposures + propensities | `[BUILT][SYNTHETIC — simulator]` | OPE math requires known propensities |
| C11 | LightGCN | graph CF | user-item graph | embeddings | `[VISION — V2]` | high-order CF on the real interaction graph |
| C12 | Drift + offline A/B replay | regression gate | scores over time | KL/PSI + SHIP/HOLD card | `[BUILT][SYNTHETIC]` | guardrail-first promotion decision |

No untagged component survives.

---

## 7. Data Flow: Output → Input Chain

1. **Reading logs** `(user, story, position, completion_depth, ts, logged_propensity)` → **C1/C2/C3** emit a merged, deduped candidate set. *Gate:* if `logged_propensity` is absent, OPE downstream is invalid — the pipeline flags the run as "naive-only." *Logged:* candidate-source, retrieval score. *Monitored:* recall@K per source.
2. Candidates + **user item-sequence** → **C4 SASRec** (sequence relevance) and **C5 LambdaRank** (listwise score) → per-candidate scores. *Logged:* model version pinned to embedding version.
3. Scores + **position** → **C7**: Route A reweights the click loss by `1/P(examined|position)`; Route B's PAL tower is dropped at serving → debiased relevance. *Gate:* `display_rank` is excluded as a ranker feature (leakage control, already enforced) `[BUILT]`.
4. Debiased relevance → **C8** predicts engagement/completion/diversity → **constrained selection** (max engagement s.t. completion ≥ τ₁, creator-diversity ≥ τ₂) → final top-N. *Logged:* the active constraint thresholds (these are what C12/PulseSignal A/B-tests).
5. Top-N shown → **interaction logged with its propensity** → **C6 OPE**: IPS/SNIPS weight by `1/p`; DR adds the reward-model residual. **The propensity comes from C10's known logging policy** — this is the load-bearing arrow; without it every downstream number measures the old policy.
6. creator/item-exposure distribution over rounds → **C9** measures **exposure concentration** (Gini/entropy/long-tail) and demonstrates the popularity ratchet → if concentration rises, trigger a CTR+exploration / IPS-corrected policy. *(Built: concentration audit; Shapley mix/rate attribution = V2.)*
7. **C12** monitors KL/PSI drift, runs offline A/B replay → SHIP / HOLD / INVALIDATE (guardrail-first).

---

## 8. Product Reasoning Kernel

*Proof that the theory shaped the build. Each row: decision → first-principles driver → alternative rejected → data/logging consequence → evaluation consequence → business consequence.*

| Product decision | First-principles driver | Alternative rejected | Data/logging consequence | Evaluation consequence | Business consequence |
|---|---|---|---|---|---|
| Build a semi-synthetic simulator with known propensities | IPS/DR are unbiased only with known exposure probabilities; observational logs make them circular | "Just compute IPS on the raw logs" | **Must log per-impression `propensity` + exploration-bucket flag** | Can validate IPS/SNIPS/DR against ground-truth policy value | Prevents shipping a recommender that only exploits historical exposure |
| Position enters via a PAL additive tower (or IPS reweight), not as a concatenated feature | PBM: click = examination(pos) × relevance; separability requires position to be additive at the logit | "Add position as one more input feature" | Log position per impression; PAL needs position as a separate tower input | NDCG must be computed with position-effect removed; can plot item movement | Stops the model rewarding prime placement instead of quality |
| SASRec as the primary DL ranker | Reading is sequential; next-item depends on order — a Markov/transformer over the sequence captures it | Static two-tower-only / DIN | Must store ordered per-user interaction sequences (not just bags) | Sequential Recall@K becomes the headline; needs leave-last-out temporal split | Better "what to read next," the core fiction-platform value |
| Cold-start = separate content retrieval channel + reserved exploration slots | TS/bandits live in the ranker; they cannot rescue an item retrieval never surfaces (no support) | "Thompson sampling handles cold start" | Log a content embedding for new items + a reserved-slot indicator with its propensity | New-item Recall@5 measurable; exploration propensities known | New/long-tail creators get guaranteed discovery → supply-side health |
| Constrained optimization for multi-objective, not enumerate-then-weight | For fixed linear weights, argmax of the weighted sum is a single frontier point — enumeration is redundant | "Compute the Pareto frontier of top-500, then apply weights" | Log objective sub-scores + active constraints | A/B the *constraints*, not opaque weights | Business-legible levers (completion/diversity floors) the org can reason about |
| Feedback-loop audit via exposure-concentration (Gini/entropy/long-tail; Shapley mix/rate = V2) | The policy sets both exposure and engagement → amplification is separable only descriptively | "Monitor aggregate engagement" | Log creator-exposure distribution snapshots over time | Decompose exposure shift; report cross-term as a confidence indicator | Detects catalog ossification months before it shows in lagging metrics |

---

## 9. Technique Tournament

Decision labels: **Use now · Baseline only · V2 · T3 · Discuss-only · Exclude**.

### 9.1 Candidate generation (retrieval)

| Method | Category | Optimizes | Data needed | Strength | Failure mode | Cost/latency | Interpretability | Decision |
|---|---|---|---|---|---|---|---|---|
| Popularity | baseline | global frequency | counts | unbeatable cold floor; the bar V1 *failed* | head-concentrated, no personalization | trivial | high | **Baseline only** (Recall@100 0.587 `[BUILT][SYNTHETIC]`) |
| Item-item co-occurrence (PMI) | classic CF | co-engagement | session pairs | cheap, explainable | sparse, weak alone (Recall@100 0.04) | low | high | **Baseline only** `[BUILT][SYNTHETIC]` |
| ALS (implicit MF) | classic CF | low-rank reconstruction | interaction matrix | strong CF baseline, from-scratch | no content; cold-start blind (Recall@100 0.347) | low | medium | **Use now (baseline)** `[BUILT][SYNTHETIC]` |
| Two-tower (DL) | modern | dot-product retrieval | features + interactions | scalable ANN, content cold-start | needs negatives done right | medium | low | **Use now (primary)** `[VISION]` |
| LightGCN | research | high-order graph CF | user-item graph | propagates multi-hop signal | over-smoothing; cost at scale | medium-high | low | **V2** |
| Content-only (metadata) | cold-start | content similarity | item metadata | best for new items (Recall@100 0.647 — **e-commerce provenance**) | ignores collaborative signal | low | medium | **Baseline (vendored)** `[BUILT][SYNTHETIC — e-commerce]`; the **fiction cold-start channel (C2) is `[VISION/BUILD-TASK]` — not built** |

### 9.2 Ranking model

| Method | Category | Optimizes | Strength | Failure mode | Decision |
|---|---|---|---|---|---|
| Popularity sort | baseline | frequency | the honest bar (NDCG 0.177) | no personalization | **Baseline only** `[BUILT]` |
| V1 heuristic | baseline | hand weights | — (NDCG 0.136, *lost*) | worse than popularity | **Deleted as a ranker; kept as the origin story** `[BUILT]` |
| LightGBM LambdaRank | industry | listwise NDCG | real trained ranker (NDCG 0.192, +8.6%) | tabular only, no sequence | **Use now** `[BUILT][SYNTHETIC]` |
| SASRec | modern | next-item likelihood | sequence-aware; fiction-native (R@10 0.066 > ALS 0.034 on smoke) | needs ordered data, more tuning | **Use now (primary DL)** `[BUILT — real data (smoke)]` |
| BERT4Rec | modern | masked-item (bidirectional) | strong on dense sequences | bidirectional leaks future at serving; heavier | **Discuss-only** (causal SASRec preferred for next-item) |
| DIN / DIEN | modern | target-aware attention | great with rich ad-style features | attention-over-history vs candidate is heavy; less transferable | **V2** |
| DLRM | modern | feature-interaction | shines with many cross-features | needs rich tabular crosses you won't have | **Discuss-only** |

### 9.3 Off-policy evaluation

| Method | Optimizes | Strength | Failure mode | Decision |
|---|---|---|---|---|
| Direct Method (reward model q̂) | model-based value | low variance | biased if q̂ misspecified | **Use now (as DR component)** |
| IPS | unbiased value | unbiased w/ known p | huge variance for small p; circular w/o known p | **Use now** `[BUILT][SYNTHETIC]` |
| SNIPS | self-normalized value | much lower variance, scale-invariant | small bias | **Use now** `[BUILT][SYNTHETIC]` |
| Doubly-Robust (DR) | unbiased + low variance | unbiased if *either* p or q̂ correct (bias −0.007; std 0.0034 < IPS 0.0042; robust under bad q̂) | both wrong in same direction under shared bias | **Use now (headline)** `[BUILT][SYNTHETIC — simulator]` |

### 9.4 Position debiasing · reranking · exploration

| Method | Role | Decision |
|---|---|---|
| IPS on examination propensity (Joachims-style) | reweight click loss by 1/P(examined) | **Use now (Route A)** `[BUILT][SYNTHETIC — simulator]` (MSE 4.7×↓ vs naive; rank-corr cost) |
| PAL additive position tower (model-based) | learn examination, drop at serving | **Use now (Route B)** `[BUILT][SYNTHETIC — simulator]` (Spearman 0.699) |
| Naive "position as a feature" | concatenate position into the net | **Exclude** (interacts, doesn't debias) |
| MMR reranking | diversity vs relevance | **Use now** `[BUILT][SYNTHETIC]` |
| Constrained optimization | floors on completion/diversity | **Use now (replaces Pareto-enumeration)** `[BUILD-TASK]` |
| ε-greedy / UCB exploration | cold-start exposure budget | **Use now** |
| Thompson sampling | posterior-based exploration | **V2** (needs a posterior the point ranker lacks — MC-dropout/ensemble) |

### 9.5 Deep-learning model roadmap (T3 tournament — considered, NOT built)

These are the DL architectures evaluated for PulseDiscover beyond the SASRec (primary) / ALS (baseline) core. **None is built now** — each is labeled and deferred per §9.5.1. The point of listing them is to show the decision landscape: why each was considered, the data it needs, and the conditions under which it would beat the SASRec/ALS core.

| Model | Why considered (optimizes) | Data it needs | When it would beat SASRec/ALS | Label | Why deferred |
|---|---|---|---|---|---|
| **Two-tower retrieval** | scalable ANN retrieval; content cold-start via item tower | user/item features + interactions; item metadata for the content tower | catalog ≫ (millions of items) where ALS/brute-force retrieval doesn't scale; cold items need a content tower | **V2** (build-if-feasible after SASRec, per Gate-0 amendment) | retrieval isn't the smoke bottleneck; needs the domain dataset + FAISS index |
| **LightGCN** | high-order collaborative signal via graph convolution on the user–item bipartite graph | the real user–item interaction graph (no node features required) | dense interaction graphs where multi-hop CF lifts Recall over ALS; warm-ish users with rich neighborhoods | **T3** | a 4th modeling paradigm; over-smoothing/compute care; only after the evidence spine + a domain dataset |
| **GraphSAGE (PyG)** | inductive graph embeddings using node features; embeds unseen nodes | item/user node features + the graph | rich node features + a true cold-start-at-graph-scale need (inductive) | **T3** | needs node features + graph infra beyond the CPU smoke |
| **PinSage** | web-scale random-walk GraphSAGE for item–item recommendation | very large item graph + production graph infra | only at web scale with billions of edges | **Discuss-only** | industrial-scale; out of solo-build scope (name-drop risk if claimed) |
| **Wide & Deep** | memorization (wide feature crosses) + generalization (deep) | many engineered cross/categorical features | rich cross-features (context/demographics) where memorized crosses matter | **Discuss-only** | MovieLens/Goodreads lack the cross-feature set it exists for |
| **DeepFM / xDeepFM** | automatic feature-interaction modeling (FM + deep; xDeepFM adds explicit high-order CIN) | many sparse categorical CTR-style features | CTR prediction with rich categorical side-info (ads/feed) | **Discuss-only** (xDeepFM); **T3** for DeepFM if a rich feature set arrives | no rich categorical feature set in the current data |
| **DIN / DIEN** | target-aware attention over behavior vs candidate (DIEN adds an interest-evolution GRU) | rich user behavior sequences + candidate-side features | candidate-aware attention with side features (e-commerce ad ranking) — can beat SASRec when those features exist | **V2** | heavier than SASRec; needs candidate+behavior features; SASRec is the simpler honest sequential model first |
| **BERT4Rec** | bidirectional (masked-item / cloze) sequential transformer | the same interaction sequences as SASRec | dense sequences where cloze training helps and serving can adapt bidirectionality | **Discuss-only** (kept as a T3 comparison) | bidirectionality leaks future for next-item serving unless adapted; causal SASRec is the honest next-item objective |
| **DLRM** | industrial CTR with massive embedding tables + dense/sparse interactions | huge sparse categorical + dense features at industrial scale | only at industrial feature scale with heavy crosses | **Exclude** | no honest fit for a fiction/portfolio dataset — would be name-dropping |
| **Transformers4Rec** | transformer sequential rec with multi-feature (side-info) sequences (the natural SASRec scale-up) | sequences + item side-features; GPU | SASRec-class sequence modeling enriched with side features, productionized | **T3** | the natural scale-up of the SASRec core; deferred until a domain dataset + GPU |

#### 9.5.1 Why ALL of these are deferred until the BeastMax evidence spine is complete
1. **The differentiator is evaluation honesty, not model count.** PulseDiscover's edge is the OPE/DR + position-bias + feedback-loop *measurement* layer. Adding modeling paradigms before that spine is proven dilutes focus and risks shipping unvalidated complexity (the exact failure the project critiques).
2. **They need richer data/compute than the CPU smoke.** Graph models need the interaction graph; W&D/DeepFM/DLRM need rich categorical features; Transformers4Rec/PinSage need a GPU. The MovieLens-1M smoke (and even a single Goodreads genre on CPU) can't fairly exercise them.
3. **They must plug into the SAME evaluation harness.** Once the spine (baselines → SASRec → known-propensity simulator → IPS/SNIPS/DM/DR → position-bias → feedback-loop) is complete and CI'd on a domain dataset, each new model is evaluated through *that* harness — inheriting honest off-policy metrics + CIs instead of vanilla NDCG. **Build the measurement first, then the models.**

---

## 10. Deep Defense Kernel

*One card per load-bearing method. Full derivations for the differentiators (IPS, SNIPS, DR, SASRec, LambdaMART, two-tower, PAL, ALS); sketches for table-stakes. Each card ends in "how it appears here" + "business decision protected" so the math is load-bearing, not decoration.*

### 10.1 — IPS (Inverse Propensity Scoring) for off-policy value `[BUILT][SYNTHETIC]`

- **Problem:** estimate the value of a new policy π from data logged by a different policy μ, without an online test.
- **Objective (full):** the value of π is `V(π) = E_{x}E_{a∼π(·|x)}[r(x,a)]`. With logged tuples `(x_i, a_i, r_i, p_i)` where `p_i = μ(a_i|x_i)` is the logging propensity, the IPS estimator is
  `V̂_IPS(π) = (1/n) Σ_i [ π(a_i|x_i) / μ(a_i|x_i) ] r_i`.
  For LTR with position bias, the per-item form weights an observed click by the inverse examination propensity: `V̂ = (1/n) Σ_i r_i / P(examined | pos_i)`.
- **Unbiasedness (the derivation that matters):** taking expectation over `a∼μ`,
  `E_{a∼μ}[ (π/μ) r ] = Σ_a μ(a|x)·(π(a|x)/μ(a|x))·r(x,a) = Σ_a π(a|x) r(x,a) = E_{a∼π}[r]`.
  So IPS is unbiased **iff** (a) propensities are known/correct and (b) **positivity** holds: `μ(a|x) > 0` wherever `π(a|x) > 0`.
- **Variance:** `Var ∝ E[(π/μ)² r²]` — blows up as `μ → 0`. A single rarely-shown item can dominate the estimate. This is *the* practical weakness.
- **Complexity:** O(n) over logged events; trivial.
- **Failure modes:** unknown propensities (must be logged/known); positivity violation under a hard top-K retriever (items outside the candidate set have `μ≈0` → estimate only valid over retrieved support); heavy tails.
- **Why over alternatives / when alternative wins:** unbiased where DM is biased; but if variance is intolerable and you trust a reward model, DM (or DR) wins.
- **How it appears here:** C6 computes IPS on simulator logs (propensity from C10). Measured honestly: raw NDCG 0.5275 vs IPS 0.5224, **bias_delta −0.0053** `[BUILT][SYNTHETIC]` — IPS revealed a *smaller* gain.
- **Business decision protected:** stops promoting a ranker whose offline lift is really exposure, not quality.
- **Hard Q / safe A:** *"Your propensities come from the model that logged the data — isn't that circular?"* → "Yes on observational data; that's why I use a known-propensity simulator with an exploration slice. On raw logs IPS is only valid with logged serving propensities or a randomization bucket."

### 10.2 — SNIPS (Self-Normalized IPS) `[BUILT][SYNTHETIC]`

- **Problem:** IPS's unbounded variance and scale sensitivity.
- **Objective (full):** `V̂_SNIPS(π) = [ Σ_i w_i r_i ] / [ Σ_i w_i ]`, with `w_i = π(a_i|x_i)/μ(a_i|x_i)`.
- **Why it helps:** the denominator `Σ w_i` has expectation n (since `E[w]=1`), so normalizing controls the case where the sampled weights don't average to 1 — drastically reducing variance and making the estimate **scale-invariant** to the weights. Cost: it introduces a small **bias** (ratio of expectations ≠ expectation of ratio), vanishing as n→∞ (consistent).
- **Complexity:** O(n).
- **Failure modes:** still sensitive to a few enormous weights; bias non-negligible at small n.
- **Why over IPS / when IPS wins:** SNIPS is the default in practice for variance; vanilla IPS wins only when you need a strictly unbiased estimate and have controlled weights.
- **How it appears here:** C6 reports SNIPS alongside IPS plus a **clipping-sensitivity sweep** over weight caps `[5,10,20,50,100]` `[BUILT][SYNTHETIC]` — the bias/variance trade made explicit.
- **Business decision protected:** prevents a single rare-but-lucky item from dominating the policy-value estimate and triggering a wrong promotion.
- **Hard Q / safe A:** *"SNIPS is biased — why use it?"* → "It trades a small, vanishing bias for a large variance reduction and scale-invariance; at finite n that trade is almost always worth it, and I show the clipping sweep."

### 10.3 — Doubly-Robust estimation (the headline) `[BUILT][SYNTHETIC — simulator]`

- **Problem:** IPS is unbiased-but-high-variance; DM (reward model) is low-variance-but-biased. Get the best of both.
- **Objective (full, contextual-bandit form):**
  `V̂_DR(π) = (1/n) Σ_i [ Ê_{a∼π}[q̂(x_i,a)] + (π(a_i|x_i)/μ(a_i|x_i))·(r_i − q̂(x_i,a_i)) ]`
  i.e. **Direct-Method estimate + IPS-weighted residual of the reward model q̂.** In the simpler logged-action form: `V̂_DR = DM + (1/p)(r − q̂)`.
- **Double robustness (both cases derived):**
  - *If q̂ is correct* (`E[r|x,a]=q̂`): the residual `(r − q̂)` has mean zero, so the second term vanishes in expectation → `V̂_DR → E_π[q̂] = V(π)`, **regardless of p**.
  - *If p is correct* (`p=μ`): the IPS-weighted-residual term is itself an unbiased estimator of `V(π) − E_π[q̂]` (same algebra as 10.1 applied to the residual), so `DM + that = V(π)`, **regardless of q̂**.
  - Hence unbiased if **either** model is correct — "doubly robust."
- **Variance:** lower than IPS because q̂ explains most of the reward and IPS only corrects the residual `(r−q̂)`, which is small when q̂ is decent.
- **Complexity:** O(n) + the cost of fitting q̂.
- **Failure modes:** **correlated misspecification** — p and q̂ trained on the same biased logs can both be wrong in the same direction; "robust to one wrong model" ≠ "robust to both." Positivity still required.
- **Why over IPS/SNIPS / when they win:** DR dominates on the bias-variance frontier when q̂ is even moderately good; pure IPS wins only if you cannot fit any reward model.
- **How it appears here:** C6 on the C10 simulator — DR bias −0.007 with bootstrap std **0.0034 < IPS 0.0042** (variance ≤ IPS, observed), and robust under a misspecified q̂ (DM_bad +0.132 vs **DR_bad −0.006**). `[BUILT][SYNTHETIC — simulator]` *Note: π and the ground-truth reward were ALS-based for the Phase-5 run; **Option B now runs π = the trained SASRec policy** through the same harness — DR bias **−0.002**, std **0.0045 < IPS 0.0051**, double-robust (DR-bad **−0.00008** vs DM-bad +0.157) — confirming **DR evaluates the actual model**, not an ALS stand-in. Ground-truth reward remains ALS-based (not real lift); SASRec-π has lower ESS (4,474) as it diverges more from μ. Evidence: `ope_comparison_sasrec_pi.json`.*
- **Business decision protected:** the central promotion decision — DR is the bridge that makes an offline number trustworthy enough to ship on.
- **Hard Q / safe A:** *"Write the DR estimator and prove double robustness."* → the two cases above. *Unsafe answer to avoid:* the earlier-circulated wrong form `IPS + (1−1/p)(DM−IPS)`, which fails its own stated property.

### 10.4 — SASRec (Self-Attentive Sequential Recommendation) `[BUILT — real data (smoke)]`

- **Problem:** predict the next item from a user's *ordered* interaction history — the dominant signal on a serialized platform.
- **Architecture (full enough to defend):** embed items + learnable positional encodings; stack causal (masked) self-attention blocks. Self-attention: `Attention(Q,K,V) = softmax(QKᵀ/√d) V`, where Q,K,V are linear projections of the sequence embeddings. The **causal mask** zeroes attention to future positions so predicting item t uses only items <t (no leakage at serving). Each block = masked self-attention + position-wise feed-forward + residual + layer-norm.
- **Loss:** for each position t, score the true next item against sampled negatives with binary cross-entropy: `−Σ_t [ log σ(r_{t, pos}) + Σ_neg log(1−σ(r_{t,neg})) ]`, where `r_{t,i} = (hidden_t)·(item_emb_i)`.
- **Complexity:** O(L²·d) per sequence (L = sequence length, d = dim) — the quadratic-in-L attention; fine for capped L (e.g., 50–200).
- **Failure modes:** short sequences (cold users) give the transformer little to attend to; popularity can dominate if negatives aren't sampled well; max-seq-len truncation drops long-range context.
- **Why over BERT4Rec / DIN:** BERT4Rec is bidirectional (masked-item) — powerful but its bidirectionality leaks future context unless carefully adapted for next-item serving; **causal SASRec is the honest next-item objective** and is simpler. DIN's target-aware attention needs rich candidate features and is heavier.
- **How it appears here:** C4, primary DL ranker; trained on MovieLens-1M (smoke, last-position): Recall@10 0.066 > ALS 0.034 with non-overlapping 95% CIs (under-trained floor; canonical full-position + Goodreads pending) `[BUILT — real data (smoke)]`.
- **Business decision protected:** quality of "what to read next," the core retention lever.
- **Hard Q / safe A:** *"Why SASRec over BERT4Rec?"* → "Next-item recommendation is causal; SASRec's left-to-right mask matches the serving objective without the future-leakage adaptation BERT4Rec needs, and it's simpler to tune."

### 10.5 — LambdaMART / LambdaRank `[BUILT][SYNTHETIC]`

- **Problem:** optimize a *listwise*, rank-based metric (NDCG) that is flat/non-differentiable in the scores.
- **Key idea (sketch+):** skip the metric's gradient; define per-pair "lambdas." For a pair (i ranked above j, i more relevant), `λ_ij = −σ / (1 + e^{σ(s_i − s_j)}) · |ΔNDCG_ij|`, where `ΔNDCG_ij` is the NDCG change from swapping i,j. Each document's gradient is `λ_i = Σ_j λ_ij`. These lambdas are the gradients fed to gradient-boosted trees (MART) — so you "boost toward better NDCG" without differentiating NDCG.
- **Complexity:** O(pairs) per query for the lambdas; GBT training as usual.
- **Failure modes:** position/exposure bias in the labels propagates into the lambdas (hence the IPS pairing); tabular only (no sequence).
- **Why over pointwise/pairwise-only:** directly weights pairs by their *metric impact*, so it optimizes the thing you report (NDCG).
- **How it appears here:** C5, LightGBM `objective: lambdarank`; NDCG@10 0.192 vs popularity 0.177 `[BUILT][SYNTHETIC]`. `display_rank` excluded as a feature (leakage control).
- **Business decision protected:** ranking quality measured by the metric the business reads, not a proxy.
- **Hard Q / safe A:** *"Why does LambdaMART optimize NDCG without its gradient?"* → "It uses |ΔNDCG|-weighted pairwise lambdas as surrogate gradients, concentrating learning on swaps that move the metric most."

### 10.6 — Two-tower retrieval `[VISION]`

- **Problem:** retrieve top-K from millions of items in milliseconds.
- **Architecture (full enough):** a user tower `f(user history)` and item tower `g(item)`; relevance = `f(u)·g(i)`. Train with **sampled softmax / in-batch negatives**: for a batch, treat other items as negatives, loss `−log[ exp(f(u)·g(i⁺)) / Σ_{j∈batch} exp(f(u)·g(j)) ]`, with a **logQ correction** subtracting `log(sampling_prob_j)` to debias popularity in the negatives. At serving, item vectors are indexed in FAISS for ANN retrieval.
- **Complexity:** training O(batch²·d) for in-batch negatives; serving O(log N) with HNSW.
- **Failure modes:** popularity bias if negatives aren't corrected; the dot-product geometry can't model fine interactions (that's the ranker's job); cold items need the content tower.
- **Why over ALS:** learns content + collaborative jointly and supports cold-start via the content tower; ALS is collaborative-only.
- **How it appears here:** C1; FAISS index; honest note — current `pulserank` "embeddings" are 8-dim *random* content vectors capturing collaborative signal only, so two-tower is a real upgrade `[VISION]`.
- **Business decision protected:** scalable recall so good items are *retrievable* to be ranked/explored.
- **Hard Q / safe A:** *"Why logQ correction?"* → "In-batch negatives oversample popular items; subtracting log sampling-probability prevents the tower from just learning popularity."

### 10.7 — ALS (implicit-feedback matrix factorization) `[BUILT][SYNTHETIC]`

- **Problem:** low-rank user/item factors from implicit (click) data with no explicit ratings.
- **Objective (full):** `min Σ_{u,i} c_{ui}(p_{ui} − xᵤᵀ y_i)² + λ(Σ‖xᵤ‖² + Σ‖y_i‖²)`, where `p_{ui}=1` if interacted else 0, and `c_{ui}=1+α·count` is the **confidence** (more interactions → more confident).
- **Why ALS (the trick):** the objective is non-convex jointly but **quadratic in xᵤ with y fixed** (and vice-versa), so each has a closed-form least-squares solution: `xᵤ = (YᵀCᵤY + λI)⁻¹ YᵀCᵤ p(u)`. Alternate until convergence — each step is convex, fast, parallelizable.
- **Complexity:** per iteration O(users·k² + nnz·k) with the standard `YᵀY + Yᵀ(Cᵤ−I)Y` factorization.
- **Failure modes:** no content → cold-start blind; linear/dot-product only.
- **How it appears here:** C3, from-scratch numpy/scipy, 32 factors; Recall@100 0.347 `[BUILT][SYNTHETIC]`. **Honest disclosure:** ALS scores are currently **0.0 in the ranker's training features** — retrieval and ranker are not wired (an achievement-moment discovery, §19).
- **Business decision protected:** an explainable collaborative baseline to measure the DL models against.
- **Hard Q / safe A:** *"Why alternating least squares, not SGD?"* → "With one factor matrix fixed the problem is a convex least-squares with a closed form; alternating gives fast, stable convergence and parallelizes per user/item."

### 10.8 — PAL position-bias-aware learning `[BUILT][SYNTHETIC — simulator]`

- **Problem:** clicks conflate examination (position) with relevance; you want pure relevance at serving.
- **Idea (full enough):** model `P(click) = P(seen | position) × P(relevant | x)`. Learn a small **ProbSeen(position)** module jointly with the relevance model; equivalently, work in logits with an **additive** position term: `logit(click) = f(x) + g(position)`. At serving, fix position (e.g., to slot 1) or drop `g` → you score pure relevance.
- **Why additive/separate matters:** if position is instead concatenated into a deep net, it **interacts nonlinearly** with every feature, so zeroing it at serving does *not* remove its influence. Separability is the whole point.
- **vs IPS (the terminology answer):** IPS **reweights** the loss by inverse examination propensity (propensity estimated from randomization); PAL **reparameterizes** the model to learn examination as a factor. IPS is a reweighting estimator; PAL is an architecture. They are complementary — "IPS position debiasing" conflates them.
- **How it appears here:** C7 Route B; built and compared head-to-head with Route A (IPS) on the simulator — PAL Spearman-vs-truth 0.699; item-movement plot produced. `[BUILT][SYNTHETIC — simulator]`
- **Business decision protected:** stops the model rewarding prime placement; lets undervalued (badly-placed) good items surface.
- **Hard Q / safe A:** *"Is PAL just IPS?"* → "No — PAL reparameterizes (a learned position factor, additive at the logit); IPS reweights the loss by inverse propensity. Different tools; I implement and compare both."

### 10.9 — Table-stakes cards (sketch depth)

- **Popularity / co-occurrence (PMI):** `PMI(i,j)=log[P(i,j)/(P(i)P(j))]`; cheap, explainable, used as floors. Failure: head bias, sparsity. **Baseline only.**
- **MMR reranking `[BUILT]`:** `argmax_{i∉S} [ λ·rel(i) − (1−λ)·max_{j∈S} sim(i,j) ]`; λ=0.7 used. Honest result: category max-share improved 0.886→0.513 but **seller Gini worsened 0.404→0.582** — diversity ≠ equality. Failure: greedy, not globally optimal.
- **ANN (FAISS) sketch:** HNSW (graph, high recall, more memory) vs IVF-PQ (quantized, compact, tunable nprobe); choose by recall/latency/memory budget. Failure: recall drops if nprobe/efSearch too low.
- **Negative sampling:** uniform vs popularity-corrected vs in-batch; wrong negatives → popularity leakage. Pairs with two-tower's logQ correction.
- **Gini / entropy (catalog health metric):** `Gini=0` equal, `1` unequal; used to *measure* exposure equity, not optimize it directly.

---

## 11. Data Layer

### 11.1 Raw signals
`user_id, story_id, chapter_id, timestamp, display_position, logged_propensity, exploration_bucket_flag, click (0/1), completion_depth (fraction of chapters read), dwell, explicit_rating/like, genre, creator_id, creator_catalog_age, item_first_seen_ts.`

### 11.2 Derived features
User-sequence embedding (SASRec hidden state); recency-weighted genre affinity; creator affinity; **completion-depth normalized by chapter count** (length de-confounding); popularity-decayed item score; cold-start metadata embedding; exploration-bucket indicator; **position/examination propensity**; session-intent features; long-tail-creator exposure features.

### 11.3 Label construction
- **Engagement label:** click/open — *exposure-confounded by position* → corrected by C7.
- **Satisfaction label:** length-normalized completion depth — *confounded by chapter length and prior fandom* → normalize; treat short-content completion with caution.
- **Observed when:** click immediate; completion delayed (chapters read over days) → **delayed/censored** labels; pick an observation window and state it.
- **Negative labels:** non-click on a shown item ≠ irrelevant (maybe not examined) — the core reason naive labels are biased.
- **Leakage risks:** `display_position`/`display_rank` must never be a ranker feature (it is *the* leakage trap, and is excluded in code `[BUILT]`); future interactions must not leak via a non-temporal split.

### 11.4 Data-quality gates
Missing-propensity check (blocks OPE → run flagged naive-only); temporal-split enforcement (no future leakage); minimum-sequence-length check (cold users routed to content channel); exploration-coverage check (are low-propensity items represented?); completion-normalization sanity (no >1.0 depths).

---

## 12. Synthetic-Data Realism Audit

The current corpus is **fully synthetic e-commerce** (`scripts/seed_demo.py`, seed 20260505): 4,132 sessions / 41,320 impressions / 900 users / 650 items / 80 sellers / 90 days `[BUILT][SYNTHETIC]`. The PulseDiscover simulator (C10) — now `[BUILT][SYNTHETIC — simulator]` (60,400 logged events on MovieLens) — makes OPE testable. Audit:

| Question | Answer |
|---|---|
| Real pattern approximated | a logging policy exposing items by position on a content feed, with skewed popularity and exploration |
| Distributions simulated | Zipf popularity skew; PBM position-examination curve; long-tail creator catalog; lognormal engagement |
| Bias intentionally injected | position bias (via `PROPENSITY_BY_RANK`), popularity exposure bias, cold-start sparsity |
| Failure mode intentionally created | exposure-amplification feedback loop (knob); cold-start items unreachable without the content channel |
| Noise / missingness / censoring | epsilon-greedy exploration (5%); delayed completion; returns/abandonment |
| What would make it invalid | if propensities weren't logged, or if "relevance" were drawn from the same signal as exposure (the circular trap NexusSupply fell into) |
| What it CAN prove | OPE-estimator *correctness* (IPS/SNIPS/DR recover known policy value), debiasing *direction*, ranking-method *ordering* |
| What it CANNOT prove | real user taste, real online lift, real retention — these need production traffic |
| How labelled in interviews | "semi-synthetic with known propensities; I claim estimator validity, not real lift" |
| Real data that replaces it in production | logged Pratilipi reading events with served propensities + an exploration bucket |

**Realism upgrades for the fiction simulator (`[BUILD-TASK]`):** chapter structure, repeat reading, creator concentration, delayed satisfaction, negative-sampling bias, feedback-loop amplification factor.

---

## 13. Evidence Ledger

*Consolidated headline rows below; full per-phase ledger with every number + safe interview line: **`docs/12_EVIDENCE_LEDGER.md`**.*

| Claim | Number | Tag | N | CI / uncertainty | Source file | Interview line |
|---|---|---|---|---|---|---|
| SASRec beats ALS (smoke) | Recall@10 0.0662 vs ALS 0.0339 | `[BUILT — real data (smoke)]` | 1,209 | boot95 [0.052, 0.081] vs ALS [0.024, 0.045] (separate) | `outputs/evidence/sasrec_report.json` | "sequential model beats ALS on the smoke; CIs don't overlap at @10" |
| ALS best baseline | Recall@20 0.0761 | `[BUILT — real data (smoke)]` | 1,209 | [0.061, 0.091] | `retrieval_baselines.json` | "ALS > popularity > co-occurrence" |
| DR low variance + small bias | est 0.2117 vs GT 0.2191 (bias −0.0074) | `[BUILT][SYNTHETIC — simulator]` | 60,400 | std 0.00345 < IPS 0.00417 | `ope_comparison.json` | "DR lower variance than IPS — expected, observed" |
| DR robust under bad q̂ | DR_bad bias −0.0058 vs DM_bad +0.1322 | `[BUILT][SYNTHETIC — simulator]` | 60,400 | DR_bad CI [0.207, 0.220] ∋ GT | `ope_comparison.json` | "double robustness — propensities exact" |
| **DR evaluates the actual SASRec policy (Option B)** | DR bias −0.002, std 0.0045 < IPS; DR-bad −0.00008 | `[BUILT][SYNTHETIC — simulator]` | 60,400 | GT_SASRec 0.138; ESS 4,474 | `ope_comparison_sasrec_pi.json` | "OPE evaluates my real model, not an ALS stand-in; GT on ALS reward, not real lift" |
| IPS position debiasing | MSE-to-truth 0.053→0.011; pos-confound −0.47→−0.25 | `[BUILT][SYNTHETIC — simulator]` | 124 items | rank-corr cost (0.71→0.63) | `position_bias_report.json` | "removes position bias at a rank-variance cost — honest" |
| Feedback-loop ratchet | Gini 0.62→0.70 (ratchet) vs 0.15 (mitigated) | `[BUILT][SYNTHETIC — simulator]` | 10 rounds | mitigation idealized | `feedback_loop_report.json` | "ratchet demoed; mitigation magnitude is idealized" |
| Simulator positivity | min propensity 0.0045 > 0; ε 0.0985 | `[BUILT][SYNTHETIC — simulator]` | 60,400 | exact sums | `simulator_sanity_report.json` | "known propensities; candidate-pool positivity" |
| e-commerce provenance | NDCG 0.192 vs 0.177; HOLD; Gini 0.404→0.582 | `[BUILT][SYNTHETIC — e-commerce]` | 4,132 | — | `archive/pulserank_evidence/*` | "earlier harness, not the fiction build" |
| "0.134 → 0.522 improvement" | — | `[DELETE]` | — | — | eval-frame artifact | never frame as improvement |
| Two-tower / LightGCN / DIN / DeepFM / PinSage / DLRM / Transformers4Rec | — | `[ROADMAP — NOT BUILT]` (§9.5) | — | — | not built | "considered, deferred; no number" |

---

## 14. Evaluation Layer

- **Primary metric:** off-policy policy value (IPS/SNIPS/**DR**) — the unbiased target. **Secondary:** NDCG@10 / Recall@K (ranking quality), examination curves, creator-exposure Gini/entropy (catalog health). **Guardrail:** return/abandonment (e-commerce) → unsubscribe / completion-without-regret (fiction).
- **Why these:** NDCG over MAP because completion depth gives graded relevance; off-policy value because NDCG measures the old policy.
- **Offline↔online gap (centerpiece):** bridged by DR on the C10 simulator — DR variance ≤ IPS (observed: std 0.0034 vs 0.0042) and DR robust under a misspecified reward model. Predicting *real* online lift remains out of scope (simulator-only). `[BUILT][SYNTHETIC — simulator]`
- **Leakage:** `display_rank` excluded as a feature; temporal split enforced `[BUILT]`.
- **N / CI:** the built smoke + OPE spine reports N + bootstrap/Wilson CIs on **every** headline number (`docs/12_EVIDENCE_LEDGER.md`: MovieLens N=1,209; simulator N=60,400). Only the older **vendored e-commerce** rows (4,132 sessions) lack CIs — kept as provenance, not headline.
- **What the metric still misses:** simulator validity ≠ production validity; PBM assumption can break (§5).
- **Robustness/ablations:** IPS clipping sweep `[5..100]` `[BUILT]`; planned PAL-vs-IPS ablation, DR-vs-IPS variance ablation, cold-start Recall@5 ablation.

---

## 15. Business / Negative-Cost Chain

| Technique | Bad decision prevented | Naive outcome | Negative cost | Senior decision | Business edge |
|---|---|---|---|---|---|
| DR / OPE | promoting a ranker on biased offline NDCG | ship the exposure-exploiter | retention flat/down, catalog concentrates | gate promotion on DR value | offline numbers you can actually ship on |
| Position debiasing (PAL/IPS) | rewarding placement, not quality | good items in bad slots never recover | long-tail suppressed | additive position tower / IPS reweight | quality surfaces regardless of slot |
| Cold-start channel | new creators invisible | catalog ossifies | supply-side churn | content channel + reserved exploration | creator growth, fresher catalog |
| Feedback-loop audit | silent diversity collapse | "engagement looks fine" for months | creator exodus, homogenized feed | exposure-concentration (Gini/entropy/long-tail) audit → retrain trigger (Shapley = V2) | catalog health protected before it's visible |
| Completion normalization | bias toward short content | short chapters game the metric | satisfaction proxy corrupted | length-normalize satisfaction | true satisfaction optimized |
| Guardrail-first replay | shipping a metric win that breaks a guardrail | +12.6% rev shipped, returns spike | churn/refund cost | HOLD on guardrail breach | governance over vanity metrics |

---

## 16. Industry / Competitor Pattern Awareness

*Industry-pattern phrasing only — no fake internal claims.*

A mature feed-ranking org would likely separate **candidate generation → ranking → reranking → exploration → online experimentation → feedback-loop monitoring**, with off-policy evaluation gating launches and a feature store serving precomputed signals at low latency. Lightweight/startup stacks often collapse this into "embed + ANN + a ranker + an A/B test," with little counterfactual evaluation. **This solo build implements the offline version of the mature stack** — candidate generation, a listwise + sequential ranker, IPS/SNIPS/DR evaluation, MMR/constrained reranking, and a feedback-loop decomposition — with **semi-synthetic propensities to make OPE testable** where real logged propensities aren't available. What requires real traffic/company data: true online lift, real exploration logs, production-scale FAISS refresh. What gives this solo build credible edge: the *evaluation* layer (DR + position debiasing + feedback-loop audit) that most portfolios — and many shipped systems — skip.

---

## 17. Operational Failure Modes (the thing that breaks in the real world)

- **Offline NDCG rises while creator diversity collapses** — the headline pathology; caught by C9's exposure decomposition, not by NDCG.
- **Popularity ratchet** — the model amplifies what it already exposed; broken by DR-corrected training + diversity floors.
- **Retrieval cold-start starvation** — a new item never enters the top-K, so the ranker/bandit never even gets to explore it ("can't explore what you can't retrieve"); fixed by the content channel + reserved slots.
- **Propensities not logged** — OPE silently becomes invalid and NDCG collapses back to old-policy measurement; the data-quality gate flags this.
- **Completion gamed by short chapters** — satisfaction proxy corrupted; fixed by length normalization.
- **Embedding/index staleness** — a new ranker uses embeddings the FAISS index doesn't have; mitigated by pinning embedding-version to ranker-version and atomic shadow rebuild.
- **Positivity violation under hard top-K** — IPS/DR "unbiased" only over retrieved support; mitigated by exploration in candidate generation.

---

## 18. Senior-vs-Naive Judgment Table

| Situation | Naive move | Senior/Staff move | Why it matters |
|---|---|---|---|
| Offline NDCG improved | ship it | check DR-corrected value first | NDCG measures the old policy |
| New ranker beats old offline | promote | is the gain exposure or quality? confirm with DR | avoids exposure-exploiter |
| Position effect | add position as a feature | additive PAL tower or IPS reweight | concatenated position interacts, doesn't debias |
| Multi-objective | enumerate Pareto + weights | constrained optimization | enumeration is redundant for fixed linear weights |
| Cold start | "Thompson sampling handles it" | content retrieval channel + reserved slots | TS can't rescue un-retrieved items |
| Diversity reranker | "we improved fairness" | report that seller Gini *worsened* (0.404→0.582) | diversity ≠ equality; honesty |
| Two eval frames | "0.134 → 0.522 improvement!" | "different frames, not a before/after" | the overclaim that ends credibility |

---

## 19. Achievement Moments

1. **"SASRec beat ALS on the smoke — and I scoped the win honestly."** Built a small CPU SASRec; Recall@10 **0.0662 vs ALS 0.0339**, 95% CIs non-overlapping (SASRec [0.052,0.081] vs ALS [0.024,0.045]). *Decision:* claim it — but label it an under-trained, **last-position** smoke, not canonical full-position SASRec. *Proves:* I can build and CI-validate a transformer recommender and still bound the claim. `[BUILT — real data (smoke)]`
2. **"DR survived a deliberately broken reward model."** Under an overconfident q̂, the direct method DM's bias exploded to **+0.132**; **DR with the same bad q̂ stayed at −0.006** (CI [0.207,0.220] contains ground truth 0.219). *Decision:* use it to *demonstrate* double robustness, not cite it. *Proves:* I understand why DR works (exact propensities carry the correction). `[BUILT][SYNTHETIC — simulator]`
3. **"IPS debiasing won on bias but lost on rank-correlation."** IPS cut MSE-to-truth **4.7×** (0.053→0.011) and halved position-confounding (−0.47→−0.25) — but its rank correlation with truth **dropped below naive** (0.71→0.63) from reweighting variance. *Decision:* report both; neither dominates every metric. *Proves:* I surface the bias-variance tradeoff instead of cherry-picking. `[BUILT][SYNTHETIC — simulator]`
4. **"My feedback-loop fix looked too good, so I flagged it."** A CTR+exploration policy flattened exposure Gini to **0.15** (ratchet 0.70). *Decision:* report the ratchet demo as real (Gini 0.62→0.70) but label the mitigation **idealized** — it ignores the relevance cost of heavy diversification, which the catalog-level model doesn't penalize. *Proves:* I don't oversell a synthetic mitigation. `[BUILT][SYNTHETIC — simulator]`

*(Earlier e-commerce-harness moments — V1 lost to popularity, IPS corrected *down* (−0.005), MMR worsened seller-Gini 0.404→0.582, ALS not wired into the ranker — are retained in the vendored provenance, `archive/pulserank_evidence/`, tagged `[BUILT][SYNTHETIC — e-commerce]`.)*

---

## 20. Tradeoffs Everywhere

- **Accuracy vs latency:** SASRec self-attention is O(L²·d); cap sequence length and batch-score candidates in one forward pass to keep serving sub-10ms.
- **Offline metric vs online validity:** NDCG is cheap but old-policy-biased; DR is the costlier-but-trustworthy bridge.
- **Personalization vs diversity:** a sharper ranker concentrates exposure; the constrained reranker trades a little engagement for catalog health.
- **Exploration vs exploitation:** reserved cold-start slots cost short-term engagement to buy long-term supply health.
- **Calibration of propensities vs variance:** vanilla IPS (unbiased, high variance) vs SNIPS/clipping (small bias, low variance).
- **Synthetic control vs real messiness:** the simulator gives known propensities (provable OPE) but cannot prove real lift.
- **Model sophistication vs evidence quality:** a fancier model with no DR validation is *worse* portfolio evidence than a simple model with honest OPE.
- **Batch vs real-time:** embeddings/FAISS rebuilt in batch; ranking real-time; the staleness window is the tension.
- **Free build vs paid infra:** all of this runs on CPU/free GPU; full-Goodreads scale is the only place a small paid GPU helps.

---

## 21. Production Behavior

*Production-shaped solo build, not deployed at scale.*

- **Batch path:** nightly embedding refresh → FAISS index rebuild (shadow → atomic swap) → candidate-generation precompute.
- **Online path:** user vector → ANN top-K → batched neural scoring (SASRec + ranker, one forward pass) → C7 debiasing → C8 constrained selection → top-N.
- **Logging path:** every impression logs `(user, item, position, propensity, exploration_flag)` — the OPE-critical record.
- **Monitoring:** KL/PSI on score+feature distributions `[BUILT]`; retrieval-recall as the leading signal, engagement as lagging; creator-exposure Gini tracked weekly.
- **Drift triggers:** PSI breach or exposure-amplification > threshold → retrain (with a CI so noise doesn't trigger).
- **Regression gate:** offline A/B replay → guardrail-first SHIP/HOLD/INVALIDATE card `[BUILT]`.
- **Fallback / abstention:** if retrieval confidence is low or the user is cold, fall back to content channel + popularity floor.
- **HITL:** a human reviews HOLD decisions and exposure-amplification alerts.
- **Registry / lineage:** model version pinned to embedding version; every evidence JSON versioned with the seed.

---

## 22. Visual / Demo Artifact Plan

**Produced (in `outputs/plots/`, real):** retrieval Recall@K bar chart (`retrieval_baselines_recall.png`) · SASRec-vs-baselines Recall@10 with CIs (`sasrec_vs_baselines_recall10.png`) · IPS/SNIPS/DM/DR estimates + clipping sweep (`ope_comparison.png`) · simulator diagnostics: examination + propensity (`simulator_diagnostics.png`) · examination curve + PAL-vs-IPS item-movement + naive-vs-IPS (`position_bias.png`) · feedback-loop Gini-over-rounds + Lorenz + buried-quality (`feedback_loop.png`). Each carries N + tag.
**Planned (NOT produced):** two-tower retrieval comparison · cold-start new-item Recall@5 · domain (Goodreads/Amazon) versions. (The e-commerce A/B-replay HOLD card lives in vendored provenance, not this build.)

---

## 23. Interview Defense Bank

| Question | What they're testing | Unsafe answer | Safe answer | Follow-up depth |
|---|---|---|---|---|
| "Derive the DR estimator." | OPE math | the wrong `IPS+(1−1/p)(DM−IPS)` form | `V_DR = DM + (1/p)(r−q̂)`; unbiased if either p or q̂ correct (both cases) | "what if both are wrong the same way?" → correlated misspecification |
| "Your propensities come from your own logging policy — circular?" | OPE validity | "the model corrects it" | "circular on raw logs; I use a known-propensity simulator + exploration slice" | "how much exploration traffic?" → 1–5%, regret cost |
| "Does adding position as a feature debias it?" | position bias | "yes, zero it at serving" | "no — it interacts in a deep net; need additive PAL tower or IPS reweight" | "show the item-movement plot" |
| "SASRec vs BERT4Rec?" | sequential DL | "SASRec is newer" | "causal next-item objective matches serving; BERT4Rec is bidirectional/leaky for next-item" | "derive the attention + causal mask" |
| "MovieLens Recall is tiny — impressive?" | judgment | defending the number | "MovieLens is my smoke test; the headline is DR validity on Goodreads + simulator" | "what's your real dataset and why Goodreads?" |
| "What does your ranker do to the catalog in 12 months?" | systems thinking | "engagement stays up" | "popularity ratchet + creator collapse; I audit exposure concentration (Gini/entropy/long-tail) over rounds and break the loop with a CTR+exploration / IPS-corrected policy (Shapley attribution = V2)" | "how do you separate amplification from real preference?" |
| "Top-K retrieval at scale?" | IR/systems | "FAISS" | "HNSW vs IVF-PQ trade; recall/latency/memory; positivity implications for OPE" | "what nprobe/efSearch?" |
| "Is 0.134→0.522 your improvement?" | honesty | "yes, 4×!" | "no — different eval frames on different subsets, not a before/after" | "so what *is* the honest IPS result?" → −0.005 |

---

## 24. Build Path: Current → BeastMax → T3

| Stage | Build | Evidence required | Interview claim enabled |
|---|---|---|---|
| **Current** | PulseRank eval harness: ALS, LambdaRank, IPS/SNIPS, MMR, A/B replay `[BUILT][SYNTHETIC]` | committed evidence JSONs (present) | "I built an off-policy-aware ranking + evaluation harness" |
| **BeastMax — smoke (DONE)** | SASRec + baselines on MovieLens-1M; known-propensity simulator; IPS/SNIPS/DM/**DR**; IPS + PAL position debiasing; feedback-loop audit | real Recall/NDCG + CIs; DR lower-variance-than-IPS (observed) + double-robustness; pos-bias MSE 4.7×↓; ratchet demo — all in `docs/12_EVIDENCE_LEDGER.md` | "I built a sequential recommender validated by doubly-robust OPE + position debiasing on a known-propensity simulator" |
| **BeastMax — domain (pending)** | re-run the spine on a Goodreads fiction genre / Amazon Books; add cold-start content channel | domain Recall/NDCG + CIs; new-item Recall@5 | "...and carried it to a fiction-domain dataset with the same honest evaluation" |
| **T3 / polished** | LightGCN; DIN; multi-objective revenue-aware ranking; Pratilipi-flavored synthetic demo; serving write-up | LightGCN Recall vs two-tower; constrained-objective A/B sim | "I extended it to graph CF and revenue-aware multi-objective ranking" |

*No T3 claim is made without its evidence row.*

### 24.1 T3 Deep-Learning Roadmap (deferred — none built now)

Maps the §9.5 models onto the build path. **All gated behind the completed evidence spine on a domain dataset**, and each must plug into the existing OPE/CI harness before any claim is made.

| Model | Label | Data / compute needed | Evidence required before claiming | Prerequisite |
|---|---|---|---|---|
| Two-tower retrieval | **V2** | domain dataset + FAISS; CPU/GPU-light | Recall@K vs ALS with CIs on domain data | after SASRec path stable |
| DIN / DIEN | **V2** | behavior + candidate-side features; GPU | OPE-validated lift vs SASRec on the harness | rich features available |
| LightGCN | **T3** | real user–item graph; GPU-light | Recall@K vs two-tower/ALS with CIs | evidence spine complete on domain data |
| GraphSAGE (PyG) | **T3** | node features + graph; GPU | inductive cold-start Recall vs content channel | node features available |
| DeepFM | **T3** (if features arrive) | rich sparse categorical features | CTR/Recall vs baselines on the harness | feature set exists |
| Transformers4Rec | **T3** | sequences + side features; GPU | OPE-validated lift vs SASRec | GPU + domain data |
| BERT4Rec | **Discuss-only** (T3 comparison) | same sequences as SASRec | side-by-side vs causal SASRec (serving-honest) | comparison only |
| Wide & Deep / xDeepFM | **Discuss-only** | rich cross/categorical features | — | features that justify it |
| PinSage | **Discuss-only** | web-scale item graph + infra | — | industrial scale (out of solo scope) |
| DLRM | **Exclude** | industrial sparse+dense at scale | — | no honest fit |

*Build the measurement spine first; deep-learning models are added afterward and evaluated through the same off-policy/CI harness — never claimed on vanilla NDCG, never built ahead of the evidence.*

---

## 25. Free vs Small-Paid-Infra Version

| Version | Infra | Cost | Proves | Cannot prove |
|---|---|---|---|---|
| Free | CPU; matplotlib (FAISS/SQLite when needed) | ₹0 | **built:** SASRec correctness, IPS/SNIPS/DM/DR validity, PAL-vs-IPS, feedback-loop audit (all on smoke + simulator) | production-scale latency; real online lift; **two-tower / cold-start channel (future, not built)** |
| Small-paid | one rented A100 hour for full-Goodreads SASRec | ~₹150 | scale headroom of the sequential model | still not real traffic |

No paid spend is required to make the headline (DR) result real.

---

## 26. No-Overclaim Boundary

**Never say:** "This was production-deployed / served real users." · "My offline NDCG proves online lift." · "0.134 → 0.522 improvement." · "IPS is valid without known propensities." · "PAL is the same as IPS." · "This synthetic Recall is a real-world number." · "Two-tower/SASRec results" (they're not built yet).

**Say instead:** "Production-shaped solo build on synthetic + public data." · "DR-corrected off-policy estimates are validated against a known-propensity simulator's ground truth — I claim estimator validity / mechanism, **not** real online lift." · "0.134 and 0.522 are different evaluation frames, not a before/after — the honest IPS delta is −0.005." · "IPS needs logged/known propensities or an exploration slice." · "PAL reparameterizes; IPS reweights." · "Synthetic, seed-pinned; I claim estimator validity, not real lift." · "SASRec/two-tower are designed and building next; no number yet."

---

## 27. Resume / LinkedIn Lines

*Format: Outcome by Method, beating Alternative by Δ, at Cost, despite Constraint. Tagged internally.*

- "Built an off-policy-aware ranking harness (LightGBM LambdaRank + from-scratch ALS) that beat a popularity baseline by +8.6% NDCG@10 on a 4,132-session corpus, and honestly reported that IPS correction *reduced* the apparent gain (−0.005) rather than inflating it." — **safe now** `[BUILT][SYNTHETIC]`
- "Shipped a guardrail-first HOLD on a +12.6% revenue change that breached a 9pp returns guardrail — governance over vanity metrics." — **safe now** `[BUILT][SYNTHETIC]`
- "Built a sequential recommender (SASRec) on MovieLens-1M and validated ranking policies with doubly-robust off-policy evaluation on a known-propensity simulator — DR lower-variance than IPS and robust under a misspecified reward model — because offline NDCG measures the logging policy, not the model." — **safe now (smoke + simulator)** `[BUILT — smoke + simulator]`
- "Corrected position bias two ways (IPS examination-reweighting vs a PAL additive tower) and audited recommendation feedback loops via an exposure-concentration (Gini/entropy/long-tail) analysis." — **safe now (simulator)** `[BUILT][SYNTHETIC — simulator]`
- "(Two-tower / Goodreads-domain versions: roadmap, not built.)" — **interview-only**
- "Improved offline NDCG 4× (0.134→0.522)." — **unsafe / delete** (eval-frame artifact).

---

## 28. Convergence Self-Score + Acceptance Test

### 28.1 Self-score (1–10; RiskFrame-gold threshold = 9+)

| Axis | Score | Note |
|---|---|---|
| Product thesis clarity | 9 | governs the promote-or-not decision, sharp north star |
| Market/JD relevance | 9 | RecSys + OPE is the scarce, hiring-relevant pairing |
| Technique tournament depth | 9 | retrieval/ranking/OPE/debiasing all toured with decision labels |
| Deep Defense Kernel | 9 | full derivations for IPS/SNIPS/DR/SASRec/LambdaMART/ALS/two-tower/PAL |
| Product Reasoning Kernel | 9 | theory→logging→eval→business traced per decision (propensity logging is the proof) |
| Data realism / feature engineering | 8 | rich features; completion-normalization + propensity logging called out; some V2 |
| Synthetic realism audit | 9 | explicit can/can't-prove, circular-trap contrast |
| Evidence honesty | 10 | every number tagged + sourced; overclaim explicitly deleted |
| Evaluation validity | 9 | DR bridge + leakage controls; CIs now present on the built spine |
| Decision economics | 9 | negative-cost chain per technique |
| Industry-pattern awareness | 8 | pattern phrasing, no fake internal claims |
| Hairy failure modes | 9 | amplification, positivity, staleness, cold-start starvation |
| Achievement moments | 9 | four real, code-grounded ownership moments |
| Tradeoff density | 9 | tradeoffs at every layer |
| Interview dominance | 9 | defense bank with derivations + unsafe/safe |

**Average ≈ 8.9.** The BeastMax-core spine (Phases 2–7) is now **built on the MovieLens-1M smoke + a known-propensity simulator, with N + CIs** (`docs/12_EVIDENCE_LEDGER.md`). The two sub-9 axes (data realism, industry-pattern) remain gated on the **domain** build (Goodreads/Amazon), reaching 9 once that lands. **Verdict: gold-threshold on the document; smoke + simulator evidence in hand; domain headline pending.**

### 28.2 Acceptance test (must all be yes)

- Every major technique has a why/why-not tree? **Yes** (§9).
- Every load-bearing method has a defense card? **Yes** (§10).
- Every headline number has tag, N, CI, source? **Yes for the built spine** — tags/N/source + bootstrap/Wilson CIs present (`docs/12_EVIDENCE_LEDGER.md`); domain (Goodreads) numbers pending.
- Synthetic data has a realism audit? **Yes** (§12).
- Hairy real-world failure modes? **Yes** (§17).
- Architecture reflects first-principles understanding? **Yes** — propensity-logging→simulator→DR→promotion gate is the spine (§8).
- Every major decision connects to business cost? **Yes** (§15).
- Achievement moments? **Yes** (§19).
- Avoids fake production ownership? **Yes** (§21, §26).
- Safe interview language present? **Yes** (§26, §23).
- Would a top-company interviewer feel depth beyond tutorials? **Yes** — DR derivation + double-robustness demo, PBM, candidate-pool positivity, PAL-vs-IPS, feedback-loop exposure-concentration audit.
- Every component traceable theory→product→data→implementation→evidence→business→defense? **Yes for built + designed components; `[VISION]` ones carry no numbers.**

**One open item:** the older **vendored e-commerce** rows lack CIs (provenance only); the built smoke + OPE spine already carries N + bootstrap/Wilson CIs. Everything else passes.

---

*Caveat: PulseRank numbers verified in `outputs/evidence/*.json`; "tests green" inferred from committed caches/CI (61 test fns), not a live run. Academic methods (Joachims 2017 unbiased LTR; Swaminathan & Joachims 2015 SNIPS; Dudík et al. DR; Kang & McNamara SASRec line; Burges LambdaMART; Hu/Koren/Volinsky implicit ALS; Guo 2019 PAL; Open Bandit Pipeline) are stable pre-2025 knowledge, not re-verified by search this session — citation-check before quoting papers by name in a room.*
