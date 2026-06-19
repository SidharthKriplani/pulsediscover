# G26A — Model / Protocol Registry

*The single source of truth that makes it **impossible to confuse V1 and V2 metrics**. Every model, protocol, and headline number is pinned to a build, an eval protocol, an evidence file, and a comparability rule.*

## 1. Model registry

| Key | Model | Build | Lane | nU / nI | Role | Headline | Evidence | Comparability |
|---|---|---|---|---|---|---|---|---|
| `d3aplus_als_f64` | ALS f64 | d3aplus tuning | **V1** | 94,185 / 41,960 | V1 warm-floor **tuning** result | **R@20 0.0846** | `d3aplus_als_tune.json` | V1 protocol ONLY |
| `sasrec_canonical_v3` | full-softmax SASRec | G20 v3 | **V1** | seq | method-depth closure | **R@20 0.065** *(plot_reconstructed)* | `g20_..._v3.json` | V1 protocol; packaging-limited |
| `c2_als_f64` | ALS f64 | c2 | **V2 (SERVED)** | 91,732 / 40,541 | the model G22–G26 serve | **R@20 ~0.0375** (G22 exact, full-catalog single-gold) | `c2_als.pkl`, `g22..json` | V2 protocol ONLY |
| `minilm_content` | MiniLM-L6-v2 | G27 | **V2** | 41,866 items / 384-d | semantic content candidate lane | cold-start reachability 54.6%; head R@20 0.010 | `g27_content_index_*`, `g27..json` | V2 candidate-gen ONLY; not a warm replacement |
| `g28_lambdamart_fusion` | LightGBM LambdaMART | G28 | **V2** | candidate-feature table | fusion ranker over ALS+semantic+popularity | **test R@20 0.036 (>ALS 0.022), cold 0.032; 81 test positives** | `g28_learned_ranker_tournament.json` | V2 offline test split ONLY; directional, not online |

**Hard rule:** `d3aplus` R@20 0.0846 and `c2` R@20 ~0.0375 are **NOT comparable** — different build, user/item universe, and eval protocol (d3aplus tuning vs V2 full-catalog single-held-out-gold). Never substitute one for the other. When asked "what's ALS recall?", answer with the *protocol*, not a bare number.

## 2. Why two ALS models exist (the honest story)
`d3aplus` was the V1 tuning model that established the warm-floor *finding* (MF is the floor; depth didn't beat it). `c2` is the V2 *served* model behind the FastAPI/FAISS stack and all serving-system evidence (G22–G26). They are intentionally separate: V1's conclusion is locked and must not be re-litigated, so V2 built fresh evidence on a fresh served model rather than retro-fitting V1's. The registry exists precisely so this design choice reads as discipline, not confusion.

## 3. Protocol registry

| Protocol | Used by | Eval set | Ranking universe | Gold | Notes |
|---|---|---|---|---|---|
| **V1 tuning** | d3aplus ALS, SASRec | 5k eval sample | catalog | held-out | source of 0.0846 / 0.065 |
| **V2 serving full-catalog single-gold** | G22, G24, G25, G26 | warm/cold/eval samples (10k cohort; 2–4k for G22/G26) | full 40,541 catalog | single held-out gold | absolute recall lower; used for RELATIVE cohort/policy structure |
| **V2 latency** | G22, G23 | 1.5–2k query users | — | — | overlap@K + p95 (G22 offline; G23 local TestClient) |
| **V2 exposure** | G25 | 10k users, K=20, ~200k slots | full catalog | — | Gini/coverage/tier-lift |

## 4. Headline-metric pin-board (with tags)

| Metric | Value | Model | Protocol | Tag | Evidence |
|---|---|---|---|---|---|
| ALS warm floor | R@20 **0.0846** | d3aplus | V1 tuning | REAL_OFFLINE | d3aplus_als_tune.json |
| SASRec converged & lost | R@20 **0.065** | sasrec v3 | V1 | PLOT_RECONSTRUCTED | g20_v3.json |
| Served ALS exact | R@20 **~0.0375** | c2 | V2 full-catalog single-gold | SERVED_MODEL · NOT_COMPARABLE_ACROSS_PROTOCOLS | g22.json |
| FlatIP speedup | ~47% p95 ↓, lossless | c2 | V2 latency | REAL_OFFLINE | g22.json |
| HNSW ef64 | overlap 0.992, ~2.4× | c2 | V2 latency | REAL_OFFLINE | g22.json |
| Warm API p95 | **3.34 ms** (local) | c2 | V2 TestClient | LOCAL_LOAD_TEST | g23.json |
| Empty-response | **0%** | c2 | V2 failure modes | LOCAL_LOAD_TEST | g23.json |
| Popularity coverage collapse | **~59×** (3532→60) | c2 | V2 fallback | SERVED_MODEL | g24.json |
| Personalization collapse | **0.68 → 0.001** | c2 | V2 fallback | SERVED_MODEL | g24.json |
| Item cold-start (offline-split) | **57.2%** unseen gold | c2 | V2 | SERVED_MODEL | g24.json |
| Exposure Gini (popularity) | **1.000** | c2 | V2 exposure | SERVED_MODEL | g25.json |
| HNSW exposure-neutral | Gini **0.986 ≈ 0.986** | c2 | V2 exposure | SERVED_MODEL | g25.json |
| Head-cap rerank | R@20 **+40%**, tail **~8×** | c2 | V2 warm pool | SERVED_MODEL · QUARANTINED_G26B | g26.json |

## 5. Quarantine flags
- `sasrec_canonical_v3`: **PLOT_RECONSTRUCTED** — decision valid, packaging one artifact short of gold_complete.
- G26 rerankers: **QUARANTINED_G26B_PENDING_G26A** — built and measured, not licensed to ship/claim until G26B.
- All OPE estimators: **INVALID_NO_PROPENSITIES** — see OPE schema.

**Lock:** V1 = `gold_candidate` 8.9 (terminal). V2 served model = `c2_als.pkl`. No cross-protocol metric substitution.
