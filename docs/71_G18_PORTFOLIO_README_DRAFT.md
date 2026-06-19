# PulseDiscover — Two-Stage Recommender & Experimentation Program (Portfolio README draft)

*Public-facing, claim-safe. Offline, evidence-backed; not deployed.*

## What it is
PulseDiscover is an **offline two-stage recommender system and experimentation program** built on the Goodreads fantasy/paranormal corpus (**2.56M interactions**, global-time split, realistic 76.5%/23.5% warm/cold traffic). It is built to read like a real company's RecSys decision process — candidate generation, ranking, slate policy, measurement, and serving — with **disciplined claim boundaries** and **documented honest negatives**.

## Two-stage architecture
1. **Candidate generation** → 2. **Learning-to-rank** → 3. **Final slate policy** → top-20, with a **serving index** and a **governance/measurement layer** wrapped around it.

## Candidate-generation tournament
- **ALS f64** (matrix factorization) — the warm collaborative floor, **R@20 0.085** (best single generator).
- **Content-hybrid cold lane** (same-author + same-series + TF-IDF) — **cold R@20 0.241** where MF is structurally zero.
- **Co-occurrence** — conditional/complementary source (+3.76pp marginal coverage; directional, not statistically established).
- **Sequence candidates** — SASRec-small (best sequence, 0.0594), GRU4Rec, canonical full-softmax SASRec.
- **Popularity** — hard-negative floor.
- Adopted default: `ALS-200 + SASRec-small-100 + content-50 + pop-30 + co-occurrence-100` (~317 candidates/user; the cost/coverage **knee**).

## Ranker
- **LambdaMART (LightGBM)** fuses all candidate signals. Within candidate sets it **~doubles** single-retriever ranking (0.038→0.079; mid config 0.120). Feature importance ≠ marginal contribution — GPU sequence features rank high but are non-additive (correlated/substitutable).

## Slate policy
- Policies tested: pure LTR, MMR diversity, cold reserve, creator cap, combined. **Pure LTR is the default** — the candidate stage already makes slates discovery-rich, so aggressive post-hoc governance mostly cost relevance for little health gain (an optional light cold reserve is the only safe add).

## OPE / position-bias / catalog-health layer
- **Off-policy evaluation** (IPS/SNIPS/DR) on a known-propensity simulator — estimator validation; logger positivity is decisive; it could **not** separate the discovery policy from control offline.
- **Position-bias correction** — naive CTR under-credits the cold lane **~11×**; IPS examination-correction recovers it (so raw CTR can't decide).
- **Catalog/creator health** — the discovery policy improves **reach** (+creators), not head **de-concentration**.

## Serving layer
- **FAISS IndexFlatIP** over ALS item factors — exact, lossless, **~39% lower p95** than naive brute-force. (IVF approximation is a scale-demonstration; exact is already sub-millisecond at 40k items.)

## Honest negatives (a feature, not a bug)
- Deep **sequence models did not beat ALS** (best 0.059 vs 0.085); full-softmax was a 6.7× lever but still short.
- A neural **two-tower was built and was an honest negative** (near-random standalone).
- **GPU sequence features added nothing** to the ranker beyond existing signals.
- These overturned intuitive expectations *with evidence*.

## Limitations / what is NOT claimed
- **Offline only** — no online A/B, **no proven business lift**, **not deployed**.
- Ranking/slate metrics are **within-candidate re-ranking**, not full-catalog retrieval.
- **Cold-start, creator fairness, and catalog health are not "solved"** — candidate lane / reach, not solutions.
- LightGCN and canonical-SASRec-to-convergence are **deferred** (documented, not done).

## How to reproduce
- Deterministic (seed 20260616; md5 user splits). CPU for ALS/content/co-occ/LTR/FAISS; GPU steps (SASRec/two-tower training) via the included Colab notebooks (`G11D_GPU_Batch.ipynb`, `G12A2_GPU_Score_Export.ipynb`).
- Evidence in `outputs/evidence/`; full gate-by-gate write-ups in `docs/` (D1→G18); consolidated ledger in `docs/12`.

## Status
**RiskFrame-gold *candidate* / claim-safe portfolio closeout** — strong across product, evidence, evaluation, governance, serving, and claim safety; gold-complete deliberately withheld because no advanced model beat the MF floor (honest negatives) and the project is offline with documented deferrals.
