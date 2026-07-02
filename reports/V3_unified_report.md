# PulseDiscover V3 — Unified Report: ALS Floor → Serving/OPE → Neural + Exploration

**Scope:** offline evaluation only. No online lift, no A/B results, no production-quality claim.
**Dataset:** 2.56M real Goodreads interactions, temporal split (unchanged from V1/V2).
**What V3 adds:** a two-tower neural retrieval model with content signals (G32) and a Thompson
Sampling exploration layer (G33), each evaluated against the existing baselines with the same
honesty standard as V1/V2.

---

## 1. The story in one paragraph

V1 established an offline modelling floor: a well-tuned ALS reached warm Recall@20 = 0.085, and a
converged SASRec sequence model *lost* to it (0.065) — documented as an honest negative. V2 turned
that floor into a served system: FAISS FlatIP exact retrieval (lossless, −47% p95 latency), IVF
rejected for candidate-pool collapse, a learned ALS+semantic fusion ranker, executed off-policy
evaluation (IPS/SNIPS/DR), a MiniLM semantic cold-start lane, and a live Cloud Run endpoint. **V3
asks the two questions the target role cares about most: can a neural two-tower model that uses
content beat ALS, and does a bandit exploration layer buy real catalog diversity? The answers are
an honest _no_ on the first and a measured _yes-with-a-cost_ on the second — and the most useful
result is that off-policy evaluation breaks down for the exploration policy exactly where the
theory says it should.**

---

## 2. G32 — Two-tower neural model vs ALS (honest negative)

**Model.** Genuine two-tower, dot-product scored:
- **User tower:** mean-pooled learned embeddings of the user's interaction history (≤50 items).
- **Item tower:** a learned item-ID embedding **plus** a projection of the item's 384-d MiniLM
  content embedding (title + genre shelves) — so content is now *inside* the retrieval model, not
  a side lane.
- **Training:** BPR loss, 8 uniform negatives per positive, 23 epochs, PyTorch (CPU).

**Protocol.** Identical to the ALS floor (`src/d3b_als_floor.py`): trained on the same 2.51M
interactions (core minus held-out golds), evaluated leave-last-out on the same 5,000-user
`eval_sample.csv`, top-200, history-masked, with per-user bootstrap 95% CIs.

| Model | Recall@20 | 95% CI | NDCG@20 | Verdict |
|---|---|---|---|---|
| ALS (F=64) | **0.0846** | [0.0766, 0.0922] | 0.0344 | baseline |
| Two-tower (ID + content) | 0.0642 | [0.0574, 0.0712] | 0.0262 | **loses to ALS** |
| Two-tower (ID only, content off) | 0.0432 | [0.0376, 0.0490] | 0.0162 | ablation |

**Reading it honestly.**
- **ALS wins, and it's not noise.** ALS's CI lower bound (0.0766) sits entirely above the
  two-tower's upper bound (0.0712) — non-overlapping, so the −0.020 gap is real.
- **Content is a real signal.** Turning the content tower off drops Recall@20 from 0.0642 to
  0.0432 (CIs do not overlap): MiniLM content contributes **+0.021 Recall@20**. Content earns its
  place in the model; it just isn't enough to overtake a well-tuned ALS on warm users.
- **This extends the SASRec story.** Two neural models — a sequence model (SASRec, 0.065) and a
  two-tower (0.064) — now both lose to matrix factorization on this dense, warm Goodreads setting.
  The lever on sparse book histories is not model depth.

**Cold-start note (honest boundary).** The `cold_eval_sample` cohort here is not a true no-history
cohort (those users have training histories; their held-out items are mostly warm catalog items),
so two-tower Recall@20 on it is ~0 and is **not** comparable to V2's MiniLM cold lane (0.241),
which was measured on genuinely unseen users/items via a different construction (G27). V3 does not
claim a cold-start improvement. What V3 *does* show is that content signals are now first-class
inside the retrieval model — the mechanism V2's cold lane relied on — rather than bolted on.

**Artifact:** `outputs/evidence/G32_two_tower_eval.json`.
**Known minor imperfection:** the pooling mask treats item index 0 as padding, affecting 2 of
5,000 eval users and 0.0014% of interactions — below the 4th decimal, no effect on the verdict.

---

## 3. G33 — Thompson Sampling exploration layer (explore/exploit, offline)

**Setup.** For each of the 5,000 warm eval users, take the two-tower top-200 candidate pool and
build a slate of 20 under two policies:
- **Greedy:** rank by two-tower score (pure exploit, static).
- **Thompson:** a per-item Beta(α,β) posterior; sample θ ~ Beta, rank by θ, update the posterior
  with a held-out-positive **proxy** reward (the user's future interactions from `test.csv`), over
  6 passes. Offline simulation — no live serving.

| Metric | Greedy | Thompson | Direction |
|---|---|---|---|
| Catalog coverage | 16.62% (6,972 items) | **50.53%** (21,201 items) | ~3× more of the catalog exposed |
| Exposure Gini | 0.9736 | **0.8063** | less concentrated (fairer exposure) |
| Intra-list diversity | 0.492 | 0.5054 | modestly higher |
| Slate hit-rate@20 (relevance) | **0.0642** | 0.0510 | **relevance drops ~1.3pp** |

**Reading it honestly.** This is the explore/exploit tradeoff, stated as a cost, not a win:
Thompson gives up ~1.3 points of top-slate relevance to roughly triple catalog coverage and
materially cut exposure inequality. We do **not** claim the bandit improves relevance — it does
not, and it isn't supposed to.

**Off-policy evaluation of the two policies (the sharp finding).** Using the existing
IPS/SNIPS/DR estimators (`src/ope/logging_policy.py`) on a single-action contextual-bandit setup
(logged action ~ ε-softmax over the pool):

| Target policy | True value | IPS | SNIPS | DR | ESS | Logged matches |
|---|---|---|---|---|---|---|
| Greedy top-1 | 0.0062 | 0.0048 | 0.0047 | 0.0050 | 168.0 | 332 |
| Thompson top-1 | (0.0178, biased) | 0.0000 | 0.0000 | −0.0005 | 21.1 | 38 |

- **OPE recovers the greedy policy** (DR 0.0050 vs true 0.0062) because the logging policy covers
  it well — 332 matched samples, healthy ESS.
- **OPE breaks down for the exploration policy.** The Thompson policy selects items the logging
  policy rarely logs, so overlap collapses (38 matches, ESS 21) and IPS/SNIPS/DR become
  unreliable — DR even goes slightly negative. This is the textbook overlap problem: you cannot
  reliably off-policy-evaluate an exploration policy with a logging policy that doesn't cover it.
  It pairs directly with V2's G29 "DR went negative" lesson.
- **The Thompson "true value" (0.0178) is optimistically biased** — its Beta posterior was updated
  on the same proxy reward it is then scored against, so argmax-posterior-mean memorizes rewarded
  pool items. It is reported for transparency and explicitly **not** counted as a relevance gain.

**Artifact:** `outputs/evidence/G33_bandit_eval.json`.

---

## 4. What V3 does and does not establish

**Does (all from real offline runs):**
- A two-tower model that fuses learned IDs with MiniLM content, trained with BPR, evaluated
  apples-to-apples against ALS — an honest negative (0.064 vs 0.085) with a real content lift (+0.021).
- A Thompson Sampling exploration layer with a measured explore/exploit tradeoff: ~3× coverage and
  lower exposure Gini at a ~1.3pp relevance cost.
- A concrete demonstration that OPE overlap collapses for an exploration policy (ESS 168 → 21).

**Does not claim:**
- Online lift, engagement, or A/B results (offline only; rewards are proxies, propensities synthetic).
- A neural win over ALS (it lost — twice now, counting SASRec).
- A bandit relevance improvement (relevance drops; that is the point).
- A cold-start improvement over V2's 0.241 lane (different cohort; not compared).
- Production quality — the Cloud Run endpoint is a demo, not a product.

---

## 5. Metric ledger (V1 → V2 → V3)

| Result | Number | Protocol |
|---|---|---|
| ALS warm floor | R@20 = 0.0846 | V1, eval_sample 5k, leave-last-out |
| SASRec, converged | R@20 = 0.065 — lost to ALS | V1, honest negative |
| Content-hybrid cold lane | cold R@20 = 0.241 (ALS = 0) | V1 cold-start |
| FAISS FlatIP (exact) | lossless, −47% p95 | V2 |
| FAISS IVF | rejected — overlap collapsed 0.28–0.78 | V2 |
| Learned fusion (LambdaMART) | test R@20 = 0.036 vs 0.022 | V2, held-out test (81 pos) |
| Off-policy evaluation | DR 0.0089 vs true 0.0106; SNIPS stable | V2 OPE |
| Semantic cold-start reach | 54.6% of unseen golds | V2 |
| **Two-tower (ID + content), BPR** | **R@20 = 0.0642** [0.0574, 0.0712] — lost to ALS | **V3 G32, honest negative** |
| **Two-tower content ablation** | **+0.021 R@20** from MiniLM content | **V3 G32** |
| **Thompson vs greedy** | **coverage 16.6%→50.5%, Gini 0.974→0.806, relevance 0.064→0.051** | **V3 G33, offline** |
| **OPE of exploration policy** | **ESS 168→21; DR recovers greedy, fails Thompson** | **V3 G33, methodology demo** |

_All V3 numbers trace to `outputs/evidence/G32_two_tower_eval.json` and
`outputs/evidence/G33_bandit_eval.json`._
