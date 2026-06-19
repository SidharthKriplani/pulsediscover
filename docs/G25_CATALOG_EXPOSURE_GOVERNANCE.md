# G25 — Catalog Exposure Fairness / Coverage Governance (V2)

*PulseDiscover **V2** lane. Moves from fallback-quality measurement (G24) to **catalog exposure governance**: does each retrieval/fallback policy concentrate exposure into head items and suppress mid/long-tail opportunity? Evidence: `outputs/evidence/g25_catalog_exposure_governance_report.json`; plots `g25_exposure_concentration_curve.png`, `g25_head_mid_tail_exposure_by_policy.png`, `g25_policy_exposure_lorenz_curve.png`; code `src/eval/catalog_exposure_governance.py`, `scripts/run_g25_catalog_exposure_governance.py`.*

> **Scope guardrails (read first):**
> - This is **catalog EXPOSURE governance** — concentration, coverage and head/tail opportunity under the **served G23 `c2_als.pkl` protocol** (ALS f64), offline, top-20, over the 10k eval-user cohort.
> - It is **NOT** legally protected-class fairness, **NOT** a fairness certification, **NOT** creator/marketplace fairness, **NOT** online diversity, **NOT** production exposure governance.
> - "item-cold-start" = **held-out gold items unseen in train under this offline split** (not real new-catalog cold-start).
> - V1 (`gold_candidate` 8.9) is untouched; no V1 result is re-litigated or upgraded.

## 1. What was measured
For each policy, exposure (times recommended across the audited population at K=20) was accumulated per item over the **full 40,541-item catalog** (zero-exposure items included), then summarized with: catalog coverage@20, unique items recommended, zero-exposure count/share, **Gini**, top-1/5/10% exposure share, head/mid/long-tail exposure share, average exposure per tier, **exposure lift vs catalog base rate**, a Lorenz curve, and a **user-cohort × item-tier** exposure matrix. Served **exact-FAISS vs HNSW** exposure was compared on a 4k warm sample.

## 2. Policy exposure concentration (the core table)
| Policy | Gini | Coverage@20 | Zero-exposure share | Top-1% share | Long-tail exposure share |
|---|---|---|---|---|---|
| item_sim | **0.976** | **9.7%** | **90.3%** | **61.5%** | **1.28%** |
| blended *(served)* | 0.982 | 8.7% | 91.3% | 71.9% | 0.96% |
| als_warm | 0.982 | 8.7% | 91.3% | 71.9% | 0.96% |
| content | 0.995 | 5.0% | 95.0% | 93.3% | 1.00% |
| popularity | **1.000** | **0.15%** | **99.9%** | **100%** | **0.0%** |

Every policy is highly concentrated (Gini > 0.97). **Popularity fallback is near-degenerate** — Gini 1.0, only ~60 unique items ever shown, 99.9% of the catalog gets zero exposure. The Lorenz curves all hug the bottom-right corner; popularity is a right angle.

## 3. Exposure lift vs catalog base rate (where the suppression is)
Relative to each tier's share of the catalog, exposure share / catalog share:

| Tier | ALS / blended lift | Popularity lift |
|---|---|---|
| head | **9.8×** | 10.5× |
| mid | 0.15× | 0.0× |
| long_tail | **0.019×** | 0.0× |

Head items receive ~10× their proportional exposure; long-tail items receive ~**1/50th** (0.019×) under ALS and **0×** under popularity. This is the measured opportunity gap — quantified, not asserted.

## 4. Served exact vs HNSW (governance-neutral approximation)
On the same warm sample, exact FAISS and HNSW ef64 produce **near-identical exposure** (Gini 0.986 vs 0.986; coverage 6.77% vs 6.80%). The approximate index used as the G22 scale option does **not** materially reshape catalog exposure — an honest reassurance, not a diversity improvement.

## 5. User-cohort × item-tier exposure (served blended)
Across every user cohort, head items dominate the slate; cold/unknown users skew **even harder** to head (their popularity fallback is head-only). Warm users receive the only meaningful mid/long-tail exposure. Concretely (blended exposure slots): warm_6plus head 137k / mid 8.6k / tail 1.3k; unknown head 19.5k / mid 1.4k / tail 0.35k.

## 6. Item-cold-start (offline-split sense)
57.2% of held-out gold items are unseen in train under this split; by construction they receive ~0 collaborative exposure and no policy here surfaces them. Offline-split property only — not a production new-catalog claim.

## 7. Policy recommendation
**Among serving-complete policies (never empty), `blended` best preserves catalog opportunity** — it inherits ALS's wider coverage (Gini 0.982, coverage 8.7%) while degrading safely. `item_sim` spreads exposure most (Gini 0.976, coverage 9.7%) but **cannot serve no-history users**, so it is a complement, not a standalone serving policy. `popularity` is the worst possible for catalog health and should be a last-resort floor only. **Honest caveat:** even the best policy is severely head-concentrated; G25 quantifies the gap, it does **not** solve long-tail discovery. A genuine fix (exposure-aware re-ranking / diversity constraint / explicit tail quota) is recommended future work, not built here.

## 8. Safe / forbidden claims
**Safe (proven):** *"PulseDiscovery G25 audits catalog exposure concentration across retrieval and fallback policies, measuring head/mid/tail exposure, catalog coverage, zero-exposure share, and concentration tradeoffs."*
**Forbidden:** fairness solved · creator fairness solved · marketplace fairness certified · online diversity improved · long-tail discovery solved · production exposure governance deployed · protected-class fairness analysis.

## 9. Limitations
Exposure over the eval-user cohort sample at top-20, offline; tiers are train-popularity percentiles (not editorial); genre/shelf cohorts not separately audited (noisy proxy); concentration is not protected-class fairness; no rebalancing intervention is built.

## 10. Status
✅ G25 done (V2). `exposure_concentration_measured:true`, `fairness_solved:false`, `marketplace_fairness_certified:false`, `long_tail_discovery_solved:false`, `production_exposure_governance_deployed:false`. V1 unchanged.

**STOP — G25 complete; awaiting review.**
