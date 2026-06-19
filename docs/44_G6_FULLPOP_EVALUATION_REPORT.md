# 44 — G6: Full / Larger-Scale Evaluation Report

*Largest feasible scale-stability test of the C2/O1b/H1 conclusions, using existing artifacts only (ALS factors + regenerated content lists from real data). **Offline only — no online A/B, no business lift, no deployment, no creator-fairness claim. Real numbers only.** Evidence: `outputs/evidence/domain_fullpop_eval_report.json`.*

---

## 1. Scale statement
| Quantity | Value |
|---|---|
| Full heldout user pool | 42,683 (36,754 warm-gold + 5,929 cold-gold) |
| Original sampled eval | 5,000 warm + 5,000 cold |
| Precomputed-list set (C2) | 6,973 users |
| Heldout warm-gold users | 36,754 |
| **G6 warm eval used (largest *valid* warm heldout set)** | **34,539 (6.9× the warm sample)** |
| **G6 cold eval used** | **5,929 cold-gold users (the *entire* cold-gold population)** |
| **G6 reach eval used** | **34,539 warm users (14.4× the H1 sample)** |

**Warm-eval attrition (36,754 → 34,539, ~2,215 excluded):** the excluded warm-gold users **have no ALS user-factor row** — they are not in the ALS training user set (`uidx`), so no collaborative score can be produced for them. This is an artifact-coverage limit, not a filtering choice; hence the warm result is reported as the **"largest valid warm heldout evaluation set"** (34,539), not the literal full heldout pool (36,754).

**Why this scale:** ALS factors cover 91,732 users, so the **warm lane scales across nearly the whole heldout pool** cheaply (matrix scoring + history exclusion rebuilt from train). The **cold-gold population is only 5,929 users total** — so evaluating all of them *is* the full cold population (the original 5k sample was already ~84% of it). Cold lists were regenerated with the exact C2 content-hybrid method. All runs completed within sandbox limits (28s / 15s / 24s).

## 2. C2 policy stability (sampled vs full-scale)
| Metric | Sampled (C2/C1) | G6 full-scale | Verdict |
|---|---|---|---|
| ALS-only warm R@20 | 0.052 | **0.0507** | stable |
| ALS-only warm R@50 | — | 0.1019 | new |
| 90/10 warm-lane R@20 | 0.048 | **0.0461** | stable |
| 90/10 warm-lane R@50 | — | 0.0988 | new |
| **warm rel-loss @20 vs ALS** | **−8.1%** | **−9.08%** | **still within ≤10% guardrail** (closer to edge) |
| warm rel-loss @50 vs ALS | — | −3.07% | comfortable |
| content cold-lane R@20 | 0.241 (C1) | 0.2125 | same order; modest drop |
| content cold-lane R@50 | 0.340 (C1) | 0.2898 | same order; modest drop |
| 90/10 policy cold recall (see note) | 0.050 (C2) | **0.0653** | stable/favorable |
| 80/20 policy cold recall (see note) | 0.092 (C2) | 0.1071 | stable/favorable |

**Naming note — "policy cold recall":** this is the **recall of the user's gold cold item within the full 20-slot slate, where only 2 positions are cold-reserved** under 90/10 (4 under 80/20). It is *not* a 20-slot cold lane. The separate **content cold-lane R@20/50** rows above *do* measure the cold lane over its own top-20/50 (the C1-style lane capability). Two different things; both reported.

## 3. Prevalence-weighted stability
At full scale the warm/cold split is **36,754 / 5,929 ≈ 86.1% / 13.9%** in the heldout pool (vs the D2 76.5/23.5 traffic mix — the heldout is more warm-heavy). The qualitative O1b conclusions hold under either weighting:
- **90/10 remains product-rational** — discovery within the warm guardrail (−9.08% at full scale).
- **90/10 is still not proven superior to ALS-only** — this remains an offline candidate; G6 adds scale robustness, not online causality or a separation result.
- **Aggressive arms remain higher-discovery but riskier** — 80/20 cold R@20 0.107 > 90/10 0.065, consistent with its larger (guardrail-breaking) warm cost.

## 4. Creator / catalog health stability
| Metric (warm lane) | ALS-only (34.5k) | 90/10 warm-lane (34.5k) |
|---|---|---|
| distinct items | 4,324 | 4,102 |
| distinct creators | 1,292 | 1,238 |
| creator Gini | 0.908 | 0.907 |
| effective creators | 132.9 | 128.3 |

Plus, from the full cold population: **90/10's 2 cold slots reach 650 distinct cold creators and 931 distinct cold items** (of 1,420).

**Reading (consistent with H1):** on the *warm lane*, ALS-only and 90/10 have ~identical creator concentration (Gini 0.908 vs 0.907) — 90/10 does **not** de-concentrate the warm head at scale either. Its reach gains come from the **cold lane** (650 cold creators), exactly H1's "reach, not de-concentration" finding. *(Absolute Gini differs from H1's 0.845 because that run was 2,400 users at prevalence with a feedback loop; this is 34.5k warm users, warm-lane only, no feedback — **not directly comparable in level, only in direction.** Direction is stable.)*

## 5. Drift analysis
| Metric | Sampled | G6 | Abs Δ | Rel Δ | Interpretation |
|---|---|---|---|---|---|
| ALS-only warm R@20 | 0.052 | 0.0507 | −0.0013 | −2.5% | stable |
| 90/10 warm R@20 | 0.048 | 0.0461 | −0.0019 | −4.0% | stable |
| warm rel-loss @20 | −8.1% | −9.08% | −0.98 pp | — | **holds < 10%, nearer edge** |
| content cold-lane R@20 | 0.241 | 0.2125 | −0.0285 | −11.8% | modest; lane still works (ALS=0) |
| 90/10 policy cold R@20 | 0.050 | 0.0653 | +0.0153 | +30.6% | favorable |
| 80/20 policy cold R@20 | 0.092 | 0.1071 | +0.0151 | +16.4% | favorable |
| creator Gini (warm) als vs q90 | ≈equal (H1 dir) | 0.908 vs 0.907 | ~0 | — | direction holds: no warm de-concentration |

## 6. Stability verdict — **PASS**
- Sampled conclusions are **directionally stable** at full scale: 90/10 stays within the ≤10% warm guardrail, cold value stays positive (policy cold R@20 actually higher), and the reach-not-de-concentration finding holds.
- **One honest caveat:** warm rel-loss drifts from −8.1% to **−9.08%** on the largest valid warm set — still passing, but **closer to the guardrail edge**. Worth flagging for the A/B (B6 prevalence/segment-mix risk in G3); not a HOLD trigger.
- **Decision implication:** because warm loss now sits at −9.08% (≈1 pp from the limit), the online A/B must treat **warm engagement/relevance as a *primary* kill/ramp guardrail, not a secondary metric** — it is the binding constraint and should gate every ramp step.
- No metric collapsed; no runtime/comparability failure. **Not a HOLD.**

## 7. Claim-boundary update
- G6 is **still offline.** No online A/B, no business lift, no deployment, no creator fairness solved.
- G6 **strengthens scale robustness, not online causality** — it shows the offline conclusions don't depend on the 5k/2.4k sampling, nothing more.
- The content-lane drift (−11.8%) and warm-loss drift (toward −9%) are reported, not hidden.

## 8. Interview-safe summary
- "I re-ran the headline metrics at the largest feasible scale — **34.5k warm users (≈7×) and the entire 5.9k cold population** — to check the 5k-sample conclusions held."
- "**The warm guardrail holds at full scale: −9.08% vs the ≤10% budget** — though it drifted slightly toward the edge, which I flag for the A/B."
- "**Cold value is stable to favorable** — the 90/10 policy's cold recall actually came in a bit higher at full scale (0.065 vs 0.050)."
- "The cold population is only ~5,900 users, so my original sample was already ~84% of it — cold metrics were near-full-scale by construction."
- "Creator concentration on the warm lane is **identical** between ALS-only and 90/10 (Gini 0.908 vs 0.907) — confirming at scale that 90/10 buys **reach via the cold lane, not warm-head de-concentration**."
- "Everything stayed **offline** — this is scale robustness, not evidence of online lift."
- "I report the drifts honestly (content-lane −11.8%, warm-loss toward −9%) rather than only the favorable ones."

## 9. Status
- ✅ G6 PASS — conclusions scale-stable; warm guardrail holds at full population.
- ⚠️ PulseDiscover is **not** declared BeastMax-complete after G6 (per the BM2-GAP stop condition: still needs the research dossier, final PDF/deck, and golden-template extraction; G5 advanced-model depth remains a documented deferral).

**STOP — awaiting G6 review.**
