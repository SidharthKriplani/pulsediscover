# 40 — G2: Product / Growth Strategy Memo

*Product/company strategy for PulseDiscover, grounded in the existing offline evidence. **No new code/experiments.** This is a product hypothesis backed by offline results — **no business lift is proven, no online A/B has run.***

---

## 1. Company question
**What product/business problem does PulseDiscover solve, and why should a content platform care about a protected discovery lane?**

Because relevance-only ranking quietly trades away the platform's future: it over-serves known/head content, starves new books and creators of the signal they need to ever rank, and lets the catalog calcify. A protected discovery lane is a small, controllable bet on long-term catalog health and creator supply — without redesigning the feed.

## 2. Product problem
- **Warm/ALS recommenders are safe but over-serve known/head content** — they rank what already has signal, so the head compounds.
- **New books/creators need exposure to get signal** — without impressions they never accumulate the interactions that collaborative ranking requires (cold-start is structural: ALS scores new items 0).
- **Pure relevance optimization can hurt long-term catalog freshness** — every slot spent on a safe pick is a slot not spent discovering tomorrow's hit.
- **The real company problem is a balance, not a single number** — short-term relevance **and** discovery **and** catalog health. "Maximize offline recall" is the wrong objective; PulseDiscover is built around the tradeoff.

## 3. Stakeholders
| Stakeholder | What they want | How PulseDiscover helps | Risk that remains |
|---|---|---|---|
| **Readers / users** | Relevant reads + fresh things to discover | 18 warm slots keep relevance; 2 cold slots add discovery | Cold picks may feel less relevant if cold quality is weak |
| **New / underexposed creators** | A first impression to earn signal | Protected cold lane guarantees exposure share (+444 new creators reached **in the H1 offline simulation**) | **Reach ≠ fairness** — head stays concentrated (H1/H2) |
| **Platform growth / retention** | More engaged, returning sessions | Discovery flywheel *hypothesis* (§5) | Flywheel is unproven; offline ≠ online |
| **Content / catalog ops** | A living, broad catalog | +26% distinct creators, 735 new books surfaced, broad coverage (**in the H1/C2 offline simulations**) | Cold exposure is structural share, not quality-guaranteed |
| **Experimentation / DS** | Trustworthy measurement | OPE + position-bias + logging schema make the test readable | Needs exploration bucket + propensity logging discipline |
| **Business leadership** | Defensible, low-risk bets | Conservative 2-of-20-slot bet with guardrails + ship/kill rules | No proven ROI yet; A/B required |

## 4. Product thesis
**90/10 reserve-2 is a conservative exploration product, not a feed redesign.**
- **18 warm slots** preserve familiar relevance (warm loss held to −8.1% **in C2 offline serving-policy evaluation**, within the ≤10% guardrail).
- **2 protected cold/discovery slots** create guaranteed new-content exposure that survives feedback amplification (H1 offline simulation: cold share held every round).
- It's **deliberately conservative** — a 10% slot reservation, reversible, guardrail-gated — so the downside is bounded and the upside (discovery) is testable.
- It is **A/B-ready, not business-proven** — OPE cannot show it beats ALS-only (O1/O1b); only a live test settles it.

## 5. Discovery flywheel *(hypothesis to test — not proven lift)*
> new-content exposure → reader engagement signal on new items → richer ranking data for those items → creator motivation + catalog freshness → more discoverable inventory → more user discovery → more sessions / retention.

This loop is the **strategic bet** behind the cold reserve. **It is a hypothesis the A/B is designed to test, not a demonstrated effect.** Each arrow is a falsifiable step: if new-content exposure doesn't produce engagement signal (position-corrected), the flywheel breaks at step one and the policy should be held or killed.

## 6. Metrics framework
- **North-star:** retained reading sessions (returning, engaged sessions over a horizon) — **a proposed *online* north-star candidate; it is NOT measured in this offline project.**
- **Primary experiment metric:** prevalence-weighted engagement on the served slate (warm + cold).
- **Secondary:** completed reading depth; discovery-engaged sessions; cold/new-item engagement; new-creator reach; distinct items/creators exposed; catalog coverage.
- **Guardrails (must-hold):** warm engagement loss ≤10% vs control; session abandonment flat; latency/fallback-rate within budget; creator Gini not materially worse (monitor).
- **Diagnostics:** position-corrected cold-lane value (P1); OPE ESS/positivity (O1); creator Gini trajectory (H1); cold-exposure share; fallback rate.

The hierarchy matters: **a CTR bump is not success** — success is north-star retention with the warm guardrail intact and *position-corrected* cold value positive.

## 7. Segment strategy
**Likely to work best:**
- **Heavy readers** — high tolerance for exploration, lots of context to keep warm slots strong.
- **Genre explorers** — actively want variety; discovery aligns with intent.
- **Users with stale recommendations** — warm lane has little new to offer; cold slots add value.
- **Users with author/series affinity** — same-author cold candidates are high-precision (C1: same-author strongest single cold lane).
- **Surfaces or periods with high new-item availability, stale recommendations, or explicit discovery intent** — discovery is the point.
- **New-content launch windows** — when fresh inventory most needs exposure.

**Potentially risky:**
- **New users with little trust** — relevance misses are costlier early; consider warming them first.
- **Highly intent-driven sessions** — a user hunting a specific next read may not want a discovery slot.
- **Sequel/known-author-only users** — low exploration appetite; cold slots may read as noise.

Implication: 90/10 is a strong candidate for a **targeted rollout** (explorers, stale-rec, launch windows) before any blanket deployment.

## 8. Product tradeoffs
- **Short-term CTR vs long-term discovery** — the cold reserve sacrifices a little immediate relevance for catalog/creator futures.
- **Relevance vs exploration** — the 10% guardrail is the explicit price ceiling on exploration.
- **Creator reach vs creator fairness** — PulseDiscover buys reach (H1) but not de-concentration (H2); don't conflate them.
- **Catalog freshness vs low-quality cold impressions** — reserving slots guarantees freshness *share* but not freshness *quality*; cold-lane precision is the safeguard.
- **Personalization vs editorial/business rules** — the reserve is a business rule layered on personalization; tune the ratio per surface.
- **Guardrail-safe rollout vs faster learning** — conservative ramp protects users but slows learning; the exploration bucket trades relevance for measurability.

## 9. Business risks
- **Cold slots may reduce perceived relevance** — if cold quality is weak, users notice the 2 slots.
- **Poor cold quality may hurt trust** — bad discovery is worse than no discovery; precision-gate the cold lane.
- **Creator exposure may rise without fairness** — reach gains could be mistaken for equity (H2 says they aren't).
- **Offline metrics may not translate online** — the entire evidence base is offline proxy reward.
- **Position bias may hide discovery value** — raw CTR under-credits cold 11× (P1); without correction the policy looks worse than it is and could be wrongly killed.
- **Exploration may be underpowered** — insufficient exploration → OPE positivity fails and the test can't conclude (O1).
- **Monetization impact is unknown** — reserving slots for discovery may shift ad inventory, subscription conversion, or content economics in either direction. **Monetization must be monitored as a separate guardrail, not assumed neutral or positive.**

## 10. Product decision logic (ship / hold / kill)
- **SHIP** if the warm guardrail holds (≤10% loss) **and** position-corrected cold/new engagement is significantly positive **and** session quality is intact.
- **HOLD** if CIs overlap but discovery diagnostics (cold engagement, new-creator reach, coverage) trend positive → extend / re-power / narrow to responsive segments.
- **KILL** if warm engagement or session quality breaks, or cold slots show position-corrected null/negative value.
- **CONTINUE RESEARCH** if creator *reach* improves but *concentration* remains unresolved → pursue training-time creator fairness (the H2 lever), not more reranking.

## 11. Claim boundaries
- **No business lift proven.** **No online A/B yet.** **Cold-start not "solved"** (a working candidate lane, offline). **Creator fairness not solved** (reach ≠ de-concentration). **No claim that 90/10 beats ALS-only overall** (OPE can't separate them). This memo is a **product strategy hypothesis backed by offline evidence**, nothing more.

## 12. Interview-safe summary
- "PulseDiscover reframes ranking from 'maximize relevance' to 'balance relevance, discovery, and catalog health' — a company tradeoff, not a metric."
- "The product is a **conservative bet**: 2 of 20 slots reserved for discovery, reversible and guardrail-gated, warm loss held under 10%."
- "The strategic thesis is a **discovery flywheel** — exposure → signal → fresher catalog → more discovery — which I frame explicitly as a **hypothesis to A/B test, not a proven lift**."
- "Success is **retained sessions with the warm guardrail intact and position-corrected cold value positive** — not a CTR bump."
- "I'd roll it out **targeted first** — explorers, stale-rec users, launch windows — before anything broad."
- "It buys creator **reach, not fairness**, and I'm careful never to claim otherwise."
- "Everything is offline evidence — the memo is a product strategy, not a demonstrated business result."

**STOP — awaiting G2 review.**
