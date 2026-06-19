# 34 — DEF1: Final Interview Defense Manual

*Spoken-answer manual for PulseDiscover. Strictly aligned with CLAIM1 (docs/33) and PACK1 (docs/32). All claims are offline/candidate-level — no executed A/B, no business lift. Use interview-safe wording only.*

---

## 1. 60-second project story

**Plain English.** Recommenders tend to keep serving the safe, popular stuff people already engage with, and they're bad at surfacing brand-new content. I built PulseDiscover on a 2.5M-interaction Goodreads dataset: a tuned collaborative model for "warm" recommendations, plus a separate content-based lane for brand-new books the collaborative model can't see at all. Then I designed a serving policy that reserves a couple of slots for discovery, and — the part I'm proudest of — a measurement layer that stops us from drawing wrong conclusions when we test it.

**Technical.** Warm retrieval is ALS (implicit MF), tuned to f64 — R@20 0.085, my best single generator. New items get zero from ALS by construction, so I added a content-hybrid lane (same-author + same-series + TF-IDF via RRF) that hits cold R@20 0.24 with full cold-pool coverage. The serving policy is a two-lane "90/10 reserve-2": 18 warm + 2 cold slots, which keeps warm-recall loss within a 10% guardrail (−8.1%). On top I built off-policy evaluation with known propensities, prevalence correction to realistic 76.5/23.5 traffic, position-bias correction (PBM/IPS), and a catalog-health simulation. All offline — it's A/B-ready, not A/B-proven.

**Stakeholder-with-money (10 seconds).** "We recommend books people already like *and* surface new ones, without hurting quality — and I built the measurement to prove it safely before we bet money on it."

---

## 2. Core defense spine (one connected story)
- **Problem:** recommenders over-serve safe/warm content and fail on new/cold content — bad for discovery, new creators, and catalog freshness.
- **Warm solution:** ALS f64, tuned and CI-evaluated — the collaborative floor (R@20 0.085).
- **Cold solution:** content-hybrid lane — non-zero cold recall (0.24@20) where ALS is structurally 0.
- **Product policy:** 90/10 reserve-2 — discovery within the ≤10% warm guardrail.
- **Governance:** OPE (with logger-positivity diagnosis), prevalence correction, position-bias correction, creator/catalog-health + reranking analysis — so we don't ship on biased or misread signals.
- **Final status:** an A/B-ready offline candidate with a complete, honest evidence trail — not proven lift, not deployed.

---

## 3. Likely questions & answers

**Why does this problem matter?** Discovery is how platforms keep catalogs fresh and give new creators a shot. Pure collaborative systems create rich-get-richer loops; if you don't deliberately reserve discovery, the head eats everything and new content never surfaces.

**Why ALS?** It's the strongest, most reliable signal on this data — implicit MF on confidence-weighted interactions. I tuned factors f16→f64 and it beat everything else I built (R@20 0.085). It's fast, well-understood, and a fair floor to test against.

**Why not just popularity / co-occurrence?** I tested both. Popularity R@20 ~0.035, co-occurrence ~0.05 — both well below ALS (0.085). They're useful as fallbacks and inside the hybrid head, but they're not the warm floor.

**Why did SASRec fail?** I trained it under CPU constraints with last-position supervision — about 30× less training signal than canonical full-position SASRec — and it landed at R@20 0.036, below ALS f64. I diagnosed it rather than buried it: last-position training, under-training, small config. The honest read is "not competitive *as I built it*," and I scoped a canonical GPU run to settle it properly.

**Why 90/10 and not 80/20 or RRF?** It's the only discovery policy that stays inside the 10% warm-relevance guardrail (−8.1%). 80/20 costs −17%, RRF −44%. Those have higher discovery value but break the guardrail, so they're exploration arms, not the default.

**Is 90/10 proven better than ALS-only?** No — and I'm careful here. My OPE can't separate them; the DR estimates overlap. It's a *product-rational* candidate — it adds discovery within guardrail — but only a live A/B can prove the edge.

**How did you evaluate cold-start?** Leave-last-out on genuinely new-in-test items, Recall@K with bootstrap CIs, plus cold-pool coverage. ALS scores 0 by construction; the content-hybrid gets 0.24@20 and surfaces 100% of the cold pool — that's the lane's justification.

**What is OPE and why did logger positivity matter?** Off-policy evaluation estimates a new policy's value from logs of a different policy, using importance weights. It only works if the logging policy actually explored the actions you care about. My ALS-heavy logger barely showed cold items → tiny effective sample, 3× biased estimates. A logger with structured cold exploration fixed it (ESS 1.4k→7.6k). Lesson: positivity is the gating factor, not the estimator choice.

**Why prevalence correction?** My first OPE used a 50/50 warm/cold population, which over-weights discovery. Re-weighting to realistic 76.5/23.5 traffic compressed the field and confirmed 90/10 and control are indistinguishable. Without it I'd have overstated discovery's value.

**What is position bias and why is raw CTR unsafe?** Users examine top slots far more than the bottom. My cold slots sit at the bottom, so raw CTR under-credited the cold lane 11× — it looked dead when it was actually the highest-relevance slot. IPS examination-correction recovered ~77% at the aggregate. If you ship on raw CTR you'd kill the discovery lane on a measurement artifact.

**Did you solve creator fairness?** No. I improved creator *reach* — 90/10 surfaces +26% creators and 444 brand-new ones — but it doesn't *de-concentrate* the head. I'm explicit about that distinction.

**Why did creator-aware reranking fail?** The mechanism works — caps de-concentrate the warm head — but the cold reserve already spends the entire 10% relevance budget, so any cap strong enough to lower concentration costs −18% to −36% recall. Real de-concentration needs a training-time fix, not a post-hoc cap.

**What would you A/B test?** Control = ALS-only, default arm = 90/10, optional exploration arms = 80/20 / RRF. User-level randomization, primary = prevalence-weighted engagement, guardrail = ≤10% warm loss, with the position-corrected cold-lane value as the deciding secondary.

**What makes you ship / hold / kill?** Ship 90/10 if the warm guardrail holds and position-corrected cold value is significantly positive. Hold if cold value is positive but underpowered. Kill if the warm guardrail breaks or corrected cold value is null/negative.

**Biggest limitations?** It's all offline — no business lift. SASRec is a documented negative. Examination propensity is known only because it's a simulator. Creator fairness is unsolved. Per-item IPS is variance-heavy.

**What would you build next?** Pick one: a canonical GPU SASRec to close the tournament gap honestly, full-population evaluation, or formalizing the A/B into a runnable design. I'd deliberately stop after one rather than overbuild.

---

## 4. Trap questions
| Trap | Risky answer | Safe answer | Why safe is better |
|---|---|---|---|
| "So your model improved business metrics?" | "Yes, it lifts discovery." | "No — everything's offline. It's an A/B-ready candidate; only a live test proves lift." | Claiming lift with no A/B is the fastest way to lose credibility. |
| "So SASRec is worse than ALS?" | "Yeah, transformers are overhyped." | "Worse *as I built it* — CPU/last-position. I diagnosed it and scoped a proper GPU run." | Bounds the claim to my setup; shows diagnosis, not dismissal. |
| "So 90/10 is the winner?" | "Yes, it's the best policy." | "It's the best *guardrail-passing candidate*. OPE can't prove it beats control — the A/B decides." | Avoids claiming a win the evidence doesn't support. |
| "So you solved cold-start?" | "Yes, 0.24 recall." | "I built a working cold-start *channel* offline — not a proven business win." | Channel ≠ solved; keeps it honest. |
| "So you solved creator fairness?" | "Mostly, reach is way up." | "No. I improved reach, not de-concentration — and I showed reranking can't fix it cheaply." | Reach vs de-concentration is the exact distinction a senior reviewer probes. |
| "Why didn't you deploy it?" | "No time / no infra." | "Scope was a rigorous offline experimentation program; deployment needs the live A/B and infra, which I specced but didn't run." | Reframes as deliberate scope, not a gap. |
| "Why no LightGCN / two-tower / BERT4Rec?" | "Didn't get to them." | "Diminishing returns for the portfolio — the value was the governance layer, not a 7th model. They're scoped as optional extensions." | Shows prioritization, not omission. |
| "Is this just a toy project?" | "It's just a demo." | "It's offline, but it's a real 2.5M-interaction domain with OPE, prevalence and position-bias correction, and honest negatives — the parts most toy projects skip." | Reframes around the differentiator. |

---

## 5. Seniority signals
- **Not hiding negatives** — SASRec and the reranking failure are reported with full diagnosis.
- **Candidate evidence vs business lift** — every positive claim is bounded; "proven" is reserved for the A/B.
- **Guardrail-based ship/kill** — decisions gated on a fixed warm-loss budget, not vibes.
- **Measurement before the test** — logging schema, positivity, and position-bias correction designed *before* any A/B.
- **Position-bias awareness** — caught that raw CTR would have killed the right policy.
- **Creator-health caveats** — reach vs de-concentration separated; fairness left honestly open.
- **Stop-control** — explicit "package and stop" instead of endless model-adding.

---

## 6. Concise answer bank (fast recall)
1. **What is it?** "An offline, A/B-ready RecSys experimentation program on 2.5M Goodreads interactions — warm + cold lanes plus a full governance layer."
2. **Warm model?** "ALS f64, R@20 0.085 — my tuned collaborative floor."
3. **Cold model?** "Content-hybrid lane, cold R@20 0.24, where ALS is structurally zero."
4. **The policy?** "90/10 reserve-2 — 18 warm + 2 cold slots, −8.1% warm, inside the 10% guardrail."
5. **Is 90/10 proven?** "No — product-rational candidate; OPE can't separate it from control."
6. **SASRec?** "Honest negative under CPU/last-position; diagnosed, GPU run scoped."
7. **Why not popularity?** "Tested it — 0.035 vs ALS 0.085. Fallback only."
8. **OPE in one line?** "Estimate a new policy from old logs; only trustworthy if the logger explored — positivity was decisive."
9. **Prevalence correction?** "Re-weighted to real 76.5/23.5 traffic; it showed discovery's value was inflated by my 50/50 sample."
10. **Position bias?** "Raw CTR under-credited the cold slots 11×; IPS recovered most of it."
11. **Why raw CTR unsafe?** "You'd kill the cold lane on a bottom-of-page artifact."
12. **Creator fairness?** "Improved reach, not de-concentration — not solved."
13. **Why reranking failed?** "Cold reserve spends the relevance budget; caps cost 18–36% recall."
14. **A/B design?** "Control ALS-only, arm 90/10, user-level, ≤10% warm guardrail, position-corrected cold value as the decider."
15. **Ship/hold/kill?** "Ship if warm holds and corrected cold value is significantly positive; kill if warm breaks or cold value is null."
16. **Biggest limitation?** "Offline — no business lift; everything is candidate-level."
17. **Next step?** "One of: canonical GPU SASRec, full-population eval, or a runnable A/B design — then stop."
18. **Why is it senior-level?** "Honest negatives, guardrail ship logic, measurement-before-test, and knowing when to stop."
19. **Toy project?** "Real domain, real OPE/position-bias/prevalence work, documented negatives — not a toy."
20. **One-line ceiling?** "Offline, evidence-backed, A/B-ready — nothing claimed as proven lift or deployed."

**STOP — awaiting DEF1 review.**
