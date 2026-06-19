# 41 — G3: Risk Register

*Company-realistic risk register for PulseDiscover's 90/10 discovery policy. Grounded in existing evidence (C1/C2/O1/O1b/P1/H1/H2/G1/G2). **No new code/experiments. No deployment or online-A/B claim; 90/10 is not proven; creator fairness is not solved.** Every risk ties to PulseDiscover evidence or a stated claim boundary.*

---

## 1. Company question
**If a platform tested or shipped the 90/10 discovery policy, what could go wrong — and how would the team detect, mitigate, or kill it?** The job here is to show PulseDiscover is *governable*: named risks, detection metrics, mitigations, and explicit ship/hold/kill links.

## 2. Risk register

### A. Product / user risks
| ID | Description | Why it matters | Evidence/source | L | I | Detection metric | Mitigation | Ship/Hold/Kill |
|---|---|---|---|---|---|---|---|---|
| A1 | Cold slots reduce perceived relevance | 2 discovery slots may feel off vs control | C2 warm −8.1% (offline); G2 §9 | M | M | warm engagement loss vs control | keep ≤10% guardrail; precision-gate cold lane | **Kill** if warm loss >10% |
| A2 | Session abandonment rises | Discovery could break flow | G2 §6 guardrails | M | H | session abandonment rate | guardrail-gate; targeted rollout | **Kill** if abandonment rises significantly |
| A3 | Helps explorers, hurts intent-driven users | One policy ≠ all sessions | G2 §7 segments | M | M | segmented engagement (explorer vs intent) | segment/contextual targeting | **Hold** → narrow to responsive segments |
| A4 | Cold-lane quality varies by segment | Same-author strong, new-author weak | C1 (same-author R@20 0.21 vs TF-IDF tail) | M | M | cold engagement by segment | route to high-precision cold sources first | **Hold**; improve cold lane |

### B. Measurement / evaluation risks
| ID | Description | Why it matters | Evidence/source | L | I | Detection metric | Mitigation | Ship/Hold/Kill |
|---|---|---|---|---|---|---|---|---|
| B1 | Offline metrics don't translate online | Entire evidence base is offline proxy | O1/O1b; G2 §11 | H | H | online vs offline metric gap | treat offline as candidate-only; A/B decides | **Hold** until A/B reads out |
| B2 | OPE positivity fails (low exploration) | ESS collapses → unreliable estimates | O1 (ESS 1.4k vs 7.6k; w_max 290) | M | H | ESS, max importance weight | structured exploration bucket (**~30–40% cold is an example design estimate, tuned by power, relevance, and segment guardrails**) | **Hold**; fix logger before trusting OPE |
| B3 | IPS/DR variance misleads | High variance → false ranking | O1; P1 (per-item IPS MSE worse) | M | M | estimator variance, CI width | SNIPS/clipping; trust aggregate not per-item | **Hold**; don't ship on noisy estimate |
| B4 | Position bias hides cold-lane value | Raw CTR under-credits cold ~11× | P1 | H | H | naive vs position-corrected cold value | mandatory IPS examination correction | **Kill-guard**: never decide on raw CTR |
| B5 | Overlapping CIs → ambiguous decision | OPE can't separate 90/10 from ALS | O1b (`dr_separates=False`) | H | M | CI overlap on primary metric | power the A/B; pre-register decision rule | **Hold**; extend/re-power |
| B6 | Online warm/cold prevalence or segment mix differs from offline D2 mix | O1b showed prevalence correction compressed the discovery advantage, so a different live mix could change 90/10's apparent value | D2 warm/cold prevalence; O1b prevalence-correct OPE | M | M/H | live warm/cold traffic mix, segment mix, treatment/control balance by segment | stratified readout, prevalence-weighted analysis, segment-level decision rules | **Hold** if result depends on unstable or mismatched prevalence assumptions |

### C. Model / retrieval risks
| ID | Description | Why it matters | Evidence/source | L | I | Detection metric | Mitigation | Ship/Hold/Kill |
|---|---|---|---|---|---|---|---|---|
| C1r | ALS stale / warm relevance degrades | Warm lane is the relevance backbone | G1 §8 (stale factors) | M | H | warm recall drift vs refit | refit cadence + drift monitor | **Kill** if warm breaks guardrail |
| C2r | Cold index stale | "New" items become old; cold value decays | G1 §8 | M | M | cold-item age distribution | freshness SLA + alert | **Hold**; refresh index |
| C3r | Content-hybrid overuses same-author | Narrow "more of the same author" discovery | C1 (same-author dominant lane) | M | M | author-diversity of cold slots | cap same-author share in hybrid | **Hold**; rebalance hybrid |
| C4r | Cold lane surfaces low-quality/repetitive items | Bad discovery erodes trust | G2 §9 | M | H | cold-item engagement/quality flags | precision threshold; quality filter | **Kill** if cold value null/negative |
| C5r | SASRec negative leaves model-depth gap | No competitive advanced model yet | D3B (R@20 0.036 < ALS 0.085) | M | M | benchmark vs floor (future GPU run) | canonical GPU SASRec or documented deferral | n/a (research item, not a ship blocker) |

### D. Creator / catalog risks
| ID | Description | Why it matters | Evidence/source | L | I | Detection metric | Mitigation | Ship/Hold/Kill |
|---|---|---|---|---|---|---|---|---|
| D1 | Reach improves but head concentration remains | Discovery ≠ fairness | H1 (Gini flat-to-worse) | H | M | creator Gini, effective creators | don't claim fairness; pursue training-time fix | **Continue research** (not ship/kill) |
| D2 | Creator Gini worsens despite more distinct creators | Long tail of tiny-exposure creators | H1 (Gini 0.845→0.861) | M | M | Gini vs distinct-creator count | monitor both; interpret jointly | **Hold** the fairness claim, not the policy |
| D3 | Exposure mistaken for fairness | Stakeholder misread | H2; G2 §11 | M | M | claim-review in readouts | enforce reach≠de-concentration language | governance, not ship |
| D4 | Feedback loops amplify popularity | Head self-concentrates over rounds | H1 (Gini rises every round) | M | H | creator Gini trajectory, coverage | protected cold reserve; anti-amplification monitor | **Hold**; add de-concentration lever |

### E. Infra / logging risks
| ID | Description | Why it matters | Evidence/source | L | I | Detection metric | Mitigation | Ship/Hold/Kill |
|---|---|---|---|---|---|---|---|---|
| E1 | Missing propensities break OPE | Can't correct or evaluate | G1 §8; P1 schema | M | H | propensity null rate | propensity logging as hard serving contract | **Kill-guard**: block test if propensities missing |
| E2 | Logging cost too high | Budget overrun | G1 §5 | M | M | log volume vs budget | tiered/sampled logging (full on exploration bucket) | **Hold**; tier logging |
| E3 | Privacy/compliance retention limits logs | Legal constraint on data | G1 §5 (retention/PII) | M | M | retention-policy compliance | PII minimization, retention windows, aggregate-safe | governance gate |
| E4 | Latency timeout / fallback overuse | Degraded slates | G1 §4,§8 | L | M | fallback rate, p95 latency | cache/popularity fallback; time-box cold lane | **Hold** if fallback rate high |
| E5 | Nearline freshness fails (session context) | Stale in-session recs at scale | G1 §6 (large-platform note) | L | M | freshness lag of session signal | nearline session folding | **Hold** at scale |

### F. Rollout / business risks
| ID | Description | Why it matters | Evidence/source | L | I | Detection metric | Mitigation | Ship/Hold/Kill |
|---|---|---|---|---|---|---|---|---|
| F1 | Broad rollout too early | Exposes all users before learning | G2 §7 | M | H | rollout %, guardrail status | targeted → ramped rollout | **Hold** ramp until guardrails hold |
| F2 | Segment targeting too narrow to learn | Underpowered, no conclusion | G2 §7; B5 | M | M | sample size / power per segment | size for power before narrowing | **Hold**; re-power |
| F3 | Monetization impact unknown | Discovery may shift ads/subs/economics | G2 §9 (monetization caution) | M | H | revenue/monetization guardrail | monitor monetization separately, not assumed | **Kill** if monetization breaks materially |
| F4 | Stakeholders misread 90/10 as proven winner | Overclaim risk | O1b; CLAIM1 | H | M | claim review in decks/readouts | enforce "candidate not proven" wording | governance |
| F5 | Team optimizes CTR instead of retention/discovery | Wrong objective; position-bias trap | P1; G2 §6 | M | H | metric-of-record audit | north-star = retained sessions; position-corrected | governance |

## 3. Risk severity summary — top 5
1. **B4 — position bias hides cold-lane value** (H/H). Raw CTR under-credits cold 11× (P1); without correction the team could *kill the right policy* on an artifact. The single most dangerous measurement trap.
2. **B1 — offline doesn't translate online** (H/H). All evidence is offline proxy; over-trusting it is the core epistemic risk. Forces "candidate-only until A/B."
3. **F3 — monetization impact unknown** (M/H). Discovery slots touch revenue surfaces; an unmonitored monetization regression could sink an otherwise "successful" test.
4. **B5 — overlapping CIs → ambiguous decision** (H/M). OPE can't separate 90/10 from control (O1b); without a pre-powered, pre-registered A/B the decision is unresolvable.
5. **D1/D4 — reach without de-concentration + amplification** (H/M–H). The fairness story is easy to overclaim and the feedback loop amplifies the head (H1); both need honest framing and monitoring, not a fairness claim.

## 4. Risk controls
- **Pre-launch checks:** propensity logging verified (E1); exploration bucket sized for OPE positivity (B2); position-bias correction wired in (B4); guardrail metrics + monetization guardrail defined (A1,F3); claim-language review (F4).
- **During-test monitors:** warm engagement loss, session abandonment, fallback rate, ESS/positivity, creator Gini trajectory, monetization, cold-item age.
- **Post-test analysis:** position-corrected cold value (not raw CTR); segmented readout; CI/power check; offline-vs-online gap.
- **Hard kill conditions:** warm guardrail >10% loss; session quality/abandonment breaks; position-corrected cold value null/negative; monetization materially breaks; propensities missing (test invalid).
- **Hold / extend conditions:** CIs overlap but discovery diagnostics trend positive; underpowered segments; logger positivity weak.
- **Safe-to-ramp conditions:** guardrails hold on a targeted segment, position-corrected cold value positive, monetization neutral-or-positive, monitoring green → widen gradually.

## 5. Owner map
| Owner | Primary risks |
|---|---|
| **DS / ML** | B1–B5, C1r–C5r, D1–D4 (measurement, retrieval, health) |
| **Product manager** | A1–A4, F1–F5 (user experience, rollout, objective discipline) |
| **Engineering** | C1r/C2r (freshness), E1/E4/E5 (serving, latency, nearline) |
| **Data platform / experimentation** | B2, E1–E2 (logging, propensities, ESS, exploration bucket) |
| **Trust / safety / policy** | E3 (privacy/retention), D3 (fairness framing) |
| **Business leadership** | F3 (monetization), F4 (claim discipline), ramp decisions |

## 6. Interview-safe summary
- "I built a real risk register, not a generic one — every risk ties back to a PulseDiscover result or claim boundary."
- "The #1 risk is **measurement, not modeling**: raw CTR under-credits the cold lane 11×, so you could kill the right policy on a position artifact."
- "Everything's offline, so the core control is treating results as **candidate-only until the A/B** — I never let offline stand in for lift."
- "I flag **monetization as a separate guardrail** — discovery slots touch revenue surfaces and can't be assumed neutral."
- "On creators I'm careful: the policy buys **reach, not fairness**, and the feedback loop amplifies the head — both are monitored, neither is claimed solved."
- "Each risk has a **detection metric and a ship/hold/kill implication**, plus hard kill conditions like a broken warm guardrail or missing propensities."
- "Ownership is mapped across DS, product, eng, data platform, trust/safety, and leadership — it's a governable system, not just an interesting model."

**STOP — awaiting G3 review.**
