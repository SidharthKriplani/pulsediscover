# 60 — G14: Final Slate Policy + MMR / Catalog-Guardrail Reranker

*The third pipeline stage: candidate generation (G13-mid) → LTR ranking (G12-B) → **final top-20 slate policy** (this gate). Five policies over the LTR-ranked mid candidate list; relevance vs governance measured on the deterministic test split (n=733). Offline re-ranking within candidates — **no online lift, no production, no RiskFrame-gold-complete claim.** Evidence: `g14_final_slate_policy_report.json`; plot `g14_relevance_governance_tradeoff.png`.*

## 1. Results (top-20 slate, test n=733; guardrail = ≤10% R@20 loss vs pure LTR)
| Policy | R@20 | rel-loss | cold-exp share | distinct cold | distinct creators | creator Gini | novelty | violation rate |
|---|---|---|---|---|---|---|---|---|
| **pure LTR** | **0.1201** | 0% | 0.282 | 586 | 573 | 0.864 | 0.618 | 0.000 |
| LTR + MMR (λ=0.7) | 0.1146 | −4.5% | 0.204 | 516 | 677 | 0.864 | 0.525 | 0.000 |
| LTR + cold reserve (2) | 0.1187 | −1.1% | 0.293 | 613 | 598 | 0.864 | 0.630 | 0.003 |
| LTR + creator cap (2) | 0.1091 | −9.1% | 0.222 | 544 | 673 | 0.847 | 0.553 | 0.000 |
| combined | 0.1132 | −5.7% | 0.227 | 550 | 714 | 0.851 | 0.553 | **0.207** |

## 2. Findings (honest)
1. **The candidate strategy already delivered the discovery — leaving the slate layer little headroom.** Pure LTR already surfaces **28.2% cold exposure, 586 distinct cold items, novelty 0.62** — because the G13-mid candidate set feeds content/cold candidates and the LTR ranks many highly. So there isn't a "warm-only, needs-a-discovery-reserve" problem to fix at the slate stage; G13 moved discovery upstream.
2. **No policy materially de-concentrates the creator head without a relevance cost.** Creator Gini barely moves: 0.864 → best **0.847** (creator cap), and that costs **−9.1% R@20** (guardrail edge). MMR adds distinct creators (573→677) but **doesn't lower Gini at all** (0.864) and costs relevance + cold exposure + novelty. Consistent with **H1/H2: head de-concentration is structurally hard and trades against relevance.**
3. **Cold reserve is nearly free but nearly pointless here.** −1.1% relevance for +1pp cold share (0.282→0.293) — because pure LTR is already cold-heavy. A safe no-op-ish guarantee, not a real gain.
4. **The combined policy has a 21% self-violation rate** — my greedy can't always satisfy creator-cap AND cold-reserve within 20 slots (the post-hoc cold injection can break the cap). An honest implementation limitation: exact multi-constraint slates need a proper constrained optimizer, not greedy.

## 3. Verdict — **honest negative for aggressive governance reranking; recommend the soft policy**
Per the decision rule (adopt only if governance improves without unacceptable relevance loss): **no policy clears the bar convincingly.** Governance gains are small (Gini −0.017 at best) and come with relevance loss; the candidate strategy already captured the discovery upside.
- **Recommended default: pure LTR, optionally + light cold reserve (−1.1%, safe)** — keeps relevance, retains strong discovery, no governance downside.
- **creator cap (2/creator)** — adopt **only if head de-concentration is an explicit product goal**; it's the best Gini lever (0.847) but sits at the −9.1% guardrail edge for a modest gain.
- **MMR and combined: not recommended** — MMR trades relevance/cold/novelty for distinct-creator count with no Gini improvement; combined violates its own constraints 21% of the time (needs a constrained optimizer).

## 4. Why this is the right (honest) outcome
It would be easy to declare a fancy "combined governance policy" the winner. The honest reading is the opposite: **once candidate generation surfaces discovery and the LTR ranks it well, bolting on slate-level governance buys little and costs relevance.** The lever for catalog health is *upstream* (candidate mix) and *training-time* (the H2 conclusion), not a post-hoc slate reranker. PulseDiscover's discovery now lives in the candidate stage, where it's cheaper.

## 5. Claim boundaries
- Offline re-ranking within candidates; governance metrics on top-20 slates over test users; **no online lift, no production, no RiskFrame-gold-complete claim.**
- Deterministic md5 split; reproducible. Combined-policy violation rate reported, not hidden.

## 6. Status & next
- ✅ G14 done. **Slate-policy layer built and stress-tested; honest negative for aggressive governance reranking — recommend pure LTR (+ optional light cold reserve). Discovery/health is best handled upstream (candidates) and at training-time, not at the slate.**
- ⏭️ Remaining path to gold: **G15** LightGCN/co-occurrence candidate depth (raise the ceiling further), **G16** FAISS serving-latency demo, **G17** Deep Defense Kernel, **G18** Final Gold Audit.

---

## 7. Acceptance addendum (ACCEPTED — PASS as honest negative)
- **Final default = pure LTR.** R@20 0.1201; already discovery-rich (cold exposure 0.282, 586 distinct cold items, novelty 0.618) because discovery was handled upstream in G13 candidate generation.
- **Optional soft policy = cold reserve (2):** R@20 0.1187 (**−1.1%**), slightly higher cold exposure — acceptable as a light guarantee, not a real gain.
- **Creator cap (2): explicit-product-goal only.** Reduces creator Gini 0.864 → 0.847 but costs **−9.1% R@20**. Use only when head de-concentration is a stated objective.
- **MMR and combined: NOT recommended.** MMR costs relevance/cold/novelty with no Gini improvement. **Combined is rejected** — 20.7% self-violation rate; exact multi-constraint slates need a **constrained optimizer**, not greedy.
- **Plot title corrected:** Gini is lower-is-healthier, so the ideal is **higher R@20 AND lower Gini** (not "top-right"); the plot now marks the ideal direction explicitly.

**Safe interview line:** *"I tested final-slate governance after the LTR ranker. The surprising result was that pure LTR was already discovery-rich because discovery had been handled upstream in candidate generation. Aggressive post-hoc rerankers improved governance only marginally and often cost relevance, so I kept pure LTR as default with an optional light cold reserve. This was an offline slate-policy test, not an online lift claim."*

**Claim boundary (unchanged):** offline re-ranking within candidates; no online lift, no production deployment, no RiskFrame-gold-complete claim.

**STOP — G14 accepted (pure LTR = default); G15 proposed (docs/61), awaiting approval.**
