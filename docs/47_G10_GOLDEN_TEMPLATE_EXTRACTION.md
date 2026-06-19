# 47 — G10: BeastMax Golden Template (extracted from PulseDiscover)

*The reusable method behind PulseDiscover, abstracted so future portfolio lanes (PulseSignal, PulseGuard, PulseKnowledge) can run the same gated, evidence-backed, claim-disciplined program. **Method only — no claims about those future projects.***

---

## 1. Gate structure (the spine)
Each unit of work is a **gate**: one company question, one approved scope, one report, one stop. Pattern:
1. **Company question** — what business decision does this answer? (never "train a model" — always "should we ship/serve/trust X?")
2. **Approved scope only** — do exactly the named gate; no scope creep.
3. **Evidence** — run the smallest experiment that answers the question; tag every claim.
4. **Honest decision** — ship-candidate / hold / kill / documented-negative / deferral.
5. **Stop** — write the named report, present, await review. One gate at a time.

Phase arc that worked: **spine (data) → baselines → honest negative → capability gate → policy gate → measurement/governance gates → company-realism gates (infra/product/risk/rollout) → scale-stability → capstone synthesis (dossier/deck/template) → closeout.**

## 2. Evidence-ladder format
One subsection per gate, five fixed fields: **question asked · method · key result · interpretation · claim boundary.** Keep results quantified with CIs/segments; keep the boundary explicit. The ladder is the backbone of the dossier (G8 §5) and the appendix of the deck.

## 3. Claim-boundary table format
Five columns: **supported claim · evidence/path · strength (strong/moderate/limited) · what we must NOT claim · interview-safe wording.** Rule: mark well-diagnosed **negatives as strong** and cap unproven **positives at moderate** with "candidate, not proven" language. This asymmetry is what makes it defensible.

## 4. Model-depth decision pattern
Don't chase models. Cap advanced modeling at **one** competitive entry; everything else is T3 or reject. Decide via an option table (question answered / value / cost-feasibility / model-chasing risk / acceptance criteria / kill condition). If resources are missing, take a **documented deferral** — and state explicitly that *deferral satisfies claim-honesty but is not an advanced-model win.*

## 5. OPE / measurement-governance pattern
Before trusting any policy comparison: (a) **known-propensity OPE** (IPS/SNIPS/DM/DR vs known truth); (b) **positivity/ESS diagnosis** — the logger makes or breaks OPE; (c) **prevalence correction** to realistic traffic; (d) **position-bias correction** (PBM/IPS) — never decide on raw CTR; (e) state plainly when estimators **cannot separate** candidates. Measurement rigor is the portfolio differentiator, not model count.

## 6. Infra / product / risk / rollout pattern
Make every project read as a company decision, not just a model:
- **Infra (G1):** serving architecture, online/offline split, latency budget (labeled estimates), cost drivers, failure modes, privacy/compliance.
- **Product (G2):** stakeholders, the value thesis/flywheel (labeled hypothesis), metrics hierarchy with a north-star and guardrails, monetization caution.
- **Risk (G3):** risk table tied to *your own evidence*, with detection metric + ship/hold/kill per risk; rank a top-5.
- **Rollout (G4):** phased (logging QA → canary → targeted A/B → ramp → long-horizon), ITT primary readout, propensity-logging contract, pre-registered decision rule.

## 7. G6 scale-stability pattern
Before the capstone, re-run the headline metrics at the **largest feasible scale** using existing artifacts; produce a **sampled-vs-scale drift table** and a PASS/HOLD verdict. Report drift honestly (including toward-the-edge guardrails) and label what is full-population vs largest-valid-set. Cheap rigor that pre-empts "did it only hold on your sample?"

## 8. Final dossier / deck structure
- **Dossier (G8):** abstract → problem → data/eval → system → evidence ladder → synthesized findings → decision logic → limitations → future work → claim table → interview defense → conclusion.
- **Deck (G9):** ~15 slides (title → problem → data → architecture → warm → cold → policy → OPE → position-bias → health → scale → company-realism → claim boundaries → decision → appendix map), each with objective/key-message/visual/bullets/evidence/boundary; plus exec page, 90s/3m/5m scripts, appendix plan, claim checklist.
- **Closeout:** checkpoint status, what's done, optional extensions, hard stop.

## 9. Reuse for PulseSignal / PulseGuard / PulseKnowledge
The template is domain-agnostic; swap the substrate, keep the spine:
| Element | PulseDiscover | Reuse hint for the next lane |
|---|---|---|
| Company question per gate | "serve a discovery lane?" | restate as the new domain's ship decision |
| Spine (data) | Goodreads global-time split | any time-honest domain dataset |
| Baselines + honest negative | ALS floor + SASRec negative | establish a tuned floor; report the negative you find |
| Capability gate | cold-start lane | the new domain's hard capability |
| Policy gate | 90/10 reserve-2 | the new domain's serving rule + guardrail |
| Measurement governance | OPE/positivity/prevalence/position-bias | reuse wholesale wherever there are logged decisions |
| Company realism | infra/product/risk/rollout | reuse the four-memo pattern verbatim |
| Scale-stability | G6 7× re-run | re-run headline metrics at scale |
| Capstone | dossier/deck/template | same three artifacts |

**Guiding rules carried forward:** one gate at a time; tag every claim; document negatives; cap models at one competitive entry; never claim lift/deployment/"solved"; become strong, package, **stop**.

**STOP — G10 golden template extracted.**
