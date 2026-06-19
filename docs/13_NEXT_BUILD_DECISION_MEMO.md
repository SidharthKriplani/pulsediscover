# 13 — Next Build Decision Memo

*Decision aid only — no execution. The BeastMax-core spine (Phases 2–7) is built and CI'd on the MovieLens-1M smoke + a known-propensity simulator; the Gold PRD and evidence ledger are updated and correction-passed. This memo compares the three candidate next steps and recommends one. The defense PDF (Phase 10) is deferred until this decision is made.*

## Where we are
- **Built & CI'd:** retrieval baselines, SASRec (smoke), simulator, IPS/SNIPS/DM/DR, position-bias (IPS+PAL), feedback-loop audit.
- **Known limits:** MovieLens is **smoke** (not the fiction domain); OPE π/truth are **ALS-based** (not the SASRec model); compute is CPU-only (3.8 GB RAM, no GPU); I **cannot fetch datasets** (platform policy) — domain data must be user-provided.
- **Biggest PRD weakness:** the **domain-headline gap** (data-realism axis scored 8, gated on a real fiction dataset).

## The three options

### Option A — Goodreads / Amazon domain headline next
Re-run the whole spine (baselines → SASRec → simulator → OPE → position-bias → feedback-loop) on a Goodreads fiction-genre subset (or Amazon Books).
- **Pros:** closes the #1 weakness — converts "smoke" into a **fiction-domain** result with authors (creators) and series (serialization); makes every phase domain-relevant; raises the data-realism + industry-pattern axes to 9; strongest credibility gain.
- **Cons / blockers:** **needs user-provided data** (I can't download; full Goodreads is 4–11 GB > 6.8 GB disk → a genre subset only); SASRec on larger data likely **needs a GPU** (Colab/Kaggle) — marginal on this CPU sandbox; largest effort (re-runs 6 phases); ingestion/processing risk.
- **Effort:** High. **Blocker:** dataset acquisition + possible GPU.

### Option B — SASRec-based π through OPE next
Replace the ALS-based target policy in Phase 5 with the **actual SASRec model's scores** as π (and optionally a SASRec-derived reward model), and re-run IPS/SNIPS/DM/DR.
- **Pros:** **fully in-sandbox, cheap, no data blocker**; directly **closes the "ALS-π" gap** flagged in the ledger; makes the headline claim *"DR evaluates my actual recommender"* instead of *"DR evaluates an ALS policy"* — tightening the core differentiator; small, low-risk, fast.
- **Cons:** still **smoke + simulator-scoped** (no domain, no online lift); incremental narrative lift; doesn't address the domain-headline gap.
- **Effort:** Low. **Blocker:** none.

### Option C — T3 model build (two-tower / LightGCN / PyG / DeepFM / DIN …)
Build one or more deferred deep-learning models.
- **Pros:** breadth / premium-skill coverage; two-tower adds genuine retrieval cold-start.
- **Cons:** **explicitly deferred by design** until the evidence spine + a domain dataset exist (§9.5.1); needs **richer data / GPU** the smoke can't fairly exercise; adds modeling paradigms **before** the domain headline → dilutes focus and risks the "technique museum" the project critiques; **lowest credibility per unit effort** right now.
- **Effort:** Medium–High. **Blocker:** data/GPU; sequencing (should follow A).

## Comparison

| Axis | A — Domain headline | B — SASRec-π OPE | C — T3 models |
|---|---|---|---|
| Closes biggest gap (domain) | ✅ strongest | ❌ | ❌ |
| Strengthens core differentiator (OPE) | ➖ (re-validates) | ✅ direct | ❌ |
| In-sandbox / no blocker | ❌ needs data+GPU | ✅ fully | ❌ needs data+GPU |
| Effort | High | **Low** | Med–High |
| Credibility gain / effort | High but blocked | **High, unblocked** | Low now |
| Risk | ingestion/compute | minimal | scope-dilution |

## Recommendation
**Do Option B next, then Option A; defer Option C.**

- **B first** because it is cheap, fully in-sandbox (no acquisition/GPU blocker), and **directly upgrades the differentiator** — making the OPE harness evaluate the *actual* SASRec policy rather than an ALS stand-in. It closes a real, named gap with ~one phase of low-risk work and improves the headline interview story immediately.
- **A second** as the high-ceiling milestone — the domain headline is the single biggest credibility lift, but it is **blocked on you providing a Goodreads-genre / Amazon Books file** (I can't fetch) and likely a free Colab/Kaggle GPU for SASRec at domain scale. Worth doing, but it needs inputs and more effort, so it shouldn't block the cheap win in B.
- **C last** — the T3 deep-learning roster stays roadmap until after A; building paradigms before the domain headline is exactly the over-engineering the PRD argues against.

**Suggested sequence:** B (in-sandbox, now) → A (when you supply a domain dataset + GPU) → C (post-domain) → defense PDF (Phase 10) once A is in hand so the PDF is domain-grounded, not smoke-only.

*No step executed. Awaiting your choice.*
