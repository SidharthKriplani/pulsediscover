# 62 — G15: Additional Candidate-Source Depth (Co-occurrence; LightGCN deferred)

*Tests whether an additional collaborative/graph candidate source raises the G13-mid ceiling and converts to higher LTR R@20. **Co-occurrence built and evaluated; LightGCN deferred** (graph-embedding training not cheap on CPU at 40k items). Offline re-ranking within candidates; deterministic split. No online/production/gold claim; no FAISS. Evidence: `g15_candidate_source_report.json`; plot `g15_coverage_contribution.png`.*

## 1. Co-occurrence generator
Item-item co-occurrence from train baskets, vocab capped to **top-6k popular items** (memory-safe on 3.8GB; full-40k OOMs — consistent with earlier D3A findings). Co-occ matrix 6000×6000 (nnz 19.3M, built in 3.4s), top-40 neighbors/item (min co-count 5). Per eval user: aggregate co-occ neighbors of history → top-100 co-occ candidates.

## 2. Coverage & overlap (5,000 users)
| Quantity | Value |
|---|---|
| G13-mid coverage | 0.3228 |
| co-occ-only coverage | 0.1454 |
| **mid + co-occ coverage** | **0.3604** |
| **marginal coverage added by co-occ** | **+0.0376** (3.76pp golds mid misses) |
| mean overlap of co-occ candidates with mid | **0.697** (70% redundant) |

Co-occ is **mostly redundant** (70% overlap with ALS/SAS/content) but **does add genuine marginal golds** (+3.76pp) — not a pure-redundancy case.

## 3. LTR conversion (retrain, deterministic split, test n=733)
| Metric | mid baseline | mid + co-occ | lift | significance |
|---|---|---|---|---|
| **R@20** | 0.1201 | **0.1378** | **+0.0177** | ~1.5 SE (not 95%) |
| R@50 | 0.1678 | 0.1924 | +0.0246 | ~1.7 SE (not 95%) |
| NDCG@20 | 0.0556 | 0.0659 | +0.0103 | — |
| test coverage | 0.2974 | 0.3329 | +0.0355 | — |
| avg cands/user | 314 | 340 | +26 | low cost |

(1.96·SE ≈ 0.023 for R@20 at n=733.)

## 4. Verdict — **WEAK / CONDITIONAL ADOPT** (not a clean win, not an honest negative)
Decision rule: adopt if it adds marginal coverage **AND** improves R@20 materially.
- **Marginal coverage: YES** (+3.76pp golds mid misses; not pure redundancy despite 70% overlap).
- **R@20 lift: +0.0177 — directional but ~1.5 SE, not 95%-significant at n=733.** R@50 (+0.025) and NDCG (+0.010) move consistently in the same direction; coverage rises +3.55pp.
- **Cost: low** (+26 candidates/user, +130k rows).

**This is materially stronger than the G12-B GPU-feature result (+0.0027, flat)** — co-occ's lifts are consistent across R@20/R@50/NDCG and it adds real marginal coverage. But it is **not statistically significant at this test size**, so the honest call is: **adopt co-occurrence as a low-cost complementary candidate source given the consistent directional gains and genuine marginal coverage — with the explicit caveat that significance is not established at n=733 and should be re-confirmed at larger N.** Not overclaimed as a clean win; not dismissed as redundant.

## 5. LightGCN — deferred (documented)
LightGCN needs trained graph-propagation embeddings over the user-item bipartite graph; at 40k items that's a non-trivial training job (GPU or long CPU) — **not "genuinely cheap,"** so per the gate's rule it is **explicitly deferred** as a documented future candidate source (a candidate for a later GPU gate alongside the canonical-model work).

## 6. Claim boundaries
- Offline re-ranking within candidates; co-occ vocab capped at 6k (golds outside top-6k unreachable via co-occ — a coverage limitation). Deterministic md5 split; reproducible.
- Significance judged vs ≈0.023 (1.96·SE, n=733); lifts reported as directional/not-95%-significant, **not** overclaimed.
- No online lift, no production, no RiskFrame-gold-complete claim. No FAISS.

## 7. Status & next
- ✅ G15 done. **Co-occurrence = weak/conditional adopt** (low-cost complementary source; directional lift, genuine marginal coverage, not 95%-significant). **LightGCN deferred.**
- ⏭️ Remaining path to gold: **G16** FAISS / serving-latency demo over the adopted candidate set; **G17** Deep Defense Kernel; **G18** Final Gold Audit.

---

## 8. Acceptance addendum (ACCEPTED — conditional adopt)
**Exact claim boundary (ledger):** *"G15 co-occurrence = conditional adopt; +3.76pp marginal coverage and directional LTR lift, but not 95%-significant at n=733."*

- Co-occ added genuine marginal coverage (0.3228 → 0.3604, +3.76pp) but is ~70% redundant with ALS/SAS/content.
- LTR moved directionally: R@20 0.1201→0.1378 (+0.0177, ~1.5 SE), R@50 0.1678→0.1924 (+0.0246), NDCG 0.0556→0.0659 (+0.0103) — **not 95%-significant** at n=733.
- **Not a clean win, not an honest negative.** Adopt as a **low-cost complementary candidate source** with the explicit caveat: directional lift, not statistically established, revalidate at larger N.

**Updated default candidate strategy (adopted):**
`ALS-200 + SASRec-small-100 + content-50 + pop-30 + co-occurrence-100`

**Status:** co-occurrence conditional/adopted (not a proven strong winner) · LightGCN **deferred** (not cheap on CPU) · FAISS = **G16** (not part of G15).

**No-overclaim (enforced):** not "significantly improved recall"; graph retrieval **not** proven; no online lift; no production; no RiskFrame-gold-complete.

**STOP — G15 accepted (conditional adopt; default updated); G16 proposed (docs/63), awaiting approval.**
