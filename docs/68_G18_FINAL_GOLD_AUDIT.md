# 68 — G18: Final Gold Audit (RiskFrame / BeastMax 15-Axis)

*Final audit + claim-safe closeout. Audit/packaging/consistency gate — **no new models, no new candidate sources, no claim changes, no fabrication.** Verdict up front: **GOLD CANDIDATE / claim-safe portfolio closeout — NOT gold_complete** (blockers in §3). Machine-readable: `outputs/evidence/g18_final_gold_audit_report.json`.*

## 1. 15-axis scorecard
| # | Axis | Before (docs/49) | Final | Evidence / strongest artifacts | Remaining weakness | Blocks gold_complete? |
|---|---|---|---|---|---|---|
| 1 | Product thesis | 9 | **9** | docs/45 dossier, G2 | minor | no |
| 2 | Buyer / JD alignment | 7 | **8** | DEF1, G17 bank (RecSys retrieval/rank/OPE roles) | no explicit JD doc | no |
| 3 | Industry realism | 8.5 | **9** | two-stage (G12–G14), FAISS serving (G16), OPE/logging (O1/P1/G1) | not deployed | no |
| 4 | Data realism | 8 | **8** | D1/D2 (2.56M, global-time, prevalence) | single domain | no |
| 5 | **Method depth** | 6 | **7.5** | G11–G11C/D seq tournament, G12 LTR, G15 co-occ, G16 FAISS | **no advanced model beats ALS; LightGCN + canonical-convergence deferred** | **yes (for gold_complete)** |
| 6 | Technique tournament quality | 5 | **9** | ALS/pop/co-occ/content/SASRec/GRU/two-tower/LTR/MMR/FAISS/OPE/PBM — decision-driven, honest negatives | — | no |
| 7 | Evidence quality | 9 | **9.5** | ledger docs/12, tagged/CI'd, deterministic split (hash bug fixed) | — | no |
| 8 | Evaluation quality | 8.5 | **9** | LLO+global-time, bootstrap CIs, G6 scale, coverage ceiling, within-candidate vs full-catalog distinction | within-candidate metrics bounded | no |
| 9 | Business consequence chain | 6 | **8** | G2 growth, G3 risk, defense-card "if-wrong" rows | qualitative (no $ model) | no |
| 10 | Failure-mode coverage | 8 | **9** | G3 risk register, per-card failure modes, documented negatives | — | no |
| 11 | Operational realism | 2 | **8** | G1 infra/cost, G16 FAISS serving index, P1 logging schema | estimates not benchmarks; not deployed | no |
| 12 | Claim safety | 10 | **10** | docs/33, docs/69 lock, scope-tagging throughout | — | no |
| 13 | Interview defensibility | 9 | **9.5** | DEF1, G17 kernel/cards/bank | — | no |
| 14 | Differentiation / memorability | 9 | **9** | honesty arc, "coverage was the bottleneck", documented negatives | — | no |
| 15 | Build feasibility / reproducibility | 8.5 | **9** | deterministic seeds/split, checkpoints, runnable scripts; GPU steps via Colab notebooks | some steps external GPU | no |

**Mean ≈ 8.7/10.** Every axis is strong; **only Method Depth (#5) carries a gold_complete-blocking weakness** — and it is an *honest* one (a documented absence of a competitive advanced model + deferred LightGCN/canonical convergence), not a fabrication or hidden gap.

## 2. Verified accepted facts (all confirmed against artifacts)
- ✅ ALS f64 = warm collaborative floor (R@20 0.085). — `c2_als.pkl`, D3A+
- ✅ Content-hybrid = cold-start *candidate lane* (cold R@20 0.241), **not** "cold-start solved." — C1
- ✅ Sequence models tested seriously, mostly underperform ALS (best SASRec-small 0.0594). — G11/G11C/D
- ✅ Full-softmax SASRec = important objective/loss finding (6.7×, 0.0068→0.0456) but **did not beat ALS.** — G11C/D
- ✅ TwoTower built; **honest negative** (R@20 0.0020). — G11D
- ✅ LTR = ranking win **only within candidate sets** (0.0382→0.0791; mid 0.1201). — G12-B/G13
- ✅ GPU sequence features **non-additive** to LTR (+0.0027, noise). — G12-B
- ✅ Candidate coverage = binding lever; G13 widening converted (+44%, 0.086→0.124); **mid = adopted knee.** — G13
- ✅ G14: **pure LTR remains default**; aggressive post-hoc governance not worth the relevance cost (combined 20.7% self-violation). — G14
- ✅ G15 co-occ = **conditional/adopted**: +3.76pp marginal coverage, directional lift, **not 95%-significant.** — G15
- ✅ G16 FAISS FlatIP exact **adopted** as ALS-slice serving index; **IVF = scale-demo only.** — G16
- ✅ G17 defense kernel accepted as interview-defense layer. — docs/65–67
- ✅ LightGCN **deferred.** — G15
- ✅ No online lift · no production deployment · no live A/B · no full-catalog claim where eval is within-candidate · no solved fairness / cold-start / creator-health. — enforced throughout

## 3. Why NOT gold_complete (blocking gaps — honest, not fabricated)
1. **No competitive advanced model beats the ALS floor** — the sequence/graph/two-tower tournament produced honest negatives + a documented full-softmax lever, but no T3 model surpassed MF. Per the decision rule ("gold_complete only if no material T3/model-depth gap remains"), this blocks gold_complete.
2. **Offline-only — no live A/B / no proven lift.** By design, but it means business value is unproven; gold_complete would over-imply.
3. **Deferred items:** LightGCN (not cheap on CPU) and canonical SASRec trained to convergence (GPU) remain open.

These are documented, defensible deferrals — they make the project a **strong gold_candidate**, not gold_complete.

## 4. Non-blocking gaps
- Co-occurrence significance not established at n=733 (conditional adopt).
- Per-item IPS variance (aggregate-only trust).
- FAISS IVF a scale-demo at 40k items.
- Single domain; cost chain qualitative.

## 5. Verdict
**RiskFrame-gold CANDIDATE / claim-safe portfolio closeout complete.** 15 axes audited; mean ≈ 8.7; all accepted facts verified against artifacts; claim safety enforced. **gold_complete withheld** solely because model depth produced honest negatives (no advanced model beats ALS) and the project is offline with documented GPU/A-B deferrals — exactly the boundaries we've maintained.

**STOP — final audit done; claim lock (docs/69), closeout (docs/70), README (docs/71), JSON (evidence) follow.**
