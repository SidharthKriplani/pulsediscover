# 02 — Evidence & Defense Plan

*Exact evidence artifacts and how each maps to a claim and to a section of the defense PDF. Rule: no claim in the PRD or PDF without a row here. Nothing below exists yet — these are the artifacts the build must produce.*

## Evidence → Claim → Defense-section map

| Evidence artifact (to be produced) | Claim it backs | Tag at completion | N / CI | Defense PDF section |
|---|---|---|---|---|
| `data_manifest.json` | "real public dataset, leakage-safe temporal split" | `[BUILT — real data]` | dataset N | §3 Dataset choices |
| `retrieval_baselines.json` + `recall_at_k.png` | "baselines establish the floor on real data" | `[BUILT — real data]` | N + bootstrap CI | §8 retrieval |
| `two_tower_report.json` | "two-tower beats ALS/popularity retrieval" | `[BUILT — real data]` | N + CI | §8 two-tower |
| `sasrec_report.json` | "sequential model beats baselines (the headline model)" | `[BUILT — real data]` | N + CI | §8 SASRec |
| `lambdarank_report.json` | "listwise ranker on real features" | `[BUILT — real data]` | N + CI | §4/§8 |
| `simulator_manifest.json` | "known-propensity OPE harness (not circular)" | `[BUILT][SYNTHETIC]` | params | §6 OPE proof |
| `ope_comparison.json` + `ope_variance.png` | "DR has lower variance than IPS; predicts sim-online lift better than NDCG" | `[BUILT][SYNTHETIC]` | bootstrap CI | §6/§7 DR proof |
| `position_bias_report.json` + `item_movement.png` | "IPS position debiasing works (PAL if built)" | `[BUILT][SYNTHETIC]` | N + CI | §9 PAL vs IPS |
| `feedback_loop_report.json` + exposure plots | "exposure concentration audit + loop demo" | `[BUILT][SYNTHETIC]` | rounds | §10 failure modes |
| `ci_summary.json` | "every headline number has N + 95% CI" | `[BUILT]` | — | §11 evidence tables |
| `seed_version_manifest.json` + `run_log.json` | "fully reproducible" | `[BUILT]` | — | §11 provenance |
| (existing) `pulserank` evidence JSONs | "IPS/SNIPS, LambdaRank, guardrail HOLD, Gini tradeoff" | `[BUILT][SYNTHETIC]` | 4,132 sessions | §6/§8/§10 |

## Defense PDF assembly rules
- Build PDF **only after** Gate 4 (all CIs present).
- Each figure embedded in the PDF must cite its `outputs/` source file.
- Each numeric claim must trace to a row in `ci_summary.json` or an existing `pulserank` evidence JSON.
- Sections with no evidence (e.g., PAL if deferred) are written as `[V2 — not built]`, never as results.
- Derivation depth (per the approved "mixed" decision): full for IPS/SNIPS/DR/SASRec; sketch for ALS/LambdaMART/two-tower/PAL.

## Evidence acceptance checklist (per artifact)
- [ ] Has a seed + version stamp.
- [ ] Reproducible by re-running the phase.
- [ ] Tagged (`[BUILT]`/`[BUILT][SYNTHETIC]`/`[BUILT — real data]`).
- [ ] Carries N and a CI where it's a metric.
- [ ] Mapped to exactly one PRD claim and one PDF section.
- [ ] Honest negatives preserved (e.g., if SASRec doesn't beat a baseline, that is reported, not hidden).

## Defense PDF outline (final target — not generated yet)
1. Thesis + one insight · 2. Architecture · 3. Datasets + truth boundaries · 4. Technique tournaments · 5. Derivations (mixed depth) · 6. OPE proof (simulator + bias/variance table) · 7. DR proof + variance plot · 8. SASRec/two-tower + real Recall/NDCG · 9. PAL vs IPS (or IPS-only) · 10. Failure modes + feedback-loop evidence · 11. Evidence tables + provenance · 12. Safe answers + never-say/say-instead.
