# PulseDiscover BeastMax Workspace — MANIFEST

*Workspace: `GitHub/beastmax/pulsediscover`. Created for the PulseDiscover BeastMax-core build. This turn = workspace setup + audit + plan only. No models built, no data generated, no training.*

## Workspace structure created
```
beastmax/pulsediscover/
  docs/        — PRD, BRD, architecture, current-state audit
  plans/       — build plan, evidence/defense plan, risk/overclaim boundaries
  src/         — (empty) build code goes here post-approval
  notebooks/   — (empty) exploration notebooks
  data/raw/    — (empty) downloaded public datasets (gitignored at build time)
  data/interim/— (empty) processed splits, simulator output
  outputs/evidence/ — (empty) reproducible *.json artifacts
  outputs/plots/    — (empty) *.png figures
  reports/     — (empty) human-readable run reports
  defense/     — (empty) defense PDF + source
  archive/     — reference copies of prior PRDs + review
```

## What was found in `extras`
| File | Relevance to PulseDiscover |
|---|---|
| `PRD_PulseDiscover.md` (v1, 31 KB) | **Relevant** — v1 RecSys PRD |
| `PRD_PulseDiscover_v2.md` (gold v2, 59 KB) | **Relevant** — the gold-standard dossier; the working PRD |
| `PulseOS_BeastMax_Architecture_Review.md` (55 KB) | **Relevant** — contains the code audit of `pulserank_platform` + RecSys section |
| `PRD_PulseSignal.md`, `PRD_PulseGuard.md`, `PRD_PulseKnowledge.md` | Not PulseDiscover — other combinations; left in extras |
| `TuneLens_PRD_v1.pdf` | Not PulseDiscover — fine-tuning (Crucible/PulseTrain, parked); left in extras |
| `metasignal_platform*.zip` (3×, ~160 MB each) | Not PulseDiscover — experimentation (PulseSignal); left in extras |
| `AI-Flagship-Knowledge-Intelligence-System*.zip` (4×) | Not PulseDiscover — RAG (PulseKnowledge); left in extras |
| `CONTRIBUTION_LOG.md`, `.DS_Store` | General/system; left in extras |

## What was copied (originals preserved in `extras`)
| Source (extras) | Destination | Why it matters |
|---|---|---|
| `PRD_PulseDiscover_v2.md` | `docs/PRD_PulseDiscover_v2.md` | The working gold PRD this build must convert from `[VISION]` to evidence-backed |
| `PRD_PulseDiscover.md` | `archive/PRD_PulseDiscover_v1.md` | History/diff reference |
| `PulseOS_BeastMax_Architecture_Review.md` | `archive/PulseOS_BeastMax_Architecture_Review.md` | Source-of-truth code audit + RecSys section |

All copies used `cp -p` (copy, not move). **Nothing in `extras` was moved or deleted.**

## What was left behind (intentionally)
- Other combinations' PRDs (PulseSignal/PulseGuard/PulseKnowledge) — out of scope this turn.
- TuneLens PDF — parked fine-tuning project, not RecSys.
- All `.zip` archives — large, and they are experimentation/RAG platforms, not PulseDiscover. (If any contains a RecSys component, see Uncertainty below.)

## The actual code spine is NOT in `extras`
The real, built PulseRank evaluation harness lives at **`GitHub/pulserank_platform`** (a separate mounted repo, already audited in `archive/PulseOS_BeastMax_Architecture_Review.md`). It is the `[BUILT][SYNTHETIC]` evidence base (ALS, LightGBM LambdaRank, IPS/SNIPS + clipping sweep, MMR, drift, offline A/B replay; 61 tests; synthetic 4,132-session e-commerce corpus).
- **Decision:** I did NOT copy/move the `pulserank_platform` repo into this workspace, because (a) it lives outside `extras` and the instruction scoped copying to `extras`, and (b) moving a git repo risks breaking it. The build plan references it as the inherited evidence spine and proposes (open question, see below) whether to vendor a read-only copy of its `outputs/evidence/` here.

## Uncertainty / open items
1. **`pulserank_platform` integration:** should we (a) reference it in place, (b) copy its `src/` + `outputs/evidence/` into this workspace as the starting point, or (c) start `src/` fresh and import PulseRank as a dependency? Recommended: (b) vendor a read-only copy of `outputs/evidence/` into `archive/pulserank_evidence/` for provenance, and reuse its IPS/SNIPS code by reference. **Awaiting your call.**
2. **Zip contents not inspected:** the `AI-Flagship-*` and `metasignal_platform` zips were not unzipped (large, and domain-mismatched). Low probability of RecSys content; flag if you know otherwise.
3. The `docs/PRD_PulseDiscover_v2.md` here is a copy; the canonical edit target post-build is this workspace copy (extras retains the snapshot).

*Status: workspace ready. No build performed.*
