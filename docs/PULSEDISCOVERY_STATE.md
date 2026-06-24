# PulseDiscover — Current State (read first)

*Single source of truth for the project's live state. Updated for statefulness across sessions. If anything here conflicts with an older doc, this + `78_CONTROL_TOWER.md` win.*

## TL;DR
PulseDiscover is **DONE and published**: an offline, honesty-gated recommender decision system (2.56M Goodreads interactions). V1 `gold_candidate` 8.9 (locked, terminal). V2 lane **offline gold-complete** (RiskFrame 9.1). The serving path is **LIVE on Cloud Run** and the repo is **published to GitHub**. Every V2 number has been audited against its evidence artifact. Status: **interview-ready**.

## Live endpoints
- **Cloud Run (serving):** `https://pulsediscover-serving-98058433335.us-central1.run.app` — `/health`, `/recommend?user_id=&k=&mode=`, `/metadata`, `/metrics`. Project `voiceoverproject-472217`, region `us-central1`, service `pulsediscover-serving`, served-c2 ALS + exact FlatIP, 512 MiB / 1 vCPU / min-instances 1, `PD_BUILD_HNSW=0`.
- **GitHub:** `https://github.com/SidharthKriplani/pulsediscover` (branch `main`).

## Paths (IMPORTANT — they drift across sessions)
- Repo (file tools): `/Users/ASUS/Documents/Professional/GitHub/beastmax (5)/pulsediscover` — note the parent has shifted `(4)`→`(5)` across reconnects; **always re-confirm the actual path**, request the folder via `request_cowork_directory` if file tools say "outside connected folders".
- Bash mount: `/sessions/<session>/mnt/pulsediscover` (use `PYTHONPATH=src` for scripts; data in `data/interim/`).

## What was built (history)
Gated program **G11–G31**. V1 (D/O/P/H gates): ALS warm floor, content-hybrid cold lane, two-lane 90/10 policy, OPE/position-bias/health governance, A/B-readiness spec. **V2 (G22–G31):** G22 FAISS frontier · G23 production-shaped FastAPI · G24 cold-start fallback quality · G25 exposure governance · G26A decision spine + claim ladder · G26B heuristic exposure rerank · G27 semantic content retrieval · G28 learned fusion tournament · G29 OPE execution · G30 final audit + interview kit · G31 search/IR front end. Then: serving stress test + hardening, Cloud Run deploy, numbers audit, portfolio README + 19-page defense PDF, GitHub publish.

## Audited headline numbers (all matched their evidence JSONs)
| Claim | Value | Source |
|---|---|---|
| dataset | 2,557,588 int · 94,185 users · 41,961 items (k-core) | `data/interim/d2_stats.json` |
| **served ALS catalog** | **40,541 items** (this is what the model ranks over) | `c2_als.pkl` / `g22` |
| V1 ALS warm floor | R@20 0.0846 (V1 d3aplus tuning) | `d3aplus_als_tune.json` |
| SASRec converged | R@20 0.065 (lost), `plot_reconstructed` | `g20_..._v3.json` |
| served exact | R@20 0.0375 (V2 protocol) | `g22` |
| FAISS | FlatIP lossless −47% p95; HNSW overlap 0.992; IVF rejected | `g22` |
| cold-start fallback | coverage collapse 58.9× (~59×); personalization 0.6758→0.0009 | `g24` |
| item-cold-start | 0.5715 (57%) | `g24` |
| semantic | reach 0.546; cov 0.626 vs ALS 0.081; head 0.0098 vs 0.0874 | `g27` |
| learned fusion | LambdaMART test R@20 0.036 vs ALS-only 0.0222; cold 0.0323; **81 test positives** | `g28` |
| ALS/semantic overlap | 6,749 (near-disjoint) | `g28` |
| OPE | DR 0.00895 vs true 0.01059; SNIPS 0.01287; IPS 0.02106 (~2× over) | `g29` |
| exposure Gini | als/blended 0.982; **popularity 0.9995** | `g25` |
| search/IR | BM25 0.0133 > dense 0.0067; hybrid 0.0117 | `g31` |
| serving | warm p95 3.336 ms; 0% empty | `g23` |

**Protocol boundary (load-bearing):** V1 d3aplus 0.0846 and served-c2 0.0375 are different builds/protocols — **NOT comparable**. Registry: `G26A_MODEL_PROTOCOL_REGISTRY.md`.

## Claim boundaries (never violate)
Forbidden anywhere: online lift · real users · improved engagement · production *quality* · solved cold-start / long-tail · fairness certified · LLM recommender / semantic taste · learned-fusion-beats-ALS beyond the offline test split · real off-policy value · cross-protocol V1/V2 comparison · `gold_complete`. "Deployed to Cloud Run, serving HTTPS" IS true; "improved engagement" is NOT.

## Interview verbal traps (materials are safe; don't say a bolder version)
1. "beats ALS" → *"beat ALS-only on the held-out test split (0.036 vs 0.022), 81 positives → directional, not production."*
2. "in production / real users" → *"deployed on Cloud Run, live HTTPS; deployed ≠ improved engagement."*
3. "0.085" for served model → *"that's V1 tuning; served-c2 is ~0.0375 under a different protocol, registry-separated."*
4. "trained a SASRec at 0.065" → *"converged and lost; direct artifact lost with the Colab session → plot_reconstructed; decision stands."*

## Key docs
`78_CONTROL_TOWER.md` (status) · `PULSEDISCOVERY_INTERVIEW_KIT.md` (pitch + claim ladder) · `PULSEDISCOVERY_UNIFIED_DEFENSE_KERNEL.md` (method defenses) · `PULSEDISCOVERY_FAILURES_AND_HARDENING.md` (failures + serving stress test) · `G26A_MODEL_PROTOCOL_REGISTRY.md` · `12_EVIDENCE_LEDGER.md` · `69_G18_CLAIM_BOUNDARY_LOCK.md` · `defense/PulseDiscover_Interview_Defense.pdf` (19pp).

## Optional / outstanding (none required for interview-ready)
- Re-derive V1 ALS f64 from raw (the f64 model wasn't saved; needs a retrain). SASRec cannot be re-run here (no GPU).
- Add a "say this, not that" cheat-sheet page to the front of the defense PDF.
- Expand the README Portfolio table (currently 4 repos).
- Out of scope by design (needs production access): online A/B, real-traffic propensity logging, real-reward OPE, fairness certification.

## Operational gotchas (for bash/git in future sessions)
- **Always `cd` into the repo before any git command** (the terminal often sits in a sibling repo → "nothing to commit").
- Sandbox mounts leave stale `.git/*.lock` files that block git on the Mac → `find .git -name "*.lock" -delete` before committing.
- GitHub web fetch is **blocked** from this environment (returns empty); no browser connected → can't inspect other repos remotely.
- Large data/binaries are **gitignored** (not in the repo); `scripts/` rebuilds them.
- The serving image needs only `c2_als.pkl` + `c2_serving_assets.pkl` (built by `scripts/build_serving_assets.py`, stdlib-only).
