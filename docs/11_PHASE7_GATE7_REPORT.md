# 11 — Phase 7 Gate 7 Report (Feedback-Loop / Catalog-Health Audit)

*Scope: catalog-health audit on the MovieLens smoke. quality = TRAIN mean rating (normalized); popularity = TRAIN count (so quality ≠ popularity → some high-quality items start low-popularity). Multi-round popularity-ratchet vs a CTR+exploration diversified policy. Simulator-scoped; no real-lift claim. No Goodreads, no two-tower, no PDF.*

---

## 1. What was built
- `src/pulsediscover/feedback_loop.py` — Gini / entropy / long-tail-share; multi-round loop (ratchet vs ctr_explore). `[BUILT]`
- `src/run_phase7.py` — driver: 10-round simulation, exposure metrics, buried-quality analysis, genre-level proxy, plots, `feedback_loop_report.json`. `[BUILT]`
- Artifacts: `outputs/evidence/feedback_loop_report.json`, `outputs/plots/feedback_loop.png`. Tag `[BUILT][SYNTHETIC — catalog-health audit on simulator]`.

## 2. The feedback loop, made visible (10 rounds, item exposure)
| Metric | Round 0 | Ratchet final | CTR+explore final |
|---|---|---|---|
| Exposure **Gini** (↓ = healthier) | 0.621 | **0.700** | **0.151** |
| Entropy (↑ = more diverse) | — | 7.251 | 8.176 |
| **Long-tail share** (↑ = healthier) | 0.357 | **0.289** | **0.766** |
| Buried high-quality/low-pop exposure share | 0.0125 (popularity baseline) | **0.0204** | **0.0976** |

## 3. Honest reading
- **The ratchet is real and harmful.** Ranking by cumulative click *volume* concentrates exposure round over round: Gini rises 0.621 → **0.700**, long-tail share falls 0.357 → **0.289**, and the 301 high-quality-but-low-popularity items stay starved (exposure share 0.0125 → only 0.0204). The recommender amplifies what it already showed — the catalog ossifies.
- **A rate-based + exploration policy breaks it.** Ranking by exposure-normalized **CTR** (not volume) plus 10% exploration drops Gini to **0.151**, raises long-tail share to **0.766**, and gives the buried high-quality items **~4.8× more** exposure (0.0204 → 0.0976). This is the "break the loop" demonstration.
- **Important caveat — the magnitude is abstraction-dependent, and full de-concentration is NOT free.** The ctr_explore policy flattening to Gini 0.151 is a *strong/idealized* anti-ratchet on a catalog-level abstraction; a production diversified policy would not (and should not) equalize this hard — pure CTR + heavy exploration trades away **relevance/engagement**, which this catalog-level model does **not** penalize. So the result faithfully shows the *mechanism and direction* (volume-ranking concentrates; rate+exploration de-concentrates and surfaces buried quality), but the exact magnitudes are artifacts of the abstraction, not a claim about optimal diversification.
- **Genre-level (category proxy) barely moves:** final Gini ratchet 0.675 vs ctr_explore 0.657 — category concentration is stickier than item concentration (popular genres dominate either way). Honest nuance: the loop bites hardest at the *item* level.

## 4. Caveats (no spin)
- **Catalog-level abstraction** (exposure ∝ score; expected clicks = exposure × quality) — not per-user session replay; chosen for a fast, transparent mechanism demo.
- **Simulator-scoped, MovieLens smoke.** No real online lift; no claim about a specific platform.
- **No real creators in MovieLens** — creator-level concentration is a Goodreads-domain feature (pending); genre primary-category used only as a proxy.
- The diversified policy's relevance cost is **not modeled** here (stated above).

## 5. No-overclaim check (per `plans/03`)
- Ratchet harm and mitigation reported with explicit magnitude + abstraction caveats. ✅
- ctr_explore's extreme equalization flagged as idealized / not free. ✅
- Creator-level deferred to Goodreads; genre proxy labeled as such. ✅
- Simulator-scoped; no Goodreads/two-tower/PDF. ✅

## 6. BeastMax-core OPE spine — now COMPLETE on the smoke
Baselines (Ph2) → SASRec (Ph3A) → known-propensity simulator (Ph4A) → IPS/SNIPS/DM/DR validation (Ph5) → position-bias correction (Ph6) → feedback-loop/catalog-health audit (Ph7). Every phase produced reproducible, CI'd, honestly-reported evidence on MovieLens-1M.

## 7. Gate 7 status & decision
- ✅ Exposure concentration (Gini/entropy/long-tail) measured; ratchet demoed over rounds; before/after shown; baseline vs diversified compared; buried-quality quantified.
- ⏸️ Domain headline (Goodreads/Amazon) still pending.

**Gate 7 decision for you:** approve next — (a) **Phase 8–9: consolidate evidence + update the PRD** (fold all smoke evidence/CIs into `docs/PRD_PulseDiscover_v2.md`, flipping `[VISION]` rows to `[BUILT][SYNTHETIC]`), (b) **Phase 10: defense PDF** grounded in the smoke outputs, (c) **load Goodreads/Amazon** to re-run the whole spine on the domain headline, or (d) **firm up Phase 7** (per-user session replay; model the diversification relevance cost).

**STOP — awaiting Gate 7 review. No further phase begins until approved.**
