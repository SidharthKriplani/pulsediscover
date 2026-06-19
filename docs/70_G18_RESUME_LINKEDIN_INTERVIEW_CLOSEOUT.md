# 70 — G18: Resume / LinkedIn / Interview Closeout

*Claim-safe surfaces. Every claim maps to an artifact or is marked NOT SAFE. Gold candidate.*

## Resume bullets (3 options)
**Conservative:**
"Built an offline two-stage book recommender (candidate generation → LambdaMART ranking) on 2.56M Goodreads interactions; tuned ALS to a R@20 0.085 warm floor and added a content-hybrid cold-start lane." *(maps: D3A+, C1, G12)*

**Strong:**
"Designed and evaluated an offline two-stage RecSys (ALS + content + co-occurrence + sequence candidates → LambdaMART): diagnosed candidate coverage as the binding bottleneck and lifted offline within-candidate R@20 ~44% by widening retrieval; added OPE, position-bias correction, and a FAISS serving index." *(maps: G12-B, G13, O1, P1, G16)*

**Max-safe (impact-flavored but bounded):**
"Built a production-shaped offline recommender pipeline (retrieval → learning-to-rank → slate policy → ANN serving) with off-policy evaluation and catalog-health governance; surfaced and documented honest negatives (CPU sequence models and a neural two-tower did not beat matrix factorization)." *(maps: G11–G17; "production-shaped" not "production")*

## LinkedIn project blurb
"**PulseDiscover** — an offline, evidence-backed recommender experimentation program on 2.5M Goodreads interactions. Two-stage architecture (warm + cold candidate generation → LambdaMART ranking → slate policy), plus the governance layer that decides whether to trust a test: off-policy evaluation, position-bias correction, catalog-health analysis, and a FAISS serving index. Built with strict claim discipline and documented honest negatives — offline by scope, not deployed."

## GitHub README headline paragraph
"PulseDiscover is an offline two-stage recommender system and experimentation program built on the Goodreads fantasy/paranormal corpus (2.56M interactions). It pairs a tuned ALS warm model with a content-hybrid cold-start lane, fuses multiple candidate sources through a LambdaMART ranker, and wraps the whole pipeline in a measurement + serving layer (OPE, position-bias correction, catalog-health, FAISS). It is deliberately honest about scope: all results are offline (recall / known-propensity OPE / simulation), ranking metrics are within-candidate, and deep sequence/graph/two-tower models were tested and reported as honest negatives where they did not beat matrix factorization."

## 60-second spoken pitch
"PulseDiscover treats recommendation as a two-stage company decision, not a single model. Stage one generates candidates — ALS matrix factorization is the warm floor at R@20 0.085, plus a content-hybrid cold lane, co-occurrence, and sequence candidates. Stage two is a LambdaMART ranker that fuses them and roughly doubles single-retriever ranking within the candidate set. The interesting findings were honest negatives: deep sequence models and a two-tower didn't beat MF, and adding their scores to the ranker didn't help — so I diagnosed that the real bottleneck was candidate coverage, proved it by widening retrieval for a 44% offline R@20 lift, and declined aggressive slate governance because the candidate stage already handled discovery. Around it I built off-policy evaluation, position-bias correction, and a FAISS serving index. Everything's offline and scope-tagged — no online lift, no deployment."

## 30-second spoken version
"An offline two-stage recommender on 2.5M Goodreads interactions: ALS + content + co-occurrence + sequence candidates feeding a LambdaMART ranker, with off-policy evaluation, position-bias correction, and a FAISS serving layer. The senior part is the honesty — deep models were honest negatives, the real bottleneck was candidate coverage, and I fixed that for a 44% offline ranking lift. Offline by scope; no production claim."

## 10 likely interviewer follow-ups (tight answers)
1. **What's the headline number?** "Widening candidate coverage lifted offline within-candidate R@20 ~44% (0.086→0.124); ALS warm floor is R@20 0.085."
2. **Did deep models beat ALS?** "No — honest negatives. Best sequence model 0.059 < ALS 0.085; two-tower near-random."
3. **Then why build them?** "Tournament breadth + ranker features; and the negatives are diagnosed, not hand-waved."
4. **Why was LTR a win if numbers are low?** "It ~2× single-retriever *within candidates*; the absolute number is capped by candidate coverage, which I then attacked."
5. **Online results?** "None — offline only. The A/B is specced (G4), not run."
6. **Is FAISS a quality gain?** "No — latency. Exact FAISS is identical results ~39% faster."
7. **Co-occurrence — real win?** "Conditional: +3.76pp coverage and a directional lift, not 95%-significant at n=733."
8. **Cold-start solved?** "No — a working cold *candidate lane* (0.24 cold recall where MF is 0)."
9. **Creator fairness?** "Improved reach (+444 creators), not de-concentration; needs a training-time fix."
10. **What next with resources?** "Canonical SASRec to convergence, LightGCN candidates, and a live A/B."

**Every claim above maps to a gate artifact (docs/12 ledger + G-series). Gold candidate; nothing marked production/online/solved.**
