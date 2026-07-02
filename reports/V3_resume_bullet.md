# PulseDiscover — V3 résumé bullet

## Full bullet (replaces the V2 bullet)

github.com/SidharthKriplani/pulsediscover — Two-stage RecSys on 2.56M Goodreads interactions
(temporal split); ALS warm-lane R@20=0.085; content-hybrid cold lane R@20=0.241 (ALS=0 on cold
users); **two neural models evaluated and both lost to ALS as documented honest negatives — SASRec
R@20=0.065, and a two-tower model (mean-pooled history + learned-ID⊕MiniLM-content item tower, BPR)
R@20=0.064 [95% CI 0.057–0.071] vs ALS 0.085 (non-overlapping CIs), with a content-ablation lift of
+0.021 R@20**; FAISS FlatIP exact retrieval lossless vs. ALS gold, −47% p95 latency; IVF rejected
(candidate overlap collapsed to 0.28–0.78); learned ALS+semantic fusion (LambdaMART) test R@20=0.036
vs. ALS-only 0.022; **Thompson Sampling exploration layer over two-tower candidates raised catalog
coverage 16.6%→50.5% and cut exposure Gini 0.97→0.81 for a ~1.3pp top-slate relevance cost
(explore/exploit tradeoff)**; off-policy evaluation (IPS/SNIPS/DR) validated — **and shown to break
down for the exploration policy as logging-policy overlap collapsed (ESS 168→21), recovering the
greedy policy but not the bandit**; semantic cold-start reach 54.6% of unseen golds; served on GCP
Cloud Run with 0% empty-response on load test.

## Compact bullet (space-constrained resumes)

github.com/SidharthKriplani/pulsediscover — Two-stage RecSys on 2.56M Goodreads interactions;
ALS warm R@20=0.085; two neural models (SASRec 0.065, two-tower ID+MiniLM-content/BPR 0.064)
evaluated as honest negatives vs ALS, content ablation +0.021 R@20; FAISS FlatIP lossless −47% p95
(IVF rejected); LambdaMART fusion test R@20=0.036 vs 0.022; Thompson Sampling exploration raised
catalog coverage 17%→51% and cut exposure Gini 0.97→0.81 at a ~1.3pp relevance cost; OPE (IPS/SNIPS/DR)
validated and shown to collapse for the exploration policy (ESS 168→21); served on GCP Cloud Run.

## One-line variant (for a skills/projects strip)

PulseDiscover — RecSys on 2.56M Goodreads: ALS floor R@20=0.085; two-tower(ID+content,BPR) & SASRec
honest negatives; Thompson Sampling exploration (coverage 17%→51%, Gini 0.97→0.81); FAISS serving;
OPE (IPS/SNIPS/DR) incl. exploration-policy overlap collapse.

---

**Note on framing (why the negatives stay in):** the honest-negative language for SASRec and the
two-tower is deliberate and load-bearing — it is the credibility signal of the project ("which
offline win actually deserves to ship"). Do not soften "lost to ALS" into a neutral "evaluated
against." Every number above traces to `outputs/evidence/` and `reports/V3_unified_report.md`.
