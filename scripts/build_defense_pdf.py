"""Build the PulseDiscover interview-defense PDF (9-section format)."""
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                PageBreak, HRFlowable, ListFlowable, ListItem)
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os, matplotlib
_F = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
pdfmetrics.registerFont(TTFont("DJ", os.path.join(_F, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DJ-Bold", os.path.join(_F, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DJ-Italic", os.path.join(_F, "DejaVuSans-Oblique.ttf")))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily("DJ", normal="DJ", bold="DJ-Bold", italic="DJ-Italic", boldItalic="DJ-Bold")

NAVY = colors.HexColor("#1f3a5f"); ACCENT = colors.HexColor("#c0392b")
GREY = colors.HexColor("#555555"); LIGHT = colors.HexColor("#eef2f7")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName="DJ-Bold", textColor=NAVY, fontSize=15, spaceBefore=14, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="DJ-Bold", textColor=ACCENT, fontSize=11.5, spaceBefore=9, spaceAfter=3)
BODY = ParagraphStyle("Body", parent=ss["BodyText"], fontName="DJ", fontSize=9.3, leading=13, spaceAfter=5, alignment=TA_LEFT)
SMALL = ParagraphStyle("Small", parent=BODY, fontName="DJ", fontSize=8.4, leading=11, textColor=GREY)
BOLDLEAD = ParagraphStyle("BoldLead", parent=BODY, fontName="DJ", fontSize=9.3, leading=13)
TITLE = ParagraphStyle("TT", parent=ss["Title"], fontName="DJ-Bold", textColor=NAVY, fontSize=22, spaceAfter=2)
SUB = ParagraphStyle("Sub", parent=ss["Normal"], fontName="DJ", fontSize=10.5, textColor=GREY, spaceAfter=2)

def P(t, s=BODY): return Paragraph(t, s)
def bullets(items, s=BODY):
    return ListFlowable([ListItem(P(x, s), leftIndent=6) for x in items],
                        bulletType="bullet", bulletColor=NAVY, leftIndent=12, bulletFontSize=6)
def rule(): return HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#cfd8e3"), spaceBefore=2, spaceAfter=8)

def tbl(data, widths, header=True, font=8.2):
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("FONTSIZE",(0,0),(-1,-1),font),("VALIGN",(0,0),(-1,-1),"TOP"),
             ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#c7d0db")),
             ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
             ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
             ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, LIGHT])]
    if header:
        style += [("BACKGROUND",(0,0),(-1,0),NAVY),("TEXTCOLOR",(0,0),(-1,0),colors.white),
                  ("FONTNAME",(0,0),(-1,0),"DJ-Bold")]
    t.setStyle(TableStyle(style)); return t

def cell(t): return Paragraph(t, ParagraphStyle("c", parent=BODY, fontSize=8.2, leading=10.5))

story = []

# ---------- Cover ----------
story += [Spacer(1, 40), P("PulseDiscover", TITLE),
          P("Interview Defense Dossier", SUB),
          P("Offline, honesty-gated recommender decision system &mdash; 2.56M Goodreads interactions", SUB),
          Spacer(1, 8), rule(),
          P("<b>Status:</b> V1 <font color='#1f3a5f'>gold_candidate 8.9</font> (locked, terminal) &middot; "
            "V2 lane <font color='#1f3a5f'>offline gold-complete</font> (RiskFrame 9.1) &middot; "
            "serving API <b>deployed to Google Cloud Run</b>.", BODY),
          P("<b>Live endpoint:</b> https://pulsediscover-serving-98058433335.us-central1.run.app "
            "(<font face='Courier'>/health</font> &rarr; ready:true, <font face='Courier'>/recommend</font> &rarr; source:als).", SMALL),
          P("<b>Reading rule:</b> every number here is offline and protocol-tagged. Served-c2 metrics and V1 "
            "d3aplus metrics are <b>different protocols and not comparable</b> (see &sect;6). No online-lift claim appears anywhere.", SMALL),
          PageBreak()]

# ---------- 1. Project Summary ----------
story += [P("1 &nbsp; Project Summary", H1), rule(),
          P("PulseDiscover is an offline, honesty-disciplined <b>recommender decision system</b> built on a real "
            "2.56M-interaction Goodreads fantasy/paranormal dataset (94k users, 42k items). It does not chase a single "
            "metric &mdash; it <b>decides which retrieval, serving, fallback, fusion and exposure policy should ship, and "
            "proves which must not</b>, by catching offline-metric bias, candidate-coverage gaps, latency&ndash;quality "
            "trade-offs, cold-start degradation, and catalog concentration before they would reach users."),
          P("It was built as a gated experimentation program (gates G11&ndash;G31) under strict claim discipline: every "
            "claim maps to an artifact, every metric carries a protocol tag, and negatives are documented rather than "
            "buried. V1 established the offline modelling floor; the V2 lane (G22&ndash;G31) added production-shaped "
            "serving, cold-start and exposure governance, a learned ALS+semantic fusion ranker, executed off-policy "
            "evaluation, and a Search/IR retrieval front end &mdash; then the serving path was deployed to Cloud Run."),
          P("<b>What it is not:</b> not an online/production-quality system, not a solved-cold-start system, not a "
            "fairness-certified system, and not an LLM recommender. Those boundaries are explicit and load-bearing.", SMALL)]

# ---------- 2. System Architecture ----------
story += [P("2 &nbsp; System Architecture", H1), rule(),
          P("A two-stage retrieve&rarr;rank pipeline wrapped in a decision/serving spine. Every stage is versioned on the "
            "<b>served c2 ALS model</b>; V1's d3aplus tuning model is quarantined to V1 evidence only."),
          tbl([["Stage","What it does","Artifact / evidence"],
               [cell("1. Candidate generation"), cell("ALS (warm) + semantic content (MiniLM) + popularity + co-occurrence"), cell("c2_als.pkl, g27 semantic index")],
               [cell("2. Retrieval"), cell("FAISS top-K by inner product; FlatIP exact (default), HNSW (scale), IVF rejected"), cell("src/serving/faiss_retriever.py (G22)")],
               [cell("3. Fusion / rank"), cell("RRF heuristic fusion; LightGBM LambdaMART learned fusion over ALS+semantic features"), cell("src/ranking/g28_fusion_ranker.py (G28)")],
               [cell("4. Rerank"), cell("Config-flagged head-cap exposure rerank (default OFF)"), cell("recommender_service.py (G26B)")],
               [cell("5. Fallback"), cell("unknown/sparse/error &rarr; popularity; never empty (0% empty verified)"), cell("fallback tree (G23/G24)")],
               [cell("6. Serving"), cell("FastAPI: /recommend /health /metadata /metrics; versioned loader; structured logs"), cell("src/serving/api.py (G23)")],
               [cell("7. Logging"), cell("Per-request JSON; OPE-ready propensity schema designed"), cell("G26A schema, G29 logging")],
               [cell("8. Evaluation"), cell("Cohort recall, exposure (Gini/coverage), IPS/SNIPS/DR, NDCG/MRR"), cell("g24&ndash;g31 reports")],
               [cell("9. Decision"), cell("ship / hold / reject per gate, behind a locked claim ladder"), cell("control tower + claim boundary")],
              ], [1.25*inch, 3.0*inch, 2.0*inch]),
          Spacer(1,6),
          P("<b>Deployed:</b> the ALS+FlatIP serving path runs on Google Cloud Run (512&nbsp;MiB / 1&nbsp;vCPU / "
            "min-instances 1 / concurrency 8, exact mode, HNSW off). A lean image (only c2_als.pkl + a 4&nbsp;MB "
            "precomputed assets bundle) loads in ~0.8&nbsp;s and serves live HTTPS.", SMALL)]

# ---------- 3. Fundamentals Bridge ----------
story += [PageBreak(), P("3 &nbsp; Fundamentals Bridge &mdash; concept extraction", H1), rule(),
          P("Each technique, from first principles, with its role in PulseDiscover.")]
concepts = [
 ("ALS (implicit-feedback matrix factorization)",
  "Minimises &Sigma; c<sub>ui</sub>(p<sub>ui</sub> &minus; x<sub>u</sub>&middot;y<sub>i</sub>)&sup2; + &lambda;&#8214;&middot;&#8214;&sup2; with confidence c=1+&alpha;r, solved by alternating closed-form ridge regressions. Captures collaborative co-read structure directly. <b>Role:</b> the warm-retrieval floor and the served model."),
 ("FAISS (ANN retrieval)",
  "Sub-linear top-K over item vectors. FlatIP = exact exhaustive inner product; HNSW = navigable small-world graph (efSearch trades recall/latency); IVF = coarse quantiser (nprobe). <b>Role:</b> serving retrieval &mdash; FlatIP exact default, HNSW as a scale option, IVF rejected."),
 ("SASRec (self-attentive sequential rec)",
  "Causal self-attention over the user's item-sequence to predict the next item; full-softmax vs sampled-softmax loss matters. <b>Role:</b> tested to convergence as the deep-model challenger &mdash; it lost to ALS (an honest negative)."),
 ("LambdaMART (learning-to-rank)",
  "Gradient-boosted trees optimising a listwise NDCG surrogate with per-query (per-user) groups. <b>Role:</b> the learned fusion ranker over ALS+semantic+popularity candidate features (G28); also the V1 within-candidate ranker."),
 ("IPS / SNIPS (off-policy estimators)",
  "IPS = mean[ 1{a=&pi;<sub>e</sub>(x)} / &pi;<sub>log</sub>(a|x) &middot; r ] &mdash; unbiased but high-variance importance weighting. SNIPS self-normalises by &Sigma; weights to cut variance at a little bias. <b>Role:</b> executed offline in G29 to estimate a policy's value from logged data; SNIPS is the recommended estimator."),
 ("RRF (reciprocal rank fusion)",
  "score(i) = &Sigma;<sub>s</sub> 1/(c + rank<sub>s</sub>(i)) across sources &mdash; score-scale-free, training-free rank combiner. <b>Role:</b> the best non-learned fusion of ALS and semantic candidates; chosen for robustness, not a recall win."),
 ("Position bias (and PAL)",
  "Users click/consume top-ranked items partly because of position, not relevance; naive click-recall is therefore confounded. Position-aware learning (PAL) / IPS de-bias examination from relevance. <b>Role:</b> PulseDiscover does position-bias-aware <i>evaluation</i> (a two-lane OPE diagnostic) &mdash; not an in-ranker correction (that is documented future work)."),
]
for name, body in concepts:
    story += [P(name, H2), P(body, BODY)]

# ---------- 4. Decision Log ----------
story += [PageBreak(), P("4 &nbsp; Decision Log &mdash; trade-offs with rationale", H1), rule(),
          tbl([["Decision","Rationale","Boundary kept"],
               [cell("<b>ALS over SASRec</b> as the warm floor"), cell("A canonical full-softmax SASRec was trained to convergence and still scored below ALS (R@20 0.065 vs 0.085). On short, sparse book histories the dominant signal is collaborative co-read, which MF captures directly; depth was not the lever."), cell("\"a converged deep model lost\" &mdash; not \"deep models are useless\"")],
               [cell("<b>FlatIP exact as default</b> (HNSW only as scale option)"), cell("FlatIP is lossless and ~47% faster than numpy brute force at this scale; HNSW ef64 is near-lossless (overlap 0.992, ~2.4x) but approximate. Correctness-by-default; opt into approximation explicitly."), cell("latency, not quality &mdash; FAISS never improves recommendations")],
               [cell("<b>RRF = robustness, not a recall win</b>"), cell("RRF recovers cold-start and ~4x coverage at ~95% of ALS relevance, but does not beat ALS-only on warm recall. Framed as robustness/coverage, never as a headline lift."), cell("no \"hybrid beats both lanes\" claim")],
               [cell("<b>IVF rejected</b> despite speed"), cell("Low-nprobe IVF kept gold Recall@20 flat while candidate-overlap@K vs exact collapsed to 0.28&ndash;0.78 &mdash; it returned a largely different (broken) candidate set. Speed that destroys candidate fidelity is a bug."), cell("overlap@K is a first-class acceptance check")],
               [cell("<b>BM25 = query-by-document</b>"), cell("PulseDiscover has no free-text queries; BM25's \"query\" is the seed item's own text (more-like-this). This is a real IR pattern, so the search framing is honest without claiming a free-text search engine."), cell("seed-item, not free-text query search")],
              ], [1.5*inch, 3.55*inch, 1.2*inch])]

# ---------- 5. Three Real Failures ----------
story += [PageBreak(), P("5 &nbsp; Three Real Failures (caught and corrected)", H1), rule(),
          P("The interview value of the project is the failures I caught myself.", SMALL)]
fails = [
 ("SASRec false convergence &mdash; the NaN val-loss patience bug",
  "<b>What happened:</b> training early-stopped at ~6 epochs &mdash; looked converged. The validation-loss curve was all-NaN (masked-attention NaN on short/padded sequences), and a NaN comparison silently satisfied the patience criterion while train loss and eval-R@20 were still rising. "
  "<b>Revealed:</b> a monitor can fire \"done\" for the wrong reason; \"early stop triggered\" is not evidence of convergence. "
  "<b>Did:</b> rejected the run as invalid, rebuilt the monitor on an eval-R@20 plateau, and only declared convergence (v3, ~95 epochs) once the metric we care about flattened. "
  "<b>Demonstrates:</b> I never accept a convenient stop; I check why a signal fired."),
 ("IVF &mdash; the metric can lie",
  "<b>What happened:</b> FAISS IVF at low nprobe looked great on gold Recall@20, but candidate-overlap@K vs exact had collapsed to 0.28&ndash;0.78 &mdash; it returned a mostly different candidate set that happened to still contain the single gold item. "
  "<b>Revealed:</b> a single-gold metric is blind to most of the slate changing, so a downstream metric can stay flat while the thing it depends on silently degrades. "
  "<b>Did:</b> rejected IVF despite its speed; kept FlatIP/HNSW and made overlap@K a first-class acceptance check. "
  "<b>Demonstrates:</b> I instrument the mechanism, not just the headline number."),
 ("ID-dtype silent zeroing &mdash; the pipeline ran clean with wrong numbers",
  "<b>What happened:</b> book_id is a string in the ALS index, but eval/train ids loaded as int64; every membership test compared int vs str keys, so every lookup missed and recall floored to ~0 &mdash; while the pipeline ran clean and produced plausible-shaped (wrong) numbers. "
  "<b>Revealed:</b> the most dangerous bugs don't crash; silent type coercion quietly degrades a metric to zero while everything looks healthy. "
  "<b>Did:</b> caught it with an overlap sanity check (the genre map had zero catalog-overlapping keys &mdash; impossible if types matched), normalised book_id to str repo-wide. "
  "<b>Demonstrates:</b> I sanity-check intermediate artifacts instead of trusting a final number."),
]
for name, body in fails:
    story += [P(name, H2), P(body, BODY)]

# ---------- 6. Eval Results ----------
story += [PageBreak(), P("6 &nbsp; Eval Results &mdash; with honest claim boundaries", H1), rule(),
          Paragraph("<b>Protocol warning (load-bearing):</b> the V1 <b>d3aplus</b> ALS tuning result (R@20 <b>0.0846</b>) and "
            "the V2 <b>served-c2</b> results (full-catalog single-held-out-gold, R@20 ~<b>0.0375</b>) are <b>different model builds "
            "and different evaluation protocols &mdash; they are NOT comparable</b>. I keep them in a model/protocol registry "
            "precisely so they are never conflated. V2 numbers are used for <i>relative</i> cross-policy/cohort structure, not as a "
            "headline that overturns V1.",
            ParagraphStyle("warn", parent=BODY, fontSize=8.8, leading=12, backColor=colors.HexColor("#fdecea"),
                           borderColor=ACCENT, borderWidth=0.6, borderPadding=6, spaceAfter=8)),
          tbl([["Result","Number","Protocol tag"],
               [cell("V1 ALS warm floor"), cell("R@20 0.0846"), cell("V1 d3aplus tuning &mdash; REAL_OFFLINE")],
               [cell("Canonical SASRec, converged"), cell("R@20 0.065 (lost to ALS)"), cell("V1 &mdash; PLOT_RECONSTRUCTED")],
               [cell("Served ALS (exact)"), cell("R@20 ~0.0375"), cell("V2 served-c2 &mdash; NOT_COMPARABLE to V1")],
               [cell("FAISS FlatIP"), cell("lossless, ~47% lower p95"), cell("V2 latency &mdash; latency only")],
               [cell("HNSW ef64"), cell("overlap 0.992, ~2.4x"), cell("V2 latency")],
               [cell("Cold-start fallback cost"), cell("coverage collapse ~59x; personalization 0.68&rarr;0.001"), cell("V2 served-c2")],
               [cell("Exposure concentration"), cell("all policies Gini&gt;0.97; popularity 1.0"), cell("V2 &mdash; concentration, not fairness")],
               [cell("Semantic cold-start reach"), cell("54.6% of unseen golds (ALS/pop = 0)"), cell("V2 &mdash; reachability, offline")],
               [cell("Learned fusion vs ALS-only"), cell("test R@20 0.036 vs 0.022; cold 0.032 vs 0"), cell("V2 &mdash; held-out test, 81 positives, directional")],
               [cell("OPE estimators"), cell("DR 0.0089 vs true 0.0106; IPS over-est; SNIPS stable"), cell("V2 &mdash; offline, synthetic props, proxy reward")],
               [cell("Search/IR retrieval"), cell("BM25 R@20 0.0133 &gt; dense 0.0067; hybrid 0.0117"), cell("V2 &mdash; seed-item, single-relevant NDCG/MRR")],
              ], [1.7*inch, 2.35*inch, 2.2*inch]),
          Spacer(1,5),
          P("Every cell above is offline. Small-sample caveats (e.g. 81 test positives in the fusion result) are stated, not hidden; "
            "the learned-beats-ALS claim is restricted to the offline test split and is directionally consistent across four model families.", SMALL)]

# ---------- 7. Truth Boundary ----------
story += [PageBreak(), P("7 &nbsp; Truth Boundary", H1), rule(),
          P("Claimable (artifact-backed):", H2),
          bullets([
           "Built an offline two-stage recommender decision system on 2.56M real Goodreads interactions.",
           "ALS is the warm-retrieval floor &mdash; earned by beating a <i>converged</i> SASRec (documented honest negative).",
           "FAISS latency&ndash;quality frontier: FlatIP exact lossless ~47% faster; HNSW near-lossless; IVF rejected for overlap collapse.",
           "Production-<i>like</i> FastAPI+FAISS serving with versioned loader, fallback tree, logging, monitoring &mdash; and it is now <b>deployed to Cloud Run serving live HTTPS</b>.",
           "Measured cold-start fallback quality and catalog exposure concentration; added a semantic content lane that uniquely reaches cold-start items.",
           "A LightGBM LambdaMART fusion beat ALS-only on the held-out test split and recovered cold-start (offline, small positive set).",
           "Executed the OPE pipeline (logging + IPS/SNIPS/DR) and validated estimators recover a known value offline.",
          ]),
          P("Forbidden (never claimed):", H2),
          bullets([
           "Online lift / engagement / A/B impact &mdash; no live experiment was run.",
           "Production <i>quality</i> or \"served real users to effect\" &mdash; deployed and callable is true; business impact is not.",
           "Solved cold-start, solved long-tail discovery, or fairness certified.",
           "LLM recommender / semantic taste understanding &mdash; the semantic lane is embedding-based candidate generation.",
           "Learned ranker beats ALS beyond the offline test split; real off-policy value (props were synthetic, reward a proxy).",
           "Any cross-protocol comparison of V1 d3aplus and served-c2 numbers.",
          ], SMALL)]

# ---------- 8. Hard Q&A ----------
story += [PageBreak(), P("8 &nbsp; Hard Q&amp;A", H1), rule()]
qa = [
 ("Why did SASRec lose to ALS?",
  "Book next-item is weakly sequence-predictable; the dominant signal is collaborative co-read, which MF captures directly. I trained a canonical full-softmax SASRec to convergence (~95 epochs, eval-R@20 plateau) and it still scored below ALS &mdash; depth was not the lever on short, sparse histories."),
 ("Was the SASRec just undertrained?",
  "No. I monitored convergence on eval-R@20 (after catching a NaN val-loss false-stop), and the full-softmax-vs-sampled-softmax gap (~6.7x) showed the loss objective was the real lever &mdash; and even fixed, it didn't beat ALS."),
 ("What's the OPE variance problem and how did SNIPS help?",
  "Raw IPS uses importance weights &pi;<sub>e</sub>/&pi;<sub>log</sub>; at modest effective sample size a few large weights blow up the variance (my IPS over-estimated ~2x). SNIPS self-normalises by the sum of weights, trading a little bias for much lower variance &mdash; it's my recommended estimator when no reward model exists; DR is lowest-bias when a calibrated reward model does."),
 ("Why was DR negative the first time?",
  "The DR reward model was a class-balanced logistic that calibrated toward ~0.5 vs the true ~0.01 base rate, so it massively over-predicted and broke the DR baseline. I caught it from the impossible negative value, switched to a base-rate-calibrated model, and DR then recovered the known value best of the three."),
 ("What is the real cost of cold-start here?",
  "Not a recall drop &mdash; recall is actually non-monotonic (cold users' next item is more popularity-predictable, so it's easier to retrieve). The real cost is that popularity fallback collapses catalog coverage ~59x and personalization from 0.68 to ~0.001. I reframed the metric around what actually degrades."),
 ("Why did you reject IVF if it was faster?",
  "At low nprobe IVF kept gold Recall@20 flat but candidate-overlap@K vs exact collapsed to 0.28&ndash;0.78 &mdash; it served a largely different, broken candidate set. Speed that silently destroys candidate fidelity is a recall bug, so I kept FlatIP/HNSW and made overlap@K an acceptance check."),
 ("Why did BM25 beat dense on this data?",
  "On short text (title + genre shelves), exact lexical overlap &mdash; shared series names, shared genre words &mdash; carries more signal than a 384-d embedding that compresses and blurs exactly those rare disambiguating tokens. Embeddings win on long, paraphrase-heavy text; lexical wins on short, keyword-dense text. So I reported BM25 as the strongest single lane and hybrid as robustness, not a win."),
 ("So is your hybrid retrieval better?",
  "On recall, no &mdash; RRF hybrid sits below BM25 alone on this seed-item task. Its value is robustness: it never collapses to either lane's weakness and it reaches cold-start. I don't claim hybrid beats both lanes."),
 ("Did the learned ranker actually beat ALS?",
  "On the held-out test split, yes (R@20 0.036 vs 0.022) and it recovered cold-start (0.032 vs 0), consistent across four model families &mdash; but on only 81 positives with a single held-out gold, offline. I treat it as directional, not a production claim, pending OPE/online evidence."),
 ("Why does fusion help at all?",
  "ALS and semantic candidate sets are almost disjoint &mdash; they overlap on only 6,749 of ~880k candidates. A learned ranker can pick the best items across both complementary sources, which is what lifted test recall and cold-start."),
 ("What does the live Cloud Run endpoint actually serve?",
  "The served-c2 ALS model with exact FlatIP FAISS retrieval: GET /recommend returns top-K items with source tags and a graceful fallback (unknown user &rarr; popularity, never empty). /health reports loader readiness. It is the real serving path &mdash; not an online-lift experiment. \"Deployed and serving HTTPS\" is true; \"improved engagement\" is not."),
 ("Is any of this production / online?",
  "No. Everything is offline by scope. The serving API is deployed and callable, but there are no real users, no logged real-traffic propensities, and no A/B. That boundary is documented as the explicit next step, not hidden."),
 ("What would falsify your headline claims?",
  "ALS-floor: a properly-tuned sequence/graph model beating it on this exact protocol. Fusion-win: it not holding on a larger positive set or a different relevance target. Semantic value: cold-start reachability not translating to any online engagement. OPE: ESS too low / no overlap. I state these kill conditions rather than hide them."),
]
for q, a in qa:
    story += [P("Q. " + q, ParagraphStyle("Q", parent=BODY, fontSize=9.4, leading=12.5, textColor=NAVY, spaceAfter=1, fontName="DJ-Bold")),
              P("A. " + a, ParagraphStyle("A", parent=BODY, fontSize=9.0, leading=12.2, spaceAfter=7))]

# ---------- 9. What I'd Build Next ----------
story += [PageBreak(), P("9 &nbsp; What I'd Build Next", H1), rule(),
          P("With production access, in priority order:"),
          bullets([
           "<b>Real-traffic propensity logging</b> &mdash; populate the OPE schema (already designed) with live &pi;<sub>log</sub>, position, exploration bucket and reward, to turn IPS/SNIPS/DR from a validated methodology into real off-policy value estimates.",
           "<b>Canary + live A/B</b> of the fusion policy and the head-cap exposure rerank, with guardrail metrics (relevance, coverage, Gini, fallback rate, p95) wired to alerts &mdash; the only way to claim online lift.",
           "<b>In-ranker position-bias correction (PAL / IPS-weighted LTR)</b> &mdash; upgrade position bias from <i>evaluated</i> to <i>corrected</i>.",
           "<b>Productionise the lexical lane</b> (Lucene/Elasticsearch/bm25s) so BM25 is sub-millisecond, and persist the FAISS index to cut cold-start build time.",
           "<b>Two-tower / GNN retrieval and graded-relevance labels</b> &mdash; only once there's signal/scale to justify them; today they're deferred, not silently skipped.",
          ]),
          Spacer(1, 10), rule(),
          P("Companion material: docs/PULSEDISCOVERY_INTERVIEW_KIT.md (pitch &amp; claim ladder), "
            "docs/PULSEDISCOVERY_UNIFIED_DEFENSE_KERNEL.md (full method cards), "
            "docs/PULSEDISCOVERY_FAILURES_AND_HARDENING.md (failures + serving stress test), "
            "docs/G26A_MODEL_PROTOCOL_REGISTRY.md (V1/V2 metric registry).", SMALL)]

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DJ", 7.5); canvas.setFillColor(GREY)
    canvas.drawString(0.75*inch, 0.5*inch, "PulseDiscover — Interview Defense Dossier")
    canvas.drawRightString(7.75*inch, 0.5*inch, "Offline by scope • every claim artifact-backed • page %d" % doc.page)
    canvas.restoreState()

doc = SimpleDocTemplate("defense/PulseDiscover_Interview_Defense.pdf", pagesize=letter,
                        leftMargin=0.75*inch, rightMargin=0.75*inch, topMargin=0.7*inch, bottomMargin=0.7*inch,
                        title="PulseDiscover Interview Defense", author="PulseDiscover")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("built defense/PulseDiscover_Interview_Defense.pdf")
