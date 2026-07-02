"""Build the PulseDiscover interview-defense PDF — comprehensive, dark, portfolio-styled."""
import os, matplotlib
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                PageBreak, Image, KeepTogether, ListFlowable, ListItem, Flowable)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily

_F = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
pdfmetrics.registerFont(TTFont("DJ", os.path.join(_F, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DJ-Bold", os.path.join(_F, "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DJ-Italic", os.path.join(_F, "DejaVuSans-Oblique.ttf")))
registerFontFamily("DJ", normal="DJ", bold="DJ-Bold", italic="DJ-Italic", boldItalic="DJ-Bold")

# palette
BG      = colors.HexColor("#0d1b2c")
CARD    = colors.HexColor("#13243a")
LINE    = colors.HexColor("#24405e")
INK     = colors.HexColor("#dbe6f3")
MUTE    = colors.HexColor("#90a5c0")
BLUE    = colors.HexColor("#6db0e8")
GREEN   = colors.HexColor("#5fc46e")
AMBER   = colors.HexColor("#f0a93a")
RED     = colors.HexColor("#e1685f")
PURPLE  = colors.HexColor("#b3a8ee")

ss = getSampleStyleSheet()
def st(name, **kw):
    base = kw.pop("parent", ss["BodyText"])
    return ParagraphStyle(name, parent=base, fontName=kw.pop("fontName", "DJ"), **kw)

TITLE = st("t", fontName="DJ-Bold", fontSize=30, textColor=INK, alignment=TA_CENTER, leading=34)
SUBT  = st("s", fontSize=12.5, textColor=MUTE, alignment=TA_CENTER, leading=17)
H1    = st("h1", fontName="DJ-Bold", fontSize=17, textColor=INK, spaceBefore=4, spaceAfter=8, leading=21)
H2    = st("h2", fontName="DJ-Bold", fontSize=12.5, textColor=BLUE, spaceBefore=10, spaceAfter=3, leading=15)
BODY  = st("b", fontSize=9.6, textColor=INK, leading=14, spaceAfter=6, alignment=TA_LEFT)
SMALL = st("sm", fontSize=8.5, textColor=MUTE, leading=11.5, spaceAfter=4)
LEADQ = st("q", fontName="DJ-Bold", fontSize=10, textColor=BLUE, leading=13, spaceBefore=6, spaceAfter=1)
ANS   = st("a", fontSize=9.3, textColor=INK, leading=12.8, spaceAfter=4)
CELL  = st("c", fontSize=8.4, textColor=INK, leading=11)
CELLm = st("cm", fontSize=8.4, textColor=MUTE, leading=11)
CHIP  = st("chip", fontName="DJ-Bold", fontSize=8.5, textColor=colors.white, alignment=TA_CENTER, leading=11)

def P(t, s=BODY): return Paragraph(t, s)
def cell(t, s=CELL): return Paragraph(t, s)

def bullets(items, s=BODY, color=BLUE):
    return ListFlowable([ListItem(P(x, s), leftIndent=4) for x in items],
                        bulletType="bullet", bulletColor=color, leftIndent=14, bulletFontSize=6, spaceAfter=0)

def divider(color=LINE):
    t = Table([[""]], colWidths=[7.3*inch], rowHeights=[1.2])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),color)])); return t

def callout(title, body, accent=AMBER, fill=colors.HexColor("#2c1f0b")):
    inner = [P(f"<font color='#{accent.hexval()[2:8]}'><b>{title}</b></font>", st("ct", fontSize=10, leading=13)),
             P(body, st("cb", fontSize=9, textColor=INK, leading=12.5))]
    t = Table([[inner]], colWidths=[7.3*inch])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),fill),("BOX",(0,0),(-1,-1),1,accent),
                           ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10),
                           ("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7)]))
    return t

def card(title, body, accent=BLUE):
    inner = [P(f"<font color='#{accent.hexval()[2:8]}'><b>{title}</b></font>", st("k", fontSize=10.5, leading=13, spaceAfter=2)),
             P(body, st("kb", fontSize=9, textColor=INK, leading=12.4))]
    t = Table([[inner]], colWidths=[7.3*inch])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),CARD),("LINEBEFORE",(0,0),(0,-1),3,accent),
                           ("BOX",(0,0),(-1,-1),0.5,LINE),
                           ("LEFTPADDING",(0,0),(-1,-1),11),("RIGHTPADDING",(0,0),(-1,-1),10),
                           ("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7)]))
    return KeepTogether([t, Spacer(1,7)])

def dtable(rows, widths, header=True, font=8.4):
    data = [[cell(c, CELL if (not header or i>0) else st("ch", fontName="DJ-Bold", fontSize=8.6, textColor=colors.white, leading=11)) for c in r] for i,r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("VALIGN",(0,0),(-1,-1),"TOP"),("GRID",(0,0),(-1,-1),0.4,LINE),
             ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
             ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
             ("ROWBACKGROUNDS",(0,1 if header else 0),(-1,-1),[BG, CARD])]
    if header: style += [("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1c3553"))]
    t.setStyle(TableStyle(style)); return t

def chips(items):
    cs = {"blue":BLUE,"green":GREEN,"amber":AMBER,"red":RED,"purple":PURPLE,"grey":colors.HexColor("#46607f")}
    cells = [Paragraph(txt, CHIP) for txt,_ in items]
    t = Table([cells], colWidths=[7.3*inch/len(items)]*len(items))
    sty = [("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
           ("LEFTPADDING",(0,0),(-1,-1),3),("RIGHTPADDING",(0,0),(-1,-1),3)]
    for i,(_,c) in enumerate(items): sty.append(("BACKGROUND",(i,0),(i,0),cs[c]))
    t.setStyle(TableStyle(sty)); return t

story = []

# ============ COVER ============
story += [Spacer(1, 24), P("PulseDiscover", TITLE),
          P("Interview Defense Dossier", SUBT),
          P("Offline, honesty-gated recommender <b>decision</b> system &mdash; 2.56M Goodreads interactions", SUBT),
          Spacer(1, 14),
          chips([("Python","blue"),("FastAPI","green"),("FAISS","blue"),("LambdaMART","green"),("OPE: IPS/SNIPS/DR","purple"),("Cloud Run: LIVE","green")]),
          Spacer(1, 14),
          Image("defense/_arch.png", width=7.0*inch, height=7.0*inch*660/1000),
          Spacer(1, 12),
          callout("STATUS",
                  "V1 <b>gold_candidate 8.9</b> (locked, terminal) &middot; V2 lane <b>offline gold-complete</b> (RiskFrame 9.1) &middot; "
                  "serving API <b>deployed to Google Cloud Run</b> (live HTTPS) &middot; V3 gate <b>two-tower + Thompson exploration</b> "
                  "(offline, honest negative + explore/exploit tradeoff).", accent=GREEN, fill=colors.HexColor("#12301c")),
          Spacer(1,6),
          callout("READING RULE",
                  "Every number in this dossier is <b>offline and protocol-tagged</b>. The V1 d3aplus result (R@20 0.0846) and the served-c2 "
                  "result (R@20 ~0.0375) are <b>different builds and protocols &mdash; NOT comparable</b>. No online-lift, real-user, or "
                  "production-quality claim appears anywhere.", accent=RED, fill=colors.HexColor("#2a1414")),
          PageBreak()]

# ============ 0. CONTENTS ============
story += [P("Contents", H1), divider(),
          dtable([["#","Section"],
                  ["1","Project summary &amp; thesis"],["2","System architecture"],
                  ["3","Fundamentals bridge &mdash; 14 method cards"],["4","Decision log"],
                  ["5","Failures I'm proud of (8 post-mortems)"],["6","Eval results &amp; model/protocol registry"],
                  ["7","Gate-by-gate walkthrough (G22&ndash;G31)"],["8","Off-policy evaluation deep dive"],
                  ["9","Cold-start &amp; exposure deep dive"],["10","Truth boundary &amp; claim ladder"],
                  ["11","Product &amp; business reasoning"],["12","Hard Q&amp;A (22 questions)"],
                  ["13","What I'd build next"],["14","Evidence artifact index"],
                  ["V3","Neural two-tower + Thompson exploration bandit (G32&ndash;G33)"]],
                 [0.5*inch, 6.8*inch]),
          PageBreak()]

# ============ 1. SUMMARY ============
story += [P("1 &nbsp; Project Summary &amp; Thesis", H1), divider(),
          P("<b>Thesis.</b> Recommenders rarely fail because the model was wrong. They fail because a metric improved while catalog health, "
            "candidate fidelity, or cold-start coverage silently collapsed &mdash; and nobody measured the gap. PulseDiscover is the "
            "measurement-and-decision layer that catches that gap before it reaches users."),
          P("<b>What it is.</b> An offline, honesty-gated two-stage recommender decision system on a real 2.56M-interaction Goodreads "
            "fantasy/paranormal corpus (94k users, ~42k items; 40,541 in the served ALS catalog). Four candidate sources (ALS, dense semantic, BM25, popularity) feed a FAISS "
            "retrieval layer, a learned ALS+semantic fusion ranker, a config-flagged exposure rerank, and a graceful fallback tree, served "
            "by a FastAPI app deployed on Cloud Run &mdash; with cohort, exposure, and off-policy evaluation governing every ship decision."),
          P("<b>How it was built.</b> As a gated experimentation program (G11&ndash;G31). Each stage was added, measured, and either adopted "
            "or rejected with an artifact in <font face='Courier'>outputs/evidence/</font>. V1 established the offline modelling floor; the V2 "
            "lane (G22&ndash;G31) added serving, cold-start/fallback quality, exposure governance, a heuristic exposure rerank, dense semantic "
            "retrieval, a learned fusion tournament, executed off-policy evaluation, and a search/IR front end &mdash; then the serving path "
            "shipped to Cloud Run."),
          P("<b>Why V1 is gold_candidate, not gold_complete.</b> The canonical SASRec convergence headline is provenance-tagged "
            "<i>plot_reconstructed</i> (the direct artifact was lost with a Colab session). The decision it supports (MF beats a converged "
            "SASRec) is artifact-backed; the packaging is one artifact short of clean &mdash; so the status is honestly held at candidate."),
          P("<b>What it deliberately is NOT.</b> Not an online/production-quality system, not solved cold-start, not fairness-certified, not "
            "an LLM recommender. Those boundaries are load-bearing and stated throughout.", SMALL),
          PageBreak()]

# ============ 2. ARCHITECTURE ============
story += [P("2 &nbsp; System Architecture", H1), divider(),
          P("Two lanes share one honest boundary. The <b>offline modelling lane</b> (V1, d3aplus) produced the findings &mdash; ALS is the "
            "warm floor, a converged SASRec still lost. The <b>served system lane</b> (V2, c2 &rarr; Cloud Run) is what actually runs. The "
            "amber bridge is the protocol boundary that keeps the two from being conflated."),
          Image("defense/_arch.png", width=7.2*inch, height=7.2*inch*660/1000),
          Spacer(1,8),
          P("Per-stage contract", H2),
          dtable([["Stage","Input &rarr; Output","Claim enabled / not enabled"],
                  ["Candidate gen","interactions &rarr; candidate pool","\"coverage is the binding lever (+~44% recall)\" / not online impact"],
                  ["FAISS retrieval","embeddings &rarr; top-K","\"FlatIP lossless ~47% faster; HNSW near-lossless\" / not quality gain"],
                  ["Fusion / rank","pool+features &rarr; ordered slate","\"learned fusion beat ALS-only on test\" / not in production"],
                  ["Exposure rerank","slate &rarr; reranked slate","\"interpretable exposure control\" / not learned LTR"],
                  ["Fallback","any request &rarr; non-empty slate","\"0% empty across failure modes\" / not fallback quality"],
                  ["Serving","user_id,k,mode &rarr; JSON","\"production-like, deployed to Cloud Run\" / not online lift"],
                  ["Evaluation","logs/splits &rarr; metrics","\"cohort + exposure + OPE\" / not real-traffic value"],
                  ["Decision","frontiers &rarr; ship/hold/reject","\"I decide what ships and what must not\" / not deployed-and-proven"]],
                 [1.05*inch, 2.6*inch, 3.65*inch]),
          PageBreak()]

# ============ 3. FUNDAMENTALS BRIDGE ============
story += [P("3 &nbsp; Fundamentals Bridge &mdash; method cards", H1), divider(),
          P("Each technique from first principles: objective &middot; mechanism &middot; assumptions &middot; failure modes &middot; why-over-alternatives &middot; role in PulseDiscover.", SMALL)]
cards = [
 ("ALS &mdash; implicit-feedback matrix factorization", BLUE,
  "<b>Objective:</b> min &Sigma; c<sub>ui</sub>(p<sub>ui</sub> &minus; x<sub>u</sub>&middot;y<sub>i</sub>)&sup2; + &lambda;(&#8214;x&#8214;&sup2;+&#8214;y&#8214;&sup2;), confidence c=1+&alpha;r, preference p=1[r&gt;0]. "
  "<b>Mechanism:</b> alternate closed-form ridge solves per side &mdash; x<sub>u</sub>=(Y<sup>T</sup>C<sub>u</sub>Y+&lambda;I)<sup>-1</sup>Y<sup>T</sup>C<sub>u</sub>p. "
  "<b>Assumes:</b> low-rank structure; missing &asymp; weak-negative. <b>Fails on:</b> cold users / niche tail. <b>Over BPR:</b> closed-form, fast, stable. "
  "<b>Role:</b> the warm-retrieval floor and the served model (c2)."),
 ("SASRec &mdash; self-attentive sequential recommendation", BLUE,
  "<b>Objective:</b> maximize next-item likelihood (full softmax over the catalog). <b>Mechanism:</b> causal self-attention over the item-embedding sequence. "
  "<b>Assumes:</b> informative order + dense-enough sequences. <b>Fails on:</b> short/sparse/padded histories (the NaN-mask bug). <b>Lever found:</b> full-softmax vs sampled-softmax (~6.7&times;) &mdash; still &lt; ALS. "
  "<b>Role:</b> the deep-model challenger, trained to convergence; a documented honest negative."),
 ("Two-tower &mdash; ID + content retrieval (V3, G32)", BLUE,
  "<b>Objective:</b> user tower = mean-pooled history item-embeddings; item tower = learned-ID embedding &oplus; a projection of the 384-d MiniLM content vector; dot-product scored, BPR loss (8 uniform negatives), 23 epochs, PyTorch CPU. "
  "<b>Result:</b> R@20 <b>0.0642</b> [0.057&ndash;0.071] on the exact ALS-floor protocol &mdash; a documented honest negative vs ALS 0.0846 (non-overlapping CIs). Content ablation: turning the content tower off drops R@20 to 0.0432, so MiniLM content contributes <b>+0.021 R@20</b>. "
  "<b>Reading:</b> content is now first-class inside retrieval and earns a real lift, but still loses to a tuned MF on dense warm book histories &mdash; the second neural model to do so (after SASRec)."),
 ("LightGCN &mdash; graph CF (deferred, documented)", BLUE,
  "<b>Mechanism:</b> light graph convolution (no feature transforms/nonlinearities) propagating CF signal over the user-item graph. "
  "<b>Status:</b> deferred &mdash; graph training over ~42k items is not cheap on CPU, and on a catalog where a converged SASRec and a content two-tower both lost to MF it is unlikely to pay. Documented, not silently skipped."),
 ("LambdaMART &mdash; learning-to-rank fusion", GREEN,
  "<b>Objective:</b> boosted trees optimizing a listwise NDCG surrogate; gradients are &lambda;-weighted by the NDCG change of swapping a pair, grouped per query (user). "
  "<b>Assumes:</b> enough positive labels/query; features carry cross-source signal. <b>Fails on:</b> extreme label sparsity (overfit). "
  "<b>Role:</b> the G28 fusion ranker over ALS+semantic+popularity candidate features &mdash; champion of the tournament."),
 ("FAISS &mdash; ANN retrieval (FlatIP / HNSW / IVF)", GREEN,
  "<b>FlatIP:</b> exact exhaustive inner product (O(N), lossless). <b>HNSW:</b> navigable-small-world graph, efSearch trades recall/latency. <b>IVF:</b> coarse quantiser, nprobe controls cells scanned. "
  "<b>Failure:</b> aggressive nprobe shreds the candidate pool while a single-gold metric stays flat. <b>Role:</b> FlatIP default, HNSW scale option, IVF rejected."),
 ("Dense semantic content retrieval", GREEN,
  "<b>Objective:</b> embed item content (title + genre shelves) with MiniLM (384-d, L2-normalized); FAISS IndexFlatIP (cosine-equiv); item-to-item from the seed item. "
  "<b>Assumes:</b> content similarity &asymp; taste at genre level; metadata text informative (100% coverage). <b>Fails on:</b> head relevance (collaborative signal dominates), zero-history users. "
  "<b>Role:</b> the only source that structurally reaches item-cold-start items (reachability 54.6%)."),
 ("RRF &mdash; reciprocal rank fusion", GREEN,
  "<b>Objective:</b> score(i)=&Sigma;<sub>s</sub> 1/(c+rank<sub>s</sub>(i)) over sources; score-scale-free, training-free. <b>Assumes:</b> rank is a comparable cross-source signal. "
  "<b>Fails on:</b> nothing catastrophic &mdash; but it buys robustness, not a recall lift. <b>Role:</b> best non-learned fusion; ALS-first fusion fails (saturates top-20), RRF lets semantic compete."),
 ("IPS &mdash; inverse propensity scoring", PURPLE,
  "<b>Objective:</b> V&#770;(&pi;<sub>e</sub>)=(1/N)&Sigma; [1{a<sub>i</sub>=&pi;<sub>e</sub>(x<sub>i</sub>)}/&pi;<sub>log</sub>(a<sub>i</sub>|x<sub>i</sub>)]&middot;r<sub>i</sub>. <b>Assumes:</b> overlap (&pi;<sub>log</sub>&gt;0 on &pi;<sub>e</sub>'s support); correct propensities. "
  "<b>Fails on:</b> high variance at modest ESS (here it over-estimated ~2&times;). <b>Role:</b> executed in G29; the cautionary high-variance baseline."),
 ("SNIPS &mdash; self-normalized IPS", PURPLE,
  "<b>Objective:</b> &Sigma; w<sub>i</sub>r<sub>i</sub> / &Sigma; w<sub>i</sub>, w<sub>i</sub>=1{a=&pi;<sub>e</sub>}/&pi;<sub>log</sub>. Trades a little bias for much lower variance. "
  "<b>Role:</b> the recommended estimator when no reward model exists &mdash; stable and close to the known value in G29."),
 ("Doubly-Robust (DR) OPE", PURPLE,
  "<b>Objective:</b> DR = mean(r&#770;(x,&pi;<sub>e</sub>)) + mean(w&middot;(r &minus; r&#770;(x,a))); unbiased if either propensities or reward model is right. "
  "<b>Fails on:</b> a mis-calibrated reward model (a class-balanced r&#770; broke it &mdash; went negative). <b>Role:</b> lowest-bias estimator once r&#770; is base-rate-calibrated (DR 0.0089 vs true 0.0106)."),
 ("Position bias &amp; PAL", AMBER,
  "<b>Problem:</b> top-ranked items get examined more, so click-recall measures position as much as relevance. PAL/IPS de-bias examination from relevance. "
  "<b>Role:</b> PulseDiscover does position-bias-aware <i>evaluation</i> (a two-lane OPE diagnostic) &mdash; not an in-ranker correction (that is labelled future work)."),
 ("Exposure governance &mdash; Gini / Lorenz / base-rate lift", AMBER,
  "<b>Objective:</b> over the full catalog (zeros included) compute Gini = 2&Sigma;i&middot;x<sub>i</sub>/(n&Sigma;x<sub>i</sub>) &minus; (n+1)/n, the Lorenz curve, top-x% share, and tier exposure-lift vs catalog base rate. "
  "<b>Failure:</b> Gini alone hides which tail &mdash; pair with tier-lift. <b>Role:</b> catalog-health as a first-class ship-gate metric (all policies Gini&gt;0.97; popularity 0.9995)."),
 ("Cold-start cohorts &amp; fallback policy", AMBER,
  "<b>Mechanism:</b> cohort users (unknown / sparse 1&ndash;2 / low 3&ndash;5 / warm 6+); evaluate popularity/content/item-sim/blended fallback; report warm-ALS and cold-fallback metrics SEPARATELY. "
  "<b>Honest nuance:</b> per-cohort recall is non-monotonic (cold gold is more popularity-predictable) &mdash; recall understates the real cost, which is coverage/personalization collapse."),
 ("Negative sampling", colors.HexColor("#46607f"),
  "<b>Idea:</b> sampled negatives approximate the full softmax; popularity-biased samplers skew the gradient. <b>Role:</b> implicit in ALS/sequence training; the full-softmax-vs-sampled lever in G20 was exactly this choice."),
]
for name, acc, body in cards:
    story.append(card(name, body, acc))
story.append(PageBreak())

# ============ 4. DECISION LOG ============
story += [P("4 &nbsp; Decision Log", H1), divider(),
          dtable([["Decision","Rationale","Boundary kept"],
                  ["ALS over SASRec (warm floor)","Converged full-softmax SASRec still lost (0.065 vs 0.085); depth isn't the lever on sparse book histories.","\"a converged deep model lost\" &mdash; not \"deep is useless\""],
                  ["FlatIP exact as default","Lossless &amp; ~47% faster; HNSW near-lossless but approximate. Correctness by default, opt into approximation.","HNSW is a flagged scale mode, not silent default"],
                  ["IVF rejected","Low-nprobe kept gold recall flat while candidate-overlap collapsed 0.28&ndash;0.78 &mdash; a broken pool.","overlap@K is a first-class acceptance check"],
                  ["RRF = robustness not win","Recovers cold-start + ~4&times; coverage at ~95% of ALS relevance; does NOT beat ALS-only on warm recall.","no \"hybrid beats both lanes\" claim"],
                  ["Semantic as complement","Uniquely reaches item-cold-start (54.6%) but loses head relevance (0.010 vs 0.087).","cold-start/tail source, not a warm-path replacement"],
                  ["BM25 = query-by-document","No free-text queries exist; the seed item's text is the query (more-like-this).","seed-item, not a free-text search engine"],
                  ["Learned fusion, bounded","LambdaMART beat ALS-only on held-out test (0.036 vs 0.022) but on 81 positives.","claimed on the offline test split only"],
                  ["OPE executed, not faked","Estimators recover a known value offline; props synthetic, reward a proxy.","methodology demonstrated, not a real value"],
                  ["V1/V2 registry","d3aplus 0.0846 and c2 0.0375 are different builds/protocols.","never conflated &mdash; kept in a registry"],
                  ["Two-tower = honest negative (V3)","ID+content BPR two-tower R@20 0.064 vs ALS 0.085 (non-overlapping CIs); content ablation adds +0.021.","content earns a lift; MF still wins warm"],
                  ["Bandit = exploration, not relevance (V3)","Thompson raised coverage 17&rarr;51% and cut Gini 0.97&rarr;0.81 for a ~1.3pp relevance cost.","explore/exploit stated as a COST, offline only"]],
                 [1.55*inch, 4.0*inch, 1.75*inch]),
          PageBreak()]

# ============ 5. FAILURES ============
story += [P("5 &nbsp; Failures I'm Proud Of", H1), divider(),
          P("The interview value of a project is the failures you catch yourself. Each: what happened &rarr; what it revealed &rarr; what I did.", SMALL)]
fails = [
 ("SASRec false convergence (NaN val-loss patience)",
  "Training early-stopped at ~6 epochs on an all-NaN val-loss curve (2-block masked-attention NaN on short/padded sequences); a NaN comparison silently satisfied patience while train loss + eval-R@20 were still rising. <b>Revealed:</b> \"early stop fired\" &ne; \"converged.\" <b>Did:</b> rejected as invalid, rebuilt the monitor on an eval-R@20 plateau (valid convergence at ~95 epochs)."),
 ("v2 eval leakage (same set for stopping + final eval)",
  "SASRec v2 used the 5k eval set for BOTH the early-stopping criterion and the final metric &mdash; model selection peeking at test. <b>Revealed:</b> reusing a held-out set for selection leaks. <b>Did:</b> split a separate validation-monitor set; fixed the epoch cap to an exact value."),
 ("Stale-script contamination on Colab",
  "Twice, Colab ran a cached v1 script instead of v3 and reported old numbers. <b>Revealed:</b> \"I updated the script\" &ne; \"the run used it.\" <b>Did:</b> hardened the notebook to write/run a distinctly-named file with assert guards; provenance-tagged the unrecoverable headline plot_reconstructed."),
 ("IVF &mdash; the metric can lie",
  "IVF looked fine on gold Recall@20 while candidate-overlap@K vs exact collapsed to 0.28&ndash;0.78 &mdash; a mostly different candidate set that happened to contain the gold. <b>Revealed:</b> a single-gold metric is blind to most of the slate changing. <b>Did:</b> rejected IVF; made overlap@K a first-class check."),
 ("ID-dtype silent zeroing",
  "book_id is a string in the ALS index but ints in eval; every membership test compared int vs str keys, so every lookup missed and recall floored to ~0 &mdash; while the pipeline ran clean with plausible numbers. <b>Revealed:</b> silent type coercion is where pipelines rot. <b>Did:</b> caught it via an overlap sanity check (0 catalog-overlapping keys = impossible if types matched); normalized to str repo-wide."),
 ("G24 recall non-monotonicity",
  "Cold cohorts scored HIGHER recall than warm &mdash; the opposite of the expected story. <b>Revealed:</b> cold users' next item is more popularity-predictable, so recall understates cold-start cost. <b>Did:</b> refused the expected narrative; reframed the real cost as coverage (~59&times;) + personalization (0.68&rarr;0.001) collapse."),
 ("G29 Doubly-Robust went negative",
  "DR returned ~&minus;0.4 (impossible for a [0,1] reward). <b>Revealed:</b> a class-balanced reward model calibrated to ~0.5 vs a true ~1% base rate broke the DR baseline. <b>Did:</b> caught it from the impossible sign; switched to a base-rate-calibrated model; DR then recovered the known value best of the three."),
 ("Serving load-path false confidence (caught in the stress test)",
  "The startup dim-assert was tautological (index built from Y, so it can't fail) and a corrupt model pickle returned 0 items (popularity not yet loaded). <b>Revealed:</b> a health check that can't fail is worse than none. <b>Did:</b> added a real X/Y dim assert; load popularity independently so a degraded service still serves a non-empty fallback."),
 ("V3 OPE overlap collapse for the exploration policy",
  "Reusing the G29 IPS/SNIPS/DR estimators on the Thompson bandit, DR went slightly negative and IPS/SNIPS hit 0 &mdash; while they cleanly recovered the greedy policy (DR 0.005 vs true 0.006). <b>Revealed:</b> the exploration policy picks items the logging policy rarely logs, so effective sample size collapsed 168&rarr;21 and the estimators lost all power. <b>Did:</b> reported it as the finding, not a bug &mdash; you cannot off-policy-evaluate an exploration policy a logging policy doesn't cover; real logged propensities are the prerequisite."),
]
for name, body in fails:
    story.append(card(name, body, RED))
story.append(PageBreak())

# ============ 6. EVAL + REGISTRY ============
story += [P("6 &nbsp; Eval Results &amp; Model/Protocol Registry", H1), divider(),
          callout("PROTOCOL WARNING (load-bearing)",
                  "V1 <b>d3aplus</b> R@20 <b>0.0846</b> and V2 <b>served-c2</b> R@20 <b>~0.0375</b> are different model builds and different "
                  "evaluation protocols (V2 = full-catalog single-held-out-gold). <b>They are NOT comparable.</b> V2 numbers are used for "
                  "relative cross-policy/cohort structure, never as a headline that overturns V1.", accent=RED, fill=colors.HexColor("#2a1414")),
          Spacer(1,6),
          P("Headline metrics", H2),
          dtable([["Result","Number","Protocol tag"],
                  ["ALS warm floor","R@20 0.0846","V1 d3aplus tuning &mdash; REAL_OFFLINE"],
                  ["SASRec converged","R@20 0.065 (lost)","V1 &mdash; PLOT_RECONSTRUCTED"],
                  ["Served ALS exact","R@20 ~0.0375","V2 served-c2 &mdash; NOT_COMPARABLE"],
                  ["FAISS FlatIP","lossless, ~47% lower p95","V2 latency"],
                  ["HNSW ef64","overlap 0.992, ~2.4&times;","V2 latency"],
                  ["Cold-start fallback","coverage ~59&times;; pers. 0.68&rarr;0.001","V2 served-c2"],
                  ["Semantic reach","54.6% unseen golds","V2 served-c2"],
                  ["Learned fusion","test 0.036 vs 0.022","V2 &mdash; 81 positives, directional"],
                  ["OPE","DR 0.0089 vs true 0.0106","V2 &mdash; offline, proxy reward"],
                  ["Exposure","Gini &gt;0.97; pop 0.9995","V2 &mdash; concentration not fairness"],
                  ["Search/IR","BM25 0.0133 &gt; dense 0.0067","V2 &mdash; seed-item NDCG/MRR"],
                  ["Two-tower (ID+content)","R@20 0.0642 (lost to ALS)","V3 G32 &mdash; ALS-floor protocol"],
                  ["Two-tower content ablation","+0.021 R@20 from MiniLM","V3 G32"],
                  ["Thompson vs greedy","cov 17&rarr;51%; Gini 0.97&rarr;0.81; rel &minus;1.3pp","V3 G33 &mdash; offline sim"],
                  ["OPE of exploration policy","ESS 168&rarr;21; DR fails bandit","V3 G33 &mdash; methodology demo"]],
                 [1.7*inch, 2.35*inch, 3.25*inch]),
          Spacer(1,8),
          P("Model registry", H2),
          dtable([["Key","Lane","Headline","Comparability"],
                  ["d3aplus_als_f64","V1","R@20 0.0846","V1 protocol ONLY"],
                  ["sasrec_canonical_v3","V1","R@20 0.065 (plot_reconstructed)","V1; packaging-limited"],
                  ["c2_als_f64","V2 SERVED","R@20 ~0.0375","V2 protocol ONLY"],
                  ["minilm_content","V2","reach 54.6%","candidate-gen only"],
                  ["g28_lambdamart_fusion","V2","test 0.036 (81 pos)","offline test split only"],
                  ["g32_two_tower","V3","R@20 0.0642 (honest negative)","ALS-floor protocol"],
                  ["g33_thompson_bandit","V3","cov 51%; Gini 0.81 (offline)","exploration sim, proxy reward"]],
                 [1.8*inch, 0.95*inch, 2.4*inch, 2.15*inch]),
          PageBreak()]

# ============ 7. GATE WALKTHROUGH ============
story += [P("7 &nbsp; Gate-by-Gate Walkthrough (V2&ndash;V3)", H1), divider(),
          dtable([["Gate","What shipped","Result / verdict"],
                  ["G22","FAISS latency&ndash;quality frontier","FlatIP default lossless &minus;47% p95; HNSW overlap 0.992; IVF rejected"],
                  ["G23","Production-shaped FastAPI + load test","warm p95 ~3.3ms; 0% empty; failure modes graceful"],
                  ["G24","Cold-start / sparse-cohort fallback quality","coverage collapse ~59&times;; recall non-monotonic; 57% item-cold-start"],
                  ["G25","Catalog exposure governance","all Gini&gt;0.97; popularity degenerate; HNSW exposure-neutral"],
                  ["G26A","Decision-architecture spine + claim ladder","model/protocol registry; OPE logging schema designed"],
                  ["G26B","Heuristic exposure rerank, in-service","head-cap: coverage 6.9&rarr;9.8%, tail ~4.3&times;, no relevance loss; default OFF"],
                  ["G27","Semantic content retrieval (MiniLM+FAISS)","reach 54.6% of unseen golds; coverage 62.6% vs 8.1%; loses head"],
                  ["G28","Learned fusion tournament (LambdaMART)","beat ALS-only on test (0.036 vs 0.022) + recovered cold-start"],
                  ["G29","Off-policy evaluation executed","IPS/SNIPS/DR recover known value; DR lowest-bias, SNIPS stable"],
                  ["G30","Final RiskFrame audit + interview kit","9.1/10; offline gold-complete"],
                  ["G31","Search/IR front end (BM25+dense+RRF)","BM25 strongest single lane; NDCG/MRR; role coverage added"],
                  ["G32","Two-tower neural retrieval (ID+content, BPR)","R@20 0.064 vs ALS 0.085 &mdash; honest negative; content ablation +0.021"],
                  ["G33","Thompson Sampling exploration bandit","coverage 17&rarr;51%, Gini 0.97&rarr;0.81, &minus;1.3pp relevance; OPE overlap collapse"]],
                 [0.55*inch, 3.0*inch, 3.75*inch]),
          Spacer(1,6),
          P("The discipline an interviewer should notice: every gate has a pre-registered question, an artifact, and an honest verdict &mdash; "
            "including the gates that produced negatives (G20 SASRec, G22 IVF) and the gate that refused an expected story (G24).", SMALL),
          PageBreak()]

# ============ V3. NEURAL + EXPLORATION ============
story += [P("V3 &nbsp; Neural Two-Tower + Thompson Exploration Bandit", H1), divider(),
          P("V3 answers the two questions the target role weighs most heavily: can a neural two-tower with content "
            "beat ALS, and does a bandit exploration layer buy real catalog diversity? The answers are an honest "
            "<b>no</b> and a measured <b>yes-with-a-cost</b> &mdash; and the sharpest result is that off-policy "
            "evaluation breaks down for the exploration policy exactly where the theory says it should."),
          P("G32 &mdash; two-tower vs ALS (identical protocol)", H2),
          P("User tower: mean-pooled learned embeddings of the user's history (&le;50 items). Item tower: a learned "
            "item-ID embedding plus a projection of the 384-d MiniLM content embedding. BPR loss, 8 uniform negatives, "
            "23 epochs. Evaluated leave-last-out on the same 5,000-user eval_sample, top-200, history-masked, with "
            "per-user bootstrap 95% CIs &mdash; the exact ALS-floor protocol.", SMALL),
          dtable([["Model","Recall@20","95% CI","NDCG@20","Verdict"],
                  ["ALS (F=64)","0.0846","[0.0766, 0.0922]","0.0344","baseline"],
                  ["Two-tower (ID + content)","0.0642","[0.0574, 0.0712]","0.0262","loses to ALS"],
                  ["Two-tower (ID only, content off)","0.0432","[0.0376, 0.0490]","0.0162","ablation"]],
                 [2.35*inch, 1.15*inch, 1.55*inch, 1.05*inch, 1.2*inch]),
          Spacer(1,4),
          bullets([
           "<b>ALS wins, not noise:</b> its CI lower bound 0.0766 is above the two-tower's upper bound 0.0712 &mdash; non-overlapping.",
           "<b>Content is a real signal:</b> the content tower adds +0.021 R@20 (0.0432 &rarr; 0.0642), CIs non-overlapping &mdash; content is now first-class inside retrieval.",
           "<b>Second neural negative:</b> SASRec (0.065) and the two-tower (0.064) both lose to MF on dense warm book histories &mdash; depth is not the lever.",
          ]),
          P("G33 &mdash; Thompson Sampling exploration (offline)", H2),
          P("For each of the 5,000 warm users, take the two-tower top-200 pool and build a size-20 slate under greedy "
            "(rank by score) vs Thompson (per-item Beta posterior, updated with a held-out-positive proxy reward over "
            "6 passes). Offline simulation &mdash; no live serving.", SMALL),
          dtable([["Metric","Greedy","Thompson","Direction"],
                  ["Catalog coverage","16.6% (6,972)","50.5% (21,201)","~3&times; more catalog exposed"],
                  ["Exposure Gini","0.9736","0.8063","less concentrated"],
                  ["Intra-list diversity","0.492","0.5054","modestly higher"],
                  ["Slate hit-rate@20 (relevance)","0.0642","0.0510","&minus;1.3pp (the cost)"]],
                 [2.5*inch, 1.55*inch, 1.55*inch, 1.7*inch]),
          Spacer(1,4),
          P("Off-policy evaluation of the two policies (IPS/SNIPS/DR, single-action bandit)", H2),
          dtable([["Target","true","IPS","SNIPS","DR","ESS","matches"],
                  ["Greedy top-1","0.0062","0.0048","0.0047","0.0050","168","332"],
                  ["Thompson top-1","(0.018, biased)","0.0000","0.0000","&minus;0.0005","21","38"]],
                 [1.7*inch, 1.15*inch, 0.85*inch, 0.95*inch, 0.9*inch, 0.75*inch, 1.0*inch]),
          Spacer(1,6),
          callout("HONESTY BOUNDARY (V3)",
                  "Offline only. The two-tower LOST to ALS (reported as an honest negative). The bandit REDUCES top-slate "
                  "relevance &mdash; it buys exploration/diversity, not accuracy. OPE recovers the greedy policy but is "
                  "unreliable for the exploration policy (ESS 168&rarr;21), so its estimates are a methodology demo, not "
                  "online value. The Thompson &lsquo;true value&rsquo; is optimistically biased (posterior fit on the same "
                  "proxy reward) and is not counted as a relevance gain. No online lift, no A/B, no production claim.",
                  accent=AMBER, fill=colors.HexColor("#2c1f0b")),
          PageBreak()]

# ============ 8. OPE DEEP DIVE ============
story += [P("8 &nbsp; Off-Policy Evaluation &mdash; deep dive", H1), divider(),
          P("<b>Setup.</b> Single-action contextual-bandit OPE. Context x = user; action a = one item drawn from the candidate pool; the "
            "stochastic logging policy &pi;<sub>0</sub> = &epsilon;-uniform (&epsilon;=0.15) + softmax over an inverse-rank blend, guaranteeing "
            "overlap (&pi;<sub>0</sub>&gt;0). Reward r = 1 if the drawn item is in the user's held-out test interactions. Evaluated on the "
            "leakage-safe TEST users; 30 Monte-Carlo draws/user &rarr; 28,320 logged samples."),
          P("Estimators", H2),
          bullets([
           "<b>IPS</b> = (1/N) &Sigma; [1{a<sub>i</sub>=&pi;<sub>e</sub>(x<sub>i</sub>)} / &pi;<sub>0</sub>(a<sub>i</sub>|x<sub>i</sub>)] r<sub>i</sub> &mdash; unbiased, high variance.",
           "<b>SNIPS</b> = &Sigma; w<sub>i</sub>r<sub>i</sub> / &Sigma; w<sub>i</sub> &mdash; self-normalized; lower variance, slight bias.",
           "<b>DR</b> = mean(r&#770;(x,&pi;<sub>e</sub>)) + mean(w<sub>i</sub>(r<sub>i</sub> &minus; r&#770;(x,a<sub>i</sub>))) &mdash; lowest bias with a calibrated reward model r&#770;.",
          ]),
          P("Result (recover a known value)", H2),
          dtable([["Target policy","true (precision@1)","IPS","SNIPS","DR","ESS"],
                  ["ALS top-1","0.0106","0.0211","0.0129","0.0089","266"],
                  ["RRF-fusion top-1","0.0117","0.0218","0.0134","0.0110","220"]],
                 [1.9*inch, 1.4*inch, 1.0*inch, 1.0*inch, 1.0*inch, 1.0*inch]),
          Spacer(1,6),
          P("<b>Reading:</b> all three land near the directly-computed true value &mdash; the logged-propensity pipeline is correct. DR is "
            "closest (its reward-model baseline reduces bias); IPS over-estimates ~2&times; (the textbook high-variance failure at modest ESS); "
            "SNIPS sits between. The estimators correctly rank RRF-fusion above ALS &mdash; the ordering OPE would use to choose a policy."),
          callout("HONESTY BOUNDARY",
                  "Propensities are synthetic (a logging policy we control) and the reward is a held-out-gold proxy, not real engagement. This "
                  "VALIDATES the estimators offline; it does NOT estimate real online value. A real off-policy number needs logged "
                  "real-traffic propensities &mdash; the labelled next step.", accent=AMBER, fill=colors.HexColor("#2c1f0b")),
          PageBreak()]

# ============ 9. COLD-START + EXPOSURE ============
story += [P("9 &nbsp; Cold-Start &amp; Exposure &mdash; deep dive", H1), divider(),
          P("Cold-start taxonomy", H2),
          P("<b>User cold-start:</b> unknown (0 history) / sparse (1&ndash;2) / low (3&ndash;5) / warm (6+). <b>Item cold-start (offline-split "
            "sense):</b> 57.2% of held-out gold items are unseen in train &mdash; no collaborative signal, so ALS/popularity reach them with "
            "probability ~0 by construction."),
          P("The honest finding (not the expected one)", H2),
          P("Per-cohort Recall@20 is <b>non-monotonic</b> &mdash; cold/sparse cohorts score equal-or-higher than warm, because a low-history "
            "user's next item is more popularity-predictable and easier to retrieve. So <b>recall understates the cold-start cost.</b> The real, "
            "measured cost of falling back to popularity is a <b>collapse of personalization and catalog coverage</b>:"),
          dtable([["Metric (top-20)","ALS warm (served)","Popularity fallback"],
                  ["Unique items recommended","3,532","60"],
                  ["Catalog coverage","8.7%","0.15%"],
                  ["Slate personalization (distinct/users)","0.68","0.001"],
                  ["Long-tail exposure share","0.4%","0.0%"]],
                 [3.1*inch, 2.1*inch, 2.1*inch]),
          Spacer(1,6),
          P("Exposure governance", H2),
          P("Over the full catalog (zeros included): every policy is concentrated (Gini &gt; 0.97); popularity is near-degenerate (Gini 0.9995, "
            "99.9% zero-exposure, ~60 unique items). Head items get ~10&times; their catalog base rate; the long tail gets ~0.02&times; "
            "(suppressed ~50&times;). HNSW is exposure-neutral vs exact (Gini 0.986 &asymp; 0.986). This is catalog-exposure <b>concentration</b> "
            "governance &mdash; explicitly NOT protected-class fairness."),
          P("Mitigation (G26): a head-item exposure cap is Pareto-dominant on the warm cohort &mdash; it improves Recall@20 +40% AND long-tail "
            "exposure ~8&times; AND lowers Gini, because head inflation was crowding niche relevant items out of the top-20. MMR and "
            "genre-diversification were honest negatives. The win is protocol-specific (warm cohort, single-held-out-gold).", SMALL),
          PageBreak()]

# ============ 10. TRUTH BOUNDARY ============
story += [P("10 &nbsp; Truth Boundary &amp; Claim Ladder", H1), divider(),
          dtable([["Real (built &amp; measured)","Simulated / proxy","NOT claimed"],
                  ["2.56M Goodreads dataset, temporal split","OPE propensities synthetic","online lift / engagement / A/B impact"],
                  ["ALS/SASRec/content/semantic/BM25 retrieval","OPE reward = held-out-gold proxy","production quality (deployed &ne; effective)"],
                  ["FAISS serving + load test + Cloud Run deploy","81 test positives in fusion eval","solved cold-start / long-tail"],
                  ["cohort recall, exposure Gini, NDCG/MRR, IPS/SNIPS/DR","&mdash;","fairness certification"],
                  ["gate-by-gate audit history","&mdash;","LLM recommender / semantic taste"],
                  ["model/protocol registry","&mdash;","learned-fusion-beats-ALS beyond the test split"]],
                 [2.7*inch, 2.0*inch, 2.6*inch]),
          Spacer(1,8),
          P("Claim ladder", H2),
          bullets([
           "<b>Locked safe:</b> offline decision dossier &middot; ALS warm floor &middot; FAISS frontier &middot; fallback quality &middot; exposure governance &middot; deployed-to-Cloud-Run.",
           "<b>Safe with boundary:</b> SASRec converged-and-lost (plot_reconstructed) &middot; production-LIKE API &middot; cold-start handled at system level &middot; OPE methodology demonstrated.",
           "<b>Forbidden:</b> online lift &middot; real users &middot; real off-policy value &middot; cold-start solved &middot; fairness certified &middot; LLM recommender &middot; learned-beats-ALS in production.",
          ]),
          PageBreak()]

# ============ 11. PRODUCT / BUSINESS ============
story += [P("11 &nbsp; Product &amp; Business Reasoning", H1), divider(),
          dtable([["Decision","First-principles driver","Business consequence"],
                  ["ALS = warm floor","depth didn't beat MF at convergence","ship a simple, cheap, strong floor"],
                  ["coverage &gt; sophistication","recall ceiling is candidate-bound","invest in candidate generation, not depth"],
                  ["FAISS = latency evidence","ANN doesn't change relevance","serving cost, not a UX lift"],
                  ["fallback existence &ne; quality","always-nonempty &ne; good","cold UX is generic; measure it"],
                  ["exposure audit &ne; fairness","concentration &ne; protected-class","catalog-health signal, not certification"],
                  ["popularity = emergency only","Gini 0.9995, zero personalization","creator/tail starvation if it dominates"],
                  ["logs for real OPE","IPS needs &pi;<sub>log</sub> overlap","online decisioning unlocked once logged"]],
                 [1.7*inch, 2.9*inch, 2.7*inch]),
          Spacer(1,6),
          P("<b>Who uses this:</b> recommender DS (protocol/metric validity), ranking ML engineer (retrieval/serving tradeoffs), search/discovery "
            "&amp; marketplace PM (catalog health, cold-start UX), platform engineer (latency, fallback, monitoring). <b>Why wrongness is "
            "expensive:</b> a recall-looking win that worsens catalog health, an ANN setting that shreds the candidate pool, or a popularity "
            "fallback that quietly becomes dominant all degrade discovery and creator opportunity at scale before any dashboard notices."),
          PageBreak()]

# ============ 12. HARD Q&A ============
story += [P("12 &nbsp; Hard Q&amp;A", H1), divider()]
qa = [
 ("Why did SASRec lose to ALS?","Book next-item is weakly sequence-predictable; the dominant signal is collaborative co-read, which MF captures directly. I trained a canonical full-softmax SASRec to convergence and it still scored below ALS &mdash; depth was not the lever on sparse histories."),
 ("Was the SASRec undertrained?","No &mdash; I monitored convergence on eval-R@20 after catching a NaN val-loss false-stop, and showed the full-softmax-vs-sampled lever (~6.7&times;) was the real driver; even fixed, it didn't beat ALS."),
 ("What's the OPE variance problem and how does SNIPS fix it?","Raw IPS weights &pi;<sub>e</sub>/&pi;<sub>0</sub> blow up variance at modest ESS (my IPS over-estimated ~2&times;). SNIPS self-normalizes by the sum of weights, trading a little bias for much lower variance &mdash; my recommended estimator when no reward model exists; DR is lowest-bias when a calibrated one does."),
 ("Why did DR go negative once?","A class-balanced reward model calibrated toward ~0.5 vs a true ~1% base rate, so it over-predicted and broke the DR baseline. I caught it from the impossible negative value and switched to a base-rate-calibrated model; DR then recovered the known value best of the three."),
 ("What's the real cost of cold-start here?","Not a recall drop &mdash; recall is non-monotonic because cold users' next item is more popularity-predictable. The real cost is coverage collapse ~59&times; and personalization 0.68&rarr;0.001. I reframed the metric around what actually degrades."),
 ("Why reject IVF if it was faster?","Low-nprobe IVF kept gold Recall@20 flat while candidate-overlap vs exact collapsed to 0.28&ndash;0.78 &mdash; a broken pool a single-gold metric can't see. Speed that destroys candidate fidelity is a recall bug."),
 ("Why did BM25 beat dense on short text?","On title+genre text, exact lexical overlap (shared series/genre words) carries more signal than a 384-d embedding that blurs the rare disambiguating tokens. Embeddings win on long paraphrase-heavy text; lexical wins on short keyword-dense text."),
 ("So is your hybrid retrieval better?","On recall, no &mdash; RRF hybrid sits below BM25 alone on this seed-item task. Its value is robustness: it never collapses to either lane's weakness and it reaches cold-start. I don't claim hybrid beats both."),
 ("Did the learned ranker actually beat ALS?","On the held-out test split, yes (0.036 vs 0.022) and it recovered cold-start (0.032 vs 0), consistent across four model families &mdash; but on 81 positives, single-gold, offline. Directional, not a production claim."),
 ("Why does fusion help at all?","ALS and semantic candidate sets are almost disjoint (overlap 6,749 of ~880k). A learned ranker picks the best across both complementary sources &mdash; that's what lifted test recall and cold-start."),
 ("What does the live Cloud Run endpoint serve?","The served-c2 ALS model with exact FlatIP retrieval: /recommend returns top-K with source tags and a graceful fallback (unknown user &rarr; popularity, never empty). /health reports readiness. \"Deployed and serving HTTPS\" is true; \"improved engagement\" is not."),
 ("Is any of this production / online?","No &mdash; offline by scope. The API is deployed and callable, but there are no real users, no logged real-traffic propensities, and no A/B. That boundary is documented as the next step."),
 ("Why is your served recall ~0.0375 when V1 was 0.0846?","Different builds and protocols &mdash; d3aplus tuning vs served-c2 full-catalog single-held-out-gold. I keep them in a registry so they're never conflated. The relative cross-policy structure is the V2 signal, not the absolute."),
 ("Is the semantic lane an LLM recommender?","No &mdash; embedding-based candidate generation (MiniLM over content text). No generation, no chat, no \"semantic taste.\" It reaches items collaborative filtering can't."),
 ("Did the V3 two-tower beat ALS?","No &mdash; and I report it as an honest negative. A proper two-tower (mean-pooled history user tower + learned-ID&oplus;MiniLM-content item tower, BPR) reached R@20 0.0642 [0.057&ndash;0.071] vs ALS 0.0846 on the identical protocol; the CIs don't overlap. The useful result is the content ablation: MiniLM content adds +0.021 R@20, so content earns its place inside retrieval even though the model still loses to MF on dense warm histories. Two neural models (SASRec, two-tower) now lose to ALS &mdash; consistent, not a fluke."),
 ("What did the Thompson Sampling bandit actually buy?","Exploration, at an explicit relevance cost &mdash; not a relevance win. Over two-tower candidate pools, Thompson vs greedy raised catalog coverage 16.6%&rarr;50.5% and cut exposure Gini 0.97&rarr;0.81, for a ~1.3pp drop in top-slate hit-rate (0.064&rarr;0.051). That's the explore/exploit tradeoff, stated as a cost. Offline simulation with a proxy reward &mdash; no online lift."),
 ("Why did OPE break for the bandit but not for greedy?","Overlap. The synthetic logging policy covers the greedy top-1 well (332 matched samples, ESS 168) so IPS/SNIPS/DR recover it (DR 0.005 vs true 0.006). The exploration policy selects items the logging policy rarely logs, so overlap collapses (38 matches, ESS 21) and the estimators become unreliable &mdash; DR even goes negative. You can't off-policy-evaluate an exploration policy a logging policy doesn't cover; it's the same lesson as the G29 DR-negative failure."),
 ("Is your exposure work a fairness claim?","No &mdash; catalog exposure concentration (Gini/coverage/tier-lift), explicitly not protected-class fairness. I don't certify fairness; I measure concentration as a ship-gate metric."),
 ("How did you prevent leakage in the fusion eval?","Split by user-hash (a user is wholly in one split); test labels never used for training or model selection; served-c2 protocol only. The fix for the v2 SASRec leakage (stopping on the eval set) was a separate validation-monitor set."),
 ("Why is the cover claim 'gold_candidate' not 'gold_complete'?","One packaging gap: the G20 SASRec headline is plot_reconstructed (the direct artifact was lost). The decision is sound; the evidence packaging is one artifact short, so I hold the status honestly at candidate."),
 ("What would a head-cap rerank cost in production?","Sub-millisecond, and it improved recall AND exposure on the warm cohort here &mdash; but the win is protocol-specific (single-held-out-gold). I ship it config-flagged, default OFF, and would tune the cap on a live relevance target."),
 ("What's your single biggest risk if I pushed you?","Small-sample fragility &mdash; the fusion win rests on 81 test positives. I state it, it's directionally consistent across four model families, and the honest resolution is propensity logging + an A/B, which I've designed but not run."),
 ("Could you have just shipped the best offline number?","That's exactly the failure mode the project targets. The IVF and head-cap stories show a better-looking number that was either broken (IVF) or protocol-specific (head-cap). Shipping a metric without its mechanism is how recsys quietly breaks."),
 ("What would falsify your headline claims?","ALS-floor: a properly-tuned sequence/graph model beating it on this exact protocol. Fusion-win: not holding on a larger positive set. Semantic value: cold-start reach not translating to online engagement. OPE: ESS too low / no overlap. I state these kill conditions, not hide them."),
]
for q, a in qa:
    story += [P("Q. " + q, LEADQ), P("A. " + a, ANS)]
story.append(PageBreak())

# ============ 13. NEXT ============
story += [P("13 &nbsp; What I'd Build Next", H1), divider(),
          P("With production access, in priority order:"),
          bullets([
           "<b>Real-traffic propensity logging</b> &mdash; populate the designed OPE schema (request/impression/position/&pi;<sub>log</sub>/reward) to turn IPS/SNIPS/DR into real off-policy value estimates.",
           "<b>Canary + live A/B</b> of the fusion policy and the head-cap exposure rerank, with guardrail metrics (relevance, coverage, Gini, fallback rate, p95) wired to alerts &mdash; the only way to claim online lift.",
           "<b>In-ranker position-bias correction (PAL / IPS-weighted LTR)</b> &mdash; upgrade position bias from <i>evaluated</i> to <i>corrected</i>.",
           "<b>Productionise the lexical lane</b> (Lucene / Elasticsearch / bm25s) for sub-ms BM25, and persist the FAISS index to cut cold-start build time.",
           "<b>Two-tower &amp; exploration are now built (V3)</b> &mdash; the two-tower is an honest negative with a real content lift, and the Thompson bandit needs real logged propensities to make its OPE trustworthy; graded-relevance labels and GNN retrieval remain the next scale-up.",
          ]),
          PageBreak()]

# ============ 14. EVIDENCE INDEX ============
story += [P("14 &nbsp; Evidence Artifact Index", H1), divider(),
          P("Every claim maps to a JSON in <font face='Courier'>outputs/evidence/</font>. Selected artifacts:", SMALL),
          dtable([["Artifact","Proves"],
                  ["g22_faiss_latency_quality_report.json","FlatIP lossless &minus;47% p95; HNSW near-lossless; IVF overlap collapse"],
                  ["g23_serving_api_report.json","production-like API, 0% empty, failure-mode coverage"],
                  ["g24_cold_start_sparse_cohort_report.json","fallback coverage collapse ~59&times;, personalization 0.68&rarr;0.001"],
                  ["g25_catalog_exposure_governance_report.json","exposure Gini/coverage; HNSW exposure-neutral"],
                  ["g26b_rerank_integration_report.json","head-cap rerank: +recall, +coverage, no relevance loss"],
                  ["g27_semantic_retrieval_report.json","semantic reaches 54.6% of item-cold-start golds"],
                  ["g28_final_ranker_fusion_decision.json","learned fusion beats ALS-only on held-out test"],
                  ["g29_ope_execution_report.json","IPS/SNIPS/DR recover a known value offline"],
                  ["g31_search_ir_report.json","BM25 + dense + RRF with NDCG/MRR"],
                  ["g30_final_gold_audit.json","RiskFrame 9.1, offline gold-complete"],
                  ["g_serving_stress_test.json","serving edge-case stress test + two load-path fixes"],
                  ["G32_two_tower_eval.json","two-tower R@20 0.064 vs ALS 0.085 + content ablation (V3)"],
                  ["G33_bandit_eval.json","Thompson coverage/Gini/diversity + OPE overlap collapse (V3)"]],
                 [3.4*inch, 3.9*inch]),
          Spacer(1,10), divider(),
          P("Companion docs in the repo: PULSEDISCOVERY_INTERVIEW_KIT.md (pitch &amp; claim ladder) &middot; "
            "PULSEDISCOVERY_UNIFIED_DEFENSE_KERNEL.md (method-by-method defense) &middot; PULSEDISCOVERY_FAILURES_AND_HARDENING.md "
            "(failures + serving stress test) &middot; G26A_MODEL_PROTOCOL_REGISTRY.md (V1/V2 registry).", SMALL),
          P("If a result isn't backed by an artifact in outputs/evidence/, it isn't claimed.", st("end", fontSize=9, textColor=GREEN, alignment=TA_CENTER, spaceBefore=8))]

def page_bg(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(BG); canvas.rect(0,0,letter[0],letter[1],fill=1,stroke=0)
    canvas.setFont("DJ", 7.5); canvas.setFillColor(MUTE)
    canvas.drawString(0.6*inch, 0.45*inch, "PulseDiscover — Interview Defense Dossier")
    canvas.drawRightString(letter[0]-0.6*inch, 0.45*inch, "Offline by scope · every claim artifact-backed · p%d" % doc.page)
    canvas.restoreState()

doc = SimpleDocTemplate("defense/PulseDiscover_Interview_Defense.pdf", pagesize=letter,
                        leftMargin=0.6*inch, rightMargin=0.6*inch, topMargin=0.6*inch, bottomMargin=0.6*inch,
                        title="PulseDiscover Interview Defense Dossier", author="Sidharth Kriplani")
doc.build(story, onFirstPage=page_bg, onLaterPages=page_bg)
print("built defense/PulseDiscover_Interview_Defense.pdf")
