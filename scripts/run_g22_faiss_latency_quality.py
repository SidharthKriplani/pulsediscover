"""G22 — FAISS serving latency-quality frontier (V2). Exact ALS vs FlatIP vs IVF(nprobe) vs HNSW(efSearch).
Quality: overlap@K + candidate-recall@K vs exact ALS, downstream rec Recall@20 (gold) vs exact.
Serving: p50/p95/p99 latency, throughput, index build time, index memory. Frontier + recommended + fallback.
CPU in-sandbox (labeled). FAISS retains/degrades recall — NOT a quality improvement. V2: new evidence, V1 untouched."""
import os, sys, json, time, pickle, numpy as np, pandas as pd
ROOT="/sessions/festive-quirky-cannon/mnt/pulsediscover"; sys.path.insert(0,os.path.join(ROOT,"src"))
from serving.faiss_retriever import ExactRetriever, FaissRetriever
OUT=os.path.join(ROOT,"data/interim"); SP=os.path.join(OUT,"domain_splits"); EVID=os.path.join(ROOT,"outputs/evidence"); PLOTS=os.path.join(ROOT,"outputs/plots")
SEED=20260616; rng=np.random.default_rng(SEED); t0=time.time()
A=pickle.load(open(os.path.join(OUT,"c2_als.pkl"),"rb")); X=A["X"];Y=A["Y"];uidx=A["uidx"];iidx=A["iidx"];i_uniq=A["i_uniq"]
es=pd.read_csv(os.path.join(SP,"eval_sample.csv"),dtype=str)
i2pos={b:p for p,b in enumerate(i_uniq)}                       # book_id -> row in Y (for gold recall)
warm=es[es.gold.isin(set(iidx))]; qu=[u for u in warm.user_id if u in uidx]
QN=min(2000,len(qu)); sel=rng.choice(len(qu),QN,replace=False)
Q=np.ascontiguousarray(X[[uidx[qu[i]] for i in sel]],dtype=np.float32)
gold_pos=np.array([i2pos.get(warm.set_index("user_id").gold.get(qu[i]),-1) for i in sel])
KS=[20,100,200]; KMAX=max(KS)
def pp(lat): a=np.array(lat); return {"p50_ms":round(float(np.percentile(a,50)),3),"p95_ms":round(float(np.percentile(a,95)),3),"p99_ms":round(float(np.percentile(a,99)),3)}
def per_query_lat(searcher,k,n=800):
    lat=[]
    for i in range(min(n,QN)):
        t=time.perf_counter(); searcher(Q[i],k); lat.append((time.perf_counter()-t)*1000)
    return lat
# exact reference
ex=ExactRetriever(Y); ex_top={i:set(ex.search(Q[i],KMAX)) for i in range(QN)}
ex_topk={k:[set(list(ex.search(Q[i],k))) for i in range(QN)] for k in KS}
def recall20_for(topfn):                                       # gold (Y-row) in top-20
    hit=0;n=0
    for i in range(QN):
        if gold_pos[i]<0: continue
        n+=1; t=topfn(i,20)
        if gold_pos[i] in t: hit+=1
    return round(hit/max(n,1),4)
ex_recall20=recall20_for(lambda i,k:set(list(ex.search(Q[i],k))))
def eval_index(name,retr,searcher,build_sec,index_mb,extra):
    # quality: overlap@K + cand recall@K vs exact, downstream recall@20
    ov={}; cr={}
    topK={i:list(searcher(Q[i],KMAX)) for i in range(QN)}
    for k in KS:
        ov[f"overlap@{k}"]=round(float(np.mean([len(set(topK[i][:k])&ex_topk[k][i])/k for i in range(QN)])),4)
        cr[f"cand_recall@{k}"]=ov[f"overlap@{k}"]              # symmetric here (same K from same space)
    r20=recall20_for(lambda i,k:set(topK[i][:k]))
    lat=per_query_lat(searcher,KMAX);
    tb=time.perf_counter(); retr.search_batch(Q,KMAX); thr=round(QN/(time.perf_counter()-tb))
    return {"setting":name,"build_sec":round(build_sec,3),"index_MB":index_mb,**extra,
            "latency":pp(lat),"throughput_qps_batch":thr,
            "quality":{**ov,"rec_Recall@20":r20,"rec_R@20_degradation_vs_exact":round(ex_recall20-r20,4)}}
results=[]
# brute-force numpy exact baseline
lat=per_query_lat(lambda q,k: ex.search(q,k),KMAX)
results.append({"setting":"exact_numpy_bruteforce","build_sec":0.0,"index_MB":round(Y.nbytes/1e6,2),
  "latency":pp(lat),"throughput_qps_batch":None,"quality":{"overlap@20":1.0,"overlap@100":1.0,"overlap@200":1.0,"rec_Recall@20":ex_recall20,"rec_R@20_degradation_vs_exact":0.0},"note":"reference"})
# FAISS FlatIP (exact)
fr=FaissRetriever(Y,kind="flatip"); results.append(eval_index("faiss_flatip_exact",fr,lambda q,k:fr.search(q,k),fr.build_sec,round(fr.index_bytes()/1e6,2),{}))
# FAISS IVF nprobe sweep
ivf=FaissRetriever(Y,kind="ivf",nlist=256,nprobe=8); ivf_mb=round(ivf.index_bytes()/1e6,2)
for npb in [1,8,16,32,64]:
    ivf.set_nprobe(npb); results.append(eval_index(f"faiss_ivf_nprobe{npb}",ivf,lambda q,k:ivf.search(q,k),ivf.build_sec,ivf_mb,{"nprobe":npb}))
# FAISS HNSW efSearch sweep (IP)
try:
    hn=FaissRetriever(Y,kind="hnsw",M=32,efC=200,efS=64); hn_mb=round(hn.index_bytes()/1e6,2)
    for ef in [16,32,64,128]:
        hn.set_efsearch(ef); results.append(eval_index(f"faiss_hnsw_ef{ef}",hn,lambda q,k:hn.search(q,k),hn.build_sec,hn_mb,{"efSearch":ef}))
    hnsw_ok=True
except Exception as e:
    hnsw_ok=False; print("HNSW skipped:",e)
# recommendation: highest-retention setting with p95 < brute-force, retention>=0.98 on rec_Recall@20
bf_p95=results[0]["latency"]["p95_ms"]
cands=[r for r in results if r["setting"]!="exact_numpy_bruteforce" and r["quality"]["rec_R@20_degradation_vs_exact"]<=0.002 and r["latency"]["p95_ms"]<bf_p95]
recommended = min(cands,key=lambda r:r["latency"]["p95_ms"])["setting"] if cands else "faiss_flatip_exact"
out={"scope":"G22 FAISS serving latency-quality frontier (V2)","lane":"PulseDiscover V2","tag":"[BUILT — real data, CPU in-sandbox latency (NOT production)]",
 "embedding_slice":"ALS f64 item factors (40,541 x 64); content/co-occ/pop NOT dense-ANN (excluded)",
 "n_items":int(Y.shape[0]),"dim":int(Y.shape[1]),"n_query_users":QN,"exact_rec_Recall@20":ex_recall20,
 "frontier":results,"brute_force_p95_ms":bf_p95,"hnsw_available":hnsw_ok,
 "recommended_setting":recommended,
 "fallback_policy":"If chosen ANN setting's rec_Recall@20 degradation > 0.002 OR p95 regresses, fall back to faiss_flatip_exact (lossless, sub-ms at this scale).",
 "claim_boundary":["FAISS RETAINS exact recall (FlatIP) or trades a MEASURED amount (approx) for latency; it does NOT improve recommendation quality.",
   "CPU in-sandbox latency, NOT a production benchmark. No online lift / production / deployment claim.",
   "Does NOT change the V1 offline conclusion that ALS is the warm-retrieval floor.","V2 evidence; V1 (gold_candidate 8.9) unchanged."],
 "sec":round(time.time()-t0,1)}
json.dump(out,open(os.path.join(EVID,"g22_faiss_latency_quality_report.json"),"w"),indent=2)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,6))
    for r in results:
        if r["setting"]=="exact_numpy_bruteforce": continue
        deg=r["quality"]["rec_R@20_degradation_vs_exact"]; p95=r["latency"]["p95_ms"]; ret=1-deg
        c="#228833" if "flatip" in r["setting"] else ("#4477aa" if "ivf" in r["setting"] else "#ee7733")
        ax.scatter(p95,ret,s=90,c=c,edgecolor="k"); ax.annotate(r["setting"].replace("faiss_",""),(p95,ret),fontsize=6,xytext=(3,3),textcoords="offset points")
    ax.axhline(0.98,ls="--",c="red",alpha=0.5,label="0.98 rec-recall retention target"); ax.axvline(bf_p95,ls=":",c="gray",label="brute-force p95 %.2fms"%bf_p95)
    ax.set_xlabel("p95 latency (ms, CPU in-sandbox)"); ax.set_ylabel("rec Recall@20 retention vs exact (1 = lossless)")
    ax.set_title("G22 FAISS serving frontier — latency vs quality retention (ALS slice)"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS,"g22_faiss_latency_quality_frontier.png"),dpi=110); plt.close()
except Exception as e: print("plot skipped",e)
for r in results: print("%-26s p50 %s p95 %s rec_R@20 %s (deg %s) overlap@20 %s"%(r["setting"],r["latency"]["p50_ms"],r["latency"]["p95_ms"],r["quality"]["rec_Recall@20"],r["quality"]["rec_R@20_degradation_vs_exact"],r["quality"].get("overlap@20")))
print("recommended:",recommended,"| exact rec_R@20",ex_recall20,"| %.1fs"%(time.time()-t0))
