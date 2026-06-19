"""G16 — FAISS / ANN serving-latency demo on the ALS f64 embedding-retrieval slice.
Systems gate ONLY: brute-force exact vs FAISS IndexFlatIP (exact) vs FAISS IVF (approx, nprobe sweep).
Measures recall RETENTION vs brute-force + latency/build/size. FAISS cannot improve ranking quality
— it can only retain brute-force recall at lower latency. CPU in-sandbox numbers (not production)."""
import os, json, time, pickle, numpy as np, pandas as pd, faiss
OUT="/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim"; SP=os.path.join(OUT,"domain_splits"); EVID="/sessions/festive-quirky-cannon/mnt/pulsediscover/outputs/evidence"; PLOTS="/sessions/festive-quirky-cannon/mnt/pulsediscover/outputs/plots"
SEED=20260616; np.random.seed(SEED); t0=time.time()
A=pickle.load(open(os.path.join(OUT,"c2_als.pkl"),"rb")); X=A["X"];Y=A["Y"];uidx=A["uidx"]
Y=np.ascontiguousarray(Y,dtype=np.float32); d=Y.shape[1]; nitems=Y.shape[0]
es=pd.read_csv(os.path.join(SP,"eval_sample.csv"),dtype=str)
qu=[u for u in es.user_id if u in uidx]; no_factor=int((~es.user_id.isin(set(uidx))).sum())
QN=min(2000,len(qu)); rng=np.random.default_rng(SEED); qsel=rng.choice(len(qu),QN,replace=False)
Q=np.ascontiguousarray(X[[uidx[qu[i]] for i in qsel]],dtype=np.float32)
KS=[100,200,400]; KMAX=max(KS)
# brute-force exact top-KMAX (reference) + per-query latency
def bf_topk(q,k):
    s=Y@q; idx=np.argpartition(-s,k)[:k]; return idx[np.argsort(-s[idx])]
bf_top={}; lat_bf=[]
for i in range(QN):
    t=time.perf_counter(); o=bf_topk(Q[i],KMAX); lat_bf.append((time.perf_counter()-t)*1000); bf_top[i]=set(o[:KMAX])
bf_topk_byK={k:[set(list(bf_topk(Q[i],k))) for i in range(QN)] for k in KS}
# FAISS exact IndexFlatIP
tb=time.perf_counter(); flat=faiss.IndexFlatIP(d); flat.add(Y); build_flat=time.perf_counter()-tb
lat_flat=[]; flat_top={}
for i in range(QN):
    t=time.perf_counter(); _,I=flat.search(Q[i:i+1],KMAX); lat_flat.append((time.perf_counter()-t)*1000); flat_top[i]=I[0]
# FAISS IVF approx (IP), nprobe sweep
nlist=256; tb=time.perf_counter(); quant=faiss.IndexFlatIP(d); ivf=faiss.IndexIVFFlat(quant,d,nlist,faiss.METRIC_INNER_PRODUCT); ivf.train(Y); ivf.add(Y); build_ivf=time.perf_counter()-tb
def retention_and_lat(searcher):
    lat=[]; topI={}
    for i in range(QN):
        t=time.perf_counter(); _,I=searcher(Q[i:i+1],KMAX); lat.append((time.perf_counter()-t)*1000); topI[i]=I[0]
    ret={}
    for k in KS:
        r=[len(set(topI[i][:k])&bf_topk_byK[k][i])/k for i in range(QN)]; ret[f"recall_retention@{k}"]=round(float(np.mean(r)),4)
    return ret,lat
def pp(lat): a=np.array(lat); return {"p50_ms":round(float(np.percentile(a,50)),3),"p95_ms":round(float(np.percentile(a,95)),3),"mean_ms":round(float(a.mean()),3)}
# flat exact retention (should be ~1.0)
flat_ret={}
for k in KS:
    r=[len(set(flat_top[i][:k])&bf_topk_byK[k][i])/k for i in range(QN)]; flat_ret[f"recall_retention@{k}"]=round(float(np.mean(r)),4)
ivf_sweep={}
for nprobe in [1,4,8,16,32,64]:
    ivf.nprobe=nprobe; ret,lat=retention_and_lat(ivf.search); ivf_sweep[nprobe]={**ret,**pp(lat)}
# index sizes
faiss.write_index(flat,"/tmp/flat.idx"); faiss.write_index(ivf,"/tmp/ivf.idx")
sz_flat=round(os.path.getsize("/tmp/flat.idx")/1e6,2); sz_ivf=round(os.path.getsize("/tmp/ivf.idx")/1e6,2)
# throughput (batch)
tb=time.perf_counter(); flat.search(Q,KMAX); thr_flat=round(QN/(time.perf_counter()-tb))
ivf.nprobe=8; tb=time.perf_counter(); ivf.search(Q,KMAX); thr_ivf=round(QN/(time.perf_counter()-tb))
out={"scope":"G16 FAISS ANN serving-latency demo (ALS f64 embedding slice)","tag":"[BUILT — real data, CPU in-sandbox latency (NOT production)]",
 "embedding_slice":"ALS f64 item factors only; content/co-occ/popularity are non-dense lanes, excluded from ANN (served separately)",
 "n_items_indexed":int(nitems),"dim":int(d),"n_query_users":QN,"K_values":KS,
 "brute_force":{"latency":pp(lat_bf),"throughput_qps_batch":None,"note":"exact reference"},
 "faiss_flat_ip_exact":{"latency":pp(lat_flat),"build_sec":round(build_flat,3),"index_MB":sz_flat,"recall_retention":flat_ret,"throughput_qps_batch":thr_flat},
 "faiss_ivf_approx":{"nlist":nlist,"build_sec":round(build_ivf,3),"index_MB":sz_ivf,"nprobe_sweep":ivf_sweep,"throughput_qps_batch_nprobe8":thr_ivf},
 "failure_cases":{"cold_items_no_embedding":"cold/new items have no ALS factor -> NOT ANN-retrievable; served by content lane (by design)",
   "users_without_ALS_factor":no_factor,"small_catalog_note":"only %d items: brute-force p50 %.3f ms already fast -> FAISS is a scale-demonstration, not a current-size win"%(nitems,pp(lat_bf)["p50_ms"]),
   "aggressive_approx":"low nprobe (1) loses recall retention -> see sweep"},
 "claim_boundaries":["FAISS RETAINS brute-force recall at best; cannot improve ranking quality.","CPU in-sandbox latency, not a production benchmark.","No online lift / production / RiskFrame-gold claim.","Candidate strategy / ranker / slate / G15 interpretation unchanged."],
 "sec":round(time.time()-t0,1)}
json.dump(out,open(os.path.join(EVID,"g16_faiss_serving_report.json"),"w"),indent=2)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    nps=sorted(ivf_sweep); ret200=[ivf_sweep[n]["recall_retention@200"] for n in nps]; p95=[ivf_sweep[n]["p95_ms"] for n in nps]
    fig,ax=plt.subplots(figsize=(9,5.5)); ax2=ax.twinx()
    ax.plot(nps,ret200,"o-",color="#4477aa",label="recall retention@200 vs brute-force")
    ax2.plot(nps,p95,"s-",color="#ee7733",label="IVF p95 latency (ms)")
    ax.axhline(0.95,ls="--",c="green",alpha=0.6,label="0.95 retention target")
    ax.axhline(flat_ret["recall_retention@200"],ls=":",c="gray");
    bf=pp(lat_bf)["p95_ms"]; ax2.axhline(bf,ls=":",c="red",alpha=0.6,label="brute-force p95 %.2fms"%bf)
    ax.set_xlabel("IVF nprobe (higher = more exact, slower)"); ax.set_ylabel("recall retention@200",color="#4477aa"); ax2.set_ylabel("p95 latency ms",color="#ee7733")
    ax.set_title("G16 FAISS IVF: recall retention vs latency (ALS slice, %d items, CPU)"%nitems)
    l1,la1=ax.get_legend_handles_labels(); l2,la2=ax2.get_legend_handles_labels(); ax.legend(l1+l2,la1+la2,fontsize=7,loc="center right")
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS,"g16_latency_recall_tradeoff.png"),dpi=110); plt.close()
except Exception as e: print("plot skipped",e)
print("brute p50/p95 ms:",pp(lat_bf)); print("flat exact:",pp(lat_flat),"build %.3fs size %sMB ret"%(build_flat,sz_flat),flat_ret)
print("IVF build %.3fs size %sMB"%(build_ivf,sz_ivf))
for n in sorted(ivf_sweep): print("  nprobe %2d: ret@200 %.3f p50 %.3f p95 %.3f"%(n,ivf_sweep[n]["recall_retention@200"],ivf_sweep[n]["p50_ms"],ivf_sweep[n]["p95_ms"]))
print("throughput qps: flat %d, ivf(nprobe8) %d | users_no_ALS %d | sec %.1f"%(thr_flat,thr_ivf,no_factor,time.time()-t0))
