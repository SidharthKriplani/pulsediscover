"""G23 (V2) — load test + failure-mode test for the production-shaped serving API.
Exercises the real FastAPI app in-process (TestClient: routing + serialization + service). Measures
p50/p95/p99 latency, throughput, error/fallback rate, warm vs cold user behavior, concurrency, and
explicit failure-mode handling. Writes g23_serving_api_report.json. NOT a production benchmark."""
import os, sys, json, time, threading, numpy as np, pandas as pd
ROOT="/sessions/festive-quirky-cannon/mnt/pulsediscover"; sys.path.insert(0, os.path.join(ROOT,"src"))
from fastapi.testclient import TestClient
from serving.api import app, svc
EVID=os.path.join(ROOT,"outputs/evidence"); SP=os.path.join(ROOT,"data/interim/domain_splits")
client=TestClient(app); t0=time.time()
# warm users (have ALS factor) + cold (synthetic unknown ids)
es=pd.read_csv(os.path.join(SP,"eval_sample.csv"),dtype=str)
warm=[u for u in es.user_id if u in svc.uidx][:1500]
cold=[f"unknown_user_{i}" for i in range(500)]
def call(uid,k=20,mode=None):
    p={"user_id":uid,"k":k};
    if mode: p["mode"]=mode
    t=time.perf_counter(); r=client.get("/recommend",params=p); return (time.perf_counter()-t)*1000, r.status_code, r.json()
def seg(uids,mode=None):
    lat=[]; fb=0; err=0; empty=0
    for u in uids:
        ms,code,body=call(u,20,mode); lat.append(ms)
        if body.get("fallback_used"): fb+=1
        if body.get("error_code"): err+=1
        if body.get("response_count",0)==0: empty+=1
    a=np.array(lat); n=len(uids)
    return {"n":n,"p50_ms":round(float(np.percentile(a,50)),3),"p95_ms":round(float(np.percentile(a,95)),3),
            "p99_ms":round(float(np.percentile(a,99)),3),"fallback_rate":round(fb/n,4),"error_rate":round(err/n,4),"empty_rate":round(empty/n,4)}
warm_seg=seg(warm); cold_seg=seg(cold); scale_seg=seg(warm[:500],mode="scale")
# throughput (sequential)
tt=time.perf_counter(); [call(u) for u in warm[:500]]; qps=round(500/(time.perf_counter()-tt))
# concurrency (threads)
def worker(uids,out,idx):
    l=[];
    for u in uids: ms,_,_=call(u); l.append(ms)
    out[idx]=l
chunks=np.array_split(warm[:800],8); outs=[None]*8; tcon=time.perf_counter()
th=[threading.Thread(target=worker,args=(list(c),outs,i)) for i,c in enumerate(chunks)]; [t.start() for t in th]; [t.join() for t in th]
con_lat=np.concatenate([np.array(o) for o in outs]); con_qps=round(800/(time.perf_counter()-tcon))
# failure-mode tests (explicit)
fm={}
fm["unknown_user"]=client.get("/recommend",params={"user_id":"nope_xyz","k":20}).json()
fm["invalid_k_high"]=client.get("/recommend",params={"user_id":warm[0],"k":9999}).status_code   # pydantic 422
fm["invalid_k_zero"]=client.get("/recommend",params={"user_id":warm[0],"k":0}).status_code
fm["health"]=client.get("/health").json()
fm["metadata"]=client.get("/metadata").json()
fm["metrics_json_keys"]=sorted(list(client.get("/metrics.json").json().keys()))
fm["debug_retrieval_ok"]=client.get("/debug/retrieval",params={"user_id":warm[0],"k":5}).status_code==200
fm["prometheus_sample"]=client.get("/metrics").text.splitlines()[:3]
out={"scope":"G23 production-shaped serving API + load test (V2)","lane":"PulseDiscover V2","tag":"[BUILT — real data, CPU in-sandbox, NOT production]",
 "api_status":"ok" if svc.ready else "degraded","metadata":svc.metadata(),
 "endpoints":["GET /recommend","/health","/metadata","/metrics","/metrics.json","/debug/retrieval"],
 "load_test":{"warm_users":warm_seg,"cold_users":cold_seg,"scale_mode_hnsw":scale_seg,
   "throughput_qps_sequential":qps,"concurrency_threads":8,"throughput_qps_concurrent":con_qps,
   "concurrent_p95_ms":round(float(np.percentile(con_lat,95)),3)},
 "monitoring_snapshot":svc.metrics(),
 "failure_mode_tests":fm,
 "fallback_decision_tree":["unknown/sparse/missing-factor user -> popularity","invalid k -> 422 (pydantic) or popularity","empty candidates -> popularity","FAISS error -> popularity","index load failure -> degraded health + popularity","missing item metadata -> creator_id null (no crash)"],
 "production_claim_allowed":False,"production_like_claim_allowed":True,
 "claim_boundary":["Production-SHAPED local serving (FastAPI in-process TestClient measurement), NOT a deployed production system.",
   "No real users, no online lift. CPU in-sandbox latency (excludes real network/infra).",
   "FlatIP exact is the DEFAULT; HNSW ef64 is an explicit scale mode, not auto-default.",
   "Does not change V1 conclusions; V1 (gold_candidate 8.9) untouched."],
 "sec":round(time.time()-t0,1)}
json.dump(out,open(os.path.join(EVID,"g23_serving_api_report.json"),"w"),indent=2)
print("warm",warm_seg); print("cold",cold_seg); print("scale",scale_seg)
print("qps seq",qps,"concurrent",con_qps,"con p95",out["load_test"]["concurrent_p95_ms"])
print("metrics",svc.metrics()); print("fm invalid_k_high",fm["invalid_k_high"],"invalid_k_zero",fm["invalid_k_zero"],"health",fm["health"]["status"])
print("%.1fs"%(time.time()-t0))
