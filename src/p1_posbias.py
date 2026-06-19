"""P1 — position-bias correction on the two-lane served policy. Simulate top-20 slates with a PBM
examination curve; observed click = examined AND item==user gold (relevance proxy); true relevance
known. Compare naive CTR vs IPS examination-correction vs true, by position/lane/item. Policies:
90/10 (primary), als_only (control), 80/20 (aggressive). No business-lift claim."""
import pandas as pd, numpy as np, os, json, pickle, sys, time
from collections import defaultdict
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from pulsediscover.feedback_loop import gini
OUT="/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim"; SP=os.path.join(OUT,"domain_splits")
EVID="/sessions/festive-quirky-cannon/mnt/pulsediscover/outputs/evidence"; PLOTS="/sessions/festive-quirky-cannon/mnt/pulsediscover/outputs/plots"
L=20; SEED=20260616; t0=time.time(); rng=np.random.default_rng(SEED)
exam=0.95*(1.0/(np.arange(L)+1.0))**0.7                  # KNOWN PBM examination propensity by position
lists=pickle.load(open(os.path.join(OUT,"c2_lists.pkl"),"rb")); WL=lists["warm_list"]; CL=lists["cold_list"]; cold_set=lists["cold_items"]; b2c=lists["b2c"]
w=pd.read_csv(os.path.join(SP,"warm_eval_sample.csv"),dtype=str); c=pd.read_csv(os.path.join(SP,"cold_eval_sample.csv"),dtype=str)
inst=[(u,g) for u,g in zip(w.user_id,w.gold)]+[(u,g) for u,g in zip(c.user_id,c.gold)]
def slate(u,name):
    wl=WL.get(u,[]); cl=CL.get(u,[])
    s={"als_only":wl[:L],"q90_10":wl[:18]+cl[:2],"q80_20":wl[:16]+cl[:4]}[name]
    return (s+[None]*L)[:L]
COLD_POS={"als_only":set(),"q90_10":{18,19},"q80_20":{16,17,18,19}}
POL=["als_only","q90_10","q80_20"]
report={}
for name in POL:
    naive=np.zeros(L); ips=np.zeros(L); true=np.zeros(L); n=0
    it_imp=defaultdict(int); it_clk=defaultdict(float); it_ips=defaultdict(float); it_true=defaultdict(int)
    for u,gold in inst:
        s=slate(u,name); n+=1
        for pos in range(L):
            a=s[pos]; rel=1 if a==gold else 0
            examined=rng.random()<exam[pos]; click=1 if (examined and rel) else 0
            naive[pos]+=click; ips[pos]+=click/exam[pos]; true[pos]+=rel
            if a is not None:
                it_imp[a]+=1; it_clk[a]+=click; it_ips[a]+=click/exam[pos]; it_true[a]+=rel
    naive/=n; ips/=n; true/=n
    # item-level (items with >=20 impressions)
    items=[a for a in it_imp if it_imp[a]>=20]
    nv=np.array([it_clk[a]/it_imp[a] for a in items]); ip=np.array([it_ips[a]/it_imp[a] for a in items]); tr=np.array([it_true[a]/it_imp[a] for a in items])
    def corr(x,y): return float(np.corrcoef(x,y)[0,1]) if len(x)>2 and x.std()>0 and y.std()>0 else None
    cold_pos=sorted(COLD_POS[name]); warm_pos=[p for p in range(L) if p not in COLD_POS[name]]
    def lane(idx): return {"naive":round(float(naive[idx].mean()),5),"ips":round(float(ips[idx].mean()),5),"true":round(float(true[idx].mean()),5)} if len(idx) else None
    # exposure concentration: clicks vs ips-credit per item
    clk_arr=np.array([it_clk[a] for a in it_imp]); ips_arr=np.array([it_ips[a] for a in it_imp])
    report[name]={
      "naive_ctr_by_pos":[round(float(x),5) for x in naive],
      "ips_rel_by_pos":[round(float(x),5) for x in ips],
      "true_rel_by_pos":[round(float(x),5) for x in true],
      "lane_warm":lane(np.array(warm_pos)),"lane_cold":lane(np.array(cold_pos)) if cold_pos else None,
      "item_naive_vs_true":{"corr":round(corr(nv,tr),4) if corr(nv,tr) else None,"mse":round(float(np.mean((nv-tr)**2)),6),"n_items":len(items)},
      "item_ips_vs_true":{"corr":round(corr(ip,tr),4) if corr(ip,tr) else None,"mse":round(float(np.mean((ip-tr)**2)),6)},
      "ips_weight_range":{"min":round(float(1/exam[0]),3),"max":round(float(1/exam[L-1]),3)},
      "credit_gini_clicks":round(gini(clk_arr),4),"credit_gini_ips":round(gini(ips_arr),4)}
out={"dataset":"goodreads_fantasy_paranormal","scope":"position-bias correction on two-lane served policy","tag":"[BUILT][SYNTHETIC — position-bias measurement]",
 "pbm_examination_by_pos":[round(float(x),4) for x in exam],"n_instances":len(inst),
 "policies":report,
 "ab_logging_schema":["timestamp","user_id","user_context","policy_id","slate_position","item_id","lane(warm/cold)",
                      "creator_id","examination_propensity(position)","logging_propensity(item)","reward(click/engagement)"],
 "honest_notes":["Examination propensity is KNOWN here (simulator PBM); in production it must be estimated via randomization/PBM-EM.",
   "Reward = examined AND item==gold (relevance proxy), NOT real engagement/lift.",
   "IPS divides click by examination(position): unbiased for true relevance, variance grows at low positions.",
   "True relevance is the latent the naive CTR is a position-biased observation of."],
 "sec":round(time.time()-t0,1)}
json.dump(out,open(os.path.join(EVID,"domain_position_bias_two_lane_report.json"),"w"),indent=2)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(13,4.8))
    r=report["q90_10"]; pos=np.arange(L)
    ax[0].plot(pos,r["naive_ctr_by_pos"],"o-",label="naive CTR (position-confounded)")
    ax[0].plot(pos,r["ips_rel_by_pos"],"s-",label="IPS-corrected relevance")
    ax[0].plot(pos,r["true_rel_by_pos"],"^--",label="true relevance (known)")
    ax[0].axvspan(17.5,19.5,color="orange",alpha=0.2,label="cold slots (18-19)")
    ax[0].set_xlabel("slate position"); ax[0].set_ylabel("rate"); ax[0].set_title("90/10: naive CTR vs IPS vs true by position"); ax[0].legend(fontsize=8)
    labels=["warm lane","cold lane"]; nv=[r["lane_warm"]["naive"],r["lane_cold"]["naive"]]; ipv=[r["lane_warm"]["ips"],r["lane_cold"]["ips"]]; tv=[r["lane_warm"]["true"],r["lane_cold"]["true"]]
    x=np.arange(2); wd=0.25
    ax[1].bar(x-wd,nv,wd,label="naive"); ax[1].bar(x,ipv,wd,label="IPS"); ax[1].bar(x+wd,tv,wd,label="true")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels); ax[1].set_title("90/10: warm vs cold lane — naive underrates cold"); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS,"domain_position_bias_two_lane.png"),dpi=110); plt.close()
except Exception as ex: print("plot skipped:",ex)
for name in POL:
    r=report[name]; print(name,"naive_pos0=%.4f pos19=%.4f | ips_pos0=%.4f pos19=%.4f | true_pos0=%.4f pos19=%.4f"%(
      r["naive_ctr_by_pos"][0],r["naive_ctr_by_pos"][19],r["ips_rel_by_pos"][0],r["ips_rel_by_pos"][19],r["true_rel_by_pos"][0],r["true_rel_by_pos"][19]))
    if r["lane_cold"]: print("   cold lane: naive %.4f ips %.4f true %.4f | warm lane: naive %.4f ips %.4f true %.4f"%(
      r["lane_cold"]["naive"],r["lane_cold"]["ips"],r["lane_cold"]["true"],r["lane_warm"]["naive"],r["lane_warm"]["ips"],r["lane_warm"]["true"]))
    print("   item corr naive-vs-true %s, ips-vs-true %s | credit Gini clicks %.3f ips %.3f"%(
      r["item_naive_vs_true"]["corr"],r["item_ips_vs_true"]["corr"],r["credit_gini_clicks"],r["credit_gini_ips"]))
