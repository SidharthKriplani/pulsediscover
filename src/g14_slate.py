"""G14 — final slate policy layer. Train LTR on the G13-mid candidate table, then apply slate
policies (pure LTR / +MMR / +cold reserve / +creator cap / combined) to build each test user's
top-20 slate and measure relevance vs governance. Offline re-ranking within candidates; deterministic
split. No online/production/gold claim; no FAISS/new candidate sources."""
import os, json, time, pickle, math, hashlib, numpy as np, pandas as pd, torch, lightgbm as lgb
from collections import defaultdict
import sys; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from pulsediscover import sasrec as S
from pulsediscover.feedback_loop import gini
OUT="/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim"; SP=os.path.join(OUT,"domain_splits")
EVID="/sessions/festive-quirky-cannon/mnt/pulsediscover/outputs/evidence"; PLOTS="/sessions/festive-quirky-cannon/mnt/pulsediscover/outputs/plots"; MOD="/sessions/festive-quirky-cannon/mnt/pulsediscover/outputs/_models/gpu"
SEED=20260616; MAXLEN=30; CONF=[("als",200),("sas",100),("content",50),("pop",30)]; t0=time.time()
def uh(u): return int(hashlib.md5(u.encode()).hexdigest(),16)%100
A=pickle.load(open(os.path.join(OUT,"c2_als.pkl"),"rb")); X=A["X"];Y=A["Y"];uidx=A["uidx"];iidx=A["iidx"]
meta=pickle.load(open(os.path.join(OUT,"d3b_seq_meta.pkl"),"rb")); item2idx=meta["item2idx"]; idx2item=meta["idx2item"]; n_items=meta["n_items"]; eval_seq=meta["eval_seq"]
st=torch.load(os.path.join(MOD,"ckpt_small.pt"),weights_only=False,map_location="cpu")
sas=S.SASRec(n_items,d=48,maxlen=MAXLEN,nblocks=1,nheads=1,dropout=0.2); sas.load_state_dict(st["model"]); sas.eval()
cm=pd.read_csv(os.path.join(OUT,"domain_content_meta.csv"),dtype=str).fillna("")
b2c=dict(zip(cm.book_id,cm.creator_id)); b2s=dict(zip(cm.book_id,cm.series_id)); b2y={b:(int(y) if y.isdigit() else 0) for b,y in zip(cm.book_id,cm.publication_year)}
pop=pd.read_csv(os.path.join(SP,"train.csv"),usecols=["book_id"],dtype=str).book_id.value_counts(); popd=pop.to_dict()
G=pickle.load(open(os.path.join(OUT,"g13_cands.pkl"),"rb")); cands=G["cands"]; poptop=G["poptop"]
es=pd.read_csv(os.path.join(SP,"eval_sample.csv"),dtype=str); users=es.user_id.tolist(); golds=dict(zip(es.user_id,es.gold))
warm=set(iidx); cold_set=set(G_cs) if False else pickle.load(open(os.path.join(OUT,"c2_lists.pkl"),"rb"))["cold_items"]  # noqa: F821 (dead branch, never evaluated)
hist_books={u:set(idx2item[i] for i in eval_seq.get(u,[]) if i<len(idx2item) and idx2item[i] is not None) for u in users}
uauth={u:set(b2c.get(b,"") for b in hist_books[u])-{""} for u in users}; user_ser={u:set(b2s.get(b,"") for b in hist_books[u])-{""} for u in users}
FN=["als_score","als_rank","sas_score","sas_rank","log_pop","pub_year","warm_flag","seen_flag","author_match","series_match","from_als","from_sas","from_content","from_pop","history_len"]
def lp(s): s=s[-MAXLEN:]; return [0]*(MAXLEN-len(s))+list(s)
# build features for all users, cache per-user candidate frame
ufeat={}; rowsX=[];rowsY=[];grp=[];split=[];order_users=[]; B=500
for s0 in range(0,len(users),B):
    ub=users[s0:s0+B]; urows=[uidx.get(u,-1) for u in ub]
    alssc=X[[r if r>=0 else 0 for r in urows]]@Y.T
    xb=torch.tensor([lp(eval_seq.get(u,[])) for u in ub],dtype=torch.long)
    with torch.no_grad(): sassc=sas.score_last(xb).numpy()
    for bi,u in enumerate(ub):
        c=cands[u]; cset=[]
        for src,k in CONF: cset+= (poptop[:k] if src=="pop" else c[src][:k])
        cset=list(dict.fromkeys(cset))
        arank={b:r for r,b in enumerate(c["als"])}; sasrank={b:r for r,b in enumerate(c["sas"])}
        aset=set(c["als"][:200]); sset=set(c["sas"][:100]); cont=set(c["content"][:50]); pset=set(poptop[:30])
        sv=sassc[bi]; row=alssc[bi]; au=uauth[u]; se=user_ser[u]; hb=hist_books[u]; hl=len(eval_seq.get(u,[]))
        feats=[]
        for b in cset:
            feats.append([float(row[iidx[b]]) if b in iidx else np.nan,arank.get(b,999),
                float(sv[item2idx[b]]) if b in item2idx else np.nan,sasrank.get(b,999),
                math.log1p(popd.get(b,0)),b2y.get(b,0),1 if b in warm else 0,1 if b in hb else 0,
                1 if b2c.get(b,"") in au else 0,1 if b2s.get(b,"") in se else 0,
                1 if b in aset else 0,1 if b in sset else 0,1 if b in cont else 0,1 if b in pset else 0,hl])
        F=np.array(feats,np.float32); ufeat[u]={"items":cset,"F":F}
        sl="tr" if uh(u)<70 else ("va" if uh(u)<85 else "te")
        rowsX.append(F); rowsY.append(np.array([1 if b==golds[u] else 0 for b in cset])); grp.append(len(cset)); split.append(sl); order_users.append(u)
Xf=np.vstack(rowsX); Yf=np.concatenate(rowsY)
sp_arr=np.array(split)
def block(mask_idx):
    sel=[i for i,s in enumerate(split) if s==mask_idx]; rows=np.concatenate([rowsX[i] for i in sel]); lab=np.concatenate([rowsY[i] for i in sel]); gp=[grp[i] for i in sel]
    return lgb.Dataset(rows,label=lab,group=gp,feature_name=FN)
params=dict(objective="lambdarank",metric="ndcg",ndcg_eval_at=[20],learning_rate=0.07,num_leaves=31,min_data_in_leaf=50,verbose=-1,seed=SEED,label_gain=[0,1])
m=lgb.train(params,block("tr"),num_boost_round=500,valid_sets=[block("va")],callbacks=[lgb.early_stopping(40,verbose=False)])
te_users=[u for u in order_users if (uh(u)>=85)]
# slate policies
def mmr_order(items,rel,lam=0.7):
    chosen=[]; rest=list(range(len(items))); cr=[b2c.get(items[i],"") for i in range(len(items))]; sr=[b2s.get(items[i],"") for i in range(len(items))]
    chosen_cr=set(); chosen_sr=set();
    while rest and len(chosen)<20:
        best=None;bs=-1e9
        for i in rest:
            sim=0.5*(cr[i] in chosen_cr)+0.5*(sr[i] in chosen_sr); val=lam*rel[i]-(1-lam)*sim
            if val>bs: bs=val;best=i
        chosen.append(best); chosen_cr.add(cr[best]); chosen_sr.add(sr[best]); rest.remove(best)
    return chosen+rest                                          # full order (top-20 are MMR-selected)
def cap_order(items,rel,cap=2):
    o=np.argsort(-rel); chosen=[];cnt=defaultdict(int);spill=[]
    for i in o:
        cr=b2c.get(items[i],"")
        if cnt[cr]<cap and len(chosen)<20: chosen.append(i);cnt[cr]+=1
        else: spill.append(i)
    return chosen+spill
def reserve_order(items,rel,k=2):
    o=list(np.argsort(-rel)); cold=[i for i in o if items[i] in cold_set]
    top=o[:20]; ncold=sum(1 for i in top if items[i] in cold_set)
    add=[i for i in cold if i not in top][:max(0,k-ncold)]
    if add:
        keep=[i for i in o[:20-len(add)]]; newtop=keep+add; rest=[i for i in o if i not in newtop]; return newtop+rest
    return o
def combined_order(items,rel,cap=2,k=2,lam=0.85):
    # cap + light mmr greedy, then ensure k cold in top-20
    cr=[b2c.get(items[i],"") for i in range(len(items))]; sr=[b2s.get(items[i],"") for i in range(len(items))]
    rest=list(np.argsort(-rel)); chosen=[];ccr=set();csr=set();cnt=defaultdict(int)
    while rest and len(chosen)<20:
        best=None;bs=-1e9
        for i in rest[:80]:
            if cnt[cr[i]]>=cap: continue
            sim=0.5*(cr[i] in ccr)+0.5*(sr[i] in csr); val=lam*rel[i]-(1-lam)*sim
            if val>bs: bs=val;best=i
        if best is None: best=rest[0]
        chosen.append(best);ccr.add(cr[best]);csr.add(sr[best]);cnt[cr[best]]+=1;rest.remove(best)
    # cold reserve
    ncold=sum(1 for i in chosen if items[i] in cold_set)
    if ncold<k:
        coldleft=[i for i in rest if items[i] in cold_set][:k-ncold]
        if coldleft: chosen=chosen[:20-len(coldleft)]+coldleft
    return chosen+[i for i in rest if i not in chosen]
POLICIES=["pure_ltr","mmr","cold_reserve","creator_cap","combined"]
def order(name,items,rel):
    if name=="pure_ltr": return list(np.argsort(-rel))
    if name=="mmr": return mmr_order(items,rel,0.7)
    if name=="cold_reserve": return reserve_order(items,rel,2)
    if name=="creator_cap": return cap_order(items,rel,2)
    return combined_order(items,rel)
res={}; pure_r20=None
for name in POLICIES:
    r20=r50=ndcg=0; cold_exp=0; slot_tot=0; viol=0; nov=[]; dcold=set(); dcre=set(); cre_exp=defaultdict(int); itemexp=defaultdict(int)
    for u in te_users:
        items=ufeat[u]["items"]; rel=m.predict(ufeat[u]["F"]); o=order(name,items,rel); top=o[:20]
        labs=[1 if items[i]==golds[u] else 0 for i in o]; pos=labs.index(1) if 1 in labs else 999
        if pos<20:r20+=1;ndcg+=1.0/math.log2(pos+2)
        if pos<50:r50+=1
        capcnt=defaultdict(int); ncold=0
        for i in top:
            b=items[i]; slot_tot+=1; cr=b2c.get(b,""); cre_exp[cr]+=1; itemexp[b]+=1; dcre.add(cr); capcnt[cr]+=1
            nov.append(1.0/math.log1p(popd.get(b,0)+1))
            if b in cold_set: cold_exp+=1; ncold+=1; dcold.add(b)
        if name in("creator_cap","combined") and max(capcnt.values())>2: viol+=1
        if name in("cold_reserve","combined") and ncold<2: viol+=1
    n=len(te_users)
    r20/=n; r50/=n; ndcg/=n
    ev=np.array(list(itemexp.values()),float); cge=np.array(list(cre_exp.values()),float)
    lt=sorted(itemexp.values(),reverse=True); ltshare=round(1-sum(lt[:max(1,int(0.01*len(lt)))])/max(sum(lt),1),4)
    res[name]={"R@20":round(r20,4),"R@50":round(r50,4),"NDCG@20":round(ndcg,4),
        "cold_exp_share":round(cold_exp/slot_tot,4),"distinct_cold_items":len(dcold),"distinct_creators":len(dcre),
        "creator_gini":round(gini(cge),4),"long_tail_share":round(ltshare,4),"mean_novelty":round(float(np.mean(nov)),4),
        "policy_violation_rate":round(viol/n,4)}
    if name=="pure_ltr": pure_r20=r20
for name in POLICIES: res[name]["warm_rel_loss_vs_pureLTR"]=round((res[name]["R@20"]-pure_r20)/pure_r20,4)
out={"scope":"G14 final slate policy layer (on G13-mid candidates + LTR)","tag":"[BUILT — real data, CPU]",
 "candidate_config":"mid (ALS200+SAS100+content50+pop30)","n_test_users":len(te_users),"split":"deterministic md5 user 70/15/15",
 "guardrail":"<=10% warm-relevance (R@20) loss vs pure LTR","policies":res,
 "honest_notes":["Offline re-ranking within candidates; no online/production/RiskFrame-gold claim.",
   "Governance metrics on top-20 slates over test users; reuses H1/H2 tradeoff on the LTR slate.",
   "Deterministic split; no FAISS/new candidate sources."],
 "sec":round(time.time()-t0,1)}
json.dump(out,open(os.path.join(EVID,"g14_final_slate_policy_report.json"),"w"),indent=2)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,6))
    for name in POLICIES:
        ax.scatter(res[name]["creator_gini"],res[name]["R@20"],s=110,label=name)
        ax.annotate(name,(res[name]["creator_gini"],res[name]["R@20"]),fontsize=8,xytext=(4,4),textcoords="offset points")
    ax.axhline(pure_r20*0.9,ls="--",c="red",alpha=0.6,label="-10% warm guardrail")
    ax.set_xlabel("creator exposure Gini (lower=healthier)"); ax.set_ylabel("R@20 (relevance)")
    ax.set_title("G14 relevance vs governance tradeoff (top-right = best)"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS,"g14_relevance_governance_tradeoff.png"),dpi=110); plt.close()
except Exception as e: print("plot skipped",e)
for name in POLICIES:
    r=res[name]; print("%-13s R@20 %.4f (loss %+.1f%%) cold%% %.3f distCold %d distCre %d gini %.3f nov %.3f viol %.3f"%(
      name,r["R@20"],100*r["warm_rel_loss_vs_pureLTR"],r["cold_exp_share"],r["distinct_cold_items"],r["distinct_creators"],r["creator_gini"],r["mean_novelty"],r["policy_violation_rate"]))
print("sec",round(time.time()-t0,1))
