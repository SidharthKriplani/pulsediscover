"""
D1 streaming extraction — Goodreads fantasy_paranormal. NO full load, NO review_text.
Reviews -> compact interactions (user_id, book_id, rating, event_ts).
Books   -> compact meta (book_id, creator_id, first series_id, title, language_code).
No k-core, no English filter, rating-0 kept (counted). Writes d1_stats.json + D1_DONE marker.
Run in background (nohup) — not bound by the shell timeout.
"""
import gzip, re, json, csv, calendar, os, time, sys

RAW = "/sessions/festive-quirky-cannon/mnt/pulsediscover/data/raw/goodreads"
OUT = "/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim"
os.makedirs(OUT, exist_ok=True)
REV = os.path.join(RAW, "goodreads_reviews_fantasy_paranormal.json.gz")
BKS = os.path.join(RAW, "goodreads_books_fantasy_paranormal.json.gz")
MON = {b'Jan':1,b'Feb':2,b'Mar':3,b'Apr':4,b'May':5,b'Jun':6,b'Jul':7,b'Aug':8,b'Sep':9,b'Oct':10,b'Nov':11,b'Dec':12}
t0 = time.time()
def log(m): open(os.path.join(OUT,"d1_progress.log"),"a").write(f"[{time.time()-t0:.0f}s] {m}\n")

# ---------------- BOOKS ----------------
nb=author_cov=series_cov=lang_cov=0
from collections import Counter
langs=Counter()
with open(os.path.join(OUT,"domain_books_meta.csv"),"w",newline="") as bf:
    bw=csv.writer(bf); bw.writerow(["book_id","creator_id","series_id","title","language_code"])
    with gzip.open(BKS,"rt",encoding="utf-8",errors="replace") as f:
        for line in f:
            try: d=json.loads(line)
            except: continue
            nb+=1
            bid=d.get("book_id") or ""
            auth=d.get("authors") or []
            cid=(auth[0].get("author_id","") if auth and isinstance(auth[0],dict) else "")
            ser=d.get("series") or []
            sid=(str(ser[0]) if ser else "")
            title=(d.get("title") or "").replace("\n"," ").replace("\r"," ")
            lang=d.get("language_code") or ""
            if cid: author_cov+=1
            if sid: series_cov+=1
            if lang: lang_cov+=1
            langs[lang]+=1
            bw.writerow([bid,cid,sid,title,lang])
log(f"books done n={nb}")

# ---------------- REVIEWS ----------------
u_re=re.compile(rb'"user_id": "([^"]*)"'); b_re=re.compile(rb'"book_id": "([^"]*)"')
r_re=re.compile(rb'"rating": (\d+)');     d_re=re.compile(rb'"date_added": "([^"]*)"')

def parse_ts(s):  # b'Sun Jul 30 07:44:10 -0700 2017'
    try:
        mon=MON[s[4:7]]; day=int(s[8:10]); hh=int(s[11:13]); mm=int(s[14:16]); ss=int(s[17:19])
        off=s[20:25]; year=int(s[26:30])
        epoch=calendar.timegm((year,mon,day,hh,mm,ss,0,0,0))
        sign=1 if off[0:1]==b'+' else -1
        return epoch - sign*(int(off[1:3])*3600+int(off[3:5])*60)
    except Exception:
        return None

n=written=miss_u=miss_b=miss_ts=rating0=0
users=set(); items=set(); pairs=set()
from collections import Counter as C2
per_user=C2(); per_item=C2(); ratings=C2()
tmin=10**12; tmax=0
with open(os.path.join(OUT,"domain_interactions.csv"),"w",newline="") as rf:
    rw=csv.writer(rf); rw.writerow(["user_id","book_id","rating","event_ts"])
    with gzip.open(REV,"rb") as f:
        for line in f:
            n+=1
            mu=u_re.search(line); mb=b_re.search(line)
            if not mu: miss_u+=1
            if not mb: miss_b+=1
            if not mu or not mb:
                if n%1000000==0: log(f"reviews {n}")
                continue
            u=mu.group(1).decode(); b=mb.group(1).decode()
            mr=r_re.search(line); rat=int(mr.group(1)) if mr else 0
            md=d_re.search(line); ts=parse_ts(md.group(1)) if md else None
            if ts is None: miss_ts+=1
            else:
                if ts<tmin: tmin=ts
                if ts>tmax: tmax=ts
            users.add(u); items.add(b); per_user[u]+=1; per_item[b]+=1
            pairs.add(u+"|"+b)
            ratings[rat]+=1
            if rat==0: rating0+=1
            rw.writerow([u,b,rat,ts if ts is not None else ""])
            written+=1
            if n%1000000==0: log(f"reviews {n} written={written}")
log(f"reviews done n={n} written={written}")

# post-filter (one-pass, non-iterative) estimate: rows with per_user>=5 AND per_item>=10
u5=sum(1 for c in per_user.values() if c>=5)
b10=sum(1 for c in per_item.values() if c>=10)
retained=0
with open(os.path.join(OUT,"domain_interactions.csv")) as rf:
    next(rf)
    for ln in rf:
        p=ln.split(",",2)
        if len(p)>=2 and per_user.get(p[0],0)>=5 and per_item.get(p[1],0)>=10:
            retained+=1
import statistics as S
cnts=list(per_user.values())
stats={
 "books_total": nb, "books_author_cov_pct": round(100*author_cov/max(1,nb),2),
 "books_series_cov_pct": round(100*series_cov/max(1,nb),2),
 "books_lang_cov_pct": round(100*lang_cov/max(1,nb),2),
 "books_top_langs": dict(langs.most_common(8)),
 "reviews_lines": n, "interactions_written": written,
 "unique_users": len(users), "unique_items": len(items), "unique_pairs": len(pairs),
 "duplicate_userbook_rows": written-len(pairs),
 "duplicate_rate_pct": round(100*(written-len(pairs))/max(1,written),3),
 "missing_user": miss_u, "missing_book": miss_b, "missing_event_ts": miss_ts,
 "missing_ts_pct": round(100*miss_ts/max(1,written),3),
 "rating0_rows": rating0, "rating0_pct": round(100*rating0/max(1,written),2),
 "rating_dist": dict(sorted(ratings.items())),
 "date_min_unix": tmin if tmax else None, "date_max_unix": tmax or None,
 "date_min": time.strftime("%Y-%m-%d", time.gmtime(tmin)) if tmax else None,
 "date_max": time.strftime("%Y-%m-%d", time.gmtime(tmax)) if tmax else None,
 "interactions_per_user_mean": round(S.mean(cnts),2) if cnts else 0,
 "interactions_per_user_median": int(S.median(cnts)) if cnts else 0,
 "users_ge5": u5, "items_ge10": b10,
 "interactions_retained_user5_item10_onepass": retained,
 "elapsed_sec": round(time.time()-t0,1),
}
json.dump(stats, open(os.path.join(OUT,"d1_stats.json"),"w"), indent=2)
open(os.path.join(OUT,"D1_DONE"),"w").write("done")
log("D1_DONE")
