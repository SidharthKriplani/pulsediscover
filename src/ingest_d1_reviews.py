"""D1 reviews extraction (lean, single synchronous pass). Reads decompressed reviews from
STDIN (piped from `gzip -dc`), extracts user_id, book_id, rating, date_added->event_ts via
bytes.find. Writes compact interactions CSV + d1_partial.json (cheap aggregates only).
Unique/dup/density stats are computed separately from the CSV (ingest_d1_stats.py)."""
import sys, os, json, calendar, time
from collections import Counter
OUT="/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim"
MON={b'Jan':1,b'Feb':2,b'Mar':3,b'Apr':4,b'May':5,b'Jun':6,b'Jul':7,b'Aug':8,b'Sep':9,b'Oct':10,b'Nov':11,b'Dec':12}
UK=b'"user_id": "'; BK=b'"book_id": "'; RK=b'"rating": '; DK=b'"date_added": "'
t0=time.time()
def parse_ts(s):
    try:
        mon=MON[s[4:7]]; day=int(s[8:10]); hh=int(s[11:13]); mm=int(s[14:16]); ss=int(s[17:19])
        off=s[20:25]; year=int(s[26:30])
        return calendar.timegm((year,mon,day,hh,mm,ss,0,0,0)) - (1 if off[0:1]==b'+' else -1)*(int(off[1:3])*3600+int(off[3:5])*60)
    except Exception: return None
n=written=miss_u=miss_b=miss_ts=rating0=0
ratings=Counter(); tmin=10**12; tmax=0
out=open(os.path.join(OUT,"domain_interactions.csv"),"w"); w=out.write
w("user_id,book_id,rating,event_ts\n")
for line in sys.stdin.buffer:
    n+=1
    i=line.find(UK)
    if i<0: miss_u+=1; continue
    i+=12; j=line.find(b'"',i); u=line[i:j]
    k=line.find(BK)
    if k<0: miss_b+=1; continue
    k+=12; mm=line.find(b'"',k); b=line[k:mm]
    r=line.find(RK); rat=0
    if r>=0:
        r+=10; e=r
        while e<len(line) and 48<=line[e]<=57: e+=1
        if e>r: rat=int(line[r:e])
    dd=line.find(DK); ts=None
    if dd>=0:
        dd+=15; q=line.find(b'"',dd); ts=parse_ts(line[dd:q])
    if ts is None: miss_ts+=1
    else:
        if ts<tmin: tmin=ts
        if ts>tmax: tmax=ts
    ratings[rat]+=1
    if rat==0: rating0+=1
    w(u.decode()); w(","); w(b.decode()); w(","); w(str(rat)); w(","); w(str(ts) if ts is not None else ""); w("\n")
    written+=1
out.close()
json.dump({"reviews_lines":n,"interactions_written":written,"missing_user":miss_u,"missing_book":miss_b,
 "missing_event_ts":miss_ts,"missing_ts_pct":round(100*miss_ts/max(1,written),3),
 "rating0_rows":rating0,"rating0_pct":round(100*rating0/max(1,written),2),
 "rating_dist":dict(sorted(ratings.items())),
 "date_min_unix":tmin if tmax else None,"date_max_unix":tmax or None,
 "date_min":time.strftime("%Y-%m-%d",time.gmtime(tmin)) if tmax else None,
 "date_max":time.strftime("%Y-%m-%d",time.gmtime(tmax)) if tmax else None,
 "extract_sec":round(time.time()-t0,1)}, open(os.path.join(OUT,"d1_partial.json"),"w"), indent=2)
print("EXTRACT_DONE", written, "rows", round(time.time()-t0,1),"s")
