"""C1 step 1 — stream raw Goodreads books gz, extract content features for cold-start:
book_id, title, description(trunc), popular_shelves(top-5 names), publication_year, language_code.
Merges author/series from the core meta. Writes domain_content_meta.csv. No modeling."""
import gzip, json, os, csv, time
OUT="/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim"
RAW="/sessions/festive-quirky-cannon/mnt/pulsediscover/data/raw/goodreads/goodreads_books_fantasy_paranormal.json.gz"
import pandas as pd
core_meta=pd.read_csv(os.path.join(OUT,"domain_core_books_meta.csv"),dtype=str).fillna("")
b2c=dict(zip(core_meta.book_id,core_meta.creator_id)); b2s=dict(zip(core_meta.book_id,core_meta.series_id))
t0=time.time(); n=0
with open(os.path.join(OUT,"domain_content_meta.csv"),"w",newline="") as f:
    w=csv.writer(f); w.writerow(["book_id","title","creator_id","series_id","language_code","publication_year","shelves","desc"])
    with gzip.open(RAW,"rt",encoding="utf-8",errors="replace") as fh:
        for line in fh:
            try: d=json.loads(line)
            except: continue
            n+=1
            bid=d.get("book_id") or ""
            title=(d.get("title") or "").replace("\n"," ").replace("\r"," ")[:200]
            lang=d.get("language_code") or ""
            py=d.get("publication_year") or ""
            sh=d.get("popular_shelves") or []
            shelves=" ".join((s.get("name","") for s in sh[:5] if isinstance(s,dict)))
            desc=(d.get("description") or "").replace("\n"," ").replace("\r"," ")[:200]
            w.writerow([bid,title,b2c.get(bid,""),b2s.get(bid,""),lang,py,shelves,desc])
print("CONTENT_EXTRACT books=%d sec=%.1f"%(n,time.time()-t0))
