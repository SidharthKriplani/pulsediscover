"""D1 stats — computed from the compact CSV (no decompression). Uniques, duplicate rate,
density, users>=5/items>=10, one-pass retained estimate; book coverage from meta CSV;
merges d1_partial.json -> d1_stats.json + D1_DONE."""
import os, json, csv, statistics as S
from collections import Counter
OUT="/sessions/festive-quirky-cannon/mnt/pulsediscover/data/interim"
INT=os.path.join(OUT,"domain_interactions.csv")
users=set(); items=set(); pairs=set(); per_user=Counter(); per_item=Counter()
with open(INT) as f:
    next(f)
    for line in f:
        c=line.find(","); u=line[:c]
        c2=line.find(",",c+1); b=line[c+1:c2]
        users.add(u); items.add(b); per_user[u]+=1; per_item[b]+=1; pairs.add(line[:c2])
written=sum(per_user.values())
retained=0
with open(INT) as f:
    next(f)
    for line in f:
        c=line.find(","); u=line[:c]; c2=line.find(",",c+1); b=line[c+1:c2]
        if per_user.get(u,0)>=5 and per_item.get(b,0)>=10: retained+=1
# book coverage
nb=auth=ser=lang=0; langs=Counter()
with open(os.path.join(OUT,"domain_books_meta.csv")) as bf:
    next(bf)
    for row in csv.reader(bf):
        if len(row)<5: continue
        nb+=1
        if row[1]: auth+=1
        if row[2]: ser+=1
        if row[4]: lang+=1; langs[row[4]]+=1
cnts=list(per_user.values())
partial=json.load(open(os.path.join(OUT,"d1_partial.json")))
stats=dict(partial)
stats.update({
 "books_total":nb,
 "books_author_cov_pct":round(100*auth/max(1,nb),2),
 "books_series_cov_pct":round(100*ser/max(1,nb),2),
 "books_lang_cov_pct":round(100*lang/max(1,nb),2),
 "books_top_langs":dict(langs.most_common(8)),
 "unique_users":len(users),"unique_items":len(items),"unique_pairs":len(pairs),
 "duplicate_userbook_rows":written-len(pairs),
 "duplicate_rate_pct":round(100*(written-len(pairs))/max(1,written),3),
 "interactions_per_user_mean":round(S.mean(cnts),2),
 "interactions_per_user_median":int(S.median(cnts)),
 "interactions_per_user_p90":int(sorted(cnts)[int(0.9*len(cnts))]),
 "users_ge5":sum(1 for v in per_user.values() if v>=5),
 "items_ge10":sum(1 for v in per_item.values() if v>=10),
 "interactions_retained_user5_item10_onepass":retained,
})
json.dump(stats, open(os.path.join(OUT,"d1_stats.json"),"w"), indent=2)
open(os.path.join(OUT,"D1_DONE"),"w").write("done")
print("STATS_DONE users=%d items=%d dup%%=%.2f retained=%d"%(len(users),len(items),stats["duplicate_rate_pct"],retained))
