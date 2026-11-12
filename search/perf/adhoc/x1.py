import json, numpy as np
R=[json.loads(l) for l in open("se1/pairs.jsonl")]
print("part  a  b  docs pos  dlog% se_boot% se_delta%  sdu  cap20 dlog/se  maxsh  geo")
for r in R:
    if abs(r["logr"])<0.04:
        print(f'{r["part"][-5:]:6s} {r["a"][:22]:22s} {r["b"].split(":")[1][:12]:12s} {r["docs"]:4d} {r["positions"]:7d} {100*r["logr"]:+6.2f} {100*r["se_boot"]:5.2f} {100*r["se_delta"]:5.2f} {r["sd_u"]:5.2f} {100*r["logr_cap20"]:+6.2f}/{100*r["se_cap20"]:4.2f} {r["logr"]/r["se_boot"]:+5.1f} {r["max_share"]:.2f} {100*r["doc_geo"]:+6.2f}/{100*r["se_doc_geo"]:4.2f}')
