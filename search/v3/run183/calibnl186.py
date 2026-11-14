import json, numpy as np, unicodedata as U, collections
G="/data/q27/challenge/golden/calib"; m=json.load(open(G+"/kld.json")); gen=np.fromfile(G+"/gen.u8",np.uint8).astype(bool)
tok=np.memmap(G+"/tokens.i32",np.int32,"r")
from transformers import AutoTokenizer; tk=AutoTokenizer.from_pretrained("/data/q27/models/qwen38-27b-bf16")
off=0; nl_rows=0; all_rows=0; nd=0; src=collections.Counter()
for d in m["docs"]:
    t=tok[off:off+d["tokens"]]; off+=d["tokens"]
    txt=tk.decode(t[:3000]); L=[c for c in txt if c.isalpha()]
    r=d["row0"]; n=d["tokens"]-1-d["score_from"]; g=int(gen[r:r+n].sum()); all_rows+=g
    if L and sum(1 for c in L if "LATIN" not in U.name(c,""))/len(L)>0.5:
        nl_rows+=g; nd+=1; src[str(d.get("source") or d.get("category"))[:30]]+=g
print(f"calib: {len(m['docs'])} docs, gen rows {all_rows}; non-Latin docs {nd}, gen rows {nl_rows} ({nl_rows/all_rows:.1%})")
print(src.most_common(12))
