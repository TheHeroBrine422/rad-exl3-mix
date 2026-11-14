import json, numpy as np, unicodedata as U
G="/data/q27/goldens/golden-ver3-8k"; m=json.load(open(G+"/kld.json")); N=m["rows"]; gen=np.fromfile(G+"/gen.u8",np.uint8).astype(bool)
tok=np.fromfile(G+"/tokens.i32",np.int32)
from transformers import AutoTokenizer; tk=AutoTokenizer.from_pretrained("/data/q27/models/qwen38-27b-bf16")
F={"ours":"srch183/scores/ref183-init-2.5-ver3-8k-m.json.rows","uni":"srch183/scores/ref183-uni-2.5-ver3-8k-ver3-8k-m.json.rows","IQR":"w134/refs/ref-IQR4-iq2r-ver3-8k-m.json.rows"}
X={k:np.fromfile("/data/q27/"+f,np.float32).reshape(N,-1)[:,0] for k,f in F.items()}
off=0; rows=[]
for j,d in enumerate(m["docs"]):
    t=tok[off:off+d["tokens"]]; off+=d["tokens"]
    txt=tk.decode(t); letters=[c for c in txt if c.isalpha()]
    nonlat=sum(1 for c in letters if not ("LATIN" in U.name(c,"") )) / max(1,len(letters))
    s=slice(d["row0"],d["row0"]+d["tokens"]-1-d["score_from"]); g=gen[s]
    if g.sum()==0: continue
    rows.append((j,d.get("source"),nonlat,{k:float(X[k][s][g].sum()) for k in X},int(g.sum())))
for name,sel in [("non-Latin >50%",lambda r:r[2]>0.5),("Latin",lambda r:r[2]<=0.5)]:
    R=[r for r in rows if sel(r)]; n=sum(r[4] for r in R)
    print(f"{name}: {len(R)} docs, {n} rows: "+"  ".join(f"{k} {sum(r[3][k] for r in R)/n:.4f}" for k in X))
    if name.startswith("non"):
        for r in sorted(R,key=lambda r:-r[3]["ours"]):
            print(f"   doc {r[0]:3d} {r[1]:12} nonlatin {r[2]:.0%} rows {r[4]:5d}  ours {r[3]['ours']/r[4]:.3f} uni {r[3]['uni']/r[4]:.3f} IQR {r[3]['IQR']/r[4]:.3f}")
R=[r for r in rows if r[2]>0.5 and r[0]!=100]; n=sum(r[4] for r in R)
print("non-Latin excl doc 100: "+"  ".join(f"{k} {sum(r[3][k] for r in R)/n:.4f}" for k in X))
