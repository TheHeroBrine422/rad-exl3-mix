import json, glob, os, numpy as np
def main_curve(f):
    L=[json.loads(l) for l in open(f)]
    d={}
    for x in L:
        if x["what"]=="main": d[x["rows"]]=(np.array(x["sums"]),np.array(x["n"]))
    return sorted(d.items())
def boot(a,b,n,R=2000):
    rng=np.random.default_rng(0); k=len(n); r=[]
    for _ in range(R):
        i=rng.integers(0,k,k); r.append(b[i].sum()/a[i].sum())
    return np.percentile(r,[2.5,97.5])
for f in sorted(glob.glob("/tmp/rp/*.fixed.jsonl")):
    c=main_curve(f); name=os.path.basename(f).split(".fixed")[0]
    if len(c)<3: continue
    base=c[0][1][0].sum()
    print(f"== {name}")
    prev=None; fired=None
    for rows,(s,n) in c:
        kl=s.sum()/n.sum(); line=f"  {rows/1e6:5.2f}M  mon {kl:.5f}  vs start {s.sum()/base:.4f}"
        if prev is not None:
            ps,pn=prev; step=s.sum()/ps.sum(); lo,hi=boot(ps,s,n); line+=f"  step ratio {step:.4f} [{lo:.4f},{hi:.4f}]"
            if fired is None and rows>=5e5 and hi>0.995 : fired=rows; line+="  <- rule fires (CI upper > 0.995: no clear 0.5 % gain)"
        print(line); prev=(s,n)
