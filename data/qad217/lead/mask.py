import numpy as np, collections, re
N=40; Z=3.0
def load(s):
    z=np.load(f"/data/q27/w217/runs/gn-{s}/gradnoise.npz"); out={}
    for k in z.keys():
        if k.endswith("|mean"):
            b=k[:-5]; m=z[k]; sd=z[b+"|std"]; out[b]=np.abs(m)>Z*sd/np.sqrt(N)
    return out
def parse(b):
    m=re.search(r"layers\.(\d+)\.(\w+)\.(\w+):(\w)",b); L=int(m.group(1)); blk=m.group(2); t=m.group(3); side={"r":"in(suh)","c":"out(svh)"}[m.group(4)]
    blk={"linear_attn":"GDN","self_attn":"attn","mlp":"mlp"}.get(blk,blk); return L,blk,t,side
M={s:load(s) for s in ("2.50","3.00","4.00")}
print(f"chance rate at {Z} se ≈ 0.27 %\n")
for s in M:
    tot=sum(v.size for v in M[s].values()); sig=sum(v.sum() for v in M[s].values()); print(f"{s}: {sig}/{tot} significant = {100*sig/tot:.2f} %")
def table(key, title):
    print(f"\n== by {title}  (% significant: 2.50 / 3.00 / 4.00; share of all scales)")
    rows=collections.OrderedDict()
    for s in M:
        for b,v in M[s].items():
            k=key(parse(b)); r=rows.setdefault(k,{}); a=r.setdefault(s,[0,0]); a[0]+=v.sum(); a[1]+=v.size
    tot=sum(v.size for v in M["2.50"].values())
    for k in sorted(rows, key=str):
        r=rows[k]; print(f"  {str(k):34s} "+" / ".join(f"{100*r[s][0]/r[s][1]:5.2f}" for s in M)+f"   ({100*r['2.50'][1]/tot:4.1f} % of scales)")
table(lambda p:(p[1],p[2]), "block / tensor")
table(lambda p:p[3], "side")
table(lambda p:(p[1],p[3]), "block × side")
table(lambda p:f"layers {16*(p[0]//16):02d}-{16*(p[0]//16)+15:02d}", "depth band")
# overlap 2.50 vs 3.00 vs 4.00
for a,b in (("2.50","3.00"),("2.50","4.00"),("3.00","4.00")):
    inter=sum((M[a][k]&M[b][k]).sum() for k in M[a]); na=sum(M[a][k].sum() for k in M[a]); nb=sum(M[b][k].sum() for k in M[b])
    print(f"\noverlap {a} & {b}: {inter} shared; {100*inter/na:.1f} % of {a}'s, {100*inter/nb:.1f} % of {b}'s (chance ~{100*nb/sum(v.size for v in M[b].values()):.2f} %)")
