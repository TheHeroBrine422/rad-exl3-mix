import json, numpy as np, collections
R=[json.loads(l) for l in open("se2/pairs.jsonl")]
print("parts:", collections.Counter(r["part"] for r in R))
for kind in ("x","c"):
    S=[r for r in R if (r["part"].endswith("-x"))==(kind=="x")]
    a=np.array([abs(r["logr"]) for r in S]); s=np.array([r["sd_u"] for r in S]); n=np.array([r["docs"] for r in S])
    print(f"== {kind}: n pairs {len(S)}")
    for lo,hi in ((0,0.005),(0.005,0.01),(0.01,0.02),(0.02,0.04),(0.04,0.08),(0.08,0.13)):
        m=(a>=lo)&(a<hi)
        if m.sum(): print(f"  |dlog| {100*lo:4.1f}-{100*hi:4.1f}%: {m.sum():4d} pairs  sd_u median {np.median(s[m]):.3f} IQR {np.percentile(s[m],25):.3f}-{np.percentile(s[m],75):.3f}   docs med {int(np.median(n[m]))}  SE@100docs {100*np.median(s[m])/10:.2f}%")
    # ratio sd_u / |dlog|
    m=a>0.005; print("  sd_u/|dlog| median (|dlog|>0.5%):", np.median(s[m]/a[m]).round(1))
