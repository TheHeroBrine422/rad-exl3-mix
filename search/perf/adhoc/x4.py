import json, numpy as np, collections
R=[json.loads(l) for l in open("se2/pairs.jsonl")]
P=collections.defaultdict(dict)
for r in R:
    g=r["part"]
    if g.endswith(("-x","-c")): P[(g[:-2], r["model_a"], r["model_b"])][g[-1]]=r
rows=[(v["x"]["logr"],v["c"]["logr"],v["x"]["se_boot"],v["c"]["se_boot"],v["x"]["sd_u"],v["c"]["sd_u"],v["x"]["docs"],v["c"]["docs"],k[0]) for k,v in P.items() if "x" in v and "c" in v]
A=np.array([r[:8] for r in rows])
x,c,sx,sc,ux,uc,nx,nc=A.T
print("pairs with both parts:", len(A))
w=1/(sc**2+sx**2)
# slope of c on x through origin, weighted; and ratio of sd_u per doc
b=np.sum(w*x*c)/np.sum(w*x*x); print("slope c~x (through 0, weighted):", round(b,3))
big=np.abs(x)>2*sx
print("pairs |x| > 2 SE:", big.sum(), " same sign on c:", np.mean(np.sign(x[big])==np.sign(c[big])).round(2), " c sig same sign:", np.mean((np.sign(x[big])==np.sign(c[big]))&(np.abs(c[big])>2*sc[big])).round(2))
z=(c-b*x)/np.sqrt(sc**2+(b*sx)**2); print("z of c - b x: sd", z.std().round(2), " (1 = differences explained by noise)")
print("sd_u x / sd_u c per doc: median", np.median(ux/uc).round(2), " IQR", np.percentile(ux/uc,[25,75]).round(2))
print("SE^2*docs ratio x/c (per-doc variance):", np.median((sx**2*nx)/(sc**2*nc)).round(2))
for s in sorted(set(r[8] for r in rows)):
    m=np.array([r[8]==s for r in rows]); print(s, m.sum(), "slope", round(np.sum(w[m]*x[m]*c[m])/np.sum(w[m]*x[m]*x[m]),3), "z sd", z[m].std().round(2))
