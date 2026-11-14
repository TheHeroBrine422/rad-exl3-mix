#!/usr/bin/env python3
"""prox151.py MADDIR I8DIR STAMPS.json OUT.json -- #177-03 (w151) item 4 per level: the quant jobs' proxy error, mad level vs
its mul1i8 twin (144dd6e1 ladder), geometric mean of the per-tensor ratio over the 400 trunk tensors (p10 / p90, worst
family), plus the level's stamps. MADDIR / I8DIR: the levels' level/proxy.jsonl copies named <level>.jsonl."""
import json, math, os, sys
from collections import defaultdict
import numpy as np
M, I, ST, OUT = sys.argv[1:5]
stamps = json.load(open(ST)); out = {}
for f in sorted(os.listdir(M), key=lambda x: float(x.split("-k")[1].split("-")[0])):
    n = f[:-6]; i = n.replace("635bac02", "144dd6e1")[:-4] + "-i8"
    em = {j["key"]: j["proxy_err"] for j in map(json.loads, open(f"{M}/{f}"))}
    ei = {j["key"]: j["proxy_err"] for j in map(json.loads, open(f"{I}/{i}.jsonl"))}
    ks = sorted(set(em) & set(ei)); lr = np.array([math.log(em[k] / ei[k]) for k in ks]); bf = defaultdict(list)
    for k, v in zip(ks, lr): bf[".".join(k.split(".")[4:])].append(v)
    fam = {k: math.exp(np.mean(v)) for k, v in bf.items()}; wf = max(fam, key=fam.get)
    out[n] = {"K": float(n.split("-k")[1].split("-")[0]), "tensors": len(ks), "geo": math.exp(lr.mean()),
              "p10": math.exp(np.percentile(lr, 10)), "p90": math.exp(np.percentile(lr, 90)), "worst_family": [wf, fam[wf]],
              **stamps.get(n, {})}
json.dump(out, open(OUT, "w"), indent=1)
for n, r in out.items():
    print(f"K{r['K']:<6g} proxy mad/i8 geo {r['geo']:.4f} (p10 {r['p10']:.3f} p90 {r['p90']:.3f}; worst {r['worst_family'][0]} {r['worst_family'][1]:.4f})"
          f"  tensors {r['tensors']}  settings {r.get('settings', '?')[:8]} code {r.get('code', '?')[:8]} scales {str(r.get('scales'))[:8]}")
g = [r["geo"] for r in out.values()]
print(f"{len(g)} levels: geo of geos {math.exp(np.mean(np.log(g))):.4f}, range {min(g):.4f} - {max(g):.4f}")
