import json, numpy as np, collections
R=[json.loads(l) for l in open("se2/pairs.jsonl")]
def cost(tok, docs): return 0.00075*tok + 0.3*docs
# efficiency of each cap relative to whole docs: var * cost ratio, median over pairs, by part kind
kinds=collections.defaultdict(list)
for r in R:
    if r["se_delta"]<=0: continue
    k = "x" if r["part"].endswith("-x") else "c"
    full = r["se_delta"]**2*cost(r["tokens"], r["docs"])
    row = {}
    for c,v in r["caps"].items():
        if v["se_delta"]>0: row[int(c)] = (v["se_delta"]**2*cost(v["tokens"], r["docs"]))/full
    # bias: logr at cap vs full, in units of full SE
    row["bias"] = {int(c): (v["logr"]-r["logr"])/r["se_boot"] for c,v in r["caps"].items()}
    row["tokfrac"] = {int(c): v["tokens"]/r["tokens"] for c,v in r["caps"].items()}
    kinds[(k, r["part"])].append(row)
for (k,p),rows in sorted(kinds.items()):
    print(f"{p:20s} n={len(rows):3d}  ", "  ".join(f"cap{c}: eff {np.median([x[c] for x in rows]):.2f} tok {np.median([x['tokfrac'][c] for x in rows]):.2f} bias|z| {np.median(np.abs([x['bias'][c] for x in rows])):.2f}" for c in (512,1024,2048,4096)))
