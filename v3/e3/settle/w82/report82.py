#!/usr/bin/env python3
"""report82.py OUTDIR... -- #117 1(b) readout (../w82.md): rel(arm) / rel(natural) per tensor (geo over samples 0 / 1),
the rule's median at K2.5 and K4, per family, parameter-weighted geo, own / glob gap, globdesc."""
import collections, glob, json, math, sys
import numpy as np

R = collections.defaultdict(dict)                       # (layer, tensor, K) -> {sample: {arm: rel}}
NP = {}
for d in sys.argv[1:]:
    for f in glob.glob(f"{d}/**/*.jsonl", recursive=True):
        o = json.loads(open(f).readline())
        R[(o["layer"], o["w"], o["bits"])][o["seed"]] = o["rel"]; NP[(o["layer"], o["w"])] = o["N"] * o["K"]
FAM = ["ssm_qkv", "ssm_z", "attn_qg", "attn_k", "attn_v", "ffn_gate", "ffn_up"]
g = lambda v: math.exp(np.mean(np.log(v)))
for K in sorted({k[2] for k in R}):
    rows = []
    for (l, w, k), s in sorted(R.items()):
        if k != K: continue
        full = [x for x in s.values() if "natural" in x]
        r = {a: g([x[a] / x["natural"] for x in full if a in x]) for a in ("glob", "own", "globdesc")}
        rows.append((l, w, len(full), r))
    n2 = sum(1 for r in rows if r[2] == 2)
    gl = np.array([r[3]["glob"] for r in rows]); ow = np.array([r[3]["own"] for r in rows]); gd = np.array([r[3]["globdesc"] for r in rows])
    wts = np.array([NP[(r[0], r[1])] for r in rows], float)
    verdict = "≤ 0.98" if np.median(gl) <= 0.98 else "> 0.98"
    print(f"## K{K:g}: {len(rows)} tensors ({n2} with both samples)")
    print(f"glob/natural median {np.median(gl):.4f} ({verdict}), geo {g(gl):.4f}, param-weighted geo "
          f"{math.exp((wts * np.log(gl)).sum() / wts.sum()):.4f}, min {gl.min():.4f}, max {gl.max():.4f}")
    print(f"own/natural median {np.median(ow):.4f} (geo {g(ow):.4f}); globdesc/natural median {np.median(gd):.4f}")
    print("family | n | glob median | own median | globdesc median")
    for f in FAM:
        m = [r for r in rows if r[1] == f]
        if m: print(f"{f} | {len(m)} | {np.median([r[3]['glob'] for r in m]):.4f} | {np.median([r[3]['own'] for r in m]):.4f} | "
                    f"{np.median([r[3]['globdesc'] for r in m]):.4f}")
    print("tensor | samples | glob | own | globdesc")
    for l, w, n, r in rows: print(f"{l}:{w} | {n} | {r['glob']:.4f} | {r['own']:.4f} | {r['globdesc']:.4f}")
    print()
