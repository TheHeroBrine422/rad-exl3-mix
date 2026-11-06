#!/usr/bin/env python3
"""chain/report.py DIR -- #102 tables from run.sh's data/*.jsonl (README.md "Rule"): Gaussian tiles (mse ratios) and
the real tensors (rel(n) / rel(n=1) per tensor; median and geometric mean over tensors, per family geometric mean,
free / TB and the share of the free-vs-TB gap recovered)."""
import glob, json, math, os, sys
from collections import defaultdict

rows = [json.loads(l) for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.jsonl"))) for l in open(f)]
gm = lambda v: math.exp(sum(math.log(x) for x in v) / len(v))
med = lambda v: sorted(v)[len(v) // 2] if len(v) % 2 else sum(sorted(v)[len(v) // 2 - 1:len(v) // 2 + 1]) / 2

G = defaultdict(dict)
for r in rows:
    if r["mode"] == "gauss": G[r["bits"]][r["arm"]] = r
if G:
    print("## Gaussian tiles (mul1, i.i.d. N(0,1), seed 7): mse / mse(TB n=1)")
    arms = ["free", "tiles_tb16"] + [f"chain{n}" for n in (2, 4, 8, 16, 32)]
    print("| K | tiles | mse TB n=1 | " + " | ".join(arms) + " | n=1 = viterbi_tiles_tb |")
    print("|---" * (len(arms) + 4) + "|")
    for K in sorted(G):
        a = G[K]; b = a["chain1"]["mse"]
        print(f"| {K:g} | {a['chain1']['tiles']} | {b:.4e} | " + " | ".join(f"{a[x]['mse'] / b:.4f}" if x in a else "–" for x in arms)
              + f" | {a['chain1'].get('identical_to_tiles_tb1', '–')} |")

T = defaultdict(dict)   # (layer, w, K) -> n -> rel
for r in rows:
    if r["mode"] == "tensor": T[(r["layer"], r["w"], r["bits"])][r["n"]] = r
if not T: sys.exit()
Ks = sorted({k[2] for k in T}); ns = (2, 4, 8)
print("\n## Real tensors (#57 set, mul1, 128 rows, block LDLQ, calib H_in): rel(n) / rel(n=1)")
print("| K | tensors | free / TB | " + " | ".join(f"n={n} median | n={n} gmean | n={n} max" for n in ns) + " | gap recovered at n=4 (gmean) | sec n=1 / free (median) |")
print("|---" * (3 + 3 * len(ns) + 2) + "|")
for K in Ks:
    ts = [k for k in T if k[2] == K and all(n in T[k] for n in (0, 1) + ns)]
    if not ts: continue
    fr = [T[k][0]["rel"] / T[k][1]["rel"] for k in ts]
    cells = []
    for n in ns:
        v = [T[k][n]["rel"] / T[k][1]["rel"] for k in ts]; cells += [f"{med(v):.4f}", f"{gm(v):.4f}", f"{max(v):.4f}"]
    r4 = gm([T[k][4]["rel"] / T[k][1]["rel"] for k in ts]); f0 = gm(fr)
    sec = med([T[k][1]["sec"] / T[k][0]["sec"] for k in ts])
    print(f"| {K:g} | {len(ts)} | {f0:.4f} | " + " | ".join(cells) + f" | {(1 - r4) / (1 - f0):.0%} | {sec:.2f} |")

print("\n## Per family, n=4 / n=1 (geometric mean over the family's tensors)")
fams = sorted({k[1] for k in T})
print("| family | " + " | ".join(f"K{K:g}" for K in Ks) + " |"); print("|---" * (len(Ks) + 1) + "|")
for fam in fams:
    c = []
    for K in Ks:
        v = [T[k][4]["rel"] / T[k][1]["rel"] for k in T if k[1] == fam and k[2] == K and 4 in T[k] and 1 in T[k]]
        c.append(f"{gm(v):.4f} ({len(v)})" if v else "–")
    print(f"| {fam} | " + " | ".join(c) + " |")

print("\n## Per tensor: rel(n=1) and rel(n)/rel(n=1) for n = 0 (free), 2, 4, 8")
print("| layer | tensor | K | rel TB n=1 | free | n=2 | n=4 | n=8 |"); print("|---" * 8 + "|")
for k in sorted(T, key=lambda k: (k[2], k[1], k[0])):
    d = T[k]
    if 1 not in d: continue
    b = d[1]["rel"]
    print(f"| {k[0]} | {k[1]} | {k[2]:g} | {b:.4e} | " + " | ".join(f"{d[n]['rel'] / b:.4f}" if n in d else "–" for n in (0, 2, 4, 8)) + " |")

print("\n## Rule (README): adopt if the n=4 median ratio <= 0.98 at K2, K2.5 and K3, and every family's n=4 gmean <= 1.00")
ok = True
for K in (2.0, 2.5, 3.0):
    v = [T[k][4]["rel"] / T[k][1]["rel"] for k in T if k[2] == K and 4 in T[k] and 1 in T[k]]
    if not v: print(f"K{K:g}: no data"); ok = False; continue
    p = med(v) <= 0.98; ok &= p; print(f"K{K:g}: median {med(v):.4f} -> {'pass' if p else 'FAIL'}")
bad = []
for fam in fams:
    for K in Ks:
        v = [T[k][4]["rel"] / T[k][1]["rel"] for k in T if k[1] == fam and k[2] == K and 4 in T[k] and 1 in T[k]]
        if v and gm(v) > 1.0: bad.append(f"{fam} K{K:g} {gm(v):.4f}")
print("families > 1.00:", ", ".join(bad) or "none"); ok &= not bad
print("VERDICT:", "ADOPT" if ok else "DO NOT ADOPT")
