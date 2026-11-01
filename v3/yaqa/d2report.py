#!/usr/bin/env python3
"""d2report.py SCORE.jsonl -- read D2. (1) ours vs turboderp at K4 / K6: median over tensors of rel(ours, damping 0.025) /
rel(his) on H_A (ours in-sample), H_B (both out of sample) and H_pub. If the ratio is clearly lower on H_A than on H_B,
part of our proxy advantage is in-sample. (2) damping: per tensor, the arm with the lowest rel on H_A vs on H_B; how
often they agree, and the median gain of the H_A-chosen arm measured on H_B (honest) vs on H_A (in-sample)."""
import collections, json, statistics as st, sys
R = collections.defaultdict(dict)
for l in open(sys.argv[1]):
    d = json.loads(l); R[d["key"]][d["arm"]] = d
fam = lambda k: k.split(".")[-1]
for K in (4, 6):
    print(f"== K{K}: ours (H_A, sigma 0.025) / turboderp, median [min, max] over tensors")
    for h in ("A", "B", "pub"):
        r = [v[f"o{K}-0.025"][f"rel_{h}"] / v[f"td{K}"][f"rel_{h}"] for v in R.values() if f"o{K}-0.025" in v and f"td{K}" in v]
        if r: print(f"   on H_{h:3s}: {st.median(r):.3f} [{min(r):.3f}, {max(r):.3f}]  n={len(r)}")
    arms = [f"o{K}-{s}" for s in ("0.025", "0.005", "0.001")]
    agree, gA, gB = 0, [], []; n = 0
    for k, v in R.items():
        if not all(a in v for a in arms): continue
        n += 1
        bA = min(arms, key=lambda a: v[a]["rel_A"]); bB = min(arms, key=lambda a: v[a]["rel_B"])
        agree += bA == bB
        gA.append(v[bA]["rel_A"] / v[arms[0]]["rel_A"]); gB.append(v[bA]["rel_B"] / v[arms[0]]["rel_B"])
    if n: print(f"   damping: best arm on H_A == best on H_B for {agree}/{n} tensors; H_A-chosen arm vs 0.025: "
                f"in-sample {st.median(gA):.3f}, out-of-sample {st.median(gB):.3f} (median)")
