#!/usr/bin/env python3
"""emb2report.py LDLQ.jsonl... -- #119 embedding LDLQ screen table: each arm vs the H = I control, in every metric.

ratio = arm error / control error at the same K; bits = the K the control needs for the arm's error (log-linear in K
between the control's points) minus the arm's K (> 0: the arm saves bits in that metric). Verdict per RULES.md
(amendments 1-2): an arm passes a rate when it is >= 10% lower (or saves >= 0.25 bit) in BOTH M0 and F; it passes when
that holds at >= 2 of the 4 rates."""
import json, sys
import numpy as np

R = {}
for fn in sys.argv[1:]:
    for l in open(fn):
        d = json.loads(l); R[(d["H"], float(d["K"]))] = d
Ks = sorted({k for h, k in R if h == "I"})
MET = ("rel_I", "rel_M0", "rel_F")


def bits(m, e, K):
    ks = np.array(Ks); le = np.log([R[("I", k)][m] for k in Ks])        # decreasing in K
    x = np.log(e); i = int(np.clip(np.searchsorted(-le, -x), 1, len(ks) - 1))   # segment (ends: extrapolated)
    return float(ks[i - 1] + (x - le[i - 1]) / (le[i] - le[i - 1]) * (ks[i] - ks[i - 1])) - K


print("| H | K | " + " | ".join(f"{m[4:]} ratio (bits)" for m in MET) + " | rate passes |")
print("|---|---|" + "---|" * (len(MET) + 1))
verdict = {}
for h in sorted({h for h, k in R if h != "I"}):
    n = 0
    for K in sorted(k for hh, k in R if hh == h):
        if ("I", K) not in R: continue
        a, c = R[(h, K)], R[("I", K)]
        cells, ok = [], True
        for m in MET:
            r = a[m] / c[m]; b = bits(m, a[m], K); cells.append(f"{r:.3f} ({b:+.2f})")
            if m != "rel_I" and not (r <= 0.90 or b >= 0.25): ok = False
        n += ok
        print(f"| {h} | {K:g} | " + " | ".join(cells) + f" | {'yes' if ok else 'no'} |")
    verdict[h] = "PASS" if n >= 2 else "FAIL"
print("\ncontrol (H = I): " + ", ".join(f"K{k:g} rel_I {R[('I', k)]['rel_I']:.5f}" for k in Ks))
print("verdicts:", verdict)
