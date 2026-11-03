#!/usr/bin/env python3
"""hratio.py K BASE.jsonl H.jsonl [BASE2.jsonl H2.jsonl ...] -- knap4 KN_HRATIO lines for level K: per linear, the
two-sided (h-weighted) error ratio of an H_out quant to the one-sided quant of the same settings, geometric mean over
the given (base, H) pairs (rotation-seed replicates). Inputs: ourq proxy.jsonl files written with --hprice (field
proxy_err2) or rowerr2.py output (field e2; an "arm" field, if several arms are in one file, must be filtered first).
Writes {"key", "K", "r", "e1_ratio", "n"} per linear to stdout: e1_ratio (one-sided) is logged beside it."""
import json, math, sys
from collections import defaultdict

K, files = float(sys.argv[1]), sys.argv[2:]
assert files and len(files) % 2 == 0, "pairs of BASE H files"
def load(f):
    out = {}
    for l in open(f):
        d = json.loads(l)
        e2 = d.get("proxy_err2", d.get("e2")); e1 = d.get("proxy_err1", d.get("e1", d.get("proxy_err")))
        if e2 is not None: out[d["key"]] = (e1, e2)
    return out
acc = defaultdict(list)
for b, h in zip(files[::2], files[1::2]):
    B, Hq = load(b), load(h)
    for k in B.keys() & Hq.keys():
        if B[k][1] > 0 and Hq[k][1] > 0: acc[k].append((math.log(Hq[k][0] / B[k][0]), math.log(Hq[k][1] / B[k][1])))
for k in sorted(acc):
    v = acc[k]
    print(json.dumps({"key": k, "K": K, "r": math.exp(sum(x[1] for x in v) / len(v)),
                      "e1_ratio": math.exp(sum(x[0] for x in v) / len(v)), "n": len(v)}))
