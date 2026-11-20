#!/usr/bin/env python3
"""merge.py R1DIR R2DIR OUT.jsonl V1OUT.jsonl -- dad1's final result set (see README "Passes"): tags every line with its pass.
S1 = the candidates rebuilt after the multiply fix; their r1 lines go to V1OUT (the v_mul_lo_u32 form)."""
import json, sys
r1, r2, out, v1 = sys.argv[1:5]
S1 = {"mul1_i8", "int4grid", "qtip_1mad", "cptc6", "cptc6full", "cptc5", "cptcmin", "lc1"}
P2 = {"raw", "base", "mad", "mad4", "qtip_3inst", "empty", "read"}
def rd(f, tag, keep):
    o = []
    try:
        for l in open(f):
            r = json.loads(l)
            if keep(r): r["pass"] = tag; o.append(r)
    except FileNotFoundError:
        print("missing", f, file=sys.stderr)
    return o
K1 = lambda r: r["K"] >= 1
L = rd(f"{r1}/p1.jsonl", "p1", lambda r: K1(r) and r["cand"] not in S1) + rd(f"{r2}/p1.jsonl", "p1", K1)
L += rd(f"{r1}/p2.jsonl", "p2", lambda r: K1(r) and r["cand"] in P2) + rd(f"{r2}/p2.jsonl", "p2", K1)
L += rd(f"{r2}/p3.jsonl", "p3", K1)
for d in "123":
    L += rd(f"{r2}/d{d}.jsonl", "d" + d, K1) + rd(f"{r2}/q{d}.jsonl", "p" + d, K1) + rd(f"{r2}/e{d}.jsonl", "d" + d, K1)
for c in "1234": L += rd(f"{r2}/c{c}.jsonl", "c" + c, lambda r: True)
open(out, "w").write("".join(json.dumps(r) + "\n" for r in L))
V = rd(f"{r1}/p1.jsonl", "v1", lambda r: K1(r) and r["cand"] in S1 and "test" not in r)
open(v1, "w").write("".join(json.dumps(r) + "\n" for r in V))
print(len(L), len(V))
