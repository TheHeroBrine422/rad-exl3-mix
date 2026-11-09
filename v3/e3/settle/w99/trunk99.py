#!/usr/bin/env python3
"""trunk99.py A.rad B.rad -- ../w99.md item 1 donor provenance: are the 336 trunk weights of two containers the same
bytes? Per weight: quantiser, options without src=, shape, encoding struct and every plane's bytes (census.py's ident).
A stored swap27 mix is reused as a donor only if its trunk equals a later fresh pack of the same level.
Last line "TRUNK SAME {json}" (exit 0) or "TRUNK DIFFERS {json}" (exit 1). Run with the staged pipeline/ on PYTHONPATH."""
import json, sys

from rxq import mix, pairs

C = pairs._census()
a, b = sys.argv[1], sys.argv[2]
A, B = C.Rad(a), C.Rad(b)
nosrc = lambda o: ",".join(t for t in o.split(",") if not t.startswith("src="))
diff = []
for n in mix.trunk_names():
    wa, wb = A.w[n], B.w[n]
    if (wa["quant"], nosrc(wa["opts"]), wa["shape"]) != (wb["quant"], nosrc(wb["opts"]), wb["shape"]) or A.ident(n) != B.ident(n):
        diff.append(n)
for n in diff[:10]: print("DIFF", n)
st = {"a": a, "b": b, "trunk": len(mix.trunk_names()), "differ": len(diff)}
print(f"TRUNK {'SAME' if not diff else 'DIFFERS'} {json.dumps(st)}")
sys.exit(1 if diff else 0)
