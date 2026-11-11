#!/usr/bin/env python3
"""table107.py DATA OUT.json -- #141-01 (../w107.md): D(K) = geo mean of the cur arm's measured d_all over the 54 tensor x
samples at each new K (DATA/out/*/LAYER-TENSOR-sS-K{K}-cur.jsonl, both boxes), T(K) = D(K) rounded to 3 decimals ->
OUT.json {K: T}. Refuses unless all 54 rows of every new K exist (phase b's input; written once, before any tab row
of the new K)."""
import glob, json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from settle107 import NEW_K  # noqa: E402

DATA, OUT = sys.argv[1:3]
T, info = {}, {}
for Ks in NEW_K:
    rows = {}
    for f in glob.glob(f"{DATA}/out/*/*-K{Ks}-cur.jsonl"):
        o = json.loads(open(f).readline()); rows[(o["layer"], o["w"], o["sample"])] = o["measured"]["d_all"]
    assert len(rows) == 54, (Ks, len(rows))
    D = math.exp(sum(math.log(v) for v in rows.values()) / len(rows))
    T[Ks] = round(D, 3); info[Ks] = {"D": D, "n": len(rows), "min": min(rows.values()), "max": max(rows.values())}
    print(f"K{Ks}: D {D:.4f} -> T {T[Ks]} (n {len(rows)}, min {info[Ks]['min']:.3f}, max {info[Ks]['max']:.3f})")
assert not os.path.exists(OUT), f"{OUT} exists: written once"
with open(OUT + ".part", "w") as f: json.dump(T, f)
os.replace(OUT + ".part", OUT)
