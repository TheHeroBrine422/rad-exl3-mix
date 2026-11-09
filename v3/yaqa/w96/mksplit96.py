#!/usr/bin/env python3
"""mksplit96.py POOLS-V3.json DOCS94.json OUT.json -- w96's split of the qad pool for tuning the Fisher smoothing:
tune half B = the 85 qad documents of w94's qad head dump (docs94.json, so the tune-half rows have head inputs) plus
qad documents drawn at random (seed 96) until B holds half of the pool; fit half A = the rest. Calib document ids."""
import json, random, sys
P = json.load(open(sys.argv[1]))["pool"]; D = json.load(open(sys.argv[2]))["sets"]["qad"]
qad = [i for i, p in enumerate(P) if p == "qad"]; dump = sorted({d["doc"] for d in D})
assert all(P[i] == "qad" for i in dump), "dump doc outside qad"
rest = [i for i in qad if i not in set(dump)]; random.Random(96).shuffle(rest)
B = sorted(dump + rest[:len(qad) // 2 - len(dump)]); A = sorted(set(qad) - set(B))
json.dump({"seed": 96, "pools": "pools-v3.json", "A": A, "B": B, "dump_docs": dump}, open(sys.argv[3], "w"))
print(f"qad {len(qad)} docs: A {len(A)}, B {len(B)} (dump {len(dump)})")
